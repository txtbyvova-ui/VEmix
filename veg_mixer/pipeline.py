"""Orchestration of the end-to-end commands.

analysis -> metadata overrides -> curation -> sequencing -> export (playlist / mix)
-> QC -> reports. The CLI (cli.py) only parses arguments and calls these functions.

Safety rules enforced here (on top of the per-module guards):
* every planned output is checked against every scanned source BEFORE any work
  (``fsutil.assert_outputs_safe``) — no output may overwrite a library file, and two
  outputs may not collide;
* our own outputs are excluded from library scans (no MP3->MP3 re-ingestion);
* ``--dry-run`` writes nothing (cache is opened read-only, no work dirs, no reports);
* a run in which a selected track or the final mix failed can never report PASS.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import GENERATOR
from .analysis import TrackAnalyzer, scan_library
from .cache import AnalysisCache
from .config import AUDIO_EXTENSIONS, AppConfig, apply_recorded_targets
from .curation import curate, default_profile, get_daypart, load_profile
from .ffmpeg_tools import FFmpegTools
from .fsutil import (
    assert_outputs_safe,
    atomic_write_json,
    is_within,
    mp3_audio_sha256,
    name_key,
    path_key,
    remove_stale_dirs,
    remove_stale_files,
)
from .loudness import LoudnessNormalizer
from .metadata import apply_metadata, export_metadata, load_metadata
from .models import (
    AnalyzedTrack,
    AudioProbe,
    ClientProfile,
    LoudnessMeasurement,
    ManifestTrack,
    MixInput,
    NormalizationResult,
    NormalizationStatus,
    OverallStatus,
    PlaylistManifest,
    QCIssue,
    QCReport,
    SequenceResult,
    SequencedTrack,
    Severity,
    TrackFeatures,
    TrackQC,
    to_jsonable,
)
from .playlist import PlaylistExportError, export_playlist
from .quality_control import overall_status, qc_mix, qc_playlist
from .renderer import render_mix, write_mix_tracklist
from .reports import compute_summary, render_text_report, write_reports
from .sequencing import build_sequence, plan_dayparts

log = logging.getLogger(__name__)

CACHE_FILE_NAME = "veg_analysis_cache.json"
LEGACY_CACHE_FILE_NAME = "bpm_cache.json"
ALL_DAYPARTS = ("morning", "afternoon", "evening")
MIX_WORK_PREFIX = ".veg_mix_"
NORM_WORK_PREFIX = "veg_norm_"          # loudness.py pass-2 folders
ENCODE_TEMP_GLOB = ".*.veg-*"           # loudness.py encode temps next to deliverables

Printer = Callable[[str], None]

# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    tools: FFmpegTools
    config: AppConfig
    jobs: int = 1
    dry_run: bool = False
    work_dir: Path | None = None
    out: Printer = print
    # config fields set explicitly on the command line ("loudness.<f>" / "encode.<f>");
    # qc keeps the manifest's recorded targets for every other field
    overrides: frozenset[str] = frozenset()

    @property
    def normalizer(self) -> LoudnessNormalizer:
        return LoudnessNormalizer(self.tools, self.config.loudness, self.config.encode, self.config.qc,
                                  work_dir=self.work_dir)


@dataclass
class LibraryAnalysis:
    folder: Path
    analyzed: list[AnalyzedTrack]
    tracks: list[TrackFeatures]                 # analysable tracks, metadata applied
    broken: list[AnalyzedTrack]
    auto_tracks: list[TrackFeatures] = field(default_factory=list)   # before curator overrides
    warnings: list[str] = field(default_factory=list)
    measurements: dict[str, LoudnessMeasurement] = field(default_factory=dict)   # track_id ->
    probes: dict[str, AudioProbe] = field(default_factory=dict)                  # track_id ->

    @property
    def broken_as_excluded(self) -> list[dict]:
        return [{"source_rel": a.features.rel_path, "stage": "analysis",
                 "reasons": [f"analysis failed: {e}" for e in a.errors]} for a in self.broken]

    @property
    def source_paths(self) -> list[Path]:
        """Every scanned file, broken ones included (outputs must never hit any of them)."""
        return [a.features.path for a in self.analyzed]


def resolve_cache_paths(folder: Path, cache_arg: str | None) -> tuple[Path, Path | None]:
    """(new cache path, legacy cache path). A legacy v1 cache passed via --cache is never
    rewritten: it becomes the read-only migration source for a sibling v2 cache."""
    legacy_default = folder / LEGACY_CACHE_FILE_NAME
    if cache_arg:
        given = Path(cache_arg).resolve()
        if given.suffix.lower() != ".json" or given.suffix.lower() in AUDIO_EXTENSIONS:
            raise ValueError(f"--cache must be a .json file, got {given.name}")
        if given.is_file() and (_is_legacy_cache(given) or given.name.lower() == LEGACY_CACHE_FILE_NAME):
            return given.with_name(given.stem + ".v2.json"), given
        return given, (legacy_default if legacy_default.is_file() else None)
    return folder / CACHE_FILE_NAME, (legacy_default if legacy_default.is_file() else None)


def _is_legacy_cache(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and "schema" not in data


def default_seed(*parts: str | None) -> int:
    """Deterministic seed from names (like automix v1's md5-of-output-name seed)."""
    text = "|".join(p or "" for p in parts)
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_library(ctx: RunContext, folder: Path, *, cache_arg: str | None = None, recursive: bool = False,
                    metadata_path: Path | None = None, measure_loudness: bool = True,
                    exclude: list[Path] | tuple[Path, ...] = ()) -> LibraryAnalysis:
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"--folder '{folder}' does not exist or is not a directory")
    files = scan_library(folder, recursive=recursive, exclude=list(exclude))
    ctx.out(f"Found {len(files)} audio file(s) in {folder}")
    cache_path, legacy_path = resolve_cache_paths(folder, cache_arg)
    cache = AnalysisCache(cache_path, legacy_path=legacy_path, read_only=ctx.dry_run)
    for w in cache.warnings:
        ctx.out(f"  WARNING: {w}")
    ctx.out(f"Cache: {cache_path}" + (f"  (migrating legacy {legacy_path.name})" if legacy_path else "")
            + ("  [read-only: dry run]" if ctx.dry_run else ""))
    analyzer = TrackAnalyzer(ctx.tools, cache, ctx.config.analysis, ctx.normalizer)

    analyzed: list[AnalyzedTrack | None] = [None] * len(files)

    def work(i: int) -> None:
        at = analyzer.analyze(files[i], folder, measure_loudness=measure_loudness)
        analyzed[i] = at
        f = at.features
        status = "ERROR " + "; ".join(at.errors) if at.errors else (
            f"BPM {_fmt(f.bpm, '6.1f')}  Key {f.camelot or '?':<4}  "
            f"E {_fmt(f.energy, '.2f')}  "
            + (f"{at.loudness.integrated_lufs:6.1f} LUFS {at.loudness.true_peak_dbtp:+5.1f} dBTP" if at.loudness else ""))
        cached = "cache" if at.from_cache and all(at.from_cache.values()) else ""
        ctx.out(f"  [{i + 1}/{len(files)}] {f.rel_path}  {status} {cached}".rstrip())

    try:
        if ctx.jobs > 1:
            with ThreadPoolExecutor(ctx.jobs) as pool:
                list(pool.map(work, range(len(files))))
        else:
            for i in range(len(files)):
                work(i)
    finally:
        # persist whatever was computed, even after a crash; a failing save (read-only
        # library, AV lock) is a warning, never a reason to lose the analysis results
        if not ctx.dry_run:
            try:
                cache.save()
            except OSError as e:
                msg = f"analysis cache {cache_path} could not be saved ({e}); it will be recomputed next run"
                ctx.out(f"  WARNING: {msg}")
                log.warning(msg)
    for w in cache.warnings:
        log.warning(w)

    done = [a for a in analyzed if a is not None]
    broken = [a for a in done if not a.ok]
    tracks = [a.features for a in done if a.ok]
    auto_tracks = list(tracks)
    warnings: list[str] = []
    if metadata_path:
        overrides = load_metadata(Path(metadata_path))
        tracks, warnings = apply_metadata(tracks, overrides)
        ctx.out(f"Curator metadata: {metadata_path} ({len(overrides)} row(s))")
        for w in warnings:
            ctx.out(f"  WARNING: {w}")
    by_rel = {a.features.rel_path: a for a in done if a.ok}
    measurements = {t.track_id: by_rel[t.rel_path].loudness for t in tracks
                    if t.rel_path in by_rel and by_rel[t.rel_path].loudness is not None}
    probes = {t.track_id: by_rel[t.rel_path].probe for t in tracks
              if t.rel_path in by_rel and by_rel[t.rel_path].probe is not None}
    if broken:
        ctx.out(f"  {len(broken)} file(s) could not be analysed (reported, not silently skipped):")
        for a in broken:
            ctx.out(f"    - {a.features.rel_path}: {'; '.join(a.errors)}")
    return LibraryAnalysis(folder, done, tracks, broken, auto_tracks, warnings, measurements, probes)


def run_analyze(ctx: RunContext, folder: Path, *, cache_arg=None, recursive=False, metadata_path=None,
                export_path: Path | None = None, measure_loudness: bool = True) -> int:
    exclude = [Path(export_path)] if export_path else []
    lib = analyze_library(ctx, folder, cache_arg=cache_arg, recursive=recursive,
                          metadata_path=metadata_path, measure_loudness=measure_loudness, exclude=exclude)
    unknown = {name: sum(1 for t in lib.tracks if getattr(t, name) in (None, "", []))
               for name in ("bpm", "camelot", "energy", "mood", "genre", "language", "familiarity")}
    ctx.out("")
    ctx.out(f"Analysed: {len(lib.tracks)} ok, {len(lib.broken)} broken")
    ctx.out("Unknown values (stay unknown until curated): "
            + ", ".join(f"{k}={v}" for k, v in unknown.items()))
    scripts = sum(1 for t in lib.tracks if t.script_hint)
    if scripts:
        ctx.out(f"Tracks with a non-Latin script hint (language unverified): {scripts}")
    if lib.measurements:
        lufs = [m.integrated_lufs for m in lib.measurements.values()]
        tps = [m.true_peak_dbtp for m in lib.measurements.values()]
        ctx.out(f"Source loudness: {min(lufs):.1f} .. {max(lufs):.1f} LUFS, max true peak {max(tps):+.1f} dBTP")
    if export_path:
        assert_outputs_safe([Path(export_path)], lib.source_paths)
        if ctx.dry_run:
            ctx.out(f"[dry-run] would export curator metadata to {export_path}")
        else:
            # export the AUTO values (before curator overrides) so the auto_* reference columns
            # stay meaningful; the curator's existing manual values are carried over
            existing = load_metadata(Path(metadata_path)) if metadata_path else None
            export_metadata(lib.auto_tracks, Path(export_path), existing=existing)
            ctx.out(f"Curator metadata exported -> {export_path}")
    return 1 if lib.broken else 0


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def run_normalize(ctx: RunContext, folder: Path, output_dir: Path, *, recursive: bool = False) -> int:
    folder, output_dir = Path(folder).resolve(), Path(output_dir).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"--folder '{folder}' does not exist or is not a directory")
    if is_within(folder, output_dir) or (recursive and is_within(output_dir, folder)):
        raise ValueError("--output must not be the library folder (or contain it, or lie inside it with --recursive): "
                         "normalized MP3s would be re-ingested as sources")
    files = scan_library(folder, recursive=recursive, exclude=[output_dir])
    ctx.out(f"Found {len(files)} audio file(s); target {ctx.config.loudness.target_lufs:.1f} LUFS, "
            f"true peak <= {ctx.config.loudness.target_true_peak_dbtp:.1f} dBTP")
    plan = [(src, output_dir / src.relative_to(folder).with_suffix(".mp3")) for src in files]
    # one output per source: different sources may not map to one name (song.wav + song.flac)
    by_name: dict[str, list[Path]] = {}
    for src, dest in plan:
        by_name.setdefault(name_key(dest), []).append(src)
    clashes = [srcs for srcs in by_name.values() if len(srcs) > 1]
    if clashes:
        raise ValueError("several sources would be written to the same output: "
                         + "; ".join(" + ".join(s.name for s in srcs) for srcs in clashes[:5]))
    assert_outputs_safe([d for _, d in plan] + [output_dir / "normalize_report.json",
                                                output_dir / "normalize_report.txt"], files)
    if ctx.dry_run:
        for src, dest in plan:
            ctx.out(f"  [dry-run] {src.relative_to(folder)} -> {dest}")
        return 0
    _sweep_stale(ctx, [output_dir], recursive=True)
    normalizer = ctx.normalizer
    results: list[NormalizationResult | None] = [None] * len(plan)

    def work(i: int) -> None:
        src, dest = plan[i]
        r = normalizer.normalize_to_mp3(src, dest)
        results[i] = r
        ctx.out(f"  [{i + 1}/{len(plan)}] {src.relative_to(folder)}: {_describe(r)}")

    with ThreadPoolExecutor(max(1, ctx.jobs)) as pool:
        list(pool.map(work, range(len(plan))))
    report = report_from_results([r for r in results if r is not None], ctx, mode="normalize",
                                 base=output_dir, folder=folder)
    json_path, txt_path = write_reports(report, output_dir, stem="normalize_report")
    ctx.out("")
    ctx.out(summary_block(render_text_report(report)))
    ctx.out(f"Reports: {json_path} | {txt_path}")
    return exit_code(report.status)


def report_from_results(results: list[NormalizationResult], ctx: RunContext, *, mode: str, base: Path,
                        folder: Path | None = None) -> QCReport:
    """QC report for a batch of normalizations (normalize command). The summary is
    computed by reports.compute_summary, like every other report."""
    cfg = ctx.config.loudness
    rows, issues = [], []
    for i, r in enumerate(results, 1):
        meta = r.to_metadata()
        file = _rel_or_abs(r.output or r.source, base)
        source_rel = _rel_or_abs(r.source, folder) if folder else r.source.name
        rows.append(TrackQC(position=i, file=file, source_rel=source_rel,
                            original_lufs=meta["original_lufs"], final_lufs=meta["measured_lufs"],
                            original_true_peak_dbtp=meta["original_true_peak_dbtp"],
                            final_true_peak_dbtp=meta["true_peak_dbtp"], duration_s=meta["duration_s"],
                            status=r.status.value, warnings=r.warnings + r.review_reasons + r.errors))
        for e in r.errors:
            issues.append(QCIssue(Severity.FAIL, "normalization_failed", e, track_position=i, file=file))
        for rr in r.review_reasons:
            issues.append(QCIssue(Severity.WARNING, "requires_review", rr, track_position=i, file=file))
    report = QCReport(
        mode=mode, generated_at=_now(), status=overall_status(issues, []), target_lufs=cfg.target_lufs,
        target_true_peak_dbtp=cfg.target_true_peak_dbtp, tracks=rows, issues=issues,
        checks_run=["integrated_loudness", "true_peak", "decode", "duration", "channels", "sample_rate"])
    report.summary = compute_summary(report)
    return report


# ---------------------------------------------------------------------------
# Curation + sequencing (shared by playlist and mix)
# ---------------------------------------------------------------------------

@dataclass
class Selection:
    profile: ClientProfile
    lib: LibraryAnalysis
    sequences: dict[str | None, SequenceResult]
    excluded: list[dict]
    editorial_scores: dict[str, float]
    # per sequence key: eligible tracks placed in NO sequence of this run (with reasons)
    not_selected: dict[str | None, list[dict]] = field(default_factory=dict)


def expand_dayparts(profile: ClientProfile, daypart: str | None) -> list[str] | None:
    """'all' = the three built-in dayparts followed by any custom ones from the profile."""
    if not daypart:
        return None
    if daypart != "all":
        return [daypart]
    names = list(ALL_DAYPARTS)
    names += [n for n in profile.dayparts if n not in names]
    return names


def select(ctx: RunContext, folder: Path, *, profile_path: Path | None, metadata_path: Path | None,
           cache_arg: str | None, recursive: bool, daypart: str | None, seed: int,
           target_duration_s: float | None, max_tracks: int | None,
           exclude: list[Path] | tuple[Path, ...] = ()) -> Selection:
    profile = load_profile(Path(profile_path)) if profile_path else default_profile()
    dayparts = expand_dayparts(profile, daypart)
    for name in dayparts or []:
        get_daypart(profile, name)       # fail fast on typos
    lib = analyze_library(ctx, folder, cache_arg=cache_arg, recursive=recursive, metadata_path=metadata_path,
                          exclude=exclude)
    cur = curate(lib.tracks, profile)
    editorial = {d.track.track_id: d.editorial_score for d in cur.decisions}
    # only explicit allowlisting exempts a track from the daypart poor-fit cut; a general
    # curator approval must not switch the daypart energy filter off (VEDIT-1)
    protected = frozenset(d.track.track_id for d in cur.decisions if d.eligible and d.allowlisted)
    excluded = [{"source_rel": d.track.rel_path, "stage": "curation", "reasons": list(d.exclusion_reasons)}
                for d in cur.excluded]
    excluded += lib.broken_as_excluded
    review = [d for d in cur.decisions if d.eligible and (d.review_reasons or d.track.requires_review)]
    ctx.out("")
    ctx.out(f"Curation ({profile.client}): {len(cur.eligible)} eligible, {len(cur.excluded)} excluded, "
            f"{len(review)} eligible but flagged for review")
    for d in cur.excluded[:50]:
        ctx.out(f"  excluded: {d.track.rel_path}: {'; '.join(d.exclusion_reasons)}")
    if len(cur.excluded) > 50:
        ctx.out(f"  ... {len(cur.excluded) - 50} more (see manifest)")
    seq_cfg = ctx.config.sequencing
    if dayparts:
        sequences: dict[str | None, SequenceResult] = dict(plan_dayparts(
            cur.eligible, seq_cfg, profile=profile, dayparts=list(dayparts), editorial_scores=editorial,
            seed=seed, target_duration_s=target_duration_s, max_tracks=max_tracks, protected_track_ids=protected))
    else:
        sequences = {None: build_sequence(cur.eligible, seq_cfg, profile=profile, editorial_scores=editorial,
                                          seed=seed, target_duration_s=target_duration_s, max_tracks=max_tracks,
                                          protected_track_ids=protected)}
    # keyed by the file path, not the track id: two different files must never collapse (VSPEC-2)
    placed = {path_key(st.track.path) for seq in sequences.values() for st in seq.tracks}
    by_rel = {t.rel_path: t for t in cur.eligible}
    not_selected: dict[str | None, list[dict]] = {}
    for name, seq in sequences.items():
        label = name or "playlist"
        ctx.out(f"Sequenced {label}: {len(seq.tracks)} track(s), {seq.total_duration_s / 60:.1f} min, "
                f"{len(seq.flagged)} flagged transition(s)")
        for fl in seq.flagged:
            ctx.out(f"  flagged {fl.from_position}->{fl.to_position}: {'; '.join(fl.reasons)}")
        dropped = [{"source_rel": ns.get("rel_path"), "track_id": ns.get("track_id"), "reason": ns.get("reason")}
                   for ns in seq.not_selected
                   if not str(ns.get("reason", "")).startswith("excluded")
                   and (ns.get("rel_path") not in by_rel or path_key(by_rel[ns["rel_path"]].path) not in placed)]
        not_selected[name] = dropped
    unplaced: dict[str, dict] = {}
    for lst in not_selected.values():
        for d in lst:
            unplaced.setdefault(str(d["source_rel"]), d)
    if unplaced:
        ctx.out(f"  {len(unplaced)} eligible track(s) were not scheduled in any output (reasons in the manifest):")
        for d in list(unplaced.values())[:20]:
            ctx.out(f"    - {d['source_rel']}: {d['reason']}")
        if len(unplaced) > 20:
            ctx.out(f"    ... {len(unplaced) - 20} more")
    return Selection(profile, lib, sequences, excluded, editorial, not_selected)


def print_sequence(ctx: RunContext, seq: SequenceResult) -> None:
    for st in seq.tracks:
        t = st.track
        score = f"{st.transition_in.total:.2f}" if st.transition_in else "  - "
        ctx.out(f"  {st.position:>3}. [{score}] E={_fmt(t.energy, '.2f')} BPM={_fmt(t.bpm, '.0f')} "
                f"{t.camelot or '?':<3} {t.display_name}")


# ---------------------------------------------------------------------------
# playlist (Mode A)
# ---------------------------------------------------------------------------

def run_playlist(ctx: RunContext, folder: Path, output_dir: Path, *, profile_path=None, metadata_path=None,
                 cache_arg=None, recursive=False, daypart: str | None = None, seed: int | None = None,
                 target_duration_s: float | None = None, max_tracks: int | None = None) -> int:
    output_dir = Path(output_dir).resolve()
    if path_key(output_dir) == path_key(Path(folder).resolve()):
        raise ValueError("--output must be a separate folder, not the music library itself")
    # a library inside the published tracks folder(s) would be re-ingested: refuse before analysis
    candidates = [output_dir] + ([output_dir / n for n in ALL_DAYPARTS] if daypart else [])
    if any(is_within(Path(folder).resolve(), c / "tracks") for c in candidates):
        raise ValueError("--folder lies inside the playlist's own tracks/ folder; use the original library")
    seed = seed if seed is not None else default_seed("playlist", str(profile_path), daypart or "")
    # The output folder is NOT excluded from the scan (a library inside --output must still be
    # found, VSAFE-1); earlier VEgroove playlists are skipped by scan_library via their manifests.
    sel = select(ctx, folder, profile_path=profile_path, metadata_path=metadata_path, cache_arg=cache_arg,
                 recursive=recursive, daypart=daypart, seed=seed, target_duration_s=target_duration_s,
                 max_tracks=max_tracks)
    targets = {name: (output_dir / name if name else output_dir) for name in sel.sequences}
    for name, out in targets.items():
        if not is_within(out, output_dir):
            raise ValueError(f"daypart name {name!r} would write outside --output ({out})")
    _sweep_stale(ctx, list(targets.values()))
    assert_outputs_safe([out / f for out in targets.values()
                         for f in ("manifest.json", "playlist.m3u8", "tracklist.txt", "qc_report.json",
                                   "qc_report.txt")], sel.lib.source_paths)
    normalizer = ctx.normalizer
    statuses: list[OverallStatus] = []
    export_errors = 0
    for name, seq in sel.sequences.items():
        out = targets[name]
        ctx.out("")
        ctx.out(f"=== Playlist{f' [{name}]' if name else ''} -> {out} ===")
        if not seq.tracks:
            ctx.out(f"  ERROR: no tracks were selected for {name or 'the playlist'} "
                    f"({len(sel.not_selected.get(name, []))} not selected, {len(sel.excluded)} excluded) — nothing exported")
            if (out / "manifest.json").is_file():
                ctx.out(f"  WARNING: {out} still holds the playlist of an EARLIER run; it was not updated")
            statuses.append(OverallStatus.FAIL)
            continue
        if ctx.dry_run:
            print_sequence(ctx, seq)
        try:
            res = export_playlist(seq, out, normalizer, profile_name=sel.profile.client, excluded=sel.excluded,
                                  input_measurements=sel.lib.measurements, source_probes=sel.lib.probes,
                                  jobs=ctx.jobs, dry_run=ctx.dry_run, progress=_progress(ctx),
                                  not_selected=sel.not_selected.get(name, []))
        except PlaylistExportError as e:
            ctx.out(f"  ERROR: playlist export failed (nothing published here): {e}")
            statuses.append(OverallStatus.FAIL)
            export_errors += 1
            continue
        if ctx.dry_run:
            ctx.out(f"[dry-run] nothing written; {len(seq.tracks)} track(s) would be exported to {out}")
            continue
        if res.failed:
            ctx.out(f"  {len(res.failed)} track(s) FAILED normalization and were left out of the playlist")
        if res.manifest_path is None:
            # nothing was published (every track failed); never QC a stale previous run instead
            ctx.out(f"  ERROR: no playlist was published to {out} — every selected track failed normalization")
            statuses.append(OverallStatus.FAIL)
            continue
        report = qc_playlist(out, normalizer, profile=sel.profile, qc_cfg=ctx.config.qc)
        json_path, txt_path = write_reports(report, out)
        statuses.append(report.status)
        ctx.out(summary_block(render_text_report(report)))
        ctx.out(f"Playlist: {res.m3u8_path}\nManifest: {res.manifest_path}\nQC: {json_path} | {txt_path}")
    if export_errors:            # locked output, rolled-back publish, foreign files: a runtime error
        return EXIT_RUNTIME
    return exit_code(_worst(statuses)) if statuses else 0


# ---------------------------------------------------------------------------
# mix (Mode B)
# ---------------------------------------------------------------------------

def mix_sidecars(out: Path) -> list[Path]:
    """Files written next to a mix MP3."""
    return [out.with_name(out.stem + ".manifest.json"), out.with_name(out.stem + "_tracklist.txt"),
            out.with_name(out.stem + "_qc_report.json"), out.with_name(out.stem + "_qc_report.txt")]


def planned_mix_outputs(output_file: Path, names: list[str | None], parts: int) -> list[Path]:
    multi = len(names) > 1
    outs = []
    for name in names:
        base = output_file.with_name(f"{output_file.stem}_{name}{output_file.suffix}") if name and multi else output_file
        for k in range(1, max(1, parts) + 1):
            outs.append(base if parts <= 1 else base.with_name(f"{base.stem}_part{k}{base.suffix}"))
    return outs


def run_mix(ctx: RunContext, folder: Path, output_file: Path, *, profile_path=None, metadata_path=None,
            cache_arg=None, recursive=False, daypart: str | None = None, parts: int = 1,
            seed: int | None = None, target_duration_s: float | None = None, max_tracks: int | None = None) -> int:
    output_file = Path(output_file).resolve()
    if output_file.suffix.lower() != ".mp3":
        raise ValueError(f"--output must be an .mp3 file, got {output_file.name}")
    if parts < 1:
        raise ValueError("--parts must be >= 1")
    seed = seed if seed is not None else default_seed(output_file.name)
    profile_for_names = load_profile(Path(profile_path)) if profile_path else default_profile()
    names = expand_dayparts(profile_for_names, daypart) or [None]
    planned = planned_mix_outputs(output_file, names, parts)
    # Earlier VEgroove mixes are skipped by scan_library via their sibling manifest. The planned
    # outputs are deliberately NOT excluded from the scan: if one of them names an existing library
    # file, that file must show up as a source so the guard below refuses to overwrite it.
    sel = select(ctx, folder, profile_path=profile_path, metadata_path=metadata_path, cache_arg=cache_arg,
                 recursive=recursive, daypart=daypart, seed=seed, target_duration_s=target_duration_s,
                 max_tracks=max_tracks)
    assert_outputs_safe([p for out in planned for p in [out, *mix_sidecars(out)]], sel.lib.source_paths)

    jobs: list[tuple[Path, SequenceResult]] = []
    for name, seq in sel.sequences.items():
        base = (output_file.with_name(f"{output_file.stem}_{name}{output_file.suffix}")
                if name and len(sel.sequences) > 1 else output_file)
        for k, part in enumerate(_split_parts(seq, parts), 1):
            out = base if parts <= 1 else base.with_name(f"{base.stem}_part{k}{base.suffix}")
            jobs.append((out, part))
    statuses = []
    for out, seq in jobs:
        ctx.out("")
        ctx.out(f"=== Mix -> {out}  ({len(seq.tracks)} tracks, {seq.total_duration_s / 3600:.2f} h, seed {seed}) ===")
        if not seq.tracks:
            ctx.out(f"  ERROR: no eligible tracks for this mix ({len(sel.excluded)} excluded, "
                    f"{len(sel.not_selected.get(seq.daypart, []))} not selected) — nothing rendered")
            _warn_earlier_mix(ctx, out)
            statuses.append(OverallStatus.FAIL)
            continue
        if ctx.dry_run:
            print_sequence(ctx, seq)
            ctx.out("[dry-run] nothing written")
            continue
        statuses.append(_render_one_mix(ctx, sel, seq, out, seed))
    return exit_code(_worst(statuses)) if statuses else 0


def _split_parts(seq: SequenceResult, parts: int) -> list[SequenceResult]:
    """Legacy --parts: contiguous chunks of the (already sequenced) order. Positions are
    renumbered per chunk and flagged transitions are remapped to the new positions."""
    if parts <= 1 or len(seq.tracks) <= 1:
        return [seq]
    size = -(-len(seq.tracks) // parts)
    chunks = []
    for k in range(parts):
        tracks = seq.tracks[k * size:(k + 1) * size]
        if not tracks:
            continue
        remap = {t.position: i + 1 for i, t in enumerate(tracks)}
        renumbered = [dataclasses.replace(t, position=i + 1, transition_in=None if i == 0 else t.transition_in)
                      for i, t in enumerate(tracks)]
        flagged = [dataclasses.replace(f, from_position=remap[f.from_position], to_position=remap[f.to_position])
                   for f in seq.flagged if f.from_position in remap and f.to_position in remap]
        chunks.append(dataclasses.replace(seq, tracks=renumbered, flagged=flagged,
                                          total_duration_s=sum(t.track.duration_s for t in tracks)))
    return chunks


def _render_one_mix(ctx: RunContext, sel: Selection, seq: SequenceResult, out: Path, seed: int) -> OverallStatus:
    normalizer = ctx.normalizer
    out.parent.mkdir(parents=True, exist_ok=True)
    work_parent = ctx.work_dir or Path(tempfile.gettempdir())
    _sweep_stale(ctx, [out.parent])
    work = Path(tempfile.mkdtemp(prefix=MIX_WORK_PREFIX, dir=work_parent))
    manifest_path, tracklist_path = mix_sidecars(out)[:2]
    try:
        # 1) normalize every track to a lossless float WAV (no MP3 generation loss)
        ctx.out("  Normalizing tracks to lossless intermediates ...")
        results: list[NormalizationResult | None] = [None] * len(seq.tracks)

        def work_one(i: int) -> None:
            t = seq.tracks[i].track
            r = normalizer.normalize_to_wav(t.path, work / f"{i + 1:04d}.wav",
                                            input_measurement=sel.lib.measurements.get(t.track_id),
                                            source_probe=sel.lib.probes.get(t.track_id))
            results[i] = r
            ctx.out(f"    [{i + 1}/{len(seq.tracks)}] {t.display_name}: {_describe(r)}")

        with ThreadPoolExecutor(max(1, ctx.jobs)) as pool:
            list(pool.map(work_one, range(len(seq.tracks))))

        inputs: list[MixInput] = []
        used: list[tuple[SequencedTrack, NormalizationResult]] = []
        failed_excluded = []
        for st, r in zip(seq.tracks, results):
            t = st.track
            if r is None or r.status is NormalizationStatus.FAILED or r.output is None:
                failed_excluded.append({"source_rel": t.rel_path, "track_id": t.track_id, "stage": "normalization",
                                        "sequence_position": st.position,
                                        "reasons": ["normalization failed: " + "; ".join(r.errors if r else ["not run"])],
                                        "loudness": r.to_metadata() if r else {}})
                continue
            inputs.append(MixInput(position=len(inputs) + 1, pcm_path=r.output, name=t.display_name,
                                   source_rel=t.rel_path, bpm=t.bpm, energy=t.energy, camelot=t.camelot))
            used.append((st, r))
        if not inputs:
            ctx.out("  ERROR: every selected track failed normalization; nothing to mix")
            for e in failed_excluded:
                ctx.out(f"    - {e['source_rel']}: {'; '.join(e['reasons'])}")
            _warn_earlier_mix(ctx, out)
            return OverallStatus.FAIL

        # 2) render (streaming, FX only on outgoing regions, overlap peak protection)
        ctx.out(f"  Rendering {len(inputs)} track(s) ...")
        tcfg = dataclasses.replace(ctx.config.transitions, seed=seed)
        render = render_mix(inputs, work / "mix.wav", tcfg, ceiling_dbtp=ctx.config.loudness.pcm_true_peak_ceiling,
                            sample_rate=ctx.config.encode.sample_rate, seed=seed, progress=_progress(ctx))

        # 3) final loudness pass + single MP3 encode + verification on the encoded mix. Never
        #    loudnorm's dynamic (AGC) mode here: it would re-balance the tracks inside the mix.
        ctx.out("  Final loudness check and MP3 encode ...")
        mix_normalizer = LoudnessNormalizer(
            ctx.tools, dataclasses.replace(ctx.config.loudness, max_dynamic_gain_db=0.0),
            ctx.config.encode, ctx.config.qc, work_dir=ctx.work_dir)
        final = mix_normalizer.normalize_to_mp3(render.wav_path, out,
                                                metadata={"title": out.stem, "artist": "VEgroove"})
        ctx.out(f"    mix: {_describe(final)}")
        published = final.status is not NormalizationStatus.FAILED and final.output is not None
        if published:
            write_mix_tracklist(render, inputs, tracklist_path)
        else:
            ctx.out(f"  ERROR: the final mix failed verification and was NOT published: {'; '.join(final.errors)}")
            # never leave an earlier run's mix/tracklist looking like this run's deliverable (DSP-3)
            _move_previous_outputs_aside(ctx, [out, tracklist_path])

        # 4) manifest (positions renumbered after failures; unscored new adjacencies flagged)
        old_to_new = {st.position: i + 1 for i, (st, _) in enumerate(used)}
        manifest_tracks = []
        flagged: list[dict] = []
        prev_old: int | None = None
        for (st, r), mt in zip(used, render.tracks):
            consecutive = prev_old is not None and st.position == prev_old + 1
            transition_in = to_jsonable(st.transition_in) if st.transition_in and consecutive else None
            if prev_old is not None and not consecutive:
                flagged.append({"from_index": mt.position - 1, "to_index": mt.position, "kind": "unscored",
                                "reasons": ["a track between these two failed normalization; this adjacency was "
                                            "never scored by the sequencer"]})
            prev_old = st.position
            manifest_tracks.append(ManifestTrack(
                index=mt.position, file="", title=st.track.title or st.track.path.stem, artist=st.track.artist,
                duration_s=mt.duration_s, source_rel=st.track.rel_path, source_path=str(st.track.path),
                track_id=st.track.track_id, loudness=r.to_metadata(), features=_features_dict(st.track),
                transition_in=transition_in,
                flags=(["loudness_review"] if r.requires_review else [])
                + (["metadata_review"] if st.track.requires_review else [])))
        for f in seq.flagged:
            a, b = old_to_new.get(f.from_position), old_to_new.get(f.to_position)
            if a is not None and b is not None and b == a + 1:
                flagged.append({"from_index": a, "to_index": b, "reasons": f.reasons, "kind": "sequencing"})
        flagged += [{"from_index": tr.position - 1, "to_index": tr.position, "reasons": tr.flags,
                     "kind": "transition"} for tr in render.transitions if tr.uncertain or tr.gain_reduction_db > 0]
        mix_info = {
            "file": out.name, "published": published,
            "expected_duration_s": render.expected_duration_s, "rendered_duration_s": render.duration_s,
            "sample_rate": render.sample_rate, "fade_in_s": tcfg.mix_fade_in_s, "fade_out_s": tcfg.mix_fade_out_s,
            "tracks": to_jsonable(render.tracks), "transitions": to_jsonable(render.transitions),
            "loudness": final.to_metadata(), "tracklist": tracklist_path.name if published else None,
            "seed": seed, "render_peak_dbfs": render.peak_dbfs, "clipped_samples": render.clipped_samples,
        }
        if published:
            # hash of the MP3 audio payload (tags excluded): a curator's tag edit stays valid
            mix_info["audio_sha256"], mix_info["audio_size_bytes"] = mp3_audio_sha256(out)
        manifest = PlaylistManifest(
            generator=GENERATOR, generated_at=_now(), mode="mix", profile=sel.profile.client,
            daypart=seq.daypart, loudness_targets=_targets(ctx), tracks=manifest_tracks,
            excluded=sel.excluded + failed_excluded, flagged_transitions=flagged,
            not_selected=sel.not_selected.get(seq.daypart, []),
            warnings=[f"{e['source_rel']}: {'; '.join(e['reasons'])}" for e in failed_excluded]
            + ([f"final mix not published: {e}" for e in final.errors] if not published else []),
            mix=mix_info)
        atomic_write_json(manifest_path, manifest.to_dict())
    finally:
        shutil.rmtree(work, ignore_errors=True)

    report = qc_mix(manifest_path, normalizer, profile=sel.profile, qc_cfg=ctx.config.qc)
    json_path, txt_path = write_reports(report, out.parent, stem=f"{out.stem}_qc_report")
    ctx.out(summary_block(render_text_report(report)))
    ctx.out(f"Mix: {out if published else '(not published)'}\nTracklist: {tracklist_path if published else '-'}\n"
            f"Manifest: {manifest_path}\nQC: {json_path} | {txt_path}")
    return report.status if published else OverallStatus.FAIL


# ---------------------------------------------------------------------------
# qc
# ---------------------------------------------------------------------------

def find_qc_targets(folder: Path) -> list[tuple[str, Path]]:
    """Playlist folders (manifest.json, also one level down for dayparts) and mix manifests."""
    folder = Path(folder).resolve()
    found: list[tuple[str, Path]] = []
    for candidate in [folder, *sorted(p for p in folder.iterdir() if p.is_dir())] if folder.is_dir() else []:
        if (candidate / "manifest.json").is_file():
            found.append(("playlist", candidate))
        for m in sorted(candidate.glob("*.manifest.json")):
            found.append(("mix", m))
    return found


def run_qc(ctx: RunContext, folder: Path, *, profile_path: Path | None = None) -> int:
    targets = find_qc_targets(folder)
    if not targets:
        raise FileNotFoundError(f"no manifest.json or *.manifest.json found in {folder}")
    profile = load_profile(Path(profile_path)) if profile_path else None
    statuses = []
    for kind, target in targets:
        manifest_file = target / "manifest.json" if kind == "playlist" else target
        normalizer = normalizer_for_manifest(ctx, manifest_file)
        ctx.out(f"=== QC ({kind}) {target} ===")
        if kind == "playlist":
            report = qc_playlist(target, normalizer, profile=profile, qc_cfg=ctx.config.qc)
            out_dir, stem = target, "qc_report"
        else:
            report = qc_mix(target, normalizer, profile=profile, qc_cfg=ctx.config.qc)
            out_dir, stem = target.parent, f"{target.name[: -len('.manifest.json')]}_qc_report"
        statuses.append(report.status)
        ctx.out(summary_block(render_text_report(report)))
        if ctx.dry_run:
            ctx.out(f"[dry-run] reports not written ({out_dir / (stem + '.json')})")
        else:
            paths = write_reports(report, out_dir, stem=stem)
            ctx.out(f"Reports: {paths[0]} | {paths[1]}")
    return exit_code(_worst(statuses))


def normalizer_for_manifest(ctx: RunContext, manifest_file: Path) -> LoudnessNormalizer:
    """QC checks against the targets recorded at export time; only fields given explicitly
    on the command line override them (field by field)."""
    lc, enc = ctx.config.loudness, ctx.config.encode
    try:
        targets = json.loads(Path(manifest_file).read_text(encoding="utf-8")).get("loudness_targets") or {}
    except (OSError, ValueError, AttributeError):
        targets = {}
    lc2, enc2, problems = apply_recorded_targets(targets, lc, enc, skip=ctx.overrides)
    for problem in problems:     # QC reports them as FAIL manifest_invalid; valid fields still apply
        ctx.out(f"  WARNING: {Path(manifest_file).name}: ignoring invalid loudness_targets.{problem}")
    return LoudnessNormalizer(ctx.tools, lc2, enc2, ctx.config.qc, work_dir=ctx.work_dir)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_ENV, EXIT_WARNING, EXIT_RUNTIME = 0, 1, 2, 3, 4, 5


def exit_code(status: OverallStatus | None) -> int:
    if status is OverallStatus.FAIL:
        return EXIT_FAIL
    if status is OverallStatus.WARNING:
        return EXIT_WARNING
    return EXIT_OK


def _worst(statuses: list[OverallStatus]) -> OverallStatus | None:
    order = [OverallStatus.PASS, OverallStatus.WARNING, OverallStatus.FAIL]
    return max(statuses, key=order.index) if statuses else None


def _targets(ctx: RunContext) -> dict:
    lc, enc = ctx.config.loudness, ctx.config.encode
    return {"target_lufs": lc.target_lufs, "target_true_peak_dbtp": lc.target_true_peak_dbtp,
            "target_lra": lc.target_lra, "lufs_tolerance": lc.lufs_tolerance,
            "true_peak_tolerance_db": lc.true_peak_tolerance_db, "bitrate": enc.bitrate,
            "sample_rate": enc.sample_rate, "channels": enc.channels, "codec": enc.codec}


def _features_dict(t: TrackFeatures) -> dict:
    d = t.to_dict()
    d.pop("path", None)
    return d


def _describe(r: NormalizationResult) -> str:
    meta = r.to_metadata()
    if r.status is NormalizationStatus.FAILED:
        return "FAILED - " + "; ".join(r.errors[:2])
    mode = meta["processing_mode"]
    txt = (f"{r.status.value.upper()} {meta['original_lufs']:.1f} -> {meta['measured_lufs']:.1f} LUFS, "
           f"TP {meta['true_peak_dbtp']:+.1f} dBTP ({mode})")
    if r.review_reasons:
        txt += " | REVIEW: " + "; ".join(r.review_reasons[:1])
    return txt


def _move_previous_outputs_aside(ctx: RunContext, paths: list[Path]) -> None:
    """A failed run must not leave an EARLIER run's deliverable looking current (DSP-3). The
    aside copy gets a hidden name ('.<name>.stale-<UTC>') that library scans never ingest."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for old in paths:
        if old.exists():
            aside = old.with_name(f".{old.stem}.stale-{stamp}{old.suffix}")
            old.replace(aside)
            ctx.out(f"  moved the previous run's {old.name} aside -> {aside.name}")


def _warn_earlier_mix(ctx: RunContext, out: Path) -> None:
    if out.exists() or mix_sidecars(out)[0].exists():
        ctx.out(f"  WARNING: {out.name} and its manifest/QC report are from an EARLIER run and were not updated")


def _sweep_stale(ctx: RunContext, output_dirs: list[Path], *, recursive: bool = False) -> None:
    """Remove temporary leftovers of KILLED earlier runs (idle > 6 h): mix work folders and
    normalizer pass-2 folders in the work/temp dir, hidden encode temps next to deliverables."""
    if ctx.dry_run:
        return
    work_parent = ctx.work_dir or Path(tempfile.gettempdir())
    removed = remove_stale_dirs(work_parent, MIX_WORK_PREFIX) + remove_stale_dirs(work_parent, NORM_WORK_PREFIX)
    for d in output_dirs:
        if Path(d).is_dir():
            removed += remove_stale_files(d, ENCODE_TEMP_GLOB, recursive=recursive)
    for r in removed:
        ctx.out(f"  removed a stale temporary item of an interrupted run: {r}")


def summary_block(text: str) -> str:
    """The required summary block of a text report (up to the 'Overall status' line)."""
    lines = []
    for line in text.splitlines():
        lines.append(line)
        if line.startswith("Overall status:"):
            break
    return "\n".join(lines)


def _progress(ctx: RunContext):
    def cb(done: int, total: int, message: str = "") -> None:
        ctx.out(f"    [{done}/{total}] {message}".rstrip())
    return cb


def _fmt(v, spec: str) -> str:
    return "?" if v is None else format(v, spec)


def _rel_or_abs(p: Path, base: Path | None) -> str:
    if base is None:
        return str(p)
    try:
        return Path(p).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return str(p)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "RunContext", "analyze_library", "run_analyze", "run_normalize", "run_playlist", "run_mix", "run_qc",
    "resolve_cache_paths", "expand_dayparts", "normalizer_for_manifest", "exit_code",
    "EXIT_OK", "EXIT_FAIL", "EXIT_USAGE", "EXIT_ENV", "EXIT_WARNING", "EXIT_RUNTIME",
]
