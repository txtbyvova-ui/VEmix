"""Orchestration of the VEmix 3.0 commands (spec §7): enrich, interpret, plan, suggest, compare, render.

They sit next to the 2.0 commands in ``pipeline.py`` and reuse its pieces (library
analysis, output-safety checks, playlist export, mix rendering, QC) without changing
them. The CLI only parses arguments and calls these functions.

Network policy: only ``enrich`` and ``interpret`` (in ``online`` mode) and ``suggest`` (one
AI request, plus at most one re-ask when the answer is unparsable or does not make a valid
scenario, or the provider error is retryable) go online; ``plan`` reads the research and AI
caches, ``render`` the plan file and the analysis cache — neither ever uses the network, so a
plan is reproducible from local data. ``--dry-run`` writes nothing: caches are read-only and
no exports are written (``suggest --dry-run`` sends no request).

Every output is checked against the audio sources AND the command's own inputs (scenario,
curator metadata, profile, plan file, analysis / research / AI caches): no flag can overwrite them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .curation import curate, default_profile, load_profile
from .dramaturgy import PRESET_ORDER, TrackContext, build_context, plan
from .fsutil import assert_outputs_safe, atomic_write_text, is_within, path_key
from .interpret_rules import derive
from .models import (
    FlaggedTransition,
    OverallStatus,
    SequencedTrack,
    SequenceResult,
    TrackFeatures,
    to_jsonable,
)
from .pipeline import (
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_USAGE,
    EXIT_WARNING,
    LibraryAnalysis,
    OptionalDependencyError,
    RunContext,
    Selection,
    _progress,
    _render_one_mix,
    _sweep_stale,
    analyze_library,
    default_seed,
    resolve_cache_paths,
    exit_code,
    mix_sidecars,
    summary_block,
)
from .plan_report import (
    PlanFileError,
    load_plan_file,
    pick_variant,
    plan_document,
    render_comparison,
    render_plan_text,
    write_plan_csv,
    write_plan_json,
    write_plan_m3u8,
)
from .playlist import PlaylistExportError, export_playlist
from .quality_control import qc_playlist
from .reports import render_text_report, write_reports
from .scenario import ScenarioError, load_scenario, resolve_scenario, with_overrides
from .sequencing import effective_config, transition_score

DATA_MODES = ("off", "cache", "online")
RESEARCH_CACHE_FALLBACK = "veg_research_cache.json"
AI_CACHE_FALLBACK = "veg_ai_cache.json"
# violation kinds that concern the transition INTO a position (become manifest flags)
TRANSITION_VIOLATIONS = ("forbidden_pair", "tempo_jump", "key_jump", "energy_jump", "artist_gap")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cache_path(folder: Path, given: str | None, default_name: str, suffix_hint: str,
                analysis_cache: str | None = None) -> Path:
    """An explicit path; else next to the analysis cache when ``--cache`` moved it out of the
    library (a read-only library stays untouched); else in the library folder."""
    if given:
        p = Path(given).resolve()
        if p.suffix.lower() != ".json":
            raise ValueError(f"{suffix_hint} must be a .json file, got {p.name}")
        return p
    if analysis_cache:
        return Path(analysis_cache).resolve().parent / default_name
    return Path(folder).resolve() / default_name


def _guard_inputs(outputs: list[Path], inputs: dict[str, Any]) -> None:
    """Refuse an output that would overwrite one of the command's inputs or caches."""
    keys = {}
    for flag, value in inputs.items():
        for item in (value if isinstance(value, (list, tuple)) else [value]):
            if item:
                keys.setdefault(path_key(Path(item).resolve()), flag)
    for out in outputs:
        flag = keys.get(path_key(Path(out).resolve()))
        if flag:
            raise ValueError(f"refusing to write {out}: it is the {flag} this command reads")


def _analysis_caches(folder: Path, cache_arg: str | None) -> list[Path]:
    new, legacy = resolve_cache_paths(Path(folder).resolve(), cache_arg)
    return [p for p in (new, legacy) if p]


def _research_records(ctx: RunContext, lib: LibraryAnalysis, mode: str, cache_path: Path) -> dict[str, dict]:
    """Research records from the cache only (plan / interpret never go online for facts)."""
    if mode == "off":
        return {}
    from .research import ResearchCache, enrich_tracks
    cache = ResearchCache(cache_path, read_only=True)
    res = enrich_tracks(lib.tracks, probes=lib.probes, mode="cache", cache=cache)
    for w in res.warnings[:10]:
        ctx.out(f"  WARNING: research: {w}")
    matched = sum(1 for r in res.records.values() if r.status == "matched")
    stale = sum(1 for r in res.records.values() if getattr(r, "stale", False))
    ctx.out(f"Research cache: {cache_path.name}: {matched} matched record(s) of {len(lib.tracks)} track(s)"
            + (f", {stale} past their time-to-live (still used; refresh with `enrich --mode online`)"
               if stale else ""))
    return {tid: r.to_dict() for tid, r in res.records.items()}


def _ai_interpretations(ctx: RunContext, lib: LibraryAnalysis, mode: str, cache_path: Path,
                        records: dict[str, dict], use_community_tags: bool) -> dict[str, dict]:
    if mode == "off":
        return {}
    from .ai import AICache, interpret_tracks
    cache = AICache(cache_path, read_only=True)
    res = interpret_tracks(lib.tracks, provider=None, mode="cache", cache=cache, segments=lib.segments,
                           research=records, use_community_tags=use_community_tags)
    for w in res.warnings[:10]:
        ctx.out(f"  WARNING: AI: {w}")
    ctx.out(f"AI cache: {cache_path.name}: interpretations for {len(res.interpretations)} track(s)")
    fake = sum(1 for i in res.interpretations.values() if str(i.get("source", "")).startswith("ai:fake/"))
    if fake:
        ctx.out(f"  WARNING: {fake} interpretation(s) come from the 'fake' demo provider (an echo of the input, "
                f"not a model); re-run `interpret` with a real provider or plan with --ai off")
    return dict(res.interpretations)


def _contexts(tracks: list[TrackFeatures], lib: LibraryAnalysis, records: dict[str, dict], interps: dict[str, dict],
              editorial: dict[str, float] | None, use_community_tags: bool) -> list[TrackContext]:
    out = []
    for t in tracks:
        seg = lib.segments.get(t.track_id)
        layers = [derive(t, seg)]
        if t.track_id in interps:
            layers.append(interps[t.track_id])
        out.append(build_context(t, segments=seg, research=records.get(t.track_id), interpretations=layers,
                                 editorial=(editorial or {}).get(t.track_id), use_community_tags=use_community_tags))
    return out


def _provider(get_provider, name: str | None, model: str | None, base_url: str | None):
    """AI provider; a missing optional package (the anthropic SDK) is an environment problem
    (exit 3), a wrong name / model / URL a usage error (exit 2)."""
    from .ai import AIUnavailableError
    try:
        return get_provider(name, model=model, base_url=base_url)
    except AIUnavailableError as e:
        raise OptionalDependencyError(str(e)) from e


def _check_mode(name: str, value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# enrich (spec §3)
# ---------------------------------------------------------------------------

def run_enrich(ctx: RunContext, folder: Path, *, cache_arg: str | None = None, recursive: bool = False,
               metadata_path: str | None = None, mode: str = "online", research_cache: str | None = None,
               sources: tuple[str, ...] | None = None, contact: str | None = None,
               review_csv: str | None = None, report_json: str | None = None) -> int:
    from .research import (RESEARCH_CACHE_NAME, ResearchCache, default_client, default_sources, enrich_tracks,
                           export_review_csv, format_conflict, report_document)
    _check_mode("--mode", mode, DATA_MODES)
    folder = Path(folder).resolve()
    # None = the module's defaults: cache mode reads every source cached (Discogs included, no
    # token needed); online mode asks MusicBrainz + Wikidata, plus Discogs with a token
    sources = tuple(sources) if sources else None
    outputs = [Path(p).resolve() for p in (review_csv, report_json) if p]
    if len({path_key(p) for p in outputs}) != len(outputs):
        raise ValueError("--review-csv and --report must be different files")
    cache_path = _cache_path(folder, research_cache, RESEARCH_CACHE_NAME, "--research-cache", cache_arg)
    _guard_inputs(outputs, {"--research-cache": cache_path, "--metadata": metadata_path,
                            "analysis cache": _analysis_caches(folder, cache_arg)})
    lib = analyze_library(ctx, folder, cache_arg=cache_arg, recursive=recursive, metadata_path=metadata_path,
                          measure_loudness=False, exclude=outputs)
    assert_outputs_safe(outputs, lib.source_paths)
    cache = ResearchCache(cache_path, read_only=ctx.dry_run)
    if ctx.dry_run and mode == "online":
        ctx.out("[dry-run] network disabled: research runs in cache-only mode")
        mode = "cache"
    client = default_client(contact or os.environ.get("VEG_CONTACT")) if mode == "online" else None
    shown = ", ".join(sources) if sources else (", ".join(default_sources()) if mode == "online" else
                                                 "every cached source")
    ctx.out(f"Research: mode {mode}, sources {shown}, cache {cache_path}")
    result = enrich_tracks(lib.tracks, probes=lib.probes, mode=mode, cache=cache, client=client, sources=sources,
                           progress=_progress(ctx))
    if not ctx.dry_run:
        try:
            cache.save()
        except OSError as e:
            ctx.out(f"  WARNING: research cache {cache_path} could not be saved ({e})")
    by_id = {t.track_id: t for t in lib.tracks}
    ctx.out("")
    for tid, rec in sorted(result.records.items(), key=lambda kv: by_id[kv[0]].rel_path if kv[0] in by_id else kv[0]):
        t = by_id.get(tid)
        conf = f"{rec.match_confidence:.2f}" if rec.match_confidence is not None else "  - "
        facts = ", ".join(f"{k}={v.get('value')}" for k, v in list((rec.facts or {}).items())[:4])
        ctx.out(f"  {rec.status:<10} {conf}  {t.display_name if t else tid}" + (f"  [{facts}]" if facts else ""))
        for c in rec.conflicts or []:
            ctx.out(f"      conflict {format_conflict(c)}")
        for err in (rec.errors or [])[:2]:
            ctx.out(f"      error: {err}")
    st = result.stats
    ctx.out("")
    ctx.out("Research summary: " + ", ".join(f"{k} {v}" for k, v in st.items()))
    for w in result.warnings:
        ctx.out(f"  WARNING: {w}")
    ctx.out("Matched facts are stored with source URL, retrieval time and the fields used; ambiguous matches are "
            "NOT applied (review them with --review-csv). MusicBrainz tags/genres (user tags, CC BY-NC-SA) and "
            "Wikidata genres (crowd-edited) are community tags, ignored by planning unless --use-community-tags "
            "is given.")
    if ctx.dry_run:
        for p in outputs:
            ctx.out(f"[dry-run] would write {p}")
    else:
        if review_csv:
            export_review_csv(result, Path(review_csv))
            ctx.out(f"Review CSV -> {review_csv}")
        if report_json:
            # Discogs values carry their attribution and expire after 6 h (API terms): pruned here
            atomic_write_text(Path(report_json), json.dumps(report_document(result), indent=2,
                                                            ensure_ascii=False) + "\n")
            ctx.out(f"Research report -> {report_json}")
    if st.get("errors", 0) and not any(st.get(k, 0) for k in ("matched", "ambiguous", "not_found", "cache_hits")):
        ctx.out("ERROR: every lookup failed (network, authentication or rate limit — see the errors above)")
        return EXIT_RUNTIME
    attention = st.get("ambiguous", 0) + st.get("errors", 0)
    return EXIT_WARNING if attention else EXIT_OK


# ---------------------------------------------------------------------------
# interpret (spec §4)
# ---------------------------------------------------------------------------

def run_interpret(ctx: RunContext, folder: Path, *, cache_arg: str | None = None, recursive: bool = False,
                  metadata_path: str | None = None, mode: str = "online", provider: str | None = None,
                  model: str | None = None, base_url: str | None = None, ai_cache: str | None = None,
                  research_cache: str | None = None, research: str = "cache", segments: str = "cache",
                  use_community_tags: bool = False, report_json: str | None = None,
                  retry_invalid: bool = False) -> int:
    from .ai import AI_CACHE_NAME, PRIVACY_NOTE, AICache, get_provider, interpret_tracks
    _check_mode("--mode", mode, DATA_MODES)
    _check_mode("--research", research, ("off", "cache"))
    folder = Path(folder).resolve()
    outputs = [Path(report_json).resolve()] if report_json else []
    research_path = _cache_path(folder, research_cache, RESEARCH_CACHE_FALLBACK, "--research-cache", cache_arg)
    cache_path = _cache_path(folder, ai_cache, AI_CACHE_NAME, "--ai-cache", cache_arg)
    _guard_inputs(outputs, {"--ai-cache": cache_path, "--research-cache": research_path, "--metadata": metadata_path,
                            "analysis cache": _analysis_caches(folder, cache_arg)})
    lib = analyze_library(ctx, folder, cache_arg=cache_arg, recursive=recursive, metadata_path=metadata_path,
                          measure_loudness=False, exclude=outputs, segments=segments)
    assert_outputs_safe(outputs, lib.source_paths)
    records = _research_records(ctx, lib, research, research_path)
    cache = AICache(cache_path, read_only=ctx.dry_run)
    prov = None
    if mode == "online" and ctx.dry_run:
        ctx.out("[dry-run] no AI requests: cache-only mode")
        mode = "cache"
    if mode == "online":
        prov = _provider(get_provider, provider, model, base_url)
        ctx.out(f"AI provider: {prov.name} / {prov.model} ({PRIVACY_NOTE})")
    result = interpret_tracks(lib.tracks, provider=prov, mode=mode, cache=cache, segments=lib.segments,
                              research=records, use_community_tags=use_community_tags, progress=_progress(ctx),
                              retry_invalid=retry_invalid)
    if not ctx.dry_run:
        try:
            cache.save()
        except OSError as e:
            ctx.out(f"  WARNING: AI cache {cache_path} could not be saved ({e})")
    by_id = {t.track_id: t for t in lib.tracks}
    ctx.out("")
    for t in sorted(lib.tracks, key=lambda x: x.rel_path):
        rules = derive(t, lib.segments.get(t.track_id))
        ai = result.interpretations.get(t.track_id)
        r_roles = ", ".join(f"{r['role']} {r['confidence']:.2f}" for r in rules["roles"][:3]) or "-"
        line = f"  {t.display_name}\n      derived roles: {r_roles}"
        if ai:
            moods = ", ".join(f"{m['label']} {m['confidence']:.2f}" for m in ai.get("moods") or []) or "-"
            roles = ", ".join(f"{r['role']} {r['confidence']:.2f}" for r in ai.get("roles") or []) or "-"
            line += f"\n      AI ({ai.get('source')}): moods {moods}; roles {roles}"
            if ai.get("dropped"):
                line += f"; {len(ai['dropped'])} unsupported claim(s) dropped"
        ctx.out(line)
    ctx.out("")
    ctx.out("AI summary: " + ", ".join(f"{k} {v}" for k, v in result.stats.items()))
    for w in result.warnings:
        ctx.out(f"  WARNING: {w}")
    if report_json:
        if ctx.dry_run:
            ctx.out(f"[dry-run] would write {report_json}")
        else:
            atomic_write_text(Path(report_json), json.dumps(
                {"interpretations": result.interpretations, "derived": {t.track_id: derive(t, lib.segments.get(t.track_id))
                                                                       for t in lib.tracks},
                 "stats": result.stats, "warnings": result.warnings}, indent=2, ensure_ascii=False,
                default=str) + "\n")
            ctx.out(f"Interpretation report -> {report_json}")
    st = result.stats
    if st.get("errors") and not (st.get("valid") or st.get("cache_hits") or st.get("invalid")):
        ctx.out("ERROR: every AI request failed (provider, key or network — see the warnings above)")
        return EXIT_RUNTIME
    return EXIT_WARNING if st.get("errors") or st.get("invalid") else EXIT_OK


# ---------------------------------------------------------------------------
# suggest: an AI-drafted scenario (spec §5 "AI may propose the structure")
# ---------------------------------------------------------------------------

def run_suggest(ctx: RunContext, folder: Path, *, brief: str, output: str, cache_arg: str | None = None,
                recursive: bool = False, metadata_path: str | None = None, duration_min: float | None = None,
                provider: str | None = None, model: str | None = None, base_url: str | None = None) -> int:
    from .ai import get_provider
    from .scenario_suggest import library_summary, suggest_scenario, write_draft
    out = Path(output).resolve()
    if out.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"--output must be a .yaml file, got {out.name}")
    if duration_min is not None and duration_min <= 0:
        raise ValueError("--duration-min must be > 0")
    _guard_inputs([out], {"--metadata": metadata_path})
    lib = analyze_library(ctx, Path(folder), cache_arg=cache_arg, recursive=recursive, metadata_path=metadata_path,
                          measure_loudness=False, exclude=[out])
    assert_outputs_safe([out], lib.source_paths)
    summary = library_summary(lib.tracks)
    ctx.out("Library summary sent to the AI (no names, paths or audio): "
            + json.dumps(summary, ensure_ascii=False))
    if ctx.dry_run:
        ctx.out(f"[dry-run] no AI request; would write a draft to {out}")
        return EXIT_OK
    prov = _provider(get_provider, provider, model, base_url)
    repaired: list[str] = []
    draft, rationale, problems = suggest_scenario(prov, brief, summary, target_duration_min=duration_min,
                                                  warnings=repaired)
    for p in problems:
        ctx.out(f"  note: rejected attempt: {p}")
    for w in repaired:
        ctx.out(f"  note: draft repaired: {w}")
    write_draft(out, draft, rationale=rationale, provider_label=f"{prov.name}/{prov.model}", brief=brief)
    ctx.out(f"Scenario draft ({len(draft['phases'])} phases) -> {out}")
    if rationale:
        ctx.out(f"AI rationale: {rationale}")
    ctx.out("Review it, add anchors / required tracks, then run `plan --scenario` with it.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# plan / dry-run (spec §5–§8)
# ---------------------------------------------------------------------------

def run_plan(ctx: RunContext, folder: Path, *, scenario_path: str, profile_path: str | None = None,
             metadata_path: str | None = None, cache_arg: str | None = None, recursive: bool = False,
             seed: int | None = None, alternatives: int | None = None, target_duration_s: float | None = None,
             max_tracks: int | None = None, variant: str | None = None, research: str = "cache", ai: str = "cache",
             segments: str = "cache", research_cache: str | None = None, ai_cache: str | None = None,
             use_community_tags: bool = False, save_path: str | None = None, m3u_path: str | None = None,
             csv_path: str | None = None, report_path: str | None = None, reasons: bool = True) -> int:
    _check_mode("--research", research, ("off", "cache"))
    _check_mode("--ai", ai, ("off", "cache"))
    _check_mode("--segments", segments, ("off", "cache", "compute"))
    folder = Path(folder).resolve()
    scenario = with_overrides(load_scenario(Path(scenario_path)), target_duration_s=target_duration_s,
                              max_tracks=max_tracks)
    if alternatives is not None and not 1 <= alternatives <= len(PRESET_ORDER):
        raise ValueError(f"--alternatives must be 1..{len(PRESET_ORDER)}")
    seed = seed if seed is not None else (scenario.seed if scenario.seed is not None
                                          else default_seed("plan", scenario.name))
    profile = load_profile(Path(profile_path)) if profile_path else default_profile()
    outputs = [Path(p).resolve() for p in (save_path, m3u_path, csv_path, report_path) if p]
    if len({path_key(p) for p in outputs}) != len(outputs):
        raise ValueError("--save, --m3u, --csv and --report must be different files")
    research_path = _cache_path(folder, research_cache, RESEARCH_CACHE_FALLBACK, "--research-cache", cache_arg)
    ai_path = _cache_path(folder, ai_cache, AI_CACHE_FALLBACK, "--ai-cache", cache_arg)
    _guard_inputs(outputs, {"--scenario": scenario_path, "--metadata": metadata_path, "--profile": profile_path,
                            "--research-cache": research_path, "--ai-cache": ai_path,
                            "analysis cache": _analysis_caches(folder, cache_arg)})

    lib = analyze_library(ctx, folder, cache_arg=cache_arg, recursive=recursive, metadata_path=metadata_path,
                          measure_loudness=False, exclude=outputs, segments=segments)
    assert_outputs_safe(outputs, lib.source_paths)
    cur = curate(lib.tracks, profile)
    editorial = {d.track.track_id: d.editorial_score for d in cur.decisions}
    cur_excluded = {d.track.track_id: list(d.exclusion_reasons) for d in cur.excluded}
    ctx.out("")
    ctx.out(f"Curation ({profile.client}): {len(cur.eligible)} eligible, {len(cur.excluded)} excluded")
    resolved = resolve_scenario(scenario, cur.eligible, curation_excluded=cur_excluded, library=lib.tracks)
    for note in resolved.notes:
        ctx.out(f"  note: {note}")
    records = _research_records(ctx, lib, research, research_path)
    interps = _ai_interpretations(ctx, lib, ai, ai_path, records, use_community_tags)
    contexts = _contexts(list(resolved.pool), lib, records, interps, editorial, use_community_tags)
    ctx.out(f"Planning '{scenario.name}': {len(resolved.pool)} candidate track(s), seed {seed}, "
            f"{alternatives or scenario.alternatives} variant(s) ...")
    result = plan(contexts, resolved, seed=seed, alternatives=alternatives, seq_cfg=ctx.config.sequencing,
                  profile=profile)
    chosen = _pick(result, variant)
    ctx.out("")
    ctx.out(render_plan_text(result, chosen, reasons=reasons))
    if len(result.variants) > 1:
        ctx.out("")
        ctx.out(render_comparison(result))

    if outputs and ctx.dry_run:
        for p in outputs:
            ctx.out(f"[dry-run] would write {p}")
    elif outputs:
        used = {tid for v in result.variants for tid in v.order}
        tracks = {t.track_id: t for t in lib.tracks if t.track_id in used}
        if save_path:
            doc = plan_document(result, folder=folder, recursive=recursive, tracks=tracks, profile=profile.client,
                                profile_path=str(Path(profile_path).resolve()) if profile_path else None,
                                metadata_path=str(Path(metadata_path).resolve()) if metadata_path else None,
                                scenario_path=Path(scenario_path).resolve(),
                                settings={"research": research, "ai": ai, "segments": segments,
                                          "use_community_tags": use_community_tags,
                                          "cache": str(Path(cache_arg).resolve()) if cache_arg else None,
                                          "variant_shown": chosen.label})
            write_plan_json(Path(save_path), doc)
            ctx.out(f"Plan -> {save_path}")
        if m3u_path:
            write_plan_m3u8(Path(m3u_path), chosen, folder=folder,
                            title=f"VEmix plan - {result.scenario} - {chosen.label}")
            ctx.out(f"M3U8 (variant {chosen.label}, points at the library files) -> {m3u_path}")
        if csv_path:
            write_plan_csv(Path(csv_path), result.variants)
            ctx.out(f"CSV (all variants) -> {csv_path}")
        if report_path:
            text = render_plan_text(result, chosen, reasons=True)
            if len(result.variants) > 1:
                text += "\n\n" + render_comparison(result)
            atomic_write_text(Path(report_path), text + "\n")
            ctx.out(f"Report -> {report_path}")
    ctx.out("")
    ctx.out("Nothing was rendered: use `render --plan <file>` to export the chosen variant.")
    return EXIT_WARNING if chosen.violations else EXIT_OK


def _pick(result, label: str | None):
    if label is None:
        return result.variants[0]
    for v in result.variants:
        if v.label.upper() == label.strip().upper() or v.preset == label.strip().lower():
            return v
    raise ValueError(f"--variant {label!r} not planned (variants: "
                     + ", ".join(f"{v.label} ({v.preset})" for v in result.variants) + ")")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def run_compare(ctx: RunContext, plan_path: Path, *, variant: str | None = None, reasons: bool = False) -> int:
    """Comparison of the variants stored in a plan file (no audio, no FFmpeg needed)."""
    doc = load_plan_file(Path(plan_path))
    result = _result_view(doc["result"])
    ctx.out(render_comparison(result))
    if variant:
        v = next((x for x in result.variants if x.label.upper() == variant.upper() or x.preset == variant.lower()), None)
        if v is None:
            raise PlanFileError(f"variant {variant!r} not in the plan")
        ctx.out("")
        ctx.out(render_plan_text(result, v, reasons=reasons))
    return EXIT_OK


class _Obj:
    """Attribute view of a plan-file dict (render_* functions read attributes)."""

    def __init__(self, d: dict) -> None:
        self.__dict__.update(d)


def _result_view(d: dict) -> Any:
    variants = []
    for v in d.get("variants") or []:
        vv = dict(v)
        vv["entries"] = [_Obj(e) for e in v.get("entries") or []]
        vv["violations"] = [_Obj(x) for x in v.get("violations") or []]
        variants.append(_Obj(vv))
    r = dict(d)
    r["variants"] = variants
    return _Obj(r)


# ---------------------------------------------------------------------------
# render the chosen variant (spec §7; long work only on this explicit command)
# ---------------------------------------------------------------------------

def sequence_from_plan(tracks: list[TrackFeatures], variant: dict, *, ctx: RunContext, profile) -> SequenceResult:
    """SequenceResult in the PLANNED order (nothing is re-sequenced). Hard-constraint
    violations of the plan become sequencing flags (QC lists them); planned contrasts do not."""
    cfg, _ = effective_config(ctx.config.sequencing, profile)
    entries = {e.get("position"): e for e in variant.get("entries") or []}
    seq: list[SequencedTrack] = []
    for k, t in enumerate(tracks):
        e = entries.get(k + 1) or {}
        notes = [f"plan: phase {e.get('phase')}, target energy {e.get('target_energy')}, "
                 f"deviation {e.get('deviation')}" + (f", anchored ({e.get('anchored')})" if e.get("anchored") else "")]
        if (e.get("transition") or {}).get("planned_contrast"):
            notes.append("plan: planned contrast (scenario phase entry)")
        seq.append(SequencedTrack(position=k + 1, track=t,
                                  transition_in=transition_score(tracks[k - 1], t, cfg) if k else None, notes=notes))
    flagged = []
    for v in variant.get("violations") or []:
        pos = v.get("position")
        if v.get("kind") in TRANSITION_VIOLATIONS and isinstance(pos, int) and 2 <= pos <= len(tracks):
            flagged.append(FlaggedTransition(from_position=pos - 1, to_position=pos,
                                             from_track=tracks[pos - 2].display_name,
                                             to_track=tracks[pos - 1].display_name,
                                             reasons=[f"plan constraint not met: {v.get('message')}"],
                                             score=round(seq[pos - 1].transition_in.total, 4)
                                             if seq[pos - 1].transition_in else None))
    return SequenceResult(daypart=None, tracks=seq, flagged=flagged, not_selected=[],
                          total_duration_s=sum(t.duration_s for t in tracks), seed=0,
                          params={"scenario": {"variant": variant.get("label"), "preset": variant.get("preset")}})


def _plan_section(doc: dict, variant: dict, plan_path: Path) -> dict:
    return {
        "schema": doc.get("schema"), "file": str(Path(plan_path).resolve()),
        "scenario": (doc.get("scenario") or {}).get("name"), "scenario_sha256": (doc.get("scenario") or {}).get("sha256"),
        "variant": variant.get("label"), "preset": variant.get("preset"), "seed": (doc.get("result") or {}).get("seed"),
        "objective": variant.get("objective"),
        "order": [{"position": e.get("position"), "track_id": e.get("track_id"), "rel_path": e.get("rel_path"),
                   "display": e.get("display"), "phase": e.get("phase"), "target_energy": e.get("target_energy"),
                   "energy": e.get("energy"), "deviation": e.get("deviation"), "anchored": e.get("anchored"),
                   "planned_contrast": bool((e.get("transition") or {}).get("planned_contrast"))}
                  for e in variant.get("entries") or []],
        "violations": variant.get("violations") or [],
    }


def run_render(ctx: RunContext, plan_path: Path, *, output: str, mode: str = "playlist", variant: str | None = None,
               cache_arg: str | None = None, profile_path: str | None = None, metadata_path: str | None = None,
               seed: int | None = None) -> int:
    if mode not in ("playlist", "mix"):
        raise ValueError(f"--mode must be playlist or mix, got {mode!r}")
    plan_path = Path(plan_path)
    doc = load_plan_file(plan_path)
    v = pick_variant(doc, variant)
    lib_info = doc["library"]
    folder = Path(lib_info["folder"])
    if not folder.is_dir():
        raise FileNotFoundError(f"the plan's library folder {folder} does not exist")
    metadata_path = metadata_path or doc.get("metadata")
    profile_path = profile_path or (doc.get("profile") or {}).get("path")
    profile = load_profile(Path(profile_path)) if profile_path else default_profile()
    if cache_arg is None:
        planned_cache = (doc.get("settings") or {}).get("cache")
        if isinstance(planned_cache, str) and Path(planned_cache).is_absolute() and Path(planned_cache).is_file():
            cache_arg = planned_cache            # the analysis cache the plan was scored with
    out = Path(output).resolve()
    if mode == "playlist":
        if path_key(out) == path_key(folder.resolve()) or is_within(folder.resolve(), out / "tracks"):
            raise ValueError("--output must be a separate folder, not the music library (or a parent of it)")
        planned_outputs = [out / f for f in ("manifest.json", "playlist.m3u8", "tracklist.txt", "qc_report.json",
                                             "qc_report.txt")]
    else:
        if out.suffix.lower() != ".mp3":
            raise ValueError(f"--output must be an .mp3 file for --mode mix, got {out.name}")
        planned_outputs = [out, *mix_sidecars(out)]
    _guard_inputs(planned_outputs, {"--plan": plan_path, "--metadata": metadata_path, "--profile": profile_path,
                                    "analysis cache": _analysis_caches(folder, cache_arg)})

    lib = analyze_library(ctx, folder, cache_arg=cache_arg, recursive=bool(lib_info.get("recursive")),
                          metadata_path=metadata_path, measure_loudness=True)
    assert_outputs_safe(planned_outputs, lib.source_paths)
    by_rel = {t.rel_path: t for t in lib.tracks}
    tracks: list[TrackFeatures] = []
    stale: list[str] = []
    for tid in v.get("order") or []:
        info = lib_info["tracks"].get(tid) or {}
        rel = info.get("rel_path")
        t = by_rel.get(rel)
        if t is None:
            stale.append(f"{rel or tid}: not found (or not analysable) in {folder}")
            continue
        try:
            st = Path(t.path).stat()
            if info.get("size") is not None and (st.st_size != info["size"] or st.st_mtime_ns != info.get("mtime_ns")):
                stale.append(f"{rel}: changed since the plan was made")
        except OSError as e:
            stale.append(f"{rel}: {e}")
        tracks.append(t)
    if stale:
        raise PlanFileError("the plan is stale — re-run `plan`: " + "; ".join(stale[:8])
                            + (f" (+{len(stale) - 8} more)" if len(stale) > 8 else ""))
    seq = sequence_from_plan(tracks, v, ctx=ctx, profile=profile)
    section = _plan_section(doc, v, plan_path)
    excluded, not_selected = _render_exclusions(lib, profile, doc, v)
    planned_seed = (doc.get("result") or {}).get("seed")
    seed = seed if seed is not None else (int(planned_seed) if planned_seed is not None else default_seed(out.name))
    ctx.out("")
    ctx.out(f"=== Render plan '{section['scenario']}' variant {v.get('label')} ({v.get('preset')}) as {mode} -> {out} ===")
    for st in seq.tracks:
        e = section["order"][st.position - 1]
        ctx.out(f"  {st.position:>3}. [{e['phase']}] {st.track.display_name}")
    if mode == "playlist":
        if not ctx.dry_run:
            _sweep_stale(ctx, [out])
        try:
            res = export_playlist(seq, out, ctx.normalizer, profile_name=profile.client, excluded=excluded,
                                  input_measurements=lib.measurements, source_probes=lib.probes, jobs=ctx.jobs,
                                  dry_run=ctx.dry_run, progress=_progress(ctx), not_selected=not_selected,
                                  plan=section)
        except PlaylistExportError as e:
            ctx.out(f"  ERROR: playlist export failed (nothing published here): {e}")
            return EXIT_RUNTIME
        if ctx.dry_run:
            ctx.out(f"[dry-run] nothing written; {len(seq.tracks)} track(s) would be exported to {out}")
            return EXIT_OK
        if res.manifest_path is None:
            ctx.out(f"  ERROR: no playlist was published to {out} — every planned track failed normalization")
            return exit_code(OverallStatus.FAIL)
        report = qc_playlist(out, ctx.normalizer, profile=profile, qc_cfg=ctx.config.qc)
        json_path, txt_path = write_reports(report, out)
        ctx.out(summary_block(render_text_report(report)))
        ctx.out(f"Playlist: {res.m3u8_path}\nManifest: {res.manifest_path}\nQC: {json_path} | {txt_path}")
        return exit_code(report.status)
    if ctx.dry_run:
        ctx.out("[dry-run] nothing rendered")
        return EXIT_OK
    sel = Selection(profile=profile, lib=lib, sequences={None: seq}, excluded=excluded, editorial_scores={},
                    not_selected={None: not_selected})
    status = _render_one_mix(ctx, sel, seq, out, seed, plan=section)
    return exit_code(status)


def _render_exclusions(lib: LibraryAnalysis, profile, doc: dict, variant: dict) -> tuple[list[dict], list[dict]]:
    """Manifest lists for a rendered plan, as the 2.0 commands write them: tracks the client
    profile or the scenario excluded, files that failed analysis, and eligible tracks this
    variant did not place."""
    cur = curate(lib.tracks, profile)
    excluded = [{"source_rel": d.track.rel_path, "stage": "curation", "reasons": list(d.exclusion_reasons)}
                for d in cur.excluded]
    excluded += lib.broken_as_excluded
    seen = {e["source_rel"] for e in excluded}
    for x in (doc.get("result") or {}).get("excluded") or []:
        rel = x.get("rel_path")
        if rel and rel not in seen:
            excluded.append({"source_rel": rel, "stage": "scenario", "reasons": [str(x.get("reason") or "scenario")]})
            seen.add(rel)
    placed = set(variant.get("order") or [])
    not_selected = [{"source_rel": t.rel_path, "track_id": t.track_id,
                     "reason": f"not placed by plan variant {variant.get('label')}"}
                    for t in cur.eligible if t.track_id not in placed and t.rel_path not in seen]
    return excluded, not_selected


__all__ = ["run_enrich", "run_interpret", "run_suggest", "run_plan", "run_compare", "run_render",
           "sequence_from_plan", "DATA_MODES", "OptionalDependencyError"]
