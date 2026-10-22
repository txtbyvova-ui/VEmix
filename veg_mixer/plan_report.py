"""Plan reports and exports (VEmix 3.0, spec §7–§8).

* ``render_plan_text`` — the dry-run report: one row per position (number, phase, track,
  energy vs target and deviation, BPM, key, role, transition score, warnings), then the
  per-position reasons, hard-constraint violations, compromises and data provenance.
  Every number comes from ``dramaturgy`` (the evaluated objective), never from text
  written after the fact.
* ``render_comparison`` — side-by-side comparison of the variants under the scenario
  weights.
* ``write_plan_json`` / ``load_plan_file`` — the plan artifact (schema ``vegroove.plan``):
  the authoritative input of ``render`` (stored order, seed, weights, scenario hash,
  library root and per-track fingerprints so a stale plan is detected, not re-planned).
* ``write_plan_m3u8`` / ``write_plan_csv`` — exports of a variant that point at the
  ORIGINAL library files (relative paths when possible). They are planning artifacts, not
  deliverables: nothing is normalized or rendered.

All files are written atomically; nothing here touches audio.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from . import GENERATOR
from .dramaturgy import TERMS, PlanResult, PlanVariant
from .fsutil import atomic_write_text

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

PLAN_SCHEMA = "vegroove.plan"
PLAN_SCHEMA_VERSION = 1
TIMELINE_NOTE = ("relative time uses SOURCE durations; a rendered continuous mix is shorter by the crossfades "
                 "and trimmed silence, so phase boundaries shift slightly there")
CATEGORY_LABELS = {"measured": "measured", "file_tags": "file tags", "curator": "curator", "external": "external",
                   "derived": "derived (rules)", "ai": "AI", "unknown": "unknown"}
CSV_COLUMNS = ("variant", "position", "phase", "start_s", "t_rel", "artist", "title", "rel_path", "duration_s",
               "energy", "energy_source", "target_energy", "deviation", "bpm", "key", "roles", "moods",
               "transition_score", "bpm_diff", "key_steps", "energy_jump", "planned_contrast", "anchored",
               "violations", "warnings")


class PlanFileError(ValueError):
    """A plan file that is missing, not a VEmix plan, or does not match the library."""


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _mmss(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _f(v: float | None, fmt: str = ".2f", signed: bool = False) -> str:
    if v is None:
        return "?"
    return format(v, ("+" if signed else "") + fmt)


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def _role_cell(entry: Any) -> str:
    wanted = [r for r in entry.roles if r.get("wanted")]
    pick = wanted[0] if wanted else (entry.roles[0] if entry.roles else None)
    if not pick:
        return "?"
    tag = {"derived": "d", "ai": "ai", "curator": "c"}.get(pick.get("category"), "?")
    return f"{pick['role']}({tag} {pick['confidence']:.2f})" + ("" if wanted else "*")


def _warn_cell(entry: Any) -> str:
    items = []
    if entry.anchored:
        items.append(f"anchor {entry.anchored}")
    if entry.transition and entry.transition.get("planned_contrast"):
        items.append("planned contrast")
    items += [f"VIOLATION: {v}" for v in entry.violations]
    items += entry.warnings
    return "; ".join(items) or "-"


# ---------------------------------------------------------------------------
# text report
# ---------------------------------------------------------------------------

def render_plan_text(result: PlanResult, variant: PlanVariant | None = None, *, reasons: bool = True,
                     target_duration_s: float | None = None) -> str:
    v = variant or result.variants[0]
    out: list[str] = []
    target = target_duration_s if target_duration_s is not None else v.stats.get("target_duration_s")
    out.append(f"VEmix plan: {result.scenario} — variant {v.label} ({v.preset}: {v.preset_description})")
    head = (f"Planned: {_mmss(v.duration_s)} ({v.duration_s / 60:.1f} min), {len(v.order)} tracks"
            + (f" | target {target / 60:.1f} min" if target else "")
            + f" | objective {v.objective:.3f} | hard-constraint violations: {len(v.violations)}")
    out.append(head)
    out.append("Phases: " + " | ".join(
        f"{p['name']} {p['start'] * 100:.0f}-{p['end'] * 100:.0f}%"
        + (f" E {p['energy'][0]:.2f}" + (f"->{p['energy'][1]:.2f}" if p['energy'][1] != p['energy'][0] else "")
           if p.get("energy") else "")
        + (" [contrast]" if p.get("contrast") else "") for p in result.phases))
    out.append(f"Seed {result.seed}; candidates {result.candidates} of {result.pool_size} eligible; "
               f"{v.evaluations} objective evaluations")
    out.append("")
    hdr = (f"{'#':>3}  {'Phase':<12} {'Start':>7}  {'Track':<36} {'E':>5} {'Tgt':>5} {'Dev':>6} {'BPM':>5} "
           f"{'Key':<4} {'Role':<24} {'Trans':>5}  Warnings")
    out.append(hdr)
    out.append("-" * len(hdr))
    for e in v.entries:
        trans = f"{e.transition['score']:.2f}" if e.transition and e.transition.get("score") is not None else "  -"
        out.append(f"{e.position:>3}  {_clip(e.phase, 12):<12} {_mmss(e.start_s):>7}  {_clip(e.display, 36):<36} "
                   f"{_f(e.energy):>5} {_f(e.target_energy):>5} {_f(e.deviation, '.2f', True):>6} "
                   f"{(f'{e.bpm:.0f}' if e.bpm is not None else '?'):>5} {(e.camelot or '?'):<4} "
                   f"{_clip(_role_cell(e), 24):<24} {trans:>5}  {_warn_cell(e)}")
    out.append("")
    out.append("Role column: (d) derived from measurements, (ai) AI interpretation, (c) curator; "
               "'*' = the track has no role the phase asks for. Dev = track energy − curve target.")
    if v.violations:
        out.append("")
        out.append("Hard-constraint violations (not avoided by the search):")
        for vio in v.violations:
            where = f"#{vio.position} " if vio.position else ""
            out.append(f"  - {where}[{vio.kind}] {vio.message}")
    else:
        out.append("Hard constraints: all satisfied.")
    if reasons:
        out.append("")
        out.append("Why each track is here (from the evaluated objective):")
        for e in v.entries:
            out.append(f"  {e.position:>2}. {e.display}")
            for line in e.why:
                out.append(f"      - {line}")
            if e.counterfactual and e.counterfactual.get("term_changes"):
                ch = ", ".join(f"{k} {d:+.3f}" for k, d in sorted(e.counterfactual["term_changes"].items(),
                                                                  key=lambda kv: -abs(kv[1]))[:4])
                out.append(f"      - term changes of that alternative: {ch}")
    out.append("")
    out.append("Objective terms (weighted, scenario weights): "
               + ", ".join(f"{k} {v.terms.get(k, 0.0):.3f}" for k in TERMS if v.terms.get(k, 0.0)))
    st = v.stats
    out.append(f"Curve: mean |dev| {_f(st.get('curve_mean_abs_dev'), '.3f')}, max {_f(st.get('curve_max_abs_dev'), '.3f')}, "
               f"energy known for {st.get('curve_coverage', 0) * 100:.0f}% of tracks; transitions mean "
               f"{_f(st.get('transition_mean'), '.2f')}, min {_f(st.get('transition_min'), '.2f')}; "
               f"{st.get('artists')} artists")
    data = st.get("data") or {}
    out.append("Data used (feature values by category): "
               + ", ".join(f"{CATEGORY_LABELS[k]} {n}" for k, n in data.items() if n))
    inp = result.inputs
    out.append(f"Inputs: {inp.get('tracks')} tracks; segment analysis for {inp.get('with_segments')}, "
               f"AI interpretation for {inp.get('with_ai')}, verified external facts for "
               f"{inp.get('with_external_facts')}, curated mood for {inp.get('with_curated_mood')}, "
               f"unknown energy for {inp.get('energy_unknown')}")
    missing: dict[str, int] = {}
    for e in v.entries:
        for w in e.warnings:
            # group warnings that differ only in their numbers ("key change of 4 steps ...")
            key = _NUMBER_RE.sub("N", w.split(" — ")[0].split(", but ")[0])
            missing[key] = missing.get(key, 0) + 1
    if missing:
        out.append("Low-confidence / missing data in this plan: "
                   + "; ".join(f"{k} ({n})" for k, n in sorted(missing.items(), key=lambda kv: -kv[1])[:8]))
    if v.tradeoffs:
        out.append("Compromises: " + " | ".join(v.tradeoffs))
    for note in result.notes:
        out.append(f"Note: {note}")
    out.append(f"Note: {TIMELINE_NOTE}")
    return "\n".join(out)


def render_comparison(result: PlanResult) -> str:
    out = [f"VEmix plan comparison: {result.scenario} ({len(result.variants)} variant(s), scenario weights)", ""]
    hdr = (f"{'Var':<4} {'Preset':<9} {'Objective':>9} {'Viol':>4} {'Min':>6} {'Tracks':>6} {'Curve dev':>11} "
           f"{'Trans':>11} {'Artists':>7} {'Shared':>6}")
    out.append(hdr)
    out.append("-" * len(hdr))
    ref = set(result.variants[0].order) if result.variants else set()
    for v in result.variants:
        st = v.stats
        out.append(f"{v.label:<4} {v.preset:<9} {v.objective:>9.3f} {len(v.violations):>4} {v.duration_s / 60:>6.1f} "
                   f"{len(v.order):>6} {_f(st.get('curve_mean_abs_dev'), '.3f'):>5}/{_f(st.get('curve_max_abs_dev'), '.2f'):<5}"
                   f" {_f(st.get('transition_mean'), '.2f'):>5}/{_f(st.get('transition_min'), '.2f'):<5} "
                   f"{st.get('artists', 0):>7} {len(set(v.order) & ref):>6}")
    out.append("")
    names = [k for k in TERMS if any(v.terms.get(k, 0.0) for v in result.variants)]
    out.append("Weighted objective terms (lower is better):")
    out.append(f"  {'term':<12}" + "".join(f"{v.label:>9}" for v in result.variants))
    for k in names:
        out.append(f"  {k:<12}" + "".join(f"{v.terms.get(k, 0.0):>9.3f}" for v in result.variants))
    out.append("")
    for v in result.variants:
        out.append(f"{v.label} ({v.preset}): {v.preset_description}")
        for t in v.tradeoffs:
            out.append(f"   - {t}")
    best = min(result.variants, key=lambda x: (round(_severity(x), 6), x.objective)) if result.variants else None
    if best:
        out.append("")
        lowest = min(result.variants, key=lambda x: x.objective)
        if lowest is best:
            out.append(f"Best under the scenario weights: variant {best.label} ({best.preset}) — lowest objective"
                       + (" and no hard-constraint violation." if not best.violations else
                          f", hard-constraint severity {_severity(best):.2f}."))
        else:
            out.append(f"Best under the scenario weights: variant {best.label} ({best.preset}) — the least severe "
                       f"hard-constraint violations ({_severity(best):.2f}); variant {lowest.label} has the lowest "
                       f"objective but breaks more ({_severity(lowest):.2f}).")
        out.append("The other variants trade that score for the qualities listed above; choose by ear.")
    return "\n".join(out)


def _severity(v: Any) -> float:
    """Summed severity of a variant's hard-constraint violations (objects or plan-file dicts)."""
    total = 0.0
    for x in v.violations or []:
        sev = x.get("severity") if isinstance(x, dict) else getattr(x, "severity", None)
        total += float(sev) if isinstance(sev, (int, float)) else 1.0
    return total


# ---------------------------------------------------------------------------
# plan artifact
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def plan_document(result: PlanResult, *, folder: Path, recursive: bool, tracks: Mapping[str, Any],
                  profile: str | None, profile_path: str | None, metadata_path: str | None,
                  scenario_path: Path | None, settings: Mapping[str, Any] | None = None) -> dict:
    """The plan file content. ``tracks`` maps track_id -> TrackFeatures for every track the
    variants use (their fingerprints let ``render`` detect a stale plan)."""
    lib: dict[str, dict] = {}
    for tid, t in tracks.items():
        try:
            stat = Path(t.path).stat()
            fp = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            fp = {}
        lib[tid] = {"rel_path": t.rel_path, **fp}
    return {
        "schema": PLAN_SCHEMA, "schema_version": PLAN_SCHEMA_VERSION, "generator": GENERATOR,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "library": {"folder": str(Path(folder).resolve()), "recursive": bool(recursive), "tracks": lib},
        "scenario": {"name": result.scenario, "path": str(scenario_path) if scenario_path else None,
                     "sha256": file_sha256(scenario_path) if scenario_path else None},
        "profile": {"name": profile, "path": profile_path}, "metadata": metadata_path,
        "settings": dict(settings or {}),
        "result": result.to_dict(),
    }


def write_plan_json(path: Path, document: dict) -> None:
    atomic_write_text(Path(path), json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_plan_file(path: Path) -> dict:
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise PlanFileError(f"{path}: plan file not found") from None
    except (OSError, UnicodeDecodeError) as e:
        raise PlanFileError(f"{path}: cannot read plan: {e}") from e
    except ValueError as e:
        raise PlanFileError(f"{path}: not valid JSON: {e}") from None
    if not isinstance(doc, dict) or doc.get("schema") != PLAN_SCHEMA:
        raise PlanFileError(f"{path}: not a VEmix plan (schema={doc.get('schema') if isinstance(doc, dict) else None!r})")
    if doc.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise PlanFileError(f"{path}: unsupported plan schema_version {doc.get('schema_version')!r}")
    result = doc.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("variants"), list) or not result["variants"]:
        raise PlanFileError(f"{path}: plan has no variants")
    lib = doc.get("library")
    if not isinstance(lib, dict) or not isinstance(lib.get("tracks"), dict) or not lib.get("folder"):
        raise PlanFileError(f"{path}: plan has no library section")
    return doc


def pick_variant(doc: dict, label: str | None) -> dict:
    """The variant ``label`` (label or preset name); without one, the variant ``plan`` showed
    and exported (``settings.variant_shown``, so render matches the reviewed M3U), else A."""
    variants = doc["result"]["variants"]
    if label is None:
        shown = (doc.get("settings") or {}).get("variant_shown")
        return next((v for v in variants if isinstance(shown, str) and v.get("label") == shown), variants[0])
    for v in variants:
        if str(v.get("label", "")).upper() == label.strip().upper() or v.get("preset") == label.strip().lower():
            return v
    raise PlanFileError(f"variant {label!r} not in the plan (available: "
                        + ", ".join(f"{v.get('label')} ({v.get('preset')})" for v in variants) + ")")


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def _entry_path(rel_path: str, folder: Path, out_dir: Path) -> str:
    src = (Path(folder) / rel_path).resolve()
    try:
        return Path(os.path.relpath(src, Path(out_dir).resolve())).as_posix()
    except ValueError:        # another drive on Windows
        return str(src)


def write_plan_m3u8(path: Path, variant: PlanVariant, *, folder: Path, title: str) -> None:
    """Extended M3U of a plan variant pointing at the original library files."""
    path = Path(path)
    lines = ["#EXTM3U", f"#PLAYLIST:{' '.join(title.split())}"]
    for e in variant.entries:
        label = " ".join(e.display.replace(",", " ").split())
        lines.append(f"#EXTINF:{int(round(e.duration_s))},{label}")
        lines.append(_entry_path(e.rel_path, folder, path.parent))
    atomic_write_text(path, "\n".join(lines) + "\n")


def plan_csv_rows(variant: PlanVariant) -> list[dict]:
    rows = []
    for e in variant.entries:
        tr = e.transition or {}
        rows.append({
            "variant": variant.label, "position": e.position, "phase": e.phase, "start_s": e.start_s, "t_rel": e.t_rel,
            "artist": e.artist or "", "title": e.title or "", "rel_path": e.rel_path, "duration_s": e.duration_s,
            "energy": "" if e.energy is None else e.energy, "energy_source": e.energy_source or "",
            "target_energy": "" if e.target_energy is None else e.target_energy,
            "deviation": "" if e.deviation is None else e.deviation, "bpm": "" if e.bpm is None else e.bpm,
            "key": e.camelot or "",
            "roles": ";".join(f"{r['role']}:{r['confidence']}:{r['category']}" for r in e.roles),
            "moods": ";".join(f"{m['label']}:{m['confidence']}:{m['category']}" for m in e.moods),
            "transition_score": tr.get("score", ""), "bpm_diff": "" if tr.get("bpm_diff") is None else tr["bpm_diff"],
            "key_steps": "" if tr.get("key_steps") is None else tr["key_steps"],
            "energy_jump": "" if tr.get("energy_jump") is None else tr["energy_jump"],
            "planned_contrast": "true" if tr.get("planned_contrast") else "",
            "anchored": e.anchored or "", "violations": " | ".join(e.violations), "warnings": " | ".join(e.warnings),
        })
    return rows


_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _csv_cell(value: Any) -> Any:
    """Text that a spreadsheet would run as a formula (artist / title / file names come from
    tags and file names, i.e. third-party data) is prefixed with an apostrophe."""
    if isinstance(value, str) and value.startswith(_FORMULA_START):
        return "'" + value
    return value


def write_plan_csv(path: Path, variants: list[PlanVariant]) -> None:
    """UTF-8 CSV with BOM (opens in Excel), one row per planned position of each variant."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator="\r\n")
    w.writeheader()
    for v in variants:
        for row in plan_csv_rows(v):
            w.writerow({k: _csv_cell(x) for k, x in row.items()})
    atomic_write_text(Path(path), "﻿" + buf.getvalue(), newline="")


__all__ = ["PLAN_SCHEMA", "PLAN_SCHEMA_VERSION", "PlanFileError", "render_plan_text", "render_comparison",
           "plan_document", "write_plan_json", "load_plan_file", "pick_variant", "write_plan_m3u8",
           "write_plan_csv", "plan_csv_rows", "CSV_COLUMNS", "TIMELINE_NOTE"]
