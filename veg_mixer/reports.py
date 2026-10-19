"""P1 reporting (spec §11): machine-readable JSON and human-readable text QC reports.

The text report starts with the fixed summary block required by the spec; the
same numbers are kept machine-readably in ``QCReport.summary`` (both are produced
from ``compute_summary`` so they cannot disagree). Files are written atomically
(temp file in the same folder + ``os.replace``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from pathlib import Path

from .models import OverallStatus, QCIssue, QCReport, Severity, TrackQC

log = logging.getLogger(__name__)

SUMMARY_TITLE = "VEgroove Audio QC"
NA = "n/a"
# INFO issue code of an eligible track the sequencer did not place (manifest.not_selected).
NOT_SELECTED_CODE = "not_selected"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_hms(seconds: float | None) -> str:
    """Seconds -> ``H:MM:SS`` (floored; the tracklist convention). None -> ``n/a``."""
    if seconds is None or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return NA
    s = int(math.floor(max(0.0, float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _num(v: float | None, digits: int = 1, unit: str = "") -> str:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return NA
    text = f"{v:.{digits}f}"
    return f"{text} {unit}" if unit else text


def _ts(v: float | None) -> str:
    return "" if v is None else f"{format_hms(v)} ({v:.1f} s)"


# ---------------------------------------------------------------------------
# Summary (single source for JSON summary and text summary)
# ---------------------------------------------------------------------------

def is_energy_jump_flag(f: dict) -> bool:
    """A sequencer-flagged transition that reports an energy JUMP (the §11
    "Suspicious energy transitions" count). Coverage notes such as "unknown energy on
    both sides" are flagged transitions too, but not energy jumps."""
    if (f.get("kind") or "sequencing") != "sequencing":
        return False
    return any("energy jump" in str(r).casefold() for r in (f.get("reasons") or []))


def excluded_stage(entry: dict) -> str:
    """Stage of a ``manifest.excluded`` entry: ``curation`` | ``analysis`` |
    ``normalization`` (or another explicit value). A failed loudness status always
    means ``normalization``; entries of older manifests without ``stage`` are
    recognised by their reason prefix, else they are curation exclusions."""
    if not isinstance(entry, dict):
        return "curation"
    loud = entry.get("loudness")
    if isinstance(loud, dict) and str(loud.get("normalization_status") or "").strip().casefold() == "failed":
        return "normalization"
    stage = str(entry.get("stage") or "").strip().casefold()
    if stage:
        return stage
    reasons = entry.get("reasons")
    reasons = [reasons] if isinstance(reasons, str) else (reasons or [])
    text = [str(r).strip().casefold() for r in reasons]
    if any(r.startswith("normalization failed") for r in text):
        return "normalization"
    if any(r.startswith("analysis failed") for r in text):
        return "analysis"
    return "curation"


def _finite(values) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def compute_summary(report: QCReport) -> dict:
    """The §11 summary numbers. LUFS/TP figures are QC's own measurements of the
    delivered audio (per-track files; for a mix: the whole mix plus each track's
    body inside it) — never the export's self-reported values. Tracks that failed
    normalization are failed track rows ("Tracks failed"), not "Excluded tracks"
    (= curation + analysis exclusions)."""
    lufs = _finite(t.final_lufs for t in report.tracks)
    tps = _finite(t.final_true_peak_dbtp for t in report.tracks)
    if report.mix:
        lufs += _finite([report.mix.get("integrated_lufs")])
        tps += _finite([report.mix.get("true_peak_dbtp")])
    by_sev = {s.value: 0 for s in Severity}
    for i in report.issues:
        by_sev[Severity(i.severity).value] += 1
    silence_warnings = sum(1 for g in report.silence_warnings if g.get("suspicious"))
    energy_jumps = sum(1 for f in report.flagged_transitions if is_energy_jump_flag(f))
    # A category whose check did not run in this mode (silence in playlist/normalize mode,
    # metadata/sequencing in normalize mode) is reported as "not checked", never as 0.
    ran = set(report.checks_run)
    not_checked = [name for name, check, value in (
        ("unknown_metadata", "metadata_coverage", report.unknown_metadata),
        ("suspicious_energy_transitions", "flagged_transitions", energy_jumps),
        ("silence_warnings", "silence", len(report.silence_warnings)),
    ) if check not in ran and not value]
    return {
        "mode": report.mode,
        "tracks_processed": len(report.tracks),
        "tracks_passed": sum(1 for t in report.tracks if t.status == "passed"),
        "tracks_failed": sum(1 for t in report.tracks if t.status == "failed"),
        "tracks_requiring_review": sum(1 for t in report.tracks if t.status == "review"),
        "target_lufs": report.target_lufs,
        "target_true_peak_dbtp": report.target_true_peak_dbtp,
        "measured_lufs_min": round(min(lufs), 2) if lufs else None,
        "measured_lufs_max": round(max(lufs), 2) if lufs else None,
        "max_true_peak_dbtp": round(max(tps), 2) if tps else None,
        "excluded_tracks": sum(1 for e in report.excluded if excluded_stage(e) != "normalization"),
        "not_selected": sum(1 for i in report.issues if i.code == NOT_SELECTED_CODE),
        "unknown_metadata": None if "unknown_metadata" in not_checked else report.unknown_metadata,
        "suspicious_energy_transitions": None if "suspicious_energy_transitions" in not_checked else energy_jumps,
        "flagged_transitions": len(report.flagged_transitions),
        "silence_warnings": None if "silence_warnings" in not_checked else silence_warnings,
        "not_checked": not_checked,
        "silence_info": len(report.silence_warnings) - silence_warnings,
        "issues_by_severity": by_sev,
        "checks_run": len(report.checks_run),
        "checks_skipped": len(report.checks_skipped),
        "overall_status": OverallStatus(report.status).value,
    }


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _count(value, s: dict) -> str:
    return f"n/a (not checked in {s.get('mode') or 'this'} mode)" if value is None else str(value)


def _summary_block(s: dict) -> list[str]:
    lo, hi = s.get("measured_lufs_min"), s.get("measured_lufs_max")
    lufs_range = f"{_num(lo)} .. {_num(hi)} LUFS" if lo is not None and hi is not None else NA
    tp = s.get("max_true_peak_dbtp")
    return [
        SUMMARY_TITLE,
        "",
        f"Tracks processed: {s['tracks_processed']}",
        f"Tracks passed: {s['tracks_passed']}",
        f"Tracks failed: {s['tracks_failed']}",
        f"Tracks requiring review: {s['tracks_requiring_review']}",
        "",
        f"Target LUFS: {_num(s.get('target_lufs'))}",
        f"Measured LUFS range: {lufs_range}",
        f"Maximum measured True Peak: {_num(tp, 2, 'dBTP')}",
        "",
        f"Excluded tracks: {s['excluded_tracks']}",
        f"Unknown metadata: {_count(s.get('unknown_metadata'), s)}",
        f"Suspicious energy transitions: {_count(s.get('suspicious_energy_transitions'), s)}",
        f"Silence warnings: {_count(s.get('silence_warnings'), s)}",
        "",
        f"Overall status: {s['overall_status']}",
    ]


def _clip(text: str | None, width: int) -> str:
    text = text or ""
    return text if len(text) <= width else text[: width - 1] + "…"


def _tracks_section(tracks: list[TrackQC]) -> list[str]:
    """Per-track table; position 0 = a track that never reached the output (failed
    normalization), shown with its source path."""
    head = (f"{'#':>4}  {'File':<44} {'Orig LUFS':>9} {'Final LUFS':>10} {'Orig TP':>8} "
            f"{'Final TP':>8} {'Dur s':>7}  Status")
    out = ["Tracks", "-" * len(head), head]
    for t in tracks:
        pos = str(t.position) if t.position > 0 else "-"
        name = t.file if t.position > 0 or t.file else f"(not published) {t.source_rel or '?'}"
        out.append(f"{pos:>4}  {_clip(name, 44):<44} {_num(t.original_lufs):>9} {_num(t.final_lufs):>10} "
                   f"{_num(t.original_true_peak_dbtp, 2):>8} {_num(t.final_true_peak_dbtp, 2):>8} "
                   f"{_num(t.duration_s):>7}  {t.status}")
        for w in t.warnings:
            out.append(f"{'':>6}- {w}")
    if not tracks:
        out.append("  (no tracks)")
    return out


def _issues_section(issues: list[QCIssue]) -> list[str]:
    """All issues, most severe first (``not_selected`` notes have their own section)."""
    out = ["Issues", "------"]
    order = {Severity.FAIL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    issues = [i for i in issues if i.code != NOT_SELECTED_CODE]
    for i in sorted(issues, key=lambda x: (order[Severity(x.severity)],
                                          x.timestamp_s if x.timestamp_s is not None else -1.0)):
        where = []
        if i.track_position is not None:
            where.append(f"#{i.track_position}")
        if i.timestamp_s is not None:
            where.append(_ts(i.timestamp_s))
        loc = f" [{', '.join(where)}]" if where else ""
        out.append(f"{Severity(i.severity).value:<7} {i.code}{loc}: {i.message}")
    if not issues:
        out.append("  (none)")
    return out


def _flagged_section(flagged: list[dict]) -> list[str]:
    out = ["Flagged transitions", "-------------------"]
    for f in flagged:
        ts = f.get("timestamp_s")
        when = f" at {_ts(ts)}" if isinstance(ts, (int, float)) else ""
        reasons = "; ".join(str(r) for r in (f.get("reasons") or [])) or "no reason given"
        out.append(f"#{f.get('from_index')} -> #{f.get('to_index')} [{f.get('kind', 'sequencing')}, "
                   f"{f.get('source', 'manifest')}]{when}: {reasons}")
    if not flagged:
        out.append("  (none)")
    return out


def _silence_section(gaps: list[dict]) -> list[str]:
    out = ["Silence", "-------"]
    for g in gaps:
        pos = g.get("positions")
        tracks = f" track {', '.join('#' + str(p) for p in pos)}" if pos else ""
        out.append(f"{g.get('severity', 'INFO'):<7} {format_hms(g.get('start_s'))} - {format_hms(g.get('end_s'))} "
                   f"({_num(g.get('start_s'))}-{_num(g.get('end_s'))} s, {_num(g.get('duration_s'))} s) "
                   f"{g.get('classification')}{tracks}: {g.get('message', '')}")
    if not gaps:
        out.append("  (none)")
    return out


def _excluded_section(excluded: list[dict]) -> list[str]:
    """Curation / analysis exclusions (normalization failures are failed track rows)."""
    rows = [e for e in excluded if excluded_stage(e) != "normalization"]
    if not rows:
        return []
    out = ["Excluded tracks", "---------------"]
    for e in rows:
        reasons = e.get("reasons")
        reasons = [reasons] if isinstance(reasons, str) else (reasons or [])
        out.append(f"{e.get('source_rel')} [{excluded_stage(e)}]: "
                   f"{'; '.join(str(r) for r in reasons) or 'no reason given'}")
    return out + [""]


def _not_selected_section(issues: list[QCIssue]) -> list[str]:
    rows = [i for i in issues if i.code == NOT_SELECTED_CODE]
    if not rows:
        return []
    out = ["Not selected (eligible, not placed by the sequencer)",
           "----------------------------------------------------"]
    out += [i.message for i in rows]
    return out + [""]


def _mix_section(mix: dict) -> list[str]:
    out = ["Continuous mix", "--------------"]
    out.append(f"File: {mix.get('file')}  (manifest {mix.get('manifest')})")
    published, final = mix.get("published"), mix.get("final_normalization_status")
    if published is not None or final:
        delivered = published is not False and final != "failed"
        out.append(f"Published: {'yes' if delivered else 'NO'} (final normalization status: {final or NA})")
    out.append(f"Duration: decoded {_num(mix.get('decoded_duration_s'), 2, 's')} "
               f"({format_hms(mix.get('decoded_duration_s'))}), expected "
               f"{_num(mix.get('expected_duration_s'), 2, 's')}")
    out.append(f"Final mix loudness: {_num(mix.get('integrated_lufs'), 2, 'LUFS')}, true peak "
               f"{_num(mix.get('true_peak_dbtp'), 2, 'dBTP')}, LRA {_num(mix.get('lra_lu'), 1, 'LU')}, "
               f"sample peak {_num(mix.get('sample_peak_dbfs'), 2, 'dBFS')}")
    events = mix.get("clip_events") or []
    clip_txt = ", ".join(f"{format_hms(e.get('time_s'))} ({_num(e.get('time_s'), 2)} s)" for e in events[:10])
    out.append(f"Clipped samples (decoded MP3): {mix.get('clipped_samples', NA)}"
               + (f" — first at {clip_txt}" if clip_txt else ""))
    for seg in mix.get("segments") or []:
        val = (f"{_num(seg.get('integrated_lufs'), 1, 'LUFS')}, TP {_num(seg.get('true_peak_dbtp'), 2, 'dBTP')}"
               if seg.get("integrated_lufs") is not None else seg.get("skipped") or seg.get("error") or NA)
        out.append(f"  body #{seg.get('position')}: {format_hms(seg.get('start_s'))} - "
                   f"{format_hms(seg.get('end_s'))}: {val}")
    for w in mix.get("manifest_warnings") or []:
        out.append(f"Export warning: {w}")
    return out


def render_text_report(report: QCReport) -> str:
    """Human-readable report; starts with the spec §11 summary block."""
    s = compute_summary(report)
    lines = _summary_block(s)
    lines += ["", f"Mode: {report.mode}    Generated: {report.generated_at}",
              f"True-peak target: {_num(report.target_true_peak_dbtp, 1, 'dBTP')}", ""]
    if report.mix:
        lines += _mix_section(report.mix) + [""]
    lines += _tracks_section(report.tracks) + [""]
    lines += _issues_section(report.issues) + [""]
    lines += _flagged_section(report.flagged_transitions) + [""]
    lines += _silence_section(report.silence_warnings) + [""]
    lines += _excluded_section(report.excluded)
    lines += _not_selected_section(report.issues)
    lines += ["Checks run: " + (", ".join(report.checks_run) or "none"),
              "Checks skipped: " + ("; ".join(report.checks_skipped) or "none")]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Writing / reading
# ---------------------------------------------------------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:   # pragma: no cover - best effort
                log.warning("could not remove temporary file %s", tmp)


def write_reports(report: QCReport, output_dir: Path, *, stem: str = "qc_report") -> tuple[Path, Path]:
    """Write ``<stem>.json`` (QCReport.to_dict) and ``<stem>.txt`` atomically."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not report.summary:
        report.summary = compute_summary(report)
    json_path, txt_path = output_dir / f"{stem}.json", output_dir / f"{stem}.txt"
    _atomic_write_text(json_path, json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(txt_path, render_text_report(report))
    log.info("QC report written: %s, %s", json_path, txt_path)
    return json_path, txt_path


def qc_report_from_dict(d: dict) -> QCReport:
    """Inverse of ``QCReport.to_dict`` (for tooling that re-reads a JSON report)."""
    return QCReport(
        mode=d["mode"],
        generated_at=d["generated_at"],
        status=OverallStatus(d["status"]),
        target_lufs=d["target_lufs"],
        target_true_peak_dbtp=d["target_true_peak_dbtp"],
        tracks=[TrackQC(**t) for t in d.get("tracks") or []],
        issues=[QCIssue(**{**i, "severity": Severity(i["severity"])}) for i in d.get("issues") or []],
        checks_run=list(d.get("checks_run") or []),
        checks_skipped=list(d.get("checks_skipped") or []),
        excluded=list(d.get("excluded") or []),
        flagged_transitions=list(d.get("flagged_transitions") or []),
        silence_warnings=list(d.get("silence_warnings") or []),
        unknown_metadata=int(d.get("unknown_metadata") or 0),
        summary=dict(d.get("summary") or {}),
        mix=d.get("mix"),
    )
