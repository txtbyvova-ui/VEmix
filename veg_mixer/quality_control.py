"""P1 quality control for both output modes (spec §10) plus the adjacent-track
short-term loudness check of spec §3 "Additional QC".

QC is an INDEPENDENT validator: it never trusts the numbers written by the export
or the renderer. Every published audio file is probed, fully decoded and
re-measured here; manifests are only used to know what *should* be there.

Playlist mode (``qc_playlist``)::

    manifest.json + playlist.m3u8 (strict parser below) + tracks/*.mp3
    -> existence / containment / decode / duration / loudness / true peak per file
    -> order (m3u8 == manifest == 1..N), duplicates, excluded tracks, profile rules
    -> adjacent short-term loudness on the FRESH measurements, flagged transitions

Continuous mix (``qc_mix``)::

    <stem>.manifest.json -> mix MP3 measured as a whole (integrated, true peak,
    duration), one streaming PCM pass for silence + clipping, per-track body
    loudness inside the mix, transition timing and tracklist timestamps.

Both modes also read the manifest's bookkeeping: tracks that FAILED normalization
(``excluded[]`` with ``stage = "normalization"``) are FAIL issues and failed track rows
(position 0 = not in the output); undecodable library files (``stage = "analysis"``)
are warnings; ``not_selected[]`` is listed as INFO. Client-profile rules are checked
by re-running ``curation.curate`` on the exported features — never by a second copy
of the rules.

Severity policy: FAIL = must not ship, WARNING = needs listening review, INFO =
listed for completeness. ``overall_status`` never returns PASS when a critical
check was skipped. Every mix warning carries a timestamp (seconds + H:MM:SS).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import MISSING, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from .config import EncodeConfig, LoudnessConfig, QCConfig, TransitionConfig, apply_recorded_targets
from .curation import ProfileError, curate
from .ffmpeg_tools import (
    FFmpegError,
    FFmpegNotFoundError,
    FFmpegTools,
    LoudnessMeasurementError,
    build_measurement,
    loudness_filter,
    probe_audio,
    run_ffmpeg,
)
from .loudness import LoudnessNormalizer, compare_adjacent_short_term
from .models import (
    AudioProbe,
    ClientProfile,
    DecodeCheck,
    LoudnessMeasurement,
    ManifestTrack,
    MixTrackEntry,
    OverallStatus,
    PlaylistManifest,
    QCIssue,
    QCReport,
    Severity,
    TrackFeatures,
    TrackQC,
    TransitionReport,
)
from .fsutil import mp3_audio_sha256
from .reports import NOT_SELECTED_CODE, compute_summary, excluded_stage, format_hms, is_energy_jump_flag
from .textnorm import normalize_rel_path

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
M3U8_NAME = "playlist.m3u8"
TRACKLIST_NAME = "tracklist.txt"

# Mix timeline consistency tolerance (renderer contract: 10 ms).
TIMING_TOLERANCE_S = 0.010
# Last track end vs expected mix duration (serialized values are rounded).
MIX_END_TOLERANCE_S = 0.05
# #EXTINF durations are whole seconds; allowed slack on top of QC duration tolerance.
EXTINF_SLACK_S = 1.0
# A track body inside the mix may deviate from the target by the loudness tolerance
# plus this much before it is worth a listening check (bodies exclude crossfades,
# intros/outros make them legitimately differ from the full-track integrated value).
SEGMENT_EXTRA_TOLERANCE_LU = 2.0
# Track bodies shorter than this are not measured (integrated loudness needs > 0.4 s
# and short-term statistics need >= 3 s).
MIN_SEGMENT_S = 1.0
# Consecutive clipped frames closer than this belong to the same clipping event.
CLIP_EVENT_GAP_S = 0.05
MAX_CLIP_EVENTS = 10
# RMS floor so that digital zero stays a finite, JSON-safe number.
LEVEL_FLOOR_DBFS = -200.0
# Share of gap windows below the digital-silence level that makes a gap "digital".
DIGITAL_SHARE = 0.5
# Silence is located on RMS sub-windows of this length (gap edges ±50 ms) instead of the
# coarse cfg.silence_window_s windows: a gap that straddles coarse window borders would
# otherwise lose up to a whole window at each end (a 1.7 s gap measured as 1.0 s).
SILENCE_SUBWINDOW_S = 0.05
# Silent runs separated by a short non-silent interruption (a click, a transient) form
# ONE region: the interruption may last up to BRIDGE_WINDOWS x cfg.silence_window_s and
# each silent run next to it must last at least as long as the interruption itself
# (silence dominates). Only windows below the silence threshold are involved, so normal
# music is never merged into a "gap".
SILENCE_BRIDGE_WINDOWS = 1.0
# A non-digital quiet region NEAR (not touching) a boundary is a "transition gap" only
# when at least this share of it lies within cfg.transition_guard_s of the boundary.
TRANSITION_GAP_MIN_GUARD_SHARE = 0.5
# A region TOUCHING a boundary is dead air at the transition (WARNING) whatever its length,
# unless it covers at least this share of its own track's span (the time between the
# boundaries around it): then the track itself is quiet throughout (INFO quiet passage).
QUIET_TRACK_MIN_SHARE = 0.6

# Checks whose omission must prevent PASS. Skipped checks are recorded as
# "<name>: <reason>"; any name NOT listed in NON_CRITICAL_CHECKS counts as critical.
NON_CRITICAL_CHECKS = frozenset({
    "profile_rules", "adjacent_short_term", "flagged_transitions", "metadata_coverage",
    "orphan_files", "m3u8_durations", "tracklist_present", "segment_loudness",
    "peak_protection", "loudness_targets", "mix_file_integrity",
})

# Issues about a transition between two tracks (not about one track's audio).
TRANSITION_CODES = frozenset({
    "adjacent_loudness_jump", "suspicious_energy_transition", "flagged_transition",
    "peak_protection_applied", "uncertain_transition", "planned_contrast_transition",
})

# VEmix 3.0 plan violations about a PAIR of tracks: they reach the manifest as sequencer flags
# (plan_pipeline.TRANSITION_VIOLATIONS); every other kind is reported from manifest.plan itself.
PLAN_TRANSITION_VIOLATIONS = frozenset({"forbidden_pair", "tempo_jump", "key_jump", "energy_jump", "artist_gap"})
# plan.order[].anchored: "position 1" / "position -1" (from the end) / "phase climax" (not positional)
_ANCHOR_POSITION_RE = re.compile(r"^position\s+([+-]?\d+)$", re.I)
# Renderer notes on an uncertain transition (transitions.plan_transition_detailed): the energy
# difference note vs a tempo / beat cause ("no beat/phrase alignment; tempo ... / BPM unknown").
_ENERGY_DIFFERENCE_FLAG_RE = re.compile(r"\benergy (?:difference|jump)\b", re.I)
_TEMPO_FLAG_RE = re.compile(r"\b(?:tempo|bpm|beats?|phrase)\b", re.I)

AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus")
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_EXTINF_RE = re.compile(r"^#EXTINF:(-?\d+(?:\.\d+)?),(.*)$")
_KNOWN_EXT_DIRECTIVES = ("#PLAYLIST:", "#EXTGRP:", "#EXTALB:", "#EXTART:", "#EXTGENRE:",
                         "#EXTIMG:", "#EXTENC:", "#EXTBYT:", "#EXTBIN:")
_TRACKLIST_ROW_RE = re.compile(r"^(\d+)\s+\|\s+(\d+:\d{2}:\d{2})\s+\|")
# curation exclusion reason -> stable QC issue code (first match wins; anything else is
# reported as "profile_excluded" with curation's own wording).
_CURATION_REASON_CODES = (
    (re.compile(r"^blocklisted\b", re.I), "blocklisted_track"),
    (re.compile(r"cyrillic", re.I), "excluded_script"),
    (re.compile(r"avoided preference", re.I), "avoided_preference"),
    (re.compile(r"allowlist\.only|not on the client allowlist", re.I), "not_allowlisted"),
    (re.compile(r"not in the allowed list", re.I), "language_not_allowed"),
    (re.compile(r"^languages?\b.*\bexcluded\b", re.I), "excluded_language"),
    (re.compile(r"approv", re.I), "not_approved"),
    (re.compile(r"^excluded in\b", re.I), "excluded_track_present"),
)


# ---------------------------------------------------------------------------
# Status logic (pure)
# ---------------------------------------------------------------------------

def skipped_check_name(entry: str) -> str:
    """Name part of a ``"<name>: <reason>"`` checks_skipped entry."""
    return entry.split(":", 1)[0].strip()


def is_critical_check(name: str) -> bool:
    return skipped_check_name(name) not in NON_CRITICAL_CHECKS


def overall_status(issues: list[QCIssue], checks_skipped: list[str]) -> OverallStatus:
    """FAIL if any FAIL issue; WARNING if any WARNING issue or any skipped critical
    check (never PASS when critical checks were skipped); otherwise PASS."""
    severities = {Severity(i.severity) for i in issues}
    if Severity.FAIL in severities:
        return OverallStatus.FAIL
    if Severity.WARNING in severities or any(is_critical_check(s) for s in checks_skipped):
        return OverallStatus.WARNING
    return OverallStatus.PASS


# ---------------------------------------------------------------------------
# Strict M3U8 parser (independent of playlist.py on purpose)
# ---------------------------------------------------------------------------

@dataclass
class M3U8Entry:
    line_no: int
    duration_s: float | None     # None = -1 (unknown) in #EXTINF
    title: str | None
    uri: str


@dataclass
class M3U8Parse:
    entries: list[M3U8Entry] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)      # syntax violations -> FAIL
    warnings: list[str] = field(default_factory=list)    # tolerated oddities -> WARNING
    playlist_title: str | None = None


def check_relative_uri(uri: str) -> str | None:
    """Reason why ``uri`` is not an acceptable relative POSIX path, else None."""
    if not uri:
        return "empty path"
    if any(ord(c) < 32 or ord(c) == 127 for c in uri):
        return "path contains control characters"
    if "\\" in uri:
        return "path uses a backslash (playlists must use POSIX '/' separators)"
    if uri.startswith("/") or PureWindowsPath(uri).drive or PurePosixPath(uri).is_absolute():
        return "absolute path (playlists must use paths relative to the playlist)"
    if _SCHEME_RE.match(uri):
        return "URL/scheme instead of a relative file path"
    return None


def parse_m3u8(path: Path) -> M3U8Parse:
    """Strict extended-M3U (UTF-8) parser.

    Rules: UTF-8 text; the first line is exactly ``#EXTM3U``; every entry is an
    ``#EXTINF:<duration>,<title>`` line immediately followed (blank lines and
    ``#`` comments aside) by a relative POSIX path; no dangling ``#EXTINF``; no
    URI without ``#EXTINF``; at least one entry.
    """
    res = M3U8Parse()
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        res.errors.append(f"playlist is not valid UTF-8 ({e.reason} at byte {e.start})")
        text = raw.decode("utf-8", errors="replace")
    if text.startswith("﻿"):
        res.warnings.append("playlist starts with a UTF-8 byte-order mark (some players reject it)")
        text = text[1:]
    lines = text.splitlines()
    if not lines or lines[0].rstrip() != "#EXTM3U":
        res.errors.append("line 1: the playlist must start with '#EXTM3U'")

    pending: tuple[int, float | None, str] | None = None
    for no, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        if no == 1 and stripped == "#EXTM3U":
            continue
        if not stripped:
            continue
        if stripped == "#EXTM3U":
            res.errors.append(f"line {no}: duplicate '#EXTM3U' header")
            continue
        if stripped.startswith("#EXTINF"):
            if pending is not None:
                res.errors.append(f"line {pending[0]}: #EXTINF is not followed by a file path")
            m = _EXTINF_RE.match(stripped)
            if not m:
                res.errors.append(f"line {no}: malformed #EXTINF (expected '#EXTINF:<seconds>,<title>')")
                pending = (no, None, "")
                continue
            duration = float(m.group(1))
            title = m.group(2).strip()
            if duration < 0 and duration != -1:
                res.errors.append(f"line {no}: negative #EXTINF duration {m.group(1)}")
            if not title:
                res.errors.append(f"line {no}: #EXTINF has an empty title")
            pending = (no, None if duration == -1 else duration, title)
            continue
        if stripped.startswith("#"):
            if stripped.startswith("#PLAYLIST:"):
                res.playlist_title = stripped[len("#PLAYLIST:"):].strip() or None
            elif stripped.startswith("#EXT") and not stripped.startswith(_KNOWN_EXT_DIRECTIVES):
                res.warnings.append(f"line {no}: unknown directive {stripped.split(':', 1)[0]!r}")
            continue   # comment
        # --- URI line ---
        if line != stripped:
            res.errors.append(f"line {no}: path has leading/trailing whitespace")
        reason = check_relative_uri(stripped)
        if reason:
            res.errors.append(f"line {no}: {reason}: {stripped!r}")
        if pending is None:
            res.errors.append(f"line {no}: entry {stripped!r} has no preceding #EXTINF")
            res.entries.append(M3U8Entry(no, None, None, stripped))
        else:
            res.entries.append(M3U8Entry(no, pending[1], pending[2] or None, stripped))
            pending = None
    if pending is not None:
        res.errors.append(f"line {pending[0]}: #EXTINF is not followed by a file path")
    if not res.entries:
        res.errors.append("playlist has no entries")
    return res


# ---------------------------------------------------------------------------
# Streaming PCM scan (silence + clipping) — never loads the whole mix into RAM
# ---------------------------------------------------------------------------

@dataclass
class PcmScan:
    sample_rate: int
    channels: int
    window_s: float
    levels_dbfs: np.ndarray            # RMS per window (last window may be partial)
    decoded_duration_s: float
    peak_dbfs: float | None
    clipped_samples: int
    clip_events: list[dict] = field(default_factory=list)   # first MAX_CLIP_EVENTS events
    clip_event_count: int = 0                                # all events
    errors: list[str] = field(default_factory=list)
    # finer RMS grid used to locate silence (SILENCE_SUBWINDOW_S; last one may be partial)
    sub_window_s: float = 0.0
    sub_levels_dbfs: np.ndarray = field(default_factory=lambda: np.empty(0))

    def silence_levels(self) -> tuple[np.ndarray, float]:
        """(levels, window length) to run ``detect_silence_gaps`` on: the fine grid when
        available, else the coarse windows."""
        if len(self.sub_levels_dbfs) and self.sub_window_s > 0:
            return self.sub_levels_dbfs, self.sub_window_s
        return self.levels_dbfs, self.window_s


def _level_dbfs(mean_square: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        lv = 10.0 * np.log10(mean_square)
    return np.maximum(lv, LEVEL_FLOOR_DBFS)


def scan_pcm(tools: FFmpegTools, path: Path, cfg: QCConfig, *, sample_rate: int | None = None,
             channels: int = 2, chunk_s: float = 10.0, timeout_s: float | None = None) -> PcmScan:
    """Decode ``path`` with ffmpeg to f32le on stdout and process it chunk by chunk:
    RMS level per ``cfg.silence_window_s`` window and per ``SILENCE_SUBWINDOW_S``
    sub-window (silence edges), sample peak, and samples with
    ``|x| >= cfg.clip_sample_threshold`` grouped into timestamped clipping events.
    Memory use is bounded by ``chunk_s`` regardless of the file length."""
    path = Path(path)
    if sample_rate is None:
        sample_rate = probe_audio(tools, path).sample_rate or 44100
    sr = int(sample_rate)
    win = max(1, int(round(cfg.silence_window_s * sr)))
    window_s = win / sr
    sub = max(1, min(win, int(round(SILENCE_SUBWINDOW_S * sr))))
    frame_bytes = 4 * channels
    read_frames = win * max(1, int(round(chunk_s / window_s)))
    cmd = [tools.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-i", str(path),
           "-map", "0:a:0", "-vn", "-ac", str(channels), "-ar", str(sr),
           "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]

    mean_squares: list[np.ndarray] = []
    carry = np.empty((0, channels), dtype=np.float64)
    sub_mean_squares: list[np.ndarray] = []
    sub_carry = np.empty((0, channels), dtype=np.float64)
    leftover = b""
    total_frames = 0
    peak = 0.0
    clipped = 0
    events: list[dict] = []
    n_events = 0                     # all clipping events, also those beyond MAX_CLIP_EVENTS
    last_clip_frame = -10**18
    gap_frames = int(CLIP_EVENT_GAP_S * sr)
    errors: list[str] = []

    with tempfile.TemporaryFile() as err_file:   # stderr to a file: no pipe deadlock
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file, bufsize=1 << 20)
        except FileNotFoundError as e:
            raise FFmpegNotFoundError(str(e)) from e
        try:
            assert proc.stdout is not None
            while True:
                buf = proc.stdout.read(read_frames * frame_bytes)
                if not buf:
                    break
                buf = leftover + buf
                usable = len(buf) - (len(buf) % frame_bytes)
                leftover = buf[usable:]
                if not usable:
                    continue
                x = np.frombuffer(buf[:usable], dtype="<f4").reshape(-1, channels).astype(np.float64)
                ax = np.abs(x)
                peak = max(peak, float(ax.max()))
                hit = ax >= cfg.clip_sample_threshold
                per_frame = np.count_nonzero(hit, axis=1)
                local = np.flatnonzero(per_frame)
                if local.size:
                    clipped += int(per_frame.sum())
                    g = local + total_frames                       # global frame numbers
                    new_event = np.diff(np.concatenate(([last_clip_frame], g))) > gap_frames
                    event_id = n_events - 1 + np.cumsum(new_event)
                    for k in np.flatnonzero(new_event):
                        if event_id[k] < MAX_CLIP_EVENTS:
                            t0 = g[k] / sr
                            events.append({"time_s": round(float(t0), 3), "time_hms": format_hms(t0),
                                           "end_s": round(float(t0), 3), "samples": 0})
                    for eid in np.unique(event_id[event_id < MAX_CLIP_EVENTS]):
                        sel = event_id == eid
                        events[eid]["samples"] += int(per_frame[local[sel]].sum())
                        events[eid]["end_s"] = round(float(g[sel][-1] / sr), 3)
                    n_events += int(new_event.sum())
                    last_clip_frame = int(g[-1])
                total_frames += len(x)
                carry = _accumulate_windows(carry, x, win, channels, mean_squares)
                sub_carry = _accumulate_windows(sub_carry, x, sub, channels, sub_mean_squares)
            proc.stdout.close()
            rc = proc.wait(timeout=timeout_s)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        err_file.seek(0)
        err_text = err_file.read().decode("utf-8", errors="replace")
    errors.extend(ln.strip() for ln in err_text.splitlines() if ln.strip())
    if rc != 0 and not errors:
        errors.append(f"ffmpeg exited with code {rc}")
    for rest, store in ((carry, mean_squares), (sub_carry, sub_mean_squares)):
        if len(rest):
            store.append(np.array([float(np.mean(rest * rest))]))
    ms = np.concatenate(mean_squares) if mean_squares else np.empty(0)
    sub_ms = np.concatenate(sub_mean_squares) if sub_mean_squares else np.empty(0)
    return PcmScan(
        sample_rate=sr, channels=channels, window_s=window_s, levels_dbfs=_level_dbfs(ms),
        decoded_duration_s=total_frames / sr,
        peak_dbfs=round(20 * math.log10(peak), 2) if peak > 0 else None,
        clipped_samples=clipped, clip_events=events, clip_event_count=n_events, errors=errors[:20],
        sub_window_s=sub / sr, sub_levels_dbfs=_level_dbfs(sub_ms))


def _accumulate_windows(carry: np.ndarray, x: np.ndarray, size: int, channels: int,
                        out: list[np.ndarray]) -> np.ndarray:
    """Append the mean square of every complete ``size``-frame window of carry + x to
    ``out``; return the incomplete remainder (carried into the next chunk)."""
    data = np.concatenate([carry, x]) if carry.size else x
    n_full = len(data) // size
    if n_full:
        block = data[: n_full * size].reshape(n_full, size * channels)
        out.append(np.mean(block * block, axis=1))
    return data[n_full * size:]


def detect_silence_gaps(levels_dbfs: np.ndarray, window_s: float, cfg: QCConfig, *,
                        total_duration_s: float | None = None, boundaries_s: list[float] = (),
                        fade_in_s: float = 0.0, fade_out_s: float = 0.0) -> list[dict]:
    """Pure gap finder/classifier on per-window RMS levels (any window length; the
    mix QC passes the fine ``SILENCE_SUBWINDOW_S`` grid so gap edges are accurate).

    Windows below ``cfg.silence_threshold_dbfs`` form silent runs. Runs separated by a
    short non-silent interruption (up to ``SILENCE_BRIDGE_WINDOWS`` x
    ``cfg.silence_window_s``, e.g. a click in a quiet bed) are merged into one region when
    each run is at least as long as the interruption — one quiet passage is one gap.
    Regions lasting at least ``cfg.silence_min_duration_s`` become gaps, classified
    (first match wins):

    * ``edge_fade``       — at the very start/end, inside the planned fade-in/out (INFO)
    * ``digital_silence`` — mostly below ``cfg.digital_silence_dbfs`` (WARNING)
    * ``transition_gap``  — touching a boundary (dead air at the transition, however long)
      unless it covers ``QUIET_TRACK_MIN_SHARE`` of its own track's span, or near a
      boundary and mostly (``TRANSITION_GAP_MIN_GUARD_SHARE``) inside the guard zone (WARNING)
    * ``quiet_passage``   — not digital silence, inside a track body (or reaching far
      into it from a boundary): likely intentional (INFO)

    Every gap also carries ``digital_silence`` / ``near_boundary`` flags so that a
    digital gap at a boundary is visible as both.
    """
    levels = np.asarray(levels_dbfs, dtype=np.float64)
    n = len(levels)
    if n == 0:
        return []
    total = float(total_duration_s) if total_duration_s is not None else n * window_s
    silent = levels < cfg.silence_threshold_dbfs
    gaps: list[dict] = []
    for i, j, runs in _silent_regions(silent, window_s, cfg):
        start, end = i * window_s, min(j * window_s, total)
        duration = end - start
        if duration + 1e-9 >= cfg.silence_min_duration_s:
            gap = _classify_gap(levels[i:j], start, end, i == 0, j == n, total, cfg,
                                boundaries_s, fade_in_s, fade_out_s, window_s)
            gap["silent_share"] = round(float(np.mean(silent[i:j])), 2)
            gap["merged_runs"] = runs
            gaps.append(gap)
    return gaps


def _silent_regions(silent: np.ndarray, window_s: float, cfg: QCConfig) -> list[tuple[int, int, int]]:
    """[(first window, end window (exclusive), number of merged silent runs)]."""
    edges = np.diff(np.concatenate(([0], silent.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    bridge = max(1, int(round(SILENCE_BRIDGE_WINDOWS * cfg.silence_window_s / window_s)))
    regions: list[list[int]] = []          # [start, end, runs, length of the last run]
    for i, j in zip(starts.tolist(), ends.tolist()):
        if regions:
            last, cur, interruption = regions[-1][3], j - i, i - regions[-1][1]
            if interruption <= bridge and min(last, cur) >= interruption:
                regions[-1][1:] = [j, regions[-1][2] + 1, cur]
                continue
        regions.append([i, j, 1, j - i])
    return [(i, j, runs) for i, j, runs, _ in regions]


def _guard_share(start: float, end: float, boundaries_s, guard_s: float) -> float:
    """Share of [start, end] lying within ``guard_s`` of any boundary."""
    if end <= start:
        return 1.0
    zones = sorted((float(b) - guard_s, float(b) + guard_s) for b in boundaries_s or ())
    covered, cur = 0.0, None
    for lo, hi in zones + [(math.inf, math.inf)]:
        if cur is not None and lo <= cur[1]:
            cur = (cur[0], max(cur[1], hi))
            continue
        if cur is not None:
            covered += max(0.0, min(end, cur[1]) - max(start, cur[0]))
        cur = (lo, hi)
    return covered / (end - start)


def _host_track_share(start: float, end: float, boundaries_s, total: float) -> float:
    """Share of the region's host track span (between the boundaries around its middle)
    that the region covers: ~1 for a track that is quiet throughout, small for dead air
    at a track's head or tail."""
    mid = (start + end) / 2.0
    cuts = sorted({0.0, float(total), *(float(b) for b in boundaries_s or ())})
    lo = max((c for c in cuts if c <= mid), default=0.0)
    hi = min((c for c in cuts if c > mid), default=float(total))
    span = hi - lo
    if span <= 0:
        return 1.0
    return max(0.0, min(end, hi) - max(start, lo)) / span


def _classify_gap(lv: np.ndarray, start: float, end: float, at_start: bool, at_end: bool,
                  total: float, cfg: QCConfig, boundaries_s, fade_in_s: float, fade_out_s: float,
                  window_s: float) -> dict:
    digital_share = float(np.mean(lv < cfg.digital_silence_dbfs))
    digital = digital_share >= DIGITAL_SHARE
    nearest, near_dist = None, math.inf
    for b in boundaries_s or ():
        d = 0.0 if start <= b <= end else min(abs(b - start), abs(b - end))
        if d < near_dist:
            nearest, near_dist = float(b), d
    near = nearest is not None and near_dist <= cfg.transition_guard_s
    share = _guard_share(start, end, boundaries_s, cfg.transition_guard_s)
    if near and near_dist <= window_s + 1e-9:          # touches / contains the boundary
        at_transition = _host_track_share(start, end, boundaries_s, total) < QUIET_TRACK_MIN_SHARE
    else:
        at_transition = near and share >= TRANSITION_GAP_MIN_GUARD_SHARE
    slack = max(window_s, cfg.silence_window_s)   # fades are planned in seconds, levels are windowed
    edge = ((at_start and fade_in_s > 0 and end <= fade_in_s + slack + 1e-9)
            or (at_end and fade_out_s > 0 and start >= total - fade_out_s - slack - 1e-9))
    if edge:
        cls, sev = "edge_fade", Severity.INFO
    elif digital:
        cls, sev = "digital_silence", Severity.WARNING
    elif at_transition:
        cls, sev = "transition_gap", Severity.WARNING
    else:
        cls, sev = "quiet_passage", Severity.INFO
    duration = end - start
    label = {"edge_fade": "silence inside the planned fade-in/out",
             "digital_silence": "digital silence",
             "transition_gap": "suspicious gap at a transition",
             "quiet_passage": "quiet passage (likely intentional)"}[cls]
    msg = (f"{label}: {duration:.1f} s from {format_hms(start)} ({start:.1f} s) to "
           f"{format_hms(end)} ({end:.1f} s), min level {float(lv.min()):.0f} dBFS")
    if near:
        msg += (f", near the transition at {format_hms(nearest)} ({nearest:.1f} s)" if cls != "quiet_passage"
                else f", starts/ends near the transition at {format_hms(nearest)} ({nearest:.1f} s) but "
                     f"extends into the track body")
    return {
        "start_s": round(start, 3), "end_s": round(end, 3), "duration_s": round(duration, 3),
        "start_hms": format_hms(start), "end_hms": format_hms(end),
        "min_level_dbfs": round(float(lv.min()), 1), "max_level_dbfs": round(float(lv.max()), 1),
        "digital_silence": digital, "digital_share": round(digital_share, 2),
        "near_boundary": near, "boundary_s": nearest if near else None,
        "classification": cls, "severity": sev.value, "suspicious": sev is Severity.WARNING,
        "message": msg,
    }


def scan_silence(tools: FFmpegTools, path: Path, cfg: QCConfig, *, boundaries_s: list[float] = (),
                 fade_in_s: float = 0.0, fade_out_s: float = 0.0,
                 sample_rate: int | None = None) -> list[dict]:
    """Stream-decode ``path`` and return classified silence gaps (see
    ``detect_silence_gaps``). Raises FFmpegError when nothing could be decoded —
    an undecodable file is never reported as "no silence"."""
    scan = scan_pcm(tools, path, cfg, sample_rate=sample_rate)
    if scan.decoded_duration_s <= 0:
        raise FFmpegError(f"could not decode {path} for the silence scan: "
                          + ("; ".join(scan.errors[:3]) or "no audio decoded"))
    if scan.errors:
        log.warning("%s: decoder reported errors during the silence scan: %s", path, scan.errors[0])
    levels, window_s = scan.silence_levels()
    return detect_silence_gaps(levels, window_s, cfg,
                               total_duration_s=scan.decoded_duration_s, boundaries_s=boundaries_s,
                               fade_in_s=fade_in_s, fade_out_s=fade_out_s)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class _Builder:
    """Collects issues / checks for one report."""

    def __init__(self) -> None:
        self.issues: list[QCIssue] = []
        self.checks_run: list[str] = []
        self.checks_skipped: list[str] = []
        # rows whose status is decided by the check that built them (tracks that never
        # reached the output, mix rows of an unpublished mix): not re-derived from issues
        self.fixed_rows: list[TrackQC] = []

    def add(self, severity: Severity, code: str, message: str, *, position: int | None = None,
            timestamp_s: float | None = None, file: str | None = None) -> QCIssue:
        issue = QCIssue(severity=severity, code=code, message=message, track_position=position,
                        timestamp_s=None if timestamp_s is None else round(float(timestamp_s), 3), file=file)
        self.issues.append(issue)
        return issue

    def ran(self, name: str) -> None:
        if name not in self.checks_run:
            self.checks_run.append(name)

    def skip(self, name: str, reason: str) -> None:
        entry = f"{name}: {reason}"
        if entry not in self.checks_skipped and name not in self.checks_run:
            self.checks_skipped.append(entry)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fmt_ts(t: float) -> str:
    return f"{format_hms(t)} ({t:.1f} s)"


def _opt_num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _resolve_inside(base: Path, rel: str) -> tuple[Path | None, str | None]:
    """(absolute path, None) when ``rel`` is a relative POSIX path that stays inside
    ``base``; otherwise (None, reason)."""
    reason = check_relative_uri(rel)
    if reason:
        return None, reason
    base_r = base.resolve()
    target = (base_r / rel).resolve()
    if not target.is_relative_to(base_r):
        return None, f"path escapes the output folder ({target})"
    return target, None


def _load_manifest(path: Path) -> tuple[PlaylistManifest | None, str | None, str | None]:
    """(manifest, issue_code, message). Never raises for bad content."""
    if not path.is_file():
        return None, "manifest_missing", f"manifest not found: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return None, "manifest_invalid", f"manifest {path.name} is not readable JSON: {e}"
    try:
        if not isinstance(data, dict):
            raise ValueError("top level is not an object")
        manifest = PlaylistManifest.from_dict(data)
        for t in manifest.tracks:
            if isinstance(t.index, bool) or not isinstance(t.index, int) or not isinstance(t.file, str):
                raise ValueError(f"track entry with invalid index/file: {t.index!r} / {t.file!r}")
            for name in ("loudness", "features"):     # null = unknown; anything else must be an object
                value = getattr(t, name)
                if value is None:
                    setattr(t, name, {})
                elif not isinstance(value, dict):
                    raise ValueError(f"track #{t.index}: '{name}' is not an object")
            if t.source_rel is not None and not isinstance(t.source_rel, str):
                raise ValueError(f"track #{t.index}: source_rel is not a string")
        for name in ("excluded", "flagged_transitions", "not_selected"):
            items = getattr(manifest, name)
            if items is None:
                setattr(manifest, name, [])
            elif not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
                raise ValueError(f"'{name}' must be a list of objects")
        for name in ("warnings", "leftover_files"):
            setattr(manifest, name, _str_list(getattr(manifest, name)))
    except (TypeError, ValueError, KeyError) as e:
        return None, "manifest_invalid", f"manifest {path.name} does not match the VEgroove schema: {e}"
    return manifest, None, None


def _verify_files(normalizer: LoudnessNormalizer, paths: list[Path]):
    """normalizer.verify_output for each path (order-preserving, threaded)."""
    def one(p: Path):
        try:
            return normalizer.verify_output(p, kind="mp3")
        except FFmpegNotFoundError:
            raise
        except (OSError, FFmpegError, LoudnessMeasurementError) as e:   # pragma: no cover - defensive
            return None, None, None, e
    if not paths:
        return []
    jobs = max(1, min(4, os.cpu_count() or 1, len(paths)))
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(one, paths))


def _tool_errors(ev) -> list[str]:
    """Probe/decode/meter failures reported by verify_output (or the exception itself)."""
    if isinstance(ev, BaseException):
        return [f"{type(ev).__name__}: {str(ev).splitlines()[0] if str(ev) else ''}"]
    return [x for x in (getattr(ev, "errors", None) or []) if "failed" in x]


def _str_list(value) -> list[str]:
    """A manifest list-of-strings field, tolerating a bare string / null."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _features_unknown(features: dict) -> bool:
    return any(features.get(f) in (None, "") for f in ("energy", "mood", "genre", "language"))


def _curation_track(t: ManifestTrack) -> TrackFeatures:
    """``TrackFeatures`` rebuilt from a manifest track, as ``curation.curate`` saw it at
    export: the exported (curated) features plus the identity fields of the manifest
    entry (``path`` = source_path; the playlist manifest keeps artist/title/rel path on
    the entry, the mix manifest inside ``features``). ``excluded`` is cleared — an
    ``excluded=true`` track is reported by the manifest presence check instead."""
    f = dict(t.features or {})
    duration = next((v for v in (_opt_num(f.get("duration_s")), _opt_num(f.get("source_duration_s")),
                                 _opt_num(t.duration_s)) if v is not None), 0.0)
    d = {k: v for k, v in f.items() if k != "path"}
    d.update(
        path=t.source_path or f.get("path") or t.source_rel or t.file or "",
        duration_s=duration,
        rel_path=f.get("rel_path") or t.source_rel or "",
        track_id=f.get("track_id") or t.track_id or "",
        artist=f["artist"] if "artist" in f else t.artist,
        title=f["title"] if "title" in f else t.title,
        excluded=False,
        vibe_tags=_str_list(f.get("vibe_tags")),
        review_reasons=_str_list(f.get("review_reasons")),
        exclusion_reasons=_str_list(f.get("exclusion_reasons")),
        feature_sources=f.get("feature_sources") if isinstance(f.get("feature_sources"), dict) else {},
    )
    return TrackFeatures.from_dict(d)


def _curation_code(reason: str) -> str:
    for pattern, code in _CURATION_REASON_CODES:
        if pattern.search(reason):
            return code
    return "profile_excluded"


def _editorial_reasons(t: ManifestTrack | None, extra: list[str] = ()) -> list[str]:
    """Editorial review reasons of a published track: the curator / curation flags
    exported in its features (``requires_review`` + ``review_reasons``) plus review
    reasons QC's own curation run found (``extra``), de-duplicated."""
    reasons: list[str] = []
    f = (t.features if t is not None else None) or {}
    if f.get("requires_review") is True:
        reasons += _str_list(f.get("review_reasons")) or ["flagged for review in the track metadata"]
    for r in extra:
        if r not in reasons:
            reasons.append(r)
    return reasons


def _loudness_targets(normalizer: LoudnessNormalizer) -> tuple[float, float, float, float]:
    cfg = normalizer.cfg
    return (cfg.target_lufs, cfg.lufs_tolerance, cfg.target_true_peak_dbtp,
            cfg.target_true_peak_dbtp + cfg.true_peak_tolerance_db)


def _check_manifest_targets(b: _Builder, manifest: PlaylistManifest, normalizer: LoudnessNormalizer) -> None:
    targets = manifest.loudness_targets or {}
    _, _, problems = apply_recorded_targets(targets, LoudnessConfig(), EncodeConfig())
    for problem in problems:
        b.add(Severity.FAIL, "manifest_invalid", f"manifest loudness_targets: {problem}")
    if problems:
        b.ran("loudness_targets")
    if not isinstance(targets, dict):
        return
    if targets.get("target_lufs") is None:
        if not problems:
            b.skip("loudness_targets", "manifest does not state its loudness target")
        return
    exported = _opt_num(targets.get("target_lufs"))
    if exported is None or isinstance(targets.get("target_lufs"), bool):
        return                          # already reported as manifest_invalid
    b.ran("loudness_targets")
    if abs(exported - normalizer.cfg.target_lufs) > 0.01:
        b.add(Severity.WARNING, "target_mismatch",
              f"manifest was exported for {exported:.1f} LUFS but QC checks against "
              f"{normalizer.cfg.target_lufs:.1f} LUFS")


# ---------------------------------------------------------------------------
# Shared manifest-level checks (both modes)
# ---------------------------------------------------------------------------

def _check_duplicates(b: _Builder, tracks: list[ManifestTrack], *, check_files: bool = True) -> None:
    """Same source (rel path / absolute path / track id) twice -> FAIL. ``check_files``
    also rejects two entries pointing at the same output file (playlist mode only)."""
    b.ran("duplicates")
    seen: dict[tuple[str, str], int] = {}
    reported: set[int] = set()
    for t in tracks:
        keys = []
        if t.source_rel:
            # identity of a FILE: case-preserving NFC path ('Straße' != 'Strasse'), like the
            # pipeline's track ids; casefolding is only for curator-facing lookups (R2B-1)
            keys.append(("source_rel", normalize_rel_path(t.source_rel)))
        if t.source_path:
            keys.append(("source_path", os.path.normcase(os.path.normpath(t.source_path))))
        if t.track_id:
            keys.append(("track_id", t.track_id))
        for kind, value in keys:
            first = seen.setdefault((kind, value), t.index)
            if first != t.index and t.index not in reported:
                reported.add(t.index)
                b.add(Severity.FAIL, "duplicate_source",
                      f"#{t.index} duplicates #{first}: same {kind} {value!r}",
                      position=t.index, file=t.file)
    if not check_files:
        return
    files: dict[str, int] = {}
    for t in tracks:
        first = files.setdefault(t.file, t.index)
        if first != t.index:
            b.add(Severity.FAIL, "duplicate_file", f"#{t.index} and #{first} reference the same file {t.file!r}",
                  position=t.index, file=t.file)


def _check_excluded(b: _Builder, manifest: PlaylistManifest, tracks: list[ManifestTrack]) -> None:
    """No published track may be listed in ``manifest.excluded`` or carry
    ``excluded=true`` in its metadata (FAIL ``excluded_track_present``)."""
    b.ran("excluded_tracks")
    excluded_rel = {}
    for e in manifest.excluded or []:
        rel = (e or {}).get("source_rel")
        if rel:
            excluded_rel[normalize_rel_path(rel)] = e.get("reasons") or []
    for t in tracks:
        rel = normalize_rel_path(t.source_rel or "")
        if rel and rel in excluded_rel:
            reasons = "; ".join(_str_list(excluded_rel[rel])) or "no reason given"
            b.add(Severity.FAIL, "excluded_track_present",
                  f"#{t.index} {t.source_rel!r} is listed as excluded in the manifest ({reasons})",
                  position=t.index, file=t.file)
        if t.features.get("excluded") is True:
            reasons = "; ".join(_str_list(t.features.get("exclusion_reasons"))) or "no reason given"
            b.add(Severity.FAIL, "excluded_track_present",
                  f"#{t.index} {t.source_rel!r} is marked excluded in its metadata ({reasons})",
                  position=t.index, file=t.file)


def _check_profile(b: _Builder, tracks: list[ManifestTrack], profile: ClientProfile | None, *,
                   unavailable: str | None = None) -> dict[int, list[str]]:
    """Client-profile rules, verified by re-running ``curation.curate`` on the exported
    features (one implementation of the rules: blocklist / allowlist / languages /
    Cyrillic hint / avoid preferences / approval). Every track curation would not make
    eligible -> one FAIL per curation reason. Returns ``{id(track): review reasons}``
    for eligible tracks curation flags for review (merged into ``requires_review``)."""
    if profile is None:
        b.skip("profile_rules", "no client profile supplied")
        return {}
    if unavailable:
        b.skip("profile_rules", unavailable)
        return {}
    rebuilt: list[tuple[ManifestTrack, TrackFeatures]] = []
    for t in tracks:
        try:
            rebuilt.append((t, _curation_track(t)))
        except (TypeError, ValueError, KeyError) as ex:
            b.add(Severity.WARNING, "profile_rules_unverifiable",
                  f"#{t.index} {t.source_rel!r}: features cannot be read back for the profile check ({ex})",
                  position=t.index, file=t.file)
    try:
        result = curate([f for _, f in rebuilt], profile)
    except ProfileError as ex:
        b.add(Severity.WARNING, "profile_rules_error", f"client profile cannot be applied: {ex}")
        b.skip("profile_rules", "client profile cannot be applied")
        return {}
    b.ran("profile_rules")
    review: dict[int, list[str]] = {}
    for (t, _), d in zip(rebuilt, result.decisions):
        if d.eligible:
            if d.review_reasons:
                review[id(t)] = list(d.review_reasons)
            continue
        for reason in d.exclusion_reasons or ["not eligible under the client profile"]:
            b.add(Severity.FAIL, _curation_code(reason),
                  f"#{t.index} {t.source_rel!r} would be excluded by profile {profile.client!r}: {reason}",
                  position=t.index, file=t.file)
    return review


def _check_unpublished(b: _Builder, manifest: PlaylistManifest) -> list[TrackQC]:
    """Tracks the export could not deliver, from ``manifest.excluded``:

    * ``stage = "normalization"`` (or loudness status ``failed``) -> FAIL
      ``normalization_failed`` + a ``failed`` TrackQC row (position 0 = not in the
      output) so that "Tracks failed" counts it and the run can never PASS;
    * ``stage = "analysis"`` (undecodable library file) -> WARNING ``source_analysis_failed``.
    """
    b.ran("unpublished_tracks")
    rows: list[TrackQC] = []
    for e in manifest.excluded or []:
        stage = excluded_stage(e)
        rel = e.get("source_rel")
        reasons = "; ".join(_str_list(e.get("reasons"))) or "no reason given"
        if stage == "normalization":
            loud = e.get("loudness") if isinstance(e.get("loudness"), dict) else {}
            planned = e.get("sequence_position")
            where = (f" (planned position {planned})"
                     if isinstance(planned, int) and not isinstance(planned, bool) else "")
            b.add(Severity.FAIL, "normalization_failed",
                  f"{rel!r}{where} failed normalization and is not in the output: {reasons}", file=rel)
            row = TrackQC(position=0, file="", source_rel=rel,
                          original_lufs=_opt_num(loud.get("original_lufs")), final_lufs=None,
                          original_true_peak_dbtp=_opt_num(loud.get("original_true_peak_dbtp")),
                          final_true_peak_dbtp=None, duration_s=None, status="failed",
                          warnings=[f"FAIL: not published — {reasons}"])
            rows.append(row)
            b.fixed_rows.append(row)
        elif stage == "analysis":
            b.add(Severity.WARNING, "source_analysis_failed",
                  f"library file {rel!r} could not be analysed and was left out: {reasons}", file=rel)
    return rows


def _list_not_selected(b: _Builder, manifest: PlaylistManifest) -> None:
    """Eligible tracks the sequencer did not place in this output (INFO, with reason)."""
    for e in manifest.not_selected or []:
        rel = e.get("source_rel") or e.get("rel_path")
        reason = "; ".join(_str_list(e.get("reason")) + _str_list(e.get("reasons"))) or "no reason given"
        b.add(Severity.INFO, NOT_SELECTED_CODE, f"{rel!r} was eligible but not selected: {reason}", file=rel)


def _collect_flagged(b: _Builder, manifest: PlaylistManifest, flagged_out: list[dict]) -> None:
    """Manifest flagged transitions -> report list, each with its reasons.

    * ``sequencing`` -> WARNING (``suspicious_energy_transition`` when a reason reports an
      energy jump — the §11 summary count — else ``flagged_transition``);
    * ``loudness`` / ``transition`` -> INFO: QC re-derives them itself (fresh short-term
      measurements / the renderer's transition reports), so they are not double-counted;
      a ``transition`` entry already derived from the renderer is not listed twice;
    * any other kind (e.g. ``unscored``: the sequencer never scored the pair) -> WARNING.
    """
    items = manifest.flagged_transitions or []
    b.ran("flagged_transitions")
    renderer_pairs = {f.get("to_index") for f in flagged_out if f.get("source") == "renderer"}
    for f in items:
        f = dict(f or {})
        frm = f.get("from_index", f.get("from_position"))
        to = f.get("to_index", f.get("to_position"))
        reasons = _str_list(f.get("reasons"))
        kind = f.get("kind") or "sequencing"
        if kind == "transition" and to in renderer_pairs:
            continue
        entry = {**f, "from_index": frm, "to_index": to, "reasons": reasons, "kind": kind,
                 "source": f.get("source") or "manifest"}
        flagged_out.append(entry)
        why = "; ".join(reasons) or "no reason given"
        if kind == "sequencing":
            # only real energy JUMPS count as "suspicious energy transitions" (§11); e.g.
            # "unknown energy on both sides" is a coverage note, reported as flagged_transition
            energy = is_energy_jump_flag({"kind": kind, "reasons": reasons})
            b.add(Severity.WARNING, "suspicious_energy_transition" if energy else "flagged_transition",
                  f"transition #{frm} -> #{to} flagged by the sequencer: {why}", position=to)
        elif kind in ("loudness", "transition"):
            b.add(Severity.INFO, "flagged_transition",
                  f"transition #{frm} -> #{to} ({kind}) flagged at export (re-checked by QC): {why}", position=to)
        else:
            b.add(Severity.WARNING, "flagged_transition",
                  f"transition #{frm} -> #{to} ({kind}) flagged at export: {why}", position=to)


def _anchor_slot(anchored, planned_position, length: int) -> int | None:
    """1-based slot an anchored plan entry must hold in an order of ``length`` tracks:
    ``position N`` counts from the start, ``position -N`` from the end (``position -1`` =
    the last track, wherever that is after a failed normalization shortened the output);
    ``phase X`` anchors are not positional (None). Any other label falls back to the
    planned position."""
    text = str(anchored).strip()
    if text.casefold().startswith("phase"):
        return None
    m = _ANCHOR_POSITION_RE.match(text)
    if m is None:
        return planned_position if isinstance(planned_position, int) and not isinstance(planned_position, bool) \
            else None
    n = int(m.group(1))
    if n > 0:
        return n
    return length + n + 1 if n < 0 else None


def _check_plan_violations(b: _Builder, manifest: PlaylistManifest, plan: dict) -> None:
    """Hard-constraint violations the planner reported (``plan.violations``): a delivery whose
    plan breaks its scenario is never a silent PASS. Transition kinds (forbidden_pair,
    tempo_jump, key_jump, energy_jump, artist_gap) reach the manifest as sequencer flags
    ('plan constraint not met: <message>', ``plan_pipeline.sequence_from_plan``) and are
    reported by ``_collect_flagged``; one that did not get there (no valid position, an
    adjacency broken by a failed normalization) and every other kind (anchor, required,
    phase_anchor, duration, artist_count, same_song, track_count, ...) is a WARNING
    ``plan_violation`` with the planner's message. A plan without ``violations`` has none."""
    b.ran("plan_constraints")
    raw = plan.get("violations")
    if raw is None:
        return
    if not isinstance(raw, list):
        b.add(Severity.WARNING, "plan_invalid",
              "manifest.plan.violations is not a list: the plan's hard constraints could not be checked")
        return
    flagged = [r for f in manifest.flagged_transitions or [] if isinstance(f, dict)
               for r in _str_list(f.get("reasons"))]
    for v in raw:
        v = v if isinstance(v, dict) else {"message": "malformed violation entry" if v is None else str(v)}
        kind = str(v.get("kind") or "unknown")
        message = str(v.get("message") or "").strip()
        if kind in PLAN_TRANSITION_VIOLATIONS and message:
            # each flag accounts for ONE violation: two identical messages need two flags
            hit = next((i for i, r in enumerate(flagged) if message in r), None)
            if hit is not None:
                flagged.pop(hit)
                continue
        pos = v.get("position")
        where = f" at planned #{pos}" if isinstance(pos, int) and not isinstance(pos, bool) else ""
        b.add(Severity.WARNING, "plan_violation",
              f"the plan breaks a hard constraint of its scenario ({kind}{where}): {message or 'no message recorded'}")


def _check_plan(b: _Builder, manifest: PlaylistManifest) -> None:
    """VEmix 3.0: a delivery rendered from a dramaturgy plan (``manifest.plan``) must play the
    planned order. A planned track that did not reach the output (normalization failure)
    shifts everything after it: WARNING, and an anchor that moved is named (``position N``
    from the start, ``position -N`` from the end; phase anchors are not positional). Any other
    difference from the plan is a FAIL. Hard-constraint violations the planner reported are
    WARNINGs (``_check_plan_violations``). Nothing runs for 2.0 manifests (no ``plan``)."""
    plan = manifest.plan
    if not isinstance(plan, dict):
        return
    planned = [p for p in plan.get("order") or [] if isinstance(p, dict)]
    if planned:
        b.add(Severity.INFO, "plan_source",
              f"rendered from plan '{plan.get('scenario')}' variant {plan.get('variant')} ({plan.get('preset')}), "
              f"{len(planned)} planned track(s)")
    _check_plan_violations(b, manifest, plan)
    if not planned:
        b.add(Severity.WARNING, "plan_invalid", "manifest.plan has no planned order to check against")
        return
    b.ran("plan_order")
    b.ran("plan_anchors")
    key = lambda d: d.get("track_id") or d.get("rel_path")                    # noqa: E731
    published = [t.track_id or t.source_rel for t in manifest.tracks]
    planned_keys = [key(p) for p in planned]
    missing = [p for p in planned if key(p) not in set(published)]
    expected = [k for k in planned_keys if k in set(published)]
    for p in missing:
        b.add(Severity.WARNING, "plan_track_missing",
              f"planned #{p.get('position')} {p.get('display') or p.get('rel_path')!r} (phase {p.get('phase')}) is not "
              f"in the output; later positions moved up by one", file=p.get("rel_path"))
    if published != expected:
        extra = [k for k in published if k not in set(planned_keys)]
        b.add(Severity.FAIL, "plan_order_mismatch",
              "the published order differs from the plan"
              + (f" ({len(extra)} track(s) that are not in the plan)" if extra else ""))
        return
    index = {k: i + 1 for i, k in enumerate(published)}
    for p in planned:
        if not p.get("anchored") or key(p) not in index:
            continue
        want = _anchor_slot(p["anchored"], p.get("position"), len(published))
        if want is None or index[key(p)] == want:
            continue
        if want < 1 or want > len(published):
            b.add(Severity.WARNING, "plan_anchor_moved",
                  f"anchored track {p.get('display') or p.get('rel_path')!r} ({p.get('anchored')}) cannot hold that "
                  f"position in a published order of {len(published)} track(s); it plays at #{index[key(p)]}",
                  position=index[key(p)])
            continue
        in_plan = _anchor_slot(p["anchored"], p.get("position"), len(planned))
        b.add(Severity.WARNING, "plan_anchor_moved",
              f"anchored track {p.get('display') or p.get('rel_path')!r} ({p.get('anchored')}) plays at "
              f"#{index[key(p)]} instead of #{want}"
              + (f" (the plan itself placed it at #{p.get('position')})" if p.get("position") != in_plan else ""),
              position=index[key(p)])


def _identities(track_id, rel_path) -> set[tuple[str, str]]:
    """Identity keys of one track (track id and/or library-relative path)."""
    out: set[tuple[str, str]] = set()
    if isinstance(track_id, str) and track_id.strip():
        out.add(("id", track_id.strip()))
    if isinstance(rel_path, str) and rel_path.strip():
        out.add(("rel", normalize_rel_path(rel_path)))
    return out


def _planned_contrast_positions(manifest: PlaylistManifest, entries: list[MixTrackEntry]) -> set[int]:
    """Mix positions whose INCOMING transition is a planned contrast of the scenario
    (``plan.order[].planned_contrast``): the track and its predecessor in the mix are matched
    to the planned pair by track id or rel_path (a planned track lost to a failed
    normalization breaks the pair). Empty for 2.0 manifests (no ``plan``) and for plans
    without the field."""
    plan = manifest.plan
    if not isinstance(plan, dict):
        return set()
    order = [p for p in plan.get("order") or [] if isinstance(p, dict)]
    pairs = [(_identities(a.get("track_id"), a.get("rel_path")), _identities(c.get("track_id"), c.get("rel_path")))
             for a, c in zip(order, order[1:]) if c.get("planned_contrast") is True]
    if not pairs:
        return set()
    by_index = {t.index: t for t in manifest.tracks or []}

    def ids(e: MixTrackEntry) -> set[tuple[str, str]]:
        t = by_index.get(e.position)
        return _identities(t.track_id if t else None, e.source_rel) | (_identities(None, t.source_rel) if t else set())

    out: set[int] = set()
    for prev, nxt in zip(entries, entries[1:]):
        pi, ni = ids(prev), ids(nxt)
        if any(pi & a and ni & c for a, c in pairs):
            out.add(nxt.position)
    return out


def _energy_only_uncertainty(flags: list[str]) -> bool:
    """True when the renderer's notes on an uncertain transition name the energy difference
    and no tempo / BPM / beat / phrase cause."""
    return (any(_ENERGY_DIFFERENCE_FLAG_RE.search(f) for f in flags)
            and not any(_TEMPO_FLAG_RE.search(f) for f in flags))


def _unknown_metadata(b: _Builder, tracks: list[ManifestTrack]) -> int:
    b.ran("metadata_coverage")
    unknown = [t.index for t in tracks if _features_unknown(t.features or {})]
    if unknown:
        b.add(Severity.INFO, "unknown_metadata",
              f"{len(unknown)} track(s) with unknown energy/mood/genre/language: "
              + ", ".join(f"#{i}" for i in unknown[:30]))
    return len(unknown)


def _assign_track_status(rows: list[TrackQC], issues: list[QCIssue], fixed: list[TrackQC] = ()) -> None:
    """Per-track status from the issues located on that track. Transition-level
    issues (they concern a PAIR of tracks; position = incoming track) are listed in
    the report but do not make a single track 'review'/'failed'. Rows in ``fixed``
    keep the status of the check that built them."""
    by_pos: dict[int, list[QCIssue]] = {}
    for i in issues:
        if i.track_position is not None and i.code not in TRANSITION_CODES:
            by_pos.setdefault(i.track_position, []).append(i)
    fixed_ids = {id(r) for r in fixed}
    for row in rows:
        if id(row) in fixed_ids:
            continue
        own = by_pos.get(row.position, [])
        sev = {Severity(i.severity) for i in own}
        row.status = ("failed" if Severity.FAIL in sev else
                      "review" if Severity.WARNING in sev else "passed")
        row.warnings = [f"{Severity(i.severity).value}: {i.message}" for i in own
                        if Severity(i.severity) is not Severity.INFO] + row.warnings


def _finalize(report: QCReport, b: _Builder) -> QCReport:
    report.issues = b.issues
    report.checks_run = b.checks_run
    report.checks_skipped = b.checks_skipped
    _assign_track_status(report.tracks, report.issues, b.fixed_rows)
    report.status = overall_status(report.issues, report.checks_skipped)
    report.summary = compute_summary(report)
    log.info("QC %s: %s (%d issue(s), %d check(s) skipped)", report.mode, report.status.value,
             len(report.issues), len(report.checks_skipped))
    return report


def _fail_report(mode: str, normalizer: LoudnessNormalizer, b: _Builder) -> QCReport:
    report = QCReport(mode=mode, generated_at=_now(), status=OverallStatus.FAIL,
                      target_lufs=normalizer.cfg.target_lufs,
                      target_true_peak_dbtp=normalizer.cfg.target_true_peak_dbtp)
    return _finalize(report, b)


# ---------------------------------------------------------------------------
# Playlist QC
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    position: int
    file: str
    track: ManifestTrack | None
    path: Path | None = None


def qc_playlist(output_dir: Path, normalizer: LoudnessNormalizer, *, profile: ClientProfile | None = None,
                qc_cfg: QCConfig | None = None) -> QCReport:
    """Independent QC of a playlist export folder (spec §10 "Playlist QC")."""
    output_dir = Path(output_dir)
    qc = qc_cfg or normalizer.qc
    b = _Builder()
    target, tol, tp_target, tp_limit = _loudness_targets(normalizer)
    report = QCReport(mode="playlist", generated_at=_now(), status=OverallStatus.FAIL,
                      target_lufs=target, target_true_peak_dbtp=tp_target)

    # --- manifest ---
    b.ran("manifest")
    manifest, code, msg = _load_manifest(output_dir / MANIFEST_NAME)
    if code:
        b.add(Severity.FAIL, code, msg, file=MANIFEST_NAME)
    elif manifest.mode != "playlist":
        b.add(Severity.FAIL, "manifest_wrong_mode", f"manifest mode is {manifest.mode!r}, expected 'playlist'",
              file=MANIFEST_NAME)
        manifest = None
    if manifest is not None:
        _check_manifest_targets(b, manifest, normalizer)
        if not manifest.tracks:
            b.add(Severity.FAIL, "empty_playlist", "manifest lists no tracks", file=MANIFEST_NAME)

    # --- m3u8 ---
    b.ran("m3u8_syntax")
    m3u8_path = output_dir / M3U8_NAME
    parsed: M3U8Parse | None = None
    if not m3u8_path.is_file():
        b.add(Severity.FAIL, "m3u8_missing", f"{M3U8_NAME} not found in {output_dir}", file=M3U8_NAME)
    else:
        try:
            parsed = parse_m3u8(m3u8_path)
        except OSError as ex:
            parsed = M3U8Parse(errors=[f"playlist cannot be read: {ex}"])
        for err in parsed.errors:
            b.add(Severity.FAIL, "m3u8_syntax", err, file=M3U8_NAME)
        for w in parsed.warnings:
            b.add(Severity.WARNING, "m3u8_warning", w, file=M3U8_NAME)

    b.ran("tracklist_present")
    if not (output_dir / TRACKLIST_NAME).is_file():
        b.add(Severity.WARNING, "tracklist_missing", f"{TRACKLIST_NAME} not found", file=TRACKLIST_NAME)

    # --- the authoritative entry list: manifest, else the playlist itself ---
    entries: list[_Entry] = []
    if manifest is not None:
        entries = [_Entry(t.index, t.file, t) for t in manifest.tracks]
    elif parsed is not None:
        entries = [_Entry(k, e.uri, None) for k, e in enumerate(parsed.entries, start=1)]

    # --- order: m3u8 == manifest == 1..N ---
    if manifest is None:
        b.skip("track_order", "manifest unavailable")
    elif parsed is None:
        b.skip("track_order", f"{M3U8_NAME} unavailable")
    else:
        b.ran("track_order")
        indices = [t.index for t in manifest.tracks]
        if indices != list(range(1, len(indices) + 1)):
            b.add(Severity.FAIL, "track_order_mismatch",
                  f"manifest indices are not the sequence 1..{len(indices)}: {indices[:20]}", file=MANIFEST_NAME)
        m3u = [_posix_clean(e.uri) for e in parsed.entries]
        man = [_posix_clean(t.file) for t in manifest.tracks]
        if m3u != man:
            b.add(Severity.FAIL, "track_order_mismatch", _order_diff(man, m3u), file=M3U8_NAME)

    # --- references exist and stay inside output_dir ---
    b.ran("files_exist")
    b.ran("paths_inside_output")
    referenced: dict[str, Path] = {}
    for e in entries:
        path, reason = _resolve_inside(output_dir, e.file)
        if reason:
            b.add(Severity.FAIL, "path_outside_output" if "escapes" in reason else "path_invalid",
                  f"#{e.position} {e.file!r}: {reason}", position=e.position, file=e.file)
            continue
        if not path.is_file():
            b.add(Severity.FAIL, "file_missing", f"#{e.position} referenced file does not exist: {e.file}",
                  position=e.position, file=e.file)
            continue
        e.path = path
        referenced[_posix_clean(e.file)] = path
    if parsed is not None and manifest is not None:     # m3u8-only references
        known = {_posix_clean(e.file) for e in entries}
        for k, pe in enumerate(parsed.entries, start=1):
            if _posix_clean(pe.uri) in known:
                continue
            path, reason = _resolve_inside(output_dir, pe.uri)
            if reason:
                b.add(Severity.FAIL, "path_outside_output" if "escapes" in reason else "path_invalid",
                      f"{M3U8_NAME} entry {k} {pe.uri!r}: {reason}", file=pe.uri)
            elif not path.is_file():
                b.add(Severity.FAIL, "file_missing", f"{M3U8_NAME} entry {k} does not exist: {pe.uri}", file=pe.uri)
            else:
                referenced[_posix_clean(pe.uri)] = path

    # --- client-profile rules: curation re-run on the exported features ---
    profile_review: dict[int, list[str]] = {}
    if manifest is not None:
        profile_review = _check_profile(b, manifest.tracks, profile)

    # --- decode + re-measure every existing file (never trust manifest numbers) ---
    to_verify = [e for e in entries if e.path is not None]
    log.info("QC: verifying %d file(s) in %s", len(to_verify), output_dir)
    results = _verify_files(normalizer, [e.path for e in to_verify])
    measurements: dict[int, LoudnessMeasurement | None] = {}
    rows: list[TrackQC] = []
    for e in entries:
        loud = (e.track.loudness if e.track else {}) or {}
        rows.append(TrackQC(
            position=e.position, file=e.file, source_rel=e.track.source_rel if e.track else None,
            original_lufs=_opt_num(loud.get("original_lufs")), final_lufs=None,
            original_true_peak_dbtp=_opt_num(loud.get("original_true_peak_dbtp")),
            final_true_peak_dbtp=None, duration_s=None, status="failed",
            warnings=[f"export: {w}" for w in _str_list(loud.get("warnings"))]))
    row_of = {id(e): r for e, r in zip(entries, rows)}   # by entry, robust to duplicated indices
    for name in ("files_decodable", "nonzero_duration", "loudness", "true_peak"):
        if to_verify:
            b.ran(name)
        else:
            b.skip(name, "no existing audio file to verify")
    for e, res in zip(to_verify, results):
        probe, decode, m, ev = res
        measurements[e.position] = m
        row = row_of[id(e)]
        editorial = _editorial_reasons(e.track, profile_review.get(id(e.track), []))
        _evaluate_playlist_file(b, e, row, probe, decode, m, ev, normalizer, qc,
                                target, tol, tp_limit, parsed, editorial)

    # --- duplicates / exclusions / unpublished tracks / metadata (manifest-level) ---
    if manifest is not None:
        _check_duplicates(b, manifest.tracks)
        _check_excluded(b, manifest, manifest.tracks)
        rows += _check_unpublished(b, manifest)
        report.unknown_metadata = _unknown_metadata(b, manifest.tracks)
        report.excluded = [dict(x) for x in manifest.excluded or []]
        _list_not_selected(b, manifest)
        _collect_flagged(b, manifest, report.flagged_transitions)
        _check_plan(b, manifest)
    else:
        for name in ("duplicates", "excluded_tracks", "unpublished_tracks"):
            b.skip(name, "manifest unavailable")
        for name in ("profile_rules", "metadata_coverage", "flagged_transitions"):
            b.skip(name, "manifest unavailable")

    # --- adjacent short-term loudness on the fresh measurements (spec §3) ---
    if len(entries) < 2:
        b.skip("adjacent_short_term", "fewer than two tracks")
    else:
        b.ran("adjacent_short_term")
        for prev, nxt in zip(entries, entries[1:]):
            mp, mn = measurements.get(prev.position), measurements.get(nxt.position)
            if mp is None or mn is None:
                continue
            reasons = compare_adjacent_short_term(mp.short_term, mn.short_term, qc)
            if reasons:
                report.flagged_transitions.append({
                    "from_index": prev.position, "to_index": nxt.position, "reasons": reasons,
                    "kind": "loudness", "source": "qc"})
                b.add(Severity.WARNING, "adjacent_loudness_jump",
                      f"transition #{prev.position} -> #{nxt.position}: " + "; ".join(reasons)
                      + " — listen before release (not auto-compressed)", position=nxt.position)

    # --- orphan / temp files in tracks/ ---
    _check_orphans(b, output_dir, set(referenced.values()))

    report.tracks = rows
    return _finalize(report, b)


def _posix_clean(rel: str) -> str:
    rel = rel.strip()
    while rel.startswith("./"):
        rel = rel[2:]
    return rel


def _order_diff(manifest_files: list[str], m3u_files: list[str]) -> str:
    if len(manifest_files) != len(m3u_files):
        only_m3u = [f for f in m3u_files if f not in manifest_files]
        only_man = [f for f in manifest_files if f not in m3u_files]
        detail = f"{M3U8_NAME} has {len(m3u_files)} entries, manifest has {len(manifest_files)}"
        if only_m3u:
            detail += f"; only in {M3U8_NAME}: {only_m3u[:5]}"
        if only_man:
            detail += f"; only in manifest: {only_man[:5]}"
        return "track order differs: " + detail
    for k, (a, c) in enumerate(zip(manifest_files, m3u_files), start=1):
        if a != c:
            return (f"track order differs at position {k}: manifest {a!r} vs {M3U8_NAME} {c!r}")
    return "track order differs"   # pragma: no cover


def _evaluate_playlist_file(b: _Builder, e: _Entry, row: TrackQC, probe: AudioProbe | None,
                            decode: DecodeCheck | None, m: LoudnessMeasurement | None, ev,
                            normalizer: LoudnessNormalizer, qc: QCConfig, target: float, tol: float,
                            tp_limit: float, parsed: M3U8Parse | None, editorial: list[str]) -> None:
    pos, f = e.position, e.file
    tool_errors = _tool_errors(ev)
    decodable = probe is not None and decode is not None and decode.ok
    if not decodable:
        detail = "; ".join((decode.errors[:2] if decode else []) + tool_errors[:2]) or "unknown error"
        b.add(Severity.FAIL, "file_not_decodable", f"#{pos} {f} is not cleanly decodable: {detail}",
              position=pos, file=f)
    duration = decode.decoded_duration_s if decode else None
    row.duration_s = round(duration, 3) if duration is not None else None
    if decode is not None and (duration is None or duration <= 0):
        b.add(Severity.FAIL, "zero_duration", f"#{pos} {f} decodes to zero/unknown duration",
              position=pos, file=f)
    if probe is not None:
        if probe.codec != "mp3":
            b.add(Severity.FAIL, "wrong_format", f"#{pos} {f} codec is {probe.codec!r}, expected 'mp3'",
                  position=pos, file=f)
        enc = normalizer.encode
        if probe.sample_rate != enc.sample_rate or probe.channels != enc.channels:
            b.add(Severity.FAIL, "format_mismatch",
                  f"#{pos} {f} is {probe.sample_rate} Hz / {probe.channels} ch, expected "
                  f"{enc.sample_rate} Hz / {enc.channels} ch", position=pos, file=f)
    track = e.track
    loud = (track.loudness if track else {}) or {}
    if duration and duration > 0:
        expected = _opt_num(track.duration_s) if track else None
        if expected is None:
            expected = _opt_num(loud.get("duration_s"))
        if expected is not None and abs(duration - expected) > qc.duration_tolerance_s:
            b.add(Severity.FAIL, "duration_mismatch",
                  f"#{pos} {f} decodes to {duration:.2f} s but the manifest says {expected:.2f} s",
                  position=pos, file=f)
        if parsed is not None:
            ext = next((pe for pe in parsed.entries if _posix_clean(pe.uri) == _posix_clean(f)), None)
            if ext is not None and ext.duration_s is not None and \
                    abs(ext.duration_s - duration) > EXTINF_SLACK_S + qc.duration_tolerance_s:
                b.ran("m3u8_durations")
                b.add(Severity.WARNING, "m3u8_duration_mismatch",
                      f"#{pos} #EXTINF says {ext.duration_s:.0f} s, file decodes to {duration:.1f} s",
                      position=pos, file=f)
            elif ext is not None:
                b.ran("m3u8_durations")
    status = str(loud.get("normalization_status") or "").lower() or None
    if status == "failed":
        b.add(Severity.FAIL, "failed_track_published",
              f"#{pos} {f} has normalization_status 'failed' in the manifest but is in the playlist",
              position=pos, file=f)
    if m is None:
        if decodable:
            detail = "; ".join(tool_errors[:2]) or "meter returned no usable value"
            b.add(Severity.FAIL, "loudness_unmeasurable", f"#{pos} {f}: loudness could not be measured ({detail})",
                  position=pos, file=f)
        return
    row.final_lufs = round(m.integrated_lufs, 2)
    row.final_true_peak_dbtp = round(m.true_peak_dbtp, 2)
    deviation = m.integrated_lufs - target
    if abs(deviation) > tol:
        if status == "review" and deviation < 0:
            reasons = "; ".join(_str_list(loud.get("review_reasons"))) or "flagged for review at export"
            b.add(Severity.WARNING, "requires_review",
                  f"#{pos} {f} measures {m.integrated_lufs:.1f} LUFS ({deviation:+.1f} LU vs target): "
                  f"requires review — {reasons}", position=pos, file=f)
        else:
            b.add(Severity.FAIL, "loudness_out_of_tolerance",
                  f"#{pos} {f} measures {m.integrated_lufs:.2f} LUFS, outside {target:.1f} ± {tol:.1f} LUFS"
                  + (f" (manifest status {status!r})" if status else ""), position=pos, file=f)
    elif status == "review" or loud.get("requires_review") is True or editorial:
        reasons = _str_list(loud.get("review_reasons")) + [f"editorial: {r}" for r in editorial]
        b.add(Severity.WARNING, "requires_review",
              f"#{pos} {f} is flagged for review: " + ("; ".join(reasons) or "no reason given"),
              position=pos, file=f)
    if m.true_peak_dbtp > tp_limit:
        b.add(Severity.FAIL, "true_peak_exceeded",
              f"#{pos} {f} true peak {m.true_peak_dbtp:+.2f} dBTP exceeds the limit {tp_limit:+.2f} dBTP",
              position=pos, file=f)
    reported = _opt_num(loud.get("measured_lufs"))
    if reported is not None and abs(reported - m.integrated_lufs) > normalizer.cfg.meter_disagreement_lu:
        b.add(Severity.WARNING, "manifest_loudness_mismatch",
              f"#{pos} {f}: manifest says {reported:.2f} LUFS, QC measures {m.integrated_lufs:.2f} LUFS "
              f"(file changed after export?)", position=pos, file=f)


def _check_orphans(b: _Builder, output_dir: Path, referenced: set[Path]) -> None:
    tracks_dir = output_dir / "tracks"
    if not tracks_dir.is_dir():
        b.skip("orphan_files", "no tracks/ folder")
        return
    b.ran("orphan_files")
    resolved = {p.resolve() for p in referenced}
    for p in sorted(tracks_dir.iterdir()):
        if not p.is_file() or p.resolve() in resolved:
            continue
        rel = p.relative_to(output_dir).as_posix()
        if ".veg-" in p.name or p.name.startswith("."):
            b.add(Severity.WARNING, "temp_file_leftover", f"leftover temporary file {rel}", file=rel)
        elif p.suffix.lower() in AUDIO_SUFFIXES:
            b.add(Severity.WARNING, "orphan_file", f"{rel} is not referenced by the playlist", file=rel)


# ---------------------------------------------------------------------------
# Continuous-mix QC
# ---------------------------------------------------------------------------

def parse_tracklist(path: Path) -> tuple[dict[int, list[str]], list[str]]:
    """Rows ``<pos> | H:MM:SS | ...`` -> {position: [timestamps]} (+ parse errors)."""
    rows: dict[int, list[str]] = {}
    errors: list[str] = []
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        return rows, [f"tracklist is not readable UTF-8 text: {e}"]
    for line in text.splitlines():
        m = _TRACKLIST_ROW_RE.match(line.strip())
        if m:
            rows.setdefault(int(m.group(1)), []).append(m.group(2))
    if not rows:
        errors.append("tracklist contains no '<n> | H:MM:SS |' rows")
    return rows, errors


_MIX_FLOATS = {
    MixTrackEntry: ("start_s", "end_s", "duration_s", "head_trim_s", "tail_trim_s", "crossfade_in_s"),
    TransitionReport: ("crossfade_s", "overlap_start_s", "gain_reduction_db"),
}


def _dataclass_from(cls, d: dict):
    """Build a mix-manifest dataclass strictly: every field without a default must be
    present, ``position`` must be an int and timing fields finite numbers (a NaN
    would silently pass every tolerance comparison)."""
    if not isinstance(d, dict):
        raise ValueError(f"{cls.__name__} entry is not an object")
    required = [f.name for f in fields(cls)
                if f.default is MISSING and f.default_factory is MISSING]
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"{cls.__name__} entry {d.get('position', '?')} lacks {missing}")
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in d.items() if k in known}
    if isinstance(kwargs["position"], bool) or not isinstance(kwargs["position"], int):
        raise ValueError(f"{cls.__name__} position {kwargs['position']!r} is not an integer")
    for name in _MIX_FLOATS[cls]:
        if name in kwargs:
            value = _opt_num(kwargs[name])
            if value is None or isinstance(kwargs[name], bool):
                raise ValueError(f"{cls.__name__} #{kwargs['position']}: {name}={kwargs[name]!r} "
                                 "is not a finite number")
            kwargs[name] = value
    return cls(**kwargs)


def _measure_segment(normalizer: LoudnessNormalizer, path: Path, start: float, end: float,
                     channels: int | None, qc: QCConfig) -> LoudnessMeasurement:
    """Integrated loudness / true peak / short-term stats of [start, end) of a file
    (input seeking; ffmpeg decodes from the seek point and trims exactly)."""
    cfg = normalizer.cfg
    af = loudness_filter(normalizer.prep_filter(channels), cfg.target_lufs, cfg.pcm_true_peak_ceiling,
                         cfg.target_lra)
    p = run_ffmpeg(normalizer.tools, ["-nostats", "-progress", "pipe:1", "-ss", f"{start:.3f}",
                                      "-to", f"{end:.3f}", "-i", str(path), "-map", "0:a:0", "-af", af,
                                      "-f", "null", "-"], check=False)
    if p.returncode != 0:
        raise FFmpegError(f"segment measurement failed for {path}", returncode=p.returncode, stderr=p.stderr)
    return build_measurement(p.stderr, p.stdout, edge_window_s=qc.edge_window_s)


def qc_mix(manifest_path: Path, normalizer: LoudnessNormalizer, *, profile: ClientProfile | None = None,
           qc_cfg: QCConfig | None = None) -> QCReport:
    """Independent QC of a continuous mix (spec §10 "Continuous Mix QC"). The FINAL
    rendered MP3 is measured as a whole — per-track normalization is no evidence
    of the mix loudness after crossfading.

    A mix the export did not publish (``mix.published = false`` or a failed final
    normalization) is FAIL ``mix_not_published`` and whatever file sits at its path is
    NOT measured (it is a previous run's audio, not this manifest's). When the manifest
    records ``sha256`` / ``size_bytes`` of the published MP3, the file on disk must
    match them (FAIL ``mix_file_mismatch``). ``profile`` re-checks the client-profile
    rules on the per-track features like playlist QC."""
    manifest_path = Path(manifest_path)
    qc = qc_cfg or normalizer.qc
    b = _Builder()
    target, tol, tp_target, tp_limit = _loudness_targets(normalizer)

    b.ran("manifest")
    manifest, code, msg = _load_manifest(manifest_path)
    if code:
        b.add(Severity.FAIL, code, msg, file=manifest_path.name)
        return _fail_report("mix", normalizer, b)
    if manifest.mode != "mix" or not isinstance(manifest.mix, dict):
        b.add(Severity.FAIL, "manifest_wrong_mode",
              f"manifest mode is {manifest.mode!r} / mix section {'missing' if not manifest.mix else 'invalid'}; "
              "expected a mix manifest", file=manifest_path.name)
        return _fail_report("mix", normalizer, b)
    mix = manifest.mix
    base = manifest_path.parent
    try:
        entries = sorted((_dataclass_from(MixTrackEntry, d) for d in mix.get("tracks") or []),
                         key=lambda t: t.position)
        transitions = [_dataclass_from(TransitionReport, d) for d in mix.get("transitions") or []]
    except (TypeError, ValueError) as e:
        b.add(Severity.FAIL, "manifest_invalid", f"mix section is malformed: {e}", file=manifest_path.name)
        return _fail_report("mix", normalizer, b)
    if not entries:
        b.add(Severity.FAIL, "manifest_invalid", "mix section lists no tracks", file=manifest_path.name)
    _check_manifest_targets(b, manifest, normalizer)

    report = QCReport(mode="mix", generated_at=_now(), status=OverallStatus.FAIL,
                      target_lufs=target, target_true_peak_dbtp=tp_target)
    mix_loud = mix.get("loudness") if isinstance(mix.get("loudness"), dict) else {}
    final_status = str(mix_loud.get("normalization_status") or "").lower() or None
    mix_info: dict = {"manifest": manifest_path.name, "file": mix.get("file"),
                      "expected_duration_s": _opt_num(mix.get("expected_duration_s")),
                      "seed": mix.get("seed"), "renderer_clipped_samples": mix.get("clipped_samples"),
                      "tracklist": mix.get("tracklist"), "published": mix.get("published"),
                      "final_normalization_status": final_status,
                      "manifest_warnings": list(manifest.warnings or [])}
    report.mix = mix_info
    rel = mix.get("file")

    # --- was the mix published at all? (a failed final encode leaves an older file in place) ---
    b.ran("mix_published")
    undelivered: str | None = None        # why the file on disk is not this manifest's mix
    if mix.get("published") is False or final_status == "failed":
        detail = ("; ".join(_str_list(mix_loud.get("errors"))[:3])
                  or "; ".join(w for w in manifest.warnings if w.startswith("final mix"))
                  or "no reason recorded")
        b.add(Severity.FAIL, "mix_not_published",
              f"the final mix was not published (published={mix.get('published')!r}, final normalization "
              f"status {final_status!r}): {detail} — any {rel!r} on disk is not the audio this manifest "
              "describes and was not measured", file=rel if isinstance(rel, str) else None)
        undelivered = "the mix was not published"

    # --- mix file: exists, is the published one, decodes, final loudness / true peak / duration ---
    b.ran("mix_file")
    mix_path: Path | None = None
    if undelivered:
        pass
    elif not isinstance(rel, str) or not rel:
        b.add(Severity.FAIL, "manifest_invalid", "mix section has no 'file'", file=manifest_path.name)
    else:
        mix_path, reason = _resolve_inside(base, rel)
        if reason:
            b.add(Severity.FAIL, "path_outside_output" if "escapes" in reason else "path_invalid",
                  f"mix file {rel!r}: {reason}", file=rel)
            mix_path = None
        elif not mix_path.is_file():
            b.add(Severity.FAIL, "file_missing", f"mix file does not exist: {rel}", file=rel)
            mix_path = None
        elif not _check_mix_integrity(b, mix, mix_path, rel, mix_info):
            undelivered = "the file on disk is not the published mix"
            mix_path = None
    if mix_path is None:
        b.skip("mix_file_integrity", undelivered or "mix file unavailable")
    unmeasured = undelivered or "mix file could not be decoded"

    probe = decode = m = None
    if mix_path is not None:
        log.info("QC: measuring mix %s", mix_path)
        probe, decode, m, ev = _verify_files(normalizer, [mix_path])[0]
        if probe is None or decode is None or not decode.ok:
            detail = "; ".join((decode.errors[:2] if decode else []) + _tool_errors(ev)[:2])
            b.add(Severity.FAIL, "file_not_decodable", f"mix {rel} is not cleanly decodable: {detail or 'unknown'}",
                  file=rel)
        if probe is not None and probe.codec != "mp3":
            b.add(Severity.FAIL, "wrong_format", f"mix codec is {probe.codec!r}, expected 'mp3'", file=rel)
        mix_info.update({
            "codec": probe.codec if probe else None, "sample_rate": probe.sample_rate if probe else None,
            "channels": probe.channels if probe else None,
            "decoded_duration_s": decode.decoded_duration_s if decode else None,
            "integrated_lufs": round(m.integrated_lufs, 2) if m else None,
            "true_peak_dbtp": round(m.true_peak_dbtp, 2) if m else None,
            "lra_lu": round(m.lra_lu, 2) if m else None,
        })

    if m is None:
        for name in ("mix_loudness", "mix_true_peak"):
            b.skip(name, undelivered or "mix file could not be measured")
        if mix_path is not None and decode is not None and decode.ok:
            b.add(Severity.FAIL, "loudness_unmeasurable", "mix loudness could not be measured", file=rel)
    else:
        b.ran("mix_loudness")
        b.ran("mix_true_peak")
        deviation = m.integrated_lufs - target
        if abs(deviation) > tol:
            if final_status == "review" and deviation < 0:
                b.add(Severity.WARNING, "requires_review",
                      f"final mix measures {m.integrated_lufs:.1f} LUFS ({deviation:+.1f} LU): requires review — "
                      + ("; ".join(_str_list(mix_loud.get("review_reasons"))) or "flagged at export"), file=rel)
            else:
                b.add(Severity.FAIL, "mix_loudness_out_of_tolerance",
                      f"final mix measures {m.integrated_lufs:.2f} LUFS, outside {target:.1f} ± {tol:.1f} LUFS",
                      file=rel)
        if m.true_peak_dbtp > tp_limit:
            b.add(Severity.FAIL, "true_peak_exceeded",
                  f"final mix true peak {m.true_peak_dbtp:+.2f} dBTP exceeds the limit {tp_limit:+.2f} dBTP",
                  file=rel)
        if abs(deviation) <= tol and (mix_loud.get("requires_review") is True
                                      or str(mix_loud.get("processing_mode") or "") == "dynamic"
                                      or str(mix_loud.get("loudnorm_reported_mode") or "") == "dynamic"):
            b.add(Severity.WARNING, "requires_review",
                  "the final mix pass was flagged for review: "
                  + ("; ".join(_str_list(mix_loud.get("review_reasons"))) or "loudnorm dynamic processing"),
                  file=rel)
        reported = _opt_num(mix_loud.get("measured_lufs"))
        if reported is not None and abs(reported - m.integrated_lufs) > normalizer.cfg.meter_disagreement_lu:
            b.add(Severity.WARNING, "manifest_loudness_mismatch",
                  f"manifest says the mix is {reported:.2f} LUFS, QC measures {m.integrated_lufs:.2f} LUFS", file=rel)

    expected = mix_info["expected_duration_s"]
    decoded = decode.decoded_duration_s if decode else None
    if expected is None:
        b.skip("mix_duration", "manifest has no expected_duration_s")
    elif decoded is None or decoded <= 0:
        b.skip("mix_duration", unmeasured)
    else:
        b.ran("mix_duration")
        if abs(decoded - expected) > qc.mix_duration_tolerance_s:
            b.add(Severity.FAIL, "mix_duration_mismatch",
                  f"mix decodes to {decoded:.2f} s ({format_hms(decoded)}) but {expected:.2f} s "
                  f"({format_hms(expected)}) was expected (tolerance {qc.mix_duration_tolerance_s:.2f} s)", file=rel)

    # --- transition timing (pure manifest consistency) ---
    _check_mix_timing(b, entries, transitions, expected)

    # --- one streaming decode: silence + clipping ---
    boundaries: list[float] = []
    for prev, nxt in zip(entries, entries[1:]):
        boundaries += [float(nxt.start_s), float(prev.end_s)]
    tcfg = TransitionConfig()
    fade_in = _opt_num(mix.get("fade_in_s"))
    fade_out = _opt_num(mix.get("fade_out_s"))
    fade_in = tcfg.mix_fade_in_s if fade_in is None else fade_in
    fade_out = tcfg.mix_fade_out_s if fade_out is None else fade_out
    scan: PcmScan | None = None
    if mix_path is not None and decode is not None and decode.ok:
        try:
            scan = scan_pcm(normalizer.tools, mix_path, qc, sample_rate=probe.sample_rate if probe else None)
        except FFmpegError as ex:
            b.add(Severity.FAIL, "decode_error", f"streaming decode of the mix failed: {str(ex).splitlines()[0]}",
                  file=rel)
            scan = None
        if scan is not None and scan.decoded_duration_s <= 0:
            scan = None
    if scan is None:
        b.skip("silence", unmeasured)
        b.skip("clipping", unmeasured)
    else:
        b.ran("silence")
        b.ran("clipping")
        for err in scan.errors[:3]:
            b.add(Severity.FAIL, "decode_error", f"decoder error while scanning the mix: {err}", file=rel)
        levels, level_window_s = scan.silence_levels()
        gaps = detect_silence_gaps(levels, level_window_s, qc, total_duration_s=scan.decoded_duration_s,
                                   boundaries_s=boundaries, fade_in_s=fade_in, fade_out_s=fade_out)
        for g in gaps:
            positions = [t.position for t in entries
                         if float(t.start_s) < g["end_s"] and float(t.end_s) > g["start_s"]]
            g["positions"] = positions
            b.add(Severity(g["severity"]), f"silence_{g['classification']}",
                  g["message"] + (f" (track {', '.join('#' + str(p) for p in positions)})" if positions else ""),
                  position=positions[0] if positions else None, timestamp_s=g["start_s"], file=rel)
        report.silence_warnings = gaps
        mix_info.update({"sample_peak_dbfs": scan.peak_dbfs, "clipped_samples": scan.clipped_samples,
                         "clip_events": scan.clip_events, "scan_window_s": scan.window_s,
                         "silence_resolution_s": round(level_window_s, 4),
                         "fade_in_s": fade_in, "fade_out_s": fade_out})
        if scan.clipped_samples:
            for ev in scan.clip_events:
                pos = _position_at(entries, ev["time_s"])
                b.add(Severity.FAIL, "clipping_detected",
                      f"clipping at {_fmt_ts(ev['time_s'])}: {ev['samples']} sample(s) at |x| >= "
                      f"{qc.clip_sample_threshold} (total {scan.clipped_samples} in the mix)",
                      position=pos, timestamp_s=ev["time_s"], file=rel)
    renderer_clips = _opt_num(mix.get("clipped_samples"))
    if renderer_clips:
        b.add(Severity.INFO, "renderer_clipped_samples",
              f"renderer reported {int(renderer_clips)} sample(s) above full scale in its float WAV "
              "(the decoded MP3 scan above is authoritative)", file=rel)

    # --- renderer peak protection / uncertain transitions ---
    # The 2.0 transition engine knows nothing about a 3.0 plan: the energy difference INTO a
    # planned contrast phase is intended (INFO); a tempo / beat cause still needs listening.
    b.ran("peak_protection")
    contrast_in = _planned_contrast_positions(manifest, entries)
    for tr in transitions:
        flags = [str(x) for x in (tr.flags or [])]
        reasons = []
        gr = _opt_num(tr.gain_reduction_db) or 0.0
        if abs(gr) > 0.01:
            reasons.append(f"peak protection reduced the overlap by {abs(gr):.1f} dB")
            b.add(Severity.WARNING, "peak_protection_applied",
                  f"transition into #{tr.position} at {_fmt_ts(float(tr.overlap_start_s))}: renderer peak "
                  f"protection reduced gain by {abs(gr):.1f} dB — listen to the overlap",
                  position=tr.position, timestamp_s=float(tr.overlap_start_s), file=rel)
        planned_contrast = tr.position in contrast_in
        if tr.uncertain and planned_contrast and _energy_only_uncertainty(flags):
            reasons.append("planned contrast of the scenario (energy change intended)")
            b.add(Severity.INFO, "planned_contrast_transition",
                  f"transition into #{tr.position} at {_fmt_ts(float(tr.overlap_start_s))} is a planned contrast "
                  f"of the scenario; the renderer marked it uncertain for the energy difference only: "
                  + "; ".join(flags),
                  position=tr.position, timestamp_s=float(tr.overlap_start_s), file=rel)
        elif tr.uncertain:
            reasons.append("uncertain transition (no reliable phrase/beat alignment)")
            b.add(Severity.WARNING, "uncertain_transition",
                  f"transition into #{tr.position} at {_fmt_ts(float(tr.overlap_start_s))} is marked uncertain"
                  + (f": {'; '.join(flags)}" if flags else "")
                  + (" (planned contrast of the scenario: only its energy change is intended — listen)"
                     if planned_contrast else ""),
                  position=tr.position, timestamp_s=float(tr.overlap_start_s), file=rel)
        if reasons:
            report.flagged_transitions.append({
                "from_index": tr.position - 1, "to_index": tr.position, "reasons": reasons + flags,
                "kind": "mix", "source": "renderer", "timestamp_s": float(tr.overlap_start_s),
                "timestamp_hms": format_hms(float(tr.overlap_start_s))})

    # --- tracklist timestamps ---
    _check_tracklist(b, base, mix.get("tracklist"), entries)

    # --- manifest-level checks shared with playlist mode ---
    mtracks = manifest.tracks or [ManifestTrack(index=t.position, file=rel or "", title=t.name, artist=None,
                                                duration_s=t.duration_s, source_rel=t.source_rel,
                                                source_path="", track_id="") for t in entries]
    _check_duplicates(b, mtracks, check_files=False)
    _check_excluded(b, manifest, mtracks)
    profile_review = _check_profile(b, manifest.tracks, profile, unavailable=None if manifest.tracks else
                                    "mix manifest carries no per-track features")
    failed_rows = _check_unpublished(b, manifest)
    report.unknown_metadata = _unknown_metadata(b, manifest.tracks) if manifest.tracks else 0
    if not manifest.tracks:
        b.skip("metadata_coverage", "mix manifest carries no per-track features")
    report.excluded = [dict(x) for x in manifest.excluded or []]
    _list_not_selected(b, manifest)
    _collect_flagged(b, manifest, report.flagged_transitions)
    _check_plan(b, manifest)

    # --- per-track body loudness inside the mix + adjacent short-term (spec §3) ---
    report.tracks = _mix_track_rows(b, normalizer, qc, mix_path if (decode and decode.ok) else None,
                                    probe, entries, manifest, target, tol, rel, mix_info,
                                    profile_review=profile_review, undelivered=undelivered,
                                    unmeasured=unmeasured) + failed_rows
    return _finalize(report, b)


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _check_mix_integrity(b: _Builder, mix: dict, path: Path, rel: str, mix_info: dict) -> bool:
    """The file on disk must carry the AUDIO the export published. Current manifests record
    ``audio_sha256`` / ``audio_size_bytes`` of the MP3 payload (ID3/APE tags excluded), so a
    curator's tag-only edit still passes; older manifests recorded whole-file ``sha256`` /
    ``size_bytes``. False (FAIL ``mix_file_mismatch``) on a difference; True when it matches
    or the manifest records nothing to compare."""
    def recorded(key: str) -> str | None:
        v = mix.get(key)
        return v.strip() if isinstance(v, str) and v.strip() else None

    audio_sha, audio_size = recorded("audio_sha256"), _opt_num(mix.get("audio_size_bytes"))
    file_sha, file_size = recorded("sha256"), _opt_num(mix.get("size_bytes"))
    if audio_sha is None and audio_size is None and file_sha is None and file_size is None:
        b.skip("mix_file_integrity", "manifest records no hash/size of the published mix")
        return True
    b.ran("mix_file_integrity")
    try:
        if audio_sha is not None or audio_size is not None:
            got_sha, got_size = mp3_audio_sha256(path)
            mix_info["audio_sha256"], mix_info["audio_size_bytes"] = got_sha, got_size
            same = ((audio_size is None or got_size == int(audio_size))
                    and (audio_sha is None or got_sha.casefold() == audio_sha.casefold()))
            what = "MP3 audio payload (tags excluded)"
        else:
            size = path.stat().st_size
            mix_info["size_bytes"] = size
            same = file_size is None or size == int(file_size)
            if same and file_sha is not None:
                got = _file_sha256(path)
                mix_info["sha256"] = got
                same = got.casefold() == file_sha.casefold()
            what = "file bytes (older manifest: tags included)"
        if not same:
            b.add(Severity.FAIL, "mix_file_mismatch",
                  f"mix file {rel!r}: the {what} differ from the mix this manifest published — a stale or "
                  "replaced file; not measured", file=rel)
            return False
    except OSError as ex:
        b.add(Severity.FAIL, "mix_file_mismatch", f"mix file {rel!r} cannot be read for verification: {ex}",
              file=rel)
        return False
    return True


def _position_at(entries: list[MixTrackEntry], t: float) -> int | None:
    hits = [e.position for e in entries if float(e.start_s) <= t < float(e.end_s)]
    return hits[-1] if hits else (entries[-1].position if entries and t >= float(entries[-1].start_s) else None)


def _check_mix_timing(b: _Builder, entries: list[MixTrackEntry], transitions: list[TransitionReport],
                      expected: float | None) -> None:
    if not entries:
        b.skip("transition_timing", "no tracks in the mix manifest")
        return
    b.ran("transition_timing")
    tol = TIMING_TOLERANCE_S

    def bad(pos: int, msg: str, ts: float | None) -> None:
        b.add(Severity.FAIL, "transition_timing", f"#{pos}: {msg}", position=pos, timestamp_s=ts)

    positions = [e.position for e in entries]
    if positions != list(range(1, len(entries) + 1)):
        b.add(Severity.FAIL, "transition_timing", f"mix track positions are not 1..{len(entries)}: {positions[:20]}")
    first = entries[0]
    if abs(float(first.start_s)) > tol:
        bad(first.position, f"first track starts at {_fmt_ts(float(first.start_s))}, expected 0:00:00", first.start_s)
    for k, e in enumerate(entries):
        start, end = float(e.start_s), float(e.end_s)
        xf = _opt_num(e.crossfade_in_s) or 0.0
        dur = _opt_num(e.duration_s)
        if end <= start:
            bad(e.position, f"end {_fmt_ts(end)} is not after start {_fmt_ts(start)}", start)
        if dur is not None and abs((end - start) - dur) > tol:
            bad(e.position, f"end - start = {end - start:.3f} s but duration_s = {dur:.3f} s", start)
        if xf < 0:
            bad(e.position, f"negative crossfade {xf:.3f} s", start)
        if k == 0:
            if xf > tol:
                bad(e.position, f"first track has a crossfade-in of {xf:.3f} s", start)
            continue
        prev = entries[k - 1]
        pstart, pend = float(prev.start_s), float(prev.end_s)
        if start <= pstart:
            bad(e.position, f"start {_fmt_ts(start)} is not after the previous start {_fmt_ts(pstart)}", start)
        want = pend - xf
        if abs(start - want) > tol:
            bad(e.position, f"starts at {start:.3f} s but previous end {pend:.3f} s minus crossfade {xf:.3f} s "
                            f"= {want:.3f} s (off by {start - want:+.3f} s)", start)
        if xf > (pend - pstart) + tol or (dur is not None and xf > dur + tol):
            bad(e.position, f"crossfade {xf:.3f} s is longer than an adjacent track", start)
    by_pos = {e.position: e for e in entries}
    for tr in transitions:
        e = by_pos.get(tr.position)
        if e is None or tr.position < 2:
            bad(tr.position, "transition report refers to no incoming track", None)
            continue
        xf = _opt_num(e.crossfade_in_s) or 0.0
        if abs(float(tr.crossfade_s) - xf) > tol:
            bad(tr.position, f"transition crossfade {float(tr.crossfade_s):.3f} s != track crossfade_in "
                             f"{xf:.3f} s", float(e.start_s))
        if abs(float(tr.overlap_start_s) - float(e.start_s)) > tol:
            bad(tr.position, f"overlap starts at {float(tr.overlap_start_s):.3f} s but the track starts at "
                             f"{float(e.start_s):.3f} s", float(tr.overlap_start_s))
        if float(tr.crossfade_s) < 0:
            bad(tr.position, f"negative transition crossfade {float(tr.crossfade_s):.3f} s", float(e.start_s))
    if expected is not None:
        last_end = float(entries[-1].end_s)
        if abs(last_end - expected) > MIX_END_TOLERANCE_S:
            bad(entries[-1].position, f"last track ends at {last_end:.3f} s but the expected mix duration is "
                                      f"{expected:.3f} s", last_end)


def _check_tracklist(b: _Builder, base: Path, rel, entries: list[MixTrackEntry]) -> None:
    if not rel:
        b.skip("tracklist_timestamps", "mix manifest names no tracklist")
        return
    b.ran("tracklist_timestamps")
    path, reason = _resolve_inside(base, str(rel))
    if reason or not path.is_file():
        b.add(Severity.FAIL, "tracklist_missing", f"tracklist {rel!r}: {reason or 'file not found'}", file=str(rel))
        return
    rows, errors = parse_tracklist(path)
    for err in errors:
        b.add(Severity.FAIL, "tracklist_invalid", err, file=str(rel))
    if errors:
        return
    wanted = {e.position for e in entries}
    for e in entries:
        stamps = rows.get(e.position)
        start = float(e.start_s)
        if not stamps:
            b.add(Severity.FAIL, "tracklist_incomplete", f"tracklist has no row for #{e.position}",
                  position=e.position, timestamp_s=start, file=str(rel))
            continue
        if len(stamps) > 1:
            b.add(Severity.FAIL, "tracklist_invalid", f"tracklist has {len(stamps)} rows for #{e.position}",
                  position=e.position, file=str(rel))
        # floor(start_s), tolerant to float noise on exact-second starts
        accepted = {format_hms(start + 1e-3), format_hms(max(0.0, start - 1e-3))}
        if stamps[0] not in accepted:
            b.add(Severity.FAIL, "tracklist_timestamp_mismatch",
                  f"tracklist says #{e.position} starts at {stamps[0]}, the mix timeline says "
                  f"{format_hms(start)} ({start:.2f} s)", position=e.position, timestamp_s=start, file=str(rel))
    extra = sorted(set(rows) - wanted)
    if extra:
        b.add(Severity.FAIL, "tracklist_invalid", f"tracklist lists positions not in the mix: {extra[:10]}",
              file=str(rel))


def _mix_track_rows(b: _Builder, normalizer: LoudnessNormalizer, qc: QCConfig, mix_path: Path | None,
                    probe: AudioProbe | None, entries: list[MixTrackEntry], manifest: PlaylistManifest,
                    target: float, tol: float, rel, mix_info: dict, *,
                    profile_review: dict[int, list[str]] | None = None, undelivered: str | None = None,
                    unmeasured: str = "mix file could not be decoded") -> list[TrackQC]:
    """One row per track of the mix timeline: its body loudness measured inside the
    mix, review flags (loudness ``review`` status and editorial flags — curator /
    curation ``requires_review``, like playlist QC) as timestamped WARNINGs. When the
    mix was not delivered (``undelivered``) every row is ``failed``: none of these
    tracks reached the listener."""
    by_index = {t.index: t for t in manifest.tracks or []}
    profile_review = profile_review or {}
    rows: list[TrackQC] = []
    seg_meas: dict[int, LoudnessMeasurement] = {}
    segments: list[dict] = []
    channels = probe.channels if probe else None
    if mix_path is None:
        b.skip("segment_loudness", unmeasured)
    elif entries:
        b.ran("segment_loudness")
    for k, e in enumerate(entries):
        mt = by_index.get(e.position)
        loud = (mt.loudness if mt else {}) or {}
        row = TrackQC(position=e.position, file=e.name or (mt.file if mt else ""), source_rel=e.source_rel,
                      original_lufs=_opt_num(loud.get("original_lufs")), final_lufs=None,
                      original_true_peak_dbtp=_opt_num(loud.get("original_true_peak_dbtp")),
                      final_true_peak_dbtp=None, duration_s=_opt_num(e.duration_s), status="passed",
                      warnings=[])
        reasons = []
        # same rule as playlist QC: status 'review' OR requires_review (e.g. opt-in loudnorm
        # dynamic/AGC processing, an unexpected dynamic fallback) — VDSP-1
        if (str(loud.get("normalization_status") or "").lower() == "review"
                or loud.get("requires_review") is True):
            reasons.append("loudness review during normalization: "
                           + ("; ".join(_str_list(loud.get("review_reasons"))) or "no reason given"))
        extra = profile_review.get(id(mt), []) if mt is not None else []
        reasons += [f"editorial: {r}" for r in _editorial_reasons(mt, extra)]
        if reasons:
            b.add(Severity.WARNING, "requires_review",
                  f"#{e.position} (starts {_fmt_ts(float(e.start_s))}) is flagged for review: " + "; ".join(reasons),
                  position=e.position, timestamp_s=float(e.start_s))
        if undelivered:
            row.status = "failed"
            row.warnings = [f"FAIL: not delivered — {undelivered}"]
            b.fixed_rows.append(row)
        rows.append(row)
        if mix_path is None:
            continue
        body_start = float(e.start_s) + (_opt_num(e.crossfade_in_s) or 0.0)
        nxt = entries[k + 1] if k + 1 < len(entries) else None
        body_end = float(e.end_s) - ((_opt_num(nxt.crossfade_in_s) or 0.0) if nxt else 0.0)
        seg = {"position": e.position, "start_s": round(body_start, 3), "end_s": round(body_end, 3)}
        if body_end - body_start < MIN_SEGMENT_S:
            seg["skipped"] = "track body shorter than the meter needs"
            segments.append(seg)
            continue
        try:
            sm = _measure_segment(normalizer, mix_path, body_start, body_end, channels, qc)
        except (FFmpegError, LoudnessMeasurementError) as ex:
            seg["error"] = str(ex).splitlines()[0]
            b.add(Severity.WARNING, "segment_unmeasurable",
                  f"#{e.position} body {_fmt_ts(body_start)} – {_fmt_ts(body_end)} could not be measured: "
                  f"{seg['error']}", position=e.position, timestamp_s=body_start, file=rel)
            segments.append(seg)
            continue
        seg_meas[e.position] = sm
        row.final_lufs = round(sm.integrated_lufs, 2)
        row.final_true_peak_dbtp = round(sm.true_peak_dbtp, 2)
        seg.update(integrated_lufs=row.final_lufs, true_peak_dbtp=row.final_true_peak_dbtp)
        segments.append(seg)
        dev = sm.integrated_lufs - target
        if abs(dev) > tol + SEGMENT_EXTRA_TOLERANCE_LU:
            b.add(Severity.WARNING, "segment_loudness_deviation",
                  f"#{e.position} plays at {sm.integrated_lufs:.1f} LUFS inside the mix "
                  f"({dev:+.1f} LU vs {target:.1f} LUFS) from {_fmt_ts(body_start)}",
                  position=e.position, timestamp_s=body_start, file=rel)
    mix_info["segments"] = segments
    if len(entries) < 2 or not seg_meas:
        b.skip("adjacent_short_term", "fewer than two measured track bodies")
        return rows
    b.ran("adjacent_short_term")
    for prev, nxt in zip(entries, entries[1:]):
        a, c = seg_meas.get(prev.position), seg_meas.get(nxt.position)
        if a is None or c is None:
            continue
        reasons = compare_adjacent_short_term(a.short_term, c.short_term, qc)
        if reasons:
            ts = float(nxt.start_s)
            b.add(Severity.WARNING, "adjacent_loudness_jump",
                  f"transition #{prev.position} -> #{nxt.position} at {_fmt_ts(ts)}: " + "; ".join(reasons),
                  position=nxt.position, timestamp_s=ts, file=rel)
    return rows
