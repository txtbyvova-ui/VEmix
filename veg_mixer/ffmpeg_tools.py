"""Thin, testable wrappers around ffmpeg / ffprobe.

Parsing helpers are pure functions (unit-tested with canned output); the ``run_*``
/ ``probe_*`` / ``measure_*`` functions shell out to the real binaries.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import AudioProbe, DecodeCheck, LoudnessMeasurement, ShortTermStats

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "FFmpeg with the libmp3lame encoder is required (ffprobe is recommended). Install it and make sure "
    "`ffmpeg`/`ffprobe` are on PATH, "
    "or point VEG_FFMPEG / VEG_FFPROBE at the executables.\n"
    "  Windows: winget install --id Gyan.FFmpeg -e   (then open a NEW console)\n"
    "  macOS:   brew install ffmpeg\n"
    "  Debian/Ubuntu: sudo apt install ffmpeg"
)

REQUIRED_FILTERS = ("loudnorm", "ebur128", "aresample", "volume", "pan")
# Every MP3 the tool publishes is encoded with LAME; a build without it is unusable.
REQUIRED_ENCODER = "libmp3lame"
# Discovery queries (-version / -filters / -encoders) return instantly on a sane build.
DISCOVERY_TIMEOUT_S = 30


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg/ffprobe missing or unusable (also when a binary vanishes after discovery)."""


class FFmpegError(RuntimeError):
    """ffmpeg/ffprobe returned an error."""

    def __init__(self, message: str, *, cmd: list[str] | None = None,
                 returncode: int | None = None, stderr: str = "") -> None:
        tail = "\n".join(stderr.strip().splitlines()[-8:]) if stderr else ""
        super().__init__(f"{message}" + (f"\n--- ffmpeg stderr (tail) ---\n{tail}" if tail else ""))
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr


class LoudnessMeasurementError(ValueError):
    """Meter output missing, malformed or numerically invalid."""


@dataclass(frozen=True)
class FFmpegTools:
    ffmpeg: str
    ffprobe: str | None
    version: str
    has_soxr: bool
    has_libmp3lame: bool


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _exe_works(path: str) -> str | None:
    try:
        p = subprocess.run([path, "-hide_banner", "-version"], capture_output=True,
                           encoding="utf-8", errors="replace", timeout=DISCOVERY_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return p.stdout


def _run_tool(cmd: list[str], *, what: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    """``subprocess.run`` for ffmpeg/ffprobe. A binary that vanished (or became unusable)
    after discovery raises FFmpegNotFoundError — never a bare OSError that callers would
    mistake for a broken input file. A timeout raises FFmpegError."""
    try:
        return subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
    except FileNotFoundError as e:   # binary vanished after discovery
        raise FFmpegNotFoundError(f"{what} at {cmd[0]!r} is no longer available: {e}\n" + INSTALL_HINT) from e
    except OSError as e:             # replaced by something that cannot be executed
        raise FFmpegNotFoundError(f"{what} at {cmd[0]!r} cannot be executed: {e}\n" + INSTALL_HINT) from e
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"{what} timed out after {timeout}s", cmd=cmd) from e


def _discovery_output(ff: str, option: str) -> str:
    """stdout of ``ffmpeg -hide_banner <option>``; any failure means the build is unusable."""
    try:
        p = _run_tool([ff, "-hide_banner", option], what="ffmpeg", timeout=DISCOVERY_TIMEOUT_S)
    except FFmpegError as e:
        raise FFmpegNotFoundError(f"ffmpeg at {ff!r} does not answer `{option}`: {e}\n" + INSTALL_HINT) from e
    if p.returncode != 0:
        raise FFmpegNotFoundError(f"ffmpeg at {ff!r} failed on `{option}` (exit code {p.returncode}).\n"
                                  + INSTALL_HINT)
    return p.stdout


def _lists_component(listing: str, name: str) -> bool:
    """True when ``-filters`` / ``-encoders`` output has ``name`` as a whole column."""
    return re.search(rf"\s{re.escape(name)}\s", listing) is not None


@functools.lru_cache(maxsize=8)
def find_ffmpeg(ffmpeg: str | None = None, ffprobe: str | None = None) -> FFmpegTools:
    """Locate ffmpeg (+ffprobe): explicit argument → VEG_FFMPEG/VEG_FFPROBE → PATH.

    Raises FFmpegNotFoundError with installation instructions when unusable: missing,
    not executable, lacking a required filter, or lacking the libmp3lame encoder.
    """
    ff_candidate = ffmpeg or os.environ.get("VEG_FFMPEG") or shutil.which("ffmpeg")
    if not ff_candidate:
        raise FFmpegNotFoundError("ffmpeg was not found.\n" + INSTALL_HINT)
    version_out = _exe_works(ff_candidate)
    if version_out is None:
        raise FFmpegNotFoundError(f"ffmpeg at {ff_candidate!r} cannot be executed.\n" + INSTALL_HINT)

    filters = _discovery_output(ff_candidate, "-filters")
    missing = [f for f in REQUIRED_FILTERS if not _lists_component(filters, f)]
    if missing:
        raise FFmpegNotFoundError(f"ffmpeg at {ff_candidate!r} lacks required filters: {missing}.\n" + INSTALL_HINT)
    encoders = _discovery_output(ff_candidate, "-encoders")
    if not _lists_component(encoders, REQUIRED_ENCODER):   # INT-20: fail at startup (exit 3), not per track
        raise FFmpegNotFoundError(
            f"ffmpeg at {ff_candidate!r} has no {REQUIRED_ENCODER} (MP3) encoder.\n" + INSTALL_HINT)

    fp_candidate = ffprobe or os.environ.get("VEG_FFPROBE")
    if not fp_candidate:
        sibling = Path(ff_candidate).with_name("ffprobe" + Path(ff_candidate).suffix)
        fp_candidate = str(sibling) if sibling.exists() else shutil.which("ffprobe")
    if fp_candidate and _exe_works(fp_candidate) is None:
        fp_candidate = None
    if not fp_candidate:
        log.warning("ffprobe not found; falling back to ffmpeg-based probing (less robust)")

    first_line = version_out.splitlines()[0] if version_out else "ffmpeg (unknown version)"
    return FFmpegTools(
        ffmpeg=ff_candidate,
        ffprobe=fp_candidate,
        version=first_line,
        has_soxr="--enable-libsoxr" in version_out,
        has_libmp3lame=True,   # checked above; hand-built FFmpegTools may still say False
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_ffmpeg(tools: FFmpegTools, args: list[str], *, check: bool = True,
               timeout: float | None = None) -> subprocess.CompletedProcess:
    cmd = [tools.ffmpeg, "-hide_banner", "-nostdin", *args]
    p = _run_tool(cmd, what="ffmpeg", timeout=timeout)
    if check and p.returncode != 0:
        raise FFmpegError(f"ffmpeg failed with exit code {p.returncode}", cmd=cmd,
                          returncode=p.returncode, stderr=p.stderr)
    return p


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def _to_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def parse_ffprobe_json(path: Path, text: str) -> AudioProbe:
    """Parse ``ffprobe -show_format -show_streams -print_format json`` output."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise FFmpegError(f"ffprobe returned invalid JSON for {path}: {e}") from e
    streams = data.get("streams") or []
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        raise FFmpegError(f"no audio stream in {path}")
    a = audio[0]
    fmt = data.get("format") or {}
    tags: dict[str, str] = {}
    for source in (fmt.get("tags") or {}, a.get("tags") or {}):
        for k, v in source.items():
            tags.setdefault(str(k).lower(), str(v))
    cover = any(s.get("codec_type") == "video" and (s.get("disposition") or {}).get("attached_pic") == 1
                for s in streams)
    duration = _to_float(a.get("duration")) or _to_float(fmt.get("duration"))
    return AudioProbe(
        path=path,
        duration_s=duration,
        sample_rate=_to_int(a.get("sample_rate")),
        channels=_to_int(a.get("channels")),
        codec=a.get("codec_name"),
        bit_rate=_to_int(a.get("bit_rate")) or _to_int(fmt.get("bit_rate")),
        tags=tags,
        has_cover_art=cover,
    )


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_AUDIO_RE = re.compile(r"Stream #\d+:\d+.*?: Audio: (\w+)[^,]*, (\d+) Hz, ([^,]+)")
_CH_NAMES = {"mono": 1, "stereo": 2, "2.1": 3, "quad": 4, "5.0": 5, "5.1": 6, "7.1": 8}


def parse_ffmpeg_banner(path: Path, stderr: str) -> AudioProbe:
    """Fallback probe from ``ffmpeg -i <file>`` stderr when ffprobe is unavailable."""
    m = _AUDIO_RE.search(stderr)
    if not m:
        raise FFmpegError(f"no audio stream in {path}", stderr=stderr)
    d = _DUR_RE.search(stderr)
    duration = None
    if d:
        duration = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))
    layout = m.group(3).strip().split("(")[0]
    channels = _CH_NAMES.get(layout)
    if channels is None:
        cm = re.match(r"(\d+) channels", layout)
        channels = int(cm.group(1)) if cm else None
    tags: dict[str, str] = {}
    meta = re.search(r"Input #0.*?\n  Metadata:\n((?:    .*\n)+)", stderr)
    if meta:
        for line in meta.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                tags.setdefault(k.strip().lower(), v.strip())
    cover = bool(re.search(r"Stream #\d+:\d+.*Video:.*\(attached pic\)", stderr))
    return AudioProbe(path=path, duration_s=duration, sample_rate=int(m.group(2)),
                      channels=channels, codec=m.group(1), tags=tags, has_cover_art=cover)


def probe_audio(tools: FFmpegTools, path: Path) -> AudioProbe:
    path = Path(path)
    if not path.is_file():
        raise FFmpegError(f"file not found: {path}")
    if tools.ffprobe:
        cmd = [tools.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
        p = _run_tool(cmd, what="ffprobe")      # SAFE-14: a vanished ffprobe is not a broken file
        if p.returncode != 0:
            raise FFmpegError(f"ffprobe failed for {path}", cmd=cmd, returncode=p.returncode, stderr=p.stderr)
        return parse_ffprobe_json(path, p.stdout)
    p = run_ffmpeg(tools, ["-i", str(path)], check=False)
    return parse_ffmpeg_banner(path, p.stderr)


def parse_progress_out_time(stdout: str) -> float | None:
    """Last ``out_time_us`` from ``-progress pipe:1`` output, in seconds."""
    values = re.findall(r"^out_time_us=(\d+)\s*$", stdout, re.M)
    return int(values[-1]) / 1e6 if values else None


def decode_check(tools: FFmpegTools, path: Path) -> DecodeCheck:
    """Fully decode the first audio stream; any decoder error fails the check."""
    p = run_ffmpeg(tools, ["-v", "error", "-nostats", "-progress", "pipe:1", "-i", str(path),
                           "-map", "0:a:0", "-f", "null", "-"], check=False)
    errors = [ln.strip() for ln in p.stderr.splitlines() if ln.strip()]
    duration = parse_progress_out_time(p.stdout)
    ok = p.returncode == 0 and not errors and duration is not None and duration > 0
    if p.returncode != 0 and not errors:
        errors.append(f"ffmpeg exited with code {p.returncode}")
    if duration is not None and duration <= 0:
        errors.append("decoded duration is zero")
    return DecodeCheck(ok=ok, decoded_duration_s=duration, errors=errors[:20])


# ---------------------------------------------------------------------------
# Loudness metering
# ---------------------------------------------------------------------------

# loudnorm JSON key -> (min, max) accepted range
_LOUDNORM_RANGES = {
    "input_i": (-99.0, 0.0),
    "input_tp": (-99.0, 30.0),
    "input_lra": (0.0, 99.0),
    "input_thresh": (-99.0, 0.0),
}
_LOUDNORM_OPTIONAL = {
    "output_i": (-99.0, 0.0),
    "output_tp": (-99.0, 30.0),
    "output_lra": (0.0, 99.0),
    "output_thresh": (-99.0, 0.0),
    "target_offset": (-99.0, 99.0),
}
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


def parse_loudnorm_json(stderr: str) -> dict:
    """Extract and validate the loudnorm ``print_format=json`` block.

    Returns floats for the numeric keys and the raw ``normalization_type`` string.
    Raises LoudnessMeasurementError on missing/malformed/non-finite/out-of-range values.
    """
    blocks = _JSON_BLOCK_RE.findall(stderr or "")
    if not blocks:
        raise LoudnessMeasurementError("loudnorm JSON block not found in ffmpeg output")
    try:
        raw = json.loads(blocks[-1])
    except json.JSONDecodeError as e:
        raise LoudnessMeasurementError(f"loudnorm JSON is malformed: {e}") from e
    if not isinstance(raw, dict):
        raise LoudnessMeasurementError("loudnorm JSON is not an object")

    out: dict = {}
    for key, (lo, hi) in {**_LOUDNORM_RANGES, **_LOUDNORM_OPTIONAL}.items():
        if key not in raw:
            if key in _LOUDNORM_RANGES:
                raise LoudnessMeasurementError(f"loudnorm JSON lacks required key {key!r}")
            continue
        try:
            value = float(raw[key])
        except (TypeError, ValueError):
            raise LoudnessMeasurementError(f"loudnorm {key}={raw[key]!r} is not a number") from None
        if not math.isfinite(value):
            hint = " (silent or empty audio?)" if key in ("input_i", "input_thresh") else ""
            raise LoudnessMeasurementError(f"loudnorm {key}={raw[key]!r} is not finite{hint}")
        if not (lo <= value <= hi):
            raise LoudnessMeasurementError(f"loudnorm {key}={value} is outside the plausible range [{lo}, {hi}]")
        out[key] = value
    ntype = raw.get("normalization_type")
    if ntype is not None and ntype not in ("linear", "dynamic"):
        raise LoudnessMeasurementError(f"unexpected normalization_type {ntype!r}")
    out["normalization_type"] = ntype
    return out


_NUM = r"(-?inf|-?nan|-?\d+(?:\.\d+)?)"
_EBU_I_RE = re.compile(r"Integrated loudness:\s*\n\s*I:\s*" + _NUM + r"\s*LUFS")
_EBU_LRA_RE = re.compile(r"Loudness range:\s*\n\s*LRA:\s*" + _NUM + r"\s*LU")
_EBU_TP_RE = re.compile(r"True peak:\s*\n\s*Peak:\s*" + _NUM + r"\s*dBFS")
_EBU_FRAME_RE = re.compile(r"\bt:\s*(\d+(?:\.\d+)?)\s+TARGET:.*?\bS:\s*" + _NUM)


def parse_ebur128_summary(stderr: str) -> dict:
    """Integrated / LRA / true peak from the ebur128 summary (None when absent or non-finite)."""
    def grab(rx: re.Pattern) -> float | None:
        found = rx.findall(stderr or "")
        return _to_float(found[-1]) if found else None
    return {"integrated": grab(_EBU_I_RE), "lra": grab(_EBU_LRA_RE), "true_peak": grab(_EBU_TP_RE)}


def parse_ebur128_frames(stderr: str) -> tuple[np.ndarray, np.ndarray]:
    """(t, S) arrays from ``ebur128=framelog=info`` per-100 ms lines."""
    pairs = [(float(t), float(s)) for t, s in _EBU_FRAME_RE.findall(stderr or "")
             if s.lstrip("-") not in ("inf", "nan")]
    if not pairs:
        return np.empty(0), np.empty(0)
    arr = np.asarray(pairs, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


SHORT_TERM_WINDOW_S = 3.0
GATE_LUFS = -70.0


def short_term_stats(t: np.ndarray, s: np.ndarray, edge_window_s: float = 10.0) -> ShortTermStats:
    """Distribution of gated short-term loudness plus head/tail medians.

    A short-term value at time t covers audio [t-3 s, t], so the head window uses
    t in [3, max(3, edge)] and the tail window t in [T-edge+3, T].
    """
    if t.size == 0:
        return ShortTermStats(None, None, None, None, None, None, frames=0)
    valid = (t >= SHORT_TERM_WINDOW_S - 0.05) & (s > GATE_LUFS)
    if not np.any(valid):
        return ShortTermStats(None, None, None, None, None, None, frames=int(t.size))
    tv, sv = t[valid], s[valid]
    end = float(t[-1])

    def med(mask: np.ndarray) -> float | None:
        return round(float(np.median(sv[mask])), 2) if np.any(mask) else None

    head = med(tv <= max(SHORT_TERM_WINDOW_S, edge_window_s) + 0.05)
    tail = med(tv >= min(end, end - edge_window_s + SHORT_TERM_WINDOW_S) - 0.05)
    p10, p50, p95 = (round(float(v), 2) for v in np.percentile(sv, [10, 50, 95]))
    return ShortTermStats(p10=p10, p50=p50, p95=p95, max=round(float(sv.max()), 2),
                          head_lufs=head, tail_lufs=tail, frames=int(t.size))


def loudness_filter(prep_filter: str | None, target_lufs: float, target_tp: float, target_lra: float) -> str:
    parts = [prep_filter] if prep_filter else []
    parts.append("ebur128=peak=true:framelog=info")
    parts.append(f"loudnorm=I={target_lufs:.2f}:TP={target_tp:.2f}:LRA={target_lra:.2f}:print_format=json")
    return ",".join(parts)


def build_measurement(stderr: str, stdout: str = "", *, edge_window_s: float = 10.0) -> LoudnessMeasurement:
    """Combine loudnorm JSON (primary) and ebur128 (cross-check + short-term) output."""
    ln = parse_loudnorm_json(stderr)
    ebu = parse_ebur128_summary(stderr)
    t, s = parse_ebur128_frames(stderr)
    tp_candidates = [v for v in (ln["input_tp"], ebu["true_peak"]) if v is not None]
    return LoudnessMeasurement(
        integrated_lufs=ln["input_i"],
        true_peak_dbtp=max(tp_candidates),   # conservative: safety first
        lra_lu=ln["input_lra"],
        threshold_lufs=ln["input_thresh"],
        target_offset_lu=ln.get("target_offset"),
        loudnorm_true_peak_dbtp=ln["input_tp"],
        ebur128_integrated_lufs=ebu["integrated"],
        ebur128_true_peak_dbtp=ebu["true_peak"],
        short_term=short_term_stats(t, s, edge_window_s),
        duration_s=parse_progress_out_time(stdout),
    )


def measure_loudness(tools: FFmpegTools, path: Path, *, prep_filter: str | None = None,
                     target_lufs: float = -16.0, target_tp: float = -1.5, target_lra: float = 11.0,
                     edge_window_s: float = 10.0) -> LoudnessMeasurement:
    """One decode pass: integrated LUFS, true peak (loudnorm 192 kHz + ebur128 4x
    oversampling, max of both), LRA, threshold, short-term distribution, decoded duration."""
    p = run_ffmpeg(tools, ["-nostats", "-progress", "pipe:1", "-i", str(path), "-map", "0:a:0",
                           "-af", loudness_filter(prep_filter, target_lufs, target_tp, target_lra),
                           "-f", "null", "-"], check=False)
    if p.returncode != 0:
        raise FFmpegError(f"loudness measurement failed for {path}", returncode=p.returncode, stderr=p.stderr)
    return build_measurement(p.stderr, p.stdout, edge_window_s=edge_window_s)
