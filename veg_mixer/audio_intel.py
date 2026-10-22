"""Segment-level audio intelligence (VEmix 3.0, spec §2): how a track's intensity,
level, density and texture evolve over time, its rough structure and a tempo
summary — all MEASURED from the file and cached in the analysis-cache section
``segments`` (independent of ``probe``/``bpm``/``key``/``energy``/``loudness``).

Method — one decode, one STFT, one HPSS, vectorised numpy (~2.7 s CPU and ~165 MB peak
for a 4-minute 44.1 kHz stereo track on a desktop CPU, two thirds of it the HPSS median
filters; ~11 s / ~245 MB at the 900 s cap — memory is bounded by block processing):

* Decode once: mono, ``SegmentConfig.sample_rate``, at most ``max_analysis_s`` seconds
  (libsndfile; formats it cannot open — m4a/aac/opus/... — through FFmpeg into a
  temporary float WAV, without assuming a known duration; FFmpeg also takes over when
  libsndfile fails mid-file or announces 0 frames for a non-trivial file, e.g. a WAV whose
  data size was never finalised, and a partial decode is noted). Deliberately NOT shared
  with the BPM/key decodes: those are pinned bit-exact to automix v1 and stay untouched.
* The analysed audio is scaled to unit RMS (level independence between tracks), then
  STFT'd (n_fft 2048, hop 512) in ~30 s blocks so memory stays bounded for long files.
  Per frame: power, bass power (< ``bass_cutoff_hz``), spectral centroid, spectral
  flatness, mel and chroma energies, and an HPSS split. HPSS is librosa's median-filter
  method (kernel 31, soft mask, power 2) with the harmonic (time) filter run on a
  spectrogram pooled 2x in time and the percussive (frequency) filter on one pooled 2x
  in frequency: ~3.5x cheaper than full resolution, percussive ratios within ~0.01 of
  ``librosa.decompose.hpss`` on test material (full resolution costs ~7 s for 4 min).
* Onset envelope, onsets and the tempogram over the whole analysed audio, with the
  librosa estimators the 2.0 energy heuristic uses — plus one guard: onsets weaker than
  ``_ONSET_MIN_STRENGTH`` are dropped (``onset_detect`` normalises the envelope to its
  maximum, so on a held, static sound it turns numerical noise into "onsets"; real
  onsets are an order of magnitude stronger, so ordinary material is unaffected).

Per segment (``segment_s``; raised for long files so there are at most ``max_segments``):

* ``intensity`` — the 2.0 energy heuristic (``analysis.compute_energy``: 0.35 onset
  density + 0.30 percussive ratio + 0.20 brightness + 0.15 tempo x onset density, the
  same constants, imported) evaluated per segment. Loudness-INDEPENDENT: a quieter
  passage of the same material scores the same, and values sit on the scale of
  ``TrackFeatures.energy``. A segment more than 50 dB below the track RMS is silent: 0.
* ``rms_db`` — segment RMS relative to the median segment (level-DEPENDENT inside the
  track: quiet intros, breaks, fades), floored at -60 dB.
* ``onset_rate`` (1/s), ``percussive_ratio``, ``bass_ratio`` (power below the cutoff /
  total), ``centroid_hz`` and ``flatness`` (medians over the segment's active frames).

Summary: ``energy_in``/``energy_out`` (intensity of the first / last ``edge_s`` of
non-silent audio), mean and peak; ``trajectory`` (linear fit + residual oscillation of
the smoothed intensity curve); ``density`` (onset density, percussive share and mel-band
occupancy); percussive and bass density; dynamic range of ``rms_db``; ``tempo`` (peaks of
the taper-corrected autocorrelation tempogram with a log-normal prior at 120 BPM,
half/double-time ambiguity — it never replaces or touches the cached ``bpm`` section);
a vocal-presence PROXY and a 6-number ``timbre`` vector (``TIMBRE_DIMENSIONS``).

Sections: Foote checkerboard novelty on a self-similarity matrix of 1 s blocks (MFCC,
chroma, level, onset activity, percussive share) plus fade-in/fade-out detection on the
block level; labels (``SECTION_LABELS``) from each section's intensity + level relative to
the track and its position; similar high-intensity sections share a ``repeat_group`` — a
chorus CANDIDATE, never claimed to be a chorus.

Limits — this is why every interpretive field carries a low confidence:

* Intensity rank-orders dense/bright/fast vs sparse/soft/slow material; it is not a
  perceptual energy measurement and ignores harmony, lyrics and context.
* Section boundaries are approximate (~1-2 s, the block size) and labels are guesses
  from relative intensity and position (confidence <= ``SECTION_MAX_CONFIDENCE``), not a
  musical analysis: there is no verse/chorus detection.
* The vocal proxy cannot tell a voice from a lead synth, guitar or strings in the same
  register (confidence <= ``VOCAL_PROXY_MAX_CONFIDENCE``).
* Tempo candidates come from onset periodicity: unreliable for rubato, beatless or very
  sparse material; half/double-time ambiguity is flagged, not resolved.
* Only the first ``max_analysis_s`` seconds are analysed (noted in ``notes``).

Nothing here is written into ``TrackFeatures``: measured segment data is a separate
category, and the 2.0 BPM/key/energy analysis is unchanged.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf

from .analysis import (
    CAMELOT_MAP,
    ENERGY_CENTROID_HZ,
    ENERGY_ONSET_RATE_SCALE,
    ENERGY_TEMPO_BPM,
    ENERGY_WEIGHTS,
    MAJOR_PROFILE,
    MINOR_PROFILE,
)
from .cache import AnalysisCache
from .ffmpeg_tools import FFmpegError, FFmpegNotFoundError, FFmpegTools, find_ffmpeg, run_ffmpeg

__all__ = [
    "DENSITY_LEVELS",
    "SECTION_LABELS",
    "SECTION_MAX_CONFIDENCE",
    "SEGMENTS_SECTION",
    "SEGMENTS_VERSION",
    "TIMBRE_DIMENSIONS",
    "TRAJECTORIES",
    "VOCAL_PROXY_MAX_CONFIDENCE",
    "VOCAL_PROXY_METHOD",
    "SegmentConfig",
    "analyze_segments",
    "compute_segments",
    "is_valid_segments",
    "segments_params",
    "timbre_similarity",
]

log = logging.getLogger(__name__)

SEGMENTS_VERSION = "segments-v1"
SEGMENTS_SECTION = "segments"

SECTION_LABELS = ("intro", "build", "peak", "break", "body", "outro")
TRAJECTORIES = ("rising", "falling", "stable", "wave")
DENSITY_LEVELS = ("sparse", "moderate", "dense")
# Order of the numbers in summary["timbre"]; every one is 0..1.
TIMBRE_DIMENSIONS = ("brightness", "noisiness", "bass", "percussive", "onset_density", "fullness")
VOCAL_PROXY_METHOD = "hpss-harmonic-share-300-3400Hz x in-band-spectral-movement (not a vocal detector)"
VOCAL_PROXY_MAX_CONFIDENCE = 0.3
SECTION_MAX_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# Fixed algorithm constants (part of segments_params — bump SEGMENTS_VERSION when the
# code changes in a way these do not capture)
# ---------------------------------------------------------------------------

_N_MELS = 128                 # librosa.feature.melspectrogram default -> the 2.0 onset envelope
_N_MFCC = 13                  # MFCC 1..12 describe timbre for the self-similarity (0 = level)
_TEMPO_AC_S = 8.0             # librosa.feature.tempo ac_size: tempogram window
_TEMPO_RANGE_BPM = (40.0, 240.0)
_TEMPO_PRIOR = (120.0, 1.0)   # log-normal prior (centre BPM, std in octaves) = librosa's default
_TEMPO_RIVAL_MARGIN = 0.2     # best must out-score a non-octave rival tempo by 20 % for full confidence
_ACTIVE_SPAN_DB = -40.0       # frames within 40 dB of a span's loudest frame are "active" (2.0 rule)
_FRAME_FLOOR_DB = -70.0       # frames this far below the track's loudest frame are silence
_EDGE_FLOOR_DB = -60.0        # first/last frame above this (re loudest) bound the audible audio
_SILENT_SEGMENT_DB = -50.0    # segment RMS this far below the track RMS -> silent segment
_RMS_DB_RANGE = (-60.0, 20.0)
_LEVEL_WEIGHT = 0.25          # section labelling: drive = intensity + 0.25 * clip(level_db / 24, -1, 0.25)
_LEVEL_SPAN_DB = 24.0
_FADE_DB = 6.0                # level change across a first/last section that marks a fade-in/out
_OCCUPANCY_DB = -30.0         # mel bands within 30 dB of the frame's loudest band count as occupied
_DENSITY_WEIGHTS = {"fullness": 0.40, "onset_density": 0.35, "percussive_ratio": 0.25}
_VOCAL_SHARE_FULL = 0.6       # in-band harmonic share mapped to 1
_VOCAL_MOVE_FULL = 0.35       # in-band harmonic movement mapped to 1
_FLATNESS_LOG_RANGE = (-5.0, -1.0)   # log10 spectral flatness mapped to 0..1 (noisiness)
_ONSET_MIN_STRENGTH = 0.5    # onsets weaker than this (mean dB rise over mel bands) are noise
_TREND_MIN_R2 = 0.5           # a monotonic trajectory must explain this share of the variance
_STRUCT_WEIGHTS = {"timbre": 1.0, "harmony": 0.8, "dynamics": 1.2}
_NOVELTY_MIN = 0.25           # absolute checkerboard novelty a boundary needs (0..1 scale) ...
_NOVELTY_MAD = 5.0            # ... above the curve's noise floor: median + 5 robust std (MAD) ...
_NOVELTY_REL = 0.30           # ... and relative to the track's strongest boundary
_REPEAT_SIMILARITY = 0.80     # cross / within-section similarity of a repeat candidate
_BLOCK_FRAMES = 1280
# Harmony summary: Krumhansl-Kessler profile correlation (the 2.0 key tables) of STFT chroma,
# for the whole analysed audio and per segment. A SEPARATE measurement: it never replaces the
# cached key-v1 ('key' section), which keeps its conservative window (project memory rule).
HARMONY_METHOD = "kk-stft-chroma-v1"
_HARMONY_MIN_CORR = 0.3        # a segment whose best correlation is below this has no key estimate          # STFT block (~30 s at 22.05 kHz / hop 512): bounded memory
# Decoding robustness (not result parameters): a file this small cannot hold min_audio_s of
# audio, so a header announcing no frames is believed without asking FFmpeg; a decode that
# hangs (stalled network share, pathological file) becomes a decode error.
_EMPTY_FILE_BYTES = 4096
_FFMPEG_TIMEOUT_S = 600.0

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentConfig:
    """Parameters of the segment analysis. Every field is part of ``segments_params``,
    so changing any of them invalidates only the ``segments`` cache section.

    ``sample_rate`` is separate from ``AnalysisConfig.sample_rate`` on purpose: the
    BPM/key analysis is pinned to automix v1 and must not follow changes made here."""

    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    segment_s: float = 5.0
    max_segments: int = 72            # longer audio -> proportionally longer segments
    max_analysis_s: float = 900.0
    min_audio_s: float = 5.0          # less audio -> {"error": ...}
    silence_dbfs: float = -70.0       # whole-audio RMS below this -> {"error": "silent"}
    edge_s: float = 20.0              # energy_in / energy_out window
    bass_cutoff_hz: float = 150.0
    hpss_kernel: int = 31             # librosa default median-filter length (full resolution)
    hpss_time_pool: int = 2           # harmonic filter runs on frames pooled by this factor
    hpss_freq_pool: int = 2           # percussive filter runs on bins pooled by this factor
    block_s: float = 1.0              # structure-analysis resolution
    novelty_kernel_s: float = 8.0     # checkerboard kernel width (both halves)
    min_section_s: float = 8.0
    max_sections: int = 16
    trend_min: float = 0.08           # intensity change over the track that counts as a trend
    wave_min: float = 0.05            # residual oscillation (std) that counts as a wave
    density_bounds: tuple[float, float] = (0.30, 0.55)    # sparse < lo <= moderate < hi <= dense
    vocal_band_hz: tuple[float, float] = (300.0, 3400.0)

    def __post_init__(self) -> None:
        positive = ("sample_rate", "n_fft", "hop_length", "segment_s", "max_segments", "max_analysis_s",
                    "min_audio_s", "edge_s", "bass_cutoff_hz", "hpss_kernel", "hpss_time_pool",
                    "hpss_freq_pool", "block_s", "novelty_kernel_s", "min_section_s", "max_sections",
                    "trend_min", "wave_min")
        for name in positive:
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0):
                raise ValueError(f"SegmentConfig.{name} must be a positive number, got {v!r}")
        # frame / kernel sizes index arrays: a float here would only fail deep inside the analysis
        for name in ("sample_rate", "n_fft", "hop_length", "max_segments", "hpss_kernel", "hpss_time_pool",
                     "hpss_freq_pool", "max_sections"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"SegmentConfig.{name} must be an integer, got {getattr(self, name)!r}")
        if self.hop_length > self.n_fft:
            raise ValueError(f"SegmentConfig.hop_length ({self.hop_length}) must not exceed n_fft ({self.n_fft})")
        if not _is_num(self.silence_dbfs):
            raise ValueError(f"SegmentConfig.silence_dbfs must be a finite number, got {self.silence_dbfs!r}")
        for name in ("density_bounds", "vocal_band_hz"):
            v = getattr(self, name)
            if not (isinstance(v, (tuple, list)) and len(v) == 2 and all(_is_num(x) for x in v)):
                raise ValueError(f"SegmentConfig.{name} must be a pair of numbers, got {v!r}")
            object.__setattr__(self, name, tuple(v))   # a list (e.g. from YAML) would make the config unhashable
        lo, hi = self.density_bounds
        if not 0.0 < lo < hi < 1.0:
            raise ValueError(f"SegmentConfig.density_bounds must satisfy 0 < lo < hi < 1, got {self.density_bounds}")
        vlo, vhi = self.vocal_band_hz
        if not 0.0 <= vlo < vhi:
            raise ValueError(f"SegmentConfig.vocal_band_hz must satisfy 0 <= lo < hi, got {self.vocal_band_hz}")


def segments_params(cfg: SegmentConfig) -> dict:
    """Cache params of the ``segments`` section: every config field plus the fixed
    algorithm constants (incl. the 2.0 energy constants the intensity reuses)."""
    params: dict[str, Any] = {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in dataclasses.asdict(cfg).items()}
    params["algo"] = {
        "energy_weights": dict(ENERGY_WEIGHTS), "onset_rate_scale": ENERGY_ONSET_RATE_SCALE,
        "centroid_hz": list(ENERGY_CENTROID_HZ), "tempo_bpm": list(ENERGY_TEMPO_BPM),
        "n_mels": _N_MELS, "n_mfcc": _N_MFCC, "tempo_ac_s": _TEMPO_AC_S,
        "tempo_range_bpm": list(_TEMPO_RANGE_BPM), "tempo_prior": list(_TEMPO_PRIOR),
        "tempo_rival_margin": _TEMPO_RIVAL_MARGIN,
        "active_span_db": _ACTIVE_SPAN_DB, "frame_floor_db": _FRAME_FLOOR_DB, "edge_floor_db": _EDGE_FLOOR_DB,
        "silent_segment_db": _SILENT_SEGMENT_DB, "rms_db_range": list(_RMS_DB_RANGE),
        "level_weight": _LEVEL_WEIGHT, "level_span_db": _LEVEL_SPAN_DB, "fade_db": _FADE_DB,
        "occupancy_db": _OCCUPANCY_DB, "density_weights": dict(_DENSITY_WEIGHTS),
        "vocal_share_full": _VOCAL_SHARE_FULL, "vocal_move_full": _VOCAL_MOVE_FULL,
        "flatness_log_range": list(_FLATNESS_LOG_RANGE), "struct_weights": dict(_STRUCT_WEIGHTS),
        "onset_min_strength": _ONSET_MIN_STRENGTH, "trend_min_r2": _TREND_MIN_R2,
        "novelty_min": _NOVELTY_MIN, "novelty_mad": _NOVELTY_MAD, "novelty_rel": _NOVELTY_REL,
        "repeat_similarity": _REPEAT_SIMILARITY,
        "harmony": HARMONY_METHOD,
    }
    return params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_segments(path: Path, cfg: SegmentConfig | None = None, *,
                     tools: FFmpegTools | None = None) -> dict:
    """Uncached segment analysis of ``path`` (see the module docstring for the method).

    Never raises for undecodable, short or silent audio: the result is then
    ``{"version", "error": str, "error_kind": "decode"|"too_short"|"silent"|"analysis",
    "duration_s", "analysed_s", "notes"}``. ``FFmpegNotFoundError`` propagates (a file
    libsndfile cannot read needs FFmpeg). The value is JSON-native (no numpy types, no NaN).
    """
    cfg = cfg or SegmentConfig()
    path = Path(path)
    try:
        dec = _decode(path, cfg, tools)
    except FFmpegNotFoundError:
        raise
    except _DecodeError as e:
        return _error_value("decode", str(e), None, 0.0, [])
    y, notes = dec.y, list(dec.notes)
    sr = cfg.sample_rate
    analysed = len(y) / sr
    if analysed < cfg.min_audio_s:
        # floored, so 4.99 s is not reported as "5.0 s (< 5 s)"
        return _error_value("too_short", f"only {math.floor(analysed * 10) / 10:.1f} s of audio "
                            f"(< {cfg.min_audio_s:g} s)", dec.duration_s, analysed, notes)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        notes.append("non-finite samples in the decoded audio were replaced by 0")
    rms = math.sqrt(_sum_squares(y) / len(y))
    if not math.isfinite(rms) or rms <= 10 ** (cfg.silence_dbfs / 20):
        return _error_value("silent", f"silent (RMS below {cfg.silence_dbfs:g} dBFS)",
                            dec.duration_s, analysed, notes)
    y *= np.float32(1.0 / rms)        # unit RMS, in place (the decoded array is ours)
    try:
        return _analyse(y, dec, cfg, notes)
    except Exception as e:  # an unexpected numeric failure must not take the library run down
        log.warning("segment analysis failed for %s (%s: %s); no segment data for this track",
                    path, type(e).__name__, _first_line(e))
        log.debug("segment analysis traceback for %s", path, exc_info=True)
        return _error_value("analysis", f"segment analysis failed: {type(e).__name__}: {_first_line(e)}",
                            dec.duration_s, analysed, notes)


def analyze_segments(path: Path, cache: AnalysisCache, cfg: SegmentConfig | None = None, *,
                     tools: FFmpegTools | None = None, compute: bool = True,
                     expected_fingerprint: dict | None = None) -> tuple[dict | None, bool]:
    """Cached segment analysis: ``(value, from_cache)``.

    Section ``SEGMENTS_SECTION``, version ``SEGMENTS_VERSION``, params
    ``segments_params(cfg)``. ``compute=False`` is a cache-only lookup: ``(None, False)``
    when nothing valid is cached. Short / silent results are cached ("unknown" is a
    result); decode and unexpected analysis failures are returned but NOT cached (they
    may be transient). ``expected_fingerprint`` (``cache.fingerprint(path)`` taken before
    the file was read) guards against storing a result of a file that changed while it
    was analysed; when omitted, it is taken here before decoding. A read-only cache keeps
    the value in memory only (``save()`` is a no-op there)."""
    cfg = cfg or SegmentConfig()
    path = Path(path)
    params = segments_params(cfg)
    hit = cache.get(path, SEGMENTS_SECTION, version=SEGMENTS_VERSION, params=params)
    if hit is not None:
        if is_valid_segments(hit):
            return hit, True
        log.warning("cache: malformed %s value for %s; %s", SEGMENTS_SECTION, path,
                    "recomputing" if compute else "ignored")
    if not compute:
        return None, False
    fp0 = expected_fingerprint if expected_fingerprint is not None else cache.fingerprint(path)
    value = compute_segments(path, cfg, tools=tools)
    if value.get("error_kind") not in ("decode", "analysis"):
        try:
            cache.put(path, SEGMENTS_SECTION, value, version=SEGMENTS_VERSION, params=params,
                      expected_fingerprint=fp0)
        except FileNotFoundError:   # the source vanished meanwhile: nothing to attach it to
            log.warning("cache: %s disappeared during the segment analysis; result not cached", path)
    return value, False


def is_valid_segments(value: Any) -> bool:
    """Structural check of a segments value (cache hits are validated with it)."""
    if not isinstance(value, dict) or value.get("version") != SEGMENTS_VERSION:
        return False
    if "error" in value:
        return isinstance(value["error"], str)
    times = value.get("times")
    if not isinstance(times, list) or not times:
        return False
    for key in ("intensity", "rms_db", "onset_rate", "percussive_ratio", "bass_ratio", "centroid_hz", "flatness"):
        arr = value.get(key)
        if not (isinstance(arr, list) and len(arr) == len(times) and all(_is_num(v) for v in arr)):
            return False
    summary = value.get("summary")
    return (isinstance(summary, dict) and isinstance(value.get("sections"), list)
            and isinstance(value.get("notes"), list)
            and all(_is_num(summary.get(k)) for k in ("energy_in", "energy_out", "energy_mean", "energy_peak")))


def timbre_similarity(a: dict | Sequence[float] | None, b: dict | Sequence[float] | None) -> float | None:
    """Similarity 0..1 of two ``summary["timbre"]`` vectors (or two segments values):
    ``1 - euclidean distance / sqrt(dims)``. None when either side is unknown."""
    va, vb = _timbre_of(a), _timbre_of(b)
    if va is None or vb is None or len(va) != len(vb):
        return None
    dist = math.dist(va, vb)       # scaled norm: no OverflowError on absurd (non 0..1) inputs
    return round(max(0.0, 1.0 - dist / math.sqrt(len(va))), 4)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

class _DecodeError(Exception):
    """The file could not be decoded (reported as an error value, never raised to callers)."""


@dataclass
class _Decoded:
    y: np.ndarray
    duration_s: float | None     # full duration (None when FFmpeg truncated and gave no duration)
    notes: list[str]


def _decode(path: Path, cfg: SegmentConfig, tools: FFmpegTools | None) -> _Decoded:
    """Mono float32 at ``cfg.sample_rate``, at most ``cfg.max_analysis_s`` seconds."""
    try:
        readable = path.is_file()
    except OSError as e:   # Path.is_file only swallows "not found"-type errors; access denied raises
        raise _DecodeError(f"cannot access the file ({type(e).__name__}: {e.strerror or e})") from e
    if not readable:
        raise _DecodeError(f"not a readable file: {path}")
    sr, cap = cfg.sample_rate, cfg.max_analysis_s
    notes: list[str] = []
    expected_s: float | None = None    # what libsndfile announced, when its own decode failed
    try:
        info = sf.info(str(path))
    except Exception:   # libsndfile cannot open it (m4a/aac/opus/... or damaged): FFmpeg below
        info = None
    if info is not None and info.samplerate > 0 and info.frames == 0:
        if _file_size(path) <= _EMPTY_FILE_BYTES:
            return _Decoded(np.zeros(0, dtype=np.float32), 0.0, notes)
        # a header that claims no audio in front of real data (a recorder that never
        # finalised the WAV data size): FFmpeg reads up to the end of the file
        notes.append("libsndfile reports no audio frames; decoded with FFmpeg")
    if info is not None and info.frames > 0 and info.samplerate > 0:
        try:
            y = _read_mono(path, sr, cap)
            full = info.frames / info.samplerate
            analysed = len(y) / sr
            truncated = full > cap + 0.05 and analysed >= cap - 0.05
            if truncated:
                notes.append(f"analysed the first {analysed:.0f} s of {full:.0f} s (max_analysis_s)")
            return _Decoded(y, round(full if truncated else analysed, 3), notes)
        except Exception as e:   # damaged data behind a valid header: FFmpeg is more tolerant
            notes.append(f"libsndfile could not decode the file ({type(e).__name__}); decoded with FFmpeg")
            expected_s = min(info.frames / info.samplerate, cap)
    tools = tools or find_ffmpeg()   # FFmpegNotFoundError propagates on purpose
    try:
        # cleanup errors (an antivirus holding the temp WAV) must not discard a finished decode
        with tempfile.TemporaryDirectory(prefix="veg_seg_", ignore_cleanup_errors=True) as td:
            wav = Path(td) / "decoded.wav"
            try:
                # no duration is assumed: -t only caps the decode at max_analysis_s
                p = run_ffmpeg(tools, ["-nostats", "-y", "-i", str(path), "-map", "0:a:0", "-vn",
                                       "-t", f"{cap:.3f}", "-ac", "1", "-ar", str(sr),
                                       "-c:a", "pcm_f32le", "-rf64", "auto", str(wav)],
                               timeout=_FFMPEG_TIMEOUT_S)
            except FFmpegError as e:
                raise _DecodeError(f"FFmpeg could not decode the file: {_first_line(e)}") from e
            try:
                y, file_sr = sf.read(str(wav), dtype="float32", always_2d=True)
            except Exception as e:
                raise _DecodeError(f"FFmpeg produced no readable audio: {type(e).__name__}") from e
    except OSError as e:   # temporary folder not creatable / removable
        raise _DecodeError(f"temporary decode file failed: {type(e).__name__}: {_first_line(e)}") from e
    y = y.mean(axis=1) if y.shape[1] > 1 else y[:, 0]
    if int(file_sr) != sr and len(y):
        y = librosa.resample(y, orig_sr=int(file_sr), target_sr=sr)
    y = np.ascontiguousarray(y, dtype=np.float32)
    analysed = len(y) / sr
    if expected_s is not None and analysed < expected_s - 1.0:
        # FFmpeg skips what it cannot decode: say so instead of passing a partial decode off as the track
        notes.append(f"FFmpeg decoded only {analysed:.1f} s of the {expected_s:.1f} s the header announces "
                     f"(damaged file?)")
    m = _DURATION_RE.search(p.stderr or "")
    banner = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None
    if analysed >= cap - 0.05:
        if banner is not None and banner > analysed + 0.05:
            notes.append(f"analysed the first {analysed:.0f} s of {banner:.0f} s (max_analysis_s)")
            return _Decoded(y, round(banner, 3), notes)
        if banner is None or banner < analysed - 0.05:   # no (or an evidently wrong) input duration
            notes.append(f"analysed the first {analysed:.0f} s (max_analysis_s); the full duration is unknown")
            return _Decoded(y, None, notes)
        # the input is max_analysis_s long (within 50 ms): nothing was cut off
    return _Decoded(y, round(analysed, 3), notes)


def _read_mono(path: Path, sr: int, cap: float) -> np.ndarray:
    """``librosa.load(path, sr=sr, mono=True, duration=cap)`` (same frames, channel mean
    and soxr_hq resampling) read in 10 s blocks, so a long multichannel file never sits
    in memory at its native width."""
    with sf.SoundFile(str(path)) as f:
        native = int(f.samplerate)
        want = int(cap * native)
        hint = min(want, max(0, int(f.frames)))
        buf = np.empty(hint, dtype=np.float32)
        extra: list[np.ndarray] = []
        filled = 0
        for block in f.blocks(blocksize=10 * native, frames=want, dtype="float32", always_2d=True):
            mono = block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]
            take = min(mono.size, hint - filled)
            buf[filled:filled + take] = mono[:take]
            filled += take
            if take < mono.size:          # frame count was an estimate (MP3): keep the rest
                extra.append(mono[take:].copy())
    y = np.concatenate([buf[:filled], *extra]) if extra else buf[:filled]
    if native != sr and y.size:
        y = librosa.resample(y, orig_sr=native, target_sr=sr, res_type="soxr_hq")
    return np.ascontiguousarray(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Frame features (block-wise STFT + HPSS)
# ---------------------------------------------------------------------------

@dataclass
class _Frames:
    n: int
    tot: np.ndarray        # power per frame
    bass: np.ndarray       # power below the bass cutoff
    harm: np.ndarray       # HPSS harmonic power
    perc: np.ndarray       # HPSS percussive power
    centroid: np.ndarray   # spectral centroid (Hz), 0 for empty frames
    flatness: np.ndarray   # spectral flatness (power)
    mel: np.ndarray        # (n_mels, n) mel power
    chroma: np.ndarray     # (12, n) chroma power
    vmid: np.ndarray       # harmonic power in the vocal band
    vflux: np.ndarray      # in-band harmonic spectral movement 0..1 (vs the previous frame)
    sq: np.ndarray         # sum of squared samples in [k*hop, (k+1)*hop) (unwindowed, for RMS)
    count: np.ndarray      # number of samples behind ``sq`` (hop, fewer for the last frame)


def _odd(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 else k + 1


def _hpss_masks(S: np.ndarray, cfg: SegmentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Soft masks (power 2) of librosa's median-filter HPSS with pooled filtering: the
    harmonic median runs along time on ``hpss_time_pool``-pooled frames, the percussive
    median along frequency on ``hpss_freq_pool``-pooled bins (pool 1 = librosa exactly)."""
    tk, fk, k = cfg.hpss_time_pool, cfg.hpss_freq_pool, cfg.hpss_kernel
    P = S * S
    nbins, nt = S.shape
    idx_t = np.arange(0, nt, tk)
    cnt_t = np.diff(np.append(idx_t, nt))
    ht = np.sqrt(np.add.reduceat(P, idx_t, axis=1) / cnt_t)
    ht = scipy.ndimage.median_filter(ht, size=(1, _odd(k // tk)), mode="reflect")
    H = np.repeat(ht, cnt_t, axis=1)
    idx_f = np.arange(0, nbins, fk)
    cnt_f = np.diff(np.append(idx_f, nbins))
    pf = np.sqrt(np.add.reduceat(P, idx_f, axis=0) / cnt_f[:, None])
    pf = scipy.ndimage.median_filter(pf, size=(_odd(k // fk), 1), mode="reflect")
    Pm = np.repeat(pf, cnt_f, axis=0)
    h2, p2 = H * H, Pm * Pm
    den = h2 + p2
    ok = den > 0
    # both estimates zero: split evenly, like librosa.util.softmask(split_zeros=True)
    mask_h = np.divide(h2, den, out=np.full_like(den, 0.5), where=ok)
    mask_p = np.divide(p2, den, out=np.full_like(den, 0.5), where=ok)
    return mask_h, mask_p


def _frame_features(y: np.ndarray, cfg: SegmentConfig, *, block_frames: int = _BLOCK_FRAMES) -> _Frames:
    """Per-frame features of ``y`` (centred frames, identical to ``librosa.stft(y)``),
    computed in blocks with enough overlap that the result does not depend on the block
    size (HPSS needs neighbouring frames, the movement measure the previous frame)."""
    sr, n_fft, hop = cfg.sample_rate, cfg.n_fft, cfg.hop_length
    tk = cfg.hpss_time_pool
    n = 1 + len(y) // hop
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=_N_MELS)
    chroma_fb = librosa.filters.chroma(sr=sr, n_fft=n_fft, tuning=0.0)
    bass_bins = freqs < cfg.bass_cutoff_hz
    vlo, vhi = cfg.vocal_band_hz
    vbins = (freqs >= vlo) & (freqs <= vhi)
    margin = tk * (_odd(cfg.hpss_kernel // tk) // 2 + 1)
    block = max(tk, (int(block_frames) // tk) * tk)

    out = _Frames(n=n, tot=np.zeros(n), bass=np.zeros(n), harm=np.zeros(n), perc=np.zeros(n),
                  centroid=np.zeros(n), flatness=np.zeros(n),
                  mel=np.zeros((_N_MELS, n), dtype=np.float32), chroma=np.zeros((12, n), dtype=np.float32),
                  vmid=np.zeros(n), vflux=np.zeros(n), sq=np.zeros(n), count=np.zeros(n))
    for f0 in range(0, n, block):
        f1 = min(n, f0 + block)
        e0, e1 = max(0, f0 - margin), min(n, f1 + margin)
        # sample energy per hop (unwindowed): RMS of segments / blocks / sections
        s0, s1 = f0 * hop, min(len(y), f1 * hop)
        full = (s1 - s0) // hop
        ys = y[s0:s0 + full * hop].reshape(full, hop)
        out.sq[f0:f0 + full] = np.einsum("ij,ij->i", ys, ys, dtype=np.float64)
        out.count[f0:f0 + full] = hop
        if f0 + full < f1:                 # the last, partial hop of the track
            rest = y[s0 + full * hop:s1]
            out.sq[f0 + full] = float(np.dot(rest.astype(np.float64), rest))
            out.count[f0 + full] = rest.size
        # centred frames of y without a padded copy of the whole signal (librosa.stft(y) frames)
        seg = _padded_slice(y, e0 * hop, (e1 - 1) * hop + n_fft, n_fft // 2)
        S = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=hop, center=False))
        mask_h, mask_p = _hpss_masks(S, cfg)
        # in-band harmonic movement needs the previous frame: computed on the extended block
        hm = S[vbins] * mask_h[vbins]
        num = np.abs(np.diff(hm, axis=1)).sum(axis=0)
        den = (hm[:, 1:] + hm[:, :-1]).sum(axis=0)
        flux = np.zeros(e1 - e0)
        flux[1:] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        sl = slice(f0 - e0, f1 - e0)
        S, mask_h, mask_p, hm, flux = S[:, sl], mask_h[:, sl], mask_p[:, sl], hm[:, sl], flux[sl]
        P = S * S
        tot = P.sum(axis=0, dtype=np.float64)
        out.tot[f0:f1] = tot
        out.bass[f0:f1] = P[bass_bins].sum(axis=0, dtype=np.float64)
        out.harm[f0:f1] = (P * mask_h * mask_h).sum(axis=0, dtype=np.float64)
        out.perc[f0:f1] = (P * mask_p * mask_p).sum(axis=0, dtype=np.float64)
        mag = S.sum(axis=0, dtype=np.float64)
        out.centroid[f0:f1] = np.divide(freqs @ S, mag, out=np.zeros_like(mag), where=mag > 0)
        Pf = np.maximum(P, 1e-10)
        out.flatness[f0:f1] = np.exp(np.log(Pf).mean(axis=0)) / Pf.mean(axis=0)
        out.mel[:, f0:f1] = mel_fb @ P
        out.chroma[:, f0:f1] = chroma_fb @ P
        out.vmid[f0:f1] = (hm * hm).sum(axis=0, dtype=np.float64)
        out.vflux[f0:f1] = flux      # frame 0 of the track has no predecessor: 0
    return out


def _padded_slice(y: np.ndarray, start: int, stop: int, pad: int) -> np.ndarray:
    """``np.pad(y, pad)[start:stop]`` (zero padding) without copying the whole signal."""
    out = np.zeros(stop - start, dtype=y.dtype)
    lo, hi = max(start - pad, 0), min(stop - pad, len(y))
    if hi > lo:
        out[lo - (start - pad):hi - (start - pad)] = y[lo:hi]
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class _Ctx:
    """Frame features + onsets of the analysed audio with O(1) span sums."""

    def __init__(self, fr: _Frames, cfg: SegmentConfig) -> None:
        self.cfg, self.fr = cfg, fr
        self.sr, self.hop = cfg.sample_rate, cfg.hop_length
        self.n = fr.n
        self.mel_db = librosa.power_to_db(fr.mel)
        self.oenv = librosa.onset.onset_strength(S=self.mel_db, sr=self.sr, hop_length=self.hop)
        onsets = librosa.onset.onset_detect(onset_envelope=self.oenv, sr=self.sr, hop_length=self.hop)
        onsets = onsets[(onsets >= 0) & (onsets < self.n)]
        # onset_detect normalises the envelope to its maximum, so on near-static material
        # (a held pad) numerical noise becomes "onsets"; real onsets are >= ~2 here
        onsets = onsets[self.oenv[onsets] >= _ONSET_MIN_STRENGTH]
        mask = np.zeros(self.n, dtype=np.int64)
        mask[onsets] = 1
        self.on_cum = np.concatenate([[0], np.cumsum(mask)])
        self.ac = int(librosa.time_to_frames(_TEMPO_AC_S, sr=self.sr, hop_length=self.hop))
        # mean tempogram over all frames (what librosa.feature.tempo aggregates) and over the
        # frames that carry an onset signal (the tempo summary's view)
        self.tg_mean, self.tg_active_mean = _tempogram_means(self.oenv, self.ac)
        # the 2.0 estimator (librosa.feature.tempo, prior at 120 BPM) for the intensity's tempo term
        self.tempo_2_0 = float(np.atleast_1d(librosa.feature.tempo(
            onset_envelope=self.oenv, sr=self.sr, hop_length=self.hop, tg=self.tg_mean[:, None]))[0])
        self.cs = {k: np.concatenate([[0.0], np.cumsum(getattr(fr, k))])
                   for k in ("tot", "bass", "harm", "perc", "sq", "count")}
        peak = float(fr.tot.max()) if self.n else 0.0
        self.frame_floor = peak * 10 ** (_FRAME_FLOOR_DB / 10)

    def frame_of(self, t: float) -> int:
        """First frame whose centre is at or after ``t`` seconds."""
        return int(min(self.n, max(0, math.ceil(t * self.sr / self.hop - 1e-9))))

    def ssum(self, key: str, a: int, b: int) -> float:
        cs = self.cs[key]
        return float(cs[b] - cs[a])

    def rms(self, t0: float, t1: float) -> float:
        """RMS of the samples in [t0, t1) at hop resolution (~23 ms)."""
        a = int(min(self.n, max(0, round(t0 * self.sr / self.hop))))
        b = int(min(self.n, max(a, round(t1 * self.sr / self.hop))))
        count = self.ssum("count", a, b)
        return math.sqrt(self.ssum("sq", a, b) / count) if count > 0 else 0.0

    def span(self, a: int, b: int, dur_s: float) -> dict[str, float]:
        """The 2.0 energy heuristic (and its parts) over frames [a, b)."""
        lo, hi = ENERGY_CENTROID_HZ
        t_lo, t_hi = ENERGY_TEMPO_BPM
        dur_s = max(dur_s, 1e-6)
        rate = float(self.on_cum[b] - self.on_cum[a]) / dur_s
        onset_density = 1.0 - math.exp(-rate / ENERGY_ONSET_RATE_SCALE)
        h, p = self.ssum("harm", a, b), self.ssum("perc", a, b)
        pr = p / (h + p) if (h + p) > 0 else 0.0
        tot = self.ssum("tot", a, b)
        bass_ratio = self.ssum("bass", a, b) / tot if tot > 0 else 0.0
        fp = self.fr.tot[a:b]
        if fp.size and fp.max() > self.frame_floor:
            active = (fp >= fp.max() * 10 ** (_ACTIVE_SPAN_DB / 10)) & (fp > self.frame_floor)
            c_hz = float(np.median(self.fr.centroid[a:b][active]))
            flat = float(np.median(self.fr.flatness[a:b][active]))
        else:
            c_hz, flat = lo, 0.0
        brightness = _clip01(math.log(max(c_hz, 1.0) / lo) / math.log(hi / lo))
        tempo = _clip01((self.tempo_2_0 - t_lo) / (t_hi - t_lo)) * onset_density
        comps = {"onset_density": onset_density, "percussive_ratio": pr, "brightness": brightness, "tempo": tempo}
        intensity = _clip01(sum(ENERGY_WEIGHTS[k] * _clip01(v) for k, v in comps.items()))
        return {"intensity": intensity, "onset_rate": rate, "onset_density": onset_density,
                "percussive_ratio": pr, "bass_ratio": bass_ratio, "centroid_hz": c_hz, "flatness": flat}


def _tempogram_means(oenv: np.ndarray, win: int, *, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray | None]:
    """Frame-means of ``librosa.feature.tempogram(onset_envelope=oenv, win_length=win)``
    (centred, Hann, inf-norm), accumulated over chunks of frames instead of materialising
    the (win x frames) matrix: (mean over all frames, mean over frames whose normalised
    lag-0 value is set, i.e. that carry an onset signal — None when there are none)."""
    n = oenv.shape[-1]
    window = scipy.signal.get_window("hann", win, fftbins=True)
    padded = np.pad(oenv, (win // 2, win // 2), mode="linear_ramp", end_values=(0, 0))
    total = np.zeros(win)
    active = np.zeros(win)
    n_active = 0
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        frames = librosa.util.frame(padded[c0:c1 - 1 + win], frame_length=win, hop_length=1)
        tg = librosa.util.normalize(librosa.autocorrelate(frames * window[:, None], axis=-2), norm=np.inf, axis=-2)
        total += tg.sum(axis=1)
        good = tg[0] > 0.5
        active += tg[:, good].sum(axis=1)
        n_active += int(good.sum())
    return total / max(n, 1), (active / n_active if n_active else None)


def _analyse(y: np.ndarray, dec: _Decoded, cfg: SegmentConfig, notes: list[str]) -> dict:
    sr = cfg.sample_rate
    analysed = len(y) / sr
    ctx = _Ctx(_frame_features(y, cfg), cfg)   # everything below works on frame features

    # --- fixed-length segments ---------------------------------------------
    seg_len = max(cfg.segment_s, analysed / cfg.max_segments)
    if seg_len > cfg.segment_s + 1e-9:
        notes.append(f"segment length raised to {seg_len:.2f} s (at most {cfg.max_segments} segments)")
    starts = _segment_starts(analysed, seg_len)
    ends = starts[1:] + [analysed]
    silent_rms = 10 ** (_SILENT_SEGMENT_DB / 20)      # y has unit RMS
    rows: list[dict[str, float]] = []
    rms_seg: list[float] = []
    for t0, t1 in zip(starts, ends):
        r = ctx.rms(t0, t1)
        rms_seg.append(r)
        if r < silent_rms:
            rows.append({"intensity": 0.0, "onset_rate": 0.0, "percussive_ratio": 0.0, "bass_ratio": 0.0,
                         "centroid_hz": 0.0, "flatness": 0.0, "silent": 1.0})
        else:
            rows.append({**ctx.span(ctx.frame_of(t0), ctx.frame_of(t1), t1 - t0), "silent": 0.0})
    silent = np.array([row["silent"] > 0 for row in rows])
    if silent.any():
        notes.append(f"{int(silent.sum())} silent segment(s): features 0, rms_db at the floor")
    rms_arr = np.array(rms_seg)
    ref = float(np.median(rms_arr[~silent])) if (~silent).any() else 1.0
    rms_db = np.clip(20 * np.log10(np.maximum(rms_arr, 1e-12) / ref), *_RMS_DB_RANGE)
    intensity = np.array([row["intensity"] for row in rows])
    centers = np.array([(a + b) / 2 for a, b in zip(starts, ends)])
    durs = np.array([b - a for a, b in zip(starts, ends)])

    # --- summary ----------------------------------------------------------
    e_in, e_out = _edge_energies(ctx, cfg)
    act = ~silent
    energy_mean = float(np.average(intensity[act], weights=durs[act])) if act.any() else 0.0
    energy_peak = float(_moving_mean(intensity, 3).max()) if len(intensity) else 0.0
    trajectory = _trajectory(centers, intensity, act, e_in, e_out, cfg)
    track = _track_features(ctx)
    density = _density(track, cfg)
    dr = rms_db[act]
    dynamic_range = float(np.percentile(dr, 95) - np.percentile(dr, 10)) if dr.size >= 2 else 0.0
    tempo = _tempo_summary(ctx, track["onset_rate"], analysed)
    if tempo["bpm"] is None:
        notes.append("no clear periodic pulse: tempo summary unknown")
    if analysed < 30.0:
        notes.append(f"short audio ({analysed:.0f} s): trajectory and sections are not meaningful")
    vocal = _vocal_proxy(ctx, track)
    harmony = _harmony_summary(ctx, starts, ends, silent)
    summary = {
        "energy_in": _r(e_in, 3), "energy_out": _r(e_out, 3),
        "energy_mean": _r(energy_mean, 3), "energy_peak": _r(energy_peak, 3),
        "trajectory": trajectory, "density": density,
        "percussive_density": _r(math.sqrt(track["percussive_ratio"] * track["onset_density"]), 3),
        "bass_density": _r(track["bass_ratio"], 3),
        "dynamic_range_db": _r(dynamic_range, 1),
        "tempo": tempo,
        "vocal_presence_proxy": vocal,
        "timbre": [_r(track[k], 3) for k in TIMBRE_DIMENSIONS],
        "harmony": harmony,
    }
    sections = _sections(ctx, cfg, ref, analysed)
    return {
        "version": SEGMENTS_VERSION,
        "duration_s": dec.duration_s,
        "analysed_s": _r(analysed, 3),
        "segment_s": _r(seg_len, 3),
        "times": [_r(t, 2) for t in starts],
        "intensity": [_r(v, 3) for v in intensity],
        "rms_db": [_r(v, 1) for v in rms_db],
        "onset_rate": [_r(row["onset_rate"], 2) for row in rows],
        "percussive_ratio": [_r(row["percussive_ratio"], 3) for row in rows],
        "bass_ratio": [_r(row["bass_ratio"], 3) for row in rows],
        "centroid_hz": [int(round(row["centroid_hz"])) for row in rows],
        "flatness": [_r(row["flatness"], 4) for row in rows],
        "summary": summary,
        "sections": sections,
        "notes": notes,
    }


def _kk_key(chroma: np.ndarray) -> tuple[str | None, float, float]:
    """(camelot, best correlation, margin to the best DIFFERENT key) of a 12-bin chroma
    against the Krumhansl-Kessler profiles; (None, corr, 0) for a flat / silent chroma."""
    c = np.asarray(chroma, dtype=np.float64)
    if not np.isfinite(c).all() or c.std() <= 1e-12:
        return None, 0.0, 0.0
    scores = []
    for shift in range(12):
        shifted = np.roll(c, -shift)
        for mode, prof in ((1, MAJOR_PROFILE), (0, MINOR_PROFILE)):
            scores.append((float(np.corrcoef(shifted, prof)[0, 1]), shift, mode))
    scores.sort(key=lambda x: -x[0])
    best, second = scores[0], scores[1]
    cam = CAMELOT_MAP.get((best[1], best[2]))
    return cam, best[0], best[0] - second[0]


def _camelot_distance(a: str, b: str) -> int:
    na, la, nb, lb = int(a[:-1]), a[-1], int(b[:-1]), b[-1]
    d = abs(na - nb) % 12
    return min(d, 12 - d) + (la != lb)


def _harmony_summary(ctx: "_Ctx", starts: list[float], ends: list[float], silent: np.ndarray) -> dict:
    """Key clarity and stability over time (spec §2 "harmonic characteristics"):

    * ``key_estimate`` / ``clarity`` / ``margin`` — best Krumhansl-Kessler key of the mean
      chroma of the analysed audio, its correlation, and how far it beats the next key;
    * ``segment_keys`` — the best key per segment (None when silent or unclear);
    * ``stability`` — share of segments with a key within one Camelot step of the estimate;
    * ``modulations`` — changes of the segment key by two or more steps that persist for at
      least two segments (a candidate, not a musical analysis).

    It is a measurement of the whole track; the planner keeps using the cached key-v1."""
    chroma = ctx.fr.chroma.astype(np.float64)
    out: dict[str, Any] = {"method": HARMONY_METHOD, "key_estimate": None, "clarity": 0.0, "margin": 0.0,
                           "stability": None, "modulations": 0, "segment_keys": []}
    if chroma.size == 0:
        return out
    key, corr, margin = _kk_key(chroma.sum(axis=1))
    out.update({"key_estimate": key, "clarity": _r(max(0.0, corr), 3), "margin": _r(max(0.0, margin), 3)})
    seg_keys: list[str | None] = []
    for (t0, t1), quiet in zip(zip(starts, ends), silent):
        f0, f1 = ctx.frame_of(t0), max(ctx.frame_of(t0) + 1, ctx.frame_of(t1))
        if quiet or f1 <= f0:
            seg_keys.append(None)
            continue
        k, c, _ = _kk_key(chroma[:, f0:f1].sum(axis=1))
        seg_keys.append(k if (k is not None and c >= _HARMONY_MIN_CORR) else None)
    out["segment_keys"] = seg_keys
    known = [k for k in seg_keys if k is not None]
    if key is not None and known:
        out["stability"] = _r(sum(1 for k in known if _camelot_distance(k, key) <= 1) / len(known), 3)
        modulations, current = 0, known[0]
        for i in range(1, len(known)):
            k = known[i]
            # a move of two or more steps counts when the next segment stays near the new key
            if _camelot_distance(k, current) >= 2 and i + 1 < len(known)                     and _camelot_distance(known[i + 1], k) <= 1:
                modulations += 1
                current = k
        out["modulations"] = modulations
    return out


def _segment_starts(analysed: float, seg_len: float) -> list[float]:
    """Start times of fixed-length segments; a tail shorter than half a segment joins
    the previous one."""
    n = max(1, int(analysed // seg_len))
    if analysed - n * seg_len >= seg_len / 2:
        n += 1
    return [k * seg_len for k in range(n)]


def _edge_energies(ctx: _Ctx, cfg: SegmentConfig) -> tuple[float, float]:
    """Intensity of the first / last ``edge_s`` seconds of audible audio (leading and
    trailing near-silence excluded)."""
    tot = ctx.fr.tot
    audible = np.flatnonzero(tot >= tot.max() * 10 ** (_EDGE_FLOOR_DB / 10))
    first, last = (int(audible[0]), int(audible[-1]) + 1) if audible.size else (0, ctx.n)
    edge = max(1, int(round(cfg.edge_s * ctx.sr / ctx.hop)))
    frame_s = ctx.hop / ctx.sr
    a_in, b_in = first, min(last, first + edge)
    a_out, b_out = max(first, last - edge), last
    e_in = ctx.span(a_in, b_in, (b_in - a_in) * frame_s)["intensity"]
    e_out = ctx.span(a_out, b_out, (b_out - a_out) * frame_s)["intensity"]
    return e_in, e_out


def _trajectory(centers: np.ndarray, intensity: np.ndarray, active: np.ndarray, e_in: float, e_out: float,
                cfg: SegmentConfig) -> dict:
    """rising / falling / stable / wave from the smoothed intensity curve of the
    non-silent segments: a linear trend (total change over the track, confirmed by the
    edge energies) unless a residual oscillation with >= 2 turning points dominates."""
    x = centers[active] / 60.0
    v = intensity[active]
    if v.size < 4 or x[-1] - x[0] <= 0:
        return {"value": None, "confidence": 0.0, "slope_per_min": None, "oscillation": None}
    sm = _moving_mean(v, 3)
    slope, icpt = np.polyfit(x, sm, 1)
    total = float(slope * (x[-1] - x[0]))
    edge = e_out - e_in
    resid = sm - (slope * x + icpt)
    osc = float(np.std(resid))
    ss_tot = float(np.sum((sm - sm.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 1e-12 else 0.0
    turns = (len(scipy.signal.find_peaks(sm, prominence=cfg.wave_min)[0])
             + len(scipy.signal.find_peaks(-sm, prominence=cfg.wave_min)[0]))
    length = _clip01(v.size / 12)
    # a trend must be large, explain most of the curve's variance and agree with the edges
    up = total >= cfg.trend_min and edge >= cfg.trend_min / 2 and r2 >= _TREND_MIN_R2
    down = total <= -cfg.trend_min and edge <= -cfg.trend_min / 2 and r2 >= _TREND_MIN_R2
    if (up or down) and (osc < 0.6 * abs(total) or turns < 2):
        value = "rising" if up else "falling"
        conf = (0.35 + 0.45 * _clip01((abs(total) - cfg.trend_min) / (2 * cfg.trend_min))) * (0.6 + 0.4 * r2)
    elif osc >= cfg.wave_min and turns >= 2:
        value = "wave"
        conf = 0.3 + 0.4 * _clip01((osc - cfg.wave_min) / cfg.wave_min) * _clip01(turns / 4)
    else:
        value = "stable"
        conf = 0.3 + 0.4 * (1.0 - _clip01(max(abs(total) / cfg.trend_min, osc / cfg.wave_min)))
    conf = min(0.8, conf * (0.5 + 0.5 * length))
    return {"value": value, "confidence": _r(conf, 2), "slope_per_min": _r(float(slope), 3),
            "oscillation": _r(osc, 3)}


def _track_features(ctx: _Ctx) -> dict[str, float]:
    """Whole-track texture over audible frames: the timbre dimensions plus the
    ingredients of density, percussive/bass density and the vocal proxy."""
    fr = ctx.fr
    act = fr.tot > ctx.frame_floor
    if not act.any():
        act = np.ones(ctx.n, dtype=bool)
    active_s = float(act.sum()) * ctx.hop / ctx.sr
    lo, hi = ENERGY_CENTROID_HZ
    c_hz = float(np.median(fr.centroid[act]))
    flat = float(np.median(fr.flatness[act]))
    f_lo, f_hi = _FLATNESS_LOG_RANGE
    tot = float(fr.tot[act].sum())
    h, p = float(fr.harm[act].sum()), float(fr.perc[act].sum())
    on = float(ctx.on_cum[-1])
    rate = on / max(active_s, 1e-6)
    mel = fr.mel[:, act]
    occupied = mel >= mel.max(axis=0, keepdims=True) * 10 ** (_OCCUPANCY_DB / 10)
    return {
        "brightness": _clip01(math.log(max(c_hz, 1.0) / lo) / math.log(hi / lo)),
        "noisiness": _clip01((math.log10(max(flat, 1e-12)) - f_lo) / (f_hi - f_lo)),
        "bass": fr.bass[act].sum() / tot if tot > 0 else 0.0,
        "percussive": p / (h + p) if (h + p) > 0 else 0.0,
        "onset_density": 1.0 - math.exp(-rate / ENERGY_ONSET_RATE_SCALE),
        "fullness": float(occupied.mean()) if occupied.size else 0.0,
        "onset_rate": rate,
        "percussive_ratio": p / (h + p) if (h + p) > 0 else 0.0,
        "bass_ratio": fr.bass[act].sum() / tot if tot > 0 else 0.0,
        "harm": h, "active_s": active_s, "active_mask": act,   # type: ignore[dict-item]
    }


def _density(track: dict, cfg: SegmentConfig) -> dict:
    """sparse / moderate / dense from mel-band occupancy (how much of the spectrum is
    filled), onset density and percussive share."""
    parts = {"fullness": track["fullness"], "onset_density": track["onset_density"],
             "percussive_ratio": track["percussive_ratio"]}
    score = _clip01(sum(_DENSITY_WEIGHTS[k] * _clip01(v) for k, v in parts.items()))
    lo, hi = cfg.density_bounds
    value = "sparse" if score < lo else ("dense" if score >= hi else "moderate")
    margin = min(abs(score - lo), abs(score - hi))
    conf = min(0.6, 0.25 + 0.35 * _clip01(margin / 0.1))
    return {"value": value, "score": _r(score, 3), "confidence": _r(conf, 2)}


def _tempo_summary(ctx: _Ctx, onset_rate: float, analysed: float) -> dict:
    """Tempo candidates from the mean autocorrelation tempogram, corrected for the Hann
    window's taper (so a steady pulse scores the same at every multiple of its period),
    peak-picked in 40-240 BPM, refined to sub-lag precision (parabolic peak, then the
    2x / 4x lag peaks) and ranked with librosa's log-normal prior at 120 BPM.
    ``half_double_ambiguous`` = a candidate at x2 or x0.5 scores >= 80 % of the best."""
    unknown = {"bpm": None, "confidence": 0.0, "candidates": [], "half_double_ambiguous": False}
    mean = ctx.tg_active_mean
    if mean is None or onset_rate < 0.2:
        return unknown
    ac = mean.size
    w = scipy.signal.get_window("hann", ac, fftbins=True)
    taper = np.correlate(w, w, mode="full")[ac - 1:] / float(w @ w)
    strength = np.divide(mean, taper, out=np.zeros_like(mean), where=taper > 0.05)
    lag_s = ctx.hop / ctx.sr
    bpm_lo, bpm_hi = _TEMPO_RANGE_BPM
    l_min = max(2, int(math.floor(60.0 / bpm_hi / lag_s)))
    l_max = min(ac - 2, int(math.ceil(60.0 / bpm_lo / lag_s)))
    if l_max <= l_min:
        return unknown
    seg = strength[l_min:l_max + 1]
    peaks, _ = scipy.signal.find_peaks(seg)
    if peaks.size == 0:
        return unknown
    background = float(np.median(seg))
    centre, octaves = _TEMPO_PRIOR
    cands = []
    for pk in peaks:
        lag = int(pk + l_min)
        s = float(strength[lag])
        exact = _refine_lag(strength, lag)
        for k in (4, 2):   # the 2x / 4x lag peak of the same pulse gives a finer period
            kl = int(round(exact * k))
            if kl + 1 < ac:
                local = kl - 2 + int(np.argmax(strength[kl - 2:kl + 3]))
                if 0 < local < ac - 1 and strength[local] >= 0.5 * s and \
                        strength[local] >= strength[local - 1] and strength[local] >= strength[local + 1]:
                    exact = _refine_lag(strength, local) / k
                    break
        bpm = 60.0 / (exact * lag_s)
        prior = math.exp(-0.5 * (math.log2(bpm / centre) / octaves) ** 2)
        # score ranks the candidates; salience (how far the peak rises above the curve's
        # background, which a DC-heavy onset envelope pushes towards 1) compares rivals
        salience = _clip01((s - background) / max(1e-6, 1.0 - background)) * prior
        cands.append({"bpm": bpm, "strength": s, "score": s * prior, "salience": salience})
    cands.sort(key=lambda c: -c["score"])
    best = cands[0]

    def octave(c: dict) -> bool:      # the same pulse at half / double time
        q = c["bpm"] / best["bpm"]
        return abs(q - 2.0) <= 0.08 or abs(q - 0.5) <= 0.02

    ambiguous = any(c is not best and c["score"] >= 0.8 * best["score"] and octave(c) for c in cands)
    contrast = _clip01((best["strength"] - background) / max(1e-6, 1.0 - background))
    if contrast < 0.05:
        return unknown
    # Another metrical level (3:2, 4:3, 5:4 ... e.g. on a 16th-note hi-hat grid) about as salient
    # makes the pick a coin toss between grid multiples: the confidence must say so. Without
    # this, 140 BPM four-on-the-floor read as 112 BPM with confidence 0.8. (The pick itself is
    # unchanged: the same autocorrelation + prior trade-off as librosa.feature.tempo.)
    rival = max((c["salience"] for c in cands[1:] if not octave(c) and abs(c["bpm"] / best["bpm"] - 1.0) > 0.04),
                default=0.0)
    unique = _clip01((1.0 - rival / best["salience"]) / _TEMPO_RIVAL_MARGIN) if best["salience"] > 0 else 0.0
    conf = (contrast * _clip01(onset_rate / 1.0) * _clip01(analysed / 30.0) * (0.6 if ambiguous else 1.0)
            * (0.15 + 0.85 * unique))
    shown: list[dict] = []
    for c in cands:
        if all(abs(c["bpm"] / s["bpm"] - 1.0) > 0.04 for s in shown):
            shown.append(c)
        if len(shown) == 3:
            break
    return {"bpm": _r(best["bpm"], 2), "confidence": _r(min(0.9, conf), 2),
            "candidates": [{"bpm": _r(c["bpm"], 2), "strength": _r(c["strength"], 3)} for c in shown],
            "half_double_ambiguous": bool(ambiguous)}


def _refine_lag(strength: np.ndarray, lag: int) -> float:
    """Parabolic interpolation of a peak at integer ``lag``."""
    if lag <= 0 or lag >= strength.size - 1:
        return float(lag)
    a, b, c = float(strength[lag - 1]), float(strength[lag]), float(strength[lag + 1])
    den = a - 2 * b + c
    if den >= 0:
        return float(lag)
    return lag + max(-0.5, min(0.5, 0.5 * (a - c) / den))


def _vocal_proxy(ctx: _Ctx, track: dict) -> dict:
    """Share of harmonic (HPSS) power in the vocal band x how much that in-band
    harmonic spectrum moves frame to frame (vibrato, glides, formants move partials;
    a sustained pad does not). A lead synth, guitar or strings in the same register
    score alike: this is a proxy with a capped, low confidence — not a vocal detector."""
    act = track["active_mask"]
    h = track["harm"]
    vmid = ctx.fr.vmid[act]
    mid = float(vmid.sum())
    if h <= 0 or mid <= 0 or track["active_s"] < 3.0:
        return {"value": None, "confidence": 0.0, "method": VOCAL_PROXY_METHOD}
    share = mid / h
    move = float((ctx.fr.vflux[act] * vmid).sum() / mid)
    value = _clip01(share / _VOCAL_SHARE_FULL) * _clip01(move / _VOCAL_MOVE_FULL)
    conf = min(VOCAL_PROXY_MAX_CONFIDENCE, 0.2 * _clip01(track["active_s"] / 60.0))
    return {"value": _r(value, 3), "confidence": _r(conf, 2), "method": VOCAL_PROXY_METHOD}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def _sections(ctx: _Ctx, cfg: SegmentConfig, rms_ref: float, analysed: float) -> list[dict]:
    """Novelty + fade boundaries, then heuristic labels and repeat candidates."""
    sr, hop = ctx.sr, ctx.hop
    bf = max(1, int(round(cfg.block_s * sr / hop)))           # frames per block
    starts = np.arange(0, ctx.n, bf)
    nb = len(starts)
    block_s = bf * hop / sr
    block_t = starts * hop / sr
    counts = np.diff(np.append(starts, ctx.n)).astype(np.float64)

    # block features: timbre (MFCC 1..12), harmony (chroma), dynamics (level, onsets, percussive share)
    mfcc = librosa.feature.mfcc(S=ctx.mel_db, n_mfcc=_N_MFCC)[1:]
    mf = np.add.reduceat(mfcc, starts, axis=1) / counts
    ch = np.add.reduceat(ctx.fr.chroma.astype(np.float64), starts, axis=1)
    ch = ch / np.maximum(ch.max(axis=0, keepdims=True), 1e-12)
    level = np.array([ctx.rms(t, min(analysed, t + block_s)) for t in block_t])
    level_db = np.clip(20 * np.log10(np.maximum(level, 1e-12) / rms_ref), *_RMS_DB_RANGE)
    onset = np.add.reduceat(ctx.oenv.astype(np.float64), starts) / counts
    h = np.add.reduceat(ctx.fr.harm, starts)
    p = np.add.reduceat(ctx.fr.perc, starts)
    perc = np.divide(p, h + p, out=np.zeros_like(p), where=(h + p) > 0)
    groups = {"timbre": mf.T, "harmony": ch.T, "dynamics": np.stack([level_db, onset, perc], axis=1)}
    feats = np.concatenate([_zscore(g) * (_STRUCT_WEIGHTS[k] / math.sqrt(g.shape[1])) for k, g in groups.items()],
                           axis=1)

    w = max(1, int(round(cfg.novelty_kernel_s / 2 / block_s)))
    sim, nov = _novelty(feats, w)
    min_blocks = max(1, int(round(cfg.min_section_s / block_s)))
    cands: list[tuple[float, float]] = []                    # (time, strength)
    if nov.size and nov.max() > 0:
        med = float(np.median(nov))
        noise = _NOVELTY_MAD * 1.4826 * float(np.median(np.abs(nov - med)))
        thr = max(_NOVELTY_MIN, med + noise, _NOVELTY_REL * float(nov.max()))
        peaks, _ = scipy.signal.find_peaks(nov, height=thr, distance=min_blocks)
        cands += [(float(block_t[i]), float(nov[i])) for i in peaks]
    smooth_db = scipy.ndimage.median_filter(level_db, size=3, mode="nearest") if nb >= 3 else level_db
    cands += [(t, 0.0) for t in _fade_boundaries(smooth_db, block_t, block_s, analysed)]
    max_sections = int(min(cfg.max_sections, max(1, analysed // cfg.min_section_s)))
    bounds: list[float] = []
    for t, _s in sorted(cands, key=lambda c: -c[1]):          # strongest first; fades last
        if len(bounds) >= max_sections - 1:
            break
        if t < cfg.min_section_s / 2 or analysed - t < cfg.min_section_s / 2:
            continue
        if all(abs(t - b) >= cfg.min_section_s for b in bounds):
            bounds.append(t)
    edges = [0.0] + sorted(bounds) + [analysed]

    # per-section measurements
    secs: list[dict[str, Any]] = []
    for t0, t1 in zip(edges[:-1], edges[1:]):
        a, b = ctx.frame_of(t0), ctx.frame_of(t1)
        inten = ctx.span(a, b, t1 - t0)["intensity"]
        lvl = 20 * math.log10(max(ctx.rms(t0, t1), 1e-12) / rms_ref)
        drive = inten + _LEVEL_WEIGHT * min(0.25, max(-1.0, lvl / _LEVEL_SPAN_DB))
        third = (t1 - t0) / 3
        rise = (ctx.span(ctx.frame_of(t1 - third), b, third)["intensity"]
                - ctx.span(a, ctx.frame_of(t0 + third), third)["intensity"])
        bi = (block_t >= t0 - 1e-9) & (block_t < t1 - 1e-9)
        sdb = smooth_db[bi]
        q = max(1, sdb.size // 4)
        level_change = float(sdb[-q:].mean() - sdb[:q].mean()) if sdb.size >= 2 else 0.0
        secs.append({"t0": t0, "t1": t1, "intensity": inten, "drive": drive, "rise": rise,
                     "level_change": level_change, "blocks": np.flatnonzero(bi)})
    _label_sections(secs)
    _repeat_groups(secs, sim)
    return [{"start_s": _r(s["t0"], 2), "end_s": _r(s["t1"], 2), "label": s["label"],
             "intensity": _r(s["intensity"], 3), "repeat_group": s["repeat_group"],
             "confidence": _r(s["confidence"], 2)} for s in secs]


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, keepdims=True)
    return (x - x.mean(axis=0, keepdims=True)) / np.where(sd > 1e-9, sd, 1.0)


def _novelty(feats: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(self-similarity of the blocks, novelty 0..1 of a boundary at the start of each
    block): Foote's Gaussian-tapered checkerboard kernel over an RBF similarity matrix
    (median-distance bandwidth) of the reflect-padded block features."""
    nb = feats.shape[0]
    if nb < 2:
        return np.ones((nb, nb)), np.zeros(nb)
    pad = min(w, nb - 1)
    fp = np.pad(feats, ((pad, pad), (0, 0)), mode="reflect")
    sq = (fp * fp).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * fp @ fp.T, 0.0)
    off = d2[np.triu_indices(len(fp), 1)]
    sigma2 = float(np.median(off)) if off.size and np.median(off) > 0 else 1.0
    sim_p = np.exp(-d2 / (2.0 * sigma2))
    sim = sim_p[pad:pad + nb, pad:pad + nb]
    if nb < 2 * w + 1:
        return sim, np.zeros(nb)
    k = np.arange(2 * w) - (w - 0.5)
    taper = np.exp(-0.5 * (k / (0.5 * w)) ** 2)
    sign = np.sign(k)
    kernel = np.outer(taper, taper) * np.outer(sign, sign)    # + within halves, - across
    kernel /= np.abs(kernel).sum() / 2.0
    idx = np.arange(nb)[:, None] + np.arange(2 * w)[None, :]   # padded index of block i is i + w
    windows = sim_p[idx[:, :, None], idx[:, None, :]]
    nov = np.maximum((windows * kernel).sum(axis=(1, 2)), 0.0)
    return sim, nov


def _fade_boundaries(level_db: np.ndarray, block_t: np.ndarray, block_s: float, analysed: float) -> list[float]:
    """Start of a terminal fade-out / end of an initial fade-in: where the smoothed
    block level leaves / reaches 3 dB below the track's loud level (70th percentile),
    when the quiet edge lasts >= 4 s and ends >= 9 dB down."""
    out: list[float] = []
    if level_db.size < 6:
        return out
    loud = float(np.percentile(level_db, 70))
    near = np.flatnonzero(level_db >= loud - 3.0)
    if not near.size:
        return out
    j = int(near[-1]) + 1                                     # first block of the fade-out
    if j < level_db.size and analysed - block_t[j] >= 4.0 and level_db[-2:].mean() <= loud - 9.0:
        out.append(float(block_t[j]))
    i = int(near[0])                                          # first block at full level
    if i > 0 and block_t[i] >= 4.0 and level_db[:2].mean() <= loud - 9.0:
        out.append(float(block_t[i]))
    return out


def _label_sections(secs: list[dict]) -> None:
    """Heuristic labels from each section's drive (intensity + level term) relative to
    the duration-weighted median drive of the track, its internal intensity rise, its
    level change and its position. Confidence <= SECTION_MAX_CONFIDENCE."""
    if len(secs) == 1:
        secs[0].update(label="body", confidence=0.2)
        return
    drives = np.array([s["drive"] for s in secs])
    durs = np.array([s["t1"] - s["t0"] for s in secs])
    ref = _weighted_median(drives, durs)
    spread = max(0.04, float(drives.max() - drives.min()))
    d_low = max(0.05, 0.25 * spread)
    d_high = max(0.04, 0.15 * spread)
    rise_min = max(0.06, 0.3 * spread)
    last = len(secs) - 1
    for i, s in enumerate(secs):
        low = (ref - s["drive"]) / d_low                      # >= 1: clearly quieter / sparser
        nxt = secs[i + 1]["drive"] if i < last else None
        if i == 0 and (low >= 1 or s["level_change"] >= _FADE_DB):
            label, margin = "intro", max(low, s["level_change"] / _FADE_DB)
        elif i == last and (low >= 1 or s["level_change"] <= -_FADE_DB):
            label, margin = "outro", max(low, -s["level_change"] / _FADE_DB)
        elif nxt is not None and s["rise"] >= rise_min and nxt >= s["drive"]:
            label, margin = "build", s["rise"] / rise_min
        elif low >= 1:
            label, margin = "break", low
        elif (s["drive"] - ref) / d_high >= 1:
            label, margin = "peak", (s["drive"] - ref) / d_high
        else:
            label, margin = "body", 1.0
        conf = 0.25 if label == "body" else 0.2 + 0.15 * min(2.0, margin)
        s.update(label=label, confidence=min(SECTION_MAX_CONFIDENCE, conf))
    for s in secs:   # repeat candidates are looked for among the loud sections only
        s["high"] = s["label"] in ("peak", "body") and s["drive"] >= ref - d_low / 2


def _repeat_groups(secs: list[dict], sim: np.ndarray) -> None:
    """High-intensity sections whose cross-similarity is >= ``_REPEAT_SIMILARITY`` of
    their within-section similarity share a group id (1, 2, ... in order of appearance);
    a chorus candidate only. Others get None."""
    for s in secs:
        s["repeat_group"] = None
    high = [i for i, s in enumerate(secs) if s.get("high") and s["blocks"].size >= 2]
    parent = {i: i for i in high}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for x in range(len(high)):
        for y in range(x + 1, len(high)):
            a, b = secs[high[x]]["blocks"], secs[high[y]]["blocks"]
            within = math.sqrt(float(sim[np.ix_(a, a)].mean()) * float(sim[np.ix_(b, b)].mean()))
            cross = float(sim[np.ix_(a, b)].mean())
            if within > 0 and cross / within >= _REPEAT_SIMILARITY:
                parent[find(high[x])] = find(high[y])
    members: dict[int, list[int]] = {}
    for i in high:
        members.setdefault(find(i), []).append(i)
    gid = 0
    for root in sorted((r for r, m in members.items() if len(m) >= 2), key=lambda r: min(members[r])):
        gid += 1
        for i in members[root]:
            secs[i]["repeat_group"] = gid


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _error_value(kind: str, message: str, duration_s: float | None, analysed_s: float, notes: list[str]) -> dict:
    return {"version": SEGMENTS_VERSION, "error": message, "error_kind": kind,
            "duration_s": _r(duration_s, 3) if duration_s is not None else None,
            "analysed_s": _r(analysed_s, 3), "notes": list(notes)}


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _sum_squares(y: np.ndarray, chunk: int = 1 << 20) -> float:
    """Sum of squares in float64 without a float64 copy of the whole signal."""
    total = 0.0
    for i in range(0, len(y), chunk):
        c = y[i:i + chunk].astype(np.float64)
        total += float(np.dot(c, c))
    return total


def _moving_mean(v: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average; the window shrinks at the edges."""
    if v.size == 0:
        return v.astype(np.float64)
    k = max(1, min(k, v.size))
    c = np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])
    half = k // 2
    i = np.arange(v.size)
    lo, hi = np.maximum(0, i - half), np.minimum(v.size, i + half + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cw, cw[-1] / 2.0)])


def _timbre_of(x: dict | Sequence[float] | None) -> list[float] | None:
    if isinstance(x, dict):
        if "summary" in x:   # a segments value (a malformed summary is just "unknown")
            summary = x["summary"]
            x = summary.get("timbre") if isinstance(summary, dict) else None
        else:
            x = x.get("timbre")
    if not isinstance(x, (list, tuple)) or not x or not all(_is_num(v) for v in x):
        return None
    return [float(v) for v in x]


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _r(v: float | None, nd: int) -> float | None:
    """JSON-native rounded float (None for unknown / non-finite)."""
    if v is None:
        return None
    f = float(v)
    return round(f, nd) + 0.0 if math.isfinite(f) else None   # + 0.0: no "-0.0" in the JSON


def _clip01(v: float) -> float:
    return 0.0 if not math.isfinite(v) else min(1.0, max(0.0, v))


def _first_line(e: BaseException) -> str:
    text = str(e).strip()
    return text.splitlines()[0] if text else type(e).__name__
