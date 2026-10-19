"""Automatic per-track analysis (spec §5, auto-analysis half) on top of the
section cache (spec §9).

* BPM / key — moved from ``legacy/automix_v1.py`` WITHOUT algorithmic changes (same
  librosa calls, windows and fallbacks; pinned by a regression test). The key window
  is conservative on purpose: a past change of detected keys produced very bad mixes,
  so it must not be widened or aligned with the BPM window.
* Energy — an explicitly LOW-CONFIDENCE heuristic (see ``estimate_energy``), never
  presented as a measurement. Danceability, vocal presence, mood and familiarity are
  not estimated at all: they stay ``None`` unless a curator supplies them.
* Artist / title / genre / language — from tags, else artist/title from the file name.
  Cyrillic script yields a review hint, NOT a language claim.
* Loudness — pass-1 measurement through ``LoudnessNormalizer.measure`` (cached).

Broken inputs are never dropped silently: ``TrackAnalyzer.analyze`` returns an
``AnalyzedTrack`` whose ``errors`` say what failed. ``FFmpegNotFoundError`` propagates.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import librosa
import numpy as np
import soundfile as sf

from .cache import AnalysisCache
from .config import AUDIO_EXTENSIONS, AnalysisConfig
from .curation import ANALYSIS_CYRILLIC_HINT_REASON
from .id3 import read_text_frames
from .ffmpeg_tools import (
    FFmpegError,
    FFmpegNotFoundError,
    FFmpegTools,
    LoudnessMeasurementError,
    decode_check,
    find_ffmpeg,
    probe_audio,
    run_ffmpeg,
)
from .models import AnalyzedTrack, AudioProbe, LoudnessMeasurement, TrackFeatures, to_jsonable
from .textnorm import normalize_language as _normalize_language

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids importing the normalizer eagerly)
    from .loudness import LoudnessNormalizer

log = logging.getLogger(__name__)

BPM_ALGO_VERSION = "bpm-v1"
KEY_ALGO_VERSION = "key-v1"
ENERGY_ALGO_VERSION = "energy-heuristic-v1"
PROBE_VERSION = "probe-v1"
LOUDNESS_VERSION = "loudness-v1"
LOUDNESS_METER = "loudnorm+ebur128-v1"

# ---------------------------------------------------------------------------
# CAMELOT / KEY (verbatim from automix v1)
# ---------------------------------------------------------------------------

CAMELOT_MAP = {
    (0,  1): '8B',  (0,  0): '5A',
    (1,  1): '3B',  (1,  0): '12A',
    (2,  1): '10B', (2,  0): '7A',
    (3,  1): '5B',  (3,  0): '2A',
    (4,  1): '12B', (4,  0): '9A',
    (5,  1): '7B',  (5,  0): '4A',
    (6,  1): '2B',  (6,  0): '11A',
    (7,  1): '9B',  (7,  0): '6A',
    (8,  1): '4B',  (8,  0): '1A',
    (9,  1): '11B', (9,  0): '8A',
    (10, 1): '6B',  (10, 0): '3A',
    (11, 1): '1B',  (11, 0): '10A',
}

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_CAMELOT_RE = re.compile(r"^(?:[1-9]|1[0-2])[AB]$")

# ---------------------------------------------------------------------------
# Energy heuristic constants (part of ENERGY_ALGO_VERSION — bump it when changing them)
# ---------------------------------------------------------------------------

ENERGY_MIN_WINDOW_S = 20.0    # offset window shorter than this -> analyse from the start
ENERGY_MIN_AUDIO_S = 5.0      # less audio than this -> no estimate (None)
ENERGY_SILENCE_DBFS = -70.0   # RMS below this -> silent -> no estimate (None)
ENERGY_WEIGHTS = {"onset_density": 0.35, "percussive_ratio": 0.30, "brightness": 0.20, "tempo": 0.15}
# onset rate (onsets/s) at which the density component reaches 1 - 1/e
ENERGY_ONSET_RATE_SCALE = 3.0
# spectral-centroid range mapped to 0..1 on a log-frequency scale
ENERGY_CENTROID_HZ = (300.0, 5000.0)
# tempo range mapped linearly to 0..1
ENERGY_TEMPO_BPM = (70.0, 140.0)
_STFT_N_FFT = 2048
_HOP = 512

# Shared with curation, which replaces this hint with its own verdict.
CYRILLIC_REVIEW_REASON = ANALYSIS_CYRILLIC_HINT_REASON
_CYRILLIC_RE = re.compile("[Ѐ-ԯᲀ-᲏ⷠ-ⷿꙀ-ꚟ]")

LANGUAGE_TAGS = ("language", "tlan", "lang")
# A source changed while it was being analysed: its results are not cached (they may mix
# the old and the new content) and the track is flagged.
CHANGED_DURING_ANALYSIS_REASON = ("source file changed while it was being analysed — results were not "
                                  "cached; re-run the analysis")

# VEgroove outputs that must never be re-ingested as library tracks (see scan_library).
VEGROOVE_MANIFEST_SCHEMA = "vegroove.manifest"
PLAYLIST_MANIFEST_NAME = "manifest.json"
MIX_MANIFEST_SUFFIX = ".manifest.json"

# "Artist - Title" (also en/em dash between spaces); leading track numbers.
_ARTIST_TITLE_SEP = re.compile(r"\s+[-–—]\s+")
_TRACKNO_DOT = re.compile(r"^\s*\d{1,3}\s*[.)]\s+")      # "01. Title", "1) Title"
_TRACKNO_DASH = re.compile(r"^\s*\d{1,3}\s+[-–—]\s+")   # "01 - Artist - Title"


# ---------------------------------------------------------------------------
# Library scanning / naming helpers
# ---------------------------------------------------------------------------

def scan_library(folder: Path, *, recursive: bool = False,
                 extensions: tuple[str, ...] = AUDIO_EXTENSIONS,
                 exclude: Iterable[Path] = ()) -> list[Path]:
    """Audio files in ``folder`` (absolute paths), sorted by case-folded relative path.

    Extensions match case-insensitively. Dot-files and dot-directories are skipped
    (our own temporary files start with '.'). Raises FileNotFoundError /
    NotADirectoryError for a bad folder instead of returning an empty library.

    Our own outputs are never library tracks (re-ingesting them would put a mix inside
    the next mix, or re-encode MP3 -> MP3):

    * ``exclude`` — files, or folders whose whole subtree is skipped (e.g. the output
      folder / planned output files of this run); compared after ``realpath`` and
      ``normcase``, so another spelling of the same location still matches;
    * an audio file with a sibling ``<stem>.manifest.json`` of schema
      ``vegroove.manifest`` (a published mix);
    * when ``recursive``, any folder (the root included) that contains a VEgroove
      ``manifest.json`` (a playlist export: its ``tracks/`` are outputs).
    """
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"library folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"library path is not a folder: {root}")
    root = root.resolve()
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    excluded = [_norm_real(Path(e)) for e in exclude]
    skipped: list[str] = []
    found: list[Path] = []

    def keep_dir(dirpath: Path, names: list[str]) -> bool:
        if PLAYLIST_MANIFEST_NAME in names and _is_vegroove_manifest(dirpath / PLAYLIST_MANIFEST_NAME):
            skipped.append(f"{dirpath} (VEgroove playlist folder)")
            return False
        return True

    def collect(dirpath: Path, filenames: list[str]) -> None:
        names = {n.casefold(): n for n in filenames}
        for f in filenames:
            if f.startswith(".") or Path(f).suffix.lower() not in exts:
                continue
            p = dirpath / f
            manifest = names.get((Path(f).stem + MIX_MANIFEST_SUFFIX).casefold())
            if manifest and _is_vegroove_manifest(dirpath / manifest):
                skipped.append(f"{p} (VEgroove mix, see {manifest})")
                continue
            if p.is_file():
                found.append(p)

    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            here = Path(dirpath)
            if _is_excluded(here, excluded):
                skipped.append(f"{here} (excluded output location)")
                dirnames[:] = []
                continue
            if not keep_dir(here, filenames):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            collect(here, filenames)
    else:
        collect(root, [p.name for p in root.iterdir()])

    files: list[Path] = []
    for p in found:
        if excluded and _is_excluded(p, excluded):
            skipped.append(f"{p} (excluded output location)")
        else:
            files.append(p)
    if skipped:
        log.warning("library scan: skipped %d VEgroove output(s) inside %s: %s", len(skipped), root,
                    "; ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""))
    return sorted(files, key=lambda p: (p.relative_to(root).as_posix().casefold(), p.as_posix()))


def _norm_real(path: Path) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _is_excluded(path: Path, excluded: list[str]) -> bool:
    """True when ``path`` is one of ``excluded`` or lies under one of them."""
    if not excluded:
        return False
    p = _norm_real(path)
    return any(p == e or p.startswith(e.rstrip(os.sep) + os.sep) for e in excluded)


def _is_vegroove_manifest(path: Path) -> bool:
    """A manifest written by VEgroove (playlist ``manifest.json`` / mix ``*.manifest.json``)."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return isinstance(data, dict) and data.get("schema") == VEGROOVE_MANIFEST_SCHEMA


def library_rel_path(path: Path, library_root: Path) -> str:
    """POSIX path of ``path`` relative to ``library_root`` (``..`` segments when outside,
    the absolute POSIX path when on another drive)."""
    p, root = Path(path).resolve(), Path(library_root).resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return Path(os.path.relpath(p, root)).as_posix()
    except ValueError:   # different drive on Windows
        return p.as_posix()


def parse_artist_title(stem: str) -> tuple[str | None, str | None]:
    """``"Artist - Title"`` -> ("Artist", "Title"), split on the FIRST separator.

    A leading track number is removed when it is clearly one ("01. Title",
    "01 - Artist - Title"); "50 Cent - Candy Shop" is left alone. Without a separator
    the whole stem is the title and the artist is unknown (None).
    """
    text = (stem or "").strip()
    text = _TRACKNO_DOT.sub("", text, count=1)
    m = _TRACKNO_DASH.match(text)
    if m and _ARTIST_TITLE_SEP.search(text[m.end():]):   # "01 - Artist - Title"
        text = text[m.end():]
    parts = _ARTIST_TITLE_SEP.split(text, maxsplit=1)
    if len(parts) == 2:
        artist, title = parts[0].strip(), parts[1].strip()
        return artist or None, title or None
    return None, text or None


def detect_script_hint(*texts: str | None) -> str | None:
    """'cyrillic' when any text contains Cyrillic letters, else None.

    This is a hint for editorial review — it is NOT a language detection (Ukrainian,
    Serbian, Kazakh, transliterated titles... all look alike here)."""
    for t in texts:
        if t and _CYRILLIC_RE.search(t):
            return "cyrillic"
    return None


def normalize_language(value: str | None) -> str | None:
    """Tag value -> canonical language set (textnorm): ISO 639-1 codes joined with ';'
    ('rus;eng' -> 'ru;en', 'English' -> 'en', 'en-US' -> 'en', 'por' -> 'pt'); values no
    table recognises are kept lower-cased after the codes (curation treats them as an
    UNKNOWN language). Placeholders ('und', 'xxx', '') mean unknown -> None."""
    return _normalize_language(value)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_window(path: Path, sr: int, offset: float, duration: float,
                 tools: FFmpegTools | None, *, allow_ffmpeg: bool = True) -> tuple[np.ndarray, int]:
    """``librosa.load(str(path), sr=sr, mono=True, offset=offset, duration=duration)`` —
    the exact call of automix v1 — plus two rules that do not change any result v1
    could compute:

    * an offset at/after the end of the file yields an EMPTY signal, so the caller's
      "too little audio -> retry from the start" fallback runs. (v1 read MP3 through
      audioread, which returned an empty array here; libsndfile raises on the seek.)
    * formats libsndfile cannot open (m4a/aac/...) are first decoded by FFmpeg into a
      temporary float WAV at the native sample rate, then loaded with the same call.
    """
    path_s = str(path)
    try:
        info = sf.info(path_s)
    except (sf.LibsndfileError, RuntimeError):
        if not allow_ffmpeg:
            raise
        return _load_via_ffmpeg(path, sr, offset, duration, tools)
    if offset > 0 and info.samplerate > 0 and int(offset * info.samplerate) >= info.frames:
        return np.zeros(0, dtype=np.float32), sr
    try:
        y, out_sr = librosa.load(path_s, sr=sr, mono=True, offset=offset, duration=duration)
    except sf.LibsndfileError:
        # MP3 frame counts can be estimates: a seek a few ms short of the reported end
        # may still fail. Treat it as an empty window like the check above.
        if offset > 0 and info.samplerate > 0 and offset >= info.frames / info.samplerate - 0.05:
            return np.zeros(0, dtype=np.float32), sr
        raise
    return y, int(out_sr)


def _load_via_ffmpeg(path: Path, sr: int, offset: float, duration: float,
                     tools: FFmpegTools | None) -> tuple[np.ndarray, int]:
    tools = tools or find_ffmpeg()   # FFmpegNotFoundError propagates on purpose
    with tempfile.TemporaryDirectory(prefix="veg_dec_") as td:
        wav = Path(td) / "decoded.wav"
        # Only [0, offset + duration] is ever read; decoding more would be wasted work.
        run_ffmpeg(tools, ["-nostats", "-y", "-i", str(path), "-map", "0:a:0", "-vn",
                           "-t", f"{offset + duration:.3f}", "-c:a", "pcm_f32le", "-rf64", "auto", str(wav)])
        y, out_sr = _load_window(wav, sr, offset, duration, None, allow_ffmpeg=False)
        return y.copy(), out_sr


# ---------------------------------------------------------------------------
# BPM / key (automix v1 algorithms, unchanged)
# ---------------------------------------------------------------------------

def bpm_params(cfg: AnalysisConfig) -> dict:
    return {"sample_rate": cfg.sample_rate, "offset_s": cfg.bpm_offset_s,
            "duration_s": cfg.bpm_duration_s, "min_window_s": cfg.bpm_min_window_s}


def key_params(cfg: AnalysisConfig) -> dict:
    return {"sample_rate": cfg.sample_rate, "offset_s": cfg.key_offset_s,
            "duration_s": cfg.key_duration_s, "min_window_s": cfg.key_min_window_s}


def energy_params(cfg: AnalysisConfig) -> dict:
    return {"sample_rate": cfg.sample_rate, "offset_s": cfg.energy_offset_s,
            "duration_s": cfg.energy_duration_s, "min_window_s": ENERGY_MIN_WINDOW_S,
            "min_audio_s": ENERGY_MIN_AUDIO_S, "silence_dbfs": ENERGY_SILENCE_DBFS,
            "weights": ENERGY_WEIGHTS, "onset_rate_scale": ENERGY_ONSET_RATE_SCALE,
            "centroid_hz": list(ENERGY_CENTROID_HZ), "tempo_bpm": list(ENERGY_TEMPO_BPM)}


def compute_bpm(path: Path, cfg: AnalysisConfig, *, tools: FFmpegTools | None = None) -> float:
    """Uncached BPM, algorithm of automix v1 ``detect_bpm``."""
    y, sr = _load_window(Path(path), cfg.sample_rate, cfg.bpm_offset_s, cfg.bpm_duration_s, tools)
    if len(y) < sr * cfg.bpm_min_window_s:  # too little / unrepresentative audio after the offset
        y, sr = _load_window(Path(path), cfg.sample_rate, 0.0, cfg.bpm_duration_s, tools)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0])


def compute_key(path: Path, cfg: AnalysisConfig, *, tools: FFmpegTools | None = None) -> tuple[str, float]:
    """Uncached key, algorithm of automix v1 ``detect_key`` (chroma CQT mean +
    Krumhansl-Kessler correlation). Returns (camelot, best correlation).

    The window (offset 30 s, 60 s, retry from the start only if < 1 s was captured) is
    deliberately conservative — do not widen it (project memory rule)."""
    y, sr = _load_window(Path(path), cfg.sample_rate, cfg.key_offset_s, cfg.key_duration_s, tools)
    if len(y) < sr * cfg.key_min_window_s:  # < 1 s captured from the offset — retry from the start
        y, sr = _load_window(Path(path), cfg.sample_rate, 0.0, cfg.key_duration_s, tools)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)

    best_corr = -1.0
    best_pitch, best_mode = 0, 1
    # errstate only silences warnings for constant chroma (silence); results are unchanged
    with np.errstate(invalid="ignore", divide="ignore"):
        for shift in range(12):
            shifted = np.roll(chroma_avg, -shift)
            maj_corr = float(np.corrcoef(shifted, MAJOR_PROFILE)[0, 1])
            min_corr = float(np.corrcoef(shifted, MINOR_PROFILE)[0, 1])
            if maj_corr > best_corr:
                best_corr, best_pitch, best_mode = maj_corr, shift, 1
            if min_corr > best_corr:
                best_corr, best_pitch, best_mode = min_corr, shift, 0

    camelot = CAMELOT_MAP.get((best_pitch, best_mode), '1A')
    return camelot, best_corr


def detect_bpm(path: Path, cache: AnalysisCache, cfg: AnalysisConfig, *,
               tools: FFmpegTools | None = None) -> float:
    """Cached BPM (section 'bpm', version BPM_ALGO_VERSION). May be 0.0 when librosa
    finds no beat — callers treat that as unknown."""
    return _detect_bpm(Path(path), cache, cfg, tools)[0]


def detect_key(path: Path, cache: AnalysisCache, cfg: AnalysisConfig, *,
               tools: FFmpegTools | None = None) -> tuple[str, float | None]:
    """Cached (camelot, confidence) (section 'key', version KEY_ALGO_VERSION).

    The confidence is the full-precision correlation (v1 cached it rounded to 3
    decimals); values migrated from a v1 cache keep their rounded value, or None when
    v1 had not stored one."""
    camelot, conf, _ = _detect_key(Path(path), cache, cfg, tools)
    return camelot, conf


def _detect_bpm(path: Path, cache: AnalysisCache, cfg: AnalysisConfig,
                tools: FFmpegTools | None, expected_fp: dict | None = None) -> tuple[float, bool]:
    params = bpm_params(cfg)
    hit = cache.get(path, "bpm", version=BPM_ALGO_VERSION, params=params)
    if hit is not None and _is_finite_number(hit.get("bpm")):
        return float(hit["bpm"]), True
    if hit is not None:
        log.warning("cache: malformed bpm value for %s; recomputing", path)
    bpm = compute_bpm(path, cfg, tools=tools)
    cache.put(path, "bpm", {"bpm": bpm}, version=BPM_ALGO_VERSION, params=params,
              expected_fingerprint=expected_fp)
    return bpm, False


def _detect_key(path: Path, cache: AnalysisCache, cfg: AnalysisConfig,
                tools: FFmpegTools | None, expected_fp: dict | None = None) -> tuple[str, float | None, bool]:
    params = key_params(cfg)
    hit = cache.get(path, "key", version=KEY_ALGO_VERSION, params=params)
    if hit is not None:
        cam, conf = hit.get("camelot"), hit.get("key_confidence")
        if isinstance(cam, str) and _CAMELOT_RE.match(cam) and (conf is None or _is_finite_number(conf)):
            return cam, (float(conf) if conf is not None else None), True
        log.warning("cache: malformed key value for %s; recomputing", path)
    camelot, conf = compute_key(path, cfg, tools=tools)
    cache.put(path, "key", {"camelot": camelot, "key_confidence": conf},
              version=KEY_ALGO_VERSION, params=params, expected_fingerprint=expected_fp)
    return camelot, conf, False


# ---------------------------------------------------------------------------
# Energy heuristic
# ---------------------------------------------------------------------------

def compute_energy(path: Path, cfg: AnalysisConfig, *, tools: FFmpegTools | None = None) -> dict:
    """Uncached energy heuristic. Returns ``{"energy": float | None, "components": {...},
    "analysed_s": float, "reason": str | None}``.

    Loudness-independent by construction: the window is scaled to unit RMS first, and
    every component is a ratio / rate / frequency, so the master's level (or a later
    normalization) does not move the score. Components, each mapped to 0..1:

    * onset_density    — onsets per second from spectral flux (onset strength),
                         ``1 - exp(-rate / 3)``                         (weight 0.35)
    * percussive_ratio — percussive share of spectral energy after HPSS  (weight 0.30)
    * brightness       — median spectral centroid on a log scale, 300 Hz -> 0,
                         5 kHz -> 1 (frames within 40 dB of the loudest) (weight 0.20)
    * tempo            — autocorrelation tempo, 70 BPM -> 0, 140 BPM -> 1, scaled by
                         onset_density (sparse material gets no tempo credit) (weight 0.15)

    It is an ESTIMATE (rank-ordering dense/bright/fast vs sparse/soft/slow material),
    not a perceptual energy measurement; callers attach a low confidence to it.
    """
    path = Path(path)
    y, sr = _load_window(path, cfg.sample_rate, cfg.energy_offset_s, cfg.energy_duration_s, tools)
    if len(y) < sr * ENERGY_MIN_WINDOW_S:
        y, sr = _load_window(path, cfg.sample_rate, 0.0, cfg.energy_duration_s, tools)
    seconds = len(y) / sr if sr else 0.0
    result: dict[str, Any] = {"energy": None, "components": {}, "analysed_s": round(seconds, 3), "reason": None}
    if seconds < ENERGY_MIN_AUDIO_S:
        result["reason"] = f"only {seconds:.1f} s of audio (< {ENERGY_MIN_AUDIO_S:.0f} s)"
        return result
    y = np.asarray(y, dtype=np.float64)
    rms = float(np.sqrt(np.mean(y * y)))
    if not math.isfinite(rms) or rms <= 10 ** (ENERGY_SILENCE_DBFS / 20):
        result["reason"] = f"silent (RMS below {ENERGY_SILENCE_DBFS:.0f} dBFS)"
        return result
    y = (y / rms).astype(np.float32)          # level independence

    S = np.abs(librosa.stft(y, n_fft=_STFT_N_FFT, hop_length=_HOP))
    harm, perc = librosa.decompose.hpss(S)
    e_h, e_p = float(np.sum(harm ** 2)), float(np.sum(perc ** 2))
    percussive_ratio = e_p / (e_h + e_p) if (e_h + e_p) > 0 else 0.0

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=_HOP, units="time")
    onset_rate = len(onsets) / seconds
    onset_density = 1.0 - math.exp(-onset_rate / ENERGY_ONSET_RATE_SCALE)

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    frame_power = np.sum(S ** 2, axis=0)
    active = frame_power >= frame_power.max() * 1e-4 if frame_power.size else frame_power.astype(bool)
    c_hz = float(np.median(centroid[active])) if np.any(active) else float(ENERGY_CENTROID_HZ[0])
    lo, hi = ENERGY_CENTROID_HZ
    brightness = _clip01(math.log(max(c_hz, 1.0) / lo) / math.log(hi / lo))

    tempo_bpm = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=_HOP))[0])
    t_lo, t_hi = ENERGY_TEMPO_BPM
    # With few onsets the tempo estimate is mostly the estimator's 120 BPM prior, so the
    # tempo component is scaled by the onset density (no pulse -> no tempo credit).
    tempo_score = _clip01((tempo_bpm - t_lo) / (t_hi - t_lo)) * onset_density

    components = {"onset_density": onset_density, "percussive_ratio": percussive_ratio,
                  "brightness": brightness, "tempo": tempo_score}
    energy = sum(ENERGY_WEIGHTS[k] * _clip01(v) for k, v in components.items())
    result["energy"] = round(_clip01(energy), 3)
    result["components"] = {k: round(float(v), 4) for k, v in components.items()}
    result["components"].update({"onset_rate_per_s": round(onset_rate, 3),
                                 "centroid_hz": round(c_hz, 1), "tempo_bpm": round(tempo_bpm, 2)})
    return result


def estimate_energy(path: Path, cache: AnalysisCache, cfg: AnalysisConfig, *,
                    tools: FFmpegTools | None = None) -> float | None:
    """Cached energy heuristic in 0..1 (section 'energy', ENERGY_ALGO_VERSION), or None
    when the audio is too short or silent ("unknown" is cached too). See
    ``compute_energy`` for the method. ``cfg.estimate_energy`` is honoured by
    ``TrackAnalyzer``; a direct call always estimates."""
    return _estimate_energy(Path(path), cache, cfg, tools)[0]


def _estimate_energy(path: Path, cache: AnalysisCache, cfg: AnalysisConfig,
                     tools: FFmpegTools | None,
                     expected_fp: dict | None = None) -> tuple[float | None, bool, str | None]:
    params = energy_params(cfg)
    hit = cache.get(path, "energy", version=ENERGY_ALGO_VERSION, params=params)
    if hit is not None:
        e = hit.get("energy")
        if e is None or (_is_finite_number(e) and 0.0 <= e <= 1.0):
            return (float(e) if e is not None else None), True, hit.get("reason")
        log.warning("cache: malformed energy value for %s; recomputing", path)
    result = compute_energy(path, cfg, tools=tools)
    cache.put(path, "energy", result, version=ENERGY_ALGO_VERSION, params=params,
              expected_fingerprint=expected_fp)
    return result["energy"], False, result["reason"]


# ---------------------------------------------------------------------------
# Track analyzer
# ---------------------------------------------------------------------------

class TrackAnalyzer:
    """Probe + tags + BPM/key/energy + pass-1 loudness for one library file, cached."""

    def __init__(self, tools: FFmpegTools | None, cache: AnalysisCache, cfg: AnalysisConfig | None = None,
                 normalizer: LoudnessNormalizer | None = None) -> None:
        self.tools = tools if tools is not None else find_ffmpeg()
        self.cache = cache
        self.cfg = cfg or AnalysisConfig()
        self.normalizer = normalizer

    # --- public ----------------------------------------------------------

    def analyze(self, path: Path, library_root: Path, *, measure_loudness: bool = True) -> AnalyzedTrack:
        """Analyze one file. Never raises for a broken file (see ``AnalyzedTrack.errors``);
        ``FFmpegNotFoundError`` propagates. The cache is not saved here.

        The file's fingerprint is taken once BEFORE anything is read; results are cached
        only while the file still matches it, and a file that changed during the analysis
        is flagged for review (its results may describe the old content)."""
        path = Path(path).resolve()
        rel_path = library_rel_path(path, library_root)
        errors: list[str] = []
        review: list[str] = []
        from_cache: dict[str, bool] = {}
        sources: dict[str, str] = {}
        fp0 = self.cache.fingerprint(path)

        probe: AudioProbe | None = None
        try:
            probe, from_cache["probe"] = self._probe(path, fp0)
        except FFmpegNotFoundError:
            raise
        except (FFmpegError, OSError, ValueError) as e:
            errors.append(f"probe failed: {_first_line(e)}")

        tags = probe.tags if probe else {}
        artist, title = _tag(tags, "artist"), _tag(tags, "title")
        if artist:
            sources["artist"] = "id3"
        if title:
            sources["title"] = "id3"
        if not (artist and title):
            f_artist, f_title = parse_artist_title(path.stem)
            if not artist and f_artist:
                artist, sources["artist"] = f_artist, "filename"
            if not title and f_title:
                title, sources["title"] = f_title, "filename"

        features = TrackFeatures(path=path, duration_s=0.0, rel_path=rel_path, artist=artist, title=title,
                                 feature_sources=sources, review_reasons=review)
        hint = detect_script_hint(artist, title, Path(rel_path).stem)
        if hint:
            features.script_hint = hint
            review.append(CYRILLIC_REVIEW_REASON)

        if probe is None:
            return AnalyzedTrack(features=features, probe=None, errors=errors, from_cache=from_cache)

        genre = _tag(tags, "genre")
        if genre:
            features.genre, sources["genre"] = genre, "id3"
        lang_raw = next((tags[k] for k in LANGUAGE_TAGS if _tag(tags, k)), None)
        if path.suffix.lower() == ".mp3":
            # ffprobe keeps only the first value of a multi-valued ID3v2.4 TLAN ('eng\0rus');
            # every language must reach the curation rules (SPEC-3)
            tlan = read_text_frames(path, ("TLAN",)).get("TLAN")
            if tlan:
                lang_raw = ";".join(tlan)
        language = normalize_language(lang_raw)
        if language:
            features.language, sources["language"] = language, "id3"
        if not probe.channels:
            errors.append("source has no usable audio channels")

        self._analyze_bpm(path, features, errors, review, from_cache, fp0)
        self._analyze_key(path, features, errors, review, from_cache, fp0)
        if self.cfg.estimate_energy:
            self._analyze_energy(path, features, review, from_cache, fp0)

        loudness: LoudnessMeasurement | None = None
        if measure_loudness and self.normalizer is not None:
            try:
                loudness, from_cache["loudness"] = self._loudness(path, probe, fp0)
            except FFmpegNotFoundError:
                raise
            except (FFmpegError, LoudnessMeasurementError, OSError, ValueError) as e:
                errors.append(f"loudness measurement failed: {_first_line(e)}")

        features.duration_s = self._duration(path, probe, loudness, errors, review)
        if fp0 is not None and not self.cache.same_state(fp0, self.cache.fingerprint(path)):
            log.warning("%s changed while it was being analysed; results not cached", rel_path)
            review.append(CHANGED_DURING_ANALYSIS_REASON)
        log.debug("analyzed %s: bpm=%s key=%s energy=%s errors=%d", rel_path, features.bpm,
                  features.camelot, features.energy, len(errors))
        return AnalyzedTrack(features=features, probe=probe, loudness=loudness, errors=errors,
                             from_cache=from_cache)

    # --- steps -----------------------------------------------------------

    def _probe(self, path: Path, expected_fp: dict | None = None) -> tuple[AudioProbe, bool]:
        params = {"prober": "ffprobe" if self.tools.ffprobe else "ffmpeg-banner"}
        hit = self.cache.get(path, "probe", version=PROBE_VERSION, params=params)
        if hit is not None:
            try:
                probe = AudioProbe.from_dict(hit)
                probe.path = path
                return probe, True
            except (KeyError, TypeError, ValueError):
                log.warning("cache: malformed probe value for %s; probing again", path)
        probe = probe_audio(self.tools, path)
        self.cache.put(path, "probe", to_jsonable(probe), version=PROBE_VERSION, params=params,
                       expected_fingerprint=expected_fp)
        return probe, False

    def _analyze_bpm(self, path: Path, features: TrackFeatures, errors: list[str], review: list[str],
                     from_cache: dict[str, bool], expected_fp: dict | None = None) -> None:
        try:
            bpm, from_cache["bpm"] = _detect_bpm(path, self.cache, self.cfg, self.tools, expected_fp)
        except FFmpegNotFoundError:
            raise
        except Exception as e:  # decoder / librosa failure: report, never drop the track
            log.debug("BPM analysis failed for %s", path, exc_info=True)
            errors.append(f"BPM analysis failed: {type(e).__name__}: {_first_line(e)}")
            return
        if bpm > 0 and math.isfinite(bpm):
            features.bpm = bpm
            features.feature_sources["bpm"] = self._source(path, "bpm", BPM_ALGO_VERSION)
        else:
            review.append("tempo could not be detected (no beat found)")

    def _analyze_key(self, path: Path, features: TrackFeatures, errors: list[str], review: list[str],
                     from_cache: dict[str, bool], expected_fp: dict | None = None) -> None:
        try:
            camelot, conf, from_cache["key"] = _detect_key(path, self.cache, self.cfg, self.tools, expected_fp)
        except FFmpegNotFoundError:
            raise
        except Exception as e:
            log.debug("key analysis failed for %s", path, exc_info=True)
            errors.append(f"key analysis failed: {type(e).__name__}: {_first_line(e)}")
            return
        if conf is not None and not conf > 0:
            # no positive correlation with any key profile (silence / noise): unknown, not '8B'
            review.append(f"key could not be determined (best profile correlation {conf:.2f})")
            return
        source = self._source(path, "key", KEY_ALGO_VERSION)
        features.camelot = camelot
        features.key_confidence = conf
        features.feature_sources["camelot"] = source
        if conf is not None:
            features.feature_sources["key_confidence"] = source

    def _analyze_energy(self, path: Path, features: TrackFeatures, review: list[str],
                        from_cache: dict[str, bool], expected_fp: dict | None = None) -> None:
        try:
            energy, from_cache["energy"], reason = _estimate_energy(path, self.cache, self.cfg, self.tools,
                                                                    expected_fp)
        except FFmpegNotFoundError:
            raise
        except Exception as e:  # an optional estimate: a failure leaves it unknown
            log.debug("energy estimate failed for %s", path, exc_info=True)
            review.append(f"energy estimate failed: {type(e).__name__}: {_first_line(e)}")
            return
        if energy is None:
            log.debug("no energy estimate for %s: %s", path, reason)
            return
        features.energy = energy
        features.energy_confidence = self.cfg.energy_confidence
        features.feature_sources["energy"] = f"auto:{ENERGY_ALGO_VERSION}"

    def _loudness(self, path: Path, probe: AudioProbe,
                  expected_fp: dict | None = None) -> tuple[LoudnessMeasurement, bool]:
        norm = self.normalizer
        assert norm is not None
        params = {"target_lufs": norm.cfg.target_lufs, "pcm_true_peak_ceiling": norm.cfg.pcm_true_peak_ceiling,
                  "target_lra": norm.cfg.target_lra, "edge_window_s": norm.qc.edge_window_s,
                  "meter": LOUDNESS_METER}
        hit = self.cache.get(path, "loudness", version=LOUDNESS_VERSION, params=params)
        if hit is not None:
            try:
                m = LoudnessMeasurement.from_dict(hit)
                if m is not None:
                    return m, True
            except (KeyError, TypeError, ValueError):
                pass
            log.warning("cache: malformed loudness value for %s; measuring again", path)
        m = norm.measure(path, channels=probe.channels)
        self.cache.put(path, "loudness", to_jsonable(m), version=LOUDNESS_VERSION, params=params,
                       expected_fingerprint=expected_fp)
        return m, False

    def _duration(self, path: Path, probe: AudioProbe, loudness: LoudnessMeasurement | None,
                  errors: list[str], review: list[str]) -> float:
        container = probe.duration_s if probe.duration_s and probe.duration_s > 0 else None
        decoded = loudness.duration_s if loudness and loudness.duration_s and loudness.duration_s > 0 else None
        if container and decoded and abs(container - decoded) > max(1.0, 0.02 * container):
            review.append(f"container duration {container:.1f} s differs from the decoded duration "
                          f"{decoded:.1f} s (truncated or damaged file?)")
        duration = container or decoded
        if duration is None and not errors:
            try:
                dc = decode_check(self.tools, path)
                if dc.ok and dc.decoded_duration_s:
                    duration = dc.decoded_duration_s
                else:
                    errors.append("file does not decode cleanly: " + ("; ".join(dc.errors[:3]) or "unknown error"))
            except FFmpegError as e:
                errors.append(f"decode check failed: {_first_line(e)}")
        if duration is None:
            errors.append("duration could not be determined")
            return 0.0   # placeholder: the track has errors and must not be used
        return float(duration)

    def _source(self, path: Path, section: str, version: str) -> str:
        migrated = self.cache.migrated_from(path, section)
        return f"auto:{version}:legacy-cache" if migrated else f"auto:{version}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tag(tags: dict[str, str], name: str) -> str | None:
    value = tags.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _clip01(v: float) -> float:
    return 0.0 if not math.isfinite(v) else min(1.0, max(0.0, v))


def _first_line(e: BaseException) -> str:
    text = str(e).strip()
    return text.splitlines()[0] if text else type(e).__name__
