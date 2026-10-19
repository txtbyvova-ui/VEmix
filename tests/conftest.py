"""Shared fixtures: synthetic audio (no copyrighted music) and FFmpeg discovery."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from veg_mixer.ffmpeg_tools import FFmpegNotFoundError, find_ffmpeg

SR = 44100

try:
    TOOLS = find_ffmpeg()
    FFMPEG_SKIP_REASON = ""
except FFmpegNotFoundError as e:  # pragma: no cover - environment dependent
    TOOLS = None
    FFMPEG_SKIP_REASON = f"FFmpeg not available: {str(e).splitlines()[0]}"

requires_ffmpeg = pytest.mark.skipif(TOOLS is None, reason=FFMPEG_SKIP_REASON or "FFmpeg not available")


@pytest.fixture(scope="session")
def ffmpeg_tools():
    if TOOLS is None:
        pytest.skip(FFMPEG_SKIP_REASON)
    return TOOLS


# ---------------------------------------------------------------------------
# Synthetic signals
# ---------------------------------------------------------------------------

def pinkish_noise(n: int, rng: np.random.Generator, channels: int = 2) -> np.ndarray:
    """~1/f noise (Paul Kellet's economy filter) — spectrally closer to music than white noise."""
    from scipy.signal import lfilter
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, rng.standard_normal((n, channels)), axis=0)


def music_like(seconds: float, *, peak_dbfs: float = -1.0, seed: int = 0, channels: int = 2,
               bpm: float = 120.0, tone_hz: float = 220.0, am_depth: float = 0.5,
               drive: float = 2.0) -> np.ndarray:
    """Pinkish noise bed + kick-like pulses at ``bpm`` + a sustained tone, slowly
    amplitude-modulated (non-zero LRA). ``drive`` > 0 applies tanh "mastering":
    drive=2 gives a true-peak-to-loudness crest of ~9-10 dB like a modern pop master,
    drive=0 leaves ~13 dB. Scaled to a sample peak."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = 0.6 * pinkish_noise(n, rng, channels)
    x /= np.max(np.abs(x)) + 1e-12
    beat = 60.0 / bpm
    env = np.exp(-((t % beat) / 0.05))                      # decaying pulse every beat
    kick = np.sin(2 * np.pi * 60 * t) * env
    tone = 0.3 * np.sin(2 * np.pi * tone_hz * t)
    sig = x * 0.5 + (kick + tone)[:, None]
    sig *= (1 - am_depth / 2 + am_depth / 2 * np.sin(2 * np.pi * 0.25 * t))[:, None]
    sig /= np.max(np.abs(sig)) + 1e-12
    if drive > 0:
        sig = np.tanh(drive * sig) / np.tanh(drive)
    return (sig * 10 ** (peak_dbfs / 20)).astype(np.float32)


def spiky(seconds: float, *, bed_peak_dbfs: float = -40.0, spike: float = 0.95, every_s: float = 2.0,
          seed: int = 0) -> np.ndarray:
    """Quiet bed with sparse near-full-scale clicks: the loudness target and the
    true-peak limit cannot both be met without heavy limiting."""
    sig = music_like(seconds, peak_dbfs=bed_peak_dbfs, seed=seed, drive=0)
    step = int(every_s * SR)
    for k in range(step // 2, len(sig), step):
        sig[k:k + 20] = spike
    return sig


def write_wav(path: Path, data: np.ndarray, sr: int = SR, subtype: str = "FLOAT") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype=subtype)
    return path


def encode_mp3(tools, wav: Path, mp3: Path, bitrate: str = "192k", tags: dict | None = None) -> Path:
    """Create an MP3 fixture from a WAV with the real encoder."""
    args = [tools.ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", str(wav)]
    for k, v in (tags or {}).items():
        args += ["-metadata", f"{k}={v}"]
    args += ["-c:a", "libmp3lame", "-b:a", bitrate, str(mp3)]
    subprocess.run(args, check=True, capture_output=True)
    return mp3


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
