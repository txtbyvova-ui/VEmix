"""Mode B (spec §4/§8): streaming continuous-mix renderer + legacy-compatible tracklist.

Memory model (automix v1 held the whole 3.5 h mix in RAM as one AudioSegment):

* the mix is written progressively to an RF64 float WAV (a 3.5 h float mix exceeds
  the 4 GB RIFF limit); only the current track, the next track (lookahead to plan the
  transition) and the mix fade-out hold-back (``mix_fade_out_s``) live in memory;
* audio is written in order and never revisited: once a sample is in the file it is
  immutable. Transition FX only ever see a copy of the outgoing track's overlap region.

Per transition (``i -> i+1``)::

    plan      transitions.plan_transition_detailed (material-aware length, curve, FX)
    region    copy of the last ``xf`` samples of track i  x fade_out   (post-fader send)
    FX        transitions.apply_fx(region) -> new array; echo taps ring out past the
              region and decay under the incoming track (never cut to zero abruptly)
    sum       + first samples of track i+1 x fade_in
    protect   true peak (x``true_peak_oversample``); above the ceiling -> smooth gain dip
              confined to the overlap, recorded in TransitionReport
    write     the block, then the body of track i+1 continues after it

Duration accounting: ``expected_duration_s = sum(trimmed durations) - sum(crossfades)``
and the written file is checked against it to the sample.
"""

from __future__ import annotations

import datetime
import logging
import math
import os
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import transitions
from .config import TransitionConfig
from .models import MixInput, MixRenderResult, MixTrackEntry, TransitionReport

log = logging.getLogger(__name__)

CHANNELS = 2
CLIP_THRESHOLD = 0.999           # |x| >= this counts as a clipped sample (QCConfig.clip_sample_threshold)
WRITE_CHUNK_FRAMES = 1 << 20     # bound temporaries while writing / measuring

# legacy trim_silence (automix v1): 50 ms chunks, 100 ms safety buffer
TRIM_CHUNK_S = 0.050
TRIM_SAFETY_S = 0.100
TRIM_SCAN_CHUNKS = 400           # chunks per vectorized scan step (20 s)

# overlap peak protection
PEAK_MARGIN_DB = 0.05            # aim this far below the ceiling (estimation slack)
PEAK_HOLD_S = 0.4                # hold + raised-cosine smoothing window of the gain dip
PEAK_EDGE_S = 0.05               # the dip is pinned to 0 dB at both overlap edges
PEAK_MAX_ATTEMPTS = 4
PEAK_RESIDUAL_FLAG_DB = 0.1
# Real neighbouring samples fed to the true-peak interpolator on each side of a block
# (cropped from the result). resample_poly's default Kaiser filter reaches ~10 input
# samples per side at any up-factor; without context it assumes zeros outside the
# block and a full-level edge sample reads up to ~1 dB too hot (Gibbs ringing).
TP_CONTEXT_N = 16

ProgressFn = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Track loading / silence trimming
# ---------------------------------------------------------------------------

@dataclass
class _Track:
    inp: MixInput
    audio: np.ndarray              # float32 (n, 2), trimmed
    head_trim: int = 0             # samples removed at the start
    tail_trim: int = 0             # samples removed at the end

    @property
    def n(self) -> int:
        return len(self.audio)


def _describe(inp: MixInput) -> str:
    return f"#{inp.position} {inp.name!r} ({inp.pcm_path})"


def _check_input(inp: MixInput, sr: int) -> None:
    path = Path(inp.pcm_path)
    if not path.is_file():
        raise FileNotFoundError(f"normalized track not found: {_describe(inp)}")
    try:
        info = sf.info(str(path))
    except RuntimeError as e:
        raise ValueError(f"cannot read normalized track {_describe(inp)}: {e}") from e
    if info.samplerate != sr:
        raise ValueError(f"sample rate mismatch for {_describe(inp)}: {info.samplerate} Hz, mix is {sr} Hz")
    if info.channels not in (1, 2):
        raise ValueError(f"{_describe(inp)} has {info.channels} channels; only mono or stereo is supported")
    if info.frames <= 0:
        raise ValueError(f"{_describe(inp)} contains no audio frames")


def _load_track(inp: MixInput, sr: int, cfg: TransitionConfig) -> _Track:
    _check_input(inp, sr)
    try:
        data, file_sr = sf.read(str(inp.pcm_path), dtype="float32", always_2d=True)
    except RuntimeError as e:
        raise ValueError(f"cannot decode normalized track {_describe(inp)}: {e}") from e
    if file_sr != sr:
        raise ValueError(f"sample rate mismatch for {_describe(inp)}: {file_sr} Hz, mix is {sr} Hz")
    if len(data) == 0:
        raise ValueError(f"{_describe(inp)} contains no audio frames")
    if not np.isfinite(data).all():
        raise ValueError(f"{_describe(inp)} contains non-finite samples (NaN/inf)")
    if data.shape[1] == 1:
        data = np.repeat(data, CHANNELS, axis=1)          # mono -> identical L/R
    if cfg.trim_silence:
        start, end = trim_silence_bounds(data, sr, thresh_dbfs=cfg.trim_silence_thresh_dbfs,
                                         min_silence_ms=cfg.trim_min_silence_ms)
        if end <= start:
            raise ValueError(f"{_describe(inp)} is silent (below {cfg.trim_silence_thresh_dbfs:g} dBFS throughout)")
    else:
        if not np.any(data):
            raise ValueError(f"{_describe(inp)} is digitally silent")
        start, end = 0, len(data)
    track = _Track(inp=inp, audio=np.ascontiguousarray(data[start:end]), head_trim=start,
                   tail_trim=len(data) - end)
    log.info("loaded %s: %.2f s (trimmed %.2f s head / %.2f s tail)", inp.name, track.n / sr,
             start / sr, (len(data) - end) / sr)
    return track


def _chunk_rms(seg: np.ndarray, c: int) -> np.ndarray:
    """RMS over all channels of consecutive ``c``-sample chunks of ``seg`` (len multiple of c)."""
    m = len(seg) // c
    if m == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(np.square(seg[:m * c].astype(np.float64)).reshape(m, c, -1), axis=(1, 2)))


def trim_silence_bounds(x: np.ndarray, sr: int, *, thresh_dbfs: float = -40.0,
                        min_silence_ms: float = 200.0) -> tuple[int, int]:
    """``(start, end)`` sample bounds after trimming leading/trailing silence.

    Same rules as automix v1 ``trim_silence``: 50 ms chunks scanned from each end until a
    chunk's RMS level exceeds ``thresh_dbfs``; a 100 ms safety buffer is kept; a trim
    shorter than ``min_silence_ms`` is not applied. ``end <= start`` means the whole
    signal is below the threshold (no chunk is louder)."""
    n = len(x)
    c = max(1, int(round(TRIM_CHUNK_S * sr)))
    safety = int(round(TRIM_SAFETY_S * sr))
    min_sil = int(round(min_silence_ms / 1000.0 * sr))
    thr = 10.0 ** (thresh_dbfs / 20.0)

    # --- head: chunks aligned to the start (the last one may be partial) ---
    start = n
    pos = 0
    while pos < n:
        seg = x[pos:pos + TRIM_SCAN_CHUNKS * c]
        rms = _chunk_rms(seg, c)
        if len(seg) % c:
            rms = np.append(rms, _chunk_rms(seg[len(seg) - len(seg) % c:], len(seg) % c))
        loud = np.nonzero(rms > thr)[0]
        if len(loud):
            start = pos + int(loud[0]) * c
            break
        pos += TRIM_SCAN_CHUNKS * c
    if start >= n:
        return 0, 0

    # --- tail: chunks aligned to the end, scanned backwards while end > start ---
    end = start
    hi = n
    while hi > start:
        lo = max(0, hi - TRIM_SCAN_CHUNKS * c)
        seg = x[lo:hi]
        full = (len(seg) // c) * c
        rms_full = _chunk_rms(seg[len(seg) - full:], c)[::-1]     # chunk ending at hi first
        ends = hi - c * np.arange(len(rms_full))
        rms_list, end_list = list(rms_full), list(ends)
        if len(seg) > full:                                        # partial chunk at the very start
            rms_list.append(float(_chunk_rms(seg[:len(seg) - full], len(seg) - full)[0]))
            end_list.append(lo + len(seg) - full)
        found = False
        for r, e in zip(rms_list, end_list):
            if e <= start:
                break
            if r > thr:
                end, found = int(e), True
                break
        if found or lo == 0:
            break
        hi = lo
    if end <= start:
        return 0, 0

    start_b = max(0, start - safety)
    end_b = min(n, end + safety)
    if start_b < min_sil:
        start_b = 0
    if n - end_b < min_sil:
        end_b = n
    return start_b, end_b


# ---------------------------------------------------------------------------
# True peak / overlap peak protection
# ---------------------------------------------------------------------------

def _db(v: float) -> float:
    return 20.0 * math.log10(v) if v > 0 else float("-inf")


def _as_frames(x: np.ndarray | None, channels: int) -> np.ndarray:
    """``x`` as float64 ``(n, channels)`` (1-D input is one channel; None is empty)."""
    if x is None:
        return np.zeros((0, channels))
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] != channels:
        raise ValueError(f"true-peak context has {x.shape[1]} channel(s), the block {channels}")
    return x


def _with_context(x: np.ndarray, before: np.ndarray | None, after: np.ndarray | None,
                  k: int) -> tuple[np.ndarray, int]:
    """``x`` extended by up to ``k`` real context samples on each side; whatever context
    is missing is filled by (even) reflection about the edge sample, which keeps a crest
    at the edge a crest instead of a step from zero. Odd reflection was rejected: on
    noise / high-frequency content it extrapolates up to 3x the amplitude.
    Returns ``(extended, offset of x[0] inside it)``."""
    ch = x.shape[1]
    b = _as_frames(before, ch)[-k:] if k else np.zeros((0, ch))
    a = _as_frames(after, ch)[:k] if k else np.zeros((0, ch))
    ext = np.concatenate([b, x, a])
    lo, hi = k - len(b), k - len(a)
    if lo or hi:
        ext = np.pad(ext, ((lo, hi), (0, 0)), mode="reflect" if len(ext) > 1 else "edge")
    return ext, k


def true_peak_envelope(x: np.ndarray, oversample: int, *, before: np.ndarray | None = None,
                       after: np.ndarray | None = None) -> np.ndarray:
    """Per-sample true-peak estimate: max |x| over the ``oversample``-times upsampled
    signal (``resample_poly``) in each original sample period, across channels.

    ``before`` / ``after`` are the real samples that precede / follow ``x`` (the last /
    first ``TP_CONTEXT_N`` are used). The interpolator runs over ``x`` plus that context
    and the result is cropped back to ``x``; missing context is reflect-padded, so the
    block edges are not mistaken for steps from silence."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    if n == 0:
        return np.zeros(0)
    sample_peak = np.max(np.abs(x), axis=1)
    if oversample <= 1:
        return sample_peak
    ext, off = _with_context(x, before, after, TP_CONTEXT_N)
    up = resample_poly(ext, oversample, 1, axis=0)
    seg = up[off * oversample:(off + n) * oversample]
    env = np.abs(seg).reshape(n, oversample, -1).max(axis=(1, 2))
    return np.maximum(env, sample_peak)


def true_peak_dbtp(x: np.ndarray, oversample: int, *, before: np.ndarray | None = None,
                   after: np.ndarray | None = None) -> float:
    env = true_peak_envelope(x, oversample, before=before, after=after)
    return _db(float(env.max())) if len(env) else float("-inf")


@dataclass(frozen=True)
class _Signal:
    """Audio plus the real samples around it (context for true-peak interpolation)."""
    audio: np.ndarray
    before: np.ndarray | None = None
    after: np.ndarray | None = None

    def true_peak_dbtp(self, oversample: int) -> float:
        return true_peak_dbtp(self.audio, oversample, before=self.before, after=self.after)


@dataclass
class _Protection:
    block: np.ndarray
    gain_reduction_db: float
    tp_before: float
    tp_after: float
    flags: list[str] = field(default_factory=list)


def _protect_overlap(block: np.ndarray, sr: int, *, ceiling_dbtp: float, oversample: int,
                     contributors: tuple[_Signal | np.ndarray, ...], before: np.ndarray | None = None,
                     after: np.ndarray | None = None) -> _Protection:
    """Keep the true peak of an overlap at or below the ceiling with a smooth gain dip.

    The required reduction per sample is held for ``PEAK_HOLD_S`` and smoothed with a
    raised-cosine (Hann) kernel of the same length (``transitions.smooth_reduction_db``;
    in the interior provably >= the requirement), then pinned to 0 dB over ``PEAK_EDGE_S``
    at both edges so the gain is continuous with the untouched audio around the overlap.
    Re-measured after applying; the margin grows until the overlap is at or below the
    allowance.

    The allowance is the ceiling, or the louder contributor's own true peak if a track
    is already hotter than the ceiling (that is a normalization issue — the overlap is
    only prevented from adding to it).

    ``before`` / ``after`` are the mix samples around the block and each contributor
    carries its own context (``_Signal``), so block edges are interpolated against real
    audio and do not read hot (DSP-4)."""
    env = true_peak_envelope(block, oversample, before=before, after=after)
    tp_before = _db(float(env.max())) if len(env) else float("-inf")
    flags: list[str] = []
    if tp_before <= ceiling_dbtp:
        return _Protection(block, 0.0, tp_before, tp_before)
    own = max((c if isinstance(c, _Signal) else _Signal(c)).true_peak_dbtp(oversample) for c in contributors)
    allowance = max(ceiling_dbtp, own)
    if own > ceiling_dbtp:
        flags.append(f"track material itself peaks at {own:+.2f} dBTP inside the overlap (ceiling "
                     f"{ceiling_dbtp:+.2f} dBTP): overlap held at that level, not louder")
    if tp_before <= allowance:
        return _Protection(block, 0.0, tp_before, tp_before, flags)

    hold_n = int(round(PEAK_HOLD_S * sr))
    edge_n = int(round(PEAK_EDGE_S * sr))
    env_db = 20.0 * np.log10(np.maximum(env, 1e-12))

    margin = PEAK_MARGIN_DB
    out, gain_db, tp_after = block, np.zeros(len(block)), tp_before
    for _ in range(PEAK_MAX_ATTEMPTS):
        gain_db = transitions.smooth_reduction_db(env_db - (allowance - margin), hold_n, edge_n=edge_n)
        out = block * (10.0 ** (-gain_db / 20.0))[:, None]
        tp_after = true_peak_dbtp(out, oversample, before=before, after=after)   # dip is 0 dB at the edges
        if tp_after <= allowance:
            break
        margin += (tp_after - allowance) + PEAK_MARGIN_DB
    reduction = float(gain_db.max())
    flags.append(f"overlap overshoot: true peak {tp_before:+.2f} dBTP > {allowance:+.2f} dBTP — smooth gain dip "
                 f"of up to {reduction:.2f} dB inside the overlap only (now {tp_after:+.2f} dBTP)")
    if tp_after > allowance + PEAK_RESIDUAL_FLAG_DB:
        flags.append(f"peak protection incomplete: overlap still {tp_after - allowance:.2f} dB above "
                     f"{allowance:+.2f} dBTP after {PEAK_MAX_ATTEMPTS} attempts — listen")
    return _Protection(out, reduction, tp_before, tp_after, flags)


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------

class _MixWriter:
    """Appends blocks to the output file. Applies the mix fade-in by absolute position
    and holds back the last ``hold_n`` frames for the mix fade-out (applied in
    ``finish``). Input blocks are never modified in place."""

    def __init__(self, f: sf.SoundFile, *, fade_in_n: int, hold_n: int) -> None:
        self.f = f
        self.fade_in = transitions.fade_curves(fade_in_n, "s_curve")[1] if fade_in_n > 0 else np.zeros(0, np.float32)
        self.hold_n = max(0, hold_n)
        self.position = 0          # frames handed to write()
        self.written = 0           # frames committed to the file
        self.peak = 0.0
        self.clipped = 0
        self._held = np.zeros((0, CHANNELS), dtype=np.float32)

    def write(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32)
        n = len(block)
        if n == 0:
            return
        if self.position < len(self.fade_in):
            k = min(n, len(self.fade_in) - self.position)
            block = block.copy()
            block[:k] *= self.fade_in[self.position:self.position + k, None]
        self.position += n
        commit = len(self._held) + n - self.hold_n
        if commit <= 0:
            self._held = np.concatenate([self._held, block])
        elif commit <= len(self._held):
            self._emit(self._held[:commit])
            self._held = np.concatenate([self._held[commit:], block])
        else:
            self._emit(self._held)
            k = commit - len(self._held)
            self._emit(block[:k])
            self._held = block[k:].copy()

    def finish(self, fade_out_n: int) -> None:
        held = self._held.copy()
        k = min(max(0, fade_out_n), len(held))
        if k:
            held[len(held) - k:] *= transitions.fade_curves(k, "s_curve")[0][:, None]
        self._emit(held)
        self._held = np.zeros((0, CHANNELS), dtype=np.float32)

    def _emit(self, x: np.ndarray) -> None:
        for i in range(0, len(x), WRITE_CHUNK_FRAMES):
            part = np.ascontiguousarray(x[i:i + WRITE_CHUNK_FRAMES], dtype=np.float32)
            self.f.write(part)
            a = np.abs(part)
            self.peak = max(self.peak, float(a.max()) if a.size else 0.0)
            self.clipped += int(np.count_nonzero(a >= CLIP_THRESHOLD))
            self.written += len(part)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_mix(inputs: list[MixInput], out_wav: Path, cfg: TransitionConfig, *,
               ceiling_dbtp: float, sample_rate: int = 44100, seed: int = 0,
               progress: ProgressFn | None = None) -> MixRenderResult:
    """Render the continuous mix of ``inputs`` (normalized lossless WAVs, play order)
    into ``out_wav`` (RF64, 32-bit float, stereo; atomic replace).

    ``ceiling_dbtp`` is the true-peak ceiling enforced inside overlaps. ``seed`` drives
    the (opt-in) random FX choice. ``progress(done, total, message)`` is called after each
    track. Raises ``ValueError`` for empty input, invalid configuration or broken tracks
    (nothing is skipped silently)."""
    if not inputs:
        raise ValueError("render_mix: no input tracks — nothing to mix")
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate!r}")
    if not isinstance(ceiling_dbtp, (int, float)) or not math.isfinite(ceiling_dbtp):
        raise ValueError(f"ceiling_dbtp must be a finite number, got {ceiling_dbtp!r}")
    transitions.validate_transition_config(cfg)
    sr = int(sample_rate)
    out_wav = Path(out_wav).resolve()
    for inp in inputs:
        _check_input(inp, sr)                                  # fail fast, before writing anything
        if Path(inp.pcm_path).resolve() == out_wav:
            raise ValueError(f"refusing to overwrite an input track with the mix: {out_wav}")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_wav.with_name(f".{out_wav.stem}.veg-{uuid.uuid4().hex[:8]}.wav")
    rng = np.random.default_rng(seed)

    try:
        with sf.SoundFile(str(tmp), "w", samplerate=sr, channels=CHANNELS, format="RF64", subtype="FLOAT") as f:
            entries, reports, writer = _render_stream(inputs, f, cfg, sr=sr, ceiling_dbtp=ceiling_dbtp,
                                                      rng=rng, progress=progress)
        expected_frames = sum(int(round(e.duration_s * sr)) for e in entries) \
            - sum(int(round(r.crossfade_s * sr)) for r in reports)
        frames = sf.info(str(tmp)).frames
        if frames != expected_frames or writer.written != expected_frames:
            raise RuntimeError(f"mix duration accounting mismatch: file has {frames} frames, "
                               f"writer {writer.written}, tracklist expects {expected_frames}")
        os.replace(tmp, out_wav)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            log.warning("could not remove temporary mix file %s", tmp)
        raise

    expected_s = sum(e.duration_s for e in entries) - sum(r.crossfade_s for r in reports)
    result = MixRenderResult(
        wav_path=out_wav, sample_rate=sr, channels=CHANNELS, duration_s=frames / sr,
        expected_duration_s=expected_s, tracks=entries, transitions=reports,
        peak_dbfs=round(_db(writer.peak), 3) if writer.peak > 0 else None,
        clipped_samples=writer.clipped, seed=seed)
    log.info("mix rendered: %s, %.1f s, %d tracks, %d transitions, peak %s dBFS, %d clipped samples",
             out_wav.name, result.duration_s, len(entries), len(reports), result.peak_dbfs, writer.clipped)
    return result


def _render_stream(inputs: list[MixInput], f: sf.SoundFile, cfg: TransitionConfig, *, sr: int,
                   ceiling_dbtp: float, rng: np.random.Generator,
                   progress: ProgressFn | None) -> tuple[list[MixTrackEntry], list[TransitionReport], _MixWriter]:
    total = len(inputs)
    declick_n = int(round(cfg.declick_ms / 1000.0 * sr))
    cur = _load_track(inputs[0], sr, cfg)
    fade_in_n = min(int(round(cfg.mix_fade_in_s * sr)), cur.n // 2)
    hold_n = int(round(cfg.mix_fade_out_s * sr))
    writer = _MixWriter(f, fade_in_n=fade_in_n, hold_n=hold_n)

    entries: list[MixTrackEntry] = []
    reports: list[TransitionReport] = []
    fx_history: list[str] = []
    cursor = 0                 # samples of `cur` already written inside the previous transition block
    start_n = 0                # absolute mix position where `cur` becomes audible
    xf_in_n = 0
    fx_in = "—"
    splice_in = False          # previous transition was a butt splice -> de-click the head of `cur`

    for i in range(total):
        nxt = _load_track(inputs[i + 1], sr, cfg) if i + 1 < total else None
        xf_n, decision, flags = 0, None, []
        if nxt is not None:
            decision = transitions.plan_transition_detailed(
                cur.inp, nxt.inp, cur.n / sr, nxt.n / sr, cfg,
                prev_tail_db=transitions.level_profile_db(cur.audio, sr, align="end"),
                next_head_db=transitions.level_profile_db(nxt.audio, sr, align="start"),
                rng=rng, fx_history=fx_history)
            flags = list(decision.flags)
            xf_n = int(round(decision.plan.crossfade_s * sr))
            room = min(cur.n - cursor, nxt.n)
            if xf_n > room:      # only reachable with exotic configs; never re-open written audio
                flags.append(f"crossfade shortened from {xf_n / sr:.2f} s to {room / sr:.2f} s "
                             f"(not enough unplayed audio)")
                xf_n = room

        # --- body of the current track: written once, never touched again ---
        body = cur.audio[cursor:cur.n - xf_n]
        d_in = min(declick_n, len(body) // 2) if splice_in and cursor == 0 else 0
        d_out = min(declick_n, len(body) // 2) if nxt is not None and xf_n == 0 else 0
        if d_in or d_out:
            body = body.copy()
            if d_in:
                body[:d_in] *= transitions.fade_curves(d_in, "s_curve")[1][:, None]
            if d_out:
                body[len(body) - d_out:] *= transitions.fade_curves(d_out, "s_curve")[0][:, None]
        writer.write(body)
        entries.append(MixTrackEntry(
            position=cur.inp.position, name=cur.inp.name, source_rel=cur.inp.source_rel,
            start_s=start_n / sr, end_s=(start_n + cur.n) / sr, duration_s=cur.n / sr,
            head_trim_s=cur.head_trim / sr, tail_trim_s=cur.tail_trim / sr,
            crossfade_in_s=xf_in_n / sr, fx_in=fx_in))

        if nxt is None:
            if progress:
                progress(i + 1, total, cur.inp.name)
            break

        # --- transition cur -> nxt ---
        plan = decision.plan
        overlap_start = writer.position
        if xf_n > 0:
            reserve = _max_outgoing_overlap(nxt, cfg, sr) if i + 2 < total else 0
            block, used, protection, block_flags = _transition_block(
                cur, nxt, xf_n, plan.curve, plan.fx, sr=sr, ceiling_dbtp=ceiling_dbtp,
                oversample=cfg.true_peak_oversample, ext_limit=max(0, nxt.n - xf_n - reserve))
            writer.write(block)
            next_cursor = used
            report = TransitionReport(
                position=nxt.inp.position, crossfade_s=xf_n / sr, fx=plan.fx, curve=plan.curve,
                overlap_start_s=overlap_start / sr,
                overlap_peak_dbfs=round(_db(float(np.max(np.abs(block)))), 3) if block.size else None,
                overlap_true_peak_dbtp=round(protection.tp_after, 3),
                gain_reduction_db=round(protection.gain_reduction_db, 3),
                uncertain=plan.uncertain, flags=flags + block_flags + protection.flags)
        else:
            next_cursor = 0
            report = TransitionReport(
                position=nxt.inp.position, crossfade_s=0.0, fx=plan.fx, curve=plan.curve,
                overlap_start_s=overlap_start / sr, overlap_peak_dbfs=None, overlap_true_peak_dbtp=None,
                gain_reduction_db=0.0, uncertain=plan.uncertain, flags=flags)
        reports.append(report)
        fx_history.append(plan.fx)
        if progress:
            progress(i + 1, total, cur.inp.name)

        start_n, xf_in_n, fx_in, splice_in = overlap_start, xf_n, plan.fx, xf_n == 0
        cur, cursor = nxt, next_cursor        # the previous track is released here

    writer.finish(min(hold_n, cur.n // 2))
    return entries, reports, writer


def _max_outgoing_overlap(track: _Track, cfg: TransitionConfig, sr: int) -> int:
    """Upper bound (samples) of the crossfade ``track`` may need for its own outgoing
    transition; the incoming echo ring-out must not reach into it."""
    dur = track.n / sr
    if not cfg.enabled or dur < cfg.min_track_s_for_crossfade:
        return 0
    return int(math.ceil(min(cfg.max_overlap_s, cfg.max_fraction_of_track * dur) * sr)) + 1


def _transition_block(cur: _Track, nxt: _Track, xf_n: int, curve: str, fx: str, *, sr: int,
                      ceiling_dbtp: float, oversample: int,
                      ext_limit: int) -> tuple[np.ndarray, int, _Protection, list[str]]:
    """Build the overlap block. Returns (block, samples of ``nxt`` consumed, protection, flags)."""
    flags: list[str] = []
    fade_out, fade_in = transitions.fade_curves(xf_n, curve)
    # A COPY of the outgoing region only (the fade is the channel fader; the echo is
    # post-fader, so its taps decay naturally instead of being cut at the region end).
    region = cur.audio[cur.n - xf_n:] * fade_out[:, None]
    processed = transitions.apply_fx(region, sr, fx)
    if len(processed) < xf_n:
        raise RuntimeError(f"apply_fx({fx!r}) returned {len(processed)} samples for a {xf_n}-sample region")
    ext = len(processed) - xf_n
    if ext > ext_limit:
        flags.append(f"FX tail shortened from {ext / sr * 1000:.0f} ms to {ext_limit / sr * 1000:.0f} ms "
                     "(next transition too close)")
        ext = ext_limit
        processed = processed[:xf_n + ext].copy()
        k = min(len(processed), int(round(0.010 * sr)))
        processed[len(processed) - k:] *= transitions.fade_curves(k, "s_curve")[0][:, None]
    total = xf_n + ext
    out_part = processed[:total].astype(np.float64)
    in_gain = np.ones(total, dtype=np.float64)
    in_gain[:xf_n] = fade_in
    in_part = nxt.audio[:total].astype(np.float64) * in_gain[:, None]
    block = out_part + in_part
    # True-peak context (DSP-4): in the mix the block is preceded by the body of `cur`
    # and followed by the body of `nxt`. The faded contributors genuinely start / end
    # in silence (fader closed), so zeros are their real context on that side.
    k = TP_CONTEXT_N
    lead = cur.audio[max(0, cur.n - xf_n - k):cur.n - xf_n]
    trail = nxt.audio[total:total + k]
    silence = np.zeros((k, cur.audio.shape[1]))
    protection = _protect_overlap(
        block, sr, ceiling_dbtp=ceiling_dbtp, oversample=oversample, before=lead, after=trail,
        contributors=(_Signal(region, before=lead, after=silence), _Signal(in_part, before=silence, after=trail)))
    return protection.block.astype(np.float32), total, protection, flags


# ---------------------------------------------------------------------------
# Tracklist (legacy-compatible)
# ---------------------------------------------------------------------------

def ms_to_hms(ms: int) -> str:
    """automix v1 ``ms_to_hms``: H:MM:SS, floored to whole seconds."""
    s = int(ms) // 1000
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _hms_floor(seconds: float, sample_rate: int) -> str:
    """H:MM:SS floored to whole seconds. Mix positions are whole samples, so half a
    sample of slack keeps float noise (3719.99999999 for 3720 s) from flooring a second down."""
    return ms_to_hms(int(math.floor(seconds + 0.5 / max(sample_rate, 1))) * 1000)


def _camelot_parse(code: str | None) -> tuple[int, str] | None:
    if not code or not isinstance(code, str):
        return None
    code = code.strip().upper()
    if len(code) < 2 or code[-1] not in "AB" or not code[:-1].isdigit():
        return None
    num = int(code[:-1])
    return (num, code[-1]) if 1 <= num <= 12 else None


def _camelot_compatible(a: str | None, b: str | None) -> bool | None:
    """automix v1 semantics (same key, adjacent on the wheel, relative major/minor);
    ``None`` when either key is unknown — never assumed compatible."""
    pa, pb = _camelot_parse(a), _camelot_parse(b)
    if pa is None or pb is None:
        return None
    (na, la), (nb, lb) = pa, pb
    return (na == nb) or (la == lb and (na - nb) % 12 in (1, 11))


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_mix_tracklist(result: MixRenderResult, inputs: list[MixInput], path: Path, *,
                        mix_name: str | None = None) -> None:
    """Human-readable tracklist, compatible with automix v1's ``*_tracklist.txt``.

    Header and the first two table columns are exactly v1's (``{num:<4} | {H:MM:SS:<10} |``)
    so existing tooling and QC (``^(\\d+)\\s+\\|\\s+(\\d+:\\d{2}:\\d{2})\\s+\\|``) keep working.
    ``mix_name`` defaults to ``<stem>.mp3`` for a ``<stem>_tracklist.txt`` path (v1 naming),
    otherwise the rendered WAV's name."""
    path = Path(path)
    if mix_name is None:
        mix_name = (path.stem[:-len("_tracklist")] + ".mp3") if path.stem.endswith("_tracklist") \
            else result.wav_path.name
    by_pos = {inp.position: inp for inp in inputs}
    trans = {t.position: t for t in result.transitions}

    sr = result.sample_rate or 44100
    total = int(math.floor(result.duration_s + 0.5 / sr))
    lines = [
        f"TRACKLIST: {mix_name}",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Random seed: {result.seed}",
        f"Total duration: {total // 3600}h {(total % 3600) // 60:02d}m {total % 60:02d}s",
        f"Total tracks: {len(result.tracks)}",
        "",
        f"{'#':<4} | {'Time':<10} | {'BPM':<6} | {'Key':<5} | {'Xfade':<6} | {'FX':<14} | Track",
        "-" * 4 + "-|-" + "-" * 10 + "-|-" + "-" * 6 + "-|-" + "-" * 5 + "-|-" + "-" * 6 + "-|-" + "-" * 14
        + "-|-" + "-" * 30,
    ]
    for entry in result.tracks:
        num = entry.position          # QC keys rows by mix position (1..N, like v1 "num")
        inp = by_pos.get(entry.position)
        bpm = inp.bpm if inp else None
        key = (inp.camelot if inp else None) or "?"
        tr = trans.get(entry.position)
        if tr is None:
            xfade, fx = "—", "—"
        else:
            xfade = f"{tr.crossfade_s:.1f}s" if tr.crossfade_s > 0 else "cut"
            fx = tr.fx
        marks = []
        if tr is not None and tr.uncertain:
            marks.append("listen: uncertain transition")
        if tr is not None and tr.gain_reduction_db > 0:
            marks.append(f"overlap peak -{tr.gain_reduction_db:.1f} dB")
        suffix = f"  [{'; '.join(marks)}]" if marks else ""
        known_bpm = isinstance(bpm, (int, float)) and math.isfinite(bpm) and bpm > 0
        bpm_s = f"{bpm:<6.1f}" if known_bpm else f"{'?':<6}"
        lines.append(f"{num:<4} | {_hms_floor(entry.start_s, sr):<10} | {bpm_s} | {key:<5} | {xfade:<6} | "
                     f"{fx:<14} | {entry.name}{suffix}")

    # --- camelot stats (unknown keys are reported, never counted as compatible) ---
    pairs = list(zip(result.tracks, result.tracks[1:]))
    verdicts = [_camelot_compatible(by_pos[a.position].camelot if a.position in by_pos else None,
                                    by_pos[b.position].camelot if b.position in by_pos else None)
                for a, b in pairs]
    n_tr = len(pairs)
    compatible = sum(1 for v in verdicts if v is True)
    incompatible = sum(1 for v in verdicts if v is False)
    unknown = sum(1 for v in verdicts if v is None)
    pct = (compatible / n_tr * 100) if n_tr else 0
    lines += ["", "=== CAMELOT STATS ===",
              f"Compatible transitions: {compatible}/{n_tr} ({pct:.0f}%)",
              f"Incompatible transitions: {incompatible}/{n_tr}",
              f"Unknown key: {unknown}/{n_tr}"]

    # --- FX ---
    counts = {fx: 0 for fx in transitions.FX_NAMES}
    for t in result.transitions:
        counts[t.fx] = counts.get(t.fx, 0) + 1
    fx_total = sum(counts.values())
    lines += ["", "=== FX SUMMARY ==="]
    for name, count in counts.items():
        lines.append(f"{name}: {count} ({(count / fx_total * 100) if fx_total else 0:.0f}%)")
    splices = sum(1 for t in result.transitions if t.crossfade_s <= 0)
    lines.append(f"Butt splices (no overlap): {splices}/{len(result.transitions)}")

    # --- transitions that need listening ---
    lines += ["", "=== TRANSITIONS TO REVIEW (no beat/phrase alignment available) ==="]
    uncertain = [t for t in result.transitions if t.uncertain]
    if not uncertain:
        lines.append("None flagged.")
    for t in uncertain:
        reasons = "; ".join(t.flags) or "uncertain transition"
        lines.append(f"#{t.position - 1} → #{t.position} at {_hms_floor(t.overlap_start_s, sr)} "
                     f"(xfade {t.crossfade_s:.1f}s): {reasons}")

    # --- peak protection ---
    lines += ["", "=== PEAK PROTECTION ==="]
    protected = [t for t in result.transitions if t.gain_reduction_db > 0]
    measured = [t.overlap_true_peak_dbtp for t in result.transitions if t.overlap_true_peak_dbtp is not None]
    lines.append(f"Overlaps with true-peak overshoot corrected: {len(protected)}/{len(result.transitions)}")
    if measured:
        lines.append(f"Max overlap true peak after protection: {max(measured):+.2f} dBTP")
    for t in protected:
        lines.append(f"#{t.position - 1} → #{t.position} at {_hms_floor(t.overlap_start_s, sr)}: "
                     f"gain dip up to {t.gain_reduction_db:.2f} dB, overlap now {t.overlap_true_peak_dbtp:+.2f} dBTP")
    peak = f"{result.peak_dbfs:+.2f} dBFS" if result.peak_dbfs is not None else "n/a (silent)"
    lines.append(f"Rendered mix sample peak: {peak}; clipped samples (|x| >= {CLIP_THRESHOLD}): "
                 f"{result.clipped_samples}")
    lines.append(f"Duration: rendered {result.duration_s:.3f} s, expected {result.expected_duration_s:.3f} s "
                 "(sum of trimmed tracks minus crossfades)")
    _atomic_write_text(path, "\n".join(lines) + "\n")
