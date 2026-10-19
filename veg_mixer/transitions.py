"""P1 transition engine (spec §8): crossfade planning, fade curves and transition FX.

Pure functions on float arrays shaped ``(n, channels)`` — no file I/O. The streaming
renderer (``renderer.py``) calls into this module once per transition.

Fixes of automix v1:

* FX operate ONLY on a copy of the outgoing track's transition region and return a
  NEW array. v1 ran ``mix = apply_fx(mix, fx)`` on the whole accumulated mix, so a
  filter sweep or echo re-processed audio that had already been rendered.
* The crossfade length follows the actual material — fade-outs, quiet intros, energy
  difference and the adjacent track durations — not the BPM difference alone.
* There is no reliable beat/phrase detection, so nothing is beat-aligned and the plan
  never pretends otherwise: a big or unknown tempo difference gets a conservative short
  overlap and ``uncertain=True`` with a human-readable reason for listening review.
* FX are opt-in: the default mode is ``clean``; ``random`` must be requested explicitly
  and is deterministic for a seed (it never picks the same effect three times in a row).
* The echo has correct tap offsets in samples, a smooth wet onset, a tail that rings out
  past the region instead of being truncated, and gain staging (dry+wet never peaks above
  the dry region).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, fftconvolve, sosfilt, sosfilt_zi
from scipy.signal.windows import hann

from .config import TransitionConfig
from .models import MixInput, TransitionPlan

log = logging.getLogger(__name__)

CURVES = ("linear", "equal_power", "s_curve")
FX_NAMES = ("clean", "filter_sweep", "echo_tail", "filter+echo")
FX_MODES = (*FX_NAMES, "random")

# --- level profiles used by the planner (see level_profile_db) ---
LEVEL_FRAME_S = 0.25        # RMS frame length of prev_tail_db / next_head_db
EDGE_ANALYSIS_S = 12.0      # how much of the outgoing tail / incoming head is profiled
EDGE_LEVEL_S = 1.0          # end level (fade-out) / start level (intro) = median over this span
FADE_DECAY_DB = 10.0        # tail decays more than this below the body => "already fades out"
FADE_EDGE_DROP_DB = 3.0     # a frame within this of the body level still counts as "full level"
SILENCE_FLOOR_DB = -120.0

# --- energy (0..1 scale, curated or heuristic) ---
ENERGY_JUMP = 0.3           # |ΔE| above this => shorter overlap (do not smear two different moods)
ENERGY_JUMP_FACTOR = 0.5

# --- filter sweep: smooth time-varying low-pass 10 kHz -> 200 Hz ---
SWEEP_START_HZ = 10_000.0
SWEEP_END_HZ = 200.0
SWEEP_STEPS = 12            # fixed Butterworth filters crossfaded sample-wise (no zipper noise)
SWEEP_ORDER = 2             # 12 dB/oct: musical, small phase difference between neighbours

# --- echo tail: (delay s, gain dB relative to the dry signal) ---
ECHO_TAPS = ((0.080, -15.0), (0.160, -21.0), (0.300, -27.0))
ECHO_RAMP_FRACTION = 0.25   # wet send ramps in (raised cosine) over the first 25 % of the region
ECHO_STAGING_HOLD_S = 0.1   # smoothing window of the compensating gain (gain staging)
ECHO_STAGING_EDGE_S = 0.01  # compensating gain pinned to 0 dB at the region start (continuity)
ECHO_SEARCH_STEPS = 40      # bisection steps of the residual wet scale


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def validate_transition_config(cfg: TransitionConfig) -> None:
    """Raise ``ValueError`` for settings the engine cannot honour (fail before rendering)."""
    if cfg.curve not in CURVES:
        raise ValueError(f"unknown fade curve {cfg.curve!r}; expected one of {', '.join(CURVES)}")
    if cfg.fx_mode not in FX_MODES:
        raise ValueError(f"unknown fx_mode {cfg.fx_mode!r}; expected one of {', '.join(FX_MODES)}")
    for name in ("default_crossfade_s", "min_crossfade_s", "max_overlap_s", "min_track_s_for_crossfade",
                 "mix_fade_in_s", "mix_fade_out_s", "declick_ms", "trim_min_silence_ms",
                 "uncertain_tempo_ratio", "conservative_crossfade_s"):
        v = getattr(cfg, name)
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"{name}={v!r} must be a finite number >= 0")
    if not (0.0 < cfg.max_fraction_of_track <= 0.5):
        raise ValueError(f"max_fraction_of_track={cfg.max_fraction_of_track!r} must be in (0, 0.5] "
                         "(an incoming and an outgoing overlap must fit into one track)")
    if not isinstance(cfg.true_peak_oversample, int) or cfg.true_peak_oversample < 1:
        raise ValueError(f"true_peak_oversample={cfg.true_peak_oversample!r} must be an integer >= 1")
    if not math.isfinite(cfg.trim_silence_thresh_dbfs):
        raise ValueError("trim_silence_thresh_dbfs must be finite")
    if cfg.fx_mode == "random":
        _fx_weights(cfg)


def _fx_weights(cfg: TransitionConfig) -> dict[str, float]:
    unknown = sorted(set(cfg.fx_weights) - set(FX_NAMES))
    if unknown:
        raise ValueError(f"fx_weights has unknown effect(s) {unknown}; expected {', '.join(FX_NAMES)}")
    weights = {k: float(v) for k, v in cfg.fx_weights.items()}
    if any(not math.isfinite(v) or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError(f"fx_weights must be finite, >= 0 and not all zero: {cfg.fx_weights!r}")
    return weights


# ---------------------------------------------------------------------------
# Fade curves
# ---------------------------------------------------------------------------

def fade_curves(n: int, curve: str) -> tuple[np.ndarray, np.ndarray]:
    """``(fade_out, fade_in)`` gain curves of length ``n`` (float32).

    Sampled at bin centres ``t = (i + 0.5) / n``, so the pair is symmetric
    (``fade_out == fade_in[::-1]``) and never exactly 0/1 inside the overlap.

    * ``linear``      — amplitude sums to 1 (correlated material stays level).
    * ``equal_power`` — ``fade_out² + fade_in² = 1`` (uncorrelated material stays level).
    * ``s_curve``     — raised cosine, amplitude sums to 1, zero slope at both ends.
    """
    if curve not in CURVES:
        raise ValueError(f"unknown fade curve {curve!r}; expected one of {', '.join(CURVES)}")
    n = int(n)
    if n <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty.copy()
    t = (np.arange(n, dtype=np.float64) + 0.5) / n
    if curve == "linear":
        fade_in = t
        fade_out = 1.0 - t
    elif curve == "equal_power":
        fade_in = np.sin(0.5 * np.pi * t)
        fade_out = np.cos(0.5 * np.pi * t)
    else:  # s_curve
        fade_in = 0.5 - 0.5 * np.cos(np.pi * t)
        fade_out = 1.0 - fade_in
    return fade_out.astype(np.float32), fade_in.astype(np.float32)


# ---------------------------------------------------------------------------
# Material analysis helpers (level profiles, fades, tempo)
# ---------------------------------------------------------------------------

def level_profile_db(x: np.ndarray, sr: int, *, frame_s: float = LEVEL_FRAME_S,
                     max_s: float = EDGE_ANALYSIS_S, align: str = "start") -> np.ndarray:
    """Short-window RMS level profile in dBFS (frames of ``frame_s``).

    ``align="start"`` profiles the first ``max_s`` seconds (incoming head);
    ``align="end"`` the last ``max_s`` seconds with the last frame ending exactly at
    the final sample (outgoing tail). This is the format ``plan_transition`` expects
    for ``prev_tail_db`` / ``next_head_db``.
    """
    if align not in ("start", "end"):
        raise ValueError(f"align must be 'start' or 'end', got {align!r}")
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    f = max(1, int(round(frame_s * sr)))
    m = min(len(x) // f, max(1, int(math.ceil(max_s / frame_s))))
    if m <= 0:
        return np.zeros(0, dtype=np.float64)
    seg = x[:m * f] if align == "start" else x[len(x) - m * f:]
    power = np.mean(np.square(seg.astype(np.float64)).reshape(m, f, -1), axis=(1, 2))
    return np.maximum(10.0 * np.log10(np.maximum(power, 1e-30)), SILENCE_FLOOR_DB)


def _clean_profile(profile: np.ndarray | None) -> np.ndarray | None:
    if profile is None:
        return None
    p = np.asarray(profile, dtype=np.float64).ravel()
    p = np.where(np.isfinite(p), p, SILENCE_FLOOR_DB)
    return np.maximum(p, SILENCE_FLOOR_DB)


def detect_fade_out(tail_db: np.ndarray | None, frame_s: float = LEVEL_FRAME_S) -> tuple[float, float] | None:
    """``(fade_length_s, decay_db)`` when the outgoing tail already fades out, else None.

    The body level is the 95th percentile of the profile; the end level the median of the
    last ``EDGE_LEVEL_S``. A decay above ``FADE_DECAY_DB`` is a fade; it starts after the
    last frame still within ``FADE_EDGE_DROP_DB`` of the body level.
    """
    p = _clean_profile(tail_db)
    edge = max(1, int(round(EDGE_LEVEL_S / frame_s)))
    if p is None or len(p) < 2 * edge:
        return None
    ref = float(np.percentile(p, 95))
    end = float(np.median(p[-edge:]))
    decay = ref - end
    if decay <= FADE_DECAY_DB:
        return None
    loud = np.nonzero(p >= ref - FADE_EDGE_DROP_DB)[0]
    last_loud = int(loud[-1]) if len(loud) else 0
    return (len(p) - last_loud - 1) * frame_s, decay


def detect_quiet_intro(head_db: np.ndarray | None, frame_s: float = LEVEL_FRAME_S) -> tuple[float, float] | None:
    """``(intro_length_s, rise_db)`` when the incoming head starts well below its body level."""
    p = _clean_profile(head_db)
    edge = max(1, int(round(EDGE_LEVEL_S / frame_s)))
    if p is None or len(p) < 2 * edge:
        return None
    ref = float(np.percentile(p, 95))
    start = float(np.median(p[:edge]))
    rise = ref - start
    if rise <= FADE_DECAY_DB:
        return None
    loud = np.nonzero(p >= ref - FADE_EDGE_DROP_DB)[0]
    first_loud = int(loud[0]) if len(loud) else len(p)
    return first_loud * frame_s, rise


def _valid_bpm(v: float | None) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def tempo_mismatch(bpm_a: float | None, bpm_b: float | None) -> float | None:
    """Relative tempo difference after half/double-time folding (0.28 = 28 % apart);
    ``None`` when either tempo is unknown."""
    if not (_valid_bpm(bpm_a) and _valid_bpm(bpm_b)):
        return None
    a, b = float(bpm_a), float(bpm_b)
    return min(abs(a - b * k) / max(a, b * k) for k in (0.5, 1.0, 2.0))


def _fmt_bpm(v: float) -> str:
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


# ---------------------------------------------------------------------------
# FX choice
# ---------------------------------------------------------------------------

def choose_fx(cfg: TransitionConfig, rng: np.random.Generator | None = None,
              history: Sequence[str] = ()) -> tuple[str, str]:
    """``(fx, reason)`` for one transition. ``history`` = FX of the previous transitions
    (most recent last); ``random`` never yields the same effect three times in a row."""
    mode = cfg.fx_mode
    if mode == "clean":
        return "clean", "FX clean (default: a clean transition is preferable to a distracting effect)"
    if mode in FX_NAMES:
        return mode, f"FX {mode} (fixed by configuration)"
    if mode != "random":
        raise ValueError(f"unknown fx_mode {mode!r}; expected one of {', '.join(FX_MODES)}")
    weights = _fx_weights(cfg)
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    banned = history[-1] if len(history) >= 2 and history[-1] == history[-2] else None
    names = [k for k in FX_NAMES if weights.get(k, 0.0) > 0 and k != banned]
    if not names:   # only the banned effect has weight: fall back to the neutral choice
        return "clean", f"FX clean (random mode: only {banned!r} is weighted and it already ran twice in a row)"
    p = np.array([weights[k] for k in names], dtype=np.float64)
    fx = names[int(rng.choice(len(names), p=p / p.sum()))]
    extra = f"; {banned!r} excluded after two in a row" if banned else ""
    return fx, f"FX {fx} (random mode, seed-controlled{extra})"


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class TransitionDecision:
    """``TransitionPlan`` plus the notable subset of its reasons (``flags``) and the
    measurements behind the decision. ``plan_transition`` returns ``.plan``."""
    plan: TransitionPlan
    flags: list[str] = field(default_factory=list)
    tempo_mismatch: float | None = None
    fade_out_s: float | None = None
    quiet_intro_s: float | None = None


def plan_transition(prev: MixInput, nxt: MixInput, prev_dur_s: float, next_dur_s: float,
                    cfg: TransitionConfig, *, prev_tail_db: np.ndarray | None = None,
                    next_head_db: np.ndarray | None = None, rng=None,
                    fx_history: Sequence[str] = ()) -> TransitionPlan:
    """Plan the transition ``prev -> nxt`` (see ``plan_transition_detailed``)."""
    return plan_transition_detailed(prev, nxt, prev_dur_s, next_dur_s, cfg, prev_tail_db=prev_tail_db,
                                    next_head_db=next_head_db, rng=rng, fx_history=fx_history).plan


def plan_transition_detailed(prev: MixInput, nxt: MixInput, prev_dur_s: float, next_dur_s: float,
                             cfg: TransitionConfig, *, prev_tail_db: np.ndarray | None = None,
                             next_head_db: np.ndarray | None = None, rng=None,
                             fx_history: Sequence[str] = ()) -> TransitionDecision:
    """Crossfade length, curve and FX for one transition, with human-readable reasons.

    Durations are the (trimmed) lengths of the adjacent tracks. ``prev_tail_db`` /
    ``next_head_db`` are ``level_profile_db`` profiles (``LEVEL_FRAME_S`` frames) of the
    outgoing tail / incoming head; ``None`` = not analysed.

    Order: disabled / too-short track -> butt splice; default length; grow to cover a
    fade-out or quiet intro (<= ``max_overlap_s``); halve on a big energy difference;
    cap at ``conservative_crossfade_s`` and flag ``uncertain`` when the tempo is unknown
    or too different (no beat alignment exists); clamp to ``max_overlap_s`` and
    ``max_fraction_of_track`` of either track; below ``min_crossfade_s`` -> butt splice.
    """
    validate_transition_config(cfg)
    for label, d in (("prev_dur_s", prev_dur_s), ("next_dur_s", next_dur_s)):
        if not isinstance(d, (int, float)) or not math.isfinite(d) or d < 0:
            raise ValueError(f"{label}={d!r} must be a finite duration >= 0")

    reasons: list[str] = []
    flags: list[str] = []

    def note(msg: str, *, flag: bool = True) -> None:
        reasons.append(msg)
        if flag:
            flags.append(msg)

    splice = f"butt splice with {cfg.declick_ms:g} ms de-click fades"
    mismatch = tempo_mismatch(prev.bpm, nxt.bpm)
    fade = detect_fade_out(prev_tail_db)
    intro = detect_quiet_intro(next_head_db)
    uncertain = False
    xf = 0.0

    if not cfg.enabled:
        note(f"crossfades disabled — {splice}")
    elif min(prev_dur_s, next_dur_s) < cfg.min_track_s_for_crossfade:
        which, d = ("outgoing", prev_dur_s) if prev_dur_s <= next_dur_s else ("incoming", next_dur_s)
        note(f"{which} track is only {d:.1f} s (< {cfg.min_track_s_for_crossfade:g} s) — no crossfade, {splice}")
    else:
        xf = float(cfg.default_crossfade_s)
        note(f"default crossfade {xf:.1f} s", flag=False)
        # --- material: an existing fade-out / quiet intro is a natural overlap zone ---
        if fade is not None and fade[0] > xf:
            grown = min(fade[0], cfg.max_overlap_s)
            note(f"outgoing track already fades out over its last {fade[0]:.1f} s ({fade[1]:.0f} dB decay) — "
                 f"overlap extended to {grown:.1f} s to cover the fade")
            xf = max(xf, grown)
        if intro is not None and intro[0] > xf:
            grown = min(intro[0], cfg.max_overlap_s)
            note(f"incoming track has a quiet intro of {intro[0]:.1f} s ({intro[1]:.0f} dB below its body) — "
                 f"overlap extended to {grown:.1f} s to cover it")
            xf = max(xf, grown)
        # --- energy: do not smear two very different energy levels over a long overlap ---
        if prev.energy is not None and nxt.energy is not None:
            d_e = abs(float(nxt.energy) - float(prev.energy))
            if d_e > ENERGY_JUMP:
                xf *= ENERGY_JUMP_FACTOR
                note(f"large energy difference {prev.energy:.2f} → {nxt.energy:.2f} (|ΔE| {d_e:.2f} > "
                     f"{ENERGY_JUMP:.2f}) — overlap shortened to {xf:.1f} s")
        # --- tempo: no beat/phrase detection exists, so long overlaps of different tempos clash ---
        tempo_detail = None
        if mismatch is None:
            unknown = [w for w, b in (("outgoing", prev.bpm), ("incoming", nxt.bpm)) if not _valid_bpm(b)]
            tempo_detail = f"BPM unknown ({' and '.join(unknown)})"
        elif mismatch > cfg.uncertain_tempo_ratio:
            tempo_detail = (f"tempo {_fmt_bpm(prev.bpm)} vs {_fmt_bpm(nxt.bpm)} BPM "
                            f"({mismatch * 100:.0f}% apart)")
        else:
            note(f"tempos close ({_fmt_bpm(prev.bpm)} vs {_fmt_bpm(nxt.bpm)} BPM, {mismatch * 100:.1f}% apart "
                 f"after half/double folding); beats are not phase-aligned (no beat detection)", flag=False)
        if tempo_detail is not None:
            uncertain = True
            xf = min(xf, cfg.conservative_crossfade_s)
        # --- hard limits ---
        limits = [(cfg.max_overlap_s, f"max_overlap_s {cfg.max_overlap_s:g} s"),
                  (cfg.max_fraction_of_track * prev_dur_s,
                   f"{cfg.max_fraction_of_track:g} × outgoing track ({prev_dur_s:.1f} s)"),
                  (cfg.max_fraction_of_track * next_dur_s,
                   f"{cfg.max_fraction_of_track:g} × incoming track ({next_dur_s:.1f} s)")]
        limit, why = min(limits, key=lambda lw: lw[0])
        if xf > limit:
            xf = max(0.0, limit)
            note(f"crossfade clamped to {xf:.2f} s by {why}")
        if 0.0 < xf < cfg.min_crossfade_s or xf <= 0.0:
            note(f"computed crossfade {xf:.2f} s < minimum {cfg.min_crossfade_s:g} s — {splice}")
            xf = 0.0
            uncertain = False   # a cut has no beat clash; the reason above documents the cut
        elif tempo_detail is not None:
            note(f"no beat/phrase alignment; {tempo_detail} — shortened to {xf:.1f} s, listen")

    if xf > 0.0:
        fx, fx_reason = choose_fx(cfg, rng, fx_history)
        note(fx_reason, flag=fx != "clean")
    else:
        fx = "clean"
        if cfg.fx_mode != "clean":
            note(f"FX not applied (fx_mode {cfg.fx_mode!r}): no overlap region at a butt splice")

    plan = TransitionPlan(position=nxt.position, crossfade_s=round(xf, 6), curve=cfg.curve, fx=fx,
                          reasons=reasons, uncertain=uncertain)
    return TransitionDecision(plan=plan, flags=flags, tempo_mismatch=mismatch,
                              fade_out_s=fade[0] if fade else None, quiet_intro_s=intro[0] if intro else None)


# ---------------------------------------------------------------------------
# FX (outgoing transition region only)
# ---------------------------------------------------------------------------

def fx_extension_samples(fx: str, sr: int) -> int:
    """Samples an effect adds after the region end (echo tail ring-out)."""
    if fx in ("echo_tail", "filter+echo"):
        return max(int(round(d * sr)) for d, _ in ECHO_TAPS)
    return 0


def apply_fx(region: np.ndarray, sr: int, fx: str) -> np.ndarray:
    """Apply a transition effect to the outgoing track's transition region.

    ``region`` is never modified: the result is always a NEW float32 array. Output
    length is ``len(region)`` except for echo effects, which append the ring-out of the
    delay taps (``fx_extension_samples``) so the echoes are not truncated.
    """
    if fx not in FX_NAMES:
        raise ValueError(f"unknown FX {fx!r}; expected one of {', '.join(FX_NAMES)}")
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError(f"sample rate must be a positive integer, got {sr!r}")
    src = np.asarray(region)
    if src.ndim not in (1, 2):
        raise ValueError(f"region must be shaped (n,) or (n, channels), got {src.shape}")
    x = np.array(src, dtype=np.float64, copy=True)     # private copy: the caller's buffer is untouched
    mono = x.ndim == 1
    if mono:
        x = x[:, None]
    if x.size and not np.isfinite(x).all():
        raise ValueError("region contains non-finite samples")
    if fx == "clean" or len(x) == 0:
        y = x
    elif fx == "filter_sweep":
        y = _filter_sweep(x, int(sr))
    elif fx == "echo_tail":
        y = _echo_tail(x, int(sr))
    else:  # filter+echo
        y = _echo_tail(_filter_sweep(x, int(sr)), int(sr))
    out = y.astype(np.float32)
    return out[:, 0] if mono else out


def _filter_sweep(x: np.ndarray, sr: int) -> np.ndarray:
    """Smooth low-pass sweep 10 kHz -> 200 Hz over the region.

    The region is filtered by ``SWEEP_STEPS`` fixed Butterworth low-passes (log-spaced
    cutoffs); the output crossfades sample-wise between neighbouring filter outputs, so
    the cutoff moves continuously (no zipper clicks from per-block coefficient changes).
    The sweep starts from the dry signal, so the first sample equals the input."""
    n = len(x)
    if n < 2:
        return x.copy()
    nyq = 0.5 * sr
    cutoffs = np.geomspace(min(SWEEP_START_HZ, 0.45 * sr), min(SWEEP_END_HZ, 0.45 * sr), SWEEP_STEPS)
    # sweep position: -1 (dry) at the first sample -> SWEEP_STEPS-1 (200 Hz) at the last
    pos = -1.0 + SWEEP_STEPS * (np.arange(n, dtype=np.float64) / (n - 1))
    out = np.zeros_like(x)
    for k in range(-1, SWEEP_STEPS):
        w = np.clip(1.0 - np.abs(pos - k), 0.0, 1.0)      # triangular weights: neighbours sum to 1
        nz = np.nonzero(w)[0]
        if len(nz) == 0:
            continue
        a, b = int(nz[0]), int(nz[-1]) + 1
        if k < 0:
            y = x[a:b]
        else:
            sos = butter(SWEEP_ORDER, cutoffs[k] / nyq, btype="lowpass", output="sos")
            # steady-state start from the region's first sample: no start-up transient
            zi = sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
            y = sosfilt(sos, x[:b], axis=0, zi=zi)[0][a:b]
        out[a:b] += w[a:b, None] * y
    return out


def smooth_reduction_db(need_db: np.ndarray, hold_n: int, *, edge_n: int = 0,
                        pin_start: bool = True, pin_end: bool = True) -> np.ndarray:
    """Smooth gain-reduction curve (dB, >= 0) that covers a per-sample requirement.

    The requirement is held (max filter) over ``hold_n`` samples and smoothed with a
    raised-cosine (Hann) kernel of the same length; wherever the kernel lies inside the
    signal the result is provably >= ``need_db`` (every averaged hold value contains the
    sample). Optionally pinned to 0 dB with raised-cosine ramps over ``edge_n`` samples
    at the start/end so the gain is continuous with untouched audio around it."""
    need = np.maximum(np.asarray(need_db, dtype=np.float64), 0.0)
    n = len(need)
    if n == 0 or not np.any(need > 0):
        return np.zeros(n)
    hold = max(1, min(int(hold_n), n))
    hold -= 1 - hold % 2                                  # odd length -> centred kernel
    kernel = hann(hold + 2)[1:-1]
    kernel /= kernel.sum()
    held = maximum_filter1d(need, size=hold, mode="constant", cval=0.0)
    red = np.clip(fftconvolve(held, kernel, mode="same"), 0.0, None)
    edge = min(int(edge_n), n // 4)
    if edge > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(edge) + 0.5) / edge)
        if pin_start:
            red[:edge] *= ramp
        if pin_end:
            red[n - edge:] *= ramp[::-1]
    return red


def _echo_tail(x: np.ndarray, sr: int) -> np.ndarray:
    """Three delay taps (``ECHO_TAPS``) at exact sample offsets.

    * the wet send ramps in with a raised cosine over the first ``ECHO_RAMP_FRACTION``
      of the region (no abrupt echo onset);
    * the output is ``len(region) + max delay`` long: echoes ring out past the region;
    * gain staging: where dry+wet would peak above the dry region's peak, a smooth
      compensating gain (``smooth_reduction_db``, 0 dB at the region start) is applied
      to the sum, so the echo stays audible on dense material; a constant wet scale
      (bisection) removes any residual excess. The peak is never increased."""
    n, ch = x.shape
    taps = [(int(round(d * sr)), 10.0 ** (g / 20.0)) for d, g in ECHO_TAPS]
    ext = max(d for d, _ in taps)
    ramp_n = max(1, int(round(n * ECHO_RAMP_FRACTION)))
    send = np.ones(n, dtype=np.float64)
    k = min(ramp_n, n)
    send[:k] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(k) + 0.5) / ramp_n)
    sent = x * send[:, None]
    wet = np.zeros((n + ext, ch), dtype=np.float64)
    for d, g in taps:
        wet[d:d + n] += g * sent
    dry = np.zeros_like(wet)
    dry[:n] = x
    dry_peak = float(np.max(np.abs(x))) if n else 0.0
    y = dry + wet
    if dry_peak > 0 and float(np.max(np.abs(y))) > dry_peak:
        env = np.max(np.abs(y), axis=1)
        need = 20.0 * np.log10(np.maximum(env, 1e-12) / dry_peak)
        red = smooth_reduction_db(need, int(round(ECHO_STAGING_HOLD_S * sr)),
                                  edge_n=int(round(ECHO_STAGING_EDGE_S * sr)), pin_end=False)
        g = (10.0 ** (-red / 20.0))[:, None]
        dry, wet = dry * g, wet * g
        y = dry + wet
        log.debug("echo gain staging: compensating gain down to %.2f dB", -float(red.max()))
        if float(np.max(np.abs(y))) > dry_peak:            # residual near the pinned start
            scale = _max_wet_scale(dry, wet, dry_peak)
            log.debug("echo gain staging: residual wet scale %.3f", scale)
            y = dry + scale * wet
    return y


def _max_wet_scale(dry: np.ndarray, wet: np.ndarray, limit: float) -> float:
    """Largest ``s`` in [0, 1] with ``max|dry + s·wet| <= limit``. The peak is convex in
    ``s`` and equals the dry peak at ``s = 0``, so the feasible set is an interval [0, s*]."""
    active = np.any(wet != 0, axis=1)
    d, w = dry[active], wet[active]
    lo, hi = 0.0, 1.0
    for _ in range(ECHO_SEARCH_STEPS):
        mid = 0.5 * (lo + hi)
        if float(np.max(np.abs(d + mid * w))) <= limit:
            lo = mid
        else:
            hi = mid
    return lo
