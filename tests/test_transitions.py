"""P1 transition engine + Mode B renderer (spec §8, §13 Transitions 1-5 and more)."""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from conftest import SR, music_like, requires_ffmpeg, sha256, write_wav
from veg_mixer import transitions
from veg_mixer.config import LoudnessConfig, TransitionConfig
from veg_mixer.models import MixInput, MixRenderResult, MixTrackEntry, TransitionReport
from veg_mixer.renderer import (
    ms_to_hms,
    render_mix,
    trim_silence_bounds,
    true_peak_envelope,
    write_mix_tracklist,
)
from veg_mixer.transitions import (
    ECHO_TAPS,
    LEVEL_FRAME_S,
    apply_fx,
    fade_curves,
    fx_extension_samples,
    plan_transition,
    plan_transition_detailed,
    tempo_mismatch,
)

CFG = TransitionConfig()
QC_ROW_RE = re.compile(r"^(\d+)\s+\|\s+(\d+:\d{2}:\d{2})\s+\|")   # quality_control.py parses rows with this


def cfg(**kw) -> TransitionConfig:
    return dataclasses.replace(CFG, **kw)


def mi(pos: int, bpm: float | None = 120.0, energy: float | None = None, camelot: str | None = "8A",
       path: Path | None = None, name: str | None = None) -> MixInput:
    return MixInput(position=pos, pcm_path=path or Path(f"t{pos}.wav"), name=name or f"Track {pos}",
                    source_rel=f"t{pos}.mp3", bpm=bpm, energy=energy, camelot=camelot)


def make_inputs(tmp_path: Path, signals: list[np.ndarray], **kw) -> list[MixInput]:
    out = []
    for i, sig in enumerate(signals, start=1):
        p = write_wav(tmp_path / "in" / f"{i:04d}.wav", sig)
        out.append(mi(i, path=p, **kw))
    return out


def tp_dbtp(x: np.ndarray, os_: int = 4) -> float:
    """Independent true-peak estimate for assertions (oversampled abs max)."""
    return 20 * math.log10(float(np.max(np.abs(resample_poly(np.asarray(x, np.float64), os_, 1, axis=0)))))


def scale_to_tp(x: np.ndarray, target_dbtp: float) -> np.ndarray:
    return (x * 10 ** ((target_dbtp - tp_dbtp(x)) / 20)).astype(np.float32)


def flat_profile(seconds: float = 12.0, level: float = -14.0) -> np.ndarray:
    return np.full(int(seconds / LEVEL_FRAME_S), level)


def read(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR
    return data


# ---------------------------------------------------------------------------
# Fade curves
# ---------------------------------------------------------------------------

def test_fade_curve_properties():
    for curve in ("linear", "equal_power", "s_curve"):
        fo, fi = fade_curves(1000, curve)
        assert fo.dtype == np.float32 and fi.dtype == np.float32 and len(fo) == len(fi) == 1000
        assert np.all(np.diff(fi) > 0) and np.all(np.diff(fo) < 0)          # monotonic
        np.testing.assert_allclose(fo, fi[::-1], atol=1e-6)                  # symmetric pair
        assert 0 < fi[0] < 0.01 and 0.99 < fi[-1] < 1
    fo, fi = fade_curves(1000, "equal_power")
    np.testing.assert_allclose(fo.astype(np.float64) ** 2 + fi.astype(np.float64) ** 2, 1.0, atol=1e-6)
    for curve in ("linear", "s_curve"):
        fo, fi = fade_curves(1000, curve)
        np.testing.assert_allclose(fo.astype(np.float64) + fi, 1.0, atol=1e-6)   # amplitude sums to 1
    lin = fade_curves(1000, "linear")[1]
    np.testing.assert_allclose(np.diff(lin), 1 / 1000, atol=1e-6)            # linear in amplitude
    s = fade_curves(1000, "s_curve")[1]
    assert np.diff(s)[0] < np.diff(s)[500] / 50                              # zero slope at the ends
    assert all(len(c) == 0 for c in fade_curves(0, "linear"))
    with pytest.raises(ValueError, match="fade curve"):
        fade_curves(10, "cubic")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def test_default_mode_is_clean_and_default_crossfade():
    assert CFG.fx_mode == "clean"
    plan = plan_transition(mi(1, 120), mi(2, 121), 200.0, 180.0, CFG,
                           prev_tail_db=flat_profile(), next_head_db=flat_profile())
    assert plan.position == 2 and plan.fx == "clean" and plan.curve == "equal_power"
    assert plan.crossfade_s == CFG.default_crossfade_s and plan.uncertain is False
    assert plan.reasons and any("default crossfade" in r for r in plan.reasons)


def test_crossfade_follows_material_not_bpm_alone():
    """Same BPM pair, different material -> different crossfades (v1 used BPM difference only)."""
    tail_fade = np.concatenate([flat_profile(4.0), np.linspace(-14.0, -45.0, int(8.0 / LEVEL_FRAME_S))])
    d = plan_transition_detailed(mi(1, 120), mi(2, 120), 200.0, 200.0, CFG,
                                 prev_tail_db=tail_fade, next_head_db=flat_profile())
    assert d.fade_out_s is not None and 6.5 <= d.plan.crossfade_s <= 8.0
    assert any("fades out" in r for r in d.plan.reasons)
    plain = plan_transition(mi(1, 120), mi(2, 120), 200.0, 200.0, CFG,
                            prev_tail_db=flat_profile(), next_head_db=flat_profile())
    assert plain.crossfade_s == 6.0 < d.plan.crossfade_s
    # a fade longer than max_overlap_s is covered only up to the limit
    long_fade = np.linspace(-14.0, -50.0, int(12.0 / LEVEL_FRAME_S))
    long_fade[:3] = -14.0
    p = plan_transition(mi(1, 120), mi(2, 120), 200.0, 200.0, cfg(max_overlap_s=9.0), prev_tail_db=long_fade)
    assert p.crossfade_s == 9.0


def test_quiet_intro_extends_overlap():
    head = np.concatenate([np.full(int(7.0 / LEVEL_FRAME_S), -34.0), flat_profile(5.0)])
    d = plan_transition_detailed(mi(1), mi(2), 200.0, 200.0, CFG, next_head_db=head)
    assert d.quiet_intro_s == pytest.approx(7.0) and d.plan.crossfade_s == pytest.approx(7.0)
    assert any("quiet intro" in r for r in d.plan.reasons)


def test_big_energy_difference_shortens_overlap():
    p = plan_transition(mi(1, energy=0.2), mi(2, energy=0.8), 200.0, 200.0, CFG)
    assert p.crossfade_s == pytest.approx(3.0) and any("energy" in r for r in p.reasons)
    same = plan_transition(mi(1, energy=0.5), mi(2, energy=0.6), 200.0, 200.0, CFG)
    assert same.crossfade_s == 6.0
    unknown = plan_transition(mi(1, energy=None), mi(2, energy=0.9), 200.0, 200.0, CFG)
    assert unknown.crossfade_s == 6.0          # unknown energy is not treated as a jump (nor as 0)


def test_uncertain_transition_for_tempo_mismatch():
    assert tempo_mismatch(92, 128) == pytest.approx(0.28125)
    assert tempo_mismatch(64, 128) == 0.0      # half/double time folds
    d = plan_transition_detailed(mi(1, 92), mi(2, 128), 200.0, 200.0, CFG)
    p = d.plan
    assert p.uncertain and p.crossfade_s == CFG.conservative_crossfade_s
    reason = next(r for r in p.reasons if "no beat/phrase alignment" in r)
    assert "92 vs 128 BPM (28% apart)" in reason and "shortened to 3.0 s" in reason and "listen" in reason
    assert reason in d.flags                   # surfaced to reports
    assert not plan_transition(mi(1, 64), mi(2, 128), 200.0, 200.0, CFG).uncertain
    assert not plan_transition(mi(1, 120), mi(2, 123), 200.0, 200.0, CFG).uncertain


def test_uncertain_transition_for_unknown_bpm():
    p = plan_transition(mi(1, None), mi(2, 120), 200.0, 200.0, CFG)
    assert p.uncertain and p.crossfade_s == 3.0
    assert any("BPM unknown (outgoing)" in r for r in p.reasons)
    p = plan_transition(mi(1, float("nan")), mi(2, None), 200.0, 200.0, CFG)
    assert p.uncertain and any("outgoing and incoming" in r for r in p.reasons)


def test_short_tracks_clamps_and_disabled():
    p = plan_transition(mi(1), mi(2), 15.0, 200.0, CFG)                      # < min_track_s_for_crossfade
    assert p.crossfade_s == 0.0 and any("butt splice" in r and "15.0 s" in r for r in p.reasons)
    p = plan_transition(mi(1), mi(2), 200.0, 0.5, CFG)
    assert p.crossfade_s == 0.0 and p.fx == "clean"
    p = plan_transition(mi(1), mi(2), 30.0, 200.0, cfg(default_crossfade_s=12.0, max_overlap_s=20.0))
    assert p.crossfade_s == pytest.approx(0.25 * 30.0)                        # max_fraction_of_track
    assert any("clamped" in r for r in p.reasons)
    p = plan_transition(mi(1), mi(2), 200.0, 200.0, cfg(default_crossfade_s=12.0, max_overlap_s=4.0))
    assert p.crossfade_s == 4.0
    p = plan_transition(mi(1), mi(2), 21.0, 200.0, cfg(min_crossfade_s=6.0))  # 5.25 s < min -> cut
    assert p.crossfade_s == 0.0 and any("minimum" in r for r in p.reasons)
    p = plan_transition(mi(1, 92), mi(2, 128), 200.0, 200.0, cfg(enabled=False, fx_mode="echo_tail"))
    assert p.crossfade_s == 0.0 and p.fx == "clean" and not p.uncertain
    assert any("disabled" in r for r in p.reasons)


@pytest.mark.parametrize("kw, fragment", [
    ({"fx_mode": "wobble"}, "fx_mode"),
    ({"curve": "cubic"}, "fade curve"),
    ({"max_fraction_of_track": 0.8}, "max_fraction_of_track"),
    ({"fx_mode": "random", "fx_weights": {"flanger": 1.0}}, "unknown effect"),
    ({"fx_mode": "random", "fx_weights": {"clean": 0.0}}, "not all zero"),
])
def test_invalid_config_raises(kw, fragment):
    with pytest.raises(ValueError, match=fragment):
        plan_transition(mi(1), mi(2), 200.0, 200.0, cfg(**kw))


def test_random_fx_is_deterministic_and_never_three_in_a_row():
    def run(seed: int, c: TransitionConfig) -> list[str]:
        rng, hist = np.random.default_rng(seed), []
        for k in range(300):
            hist.append(plan_transition(mi(k + 1), mi(k + 2), 200.0, 200.0, c, rng=rng, fx_history=hist).fx)
        return hist

    c = cfg(fx_mode="random")
    a, b = run(7, c), run(7, c)
    assert a == b                                    # same seed -> same choices
    assert run(8, c) != a
    assert set(a) == set(transitions.FX_NAMES)       # weights include all four effects
    biased = run(1, cfg(fx_mode="random", fx_weights={"echo_tail": 100.0, "clean": 1.0}))
    for s in (a, biased):
        assert not any(s[i] == s[i + 1] == s[i + 2] for i in range(len(s) - 2))
    assert plan_transition(mi(1), mi(2), 200.0, 200.0, cfg(fx_mode="filter_sweep")).fx == "filter_sweep"


# ---------------------------------------------------------------------------
# FX
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", transitions.FX_NAMES)
def test_apply_fx_returns_new_array_and_never_mutates_input(fx):
    region = music_like(1.5, peak_dbfs=-6, seed=3)
    before = region.copy()
    out = apply_fx(region, SR, fx)
    assert np.array_equal(region, before)
    assert not np.shares_memory(out, region) and out.dtype == np.float32
    assert len(out) == len(region) + fx_extension_samples(fx, SR)
    if fx == "clean":
        assert np.array_equal(out, region)
    with pytest.raises(ValueError, match="unknown FX"):
        apply_fx(region, SR, "flanger")


def test_echo_impulse_taps_offsets_gains_and_tail():
    n = SR                                   # 1 s region
    i0 = int(0.9 * SR)                       # after the 25 % send ramp
    x = np.zeros((n, 2), np.float32)
    x[i0] = 0.8
    y = apply_fx(x, SR, "echo_tail").astype(np.float64)
    ext = int(round(0.300 * SR))
    assert len(y) == n + ext                 # ring-out appended, not truncated
    expected = np.zeros((n + ext, 2))
    expected[i0] = 0.8
    for d_s, g_db in ECHO_TAPS:
        d = int(round(d_s * SR))
        expected[i0 + d] = 0.8 * 10 ** (g_db / 20)
    np.testing.assert_allclose(y, expected, atol=1e-6)
    assert np.count_nonzero(np.abs(y[:, 0]) > 1e-9) == 1 + len(ECHO_TAPS)
    assert (i0 + int(round(0.160 * SR))) >= n and np.abs(y[n:]).max() > 0.01   # taps beyond the region end
    assert np.abs(y).max() <= 0.8 + 1e-7                                       # peak not increased
    # smooth onset: an impulse at the very start of the region produces (almost) no echo
    x2 = np.zeros((n, 2), np.float32)
    x2[0] = 0.8
    y2 = apply_fx(x2, SR, "echo_tail")
    assert np.abs(y2[1:]).max() < 1e-3


def test_echo_gain_staging_never_raises_the_peak():
    n = 2 * SR
    x = np.zeros((n, 2), np.float32)
    x[::int(round(0.080 * SR))] = 0.9         # pulses every 80 ms: taps land on later pulses
    naive = x.astype(np.float64).copy()
    for d_s, g_db in ECHO_TAPS:
        d = int(round(d_s * SR))
        naive[d:] += 10 ** (g_db / 20) * x[:n - d]
    assert np.abs(naive).max() > 1.0          # without gain staging it would overshoot
    y = apply_fx(x, SR, "echo_tail")
    assert np.abs(y).max() <= 0.9 + 1e-6
    assert np.abs(y[n:]).max() > 0.01         # the wet is still there, just staged


def test_filter_sweep_attenuates_hf_toward_the_end_without_zipper():
    rng = np.random.default_rng(5)
    noise = (0.3 * rng.standard_normal((2 * SR, 2))).astype(np.float32)
    y = apply_fx(noise, SR, "filter_sweep")
    assert np.array_equal(y[0], noise[0])     # starts from the dry signal

    def hf_db(seg):
        spec = np.abs(np.fft.rfft(seg[:, 0].astype(np.float64))) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        return 10 * math.log10(spec[f > 4000].sum() + 1e-30)

    # HF energy falls monotonically along the region (the cutoff only moves down) ...
    tenth = len(noise) // 10
    hf = [hf_db(y[k * tenth:(k + 1) * tenth]) for k in range(10)]
    assert all(b < a for a, b in zip(hf, hf[1:]))
    assert hf[-1] < hf[0] - 30
    # ... and the first few ms are (almost) the dry signal: seamless with the audio before the region
    head = int(0.005 * SR)
    assert np.abs(y[:head] - noise[:head]).max() < 0.05 * np.abs(noise[:head]).max()
    # a 100 Hz tone (below every cutoff) passes with a smooth envelope: no zipper steps
    t = np.arange(2 * SR) / SR
    tone = np.repeat((0.5 * np.sin(2 * np.pi * 100 * t))[:, None], 2, axis=1).astype(np.float32)
    z = apply_fx(tone, SR, "filter_sweep").astype(np.float64)
    frame = int(0.02 * SR)
    rms = np.sqrt(np.mean(z[: len(z) // frame * frame, 0].reshape(-1, frame) ** 2, axis=1))
    rms_db = 20 * np.log10(rms[1:-1])
    assert np.abs(np.diff(rms_db)).max() < 0.1
    assert rms_db.min() > 20 * math.log10(0.5 / math.sqrt(2)) - 1.0


# ---------------------------------------------------------------------------
# Rendering: durations, accounting, trimming
# ---------------------------------------------------------------------------

def test_transition_duration_overlap_equals_planned_crossfade(tmp_path):
    """Spec Transitions 1: DC tracks make the overlap exactly visible in the output."""
    a = np.full((25 * SR, 2), 0.3, np.float32)
    b = np.full((25 * SR, 2), -0.3, np.float32)
    inputs = make_inputs(tmp_path, [a, b])
    c = cfg(mix_fade_in_s=0.0, mix_fade_out_s=0.0, curve="linear")
    r = render_mix(inputs, tmp_path / "out" / "mix.wav", c, ceiling_dbtp=-1.0)
    tr = r.transitions[0]
    xf_n = int(round(CFG.default_crossfade_s * SR))
    assert tr.crossfade_s * SR == xf_n and tr.fx == "clean" and tr.gain_reduction_db == 0.0
    s = int(round(tr.overlap_start_s * SR))
    assert s == len(a) - xf_n == int(round(r.tracks[1].start_s * SR))
    y = read(r.wav_path)
    fo, fi = fade_curves(xf_n, "linear")
    assert np.all(y[:s] == np.float32(0.3)) and np.all(y[s + xf_n:] == np.float32(-0.3))
    np.testing.assert_allclose(y[s:s + xf_n, 0], 0.3 * fo - 0.3 * fi, atol=2e-7)
    assert len(y) == len(a) + len(b) - xf_n


def test_output_duration_accounting_with_trim(tmp_path):
    """Spec Transitions 2: file frames == sum(trimmed durations) - sum(crossfades)."""
    def padded(seconds, head, tail, seed):
        core = music_like(seconds, peak_dbfs=-4, seed=seed)
        return np.concatenate([np.zeros((int(head * SR), 2), np.float32), core,
                               np.zeros((int(tail * SR), 2), np.float32)])

    sigs = [padded(27.3, 1.0, 0.5, 1), padded(31.1, 0.0, 2.0, 2), padded(24.6, 0.3, 0.0, 3), padded(22.2, 0, 0, 4)]
    inputs = make_inputs(tmp_path, sigs)
    calls = []
    r = render_mix(inputs, tmp_path / "out" / "mix.wav", CFG, ceiling_dbtp=-1.0,
                   progress=lambda done, total, msg: calls.append((done, total, msg)))
    assert calls == [(k, 4, f"Track {k}") for k in range(1, 5)]
    info = sf.info(str(r.wav_path))
    assert info.frames == round(r.expected_duration_s * SR)
    assert r.duration_s == pytest.approx(r.expected_duration_s, abs=1e-9)
    assert r.expected_duration_s == pytest.approx(
        sum(t.duration_s for t in r.tracks) - sum(t.crossfade_s for t in r.transitions), abs=1e-9)
    first = r.tracks[0]
    assert first.head_trim_s == pytest.approx(0.9) and first.tail_trim_s == pytest.approx(0.4)
    assert first.duration_s == pytest.approx(len(sigs[0]) / SR - 1.3)
    assert r.tracks[2].head_trim_s == pytest.approx(0.2)        # 0.3 s silence - 100 ms safety
    for prev, nxt, tr in zip(r.tracks, r.tracks[1:], r.transitions):
        assert nxt.start_s == pytest.approx(prev.start_s + prev.duration_s - tr.crossfade_s, abs=1e-9)
        assert nxt.start_s == pytest.approx(tr.overlap_start_s, abs=1e-9)
        assert nxt.crossfade_in_s == tr.crossfade_s and nxt.fx_in == tr.fx
        assert nxt.end_s == pytest.approx(nxt.start_s + nxt.duration_s, abs=1e-9)
    assert r.tracks[-1].end_s == pytest.approx(r.duration_s, abs=1e-9)
    assert r.tracks[0].fx_in == "—" and r.tracks[0].start_s == 0.0
    assert not list((tmp_path / "out").glob(".*veg-*"))          # temp file replaced atomically


def test_trim_silence_bounds_follow_legacy_rules():
    core = music_like(5, peak_dbfs=-6, seed=9)
    z = lambda s: np.zeros((int(s * SR), 2), np.float32)        # noqa: E731
    start, end = trim_silence_bounds(np.concatenate([z(1.0), core, z(0.5)]), SR)
    assert start == int(0.9 * SR) and end == int(0.9 * SR) + int(0.2 * SR) + len(core)
    # silence shorter than 200 ms (+100 ms safety) is left alone
    x = np.concatenate([z(0.15), core, z(0.1)])
    assert trim_silence_bounds(x, SR) == (0, len(x))
    assert trim_silence_bounds(z(3.0), SR) == (0, 0)              # silent: nothing to keep


def test_mix_fade_in_and_out(tmp_path):
    inputs = make_inputs(tmp_path, [np.full((25 * SR, 2), 0.25, np.float32)] * 2)
    r = render_mix(inputs, tmp_path / "mix.wav", CFG, ceiling_dbtp=-1.0)
    y = read(r.wav_path)[:, 0]
    assert abs(y[0]) < 1e-4 and abs(y[-1]) < 1e-4
    assert y[int(1.5 * SR)] == pytest.approx(0.125, abs=0.01)    # half-way through the 3 s fade-in
    assert y[int(3.0 * SR) + 10] == np.float32(0.25)
    assert y[-int(2.5 * SR)] == pytest.approx(0.125, abs=0.01)   # half-way through the 5 s fade-out


# ---------------------------------------------------------------------------
# FX never alters earlier rendered audio (spec Transitions 3)
# ---------------------------------------------------------------------------

def test_fx_never_alters_audio_before_the_overlap(tmp_path):
    inputs = make_inputs(tmp_path, [music_like(25, peak_dbfs=-6, seed=1), music_like(25, peak_dbfs=-6, seed=2)])
    renders = {fx: render_mix(inputs, tmp_path / f"{fx.replace('+', '_')}.wav", cfg(fx_mode=fx), ceiling_dbtp=-1.0)
               for fx in transitions.FX_NAMES}
    audio = {fx: read(r.wav_path) for fx, r in renders.items()}
    s = int(round(renders["clean"].transitions[0].overlap_start_s * SR))
    xf_n = int(round(renders["clean"].transitions[0].crossfade_s * SR))
    for fx in ("filter_sweep", "echo_tail", "filter+echo"):
        assert renders[fx].transitions[0].fx == fx
        assert len(audio[fx]) == len(audio["clean"])
        assert np.array_equal(audio[fx][:s], audio["clean"][:s])            # bit-identical before
        assert not np.array_equal(audio[fx][s:s + xf_n], audio["clean"][s:s + xf_n])
        zone_end = s + xf_n + fx_extension_samples(fx, SR)
        assert np.array_equal(audio[fx][zone_end:], audio["clean"][zone_end:])  # and after the zone
        if fx != "filter_sweep":
            # the echo rings out under the incoming track (not truncated) without a jump at the region end
            wet = audio[fx].astype(np.float64) - audio["clean"]
            assert np.abs(wet[s + xf_n:zone_end]).max() > 0
            assert np.abs(np.diff(wet[s + xf_n - 100:s + xf_n + 100], axis=0)).max() < 1e-3


def test_fx_only_touch_transition_zones_in_longer_mix(tmp_path):
    sigs = [music_like(s, peak_dbfs=-6, seed=k) for k, s in enumerate((24, 22, 26), start=1)]
    inputs = make_inputs(tmp_path, sigs)
    clean = render_mix(inputs, tmp_path / "clean.wav", CFG, ceiling_dbtp=-1.0)
    y_clean = read(clean.wav_path)
    ext = fx_extension_samples("echo_tail", SR)
    outside = np.ones(len(y_clean), bool)
    for tr in clean.transitions:
        s = int(round(tr.overlap_start_s * SR))
        outside[s:s + int(round(tr.crossfade_s * SR)) + ext] = False
    for fx in ("echo_tail", "filter+echo"):
        r = render_mix(inputs, tmp_path / f"{fx.replace('+', '_')}.wav", cfg(fx_mode=fx), ceiling_dbtp=-1.0)
        y = read(r.wav_path)
        assert [t.overlap_start_s for t in r.transitions] == [t.overlap_start_s for t in clean.transitions]
        assert np.array_equal(y[outside], y_clean[outside])
        assert not np.array_equal(y[~outside], y_clean[~outside])


def test_apply_fx_receives_only_a_copy_of_the_outgoing_region(tmp_path, monkeypatch):
    a, b = music_like(25, peak_dbfs=-6, seed=1), music_like(25, peak_dbfs=-6, seed=2)
    inputs = make_inputs(tmp_path, [a, b])
    calls = []
    original = transitions.apply_fx

    def spy(region, sr, fx):
        snapshot = region.copy()
        calls.append((snapshot, bool(region.flags.owndata), fx))
        out = original(region, sr, fx)
        assert np.array_equal(region, snapshot)          # not mutated
        return out

    monkeypatch.setattr(transitions, "apply_fx", spy)
    r = render_mix(inputs, tmp_path / "mix.wav", cfg(fx_mode="echo_tail"), ceiling_dbtp=-1.0)
    assert len(calls) == 1
    region, owndata, fx = calls[0]
    xf_n = int(round(r.transitions[0].crossfade_s * SR))
    assert fx == "echo_tail" and owndata and region.shape == (xf_n, 2)
    fo, _ = fade_curves(xf_n, "equal_power")
    assert np.array_equal(region, a[-xf_n:] * fo[:, None])   # exactly the faded outgoing tail, nothing else


# ---------------------------------------------------------------------------
# Peak handling during overlap (spec Transitions 4)
# ---------------------------------------------------------------------------

def test_overlap_peak_overshoot_is_detected_and_smoothly_reduced(tmp_path):
    ceiling = -2.0
    xf_n = int(round(CFG.default_crossfade_s * SR))
    a = scale_to_tp(music_like(30, seed=11), ceiling - 0.3)
    b = np.concatenate([a[-xf_n:], scale_to_tp(music_like(24, seed=12), ceiling - 0.3)])   # head == tail of a
    inputs = make_inputs(tmp_path, [a, b])
    r = render_mix(inputs, tmp_path / "mix.wav", CFG, ceiling_dbtp=ceiling)
    tr = r.transitions[0]
    assert tr.crossfade_s * SR == xf_n and tr.curve == "equal_power"
    fo, fi = fade_curves(xf_n, "equal_power")
    naive = a[-xf_n:].astype(np.float64) * fo[:, None] + b[:xf_n].astype(np.float64) * fi[:, None]
    assert tp_dbtp(naive) > ceiling + 1.5                       # the correlated overlap really overshoots
    assert tr.gain_reduction_db > 1.5 and any("overshoot" in f for f in tr.flags)
    assert tr.overlap_true_peak_dbtp <= ceiling + 0.1

    y = read(r.wav_path)
    s = int(round(tr.overlap_start_s * SR))
    assert tp_dbtp(y[s:s + xf_n]) <= ceiling + 0.1             # independent measurement of the file
    fade_in_n = int(CFG.mix_fade_in_s * SR)
    assert np.array_equal(y[fade_in_n:s], a[fade_in_n:s])      # outside the overlap: untouched
    assert np.array_equal(y[s + xf_n:s + xf_n + SR], b[xf_n:xf_n + SR])
    # the applied gain is a smooth dip that returns to 0 dB at both overlap edges
    valid = np.abs(naive[:, 0]) > 0.05
    g_db = 20 * np.log10(np.abs(y[s:s + xf_n, 0][valid]) / np.abs(naive[valid, 0]))
    assert np.abs(np.diff(g_db)).max() < 0.02
    assert abs(g_db[:20]).max() < 0.02 and abs(g_db[-20:]).max() < 0.02
    assert g_db.min() == pytest.approx(-tr.gain_reduction_db, abs=0.05)


def _db(v: float) -> float:
    return 20 * math.log10(v)


def test_true_peak_envelope_block_edges_are_not_read_hot():
    # DSP-4: resample_poly on the isolated block assumed silence outside it, so a block
    # starting on a full-level bass crest read ~1.1 dB hot (Gibbs ringing of the 0 -> x step).
    t = np.arange(4 * SR) / SR
    sig = 10 ** (-2.1 / 20) * np.cos(2 * np.pi * 40 * t)          # crest every 25 ms, true peak -2.10 dBTP
    x = np.stack([sig, sig], 1)
    a, b = SR, 2 * SR                                              # block starts on a crest
    assert _db(true_peak_envelope(x[a:b], 4).max()) == pytest.approx(-2.1, abs=0.02)        # no context
    with_ctx = true_peak_envelope(x[a:b], 4, before=x[:a], after=x[b:])
    np.testing.assert_allclose(with_ctx, true_peak_envelope(x, 4)[a:b], rtol=0, atol=1e-9)   # = in-signal estimate
    # missing context is filled by reflection, which does not inflate noise-like material either
    noise = np.random.default_rng(3).standard_normal((SR, 2)) * 0.1
    blk = noise[1000:9000]
    exact = true_peak_envelope(blk, 4, before=noise[:1000], after=noise[9000:]).max()
    assert true_peak_envelope(blk, 4).max() <= exact * 10 ** (0.05 / 20)


def test_overlap_of_bass_heavy_tracks_is_not_held_at_an_inflated_own_peak(tmp_path):
    # DSP-4: two in-phase 50 Hz tracks (true peak -2.10 dBTP) used to report "track material
    # itself peaks at -1.01 dBTP" and leave the overlap at ~-1.2 dBTP against a -2.0 ceiling.
    ceiling = -2.0
    t = np.arange(25 * SR) / SR
    sig = 10 ** (-2.1 / 20) * np.cos(2 * np.pi * 50 * t)
    track = np.stack([sig, sig], 1).astype(np.float32)
    inputs = make_inputs(tmp_path, [track, track])
    r = render_mix(inputs, tmp_path / "mix.wav", cfg(trim_silence=False, mix_fade_in_s=0, mix_fade_out_s=0),
                   ceiling_dbtp=ceiling)
    tr = r.transitions[0]
    assert not any("track material itself" in f for f in tr.flags), tr.flags
    assert any("overshoot" in f for f in tr.flags) and tr.overlap_true_peak_dbtp <= ceiling
    y = read(r.wav_path).astype(np.float64)
    s, e = int(round(tr.overlap_start_s * SR)), int(round((tr.overlap_start_s + tr.crossfade_s) * SR))
    ctx = 4000                                                     # independent x16 measurement with real context
    env16 = true_peak_envelope(y[s - ctx:e + ctx], 16)[ctx:-ctx]
    assert _db(env16.max()) <= ceiling + 0.02


def test_no_gain_change_when_overlap_stays_below_ceiling(tmp_path):
    inputs = make_inputs(tmp_path, [music_like(25, peak_dbfs=-10, seed=1), music_like(25, peak_dbfs=-10, seed=2)])
    r = render_mix(inputs, tmp_path / "mix.wav", CFG, ceiling_dbtp=-1.0)
    tr = r.transitions[0]
    assert tr.gain_reduction_db == 0.0 and not any("overshoot" in f for f in tr.flags)
    assert tr.overlap_true_peak_dbtp < -1.0 and tr.overlap_peak_dbfs <= tr.overlap_true_peak_dbtp + 1e-6
    assert r.peak_dbfs < -1.0 and r.clipped_samples == 0


# ---------------------------------------------------------------------------
# Short tracks / disabled crossfades (spec Transitions 5)
# ---------------------------------------------------------------------------

def test_short_track_gets_butt_splices_and_renders(tmp_path):
    sigs = [music_like(25, peak_dbfs=-4, seed=1), music_like(0.5, peak_dbfs=-4, seed=2),
            music_like(25, peak_dbfs=-4, seed=3)]
    inputs = make_inputs(tmp_path, sigs)
    r = render_mix(inputs, tmp_path / "mix.wav", cfg(fx_mode="echo_tail"), ceiling_dbtp=-1.0)
    assert [t.crossfade_s for t in r.transitions] == [0.0, 0.0]
    assert all(t.fx == "clean" and t.overlap_true_peak_dbtp is None for t in r.transitions)
    assert any("butt splice" in f for f in r.transitions[0].flags)
    assert r.tracks[1].duration_s == pytest.approx(0.5) and r.tracks[1].start_s == pytest.approx(25.0)
    y = read(r.wav_path)
    assert len(y) == sum(len(s) for s in sigs) == round(r.expected_duration_s * SR)
    for boundary in (25 * SR, int(25.5 * SR)):                  # de-click fades on both sides of each cut
        assert np.abs(y[boundary - 1]).max() < 1e-3 and np.abs(y[boundary]).max() < 1e-3
    d = int(CFG.declick_ms / 1000 * SR)
    assert np.array_equal(y[25 * SR + d:int(25.5 * SR) - d], sigs[1][d:-d])   # short track itself intact


def test_crossfade_disabled(tmp_path):
    sigs = [music_like(s, peak_dbfs=-4, seed=k) for k, s in enumerate((21, 23, 22), start=1)]
    inputs = make_inputs(tmp_path, sigs)
    r = render_mix(inputs, tmp_path / "mix.wav", cfg(enabled=False), ceiling_dbtp=-1.0)
    assert all(t.crossfade_s == 0.0 and t.gain_reduction_db == 0.0 for t in r.transitions)
    assert sf.info(str(r.wav_path)).frames == sum(len(s) for s in sigs)
    assert r.expected_duration_s == pytest.approx(sum(len(s) for s in sigs) / SR)
    assert [t.start_s for t in r.tracks] == pytest.approx([0.0, 21.0, 44.0])


# ---------------------------------------------------------------------------
# Determinism, validation, formats
# ---------------------------------------------------------------------------

def test_render_random_fx_is_deterministic_per_seed(tmp_path):
    sigs = [music_like(21, peak_dbfs=-6, seed=k) for k in range(1, 6)]
    inputs = make_inputs(tmp_path, sigs)
    c = cfg(fx_mode="random")
    r1 = render_mix(inputs, tmp_path / "a.wav", c, ceiling_dbtp=-1.0, seed=42)
    r2 = render_mix(inputs, tmp_path / "b.wav", c, ceiling_dbtp=-1.0, seed=42)
    assert [t.fx for t in r1.transitions] == [t.fx for t in r2.transitions]
    assert np.array_equal(read(r1.wav_path), read(r2.wav_path)) and r1.seed == 42
    seqs = {tuple(t.fx for t in render_mix(inputs, tmp_path / f"s{s}.wav", c, ceiling_dbtp=-1.0, seed=s).transitions)
            for s in range(6)}
    assert len(seqs) > 1                                           # the seed actually drives the choice
    default = render_mix(inputs, tmp_path / "d.wav", CFG, ceiling_dbtp=-1.0, seed=42)
    assert {t.fx for t in default.transitions} == {"clean"}      # commercial default: no FX


def test_render_rejects_broken_inputs(tmp_path):
    good = write_wav(tmp_path / "in" / "good.wav", music_like(21, peak_dbfs=-6, seed=1))
    with pytest.raises(ValueError, match="no input tracks"):
        render_mix([], tmp_path / "mix.wav", CFG, ceiling_dbtp=-1.0)
    other_sr = write_wav(tmp_path / "in" / "48k.wav", music_like(21, peak_dbfs=-6, seed=2), sr=48000)
    with pytest.raises(ValueError, match="sample rate mismatch"):
        render_mix([mi(1, path=good), mi(2, path=other_sr)], tmp_path / "mix.wav", CFG, ceiling_dbtp=-1.0)
    silent = write_wav(tmp_path / "in" / "silent.wav", np.zeros((21 * SR, 2), np.float32))
    with pytest.raises(ValueError, match="silent"):
        render_mix([mi(1, path=good), mi(2, path=silent)], tmp_path / "mix.wav", CFG, ceiling_dbtp=-1.0)
    with pytest.raises(FileNotFoundError):
        render_mix([mi(1, path=good), mi(2, path=tmp_path / "missing.wav")], tmp_path / "mix.wav", CFG,
                   ceiling_dbtp=-1.0)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        render_mix([mi(1, path=good)], good, CFG, ceiling_dbtp=-1.0)
    with pytest.raises(ValueError, match="fx_mode"):
        render_mix([mi(1, path=good)], tmp_path / "mix.wav", cfg(fx_mode="dj"), ceiling_dbtp=-1.0)
    assert not (tmp_path / "mix.wav").exists() and not list(tmp_path.glob(".*veg-*"))


def test_mono_input_is_duplicated_and_sources_untouched(tmp_path):
    mono = write_wav(tmp_path / "in" / "mono.wav", music_like(21, peak_dbfs=-6, seed=1, channels=1))
    stereo = write_wav(tmp_path / "in" / "stereo.wav", music_like(21, peak_dbfs=-6, seed=2))
    hashes = {p: sha256(p) for p in (mono, stereo)}
    r = render_mix([mi(1, path=mono), mi(2, path=stereo)], tmp_path / "mix.wav", CFG, ceiling_dbtp=-1.0)
    y = read(r.wav_path)
    assert r.channels == 2 and y.shape[1] == 2
    s = int(round(r.transitions[0].overlap_start_s * SR))
    assert np.array_equal(y[:s, 0], y[:s, 1])
    assert {p: sha256(p) for p in (mono, stereo)} == hashes


def test_rf64_float_output_with_unicode_paths(tmp_path):
    d = tmp_path / "микс é"
    sigs = [music_like(21, peak_dbfs=-6, seed=1), music_like(22, peak_dbfs=-6, seed=2)]
    inputs = [mi(i, path=write_wav(d / f"трек {i} – café.wav", s), name=f"Артист {i} – Café")
              for i, s in enumerate(sigs, start=1)]
    r = render_mix(inputs, d / "микс.wav", CFG, ceiling_dbtp=-1.0)
    info = sf.info(str(r.wav_path))
    assert info.format == "RF64" and info.subtype == "FLOAT" and info.channels == 2 and info.samplerate == SR
    assert r.wav_path.name == "микс.wav" and len(read(r.wav_path)) == info.frames


# ---------------------------------------------------------------------------
# Tracklist
# ---------------------------------------------------------------------------

def test_tracklist_rows_match_legacy_columns_and_qc_regex(tmp_path):
    sigs = [music_like(s, peak_dbfs=-6, seed=k) for k, s in enumerate((25.37, 23.81, 26.02), start=1)]
    inputs = make_inputs(tmp_path, sigs)
    inputs[1] = dataclasses.replace(inputs[1], bpm=92.0, camelot=None, name="Артист – Песня")
    inputs[2] = dataclasses.replace(inputs[2], bpm=128.0)
    r = render_mix(inputs, tmp_path / "work" / "mix.wav", CFG, ceiling_dbtp=-1.0, seed=17)
    path = tmp_path / "out" / "mix_tracklist.txt"
    write_mix_tracklist(r, inputs, path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "TRACKLIST: mix.mp3" and lines[2] == "Random seed: 17"
    assert lines[1].startswith("Generated: ") and lines[4] == "Total tracks: 3"
    assert re.fullmatch(r"Total duration: 0h 01m \d{2}s", lines[3])
    rows = [ln for ln in lines if QC_ROW_RE.match(ln)]
    assert len(rows) == 3
    for num, (row, entry) in enumerate(zip(rows, r.tracks), start=1):
        s = math.floor(entry.start_s + 0.5 / SR)
        hms = f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
        assert row.startswith(f"{num:<4} | {hms:<10} | ")
        m = QC_ROW_RE.match(row)
        assert int(m.group(1)) == num and m.group(2) == hms
    assert "Артист – Песня" in rows[1] and "| ?     |" in rows[1]        # unknown key stays unknown
    assert "listen: uncertain transition" in rows[1] and "listen" in rows[2]
    assert "=== CAMELOT STATS ===" in text and "Unknown key: 2/2" in text
    assert "=== FX SUMMARY ===" in text and "clean: 2 (100%)" in text
    assert "92 vs 128 BPM (28% apart)" in text                          # uncertain reasons listed
    assert "=== PEAK PROTECTION ===" in text
    assert not list(path.parent.glob(".*.tmp"))


def test_tracklist_timestamps_past_one_hour(tmp_path):
    assert ms_to_hms(3_725_999) == "1:02:05" and ms_to_hms(59_999) == "0:00:59"
    entries = [MixTrackEntry(position=1, name="A", source_rel="a", start_s=0.0, end_s=3726.0, duration_s=3726.0),
               MixTrackEntry(position=2, name="B", source_rel="b", start_s=3719.9999999, end_s=3900.0,
                             duration_s=180.1, crossfade_in_s=6.0, fx_in="clean")]
    trs = [TransitionReport(position=2, crossfade_s=6.0, fx="clean", curve="equal_power", overlap_start_s=3720.0,
                            overlap_peak_dbfs=-3.0, overlap_true_peak_dbtp=-2.9)]
    res = MixRenderResult(wav_path=Path("x.wav"), sample_rate=SR, channels=2, duration_s=3900.0,
                          expected_duration_s=3900.0, tracks=entries, transitions=trs, seed=0)
    out = tmp_path / "set_tracklist.txt"
    write_mix_tracklist(res, [mi(1), mi(2, bpm=None)], out)
    text = out.read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if QC_ROW_RE.match(ln)]
    assert QC_ROW_RE.match(rows[1]).group(2) == "1:02:00"     # floating error does not floor 3720 to 3719
    assert "| ?      |" in rows[1]                              # unknown BPM printed as '?', not 0
    assert "Total duration: 1h 05m 00s" in text and text.startswith("TRACKLIST: set.mp3")


# ---------------------------------------------------------------------------
# Real FFmpeg integration: normalized intermediates -> render -> FFmpeg reads the RF64 mix
# ---------------------------------------------------------------------------

@requires_ffmpeg
@pytest.mark.ffmpeg
def test_normalized_tracks_render_to_a_mix_ffmpeg_can_verify(ffmpeg_tools, tmp_path):
    from veg_mixer.ffmpeg_tools import decode_check, probe_audio
    from veg_mixer.loudness import LoudnessNormalizer
    from veg_mixer.models import NormalizationStatus

    lc = LoudnessConfig()
    norm = LoudnessNormalizer(ffmpeg_tools, lc)
    srcs = [write_wav(tmp_path / "src" / f"src{k}.wav", music_like(24, peak_dbfs=p, seed=k))
            for k, p in ((1, -18.0), (2, -2.0), (3, -9.0))]
    hashes = {p: sha256(p) for p in srcs}
    inputs = []
    for k, src in enumerate(srcs, start=1):
        res = norm.normalize_to_wav(src, tmp_path / "norm" / f"{k:04d}.wav")
        assert res.status is NormalizationStatus.PASSED, res.errors
        inputs.append(mi(k, path=res.output))
    r = render_mix(inputs, tmp_path / "out" / "mix.wav", CFG, ceiling_dbtp=lc.pcm_true_peak_ceiling)

    probe = probe_audio(ffmpeg_tools, r.wav_path)
    assert probe.channels == 2 and probe.sample_rate == SR and probe.codec == "pcm_f32le"
    dec = decode_check(ffmpeg_tools, r.wav_path)
    assert dec.ok, dec.errors
    assert abs(dec.decoded_duration_s - r.expected_duration_s) < 0.01
    m = norm.measure(r.wav_path)
    assert m.true_peak_dbtp <= lc.pcm_true_peak_ceiling + lc.true_peak_tolerance_db
    assert abs(m.integrated_lufs - lc.target_lufs) <= lc.lufs_tolerance
    for tr in r.transitions:
        assert tr.overlap_true_peak_dbtp <= lc.pcm_true_peak_ceiling + lc.true_peak_tolerance_db
    assert {p: sha256(p) for p in srcs} == hashes
