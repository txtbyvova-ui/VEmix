"""Segment-level audio intelligence (VEmix 3.0, spec §2): trajectory, structure,
density, tempo summary, the vocal proxy, JSON-nativeness, robustness to broken input
and the ``segments`` cache section. Synthetic signals only (no copyrighted music);
FFmpeg-dependent decoding is marked ``ffmpeg``."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf

from conftest import SR, encode_mp3, music_like, requires_ffmpeg, sha256, write_wav
from veg_mixer import audio_intel
from veg_mixer.analysis import (
    BPM_ALGO_VERSION,
    KEY_ALGO_VERSION,
    bpm_params,
    compute_energy,
    key_params,
)
from veg_mixer.audio_intel import (
    DENSITY_LEVELS,
    SECTION_LABELS,
    SECTION_MAX_CONFIDENCE,
    SEGMENTS_SECTION,
    SEGMENTS_VERSION,
    TIMBRE_DIMENSIONS,
    TRAJECTORIES,
    VOCAL_PROXY_MAX_CONFIDENCE,
    SegmentConfig,
    analyze_segments,
    compute_segments,
    is_valid_segments,
    segments_params,
    timbre_similarity,
)
from veg_mixer.cache import AnalysisCache, params_hash
from veg_mixer.config import AnalysisConfig
from veg_mixer.ffmpeg_tools import FFmpegError, FFmpegNotFoundError, FFmpegTools

SEG_SR = 22050          # synthetic material at the analysis rate: no resampling cost in the pure tests
CFG = SegmentConfig()
ARRAY_KEYS = ("times", "intensity", "rms_db", "onset_rate", "percussive_ratio", "bass_ratio",
              "centroid_hz", "flatness")
FAKE_TOOLS = FFmpegTools(ffmpeg="ffmpeg-not-used", ffprobe=None, version="test", has_soxr=False,
                         has_libmp3lame=True)


# ---------------------------------------------------------------------------
# synthetic material (22.05 kHz mono unless stated otherwise)
# ---------------------------------------------------------------------------

def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SEG_SR)) / SEG_SR


def _norm(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    return (peak * y / np.max(np.abs(y))).astype(np.float32)


def pad(t: np.ndarray, amp: float = 0.25) -> np.ndarray:
    """Sustained triad (A, C#, E) with an octave partial: no onsets, no percussion."""
    return amp * sum(np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
                     for f in (220.0, 277.18, 329.63)) / 3


def groove(t: np.ndarray, bpm: float = 124.0, seed: int = 0) -> np.ndarray:
    """Kick on every beat, off-beat hats, gated bass and a bright gated saw."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    kick = np.sin(2 * np.pi * 55 * t) * np.exp(-(t % beat) / 0.06)
    hat = np.diff(rng.standard_normal(t.size), prepend=0.0) * np.exp(-(t % (beat / 2)) / 0.02)
    bass = np.sin(2 * np.pi * 55 * t) * (((t + beat / 2) % beat) < beat / 2)
    saw = sum(np.sin(2 * np.pi * k * 440 * t) / k for k in range(1, 12)) * (np.sin(np.pi * t / beat) > 0)
    return 0.9 * kick + 0.25 * hat + 0.35 * bass + 0.08 * saw


def structured() -> np.ndarray:
    """quiet pad intro 0-15 s | groove 15-45 | quiet pad break 45-60 | groove 60-85 |
    groove fading out by 34 dB over 85-100 s."""
    t = _t(100.0)
    body = groove(t) + pad(t)
    y = np.where((t < 15) | ((t >= 45) & (t < 60)), 0.25 * pad(t, amp=0.35), body)
    fade = t >= 85
    y[fade] *= 10 ** (-34.0 * (t[fade] - 85) / 15 / 20)
    return _norm(y)


def ramp(seconds: float = 60.0, *, rising: bool = True, seed: int = 0) -> np.ndarray:
    """Crescendo (or decrescendo) in arrangement AND level: kick+noise events whose rate
    goes 0.5 -> 7 per second, brighter bursts and +16 dB of gain (reversed when falling)."""
    rng = np.random.default_rng(seed)
    t = _t(seconds)

    def frac(x: float) -> float:
        return x / seconds if rising else 1.0 - x / seconds

    ev = np.zeros_like(t)
    x = 0.0
    while x < seconds:
        i = int(x * SEG_SR)
        n = min(int(0.12 * SEG_SR), t.size - i)
        if n > 0:
            tau = np.arange(n) / SEG_SR
            burst = np.diff(rng.standard_normal(n), prepend=0.0) * np.exp(-tau / 0.02) * (0.2 + 0.8 * frac(x))
            ev[i:i + n] += 0.6 * np.sin(2 * np.pi * 60 * tau) * np.exp(-tau / 0.05) + 0.5 * burst
        x += 1.0 / (0.5 + 6.5 * frac(x))
    u = t / seconds if rising else 1.0 - t / seconds
    gain = 10 ** ((-16.0 + 16.0 * u) / 20)
    return _norm((pad(t, amp=0.2) + ev) * gain)


def sparse(seconds: float = 30.0) -> np.ndarray:
    """Slow-attack low sine notes, one every 3 s."""
    t = _t(seconds)
    ph = t % 3.0
    env = (1 - np.exp(-ph / 0.8)) * np.exp(-ph / 2.5)
    f = np.array([196.0, 174.61, 220.0, 164.81])[(t // 3.0).astype(int) % 4]
    return _norm((np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * env, 0.5)


def dense(seconds: float = 30.0, *, bpm: float = 128.0, seed: int = 0) -> np.ndarray:
    """Four-on-the-floor kick, 16th hats, bright gated saw chord and a noise floor."""
    rng = np.random.default_rng(seed)
    t = _t(seconds)
    beat = 60.0 / bpm
    kick = np.sin(2 * np.pi * 55 * t) * np.exp(-(t % beat) / 0.06)
    hat = np.diff(rng.standard_normal(t.size) * np.exp(-(t % (beat / 4)) / 0.015), prepend=0.0)
    saw = sum(np.sin(2 * np.pi * k * f * t) / k for f in (220.0, 277.18, 329.63) for k in range(1, 30))
    gate = np.exp(-((t + beat / 2) % (beat / 2)) / 0.12)
    return _norm(0.9 * kick + 0.35 * hat + 0.08 * saw * gate + 0.05 * rng.standard_normal(t.size))


def voice_like(seconds: float = 30.0, seed: int = 0) -> np.ndarray:
    """A harmonic tone with vibrato, glides between notes and a moving formant — the
    spectral behaviour the vocal proxy looks for (it is not a voice, which is the point)."""
    rng = np.random.default_rng(seed)
    t = _t(seconds)
    notes = 220 * 2 ** (rng.integers(0, 8, int(seconds / 0.6) + 2) / 12)
    f0 = np.convolve(notes[(t // 0.6).astype(int)], np.ones(800) / 800, mode="same")
    f0 = f0 * 2 ** (0.4 / 12 * np.sin(2 * np.pi * 5.5 * t))
    ph = 2 * np.pi * np.cumsum(f0) / SEG_SR
    formant = 700 + 400 * np.sin(2 * np.pi * 0.7 * t)
    return _norm(sum(np.sin(k * ph) * np.exp(-((k * f0 - formant) / 500) ** 2) for k in range(1, 20)), 0.6)


def steady(seconds: float = 30.0) -> np.ndarray:
    """A held harmonic tone with a static formant: no onsets, no movement."""
    t = _t(seconds)
    return _norm(sum(np.sin(2 * np.pi * k * 220 * t) * np.exp(-((k * 220 - 900) / 500) ** 2)
                     for k in range(1, 20)), 0.6)


def wav(path: Path, y: np.ndarray, sr: int = SEG_SR) -> Path:
    return write_wav(path, y, sr=sr)


# ---------------------------------------------------------------------------
# fixtures: every synthetic track is analysed once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lib(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("audio_intel")


@pytest.fixture(scope="module")
def analysed(lib):
    """name -> (path, value) for the shared synthetic tracks."""
    tracks = {"structured": structured(), "rise": ramp(rising=True), "fall": ramp(rising=False),
              "sparse": sparse(), "dense": dense(), "dense2": dense(seed=5, bpm=126.0),
              "voice": voice_like(), "steady": steady()}
    out = {}
    for name, y in tracks.items():
        p = wav(lib / f"{name}.wav", y)
        out[name] = (p, compute_segments(p))
    return out


def walk_json_native(obj, where="value"):
    """Only plain JSON types; floats finite."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), where
            walk_json_native(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk_json_native(v, f"{where}[{i}]")
    elif isinstance(obj, float):
        assert math.isfinite(obj), where
    else:
        assert obj is None or type(obj) in (str, int, bool), f"{where}: {type(obj).__name__}"


# ---------------------------------------------------------------------------
# trajectory / energy edges
# ---------------------------------------------------------------------------

def test_crescendo_rises_and_decrescendo_falls(analysed):
    _, rise = analysed["rise"]
    _, fall = analysed["fall"]
    sr_, sf_ = rise["summary"], fall["summary"]
    assert sr_["trajectory"]["value"] == "rising" and sr_["trajectory"]["slope_per_min"] > 0
    assert sf_["trajectory"]["value"] == "falling" and sf_["trajectory"]["slope_per_min"] < 0
    assert sr_["energy_out"] > sr_["energy_in"] + 0.15
    assert sf_["energy_in"] > sf_["energy_out"] + 0.15
    for s in (sr_, sf_):
        assert 0.3 <= s["trajectory"]["confidence"] <= 0.8
        assert s["energy_peak"] >= s["energy_mean"]
    # the per-segment curve itself goes the right way
    assert np.mean(rise["intensity"][-3:]) > np.mean(rise["intensity"][:3]) + 0.2
    assert np.mean(fall["intensity"][:3]) > np.mean(fall["intensity"][-3:]) + 0.2
    # rms_db is level-dependent inside the track: the crescendo gains ~16 dB
    assert rise["rms_db"][-1] - rise["rms_db"][0] > 12


def test_stable_material_is_stable_and_structured_material_is_a_wave(analysed):
    for name in ("dense", "sparse", "steady"):
        assert analysed[name][1]["summary"]["trajectory"]["value"] == "stable", name
    # intro low -> body -> break -> body (+ a fade whose material stays dense): up-down-up
    traj = analysed["structured"][1]["summary"]["trajectory"]
    assert traj["value"] == "wave" and traj["oscillation"] >= CFG.wave_min


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def test_sections_find_intro_break_outro_and_repeat_candidates(analysed):
    _, v = analysed["structured"]
    secs = v["sections"]
    labels = [s["label"] for s in secs]
    assert set(labels) <= set(SECTION_LABELS)
    # contiguous cover of the analysed audio
    assert secs[0]["start_s"] == 0.0 and secs[-1]["end_s"] == pytest.approx(v["analysed_s"], abs=0.01)
    assert all(a["end_s"] == b["start_s"] for a, b in zip(secs, secs[1:]))

    def one(label):
        found = [s for s in secs if s["label"] == label]
        assert len(found) == 1, (label, labels)
        return found[0]

    intro, brk, outro = one("intro"), one("break"), one("outro")
    assert intro["start_s"] == 0.0 and 12.0 <= intro["end_s"] <= 18.0
    assert 42.0 <= brk["start_s"] <= 48.0 and 57.0 <= brk["end_s"] <= 63.0
    assert 79.0 <= outro["start_s"] <= 91.0 and outro["end_s"] == secs[-1]["end_s"]
    assert intro["intensity"] < 0.2 and brk["intensity"] < 0.2
    # the two grooves are the same material: one repeat group (a chorus CANDIDATE)
    loud = [s for s in secs if s["label"] in ("body", "peak") and s["start_s"] < 85.0]
    assert len(loud) == 2 and loud[0]["repeat_group"] is not None
    assert loud[0]["repeat_group"] == loud[1]["repeat_group"]
    assert all(s["repeat_group"] is None for s in (intro, brk, outro))
    assert all(0.0 < s["confidence"] <= SECTION_MAX_CONFIDENCE for s in secs)
    # level-dependent rms_db shows the quiet intro / break and the fade; intensity does not see the fade
    times = np.array(v["times"])
    rms = np.array(v["rms_db"])
    assert rms[times < 15].max() < -10 and rms[(times >= 45) & (times < 60)].max() < -10
    assert rms[-1] < -20
    assert v["summary"]["dynamic_range_db"] > 10


def test_homogeneous_material_is_one_section(analysed):
    for name in ("dense", "sparse", "steady", "voice"):
        secs = analysed[name][1]["sections"]
        assert len(secs) == 1 and secs[0]["label"] == "body" and secs[0]["repeat_group"] is None, name


# ---------------------------------------------------------------------------
# density / texture / tempo / vocal proxy
# ---------------------------------------------------------------------------

def test_density_orders_sparse_below_dense(analysed):
    s_sparse = analysed["sparse"][1]["summary"]
    s_dense = analysed["dense"][1]["summary"]
    assert s_sparse["density"]["value"] == "sparse" and s_dense["density"]["value"] == "dense"
    assert s_sparse["density"]["score"] + 0.4 < s_dense["density"]["score"]
    assert s_sparse["percussive_density"] < s_dense["percussive_density"]
    assert s_sparse["energy_mean"] < s_dense["energy_mean"]
    for name in analysed:
        d = analysed[name][1]["summary"]["density"]
        assert d["value"] in DENSITY_LEVELS and 0.0 <= d["score"] <= 1.0 and 0.0 < d["confidence"] <= 0.6


def test_tempo_summary_finds_the_pulse_and_admits_none(analysed):
    t = analysed["dense"][1]["summary"]["tempo"]
    assert t["bpm"] is not None and abs(t["bpm"] - 128.0) / 128.0 < 0.03
    assert 0.3 <= t["confidence"] <= 0.9
    assert t["candidates"] and t["candidates"][0]["bpm"] == t["bpm"]
    assert all(set(c) == {"bpm", "strength"} for c in t["candidates"])
    assert isinstance(t["half_double_ambiguous"], bool)
    t2 = analysed["structured"][1]["summary"]["tempo"]
    assert t2["bpm"] is not None and abs(t2["bpm"] - 124.0) / 124.0 < 0.03
    # a held tone has no pulse: unknown, not the estimator's 120 BPM prior
    for name in ("steady", "sparse"):
        t3 = analysed[name][1]["summary"]["tempo"]
        assert t3 == {"bpm": None, "confidence": 0.0, "candidates": [], "half_double_ambiguous": False}, name
        assert any("tempo" in n for n in analysed[name][1]["notes"])


@pytest.mark.parametrize("bpm", [140.0, 170.0])
def test_tempo_on_a_16th_grid_is_never_confidently_wrong(lib, bpm):
    """Review regression: four-on-the-floor with 16th hats makes every 16th multiple a
    periodicity; the prior then picked 112 BPM for 140 (and 113 for 170) with confidence
    0.8-0.9. The pick may still land on another metrical level (librosa.feature.tempo does
    too), but then the confidence must be low and the true tempo among the candidates."""
    t = compute_segments(wav(lib / f"grid{bpm:.0f}.wav", dense(30.0, bpm=bpm, seed=9)))["summary"]["tempo"]
    q = t["bpm"] / bpm
    if not (abs(q - 1) < 0.03 or abs(q - 2) < 0.06 or abs(q - 0.5) < 0.02):
        assert t["confidence"] <= 0.4, t
        assert any(abs(c["bpm"] / bpm - 1) < 0.03 for c in t["candidates"]), t


def test_vocal_proxy_is_a_low_confidence_proxy(analysed):
    voice = analysed["voice"][1]["summary"]["vocal_presence_proxy"]
    held = analysed["steady"][1]["summary"]["vocal_presence_proxy"]
    assert voice["value"] > held["value"] + 0.3
    for name in analysed:
        vp = analysed[name][1]["summary"]["vocal_presence_proxy"]
        assert 0.0 <= vp["confidence"] <= VOCAL_PROXY_MAX_CONFIDENCE
        assert "not a vocal detector" in vp["method"]
        assert vp["value"] is None or 0.0 <= vp["value"] <= 1.0


def test_timbre_vector_and_similarity(analysed):
    for name in analysed:
        tv = analysed[name][1]["summary"]["timbre"]
        assert len(tv) == len(TIMBRE_DIMENSIONS) and all(0.0 <= x <= 1.0 for x in tv), name
    d1, d2, sp = (analysed[n][1] for n in ("dense", "dense2", "sparse"))
    assert timbre_similarity(d1, d1) == 1.0
    assert timbre_similarity(d1, d2) > timbre_similarity(d1, sp) + 0.2
    assert timbre_similarity(d1["summary"]["timbre"], d2["summary"]) == timbre_similarity(d1, d2)
    assert timbre_similarity(d1, None) is None
    assert timbre_similarity(d1, {"version": SEGMENTS_VERSION, "error": "x"}) is None


def test_intensity_is_on_the_2_0_energy_scale(lib):
    """Per-segment intensity is the 2.0 energy heuristic per segment: over the 2.0 window
    (30-90 s) its mean matches ``analysis.compute_energy`` closely (44.1 kHz stereo input,
    so the streaming decode + resampling path is exercised as well)."""
    src = write_wav(lib / "fidelity.wav", music_like(92.0, seed=7, bpm=120.0), sr=SR)
    energy = compute_energy(src, AnalysisConfig())["energy"]
    v = compute_segments(src)
    times = np.array(v["times"])
    window = (times >= 30.0) & (times < 90.0)
    assert abs(float(np.mean(np.array(v["intensity"])[window])) - energy) < 0.06, energy


def test_intensity_is_loudness_independent(lib, analysed):
    src, ref = analysed["dense"]
    y, _ = sf.read(str(src), dtype="float32")
    quiet = compute_segments(wav(lib / "dense_quiet.wav", y * 10 ** (-20 / 20)))
    assert np.allclose(quiet["intensity"], ref["intensity"], atol=0.01)
    assert np.allclose(quiet["rms_db"], ref["rms_db"], atol=0.11)
    assert quiet["summary"]["energy_in"] == pytest.approx(ref["summary"]["energy_in"], abs=0.01)


# ---------------------------------------------------------------------------
# value shape: JSON-native, compact, deterministic
# ---------------------------------------------------------------------------

def test_value_is_json_native_and_compact(analysed):
    for name, (_, v) in analysed.items():
        assert "error" not in v, (name, v.get("error"))
        assert v["version"] == SEGMENTS_VERSION and is_valid_segments(v)
        walk_json_native(v, name)
        text = json.dumps(v, allow_nan=False)
        assert json.loads(text) == v
        n = len(v["times"])
        assert all(len(v[k]) == n for k in ARRAY_KEYS), name
        assert all(0.0 <= x <= 1.0 for x in v["intensity"]), name
        assert all(-60.0 <= x <= 20.0 for x in v["rms_db"]), name
        assert v["times"][0] == 0.0 and v["segment_s"] == CFG.segment_s
        assert v["duration_s"] == pytest.approx(v["analysed_s"], abs=0.01)
        s = v["summary"]
        assert s["trajectory"]["value"] in TRAJECTORIES
        assert all(isinstance(note, str) for note in v["notes"])
    # 100 s of material: a few KB per track in the cache
    assert len(json.dumps(analysed["structured"][1], separators=(",", ":"))) < 3000


def test_deterministic(analysed):
    src, v = analysed["dense"]
    assert compute_segments(src) == v


def test_long_audio_gets_longer_segments_and_truncation_is_noted(lib):
    src = wav(lib / "long.wav", dense(40.0, seed=2))
    v = compute_segments(src, dataclasses.replace(CFG, max_segments=4))
    assert v["segment_s"] == 10.0 and len(v["times"]) == 4
    assert any("segment length raised" in n for n in v["notes"])
    v2 = compute_segments(src, dataclasses.replace(CFG, max_analysis_s=15.0))
    assert v2["analysed_s"] == pytest.approx(15.0, abs=0.01) and v2["duration_s"] == pytest.approx(40.0, abs=0.01)
    assert v2["times"][-1] < 15.0 and v2["sections"][-1]["end_s"] == pytest.approx(15.0, abs=0.01)
    assert any("max_analysis_s" in n for n in v2["notes"])


def test_silent_gap_inside_a_track(lib):
    t = dense(10.0)
    y = np.concatenate([t, np.zeros(10 * SEG_SR, np.float32), t])
    v = compute_segments(wav(lib / "gap.wav", y))
    assert "error" not in v
    times = np.array(v["times"])
    gap = (times >= 10.0) & (times < 20.0)
    assert np.all(np.array(v["intensity"])[gap] == 0.0)
    assert np.all(np.array(v["rms_db"])[gap] == -60.0)
    assert any("silent segment" in n for n in v["notes"])
    walk_json_native(v)


# ---------------------------------------------------------------------------
# broken / degenerate input never raises
# ---------------------------------------------------------------------------

def test_short_silent_and_empty_audio_return_error_values(lib):
    short = compute_segments(wav(lib / "short.wav", dense(2.0)))
    silent = compute_segments(wav(lib / "silent.wav", np.zeros(12 * SEG_SR, np.float32)))
    empty = compute_segments(wav(lib / "empty.wav", np.zeros(0, np.float32)))
    assert short["error_kind"] == "too_short" and short["analysed_s"] == pytest.approx(2.0, abs=0.01)
    assert silent["error_kind"] == "silent" and silent["duration_s"] == pytest.approx(12.0, abs=0.01)
    assert empty["error_kind"] == "too_short"
    for v in (short, silent, empty):
        assert v["version"] == SEGMENTS_VERSION and isinstance(v["error"], str) and is_valid_segments(v)
        walk_json_native(v)


def test_very_short_audio_gives_a_value_without_trajectory(lib):
    v = compute_segments(wav(lib / "six.wav", dense(6.0)))
    assert "error" not in v and len(v["times"]) == 1 and len(v["sections"]) == 1
    traj = v["summary"]["trajectory"]
    assert traj == {"value": None, "confidence": 0.0, "slope_per_min": None, "oscillation": None}
    assert any("short audio" in n for n in v["notes"])
    walk_json_native(v)


def test_unexpected_analysis_failure_is_reported_not_raised_nor_cached(tmp_path, small, monkeypatch):
    def broken(*a, **kw):
        raise ValueError("synthetic numeric failure")

    monkeypatch.setattr(audio_intel, "_analyse", broken)
    cache = AnalysisCache(tmp_path / "c.json")
    v, hit = analyze_segments(small, cache)
    assert v["error_kind"] == "analysis" and "synthetic numeric failure" in v["error"] and not hit
    assert analyze_segments(small, cache, compute=False) == (None, False)


def test_non_finite_samples_are_replaced(lib):
    y = dense(12.0)
    y[SEG_SR * 3:SEG_SR * 3 + 100] = np.nan
    v = compute_segments(wav(lib / "nan.wav", y))
    assert "error" not in v and any("non-finite" in n for n in v["notes"])
    walk_json_native(v)
    json.dumps(v, allow_nan=False)


def test_corrupt_and_missing_files_return_decode_errors(lib, monkeypatch):
    corrupt = lib / "corrupt.wav"
    corrupt.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk" + bytes(range(256)) * 8)
    calls = []

    def ffmpeg_fails(tools, args, **kw):
        calls.append(args)
        raise FFmpegError("ffmpeg failed with exit code 1", stderr="Invalid data found when processing input")

    monkeypatch.setattr(audio_intel, "run_ffmpeg", ffmpeg_fails)
    before = sha256(corrupt)
    v = compute_segments(corrupt, tools=FAKE_TOOLS)
    assert v["error_kind"] == "decode" and "FFmpeg could not decode" in v["error"]
    assert len(calls) == 1 and "-t" in calls[0]          # FFmpeg fallback, capped at max_analysis_s
    assert sha256(corrupt) == before                      # the source is never written
    missing = compute_segments(lib / "does-not-exist.wav", tools=FAKE_TOOLS)
    assert missing["error_kind"] == "decode" and len(calls) == 1


def test_ffmpeg_not_found_propagates(lib, monkeypatch):
    corrupt = lib / "needs_ffmpeg.m4a"
    corrupt.write_bytes(b"\x00\x00\x00\x18ftypM4A " + bytes(64))

    def gone(*a, **kw):
        raise FFmpegNotFoundError("ffmpeg was not found")

    monkeypatch.setattr(audio_intel, "find_ffmpeg", gone)
    with pytest.raises(FFmpegNotFoundError):
        compute_segments(corrupt)
    monkeypatch.setattr(audio_intel, "run_ffmpeg", gone)
    with pytest.raises(FFmpegNotFoundError):
        compute_segments(corrupt, tools=FAKE_TOOLS)


def test_config_validation():
    with pytest.raises(ValueError):
        SegmentConfig(segment_s=0)
    with pytest.raises(ValueError):
        SegmentConfig(density_bounds=(0.6, 0.3))
    with pytest.raises(ValueError):
        SegmentConfig(hop_length=-512)
    # review regressions: bad types fail at construction, not deep inside the analysis
    for bad in ({"hop_length": 512.0}, {"n_fft": 1024, "hop_length": 2048}, {"density_bounds": (0.3,)},
                {"density_bounds": ("a", "b")}, {"vocal_band_hz": None}, {"silence_dbfs": float("nan")}):
        with pytest.raises(ValueError):
            SegmentConfig(**bad)
    from_yaml = SegmentConfig(density_bounds=[0.3, 0.55], vocal_band_hz=[300.0, 3400.0])   # lists from YAML
    assert from_yaml == SegmentConfig() and hash(from_yaml) == hash(SegmentConfig())
    assert segments_params(from_yaml) == segments_params(SegmentConfig())


def test_access_denied_is_a_decode_error_never_a_bare_oserror(tmp_path, small, monkeypatch):
    """Review regression: ``Path.is_file`` re-raises PermissionError (only "not found"-type
    errors are swallowed), which escaped compute_segments and would abort a library run."""
    real_is_file = Path.is_file

    def denied(self):
        if self.name == small.name:
            raise PermissionError(13, "Access is denied", str(self))
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", denied)
    v = compute_segments(small, tools=FAKE_TOOLS)
    assert v["error_kind"] == "decode" and "Access is denied" in v["error"] and str(small) not in v["error"]
    cache = AnalysisCache(tmp_path / "c.json")
    value, hit = analyze_segments(small, cache, tools=FAKE_TOOLS)
    assert value["error_kind"] == "decode" and not hit
    assert analyze_segments(small, cache, compute=False) == (None, False)      # not cached


def _fake_ffmpeg(audio: np.ndarray, duration_banner: str | None):
    """run_ffmpeg stand-in: writes ``audio`` (22.05 kHz mono) to the output path and prints
    an input banner with ``duration_banner`` (``None`` -> "N/A")."""
    calls: list[list[str]] = []

    def run(tools, args, **kw):
        calls.append(list(args))
        sf.write(args[-1], audio, SEG_SR, subtype="FLOAT")
        return subprocess.CompletedProcess(args, 0, "", f"  Duration: {duration_banner or 'N/A'}, start: 0.000000\n")

    return run, calls


def test_wav_header_without_frames_is_decoded_with_ffmpeg(tmp_path, monkeypatch):
    """Review regression: a WAV whose data size was never finalised (0) has real audio that
    libsndfile reports as 0 frames — it was cached as "too_short"; FFmpeg reads it."""
    y = dense(10.0, seed=6)
    src = wav(tmp_path / "rec.wav", y)
    raw = bytearray(src.read_bytes())
    i = raw.find(b"data")
    raw[i + 4:i + 8] = (0).to_bytes(4, "little")
    src.write_bytes(bytes(raw))
    assert sf.info(str(src)).frames == 0                         # precondition
    run, calls = _fake_ffmpeg(y, "00:00:10.00")
    monkeypatch.setattr(audio_intel, "run_ffmpeg", run)
    v = compute_segments(src, tools=FAKE_TOOLS)
    assert "error" not in v and v["analysed_s"] == pytest.approx(10.0, abs=0.01) and len(calls) == 1
    assert any("no audio frames" in n for n in v["notes"])
    # a file too small to hold min_audio_s is believed: no FFmpeg (empty.wav in the pure tests)
    empty = wav(tmp_path / "empty.wav", np.zeros(0, np.float32))
    assert compute_segments(empty, tools=FAKE_TOOLS)["error_kind"] == "too_short" and len(calls) == 1


def test_partial_ffmpeg_decode_after_a_libsndfile_failure_is_noted(tmp_path, monkeypatch):
    src = wav(tmp_path / "damaged.wav", dense(20.0, seed=7))

    def broken(*a, **kw):
        raise RuntimeError("Internal psf_fseek() failed")

    monkeypatch.setattr(audio_intel, "_read_mono", broken)
    run, calls = _fake_ffmpeg(dense(8.0, seed=7), "00:00:20.00")
    monkeypatch.setattr(audio_intel, "run_ffmpeg", run)
    v = compute_segments(src, tools=FAKE_TOOLS)
    assert "error" not in v and v["analysed_s"] == pytest.approx(8.0, abs=0.01)
    assert any("libsndfile could not decode" in n for n in v["notes"])
    assert any("decoded only 8.0 s of the 20.0 s" in n for n in v["notes"])


@pytest.mark.parametrize("banner, duration, unknown_note", [
    ("00:00:10.01", 10.0, False),      # exactly max_analysis_s long: nothing was cut off
    ("00:00:30.00", 30.0, False),      # truncated: the full duration comes from the banner
    (None, None, True),                # truncated or not: unknown
    ("00:00:04.00", None, True),       # an evidently wrong (estimated) banner is not believed
])
def test_ffmpeg_duration_at_the_analysis_cap(tmp_path, monkeypatch, banner, duration, unknown_note):
    src = tmp_path / "x.m4a"
    src.write_bytes(b"\x00\x00\x00\x18ftypM4A " + bytes(64))       # libsndfile cannot open it
    run, calls = _fake_ffmpeg(dense(10.0, seed=8), banner)
    monkeypatch.setattr(audio_intel, "run_ffmpeg", run)
    v = compute_segments(src, dataclasses.replace(CFG, max_analysis_s=10.0), tools=FAKE_TOOLS)
    assert "error" not in v and v["duration_s"] == duration
    assert any("full duration is unknown" in n for n in v["notes"]) is unknown_note
    assert calls[0][calls[0].index("-t") + 1] == "10.000"


def test_timbre_similarity_never_raises():
    assert timbre_similarity({"summary": "not a dict"}, [0.5]) is None
    assert timbre_similarity({"summary": None, "timbre": [0.5]}, [0.5]) is None
    assert timbre_similarity({"timbre": [0.1, 0.2]}, [0.1, 0.2]) == 1.0
    assert timbre_similarity([1e308, 1e308], [-1e308, 0.0]) == 0.0         # no OverflowError
    assert timbre_similarity([float("nan")], [0.1]) is None
    assert timbre_similarity([True, 0.1], [0.1, 0.1]) is None


def test_rounding_never_emits_negative_zero():
    assert math.copysign(1.0, audio_intel._r(-0.00001, 3)) == 1.0
    assert json.dumps({"x": audio_intel._r(-1e-9, 3)}) == '{"x": 0.0}'


# ---------------------------------------------------------------------------
# implementation invariants
# ---------------------------------------------------------------------------

def test_block_processing_and_hpss_match_full_resolution():
    y = dense(12.0) / 0.3
    a = audio_intel._frame_features(y, CFG)
    b = audio_intel._frame_features(y, CFG, block_frames=100)
    for k in ("tot", "bass", "harm", "perc", "centroid", "flatness", "vmid", "vflux", "sq", "count", "mel", "chroma"):
        assert np.allclose(getattr(a, k), getattr(b, k), rtol=1e-4, atol=1e-7), k
    S = np.abs(librosa.stft(y, n_fft=CFG.n_fft, hop_length=CFG.hop_length))
    assert np.allclose(a.tot, (S ** 2).sum(axis=0), rtol=1e-4)
    # pool factor 1 is librosa.decompose.hpss exactly
    full = audio_intel._frame_features(y, dataclasses.replace(CFG, hpss_time_pool=1, hpss_freq_pool=1))
    H, P = librosa.decompose.hpss(S)
    assert np.allclose(full.harm, (H ** 2).sum(axis=0), rtol=1e-3, atol=1e-6)
    assert np.allclose(full.perc, (P ** 2).sum(axis=0), rtol=1e-3, atol=1e-6)
    ratio = lambda f: f.perc.sum() / (f.harm.sum() + f.perc.sum())   # noqa: E731
    assert abs(ratio(a) - ratio(full)) < 0.03          # the pooled default stays on librosa's scale


def test_tempogram_means_match_librosa():
    y = dense(20.0)
    oenv = librosa.onset.onset_strength(y=y, sr=SEG_SR)
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=SEG_SR, win_length=344)
    total, active = audio_intel._tempogram_means(oenv, 344, chunk=128)
    assert np.allclose(total, tg.mean(axis=1)) and np.allclose(active, tg[:, tg[0] > 0.5].mean(axis=1))


def test_segments_params_cover_every_config_field():
    params = segments_params(CFG)
    assert {f.name for f in dataclasses.fields(SegmentConfig)} <= set(params)
    json.dumps(params, allow_nan=False)
    assert params_hash(params) == params_hash(segments_params(SegmentConfig()))
    assert params_hash(params) != params_hash(segments_params(dataclasses.replace(CFG, segment_s=4.0)))
    assert params_hash(params) != params_hash(segments_params(dataclasses.replace(CFG, density_bounds=(0.2, 0.5))))


# ---------------------------------------------------------------------------
# cache section "segments"
# ---------------------------------------------------------------------------

@pytest.fixture
def small(tmp_path) -> Path:
    return wav(tmp_path / "lib" / "small.wav", dense(12.0, seed=3))


@pytest.fixture
def count_computes(monkeypatch):
    calls: list[Path] = []
    real = audio_intel.compute_segments

    def counted(path, cfg=None, **kw):
        calls.append(Path(path))
        return real(path, cfg, **kw)

    monkeypatch.setattr(audio_intel, "compute_segments", counted)
    return calls


def test_second_call_is_a_cache_hit(tmp_path, small, count_computes):
    cache_path = tmp_path / "analysis_cache.json"
    cache = AnalysisCache(cache_path)
    before = sha256(small)
    v1, hit1 = analyze_segments(small, cache)
    assert sha256(small) == before                        # the source is never written
    v2, hit2 = analyze_segments(small, cache)
    assert (hit1, hit2) == (False, True) and v1 == v2 and len(count_computes) == 1
    cache.save()
    cache2 = AnalysisCache(cache_path)                    # from disk
    v3, hit3 = analyze_segments(small, cache2, compute=False)
    assert hit3 and v3 == v1 and len(count_computes) == 1
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    (entry,) = raw["entries"].values()
    assert entry["sections"][SEGMENTS_SECTION]["version"] == SEGMENTS_VERSION


def test_cache_only_lookup_returns_none_when_missing(tmp_path, small, count_computes):
    cache = AnalysisCache(tmp_path / "c.json")
    assert analyze_segments(small, cache, compute=False) == (None, False)
    assert count_computes == [] and not cache.dirty


def test_read_only_cache_never_writes(tmp_path, small):
    cache_path = tmp_path / "ro.json"
    ro = AnalysisCache(cache_path, read_only=True)
    v, hit = analyze_segments(small, ro)
    assert not hit and "error" not in v
    ro.save()
    assert not cache_path.exists()
    # an existing cache file stays byte-identical
    rw = AnalysisCache(cache_path)
    analyze_segments(small, rw)
    rw.save()
    before = cache_path.read_bytes()
    ro2 = AnalysisCache(cache_path, read_only=True)
    analyze_segments(small, ro2, dataclasses.replace(CFG, segment_s=4.0))
    ro2.save()
    assert cache_path.read_bytes() == before


def test_config_change_invalidates_only_the_segments_section(tmp_path, small, count_computes):
    cache_path = tmp_path / "c.json"
    acfg = AnalysisConfig()
    cache = AnalysisCache(cache_path)
    cache.put(small, "bpm", {"bpm": 127.5}, version=BPM_ALGO_VERSION, params=bpm_params(acfg))
    cache.put(small, "key", {"camelot": "8A", "key_confidence": 0.7}, version=KEY_ALGO_VERSION,
              params=key_params(acfg))
    analyze_segments(small, cache)
    cache.save()

    cache2 = AnalysisCache(cache_path)
    v, hit = analyze_segments(small, cache2, dataclasses.replace(CFG, segment_s=4.0))
    assert not hit and v["segment_s"] == 4.0 and len(count_computes) == 2
    assert cache2.stats["invalidated"] == 1
    assert cache2.get(small, "bpm", version=BPM_ALGO_VERSION, params=bpm_params(acfg)) == {"bpm": 127.5}
    assert cache2.get(small, "key", version=KEY_ALGO_VERSION, params=key_params(acfg)) == {
        "camelot": "8A", "key_confidence": 0.7}
    # the default config is now stale too (one section per file and name), recomputed on demand
    assert analyze_segments(small, cache2, compute=False) == (None, False)


def test_unknown_results_are_cached_but_decode_failures_are_not(tmp_path, monkeypatch, count_computes):
    cache = AnalysisCache(tmp_path / "c.json")
    silent = wav(tmp_path / "silent.wav", np.zeros(8 * SEG_SR, np.float32))
    v, _ = analyze_segments(silent, cache)
    assert v["error_kind"] == "silent"
    assert analyze_segments(silent, cache) == (v, True)

    corrupt = tmp_path / "corrupt.mp3"
    corrupt.write_bytes(b"\x00junk" * 600)
    with pytest.raises(sf.LibsndfileError):
        sf.info(str(corrupt))                              # precondition: the FFmpeg fallback runs

    def ffmpeg_fails(tools, args, **kw):
        raise FFmpegError("ffmpeg failed with exit code 1")

    monkeypatch.setattr(audio_intel, "run_ffmpeg", ffmpeg_fails)
    v1, hit1 = analyze_segments(corrupt, cache, tools=FAKE_TOOLS)
    v2, hit2 = analyze_segments(corrupt, cache, tools=FAKE_TOOLS)
    assert v1["error_kind"] == "decode" and not hit1 and not hit2     # retried, never cached
    assert analyze_segments(corrupt, cache, compute=False) == (None, False)


def test_malformed_cached_value_is_recomputed(tmp_path, small, count_computes):
    cache = AnalysisCache(tmp_path / "c.json")
    cache.put(small, SEGMENTS_SECTION, {"version": SEGMENTS_VERSION, "times": [0.0], "intensity": ["x"]},
              version=SEGMENTS_VERSION, params=segments_params(CFG))
    assert analyze_segments(small, cache, compute=False) == (None, False)
    v, hit = analyze_segments(small, cache)
    assert not hit and is_valid_segments(v) and len(count_computes) == 1
    assert analyze_segments(small, cache)[1] is True


def test_result_of_a_changed_file_is_not_cached(tmp_path, small):
    cache = AnalysisCache(tmp_path / "c.json")
    stale = dict(cache.fingerprint(small))
    stale["mtime_ns"] -= 10**9                                           # "taken before a modification"
    v, hit = analyze_segments(small, cache, expected_fingerprint=stale)
    assert "error" not in v and not hit
    assert analyze_segments(small, cache, compute=False) == (None, False)


def test_fingerprint_is_taken_before_decoding_when_not_given(tmp_path, small, monkeypatch):
    """Without ``expected_fingerprint`` the file state is captured BEFORE the decode: a
    file modified while it is analysed does not get the old content's result cached."""
    real = audio_intel.compute_segments

    def modified_meanwhile(path, cfg=None, **kw):
        value = real(path, cfg, **kw)
        st = Path(path).stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5 * 10**9))
        return value

    monkeypatch.setattr(audio_intel, "compute_segments", modified_meanwhile)
    cache = AnalysisCache(tmp_path / "c.json")
    v, hit = analyze_segments(small, cache)
    assert "error" not in v and not hit
    assert analyze_segments(small, cache, compute=False) == (None, False)


# ---------------------------------------------------------------------------
# FFmpeg decoding (formats libsndfile cannot read, real damaged files)
# ---------------------------------------------------------------------------

@requires_ffmpeg
@pytest.mark.ffmpeg
def test_m4a_is_decoded_by_ffmpeg_and_matches_the_wav(ffmpeg_tools, tmp_path):
    src = wav(tmp_path / "a.wav", structured()[: 45 * SEG_SR])
    m4a = tmp_path / "a.m4a"
    subprocess.run([ffmpeg_tools.ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", str(src), "-c:a", "aac",
                    "-b:a", "192k", str(m4a)], check=True, capture_output=True)
    with pytest.raises(sf.LibsndfileError):
        sf.info(str(m4a))                                  # precondition: FFmpeg path
    before = sha256(m4a)
    ref = compute_segments(src)
    v = compute_segments(m4a, tools=ffmpeg_tools)
    assert "error" not in v and is_valid_segments(v)
    assert v["duration_s"] == pytest.approx(45.0, abs=0.1)
    assert abs(v["summary"]["energy_mean"] - ref["summary"]["energy_mean"]) < 0.05
    assert v["summary"]["tempo"]["bpm"] == pytest.approx(ref["summary"]["tempo"]["bpm"], rel=0.02)
    assert [s["label"] for s in v["sections"]][:2] == [s["label"] for s in ref["sections"]][:2]
    assert sha256(m4a) == before
    # truncated by max_analysis_s: the full duration comes from FFmpeg's input banner
    short = compute_segments(m4a, dataclasses.replace(CFG, max_analysis_s=20.0), tools=ffmpeg_tools)
    assert short["analysed_s"] == pytest.approx(20.0, abs=0.05) and short["duration_s"] == pytest.approx(45.0, abs=0.1)
    assert any("max_analysis_s" in n for n in short["notes"])


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_mp3_and_corrupt_mp3(ffmpeg_tools, tmp_path):
    src = wav(tmp_path / "fx" / "d.wav", dense(20.0, seed=4))
    mp3 = encode_mp3(ffmpeg_tools, src, tmp_path / "d.mp3")
    v = compute_segments(mp3, tools=ffmpeg_tools)
    assert "error" not in v and v["duration_s"] == pytest.approx(20.0, abs=0.2)
    assert v["summary"]["density"]["value"] == "dense"
    corrupt = tmp_path / "broken.mp3"
    corrupt.write_bytes(b"ID3" + bytes(range(256)) * 40)
    bad = compute_segments(corrupt, tools=ffmpeg_tools)       # must not raise
    assert "error" in bad and bad["version"] == SEGMENTS_VERSION
    walk_json_native(bad)


# ---------------------------------------------------------------------------
# harmony summary (spec §2 "harmonic characteristics")
# ---------------------------------------------------------------------------

def _chords(seconds: float, roots: list[int], minor: bool, sr: int = 22050) -> np.ndarray:
    """Triads (sine partials) cycling through ``roots`` (semitones above C4), 2 s per chord."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for k in range(int(np.ceil(seconds / 2.0))):
        s0, s1 = int(k * 2.0 * sr), min(n, int((k + 1) * 2.0 * sr))
        root = roots[k % len(roots)]
        third = 3 if minor else 4
        for semi in (root, root + third, root + 7):
            f = 261.63 * 2 ** (semi / 12)
            y[s0:s1] += np.sin(2 * np.pi * f * t[s0:s1]) + 0.3 * np.sin(4 * np.pi * f * t[s0:s1])
    return (0.2 * y / np.max(np.abs(y))).astype(np.float32)


def test_harmony_summary_finds_a_stable_key(tmp_path):
    from veg_mixer.audio_intel import compute_segments
    path = tmp_path / "cmaj.wav"
    sf.write(str(path), _chords(60.0, [0, 7, 9, 5], minor=False), 22050)    # I-V-vi-IV in C major
    h = compute_segments(path)["summary"]["harmony"]
    assert h["method"] == "kk-stft-chroma-v1"
    assert h["key_estimate"] in ("8B", "9A")                    # C major (or its relative A minor)
    assert h["clarity"] > 0.6 and h["stability"] >= 0.8 and h["modulations"] == 0
    assert json.dumps(h)


def test_harmony_summary_counts_a_modulation(tmp_path):
    from veg_mixer.audio_intel import compute_segments
    first = _chords(40.0, [0, 7, 9, 5], minor=False)                       # C major
    second = _chords(40.0, [6, 13, 15, 11], minor=False)                   # F# major (tritone away)
    path = tmp_path / "mod.wav"
    sf.write(str(path), np.concatenate([first, second]), 22050)
    h = compute_segments(path)["summary"]["harmony"]
    assert h["modulations"] >= 1
    assert h["stability"] < 0.8
    keys = [k for k in h["segment_keys"] if k]
    assert len(set(keys)) >= 2
