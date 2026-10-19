"""P0 loudness normalization: pure planning/parsing units + real-FFmpeg integration."""

from __future__ import annotations

import dataclasses
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import SR, encode_mp3, music_like, requires_ffmpeg, sha256, spiky, write_wav
from veg_mixer import ffmpeg_tools
from veg_mixer.config import ConfigError, EncodeConfig, LoudnessConfig, QCConfig
from veg_mixer.ffmpeg_tools import (
    REQUIRED_FILTERS,
    FFmpegNotFoundError,
    FFmpegTools,
    LoudnessMeasurementError,
    build_measurement,
    decode_check,
    parse_ebur128_frames,
    parse_ebur128_summary,
    parse_loudnorm_json,
    probe_audio,
    run_ffmpeg,
    short_term_stats,
)
from veg_mixer.loudness import (
    DYNAMIC_MODE_DESCRIPTION,
    LoudnessNormalizer,
    NormalizationPlanError,
    compare_adjacent_short_term,
    evaluate_output,
    plan_normalization,
)
from veg_mixer.models import (
    AudioProbe,
    DecodeCheck,
    LoudnessMeasurement,
    LoudnormMode,
    NormalizationStatus,
    ShortTermStats,
)

CFG = LoudnessConfig()
TP_LIMIT = CFG.target_true_peak_dbtp + CFG.true_peak_tolerance_db
OPT_IN = dataclasses.replace(CFG, max_dynamic_gain_db=2.0)   # loudnorm dynamic mode explicitly allowed

LOUDNORM_OK = """
[Parsed_loudnorm_1 @ 000001]
{
	"input_i" : "-23.45",
	"input_tp" : "-4.10",
	"input_lra" : "6.30",
	"input_thresh" : "-33.80",
	"output_i" : "-16.02",
	"output_tp" : "-2.00",
	"output_lra" : "6.10",
	"output_thresh" : "-26.40",
	"normalization_type" : "linear",
	"target_offset" : "0.02"
}
"""

EBU_OK = """
[Parsed_ebur128_0 @ 0] t: 3.0  TARGET:-23 LUFS    M: -20.0 S: -21.0     I: -23.0 LUFS       LRA:   0.0 LU
[Parsed_ebur128_0 @ 0] t: 3.1  TARGET:-23 LUFS    M: -20.0 S: -22.0     I: -23.0 LUFS       LRA:   0.0 LU
[Parsed_ebur128_0 @ 0] t: 3.2  TARGET:-23 LUFS    M: -20.0 S: -inf     I: -23.0 LUFS       LRA:   0.0 LU
[Parsed_ebur128_0 @ 0] Summary:

  Integrated loudness:
    I:         -23.4 LUFS
    Threshold: -33.8 LUFS

  Loudness range:
    LRA:         6.3 LU
    Threshold: -43.8 LUFS
    LRA low:   -27.0 LUFS
    LRA high:  -20.7 LUFS

  True peak:
    Peak:       -3.9 dBFS
"""


def meas(i=-23.0, tp=-4.0, lra=6.0, thresh=-33.0, **kw) -> LoudnessMeasurement:
    return LoudnessMeasurement(integrated_lufs=i, true_peak_dbtp=tp, lra_lu=lra, threshold_lufs=thresh,
                               loudnorm_true_peak_dbtp=tp, **kw)


# ---------------------------------------------------------------------------
# Parsing / validation (malformed FFmpeg output — spec test 6)
# ---------------------------------------------------------------------------

def test_parse_loudnorm_json_valid():
    d = parse_loudnorm_json("noise before\n" + LOUDNORM_OK + "\nnoise after")
    assert d["input_i"] == -23.45 and d["input_tp"] == -4.10 and d["input_lra"] == 6.30
    assert d["input_thresh"] == -33.80 and d["target_offset"] == 0.02
    assert d["normalization_type"] == "linear"


@pytest.mark.parametrize("stderr, fragment", [
    ("", "not found"),
    ("Error opening input file", "not found"),
    ('{\n"input_i" : "-23.0",\n"input_tp" : "-1.0",\n', "not found"),                      # truncated
    ('{"input_i" : "-23.0", "input_tp" : "-1.0", "input_lra" : "5.0"}', "input_thresh"),    # missing key
    ('{"input_i" : "-inf", "input_tp" : "-1.0", "input_lra" : "5.0", "input_thresh" : "-70"}', "not finite"),
    ('{"input_i" : "abc", "input_tp" : "-1.0", "input_lra" : "5.0", "input_thresh" : "-30"}', "not a number"),
    ('{"input_i" : "12.0", "input_tp" : "-1.0", "input_lra" : "5.0", "input_thresh" : "-30"}', "plausible range"),
    ('{"input_i" : "-20.0", "input_tp" : "-1.0", "input_lra" : "nan", "input_thresh" : "-30"}', "not finite"),
    ('{"input_i" : "-20.0", "input_tp" : "-1.0", "input_lra" : "5", "input_thresh" : "-30", '
     '"normalization_type" : "weird"}', "normalization_type"),
    ('{"input_i" : "-20.0" "input_tp" : "-1.0", "input_lra" : "5", "input_thresh" : "-30"}', "malformed"),
])
def test_parse_loudnorm_json_rejects_malformed(stderr, fragment):
    with pytest.raises(LoudnessMeasurementError, match=fragment):
        parse_loudnorm_json(stderr)


def test_parse_ebur128_summary_and_frames():
    summary = parse_ebur128_summary(EBU_OK)
    assert summary == {"integrated": -23.4, "lra": 6.3, "true_peak": -3.9}
    t, s = parse_ebur128_frames(EBU_OK)
    assert list(t) == [3.0, 3.1] and list(s) == [-21.0, -22.0]   # -inf frame dropped


def test_build_measurement_uses_conservative_true_peak():
    m = build_measurement(LOUDNORM_OK + EBU_OK, "out_time_us=20000000\n")
    assert m.integrated_lufs == -23.45
    assert m.loudnorm_true_peak_dbtp == -4.10 and m.ebur128_true_peak_dbtp == -3.9
    assert m.true_peak_dbtp == -3.9          # max of the two meters
    assert m.duration_s == 20.0


def test_short_term_stats_head_tail_and_gate():
    t = np.round(np.arange(1, 301) * 0.1, 3)        # 30 s of 100 ms frames
    s = np.full_like(t, -20.0)
    s[t <= 10] = -30.0                               # quiet intro
    s[t >= 25] = -14.0                               # loud ending
    s[t < 3] = -120.0                                # warm-up frames are ignored
    st = short_term_stats(t, s, edge_window_s=10.0)
    assert st.head_lufs == -30.0 and st.tail_lufs == -14.0
    assert st.p95 == -14.0 and st.max == -14.0
    empty = short_term_stats(np.array([1.0, 2.0]), np.array([-120.0, -120.0]))
    assert empty.p50 is None and empty.head_lufs is None


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def test_plan_linear_when_peak_allows():
    p = plan_normalization(meas(i=-23.0, tp=-12.0), CFG)
    assert p.mode is LoudnormMode.LINEAR
    assert p.effective_target_lufs == CFG.target_lufs
    assert p.gain_needed_db == 7.0 and p.shortfall_db == 0.0
    assert p.true_peak_ceiling_dbtp == CFG.target_true_peak_dbtp - CFG.encode_headroom_db


def test_plan_dynamic_for_small_shortfall():
    # DSP-2: dynamic mode is gain riding (AGC), not a limiter -> opt-in only, and described as such.
    # gain +7 dB, TP -6 -> predicted +1; ceiling -2 => shortfall ~3.05 > 2 -> peak-limited even when opted in
    assert plan_normalization(meas(i=-23.0, tp=-6.0), OPT_IN).mode is LoudnormMode.PEAK_LIMITED_LINEAR
    # TP -7.5 -> shortfall ~1.55 dB: beyond the 1 LU tolerance, inside the opt-in budget -> dynamic
    p = plan_normalization(meas(i=-23.0, tp=-7.5), OPT_IN)
    assert p.mode is LoudnormMode.DYNAMIC and p.effective_target_lufs == CFG.target_lufs
    assert CFG.lufs_tolerance < p.shortfall_db <= OPT_IN.max_dynamic_gain_db
    assert any(DYNAMIC_MODE_DESCRIPTION in r for r in p.reasons)
    assert not any("true-peak limiting applied" in r for r in p.reasons)   # the old, misleading wording


def test_plan_default_config_never_uses_dynamic_mode_for_a_peak_shortfall():
    # DSP-2: max_dynamic_gain_db defaults to 0 -> the same 1.55 dB shortfall is delivered peak-safe
    assert CFG.max_dynamic_gain_db == 0.0
    p = plan_normalization(meas(i=-23.0, tp=-7.5), CFG)
    assert p.mode is LoudnormMode.PEAK_LIMITED_LINEAR
    assert p.effective_target_lufs == pytest.approx(-23.0 + p.peak_safe_gain_db, abs=0.01)
    assert not any(DYNAMIC_MODE_DESCRIPTION in r for r in p.reasons)


def test_plan_shortfall_within_tolerance_is_linear_at_peak_safe_gain():
    # DSP-2: TP -8.5 -> the peak-safe gain is ~0.55 dB short of the target, inside ±1 LU:
    # pure gain, delivered slightly below target, PASS — even when dynamic mode is opted in.
    for cfg in (CFG, OPT_IN):
        p = plan_normalization(meas(i=-23.0, tp=-8.5), cfg)
        assert p.mode is LoudnormMode.LINEAR
        assert 0 < p.shortfall_db <= cfg.lufs_tolerance
        assert p.effective_target_lufs == pytest.approx(-23.0 + p.peak_safe_gain_db, abs=0.01)
        assert -8.5 + (p.effective_target_lufs + 23.0) <= p.true_peak_ceiling_dbtp      # peak-safe
        assert p.lra_target > p.measured_lra_for_filter                                   # loudnorm stays linear
        assert any("tolerance" in r and "no limiting" in r for r in p.reasons)
        ev = evaluate(meas(i=p.effective_target_lufs, tp=-2.0), plan=p)
        assert ev.status is NormalizationStatus.PASSED and not ev.review_reasons


def test_lra_request_never_bypasses_the_dynamic_budget():
    # DSP-6: preserve_source_lra=False with LRA 16 > 11 used to force DYNAMIC whatever the
    # shortfall (~13 dB "of true-peak limiting" with a 2 dB budget).
    for budget in (0.0, 2.0):
        cfg = dataclasses.replace(CFG, preserve_source_lra=False, max_dynamic_gain_db=budget)
        p = plan_normalization(meas(i=-28.0, tp=-1.0, lra=16.0), cfg)
        assert p.mode is LoudnormMode.PEAK_LIMITED_LINEAR and p.shortfall_db > 13
        assert p.lra_target > 16.0                       # LRA passed so that loudnorm stays linear
        assert any("LRA compression requested" in r for r in p.reasons)
        assert not any(DYNAMIC_MODE_DESCRIPTION in r for r in p.reasons)
    # inside the budget the explicit LRA request is honoured — and described as dynamic processing
    cfg = dataclasses.replace(CFG, preserve_source_lra=False, max_dynamic_gain_db=2.0)
    p = plan_normalization(meas(i=-23.0, tp=-7.5, lra=16.0), cfg)
    assert p.mode is LoudnormMode.DYNAMIC and p.lra_target == CFG.target_lra
    assert sum(DYNAMIC_MODE_DESCRIPTION in r for r in p.reasons) == 2     # peak gap + LRA


def test_plan_peak_limited_when_target_unreachable():
    p = plan_normalization(meas(i=-28.0, tp=0.3), CFG)
    assert p.mode is LoudnormMode.PEAK_LIMITED_LINEAR
    # delivered loudness keeps the true peak under the ceiling
    assert p.effective_target_lufs + (0.3 - (-28.0)) <= p.true_peak_ceiling_dbtp + 1e-6
    assert p.effective_target_lufs < CFG.target_lufs - CFG.lufs_tolerance
    assert "limiting" in p.reasons[0]


def test_plan_never_dynamic_when_disabled():
    cfg = dataclasses.replace(CFG, max_dynamic_gain_db=0.0)
    # DSP-2 changed the expectation for TP -8.5 (shortfall ~0.55 dB inside the tolerance):
    # it is now LINEAR at the peak-safe gain instead of PEAK_LIMITED_LINEAR — never DYNAMIC.
    assert plan_normalization(meas(i=-23.0, tp=-8.5), cfg).mode is LoudnormMode.LINEAR
    assert plan_normalization(meas(i=-23.0, tp=-7.5), cfg).mode is LoudnormMode.PEAK_LIMITED_LINEAR


def test_plan_zero_lra_still_linear():
    p = plan_normalization(meas(i=-26.0, tp=-26.0, lra=0.0), CFG)
    assert p.mode is LoudnormMode.LINEAR and p.measured_lra_for_filter > 0


def test_plan_preserves_dynamic_masters():
    p = plan_normalization(meas(i=-23.0, tp=-12.0, lra=18.0), CFG)
    assert p.mode is LoudnormMode.LINEAR and p.lra_target >= 18.0
    q = plan_normalization(meas(i=-23.0, tp=-12.0, lra=18.0), dataclasses.replace(CFG, preserve_source_lra=False))
    assert q.mode is LoudnormMode.DYNAMIC        # LRA compression only when explicitly requested
    assert any(DYNAMIC_MODE_DESCRIPTION in r for r in q.reasons)


@pytest.mark.parametrize("m, fragment", [
    (meas(i=-75.0, thresh=-85.0), "near-silent"),
    (meas(i=float("-inf")), "not a finite"),
    (meas(tp=float("nan")), "not a finite"),
])
def test_plan_rejects_unusable_measurements(m, fragment):
    with pytest.raises(NormalizationPlanError, match=fragment):
        plan_normalization(m, CFG)


def test_config_validation():
    with pytest.raises(ConfigError):
        LoudnessConfig(target_lufs=-3.0).validate()
    with pytest.raises(ConfigError):
        LoudnessConfig(target_true_peak_dbtp=0.5).validate()
    with pytest.raises(ConfigError):
        EncodeConfig(channels=1).validate()
    LoudnessConfig(target_lufs=-14.0, target_true_peak_dbtp=-1.0).validate()


# ---------------------------------------------------------------------------
# Output evaluation (pure)
# ---------------------------------------------------------------------------

GOOD_PROBE = AudioProbe(path=Path("x.mp3"), duration_s=20.0, sample_rate=44100, channels=2, codec="mp3")
GOOD_DECODE = DecodeCheck(ok=True, decoded_duration_s=20.0)


def evaluate(m, plan=None, probe=GOOD_PROBE, decode=GOOD_DECODE, expected=20.0):
    return evaluate_output(plan=plan, measurement=m, probe=probe, decode=decode, expected_duration_s=expected,
                           cfg=CFG, encode=EncodeConfig(), qc=QCConfig(), kind="mp3")


def test_evaluate_passes_on_target():
    ev = evaluate(meas(i=-16.3, tp=-1.8))
    assert ev.status is NormalizationStatus.PASSED and not ev.errors and ev.correction_db == 0.0


def test_evaluate_true_peak_over_limit_fails_and_proposes_attenuation():
    ev = evaluate(meas(i=-16.0, tp=-1.0))
    assert ev.status is NormalizationStatus.FAILED
    assert any("true peak" in e for e in ev.errors)
    assert ev.correction_db < 0 and -1.0 + ev.correction_db <= CFG.target_true_peak_dbtp


def test_evaluate_peak_limited_shortfall_is_review_not_pass():
    plan = plan_normalization(meas(i=-28.0, tp=0.3), CFG)
    ev = evaluate(meas(i=-30.6, tp=-1.55), plan=plan)
    assert ev.status is NormalizationStatus.REVIEW
    assert ev.review_reasons and "true-peak" in ev.review_reasons[0]


def test_evaluate_unexplained_loudness_error_fails():
    ev = evaluate(meas(i=-19.0, tp=-8.0), plan=plan_normalization(meas(i=-25, tp=-15), CFG))
    # TP has room -> a boost is proposed; if it were final, the result must not pass
    assert ev.status is NormalizationStatus.FAILED and ev.correction_db > 0


@pytest.mark.parametrize("probe, decode, expected, fragment", [
    (GOOD_PROBE, DecodeCheck(ok=False, decoded_duration_s=None, errors=["Header missing"]), 20.0, "decodable"),
    (GOOD_PROBE, DecodeCheck(ok=True, decoded_duration_s=17.0), 20.0, "duration mismatch"),
    (dataclasses.replace(GOOD_PROBE, channels=1), GOOD_DECODE, 20.0, "channel"),
    (dataclasses.replace(GOOD_PROBE, sample_rate=22050), GOOD_DECODE, 20.0, "sample rate"),
    (dataclasses.replace(GOOD_PROBE, codec="pcm_s16le"), GOOD_DECODE, 20.0, "codec"),
])
def test_evaluate_integrity_failures(probe, decode, expected, fragment):
    ev = evaluate(meas(i=-16.0, tp=-2.0), probe=probe, decode=decode, expected=expected)
    assert ev.status is NormalizationStatus.FAILED
    assert any(fragment in e for e in ev.errors)
    assert ev.correction_db == 0.0     # re-encoding cannot fix integrity problems


def test_compare_adjacent_short_term_flags_big_differences():
    qc = QCConfig()
    a = ShortTermStats(p10=-20, p50=-17, p95=-14, max=-13, head_lufs=-18, tail_lufs=-30)
    b = ShortTermStats(p10=-12, p50=-10, p95=-8, max=-7, head_lufs=-9, tail_lufs=-12)
    reasons = compare_adjacent_short_term(a, b, qc)
    assert len(reasons) == 2
    steady = ShortTermStats(p10=-20, p50=-17, p95=-14, max=-13, head_lufs=-17, tail_lufs=-17)
    assert compare_adjacent_short_term(steady, steady, qc) == []
    assert compare_adjacent_short_term(None, b, qc) == []


# ---------------------------------------------------------------------------
# Malformed measurement inside the normalizer -> FAILED result, nothing published
# ---------------------------------------------------------------------------

def test_malformed_ffmpeg_measurement_yields_failed_result(tmp_path, monkeypatch):
    src = write_wav(tmp_path / "src.wav", music_like(2.0))
    before = sha256(src)
    fake = FFmpegTools(ffmpeg="ffmpeg", ffprobe=None, version="fake", has_soxr=False, has_libmp3lame=True)

    def garbage(tools, args, check=True, timeout=None):
        return subprocess.CompletedProcess(args, 0, stdout="", stderr='{"input_i": "-inf", "garbage": 1}')

    monkeypatch.setattr(ffmpeg_tools, "run_ffmpeg", garbage)
    probe = AudioProbe(path=src, duration_s=2.0, sample_rate=SR, channels=2, codec="pcm_f32le")
    r = LoudnessNormalizer(fake).normalize_to_mp3(src, tmp_path / "out.mp3", source_probe=probe)
    assert r.status is NormalizationStatus.FAILED and r.requires_review
    assert r.output is None and not (tmp_path / "out.mp3").exists()
    assert any("loudnorm" in e for e in r.errors)
    assert r.to_metadata()["normalization_status"] == "failed"
    assert sha256(src) == before


def test_refuses_to_overwrite_source(tmp_path):
    src = write_wav(tmp_path / "a.wav", music_like(1.0))
    fake = FFmpegTools(ffmpeg="ffmpeg", ffprobe=None, version="fake", has_soxr=False, has_libmp3lame=True)
    r = LoudnessNormalizer(fake).normalize_to_wav(src, src)
    assert r.status is NormalizationStatus.FAILED and "overwrite the source" in r.errors[0]


# ---------------------------------------------------------------------------
# FFmpeg environment errors (exit 3 at the CLI, never "broken track")
# ---------------------------------------------------------------------------

def test_vanished_ffprobe_or_ffmpeg_is_an_environment_error_not_a_broken_file(tmp_path):
    # SAFE-14: a binary that disappears after discovery used to surface as a bare
    # FileNotFoundError (OSError), which callers treat as a broken input file.
    src = write_wav(tmp_path / "a.wav", music_like(1.0))
    gone = FFmpegTools(ffmpeg=str(tmp_path / "gone" / "ffmpeg.exe"), ffprobe=str(tmp_path / "gone" / "ffprobe.exe"),
                       version="fake", has_soxr=False, has_libmp3lame=True)
    with pytest.raises(FFmpegNotFoundError, match="ffprobe") as exc:
        probe_audio(gone, src)
    assert "winget" in str(exc.value)                                   # installation hint
    with pytest.raises(FFmpegNotFoundError, match="ffmpeg"):
        run_ffmpeg(gone, ["-version"])
    with pytest.raises(FFmpegNotFoundError):
        decode_check(gone, src)
    with pytest.raises(FFmpegNotFoundError):                            # propagates, not a FAILED track
        LoudnessNormalizer(gone).normalize_to_mp3(src, tmp_path / "out.mp3")
    # replaced by something that cannot be executed (here: a directory -> PermissionError)
    unusable = dataclasses.replace(gone, ffprobe=str(tmp_path))
    with pytest.raises(FFmpegNotFoundError, match="cannot be executed"):
        probe_audio(unusable, src)


def _fake_discovery(monkeypatch, *, encoders: str, fail_on: str | None = None) -> None:
    filters = "\n".join(f" ..C {name:<16} A->A       fake filter" for name in REQUIRED_FILTERS)
    outputs = {"-version": "ffmpeg version fake-build\n", "-filters": filters, "-encoders": encoders}

    def fake_run(cmd, **kw):
        if cmd[-1] == fail_on:
            raise FileNotFoundError(2, "The system cannot find the file specified", cmd[0])
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs[cmd[-1]], stderr="")

    monkeypatch.setattr(ffmpeg_tools.subprocess, "run", fake_run)


def test_find_ffmpeg_rejects_a_build_without_libmp3lame(monkeypatch):
    # INT-20: no MP3 encoder must be an environment error at startup (exit 3), not a
    # per-track FAILED (exit 1)
    find = ffmpeg_tools.find_ffmpeg.__wrapped__          # bypass the lru_cache
    _fake_discovery(monkeypatch, encoders=" A....D mp2                  MP2 (MPEG audio layer 2)\n")
    with pytest.raises(FFmpegNotFoundError, match="libmp3lame") as exc:
        find("fake-ffmpeg", "fake-ffprobe")
    assert "winget" in str(exc.value)
    _fake_discovery(monkeypatch, encoders=" A....D libmp3lame           libmp3lame MP3 (MPEG audio layer 3)\n")
    tools = find("fake-ffmpeg", "fake-ffprobe")
    assert tools.has_libmp3lame and tools.ffprobe == "fake-ffprobe"
    # SAFE-14: vanishing between the discovery queries is an environment error too
    _fake_discovery(monkeypatch, encoders="", fail_on="-encoders")
    with pytest.raises(FFmpegNotFoundError, match="no longer available"):
        find("fake-ffmpeg", "fake-ffprobe")


# ---------------------------------------------------------------------------
# Real FFmpeg integration
# ---------------------------------------------------------------------------

def _mp3_fixture(tools, tmp_path: Path, name: str, data: np.ndarray, **tags) -> Path:
    wav = write_wav(tmp_path / "fixtures" / f"{name}.wav", data)
    return encode_mp3(tools, wav, tmp_path / "fixtures" / f"{name}.mp3", tags=tags or None)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_two_signals_with_different_loudness_reach_target(ffmpeg_tools, tmp_path):
    """Spec tests 1-4 + 7: different inputs -> both within tolerance on the FINAL MP3."""
    quiet = _mp3_fixture(ffmpeg_tools, tmp_path, "quiet", music_like(20, peak_dbfs=-20, seed=1))
    loud = _mp3_fixture(ffmpeg_tools, tmp_path, "loud", music_like(20, peak_dbfs=-0.5, seed=2))
    hashes = {p: sha256(p) for p in (quiet, loud)}
    norm = LoudnessNormalizer(ffmpeg_tools)

    m_quiet, m_loud = norm.measure(quiet), norm.measure(loud)
    assert m_loud.integrated_lufs - m_quiet.integrated_lufs > 10     # genuinely different inputs

    results = [norm.normalize_to_mp3(src, tmp_path / "out" / f"{src.stem}.mp3") for src in (quiet, loud)]
    for r in results:
        assert r.status is NormalizationStatus.PASSED, (r.errors, r.review_reasons)
        assert r.plan.mode is LoudnormMode.LINEAR and r.loudnorm_reported_mode == "linear"
        assert r.output.suffix == ".mp3" and r.output_probe.codec == "mp3"
        assert abs(r.measured_lufs - CFG.target_lufs) <= CFG.lufs_tolerance
        assert r.true_peak_dbtp <= TP_LIMIT
        # the stored numbers are a measurement of the published MP3 itself
        again = norm.measure(r.output)
        assert math.isclose(again.integrated_lufs, r.measured_lufs, abs_tol=0.05)
        assert math.isclose(again.true_peak_dbtp, r.true_peak_dbtp, abs_tol=0.05)
        meta = r.to_metadata()
        assert meta["normalization_status"] == "passed" and meta["requires_review"] is False
        assert meta["original_lufs"] is not None and meta["measured_lufs"] == r.measured_lufs
    assert abs(results[0].measured_lufs - results[1].measured_lufs) < 1.0
    assert {p: sha256(p) for p in (quiet, loud)} == hashes          # sources untouched
    assert not list((tmp_path / "out").glob(".*veg-*"))              # no temp leftovers


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_loudness_and_peak_conflict_is_flagged_not_squashed(ffmpeg_tools, tmp_path):
    """Spec test 5: the target needs ~14 dB of limiting -> peak-safe output, REVIEW."""
    src = _mp3_fixture(ffmpeg_tools, tmp_path, "spiky", spiky(20))
    r = LoudnessNormalizer(ffmpeg_tools).normalize_to_mp3(src, tmp_path / "out" / "spiky.mp3")
    assert r.plan.mode is LoudnormMode.PEAK_LIMITED_LINEAR
    assert r.status is NormalizationStatus.REVIEW and r.requires_review
    assert r.true_peak_dbtp <= TP_LIMIT                              # true peak wins
    assert r.measured_lufs < CFG.target_lufs - CFG.lufs_tolerance    # ...over the LUFS number
    assert r.review_reasons and "true-peak" in r.review_reasons[0]
    assert r.loudnorm_reported_mode == "linear"                      # no hidden limiting
    assert r.to_metadata()["normalization_status"] == "review"


def _unmastered(ffmpeg_tools, tmp_path: Path, louder_than_peak_allows_db: float):
    """An unmastered source and a target ``louder_than_peak_allows_db`` above what its
    true peak allows linearly (the plan's shortfall is that plus PEAK_SAFETY_DB)."""
    src = write_wav(tmp_path / "fx" / "unmastered.wav", music_like(20, peak_dbfs=-20, seed=4, drive=0))
    m = LoudnessNormalizer(ffmpeg_tools).measure(src)
    crest = m.true_peak_dbtp - m.integrated_lufs
    return src, m, round(CFG.pcm_true_peak_ceiling - crest + louder_than_peak_allows_db, 1)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_small_shortfall_uses_dynamic_mode_and_is_reported(ffmpeg_tools, tmp_path):
    # DSP-2: dynamic mode only when explicitly opted in (max_dynamic_gain_db=2.0), reported
    # as gain riding / AGC and flagged for listening review (review_dynamic defaults to True).
    src, m, target = _unmastered(ffmpeg_tools, tmp_path, 1.5)     # shortfall ~1.5 dB > 1 LU tolerance
    cfg = dataclasses.replace(OPT_IN, target_lufs=target)
    r = LoudnessNormalizer(ffmpeg_tools, cfg).normalize_to_mp3(src, tmp_path / "out" / "dyn.mp3", input_measurement=m)
    assert r.plan.mode is LoudnormMode.DYNAMIC and r.loudnorm_reported_mode == "dynamic"
    assert any(DYNAMIC_MODE_DESCRIPTION in w for w in r.warnings)
    assert r.status is NormalizationStatus.PASSED, (r.errors, r.review_reasons)
    assert r.requires_review and any(DYNAMIC_MODE_DESCRIPTION in x for x in r.review_reasons)
    assert r.to_metadata()["requires_review"] is True
    assert abs(r.measured_lufs - target) <= cfg.lufs_tolerance
    assert r.true_peak_dbtp <= cfg.target_true_peak_dbtp + cfg.true_peak_tolerance_db


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_default_config_delivers_the_same_shortfall_without_dynamic_mode(ffmpeg_tools, tmp_path):
    # DSP-2: by default (max_dynamic_gain_db=0) loudnorm never rides the gain
    src, m, target = _unmastered(ffmpeg_tools, tmp_path, 1.5)
    cfg = dataclasses.replace(CFG, target_lufs=target)
    r = LoudnessNormalizer(ffmpeg_tools, cfg).normalize_to_mp3(src, tmp_path / "out" / "lin.mp3", input_measurement=m)
    assert r.plan.mode is LoudnormMode.PEAK_LIMITED_LINEAR and r.loudnorm_reported_mode == "linear"
    assert r.status is not NormalizationStatus.FAILED, r.errors
    assert not any(DYNAMIC_MODE_DESCRIPTION in x for x in r.warnings + r.review_reasons)
    assert r.true_peak_dbtp <= TP_LIMIT


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_shortfall_within_tolerance_is_delivered_linearly_and_passes(ffmpeg_tools, tmp_path):
    # DSP-2: ~0.5 dB short of the target with a peak-safe linear gain -> LINEAR, PASS, no review
    src, m, target = _unmastered(ffmpeg_tools, tmp_path, 0.5)
    for cfg in (dataclasses.replace(CFG, target_lufs=target), dataclasses.replace(OPT_IN, target_lufs=target)):
        r = LoudnessNormalizer(ffmpeg_tools, cfg).normalize_to_mp3(
            src, tmp_path / "out" / f"lin{cfg.max_dynamic_gain_db:g}.mp3", input_measurement=m)
        assert r.plan.mode is LoudnormMode.LINEAR and r.loudnorm_reported_mode == "linear"
        assert 0 < r.plan.shortfall_db <= cfg.lufs_tolerance
        assert r.status is NormalizationStatus.PASSED and not r.requires_review, (r.errors, r.review_reasons)
        assert abs(r.measured_lufs - target) <= cfg.lufs_tolerance
        assert r.true_peak_dbtp <= TP_LIMIT


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_mp3_verification_is_measured_on_the_mp3_not_the_pcm(ffmpeg_tools, tmp_path):
    """Spec test 7: closed loop on the encoded file. A hot input whose PCM sits just at
    the ceiling must still end below the limit after MP3 encoding."""
    src = _mp3_fixture(ffmpeg_tools, tmp_path, "hot", music_like(20, peak_dbfs=-0.1, seed=5, drive=6))
    r = LoudnessNormalizer(ffmpeg_tools).normalize_to_mp3(src, tmp_path / "out" / "hot.mp3")
    assert r.status is NormalizationStatus.PASSED, (r.errors, r.review_reasons)
    probe = probe_audio(ffmpeg_tools, r.output)
    assert probe.codec == "mp3" and probe.channels == 2 and probe.sample_rate == 44100
    assert r.output_measurement.duration_s == pytest.approx(r.decode_check.decoded_duration_s, abs=0.05)
    assert r.true_peak_dbtp <= TP_LIMIT
    assert abs(r.decode_check.decoded_duration_s - r.expected_duration_s) <= QCConfig().duration_tolerance_s


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_lossless_source_mono_input_and_wav_output(ffmpeg_tools, tmp_path):
    mono = write_wav(tmp_path / "fx" / "mono.wav", music_like(15, peak_dbfs=-12, seed=6, channels=1))
    norm = LoudnessNormalizer(ffmpeg_tools)
    r_mp3 = norm.normalize_to_mp3(mono, tmp_path / "out" / "mono.mp3")       # inline (no extra WAV)
    assert r_mp3.status is NormalizationStatus.PASSED, r_mp3.errors
    assert r_mp3.output_probe.channels == 2
    r_wav = norm.normalize_to_wav(mono, tmp_path / "out" / "mono.wav")
    assert r_wav.status is NormalizationStatus.PASSED, r_wav.errors
    assert r_wav.output_probe.codec == "pcm_f32le" and r_wav.output_probe.channels == 2
    assert abs(r_wav.measured_lufs - CFG.target_lufs) <= CFG.lufs_tolerance
    assert r_wav.true_peak_dbtp <= CFG.pcm_true_peak_ceiling + CFG.true_peak_tolerance_db


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_steady_tone_with_zero_lra_stays_linear(ffmpeg_tools, tmp_path):
    t = np.arange(15 * SR) / SR
    tone = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    src = write_wav(tmp_path / "fx" / "tone.wav", np.stack([tone, tone], 1))
    r = LoudnessNormalizer(ffmpeg_tools).normalize_to_mp3(src, tmp_path / "out" / "tone.mp3")
    assert r.input_measurement.lra_lu == 0.0
    assert r.loudnorm_reported_mode == "linear", r.warnings
    assert r.status is NormalizationStatus.PASSED, r.errors


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_tags_are_preserved_and_stale_replaygain_removed(ffmpeg_tools, tmp_path):
    src = _mp3_fixture(ffmpeg_tools, tmp_path, "tagged", music_like(10, peak_dbfs=-3, seed=7),
                       title="Été à Paris", artist="Ünïcode Artist", REPLAYGAIN_TRACK_GAIN="-7.5 dB")
    r = LoudnessNormalizer(ffmpeg_tools).normalize_to_mp3(
        src, tmp_path / "out" / "été.mp3", metadata={"track": "3"})
    assert r.status is NormalizationStatus.PASSED, r.errors
    tags = probe_audio(ffmpeg_tools, r.output).tags
    assert tags.get("title") == "Été à Paris" and tags.get("artist") == "Ünïcode Artist"
    assert tags.get("track", "").startswith("3")
    assert "replaygain_track_gain" not in tags


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_silent_and_corrupt_inputs_fail_explicitly(ffmpeg_tools, tmp_path):
    silent = write_wav(tmp_path / "fx" / "silent.wav", np.zeros((10 * SR, 2), np.float32))
    corrupt = tmp_path / "fx" / "corrupt.mp3"
    corrupt.write_bytes(b"ID3" + bytes(range(256)) * 40)
    norm = LoudnessNormalizer(ffmpeg_tools)
    for src in (silent, corrupt):
        r = norm.normalize_to_mp3(src, tmp_path / "out" / f"{src.stem}.mp3")
        assert r.status is NormalizationStatus.FAILED and r.errors, src
        assert r.output is None and not (tmp_path / "out" / f"{src.stem}.mp3").exists()
