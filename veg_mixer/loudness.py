"""P0: perceptual loudness normalization (EBU R128, FFmpeg loudnorm, two-pass)
with post-export verification of the actual encoded file.

Pipeline per track::

    pass 1   measure source (loudnorm JSON + ebur128)            -> LoudnessMeasurement
    plan     pure decision: linear / dynamic / peak-limited      -> NormalizationPlan
    pass 2   loudnorm with measured values -> lossless WAV (float, resampled once)
    encode   WAV -> MP3 (the ONLY lossy step; tags/cover copied from the source)
    verify   re-measure the ENCODED file: LUFS, true peak, duration, channels,
             sample rate, full decode. Closed-loop gain correction re-encodes from
             the lossless WAV (never from a previous MP3).
    commit   atomic os.replace into place — failed outputs are never published.

True-peak safety takes precedence over hitting the LUFS number. When the peak-safe
linear gain lands inside the LUFS tolerance the track is delivered linearly, slightly
below target. A larger shortfall is delivered peak-safe (linear, quieter) with status
REVIEW instead of being squashed. loudnorm's dynamic mode is NOT a transparent
limiter (it rides gain over the whole track like an AGC and limits true peaks), so
it is used only when opted in (``max_dynamic_gain_db`` > 0) or when LRA compression
was explicitly requested, and it is flagged for review (``review_dynamic``).
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    LOSSLESS_EXTENSIONS,
    LOUDNORM_I_RANGE,
    LOUDNORM_LRA_RANGE,
    EncodeConfig,
    LoudnessConfig,
    QCConfig,
)
from .ffmpeg_tools import (
    FFmpegError,
    FFmpegTools,
    LoudnessMeasurementError,
    decode_check,
    measure_loudness,
    parse_loudnorm_json,
    probe_audio,
    run_ffmpeg,
)
from .models import (
    AudioProbe,
    DecodeCheck,
    LoudnessMeasurement,
    LoudnormMode,
    NormalizationPlan,
    NormalizationResult,
    NormalizationStatus,
    ShortTermStats,
)

log = logging.getLogger(__name__)

# Keeps loudnorm's internal `measured_tp + offset <= TP` linear-mode test true
# despite the 2-decimal rounding of the values we pass on the command line.
PEAK_SAFETY_DB = 0.05
# Closed-loop corrections aim this far below the true-peak target.
CORRECTION_TP_MARGIN_DB = 0.1
# loudnorm treats measured_LRA == 0 as "not supplied" and then refuses linear mode
# (verified on FFmpeg 9.0.2 with a steady tone). 0.01 LU is below meter resolution.
MIN_FILTER_LRA = 0.01
# loudnorm only stays linear while measured_LRA <= LRA: the LRA passed for a linear
# plan sits this far above the measured value.
LINEAR_LRA_MARGIN = 0.1
# What loudnorm's dynamic mode really does (reported in reasons / review reasons).
DYNAMIC_MODE_DESCRIPTION = "loudnorm dynamic mode (gain riding / AGC + true-peak limiting)"

# Tags whose values become wrong after loudness processing; cleared on export.
STALE_LOUDNESS_TAGS = (
    "replaygain_track_gain", "replaygain_track_peak", "replaygain_album_gain",
    "replaygain_album_peak", "r128_track_gain", "r128_album_gain", "itunnorm",
)


class NormalizationPlanError(ValueError):
    """The measured input cannot be normalized (silent, non-finite values, ...)."""


# ---------------------------------------------------------------------------
# Planning (pure)
# ---------------------------------------------------------------------------

def plan_normalization(m: LoudnessMeasurement, cfg: LoudnessConfig) -> NormalizationPlan:
    """Decide how to reach the target from a pass-1 measurement.

    * LINEAR — the required gain keeps the true peak under the PCM ceiling; or the
      peak-safe gain falls short of the target by no more than ``lufs_tolerance``
      (then the peak-safe gain is applied: slightly below target, still a pass).
    * DYNAMIC — loudnorm's dynamic mode, which rides the gain over the whole track
      (AGC) and limits true peaks, so it changes the master's macro-dynamics. Used
      only when opted in (``max_dynamic_gain_db`` > 0, shortfall beyond the
      tolerance but within that budget) or when LRA compression was explicitly
      requested (``preserve_source_lra=False``) and the shortfall fits the budget.
    * PEAK_LIMITED_LINEAR — a shortfall above ``max_dynamic_gain_db`` (whatever the
      LRA) and beyond the tolerance: apply only the peak-safe gain and report the
      track for review instead of limiting it hard.

    Every non-DYNAMIC plan passes an LRA of at least the measured LRA so loudnorm's
    linear-mode test holds and it does not fall back to dynamic processing.
    """
    values = {"integrated loudness": m.integrated_lufs, "true peak": m.true_peak_dbtp,
              "loudness range": m.lra_lu, "gating threshold": m.threshold_lufs}
    for name, v in values.items():
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise NormalizationPlanError(f"measured {name} is not a finite number ({v!r})")
    if m.integrated_lufs < cfg.min_input_lufs:
        raise NormalizationPlanError(
            f"input is near-silent ({m.integrated_lufs:.1f} LUFS < {cfg.min_input_lufs:.1f} LUFS)")

    ceiling = cfg.pcm_true_peak_ceiling
    gain = cfg.target_lufs - m.integrated_lufs
    predicted_tp = m.true_peak_dbtp + gain
    peak_safe_gain = ceiling - m.true_peak_dbtp - PEAK_SAFETY_DB

    lra_for_filter = max(m.lra_lu, MIN_FILTER_LRA)
    # LRA that keeps loudnorm linear (its linear test requires measured_LRA <= LRA).
    lra_linear = _clamp_lra(max(cfg.target_lra, lra_for_filter + LINEAR_LRA_MARGIN))
    lra_requested = lra_linear if cfg.preserve_source_lra else _clamp_lra(cfg.target_lra)
    # LRA compression explicitly requested (preserve_source_lra=False) and needed — or
    # unavoidable (measured LRA above loudnorm's 50 LU maximum).
    lra_compression = lra_for_filter > lra_requested

    reasons: list[str] = []
    shortfall = max(0.0, gain - peak_safe_gain)
    within_budget = shortfall <= cfg.max_dynamic_gain_db      # DSP-6: checked before any LRA rule
    if shortfall == 0.0 and not lra_compression:
        mode, effective, lra_target = LoudnormMode.LINEAR, cfg.target_lufs, lra_requested
    elif within_budget and (lra_compression or shortfall > cfg.lufs_tolerance):
        mode, effective, lra_target = LoudnormMode.DYNAMIC, cfg.target_lufs, lra_requested
        if shortfall > 0:
            reasons.append(
                f"{DYNAMIC_MODE_DESCRIPTION}: target {cfg.target_lufs:.1f} LUFS needs {gain:+.1f} dB but the "
                f"true peak ({m.true_peak_dbtp:+.1f} dBTP) allows only {peak_safe_gain:+.1f} dB linearly; the "
                f"remaining ~{shortfall:.1f} dB (opt-in budget {cfg.max_dynamic_gain_db:.1f} dB) comes from "
                f"riding the gain over the whole track, which changes its macro-dynamics")
        if lra_compression:
            why = ("LRA compression was requested (preserve_source_lra=False)" if not cfg.preserve_source_lra
                   else "that is above loudnorm's maximum LRA, so linear mode is impossible")
            reasons.append(
                f"{DYNAMIC_MODE_DESCRIPTION}: measured LRA {m.lra_lu:.1f} LU exceeds the LRA target "
                f"{lra_requested:.1f} LU and {why}")
    elif shortfall <= cfg.lufs_tolerance:
        mode, lra_target = LoudnormMode.LINEAR, lra_linear
        effective = m.integrated_lufs + peak_safe_gain
        reasons.append(
            f"target {cfg.target_lufs:.1f} LUFS needs {gain:+.1f} dB but the true peak "
            f"({m.true_peak_dbtp:+.1f} dBTP) allows only {peak_safe_gain:+.1f} dB linearly; delivering "
            f"~{effective:.1f} LUFS with pure gain ({shortfall:.1f} LU below target, inside the "
            f"±{cfg.lufs_tolerance:.1f} LU tolerance) — no limiting or dynamic processing")
    else:
        mode, lra_target = LoudnormMode.PEAK_LIMITED_LINEAR, lra_linear
        effective = m.integrated_lufs + peak_safe_gain
        reasons.append(
            f"target {cfg.target_lufs:.1f} LUFS needs {gain:+.1f} dB but the true peak "
            f"({m.true_peak_dbtp:+.1f} dBTP) allows only {peak_safe_gain:+.1f} dB; reaching it would need "
            f"~{shortfall:.1f} dB of limiting (max allowed {cfg.max_dynamic_gain_db:.1f} dB) — "
            f"delivering ~{effective:.1f} LUFS peak-safe instead")
    if lra_compression and mode is not LoudnormMode.DYNAMIC and not cfg.preserve_source_lra:
        reasons.append(
            f"LRA compression requested (measured LRA {m.lra_lu:.1f} LU > target {lra_requested:.1f} LU) "
            f"but not applied: it needs loudnorm's dynamic mode and the true-peak shortfall "
            f"{shortfall:.1f} dB exceeds max_dynamic_gain_db {cfg.max_dynamic_gain_db:.1f} dB")

    if not (LOUDNORM_I_RANGE[0] <= effective <= LOUDNORM_I_RANGE[1]):
        raise NormalizationPlanError(
            f"effective target {effective:.1f} LUFS is outside loudnorm's range {LOUDNORM_I_RANGE}")

    return NormalizationPlan(
        mode=mode,
        target_lufs=cfg.target_lufs,
        effective_target_lufs=round(effective, 2),
        true_peak_ceiling_dbtp=round(ceiling, 2),
        lra_target=round(lra_target, 2),
        gain_needed_db=round(gain, 2),
        predicted_true_peak_dbtp=round(predicted_tp, 2),
        peak_safe_gain_db=round(peak_safe_gain, 2),
        shortfall_db=round(shortfall, 2),
        measured_lra_for_filter=round(lra_for_filter, 2),
        reasons=reasons,
    )


def _clamp_lra(value: float) -> float:
    return min(max(value, LOUDNORM_LRA_RANGE[0]), LOUDNORM_LRA_RANGE[1])


# ---------------------------------------------------------------------------
# Output evaluation (pure)
# ---------------------------------------------------------------------------

@dataclass
class OutputEvaluation:
    status: NormalizationStatus
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)
    correction_db: float = 0.0   # proposed gain change for another attempt (0 = none)


def evaluate_output(
    *,
    plan: NormalizationPlan | None,
    measurement: LoudnessMeasurement | None,
    probe: AudioProbe | None,
    decode: DecodeCheck | None,
    expected_duration_s: float | None,
    cfg: LoudnessConfig,
    encode: EncodeConfig,
    qc: QCConfig,
    kind: str = "mp3",
    extra_errors: list[str] | None = None,
) -> OutputEvaluation:
    """Compare the measured output with the targets. A clean ffmpeg exit code is not
    evidence; only the measured encoded file is."""
    ev = OutputEvaluation(status=NormalizationStatus.FAILED, errors=list(extra_errors or []))

    # --- integrity ---
    if decode is None:
        ev.errors.append("output decode check did not run")
    elif not decode.ok:
        detail = "; ".join(decode.errors[:3]) or "unknown decoder error"
        ev.errors.append(f"output is not cleanly decodable: {detail}")
    duration = decode.decoded_duration_s if decode else None
    if duration is None or duration <= 0:
        ev.errors.append("output has zero or unknown decoded duration")
    elif expected_duration_s and abs(duration - expected_duration_s) > qc.duration_tolerance_s:
        ev.errors.append(f"duration mismatch: output {duration:.3f} s vs source {expected_duration_s:.3f} s")
    if probe is None:
        ev.errors.append("output could not be probed")
    else:
        expected_codec = "mp3" if kind == "mp3" else None
        if expected_codec and probe.codec != expected_codec:
            ev.errors.append(f"output codec is {probe.codec!r}, expected {expected_codec!r}")
        if probe.sample_rate != encode.sample_rate:
            ev.errors.append(f"output sample rate {probe.sample_rate} Hz, expected {encode.sample_rate} Hz")
        if probe.channels != encode.channels:
            ev.errors.append(f"output has {probe.channels} channel(s), expected {encode.channels}")
    if measurement is None:
        ev.errors.append("output loudness could not be measured")
        return ev
    if ev.errors:   # integrity problems: re-encoding with another gain would not help
        return ev

    # --- loudness / true peak ---
    integrated, tp = measurement.integrated_lufs, measurement.true_peak_dbtp
    tp_limit = cfg.target_true_peak_dbtp + cfg.true_peak_tolerance_db
    tp_aim = (cfg.target_true_peak_dbtp - CORRECTION_TP_MARGIN_DB) if kind == "mp3" else cfg.pcm_true_peak_ceiling
    deviation = integrated - cfg.target_lufs
    desired = -deviation               # gain that would hit the target exactly
    max_boost = tp_aim - tp            # negative => attenuation required for TP safety
    tp_over = tp > tp_limit
    lufs_off = abs(deviation) > cfg.lufs_tolerance

    correction = 0.0
    if tp_over:
        correction = min(desired, max_boost)
    elif lufs_off:
        correction = desired if desired < 0 else max(0.0, min(desired, max_boost))
    if abs(correction) >= cfg.min_correction_db:
        ev.correction_db = round(correction, 2)

    if (measurement.ebur128_integrated_lufs is not None
            and abs(measurement.ebur128_integrated_lufs - integrated) > cfg.meter_disagreement_lu):
        ev.warnings.append(
            f"meters disagree: loudnorm {integrated:.2f} LUFS vs ebur128 {measurement.ebur128_integrated_lufs:.2f} LUFS")

    if tp_over:
        ev.errors.append(f"true peak {tp:+.2f} dBTP exceeds the limit {tp_limit:+.2f} dBTP")
        return ev
    if lufs_off:
        peak_bound = deviation < 0 and (
            (plan is not None and plan.mode is LoudnormMode.PEAK_LIMITED_LINEAR)
            or max_boost < cfg.min_correction_db)
        if peak_bound:
            ev.review_reasons.append(
                f"integrated loudness {integrated:.1f} LUFS is {abs(deviation):.1f} LU below the "
                f"{cfg.target_lufs:.1f} LUFS target: not reachable within the "
                f"{cfg.target_true_peak_dbtp:.1f} dBTP true-peak limit without heavy limiting; "
                f"delivered peak-safe — needs listening review")
            ev.status = NormalizationStatus.REVIEW
        else:
            ev.errors.append(
                f"integrated loudness {integrated:.2f} LUFS is outside {cfg.target_lufs:.1f} ± "
                f"{cfg.lufs_tolerance:.1f} LUFS")
        return ev
    ev.status = NormalizationStatus.PASSED
    return ev


def compare_adjacent_short_term(prev: ShortTermStats | None, nxt: ShortTermStats | None,
                                qc: QCConfig) -> list[str]:
    """Reasons to listen to a transition: big short-term loudness differences between
    adjacent (normalized) tracks. Detection only — nothing is compressed to 'fix' it."""
    if prev is None or nxt is None:
        return []
    reasons = []
    if prev.p95 is not None and nxt.p95 is not None:
        delta = nxt.p95 - prev.p95
        if abs(delta) > qc.adjacent_body_delta_lu:
            reasons.append(f"short-term loudness (p95) changes {delta:+.1f} LU between adjacent tracks")
    if prev.tail_lufs is not None and nxt.head_lufs is not None:
        delta = nxt.head_lufs - prev.tail_lufs
        if abs(delta) > qc.adjacent_edge_delta_lu:
            reasons.append(f"loudness jump at the boundary: outgoing tail {prev.tail_lufs:.1f} → "
                           f"incoming head {nxt.head_lufs:.1f} LUFS ({delta:+.1f} LU)")
    return reasons


# ---------------------------------------------------------------------------
# Normalizer service
# ---------------------------------------------------------------------------

class LoudnessNormalizer:
    """Two-pass EBU R128 normalizer with verification of the encoded output."""

    def __init__(self, tools: FFmpegTools, cfg: LoudnessConfig | None = None,
                 encode: EncodeConfig | None = None, qc: QCConfig | None = None,
                 *, work_dir: Path | None = None) -> None:
        self.tools = tools
        self.cfg = cfg or LoudnessConfig()
        self.encode = encode or EncodeConfig()
        self.qc = qc or QCConfig()
        self.cfg.validate()
        self.encode.validate()
        self.work_dir = Path(work_dir) if work_dir else None

    # --- filters ---------------------------------------------------------

    @staticmethod
    def prep_filter(channels: int | None) -> str | None:
        """Bring any input to stereo BEFORE metering, so pass 1, pass 2 and the
        verification all measure the same signal layout."""
        if channels == 1:
            return "pan=stereo|c0=c0|c1=c0"
        if channels is not None and channels > 2:
            return "aformat=channel_layouts=stereo"
        return None

    def _resample_filter(self) -> str:
        sr = self.encode.sample_rate
        if self.encode.prefer_soxr and self.tools.has_soxr:
            return f"aresample={sr}:resampler=soxr:precision=28"
        return f"aresample={sr}"

    def _loudnorm_pass2(self, plan: NormalizationPlan, m: LoudnessMeasurement) -> str:
        linear = plan.mode is not LoudnormMode.DYNAMIC
        measured_tp = m.loudnorm_true_peak_dbtp if m.loudnorm_true_peak_dbtp is not None else m.true_peak_dbtp
        opts = [
            f"I={plan.effective_target_lufs:.2f}",
            f"TP={plan.true_peak_ceiling_dbtp:.2f}",
            f"LRA={plan.lra_target:.2f}",
            f"measured_I={m.integrated_lufs:.2f}",
            f"measured_TP={measured_tp:.2f}",
            f"measured_LRA={plan.measured_lra_for_filter:.2f}",
            f"measured_thresh={m.threshold_lufs:.2f}",
        ]
        if not linear and m.target_offset_lu is not None:
            opts.append(f"offset={m.target_offset_lu:.2f}")
        opts += [f"linear={'true' if linear else 'false'}", "print_format=json"]
        return "loudnorm=" + ":".join(opts)

    def _pass2_chain(self, plan: NormalizationPlan, m: LoudnessMeasurement, channels: int | None) -> list[str]:
        chain = [self.prep_filter(channels), self._loudnorm_pass2(plan, m), self._resample_filter()]
        return [f for f in chain if f]

    # --- measurement / verification --------------------------------------

    def measure(self, path: Path, *, channels: int | None = None) -> LoudnessMeasurement:
        if channels is None:
            channels = probe_audio(self.tools, Path(path)).channels
        return measure_loudness(
            self.tools, Path(path), prep_filter=self.prep_filter(channels),
            target_lufs=self.cfg.target_lufs, target_tp=self.cfg.pcm_true_peak_ceiling,
            target_lra=self.cfg.target_lra, edge_window_s=self.qc.edge_window_s)

    def plan(self, m: LoudnessMeasurement) -> NormalizationPlan:
        return plan_normalization(m, self.cfg)

    def verify_output(self, path: Path, *, plan: NormalizationPlan | None = None,
                      expected_duration_s: float | None = None, kind: str = "mp3",
                      ) -> tuple[AudioProbe | None, DecodeCheck | None, LoudnessMeasurement | None, OutputEvaluation]:
        """Probe + full decode + loudness/true-peak measurement of an existing output file."""
        errors: list[str] = []
        probe = decode = measurement = None
        try:
            probe = probe_audio(self.tools, Path(path))
        except FFmpegError as e:
            errors.append(f"probe failed: {_first_line(e)}")
        try:
            decode = decode_check(self.tools, Path(path))
        except FFmpegError as e:
            errors.append(f"decode check failed: {_first_line(e)}")
        try:
            measurement = self.measure(Path(path), channels=probe.channels if probe else 2)
        except (FFmpegError, LoudnessMeasurementError) as e:
            errors.append(f"loudness measurement of the output failed: {_first_line(e)}")
        ev = evaluate_output(plan=plan, measurement=measurement, probe=probe, decode=decode,
                             expected_duration_s=expected_duration_s, cfg=self.cfg,
                             encode=self.encode, qc=self.qc, kind=kind, extra_errors=errors)
        return probe, decode, measurement, ev

    # --- processing ------------------------------------------------------

    def normalize_to_mp3(self, source: Path, dest: Path, *,
                         input_measurement: LoudnessMeasurement | None = None,
                         source_probe: AudioProbe | None = None,
                         metadata: dict[str, str] | None = None) -> NormalizationResult:
        """Normalize ``source`` into a verified MP3 at ``dest`` (atomic replace)."""
        return self._process(Path(source), Path(dest), "mp3", input_measurement, source_probe, metadata)

    def normalize_to_wav(self, source: Path, dest: Path, *,
                         input_measurement: LoudnessMeasurement | None = None,
                         source_probe: AudioProbe | None = None) -> NormalizationResult:
        """Normalize ``source`` into a verified float WAV (lossless; used by the mix renderer)."""
        return self._process(Path(source), Path(dest), "wav", input_measurement, source_probe, None)

    def _process(self, source: Path, dest: Path, kind: str,
                 input_measurement: LoudnessMeasurement | None,
                 source_probe: AudioProbe | None,
                 metadata: dict[str, str] | None) -> NormalizationResult:
        source, dest = source.resolve(), dest.resolve()
        result = NormalizationResult(
            source=source, output=None, status=NormalizationStatus.FAILED, requires_review=True,
            target_lufs=self.cfg.target_lufs, target_true_peak_dbtp=self.cfg.target_true_peak_dbtp)
        tmp_dir: Path | None = None
        tmp_out: Path | None = None
        try:
            _check_paths(source, dest, kind)
            if kind == "mp3" and not self.tools.has_libmp3lame:
                raise FFmpegError("this ffmpeg build has no libmp3lame encoder")
            probe = source_probe or probe_audio(self.tools, source)
            if not probe.channels:
                raise NormalizationPlanError("source has no usable audio channels")
            m_in = input_measurement or self.measure(source, channels=probe.channels)
            result.input_measurement = m_in
            result.expected_duration_s = m_in.duration_s or probe.duration_s
            plan = plan_normalization(m_in, self.cfg)
            result.plan = plan
            result.warnings.extend(plan.reasons)
            if plan.mode is LoudnormMode.DYNAMIC and self.cfg.review_dynamic:
                result.review_reasons.append(
                    f"{DYNAMIC_MODE_DESCRIPTION} was applied: the track's macro-dynamics may have "
                    "changed — needs listening review")

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(tempfile.mkdtemp(prefix="veg_norm_", dir=self.work_dir))
            inline = kind == "mp3" and source.suffix.lower() in LOSSLESS_EXTENSIONS
            pcm: Path | None = None
            if not inline:
                # pass 2 -> lossless intermediate (float WAV; RF64 if > 4 GB)
                pcm = tmp_dir / "pass2.wav"
                p = run_ffmpeg(self.tools, [
                    "-nostats", "-y", "-i", str(source), "-map", "0:a:0", "-vn", "-map_metadata", "-1",
                    "-af", ",".join(self._pass2_chain(plan, m_in, probe.channels)),
                    "-c:a", "pcm_f32le", "-ac", str(self.encode.channels), "-rf64", "auto", str(pcm)])
                self._record_loudnorm_mode(p.stderr, plan, result)

            correction = 0.0
            ev: OutputEvaluation | None = None
            for attempt in range(1, self.cfg.max_encode_attempts + 1):
                tmp_out = dest.with_name(f".{dest.stem}.veg-{uuid.uuid4().hex[:8]}{dest.suffix}")
                if kind == "mp3":
                    p = run_ffmpeg(self.tools, self._encode_args(
                        source=source, audio_in=source if inline else pcm, inline=inline, plan=plan,
                        m_in=m_in, probe=probe, correction=correction, metadata=metadata, out=tmp_out))
                    if inline:
                        self._record_loudnorm_mode(p.stderr, plan, result)
                elif correction == 0.0:
                    shutil.copyfile(pcm, tmp_out)
                else:
                    run_ffmpeg(self.tools, ["-nostats", "-y", "-i", str(pcm), "-af", f"volume={correction:.2f}dB",
                                            "-c:a", "pcm_f32le", "-rf64", "auto", str(tmp_out)])
                result.encode_attempts = attempt
                probe_out, decode, m_out, ev = self.verify_output(
                    tmp_out, plan=plan, expected_duration_s=result.expected_duration_s, kind=kind)
                result.output_probe, result.decode_check, result.output_measurement = probe_out, decode, m_out
                if ev.correction_db and attempt < self.cfg.max_encode_attempts:
                    log.info("%s: attempt %d off target, correcting %+.2f dB", source.name, attempt, ev.correction_db)
                    correction += ev.correction_db
                    _unlink(tmp_out)
                    tmp_out = None
                    continue
                break

            assert ev is not None
            result.applied_correction_db = correction
            result.errors.extend(ev.errors)
            result.warnings.extend(ev.warnings)
            result.review_reasons.extend(ev.review_reasons)
            if correction:
                result.warnings.append(f"closed-loop correction of {correction:+.2f} dB applied after measuring the output")
            result.status = ev.status
            if result.status is not NormalizationStatus.FAILED:
                os.replace(tmp_out, dest)
                tmp_out = None
                result.output = dest
        except (FFmpegError, LoudnessMeasurementError, NormalizationPlanError, OSError, ValueError) as e:
            result.errors.append(f"{type(e).__name__}: {_first_line(e)}")
            result.status = NormalizationStatus.FAILED
        finally:
            if tmp_out is not None:
                _unlink(tmp_out)
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        result.requires_review = result.status is not NormalizationStatus.PASSED or bool(result.review_reasons)
        return result

    def _record_loudnorm_mode(self, stderr: str, plan: NormalizationPlan, result: NormalizationResult) -> None:
        try:
            reported = parse_loudnorm_json(stderr).get("normalization_type")
        except LoudnessMeasurementError as e:
            reported = None
            result.warnings.append(f"loudnorm pass-2 statistics unavailable: {e}")
        result.loudnorm_reported_mode = reported
        if reported is None:
            msg = "loudnorm did not report its processing mode"
        elif plan.mode is not LoudnormMode.DYNAMIC and reported == "dynamic":
            msg = (f"loudnorm fell back to {DYNAMIC_MODE_DESCRIPTION} although linear "
                   "normalization was planned")
        else:
            return
        if msg not in result.review_reasons:
            result.warnings.append(msg)
            result.review_reasons.append(msg)

    def _encode_args(self, *, source: Path, audio_in: Path, inline: bool, plan: NormalizationPlan,
                     m_in: LoudnessMeasurement, probe: AudioProbe, correction: float,
                     metadata: dict[str, str] | None, out: Path) -> list[str]:
        enc = self.encode
        args = ["-nostats", "-y", "-i", str(audio_in)]
        tag_input = 0
        if not inline and (enc.copy_tags or enc.copy_cover_art):
            args += ["-i", str(source)]
            tag_input = 1
        args += ["-map", "0:a:0"]
        if enc.copy_cover_art and probe.has_cover_art:
            args += ["-map", f"{tag_input}:v:0?", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
        args += ["-map_metadata", str(tag_input) if enc.copy_tags else "-1"]
        filters = self._pass2_chain(plan, m_in, probe.channels) if inline else []
        if correction:
            filters.append(f"volume={correction:.2f}dB")
        if filters:
            args += ["-af", ",".join(filters)]
        args += ["-c:a", enc.codec, "-b:a", enc.bitrate, "-ar", str(enc.sample_rate),
                 "-ac", str(enc.channels), "-id3v2_version", str(enc.id3v2_version)]
        for tag in STALE_LOUDNESS_TAGS:
            args += ["-metadata", f"{tag}="]
        for key, value in (metadata or {}).items():
            if value is not None:
                args += ["-metadata", f"{key}={value}"]
        args += ["-f", "mp3", str(out)]
        return args


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check_paths(source: Path, dest: Path, kind: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    if dest == source or (dest.exists() and os.path.samefile(dest, source)):
        raise ValueError(f"refusing to overwrite the source file {source}")
    expected = ".mp3" if kind == "mp3" else ".wav"
    if dest.suffix.lower() != expected:
        raise ValueError(f"output must be a {expected} file, got {dest.name}")


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("could not remove temporary file %s", path)


def _first_line(e: BaseException) -> str:
    text = str(e).strip()
    return text.splitlines()[0] if text else type(e).__name__
