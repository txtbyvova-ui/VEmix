"""Engine configuration (tuning knobs, not runtime state).

All dataclasses are frozen; derive variants with ``dataclasses.replace``.
Editorial / client-specific rules live in the client profile (see ``curation.py``),
not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# ---------------------------------------------------------------------------
# Loudness defaults (spec §3)
# ---------------------------------------------------------------------------

TARGET_LUFS = -16.0
TARGET_TRUE_PEAK_DBTP = -1.5
TARGET_LRA = 11.0

LUFS_TOLERANCE = 1.0
TRUE_PEAK_TOLERANCE_DB = 0.2

# loudnorm option ranges (ffmpeg af_loudnorm)
LOUDNORM_I_RANGE = (-70.0, -5.0)
LOUDNORM_TP_RANGE = (-9.0, 0.0)
LOUDNORM_LRA_RANGE = (1.0, 50.0)

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif")
LOSSLESS_EXTENSIONS = (".wav", ".flac", ".aiff", ".aif", ".w64", ".rf64")


class ConfigError(ValueError):
    """Invalid configuration value."""


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise ConfigError(f"{name}={value} is outside the allowed range [{lo}, {hi}]")


@dataclass(frozen=True)
class LoudnessConfig:
    target_lufs: float = TARGET_LUFS
    target_true_peak_dbtp: float = TARGET_TRUE_PEAK_DBTP
    # LRA is a processing target for loudnorm, not a guarantee for every track.
    target_lra: float = TARGET_LRA
    lufs_tolerance: float = LUFS_TOLERANCE
    true_peak_tolerance_db: float = TRUE_PEAK_TOLERANCE_DB

    # Peak headroom reserved for lossy encoding: MP3 decoding can overshoot the PCM
    # true peak, so the PCM stage aims below the ceiling.
    encode_headroom_db: float = 0.5
    # How much gain beyond the peak-safe linear gain may be obtained with loudnorm's
    # dynamic mode. That mode is NOT a transparent limiter: it rides gain over the whole
    # track (AGC) and limits true peaks, so it is OFF by default (0). Shortfalls inside
    # the LUFS tolerance are delivered linearly (peak-safe gain); larger ones are
    # delivered peak-safe (quieter) and flagged for review. Opt in with e.g. 2.0.
    max_dynamic_gain_db: float = 0.0
    # Mark tracks that needed dynamic processing as requires_review (always on by default:
    # dynamic processing changes the master's macro-dynamics).
    review_dynamic: bool = True
    # Never squeeze a dynamic master to the LRA target: the LRA passed to loudnorm is
    # max(target_lra, measured_lra), so loudnorm stays linear instead of compressing.
    preserve_source_lra: bool = True
    # Inputs quieter than this are treated as silent / broken.
    min_input_lufs: float = -60.0
    # Closed-loop corrections from the measured encoded output (each re-encodes from the
    # lossless intermediate, never from a previous MP3).
    max_encode_attempts: int = 3
    # Corrections smaller than this are not worth a re-encode.
    min_correction_db: float = 0.1
    # Disagreement between loudnorm and ebur128 integrated readings that triggers a warning.
    meter_disagreement_lu: float = 0.5

    def validate(self) -> None:
        _check_range("target_lufs", self.target_lufs, *LOUDNORM_I_RANGE)
        _check_range("target_true_peak_dbtp", self.target_true_peak_dbtp, *LOUDNORM_TP_RANGE)
        _check_range("target_lra", self.target_lra, *LOUDNORM_LRA_RANGE)
        _check_range("lufs_tolerance", self.lufs_tolerance, 0.0, 6.0)
        _check_range("true_peak_tolerance_db", self.true_peak_tolerance_db, 0.0, 3.0)
        _check_range("encode_headroom_db", self.encode_headroom_db, 0.0, 3.0)
        _check_range("max_dynamic_gain_db", self.max_dynamic_gain_db, 0.0, 12.0)
        _check_range("max_encode_attempts", self.max_encode_attempts, 1, 10)
        if self.target_true_peak_dbtp - self.encode_headroom_db < LOUDNORM_TP_RANGE[0]:
            raise ConfigError("target_true_peak_dbtp - encode_headroom_db is below loudnorm's -9 dBTP limit")

    @property
    def pcm_true_peak_ceiling(self) -> float:
        """True-peak ceiling used for the lossless (pre-encode) stage."""
        return self.target_true_peak_dbtp - self.encode_headroom_db


@dataclass(frozen=True)
class EncodeConfig:
    codec: str = "libmp3lame"
    bitrate: str = "320k"
    sample_rate: int = 44100
    channels: int = 2
    id3v2_version: int = 3
    copy_tags: bool = True
    copy_cover_art: bool = True
    # Use libsoxr for the post-loudnorm resample when the ffmpeg build has it.
    prefer_soxr: bool = True

    def validate(self) -> None:
        if self.channels != 2:
            raise ConfigError("only stereo output is supported (channels=2)")
        if self.sample_rate not in (32000, 44100, 48000):
            raise ConfigError(f"unsupported MP3 sample rate {self.sample_rate}")
        if not self.bitrate.rstrip("kK").isdigit():
            raise ConfigError(f"bitrate must look like '320k', got {self.bitrate!r}")


@dataclass(frozen=True)
class AnalysisConfig:
    """Audio-analysis parameters. Changing any of these invalidates only the
    analysis section that depends on it (see ``cache.py``)."""

    sample_rate: int = 22050  # LIBROSA_SR in automix v1
    # BPM window (automix v1 P1.4 + post-review hardening)
    bpm_offset_s: float = 30.0
    bpm_duration_s: float = 90.0
    bpm_min_window_s: float = 30.0
    # Key window — kept conservative ON PURPOSE (a past change of detected keys
    # produced very bad mixes). Do not widen / align with BPM.
    key_offset_s: float = 30.0
    key_duration_s: float = 60.0
    key_min_window_s: float = 1.0
    # Heuristic energy estimate (explicitly low confidence, never presented as measured).
    estimate_energy: bool = True
    energy_offset_s: float = 30.0
    energy_duration_s: float = 60.0
    energy_confidence: float = 0.4


@dataclass(frozen=True)
class SequencingConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "vibe_similarity": 0.35,
        "energy_continuity": 0.30,
        "bpm_continuity": 0.15,
        "harmonic_compatibility": 0.10,
        "genre_continuity": 0.10,
    })
    beam_width: int = 8
    # Energy deltas on the 0..1 scale.
    flag_energy_jump: float = 0.25      # flagged for review above this
    max_energy_jump: float = 0.40       # avoided whenever an alternative exists
    # BPM difference (after half/double-time folding) at which continuity reaches 0.
    bpm_span: float = 16.0
    # Minimum number of other tracks between two tracks of the same artist.
    artist_min_gap: int = 2
    # "Excessively similar" run control.
    similar_threshold: float = 0.92
    max_similar_run: int = 4
    # Seeded jitter amplitude for controlled variation (0 = fully deterministic greedy/beam).
    variation: float = 0.04
    # Neutral value used when a component cannot be computed (unknown features).
    neutral_score: float = 0.5


@dataclass(frozen=True)
class TransitionConfig:
    enabled: bool = True
    default_crossfade_s: float = 6.0
    min_crossfade_s: float = 1.0
    max_overlap_s: float = 10.0
    # Crossfade may not exceed this fraction of either adjacent (trimmed) track.
    max_fraction_of_track: float = 0.25
    # Tracks shorter than this get no crossfade (butt splice with de-click fades).
    min_track_s_for_crossfade: float = 20.0
    curve: str = "equal_power"  # linear | equal_power | s_curve
    # clean | filter_sweep | echo_tail | filter+echo | random (random is opt-in only)
    fx_mode: str = "clean"
    fx_weights: dict[str, float] = field(default_factory=lambda: {
        "clean": 0.25, "filter_sweep": 0.35, "echo_tail": 0.25, "filter+echo": 0.15,
    })
    mix_fade_in_s: float = 3.0
    mix_fade_out_s: float = 5.0
    declick_ms: float = 10.0
    trim_silence: bool = True
    trim_silence_thresh_dbfs: float = -40.0
    trim_min_silence_ms: float = 200.0
    # Tempo mismatch (relative, after half/double folding) above which a long overlap
    # risks audible beat clashes without phrase/beat alignment.
    uncertain_tempo_ratio: float = 0.06
    conservative_crossfade_s: float = 3.0
    # Oversampling factor for true-peak estimation of overlaps.
    true_peak_oversample: int = 4
    seed: int = 0


@dataclass(frozen=True)
class QCConfig:
    # Adjacent-track short-term loudness comparison (EBU R128 S, 3 s window).
    adjacent_body_delta_lu: float = 4.0     # difference of p95 short-term loudness
    adjacent_edge_delta_lu: float = 8.0     # outgoing tail vs incoming head
    edge_window_s: float = 10.0
    # Silence scanning (continuous mix).
    silence_window_s: float = 0.5
    silence_threshold_dbfs: float = -50.0
    silence_min_duration_s: float = 1.5
    digital_silence_dbfs: float = -90.0
    # Gaps within this distance of a transition boundary are "suspicious transition gaps".
    transition_guard_s: float = 4.0
    # Decoded-vs-expected duration tolerance (MP3 priming/padding).
    duration_tolerance_s: float = 0.25
    # Mix duration tolerance vs tracklist accounting.
    mix_duration_tolerance_s: float = 1.0
    clip_sample_threshold: float = 0.999


@dataclass(frozen=True)
class AppConfig:
    loudness: LoudnessConfig = field(default_factory=LoudnessConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    sequencing: SequencingConfig = field(default_factory=SequencingConfig)
    transitions: TransitionConfig = field(default_factory=TransitionConfig)
    qc: QCConfig = field(default_factory=QCConfig)

    def validate(self) -> None:
        self.loudness.validate()
        self.encode.validate()


# ---------------------------------------------------------------------------
# Targets recorded in a manifest (qc re-checks a delivery against them)
# ---------------------------------------------------------------------------

MANIFEST_LOUDNESS_FIELDS = {
    "target_lufs": "target_lufs", "target_true_peak_dbtp": "target_true_peak_dbtp", "target_lra": "target_lra",
    "lufs_tolerance": "lufs_tolerance", "true_peak_tolerance_db": "true_peak_tolerance_db",
}
MANIFEST_ENCODE_FIELDS = {"bitrate": "bitrate", "sample_rate": "sample_rate", "channels": "channels"}


def apply_recorded_targets(targets: object, loudness: LoudnessConfig, encode: EncodeConfig, *,
                           skip: frozenset[str] = frozenset()
                           ) -> tuple[LoudnessConfig, EncodeConfig, list[str]]:
    """Apply a manifest's ``loudness_targets`` field by field. A missing field keeps the
    given config; an invalid one (wrong type, boolean, out of range) is NOT applied and is
    reported in the returned problem list — one bad field never discards the valid ones.
    ``skip`` holds "loudness.<field>" / "encode.<field>" names set explicitly elsewhere."""
    if not isinstance(targets, dict):
        return loudness, encode, [f"loudness_targets must be an object, got {type(targets).__name__}"]
    problems: list[str] = []
    for key, fld in MANIFEST_LOUDNESS_FIELDS.items():
        raw = targets.get(key)
        if raw is None or f"loudness.{fld}" in skip:
            continue
        try:
            if isinstance(raw, bool):
                raise ConfigError("a boolean is not a number")
            candidate = replace(loudness, **{fld: float(raw)})
            candidate.validate()
            loudness = candidate
        except (TypeError, ValueError) as e:
            problems.append(f"{key}={raw!r}: {e}")
    for key, fld in MANIFEST_ENCODE_FIELDS.items():
        raw = targets.get(key)
        if raw is None or f"encode.{fld}" in skip:
            continue
        try:
            if isinstance(raw, bool):
                raise ConfigError("a boolean is not a valid value")
            if fld == "bitrate":
                if not isinstance(raw, str):
                    raise ConfigError("bitrate must be a string such as '320k'")
                value: object = raw
            else:
                if isinstance(raw, float) and not raw.is_integer():
                    raise ConfigError("must be an integer")
                value = int(raw)
            candidate_enc = replace(encode, **{fld: value})
            candidate_enc.validate()
            encode = candidate_enc
        except (TypeError, ValueError) as e:
            problems.append(f"{key}={raw!r}: {e}")
    return loudness, encode, problems
