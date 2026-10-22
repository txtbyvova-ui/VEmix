"""Shared data contracts.

Everything that crosses a module boundary (analysis → curation → sequencing →
playlist/renderer → QC → reports) is defined here, together with JSON
(de)serialization helpers. Unknown values are ``None`` — never a fabricated default.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / enums / paths / tuples into JSON-safe values.
    Non-finite floats become ``None`` (JSON has no NaN/inf)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _opt_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def make_track_id(rel_path: str) -> str:
    """Stable id derived from the library-relative path (POSIX separators, Unicode NFC).
    Case is preserved: 'Straße.mp3' and 'Strasse.mp3' (or case variants on a case-sensitive
    file system) are different files and must get different ids. Curator metadata is matched
    case-insensitively by path, not by this id."""
    norm = unicodedata.normalize("NFC", rel_path.replace("\\", "/"))
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Probing / loudness
# ---------------------------------------------------------------------------

@dataclass
class AudioProbe:
    path: Path
    duration_s: float | None
    sample_rate: int | None
    channels: int | None
    codec: str | None
    bit_rate: int | None = None
    tags: dict[str, str] = field(default_factory=dict)  # lower-cased keys
    has_cover_art: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "AudioProbe":
        return cls(
            path=Path(d["path"]),
            duration_s=_opt_float(d.get("duration_s")),
            sample_rate=d.get("sample_rate"),
            channels=d.get("channels"),
            codec=d.get("codec"),
            bit_rate=d.get("bit_rate"),
            tags=dict(d.get("tags") or {}),
            has_cover_art=bool(d.get("has_cover_art", False)),
        )


@dataclass
class DecodeCheck:
    ok: bool
    decoded_duration_s: float | None
    errors: list[str] = field(default_factory=list)


@dataclass
class ShortTermStats:
    """Distribution of EBU R128 short-term (3 s) loudness, gated at -70 LUFS."""
    p10: float | None
    p50: float | None
    p95: float | None
    max: float | None
    head_lufs: float | None   # median S over the first QC edge window
    tail_lufs: float | None   # median S over the last QC edge window
    frames: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "ShortTermStats | None":
        if not d:
            return None
        return cls(**{k: d.get(k) for k in ("p10", "p50", "p95", "max", "head_lufs", "tail_lufs")},
                   frames=int(d.get("frames", 0)))


@dataclass
class LoudnessMeasurement:
    integrated_lufs: float
    true_peak_dbtp: float          # conservative: max(loudnorm TP, ebur128 TP)
    lra_lu: float
    threshold_lufs: float
    target_offset_lu: float | None = None          # loudnorm pass-1 target_offset
    loudnorm_true_peak_dbtp: float | None = None
    ebur128_integrated_lufs: float | None = None
    ebur128_true_peak_dbtp: float | None = None
    short_term: ShortTermStats | None = None
    duration_s: float | None = None                # decoded duration seen by the meter

    @classmethod
    def from_dict(cls, d: dict | None) -> "LoudnessMeasurement | None":
        if not d:
            return None
        return cls(
            integrated_lufs=float(d["integrated_lufs"]),
            true_peak_dbtp=float(d["true_peak_dbtp"]),
            lra_lu=float(d["lra_lu"]),
            threshold_lufs=float(d["threshold_lufs"]),
            target_offset_lu=_opt_float(d.get("target_offset_lu")),
            loudnorm_true_peak_dbtp=_opt_float(d.get("loudnorm_true_peak_dbtp")),
            ebur128_integrated_lufs=_opt_float(d.get("ebur128_integrated_lufs")),
            ebur128_true_peak_dbtp=_opt_float(d.get("ebur128_true_peak_dbtp")),
            short_term=ShortTermStats.from_dict(d.get("short_term")),
            duration_s=_opt_float(d.get("duration_s")),
        )


class NormalizationStatus(str, enum.Enum):
    PASSED = "passed"    # verified on the encoded output, within tolerances
    REVIEW = "review"    # peak-safe output delivered, but target not met → human review
    FAILED = "failed"    # verification failed or processing error; do not ship


class LoudnormMode(str, enum.Enum):
    LINEAR = "linear"                          # pure gain, target reachable
    DYNAMIC = "dynamic"                        # loudnorm dynamic mode (AGC + TP limiting); opt-in, review
    PEAK_LIMITED_LINEAR = "peak_limited_linear"  # pure gain capped by TP; target NOT reached


@dataclass
class NormalizationPlan:
    mode: LoudnormMode
    target_lufs: float             # configured target
    effective_target_lufs: float   # I passed to loudnorm
    true_peak_ceiling_dbtp: float  # TP passed to loudnorm (target minus encode headroom)
    lra_target: float              # LRA passed to loudnorm
    gain_needed_db: float          # target - measured integrated
    predicted_true_peak_dbtp: float
    peak_safe_gain_db: float       # max linear gain keeping TP under the ceiling
    shortfall_db: float            # gain that linear processing cannot deliver (>= 0)
    measured_lra_for_filter: float  # LRA value passed as measured_LRA (see loudness.py)
    reasons: list[str] = field(default_factory=list)


@dataclass
class NormalizationResult:
    source: Path
    output: Path | None
    status: NormalizationStatus
    requires_review: bool
    target_lufs: float
    target_true_peak_dbtp: float
    input_measurement: LoudnessMeasurement | None = None
    output_measurement: LoudnessMeasurement | None = None
    plan: NormalizationPlan | None = None
    loudnorm_reported_mode: str | None = None   # normalization_type from pass-2 JSON
    applied_correction_db: float = 0.0          # closed-loop gain applied after pass 2
    encode_attempts: int = 0
    output_probe: AudioProbe | None = None
    decode_check: DecodeCheck | None = None
    expected_duration_s: float | None = None
    warnings: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def measured_lufs(self) -> float | None:
        return self.output_measurement.integrated_lufs if self.output_measurement else None

    @property
    def true_peak_dbtp(self) -> float | None:
        return self.output_measurement.true_peak_dbtp if self.output_measurement else None

    def to_metadata(self) -> dict:
        """Spec §3 'Required output metadata' plus provenance details."""
        inp, out = self.input_measurement, self.output_measurement
        return {
            "target_lufs": self.target_lufs,
            "target_true_peak_dbtp": self.target_true_peak_dbtp,
            "measured_lufs": out.integrated_lufs if out else None,
            "true_peak_dbtp": out.true_peak_dbtp if out else None,
            "original_lufs": inp.integrated_lufs if inp else None,
            "original_true_peak_dbtp": inp.true_peak_dbtp if inp else None,
            "original_lra_lu": inp.lra_lu if inp else None,
            "final_lra_lu": out.lra_lu if out else None,
            "normalization_status": self.status.value,
            "requires_review": self.requires_review,
            "processing_mode": self.plan.mode.value if self.plan else None,
            "loudnorm_reported_mode": self.loudnorm_reported_mode,
            "applied_correction_db": round(self.applied_correction_db, 2),
            "encode_attempts": self.encode_attempts,
            "duration_s": (self.decode_check.decoded_duration_s if self.decode_check else None),
            "sample_rate": self.output_probe.sample_rate if self.output_probe else None,
            "channels": self.output_probe.channels if self.output_probe else None,
            "warnings": list(self.warnings),
            "review_reasons": list(self.review_reasons),
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Track features (spec §5)
# ---------------------------------------------------------------------------

# Fields a curator may override via CSV/JSON (see metadata.py).
EDITABLE_FIELDS = (
    "artist", "title", "bpm", "camelot", "key_confidence", "energy", "danceability",
    "vocal_presence", "genre", "mood", "vibe_tags", "language", "familiarity",
    "approved", "excluded", "requires_review", "notes",
)


@dataclass
class TrackFeatures:
    path: Path
    duration_s: float

    bpm: float | None = None
    camelot: str | None = None
    key_confidence: float | None = None

    energy: float | None = None           # 0..1
    danceability: float | None = None     # 0..1
    vocal_presence: float | None = None   # 0..1

    genre: str | None = None
    mood: str | None = None
    vibe_tags: list[str] = field(default_factory=list)

    language: str | None = None   # ISO 639-1, lower case
    familiarity: float | None = None   # 0..1, curated only — never estimated

    approved: bool = False
    excluded: bool = False
    requires_review: bool = False

    # --- provenance / identity (beyond the spec's minimum) ---
    rel_path: str = ""                     # library-relative POSIX path (metadata key)
    track_id: str = ""
    artist: str | None = None
    title: str | None = None
    energy_confidence: float | None = None
    # field name -> "manual" | "id3" | "filename" | "auto:<algo-version>"
    feature_sources: dict[str, str] = field(default_factory=dict)
    review_reasons: list[str] = field(default_factory=list)
    exclusion_reasons: list[str] = field(default_factory=list)
    script_hint: str | None = None         # e.g. "cyrillic" — a hint, NOT a language claim
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.rel_path:
            self.rel_path = self.path.name
        if not self.track_id:
            self.track_id = make_track_id(self.rel_path)

    @property
    def display_name(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} - {self.title}"
        return self.title or self.path.stem

    def confidence(self, name: str) -> float | None:
        """Confidence of a feature: manual metadata = 1.0, otherwise per-field."""
        if getattr(self, name, None) in (None, [], ""):
            return None
        if self.feature_sources.get(name) == "manual":
            return 1.0
        if name == "camelot":
            return self.key_confidence
        if name == "energy":
            return self.energy_confidence
        source = self.feature_sources.get(name, "")
        if source == "id3":
            return 0.6
        if source.startswith("auto:"):
            return 0.5
        return 0.8

    def to_dict(self) -> dict:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrackFeatures":
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["path"] = Path(kwargs["path"])
        kwargs["duration_s"] = float(kwargs["duration_s"])
        for k in ("bpm", "key_confidence", "energy", "danceability", "vocal_presence",
                  "familiarity", "energy_confidence"):
            if k in kwargs:
                kwargs[k] = _opt_float(kwargs[k])
        for k in ("vibe_tags", "review_reasons", "exclusion_reasons"):
            kwargs[k] = list(kwargs.get(k) or [])
        kwargs["feature_sources"] = dict(kwargs.get("feature_sources") or {})
        return cls(**kwargs)


@dataclass
class AnalyzedTrack:
    """Result of analysing one library file (analysis.py)."""
    features: TrackFeatures
    probe: AudioProbe | None
    loudness: LoudnessMeasurement | None = None     # pass-1 measurement (cached)
    errors: list[str] = field(default_factory=list)  # non-empty => track is broken, report it
    from_cache: dict[str, bool] = field(default_factory=dict)   # section -> cache hit

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Client profile (spec §6, §7 dayparts) — loaded/applied in curation.py
# ---------------------------------------------------------------------------

class PreferenceLevel(str, enum.Enum):
    AVOID = "avoid"    # excluded
    LOW = "low"        # penalized in selection
    MEDIUM = "medium"  # neutral
    HIGH = "high"      # favoured in selection


class LevelSetting(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class LanguageRules:
    allowed: list[str] = field(default_factory=list)   # empty = any known language allowed
    excluded: list[str] = field(default_factory=list)
    unknown_policy: str = "review"          # review | exclude | allow
    cyrillic_script_policy: str = "review"  # exclude | review | ignore


@dataclass
class ListRules:
    tracks: list[str] = field(default_factory=list)    # rel paths, file names or "Artist - Title"
    artists: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    only: bool = False   # allowlist only: when true nothing outside the allowlist is eligible


@dataclass
class DaypartProfile:
    name: str
    energy_range: tuple[float, float] = (0.0, 1.0)
    energy_curve: tuple[float, float] | None = None   # target energy at (start, end)
    opening_tracks: int = 3
    opening_max_energy: float | None = None
    bpm_range: tuple[float, float] | None = None
    preferred_tags: list[str] = field(default_factory=list)
    avoid_tags: list[str] = field(default_factory=list)
    preferred_moods: list[str] = field(default_factory=list)
    preferred_genres: list[str] = field(default_factory=list)
    target_duration_s: float | None = None
    max_tracks: int | None = None


@dataclass
class ClientProfile:
    client: str
    language: LanguageRules = field(default_factory=LanguageRules)
    preferences: dict[str, PreferenceLevel] = field(default_factory=dict)
    # preference name -> tags / genres / moods that identify matching tracks
    preference_tags: dict[str, list[str]] = field(default_factory=dict)
    vibe_consistency: LevelSetting = LevelSetting.MEDIUM
    energy_jumps: LevelSetting = LevelSetting.MEDIUM   # tolerated size of energy jumps
    manual_review_required: bool = False
    require_approved: bool = False
    blocklist: ListRules = field(default_factory=ListRules)
    allowlist: ListRules = field(default_factory=ListRules)
    dayparts: dict[str, DaypartProfile] = field(default_factory=dict)
    source_path: Path | None = None


@dataclass
class CurationDecision:
    track: TrackFeatures
    eligible: bool
    exclusion_reasons: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)
    editorial_score: float = 0.0   # preference-based bonus/penalty (-1..+1)
    allowlisted: bool = False


@dataclass
class CurationResult:
    eligible: list[TrackFeatures]
    excluded: list[CurationDecision]
    decisions: list[CurationDecision]

    @property
    def unknown_metadata_count(self) -> int:
        fields_ = ("energy", "mood", "genre", "language")
        return sum(1 for d in self.decisions if any(getattr(d.track, f) in (None, "") for f in fields_))


# ---------------------------------------------------------------------------
# Sequencing (spec §7)
# ---------------------------------------------------------------------------

@dataclass
class TransitionScore:
    total: float                                   # 0..1, higher = better
    components: dict[str, float | None]            # 0..1 per component, None = unknown
    coverage: float                                # share of weight backed by known data
    flags: list[str] = field(default_factory=list)
    # Confidence-weighted mean of the KNOWN components (not shrunk towards neutral);
    # None when coverage is too low to judge. Used for "excessively similar run" limits.
    similarity: float | None = None


@dataclass
class SequencedTrack:
    position: int                    # 1-based
    track: TrackFeatures
    transition_in: TransitionScore | None = None
    daypart_fit: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class FlaggedTransition:
    from_position: int
    to_position: int
    from_track: str
    to_track: str
    reasons: list[str]
    score: float | None = None


@dataclass
class SequenceResult:
    daypart: str | None
    tracks: list[SequencedTrack]
    flagged: list[FlaggedTransition] = field(default_factory=list)
    not_selected: list[dict] = field(default_factory=list)   # {"rel_path", "reason"}
    total_duration_s: float = 0.0
    seed: int = 0
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Continuous mix (spec §8)
# ---------------------------------------------------------------------------

@dataclass
class MixInput:
    """One normalized lossless track handed to the renderer."""
    position: int
    pcm_path: Path            # normalized WAV (lossless intermediate)
    name: str
    source_rel: str
    bpm: float | None = None
    energy: float | None = None
    camelot: str | None = None


@dataclass
class TransitionPlan:
    position: int             # position of the INCOMING track (>= 2)
    crossfade_s: float
    curve: str
    fx: str
    reasons: list[str] = field(default_factory=list)
    uncertain: bool = False


@dataclass
class TransitionReport:
    position: int
    crossfade_s: float
    fx: str
    curve: str
    overlap_start_s: float
    overlap_peak_dbfs: float | None
    overlap_true_peak_dbtp: float | None
    gain_reduction_db: float = 0.0
    uncertain: bool = False
    flags: list[str] = field(default_factory=list)


@dataclass
class MixTrackEntry:
    position: int
    name: str
    source_rel: str
    start_s: float            # where the track becomes audible in the mix
    end_s: float
    duration_s: float         # trimmed duration of the track itself
    head_trim_s: float = 0.0
    tail_trim_s: float = 0.0
    crossfade_in_s: float = 0.0
    fx_in: str = "—"


@dataclass
class MixRenderResult:
    wav_path: Path
    sample_rate: int
    channels: int
    duration_s: float
    expected_duration_s: float
    tracks: list[MixTrackEntry]
    transitions: list[TransitionReport]
    peak_dbfs: float | None = None
    clipped_samples: int = 0
    seed: int = 0


# ---------------------------------------------------------------------------
# Manifests (playlist + mix). Schema documented in README.
# ---------------------------------------------------------------------------

@dataclass
class ManifestTrack:
    index: int                       # 1-based play order
    file: str                        # path relative to the manifest directory (POSIX)
    title: str
    artist: str | None
    duration_s: float | None
    source_rel: str
    source_path: str
    track_id: str
    loudness: dict = field(default_factory=dict)       # NormalizationResult.to_metadata()
    features: dict = field(default_factory=dict)       # curated + auto features (with sources)
    transition_in: dict | None = None                  # TransitionScore as dict
    flags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestTrack":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PlaylistManifest:
    generator: str
    generated_at: str
    mode: str                         # "playlist" | "mix"
    profile: str | None
    daypart: str | None
    loudness_targets: dict
    tracks: list[ManifestTrack]
    # {"source_rel", "reasons", "stage": "curation" | "analysis" | "normalization"}
    excluded: list[dict] = field(default_factory=list)
    flagged_transitions: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # eligible tracks the sequencer did not place here: {"source_rel", "track_id", "reason"}
    not_selected: list[dict] = field(default_factory=list)
    leftover_files: list[str] = field(default_factory=list)    # playlist: undeletable files of a previous run
    mix: dict | None = None           # mix mode: file, duration, transitions, timestamps
    schema: str = "vegroove.manifest"
    schema_version: int = SCHEMA_VERSION
    # VEmix 3.0: the dramaturgy plan this delivery was rendered from (scenario, variant, planned
    # order with phases/targets, anchors). Written only by `render`; absent from 2.0 outputs.
    plan: dict | None = None

    def to_dict(self) -> dict:
        d = to_jsonable(self)
        if d.get("plan") is None:
            d.pop("plan", None)        # 2.0 manifests stay byte-for-byte as before
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PlaylistManifest":
        if d.get("schema") != "vegroove.manifest":
            raise ValueError(f"not a VEgroove manifest (schema={d.get('schema')!r})")
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["tracks"] = [ManifestTrack.from_dict(t) for t in d.get("tracks", [])]
        return cls(**kwargs)


@dataclass
class PlaylistExportResult:
    output_dir: Path
    manifest_path: Path | None
    m3u8_path: Path | None
    tracklist_path: Path | None
    manifest: PlaylistManifest | None
    results: list[NormalizationResult] = field(default_factory=list)   # sequence order
    dry_run: bool = False

    @property
    def failed(self) -> list[NormalizationResult]:
        return [r for r in self.results if r.status is NormalizationStatus.FAILED]


# ---------------------------------------------------------------------------
# QC (spec §10/§11)
# ---------------------------------------------------------------------------

class Severity(str, enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    FAIL = "FAIL"


class OverallStatus(str, enum.Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class QCIssue:
    severity: Severity
    code: str                         # stable machine code, e.g. "true_peak_exceeded"
    message: str
    track_position: int | None = None
    timestamp_s: float | None = None
    file: str | None = None


@dataclass
class TrackQC:
    position: int
    file: str
    source_rel: str | None
    original_lufs: float | None
    final_lufs: float | None
    original_true_peak_dbtp: float | None
    final_true_peak_dbtp: float | None
    duration_s: float | None
    status: str                       # passed | review | failed
    warnings: list[str] = field(default_factory=list)


@dataclass
class QCReport:
    mode: str                         # playlist | mix | normalize
    generated_at: str
    status: OverallStatus
    target_lufs: float
    target_true_peak_dbtp: float
    tracks: list[TrackQC] = field(default_factory=list)
    issues: list[QCIssue] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    checks_skipped: list[str] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    flagged_transitions: list[dict] = field(default_factory=list)
    silence_warnings: list[dict] = field(default_factory=list)
    unknown_metadata: int = 0
    summary: dict = field(default_factory=dict)
    mix: dict | None = None

    def to_dict(self) -> dict:
        return to_jsonable(self)
