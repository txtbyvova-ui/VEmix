"""Playlist scenarios (VEmix 3.0, spec §6): the dramaturgy a plan must follow.

A scenario (YAML or JSON, examples in ``scenarios/``) is EDITORIAL input,
like the client profile: hard anchors, required/forbidden tracks and pairs, the target
energy curve as phases with relative durations, mood preferences, tempo/harmonic
tolerances, the target duration and diversity rules. Nothing about specific tracks is
built into the planner — every track name comes from here.

Validation follows the client profile loader (``curation``): unknown keys are errors
with a "did you mean" hint, duplicate keys are rejected, a YAML boolean (unquoted
yes/no/on/off) in a choice field (``curve_axis``) is explained; in text fields (names,
track references, mood / tag lists, description) it is rejected without that hint (e.g.
"expected non-empty text, got True"), and every message names the file and the key path.

Track references ("Artist - Title", a library-relative path or a file name) are matched
with the blocklist/allowlist rule (``curation.track_match_keys``); when nothing matches
exactly, a relaxed match ignores bracketed version qualifiers ("(Remastered 2008)") but
still requires the artist. An anchor or a required track that matches no track — or
more than one — is an error: a hard constraint is never silently dropped.
"""

from __future__ import annotations

import dataclasses
import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .curation import _did_you_mean, _Parser, load_config_document, reference_keys, track_match_keys
from .models import TrackFeatures
from .moods import is_known, mood_labels
from .textnorm import label_keys, normalize_artist, normalize_tag, primary_artist

log = logging.getLogger(__name__)


class ScenarioError(ValueError):
    """Invalid scenario, or a scenario that cannot be applied to the library."""


# Dramaturgical roles (spec §4/§5). Names are stable identifiers used in scenarios,
# interpretations and reports.
ROLES = ("opener", "tension_builder", "groove_builder", "release", "breather", "peak",
         "transition", "comedown", "closer")

# Objective weights (dramaturgy.evaluate). Every term is normalized to 0..1 before
# weighting, so the numbers say how much one unit of each kind of compromise costs.
DEFAULT_WEIGHTS: dict[str, float] = {
    "curve": 3.0,        # |track energy − target curve| / curve tolerance (confidence-weighted)
    "band": 2.0,         # energy outside the phase's energy_range
    "transition": 2.0,   # 1 − 2.0 transition score (mood/tags, energy, tempo, key, genre)
    "edge_energy": 0.5,  # outgoing tail vs incoming head energy (segment analysis)
    "tempo": 1.0,        # folded BPM difference relative to max_bpm_jump
    "harmony": 0.5,      # Camelot wheel distance beyond a compatible step
    "role": 1.0,         # phase roles vs the track's dramaturgical roles
    "mood": 1.5,         # phase / scenario mood preferences (explicit editorial requests)
    "timbre": 0.3,       # spectral / density contrast between neighbours (segment analysis)
    "mood_flow": 0.4,    # mood distance between neighbours (curator / AI / derived moods)
    "diversity": 0.5,    # same artist close together (beyond the hard minimum gap)
    "adjacency": 1.0,    # preferred / avoided neighbours from the scenario
    "duration": 2.0,     # distance from the target duration
    "editorial": 0.5,    # client-profile editorial score (curation)
    "uncertainty": 0.3,  # share of the plan resting on unknown or low-confidence data
}

TOP_LEVEL_KEYS = ("scenario", "target_duration_min", "duration_tolerance_min", "max_tracks", "min_tracks",
                  "use_all_tracks", "alternatives", "seed", "pool", "anchors", "required", "excluded",
                  "adjacency", "curve", "curve_axis", "phases", "moods", "transitions", "diversity", "weights")
POOL_KEYS = ("tracks", "include_tags", "exclude_tags")
ANCHOR_KEYS = ("track", "position", "phase")
ADJACENCY_KEYS = ("forbid", "avoid", "prefer")
PHASE_KEYS = ("name", "share", "energy", "energy_range", "roles", "moods", "contrast", "ai_positions",
              "description")
MOOD_KEYS = ("prefer", "avoid")
TRANSITION_KEYS = ("max_bpm_jump", "max_key_steps", "max_energy_jump", "contrast_energy_jump", "contrast_bpm_jump")
DIVERSITY_KEYS = ("artist_min_gap", "max_tracks_per_artist", "avoid_same_song")

_PHASE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,47}$")
MAX_ALTERNATIVES = 6
CURVE_AXES = ("time", "tracks")
# the position vocabulary of AI interpretations (veg_mixer.ai.schema.PHASES): a phase with one of
# these names uses the matching hints; any other phase names its equivalents in 'ai_positions'
AI_POSITIONS = ("intro", "build", "first_peak", "breather", "second_build", "climax", "descent", "closer")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ScenarioAnchor:
    track: str                      # reference as written
    position: int | None = None     # 1-based; negative counts from the end (-1 = last)
    phase: str | None = None        # alternatively: the track must play inside this phase


@dataclass
class ScenarioPhase:
    name: str
    share: float                                   # normalized relative duration (sums to 1)
    energy: tuple[float, float] | None = None      # target energy at the phase start / end
    energy_range: tuple[float, float] | None = None
    roles: list[str] = field(default_factory=list)
    prefer_moods: list[str] = field(default_factory=list)
    avoid_moods: list[str] = field(default_factory=list)
    contrast: bool = False                         # a controlled contrast is allowed on entry
    description: str | None = None
    # AI position hints that count for this phase (AI_POSITIONS); default: the phase's own name
    ai_positions: list[str] = field(default_factory=list)


@dataclass
class TransitionRules:
    max_bpm_jump: float | None = 16.0     # folded BPM difference (70 ≙ 140); None = no limit
    max_key_steps: int | None = None      # Camelot wheel steps (+1 for a mode change); None = no limit
    max_energy_jump: float = 0.30
    contrast_energy_jump: float = 0.50    # limit when entering a phase marked ``contrast``
    # tempo limit when entering a ``contrast`` phase; None = a planned contrast may change tempo freely
    contrast_bpm_jump: float | None = None


@dataclass
class DiversityRules:
    artist_min_gap: int = 2               # other tracks between two tracks of the same artist
    max_tracks_per_artist: int | None = None
    avoid_same_song: bool = True          # one version of a song per plan


@dataclass
class Scenario:
    name: str
    description: str | None = None
    target_duration_s: float | None = None
    duration_tolerance_s: float | None = None
    max_tracks: int | None = None
    min_tracks: int | None = None
    use_all_tracks: bool = False
    alternatives: int = 3
    seed: int | None = None
    pool_tracks: list[str] = field(default_factory=list)
    include_tags: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    anchors: list[ScenarioAnchor] = field(default_factory=list)
    required: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    forbid_pairs: list[tuple[str, str]] = field(default_factory=list)
    avoid_pairs: list[tuple[str, str]] = field(default_factory=list)
    prefer_pairs: list[tuple[str, str]] = field(default_factory=list)   # (first, then)
    curve: list[tuple[float, float]] | None = None    # explicit (relative position, energy) points
    # 'time': phases and the curve follow relative TIME (track midpoints, source durations);
    # 'tracks': they follow the relative POSITION in the list (k + 0.5) / n, whatever the lengths
    curve_axis: str = "time"
    phases: list[ScenarioPhase] = field(default_factory=list)
    prefer_moods: list[str] = field(default_factory=list)
    avoid_moods: list[str] = field(default_factory=list)
    transitions: TransitionRules = field(default_factory=TransitionRules)
    diversity: DiversityRules = field(default_factory=DiversityRules)
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    source_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def effective_tolerance_s(self) -> float | None:
        if self.target_duration_s is None:
            return None
        if self.duration_tolerance_s is not None:
            return self.duration_tolerance_s
        return max(120.0, 0.05 * self.target_duration_s)

    def phase(self, name: str) -> ScenarioPhase | None:
        return next((p for p in self.phases if p.name == name), None)

    def phase_at(self, t: float) -> tuple[int, ScenarioPhase]:
        """(index, phase) at relative time t (0..1)."""
        acc = 0.0
        for i, p in enumerate(self.phases):
            acc += p.share
            if t < acc - 1e-12:
                return i, p
        return len(self.phases) - 1, self.phases[-1]

    def phase_bounds(self) -> list[tuple[float, float]]:
        out, acc = [], 0.0
        for p in self.phases:
            out.append((acc, min(1.0, acc + p.share)))
            acc += p.share
        if out:
            out[-1] = (out[-1][0], 1.0)
        return out

    def target_energy(self, t: float) -> float | None:
        """Target energy of the dramaturgical curve at relative time t (0..1)."""
        t = min(1.0, max(0.0, t))
        if self.curve:
            pts = self.curve
            if t <= pts[0][0]:
                return pts[0][1]
            for (t0, e0), (t1, e1) in zip(pts, pts[1:]):
                if t <= t1:
                    return e0 if t1 <= t0 else e0 + (e1 - e0) * (t - t0) / (t1 - t0)
            return pts[-1][1]
        i, p = self.phase_at(t)
        if p.energy is None:
            return None
        lo, hi = self.phase_bounds()[i]
        frac = 0.0 if hi <= lo else (t - lo) / (hi - lo)
        return p.energy[0] + (p.energy[1] - p.energy[0]) * min(1.0, max(0.0, frac))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scenario(path: Path) -> Scenario:
    """Load and strictly validate a scenario (.yaml/.yml or .json)."""
    path = Path(path)
    data = load_config_document(path, noun="scenario", error=ScenarioError)
    scenario = parse_scenario(data, source=str(path))
    scenario.source_path = path
    log.info("scenario '%s' loaded from %s", scenario.name, path)
    return scenario


def parse_scenario(data: Any, *, source: str = "<scenario>") -> Scenario:
    """Validate an already-parsed scenario mapping."""
    p = _Parser(source, ScenarioError)
    if data is None:
        raise ScenarioError(f"{source}: scenario is empty")
    p.mapping(data, "", TOP_LEVEL_KEYS)
    name = data.get("scenario")
    if isinstance(name, bool) or not isinstance(name, (str, int, float)) or not str(name).strip():
        p.fail("scenario", "required: a non-empty scenario name")
    sc = Scenario(name=str(name).strip(), description=_text(data.get("description")))

    if data.get("target_duration_min") is not None:
        sc.target_duration_s = p.number(data["target_duration_min"], "target_duration_min", 0.0, 24 * 60.0,
                                        lo_exclusive=True) * 60.0
    if data.get("duration_tolerance_min") is not None:
        if sc.target_duration_s is None:
            p.fail("duration_tolerance_min", "needs target_duration_min")
        sc.duration_tolerance_s = p.number(data["duration_tolerance_min"], "duration_tolerance_min",
                                           0.0, 24 * 60.0) * 60.0
    if data.get("max_tracks") is not None:
        sc.max_tracks = p.integer(data["max_tracks"], "max_tracks", 1)
    if data.get("min_tracks") is not None:
        sc.min_tracks = p.integer(data["min_tracks"], "min_tracks", 1)
    if sc.min_tracks and sc.max_tracks and sc.min_tracks > sc.max_tracks:
        p.fail("min_tracks", f"{sc.min_tracks} is greater than max_tracks {sc.max_tracks}")
    if "use_all_tracks" in data:
        sc.use_all_tracks = p.boolean(data["use_all_tracks"], "use_all_tracks")
    if sc.use_all_tracks and (sc.target_duration_s is not None or sc.max_tracks is not None):
        p.fail("use_all_tracks", "places every eligible track; remove target_duration_min / max_tracks")
    if data.get("alternatives") is not None:
        n = p.integer(data["alternatives"], "alternatives", 1)
        if n > MAX_ALTERNATIVES:
            p.fail("alternatives", f"{n} is out of range (1..{MAX_ALTERNATIVES})")
        sc.alternatives = n
    if data.get("seed") is not None:
        sc.seed = p.integer(data["seed"], "seed", 0)

    pool = data.get("pool")
    if pool is not None:
        p.mapping(pool, "pool", POOL_KEYS)
        sc.pool_tracks = p.strings(pool.get("tracks"), "pool.tracks")
        sc.include_tags = p.strings(pool.get("include_tags"), "pool.include_tags")
        sc.exclude_tags = p.strings(pool.get("exclude_tags"), "pool.exclude_tags")

    sc.required = p.strings(data.get("required"), "required")
    sc.excluded = p.strings(data.get("excluded"), "excluded")
    clash = sorted({_ref_key(x) for x in sc.required} & {_ref_key(x) for x in sc.excluded})
    if clash:
        p.fail("excluded", f"track(s) both required and excluded: {', '.join(clash)}")

    adj = data.get("adjacency")
    if adj is not None:
        p.mapping(adj, "adjacency", ADJACENCY_KEYS)
        sc.forbid_pairs = _pairs(p, adj.get("forbid"), "adjacency.forbid")
        sc.avoid_pairs = _pairs(p, adj.get("avoid"), "adjacency.avoid")
        sc.prefer_pairs = _pairs(p, adj.get("prefer"), "adjacency.prefer")

    if data.get("curve") is not None:
        sc.curve = _curve(p, data["curve"])
    if data.get("curve_axis") is not None:
        sc.curve_axis = p.choice(data["curve_axis"], "curve_axis", CURVE_AXES)
    phases = data.get("phases")
    if phases is not None:
        sc.phases = _phases(p, phases, has_curve=sc.curve is not None)
    elif sc.curve is not None:
        sc.phases = [ScenarioPhase(name="main", share=1.0)]
    else:
        p.fail("phases", "required: a list of phases (or an explicit 'curve')")

    moods = data.get("moods")
    if moods is not None:
        p.mapping(moods, "moods", MOOD_KEYS)
        sc.prefer_moods = _moods(p, moods.get("prefer"), "moods.prefer", sc.warnings)
        sc.avoid_moods = _moods(p, moods.get("avoid"), "moods.avoid", sc.warnings)

    tr = data.get("transitions")
    if tr is not None:
        p.mapping(tr, "transitions", TRANSITION_KEYS)
        rules = TransitionRules()
        if "max_bpm_jump" in tr:
            rules.max_bpm_jump = (None if tr["max_bpm_jump"] is None
                                  else p.number(tr["max_bpm_jump"], "transitions.max_bpm_jump", 0.0, 200.0,
                                                lo_exclusive=True))
        if "max_key_steps" in tr:
            rules.max_key_steps = (None if tr["max_key_steps"] is None
                                   else p.integer(tr["max_key_steps"], "transitions.max_key_steps", 0))
        if tr.get("max_energy_jump") is not None:
            rules.max_energy_jump = p.number(tr["max_energy_jump"], "transitions.max_energy_jump", 0.0, 1.0,
                                             lo_exclusive=True)
        if tr.get("contrast_energy_jump") is not None:
            rules.contrast_energy_jump = p.number(tr["contrast_energy_jump"], "transitions.contrast_energy_jump",
                                                  0.0, 1.0, lo_exclusive=True)
        if "contrast_bpm_jump" in tr:
            rules.contrast_bpm_jump = (None if tr["contrast_bpm_jump"] is None
                                       else p.number(tr["contrast_bpm_jump"], "transitions.contrast_bpm_jump", 0.0,
                                                     200.0, lo_exclusive=True))
        if rules.contrast_energy_jump < rules.max_energy_jump:
            p.fail("transitions.contrast_energy_jump",
                   f"{rules.contrast_energy_jump:g} is below max_energy_jump {rules.max_energy_jump:g} "
                   "(a contrast is a LARGER jump the scenario allows on purpose)")
        sc.transitions = rules

    dv = data.get("diversity")
    if dv is not None:
        p.mapping(dv, "diversity", DIVERSITY_KEYS)
        rules_d = DiversityRules()
        if dv.get("artist_min_gap") is not None:
            rules_d.artist_min_gap = p.integer(dv["artist_min_gap"], "diversity.artist_min_gap", 0)
        if dv.get("max_tracks_per_artist") is not None:
            rules_d.max_tracks_per_artist = p.integer(dv["max_tracks_per_artist"],
                                                      "diversity.max_tracks_per_artist", 1)
        if "avoid_same_song" in dv:
            rules_d.avoid_same_song = p.boolean(dv["avoid_same_song"], "diversity.avoid_same_song")
        sc.diversity = rules_d

    w = data.get("weights")
    if w is not None:
        p.mapping(w, "weights", tuple(DEFAULT_WEIGHTS))
        for k, v in w.items():
            if k in ("description", "notes"):
                continue
            sc.weights[k] = p.number(v, f"weights.{k}", 0.0, 100.0)

    sc.anchors = _anchors(p, data.get("anchors"), sc)
    _check_track_counts(sc, p.fail, "max_tracks")
    return sc


def _check_track_counts(sc: Scenario, fail, where: str) -> None:
    """max_tracks must leave room for every anchor and required track."""
    if sc.max_tracks is None:
        return
    if sc.min_tracks and sc.min_tracks > sc.max_tracks:
        fail("min_tracks", f"{sc.min_tracks} is greater than {where} {sc.max_tracks}")
    beyond = [a.position for a in sc.anchors if a.position is not None and abs(a.position) > sc.max_tracks]
    if beyond:
        fail("anchors", f"position {beyond[0]} is beyond {where} {sc.max_tracks}")
    if len(sc.anchors) > sc.max_tracks:
        fail("anchors", f"{len(sc.anchors)} anchors but {where} is {sc.max_tracks}")
    needed = {_ref_key(a.track) for a in sc.anchors} | {_ref_key(r) for r in sc.required}
    if len(needed) > sc.max_tracks:
        fail("required", f"{len(needed)} anchored / required tracks but {where} is {sc.max_tracks}")


def with_overrides(sc: Scenario, *, target_duration_s: float | None = None,
                   max_tracks: int | None = None) -> Scenario:
    """The scenario with command-line overrides applied and re-validated: ``max_tracks`` is
    checked against the anchors, required tracks and ``min_tracks`` like the file's field;
    ``target_duration_s`` must be > 0 and at most 24 h like ``target_duration_min`` and drops
    the file's ``duration_tolerance_min`` (the default tolerance applies). Either override
    switches the scenario's ``use_all_tracks`` off — an explicit length / count on the command
    line wins, whereas the file itself may not combine them."""
    def fail(key: str, msg: str) -> None:
        flag = "--max-tracks" if max_tracks is not None else "--duration-min"
        raise ScenarioError(f"{flag} contradicts the scenario '{sc.name}': {key}: {msg}")

    out = dataclasses.replace(sc)
    if target_duration_s is not None:
        if not 0 < target_duration_s <= 24 * 3600:
            raise ScenarioError("--duration-min must be > 0 and at most 1440 (24 h)")
        out = dataclasses.replace(out, target_duration_s=target_duration_s, duration_tolerance_s=None,
                                  use_all_tracks=False)
    if max_tracks is not None:
        if max_tracks < 1:
            raise ScenarioError("--max-tracks must be >= 1")
        out = dataclasses.replace(out, max_tracks=max_tracks, use_all_tracks=False)
        _check_track_counts(out, fail, "--max-tracks")
    return out


def _text(value: Any) -> str | None:
    return " ".join(str(value).split()) if isinstance(value, str) and value.strip() else None


def _ref_key(entry: str) -> str:
    return min(reference_keys(entry) or {""})


def _pairs(p: _Parser, value: Any, key: str) -> list[tuple[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        p.fail(key, f"expected a list of [track, track] pairs, got {type(value).__name__}")
    out: list[tuple[str, str]] = []
    for i, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            p.fail(f"{key}[{i}]", f"expected a pair [\"Artist - Title\", \"Artist - Title\"], got {item!r}")
        both = p.strings(item, f"{key}[{i}]")
        if len(both) != 2 or _ref_key(both[0]) == _ref_key(both[1]):
            p.fail(f"{key}[{i}]", "a pair needs two different tracks")
        out.append((both[0], both[1]))
    return out


def _curve(p: _Parser, value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list) or len(value) < 2:
        p.fail("curve", "expected at least two points [relative_time, energy]")
    pts = [p.pair(v, f"curve[{i}]", 0.0, 1.0, ordered=False) for i, v in enumerate(value)]
    for i, (a, b) in enumerate(zip(pts, pts[1:]), 1):
        if b[0] < a[0]:
            p.fail(f"curve[{i}]", f"relative time {b[0]:g} goes backwards (previous point {a[0]:g})")
    return pts


def _moods(p: _Parser, value: Any, key: str, warnings: list[str]) -> list[str]:
    labels = mood_labels(p.strings(value, key))
    unknown = [m for m in labels if not is_known(m)]
    if unknown:
        warnings.append(f"{p.source}: {key}: mood(s) outside the built-in vocabulary match only by exact "
                        f"name: {', '.join(unknown)}")
    return labels


def _phases(p: _Parser, value: Any, *, has_curve: bool) -> list[ScenarioPhase]:
    if not isinstance(value, list) or not value:
        p.fail("phases", "expected a non-empty list of phases")
    phases: list[ScenarioPhase] = []
    seen: set[str] = set()
    for i, spec in enumerate(value):
        key = f"phases[{i}]"
        p.mapping(spec, key, PHASE_KEYS)
        name = spec.get("name")
        if not isinstance(name, str) or not _PHASE_NAME_RE.match(name.strip()):
            p.fail(f"{key}.name", f"expected a short identifier (letters, digits, '_' or '-'), got {name!r}")
        name = name.strip().lower()
        if name in seen:
            p.fail(f"{key}.name", f"duplicate phase name '{name}'")
        seen.add(name)
        key = f"phases.{name}"
        if spec.get("share") is None:
            p.fail(f"{key}.share", "required: the phase's relative duration (any positive number)")
        share = p.number(spec["share"], f"{key}.share", 0.0, 1000.0, lo_exclusive=True)
        energy = None
        if spec.get("energy") is not None:
            if has_curve:
                p.fail(f"{key}.energy", "the scenario has an explicit 'curve'; give phase energies OR a curve")
            raw = spec["energy"]
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                e = p.number(raw, f"{key}.energy", 0.0, 1.0)
                energy = (e, e)
            else:
                energy = p.pair(raw, f"{key}.energy", 0.0, 1.0, ordered=False)
        elif not has_curve:
            p.fail(f"{key}.energy", "required: target energy [at start, at end] (0..1) or a single number")
        rng = None
        if spec.get("energy_range") is not None:
            rng = p.pair(spec["energy_range"], f"{key}.energy_range", 0.0, 1.0, ordered=True)
            if energy and not all(rng[0] - 1e-9 <= e <= rng[1] + 1e-9 for e in energy):
                p.fail(f"{key}.energy_range", f"target energy {energy} lies outside the range {rng}")
        roles = []
        for j, r in enumerate(p.strings(spec.get("roles"), f"{key}.roles")):
            norm = normalize_tag(r).replace(" ", "_")
            if norm not in ROLES:
                p.fail(f"{key}.roles[{j}]", f"unknown role {r!r}" + _did_you_mean(norm, ROLES)
                       + f"; roles: {', '.join(ROLES)}")
            if norm not in roles:
                roles.append(norm)
        prefer: list[str] = []
        avoid: list[str] = []
        moods = spec.get("moods")
        if moods is not None:
            p.mapping(moods, f"{key}.moods", MOOD_KEYS)
            warnings: list[str] = []
            prefer = _moods(p, moods.get("prefer"), f"{key}.moods.prefer", warnings)
            avoid = _moods(p, moods.get("avoid"), f"{key}.moods.avoid", warnings)
            for w in warnings:
                log.info(w)
        contrast = p.boolean(spec.get("contrast", False), f"{key}.contrast")
        if "ai_positions" in spec:
            ai_positions = []
            for j, a in enumerate(p.strings(spec.get("ai_positions"), f"{key}.ai_positions")):
                norm = normalize_tag(a).replace(" ", "_")
                if norm not in AI_POSITIONS:
                    p.fail(f"{key}.ai_positions[{j}]", f"unknown AI position {a!r}" + _did_you_mean(norm, AI_POSITIONS)
                           + f"; positions: {', '.join(AI_POSITIONS)}")
                if norm not in ai_positions:
                    ai_positions.append(norm)
        else:
            ai_positions = [name] if name in AI_POSITIONS else []
        phases.append(ScenarioPhase(name=name, share=share, energy=energy, energy_range=rng, roles=roles,
                                    prefer_moods=prefer, avoid_moods=avoid, contrast=contrast,
                                    description=_text(spec.get("description")), ai_positions=ai_positions))
    total = sum(ph.share for ph in phases)
    for ph in phases:
        ph.share = ph.share / total
    return phases


def _anchors(p: _Parser, value: Any, sc: Scenario) -> list[ScenarioAnchor]:
    if value is None:
        return []
    if not isinstance(value, list):
        p.fail("anchors", f"expected a list of anchors, got {type(value).__name__}")
    out: list[ScenarioAnchor] = []
    positions: dict[int, int] = {}
    tracks: dict[str, int] = {}
    for i, spec in enumerate(value):
        key = f"anchors[{i}]"
        p.mapping(spec, key, ANCHOR_KEYS)
        ref = spec.get("track")
        if isinstance(ref, bool) or not isinstance(ref, (str, int, float)) or not str(ref).strip():
            p.fail(f"{key}.track", "required: the anchored track ('Artist - Title', path or file name)")
        ref = " ".join(str(ref).split())
        pos, phase = spec.get("position"), spec.get("phase")
        if (pos is None) == (phase is None):
            p.fail(key, "give exactly one of 'position' (1 = first, -1 = last) or 'phase'")
        if pos is not None:
            if isinstance(pos, str) and pos.strip().lower() in ("first", "last"):
                pos = 1 if pos.strip().lower() == "first" else -1
            if isinstance(pos, bool) or not isinstance(pos, int) or pos == 0:
                p.fail(f"{key}.position", f"expected a non-zero whole number (1 = first, -1 = last), got {pos!r}")
            if sc.max_tracks is not None and abs(pos) > sc.max_tracks:
                p.fail(f"{key}.position", f"{pos} is beyond max_tracks {sc.max_tracks}")
            if pos in positions:
                p.fail(f"{key}.position", f"position {pos} is already anchored by anchors[{positions[pos]}]")
            positions[pos] = i
        if phase is not None:
            if not isinstance(phase, str) or sc.phase(phase.strip().lower()) is None:
                p.fail(f"{key}.phase", f"unknown phase {phase!r}" + _did_you_mean(phase, [ph.name for ph in sc.phases]))
            phase = phase.strip().lower()
        rk = _ref_key(ref)
        if rk in tracks:
            p.fail(f"{key}.track", f"track '{ref}' is already anchored by anchors[{tracks[rk]}]")
        tracks[rk] = i
        if rk in {_ref_key(x) for x in sc.excluded}:
            p.fail(f"{key}.track", f"'{ref}' is anchored and excluded at the same time")
        out.append(ScenarioAnchor(track=ref, position=pos, phase=phase))
    return out


# ---------------------------------------------------------------------------
# Resolving references against the library
# ---------------------------------------------------------------------------

_BRACKETS_RE = re.compile(r"\s*[(\[]([^)\]]*)[)\]]")
# Bracketed text is a VERSION qualifier (dropped when comparing songs) only when it says so:
# '(2008 Remaster)', '[Radio Edit]', '(feat. X)', '(Live at ...)'. '(Part 1)' / '(Reprise)' /
# '(Interlude)' name a different piece and are kept.
_VERSION_WORDS_RE = re.compile(
    r"\b(?:remaster(?:ed)?|live|edit|radio|single|album|version|mix|remix|mono|stereo|instrumental|"
    r"acoustic|demo|extended|explicit|clean|bonus|deluxe|original|feat|ft|featuring|with|from|"
    r"soundtrack|ost|anniversary|edition|(?:19|20)\d\d)\b", re.IGNORECASE)
_VERSION_TAIL_RE = re.compile(r"\s+-\s+(?:\d{4}\s+)?(?:remaster(?:ed)?|single|radio|album|edit|mono|stereo)"
                              r"\b.*$", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")


def _relaxed_title(text: str | None) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text)).casefold()
    s = _BRACKETS_RE.sub(lambda m: "" if _VERSION_WORDS_RE.search(m.group(1)) else f" {m.group(1)} ", s)
    s = _VERSION_TAIL_RE.sub("", s)
    return " ".join(_PUNCT_RE.sub(" ", s).split())


def song_key(track: TrackFeatures) -> str | None:
    """'artist|title' with version qualifiers removed — two versions of one song (remaster,
    remix, radio edit) share it. None when artist or title is unknown."""
    artist = primary_artist(track.artist)
    title = _relaxed_title(track.title)
    if not artist or not title:
        return None
    return f"{artist}|{title}"


def match_reference(ref: str, tracks: list[TrackFeatures]) -> tuple[list[TrackFeatures], str]:
    """Tracks matching a scenario reference and how ('exact' / 'relaxed' / 'none')."""
    keys = reference_keys(ref)
    exact = [t for t in tracks if track_match_keys(t) & keys]
    if exact:
        return exact, "exact"
    m = re.split(r"\s+[-–—]\s+", ref.strip(), maxsplit=1)
    if len(m) == 2:
        # the whole credit must match (primary artist or full credit): 'Fire' must not match
        # 'Earth, Wind & Fire' (list rules match every credited part; a hard anchor must not)
        wanted = {primary_artist(m[0]), normalize_artist(m[0])} - {None, ""}
        title = _relaxed_title(m[1])
        if wanted and title:
            relaxed = [t for t in tracks if t.title and _relaxed_title(t.title) == title
                       and wanted & ({primary_artist(t.artist), normalize_artist(t.artist)} - {None, ""})]
            if relaxed:
                return relaxed, "relaxed"
    return [], "none"


def suggest(ref: str, tracks: list[TrackFeatures], n: int = 3) -> list[str]:
    names = {t.display_name: t for t in tracks}
    return difflib.get_close_matches(ref, list(names), n=n, cutoff=0.5)


@dataclass
class ResolvedScenario:
    """A scenario bound to concrete library tracks (track ids)."""
    scenario: Scenario
    pool: list[TrackFeatures]                      # candidates after pool filters / exclusions
    anchors: list[tuple[ScenarioAnchor, str]]      # (anchor, track_id)
    required: list[str]
    excluded: dict[str, str]                       # track_id -> reason
    forbid_pairs: set[frozenset[str]]
    avoid_pairs: set[frozenset[str]]
    prefer_pairs: set[tuple[str, str]]
    notes: list[str] = field(default_factory=list)
    rel_paths: dict[str, str] = field(default_factory=dict)      # track_id -> rel_path of excluded tracks


def resolve_scenario(scenario: Scenario, eligible: list[TrackFeatures], *,
                     curation_excluded: dict[str, list[str]] | None = None,
                     library: list[TrackFeatures] | None = None) -> ResolvedScenario:
    """Bind a scenario to the curated library.

    ``eligible`` are the tracks curation let through; ``library`` (optional) is every
    analysed track, used to explain an anchor that exists but was excluded; and
    ``curation_excluded`` maps such track ids to the curation reasons. Raises
    ScenarioError for an anchor / required track that matches nothing, matches several
    tracks, or was excluded by the client profile."""
    sc = scenario
    library = library or eligible
    curation_excluded = curation_excluded or {}
    notes: list[str] = list(sc.warnings)

    def one(ref: str, what: str) -> TrackFeatures:
        found, how = match_reference(ref, eligible)
        if len(found) == 1:
            if how == "relaxed":
                notes.append(f"{what} '{ref}' matched '{found[0].display_name}' ignoring version qualifiers")
            return found[0]
        if len(found) > 1:
            listed = "; ".join(t.rel_path for t in found[:6])
            raise ScenarioError(f"{what} '{ref}' is ambiguous — it matches {len(found)} tracks ({listed}); "
                                f"use the library-relative path instead")
        in_library, _ = match_reference(ref, library)
        for t in in_library:
            reasons = curation_excluded.get(t.track_id)
            if reasons:
                raise ScenarioError(f"{what} '{ref}' ({t.rel_path}) is excluded by the client profile: "
                                    f"{'; '.join(reasons)}")
        hint = suggest(ref, eligible)
        raise ScenarioError(f"{what} '{ref}' matches no eligible track in the library"
                            + (f"; closest: {', '.join(hint)}" if hint else ""))

    def soft(refs: list[str], what: str, *, exact_only: bool = False) -> list[TrackFeatures]:
        out: list[TrackFeatures] = []
        for ref in refs:
            found, how = match_reference(ref, library)
            if how == "relaxed" and exact_only:
                # excluding 'Song (Live)' must never exclude the studio 'Song' the library holds
                found = []
            if not found:
                notes.append(f"{what} '{ref}' matches no track in the library (ignored)")
            elif how == "relaxed":
                notes.append(f"{what} '{ref}' matched {', '.join(repr(t.display_name) for t in found[:3])} "
                             f"ignoring version qualifiers")
            out.extend(found)
        return out

    excluded: dict[str, str] = {}
    for t in soft(sc.excluded, "excluded track", exact_only=True):
        excluded[t.track_id] = "excluded by the scenario"

    pool = [t for t in eligible if t.track_id not in excluded]
    if sc.pool_tracks:
        listed = {t.track_id for t in soft(sc.pool_tracks, "pool track")}
        for t in pool:
            if t.track_id not in listed:
                excluded.setdefault(t.track_id, "not in the scenario's pool.tracks list")
    inc = {normalize_tag(x) for x in sc.include_tags}
    exc = {normalize_tag(x) for x in sc.exclude_tags}
    for t in pool:
        labels = set()
        for v in [*(t.vibe_tags or []), t.genre, t.mood]:
            labels |= label_keys(v)
        if exc and labels & exc:
            excluded.setdefault(t.track_id, f"tag excluded by the scenario: {', '.join(sorted(labels & exc))}")
        elif inc and not labels & inc:
            excluded.setdefault(t.track_id, "none of the scenario's pool.include_tags")

    anchors = [(a, one(a.track, "anchor").track_id) for a in sc.anchors]
    required = []
    for ref in sc.required:
        tid = one(ref, "required track").track_id
        if tid not in required:
            required.append(tid)
    # anchored / required tracks are never filtered out by pool rules
    names = {t.track_id: t.display_name for t in library}
    for tid in [tid for _, tid in anchors] + required:
        if excluded.get(tid, "").startswith("excluded by the scenario"):
            raise ScenarioError(f"track '{names.get(tid, tid)}' is both anchored/required and excluded by the "
                                f"scenario")
        excluded.pop(tid, None)
    pool = [t for t in eligible if t.track_id not in excluded]

    def pairs(items: list[tuple[str, str]], what: str):
        for a, b in items:
            fa, fb = soft([a], what), soft([b], what)
            for x in fa:
                for y in fb:
                    if x.track_id != y.track_id:
                        yield x.track_id, y.track_id

    forbid = {frozenset(p) for p in pairs(sc.forbid_pairs, "adjacency.forbid track")}
    avoid = {frozenset(p) for p in pairs(sc.avoid_pairs, "adjacency.avoid track")}
    prefer = set(pairs(sc.prefer_pairs, "adjacency.prefer track"))
    paths = {t.track_id: t.rel_path for t in library if t.track_id in excluded}
    return ResolvedScenario(scenario=sc, pool=pool, anchors=anchors, required=required, excluded=excluded,
                            forbid_pairs=forbid, avoid_pairs=avoid, prefer_pairs=prefer, notes=notes,
                            rel_paths=paths)


__all__ = ["ScenarioError", "ROLES", "AI_POSITIONS", "DEFAULT_WEIGHTS", "ScenarioAnchor", "ScenarioPhase", "TransitionRules",
           "DiversityRules", "Scenario", "load_scenario", "parse_scenario", "with_overrides", "match_reference",
           "song_key",
           "ResolvedScenario", "resolve_scenario"]
