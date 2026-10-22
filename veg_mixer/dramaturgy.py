"""Dramaturgy engine (VEmix 3.0, spec §5 and §8): scenario-driven playlist planning.

The 2.0 sequencer (``sequencing.build_sequence``) orders a pool for smooth transitions
along a two-point energy line. This module plans a PERFORMANCE: phases with relative
durations, a target energy curve over relative TIME, hard anchors, required / excluded
tracks, forbidden and preferred neighbours, dramaturgical roles, moods, tempo and
harmonic tolerances, artist diversity and a target duration — and it can choose a
subset of the pool. ``build_sequence`` is untouched; 2.0 commands behave as before.

Objective (``Model.evaluate``) — an explicit, decomposed cost. Every term is normalized
to 0..1 (per position, per transition, or global; two exceptions: ``diversity`` adds up the
close same-artist pairs of a position and can exceed 1, ``adjacency`` is −1 for a preferred
pair) and weighted by ``scenario.DEFAULT_WEIGHTS`` (overridable in the scenario)::

    J = Σ_w  w · term_w          position terms: mean over positions
                                 transition terms: mean over transitions

    curve        trust·min(1, |E − target(t)| / CURVE_TOLERANCE) + (1 − trust)·UNKNOWN_CURVE_COST,
                 trust = 0.5 + 0.5·energy confidence; UNKNOWN_CURVE_COST for unknown energy
    band         trust·min(1, distance outside the phase energy_range / BAND_FALLOFF)
    role         1 − min(1, max(best wanted-role confidence, AI position hint for the phase's
                 ai_positions) / ROLE_FULL); UNKNOWN_ROLE_COST (0.5) when no interpretation layer
                 judged any wanted role (role_judged) and there is no hint; 0 when the phase wants
                 no roles
    mood         preferred-mood affinity / avoided-mood affinity (moods.py), 0.5 when unknown
    editorial    client-profile editorial score mapped to 0..1 (0 = best)
    uncertainty  mean (1 − confidence) of energy, tempo, key, mood and roles
    transition   1 − 2.0 transition score (mood/tags, energy, tempo, key, genre; unknown → neutral);
                 ×0.7 when entering a 'contrast' phase
    edge_energy  |head energy of the next − tail energy of the previous| (segment analysis)
    tempo        folded BPM difference / max_bpm_jump (×0.3 entering a 'contrast' phase)
    harmony      Camelot wheel distance (0 for ≤ 1 step, 0.5 for 2, 1 beyond)
    timbre       distance of the segment-analysis timbre vectors
    mood_flow    1 − mood similarity of neighbours; ×0.6 at a phase boundary, ×0.3 entering a contrast
    diversity    same artist within a few positions (beyond the hard minimum gap)
    adjacency    scenario 'avoid' pair (+1) / 'prefer' pair (−1)
    duration     distance from the target duration (global)

``t`` is the relative time of a track's MIDPOINT in the plan (source durations; a
rendered mix is shorter by the crossfades — the report says so) with the default
``curve_axis: time``; with ``curve_axis: tracks`` it is the relative POSITION (k + 0.5) / n,
whatever the track lengths (``Model.timeline``). Entering a phase
marked ``contrast`` relaxes the energy-jump limit (``contrast_energy_jump``) and the tempo
limit (``contrast_bpm_jump``, unlimited by default) and discounts the transition, edge-energy,
tempo and mood-flow costs: a controlled contrast is planned, not flagged.

Hard constraints always dominate: the optimizer minimizes (violation severity, J)
lexicographically. Fixed-position anchors and required tracks are structural (moves
that would break them are never made); phase anchors, forbidden pairs, energy / tempo /
key limits, artist gap and count, one version per song and the duration window are
counted as violations and reported with the measured numbers when they cannot be
avoided ("not avoided by the search" — the planner never claims a proof it did not run).

Search: starting points from a beam construction over relative time (anchors placed at
their positions, required and phase-anchored tracks forced before their deadline) and a
curve-driven assignment of tracks to equal-time slots (Hungarian algorithm on the
position costs); the best ones are improved by a seeded, deterministic iterated local
search (swap, relocate, block relocate / swap, reverse, replace, insert, remove). The
first stages of that search price hard constraints instead of forbidding them (penalty
annealing), so it can cross an infeasible plan on the way to a better feasible one; the
result is always chosen lexicographically. The budget is a number of objective
evaluations — no wall-clock limits, so the same inputs and seed give the same plan. Alternatives re-run the search with weight presets ('dramatic',
'smooth', 'harmonic', ...) plus a novelty penalty against the earlier variants, and are
all re-scored with the scenario's own weights for an honest comparison.

Explanations (``explain``) are built only from the terms above, evaluated for the final
order, plus counterfactuals: the best replacement / swap for each slot and how much
each term would change. Every feature carries its provenance category: measured,
file tags, curator, external, derived (rules) or AI.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .config import SequencingConfig
from .models import ClientProfile, TrackFeatures, to_jsonable
from .moods import best_affinity, label_affinity, mood_labels, mood_similarity
from .scenario import DEFAULT_WEIGHTS, ROLES, ResolvedScenario, Scenario, ScenarioError, song_key
from .sequencing import _FeatureMatrix, artist_key, camelot_steps, effective_config, parse_camelot, transition_score

log = logging.getLogger(__name__)

ENGINE_VERSION = "dramaturgy-v1"

CURVE_TOLERANCE = 0.50          # |E − target| at which the curve cost reaches 1 (large misses must stay distinguishable)
UNKNOWN_CURVE_COST = 0.35       # unknown energy: neither a good nor a bad fit
# Energy is the backbone of a dramaturgy, so even the low-confidence 2.0 heuristic (0.4) keeps
# most of its say in the curve term: trust = CURVE_MIN_TRUST + (1 − CURVE_MIN_TRUST) · confidence.
CURVE_MIN_TRUST = 0.5
BAND_FALLOFF = 0.15
ROLE_FULL = 0.5                 # a role at this confidence counts as a full fit
MOOD_FULL = 0.8
UNKNOWN_ROLE_COST = 0.5
UNKNOWN_MOOD_COST = 0.5
UNKNOWN_EDGE_COST = 0.3
UNKNOWN_TEMPO_COST = 0.3
UNKNOWN_KEY_COST = 0.2
UNKNOWN_TIMBRE_COST = 0.3
UNKNOWN_FLOW_COST = 0.4
CONTRAST_DISCOUNT = 0.3         # edge-energy / mood-flow cost kept when a contrast is planned
CONTRAST_TRANSITION_DISCOUNT = 0.7   # 2.0 transition cost kept at a planned contrast
BOUNDARY_FLOW_DISCOUNT = 0.6    # mood-flow cost kept at any phase boundary (phases change mood on purpose)
DEFAULT_BPM_SPAN = 16.0
AI_CONFIDENCE_CAP = 0.8         # an AI label never counts as a curator's
COMMUNITY_TAG_CONFIDENCE = 0.3
# Tempo / key limits of a scenario are HARD only when both sides are known with at least this
# confidence: a mis-detected value must not be reported as a broken rule. Below it the limit still
# costs through the tempo / harmony terms and the report marks the transition as unverified.
# Auto BPM (0.5) qualifies unless the segment analysis doubts it (TEMPO_DOUBT_CONFIDENCE); an
# automatically detected key never does (2.0 calibrates it to <= 0.5): only a curated key is hard.
HARD_LIMIT_CONFIDENCE = 0.45
KEY_HARD_LIMIT_CONFIDENCE = 0.6
TEMPO_DOUBT_CONFIDENCE = 0.3    # auto BPM the segment analysis contradicts (or finds no pulse for)
TEMPO_SAME_PULSE = 0.04         # relative BPM tolerance for "the same pulse" (also at x2 / x0.5)
ROLE_TAG_PREFIX = "role:"       # curator vibe tag 'role:opener' assigns a role (confidence 1.0, manual)
VIOLATION_SCALE = 1000.0        # key = severity · scale + J (hard constraints dominate)
# An anchor or a required track is an explicit instruction about one track: breaking it weighs
# more than breaking a general tolerance (energy / tempo limits, artist spacing ...).
ANCHOR_SEVERITY = 10.0
POOL_CAP = 600                  # candidates kept for the n × n matrices (anchors/required always kept)
BEAM_WIDTH = 10
BEAM_SHORTLIST = 24
NOVELTY_WEIGHT = 0.6            # per unit of overlap with earlier variants (0..1)

PRESETS: dict[str, dict[str, float]] = {
    "balanced": {},
    "dramatic": {"curve": 1.8, "band": 1.5, "role": 1.6, "mood": 1.5, "transition": 0.6, "mood_flow": 0.6},
    "smooth": {"transition": 2.0, "tempo": 2.0, "harmony": 1.8, "edge_energy": 2.0, "mood_flow": 1.5,
               "timbre": 1.5, "curve": 0.6, "band": 0.8},
    "harmonic": {"harmony": 3.0, "tempo": 1.6, "transition": 1.2},
    "fresh": {},
    "varied": {"diversity": 2.5, "mood": 1.2},
}
PRESET_ORDER = ("balanced", "dramatic", "smooth", "harmonic", "fresh", "varied")
PRESET_DESCRIPTIONS = {
    "balanced": "the scenario's own weights",
    "dramatic": "weights favour the curve, roles and moods over transition smoothness",
    "smooth": "weights favour transitions (tempo, key, edge energy, timbre) over the curve",
    "harmonic": "weights favour harmonic and tempo compatibility",
    "fresh": "the scenario's weights, pushed away from the variants above",
    "varied": "weights favour spreading artists and moods",
}
# Every variant after the first is also pushed away from the earlier ones (novelty term), so the
# alternatives differ; the comparison re-scores all of them with the scenario's own weights.
POSITION_TERMS = ("curve", "band", "role", "mood", "editorial", "uncertainty")
TRANSITION_TERMS = ("transition", "edge_energy", "tempo", "harmony", "timbre", "mood_flow", "diversity", "adjacency")
GLOBAL_TERMS = ("duration",)
TERMS = POSITION_TERMS + TRANSITION_TERMS + GLOBAL_TERMS


def provenance_category(source: str | None) -> str:
    """measured | file_tags | curator | external | derived | ai | unknown."""
    if not source:
        return "unknown"
    if source == "manual":
        return "curator"
    if source in ("id3", "filename"):
        return "file_tags"
    if source.startswith("auto:"):
        return "measured"
    if source.startswith("research:"):
        return "external"
    if source.startswith("rules:"):
        return "derived"
    if source.startswith("ai:"):
        return "ai"
    return "unknown"


# ---------------------------------------------------------------------------
# Track context: everything the planner knows about one track, with provenance
# ---------------------------------------------------------------------------

@dataclass
class TrackContext:
    track: TrackFeatures
    energy: float | None = None
    energy_conf: float = 0.0
    energy_source: str | None = None
    energy_in: float | None = None
    energy_out: float | None = None
    bpm: float | None = None
    bpm_conf: float = 0.0
    bpm_source: str | None = None
    camelot: str | None = None
    key_conf: float = 0.0
    key_source: str | None = None
    moods: dict[str, float] = field(default_factory=dict)
    mood_sources: dict[str, str] = field(default_factory=dict)
    roles: dict[str, float] = field(default_factory=dict)
    role_sources: dict[str, str] = field(default_factory=dict)
    # roles some interpretation layer could judge for this track: a wanted role outside this set
    # is UNKNOWN for the track (no evidence either way), not a misfit
    role_judged: set[str] = field(default_factory=set)
    phase_hints: dict[str, float] = field(default_factory=dict)     # AI position (AI_POSITIONS) -> confidence
    trajectory: dict | None = None
    density: dict | None = None
    harmony: dict | None = None                 # segment analysis: key clarity / stability over time
    timbre: list[float] | None = None
    tempo_notes: list[str] = field(default_factory=list)       # doubts about the BPM (segment analysis)
    artist: str | None = None                   # primary-artist name key ('earth, wind & fire')
    song: str | None = None                     # 'artist|title' without version qualifiers
    # every identity key of the artist / the song: the name keys plus MusicBrainz artist / work
    # ids of a matched record. Tracks sharing ANY key are one artist / one song, so a partly
    # researched library still applies the diversity rules across matched and unmatched tracks.
    artist_ids: set[str] = field(default_factory=set)
    song_ids: set[str] = field(default_factory=set)
    editorial: float | None = None
    facts: dict[str, Any] = field(default_factory=dict)        # matched external facts (field -> value)
    interpretation_sources: list[str] = field(default_factory=list)   # 'rules:…' / 'ai:…' layers received
    provenance: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    low_confidence: list[str] = field(default_factory=list)

    @property
    def track_id(self) -> str:
        return self.track.track_id


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def build_context(track: TrackFeatures, *, segments: dict | None = None, research: dict | None = None,
                  interpretations: Sequence[dict] = (), editorial: float | None = None,
                  use_community_tags: bool = False) -> TrackContext:
    """Assemble a TrackContext. ``segments`` is an ``audio_intel`` value, ``research`` a
    ``ResearchRecord.to_dict()``, ``interpretations`` derived/AI interpretations (each with a
    ``source``). Only MATCHED research facts are used; community tags only when allowed."""
    src = track.feature_sources or {}
    ctx = TrackContext(track=track, editorial=_num(editorial))
    prov = ctx.provenance

    def note(feature: str, value: Any, source: str | None, conf: float | None) -> None:
        prov.append({"feature": feature, "value": value, "source": source, "category": provenance_category(source),
                     "confidence": None if conf is None else round(float(conf), 3)})

    e = _num(track.energy)
    if e is not None:
        ctx.energy = _clip01(e)
        ctx.energy_source = src.get("energy")
        c = track.confidence("energy")
        ctx.energy_conf = _clip01(c if c is not None else 0.5)
        note("energy", round(ctx.energy, 3), ctx.energy_source, ctx.energy_conf)
    else:
        ctx.missing.append("energy")
    b = _num(track.bpm)
    if b is not None and b > 0:
        ctx.bpm, ctx.bpm_source = b, src.get("bpm")
        c = track.confidence("bpm")
        ctx.bpm_conf = _clip01(c if c is not None else 0.5)
        note("bpm", round(b, 1), ctx.bpm_source, ctx.bpm_conf)
    else:
        ctx.missing.append("bpm")
    if track.camelot and parse_camelot(track.camelot):
        ctx.camelot, ctx.key_source = track.camelot, src.get("camelot")
        from .sequencing import _conf as seq_conf      # calibrated (auto keys <= 0.5), the 2.0 rule
        ctx.key_conf = _clip01(seq_conf(track, "camelot"))
        note("key", track.camelot, ctx.key_source, ctx.key_conf)
    else:
        ctx.missing.append("key")

    # Moods and roles by tier: the curator's labels REPLACE every interpretation (the manual value
    # always outranks AI); without them AI (capped), derived rules, file tags and community tags
    # (opt-in) are merged, each label keeping its best confidence. Vibe tags that are mood words
    # count as moods; 'role:<role>' vibe tags assign roles.
    tiers: dict[str, dict[str, dict[str, tuple[float, str]]]] = {
        "curator": {"moods": {}, "roles": {}}, "other": {"moods": {}, "roles": {}}}
    judged: dict[str, set[str]] = {"curator": set(), "other": set()}

    def tier_of(source: str) -> str:
        return "curator" if provenance_category(source) == "curator" else "other"

    def add_label(kind: str, label: str, conf: float, source: str) -> None:
        if conf <= 0 or not label:
            return
        bucket = tiers[tier_of(source)][kind]
        if conf > bucket.get(label, (0.0, ""))[0]:
            bucket[label] = (round(_clip01(conf), 3), source)

    def add_mood(label: str, conf: float, source: str) -> None:
        add_label("moods", label, conf, source)

    if track.mood:
        conf = track.confidence("mood") or 0.5
        for lab in mood_labels(track.mood):
            add_mood(lab, conf, src.get("mood") or "unknown")
    for tag in track.vibe_tags or []:
        text = str(tag).strip()
        if text.lower().startswith(ROLE_TAG_PREFIX):
            role = text[len(ROLE_TAG_PREFIX):].strip().lower().replace(" ", "_").replace("-", "_")
            if role in ROLES:
                source = src.get("vibe_tags") or "unknown"
                add_label("roles", role, track.confidence("vibe_tags") or 0.5, source)
                judged[tier_of(source)].add(role)        # a tag judges only the role it names
            continue
        for lab in mood_labels(tag):
            if label_affinity(lab, lab) is not None and _is_mood_word(lab):
                add_mood(lab, 0.8 * (track.confidence("vibe_tags") or 0.5), src.get("vibe_tags") or "unknown")

    if segments and isinstance(segments.get("summary"), dict):
        s = segments["summary"]
        e_in, e_out, e_mean = _num(s.get("energy_in")), _num(s.get("energy_out")), _num(s.get("energy_mean"))
        seg_src = f"auto:{segments.get('version', 'segments')}"
        if ctx.energy is not None and e_in is not None and e_mean is not None:
            ctx.energy_in = _clip01(ctx.energy + (e_in - e_mean))
        if ctx.energy is not None and e_out is not None and e_mean is not None:
            ctx.energy_out = _clip01(ctx.energy + (e_out - e_mean))
        if isinstance(s.get("trajectory"), dict) and s["trajectory"].get("value"):
            ctx.trajectory = dict(s["trajectory"], source=seg_src)
            note("trajectory", s["trajectory"].get("value"), seg_src, _num(s["trajectory"].get("confidence")))
        if isinstance(s.get("density"), dict) and s["density"].get("value"):
            ctx.density = dict(s["density"], source=seg_src)
            note("density", s["density"].get("value"), seg_src, _num(s["density"].get("confidence")))
        h = s.get("harmony")
        if isinstance(h, dict) and h.get("stability") is not None:
            ctx.harmony = {k: h.get(k) for k in ("key_estimate", "clarity", "stability", "modulations")}
            note("key_stability", h.get("stability"), seg_src, None)
        tv = s.get("timbre")
        if isinstance(tv, dict):                 # {dimension: 0..1} — compared in a fixed key order
            tv = [tv[k] for k in sorted(tv)]
        if isinstance(tv, list) and tv and all(_num(x) is not None for x in tv):
            ctx.timbre = [float(x) for x in tv]
        _check_tempo(ctx, s.get("tempo"), seg_src, note)
        if ctx.energy_in is not None:
            note("energy_in", round(ctx.energy_in, 3), seg_src, ctx.energy_conf)
            note("energy_out", round(ctx.energy_out, 3) if ctx.energy_out is not None else None, seg_src,
                 ctx.energy_conf)
    else:
        ctx.missing.append("segments")

    for interp in interpretations or ():
        if not isinstance(interp, dict):
            continue
        source = str(interp.get("source") or "unknown")
        ctx.interpretation_sources.append(source)
        cap = AI_CONFIDENCE_CAP if source.startswith("ai:") else 1.0
        for m in interp.get("moods") or []:
            if isinstance(m, dict):
                for lab in mood_labels(m.get("label")):
                    add_mood(lab, min(cap, _num(m.get("confidence")) or 0.0), source)
        for r in interp.get("roles") or []:
            if isinstance(r, dict) and r.get("role") in ROLES:
                add_label("roles", r["role"], min(cap, _num(r.get("confidence")) or 0.0), source)
        # an AI interpretation judges every role; derived rules list what their data allowed
        listed = interp.get("judged_roles")
        judged[tier_of(source)] |= ({x for x in listed if x in ROLES} if isinstance(listed, list) else set(ROLES))
        for p in interp.get("positions") or []:
            if isinstance(p, dict) and isinstance(p.get("phase"), str):
                conf = min(cap, _num(p.get("confidence")) or 0.0)
                ctx.phase_hints[p["phase"]] = max(ctx.phase_hints.get(p["phase"], 0.0), round(conf, 3))
        if ctx.trajectory is None and isinstance(interp.get("energy_trajectory"), dict):
            t = interp["energy_trajectory"]
            if t.get("value") not in (None, "unknown"):
                ctx.trajectory = dict(t, source=source)
        if ctx.density is None and isinstance(interp.get("sonic_density"), dict):
            d = interp["sonic_density"]
            if d.get("value") not in (None, "unknown"):
                ctx.density = dict(d, source=source)

    if research and research.get("status") == "matched":
        # a field the sources disagree on is not a verified fact: it is reported, never used
        disputed = {str(c.get("field")) for c in research.get("conflicts") or [] if isinstance(c, dict)}
        if disputed & {"year", "first_release_date", "publication_date_wikidata"}:
            disputed |= {"year", "first_release_date", "publication_date_wikidata"}
        for name, fact in (research.get("facts") or {}).items():
            if isinstance(fact, dict) and fact.get("value") is not None:
                source = f"research:{fact.get('source', 'unknown')}"
                if name in disputed:
                    note(f"disputed:{name}", fact.get("value"), source, None)
                    continue
                ctx.facts[name] = fact.get("value")
                note(f"fact:{name}", fact.get("value"), source, _num(research.get("match_confidence")))
        if use_community_tags:
            for tag in research.get("user_tags") or []:
                if isinstance(tag, dict) and tag.get("scope") != "work":     # not the song's original
                    for lab in mood_labels(tag.get("name")):
                        if _is_mood_word(lab):
                            add_mood(lab, COMMUNITY_TAG_CONFIDENCE, f"research:{tag.get('source', 'tags')}")

    for kind in ("moods", "roles"):
        use = "curator" if tiers["curator"][kind] else "other"
        target, sources = (ctx.moods, ctx.mood_sources) if kind == "moods" else (ctx.roles, ctx.role_sources)
        for label, (conf, source) in tiers[use][kind].items():
            target[label], sources[label] = conf, source
        if use == "curator":
            for label, (conf, source) in tiers["other"][kind].items():
                note(f"{kind[:-1]}_set_aside", label, source, conf)      # the curator's labels outrank it
    ctx.role_judged = set(judged["curator"]) if tiers["curator"]["roles"] else set(judged["other"])
    if tiers["curator"]["roles"] and ctx.phase_hints:
        for pos, conf in ctx.phase_hints.items():
            note("position_hint_set_aside", pos, "ai", conf)
        ctx.phase_hints = {}

    for lab, conf in ctx.moods.items():
        note("mood", lab, ctx.mood_sources.get(lab), conf)
    for role, conf in ctx.roles.items():
        note("role", role, ctx.role_sources.get(role), conf)
    if not ctx.moods:
        ctx.missing.append("mood")
    if not ctx.roles:
        ctx.missing.append("roles")

    ctx.artist = artist_key(track.artist)
    ctx.song = song_key(track)
    ctx.artist_ids = {f"name:{ctx.artist}"} if ctx.artist else set()
    ctx.song_ids = {f"name:{ctx.song}"} if ctx.song else set()
    artist_id = ctx.facts.get("artist_mbid")
    if isinstance(artist_id, str) and artist_id:
        ctx.artist_ids.add(f"mbid:{artist_id}")
    work = ctx.facts.get("work_mbid")
    if isinstance(work, str) and work:
        ctx.song_ids.add(f"work:{work}")
    for name, conf in (("energy", ctx.energy_conf if ctx.energy is not None else None),
                       ("bpm", ctx.bpm_conf if ctx.bpm is not None else None),
                       ("key", ctx.key_conf if ctx.camelot else None)):
        if conf is not None and conf < 0.5:
            ctx.low_confidence.append(name)
    return ctx


def _check_tempo(ctx: TrackContext, tempo: Any, seg_src: str, note: Callable[..., None]) -> None:
    """Cross-check the track BPM with the segment analysis' own tempo estimate. A measured
    BPM the analysis contradicts (another pulse, not just half / double time) or finds no
    pulse for drops below HARD_LIMIT_CONFIDENCE: its tempo limits are then not enforced, and
    the report says why. A curated BPM is never second-guessed. Half / double-time ambiguity
    is reported but changes nothing: the tempo limits compare folded BPM differences."""
    if not isinstance(tempo, dict) or ctx.bpm is None or provenance_category(ctx.bpm_source) == "curator":
        return
    seg_bpm, seg_conf = _num(tempo.get("bpm")), _num(tempo.get("confidence")) or 0.0
    doubt = None
    if seg_bpm is None or seg_bpm <= 0:
        doubt = "the segment analysis found no clear pulse"
    else:
        ratio = ctx.bpm / seg_bpm
        if not any(abs(ratio / f - 1.0) <= TEMPO_SAME_PULSE for f in (1.0, 2.0, 0.5)) and seg_conf >= 0.3:
            doubt = f"the segment analysis measured {seg_bpm:.0f} BPM (confidence {seg_conf:.2f})"
        elif tempo.get("half_double_ambiguous"):
            ctx.tempo_notes.append(f"half / double time ambiguous ({seg_bpm:.0f} BPM or its double / half; "
                                   f"tempo limits compare folded differences)")
    note("tempo_check", None if seg_bpm is None else round(seg_bpm, 1), seg_src, seg_conf)
    if doubt and ctx.bpm_conf > TEMPO_DOUBT_CONFIDENCE:
        ctx.bpm_conf = TEMPO_DOUBT_CONFIDENCE
        for entry in ctx.provenance:
            if entry["feature"] == "bpm":
                entry["confidence"] = TEMPO_DOUBT_CONFIDENCE
        ctx.tempo_notes.append(f"tempo {ctx.bpm:.0f} BPM is doubtful: {doubt} — tempo limits not enforced")


def _identity_groups(id_sets: Sequence[set[str]]) -> np.ndarray:
    """Group index per track (−1 = no identity): tracks sharing any identity key are one group
    (union-find over the keys, deterministic: groups numbered in track order)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ids in id_sets:
        keys = sorted(ids)
        for a, b in zip(keys, keys[1:]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    numbers: dict[str, int] = {}
    out = []
    for ids in id_sets:
        if not ids:
            out.append(-1)
            continue
        root = find(sorted(ids)[0])
        out.append(numbers.setdefault(root, len(numbers)))
    return np.array(out, dtype=int)


def _is_mood_word(label: str) -> bool:
    from .moods import is_known
    return is_known(label)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    kind: str                 # anchor | phase_anchor | required | forbidden_pair | energy_jump | tempo_jump |
                              # key_jump | artist_gap | artist_count | same_song | duration | track_count
    severity: float
    position: int | None      # 1-based (incoming track for transition constraints)
    message: str


@dataclass
class PlanEntry:
    position: int
    track_id: str
    rel_path: str
    display: str
    artist: str | None
    title: str | None
    duration_s: float
    start_s: float
    t_rel: float
    phase: str
    target_energy: float | None
    energy: float | None
    energy_conf: float | None
    energy_source: str | None
    deviation: float | None
    bpm: float | None
    camelot: str | None
    roles: list[dict]                 # [{role, confidence, source, category}], wanted roles first
    moods: list[dict]
    anchored: str | None = None
    transition: dict | None = None    # incoming transition details
    costs: dict[str, float] = field(default_factory=dict)       # weighted contribution of each term here
    why: list[str] = field(default_factory=list)
    counterfactual: dict | None = None
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)


@dataclass
class PlanVariant:
    label: str
    preset: str
    preset_description: str
    seed: int
    order: list[str]                     # track ids
    entries: list[PlanEntry]
    objective: float                     # J under the SCENARIO weights (comparable across variants)
    objective_preset: float              # J under the variant's own weights
    terms: dict[str, float]              # weighted term contributions under the scenario weights
    raw_terms: dict[str, float]          # unweighted terms (0..1)
    violations: list[Violation]
    duration_s: float
    stats: dict[str, Any]
    tradeoffs: list[str] = field(default_factory=list)
    evaluations: int = 0
    duplicate_of: str | None = None


@dataclass
class PlanResult:
    engine: str
    scenario: str
    scenario_path: str | None
    seed: int
    weights: dict[str, float]
    variants: list[PlanVariant]
    pool_size: int
    candidates: int
    excluded: list[dict]
    notes: list[str]
    inputs: dict[str, Any]
    phases: list[dict]

    def to_dict(self) -> dict:
        return to_jsonable(self)


# ---------------------------------------------------------------------------
# Model: arrays + the objective
# ---------------------------------------------------------------------------

@dataclass
class Evaluation:
    key: float
    cost: float
    severity: float
    terms: dict[str, float]           # unweighted term values
    weighted: dict[str, float]
    violations: list[Violation] | None = None
    detail: dict | None = None


class Model:
    """Arrays over the candidate pool plus the scenario, and ``evaluate(order)``."""

    def __init__(self, contexts: Sequence[TrackContext], resolved: ResolvedScenario, *,
                 seq_cfg: SequencingConfig | None = None, profile: ClientProfile | None = None,
                 pairwise: bool = True) -> None:
        """``pairwise=False`` skips the n × n transition matrices (per-position costs only:
        used to pre-rank a large library before the real model is built)."""
        sc = self.scenario = resolved.scenario
        self.resolved = resolved
        self.ctx = list(contexts)
        n = self.n = len(self.ctx)
        if n == 0:
            raise ScenarioError(f"scenario '{sc.name}': no eligible track left to plan with")
        self.index = {c.track_id: i for i, c in enumerate(self.ctx)}
        tracks = [c.track for c in self.ctx]
        cfg, _ = effective_config(seq_cfg or SequencingConfig(), profile)
        self.seq_cfg = cfg

        self.D = np.array([max(1.0, float(t.duration_s or 0.0)) for t in tracks])
        self.E = np.array([np.nan if c.energy is None else c.energy for c in self.ctx])
        self.Ec = np.array([c.energy_conf if c.energy is not None else 0.0 for c in self.ctx])
        self.Ein = np.array([np.nan if c.energy_in is None else c.energy_in for c in self.ctx])
        self.Eout = np.array([np.nan if c.energy_out is None else c.energy_out for c in self.ctx])
        self.B = np.array([np.nan if c.bpm is None else c.bpm for c in self.ctx])
        self.Bc = np.array([c.bpm_conf for c in self.ctx])
        kn = np.full(n, -1)
        kl = np.zeros(n, dtype=int)
        for i, c in enumerate(self.ctx):
            k = parse_camelot(c.camelot) if c.camelot else None
            if k:
                kn[i], kl[i] = k[0], (0 if k[1] == "A" else 1)
        self.Kc = np.array([c.key_conf for c in self.ctx])
        self.A = _identity_groups([c.artist_ids for c in self.ctx])
        self.S = _identity_groups([c.song_ids for c in self.ctx])
        ed = [c.editorial for c in self.ctx]
        self.has_editorial = any(x is not None for x in ed)
        self.EDC = np.array([0.5 if x is None else (1.0 - (_clip01((x + 1.0) / 2.0))) for x in ed])

        # phases
        self.P = len(sc.phases)
        bounds = sc.phase_bounds()
        self.ph_lo = np.array([b[0] for b in bounds])
        self.ph_hi = np.array([b[1] for b in bounds])
        self.ph_hi_search = self.ph_hi.copy()
        self.ph_hi_search[-1] = 1.0 + 1e-9
        self.ph_e0 = np.array([np.nan if p.energy is None else p.energy[0] for p in sc.phases])
        self.ph_e1 = np.array([np.nan if p.energy is None else p.energy[1] for p in sc.phases])
        self.ph_rlo = np.array([np.nan if p.energy_range is None else p.energy_range[0] for p in sc.phases])
        self.ph_rhi = np.array([np.nan if p.energy_range is None else p.energy_range[1] for p in sc.phases])
        self.ph_contrast = np.array([bool(p.contrast) for p in sc.phases])
        self.curve_t = np.array([p[0] for p in sc.curve]) if sc.curve else None
        self.curve_e = np.array([p[1] for p in sc.curve]) if sc.curve else None

        # per-track × per-phase costs
        self.ROLEC = np.zeros((n, self.P))
        self.MOODC = np.zeros((n, self.P))
        for i, c in enumerate(self.ctx):
            for p, ph in enumerate(sc.phases):
                self.ROLEC[i, p] = self._role_cost(c, ph.roles, ph.ai_positions)
                self.MOODC[i, p] = self._mood_cost(c, ph.prefer_moods + [m for m in sc.prefer_moods
                                                                         if m not in ph.prefer_moods],
                                                   ph.avoid_moods + [m for m in sc.avoid_moods
                                                                     if m not in ph.avoid_moods])
        self.U = np.array([self._uncertainty(c) for c in self.ctx])
        if not pairwise:
            return

        # pairwise matrices
        fm = _FeatureMatrix(tracks, cfg)
        self.TT = np.vstack([fm.row(i)[0] for i in range(n)]) if n else np.zeros((0, 0))
        with np.errstate(invalid="ignore"):
            Bi, Bj = self.B[:, None], self.B[None, :]
            self.TEMPO = np.minimum(np.abs(Bj - Bi), np.minimum(np.abs(Bj - 2 * Bi), np.abs(2 * Bj - Bi)))
        span = sc.transitions.max_bpm_jump or DEFAULT_BPM_SPAN
        bconf = np.minimum(1.0, 2.0 * np.minimum(self.Bc[:, None], self.Bc[None, :]))
        self.TEMPOC = np.where(np.isnan(self.TEMPO), UNKNOWN_TEMPO_COST,
                               np.minimum(1.0, np.nan_to_num(self.TEMPO) / span) * bconf
                               + (1 - bconf) * UNKNOWN_TEMPO_COST)
        known_k = kn >= 0
        dist = np.abs(kn[:, None] - kn[None, :]) % 12
        steps = np.minimum(dist, 12 - dist) + (kl[:, None] != kl[None, :])
        self.KEYS = np.where(known_k[:, None] & known_k[None, :], steps, -1)
        hv = np.where(self.KEYS <= 1, 0.0, np.where(self.KEYS == 2, 0.5, 1.0))
        kconf = np.minimum(1.0, 2.0 * np.minimum(self.Kc[:, None], self.Kc[None, :]))
        self.HARMC = np.where(self.KEYS < 0, UNKNOWN_KEY_COST, hv * kconf + (1 - kconf) * UNKNOWN_KEY_COST)
        with np.errstate(invalid="ignore"):
            self.EDGE = np.abs(self.Ein[None, :] - self.Eout[:, None])
        mj = sc.transitions.max_energy_jump
        self.EDGEC = np.where(np.isnan(self.EDGE), UNKNOWN_EDGE_COST, np.minimum(1.0, np.nan_to_num(self.EDGE) / mj))
        self.TIMBREC = np.full((n, n), UNKNOWN_TIMBRE_COST)
        tv = [np.array(c.timbre) if c.timbre else None for c in self.ctx]
        dims = {len(v) for v in tv if v is not None}
        if len(dims) == 1:
            dim = dims.pop()
            idx = [i for i, v in enumerate(tv) if v is not None]
            M = np.vstack([np.clip(tv[i], 0.0, 1.0) for i in idx])
            dd = np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).sum(-1)) / math.sqrt(dim)
            self.TIMBREC[np.ix_(idx, idx)] = np.clip(dd, 0.0, 1.0)
        self.FLOWC = self._flow_matrix()
        self.FORBID = np.zeros((n, n), dtype=bool)
        self.ADJ = np.zeros((n, n))
        for pair in resolved.forbid_pairs:
            a, b = (self.index.get(x) for x in pair)
            if a is not None and b is not None:
                self.FORBID[a, b] = self.FORBID[b, a] = True
        for pair in resolved.avoid_pairs:
            a, b = (self.index.get(x) for x in pair)
            if a is not None and b is not None:
                self.ADJ[a, b] = self.ADJ[b, a] = 1.0
        for x, y in resolved.prefer_pairs:
            a, b = self.index.get(x), self.index.get(y)
            if a is not None and b is not None:
                self.ADJ[a, b] = -1.0

        # anchors / required
        self.fixed: dict[int, int] = {}            # position (1-based, negative from the end) -> track index
        self.phase_anchor: dict[int, int] = {}     # track index -> phase index
        for anchor, tid in resolved.anchors:
            i = self.index.get(tid)
            if i is None:
                raise ScenarioError(f"anchor '{anchor.track}' is not among the planning candidates")
            if anchor.position is not None:
                self.fixed[anchor.position] = i
            else:
                self.phase_anchor[i] = [p.name for p in sc.phases].index(anchor.phase)
        self.required = {self.index[t] for t in resolved.required if t in self.index}
        self.required |= set(self.fixed.values()) | set(self.phase_anchor)
        self.locked = set(self.fixed.values())
        self.target_s = sc.target_duration_s
        self.tol_s = sc.effective_tolerance_s

    def _flow_matrix(self) -> np.ndarray:
        """1 − mood similarity for every ordered pair (``moods.mood_similarity`` semantics),
        computed through a label × label affinity table so it stays fast for large pools."""
        n = self.n
        flow = np.full((n, n), UNKNOWN_FLOW_COST)
        labels: dict[str, int] = {}
        moods = [[(labels.setdefault(lab, len(labels)), conf) for lab, conf in c.moods.items()] for c in self.ctx]
        names = list(labels)
        aff = np.full((len(names), len(names)), -1.0)       # -1 = relation unknown
        for a, la in enumerate(names):
            for b, lb in enumerate(names):
                v = label_affinity(la, lb)
                if v is not None:
                    aff[a, b] = v
        for i in range(n):
            mi = moods[i]
            if not mi:
                continue
            for j in range(n):
                mj = moods[j]
                if not mj or i == j:
                    continue
                best = -1.0
                for a, ca in mi:
                    for b, cb in mj:
                        v = aff[a, b]
                        if v >= 0:
                            score = v * min(1.0, max(0.0, ca if ca < cb else cb))
                            if score > best:
                                best = score
                if best >= 0:
                    flow[i, j] = 1.0 - best
        return flow

    # --- per-track costs ------------------------------------------------------

    @staticmethod
    def _role_cost(c: TrackContext, wanted: Sequence[str], ai_positions: Sequence[str] = ()) -> float:
        if not wanted:
            return 0.0
        hint = max([c.phase_hints.get(a, 0.0) for a in ai_positions] + [0.0])
        if hint <= 0 and not (set(wanted) & c.role_judged):
            return UNKNOWN_ROLE_COST         # no layer could judge these roles for this track
        fit = max([c.roles.get(r, 0.0) for r in wanted] + [hint])
        return 1.0 - min(1.0, fit / ROLE_FULL)

    @staticmethod
    def _mood_cost(c: TrackContext, prefer: Sequence[str], avoid: Sequence[str]) -> float:
        if not prefer and not avoid:
            return 0.0
        if not c.moods:
            return UNKNOWN_MOOD_COST if prefer else 0.2
        s = best_affinity(c.moods, prefer) if prefer else None
        a = best_affinity(c.moods, avoid) if avoid else None
        if prefer and s is None:
            pref_cost = UNKNOWN_MOOD_COST
        else:
            pref_cost = 1.0 - min(1.0, (s[0] if s else 0.0) / MOOD_FULL) if prefer else 0.0
        avoid_cost = a[0] if a else 0.0
        if prefer and avoid:
            return 0.7 * pref_cost + 0.3 * avoid_cost
        return pref_cost if prefer else avoid_cost

    @staticmethod
    def _uncertainty(c: TrackContext) -> float:
        parts = [1.0 - c.energy_conf if c.energy is not None else 1.0,
                 1.0 - c.bpm_conf if c.bpm is not None else 1.0,
                 1.0 - min(1.0, 2.0 * c.key_conf) if c.camelot else 1.0,
                 1.0 - max(c.moods.values()) if c.moods else 1.0,
                 1.0 - min(1.0, max(c.roles.values()) / ROLE_FULL) if c.roles else 1.0]
        return float(np.mean(parts))

    # --- timeline -------------------------------------------------------------

    def timeline(self, o: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(start seconds, relative position, phase index, target energy) per position. The
        relative position is the track midpoint in TIME, or (k + 0.5) / n with
        ``curve_axis: tracks``."""
        d = self.D[o]
        total = float(d.sum())
        starts = np.cumsum(d) - d
        if self.scenario.curve_axis == "tracks":
            t = (np.arange(len(o)) + 0.5) / max(1, len(o))
        else:
            t = (starts + d / 2.0) / total if total > 0 else np.zeros(len(o))
        return starts, t, self.phase_of(t), self.target_of(t)

    def phase_of(self, t: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.ph_hi_search, t, side="right"), 0, self.P - 1)

    def target_of(self, t: np.ndarray) -> np.ndarray:
        if self.curve_t is not None:
            return np.interp(t, self.curve_t, self.curve_e)
        ph = self.phase_of(t)
        lo, hi = self.ph_lo[ph], self.ph_hi[ph]
        frac = np.clip((t - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0, 1.0)
        return self.ph_e0[ph] + (self.ph_e1[ph] - self.ph_e0[ph]) * frac

    # --- position / transition costs (vectorized, shared by search and evaluation) ----

    def position_costs(self, idx: np.ndarray, t: np.ndarray, ph: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
        e = self.E[idx]
        conf = CURVE_MIN_TRUST + (1.0 - CURVE_MIN_TRUST) * self.Ec[idx]
        known = ~np.isnan(e)
        dev = np.abs(np.where(known, e, 0.0) - target)
        curve = np.where(known, conf * np.minimum(1.0, dev / CURVE_TOLERANCE) + (1 - conf) * UNKNOWN_CURVE_COST,
                         UNKNOWN_CURVE_COST)
        rlo, rhi = self.ph_rlo[ph], self.ph_rhi[ph]
        has_r = ~np.isnan(rlo)
        ev = np.where(known, e, 0.0)
        out = np.maximum(np.maximum(np.nan_to_num(rlo) - ev, ev - np.nan_to_num(rhi, nan=1.0)), 0.0)
        band = np.where(known & has_r, conf * np.minimum(1.0, out / BAND_FALLOFF), 0.0)
        return {"curve": curve, "band": band, "role": self.ROLEC[idx, ph], "mood": self.MOODC[idx, ph],
                "editorial": self.EDC[idx] if self.has_editorial else np.zeros(len(idx)),
                "uncertainty": self.U[idx]}

    def transition_costs(self, a: np.ndarray, b: np.ndarray, contrast_entry: np.ndarray,
                         boundary: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Costs of the transitions a[k] -> b[k]. ``contrast_entry``: b enters a phase marked
        ``contrast``; ``boundary``: a and b are in different phases (moods change on purpose)."""
        edge = self.EDGEC[a, b]
        flow = self.FLOWC[a, b]
        if boundary is not None:
            flow = np.where(boundary, flow * BOUNDARY_FLOW_DISCOUNT, flow)
        return {"transition": np.where(contrast_entry, (1.0 - self.TT[a, b]) * CONTRAST_TRANSITION_DISCOUNT,
                                       1.0 - self.TT[a, b]),
                "edge_energy": np.where(contrast_entry, edge * CONTRAST_DISCOUNT, edge),
                "tempo": np.where(contrast_entry, self.TEMPOC[a, b] * CONTRAST_DISCOUNT, self.TEMPOC[a, b]),
                "harmony": self.HARMC[a, b], "timbre": self.TIMBREC[a, b],
                "mood_flow": np.where(contrast_entry, flow * CONTRAST_DISCOUNT / BOUNDARY_FLOW_DISCOUNT, flow),
                "adjacency": self.ADJ[a, b]}

    def tempo_limits(self, contrast_entry: np.ndarray) -> np.ndarray:
        """Hard tempo limit per transition: ``max_bpm_jump``, or ``contrast_bpm_jump`` when the
        transition enters a contrast phase (inf = not limited)."""
        tr = self.scenario.transitions
        base = float(tr.max_bpm_jump) if tr.max_bpm_jump else math.inf
        contrast = float(tr.contrast_bpm_jump) if tr.contrast_bpm_jump else math.inf
        return np.where(contrast_entry, contrast, base)

    # --- evaluation -------------------------------------------------------------

    def evaluate(self, order: Sequence[int], weights: Mapping[str, float], *, novelty: "Novelty | None" = None,
                 detail: bool = False) -> Evaluation:
        o = np.asarray(order, dtype=int)
        m = len(o)
        sc = self.scenario
        if m == 0:
            return Evaluation(key=math.inf, cost=math.inf, severity=math.inf, terms={}, weighted={})
        starts, t, ph, target = self.timeline(o)
        pos = self.position_costs(o, t, ph, target)
        terms: dict[str, float] = {k: float(v.mean()) for k, v in pos.items()}
        contrast_entry = np.zeros(max(0, m - 1), dtype=bool)
        if m > 1:
            a, b = o[:-1], o[1:]
            boundary = ph[1:] != ph[:-1]
            contrast_entry = self.ph_contrast[ph[1:]] & boundary
            tr = self.transition_costs(a, b, contrast_entry, boundary)
            for k, v in tr.items():
                terms[k] = float(v.mean())
        else:
            tr = {}
            for k in TRANSITION_TERMS:
                terms[k] = 0.0
        # soft diversity: same artist close together
        art = self.A[o]
        gap = sc.diversity.artist_min_gap
        window = gap + 3
        div_pos = np.zeros(m)                      # charged to the LATER track of a close pair
        for lag in range(1, min(window, m - 1) + 1):
            same = (art[lag:] == art[:-lag]) & (art[lag:] >= 0)
            div_pos[lag:] += same * (window - lag + 1) / window
        terms["diversity"] = float(div_pos.sum()) / max(1, m - 1)
        total = float(self.D[o].sum())
        dur_cost = 0.0
        if self.target_s:
            dev = abs(total - self.target_s)
            tol = max(1.0, self.tol_s or 0.0)
            dur_cost = 0.5 * min(1.0, dev / tol) if dev <= tol else 0.5 + min(0.5, 2.0 * (dev - tol) / self.target_s)
        terms["duration"] = dur_cost
        weighted = {k: float(weights.get(k, 0.0)) * terms.get(k, 0.0) for k in TERMS}
        cost = sum(weighted.values())
        if novelty is not None and m > 1:
            nov = novelty.cost(o, t)
            terms["novelty"] = nov
            weighted["novelty"] = NOVELTY_WEIGHT * nov
            cost += weighted["novelty"]

        violations: list[Violation] | None = [] if detail else None
        sev = 0.0

        def viol(kind: str, s: float, position: int | None, msg: Callable[[], str]) -> None:
            nonlocal sev
            sev += s
            if violations is not None:
                violations.append(Violation(kind, round(s, 3), position, msg()))

        # structural: fixed anchors, required
        for p, i in self.fixed.items():
            k = p - 1 if p > 0 else m + p
            if not (0 <= k < m) or o[k] != i:
                viol("anchor", ANCHOR_SEVERITY, p if p > 0 else None,
                     lambda i=i, p=p: f"anchor '{self.ctx[i].track.display_name}' is not at position {p}")
        present = set(o.tolist())
        for i in self.required - present - self.locked:          # a missing fixed anchor is reported above
            viol("required", ANCHOR_SEVERITY, None,
                 lambda i=i: f"required track '{self.ctx[i].track.display_name}' is missing")
        for i, p in self.phase_anchor.items():
            if i in present:
                k = int(np.where(o == i)[0][0])
                if ph[k] != p:
                    viol("phase_anchor", ANCHOR_SEVERITY, k + 1,
                         lambda i=i, p=p, k=k: f"'{self.ctx[i].track.display_name}' should play in phase "
                                               f"'{sc.phases[p].name}' but is in '{sc.phases[ph[k]].name}'")
        if m > 1:
            a, b = o[:-1], o[1:]
            for k in np.where(self.FORBID[a, b])[0]:
                viol("forbidden_pair", 1.0, int(k) + 2,
                     lambda k=k: f"forbidden neighbours: '{self.ctx[a[k]].track.display_name}' -> "
                                 f"'{self.ctx[b[k]].track.display_name}'")
            if sc.transitions.max_bpm_jump or sc.transitions.contrast_bpm_jump:
                lim_v = self.tempo_limits(contrast_entry)
                bj = self.TEMPO[a, b]
                sure = np.minimum(self.Bc[a], self.Bc[b]) >= HARD_LIMIT_CONFIDENCE
                for k in np.where(~np.isnan(bj) & (bj > lim_v + 1e-9) & sure)[0]:
                    viol("tempo_jump", 1.0 + (bj[k] - lim_v[k]) / lim_v[k], int(k) + 2,
                         lambda k=k: f"tempo jump {self.B[a[k]]:.0f} -> {self.B[b[k]]:.0f} BPM (folded difference "
                                     f"{bj[k]:.1f} > {lim_v[k]:g}"
                                     + (" for a planned contrast)" if contrast_entry[k] else ")"))
            ks = sc.transitions.max_key_steps
            if ks is not None:
                kk = self.KEYS[a, b]
                sure = np.minimum(self.Kc[a], self.Kc[b]) >= KEY_HARD_LIMIT_CONFIDENCE
                for k in np.where((kk > ks) & sure)[0]:
                    viol("key_jump", 1.0, int(k) + 2,
                         lambda k=k: f"key change {self.ctx[a[k]].camelot} -> {self.ctx[b[k]].camelot} "
                                     f"({kk[k]} steps > {ks})")
        # energy limits across known energies (an unknown-energy track is no bridge)
        e = self.E[o]
        kpos = np.where(~np.isnan(e))[0]
        if len(kpos) > 1:
            ek = e[kpos]
            jumps = np.abs(np.diff(ek))
            later, earlier = kpos[1:], kpos[:-1]
            entry = self.ph_contrast[ph[later]] & (ph[later] != ph[earlier])
            limit = np.where(entry, sc.transitions.contrast_energy_jump, sc.transitions.max_energy_jump)
            for k in np.where(jumps > limit + 1e-9)[0]:
                viol("energy_jump", 1.0 + (jumps[k] - limit[k]) / limit[k], int(later[k]) + 1,
                     lambda k=k: (f"energy jump {ek[k + 1] - ek[k]:+.2f} ({ek[k]:.2f} -> {ek[k + 1]:.2f}) exceeds "
                                  f"the scenario limit {limit[k]:.2f}"
                                  + (" for a planned contrast" if entry[k] else "")
                                  + (f" (across {later[k] - earlier[k] - 1} unknown-energy track(s))"
                                     if later[k] - earlier[k] > 1 else "")))
        # artist gap / count, one version per song
        for lag in range(1, min(gap, m - 1) + 1):
            same = (art[lag:] == art[:-lag]) & (art[lag:] >= 0)
            for k in np.where(same)[0]:
                viol("artist_gap", 1.0, int(k) + lag + 1,
                     lambda k=k, lag=lag: f"same artist '{self.ctx[o[k + lag]].track.artist}' with only {lag - 1} "
                                          f"track(s) in between (minimum {gap})")
        maxc = sc.diversity.max_tracks_per_artist
        if maxc and (art >= 0).any():
            counts = np.bincount(art[art >= 0])
            for aid in np.where(counts > maxc)[0]:
                viol("artist_count", float(counts[aid] - maxc), None,
                     lambda aid=aid: f"{counts[aid]} tracks by '{self._artist_label(aid, o)}' (maximum {maxc})")
        if sc.diversity.avoid_same_song:
            song = self.S[o]
            if (song >= 0).any():
                counts = np.bincount(song[song >= 0])
                for sid in np.where(counts > 1)[0]:
                    viol("same_song", float(counts[sid] - 1), None,
                         lambda sid=sid: "several versions of one song: "
                                         + ", ".join(self.ctx[i].track.display_name for i in o if self.S[i] == sid))
        if self.target_s:
            tol = self.tol_s or 0.0
            dev = total - self.target_s
            if abs(dev) > tol + 1e-6:
                viol("duration", 1.0 + (abs(dev) - tol) / max(tol, 60.0), None,
                     lambda dev=dev: f"duration {total / 60:.1f} min is {abs(dev) / 60:.1f} min "
                                     f"{'over' if dev > 0 else 'under'} the target "
                                     f"{self.target_s / 60:.1f} ± {tol / 60:.1f} min")
        if sc.max_tracks and m > sc.max_tracks:
            viol("track_count", float(m - sc.max_tracks), None, lambda: f"{m} tracks > max_tracks {sc.max_tracks}")
        if sc.min_tracks and m < sc.min_tracks:
            viol("track_count", float(sc.min_tracks - m), None, lambda: f"{m} tracks < min_tracks {sc.min_tracks}")

        ev = Evaluation(key=sev * VIOLATION_SCALE + cost, cost=cost, severity=sev, terms=terms,
                        weighted=weighted, violations=violations)
        if detail:
            ev.detail = {"starts": starts, "t": t, "phase": ph, "target": target, "pos": pos, "tr": tr,
                         "contrast_entry": contrast_entry, "total_s": total, "diversity": div_pos}
        return ev

    def _artist_label(self, aid: int, o: np.ndarray) -> str:
        for i in o:
            if self.A[i] == aid:
                return self.ctx[i].track.artist or "?"
        return "?"

    # --- structural checks for local search ---------------------------------------

    def anchors_ok(self, o: Sequence[int]) -> bool:
        m = len(o)
        for p, i in self.fixed.items():
            k = p - 1 if p > 0 else m + p
            if not (0 <= k < m) or o[k] != i:
                return False
        return True


# ---------------------------------------------------------------------------
# Novelty (alternatives must differ from earlier variants)
# ---------------------------------------------------------------------------

class Novelty:
    BUCKETS = 10

    def __init__(self, n: int, model: Model, previous: Sequence[Sequence[int]]) -> None:
        self.pos = np.zeros((n, self.BUCKETS), dtype=bool)
        self.pair = np.zeros((n, n), dtype=bool)
        for prev in previous:
            p = np.asarray(prev, dtype=int)
            if len(p) == 0:
                continue
            _, t, _, _ = model.timeline(p)
            self.pos[p, np.minimum(self.BUCKETS - 1, (t * self.BUCKETS).astype(int))] = True
            if len(p) > 1:
                self.pair[p[:-1], p[1:]] = True

    def cost(self, o: np.ndarray, t: np.ndarray) -> float:
        b = np.minimum(self.BUCKETS - 1, (t * self.BUCKETS).astype(int))
        pos = float(self.pos[o, b].mean())
        pair = float(self.pair[o[:-1], o[1:]].mean()) if len(o) > 1 else 0.0
        return 0.5 * pos + 0.5 * pair


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass
class SearchSettings:
    beam_width: int = BEAM_WIDTH
    shortlist: int = BEAM_SHORTLIST
    max_evals: int | None = None          # default: 10000 + 200 · expected length, capped at 30000
    restarts: int = 4
    patience: int = 400                   # non-improving moves before a perturbation
    jitter: float = 0.01
    # Penalty annealing: the first stages accept plans that break a hard constraint at
    # this cost per unit of severity, divided by the plan length (so the search can pass
    # through them, e.g. to rebuild an energy ramp); the last stage is lexicographic. The
    # BEST plan is always chosen lexicographically (severity first).
    penalty_schedule: tuple[float, ...] = (1.0, 4.0, 16.0)
    penalty_share: float = 0.45           # share of the budget spent in the penalty stages
    starts_improved: int = 2              # best starting points that get a share of the budget


class _Search:
    def __init__(self, model: Model, weights: Mapping[str, float], seed: int, settings: SearchSettings,
                 novelty: Novelty | None) -> None:
        self.m = model
        self.w = dict(weights)
        self.rng = np.random.default_rng(seed)
        self.s = settings
        self.nov = novelty
        self.evals = 0
        sc = model.scenario
        n = model.n
        self.mean_d = float(model.D.mean())
        if sc.use_all_tracks or (not sc.target_duration_s and not sc.max_tracks):
            self.mode, self.count = "count", n
        elif sc.target_duration_s:
            self.mode, self.count = "duration", None
        else:
            self.mode, self.count = "count", min(n, sc.max_tracks or n)
        self.expected = (self.count if self.count is not None
                         else max(2, int(round(sc.target_duration_s / self.mean_d))))
        if sc.max_tracks and self.mode == "duration":
            self.expected = min(self.expected, sc.max_tracks)
        self.max_evals = settings.max_evals or min(30000, 10000 + 200 * self.expected)
        # fixed anchors by 0-based index (positive) and the end block (negative, in order)
        self.head = {p - 1: i for p, i in model.fixed.items() if p > 0}
        self.tail = [model.fixed[p] for p in sorted((p for p in model.fixed if p < 0))]
        self.tail_d = float(sum(model.D[i] for i in self.tail))
        self.placeable = np.array([i for i in range(n) if i not in model.locked], dtype=int)

    def score(self, order: Sequence[int]) -> Evaluation:
        self.evals += 1
        return self.m.evaluate(order, self.w, novelty=self.nov)

    # --- construction -----------------------------------------------------------

    def construct(self) -> list[list[int]]:
        m, sc = self.m, self.m.scenario
        target_s = sc.target_duration_s
        beams: list[tuple[float, list[int], float]] = [(0.0, [], 0.0)]     # (cost, order, elapsed)
        complete: list[list[int]] = []
        limit = self.count if self.mode == "count" else (sc.max_tracks or m.n)
        body_slots = (limit - len(self.tail)) if self.count is not None else None
        steps = 0
        while beams and steps < m.n + 1:
            steps += 1
            children: list[tuple[float, list[int], float]] = []
            for cost, order, elapsed in beams:
                k = len(order)
                if self.mode == "count" and k >= body_slots:
                    complete.append(order + self.tail)
                    continue
                if self.mode == "duration" and target_s and elapsed + self.tail_d >= target_s - 0.5 * self.mean_d:
                    complete.append(order + self.tail)
                    if elapsed + self.tail_d >= target_s or k >= limit - len(self.tail):
                        continue
                used = set(order)
                if k in self.head:
                    cands = np.array([self.head[k]]) if self.head[k] not in used else np.array([], dtype=int)
                else:
                    cands = np.array([i for i in self.placeable if i not in used], dtype=int)
                    forced = self._forced(order, elapsed, cands)
                    if forced is not None:
                        cands = forced
                if len(cands) == 0:
                    # nothing free to place here: the remaining fixed anchors close the order
                    # (their positions may then be unreachable; evaluate reports it)
                    complete.append(order + self._pending_head(used) + self.tail)
                    continue
                inc = self._incremental(order, elapsed, cands)
                inc = inc + self.rng.uniform(0.0, self.s.jitter, len(inc))
                take = np.argsort(inc, kind="stable")[: self.s.shortlist]
                for j in take:
                    nxt = int(cands[j])
                    children.append((cost + float(inc[j]), order + [nxt], elapsed + float(m.D[nxt])))
            if not children:
                break
            children.sort(key=lambda c: c[0])
            seen: set[tuple] = set()
            beams = []
            for c in children:
                sig = (frozenset(c[1]), c[1][-1])
                if sig in seen:
                    continue
                seen.add(sig)
                beams.append(c)
                if len(beams) >= self.s.beam_width:
                    break
        for _, order, _ in beams:
            complete.append(order + self.tail)
        uniq: dict[tuple, list[int]] = {}
        for c in complete:
            if c:
                uniq.setdefault(tuple(c), list(c))
        ranked = sorted(uniq.values(), key=lambda o: self.score(o).key)
        return ranked[:3] or [self._pending_head(set()) + self.tail]

    def _pending_head(self, used: set[int]) -> list[int]:
        """Positive-position anchors not yet in the order, by position."""
        return [i for _, i in sorted(self.head.items()) if i not in used]

    def _deadline_t(self, i: int) -> float:
        p = self.m.phase_anchor.get(i)
        return float(self.m.ph_hi[p]) if p is not None else 1.0

    def _forced(self, order: list[int], elapsed: float, cands: np.ndarray) -> np.ndarray | None:
        """Required / phase-anchored tracks that must be placed now (their last chance)."""
        m = self.m
        missing = [i for i in m.required if i not in order and i not in self.tail and i not in self.head.values()]
        if not missing:
            return None
        if self.mode == "count":
            slots_left = self.count - len(self.tail) - len(order)
            if slots_left <= len(missing):
                return np.array(sorted(missing), dtype=int)
            due = [i for i in missing if i in m.phase_anchor
                   and (len(order) + 1.5) / self.count >= self._deadline_t(i)]
            return np.array(due, dtype=int) if due else None
        total = self.m.scenario.target_duration_s or float(m.D.sum())
        left = total - elapsed - self.tail_d
        need = float(sum(m.D[i] for i in missing))
        if left <= need + 0.5 * self.mean_d:
            return np.array(sorted(missing), dtype=int)
        due = [i for i in missing if i in m.phase_anchor and (elapsed + 1.5 * m.D[i]) / total >= self._deadline_t(i)]
        return np.array(due, dtype=int) if due else None

    def _incremental(self, order: list[int], elapsed: float, cands: np.ndarray) -> np.ndarray:
        m, w, sc = self.m, self.w, self.m.scenario
        k = len(order)
        by_position = self.mode == "count" or sc.curve_axis == "tracks"
        slots = max(1, self.count if self.mode == "count" else self.expected)
        if by_position:
            t = np.full(len(cands), (k + 0.5) / slots)
        else:
            total = sc.target_duration_s or float(m.D.sum())
            t = (elapsed + m.D[cands] / 2.0) / total
        t = np.clip(t, 0.0, 1.0)
        ph = m.phase_of(t)
        pos = m.position_costs(cands, t, ph, m.target_of(t))
        n_est = max(2, self.expected)
        inc = sum(w.get(name, 0.0) * v for name, v in pos.items()) / n_est
        # phase anchors: placing them inside their phase is the plan
        for j, i in enumerate(cands):
            p = m.phase_anchor.get(int(i))
            if p is not None:
                inc[j] += -2.0 / n_est if ph[j] == p else 4.0 / n_est
        if k > 0:
            a = np.full(len(cands), order[-1])
            prev_t = (k - 0.5) / slots if by_position else \
                (elapsed - m.D[order[-1]] / 2.0) / (sc.target_duration_s or float(m.D.sum()))
            prev_ph = m.phase_of(np.array([prev_t]))[0]
            entry = m.ph_contrast[ph] & (ph != prev_ph)
            tr = m.transition_costs(a, cands, entry, ph != prev_ph)
            inc = inc + sum(w.get(name, 0.0) * v for name, v in tr.items()) / max(1, n_est - 1)
            # hard limits as large penalties (the final evaluation decides)
            pen = np.zeros(len(cands))
            pen += m.FORBID[a, cands]
            if sc.transitions.max_bpm_jump or sc.transitions.contrast_bpm_jump:
                lim_v = m.tempo_limits(entry)
                bj = m.TEMPO[a, cands]
                sure = np.minimum(m.Bc[a], m.Bc[cands]) >= HARD_LIMIT_CONFIDENCE
                pen += np.where(~np.isnan(bj) & (bj > lim_v) & sure, 1.0, 0.0)
            if sc.transitions.max_key_steps is not None:
                sure = np.minimum(m.Kc[a], m.Kc[cands]) >= KEY_HARD_LIMIT_CONFIDENCE
                pen += (m.KEYS[a, cands] > sc.transitions.max_key_steps) & sure
            last_e = next((m.E[i] for i in reversed(order) if not np.isnan(m.E[i])), np.nan)
            if not np.isnan(last_e):
                lim_e = np.where(entry, sc.transitions.contrast_energy_jump, sc.transitions.max_energy_jump)
                ec = m.E[cands]
                pen += np.where(~np.isnan(ec) & (np.abs(ec - last_e) > lim_e), 1.0, 0.0)
            gap = sc.diversity.artist_min_gap
            if gap:
                recent = {int(m.A[i]) for i in order[-gap:] if m.A[i] >= 0}
                pen += np.isin(m.A[cands], list(recent)) & (m.A[cands] >= 0)
            if sc.diversity.avoid_same_song:
                songs = {int(m.S[i]) for i in order if m.S[i] >= 0}
                pen += np.isin(m.S[cands], list(songs)) & (m.S[cands] >= 0)
            maxc = sc.diversity.max_tracks_per_artist
            if maxc:
                arts = m.A[np.array(order)]
                cnt = {a_: int((arts == a_).sum()) for a_ in set(arts.tolist()) if a_ >= 0}
                pen += np.array([cnt.get(int(m.A[c]), 0) >= maxc if m.A[c] >= 0 else False for c in cands])
            inc = inc + 10.0 * pen
        return inc

    # --- local search -------------------------------------------------------------

    def polish(self, order: list[int], budget: int | None = None) -> list[int]:
        """Apply the best single replacement / swap per slot while it lowers this search's own
        objective (the neighbourhood the report's counterfactuals examine), so a variant never
        ships a one-move improvement of its own objective that the report would then point at."""
        m = self.m
        order = list(order)
        if len(order) < 2:
            return order
        budget = budget if budget is not None else max(2000, self.max_evals // 4)
        start = self.evals

        def detailed(o: list[int]) -> Evaluation:      # slot times / phases rank the replacements
            self.evals += 1
            return m.evaluate(o, self.w, novelty=self.nov, detail=True)

        cur = detailed(order)
        improved = True
        while improved and self.evals - start < budget:
            improved = False
            for k in range(len(order)):
                if order[k] in m.locked or self.evals - start >= budget:
                    continue
                best = None
                for cand in _single_moves(m, order, k, self.w, cur):
                    e2 = self.score(cand)
                    if _better(e2, cur) and (best is None or _better(e2, best[1])):
                        best = (cand, e2)
                if best is not None:
                    order, cur = best[0], detailed(best[0])
                    improved = True
        return order

    def improve(self, order: list[int], budget: int | None = None) -> tuple[list[int], Evaluation]:
        """Iterated local search from ``order`` within ``budget`` evaluations: penalty stages
        (hard constraints priced, so the walk may cross infeasible plans), then a
        lexicographic stage with perturbation restarts. Returns the lexicographically best plan."""
        m = self.m
        stop = self.evals + (budget if budget is not None else self.max_evals - self.evals)
        best = list(order)
        best_ev = self.score(best)
        cur, cur_ev = list(best), best_ev
        stages = list(self.s.penalty_schedule) + [None]
        pen_budget = int((stop - self.evals) * self.s.penalty_share)
        per_pen = pen_budget // max(1, len(stages) - 1)

        def value(ev: Evaluation, lam: float | None) -> float:
            return ev.key if lam is None else ev.cost + lam * ev.severity

        # J is a mean over positions: one position moves it by ~1/m, so the price of one unit of
        # severity is scaled the same way (otherwise no stepping-stone move is ever affordable)
        scale = 1.0 / max(1, len(order))
        stages = [None if lam is None else lam * scale for lam in stages]
        for lam in stages:
            stage_stop = stop if lam is None else min(stop, self.evals + per_pen)
            restarts = self.s.restarts if lam is None else 0
            cur_val = value(cur_ev, lam)
            rounds = 0
            while self.evals < stage_stop:
                stale = 0
                while self.evals < stage_stop and stale < self.s.patience:
                    cand = self._repair_move(cur, cur_ev) if cur_ev.severity > 0 and self.rng.random() < 0.5 \
                        else self._move(cur)
                    if cand is None or not m.anchors_ok(cand):
                        stale += 1
                        continue
                    ev = self.score(cand)
                    v = value(ev, lam)
                    if v < cur_val - 1e-12:
                        cur, cur_ev, cur_val = cand, ev, v
                        stale = 0
                        if ev.key < best_ev.key - 1e-12:
                            best, best_ev = list(cand), ev
                    else:
                        stale += 1
                rounds += 1
                if rounds > restarts or self.evals >= stage_stop:
                    break
                cur = self._perturb(best)
                cur_ev = self.score(cur)
                cur_val = value(cur_ev, lam)
            if lam is not None and best_ev.key < cur_ev.key:
                cur, cur_ev = list(best), best_ev       # each stage starts from the best plan so far
        return best, best_ev

    def search(self, extra_starts: Sequence[Sequence[int]] = ()) -> tuple[list[int], Evaluation]:
        """Starting points (beam constructions + the curve assignment + ``extra_starts``, e.g.
        the earlier variants), the best few improved with an equal share of the budget; the
        lexicographically best result wins — so a later variant never ends with more
        hard-constraint violations than an earlier one it started from."""
        starts = self.construct()
        assigned = self.assignment_start()
        if assigned and assigned not in starts:
            starts.append(assigned)
        extra = [list(o) for o in extra_starts if o and self.m.anchors_ok(o)]
        ranked = sorted(starts + [o for o in extra if o not in starts], key=lambda o: self.score(o).key)
        chosen = ranked[: max(1, self.s.starts_improved)]
        if assigned and assigned not in chosen:
            chosen[-1] = assigned                   # the curve-driven start is always explored
        for o in extra:                             # ... and so are the earlier variants
            if o not in chosen:
                chosen.append(o)
        best_o, best_ev = None, None
        for k, start in enumerate(chosen):
            share = (self.max_evals - self.evals) // (len(chosen) - k)
            o, ev = self.improve(start, budget=share)
            if best_ev is None or ev.key < best_ev.key:
                best_o, best_ev = o, ev
        return best_o, best_ev

    def assignment_start(self) -> list[int] | None:
        """Curve-driven start: split the plan into equal-time slots, then assign tracks to slots
        by minimum total position cost (curve, band, role, mood, editorial, certainty) —
        ``scipy.optimize.linear_sum_assignment``. Fixed anchors keep their slots; required and
        phase-anchored tracks are forced in. Transitions are left to the local search."""
        from scipy.optimize import linear_sum_assignment
        m, sc = self.m, self.m.scenario
        if self.mode == "count":
            n_slots = self.count
        else:
            n_slots = max(1, int(round(sc.target_duration_s / self.mean_d)))
            if sc.max_tracks:
                n_slots = min(n_slots, sc.max_tracks)
        n_slots = min(n_slots, m.n)
        if n_slots < 1:
            return None
        t = (np.arange(n_slots) + 0.5) / n_slots
        ph, tgt = m.phase_of(t), m.target_of(t)
        fixed: dict[int, int] = {}
        for p, i in m.fixed.items():
            k = p - 1 if p > 0 else n_slots + p
            if 0 <= k < n_slots:
                fixed[k] = i
        free_slots = [k for k in range(n_slots) if k not in fixed]
        cand = [i for i in range(m.n) if i not in m.locked]
        # one version per song in the starting point (a required version if any, else the first
        # in pool order): the others would only be a guaranteed 'same_song' violation
        if m.scenario.diversity.avoid_same_song:
            locked_songs = {int(m.S[i]) for i in m.locked if m.S[i] >= 0}
            by_song: dict[int, list[int]] = {}
            for i in cand:
                if m.S[i] >= 0:
                    by_song.setdefault(int(m.S[i]), []).append(i)
            drop: set[int] = set()
            for sid, members in by_song.items():
                keep = [] if sid in locked_songs else [min(members, key=lambda i: (i not in m.required, i))]
                drop |= {i for i in members if i not in keep and i not in m.required}
            cand = [i for i in cand if i not in drop]
        if not free_slots or not cand:
            return [fixed[k] for k in sorted(fixed)] or None
        ci = np.array(cand)
        C = np.zeros((len(cand), len(free_slots)))
        for j, k in enumerate(free_slots):
            pos = m.position_costs(ci, np.full(len(ci), t[k]), np.full(len(ci), ph[k]), np.full(len(ci), tgt[k]))
            C[:, j] = sum(self.w.get(name, 0.0) * v for name, v in pos.items())
        slot_phase = ph[np.array(free_slots)]
        for r, i in enumerate(cand):
            p = m.phase_anchor.get(i)
            if p is not None:
                C[r, slot_phase != p] += 1e3
            if i in m.required:
                C[r, :] -= 1e3
        if len(cand) < len(free_slots):
            C = C.T
            cols, rows = linear_sum_assignment(C)
        else:
            rows, cols = linear_sum_assignment(C)
        slots: list[int | None] = [None] * n_slots
        for k, i in fixed.items():
            slots[k] = i
        for r, c in zip(rows, cols):
            slots[free_slots[int(c)]] = cand[int(r)]
        order = [i for i in slots if i is not None]
        return order if m.anchors_ok(order) else None

    def _repair_move(self, o: list[int], ev: Evaluation, depth: int = 0) -> list[int] | None:
        """A move aimed at a broken hard constraint of ``o`` (the 2.0 sequencer's repair idea):
        relocate / swap / replace a track at a violating position, bridge a tempo / energy
        jump, drop or replace a second version of a song or an artist's extra track, grow or
        shrink towards the duration. When the step does not lower the violation severity
        (e.g. dropping a duplicate opens a new tempo jump), up to two follow-up repairs are
        chained into one compound move."""
        c = self._repair_step(o, ev)
        if c is None or depth >= 2 or not self.m.anchors_ok(c):
            return c
        ec = self.m.evaluate(c, self.w, detail=True)
        self.evals += 1
        if ec.severity >= ev.severity - 1e-9 and ec.violations:
            follow = self._repair_move(c, ec, depth + 1)
            if follow is not None:
                return follow
        return c

    def _repair_step(self, o: list[int], ev: Evaluation) -> list[int] | None:
        m, rng = self.m, self.rng
        if ev.violations is None:
            ev = m.evaluate(o, self.w, detail=True)
            self.evals += 1
        if not ev.violations:
            return None
        v = ev.violations[int(rng.integers(0, len(ev.violations)))]
        free = self._free_positions(o)
        used = set(o)
        unused = [i for i in self.m_unused_ranked(o, used)]
        targets: list[int] = []
        if v.position is not None:
            targets = [k for k in (v.position - 1, v.position - 2) if 0 <= k < len(o) and o[k] not in m.locked]
        elif v.kind == "same_song":
            songs = m.S[np.asarray(o)]
            dup = [sid for sid in set(songs.tolist()) if sid >= 0 and (songs == sid).sum() > 1]
            targets = [k for k in range(len(o)) if songs[k] in dup and o[k] not in m.locked]
        elif v.kind == "artist_count":
            arts = m.A[np.asarray(o)]
            counts = {a: int((arts == a).sum()) for a in set(arts.tolist()) if a >= 0}
            limit = m.scenario.diversity.max_tracks_per_artist or 0
            targets = [k for k in range(len(o)) if counts.get(int(arts[k]), 0) > limit and o[k] not in m.locked]
        elif v.kind == "duration" and self.mode == "duration":
            total = float(m.D[np.asarray(o)].sum())
            if total > (m.target_s or 0) and self._can_shrink(o):
                removable = [k for k in free if o[k] not in m.required]
                if removable:
                    c = list(o)
                    c.pop(int(rng.choice(removable)))
                    return c
            elif unused and self._can_grow(o):
                c = list(o)
                c.insert(int(rng.integers(0, len(o) + 1)), int(rng.choice(unused[: max(8, len(unused) // 3)])))
                return c
            return None
        elif v.kind == "required":
            missing = [i for i in m.required if i not in used and i not in m.locked]
            if missing:
                c = list(o)
                k = int(rng.choice(free)) if free else len(c)
                if self.mode == "count" and free:
                    c[k] = int(rng.choice(missing))
                else:
                    c.insert(k, int(rng.choice(missing)))
                return c
            return None
        if v.kind in ("tempo_jump", "energy_jump") and v.position is not None and rng.random() < 0.5:
            bridged = self._bridge(o, v.position - 1, used)
            if bridged is not None:
                return bridged
        if not targets:
            return None
        k = int(rng.choice(targets))
        r = rng.random()
        c = list(o)
        if r < 0.35 and len(free) >= 2:                     # relocate the offending track
            x = c.pop(k)
            c.insert(int(rng.integers(0, len(c) + 1)), x)
            return c
        if r < 0.65 and len(free) >= 2:                     # swap it with another free position
            k2 = int(rng.choice(free))
            c[k], c[k2] = c[k2], c[k]
            return c
        if r < 0.9 and unused and o[k] not in m.required:   # replace it
            c[k] = int(rng.choice(unused[: max(8, len(unused) // 2)]))
            return c
        if self._can_shrink(o) and o[k] not in m.required:  # or drop it
            c.pop(k)
            return c
        return None

    def _bridge(self, o: list[int], k: int, used: set[int]) -> list[int] | None:
        """Put a stepping stone between o[k-1] and o[k]: a track whose energy and tempo lie
        between theirs, moved from elsewhere in the plan or taken from the unused pool."""
        m, rng = self.m, self.rng
        if not 1 <= k < len(o):
            return None
        a, b = o[k - 1], o[k]
        lo_e, hi_e = sorted((m.E[a], m.E[b]))
        candidates = []
        for i in range(m.n):
            if i in (a, b) or i in m.locked:
                continue
            e, bpm = m.E[i], m.B[i]
            if not np.isnan(lo_e) and not (np.isnan(e) or lo_e <= e <= hi_e):
                continue
            ta, tb = m.TEMPO[a, i], m.TEMPO[i, b]
            if not (np.isnan(ta) or np.isnan(tb)) and max(ta, tb) >= m.TEMPO[a, b]:
                continue
            candidates.append(i)
        if not candidates:
            return None
        i = int(rng.choice(candidates))
        c = list(o)
        if i in used:
            j = c.index(i)
            if j in (k - 1, k):
                return None
            c.pop(j)
            c.insert(k if j > k else k - 1, i)
            return c
        if self.mode == "count":
            free = [j for j in self._free_positions(o) if j not in (k - 1, k) and o[j] not in m.required]
            if not free:
                return None
            j = int(rng.choice(free))           # keep the count: drop another track, insert the bridge
            c.pop(j)
            c.insert(k if j > k else k - 1, i)
            return c
        if not self._can_grow(o):
            return None
        c.insert(k, i)
        return c

    def _free_positions(self, o: list[int]) -> list[int]:
        return [k for k, i in enumerate(o) if i not in self.m.locked]

    def _move(self, o: list[int]) -> list[int] | None:
        m, rng = self.m, self.rng
        free = self._free_positions(o)
        if not free:
            return None
        used = set(o)
        unused = [int(i) for i in self.m_unused_ranked(o, used)]
        r = rng.random()
        if r < 0.22 and len(free) >= 2:                     # swap
            a, b = rng.choice(free, 2, replace=False)
            c = list(o)
            c[a], c[b] = c[b], c[a]
            return c
        if r < 0.38 and len(free) >= 2:                     # relocate
            a = int(rng.choice(free))
            b = int(rng.choice(free))
            if a == b:
                return None
            c = list(o)
            x = c.pop(a)
            c.insert(b, x)
            return c
        if r < 0.48 and len(o) >= 4:                        # relocate a block of 2-4 tracks
            size = int(rng.integers(2, 5))
            a = int(rng.integers(0, max(1, len(o) - size + 1)))
            if any(o[k] in m.locked for k in range(a, min(len(o), a + size))):
                return None
            block, rest = o[a:a + size], o[:a] + o[a + size:]
            b = int(rng.integers(0, len(rest) + 1))
            return rest[:b] + block + rest[b:]
        if r < 0.54 and len(o) >= 4:                        # swap two blocks of equal length
            size = int(rng.integers(2, 4))
            if len(o) < 2 * size:
                return None
            a = int(rng.integers(0, len(o) - 2 * size + 1))
            b = int(rng.integers(a + size, len(o) - size + 1))
            if any(o[k] in m.locked for k in list(range(a, a + size)) + list(range(b, b + size))):
                return None
            return o[:a] + o[b:b + size] + o[a + size:b] + o[a:a + size] + o[b + size:]
        if r < 0.64 and len(free) >= 3:                     # reverse a block of free positions
            a, b = sorted(rng.choice(free, 2, replace=False))
            if any(o[k] in m.locked for k in range(a, b + 1)):
                return None
            return o[:a] + o[a:b + 1][::-1] + o[b + 1:]
        if r < 0.86 and unused:                             # replace
            a = int(rng.choice(free))
            if o[a] in m.required:
                return None
            c = list(o)
            c[a] = int(rng.choice(unused[: max(8, len(unused) // 3)]))
            return c
        if r < 0.94 and unused and self._can_grow(o):       # insert
            b = int(rng.integers(0, len(o) + 1))
            c = list(o)
            c.insert(b, int(rng.choice(unused[: max(8, len(unused) // 3)])))
            return c
        if self._can_shrink(o):                             # remove
            removable = [k for k in free if o[k] not in m.required]
            if not removable:
                return None
            c = list(o)
            c.pop(int(rng.choice(removable)))
            return c
        return None

    def m_unused_ranked(self, o: list[int], used: set[int]) -> list[int]:
        cache = getattr(self, "_rank_cache", None)
        if cache is None:
            m = self.m
            t = (np.arange(m.P) + 0.5) / m.P
            best = np.full(m.n, np.inf)
            for p in range(m.P):
                tp = np.full(m.n, (m.ph_lo[p] + m.ph_hi[p]) / 2.0)
                pos = m.position_costs(np.arange(m.n), tp, np.full(m.n, p), m.target_of(tp))
                cost = sum(self.w.get(k, 0.0) * v for k, v in pos.items())
                best = np.minimum(best, cost)
            del t
            cache = self._rank_cache = [int(i) for i in np.argsort(best, kind="stable")]
        return [i for i in cache if i not in used and i not in self.m.locked]

    def _can_grow(self, o: list[int]) -> bool:
        sc = self.m.scenario
        if self.mode == "count":
            return False
        if sc.max_tracks and len(o) >= sc.max_tracks:
            return False
        return True

    def _can_shrink(self, o: list[int]) -> bool:
        sc = self.m.scenario
        if self.mode == "count" or len(o) <= 2:
            return False
        if sc.min_tracks and len(o) <= sc.min_tracks:
            return False
        return True

    def _perturb(self, o: list[int]) -> list[int]:
        c = list(o)
        free = self._free_positions(c)
        for _ in range(max(2, len(free) // 6)):
            if len(free) < 2:
                break
            a, b = self.rng.choice(free, 2, replace=False)
            c[a], c[b] = c[b], c[a]
        return c


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def preset_weights(base: Mapping[str, float], preset: str) -> dict[str, float]:
    mult = PRESETS.get(preset, {})
    return {k: float(base.get(k, 0.0)) * mult.get(k, 1.0) for k in DEFAULT_WEIGHTS}


def select_candidates(contexts: list[TrackContext], resolved: ResolvedScenario, *, cap: int = POOL_CAP,
                      notes: list[str] | None = None) -> list[TrackContext]:
    """Keep the n × n matrices bounded: anchors and required tracks always stay; the rest
    are ranked by their best phase fit (curve, role, mood). The cut is reported."""
    if len(contexts) <= cap:
        return contexts
    must = {tid for _, tid in resolved.anchors} | set(resolved.required)
    probe = Model(contexts, dataclasses.replace(resolved, forbid_pairs=set(), avoid_pairs=set(),
                                                prefer_pairs=set(), anchors=[], required=[]), pairwise=False)
    best = np.full(probe.n, np.inf)
    w = resolved.scenario.weights
    for p in range(probe.P):
        tp = np.full(probe.n, (probe.ph_lo[p] + probe.ph_hi[p]) / 2.0)
        pos = probe.position_costs(np.arange(probe.n), tp, np.full(probe.n, p), probe.target_of(tp))
        best = np.minimum(best, sum(w.get(k, 0.0) * v for k, v in pos.items()))
    order = np.argsort(best, kind="stable")
    keep = {contexts[i].track_id for i in order[:cap]} | must
    if notes is not None:
        notes.append(f"candidate pool reduced from {len(contexts)} to {len(keep)} tracks by phase fit "
                     f"(the n x n transition matrices are capped at {cap})")
    return [c for c in contexts if c.track_id in keep]


def plan(contexts: Sequence[TrackContext], resolved: ResolvedScenario, *, seed: int = 0, alternatives: int | None = None,
         seq_cfg: SequencingConfig | None = None, profile: ClientProfile | None = None,
         settings: SearchSettings | None = None) -> PlanResult:
    """Plan ``alternatives`` variants (default: the scenario's) for the resolved scenario."""
    sc = resolved.scenario
    notes = list(resolved.notes)
    pool_ctx = [c for c in contexts if c.track_id not in resolved.excluded]
    if sc.use_all_tracks and len(pool_ctx) > POOL_CAP:
        raise ScenarioError(f"scenario '{sc.name}': use_all_tracks would place all {len(pool_ctx)} eligible tracks, "
                            f"but the planner handles at most {POOL_CAP} candidates; set max_tracks or "
                            f"target_duration_min, or narrow the pool (pool.include_tags / pool.tracks)")
    cands = select_candidates(list(pool_ctx), resolved, notes=notes)
    model = Model(cands, resolved, seq_cfg=seq_cfg, profile=profile)
    settings = settings or SearchSettings()
    k = max(1, min(len(PRESET_ORDER), alternatives or sc.alternatives))
    base = {name: float(sc.weights.get(name, 0.0)) for name in DEFAULT_WEIGHTS}
    variants: list[PlanVariant] = []
    orders: list[list[int]] = []
    for v in range(k):
        preset = PRESET_ORDER[v]
        weights = preset_weights(base, preset)
        novelty = Novelty(model.n, model, orders) if orders else None
        best_order, evals, dup = None, 0, None
        for attempt in range(2):
            search = _Search(model, weights, seed + 7919 * v + 104729 * attempt, settings, novelty)
            best_order, _ = search.search(extra_starts=orders)
            best_order = search.polish(best_order)
            evals += search.evals
            dup = next((variants[i].label for i, prev in enumerate(orders) if prev == best_order), None)
            if dup is None:
                break
        orders.append(list(best_order))
        variants.append(_variant(model, best_order, base, weights, preset, chr(ord("A") + v), seed, evals, dup,
                                 novelty))
    _tradeoffs(variants)
    paths = {c.track_id: c.track.rel_path for c in contexts}
    excluded = [{"track_id": tid, "rel_path": resolved.rel_paths.get(tid) or paths.get(tid), "reason": why}
                for tid, why in resolved.excluded.items()]
    inputs = {
        "tracks": len(contexts),
        "with_segments": sum(1 for c in cands if c.energy_in is not None or c.trajectory is not None),
        "with_ai": sum(1 for c in cands if any(s.startswith("ai:") for s in c.interpretation_sources)),
        "with_external_facts": sum(1 for c in cands if c.facts),
        "with_curated_mood": sum(1 for c in cands if "manual" in c.mood_sources.values()),
        "energy_unknown": sum(1 for c in cands if c.energy is None),
    }
    phases = [{"name": p.name, "start": round(lo, 4), "end": round(hi, 4),
               "energy": list(p.energy) if p.energy else None,
               "energy_range": list(p.energy_range) if p.energy_range else None, "roles": p.roles,
               "prefer_moods": p.prefer_moods, "avoid_moods": p.avoid_moods, "contrast": p.contrast}
              for p, (lo, hi) in zip(sc.phases, sc.phase_bounds())]
    return PlanResult(engine=ENGINE_VERSION, scenario=sc.name,
                      scenario_path=str(sc.source_path) if sc.source_path else None, seed=seed, weights=base,
                      variants=variants, pool_size=len(pool_ctx), candidates=model.n, excluded=excluded, notes=notes,
                      inputs=inputs, phases=phases)


def evaluate_order(model: Model, track_ids: Sequence[str], weights: Mapping[str, float] | None = None) -> Evaluation:
    """Re-score a stored plan (deterministic verification of a plan file)."""
    missing = [t for t in track_ids if t not in model.index]
    if missing:
        raise ScenarioError(f"plan references {len(missing)} track(s) that are not planning candidates: "
                            + ", ".join(missing[:5]))
    w = weights or {k: float(model.scenario.weights.get(k, 0.0)) for k in DEFAULT_WEIGHTS}
    return model.evaluate([model.index[t] for t in track_ids], w, detail=True)


# ---------------------------------------------------------------------------
# Explanations (built from the evaluated terms only)
# ---------------------------------------------------------------------------

def _variant(model: Model, order: list[int], base: dict, weights: dict, preset: str, label: str, seed: int,
             evals: int, dup: str | None, novelty: Novelty | None = None) -> PlanVariant:
    ev = model.evaluate(order, base, detail=True)
    own = model.evaluate(order, weights)
    searched_with_base = novelty is None and all(abs(weights.get(k, 0.0) - base.get(k, 0.0)) < 1e-12 for k in base)
    entries = explain(model, order, base, ev, preset=preset,
                      own=None if searched_with_base else (lambda o: model.evaluate(o, weights, novelty=novelty)))
    det = ev.detail
    known = ~np.isnan(model.E[np.asarray(order)])
    devs = np.abs(model.E[np.asarray(order)] - det["target"])[known]
    trans = [e.transition["score"] for e in entries if e.transition and e.transition.get("score") is not None]
    stats = {
        "tracks": len(order), "duration_s": round(det["total_s"], 1),
        "target_duration_s": model.scenario.target_duration_s,
        "curve_mean_abs_dev": round(float(devs.mean()), 4) if devs.size else None,
        "curve_max_abs_dev": round(float(devs.max()), 4) if devs.size else None,
        "curve_coverage": round(float(known.mean()), 3),
        "transition_mean": round(float(np.mean(trans)), 4) if trans else None,
        "transition_min": round(float(np.min(trans)), 4) if trans else None,
        "artists": len({model.A[i] for i in order if model.A[i] >= 0}),
        "violations": len(ev.violations or []),
        "data": {cat: sum(1 for e in entries for p in e.provenance if p["category"] == cat)
                 for cat in ("measured", "file_tags", "curator", "external", "derived", "ai")},
    }
    return PlanVariant(label=label, preset=preset, preset_description=PRESET_DESCRIPTIONS[preset], seed=seed,
                       order=[model.ctx[i].track_id for i in order], entries=entries, objective=round(ev.cost, 6),
                       objective_preset=round(own.cost, 6),
                       terms={k: round(v, 6) for k, v in ev.weighted.items() if k in TERMS},
                       raw_terms={k: round(v, 6) for k, v in ev.terms.items() if k in TERMS},
                       violations=ev.violations or [], duration_s=round(det["total_s"], 1), stats=stats,
                       evaluations=evals, duplicate_of=dup)


def explain(model: Model, order: list[int], weights: Mapping[str, float], ev: Evaluation | None = None,
            *, counterfactuals: bool = True, own: Callable[[list[int]], Evaluation] | None = None,
            preset: str | None = None) -> list[PlanEntry]:
    sc = model.scenario
    ev = ev or model.evaluate(order, weights, detail=True)
    det = ev.detail
    o = np.asarray(order, dtype=int)
    m = len(o)
    n_pos, n_tr = max(1, m), max(1, m - 1)
    viol_at: dict[int, list[str]] = {}
    for v in ev.violations or []:
        if v.position:
            viol_at.setdefault(v.position, []).append(v.message)
    anchor_pos = {i: p for p, i in model.fixed.items()}
    entries: list[PlanEntry] = []
    for k, i in enumerate(o):
        c = model.ctx[i]
        t = c.track
        p = int(det["phase"][k])
        phase = sc.phases[p]
        target = float(det["target"][k]) if not np.isnan(det["target"][k]) else None
        dev = (c.energy - target) if (c.energy is not None and target is not None) else None
        costs = {name: round(float(weights.get(name, 0.0)) * float(det["pos"][name][k]) / n_pos, 5)
                 for name in POSITION_TERMS}
        trans = None
        why: list[str] = []
        warnings: list[str] = []
        if k > 0:
            a = o[k - 1]
            prev = model.ctx[a]
            ts = transition_score(prev.track, t, model.seq_cfg)
            entry = bool(det["contrast_entry"][k - 1])
            tr_costs = {name: round(float(weights.get(name, 0.0)) * float(det["tr"][name][k - 1]) / n_tr, 5)
                        for name in det["tr"]}
            costs.update(tr_costs)
            costs["diversity"] = round(float(weights.get("diversity", 0.0)) * float(det["diversity"][k]) / n_tr, 5)
            bpm_diff = None if np.isnan(model.TEMPO[a, i]) else round(float(model.TEMPO[a, i]), 1)
            steps = int(model.KEYS[a, i]) if model.KEYS[a, i] >= 0 else None
            e_jump = (round(c.energy - prev.energy, 3) if c.energy is not None and prev.energy is not None else None)
            trans = {"from": prev.track.display_name, "score": round(ts.total, 4),
                     "components": {k2: (None if v is None else round(v, 3)) for k2, v in ts.components.items()},
                     "coverage": round(ts.coverage, 3), "flags": list(ts.flags), "bpm_diff": bpm_diff,
                     "key_steps": steps, "energy_jump": e_jump,
                     "edge_energy": (None if np.isnan(model.EDGE[a, i]) else round(float(model.EDGE[a, i]), 3)),
                     "planned_contrast": entry}
            parts = [f"transition {ts.total:.2f} (coverage {ts.coverage:.0%})"]
            if bpm_diff is not None:
                parts.append(f"tempo {prev.bpm:.0f}->{c.bpm:.0f} BPM (folded diff {bpm_diff:g})")
            if steps is not None:
                parts.append(f"key {prev.camelot}->{c.camelot} ({steps} step{'s' if steps != 1 else ''})")
            if e_jump is not None:
                parts.append(f"energy {e_jump:+.2f}")
            why.append("from previous: " + ", ".join(parts))
            if entry:
                bpm_rule = (f"tempo limit {sc.transitions.contrast_bpm_jump:g} BPM" if sc.transitions.contrast_bpm_jump
                            else "tempo not limited")
                why.append(f"planned contrast: entering phase '{phase.name}' (contrast allowed: energy-jump limit "
                           f"{sc.transitions.contrast_energy_jump:.2f}, {bpm_rule})")
            lim = float(model.tempo_limits(np.array([entry]))[0])
            if math.isfinite(lim) and bpm_diff is not None and bpm_diff > lim and min(prev.bpm_conf, c.bpm_conf) < HARD_LIMIT_CONFIDENCE:
                warnings.append(f"tempo difference {bpm_diff:g} above the scenario limit {lim:g}, but the BPM "
                                f"confidence is low — not enforced, listen")
            ks = sc.transitions.max_key_steps
            if ks is not None and steps is not None and steps > ks and min(prev.key_conf, c.key_conf) < KEY_HARD_LIMIT_CONFIDENCE:
                warnings.append(f"key change of {steps} steps above the scenario limit {ks}, but the keys are "
                                f"low-confidence automatic estimates — not enforced")
            if ts.coverage < 0.3:
                warnings.append("transition rests on little measured data (coverage "
                                f"{ts.coverage:.0%}: {', '.join(f for f in ts.flags if f.startswith('unknown:'))})")
        if i in anchor_pos:
            why.insert(0, f"hard anchor: position {anchor_pos[i]} (scenario)")
        if i in model.phase_anchor:
            why.insert(0, f"anchored to phase '{sc.phases[model.phase_anchor[i]].name}' (scenario)")
        if i in model.required and i not in anchor_pos and i not in model.phase_anchor:
            why.insert(0, "required by the scenario")
        if target is not None:
            if c.energy is None:
                why.append(f"phase '{phase.name}' target energy {target:.2f}; track energy unknown — curve fit "
                           f"not verified")
            else:
                why.append(f"phase '{phase.name}' target energy {target:.2f}; track {c.energy:.2f} "
                           f"({provenance_category(c.energy_source)}, confidence {c.energy_conf:.1f}) -> deviation "
                           f"{dev:+.2f}")
        wanted = phase.roles
        role_list = []
        for r, conf in sorted(c.roles.items(), key=lambda kv: (kv[0] not in wanted, -kv[1])):
            src = c.role_sources.get(r)
            role_list.append({"role": r, "confidence": conf, "source": src, "category": provenance_category(src),
                              "wanted": r in wanted})
        if wanted:
            hits = [f"{r['role']} {r['confidence']:.2f} ({r['category']})" for r in role_list if r["wanted"]]
            hints = [f"AI position hint '{a}' {c.phase_hints[a]:.2f} (ai)" for a in phase.ai_positions
                     if c.phase_hints.get(a, 0.0) > 0]
            if hits or hints:
                found = f"track: {', '.join(hits + hints)}"
            elif not set(wanted) & c.role_judged:
                found = "no evidence about these roles for this track (unknown)"
            else:
                found = "track has no matching role"
            why.append(f"phase roles wanted: {', '.join(wanted)}; {found}")
        mood_list = [{"label": lab, "confidence": conf, "source": c.mood_sources.get(lab),
                      "category": provenance_category(c.mood_sources.get(lab))}
                     for lab, conf in sorted(c.moods.items(), key=lambda kv: -kv[1])]
        prefer = phase.prefer_moods + [x for x in sc.prefer_moods if x not in phase.prefer_moods]
        if prefer:
            hit = best_affinity(c.moods, prefer) if c.moods else None
            why.append(f"moods wanted: {', '.join(prefer)}; " + (
                f"best match {hit[1]}~{hit[2]} ({hit[0]:.2f}, "
                f"{provenance_category(c.mood_sources.get(hit[1]))})" if hit else "track moods unknown"))
        if c.energy is None:
            warnings.append("energy unknown")
        stab = _num((c.harmony or {}).get("stability"))
        if stab is not None and stab < 0.6:
            warnings.append(f"key unstable over time (stability {stab:.2f}, "
                            f"{(c.harmony or {}).get('modulations') or 0} modulation candidate(s)) — "
                            f"harmonic transitions are uncertain")
        for lc in c.low_confidence:
            warnings.append(f"low-confidence {lc}")
        warnings.extend(c.tempo_notes)
        if c.moods and all(provenance_category(s) == "ai" for s in c.mood_sources.values()):
            warnings.append("moods come only from AI interpretation")
        entry = PlanEntry(position=k + 1, track_id=c.track_id, rel_path=t.rel_path, display=t.display_name,
                          artist=t.artist, title=t.title, duration_s=round(float(model.D[i]), 2),
                          start_s=round(float(det["starts"][k]), 2), t_rel=round(float(det["t"][k]), 4),
                          phase=phase.name, target_energy=None if target is None else round(target, 3),
                          energy=None if c.energy is None else round(c.energy, 3),
                          energy_conf=round(c.energy_conf, 3) if c.energy is not None else None,
                          energy_source=c.energy_source, deviation=None if dev is None else round(dev, 3),
                          bpm=None if c.bpm is None else round(c.bpm, 1), camelot=c.camelot, roles=role_list,
                          moods=mood_list, anchored=(f"position {anchor_pos[i]}" if i in anchor_pos else
                                                     (f"phase {sc.phases[model.phase_anchor[i]].name}"
                                                      if i in model.phase_anchor else None)),
                          transition=trans, costs=costs, why=why, violations=viol_at.get(k + 1, []),
                          warnings=warnings, provenance=c.provenance)
        entries.append(entry)
    if counterfactuals:
        _counterfactuals(model, order, weights, ev, entries, own=own, preset=preset)
    return entries


SEVERITY_EPS = 1e-6


def _better(a: Evaluation, b: Evaluation) -> bool:
    """Lexicographic (severity, J) comparison with a tolerance on the summed severities."""
    if a.severity < b.severity - SEVERITY_EPS:
        return True
    return abs(a.severity - b.severity) <= SEVERITY_EPS and a.cost < b.cost - 1e-9


def _single_moves(model: Model, order: list[int], k: int, weights: Mapping[str, float], ev: Evaluation,
                  max_replacements: int = 20) -> Iterable[list[int]]:
    """Orders that differ from ``order`` in slot ``k`` only: the best-ranked replacements by an
    unused track (cheap position cost first) and every swap with another free slot."""
    used = set(order)
    unused = [i for i in range(model.n) if i not in used and i not in model.locked]
    if unused and ev.detail is not None:
        t = np.full(len(unused), ev.detail["t"][k])
        ph = np.full(len(unused), ev.detail["phase"][k])
        pos = model.position_costs(np.array(unused), t, ph, np.full(len(unused), ev.detail["target"][k]))
        quick = sum(float(weights.get(n_, 0.0)) * v for n_, v in pos.items())
        for j in np.argsort(quick, kind="stable")[:max_replacements]:
            cand = list(order)
            cand[k] = unused[int(j)]
            yield cand
    for k2, i2 in enumerate(order):
        if k2 != k and i2 not in model.locked:
            cand = list(order)
            cand[k], cand[k2] = cand[k2], cand[k]
            yield cand


def _move_label(model: Model, order: list[int], cand: list[int], k: int) -> str:
    if cand[k] not in order:
        return f"replace with '{model.ctx[cand[k]].track.display_name}'"
    k2 = order.index(cand[k])
    return f"swap with #{k2 + 1} '{model.ctx[order[k2]].track.display_name}'"


def _counterfactuals(model: Model, order: list[int], weights: Mapping[str, float], ev: Evaluation,
                     entries: list[PlanEntry], *, own: Callable[[list[int]], Evaluation] | None = None,
                     preset: str | None = None) -> None:
    """For each free slot: the best replacement (unused track) and the best swap, with the
    change of every objective term — the "why not something else" part of the report.
    ``own`` scores an order with the objective the variant was SEARCHED with (preset weights
    + novelty); a move that helps the scenario's objective but hurts that one is reported as
    the preset's trade-off, not as something the search missed."""
    if ev.detail is None:
        ev = model.evaluate(order, weights, detail=True)
    for k, i in enumerate(order):
        if i in model.locked:
            entries[k].counterfactual = {"note": "anchored: not movable"}
            continue
        best: tuple[list[int], Evaluation] | None = None
        for cand in _single_moves(model, order, k, weights, ev):
            e2 = model.evaluate(cand, weights)
            if best is None or _better(e2, best[1]):
                best = (cand, e2)
        if best is None:
            continue
        cand, e2 = best
        label = _move_label(model, order, cand, k)
        d_cost = e2.cost - ev.cost
        d_sev = e2.severity - ev.severity
        if abs(d_sev) <= SEVERITY_EPS:
            d_sev = 0.0
        diffs = {name: round(e2.weighted.get(name, 0.0) - ev.weighted.get(name, 0.0), 4)
                 for name in TERMS if abs(e2.weighted.get(name, 0.0) - ev.weighted.get(name, 0.0)) >= 5e-4}
        info: dict[str, Any] = {"best_alternative": label, "objective_change": round(d_cost, 5),
                                "violation_change": round(d_sev, 3), "term_changes": diffs}
        if d_sev > 0:
            new = Counter(v.kind for v in model.evaluate(cand, weights, detail=True).violations or []) \
                - Counter(v.kind for v in ev.violations or [])
            info["violations_added"] = sorted(new)
        entries[k].counterfactual = info
        if _better(e2, ev):
            gain = (f"would improve the objective by {-d_cost:.3f}" if d_sev == 0
                    else f"would remove hard-constraint severity {-d_sev:.2f} (objective {d_cost:+.3f})")
            if own is not None and not _better(own(cand), own(order)):
                entries[k].why.append(f"note: {label} {gain} under the scenario weights; preset '{preset}' "
                                      f"prefers the current track (its trade-off)")
            else:
                entries[k].why.append(f"note: {label} {gain} (not found within the search budget)")
            continue
        if d_sev > 0:
            kinds = ", ".join(info.get("violations_added") or []) or "a hard constraint"
            reason = f"would break {kinds}"
        else:
            worse = sorted(((v, n_) for n_, v in diffs.items() if v > 0), reverse=True)[:2]
            reason = ("would worsen " + ", ".join(f"{n_} by {v:.3f}" for v, n_ in worse)) if worse else \
                "no better option found"
        entries[k].why.append(f"best alternative ({label}): objective {d_cost:+.3f}, {reason}")


def _tradeoffs(variants: list[PlanVariant]) -> None:
    if not variants:
        return
    ref = variants[0]
    names = {"curve": "energy-curve fit", "band": "phase energy ranges", "role": "role fit", "mood": "mood fit",
             "transition": "transition quality", "edge_energy": "edge energy", "tempo": "tempo continuity",
             "harmony": "harmonic continuity", "timbre": "timbre continuity", "mood_flow": "mood flow",
             "diversity": "artist spacing", "adjacency": "neighbour preferences", "duration": "duration",
             "editorial": "editorial preference", "uncertainty": "data certainty"}
    for v in variants:
        notes = []
        if v.duplicate_of:
            notes.append(f"identical to variant {v.duplicate_of}: the scenario leaves little freedom")
        if v is ref:
            notes.append(f"reference: objective {v.objective:.3f} with the scenario weights"
                         + (f", {len(v.violations)} hard-constraint violation(s)" if v.violations else ""))
            v.tradeoffs = notes
            continue
        better, worse = [], []
        for k, label in names.items():
            d = v.terms.get(k, 0.0) - ref.terms.get(k, 0.0)
            if d <= -0.005:
                better.append((d, label))
            elif d >= 0.005:
                worse.append((d, label))
        if better:
            notes.append("better than A in " + ", ".join(f"{lab} ({d:+.3f})" for d, lab in sorted(better)[:3]))
        if worse:
            notes.append("worse than A in " + ", ".join(f"{lab} ({d:+.3f})" for d, lab in
                                                        sorted(worse, reverse=True)[:3]))
        dv = len(v.violations) - len(ref.violations)
        if dv:
            notes.append(f"{'+' if dv > 0 else ''}{dv} hard-constraint violation(s) compared with A")
        shared = len(set(v.order) & set(ref.order))
        notes.append(f"objective {v.objective:.3f} vs {ref.objective:.3f} (scenario weights); "
                     f"{shared}/{len(v.order)} tracks shared with A")
        v.tradeoffs = notes


__all__ = ["ENGINE_VERSION", "PRESETS", "PRESET_ORDER", "TERMS", "TrackContext", "build_context", "provenance_category",
           "Violation", "PlanEntry", "PlanVariant", "PlanResult", "Model", "Evaluation", "SearchSettings", "plan",
           "evaluate_order", "explain", "preset_weights", "select_candidates"]
