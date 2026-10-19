"""P1 (spec §7): vibe-aware sequencing and daypart profiles.

Replaces the legacy greedy harmonic sorter of ``legacy/automix_v1.py``
(``score = bpm_diff - 50 if camelot_ok``), which let key compatibility dominate,
ignored mood/energy and produced ascending-BPM chains. The Camelot compatibility
rules themselves are kept unchanged (``camelot_compatible``).

Transition scoring — ``transition_score(a, b, cfg)``
    Five components, each normalized to 0..1 and ``None`` when unknown:

    ====================== ===============================================================
    vibe_similarity        mood (1.0 same, 0.7 same mood family, 0.35 adjacent families,
                           0.3 unrelated/unknown vocabulary, 0.0 opposed families) and
                           Jaccard overlap of ``vibe_tags``; 0.6·mood + 0.4·tags when both
    energy_continuity      1 − |ΔE| / max_energy_jump (clipped); a jump at the hard limit = 0
    bpm_continuity         1 − Δ / bpm_span, Δ after half/double-time folding (70 ≙ 140)
    harmonic_compatibility 1.0 compatible (legacy rules), 0.5 two wheel steps, 0.0 clash
    genre_continuity       1.0 same, 0.5 related (small table / "house" ⊂ "deep house"), 0.0
    ====================== ===============================================================

    Labels are normalized like curation does (``textnorm``: 'Feel-Good' == 'feel good');
    multi-valued genres/moods ('Pop/Rock', 'pop; dance') score their best-matching parts.

    Each known component carries a confidence ``c`` = the lower of both tracks'
    ``TrackFeatures.confidence()`` (manual metadata 1.0, energy = energy_confidence, ...).
    An automatically detected key is calibrated (``auto_key_confidence``: the profile
    correlation mapped onto 0..0.5, the scale of the other automatic estimates), so harmony
    — the lowest spec §7 priority — cannot outweigh auto tempo or ID3 genre. With weights ``w``::

        coverage   = Σ w·c (known components) / Σ w (all components)
        similarity = Σ w·c·v / Σ w·c   (None below MIN_SIMILARITY_COVERAGE or with < 2 components)
        total      = coverage · similarity + (1 − coverage) · neutral_score

    An unknown feature is therefore neither "0 similarity" nor "perfect match": it pulls
    the total towards ``neutral_score`` in proportion to the weight it would have carried.
    An all-unknown pair scores exactly ``neutral_score`` with coverage 0. The total ranks
    transitions; ``similarity`` (known data only) decides "excessively similar" runs.

Client profile → effective configuration (``effective_config``)
    =================  ======  ====================================================
    vibe_consistency   high    vibe_similarity weight ×1.3
                       medium  unchanged
                       low     vibe_similarity weight ×0.7, max_similar_run − 1 (min 1)
    energy_jumps       low     max_energy_jump ×0.75, flag_energy_jump ×0.8
                       medium  unchanged
                       high    max_energy_jump ×1.5, flag_energy_jump ×1.5
    =================  ======  ====================================================

    Weights are renormalized to sum to 1 afterwards. High vibe consistency does NOT
    change the similar-run limit: the client asks for coherent vibe, so near-identical
    neighbours are expected; the default limit still prevents monotony. Low consistency
    asks for variety, so the run limit is tightened.

Sequencing — ``build_sequence``
    1. Defensive filtering: ``excluded`` tracks, ``exclude_track_ids``, duplicate track
       ids and invalid durations are never placed; each is listed in ``not_selected``.
    2. Selection (only with a daypart and/or limits). Daypart fit = energy inside
       ``energy_range`` (1.0, decaying to 0 at ``ENERGY_FIT_FALLOFF`` outside), BPM range,
       preferred tags/genres/moods; each shrunk towards 0.5 by its confidence. Unknown
       energy → neutral 0.5 plus a note — never assumed to fit. ``avoid_tags`` exclude a
       track from the daypart. Selection score = 0.65·fit + 0.35·editorial (−1..+1 →
       0..1) + seeded jitter, with a small penalty for repeating an artist. A track whose
       selection score stays below ``MIN_SELECTION_SCORE`` even with every UNKNOWN fit
       component counted as perfect is left out as a poor fit — never a protected
       (allowlisted) one; a general curator approval does not switch the daypart filter off. Tracks are taken best-first until the target duration /
       track count is reached; enough calm openers are reserved when the daypart has
       ``opening_max_energy``.
    3. Ordering: beam search (``beam_width``) maximizing Σ transition total + energy-curve
       adherence (linear from ``energy_curve`` start to end; unknown energy neither
       rewarded nor punished) + seeded jitter (``variation``). HARD constraints whenever
       an alternative exists: same artist (``textnorm.primary_artist``; unknown artists
       never match) not within ``artist_min_gap`` positions, |ΔE| ≤ ``max_energy_jump``
       between consecutive KNOWN energies (an unknown-energy track is no invisible bridge),
       at most ``max_similar_run`` consecutive transitions with similarity ≥
       ``similar_threshold``, the first ``opening_tracks`` tracks with KNOWN energy ≤
       ``opening_max_energy``. Without a feasible candidate the least-bad one is
       placed (penalized by severity) and the transition is FLAGGED. Look-ahead against
       beam myopia: an energy-isolation bonus (Warnsdorff's rule — place tracks with few
       remaining energy-compatible neighbours early); artist urgency (an artist whose
       remaining tracks only just fit the remaining slots is placed now — hard when
       feasible); curve regret (children are ranked by score minus the curve adherence
       remaining tracks have irrevocably lost, so off-curve tracks are not deferred for
       free). If the chosen path still breaks a hard constraint: a deterministic local
       repair (relocations, swaps, 2-opt reversals, block moves; budgeted), a wider beam
       pass (n ≤ ``RETRY_MAX_TRACKS``) and — unless an energy gap or an artist's track count
       already forces a violation — an exhaustive feasibility search (n ≤
       ``EXACT_MAX_TRACKS``) plus seeded min-conflicts walks. Candidates are pruned to a
       top-K shortlist per expansion; transition rows are vectorized (~1 s for 2000 tracks).
    4. Report: every transition carries its exact ``TransitionScore``. A transition is
       flagged for review when it breaks a hard constraint ("no alternative found" only when
       proven — exhaustive search or a bound that holds for every order — else "the search
       could not avoid this"), its energy jump (also across unknown-energy tracks: "energy
       continuity not verified across ...") exceeds ``flag_energy_jump``, the folded tempo
       difference exceeds ``TEMPO_FLAG_FACTOR × bpm_span`` (the score itself already carries
       ``tempo_jump`` beyond ``bpm_span``), energy is unknown on both sides, or its total is
       below ``LOW_TRANSITION_SCORE``. A harmonic clash alone is NOT flagged (lowest editorial
       priority); it is listed as context when the transition is flagged for another reason.

Dayparts — ``plan_dayparts``
    Each track is assigned to at most ONE daypart by a global fit-greedy assignment
    (best (track, daypart) fit first, respecting each daypart's limits), so morning cannot
    starve evening of its best tracks and nothing is reused. A daypart short of calm
    openers then takes them — unassigned first, else from another daypart that does not
    need them for its own opening and either fits them worse or can be refilled — and a
    second greedy pass re-places displaced tracks. Each daypart is then sequenced on its
    own. Without limits every track goes to the daypart it fits best — never an equal
    split of one global list.

Determinism: all randomness comes from ``numpy.random.default_rng(seed)``; the same
inputs and seed give an identical result.
"""

from __future__ import annotations

import dataclasses
import functools
import heapq
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import ConfigError, SequencingConfig
from .models import (
    ClientProfile,
    DaypartProfile,
    FlaggedTransition,
    LevelSetting,
    SequencedTrack,
    SequenceResult,
    TrackFeatures,
    TransitionScore,
    to_jsonable,
)
from .textnorm import label_keys, normalize_tag, primary_artist, split_values

log = logging.getLogger(__name__)

ALGORITHM = "beam-v1"

COMPONENTS = (
    "vibe_similarity",
    "energy_continuity",
    "bpm_continuity",
    "harmonic_compatibility",
    "genre_continuity",
)
# Short feature names used in "unknown:<x>" flags.
COMPONENT_FEATURE = {
    "vibe_similarity": "vibe",
    "energy_continuity": "energy",
    "bpm_continuity": "bpm",
    "harmonic_compatibility": "key",
    "genre_continuity": "genre",
}

# --- component details -------------------------------------------------------
DEFAULT_CONFIDENCE = 0.5        # known value without a stated confidence (e.g. key w/o key_confidence)
# Automatic key detection reports the best Krumhansl-profile correlation, which is not a
# probability (0.6-0.9 is typical even for wrong keys). It is mapped linearly from
# KEY_CORRELATION_FLOOR..KEY_CORRELATION_FULL onto 0..AUTO_KEY_CONFIDENCE_MAX, the scale of the
# other automatic estimates (auto BPM 0.5, id3 genre 0.6), so harmonic compatibility (spec §7
# priority 6) cannot outweigh tempo (5) or genre (4) for an uncurated library.
AUTO_KEY_CONFIDENCE_MAX = 0.5
KEY_CORRELATION_FLOOR = 0.3
KEY_CORRELATION_FULL = 0.9
# "Excessively similar" (similar-run limit) is judged on the confidence-weighted mean of the
# KNOWN components (``TransitionScore.similarity``), not on the total shrunk towards neutral,
# which an uncurated library could never push above ``similar_threshold``. It needs at least
# this share of the weight backed by data and at least two known components.
# 0.15 is reachable by auto-analysed energy + BPM alone (0.30*0.4 + 0.15*0.5 = 0.195), so the
# similar-run limit also works for untagged libraries (SPEC-8); one component is never enough.
MIN_SIMILARITY_COVERAGE = 0.15
MIN_SIMILARITY_COMPONENTS = 2
MOOD_WEIGHT_IN_VIBE = 0.6       # vibe = 0.6·mood + 0.4·tag Jaccard when both are known
SAME_MOOD_FAMILY = 0.7
ADJACENT_MOOD_FAMILY = 0.35
UNRELATED_MOOD = 0.3            # different labels whose relation we cannot judge (not in the table)
RELATED_GENRE = 0.5
HARMONIC_TWO_STEPS = 0.5
LOW_TRANSITION_SCORE = 0.35     # flag transitions scoring below this
TEMPO_FLAG_FACTOR = 2.0         # flag tempo breaks beyond this × bpm_span (folded BPM difference)

# --- daypart fit / selection -------------------------------------------------
NEUTRAL_FIT = 0.5
ENERGY_FIT_FALLOFF = 0.30       # energy distance outside the daypart range at which fit reaches 0
# Inside the range, fit falls off with the distance to the daypart's core band (the span
# of its energy curve), so a 0.25 track prefers morning (curve 0.30-0.50) over afternoon
# (0.45-0.55) instead of tying at 1.0 everywhere inside overlapping ranges.
ENERGY_BAND_FALLOFF = 0.40
# Daypart selection: tracks whose selection score (fit + editorial, without jitter) stays
# below this are left out — e.g. a low-preference experimental piece that suits no daypart.
MIN_SELECTION_SCORE = 0.40
# Opening positions reward editorially strong, well-fitting tracks (selection quality)
# so that misfits are not parked at the start, where they cost only one transition.
OPENING_QUALITY_WEIGHT = 0.8
FIT_WEIGHTS = {"energy": 0.5, "bpm": 0.2, "preferences": 0.3}
PREF_PARTIAL_MOOD = 0.7         # track mood/tag in the same mood family as a preferred mood
PREF_PARTIAL_GENRE = 0.7        # related genre
PREF_MISMATCH_FIT = 0.2         # labels known, none preferred
SELECT_EDITORIAL_WEIGHT = 0.35
ARTIST_REPEAT_PENALTY = 0.08    # per already-selected track of the same artist

# --- ordering ----------------------------------------------------------------
CURVE_WEIGHT = 0.6              # energy-curve term = CURVE_WEIGHT·conf·(adherence − 0.5)
CURVE_TOLERANCE = 0.30          # |E − curve target| at which adherence reaches 0
VIOLATION_PENALTY = 10.0        # per unit of hard-constraint severity (only when unavoidable)
ISOLATION_BONUS = 0.4           # / remaining energy-compatible neighbours (Warnsdorff)
ISOLATION_MAX_DEGREE = 3        # tracks with more remaining neighbours get no bonus
ARTIST_URGENCY_BONUS = 3.0
ROW_CACHE_MAX = 1024            # cached transition rows (each 3 × n float64)
REPAIR_MAX_ROUNDS = 60          # local-repair moves after the beam (only when violations remain)
REPAIR_MAX_POSITIONS = 6        # violating positions examined per round
REPAIR_MAX_CANDIDATES = 1000    # candidate moves per violating position (deterministic stride sample)
REPAIR_SWEEPS = 10              # position sweeps per violation in _repair_moves (sets the stride)
REPAIR_MAX_TRACKS = 3000        # the repair needs an n × n transition matrix
REPAIR_MAX_EVALS = 30000        # total candidate evaluations (deterministic budget, ~2 s at n = 2000)
REPAIR_SEGMENT_LENGTHS = (2, 3)  # block lengths tried by segment moves
RETRY_MAX_TRACKS = 400          # wider second beam pass when violations remain (cost ~ n² · width)
RETRY_WIDTH_FACTOR = 4
EXACT_MAX_TRACKS = 30           # exhaustive feasibility search (proves "no alternative") up to here
EXACT_MAX_NODES = 20000         # node budget of that search (deterministic)
CONFLICT_RESTARTS = 16          # seeded min-conflicts walks when violations remain (n <= RETRY_MAX_TRACKS)
CONFLICT_MAX_EVALS = 1500       # evaluations per walk: many short walks beat few long ones (heavy-tailed)

_EPS = 1e-9

MOOD_FAMILIES: dict[str, frozenset[str]] = {
    "calm": frozenset({"calm", "chill", "chilled", "relaxed", "relaxing", "mellow", "peaceful", "soft",
                       "gentle", "serene", "laid back", "laidback", "ambient", "dreamy", "easy"}),
    "warm": frozenset({"warm", "romantic", "tender", "sentimental", "nostalgic", "melancholic",
                       "melancholy", "sad", "bittersweet", "emotional", "intimate"}),
    "bright": frozenset({"happy", "upbeat", "uplifting", "cheerful", "joyful", "feel good", "feelgood",
                         "positive", "sunny", "optimistic", "bright", "fun", "playful"}),
    "groove": frozenset({"groovy", "groove", "funky", "sensual", "sexy", "smooth", "cool", "seductive"}),
    "energetic": frozenset({"energetic", "party", "euphoric", "driving", "powerful", "intense",
                            "exciting", "hype", "danceable"}),
    "dark": frozenset({"dark", "aggressive", "angry", "tense", "menacing", "heavy", "gloomy"}),
}
ADJACENT_MOOD_FAMILIES = frozenset(frozenset(p) for p in (
    ("calm", "warm"), ("calm", "groove"), ("calm", "bright"), ("warm", "groove"),
    ("bright", "groove"), ("bright", "energetic"), ("groove", "energetic"),
))
_MOOD_FAMILY = {normalize_tag(label): fam for fam, labels in MOOD_FAMILIES.items() for label in labels}

_RELATED_GENRE_PAIRS = (
    ("pop", "dance"), ("pop", "r&b"), ("pop", "soul"), ("pop", "electropop"), ("pop", "synthpop"),
    ("dance", "house"), ("dance", "electronic"), ("dance", "disco"), ("house", "disco"),
    ("house", "electronic"), ("disco", "funk"), ("funk", "soul"), ("soul", "r&b"), ("r&b", "hip hop"),
    ("jazz", "soul"), ("jazz", "lounge"), ("lounge", "chillout"), ("chillout", "downtempo"),
    ("lounge", "downtempo"), ("indie", "indie pop"), ("indie", "indie rock"), ("rock", "pop rock"),
    ("latin", "reggaeton"), ("latin", "latin pop"), ("electronic", "synthpop"),
)

_CAMELOT_RE = re.compile(r"^0?(1[0-2]|[1-9])([AB])$")
_GENRE_ALIASES = {"rnb": "r&b", "r'n'b": "r&b", "r and b": "r&b", "rhythm and blues": "r&b",
                  "hiphop": "hip hop", "synth pop": "synthpop", "electro pop": "electropop"}


# ---------------------------------------------------------------------------
# Label / value normalization (shared by the scalar and vectorized scorers)
#
# Same rules as curation (textnorm): 'Feel-Good' == 'feel good' == 'FEEL_GOOD', and
# multi-valued genre/mood strings ('Pop/Rock', 'pop; dance') are compared by their parts.
# ---------------------------------------------------------------------------

def _norm_label(s: Any) -> str | None:
    """Normalized whole label (textnorm.normalize_tag); None when blank."""
    return normalize_tag(s) or None


def _alias_genre(g: str) -> str:
    return _GENRE_ALIASES.get(g, g)


@functools.lru_cache(maxsize=8192)
def _parts(value: str, genre: bool) -> tuple[str, ...]:
    out = {normalize_tag(p) for p in split_values(value)}
    out.discard("")
    if genre:
        out = {_alias_genre(p) for p in out}
    return tuple(sorted(out))


def _genre_parts(s: Any) -> tuple[str, ...]:
    """Normalized, alias-resolved parts of a (possibly multi-valued) genre, sorted, unique."""
    return _parts(str(s), True) if s is not None else ()


def _mood_parts(s: Any) -> tuple[str, ...]:
    """Normalized parts of a (possibly multi-valued) mood, sorted, unique."""
    return _parts(str(s), False) if s is not None else ()


def _norm_genre(s: Any) -> str | None:
    """Canonical key of a genre value: its parts joined by '/' ('Rock/Pop' == 'pop; rock')."""
    return "/".join(_genre_parts(s)) or None


def _mood_key(s: Any) -> str | None:
    return "/".join(_mood_parts(s)) or None


_RELATED_GENRES = frozenset(frozenset((_alias_genre(normalize_tag(a)), _alias_genre(normalize_tag(b))))
                            for a, b in _RELATED_GENRE_PAIRS)


def artist_key(artist: str | None) -> str | None:
    """Case-insensitive identity of the primary artist ("X feat. Y" → "x"; textnorm.primary_artist,
    the rule curation and QC use); None when unknown. Unknown artists never match each other."""
    if not artist:
        return None
    return primary_artist(artist)


def _finite(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _energy_of(t: TrackFeatures) -> float | None:
    e = _finite(t.energy)
    return None if e is None else _clip01(e)


def _bpm_of(t: TrackFeatures) -> float | None:
    b = _finite(t.bpm)
    return b if b is not None and b > 0 else None


def _tags_of(t: TrackFeatures) -> frozenset[str]:
    return frozenset(x for x in (_norm_label(v) for v in (t.vibe_tags or [])) if x)


def auto_key_confidence(correlation: float | None) -> float:
    """Calibrated confidence of an automatically detected key (see AUTO_KEY_CONFIDENCE_MAX):
    the profile correlation mapped onto 0..AUTO_KEY_CONFIDENCE_MAX; unknown → the same cap."""
    r = _finite(correlation)
    if r is None:
        return min(DEFAULT_CONFIDENCE, AUTO_KEY_CONFIDENCE_MAX)
    span = KEY_CORRELATION_FULL - KEY_CORRELATION_FLOOR
    return AUTO_KEY_CONFIDENCE_MAX * _clip01((r - KEY_CORRELATION_FLOOR) / span)


def _conf(t: TrackFeatures, name: str) -> float:
    """Confidence of a KNOWN feature, clipped to 0..1. A known value without a stated
    confidence gets DEFAULT_CONFIDENCE. A key that is neither set nor confirmed by the
    curator (``camelot`` / ``key_confidence`` source 'manual') is calibrated by
    ``auto_key_confidence`` — the raw correlation would rank harmony above tempo and genre."""
    sources = t.feature_sources or {}
    if name == "camelot" and "manual" not in (sources.get("camelot"), sources.get("key_confidence")):
        return auto_key_confidence(t.key_confidence)
    try:
        c = _finite(t.confidence(name))
    except Exception:   # defensive: malformed provenance must not break sequencing
        c = None
    return DEFAULT_CONFIDENCE if c is None else _clip01(c)


def parse_camelot(code: Any) -> tuple[int, str] | None:
    """'8A' / '08a' / ' 12B ' → (8, 'A'); None when not a valid Camelot code."""
    if not isinstance(code, str):
        return None
    m = _CAMELOT_RE.match(code.strip().upper().replace(" ", ""))
    return (int(m.group(1)), m.group(2)) if m else None


def _camelot_of(t: TrackFeatures) -> tuple[int, str] | None:
    return parse_camelot(t.camelot) if t.camelot else None


def _fmt_key(k: tuple[int, str]) -> str:
    return f"{k[0]}{k[1]}"


def _folded_bpm_diff(a: float, b: float) -> float:
    """Tempo distance after half/double-time folding (70 vs 140 → 0)."""
    return min(abs(b - a), abs(b - 2 * a), abs(2 * b - a))


def _key_steps(na: int, la: str, nb: int, lb: str) -> int:
    dist = abs(na - nb) % 12
    dist = min(dist, 12 - dist)
    return dist + (1 if la != lb else 0)


def _harmonic_value(steps: int) -> float:
    if steps <= 1:
        return 1.0
    if steps == 2:
        return HARMONIC_TWO_STEPS
    return 0.0


# ---------------------------------------------------------------------------
# Public similarity primitives
# ---------------------------------------------------------------------------

def camelot_compatible(a: str, b: str) -> bool:
    """Legacy semantics (automix v1): same key, ±1 on the wheel with the same letter
    (12A↔1A wraps), or relative major/minor (same number, other letter).
    Raises ValueError for strings that are not Camelot codes."""
    pa, pb = parse_camelot(a), parse_camelot(b)
    if pa is None or pb is None:
        raise ValueError(f"invalid Camelot code: {a if pa is None else b!r}")
    (na, la), (nb, lb) = pa, pb
    if na == nb:                                   # same key or relative major/minor
        return True
    return la == lb and (na - nb) % 12 in (1, 11)  # adjacent on the wheel, same mode


def camelot_steps(a: str, b: str) -> int | None:
    """Wheel distance (+1 for a mode change) between two Camelot codes; None if invalid."""
    pa, pb = parse_camelot(a), parse_camelot(b)
    if pa is None or pb is None:
        return None
    return _key_steps(pa[0], pa[1], pb[0], pb[1])


def _mood_pair(na: str, nb: str) -> float:
    if na == nb:
        return 1.0
    fa, fb = _MOOD_FAMILY.get(na), _MOOD_FAMILY.get(nb)
    if fa is None or fb is None:
        return UNRELATED_MOOD
    if fa == fb:
        return SAME_MOOD_FAMILY
    if frozenset((fa, fb)) in ADJACENT_MOOD_FAMILIES:
        return ADJACENT_MOOD_FAMILY
    return 0.0


def mood_similarity(a: str, b: str) -> float:
    """1.0 same mood, 0.7 same family, 0.35 adjacent families, 0.3 relation unknown, else 0.
    Labels are compared like curation does ('feel-good' == 'Feel Good'); multi-valued moods
    ('calm; warm') score their best-matching pair of parts."""
    pa, pb = _mood_parts(a), _mood_parts(b)
    if pa == pb:
        return 1.0
    if not pa or not pb:
        return UNRELATED_MOOD
    return max(_mood_pair(x, y) for x in pa for y in pb)


def _genre_pair(ga: str, gb: str) -> float:
    if ga == gb:
        return 1.0
    if frozenset((ga, gb)) in _RELATED_GENRES:
        return RELATED_GENRE
    ta, tb = set(ga.split()), set(gb.split())
    if ta <= tb or tb <= ta:
        return RELATED_GENRE
    return 0.0


def genre_similarity(a: str, b: str) -> float:
    """1.0 same genre, 0.5 related (table or one name contained in the other), else 0.
    Multi-valued genres ('Pop/Rock', 'pop; dance' — common in ID3) score their best-matching
    pair of parts: 'Pop/Rock' → 'Pop' shares 'pop' (1.0); unknown → 0."""
    pa, pb = _genre_parts(a), _genre_parts(b)
    if not pa or not pb:
        return 0.0
    if pa == pb:
        return 1.0
    return max(_genre_pair(x, y) for x in pa for y in pb)


# ---------------------------------------------------------------------------
# Transition scoring (scalar, exact; the beam search uses the vectorized twin)
# ---------------------------------------------------------------------------

def _vibe_component(a: TrackFeatures, b: TrackFeatures) -> tuple[float | None, float | None]:
    ma, mb = _mood_key(a.mood), _mood_key(b.mood)
    ta, tb = _tags_of(a), _tags_of(b)
    mood = tags = None
    if ma and mb:
        mood = (mood_similarity(ma, mb), min(_conf(a, "mood"), _conf(b, "mood")))
    if ta and tb:
        tags = (len(ta & tb) / len(ta | tb), min(_conf(a, "vibe_tags"), _conf(b, "vibe_tags")))
    if mood and tags:
        return (MOOD_WEIGHT_IN_VIBE * mood[0] + (1.0 - MOOD_WEIGHT_IN_VIBE) * tags[0],
                MOOD_WEIGHT_IN_VIBE * mood[1] + (1.0 - MOOD_WEIGHT_IN_VIBE) * tags[1])
    if mood:
        return mood
    if tags:
        return tags
    return None, None


def _weights(cfg: SequencingConfig) -> list[float]:
    return [max(0.0, float(cfg.weights.get(k, 0.0))) for k in COMPONENTS]


def transition_score(a: TrackFeatures, b: TrackFeatures, cfg: SequencingConfig) -> TransitionScore:
    """Score the transition a → b (see the module docstring for the formula)."""
    comps: dict[str, float | None] = dict.fromkeys(COMPONENTS)
    confs: dict[str, float] = {}
    flags: list[str] = []

    v, c = _vibe_component(a, b)
    if v is not None:
        comps["vibe_similarity"], confs["vibe_similarity"] = v, c

    ea, eb = _energy_of(a), _energy_of(b)
    if ea is not None and eb is not None:
        d = eb - ea
        comps["energy_continuity"] = _clip01(1.0 - abs(d) / max(float(cfg.max_energy_jump), _EPS))
        confs["energy_continuity"] = min(_conf(a, "energy"), _conf(b, "energy"))
        if abs(d) > cfg.flag_energy_jump + _EPS:
            flags.append(f"energy_jump:{d:+.2f}")

    ba, bb = _bpm_of(a), _bpm_of(b)
    if ba is not None and bb is not None:
        diff = _folded_bpm_diff(ba, bb)
        comps["bpm_continuity"] = _clip01(1.0 - diff / max(float(cfg.bpm_span), _EPS))
        confs["bpm_continuity"] = min(_conf(a, "bpm"), _conf(b, "bpm"))
        if diff > cfg.bpm_span:
            flags.append(f"tempo_jump:{ba:.0f}->{bb:.0f}")

    ka, kb = _camelot_of(a), _camelot_of(b)
    for t, k in ((a, ka), (b, kb)):
        if t.camelot and k is None:
            flags.append(f"invalid_key:{t.camelot}")
    if ka is not None and kb is not None:
        hv = _harmonic_value(_key_steps(ka[0], ka[1], kb[0], kb[1]))
        comps["harmonic_compatibility"] = hv
        confs["harmonic_compatibility"] = min(_conf(a, "camelot"), _conf(b, "camelot"))
        if hv == 0.0:
            flags.append(f"harmonic_clash:{_fmt_key(ka)}->{_fmt_key(kb)}")

    ga, gb = _norm_genre(a.genre), _norm_genre(b.genre)
    if ga and gb:
        comps["genre_continuity"] = genre_similarity(ga, gb)
        confs["genre_continuity"] = min(_conf(a, "genre"), _conf(b, "genre"))

    weights = _weights(cfg)
    total_w = sum(weights)
    if total_w <= 0:
        raise ConfigError("all sequencing weights are zero")
    num = den = 0.0
    known = 0
    for w, name in zip(weights, COMPONENTS):
        value = comps[name]
        if value is None:
            flags.append(f"unknown:{COMPONENT_FEATURE[name]}")
            continue
        wc = w * confs[name]
        num += wc * value
        den += wc
        known += wc > 0
    total =(num + float(cfg.neutral_score) * (total_w - den)) / total_w
    coverage = den / total_w
    similarity = (num / den if den > 0 and coverage >= MIN_SIMILARITY_COVERAGE - _EPS
                  and known >= MIN_SIMILARITY_COMPONENTS else None)
    return TransitionScore(total=total, components=comps, coverage=coverage, flags=flags, similarity=similarity)


# ---------------------------------------------------------------------------
# Profile → effective configuration
# ---------------------------------------------------------------------------

def _level(value: Any) -> LevelSetting:
    if isinstance(value, LevelSetting):
        return value
    try:
        return LevelSetting(str(value).strip().lower())
    except ValueError:
        raise ConfigError(f"invalid level {value!r} (expected low / medium / high)") from None


def _validate_cfg(cfg: SequencingConfig) -> None:
    bad = [k for k, w in cfg.weights.items() if _finite(w) is None or float(w) < 0]
    if bad:
        raise ConfigError(f"sequencing weights must be finite and >= 0: {bad}")
    unknown = sorted(set(cfg.weights) - set(COMPONENTS))
    if unknown:
        log.warning("ignoring unknown sequencing weight(s): %s", ", ".join(unknown))
    checks = (
        ("beam_width", cfg.beam_width >= 1),
        ("max_energy_jump", cfg.max_energy_jump > 0),
        ("flag_energy_jump", cfg.flag_energy_jump > 0),
        ("bpm_span", cfg.bpm_span > 0),
        ("artist_min_gap", cfg.artist_min_gap >= 0),
        ("max_similar_run", cfg.max_similar_run >= 1),
        ("variation", cfg.variation >= 0),
        ("neutral_score", 0.0 <= cfg.neutral_score <= 1.0),
    )
    for name, ok in checks:
        if not ok:
            raise ConfigError(f"invalid sequencing setting {name}={getattr(cfg, name)!r}")


def effective_config(cfg: SequencingConfig, profile: ClientProfile | None = None
                     ) -> tuple[SequencingConfig, dict[str, str]]:
    """Apply the client profile's vibe_consistency / energy_jumps (mapping in the module
    docstring) and renormalize the weights. Returns (config, applied adjustments)."""
    _validate_cfg(cfg)
    weights = dict(zip(COMPONENTS, _weights(cfg)))
    max_jump, flag_jump, max_run = cfg.max_energy_jump, cfg.flag_energy_jump, cfg.max_similar_run
    applied: dict[str, str] = {}
    if profile is not None:
        vc = _level(profile.vibe_consistency)
        if vc is LevelSetting.HIGH:
            weights["vibe_similarity"] *= 1.3
            applied["vibe_consistency"] = "high: vibe_similarity weight x1.3"
        elif vc is LevelSetting.LOW:
            weights["vibe_similarity"] *= 0.7
            max_run = max(1, max_run - 1)
            applied["vibe_consistency"] = "low: vibe_similarity weight x0.7, max_similar_run -1"
        ej = _level(profile.energy_jumps)
        if ej is LevelSetting.LOW:
            max_jump, flag_jump = max_jump * 0.75, flag_jump * 0.8
            applied["energy_jumps"] = "low: max_energy_jump x0.75, flag_energy_jump x0.8"
        elif ej is LevelSetting.HIGH:
            max_jump, flag_jump = max_jump * 1.5, flag_jump * 1.5
            applied["energy_jumps"] = "high: max_energy_jump x1.5, flag_energy_jump x1.5"
    total = sum(weights.values())
    if total <= 0:
        raise ConfigError("all sequencing weights are zero")
    weights = {k: v / total for k, v in weights.items()}
    eff = dataclasses.replace(cfg, weights=weights, max_energy_jump=round(max_jump, 6),
                              flag_energy_jump=round(flag_jump, 6), max_similar_run=max_run)
    return eff, applied


# ---------------------------------------------------------------------------
# Daypart fit
# ---------------------------------------------------------------------------

@dataclass
class DaypartFit:
    """How well one track suits one daypart (selection input, reported per track)."""
    fit: float                                  # 0..1
    components: dict[str, float] = field(default_factory=dict)
    avoided_tag: str | None = None              # set → track is excluded from this daypart
    notes: list[str] = field(default_factory=list)       # unknowns (neutral, not assumed)
    mismatches: list[str] = field(default_factory=list)  # human-readable misfit reasons
    # Fit with every UNKNOWN component at 1.0: the poor-fit cut (MIN_SELECTION_SCORE) uses it,
    # so a track is left out only for what is known about it — unknown energy is not a misfit.
    best_case: float | None = None
    unknown: list[str] = field(default_factory=list)     # components that rest on no data


def _labels(t: TrackFeatures) -> set[str]:
    """Normalized labels of a track as curation sees them (textnorm.label_keys: whole values
    and the parts of multi-valued ones) plus alias-resolved genre parts ('rnb' → 'r&b')."""
    out: set[str] = set()
    for r in [*(t.vibe_tags or []), t.genre, t.mood]:
        out |= label_keys(r)
    out.update(_genre_parts(t.genre))
    return out


def _range_dist(x: float, lo: float, hi: float) -> float:
    return max(lo - x, 0.0, x - hi)


def _check_daypart(dp: DaypartProfile) -> None:
    lo, hi = dp.energy_range
    if not (0.0 <= lo <= hi <= 1.0):
        raise ConfigError(f"daypart {dp.name!r}: energy_range {dp.energy_range} must be 0 <= low <= high <= 1")
    if dp.energy_curve is not None and not all(0.0 <= v <= 1.0 for v in dp.energy_curve):
        raise ConfigError(f"daypart {dp.name!r}: energy_curve {dp.energy_curve} must lie within 0..1")
    if dp.bpm_range is not None and not (0 < dp.bpm_range[0] <= dp.bpm_range[1]):
        raise ConfigError(f"daypart {dp.name!r}: invalid bpm_range {dp.bpm_range}")
    if dp.opening_tracks < 0:
        raise ConfigError(f"daypart {dp.name!r}: opening_tracks must be >= 0")


def daypart_fit(track: TrackFeatures, daypart: DaypartProfile, *, bpm_span: float = 16.0) -> DaypartFit:
    """Fit of a track to a daypart. Each component is shrunk towards 0.5 by the
    confidence of the feature it rests on; unknown features give a neutral 0.5 and a
    note — an unknown track is never assumed to fit."""
    labels = _labels(track)
    res = DaypartFit(fit=NEUTRAL_FIT)
    for tag in daypart.avoid_tags:
        norm = normalize_tag(tag)
        forms = {f for f in (norm, _alias_genre(norm)) if f}
        if forms & labels:
            res.avoided_tag = str(tag)
            break

    comps = res.components
    lo, hi = daypart.energy_range
    e = _energy_of(track)
    if e is None:
        comps["energy"] = NEUTRAL_FIT
        res.unknown.append("energy")
        res.notes.append(f"energy unknown — neutral daypart fit {NEUTRAL_FIT:.2f} "
                         f"(not assumed to suit '{daypart.name}')")
    else:
        dist = _range_dist(e, lo, hi)
        band = (min(daypart.energy_curve), max(daypart.energy_curve)) if daypart.energy_curve else (lo, hi)
        band_fit = _clip01(1.0 - _range_dist(e, *band) / ENERGY_BAND_FALLOFF)
        c = _conf(track, "energy")
        comps["energy"] = c * band_fit * _clip01(1.0 - dist / ENERGY_FIT_FALLOFF) + (1.0 - c) * NEUTRAL_FIT
        if dist > 0:
            side = "below" if e < lo else "above"
            res.mismatches.append(f"energy {e:.2f} {side} the {daypart.name} range {lo:.2f}-{hi:.2f}")

    if daypart.bpm_range is not None:
        b = _bpm_of(track)
        blo, bhi = daypart.bpm_range
        if b is None:
            comps["bpm"] = NEUTRAL_FIT
            res.unknown.append("bpm")
            res.notes.append("BPM unknown — neutral tempo fit")
        else:
            dist = min(_range_dist(x, blo, bhi) for x in (b, b / 2, b * 2))
            c = _conf(track, "bpm")
            comps["bpm"] = c * _clip01(1.0 - dist / max(bpm_span, _EPS)) + (1.0 - c) * NEUTRAL_FIT
            if dist > 0:
                res.mismatches.append(f"tempo {b:.0f} BPM outside {blo:.0f}-{bhi:.0f}")

    pref_tags = {x for x in (_norm_label(v) for v in daypart.preferred_tags) if x}
    pref_moods = {x for x in (_norm_label(v) for v in daypart.preferred_moods) if x}
    pref_genres = {_alias_genre(x) for x in (_norm_label(v) for v in daypart.preferred_genres) if x}
    preferred = pref_tags | pref_moods | pref_genres
    if preferred:
        if labels & preferred:
            comps["preferences"] = 1.0
        else:
            partial = 0.0
            mood_like = [*_mood_parts(track.mood), *_tags_of(track)]
            for m in mood_like:
                for pm in pref_moods | pref_tags:
                    if mood_similarity(m, pm) == SAME_MOOD_FAMILY:
                        partial = max(partial, PREF_PARTIAL_MOOD)
            if track.genre and any(genre_similarity(track.genre, pg) >= RELATED_GENRE for pg in pref_genres):
                partial = max(partial, PREF_PARTIAL_GENRE)
            if partial > 0:
                comps["preferences"] = partial
            elif labels:
                comps["preferences"] = PREF_MISMATCH_FIT
                res.mismatches.append(f"no preferred tag/genre/mood ({', '.join(sorted(preferred))})")
            else:
                comps["preferences"] = NEUTRAL_FIT
                res.unknown.append("preferences")
                res.notes.append("tags/genre/mood unknown — neutral preference fit")

    total_w = sum(FIT_WEIGHTS[k] for k in comps)
    res.fit = sum(FIT_WEIGHTS[k] * v for k, v in comps.items()) / total_w
    res.best_case = sum(FIT_WEIGHTS[k] * (1.0 if k in res.unknown else v) for k, v in comps.items()) / total_w
    return res


def _editorial01(t: TrackFeatures, scores: Mapping[str, float] | None) -> tuple[float, float | None]:
    """Editorial score (−1..+1, from curation) mapped to 0..1; missing → neutral 0.5."""
    if not scores:
        return 0.5, None
    raw = scores.get(t.track_id)
    if raw is None:
        raw = scores.get(t.rel_path)
    s = _finite(raw)
    if s is None:
        return 0.5, None
    s = max(-1.0, min(1.0, s))
    return (s + 1.0) / 2.0, s


def _selection_score(fit: DaypartFit | None, ed01: float) -> float:
    if fit is None:
        return ed01
    return (1.0 - SELECT_EDITORIAL_WEIGHT) * fit.fit + SELECT_EDITORIAL_WEIGHT * ed01


def _poor_fit(fit: DaypartFit | None, ed01: float, protected: bool) -> bool:
    """The MIN_SELECTION_SCORE cut: only when the misfit rests on KNOWN data (unknown
    components count as a perfect fit here) and never for protected tracks (explicitly
    allowlisted — a 'low' preference must not turn into an exclusion for them)."""
    if fit is None or protected:
        return False
    best = fit.best_case if fit.best_case is not None else fit.fit
    return (1.0 - SELECT_EDITORIAL_WEIGHT) * best + SELECT_EDITORIAL_WEIGHT * ed01 < MIN_SELECTION_SCORE


# ---------------------------------------------------------------------------
# Vectorized transition rows
# ---------------------------------------------------------------------------

class _FeatureMatrix:
    """Vectorized copy of the track features: ``row(i)`` returns, for i → every track, the
    transition total (identical to ``transition_score(...).total``), |ΔE| (0 when either
    energy is unknown) and the similarity (``transition_score(...).similarity``, −1 = None)."""

    def __init__(self, tracks: Sequence[TrackFeatures], cfg: SequencingConfig) -> None:
        n = self.n = len(tracks)
        self.w = _weights(cfg)
        self.W = float(sum(self.w))
        if self.W <= 0:
            raise ConfigError("all sequencing weights are zero")
        self.neutral = float(cfg.neutral_score)
        self.max_jump = max(float(cfg.max_energy_jump), _EPS)
        self.bpm_span = max(float(cfg.bpm_span), _EPS)

        self.E = np.full(n, np.nan)
        self.Ec = np.zeros(n)
        self.B = np.full(n, np.nan)
        self.Bc = np.zeros(n)
        self.kn = np.full(n, -1, dtype=np.int64)
        self.kl = np.zeros(n, dtype=np.int64)
        self.kc = np.zeros(n)
        self.gid = np.full(n, -1, dtype=np.int64)
        self.gc = np.zeros(n)
        self.mid = np.full(n, -1, dtype=np.int64)
        self.mc = np.zeros(n)
        self.tc = np.zeros(n)
        self.aid = np.full(n, -1, dtype=np.int64)
        genres: dict[str, int] = {}
        moods: dict[str, int] = {}
        vocab: dict[str, int] = {}
        artists: dict[str, int] = {}
        tag_rows: list[list[int]] = []
        for i, t in enumerate(tracks):
            e = _energy_of(t)
            if e is not None:
                self.E[i], self.Ec[i] = e, _conf(t, "energy")
            b = _bpm_of(t)
            if b is not None:
                self.B[i], self.Bc[i] = b, _conf(t, "bpm")
            k = _camelot_of(t)
            if k is not None:
                self.kn[i], self.kl[i], self.kc[i] = k[0], (0 if k[1] == "A" else 1), _conf(t, "camelot")
            g = _norm_genre(t.genre)
            if g:
                self.gid[i], self.gc[i] = genres.setdefault(g, len(genres)), _conf(t, "genre")
            m = _mood_key(t.mood)
            if m:
                self.mid[i], self.mc[i] = moods.setdefault(m, len(moods)), _conf(t, "mood")
            tags = _tags_of(t)
            tag_rows.append([vocab.setdefault(x, len(vocab)) for x in sorted(tags)])
            if tags:
                self.tc[i] = _conf(t, "vibe_tags")
            a = artist_key(t.artist)
            if a:
                self.aid[i] = artists.setdefault(a, len(artists))
        self.n_artists = len(artists)
        self.Ek = ~np.isnan(self.E)
        self.Bk = ~np.isnan(self.B)
        self.Kk = self.kn >= 0
        self.Gk = self.gid >= 0
        self.Mk = self.mid >= 0
        glabels, mlabels = list(genres), list(moods)
        self.G = np.array([[genre_similarity(x, y) for y in glabels] for x in glabels]).reshape(len(glabels), len(glabels))
        self.M = np.array([[mood_similarity(x, y) for y in mlabels] for x in mlabels]).reshape(len(mlabels), len(mlabels))
        self.T = np.zeros((n, max(1, len(vocab))))
        for i, cols in enumerate(tag_rows):
            self.T[i, cols] = 1.0
        self.tcount = self.T.sum(axis=1)
        self.Tk = self.tcount > 0

    def _acc(self, num: np.ndarray, den: np.ndarray, cnt: np.ndarray, ci: int, v: np.ndarray, c: np.ndarray,
             known: np.ndarray) -> None:
        wc = np.where(known, self.w[ci] * c, 0.0)
        num += wc * np.where(known, v, 0.0)
        den += wc
        cnt += wc > 0

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n
        num = np.zeros(n)
        den = np.zeros(n)
        cnt = np.zeros(n, dtype=np.int64)
        jump = np.zeros(n)
        acc = functools.partial(self._acc, num, den, cnt)
        with np.errstate(invalid="ignore", divide="ignore"):
            # vibe: mood and/or tag overlap
            mood = tags = None
            if self.Mk[i]:
                mood = (self.M[self.mid[i], np.maximum(self.mid, 0)], np.minimum(self.mc, self.mc[i]), self.Mk)
            if self.Tk[i]:
                inter = self.T @ self.T[i]
                union = self.tcount + self.tcount[i] - inter
                tv = inter / np.where(union > 0, union, 1.0)
                tags = (tv, np.minimum(self.tc, self.tc[i]), self.Tk)
            if mood is not None and tags is not None:
                both = mood[2] & tags[2]
                v = np.where(both, MOOD_WEIGHT_IN_VIBE * mood[0] + (1.0 - MOOD_WEIGHT_IN_VIBE) * tags[0],
                             np.where(mood[2], mood[0], tags[0]))
                c = np.where(both, MOOD_WEIGHT_IN_VIBE * mood[1] + (1.0 - MOOD_WEIGHT_IN_VIBE) * tags[1],
                             np.where(mood[2], mood[1], tags[1]))
                acc(0, v, c, mood[2] | tags[2])
            elif mood is not None or tags is not None:
                acc(0, *(mood if mood is not None else tags))
            # energy
            if self.Ek[i]:
                d = np.abs(self.E - self.E[i])
                v = np.clip(1.0 - d / self.max_jump, 0.0, 1.0)
                acc(1, v, np.minimum(self.Ec, self.Ec[i]), self.Ek)
                jump = np.where(self.Ek, d, 0.0)
            # tempo (half/double-time folded)
            if self.Bk[i]:
                bi = self.B[i]
                d = np.minimum(np.abs(self.B - bi), np.minimum(np.abs(self.B - 2 * bi), np.abs(2 * self.B - bi)))
                v = np.clip(1.0 - d / self.bpm_span, 0.0, 1.0)
                acc(2, v, np.minimum(self.Bc, self.Bc[i]), self.Bk)
            # key
            if self.Kk[i]:
                dist = np.abs(self.kn - self.kn[i]) % 12
                steps = np.minimum(dist, 12 - dist) + (self.kl != self.kl[i])
                v = np.where(steps <= 1, 1.0, np.where(steps == 2, HARMONIC_TWO_STEPS, 0.0))
                acc(3, v, np.minimum(self.kc, self.kc[i]), self.Kk)
            # genre
            if self.Gk[i]:
                v = self.G[self.gid[i], np.maximum(self.gid, 0)]
                acc(4, v, np.minimum(self.gc, self.gc[i]), self.Gk)
            total = (num + self.neutral * (self.W - den)) / self.W
            ok = (den > 0) & (den / self.W >= MIN_SIMILARITY_COVERAGE - _EPS) & (cnt >= MIN_SIMILARITY_COMPONENTS)
            sim = np.where(ok, num / np.where(den > 0, den, 1.0), -1.0)
        return total, jump, sim


class _RowCache:
    def __init__(self, fm: _FeatureMatrix, cap: int = ROW_CACHE_MAX) -> None:
        self.fm, self.cap = fm, cap
        self.rows: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self.rows.get(i)
        if r is None:
            r = self.rows[i] = self.fm.row(i)
        return r

    def trim(self, keep: Iterable[int]) -> None:
        if len(self.rows) > self.cap:
            keep = set(keep)
            self.rows = {k: v for k, v in self.rows.items() if k in keep}


# ---------------------------------------------------------------------------
# Beam search ordering
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Node:
    last: int
    parent: "_Node | None"
    length: int
    score: float
    avail: np.ndarray            # bool[n]: not yet placed
    artist_used: np.ndarray      # int[n_artists]: placed tracks per artist
    recent: tuple[int, ...]      # artist ids of the last artist_min_gap tracks (-1 = unknown)
    run: int                     # current run of "very similar" transitions
    zh: int                      # Zobrist hash of the placed set (beam de-duplication)
    viol: float = 0.0            # cumulative hard-constraint severity along the path
    last_e: float = math.nan     # last KNOWN energy on the path (energy is checked across unknowns)


@dataclass
class _Ordering:
    order: list[int]
    severity: float = 0.0        # hard-constraint severity left in ``order`` (0 = none)
    proven: bool = False         # exhaustive search: no order without a hard violation exists


def _opening(fm: _FeatureMatrix, daypart: DaypartProfile | None) -> tuple[np.ndarray | None, int]:
    """(opener mask, number of opening positions): openers have KNOWN energy <= opening_max_energy."""
    if daypart is None or daypart.opening_max_energy is None or daypart.opening_tracks <= 0:
        return None, 0
    openers = fm.Ek & (np.nan_to_num(fm.E, nan=np.inf) <= daypart.opening_max_energy + _EPS)
    return openers, min(int(daypart.opening_tracks), fm.n)


def _beam_search(fm: _FeatureMatrix, cfg: SequencingConfig, daypart: DaypartProfile | None,
                 rng: np.random.Generator, quality: Sequence[float] | None, beam_width: int,
                 shortlist: int) -> tuple[list[int], float]:
    """One beam pass; returns (order, cumulative forced hard-constraint severity)."""
    n = fm.n
    # Opening slots favour editorially strong, well-fitting tracks (selection quality 0..1).
    opening_slots = max(1, int(daypart.opening_tracks)) if daypart is not None else 1
    opening_bonus = (OPENING_QUALITY_WEIGHT * (np.asarray(quality, dtype=np.float64) - 0.5)
                     if quality is not None else np.zeros(n))
    gap = max(0, int(cfg.artist_min_gap))
    thr = float(cfg.similar_threshold)
    max_run = max(1, int(cfg.max_similar_run))
    mj = fm.max_jump
    var = max(0.0, float(cfg.variation))
    E, Ek, Ec, aid = fm.E, fm.Ek, fm.Ec, fm.aid
    rows = _RowCache(fm)
    openers, opening_n = _opening(fm, daypart)

    # Energy curve. Placement term: CURVE_WEIGHT·conf·(adherence − 0.5), adherence = 1 − |E − target|/τ
    # (clipped to −1..1); unknown energy → 0, i.e. neither rewarded nor punished.
    # Look-ahead ("regret"): the targets still ahead of position p span [min(t_p, t_end), max(t_p, t_end)];
    # a remaining track's best reachable deviation can only grow as p advances, so the adherence it has
    # irrevocably lost is a lower bound on future cost. Beam children are ranked by score − Σ regret,
    # otherwise the beam would defer off-curve tracks for free and pile them up at the end.
    curve = tuple(daypart.energy_curve) if daypart is not None and daypart.energy_curve else None
    if curve is not None:
        targets = curve[0] + (curve[1] - curve[0]) * np.arange(n) / (n - 1)
        e_safe_c = np.nan_to_num(E, nan=0.0)
        curve_w = np.where(Ek, CURVE_WEIGHT * Ec / CURVE_TOLERANCE, 0.0)

        def reachable_dev(p: int) -> np.ndarray:
            lo_t, hi_t = min(targets[p], targets[-1]), max(targets[p], targets[-1])
            return np.maximum(0.0, np.maximum(lo_t - e_safe_c, e_safe_c - hi_t))

        dev0 = reachable_dev(0)

    # Warnsdorff energy-isolation bonus: neighbours within the hard jump limit.
    use_iso = int(Ek.sum()) >= 2 and mj < 1.0
    if use_iso:
        order_e = np.flatnonzero(Ek)[np.argsort(E[Ek], kind="stable")]
        e_sorted = E[order_e]
        e_safe = np.nan_to_num(E, nan=0.0)
        lo_idx = np.searchsorted(e_sorted, e_safe - mj, side="left")
        hi_idx = np.searchsorted(e_sorted, e_safe + mj, side="right")

    # Artist urgency: an artist with c tracks left needs (c − 1)·(gap + 1) + 1 slots; when that
    # reaches the remaining slots it must be placed NOW (hard, when feasible) — skipping it would
    # make a same-artist violation inevitable.
    use_urgency = gap > 0 and fm.n_artists > 0
    if use_urgency:
        art_total = np.bincount(aid[aid >= 0], minlength=fm.n_artists)
        art_need = (art_total - 1) * (gap + 1) + 1
        use_urgency = bool((art_total >= 2).any())

    zkeys = [int(x) for x in np.random.default_rng(0x5E9).integers(1, 2**62, size=n)]
    root = _Node(last=-1, parent=None, length=0, score=0.0, avail=np.ones(n, dtype=bool),
                 artist_used=np.zeros(max(1, fm.n_artists), dtype=np.int64), recent=(), run=0, zh=0)
    beam = [root]
    for pos in range(n):
        remaining = n - pos
        add = rng.uniform(-var, var, n) if var > 0 else np.zeros(n)
        if pos < opening_slots:
            add = add + opening_bonus
        regret = None
        if curve is not None:
            with np.errstate(invalid="ignore"):
                adherence = np.clip(1.0 - np.abs(E - targets[pos]) / CURVE_TOLERANCE, -1.0, 1.0)
            add = add + np.where(Ek, CURVE_WEIGHT * Ec * (adherence - 0.5), 0.0)
            if pos + 1 < n:
                regret = curve_w * (reachable_dev(pos + 1) - dev0)
        urgent_pool = np.flatnonzero(art_need >= remaining) if use_urgency else None

        cands: list[tuple[float, float, int, int, float]] = []   # (rank, score, state, track, violations)
        for si, st in enumerate(beam):
            obj = add.copy()
            forced = None
            mask = st.avail
            if openers is not None and pos < opening_n:
                opening_mask = mask & openers
                if opening_mask.any():
                    mask = opening_mask
            cmask = mask
            if st.last >= 0:
                tot, jump, sim = rows.get(st.last)
                obj += tot
                if not Ek[st.last] and not math.isnan(st.last_e):
                    # unknown-energy track: continuity is checked against the last KNOWN energy,
                    # so an unknown track cannot hide a jump (it is no "maximum compatibility")
                    jump = np.where(Ek, np.abs(E - st.last_e), 0.0)
                sev = np.zeros(n)
                if gap > 0:
                    for r, a in enumerate(reversed(st.recent)):
                        if a >= 0:
                            sev = np.maximum(sev, np.where(aid == a, 3.0 + (gap - r), 0.0))
                over = jump > mj
                if over.any():
                    sev += np.where(over, 2.0 + 4.0 * (jump - mj), 0.0)
                if st.run >= max_run:
                    sev += sim >= thr
                feasible = mask & (sev == 0.0)
                if feasible.any():
                    cmask = feasible
                else:   # no alternative: least-bad candidate, penalized (and flagged later)
                    obj -= VIOLATION_PENALTY * sev
                    forced = sev
            if use_iso:
                csum = np.concatenate(([0], np.cumsum(st.avail[order_e])))
                deg = csum[hi_idx] - csum[lo_idx] - st.avail + np.count_nonzero(st.avail & ~Ek)
                obj += np.where(Ek & (deg >= 1) & (deg <= ISOLATION_MAX_DEGREE),
                                ISOLATION_BONUS / np.maximum(deg, 1), 0.0)
            if urgent_pool is not None and urgent_pool.size:
                left = art_total[urgent_pool] - st.artist_used[urgent_pool]
                urgent = urgent_pool[(left > 0) & ((left - 1) * (gap + 1) + 1 >= remaining)]
                if urgent.size:
                    is_urgent = np.isin(aid, urgent)
                    obj += ARTIST_URGENCY_BONUS * is_urgent
                    if (cmask & is_urgent).any():
                        cmask = cmask & is_urgent
            # rank(child) = score − Σ regret(remaining); the child's own regret leaves the
            # remaining set, hence the per-candidate term.
            base, rank = st.score, obj
            if regret is not None:
                base -= float(regret[st.avail].sum())
                rank = obj + regret
            idx = np.flatnonzero(cmask)
            if idx.size > shortlist:
                idx = idx[np.argpartition(-rank[idx], shortlist - 1)[:shortlist]]
            for j in idx.tolist():
                v = st.viol + (float(forced[j]) if forced is not None else 0.0)
                cands.append((base + float(rank[j]), st.score + float(obj[j]), si, j, v))

        cands.sort(key=lambda c: (-c[0], c[2], c[3]))
        new_beam: list[_Node] = []
        seen: set[tuple[int, int]] = set()
        for _, score, si, j, viol in cands:
            st = beam[si]
            zh = st.zh ^ zkeys[j]
            if (zh, j) in seen:
                continue
            seen.add((zh, j))
            avail = st.avail.copy()
            avail[j] = False
            a = int(aid[j])
            used = st.artist_used
            if a >= 0:
                used = used.copy()
                used[a] += 1
            run = 0
            if st.last >= 0 and rows.get(st.last)[2][j] >= thr:
                run = st.run + 1
            recent = (st.recent + (a,))[-gap:] if gap > 0 else ()
            new_beam.append(_Node(last=j, parent=st, length=st.length + 1, score=score, avail=avail,
                                  artist_used=used, recent=recent, run=run, zh=zh, viol=viol,
                                  last_e=float(E[j]) if Ek[j] else st.last_e))
            if len(new_beam) >= beam_width:
                break
        beam = new_beam
        rows.trim(s.last for s in beam)

    order: list[int] = []
    node: _Node | None = beam[0]
    while node is not None and node.last >= 0:
        order.append(node.last)
        node = node.parent
    order.reverse()
    return order, beam[0].viol


def _beam_order(tracks: Sequence[TrackFeatures], cfg: SequencingConfig, daypart: DaypartProfile | None,
                rng: np.random.Generator, quality: Sequence[float] | None = None) -> _Ordering:
    """Beam search. When the chosen path breaks a hard constraint: local repair (relocations,
    swaps, 2-opt, block moves); for moderate n a wider beam pass; unless a violation is already
    forced by an energy gap or an artist's track count, an exhaustive feasibility search for
    small pools (finds a violation-free order or proves that none exists, ``_Ordering.proven``)
    and seeded min-conflicts walks."""
    fm = _FeatureMatrix(tracks, cfg)
    n = fm.n
    if n <= 1:
        return _Ordering(list(range(n)))
    width = max(1, int(cfg.beam_width))
    order, viol = _beam_search(fm, cfg, daypart, rng, quality, width, width)
    if viol <= 0 or n > REPAIR_MAX_TRACKS:
        return _Ordering(order, viol)
    ctx = _RepairContext(fm, cfg, daypart)
    order, sev = _repair(order, ctx)
    if sev > 0 and n <= RETRY_MAX_TRACKS:
        wide = width * RETRY_WIDTH_FACTOR
        order2, viol2 = _beam_search(fm, cfg, daypart, rng, quality, wide, 2 * width)
        order2, sev2 = _repair(order2, ctx) if viol2 > 0 else (order2, 0.0)
        if sev2 < sev - _EPS:
            order, sev = order2, sev2
    proven = False
    # No violation-free order exists when an energy gap or an artist's track count already forces
    # a violation (the report then words each kind by those bounds): nothing to search for.
    if sev > 0 and n <= RETRY_MAX_TRACKS and not _bounds_infeasible(ctx):
        if n <= EXACT_MAX_TRACKS:
            found, proven = _exact_feasible(ctx, order)
            if found is not None:
                order, sev, proven = found, 0.0, False
        if sev > 0 and not proven:
            found = _min_conflicts(order, ctx)
            if found is not None:
                order, sev = found, 0.0
    if sev > 0:
        log.debug("ordering: hard-constraint severity %.2f left (%s)", sev,
                  "proven unavoidable" if proven else "search exhausted")
    return _Ordering(order, sev, proven)


class _RepairContext:
    """Dense n × n transition data and the hard-constraint settings shared by the local
    repair and the exhaustive search (built only when the beam left a violation)."""

    def __init__(self, fm: _FeatureMatrix, cfg: SequencingConfig, daypart: DaypartProfile | None) -> None:
        self.fm = fm
        self.gap = max(0, int(cfg.artist_min_gap))
        self.thr = float(cfg.similar_threshold)
        self.max_run = max(1, int(cfg.max_similar_run))
        self.mj = fm.max_jump
        rows = [fm.row(i) for i in range(fm.n)]
        self.tot_m = np.vstack([r[0] for r in rows])
        self.simf_m = np.vstack([r[2] >= self.thr for r in rows])
        openers, opening_n = _opening(fm, daypart)
        self.openers = openers
        # Opening positions filled with calm openers are never touched by the repair.
        self.lock = min(opening_n, int(openers.sum())) if openers is not None else 0

    def violations(self, o: np.ndarray) -> tuple[np.ndarray, float]:
        return _violations(o, self.tot_m, self.simf_m, self.fm, self.gap, self.max_run, self.mj)


def _violations(o: np.ndarray, tot_m: np.ndarray, simf_m: np.ndarray, fm: _FeatureMatrix, gap: int,
                max_run: int, mj: float) -> tuple[np.ndarray, float]:
    """Hard-constraint severity per position (index of the incoming track) and Σ transition totals.
    Energy is checked against the last KNOWN energy, i.e. across unknown-energy tracks."""
    m = len(o)
    sev = np.zeros(m)
    if m < 2:
        return sev, 0.0
    a, b = o[:-1], o[1:]
    tot = tot_m[a, b]
    for k in range(1, gap + 1):
        if m > k:
            x, y = fm.aid[o[k:]], fm.aid[o[:-k]]
            sev[k:] += np.where((x == y) & (x >= 0), 3.0 + (gap - (k - 1)), 0.0)
    e = fm.E[o]
    known = fm.Ek[o]
    last = np.maximum.accumulate(np.where(known, np.arange(m), -1))
    prev = np.concatenate(([-1], last[:-1]))
    valid = known & (prev >= 0)
    if valid.any():
        jump = np.abs(e - e[np.maximum(prev, 0)])
        over = valid & (np.nan_to_num(jump, nan=0.0) > mj)
        sev += np.where(over, 2.0 + 4.0 * (np.nan_to_num(jump, nan=0.0) - mj), 0.0)
    sim = simf_m[a, b].astype(np.int64)
    if len(sim) > max_run:
        window = np.convolve(sim, np.ones(max_run + 1, dtype=np.int64), mode="valid")
        sev[max_run + 1:] += window == max_run + 1
    return sev, float(tot.sum())


def _repair_moves(o: np.ndarray, p: int, lock: int, stride: int) -> Iterable[np.ndarray]:
    """Candidate orders around the violation at position p (transition o[p-1] → o[p]):
    relocations, swaps, 2-opt reversals and block moves; positions < lock stay fixed."""
    n = len(o)

    def relocate(q: int, t: int) -> np.ndarray:      # move o[q] to slot t (before the old o[t])
        return np.insert(np.delete(o, q), t - 1 if q < t else t, o[q])

    def swap(i: int, j: int) -> np.ndarray:
        c = o.copy()
        c[i], c[j] = c[j], c[i]
        return c

    def reverse(i: int, j: int) -> np.ndarray:        # reverse o[i..j] inclusive
        c = o.copy()
        c[i:j + 1] = c[i:j + 1][::-1]
        return c

    def move_block(s: int, length: int, t: int) -> np.ndarray:   # move o[s:s+length] before old o[t]
        block = o[s:s + length]
        rest = np.concatenate((o[:s], o[s + length:]))
        at = t - length if t > s else t
        return np.concatenate((rest[:at], block, rest[at:]))

    slots = range(lock, n + 1, stride)
    others = range(lock, n, stride)
    for q in others:                                    # any track into the gap
        if q not in (p - 1, p):
            yield relocate(q, p)
    for src in (p, p - 1):                              # either end of the violation elsewhere
        if src < lock:
            continue
        for t in slots:
            if t not in (src, src + 1):
                yield relocate(src, t)
        for q in others:
            if q != src:
                yield swap(src, q)
    for q in others:                                    # 2-opt: new neighbours at the violation
        if q > p:
            yield reverse(p, q)
        elif q < p - 1:
            yield reverse(q, p - 1)
    for length in REPAIR_SEGMENT_LENGTHS:               # block starting / ending at the violation
        for s in (p, p - length):
            if s < lock or s + length > n:
                continue
            for t in slots:
                if t < s or t > s + length:
                    yield move_block(s, length, t)


def _repair(order: list[int], ctx: _RepairContext) -> tuple[list[int], float]:
    """Local repair of hard-constraint violations left by the (myopic) beam, e.g. when it spent
    scarce "separator" tracks early. Each round tries relocations, swaps, 2-opt reversals and
    block moves around the violating positions; the move that lowers total severity most (then
    keeps Σ score highest) wins. Positions < ``ctx.lock`` (the protected calm opening) are never
    touched. Deterministic, budgeted. Returns (order, remaining severity)."""
    n = len(order)
    o = np.asarray(order, dtype=np.int64)
    sev, _ = ctx.violations(o)
    cur = float(sev.sum())
    if n < 3 or cur <= 0:
        return order, cur
    stride = max(1, math.ceil(REPAIR_SWEEPS * n / REPAIR_MAX_CANDIDATES))
    evals = 0
    for _ in range(REPAIR_MAX_ROUNDS):
        if cur <= 0 or evals >= REPAIR_MAX_EVALS:
            break
        best: tuple[tuple[float, float], np.ndarray, np.ndarray] | None = None
        positions = [p for p in np.flatnonzero(sev > 0).tolist()
                     if p >= max(ctx.lock, 1)][:REPAIR_MAX_POSITIONS]
        for p in positions:
            for cand in _repair_moves(o, p, ctx.lock, stride):
                evals += 1
                c_sev, c_tot = ctx.violations(cand)
                key = (float(c_sev.sum()), -c_tot)
                if key[0] < cur - _EPS and (best is None or key < best[0]):
                    best = (key, cand, c_sev)
                if evals >= REPAIR_MAX_EVALS:
                    break
            if evals >= REPAIR_MAX_EVALS:
                break
        if best is None:
            break
        cur, o, sev = best[0][0], best[1], best[2]
    log.debug("repair: violation severity %.2f after local moves (%d evaluations)", cur, evals)
    return o.tolist(), cur


def _bounds_infeasible(ctx: _RepairContext) -> bool:
    """True when every order breaks a hard constraint for a reason visible without search: a gap
    wider than the jump limit between the sorted known energies, or an artist with more tracks
    than ``artist_min_gap`` can space out (same bounds as ``_unavoidable``)."""
    fm = ctx.fm
    es = np.sort(fm.E[fm.Ek])
    if es.size > 1 and float(np.diff(es).max()) > ctx.mj:
        return True
    if ctx.gap > 0 and fm.n_artists:
        counts = np.bincount(fm.aid[fm.aid >= 0], minlength=fm.n_artists)
        if int(((counts - 1) * (ctx.gap + 1) + 1).max()) > fm.n:
            return True
    return False


def _min_conflicts(order: Sequence[int], ctx: _RepairContext) -> list[int] | None:
    """Seeded min-conflicts walks: swap a track of a random violating transition with a random
    other track, keep the swap unless the number of violating transitions grows (sideways steps
    cross the plateaus where the deterministic repair stops). The first walk starts from
    ``order``, later ones from seeded shuffles (positions < ``ctx.lock`` stay). Returns a
    violation-free order, or None. Deterministic (fixed seed), budgeted."""
    rng = np.random.default_rng(0xC0F1)
    start = np.asarray(order, dtype=np.int64)
    n, lock = len(start), ctx.lock
    if n - lock < 2:
        return None
    for walk in range(CONFLICT_RESTARTS):
        o = start.copy()
        if walk:
            o[lock:] = rng.permutation(o[lock:])
        sev, _ = ctx.violations(o)
        bad = int(np.count_nonzero(sev))
        for _ in range(CONFLICT_MAX_EVALS):
            if bad == 0:
                return o.tolist()
            pos = np.flatnonzero(sev)
            p = int(pos[rng.integers(0, pos.size)])
            src = p - 1 if p - 1 >= lock and rng.random() < 0.5 else p
            q = int(rng.integers(lock, n))
            if q == src or src < lock:
                continue
            cand = o.copy()
            cand[src], cand[q] = cand[q], cand[src]
            c_sev, _ = ctx.violations(cand)
            c_bad = int(np.count_nonzero(c_sev))
            if c_bad <= bad:
                o, sev, bad = cand, c_sev, c_bad
        if bad == 0:
            return o.tolist()
    return None


def _exact_feasible(ctx: _RepairContext, preferred: Sequence[int]) -> tuple[list[int] | None, bool]:
    """Exhaustive depth-first search (small pools only) for an order WITHOUT any hard-constraint
    violation (same rules as ``_violations``; opening positions need openers). Children are tried
    best transition first, the root in ``preferred`` order; branches are pruned when an artist's
    remaining tracks no longer fit the remaining slots, or the remaining known energies cannot be
    chained within the jump limit. Returns (order or None, complete); ``complete`` with None
    proves that every order breaks a hard constraint. Deterministic, node-budgeted."""
    fm = ctx.fm
    n = fm.n
    gap, max_run, mj, lock = ctx.gap, ctx.max_run, ctx.mj, ctx.lock
    energy = [float(x) if k else None for x, k in zip(fm.E.tolist(), fm.Ek.tolist())]
    aid = fm.aid.tolist()
    tot = ctx.tot_m.tolist()
    simf = ctx.simf_m.tolist()
    opener = ctx.openers.tolist() if ctx.openers is not None else [True] * n
    rank = {j: r for r, j in enumerate(preferred)}
    left = [0] * max(1, fm.n_artists)          # unplaced tracks per artist
    for a in aid:
        if a >= 0:
            left[a] += 1
    last_at = [-n - gap - 1] * max(1, fm.n_artists)
    path: list[int] = []
    used = [False] * n
    nodes = 0

    def forced_artist(p: int) -> int | None:
        """-1 = dead end; an artist id that must take slot p; None = free choice."""
        forced = None
        for a, c in enumerate(left):
            if c <= 0 or gap <= 0:
                continue
            earliest = max(p, last_at[a] + gap + 1)
            last_needed = earliest + (c - 1) * (gap + 1)
            if last_needed > n - 1:
                return -1
            if last_needed == n - 1 and earliest == p:
                if forced is not None:
                    return -1
                forced = a
        return forced

    def energy_chain_ok(last_e: float | None) -> bool:
        es = sorted([e for j, e in enumerate(energy) if e is not None and not used[j]]
                    + ([last_e] if last_e is not None else []))
        return all(y - x <= mj for x, y in zip(es, es[1:]))

    def ok(j: int, last_e: float | None, run: int) -> tuple[bool, float | None, int]:
        p = len(path)
        if p < lock and not opener[j]:
            return False, None, 0
        a = aid[j]
        if a >= 0 and p - last_at[a] <= gap:
            return False, None, 0
        e = energy[j]
        if e is not None and last_e is not None and abs(e - last_e) > mj:
            return False, None, 0
        new_run = run + 1 if p > 0 and simf[path[-1]][j] else 0
        if new_run > max_run:
            return False, None, 0
        return True, (e if e is not None else last_e), new_run

    def dfs(last_e: float | None, run: int) -> bool:
        nonlocal nodes
        p = len(path)
        if p == n:
            return True
        nodes += 1
        if nodes > EXACT_MAX_NODES:
            return False
        forced = forced_artist(p)
        if forced == -1 or not energy_chain_ok(last_e):
            return False
        if path:
            row = tot[path[-1]]
            cands = sorted((j for j in range(n) if not used[j]), key=lambda j: (-row[j], j))
        else:
            cands = sorted(range(n), key=lambda j: rank.get(j, n))
        for j in cands:
            if forced is not None and aid[j] != forced:
                continue
            good, e_next, run_next = ok(j, last_e, run)
            if not good:
                continue
            a = aid[j]
            prev_at = last_at[a] if a >= 0 else 0
            path.append(j)
            used[j] = True
            if a >= 0:
                left[a] -= 1
                last_at[a] = p
            if dfs(e_next, run_next):
                return True
            path.pop()
            used[j] = False
            if a >= 0:
                left[a] += 1
                last_at[a] = prev_at
            if nodes > EXACT_MAX_NODES:
                return False
        return False

    if dfs(None, 0):
        return list(path), True
    return None, nodes <= EXACT_MAX_NODES


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _context_reasons(a: TrackFeatures, b: TrackFeatures, ts: TransitionScore) -> list[str]:
    """Descriptive extras listed when a transition is flagged for another reason."""
    out = []
    ka, kb = _camelot_of(a), _camelot_of(b)
    if ts.components.get("harmonic_compatibility") == 0.0 and ka and kb:
        out.append(f"harmonic clash {_fmt_key(ka)} -> {_fmt_key(kb)}")
    ma, mb = _norm_label(a.mood), _norm_label(b.mood)
    if ma and mb and mood_similarity(ma, mb) < ADJACENT_MOOD_FAMILY:
        out.append(f"mood change {a.mood} -> {b.mood}")
    if ts.components.get("genre_continuity") == 0.0:
        out.append(f"genre change {a.genre} -> {b.genre}")
    return out


NO_ALTERNATIVE = "no alternative found"                 # proven: no order avoids it
NOT_AVOIDED = "the search could not avoid this"         # heuristic search; not proven unavoidable


def _energy_steps(ordered: Sequence[TrackFeatures]) -> dict[int, tuple[float, float, int]]:
    """Position -> (previous KNOWN energy, its energy, unknown-energy tracks in between) for
    every track with known energy after the first: continuity is judged across unknowns."""
    out: dict[int, tuple[float, float, int]] = {}
    last_e, last_p = None, -1
    for p, t in enumerate(ordered):
        e = _energy_of(t)
        if e is None:
            continue
        if last_e is not None:
            out[p] = (last_e, e, p - last_p - 1)
        last_e, last_p = e, p
    return out


def _unavoidable(ordered: Sequence[TrackFeatures], gap: int, mj: float) -> tuple[int, set[str]]:
    """Lower bounds that hold for EVERY order of these tracks: the number of hard energy
    violations forced by gaps wider than ``mj`` between the sorted known energies (each gap
    must be crossed once), and the artists with more tracks than ``gap`` can space out."""
    es = sorted(e for e in (_energy_of(t) for t in ordered) if e is not None)
    energy_min = sum(1 for x, y in zip(es, es[1:]) if y - x > mj + _EPS)
    counts: dict[str, int] = {}
    for t in ordered:
        a = artist_key(t.artist)
        if a:
            counts[a] = counts.get(a, 0) + 1
    artists = {a for a, c in counts.items() if gap > 0 and (c - 1) * (gap + 1) + 1 > len(ordered)}
    return energy_min, artists


def _assemble(ordered: Sequence[TrackFeatures], fits: Sequence[DaypartFit | None], cfg: SequencingConfig,
              daypart: DaypartProfile | None, *, proven: bool = False,
              opening_context: str | None = None) -> tuple[list[SequencedTrack], list[FlaggedTransition]]:
    """Positions, notes and flagged transitions. A broken hard constraint says "no alternative
    found" only when that is proven (exhaustive search, or a bound that holds for every order),
    otherwise "the search could not avoid this"."""
    gap = max(0, int(cfg.artist_min_gap))
    thr, max_run = float(cfg.similar_threshold), max(1, int(cfg.max_similar_run))
    mj, fj = float(cfg.max_energy_jump), float(cfg.flag_energy_jump)
    opening_max = daypart.opening_max_energy if daypart is not None else None
    opening_n = daypart.opening_tracks if daypart is not None else 0
    if opening_max is not None and opening_context is None:
        opening_context = f"no further track with known energy <= {opening_max:.2f} is available to this daypart"
    steps = _energy_steps(ordered)
    energy_min, artists_forced = _unavoidable(ordered, gap, mj)
    energy_viol = sum(1 for ea, eb, _ in steps.values() if abs(eb - ea) > mj + _EPS)
    # An exhaustive proof is made under the calm-opening reservation; say so (VEDIT-3).
    # Bounds that hold for EVERY order (energy gaps, artist counts) need no qualifier.
    proven_why = (f"{NO_ALTERNATIVE} while keeping the calm opening (first {opening_n} track(s) at energy "
                  f"<= {opening_max:.2f})" if proven and opening_max is not None and opening_n > 0
                  else NO_ALTERNATIVE)
    energy_why = (NO_ALTERNATIVE if energy_viol <= energy_min
                  else proven_why if proven else NOT_AVOIDED)
    seq: list[SequencedTrack] = []
    flagged: list[FlaggedTransition] = []
    run = 0
    for p, t in enumerate(ordered):
        fit = fits[p]
        notes = list(fit.notes) if fit is not None else []
        ts = None
        if p > 0:
            prev = ordered[p - 1]
            ts = transition_score(prev, t, cfg)
            hard: list[str] = []
            soft: list[str] = []
            ak = artist_key(t.artist)
            if ak and gap > 0:
                for back in range(1, gap + 1):
                    if p - back < 0:
                        break
                    if artist_key(ordered[p - back].artist) == ak:
                        why = (NO_ALTERNATIVE if ak in artists_forced
                               else proven_why if proven else NOT_AVOIDED)
                        hard.append(f"same artist '{t.artist}' with only {back - 1} track(s) in between "
                                    f"(minimum {gap}) — {why}")
                        break
            if p in steps:
                ea, eb, unknown_between = steps[p]
                d = eb - ea
                if unknown_between == 0:
                    if abs(d) > mj + _EPS:
                        hard.append(f"energy jump {d:+.2f} ({ea:.2f} -> {eb:.2f}) exceeds the hard limit "
                                    f"{mj:.2f} — {energy_why}")
                    elif abs(d) > fj + _EPS:
                        soft.append(f"energy jump {d:+.2f} ({ea:.2f} -> {eb:.2f}) exceeds {fj:.2f}")
                else:
                    # an unknown-energy track is no bridge: the span between the known
                    # neighbours is what the listener may hear, and it is unverified
                    span = (f"energy continuity not verified across {unknown_between} unknown-energy "
                            f"track(s): energy jump {d:+.2f} ({ea:.2f} -> ? -> {eb:.2f})")
                    if abs(d) > mj + _EPS:
                        hard.append(f"{span} exceeds the hard limit {mj:.2f} — {energy_why}")
                    elif abs(d) > fj + _EPS:
                        soft.append(f"{span} exceeds {fj:.2f}")
            elif _energy_of(prev) is None and _energy_of(t) is None:
                soft.append("unknown energy on both sides — energy continuity not verified")
            run = run + 1 if ts.similarity is not None and ts.similarity >= thr else 0
            if run > max_run:
                hard.append(f"{run} very similar transitions in a row (similarity >= {thr:.2f}, maximum "
                            f"{max_run}) — {proven_why if proven else NOT_AVOIDED}")
            ba, bb = _bpm_of(prev), _bpm_of(t)
            if ba is not None and bb is not None and _folded_bpm_diff(ba, bb) > TEMPO_FLAG_FACTOR * cfg.bpm_span:
                soft.append(f"tempo jump {ba:.0f} -> {bb:.0f} BPM")
            if ts.total < LOW_TRANSITION_SCORE:
                soft.append(f"low transition score {ts.total:.2f}")
            reasons = hard + soft
            if reasons:
                reasons += _context_reasons(prev, t, ts)
                flagged.append(FlaggedTransition(
                    from_position=p, to_position=p + 1, from_track=prev.display_name,
                    to_track=t.display_name, reasons=reasons, score=round(ts.total, 4)))
                notes.extend(f"transition in: {r}" for r in hard)
        if opening_max is not None and p < opening_n:
            e = _energy_of(t)
            if e is None:
                notes.append(f"opening position {p + 1}: energy unknown — {opening_context}")
            elif e > opening_max + _EPS:
                notes.append(f"opening position {p + 1}: energy {e:.2f} above the opening maximum "
                             f"{opening_max:.2f} — {opening_context}")
        seq.append(SequencedTrack(position=p + 1, track=t, transition_in=ts,
                                  daypart_fit=(round(fit.fit, 4) if fit is not None else None), notes=notes))
    return seq, flagged


# ---------------------------------------------------------------------------
# Pool preparation and selection
# ---------------------------------------------------------------------------

def _not_selected(t: TrackFeatures, reason: str) -> dict:
    return {"rel_path": t.rel_path, "track_id": t.track_id, "reason": reason}


def _prepare_pool(pool: Iterable[TrackFeatures], exclude: Mapping[str, str]
                  ) -> tuple[list[TrackFeatures], list[dict]]:
    """Drop (and record) everything that must never be placed."""
    eligible: list[TrackFeatures] = []
    rejected: list[dict] = []
    seen: dict[str, str] = {}
    for t in pool:
        if not isinstance(t, TrackFeatures):
            raise TypeError(f"sequencing expects TrackFeatures, got {type(t).__name__}")
        dur = _finite(t.duration_s)
        if t.excluded:
            why = "; ".join(t.exclusion_reasons) if t.exclusion_reasons else "excluded flag set in metadata"
            reason = f"excluded: {why}"
        elif t.track_id in exclude:
            reason = exclude[t.track_id]
        elif t.track_id in seen:
            reason = f"duplicate of '{seen[t.track_id]}' (same track id {t.track_id})"
        elif dur is None or dur <= 0:
            reason = f"invalid duration {t.duration_s!r} s"
        else:
            eligible.append(t)
            seen[t.track_id] = t.rel_path
            continue
        rejected.append(_not_selected(t, reason))
    return eligible, rejected


def _is_opener(t: TrackFeatures, dp: DaypartProfile) -> bool:
    e = _energy_of(t)
    return e is not None and dp.opening_max_energy is not None and e <= dp.opening_max_energy + _EPS


def _full(count: int, total_s: float, target_s: float | None, max_n: int | None) -> bool:
    return (max_n is not None and count >= max_n) or (target_s is not None and total_s >= target_s)


def _greedy_select(cands: Sequence[int], tracks: Sequence[TrackFeatures], scores: Sequence[float],
                   target_s: float | None, max_n: int | None) -> tuple[list[int], list[int]]:
    """Best-first selection until the limits are reached; repeated artists are
    penalized lazily (scores only decrease, so a lazy heap stays exact)."""
    heap = [(-scores[i], i, 0) for i in cands]
    heapq.heapify(heap)
    counts: dict[str, int] = {}
    selected: list[int] = []
    total = 0.0
    while heap and not _full(len(selected), total, target_s, max_n):
        _, i, seen_count = heapq.heappop(heap)
        a = artist_key(tracks[i].artist)
        cur = counts.get(a, 0) if a else 0
        if cur != seen_count:
            heapq.heappush(heap, (-(scores[i] - ARTIST_REPEAT_PENALTY * cur), i, cur))
            continue
        selected.append(i)
        total += float(tracks[i].duration_s)
        if a:
            counts[a] = cur + 1
    chosen = set(selected)
    return selected, [i for i in cands if i not in chosen]


def _ensure_openers(selected: list[int], rest: Sequence[int], tracks: Sequence[TrackFeatures],
                    scores: Sequence[float], dp: DaypartProfile | None, target_s: float | None,
                    max_n: int | None) -> tuple[list[int], list[int], list[int]]:
    """Reserve calm openers: when the selection holds fewer than ``opening_tracks`` tracks
    with known energy <= opening_max_energy, pull the best ones from ``rest`` and drop the
    weakest non-openers that the limits no longer allow. Returns (selected, added, removed)."""
    want = _opener_want(dp, max_n)
    if want <= 0:
        return selected, [], []
    need = want - sum(1 for i in selected if _is_opener(tracks[i], dp))
    if need <= 0:
        return selected, [], []
    added = sorted((i for i in rest if _is_opener(tracks[i], dp)), key=lambda i: (-scores[i], i))[:need]
    if not added:
        return selected, [], []
    out = list(selected) + added
    removed = _trim_non_openers(out, tracks, scores, dp, target_s, max_n)
    return out, added, removed


def _opener_want(dp: DaypartProfile | None, max_n: int | None) -> int:
    """Calm openers a daypart needs. Never more than the track limit allows: every track
    dropped for an opener is a non-opener, so the selection stays within max_n."""
    if dp is None or dp.opening_max_energy is None or dp.opening_tracks <= 0:
        return 0
    return int(dp.opening_tracks) if max_n is None else min(int(dp.opening_tracks), max_n)


def _trim_non_openers(group: list[int], tracks: Sequence[TrackFeatures], scores: Sequence[float],
                      dp: DaypartProfile, target_s: float | None, max_n: int | None) -> list[int]:
    """Drop (in place) the weakest non-openers that the limits no longer allow; returns them."""
    total = sum(float(tracks[i].duration_s) for i in group)
    removed: list[int] = []
    for i in sorted((i for i in group if not _is_opener(tracks[i], dp)), key=lambda i: (scores[i], i)):
        dur = float(tracks[i].duration_s)
        too_many = max_n is not None and len(group) > max_n
        still_long_enough = target_s is not None and total - dur >= target_s
        if not (too_many or still_long_enough):
            break
        group.remove(i)
        removed.append(i)
        total -= dur
    return removed


def _limit_text(target_s: float | None, max_n: int | None) -> str:
    parts = []
    if target_s is not None:
        parts.append(f"target duration {target_s:.0f} s")
    if max_n is not None:
        parts.append(f"max {max_n} tracks")
    return " / ".join(parts)


def _fit_text(fit: DaypartFit | None, ed: float | None) -> str:
    bits = []
    if fit is not None:
        bits.append(f"fit {fit.fit:.2f}")
        bits.extend(fit.mismatches)
        bits.extend(n for n in fit.notes if n.startswith("energy unknown"))
    if ed is not None:
        bits.append(f"editorial score {ed:+.2f}")
    return "; ".join(bits)


def _check_limits(target_s: float | None, max_n: int | None) -> None:
    if target_s is not None and not (_finite(target_s) is not None and target_s > 0):
        raise ValueError(f"target_duration_s must be > 0, got {target_s!r}")
    if max_n is not None and (not isinstance(max_n, int) or max_n < 1):
        raise ValueError(f"max_tracks must be a positive integer, got {max_n!r}")


def _params(cfg: SequencingConfig, applied: dict[str, str], profile: ClientProfile | None,
            daypart: DaypartProfile | None, target_s: float | None, max_n: int | None,
            counts: dict[str, int]) -> dict:
    return {
        "algorithm": ALGORITHM,
        "weights": {k: round(v, 6) for k, v in cfg.weights.items()},
        "max_energy_jump": cfg.max_energy_jump,
        "flag_energy_jump": cfg.flag_energy_jump,
        "bpm_span": cfg.bpm_span,
        "artist_min_gap": cfg.artist_min_gap,
        "similar_threshold": cfg.similar_threshold,
        "max_similar_run": cfg.max_similar_run,
        "beam_width": cfg.beam_width,
        "variation": cfg.variation,
        "neutral_score": cfg.neutral_score,
        "profile": None if profile is None else {
            "client": profile.client,
            "vibe_consistency": _level(profile.vibe_consistency).value,
            "energy_jumps": _level(profile.energy_jumps).value,
        },
        "profile_adjustments": dict(applied),
        "daypart": to_jsonable(daypart) if daypart is not None else None,
        "target_duration_s": target_s,
        "max_tracks": max_n,
        **counts,
    }


def _finish(selected: Sequence[TrackFeatures], fits: Sequence[DaypartFit | None], cfg: SequencingConfig,
            daypart: DaypartProfile | None, seed: int, not_selected: list[dict], params: dict,
            quality: Sequence[float] | None = None, opening_context: str | None = None) -> SequenceResult:
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    ordering = _beam_order(selected, cfg, daypart, rng, quality)
    ordered = [selected[i] for i in ordering.order]
    seq, flagged = _assemble(ordered, [fits[i] for i in ordering.order], cfg, daypart,
                             proven=ordering.proven, opening_context=opening_context)
    total = sum(float(t.duration_s) for t in ordered)
    log.info("sequenced %d track(s)%s in %.2f s: %d flagged transition(s), %d not selected",
             len(seq), f" for daypart '{daypart.name}'" if daypart else "", time.perf_counter() - started,
             len(flagged), len(not_selected))
    return SequenceResult(daypart=daypart.name if daypart is not None else None, tracks=seq, flagged=flagged,
                          not_selected=not_selected, total_duration_s=total, seed=seed, params=params)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_sequence(pool: list[TrackFeatures], cfg: SequencingConfig, *,
                   profile: ClientProfile | None = None, daypart: DaypartProfile | None = None,
                   editorial_scores: dict[str, float] | None = None, seed: int = 0,
                   target_duration_s: float | None = None, max_tracks: int | None = None,
                   exclude_track_ids: frozenset[str] = frozenset(),
                   protected_track_ids: frozenset[str] = frozenset()) -> SequenceResult:
    """Select (daypart / limits) and order tracks into a curated sequence.

    ``protected_track_ids`` (explicitly allowlisted tracks) are exempt from the
    MIN_SELECTION_SCORE poor-fit cut; they still compete for the limits on their score."""
    cfg_eff, applied = effective_config(cfg, profile)
    if daypart is not None:
        _check_daypart(daypart)
    target_s = target_duration_s if target_duration_s is not None else (
        daypart.target_duration_s if daypart is not None else None)
    max_n = max_tracks if max_tracks is not None else (daypart.max_tracks if daypart is not None else None)
    _check_limits(target_s, max_n)

    exclude = {tid: "excluded for this run (exclude_track_ids)" for tid in exclude_track_ids}
    eligible, not_selected = _prepare_pool(pool, exclude)
    rng = np.random.default_rng(seed)
    var = max(0.0, float(cfg_eff.variation))
    jitter = rng.uniform(-var, var, len(eligible)) if var > 0 and eligible else np.zeros(len(eligible))

    fits = [daypart_fit(t, daypart, bpm_span=cfg_eff.bpm_span) if daypart is not None else None for t in eligible]
    eds = [_editorial01(t, editorial_scores) for t in eligible]
    base = [_selection_score(f, e[0]) for f, e in zip(fits, eds)]
    scores = [b + float(j) for b, j in zip(base, jitter)]

    cands: list[int] = []
    for i, (t, f) in enumerate(zip(eligible, fits)):
        if f is not None and f.avoided_tag is not None:
            not_selected.append(_not_selected(t, f"daypart '{daypart.name}' avoids tag '{f.avoided_tag}'"))
        elif _poor_fit(f, eds[i][0], t.track_id in protected_track_ids):
            not_selected.append(_not_selected(t, f"poor fit for daypart '{daypart.name}': selection score "
                                                 f"{base[i]:.2f} < {MIN_SELECTION_SCORE:.2f} ({_fit_text(f, eds[i][1])})"))
        else:
            cands.append(i)

    if target_s is None and max_n is None:
        selected, rest = list(cands), []
    else:
        selected, rest = _greedy_select(cands, eligible, scores, target_s, max_n)
    selected, added, removed = _ensure_openers(selected, rest, eligible, scores, daypart, target_s, max_n)
    added_set = set(added)
    rest = [i for i in rest if i not in added_set] + removed
    where = f"daypart '{daypart.name}'" if daypart is not None else "this sequence"
    for i in sorted(rest):
        if i in removed:
            reason = f"not selected for {where}: replaced by a calmer opening track ({_fit_text(fits[i], eds[i][1])})"
        else:
            detail = _fit_text(fits[i], eds[i][1])
            reason = (f"not selected for {where}: {_limit_text(target_s, max_n)} reached"
                      + (f" ({detail})" if detail else ""))
        not_selected.append(_not_selected(eligible[i], reason))

    selected = sorted(selected)   # pool order: ordering must not depend on selection order
    params = _params(cfg_eff, applied, profile, daypart, target_s, max_n, {
        "pool_size": len(pool), "eligible": len(eligible), "selected": len(selected)})
    return _finish([eligible[i] for i in selected], [fits[i] for i in selected], cfg_eff, daypart, seed,
                   not_selected, params, quality=[base[i] for i in selected])


def _resolve_daypart(profile: ClientProfile | None, item: str | DaypartProfile) -> DaypartProfile:
    if isinstance(item, DaypartProfile):
        return item
    name = str(item)
    if profile is not None and name in profile.dayparts:
        return profile.dayparts[name]
    # Built-in defaults live in curation.py (ARCHITECTURE.md); imported lazily so this
    # module does not depend on it when the profile defines its dayparts.
    try:
        from . import curation
    except ImportError as e:
        raise ValueError(f"daypart {name!r} is not defined in the client profile") from e
    try:
        if profile is not None:
            return curation.get_daypart(profile, name)
        return curation.DEFAULT_DAYPARTS[name]
    except KeyError as e:   # curation lists the available names in its message
        raise ValueError(str(e.args[0]) if e.args else f"unknown daypart {name!r}") from None


def plan_dayparts(pool: list[TrackFeatures], cfg: SequencingConfig, *, profile: ClientProfile | None,
                  dayparts: list[str | DaypartProfile], editorial_scores: dict[str, float] | None = None,
                  seed: int = 0, target_duration_s: float | None = None,
                  max_tracks: int | None = None,
                  protected_track_ids: frozenset[str] = frozenset()) -> dict[str, SequenceResult]:
    """Sequence several dayparts from one pool; no track is used twice.

    1. Global fit-greedy pass: best (track, daypart) score first, each daypart up to its own
       limits (``target_duration_s`` / ``max_tracks`` arguments, else the daypart's own).
    2. Calm openers (spec §7 "avoid aggressive openings"): a daypart with ``opening_max_energy``
       that holds fewer openers than it needs takes the best ones — unassigned tracks first,
       then tracks of another daypart that it does not need for its own opening — and drops
       its weakest non-openers beyond its limits.
    3. Second greedy pass: tracks displaced in step 2 and the room freed in donor dayparts are
       re-filled into OTHER dayparts that allow the track and have room. Known limitation
       (EDIT-11): a track is not offered back to the daypart it was displaced from, even if
       that daypart later regains room because another daypart took one of its openers.
    Each daypart is then ordered on its own. ``protected_track_ids`` (explicitly allowlisted)
    are exempt from the MIN_SELECTION_SCORE poor-fit cut."""
    if not dayparts:
        raise ValueError("plan_dayparts needs at least one daypart")
    dps = [_resolve_daypart(profile, d) for d in dayparts]
    names = [dp.name for dp in dps]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate daypart names: {names}")
    for dp in dps:
        _check_daypart(dp)
    cfg_eff, applied = effective_config(cfg, profile)
    caps = [(target_duration_s if target_duration_s is not None else dp.target_duration_s,
             max_tracks if max_tracks is not None else dp.max_tracks) for dp in dps]
    for target_s, max_n in caps:
        _check_limits(target_s, max_n)

    eligible, base_rejected = _prepare_pool(pool, {})
    n, n_dp = len(eligible), len(dps)
    rng = np.random.default_rng(seed)
    var = max(0.0, float(cfg_eff.variation))
    jitter = rng.uniform(-var, var, (n_dp, n)) if var > 0 and n else np.zeros((n_dp, n))
    fits = [[daypart_fit(t, dp, bpm_span=cfg_eff.bpm_span) for t in eligible] for dp in dps]
    eds = [_editorial01(t, editorial_scores) for t in eligible]
    base = [[_selection_score(fits[d][i], eds[i][0]) for i in range(n)] for d in range(n_dp)]
    scores = [[base[d][i] + float(jitter[d, i]) for i in range(n)] for d in range(n_dp)]
    akeys = [artist_key(t.artist) for t in eligible]
    durs = [float(t.duration_s) for t in eligible]
    opener = [[_is_opener(t, dp) for t in eligible] for dp in dps]
    wants = [_opener_want(dp, caps[d][1]) for d, dp in enumerate(dps)]
    allow = [[fits[d][i].avoided_tag is None
              and not _poor_fit(fits[d][i], eds[i][0], eligible[i].track_id in protected_track_ids)
              for i in range(n)] for d in range(n_dp)]

    assigned = [-1] * n
    groups: list[list[int]] = [[] for _ in dps]
    totals = [0.0] * n_dp
    art_counts: list[dict[str, int]] = [{} for _ in dps]

    def place(d: int, i: int) -> None:
        assigned[i] = d
        groups[d].append(i)
        totals[d] += durs[i]
        if akeys[i]:
            art_counts[d][akeys[i]] = art_counts[d].get(akeys[i], 0) + 1

    def unplace(i: int) -> None:
        d = assigned[i]
        assigned[i] = -1
        groups[d].remove(i)
        totals[d] -= durs[i]
        if akeys[i]:
            art_counts[d][akeys[i]] -= 1

    def full(d: int) -> bool:
        return _full(len(groups[d]), totals[d], *caps[d])

    def greedy(pairs: Iterable[tuple[int, int]]) -> None:
        """Fit-greedy assignment (lazy heap; artist repeats penalized per daypart — scores only
        decrease, so the lazy heap stays exact)."""
        heap = [(-scores[d][i], d, i, 0) for d, i in pairs]
        heapq.heapify(heap)
        while heap:
            _, d, i, seen_count = heapq.heappop(heap)
            if assigned[i] >= 0 or full(d):
                continue
            a = akeys[i]
            cur = art_counts[d].get(a, 0) if a else 0
            if cur != seen_count:
                heapq.heappush(heap, (-(scores[d][i] - ARTIST_REPEAT_PENALTY * cur), d, i, cur))
                continue
            place(d, i)

    def openers_in(d: int) -> int:
        return sum(1 for i in groups[d] if opener[d][i])

    greedy((d, i) for d in range(n_dp) for i in range(n) if allow[d][i])

    def needed_opener(o: int, i: int) -> bool:
        return opener[o][i] and openers_in(o) <= wants[o]

    owed = [0] * n_dp   # tracks taken from a daypart for another one's opening

    def refillable(o: int) -> bool:
        """Daypart o can be given a replacement in the second pass (an unassigned track it allows)."""
        return sum(1 for j in range(n) if assigned[j] < 0 and allow[o][j]) > owed[o]

    replaced: dict[int, int] = {}   # track -> daypart it was dropped from for an opener
    for d in range(n_dp):
        need = wants[d] - openers_in(d)
        if need <= 0:
            continue
        cands = sorted((i for i in range(n) if assigned[i] != d and opener[d][i] and allow[d][i]),
                       key=lambda i: (assigned[i] >= 0, -scores[d][i], i))
        added = 0
        for i in cands:
            if added >= need:
                break
            donor = assigned[i]
            if donor >= 0:
                if needed_opener(donor, i):
                    continue   # the other daypart needs it for its own calm opening
                if scores[d][i] < scores[donor][i] and not refillable(donor):
                    continue   # fits its daypart better, and that daypart could not be refilled
                unplace(i)
                owed[donor] += 1
            place(d, i)
            added += 1
        if added:
            for i in _trim_non_openers(list(groups[d]), eligible, scores[d], dps[d], *caps[d]):
                unplace(i)
                replaced[i] = d

    # Second pass: tracks displaced by openers and the room freed in donor dayparts.
    greedy((d, i) for i in range(n) if assigned[i] < 0
           for d in range(n_dp) if allow[d][i] and replaced.get(i) != d)

    def elsewhere(i: int, d: int) -> str:
        """Why the track is in no other daypart either (state after the assignment)."""
        bits = []
        for o in range(n_dp):
            if o == d:
                continue
            if fits[o][i].avoided_tag is not None:
                bits.append(f"'{names[o]}' avoids tag '{fits[o][i].avoided_tag}'")
            elif not allow[o][i]:
                bits.append(f"'{names[o]}' poor fit")
            elif replaced.get(i) == o:
                bits.append(f"'{names[o]}' replaced it by a calmer opening track")
            elif full(o):
                bits.append(f"'{names[o]}' full ({_limit_text(*caps[o])} reached)")
            else:
                bits.append(f"'{names[o]}' had other tracks with a better score")
        return f"; other dayparts: {', '.join(bits)}" if bits else ""

    results: dict[str, SequenceResult] = {}
    for d, dp in enumerate(dps):
        not_selected = list(base_rejected)
        for i, t in enumerate(eligible):
            if assigned[i] == d:
                continue
            f = fits[d][i]
            fit_txt = _fit_text(f, eds[i][1])
            if f.avoided_tag is not None:
                reason = f"daypart '{dp.name}' avoids tag '{f.avoided_tag}'"
            elif assigned[i] >= 0:
                other = assigned[i]
                reason = (f"scheduled in daypart '{names[other]}' (fit there {fits[other][i].fit:.2f}, "
                          f"here {f.fit:.2f})")
            elif not any(allow[o][i] for o in range(n_dp)):
                reason = (f"poor fit for every daypart (best selection score "
                          f"{max(base[o][i] for o in range(n_dp)):.2f} < {MIN_SELECTION_SCORE:.2f}; "
                          f"here: {fit_txt})")
            elif replaced.get(i) == d:
                reason = (f"not selected for daypart '{dp.name}': replaced by a calmer opening track "
                          f"({fit_txt}){elsewhere(i, d)}")
            elif not allow[d][i]:
                reason = (f"poor fit for daypart '{dp.name}': selection score {base[d][i]:.2f} < "
                          f"{MIN_SELECTION_SCORE:.2f} ({fit_txt}){elsewhere(i, d)}")
            elif full(d):
                reason = (f"not selected for daypart '{dp.name}': {_limit_text(*caps[d])} reached "
                          f"({fit_txt}){elsewhere(i, d)}")
            else:   # defensive: the second pass places every allowed track while there is room
                reason = f"not selected for daypart '{dp.name}' ({fit_txt}){elsewhere(i, d)}"
            not_selected.append(_not_selected(t, reason))
        opening_context = None
        if dp.opening_max_energy is not None and openers_in(d) < wants[d]:
            held = [i for i in range(n) if assigned[i] not in (-1, d) and opener[d][i] and allow[d][i]]
            if held:
                n_open = sum(1 for i in held if needed_opener(assigned[i], i))
                parts = ([f"{n_open} open their own daypart"] if n_open else []) + (
                    [f"{len(held) - n_open} fit their daypart better"] if len(held) > n_open else [])
                opening_context = (f"the other {len(held)} track(s) with known energy <= {dp.opening_max_energy:.2f} "
                                   f"are scheduled in other dayparts ({', '.join(parts)})")
        group = sorted(groups[d])
        params = _params(cfg_eff, applied, profile, dp, caps[d][0], caps[d][1], {
            "pool_size": len(pool), "eligible": n, "selected": len(group)})
        params["plan"] = {"dayparts": names,
                          "assignment": "global fit-greedy, calm openers per daypart, no reuse"}
        results[dp.name] = _finish([eligible[i] for i in group], [fits[d][i] for i in group], cfg_eff, dp,
                                   seed, not_selected, params, quality=[base[d][i] for i in group],
                                   opening_context=opening_context)
    return results


__all__ = [
    "ALGORITHM",
    "COMPONENTS",
    "DaypartFit",
    "artist_key",
    "auto_key_confidence",
    "build_sequence",
    "camelot_compatible",
    "camelot_steps",
    "daypart_fit",
    "effective_config",
    "genre_similarity",
    "mood_similarity",
    "parse_camelot",
    "plan_dayparts",
    "transition_score",
]
