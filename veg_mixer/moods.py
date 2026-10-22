"""Mood vocabulary for scenarios, dramaturgy and AI interpretation (VEmix 3.0).

Each known mood label sits on the valence/arousal plane (Russell's circumplex,
−1..+1 on both axes). The coordinates are an EDITORIAL convention — a shared
vocabulary for comparing labels such as 'ominous' and 'sinister' — not a
measurement and not a claim about any recording. A label outside the table is
compared by exact (normalized) text only.

The 2.0 sequencer keeps its own mood-family table (``sequencing.MOOD_FAMILIES``)
unchanged, so playlists built without a scenario sort exactly as before; this module
is used only by the scenario planner, the rule-based interpreter and the AI schema.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

from .textnorm import normalize_tag, split_values

# label -> (valence, arousal)
MOOD_COORDINATES: dict[str, tuple[float, float]] = {
    # dark / tense quadrant
    "ominous": (-0.60, 0.30), "sinister": (-0.70, 0.40), "menacing": (-0.70, 0.60),
    "dark": (-0.60, 0.20), "spooky": (-0.40, 0.35), "creepy": (-0.60, 0.25), "eerie": (-0.50, 0.05),
    "haunting": (-0.40, -0.15), "mysterious": (-0.20, 0.05), "suspenseful": (-0.30, 0.55),
    "tense": (-0.40, 0.60), "dramatic": (-0.10, 0.60), "heavy": (-0.40, 0.70), "aggressive": (-0.60, 0.90),
    "angry": (-0.80, 0.80), "gloomy": (-0.70, -0.30),
    # sad / reflective
    "melancholic": (-0.60, -0.50), "melancholy": (-0.60, -0.50), "sad": (-0.70, -0.50),
    "bittersweet": (-0.20, -0.35), "nostalgic": (0.00, -0.40), "sentimental": (0.20, -0.40),
    "emotional": (-0.10, 0.10),
    # calm / soft
    "calm": (0.30, -0.70), "chill": (0.30, -0.50), "relaxed": (0.40, -0.60), "peaceful": (0.50, -0.70),
    "dreamy": (0.30, -0.50), "mellow": (0.30, -0.55), "soft": (0.30, -0.60), "ambient": (0.10, -0.70),
    "warm": (0.50, -0.30), "romantic": (0.50, -0.20), "tender": (0.50, -0.40), "intimate": (0.40, -0.40),
    # groove / sensual
    "sensual": (0.40, 0.00), "sexy": (0.40, 0.20), "seductive": (0.30, 0.15), "smooth": (0.40, -0.10),
    "groovy": (0.50, 0.40), "funky": (0.50, 0.50), "cool": (0.30, 0.10),
    # bright / playful
    "playful": (0.60, 0.50), "fun": (0.70, 0.60), "quirky": (0.40, 0.40), "campy": (0.60, 0.50),
    "theatrical": (0.20, 0.50), "happy": (0.80, 0.40), "cheerful": (0.80, 0.40), "joyful": (0.85, 0.50),
    "upbeat": (0.70, 0.50), "uplifting": (0.70, 0.50), "bright": (0.70, 0.30), "positive": (0.70, 0.30),
    "feel good": (0.75, 0.40),
    # high arousal
    "euphoric": (0.90, 0.90), "energetic": (0.50, 0.80), "driving": (0.20, 0.70), "intense": (-0.10, 0.90),
    "powerful": (0.20, 0.80), "party": (0.70, 0.80), "epic": (0.30, 0.80), "triumphant": (0.70, 0.70),
    "exciting": (0.60, 0.75), "hype": (0.60, 0.90), "danceable": (0.60, 0.60),
}
_COORDS = {normalize_tag(k): v for k, v in MOOD_COORDINATES.items()}
# Largest distance on the plane (corner to corner).
_MAX_DIST = math.hypot(2.0, 2.0)
# Labels further apart than this share no affinity at all.
AFFINITY_RADIUS = 1.2


def mood_labels(value: str | Iterable[str] | None) -> list[str]:
    """Normalized labels of a mood value ('Dark; Mysterious' -> ['dark', 'mysterious'])."""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    out: list[str] = []
    for item in items:
        for part in split_values(item):
            key = normalize_tag(part)
            if key and key not in out:
                out.append(key)
    return out


def is_known(label: str) -> bool:
    return normalize_tag(label) in _COORDS


def coordinates(label: str) -> tuple[float, float] | None:
    return _COORDS.get(normalize_tag(label))


def label_affinity(a: str, b: str) -> float | None:
    """1.0 for the same label, falling linearly to 0 at ``AFFINITY_RADIUS`` on the
    valence/arousal plane; None when either label is outside the vocabulary (and the
    labels differ) — an unknown relation is not a mismatch."""
    na, nb = normalize_tag(a), normalize_tag(b)
    if not na or not nb:
        return None
    if na == nb:
        return 1.0
    ca, cb = _COORDS.get(na), _COORDS.get(nb)
    if ca is None or cb is None:
        return None
    d = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    return max(0.0, 1.0 - d / AFFINITY_RADIUS)


def best_affinity(track_moods: Mapping[str, float], wanted: Iterable[str]) -> tuple[float, str, str] | None:
    """Best confidence-weighted match between a track's moods ({label: confidence}) and a
    list of wanted labels: (affinity × confidence, track label, wanted label), or None when
    no pair can be judged."""
    best: tuple[float, str, str] | None = None
    for w in wanted:
        for label, conf in track_moods.items():
            aff = label_affinity(label, w)
            if aff is None:
                continue
            score = aff * max(0.0, min(1.0, conf))
            if best is None or score > best[0]:
                best = (score, label, normalize_tag(w))
    return best


def mood_similarity(a: Mapping[str, float], b: Mapping[str, float]) -> float | None:
    """Similarity of two tracks' mood sets (best pair, weighted by the lower confidence);
    None when no pair can be judged."""
    best: float | None = None
    for la, ca in a.items():
        for lb, cb in b.items():
            aff = label_affinity(la, lb)
            if aff is None:
                continue
            score = aff * max(0.0, min(1.0, min(ca, cb)))
            best = score if best is None else max(best, score)
    return best


def arousal_of(moods: Mapping[str, float]) -> float | None:
    """Confidence-weighted mean arousal of the known labels (−1..+1), None when unknown."""
    num = den = 0.0
    for label, conf in moods.items():
        c = _COORDS.get(normalize_tag(label))
        if c is None or conf <= 0:
            continue
        num += c[1] * conf
        den += conf
    return num / den if den > 0 else None


__all__ = ["MOOD_COORDINATES", "AFFINITY_RADIUS", "mood_labels", "is_known", "coordinates",
           "label_affinity", "best_affinity", "mood_similarity", "arousal_of"]
