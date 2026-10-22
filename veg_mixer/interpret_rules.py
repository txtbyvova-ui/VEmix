"""Rule-based (derived) track interpretation — VEmix 3.0, spec §4 without AI.

Turns MEASURED data into dramaturgical role candidates with explicitly low confidence,
so the scenario planner works fully offline and every role it uses can be traced to
numbers. The output has the same shape as an AI interpretation (``veg_mixer.ai``), with
the source ``rules:interpret-v2`` (``RULES_SOURCE``; category "derived").

What the rules use, and nothing else:

* track energy (curated or the 2.0 heuristic) and its confidence — cited as
  ``curator.energy`` or ``measured.energy`` by where it came from;
* segment analysis (``audio_intel``, when computed): energy at the start/end, the
  measured trajectory (rising / falling / stable / wave), arrangement density,
  percussive density and the detected sections (intro / break / outro ...);
* the track duration.

Mood is NOT derived: no rule maps audio measurements to "ominous" or "playful"
reliably, so moods come only from the curator, verified sources or an AI provider.
Roles are hypotheses for the optimizer (confidence <= ``MAX_RULE_CONFIDENCE``), never
facts; the plan report shows them as "derived". ``judged_roles`` lists the roles the
available data could support at all (tension_builder / comedown / release need the segment
analysis): when no layer judged any of a phase's roles for the track and there is no AI
position hint, the planner treats the role fit as unknown, not as a misfit.
"""

from __future__ import annotations

from typing import Any

from .models import TrackFeatures

RULES_VERSION = "interpret-v2"
RULES_SOURCE = f"rules:{RULES_VERSION}"
MAX_RULE_CONFIDENCE = 0.5

# energy bands on the 0..1 scale of TrackFeatures.energy
LOW, MID_LOW, MID_HIGH, HIGH = 0.35, 0.45, 0.65, 0.72
SHORT_TRACK_S = 150.0


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def _band(x: float | None, lo: float, hi: float, soft: float = 0.15) -> float:
    """1 inside [lo, hi], falling linearly to 0 at ``soft`` outside; 0 for unknown."""
    if x is None:
        return 0.0
    if lo <= x <= hi:
        return 1.0
    d = lo - x if x < lo else x - hi
    return max(0.0, 1.0 - d / soft)


def derive(track: TrackFeatures, segments: dict | None = None) -> dict:
    """Derived interpretation of one track (same keys as an AI interpretation)."""
    e = _f(track.energy)
    curated = track.feature_sources.get("energy") == "manual"
    e_conf = _f(track.energy_confidence) if not curated else 1.0
    e_conf = 0.5 if e_conf is None else max(0.0, min(1.0, e_conf))
    summary = (segments or {}).get("summary") or {}
    sections = [s.get("label") for s in (segments or {}).get("sections") or [] if isinstance(s, dict)]
    traj = summary.get("trajectory") if isinstance(summary.get("trajectory"), dict) else None
    dens = summary.get("density") if isinstance(summary.get("density"), dict) else None
    t_val = traj.get("value") if traj else None
    t_conf = _f(traj.get("confidence")) if traj else None
    d_val = dens.get("value") if dens else None
    # the segment intensities are on the analyser's own scale: shift them onto the track's
    # energy (curated or measured), as the planner does, so a scale offset is not read as
    # 'starts quieter' / 'ends quietly'
    e_in, e_out, e_mean = _f(summary.get("energy_in")), _f(summary.get("energy_out")), _f(summary.get("energy_mean"))
    if e is not None and e_mean is not None:
        e_in = None if e_in is None else max(0.0, min(1.0, e + (e_in - e_mean)))
        e_out = None if e_out is None else max(0.0, min(1.0, e + (e_out - e_mean)))
    else:
        e_in = e_out = None
    perc = _f(summary.get("percussive_density"))
    duration = _f(track.duration_s)
    has_seg = bool(summary)

    roles: dict[str, tuple[float, list[str]]] = {}

    def add(role: str, score: float, evidence: list[str]) -> None:
        # data confidence: energy confidence, and the segment analysis when used
        conf = MAX_RULE_CONFIDENCE * max(0.0, min(1.0, score)) * (0.5 + 0.5 * e_conf)
        if not has_seg:
            conf *= 0.6
        conf = round(conf, 3)
        if conf >= 0.05 and conf > roles.get(role, (0.0, []))[0]:
            roles[role] = (conf, evidence)

    ev_e = (["curator.energy"] if curated else ["measured.energy"]) if e is not None else []
    rising = t_val == "rising"
    falling = t_val == "falling"
    stable = t_val == "stable"
    tw = t_conf if t_conf is not None else 0.0

    if e is not None:
        add("peak", _band(e, HIGH, 1.0) * (1.0 if d_val in (None, "dense", "moderate") else 0.6),
            ev_e + (["measured.segments.density"] if d_val else []))
        add("breather", _band(e, 0.0, LOW) * (1.0 if d_val in (None, "sparse", "moderate") else 0.5),
            ev_e + (["measured.segments.density"] if d_val else []))
        opener = _band(e, 0.30, 0.70)
        if "intro" in sections:
            opener = min(1.0, opener + 0.3)
        if e_in is not None and e_in < e - 0.05:
            opener = min(1.0, opener + 0.2)
        add("opener", opener * 0.8, ev_e + (["measured.segments.sections"] if "intro" in sections else [])
            + (["measured.segments.energy_in"] if e_in is not None else []))
        closer = _band(e_out if e_out is not None else e, 0.0, 0.45)
        if "outro" in sections:
            closer = min(1.0, closer + 0.3)
        add("closer", closer * 0.8, ev_e + (["measured.segments.energy_out"] if e_out is not None else [])
            + (["measured.segments.sections"] if "outro" in sections else []))
        if rising:
            add("tension_builder", _band(e, MID_LOW - 0.1, HIGH) * (0.6 + 0.4 * tw),
                ev_e + ["measured.segments.trajectory"])
        elif has_seg and e_in is not None and e_out is not None and e_out - e_in > 0.08:
            add("tension_builder", _band(e, MID_LOW - 0.1, HIGH) * 0.7,
                ev_e + ["measured.segments.energy_in", "measured.segments.energy_out"])
        groove = _band(e, MID_LOW, 0.85) * (1.0 if stable else 0.6)
        if perc is not None:
            groove *= 0.5 + 0.5 * min(1.0, perc / 0.5)
        add("groove_builder", groove * 0.9, ev_e + (["measured.segments.trajectory"] if stable else [])
            + (["measured.segments.percussive_density"] if perc is not None else []))
        if falling:
            add("comedown", _band(e, 0.25, 0.65) * (0.6 + 0.4 * tw), ev_e + ["measured.segments.trajectory"])
            add("release", _band(e, 0.55, 1.0) * (0.5 + 0.3 * tw), ev_e + ["measured.segments.trajectory"])
        elif t_val == "wave":
            add("release", _band(e, 0.55, 1.0) * 0.6, ev_e + ["measured.segments.trajectory"])
        if "break" in sections and e >= MID_LOW:
            add("release", 0.5, ev_e + ["measured.segments.sections"])
    if duration is not None and duration < SHORT_TRACK_S:
        add("transition", 0.6 * (_band(e, 0.2, 0.8) if e is not None else 0.5), ["measured.duration_s"] + ev_e)
    elif e is not None and stable:
        add("transition", 0.35 * _band(e, 0.3, 0.7), ev_e + ["measured.segments.trajectory"])

    judged: set[str] = set()
    if e is not None:
        judged |= {"peak", "breather", "opener", "closer", "groove_builder", "transition"}
        if has_seg and (t_val or (e_in is not None and e_out is not None) or sections):
            judged |= {"tension_builder", "comedown", "release"}
    elif duration is not None:
        judged.add("transition")
    out: dict[str, Any] = {
        "source": RULES_SOURCE,
        "moods": [],
        "roles": [{"role": r, "confidence": c, "evidence": ev}
                  for r, (c, ev) in sorted(roles.items(), key=lambda kv: (-kv[1][0], kv[0]))],
        "energy_trajectory": ({"value": t_val, "confidence": round(t_conf or 0.0, 3),
                               "evidence": ["measured.segments.trajectory"], "source": "auto:segments"}
                              if t_val else {"value": "unknown", "confidence": 0.0, "evidence": []}),
        "sonic_density": ({"value": d_val, "confidence": round(_f(dens.get("confidence")) or 0.0, 3),
                           "evidence": ["measured.segments.density"], "source": "auto:segments"}
                          if d_val else {"value": "unknown", "confidence": 0.0, "evidence": []}),
        "danceability": {"value": None, "confidence": 0.0, "evidence": []},
        "groove": {"value": None, "confidence": 0.0, "evidence": []},
        "positions": [],
        "judged_roles": sorted(judged),
        "risky_pairings": [],
        "notes": (f"derived from {'curated' if curated else 'measured'} energy"
                  + (" and segment analysis" if has_seg else " only (no segment analysis: lower confidence)")),
    }
    return out


__all__ = ["RULES_VERSION", "RULES_SOURCE", "MAX_RULE_CONFIDENCE", "derive"]
