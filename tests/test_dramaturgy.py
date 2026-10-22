"""VEmix 3.0 dramaturgy engine (spec §5, §8): hard constraints before soft preferences,
curve following, determinism, alternatives, verifiable explanations.

Pure Python/NumPy (no audio, no FFmpeg)."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest

from veg_mixer.dramaturgy import (
    HARD_LIMIT_CONFIDENCE,
    Model,
    SearchSettings,
    build_context,
    evaluate_order,
    plan,
    provenance_category,
)
from veg_mixer.interpret_rules import derive
from veg_mixer.models import TrackFeatures
from veg_mixer.scenario import parse_scenario, resolve_scenario

MOODS = ["ominous", "playful", "tense", "euphoric", "mysterious", "dark", "spooky", "energetic"]
KEYS = [f"{n}{m}" for n in range(1, 13) for m in "AB"]


def mk(i: int, energy: float | None, *, artist: str | None = None, title: str | None = None, bpm: float = 110.0,
       key: str | None = "8A", mood: str | None = None, duration: float = 200.0, manual: bool = True,
       vibe: list[str] | None = None, bpm_source: str = "auto:bpm-v1") -> TrackFeatures:
    artist = artist or f"Artist {i}"
    title = title or f"Song {i}"
    src = {"artist": "id3", "title": "id3", "bpm": bpm_source, "camelot": "manual", "key_confidence": "manual"}
    if energy is not None:
        src["energy"] = "manual" if manual else "auto:energy-heuristic-v1"
    if mood:
        src["mood"] = "manual"
    if vibe:
        src["vibe_tags"] = "manual"
    rel = f"{artist} - {title}.mp3"
    return TrackFeatures(path=Path("/lib") / rel, duration_s=duration, rel_path=rel, artist=artist, title=title,
                         energy=energy, energy_confidence=(1.0 if manual else 0.4) if energy is not None else None,
                         bpm=bpm, camelot=key, key_confidence=1.0 if key else None, mood=mood,
                         vibe_tags=vibe or [], feature_sources=src)


def pool(n: int = 40, seed: int = 0, **kw) -> list[TrackFeatures]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        e = float(np.round(np.linspace(0.1, 0.95, n)[i], 3))
        out.append(mk(i, e, artist=f"Artist {i % 15}", bpm=float(np.round(rng.uniform(100, 124), 1)),
                      key=str(rng.choice(KEYS)), mood=str(rng.choice(MOODS)),
                      duration=float(np.round(rng.uniform(170, 230), 1)), **kw))
    return out


ARC = {
    "scenario": "arc",
    "target_duration_min": 40,
    "duration_tolerance_min": 3,
    "phases": [
        {"name": "intro", "share": 0.15, "energy": [0.35, 0.40], "roles": ["opener"]},
        {"name": "build", "share": 0.35, "energy": [0.40, 0.75], "roles": ["tension_builder"]},
        {"name": "climax", "share": 0.25, "energy": [0.85, 0.90], "roles": ["peak"]},
        {"name": "breather", "share": 0.15, "energy": [0.30, 0.25], "contrast": True,
         "moods": {"prefer": ["ominous"], "avoid": ["euphoric"]}},
        {"name": "closer", "share": 0.10, "energy": [0.30, 0.20], "roles": ["closer"]},
    ],
    "transitions": {"max_bpm_jump": 30, "max_energy_jump": 0.30, "contrast_energy_jump": 0.70},
    "diversity": {"artist_min_gap": 2},
}
FAST = SearchSettings(max_evals=2500, restarts=2)


def run(data: dict, tracks: list[TrackFeatures], *, seed: int = 1, alternatives: int = 1, settings=FAST, **ctx_kw):
    sc = parse_scenario(json.loads(json.dumps(data)))
    resolved = resolve_scenario(sc, tracks)
    contexts = [build_context(t, interpretations=[derive(t)], **ctx_kw) for t in resolved.pool]
    return plan(contexts, resolved, seed=seed, alternatives=alternatives, settings=settings), resolved, contexts


def order_titles(variant) -> list[str]:
    return [e.display for e in variant.entries]


# ---------------------------------------------------------------------------
# hard constraints
# ---------------------------------------------------------------------------

def test_anchors_required_and_duration():
    tracks = pool()
    data = {**ARC, "anchors": [{"track": "Artist 3 - Song 3", "position": 1},
                               {"track": "Artist 1 - Song 1", "position": -1}],
            "required": ["Artist 5 - Song 35"]}
    res, _, _ = run(data, tracks)
    v = res.variants[0]
    names = order_titles(v)
    assert names[0] == "Artist 3 - Song 3" and names[-1] == "Artist 1 - Song 1"
    assert "Artist 5 - Song 35" in names
    assert abs(v.duration_s - 2400) <= 180
    assert not [x for x in v.violations if x.kind in ("anchor", "required", "duration")]
    assert v.entries[0].anchored == "position 1" and v.entries[-1].anchored == "position -1"


def test_phase_anchor_plays_inside_its_phase():
    tracks = pool()
    data = {**ARC, "anchors": [{"track": "Artist 2 - Song 2", "phase": "climax"}]}   # a LOW-energy track
    v = run(data, tracks)[0].variants[0]
    entry = next(e for e in v.entries if e.display == "Artist 2 - Song 2")
    assert entry.phase == "climax"
    assert not [x for x in v.violations if x.kind == "phase_anchor"]


def test_forbidden_pair_beats_soft_preferences():
    # two tracks that fit each other perfectly (same energy, key, mood) are forbidden neighbours
    tracks = pool(30)
    a, b = mk(100, 0.6, mood="tense"), mk(101, 0.6, mood="tense")
    data = {**ARC, "required": [a.display_name, b.display_name], "adjacency": {"forbid": [[a.display_name,
                                                                                            b.display_name]]}}
    v = run(data, tracks + [a, b])[0].variants[0]
    names = order_titles(v)
    ia, ib = names.index(a.display_name), names.index(b.display_name)
    assert abs(ia - ib) > 1
    assert not v.violations


def test_artist_spacing_count_and_one_version_per_song():
    tracks = [mk(i, 0.3 + 0.02 * i, artist="Same") for i in range(4)] + [mk(10 + i, 0.4 + 0.02 * i) for i in range(12)]
    tracks += [mk(50, 0.5, artist="Band", title="Hit"), mk(51, 0.5, artist="Band", title="Hit (2010 Remaster)")]
    data = {**ARC, "target_duration_min": None, "duration_tolerance_min": None, "max_tracks": 10,
            "diversity": {"artist_min_gap": 2, "max_tracks_per_artist": 2, "avoid_same_song": True},
            "required": ["Band - Hit"]}
    data = {k: v for k, v in data.items() if v is not None}
    v = run(data, tracks)[0].variants[0]
    names = order_titles(v)
    assert len(names) == 10
    assert sum(1 for x in v.entries if x.artist == "Same") <= 2
    assert "Band - Hit (2010 Remaster)" not in names
    pos = [k for k, e in enumerate(v.entries) if e.artist == "Same"]
    assert all(b - a > 2 for a, b in zip(pos, pos[1:]))
    assert not v.violations


def test_unavoidable_violation_is_reported_with_measured_numbers():
    tracks = [mk(i, 0.5, artist="Solo") for i in range(5)]
    data = {"scenario": "solo", "use_all_tracks": True, "phases": [{"name": "all", "share": 1, "energy": 0.5}],
            "diversity": {"artist_min_gap": 2}}
    v = run(data, tracks)[0].variants[0]
    kinds = {x.kind for x in v.violations}
    assert kinds == {"artist_gap"}
    assert all("only" in x.message and "minimum 2" in x.message for x in v.violations)


def test_energy_limit_respected_when_feasible_and_contrast_allowed():
    tracks = pool()
    v = run(ARC, tracks)[0].variants[0]
    jumps = [e.transition["energy_jump"] for e in v.entries if e.transition and e.transition["energy_jump"] is not None]
    for e in v.entries[1:]:
        if e.transition["energy_jump"] is None:
            continue
        limit = 0.70 if e.transition["planned_contrast"] else 0.30
        assert abs(e.transition["energy_jump"]) <= limit + 1e-9
    assert jumps
    assert not [x for x in v.violations if x.kind == "energy_jump"]


def test_low_confidence_tempo_is_not_a_hard_violation():
    a = mk(1, 0.5, bpm=80.0, bpm_source="manual")
    b = mk(2, 0.5, bpm=125.0, bpm_source="filename")      # confidence 0.8 -> enforced
    c = mk(3, 0.5, bpm=125.0)
    data = {"scenario": "t", "use_all_tracks": True, "phases": [{"name": "a", "share": 1, "energy": 0.5}],
            "transitions": {"max_bpm_jump": 10}, "anchors": [{"track": a.display_name, "position": 1},
                                                              {"track": b.display_name, "position": 2}]}
    v = run(data, [a, b, c])[0].variants[0]
    assert [x.kind for x in v.violations] == ["tempo_jump"]
    # an auto BPM below the hard-limit confidence (here forced low) only warns
    lowc = build_context(mk(4, 0.5, bpm=150.0))
    assert lowc.bpm_conf >= HARD_LIMIT_CONFIDENCE          # auto BPM (0.5) is enforced by design
    assert provenance_category("auto:bpm-v1") == "measured"


# ---------------------------------------------------------------------------
# soft preferences: curve, moods, roles
# ---------------------------------------------------------------------------

def test_plan_follows_the_energy_curve():
    tracks = pool(48)
    v = run(ARC, tracks)[0].variants[0]
    by_phase: dict[str, list[float]] = {}
    for e in v.entries:
        by_phase.setdefault(e.phase, []).append(e.energy)
    assert np.mean(by_phase["climax"]) > np.mean(by_phase["build"]) > np.mean(by_phase["intro"]) - 0.05
    assert np.mean(by_phase["breather"]) < np.mean(by_phase["climax"]) - 0.3
    assert v.stats["curve_mean_abs_dev"] < 0.12


def test_breather_prefers_wanted_mood():
    # neutral pool: only the two candidates carry a mood the breather phase cares about
    tracks = [dataclasses.replace(t, mood="calm") for t in pool(30)]
    om = mk(200, 0.28, mood="ominous", title="Omen")
    eu = mk(201, 0.28, mood="euphoric", title="Party")
    res, resolved, contexts = run(ARC, tracks + [om, eu])
    v = res.variants[0]
    phases = {e.display: e.phase for e in v.entries}
    assert phases.get(eu.display_name) != "breather"           # an avoided mood never lands there
    # on the same breather slot the wanted mood is strictly cheaper than the avoided one
    model = Model(contexts, resolved)
    idx = {c.track_id: i for i, c in enumerate(contexts)}
    order = [idx[t] for t in v.order]
    slot = next(k for k, e in enumerate(v.entries) if e.phase == "breather")
    w = {k: float(resolved.scenario.weights[k]) for k in resolved.scenario.weights}

    def with_(track):
        cand = [i for i in order if i != idx[track.track_id]]
        cand.insert(slot, idx[track.track_id])
        cand = [i for k, i in enumerate(cand) if k != slot + 1] if len(cand) > len(order) else cand
        return model.evaluate(cand, w)

    assert with_(om).weighted["mood"] < with_(eu).weighted["mood"]


def test_curator_role_tag_assigns_a_role():
    t = mk(1, 0.6, vibe=["role:closer", "spooky"])
    ctx = build_context(t)
    assert ctx.roles["closer"] == 1.0 and ctx.role_sources["closer"] == "manual"
    assert "spooky" in ctx.moods


def test_community_tags_only_when_allowed_and_ai_confidence_capped():
    t = mk(1, 0.6)
    research = {"status": "matched", "match_confidence": 0.95,
                "facts": {"artist_mbid": {"value": "abc", "source": "musicbrainz"}},
                "user_tags": [{"name": "spooky", "count": 5, "source": "musicbrainz"}]}
    assert "spooky" not in build_context(t, research=research).moods
    assert build_context(t, research=research, use_community_tags=True).moods["spooky"] <= 0.3
    assert build_context(t, research=research).artist_ids == {"name:artist 1", "mbid:abc"}
    ambiguous = dict(research, status="ambiguous")
    assert build_context(t, research=ambiguous).facts == {}
    ai = {"source": "ai:fake/model", "moods": [{"label": "ominous", "confidence": 0.99}],
          "roles": [{"role": "peak", "confidence": 0.95}]}
    c = build_context(t, interpretations=[ai])
    assert c.moods["ominous"] == 0.8 and c.roles["peak"] == 0.8
    assert {p["category"] for p in c.provenance} >= {"curator", "ai"}


# ---------------------------------------------------------------------------
# determinism, alternatives, verifiability, explanations
# ---------------------------------------------------------------------------

def test_same_seed_same_plan_and_reproducible_objective():
    tracks = pool()
    r1, resolved, contexts = run(ARC, tracks, seed=5, alternatives=2)
    r2, _, _ = run(ARC, tracks, seed=5, alternatives=2)
    assert json.dumps(r1.to_dict(), sort_keys=True) == json.dumps(r2.to_dict(), sort_keys=True)
    model = Model([c for c in contexts], resolved)
    ev = evaluate_order(model, r1.variants[0].order)
    assert ev.cost == pytest.approx(r1.variants[0].objective, abs=1e-6)
    assert len(ev.violations) == len(r1.variants[0].violations)


def test_alternatives_differ_and_explain_tradeoffs():
    res = run(ARC, pool(48), alternatives=3)[0]
    a, b, c = res.variants
    assert (a.preset, b.preset, c.preset) == ("balanced", "dramatic", "smooth")
    assert a.order != b.order and a.order != c.order
    assert a.tradeoffs and b.tradeoffs and c.tradeoffs
    assert any("worse than A" in t or "better than A" in t for t in b.tradeoffs)


def test_explanations_come_from_the_objective():
    res = run(ARC, pool(), alternatives=1)[0]
    v = res.variants[0]
    total_costs = sum(sum(e.costs.values()) for e in v.entries)
    # per-position contributions add up to the objective (duration is a global term)
    assert total_costs + v.terms.get("duration", 0.0) == pytest.approx(v.objective, abs=1e-3)
    e = v.entries[2]
    assert any(line.startswith("from previous: transition") for line in e.why)
    assert any("target energy" in line for line in e.why)
    assert e.counterfactual and "objective_change" in e.counterfactual
    assert e.transition["score"] is not None and set(e.transition["components"]) >= {"energy_continuity"}
    cats = {p["category"] for p in e.provenance}
    assert "curator" in cats


def test_unknown_energy_is_flagged_not_assumed():
    tracks = pool(20) + [mk(300, None, title="Mystery")]
    data = {**ARC, "required": ["Artist 300 - Mystery"]}
    v = run(data, tracks)[0].variants[0]
    e = next(x for x in v.entries if x.display == "Artist 300 - Mystery")
    assert e.energy is None and e.deviation is None
    assert "energy unknown" in e.warnings
    assert any("curve fit not verified" in line for line in e.why)


def test_use_all_tracks_and_max_tracks_modes():
    tracks = pool(12)
    data = {**ARC, "use_all_tracks": True}
    data.pop("target_duration_min"), data.pop("duration_tolerance_min")
    assert len(run(data, tracks)[0].variants[0].order) == 12
    data = {**ARC, "max_tracks": 7}
    data.pop("target_duration_min"), data.pop("duration_tolerance_min")
    assert len(run(data, tracks)[0].variants[0].order) == 7


def test_planning_a_larger_library_is_fast():
    tracks = pool(400, seed=3)
    t0 = time.perf_counter()
    res = run(ARC, tracks, alternatives=1, settings=SearchSettings())[0]
    elapsed = time.perf_counter() - t0
    assert res.variants[0].order and elapsed < 30, elapsed


def test_plan_result_is_json_native():
    res = run(ARC, pool(), alternatives=2)[0]
    text = json.dumps(res.to_dict(), allow_nan=False)
    assert '"variants"' in text


def test_flow_matrix_matches_mood_similarity():
    from veg_mixer.moods import mood_similarity
    tracks = pool(25)
    sc = parse_scenario(json.loads(json.dumps(ARC)))
    resolved = resolve_scenario(sc, tracks)
    ctx = [build_context(t) for t in resolved.pool]
    m = Model(ctx, resolved)
    for i, a in enumerate(ctx):
        for j, b in enumerate(ctx):
            if i != j and a.moods and b.moods:
                s = mood_similarity(a.moods, b.moods)
                if s is not None:
                    assert m.FLOWC[i, j] == pytest.approx(1 - s)


def test_large_library_is_prefiltered_and_reported():
    tracks = pool(2000, seed=7)
    t0 = time.perf_counter()
    res = run(ARC, tracks, alternatives=1, settings=SearchSettings())[0]
    assert time.perf_counter() - t0 < 60
    assert any("reduced from 2000 to 600" in n for n in res.notes)
    assert res.variants[0].order


def test_contrast_with_tempo_correlated_energy_follows_the_curve():
    """Regression (Halloween demo): quiet tracks are slow, loud ones fast, the tempo limit is
    strict. A planned contrast may change tempo freely (contrast_bpm_jump unset), so the plan
    must keep the breather quiet and the climax loud instead of trading the curve for tempo."""
    tracks = []
    for i in range(26):
        e = round(0.15 + 0.8 * i / 25, 3)
        tracks.append(mk(i, e, artist=f"Artist {i % 13}", bpm=round(80 + 52 * e, 1), duration=150.0,
                         key=KEYS[i % len(KEYS)]))
    data = {**ARC, "target_duration_min": 45, "duration_tolerance_min": 4,
            "phases": [
                {"name": "build", "share": 0.3, "energy": [0.45, 0.75]},
                {"name": "peak", "share": 0.2, "energy": [0.80, 0.85]},
                {"name": "breather", "share": 0.2, "energy": [0.30, 0.25], "contrast": True},
                {"name": "climax", "share": 0.2, "energy": [0.70, 0.90]},
                {"name": "closer", "share": 0.1, "energy": [0.40, 0.30]}],
            "transitions": {"max_bpm_jump": 14, "max_energy_jump": 0.30, "contrast_energy_jump": 0.60}}
    v = run(data, tracks, settings=SearchSettings())[0].variants[0]
    by_phase: dict[str, list[float]] = {}
    for e in v.entries:
        by_phase.setdefault(e.phase, []).append(e.energy)
    assert not v.violations, [x.message for x in v.violations]
    assert max(by_phase["breather"]) < 0.45 < min(by_phase["peak"])
    # the climax climbs out of the breather (how far is a trade-off of the weights: tempo and
    # energy steps are limited, and a steep ramp costs transition quality)
    assert min(by_phase["climax"]) > max(by_phase["breather"])
    assert v.stats["curve_mean_abs_dev"] < 0.15


def test_disputed_facts_are_reported_not_used():
    research = {"status": "matched", "match_confidence": 0.9,
                "facts": {"year": {"value": 1982, "source": "musicbrainz"},
                          "first_release_date": {"value": "1982-11-30", "source": "musicbrainz"},
                          "work_mbid": {"value": "w1", "source": "musicbrainz"}},
                "conflicts": [{"field": "year", "values": [{"source": "musicbrainz", "value": 1982},
                                                          {"source": "wikidata", "value": 1983}]}]}
    c = build_context(mk(1, 0.5), research=research)
    assert "year" not in c.facts and "first_release_date" not in c.facts and c.facts["work_mbid"] == "w1"
    assert any(p["feature"] == "disputed:year" for p in c.provenance)
    assert "work:w1" in c.song_ids


def test_partly_researched_library_keeps_diversity_rules():
    """A matched track (MusicBrainz ids) and unmatched tracks of the same artist / song are still
    one artist / one song: research must never switch the diversity rules off."""
    matched = {"status": "matched", "match_confidence": 0.95,
               "facts": {"artist_mbid": {"value": "a-1", "source": "musicbrainz"},
                         "work_mbid": {"value": "w-1", "source": "musicbrainz"}}}
    hit, other, remaster = (mk(1, 0.5, artist="Band", title="Hit"), mk(2, 0.5, artist="Band", title="Other"),
                            mk(3, 0.5, artist="Band", title="Hit (2010 Remaster)"))
    # a second artist spelling that only the MusicBrainz id ties to the same artist
    alias = mk(4, 0.5, artist="The Band", title="Third")
    contexts = [build_context(hit, research=matched), build_context(other), build_context(remaster),
                build_context(alias, research={**matched, "facts": {"artist_mbid": {"value": "a-1",
                                                                                     "source": "musicbrainz"}}})]
    sc = parse_scenario({"scenario": "d", "use_all_tracks": True,
                         "phases": [{"name": "all", "share": 1, "energy": 0.5}],
                         "diversity": {"artist_min_gap": 1, "avoid_same_song": True}})
    resolved = resolve_scenario(sc, [c.track for c in contexts])
    model = Model(contexts, resolved)
    assert len(set(model.A.tolist())) == 1                       # name keys + the shared artist id
    assert model.S[0] == model.S[2] != model.S[1]               # 'Hit' == 'Hit (2010 Remaster)'
    ev = evaluate_order(model, [contexts[i].track.track_id for i in (0, 2, 1, 3)])
    kinds = {x.kind for x in ev.violations}
    assert {"artist_gap", "same_song"} <= kinds, kinds


# ---------------------------------------------------------------------------
# final review regressions
# ---------------------------------------------------------------------------

def test_pool_of_anchors_only_not_at_position_one_is_planned():
    """A pool that holds only fixed anchors (none at position 1) used to crash with IndexError."""
    t = mk(1, 0.5, artist="Solo", title="Only")
    data = {"scenario": "a", "phases": [{"name": "all", "share": 1, "energy": 0.5}],
            "anchors": [{"track": "Solo - Only", "position": 2}], "pool": {"tracks": ["Solo - Only"]}}
    v = run(data, [t, mk(2, 0.5)])[0].variants[0]
    assert v.order == [t.track_id]
    kinds = [x.kind for x in v.violations]
    assert kinds == ["anchor"]                  # reported once (not also as 'required')


def test_contrast_tempo_limit_applies_without_a_general_limit():
    tracks = [mk(1, 0.8, bpm=100.0), mk(2, 0.3, bpm=140.0)]
    data = {"scenario": "c", "use_all_tracks": True,
            "phases": [{"name": "peak", "share": 1, "energy": 0.8},
                       {"name": "calm", "share": 1, "energy": 0.3, "contrast": True}],
            "transitions": {"max_bpm_jump": None, "contrast_bpm_jump": 10, "contrast_energy_jump": 0.6}}
    v = run(data, tracks)[0].variants[0]
    assert "tempo_jump" in {x.kind for x in v.violations}


def test_automatic_keys_never_become_hard_key_limits():
    a, b = mk(1, 0.5, key="1A"), mk(2, 0.5, key="7B")
    for t in (a, b):
        t.feature_sources["camelot"] = t.feature_sources["key_confidence"] = "auto:key-v1"
        t.key_confidence = 0.95                  # correlation-like value; 2.0 calibrates it to <= 0.5
    data = {"scenario": "k", "use_all_tracks": True, "phases": [{"name": "all", "share": 1, "energy": 0.5}],
            "transitions": {"max_key_steps": 1}}
    v = run(data, [a, b])[0].variants[0]
    assert "key_jump" not in {x.kind for x in v.violations}
    assert any("not enforced" in w for e in v.entries for w in e.warnings)
    # a curated key is a hard limit
    for t in (a, b):
        t.feature_sources["camelot"] = t.feature_sources["key_confidence"] = "manual"
        t.key_confidence = 1.0
    v = run(data, [a, b])[0].variants[0]
    assert "key_jump" in {x.kind for x in v.violations}


def _segments(**summary) -> dict:
    return {"version": "segments-v1", "summary": {"energy_in": 0.5, "energy_out": 0.5, "energy_mean": 0.5, **summary}}


def test_segment_tempo_doubt_lowers_bpm_confidence_below_the_hard_limit():
    t = mk(1, 0.5, bpm=126.0)
    agree = build_context(t, segments=_segments(tempo={"bpm": 63.2, "confidence": 0.6,
                                                         "half_double_ambiguous": True}))
    assert agree.bpm_conf >= HARD_LIMIT_CONFIDENCE                     # same pulse at half time
    assert any("half / double" in n for n in agree.tempo_notes)
    other = build_context(t, segments=_segments(tempo={"bpm": 84.0, "confidence": 0.6}))
    assert other.bpm_conf < HARD_LIMIT_CONFIDENCE and "bpm" in other.low_confidence
    assert any("doubtful" in n and "84" in n for n in other.tempo_notes)
    none = build_context(t, segments=_segments(tempo={"bpm": None, "confidence": 0.0}))
    assert none.bpm_conf < HARD_LIMIT_CONFIDENCE
    curated = mk(2, 0.5, bpm=126.0, bpm_source="manual")
    assert build_context(curated, segments=_segments(tempo={"bpm": 84.0, "confidence": 0.9})).bpm_conf >= 0.5


def test_curator_mood_and_roles_outrank_ai_labels():
    t = mk(1, 0.5, mood="playful", vibe=["role:opener"])
    ai = {"source": "ai:fake/model", "moods": [{"label": "ominous", "confidence": 0.8}],
          "roles": [{"role": "peak", "confidence": 0.8}],
          "positions": [{"phase": "climax", "confidence": 0.8}]}
    c = build_context(t, interpretations=[ai])
    assert set(c.moods) == {"playful"} and set(c.roles) == {"opener"} and not c.phase_hints
    assert {p["feature"] for p in c.provenance} >= {"mood_set_aside", "role_set_aside"}
    # without curator labels the AI labels are used (capped)
    c2 = build_context(mk(2, 0.5), interpretations=[ai])
    assert c2.moods == {"ominous": 0.8} and c2.roles["peak"] == 0.8


def test_roles_no_layer_could_judge_are_unknown_not_misfits():
    """Without segment analysis the rules cannot judge tension_builder: unknown (0.5), and a
    track of unknown energy must not win the phase just for having no roles."""
    known = build_context(mk(1, 0.6), interpretations=[derive(mk(1, 0.6))])
    assert "tension_builder" not in known.role_judged
    assert Model._role_cost(known, ["tension_builder"]) == pytest.approx(0.5)
    assert "peak" in known.role_judged          # judged from the energy alone
    assert Model._role_cost(known, ["peak", "tension_builder"]) == pytest.approx(
        1.0 - min(1.0, known.roles.get("peak", 0.0) / 0.5))
    unknown = build_context(mk(2, None), interpretations=[derive(mk(2, None))])
    assert Model._role_cost(unknown, ["peak"]) == pytest.approx(0.5)


def test_ai_position_hints_follow_ai_positions_of_the_phase():
    ai = {"source": "ai:fake/model", "positions": [{"phase": "climax", "confidence": 0.8}]}
    c = build_context(mk(1, None), interpretations=[ai])
    assert Model._role_cost(c, ["peak"], ["climax"]) == 0.0
    assert Model._role_cost(c, ["peak"], []) == 1.0          # an AI layer judged every role, gave none
    sc = parse_scenario({"scenario": "p", "phases": [
        {"name": "warmup", "share": 1, "energy": 0.4},
        {"name": "top", "share": 1, "energy": 0.9, "roles": ["peak"], "ai_positions": ["climax", "first_peak"]},
        {"name": "climax", "share": 1, "energy": 0.9}]})
    assert sc.phase("warmup").ai_positions == [] and sc.phase("climax").ai_positions == ["climax"]
    assert sc.phase("top").ai_positions == ["climax", "first_peak"]
    with pytest.raises(Exception, match="unknown AI position"):
        parse_scenario({"scenario": "p", "phases": [{"name": "a", "share": 1, "energy": 0.4,
                                                     "ai_positions": ["peak"]}]})


def test_counterfactual_numbers_separate_objective_and_violations():
    tracks = pool(30)
    res, _, _ = run(ARC, tracks, alternatives=2)
    for v in res.variants:
        for e in v.entries:
            cf = e.counterfactual or {}
            if "objective_change" in cf:
                assert abs(cf["objective_change"]) < 100        # J only, never the 1000x violation scale
                if cf["violation_change"] > 0:
                    assert cf.get("violations_added")
    # variant A was polished under its own (scenario) objective: no one-move improvement is left
    assert not [w for e in res.variants[0].entries for w in e.why if w.startswith("note:")]
    # a preset variant never claims the search budget missed a move its own objective rejects
    for e in res.variants[1].entries:
        for w in e.why:
            if w.startswith("note:") and "preset" not in w:
                assert "not found within the search budget" in w


def test_use_all_tracks_beyond_the_pool_cap_is_an_error(monkeypatch):
    import veg_mixer.dramaturgy as dram
    monkeypatch.setattr(dram, "POOL_CAP", 5)
    data = {"scenario": "all", "use_all_tracks": True, "phases": [{"name": "a", "share": 1, "energy": 0.5}]}
    with pytest.raises(Exception, match="handles at most 5"):
        run(data, pool(8))


def test_excluded_tracks_keep_their_paths():
    tracks = pool(12)
    data = {**ARC, "excluded": [tracks[4].display_name]}
    res, _, _ = run(data, tracks)
    ex = {x["track_id"]: x for x in res.excluded}
    assert ex[tracks[4].track_id]["rel_path"] == tracks[4].rel_path
