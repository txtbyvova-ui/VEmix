"""P1 sequencing (spec §7, §13 Sequencing 1-6): transition scoring with unknowns,
hard constraints, daypart selection/ordering, determinism and performance.

Pure-Python/NumPy module: no audio I/O, so no FFmpeg integration test is needed here."""

from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from veg_mixer.config import ConfigError, SequencingConfig
from veg_mixer.models import (
    ClientProfile,
    DaypartProfile,
    LevelSetting,
    SequenceResult,
    TrackFeatures,
    to_jsonable,
)
from veg_mixer.sequencing import (
    AUTO_KEY_CONFIDENCE_MAX,
    COMPONENTS,
    _assemble,
    _FeatureMatrix,
    artist_key,
    auto_key_confidence,
    build_sequence,
    camelot_compatible,
    camelot_steps,
    daypart_fit,
    effective_config,
    genre_similarity,
    mood_similarity,
    plan_dayparts,
    transition_score,
)
from veg_mixer.textnorm import primary_artist

CFG = SequencingConfig()
LEGACY_PATH = Path(__file__).resolve().parents[1] / "legacy" / "automix_v1.py"

MORNING = DaypartProfile(name="morning", energy_range=(0.10, 0.45), energy_curve=(0.25, 0.40),
                         opening_tracks=3, opening_max_energy=0.30, preferred_moods=["calm", "warm"],
                         avoid_tags=["aggressive"])
AFTERNOON = DaypartProfile(name="afternoon", energy_range=(0.35, 0.65), energy_curve=(0.45, 0.55),
                           opening_tracks=2, opening_max_energy=0.55, preferred_genres=["pop"])
EVENING = DaypartProfile(name="evening", energy_range=(0.55, 0.85), energy_curve=(0.60, 0.80),
                         opening_tracks=2, opening_max_energy=0.65, preferred_tags=["groovy"],
                         preferred_moods=["groovy"])
PROFILE = ClientProfile(client="test", dayparts={d.name: d for d in (MORNING, AFTERNOON, EVENING)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mk(name: str, *, artist: str | None = None, energy: float | None = None, bpm: float | None = None,
       camelot: str | None = None, mood: str | None = None, tags: tuple[str, ...] | list[str] = (),
       genre: str | None = None, duration: float = 200.0, source: str | None = "manual",
       key_conf: float | None = None, energy_conf: float | None = None, excluded: bool = False,
       rel_path: str | None = None, **kw) -> TrackFeatures:
    """Synthetic track; every provided feature is attributed to ``source``."""
    sources = {}
    if source:
        for k, v in dict(energy=energy, bpm=bpm, camelot=camelot, mood=mood, genre=genre).items():
            if v is not None:
                sources[k] = source
        if tags:
            sources["vibe_tags"] = source
    return TrackFeatures(path=Path(f"{name}.mp3"), duration_s=duration, bpm=bpm, camelot=camelot,
                         key_confidence=key_conf, energy=energy, energy_confidence=energy_conf, genre=genre,
                         mood=mood, vibe_tags=list(tags), artist=artist, title=name, excluded=excluded,
                         feature_sources=sources, rel_path=rel_path or f"{name}.mp3", **kw)


MOODS = ["calm", "happy", "groovy", "energetic", "romantic", "dark", "chill", "upbeat", "aggressive", None]
TAGS = ["summer", "acoustic", "groovy", "vocal", "retro", "dance", "night", "soft", "piano"]
GENRES = ["pop", "dance", "house", "r&b", "soul", "indie pop", "rock", "jazz", None]


def random_pool(n: int, seed: int, *, artists: int | None = None, unknown: float = 0.1,
                all_keys: bool = False) -> list[TrackFeatures]:
    rng = np.random.default_rng(seed)
    n_art = artists or max(2, n // 4)
    out = []
    for i in range(n):
        out.append(mk(
            f"t{i:05d}",
            artist=f"Artist {int(rng.integers(0, n_art))}",
            energy=float(rng.uniform(0.05, 0.95)) if rng.random() > unknown else None,
            bpm=float(rng.uniform(70, 140)) if all_keys or rng.random() > unknown else None,
            camelot=(f"{int(rng.integers(1, 13))}{'AB'[int(rng.integers(0, 2))]}"
                     if all_keys or rng.random() > unknown else None),
            key_conf=float(rng.uniform(0.3, 0.9)),
            mood=MOODS[int(rng.integers(0, len(MOODS)))],
            tags=list(rng.choice(TAGS, size=int(rng.integers(0, 3)), replace=False)),
            genre=GENRES[int(rng.integers(0, len(GENRES)))],
            duration=float(rng.uniform(150, 300)),
            source="manual" if i % 2 else "auto:test",
            energy_conf=0.4,
        ))
    return out


def titles(res: SequenceResult) -> list[str]:
    return [s.track.title for s in res.tracks]


def energies(res: SequenceResult) -> list[float | None]:
    return [s.track.energy for s in res.tracks]


def all_reasons(res: SequenceResult) -> list[str]:
    return [r for f in res.flagged for r in f.reasons]


def similar_runs(res: SequenceResult, thr: float) -> list[int]:
    """Runs of "very similar" transitions as the sequencer judges them (EDIT-5/SPEC-8: on the
    confidence-weighted similarity of the known components; >= the total whenever total >= thr)."""
    runs, run = [], 0
    for s in res.tracks[1:]:
        sim = s.transition_in.similarity
        run = run + 1 if sim is not None and sim >= thr else 0
        runs.append(run)
    return runs


@pytest.fixture(scope="module")
def legacy():
    """The original prototype (legacy/automix_v1.py), loaded by path for regression comparisons."""
    name = "automix_v1_legacy"
    spec = importlib.util.spec_from_file_location(name, LEGACY_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses resolve annotations through sys.modules
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:   # pragma: no cover - environment dependent
        sys.modules.pop(name, None)
        pytest.skip(f"legacy automix_v1 is not importable: {e}")
    yield mod
    sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# §13 Sequencing 1: harmonic compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    ("8A", "8A", True), ("8A", "9A", True), ("8A", "7A", True),
    ("12A", "1A", True), ("1A", "12A", True), ("12B", "1B", True),   # wheel wrap-around
    ("8A", "8B", True), ("8B", "8A", True),                          # relative major/minor
    ("8A", "10A", False), ("8A", "9B", False), ("8A", "3B", False), ("1B", "11B", False),
    ("08a", " 9A ", True),                                           # tolerant parsing
])
def test_camelot_compatible_table(a, b, expected):
    assert camelot_compatible(a, b) is expected


def test_camelot_rejects_invalid_codes():
    for bad in ("13A", "0B", "8C", "", "A8", "x"):
        with pytest.raises(ValueError):
            camelot_compatible(bad, "8A")
    assert camelot_steps("8A", "10A") == 2
    assert camelot_steps("8A", "9B") == 2
    assert camelot_steps("8A", "nope") is None


def test_camelot_matches_legacy_semantics_on_full_wheel(legacy):
    codes = [f"{n}{m}" for n in range(1, 13) for m in "AB"]
    for a in codes:
        for b in codes:
            assert camelot_compatible(a, b) == legacy.camelot_compatible(a, b), (a, b)


def test_harmonic_component_levels_and_clash_flag():
    base = mk("a", camelot="8A")
    comp = lambda code: transition_score(base, mk("b", camelot=code), CFG)
    assert comp("9A").components["harmonic_compatibility"] == 1.0
    assert comp("8B").components["harmonic_compatibility"] == 1.0
    assert comp("10A").components["harmonic_compatibility"] == 0.5
    assert comp("9B").components["harmonic_compatibility"] == 0.5
    clash = comp("3B")
    assert clash.components["harmonic_compatibility"] == 0.0
    assert "harmonic_clash:8A->3B" in clash.flags
    # invalid code = unknown, explicitly flagged — never "compatible"
    bad = comp("13C")
    assert bad.components["harmonic_compatibility"] is None
    assert "invalid_key:13C" in bad.flags and "unknown:key" in bad.flags


def test_bpm_half_double_time_folding():
    s = lambda x, y: transition_score(mk("a", bpm=x), mk("b", bpm=y), CFG).components["bpm_continuity"]
    assert s(70, 140) == pytest.approx(1.0)
    assert s(140, 70) == pytest.approx(1.0)
    assert s(100, 108) == pytest.approx(0.5)
    assert s(100, 116) == pytest.approx(0.0)
    assert s(92, 128) == 0.0
    assert "tempo_jump:92->128" in transition_score(mk("a", bpm=92), mk("b", bpm=128), CFG).flags


def test_mood_and_genre_similarity_tables():
    assert mood_similarity("Calm", "calm") == 1.0
    assert mood_similarity("calm", "chill") == pytest.approx(0.7)
    assert mood_similarity("calm", "aggressive") == 0.0
    assert 0.0 < mood_similarity("calm", "whimsical-unknown") < 0.5
    assert genre_similarity("Pop", "pop") == 1.0
    assert genre_similarity("house", "deep house") == pytest.approx(0.5)
    assert genre_similarity("R&B", "soul") == pytest.approx(0.5)
    assert genre_similarity("pop", "metal") == 0.0


# ---------------------------------------------------------------------------
# §13 Sequencing 2: mood + energy can outweigh harmony
# ---------------------------------------------------------------------------

def _mood_vs_harmony_tracks():
    a = mk("a", mood="calm", tags=["soft", "acoustic"], energy=0.30, bpm=100, camelot="8A", genre="pop",
           artist="A")
    x = mk("x_mood_match", mood="calm", tags=["soft", "acoustic"], energy=0.32, bpm=102, camelot="3B",
           genre="pop", artist="X")
    y = mk("y_key_match", mood="aggressive", tags=["hard", "loud"], energy=0.60, bpm=100, camelot="8A",
           genre="pop", artist="Y")
    return a, x, y


def test_mood_and_energy_outweigh_harmonic_compatibility(legacy):
    a, x, y = _mood_vs_harmony_tracks()
    assert camelot_compatible(a.camelot, y.camelot) and not camelot_compatible(a.camelot, x.camelot)
    sx, sy = transition_score(a, x, CFG), transition_score(a, y, CFG)
    assert sx.total > sy.total + 0.2, (sx, sy)
    assert sx.components["harmonic_compatibility"] == 0.0 and sy.components["harmonic_compatibility"] == 1.0

    # The legacy greedy sorter chains the harmonically compatible but mismatched track.
    legacy_order = legacy.sort_tracks_harmonic([
        {"name": t.title, "bpm": t.bpm, "camelot": t.camelot} for t in (a, x, y)])
    assert [t["name"] for t in legacy_order[:2]] == ["a", "y_key_match"]

    # The new sequencer keeps the mood/energy match adjacent, for any seed.
    for seed in range(5):
        order = titles(build_sequence([a, x, y], CFG, seed=seed))
        assert abs(order.index("a") - order.index("x_mood_match")) == 1, order


def test_output_is_not_an_ascending_bpm_spreadsheet(legacy):
    pool = random_pool(120, seed=21, unknown=0.0, all_keys=True)
    res = build_sequence(pool, CFG, seed=3)
    bpms = [s.track.bpm for s in res.tracks]
    rho = spearmanr(np.arange(len(bpms)), bpms).statistic
    assert abs(rho) < 0.5, rho

    legacy_order = legacy.sort_tracks_harmonic([{"t": t, "bpm": t.bpm, "camelot": t.camelot} for t in pool])
    legacy_tracks = [d["t"] for d in legacy_order]

    def mean_of(seq, comp=None):
        vals = [transition_score(p, q, CFG) for p, q in zip(seq, seq[1:])]
        if comp is None:
            return float(np.mean([v.total for v in vals]))
        return float(np.mean([v.components[comp] for v in vals if v.components[comp] is not None]))

    new_tracks = [s.track for s in res.tracks]
    assert mean_of(new_tracks) > mean_of(legacy_tracks) + 0.05
    assert mean_of(new_tracks, "vibe_similarity") > mean_of(legacy_tracks, "vibe_similarity")
    assert mean_of(new_tracks, "energy_continuity") > mean_of(legacy_tracks, "energy_continuity")


# ---------------------------------------------------------------------------
# §13 Sequencing 4: unknown metadata
# ---------------------------------------------------------------------------

def test_all_unknown_pair_scores_exactly_neutral():
    for neutral in (0.5, 0.4):
        cfg = dataclasses.replace(CFG, neutral_score=neutral)
        ts = transition_score(mk("a"), mk("b"), cfg)
        assert ts.total == pytest.approx(neutral)
        assert ts.coverage == 0.0
        assert all(v is None for v in ts.components.values())
        assert set(ts.components) == set(COMPONENTS)
        for feat in ("vibe", "energy", "bpm", "key", "genre"):
            assert f"unknown:{feat}" in ts.flags


def test_unknown_energy_is_neither_zero_jump_nor_maximum_jump():
    a = mk("a", energy=0.20, mood="calm", genre="pop")
    same = mk("same", energy=0.20, mood="calm", genre="pop")
    unknown = mk("unknown", energy=None, mood="calm", genre="pop")
    far = mk("far", energy=0.20 + CFG.max_energy_jump, mood="calm", genre="pop")
    s_same, s_unknown, s_far = (transition_score(a, b, CFG) for b in (same, unknown, far))
    assert s_same.total > s_unknown.total > s_far.total
    assert s_unknown.components["energy_continuity"] is None
    assert "unknown:energy" in s_unknown.flags
    assert s_far.components["energy_continuity"] == pytest.approx(0.0)
    # known: vibe (w .35, value 1) and genre (w .10, value 1); the rest pulled to neutral 0.5
    assert s_unknown.coverage == pytest.approx(0.45)
    assert s_unknown.total == pytest.approx(0.45 + 0.5 * 0.55)


def test_low_confidence_estimates_pull_towards_neutral():
    manual = transition_score(mk("a", energy=0.2), mk("b", energy=0.6), CFG)
    heuristic = transition_score(mk("a", energy=0.2, source="auto:energy-heuristic-v1", energy_conf=0.4),
                                 mk("b", energy=0.6, source="auto:energy-heuristic-v1", energy_conf=0.4), CFG)
    assert manual.components["energy_continuity"] == pytest.approx(0.0, abs=1e-12)
    assert heuristic.components["energy_continuity"] == manual.components["energy_continuity"]
    assert heuristic.coverage == pytest.approx(0.4 * manual.coverage)
    assert manual.total < heuristic.total < CFG.neutral_score
    assert "energy_jump:+0.40" in manual.flags


def test_tracks_with_unknowns_are_placed_without_fake_energy_flags():
    pool = [mk("k1", energy=0.2, mood="calm"), mk("u1", mood="calm"), mk("k2", energy=0.9, mood="happy"),
            mk("u2", genre="pop"), mk("u3"), mk("k3", energy=0.25, mood="calm")]
    res = build_sequence(pool, CFG, seed=0)
    assert sorted(titles(res)) == sorted(t.title for t in pool)
    for f in res.flagged:
        a, b = res.tracks[f.from_position - 1].track, res.tracks[f.to_position - 1].track
        if a.energy is None or b.energy is None:
            assert not any(r.startswith("energy jump") for r in f.reasons), f
        if a.energy is None and b.energy is None:
            assert "unknown energy on both sides — energy continuity not verified" in f.reasons


# ---------------------------------------------------------------------------
# §13 Sequencing 3: exclusion rules
# ---------------------------------------------------------------------------

def test_excluded_and_run_excluded_tracks_never_appear():
    keep = [mk(f"ok{i}", energy=0.3 + 0.05 * i, mood="calm") for i in range(5)]
    banned = mk("banned", energy=0.3, excluded=True, exclusion_reasons=["blocklisted artist 'Nope'"])
    flagged_only = mk("flagged_only", energy=0.3, excluded=True)
    run_excluded = mk("used_elsewhere", energy=0.3)
    duplicate = mk("dup", energy=0.3, rel_path="ok0.mp3")           # same track id as ok0
    zero = mk("zero", energy=0.3, duration=0.0)
    pool = keep + [banned, flagged_only, run_excluded, duplicate, zero]
    res = build_sequence(pool, CFG, exclude_track_ids=frozenset({run_excluded.track_id}), seed=1)

    placed = {s.track.title for s in res.tracks}
    assert placed == {t.title for t in keep}
    reasons = {d["rel_path"]: d["reason"] for d in res.not_selected}
    assert reasons["banned.mp3"] == "excluded: blocklisted artist 'Nope'"
    assert reasons["flagged_only.mp3"] == "excluded: excluded flag set in metadata"
    assert reasons["used_elsewhere.mp3"] == "excluded for this run (exclude_track_ids)"
    assert reasons["ok0.mp3"].startswith("duplicate of 'ok0.mp3'")
    assert reasons["zero.mp3"].startswith("invalid duration")
    assert all({"rel_path", "track_id", "reason"} <= set(d) for d in res.not_selected)
    assert res.total_duration_s == pytest.approx(sum(t.duration_s for t in keep))

    plans = plan_dayparts(pool, CFG, profile=PROFILE, dayparts=["morning", "evening"], seed=1)
    for r in plans.values():
        assert not {"banned", "flagged_only", "dup", "zero"} & set(titles(r))


def test_avoid_tags_exclude_tracks_from_the_daypart_and_its_opening():
    calm_aggressive = [mk(f"agg{i}", energy=0.15, mood="calm", tags=["aggressive"]) for i in range(3)]
    others = [mk(f"m{i}", energy=0.2 + 0.03 * i, mood="calm", tags=["soft"]) for i in range(6)]
    res = build_sequence(calm_aggressive + others, CFG, daypart=MORNING, seed=0)
    assert not any(t.startswith("agg") for t in titles(res))
    avoided = [d for d in res.not_selected if d["rel_path"].startswith("agg")]
    assert len(avoided) == 3 and all(d["reason"] == "daypart 'morning' avoids tag 'aggressive'" for d in avoided)


# ---------------------------------------------------------------------------
# §13 Sequencing 5: artist repetition
# ---------------------------------------------------------------------------

def test_artist_key_is_case_insensitive_and_strips_features():
    assert artist_key("The Band") == artist_key("  the   band ") == artist_key("THE BAND feat. Guest")
    assert artist_key("The Band (ft. Guest)") == "the band"
    assert artist_key(None) is None and artist_key("") is None


def test_same_artist_kept_apart_when_alternatives_exist():
    band = [mk(f"band{i}", artist=name, energy=0.40, mood="calm", tags=["soft"], bpm=100, camelot="8A",
               genre="pop")
            for i, name in enumerate(["The Band", "the band ", "THE BAND feat. Guest", "The Band"])]
    others = [mk(f"o{i}", artist=f"Other {i}", energy=0.42 + 0.01 * i, mood="chill", tags=["soft", "night"],
                 bpm=104, camelot="9A", genre="pop") for i in range(8)]
    for seed in range(5):
        res = build_sequence(band + others, CFG, seed=seed)
        pos = [s.position for s in res.tracks if s.track.title.startswith("band")]
        assert min(np.diff(sorted(pos))) >= CFG.artist_min_gap + 1, (seed, titles(res))
        assert not any("same artist" in r for r in all_reasons(res))


def test_artist_urgency_finds_the_only_feasible_pattern():
    # 4 tracks of one artist in 10 slots with gap 2: only X _ _ X _ _ X _ _ X works.
    pool = [mk(f"x{i}", artist="X") for i in range(4)] + [mk(f"y{i}", artist=f"Y{i}") for i in range(6)]
    for seed in range(3):
        res = build_sequence(pool, CFG, seed=seed)
        assert [s.position for s in res.tracks if s.track.artist == "X"] == [1, 4, 7, 10]
        assert not any("same artist" in r for r in all_reasons(res))


def test_unknown_artists_never_match_each_other():
    pool = [mk(f"u{i}", artist=None, energy=0.4, mood="calm") for i in range(6)]
    res = build_sequence(pool, CFG, seed=0)
    assert len(res.tracks) == 6
    assert not any("same artist" in r for r in all_reasons(res))


def test_single_artist_pool_is_flagged_not_crashed():
    pool = [mk(f"s{i}", artist="Solo", energy=0.4 + 0.02 * i, mood="calm") for i in range(5)]
    res = build_sequence(pool, CFG, seed=0)
    assert len(res.tracks) == 5
    artist_flags = [r for r in all_reasons(res) if r.startswith("same artist 'Solo'")]
    assert len(artist_flags) == 4
    assert all("no alternative found" in r for r in artist_flags)
    assert any("transition in: same artist" in n for s in res.tracks for n in s.notes)


# ---------------------------------------------------------------------------
# Energy jumps and similar runs
# ---------------------------------------------------------------------------

def test_energy_jumps_beyond_the_limit_are_avoided_when_possible():
    cfg = dataclasses.replace(CFG, max_energy_jump=0.15, flag_energy_jump=0.15)
    levels = [0.05 + 0.1 * i for i in range(10)]
    rng = np.random.default_rng(4)
    pool = [mk(f"e{i}", energy=e, artist=f"A{i}") for i, e in enumerate(rng.permutation(levels))]
    for seed in range(5):
        res = build_sequence(pool, cfg, seed=seed)
        jumps = np.abs(np.diff(energies(res)))
        assert jumps.max() <= 0.15 + 1e-9, energies(res)
        assert not res.flagged

    # Natural pool where a smooth path exists: no hard-limit violation either.
    natural = random_pool(60, seed=8, unknown=0.0)
    assert np.diff(sorted(t.energy for t in natural)).max() < CFG.max_energy_jump
    res = build_sequence(natural, CFG, seed=2)
    assert not any("hard limit" in r for r in all_reasons(res))


def test_unavoidable_energy_jump_is_placed_and_flagged():
    pool = [mk("low1", energy=0.10), mk("low2", energy=0.15), mk("peak", energy=0.90)]
    res = build_sequence(pool, CFG, seed=0)
    assert len(res.tracks) == 3
    hard = [r for r in all_reasons(res) if "hard limit" in r]
    assert len(hard) == 1
    assert re.fullmatch(r"energy jump [+-]0\.75 \((0\.15 -> 0\.90|0\.90 -> 0\.15)\) exceeds the hard limit "
                        r"0\.40 — no alternative found", hard[0]), hard


def test_moderate_energy_jump_is_flagged_with_readable_reason():
    res = build_sequence([mk("a", energy=0.30), mk("b", energy=0.60)], CFG, seed=0)
    assert len(res.flagged) == 1
    f = res.flagged[0]
    assert (f.from_position, f.to_position) == (1, 2)
    assert any(re.fullmatch(r"energy jump [+-]0\.30 \((0\.30 -> 0\.60|0\.60 -> 0\.30)\) exceeds 0\.25", r)
               for r in f.reasons), f.reasons


def test_tempo_and_harmonic_context_in_flag_reasons():
    a = mk("a", bpm=92, camelot="8A", mood="calm", energy=0.3, genre="pop")
    b = mk("b", bpm=128, camelot="3B", mood="aggressive", energy=0.45, genre="metal")
    res = build_sequence([a, b], CFG, seed=0)
    reasons = all_reasons(res)
    assert any(r in ("tempo jump 92 -> 128 BPM", "tempo jump 128 -> 92 BPM") for r in reasons), reasons
    assert any(r.startswith("harmonic clash") for r in reasons)
    assert any(r.startswith("low transition score") for r in reasons)


def test_similar_run_limit():
    cfg = dataclasses.replace(CFG, max_similar_run=2)
    clone = dict(mood="calm", tags=["soft"], energy=0.40, bpm=100, camelot="8A", genre="pop")
    clones = [mk(f"a{i}", **clone) for i in range(9)]
    breakers = [mk(f"b{i}", mood=m, tags=[f"tag{i}"], energy=e, bpm=104, camelot="9A", genre="pop")
                for i, (m, e) in enumerate([("happy", 0.45), ("groovy", 0.42), ("romantic", 0.48),
                                            ("energetic", 0.44)])]
    assert transition_score(clones[0], clones[1], cfg).total >= cfg.similar_threshold
    assert transition_score(clones[0], breakers[0], cfg).total < cfg.similar_threshold
    for seed in range(4):
        res = build_sequence(clones + breakers, cfg, seed=seed)
        assert max(similar_runs(res, cfg.similar_threshold)) <= 2, titles(res)
        assert not any("very similar" in r for r in all_reasons(res))

    only_clones = build_sequence(clones[:6], cfg, seed=0)
    assert any("very similar transitions in a row" in r and "no alternative found" in r
               for r in all_reasons(only_clones))


# ---------------------------------------------------------------------------
# Determinism / controlled variation
# ---------------------------------------------------------------------------

def test_same_seed_same_result_and_seed_changes_variation():
    pool = random_pool(40, seed=5)
    r1, r2 = build_sequence(pool, CFG, seed=7), build_sequence(pool, CFG, seed=7)
    assert to_jsonable(r1) == to_jsonable(r2)
    orders = {tuple(titles(build_sequence(pool, CFG, seed=s))) for s in range(7, 12)}
    assert len(orders) > 1

    still = dataclasses.replace(CFG, variation=0.0)
    fixed = {tuple(titles(build_sequence(pool, still, seed=s))) for s in range(3)}
    assert len(fixed) == 1   # the seed only matters through the variation jitter


def test_variation_keeps_quality():
    pool = random_pool(60, seed=6)
    still = build_sequence(pool, dataclasses.replace(CFG, variation=0.0), seed=0)
    varied = build_sequence(pool, CFG, seed=0)
    mean = lambda r: np.mean([s.transition_in.total for s in r.tracks[1:]])
    assert mean(varied) > mean(still) - 0.03


# ---------------------------------------------------------------------------
# §13 Sequencing 6: daypart profiles
# ---------------------------------------------------------------------------

def test_daypart_fit_components():
    inside = daypart_fit(mk("in", energy=0.3, mood="calm"), MORNING)
    assert inside.components["energy"] == pytest.approx(1.0) and inside.components["preferences"] == 1.0
    outside = daypart_fit(mk("out", energy=0.9, mood="energetic"), MORNING)
    assert outside.components["energy"] == pytest.approx(0.0)
    assert "energy 0.90 above the morning range 0.10-0.45" in outside.mismatches
    unknown = daypart_fit(mk("unk"), MORNING)
    assert unknown.components["energy"] == 0.5 and unknown.components["preferences"] == 0.5
    assert any(n.startswith("energy unknown — neutral daypart fit") for n in unknown.notes)
    assert outside.fit < unknown.fit < inside.fit
    family = daypart_fit(mk("fam", energy=0.3, mood="chill"), MORNING)   # calm family, not listed
    assert family.components["preferences"] == pytest.approx(0.7)


def test_unknown_energy_is_not_assumed_to_fit_a_daypart():
    known = [mk(f"k{i}", energy=0.25 + 0.02 * i, mood="calm") for i in range(4)]
    unknown = [mk(f"u{i}", mood="calm") for i in range(4)]
    res = build_sequence(unknown + known, CFG, daypart=MORNING, max_tracks=4, seed=0)
    assert sorted(titles(res)) == [f"k{i}" for i in range(4)]
    assert all("not selected for daypart 'morning'" in d["reason"] and "energy unknown" in d["reason"]
               for d in res.not_selected)


def test_morning_is_calmer_than_evening_from_the_same_pool():
    pool = random_pool(120, seed=11)
    morning = build_sequence(pool, CFG, daypart=MORNING, max_tracks=15, seed=1)
    evening = build_sequence(pool, CFG, daypart=EVENING, max_tracks=15, seed=1)
    m = [e for e in energies(morning) if e is not None]
    ev = [e for e in energies(evening) if e is not None]
    assert len(morning.tracks) == len(evening.tracks) == 15
    assert np.mean(m) < np.mean(ev) - 0.25, (np.mean(m), np.mean(ev))
    assert set(titles(morning)) != set(titles(evening))
    for s in morning.tracks[:MORNING.opening_tracks]:
        assert s.track.energy is not None and s.track.energy <= MORNING.opening_max_energy
    for s in evening.tracks[:EVENING.opening_tracks]:
        assert s.track.energy is not None and s.track.energy <= EVENING.opening_max_energy
    assert morning.daypart == "morning" and morning.params["daypart"]["name"] == "morning"
    assert all(s.daypart_fit is not None for s in morning.tracks)
    assert not any("aggressive" in (s.track.mood or "") for s in morning.tracks)
    # energy curve: the morning rises gently rather than wandering
    first, last = np.mean(m[:5]), np.mean(m[-5:])
    assert first <= last + 0.05


def test_calm_openers_are_reserved_in_the_selection():
    dp = DaypartProfile(name="lounge", energy_range=(0.0, 1.0), opening_tracks=2, opening_max_energy=0.30,
                        preferred_tags=["groovy"])
    groovy = [mk(f"g{i}", energy=0.6 + 0.02 * i, tags=["groovy"], artist=f"G{i}") for i in range(6)]
    calm = [mk(f"c{i}", energy=0.2, tags=["soft"], artist=f"C{i}") for i in range(3)]
    res = build_sequence(groovy + calm, CFG, daypart=dp, max_tracks=4, seed=0)
    assert len(res.tracks) == 4
    assert [s.track.energy for s in res.tracks[:2]] == [0.2, 0.2]
    replaced = [d for d in res.not_selected if "replaced by a calmer opening track" in d["reason"]]
    assert len(replaced) == 2


def test_editorial_scores_drive_selection():
    a, b = mk("liked", energy=0.3, mood="calm"), mk("disliked", energy=0.3, mood="calm")
    res = build_sequence([b, a], CFG, daypart=MORNING, max_tracks=1, seed=0,
                         editorial_scores={a.track_id: 1.0, b.track_id: -1.0})
    assert titles(res) == ["liked"]
    assert "editorial score -1.00" in res.not_selected[0]["reason"]


def test_target_duration_selection():
    pool = [mk(f"d{i}", energy=0.3, mood="calm", artist=f"A{i}", duration=200.0) for i in range(20)]
    res = build_sequence(pool, CFG, target_duration_s=1000.0, seed=0)
    assert 1000.0 <= res.total_duration_s < 1200.0
    assert len(res.not_selected) == 20 - len(res.tracks)
    assert all("target duration 1000 s reached" in d["reason"] for d in res.not_selected)
    with pytest.raises(ValueError):
        build_sequence(pool, CFG, max_tracks=0)


def test_plan_dayparts_changes_selection_and_never_reuses_tracks():
    pool = random_pool(90, seed=13)
    eligible = [t for t in pool if not t.excluded]
    k = len(eligible) // 3
    plans = plan_dayparts(pool, CFG, profile=PROFILE, dayparts=["morning", "afternoon", "evening"],
                          seed=2, max_tracks=k)
    sets = {name: set(titles(r)) for name, r in plans.items()}
    assert not (sets["morning"] & sets["afternoon"] or sets["morning"] & sets["evening"]
                or sets["afternoon"] & sets["evening"])
    mean = {name: np.mean([e for e in energies(r) if e is not None]) for name, r in plans.items()}
    assert mean["morning"] < mean["afternoon"] < mean["evening"], mean
    assert mean["evening"] - mean["morning"] > 0.25
    for s in plans["morning"].tracks[:MORNING.opening_tracks]:
        assert s.track.energy is not None and s.track.energy <= MORNING.opening_max_energy

    # Not an equal three-way split of one global list.
    global_order = titles(build_sequence(pool, CFG, seed=2))
    thirds = [set(global_order[i * k:(i + 1) * k]) for i in range(3)]
    assert sets["morning"] != thirds[0] and sets["afternoon"] != thirds[1] and sets["evening"] != thirds[2]
    by_title = {t.title: t for t in eligible}

    def mean_fit(names, dp):
        return np.mean([daypart_fit(by_title[x], dp).fit for x in names])

    for (name, dp), third in zip((("morning", MORNING), ("afternoon", AFTERNOON), ("evening", EVENING)), thirds):
        assert mean_fit(sets[name], dp) > mean_fit(third, dp) + 0.03, name   # min 0.065 over 20 seeds

    # Each daypart explains where the other tracks went.
    reasons = [d["reason"] for d in plans["morning"].not_selected]
    assert any(r.startswith("scheduled in daypart 'evening'") for r in reasons)
    assert plans["morning"].params["plan"]["dayparts"] == ["morning", "afternoon", "evening"]


def test_plan_dayparts_without_limits_assigns_every_track_once_by_fit():
    low = [mk(f"low{i}", energy=0.15 + 0.01 * i, mood="calm", artist=f"L{i}") for i in range(5)]
    mid = [mk(f"mid{i}", energy=0.50 + 0.01 * i, genre="pop", artist=f"M{i}") for i in range(5)]
    high = [mk(f"high{i}", energy=0.75 + 0.01 * i, tags=["groovy"], artist=f"H{i}") for i in range(5)]
    angry = mk("angry", energy=0.2, mood="aggressive")
    pool = low + mid + high + [angry]
    plans = plan_dayparts(pool, CFG, profile=PROFILE, dayparts=["morning", "afternoon", "evening"], seed=0)
    placed = [t for r in plans.values() for t in titles(r)]
    assert len(placed) == len(set(placed))                # no reuse
    # every track is placed except 'angry', which may be left out as a poor fit everywhere
    # (avoided by morning, far below the afternoon/evening energy bands) — never silently
    assert set(placed) | {"angry"} == {t.title for t in pool}
    if "angry" not in placed:
        reasons = [d["reason"] for d in plans["afternoon"].not_selected if d["rel_path"].startswith("angry")]
        assert reasons and "poor fit for every daypart" in reasons[0]
    assert set(titles(plans["morning"])) == {t.title for t in low}
    assert set(titles(plans["afternoon"])) >= {t.title for t in mid}
    assert set(titles(plans["evening"])) == {t.title for t in high}
    assert "angry" not in titles(plans["morning"])


def test_energy_band_breaks_ties_inside_overlapping_ranges():
    """Inside overlapping ranges the daypart whose energy band is closest wins (no flat 1.0 ties)."""
    calm, groove = mk("calm", energy=0.25), mk("groove", energy=0.75)
    dps = {"morning": MORNING, "afternoon": AFTERNOON, "evening": EVENING}
    calm_fits = {n: daypart_fit(calm, d).fit for n, d in dps.items()}
    groove_fits = {n: daypart_fit(groove, d).fit for n, d in dps.items()}
    assert max(calm_fits, key=calm_fits.get) == "morning"
    assert max(groove_fits, key=groove_fits.get) == "evening"
    # same artist does not push a better-fitting track into another daypart
    pool = [mk("a1", energy=0.30, artist="A", mood="calm"), mk("a2", energy=0.28, artist="A", mood="calm"),
            mk("m1", energy=0.50, artist="M"), mk("e1", energy=0.75, artist="E")]
    plans = plan_dayparts(pool, CFG, profile=PROFILE, dayparts=["morning", "afternoon", "evening"], seed=0)
    assert {"a1", "a2"} <= set(titles(plans["morning"]))
    assert "e1" in titles(plans["evening"])


def test_poor_fit_low_preference_track_is_left_out_and_never_opens():
    """A low-preference experimental piece below every energy band is not used to open a daypart
    (it would cost only one transition there) — it is left out with an explicit reason."""
    drone = mk("drone", energy=0.02, mood="dark", tags=["experimental"], artist="D")
    good = [mk(f"g{i}", energy=0.25 + 0.03 * i, mood="calm", artist=f"G{i}") for i in range(5)]
    editorial = {drone.track_id: -0.6, **{t.track_id: 1.0 for t in good}}
    res = build_sequence(good + [drone], CFG, profile=PROFILE, daypart=MORNING, editorial_scores=editorial, seed=0)
    assert "drone" not in titles(res)
    reason = next(d["reason"] for d in res.not_selected if d["rel_path"] == "drone.mp3")
    assert reason.startswith("poor fit for daypart 'morning'")
    # without a daypart nothing is dropped, but the editorially weakest track does not open
    res2 = build_sequence(good + [drone], CFG, profile=PROFILE, editorial_scores=editorial, seed=0)
    assert "drone" in titles(res2) and titles(res2)[0] != "drone"


def test_plan_dayparts_accepts_profile_objects_and_rejects_unknown_names():
    pool = random_pool(12, seed=1)
    plans = plan_dayparts(pool, CFG, profile=None, dayparts=[MORNING, EVENING], seed=0, max_tracks=4)
    assert list(plans) == ["morning", "evening"]
    with pytest.raises(ValueError):
        plan_dayparts(pool, CFG, profile=PROFILE, dayparts=["no-such-daypart"])
    with pytest.raises(ValueError):
        plan_dayparts(pool, CFG, profile=PROFILE, dayparts=["morning", "morning"])


# ---------------------------------------------------------------------------
# Client profile → effective configuration
# ---------------------------------------------------------------------------

def test_profile_effects_on_limits_and_weights():
    low = ClientProfile(client="c", energy_jumps=LevelSetting.LOW, vibe_consistency=LevelSetting.HIGH)
    cfg, applied = effective_config(CFG, low)
    assert cfg.max_energy_jump == pytest.approx(0.30) and cfg.flag_energy_jump == pytest.approx(0.20)
    assert sum(cfg.weights.values()) == pytest.approx(1.0)
    assert cfg.weights["vibe_similarity"] == pytest.approx(0.35 * 1.3 / (1 + 0.35 * 0.3))
    assert set(applied) == {"energy_jumps", "vibe_consistency"}

    high = ClientProfile(client="c", energy_jumps="high", vibe_consistency="low")
    cfg_h, _ = effective_config(CFG, high)
    assert cfg_h.max_energy_jump == pytest.approx(0.60) and cfg_h.flag_energy_jump == pytest.approx(0.375)
    assert cfg_h.max_similar_run == CFG.max_similar_run - 1
    assert cfg_h.weights["vibe_similarity"] < CFG.weights["vibe_similarity"]

    # The profile changes what gets flagged: a 0.22 jump is fine by default, not for "low" jumps.
    pair = [mk("a", energy=0.30), mk("b", energy=0.52)]
    assert not build_sequence(pair, CFG, seed=0).flagged
    strict = build_sequence(pair, CFG, profile=low, seed=0)
    assert any("exceeds 0.20" in r for r in all_reasons(strict))
    assert strict.params["max_energy_jump"] == pytest.approx(0.30)

    with pytest.raises(ConfigError):
        effective_config(dataclasses.replace(CFG, weights={k: 0.0 for k in COMPONENTS}))
    with pytest.raises(ConfigError):
        effective_config(dataclasses.replace(CFG, beam_width=0))


# ---------------------------------------------------------------------------
# Internals: vectorized rows == scalar scorer; performance; serialization
# ---------------------------------------------------------------------------

def test_vectorized_rows_match_scalar_transition_score():
    pool = random_pool(50, seed=3, unknown=0.3)
    pool[0].camelot = "13C"           # invalid key
    pool[1].key_confidence = -0.2     # correlation below zero
    pool[2].mood = "  "               # blank = unknown
    pool[3].bpm = -5.0                # invalid tempo
    pool[4].genre, pool[5].genre = "Pop/Rock", "pop; dance"     # multi-valued labels (EDIT-13)
    pool[6].mood, pool[7].mood = "Feel-Good", "calm, warm"
    for t in pool[8:14]:                                        # automatic key: calibrated confidence (EDIT-6)
        t.feature_sources["camelot"] = "auto:key-v1"
    for cfg in (CFG, effective_config(CFG, ClientProfile(client="c", energy_jumps="low"))[0]):
        fm = _FeatureMatrix(pool, cfg)
        for i in range(len(pool)):
            row, jump, sim = fm.row(i)
            for j in range(len(pool)):
                ts = transition_score(pool[i], pool[j], cfg)
                assert row[j] == pytest.approx(ts.total, abs=1e-12), (i, j)
                if ts.similarity is None:
                    assert sim[j] == -1.0, (i, j)
                else:
                    assert sim[j] == pytest.approx(ts.similarity, abs=1e-12), (i, j)


def test_large_pool_is_fast_and_complete():
    pool = random_pool(1500, seed=99)
    start = time.perf_counter()
    res = build_sequence(pool, CFG, seed=1)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, elapsed
    assert len(res.tracks) == 1500 and len({s.track.track_id for s in res.tracks}) == 1500
    assert [s.position for s in res.tracks] == list(range(1, 1501))
    assert res.tracks[0].transition_in is None
    assert all(s.transition_in is not None for s in res.tracks[1:])
    assert not any("hard limit" in r or "same artist" in r for r in all_reasons(res))


def test_result_is_json_serializable():
    res = build_sequence(random_pool(15, seed=2), CFG, daypart=MORNING, profile=PROFILE, seed=4)
    data = json.loads(json.dumps(to_jsonable(res), ensure_ascii=False))
    assert data["seed"] == 4 and data["daypart"] == "morning"
    assert data["params"]["weights"].keys() == set(COMPONENTS)
    assert data["tracks"][0]["position"] == 1


def test_empty_and_single_track_pools():
    empty = build_sequence([], CFG)
    assert empty.tracks == [] and empty.total_duration_s == 0.0
    one = build_sequence([mk("solo", energy=0.5)], CFG)
    assert titles(one) == ["solo"] and one.tracks[0].transition_in is None and not one.flagged


# ---------------------------------------------------------------------------
# Review regressions (finding ids in the test names)
# ---------------------------------------------------------------------------

EFF = effective_config(CFG)[0]


def hard_violations(seq: list[TrackFeatures], cfg: SequencingConfig = EFF) -> int:
    """Hard-constraint violations of an order, judged without the sequencer's internals: same
    artist within ``artist_min_gap``, |ΔE| > ``max_energy_jump`` between consecutive KNOWN
    energies (across unknown-energy tracks), more than ``max_similar_run`` very similar
    transitions in a row."""
    gap, mj = cfg.artist_min_gap, cfg.max_energy_jump
    v, run, last_e = 0, 0, None
    for p, t in enumerate(seq):
        a = artist_key(t.artist)
        if a and any(artist_key(seq[p - k].artist) == a for k in range(1, gap + 1) if p - k >= 0):
            v += 1
        if t.energy is not None:
            if last_e is not None and abs(t.energy - last_e) > mj + 1e-9:
                v += 1
            last_e = t.energy
        if p:
            sim = transition_score(seq[p - 1], t, cfg).similarity
            run = run + 1 if sim is not None and sim >= cfg.similar_threshold else 0
            v += run > cfg.max_similar_run
    return v


def brute_force_feasible(pool: list[TrackFeatures], cfg: SequencingConfig = EFF) -> bool:
    """Does ANY order of the pool satisfy every hard constraint? (exhaustive; small pools only)"""
    n = len(pool)
    art = [artist_key(t.artist) for t in pool]
    sim = [[(s := transition_score(a, b, cfg).similarity) is not None and s >= cfg.similar_threshold
            for b in pool] for a in pool]
    gap, mj, max_run = cfg.artist_min_gap, cfg.max_energy_jump, cfg.max_similar_run
    for perm in itertools.permutations(range(n)):
        run, last_e, ok = 0, None, True
        for p, j in enumerate(perm):
            if art[j] and any(art[perm[p - k]] == art[j] for k in range(1, gap + 1) if p - k >= 0):
                ok = False
                break
            e = pool[j].energy
            if e is not None:
                if last_e is not None and abs(e - last_e) > mj + 1e-9:
                    ok = False
                    break
                last_e = e
            run = run + 1 if p and sim[perm[p - 1]][j] else 0
            if run > max_run:
                ok = False
                break
        if ok:
            return True
    return False


def hard_reasons(res: SequenceResult) -> list[str]:
    return [r for r in all_reasons(res)
            if "hard limit" in r or r.startswith("same artist") or "very similar transitions" in r]


def test_edit3_review_case_with_a_valid_order_is_not_violated():
    """EDIT-3: the reviewer's 7-track pool has a violation-free order; the beam alone emitted
    'energy jump +0.58 ... no alternative found'."""
    spec = [("A3", .16, "upbeat"), ("A1", .42, "warm"), ("A3", .18, "upbeat"), ("A1", .37, "warm"),
            ("A2", .39, "warm"), ("A2", .76, "upbeat"), ("A2", .54, "warm")]
    pool = [mk(f"t{i}", artist=a, energy=e, mood=m) for i, (a, e, m) in enumerate(spec)]
    assert brute_force_feasible(pool)
    for cfg in (CFG, dataclasses.replace(CFG, variation=0.0)):
        for seed in range(4):
            res = build_sequence(pool, cfg, seed=seed)
            assert hard_violations([s.track for s in res.tracks]) == 0, titles(res)
            assert not hard_reasons(res)


def test_edit3_matches_brute_force_on_small_pools():
    """EDIT-3: whenever an exhaustive search finds an order without hard violations, so does
    the sequencer (n <= 7, all three hard constraints)."""
    rng = np.random.default_rng(5)
    feasible = 0
    for trial in range(60):
        n = 7
        pool = []
        for i in range(n):
            band = int(rng.integers(0, 3))
            e = round(float(rng.uniform(*((0.05, 0.3), (0.3, 0.6), (0.6, 0.95))[band])), 2)
            pool.append(mk(f"t{i}", artist=f"A{int(rng.integers(0, 4))}", energy=e,
                           mood=str(rng.choice(["calm", "warm", "upbeat", "groovy"]))))
        res = build_sequence(pool, CFG, seed=trial)
        for r in hard_reasons(res):
            assert r.endswith(("no alternative found", "the search could not avoid this")), r
        if brute_force_feasible(pool):
            feasible += 1
            assert hard_violations([s.track for s in res.tracks]) == 0, (trial, titles(res), hard_reasons(res))
        else:
            assert hard_reasons(res), trial
    assert feasible >= 20


def test_edit3_repair_finds_valid_orders_in_tight_medium_pools():
    """EDIT-3: tight pools (3-5 artists, energy steps close to the limit) built around a known
    valid order; the beam alone missed such orders in ~1/3 of the reviewer's pools (n = 12..24)."""
    rng = np.random.default_rng(41)
    moods = ["calm", "warm", "upbeat", "groovy", "dark", "happy"]
    checked = 0
    while checked < 12:
        n, n_art = int(rng.integers(12, 25)), int(rng.integers(3, 6))
        order, e = [], float(rng.uniform(0.1, 0.9))
        for i in range(n):
            recent = {t.artist for t in order[-CFG.artist_min_gap:]}
            artist = str(rng.choice([f"A{k}" for k in range(n_art) if f"A{k}" not in recent]))
            e = float(np.clip(e + rng.uniform(-0.38, 0.38), 0.05, 0.95))
            order.append(mk(f"t{i}", artist=artist, energy=round(e, 2), mood=str(rng.choice(moods))))
        if hard_violations(order):
            continue   # the random walk itself broke a run limit: draw again
        checked += 1
        pool = [order[i] for i in rng.permutation(n)]
        res = build_sequence(pool, CFG, seed=checked)
        assert hard_violations([s.track for s in res.tracks]) == 0, (n, n_art, hard_reasons(res))


def test_edit3_wording_says_unavoidable_only_when_proven():
    """EDIT-3: 'no alternative found' only when proven (exhaustive search, or a bound that holds
    for every order); otherwise 'the search could not avoid this'."""
    a, b = mk("a", energy=0.10, artist="X"), mk("b", energy=0.30, artist="Y")
    c, d = mk("c", energy=0.75, artist="Z"), mk("d", energy=0.50, artist="W")
    _, flagged = _assemble([a, b, c], [None] * 3, EFF, None)       # the 0.30 -> 0.75 gap is forced
    hard = [r for f in flagged for r in f.reasons if "hard limit" in r]
    assert len(hard) == 1 and hard[0].endswith("no alternative found"), hard
    _, flagged = _assemble([a, c, b, d], [None] * 4, EFF, None)    # avoidable: a, b, d, c is smooth
    hard = [r for f in flagged for r in f.reasons if "hard limit" in r]
    assert len(hard) == 2 and all(r.endswith("the search could not avoid this") for r in hard), hard
    _, flagged = _assemble([a, c, b, d], [None] * 4, EFF, None, proven=True)
    assert all(r.endswith("no alternative found") for f in flagged for r in f.reasons if "hard limit" in r)
    # an artist with more tracks than the gap rule can space out: forced, hence "no alternative"
    solo = [mk(f"s{i}", artist="Solo", energy=0.4, mood=m) for i, m in enumerate(["calm", "happy", "dark", "groovy"])]
    res = build_sequence(solo + [mk("o", artist="O", energy=0.4)], CFG, seed=0)
    assert hard_reasons(res) and all(r.endswith("no alternative found") for r in hard_reasons(res))


def test_edit4_unknown_energy_track_cannot_hide_an_energy_jump():
    """EDIT-4: an unknown-energy track between 0.90 and 0.15 used to satisfy the hard limit
    with no flag at all; continuity is now checked against the last KNOWN energy."""
    low = [mk(f"low{i}", artist=f"L{i}", energy=e, mood="calm", genre="pop")
           for i, e in enumerate([.10, .12, .15, .18])]
    high = [mk(f"high{i}", artist=f"H{i}", energy=e, mood="calm", genre="pop")
            for i, e in enumerate([.82, .85, .88, .90])]
    unknown = mk("unknownE", artist="U", mood="calm", genre="pop")
    res = build_sequence(low + high + [unknown], CFG, seed=0)
    known = [e for e in energies(res) if e is not None]
    big = [(x, y) for x, y in zip(known, known[1:]) if abs(y - x) > CFG.max_energy_jump]
    assert len(big) == 1                       # the low/high gap is crossed exactly once ...
    assert len([r for r in all_reasons(res) if "hard limit" in r]) == 1   # ... and that crossing is flagged
    assert hard_violations([s.track for s in res.tracks]) == 1

    # where a smooth path exists the unknown track is not used as an invisible bridge
    mids = [mk(f"mid{i}", artist=f"M{i}", energy=e, mood="warm", genre="pop") for i, e in enumerate([.40, .50, .60])]
    for seed in range(3):
        res = build_sequence(low + mids + high + [unknown], CFG, seed=seed)
        assert hard_violations([s.track for s in res.tracks]) == 0, energies(res)

    # the span across unknown tracks is reported in its own words (soft and hard)
    a, u, b = mk("a", energy=0.30), mk("u"), mk("b", energy=0.60)
    _, flagged = _assemble([a, u, b], [None] * 3, EFF, None)
    assert flagged[-1].reasons[0] == ("energy continuity not verified across 1 unknown-energy track(s): "
                                      "energy jump +0.30 (0.30 -> ? -> 0.60) exceeds 0.25")
    _, flagged = _assemble([a, u, mk("u2"), mk("c", energy=0.95)], [None] * 4, EFF, None)
    assert any(r.startswith("energy continuity not verified across 2 unknown-energy track(s): energy jump +0.65")
               and "exceeds the hard limit 0.40" in r for f in flagged for r in f.reasons)


def _auto_track(name: str, **kw) -> TrackFeatures:
    """Uncurated track: automatic BPM / key / energy estimates, ID3 genre."""
    src = {}
    if "bpm" in kw:
        src["bpm"] = "auto:bpm-v1"
    if "camelot" in kw:
        src["camelot"] = src["key_confidence"] = "auto:key-v1"
    if "energy" in kw:
        src["energy"] = "auto:energy-heuristic-v1"
        kw.setdefault("energy_confidence", 0.4)
    if "genre" in kw:
        src["genre"] = "id3"
    return TrackFeatures(path=Path(f"{name}.mp3"), duration_s=200.0, title=name, feature_sources=src, **kw)


def test_spec8_edit5_similar_run_limit_applies_to_uncurated_libraries():
    """SPEC-8 / EDIT-5: identical auto-analysed tracks scored ~0.66 (shrunk towards neutral) and
    could never reach similar_threshold 0.92, so they were chained without limit."""
    clone = dict(bpm=120.0, camelot="8A", key_confidence=0.9, energy=0.5, genre="Pop")
    clones = [_auto_track(f"c{i}", artist=f"C{i}", **clone) for i in range(8)]
    varied = [_auto_track(f"v{i}", artist=f"V{i}", bpm=100.0 + 9 * i, camelot=f"{1 + i}B", key_confidence=0.9,
                          energy=0.3 + 0.05 * i, genre=g) for i, g in enumerate(["Rock", "Jazz", "Soul", "Funk"] * 2)]
    ts = transition_score(clones[0], clones[1], EFF)
    assert ts.total < EFF.similar_threshold                  # the old metric could not see it
    assert ts.similarity == pytest.approx(1.0) and ts.coverage >= 0.25
    for seed in range(3):
        res = build_sequence(clones + varied, CFG, seed=seed)
        assert max(similar_runs(res, EFF.similar_threshold)) <= EFF.max_similar_run, titles(res)
        longest = run = 0
        for s in res.tracks:
            run = run + 1 if s.track.title.startswith("c") else 0
            longest = max(longest, run)
        assert longest <= EFF.max_similar_run + 1, titles(res)
        assert not any("very similar" in r for r in all_reasons(res))
    # too little evidence to call two tracks "excessively similar": one component, or low coverage
    assert transition_score(mk("a", energy=0.5), mk("b", energy=0.5), EFF).similarity is None
    assert transition_score(_auto_track("a", energy=0.5), _auto_track("b", energy=0.5), EFF).similarity is None
    # two automatic components (energy + ID3 genre, coverage 0.18) are enough since the SPEC-8
    # re-verification lowered MIN_SIMILARITY_COVERAGE to 0.15 (untagged libraries need it)
    assert transition_score(_auto_track("a", energy=0.5, genre="Pop"),
                            _auto_track("b", energy=0.5, genre="Pop"), EFF).similarity == pytest.approx(1.0)
    assert transition_score(_auto_track("a", energy=0.5, bpm=100.0, genre="Pop"),
                            _auto_track("b", energy=0.5, bpm=100.0, genre="Pop"), EFF).similarity == pytest.approx(1.0)


def test_edit6_harmony_does_not_outweigh_tempo_or_genre_for_auto_features():
    """EDIT-6: the raw key correlation (0.85) gave harmony a larger effective weight than auto
    tempo and ID3 genre, inverting spec §7 priorities 4 (genre) and 5 (tempo) over 6 (harmony)."""
    common = dict(key_confidence=0.85)
    a = _auto_track("a", bpm=120.0, camelot="8A", genre="Pop", **common)
    same_tempo_clash = _auto_track("x", bpm=120.0, camelot="2B", genre="Pop", **common)
    tempo_break_key_ok = _auto_track("y", bpm=136.0, camelot="8A", genre="Pop", **common)
    genre_change_key_ok = _auto_track("g", bpm=120.0, camelot="8A", genre="Metal", **common)
    key_ok = _auto_track("k", bpm=120.0, camelot="9A", genre="Pop", **common)
    s = lambda b: transition_score(a, b, EFF).total
    assert s(same_tempo_clash) > s(tempo_break_key_ok)       # tempo (5) over harmony (6)
    assert s(same_tempo_clash) > s(genre_change_key_ok)      # genre (4) over harmony (6)
    assert s(key_ok) > s(same_tempo_clash)                   # harmony still counts, just last
    # calibration: bounded by the automatic-estimate scale, monotonic, unknown -> the cap
    assert auto_key_confidence(0.99) == pytest.approx(AUTO_KEY_CONFIDENCE_MAX)
    assert 0.0 == auto_key_confidence(0.2) < auto_key_confidence(0.6) < auto_key_confidence(0.85) <= 0.5
    assert auto_key_confidence(None) == pytest.approx(AUTO_KEY_CONFIDENCE_MAX)
    # a curator-set key (or a curator-stated key_confidence) keeps its confidence
    manual = mk("m", camelot="8A")
    stated = _auto_track("st", camelot="8A", key_confidence=0.9)
    stated.feature_sources["key_confidence"] = "manual"
    for t, conf in ((manual, 1.0), (stated, 0.9)):
        ts = transition_score(t, t, EFF)
        assert ts.coverage == pytest.approx(EFF.weights["harmonic_compatibility"] * conf)


def _sample_dayparts(morning_max: int | None = 3, evening_opening: float | None = 0.60) -> ClientProfile:
    """Dayparts shaped like profiles/client.yaml (built here so the test does not depend on it)."""
    morning = DaypartProfile(name="morning", energy_range=(0.15, 0.60), energy_curve=(0.30, 0.50),
                             opening_tracks=3, opening_max_energy=0.40, preferred_tags=["melodic"],
                             max_tracks=morning_max)
    afternoon = DaypartProfile(name="afternoon", energy_range=(0.30, 0.70), energy_curve=(0.45, 0.55),
                               opening_tracks=3, opening_max_energy=None, preferred_genres=["pop"])
    evening = DaypartProfile(name="evening", energy_range=(0.40, 0.85), energy_curve=(0.50, 0.70),
                             opening_tracks=3, opening_max_energy=evening_opening, preferred_tags=["groove"])
    return ClientProfile(client="sample", dayparts={d.name: d for d in (morning, afternoon, evening)})


def test_edit10_limited_morning_still_gets_its_calm_openers():
    """EDIT-10: openers were reserved only from tracks left over after the global assignment, so
    a limited morning filled with 0.45 tracks and opened above its 0.40 maximum, claiming
    'no calmer alternative in the selection' while the calm tracks sat in the afternoon."""
    prof = _sample_dayparts()
    pool = [mk(f"mid{i}", artist=f"M{i}", energy=0.45, tags=["melodic"]) for i in range(4)]
    pool += [mk(f"calm{i}", artist=f"C{i}", energy=0.25) for i in range(3)]
    plans = plan_dayparts(pool, CFG, profile=prof, dayparts=["morning", "afternoon", "evening"], seed=0)
    alone = build_sequence(pool, CFG, profile=prof, daypart=prof.dayparts["morning"], seed=0)
    assert sorted(titles(plans["morning"])) == sorted(titles(alone)) == ["calm0", "calm1", "calm2"]
    placed = [t for r in plans.values() for t in titles(r)]
    assert sorted(placed) == sorted(t.title for t in pool)               # nothing lost, nothing reused
    for r in plans.values():
        for s in r.tracks:
            assert not any("no calmer alternative" in n for n in s.notes)


def test_edit11_track_displaced_by_an_opener_goes_to_a_daypart_with_room():
    """EDIT-11: a track dropped for an opener was never offered to other dayparts and was
    reported as 'no daypart had room left (limits reached here ...)' for an unlimited one."""
    prof = _sample_dayparts(evening_opening=None)
    pool = [mk(f"mid{i}", artist=f"M{i}", energy=0.45, tags=["melodic"]) for i in range(3)]
    pool.append(mk("calm_rock", artist="R", energy=0.18, genre="metal"))
    plans = plan_dayparts(pool, CFG, profile=prof, dayparts=["morning", "afternoon", "evening"], seed=0)
    placed = {t for r in plans.values() for t in titles(r)}
    assert placed == {t.title for t in pool}
    assert titles(plans["morning"])[0] == "calm_rock"
    assert plans["afternoon"].tracks                                   # no longer left empty
    reasons = [d["reason"] for r in plans.values() for d in r.not_selected]
    assert not any("limits reached" in r or "no daypart had room" in r for r in reasons), reasons

    # a genuinely full plan names the real limits of every daypart
    one = DaypartProfile(name="one", energy_range=(0.0, 1.0), opening_max_energy=None, max_tracks=1)
    two = DaypartProfile(name="two", energy_range=(0.0, 1.0), opening_max_energy=None, max_tracks=1)
    trio = [mk(f"x{i}", artist=f"X{i}", energy=0.5, mood="calm") for i in range(3)]
    plans = plan_dayparts(trio, CFG, profile=None, dayparts=[one, two], seed=0)
    left = [d["reason"] for d in plans["one"].not_selected if "scheduled" not in d["reason"]]
    assert len(left) == 1
    assert left[0].startswith("not selected for daypart 'one': max 1 tracks reached")
    assert "other dayparts: 'two' full (max 1 tracks reached)" in left[0]


def test_edit12_protected_and_unknown_energy_tracks_are_not_cut_as_poor_fits():
    """EDIT-12: MIN_SELECTION_SCORE turned a 'low' preference into an exclusion — even for an
    allowlisted track and for tracks whose only misfit was UNKNOWN energy (neutral fit)."""
    prof = _sample_dayparts(morning_max=None)
    approved = mk("approved_instrumental", artist="Example Artist", tags=["instrumental"])   # energy unknown
    known_misfit = mk("drone", energy=0.02, mood="dark", tags=["experimental"], artist="D")
    others = [mk(f"g{i}", energy=0.3 + 0.1 * i, tags=["melodic"], artist=f"G{i}") for i in range(4)]
    editorial = {approved.track_id: -0.6, known_misfit.track_id: -0.6}
    names = ["morning", "afternoon", "evening"]

    # unknown energy alone is not a known misfit: the low-preference track stays selectable
    plans = plan_dayparts(others + [approved], CFG, profile=prof, dayparts=names, editorial_scores=editorial, seed=0)
    assert "approved_instrumental" in {t for r in plans.values() for t in titles(r)}
    res = build_sequence(others + [approved], CFG, profile=prof, daypart=prof.dayparts["morning"],
                         editorial_scores=editorial, seed=0)
    assert "approved_instrumental" in titles(res)

    # a misfit on KNOWN data is still left out ... unless the curator protected the track
    plans = plan_dayparts(others + [known_misfit], CFG, profile=prof, dayparts=names,
                          editorial_scores=editorial, seed=0)
    assert "drone" not in {t for r in plans.values() for t in titles(r)}
    protected = frozenset({known_misfit.track_id})
    plans = plan_dayparts(others + [known_misfit], CFG, profile=prof, dayparts=names, editorial_scores=editorial,
                          seed=0, protected_track_ids=protected)
    assert "drone" in {t for r in plans.values() for t in titles(r)}
    res = build_sequence(others + [known_misfit], CFG, profile=prof, daypart=prof.dayparts["morning"],
                         editorial_scores=editorial, seed=0, protected_track_ids=protected)
    assert "drone" in titles(res)


def test_edit13_multi_valued_labels_and_spellings_match_curation():
    """EDIT-13: 'Pop/Rock' vs 'Pop' scored like Pop -> Metal, and 'feel-good' != 'feel good'."""
    assert genre_similarity("Pop/Rock", "Pop") == 1.0
    assert genre_similarity("Pop; Dance", "pop") == 1.0
    assert genre_similarity("Rock/Pop", "pop; rock") == 1.0
    assert genre_similarity("Hip-Hop", "hiphop") == 1.0
    assert genre_similarity("Pop/Rock", "Metal") == 0.0 == genre_similarity("Pop", "Metal")
    assert genre_similarity("Deep House; Ambient", "House") == pytest.approx(0.5)
    assert mood_similarity("Feel-Good", "feel good") == 1.0 == mood_similarity("FEEL_GOOD", "feel-good")
    assert mood_similarity("calm; warm", "Warm") == 1.0
    ts = transition_score(mk("a", genre="Pop/Rock"), mk("b", genre="Pop"), CFG)
    assert ts.components["genre_continuity"] == 1.0
    tags = transition_score(mk("a", tags=["feel-good"]), mk("b", tags=["Feel Good"]), CFG)
    assert tags.components["vibe_similarity"] == 1.0
    dp = DaypartProfile(name="day", energy_range=(0.0, 1.0), preferred_genres=["pop"], preferred_tags=["feel-good"],
                        avoid_tags=["Hip-Hop"])
    assert daypart_fit(mk("g", genre="Pop/Rock", energy=0.5), dp).components["preferences"] == 1.0
    assert daypart_fit(mk("t", tags=["feel good"], energy=0.5), dp).components["preferences"] == 1.0
    assert daypart_fit(mk("h", genre="Rap/Hip Hop", energy=0.5), dp).avoided_tag == "Hip-Hop"


def test_int6_artist_key_is_the_shared_primary_artist():
    """INT-6: one artist rule for curation, sequencing and QC (textnorm)."""
    for raw in ("Bravo/Zulu", "The Band (ft. Guest)", "A feat. B", "  the   BAND ", "AC/DC", "Simon & Garfunkel"):
        assert artist_key(raw) == primary_artist(raw), raw
    assert artist_key("Bravo feat. Zulu") == artist_key("bravo") == "bravo"


def test_similar_run_limit_works_for_untagged_auto_analysed_clones():
    """SPEC-8 / EDIT-5 (re-verification): auto-only energy + BPM (+ auto key) is enough to
    detect 'excessively similar' runs; 8 near-identical untagged tracks are not chained."""
    clones = [mk(f"same{i}", artist=f"S{i}", energy=0.5, bpm=120.0, camelot="8A", key_conf=0.6,
                 source="auto:test", energy_conf=0.4) for i in range(8)]
    others = [mk(f"diff{i}", artist=f"D{i}", energy=0.3 + 0.1 * i, bpm=90.0 + 9 * i, camelot=f"{i + 1}B",
                 key_conf=0.6, source="auto:test", energy_conf=0.4) for i in range(4)]
    ts = transition_score(clones[0], clones[1], CFG)
    assert ts.similarity is not None and ts.similarity >= CFG.similar_threshold
    res = build_sequence(clones + others, CFG, seed=0)
    names = titles(res)
    run = longest = 0
    for a, b in zip(names, names[1:]):
        run = run + 1 if a.startswith("same") and b.startswith("same") else 0
        longest = max(longest, run)
    assert longest <= CFG.max_similar_run


def test_proven_violation_under_the_calm_opening_says_so():
    """VEDIT-3: when the only calm openers include two tracks of one artist, the artist-gap
    violation is unavoidable only BECAUSE of the calm-opening reservation; the reason says so."""
    dp = DaypartProfile(name="morning", energy_range=(0.0, 1.0), energy_curve=(0.3, 0.6),
                        opening_tracks=3, opening_max_energy=0.4)
    pool = [mk("a1", artist="A", energy=0.25), mk("s1", artist="S", energy=0.30),
            mk("a2", artist="A", energy=0.35), mk("x1", artist="X", energy=0.55)]
    res = build_sequence(pool, CFG, daypart=dp, seed=0)
    reasons = [r for f in res.flagged for r in f.reasons if "same artist" in r]
    assert reasons and all("calm opening" in r or "could not avoid" in r for r in reasons), reasons
