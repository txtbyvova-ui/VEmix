"""VEmix 3.0 scenarios (spec §6): strict loading, the curve, track references.

Pure Python (no audio, no FFmpeg)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml

from veg_mixer.models import TrackFeatures
from veg_mixer.scenario import (
    DEFAULT_WEIGHTS,
    ScenarioError,
    load_scenario,
    match_reference,
    parse_scenario,
    resolve_scenario,
    song_key,
    with_overrides,
)

ROOT = Path(__file__).resolve().parents[1]
HALLOWEEN = ROOT / "scenarios" / "halloween_special.yaml"

BASE = {
    "scenario": "t",
    "phases": [
        {"name": "intro", "share": 1, "energy": [0.3, 0.5], "roles": ["opener"]},
        {"name": "peak", "share": 3, "energy": 0.9, "energy_range": [0.7, 1.0], "roles": ["peak"]},
    ],
}


def mk(artist: str, title: str, rel: str | None = None, **kw) -> TrackFeatures:
    rel = rel or f"{artist} - {title}.mp3"
    return TrackFeatures(path=Path("/lib") / rel, duration_s=kw.pop("duration_s", 200.0), rel_path=rel,
                         artist=artist, title=title, **kw)


def scenario(**extra):
    data = json.loads(json.dumps(BASE))
    data.update(extra)
    return parse_scenario(data, source="s.yaml")


def test_halloween_example_loads_cleanly(caplog):
    caplog.set_level(logging.INFO)
    sc = load_scenario(HALLOWEEN)
    assert sc.name == "halloween_special"
    assert [p.name for p in sc.phases] == ["intro", "build", "first_peak", "breather", "second_build", "climax",
                                          "descent", "closer"]
    assert abs(sum(p.share for p in sc.phases) - 1.0) < 1e-9
    assert sc.anchors[0].track == "Michael Jackson - Thriller" and sc.anchors[0].position == 1
    assert sc.phase("breather").contrast is True
    assert sc.target_duration_s == 3600 and sc.effective_tolerance_s == 240
    assert sc.weights["curve"] == 3.5 and sc.weights["transition"] == DEFAULT_WEIGHTS["transition"]
    assert sc.warnings == []          # every mood word of the example is in the vocabulary


def test_halloween_track_names_live_only_in_configuration():
    """Spec §6: the example's tracks are configuration, never part of the algorithm."""
    for name in ("dramaturgy.py", "scenario.py", "plan_pipeline.py", "plan_report.py", "moods.py",
                 "interpret_rules.py"):
        text = (ROOT / "veg_mixer" / name).read_text(encoding="utf-8").lower()
        assert "thriller" not in text and "michael jackson" not in text and "halloween" not in text


def test_phase_shares_are_normalized_and_curve_interpolates():
    sc = scenario()
    assert [round(p.share, 3) for p in sc.phases] == [0.25, 0.75]
    assert sc.target_energy(0.0) == pytest.approx(0.3)
    assert sc.target_energy(0.125) == pytest.approx(0.4)
    assert sc.target_energy(0.5) == pytest.approx(0.9)
    assert sc.phase_at(0.24)[1].name == "intro" and sc.phase_at(0.25)[1].name == "peak"
    assert sc.phase_at(1.0)[1].name == "peak"


def test_explicit_curve_replaces_phase_energies():
    sc = parse_scenario({"scenario": "c", "curve": [[0, 0.2], [0.5, 0.8], [1, 0.4]]})
    assert [p.name for p in sc.phases] == ["main"]
    assert sc.target_energy(0.25) == pytest.approx(0.5)
    assert sc.target_energy(0.75) == pytest.approx(0.6)
    with pytest.raises(ScenarioError, match="OR a curve"):
        parse_scenario({"scenario": "c", "curve": [[0, 0.2], [1, 0.4]],
                        "phases": [{"name": "a", "share": 1, "energy": 0.5}]})


@pytest.mark.parametrize("data, fragment", [
    ({"scenari": "x"}, "did you mean 'scenario'"),
    ({"scenario": "x"}, "phases: required"),
    ({**BASE, "phases": [{"name": "a", "share": 1}]}, "phases.a.energy: required"),
    ({**BASE, "phases": [{"name": "a", "energy": 0.5}]}, "phases.a.share: required"),
    ({**BASE, "phases": [{"name": "a", "share": 1, "energy": 0.5, "roles": ["peek"]}]}, "did you mean 'peak'"),
    ({**BASE, "phases": [{"name": "a", "share": 1, "energy": 0.5}, {"name": "A", "share": 1, "energy": 0.5}]},
     "duplicate phase name"),
    ({**BASE, "phases": [{"name": "a", "share": 1, "energy": 0.9, "energy_range": [0.1, 0.5]}]},
     "lies outside the range"),
    ({**BASE, "anchors": [{"track": "A - B"}]}, "exactly one of 'position'"),
    ({**BASE, "anchors": [{"track": "A - B", "position": 0}]}, "non-zero whole number"),
    ({**BASE, "anchors": [{"track": "A - B", "position": 1}, {"track": "C - D", "position": 1}]}, "already anchored"),
    ({**BASE, "anchors": [{"track": "A - B", "phase": "climax"}]}, "unknown phase"),
    ({**BASE, "anchors": [{"track": "A - B", "position": 1}, {"track": "a - b", "position": 2}]},
     "already anchored"),
    ({**BASE, "required": ["A - B"], "excluded": ["a - b"]}, "both required and excluded"),
    ({**BASE, "transitions": {"max_energy_jump": 0.4, "contrast_energy_jump": 0.3}}, "LARGER jump"),
    ({**BASE, "use_all_tracks": True, "target_duration_min": 30}, "places every eligible track"),
    ({**BASE, "alternatives": 9}, "out of range"),
    ({**BASE, "adjacency": {"forbid": [["A - B", "a - b"]]}}, "two different tracks"),
    ({**BASE, "weights": {"curv": 2}}, "did you mean 'curve'"),
    ({**BASE, "use_all_tracks": "yes"}, "expected true or false"),
])
def test_invalid_scenarios_name_the_key(data, fragment):
    with pytest.raises(ScenarioError) as ei:
        parse_scenario(data, source="s.yaml")
    assert fragment in str(ei.value)
    assert str(ei.value).startswith("s.yaml: ")


def test_loader_rejects_duplicate_keys_and_bad_formats(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("scenario: a\nscenario: b\n", encoding="utf-8")
    with pytest.raises(ScenarioError, match="duplicate key"):
        load_scenario(p)
    q = tmp_path / "s.txt"
    q.write_text("x", encoding="utf-8")
    with pytest.raises(ScenarioError, match="unsupported scenario format"):
        load_scenario(q)
    j = tmp_path / "s.json"
    j.write_text(json.dumps(BASE), encoding="utf-8")
    assert load_scenario(j).name == "t"
    j.write_text('{"scenario": "a", "scenario": "b"}', encoding="utf-8")
    with pytest.raises(ScenarioError, match="duplicate key 'scenario'"):
        load_scenario(j)


def test_unknown_moods_are_reported_not_rejected():
    sc = scenario(moods={"prefer": ["ominous", "glittery"]})
    assert sc.prefer_moods == ["ominous", "glittery"]
    assert any("glittery" in w for w in sc.warnings)


def test_match_reference_rules():
    lib = [mk("Michael Jackson", "Thriller"), mk("Earth, Wind & Fire", "September"),
           mk("Jay-Z", "99 Problems"), mk("Other", "Thriller (2008 Remaster)", rel="sub/x.mp3")]
    found, how = match_reference("Michael Jackson - Thriller", lib)
    assert [t.title for t in found] == ["Thriller"] and how == "exact"
    assert match_reference("michael jackson – thriller", lib)[0][0].title == "Thriller"
    assert match_reference("sub/x.mp3", lib)[0][0].artist == "Other"
    assert match_reference("Jay-Z - 99 Problems", lib)[0][0].artist == "Jay-Z"
    # a hard anchor never matches only one credited part of a group name
    assert match_reference("Fire - September", lib) == ([], "none")
    # version qualifiers are ignored only when nothing matches exactly, and the artist must match
    lib2 = [mk("Michael Jackson", "Thriller (2008 Remaster)")]
    found, how = match_reference("Michael Jackson - Thriller", lib2)
    assert how == "relaxed" and len(found) == 1
    assert match_reference("Someone Else - Thriller", lib2) == ([], "none")


def test_song_key_groups_versions():
    a, b = mk("Michael Jackson", "Thriller"), mk("Michael Jackson feat. X", "Thriller - 2008 Remaster")
    c = mk("Michael Jackson", "Beat It")
    assert song_key(a) == song_key(b) != song_key(c)
    assert song_key(mk("A", "")) is None


def test_resolve_errors_and_pool_filters():
    lib = [mk("Michael Jackson", "Thriller"), mk("A", "One", vibe_tags=["spooky"]), mk("B", "Two", genre="pop"),
           mk("C", "Three", mood="ominous")]
    sc = scenario(anchors=[{"track": "Michael Jackson - Thriller", "position": 1}],
                  pool={"include_tags": ["spooky", "ominous"]}, excluded=["Nobody - Nothing"])
    r = resolve_scenario(sc, lib)
    assert {t.title for t in r.pool} == {"Thriller", "One", "Three"}       # the anchor bypasses the pool filter
    assert any("Nobody - Nothing" in n for n in r.notes)
    # missing anchor: an error with a suggestion
    with pytest.raises(ScenarioError, match="matches no eligible track.*closest"):
        resolve_scenario(scenario(anchors=[{"track": "Michael Jackson - Thrillr", "position": 1}]), lib)
    # anchor excluded by the client profile: the curation reason is named
    with pytest.raises(ScenarioError, match="excluded by the client profile: blocklisted"):
        resolve_scenario(scenario(anchors=[{"track": "Michael Jackson - Thriller", "position": 1}]), lib[1:],
                         curation_excluded={lib[0].track_id: ["blocklisted track"]}, library=lib)
    # ambiguous anchor
    dup = lib + [mk("Michael Jackson", "Thriller", rel="other/Michael Jackson - Thriller.mp3")]
    with pytest.raises(ScenarioError, match="ambiguous"):
        resolve_scenario(scenario(anchors=[{"track": "Michael Jackson - Thriller", "position": 1}]), dup)


def test_pairs_resolve_to_track_ids():
    lib = [mk("A", "One"), mk("B", "Two"), mk("C", "Three")]
    sc = scenario(adjacency={"forbid": [["A - One", "B - Two"]], "prefer": [["B - Two", "C - Three"]],
                             "avoid": [["A - One", "Z - None"]]})
    r = resolve_scenario(sc, lib)
    assert r.forbid_pairs == {frozenset((lib[0].track_id, lib[1].track_id))}
    assert r.prefer_pairs == {(lib[1].track_id, lib[2].track_id)}
    assert r.avoid_pairs == set() and any("Z - None" in n for n in r.notes)


def test_example_scenario_is_valid_yaml_for_plain_loaders():
    """The example stays readable by any YAML tool (no custom tags)."""
    assert yaml.safe_load(HALLOWEEN.read_text(encoding="utf-8"))["scenario"] == "halloween_special"


def test_song_key_keeps_parts_and_reprises_apart():
    """Only VERSION qualifiers are dropped: '(Part 1)' and '(Part 2)' are different pieces."""
    k = lambda title: song_key(mk("Band", title))          # noqa: E731
    assert k("Odyssey (Part 1)") != k("Odyssey (Part 2)")
    assert k("Theme") != k("Theme (Reprise)") and k("Alive") != k("Alive (Interlude)")
    assert k("Theme") == k("Theme (2011 Remaster)") == k("Theme [Radio Edit]") == k("Theme (feat. Guest)") \
        == k("Theme (Live at Wembley)") == k("Theme - 2011 Remastered") == k("Theme (Acoustic Version)")


def test_excluding_a_missing_version_never_excludes_the_original():
    lib = [mk("Michael Jackson", "Billie Jean"), mk("Queen", "Bohemian Rhapsody")]
    sc = scenario(excluded=["Michael Jackson - Billie Jean (Live)"])
    r = resolve_scenario(sc, lib)
    assert lib[0].track_id not in r.excluded
    assert any("matches no track" in n for n in r.notes)
    # an anchor plus a relaxed-only exclusion is no conflict any more
    sc = scenario(excluded=["Michael Jackson - Billie Jean (Live)"],
                  anchors=[{"track": "Michael Jackson - Billie Jean", "position": 1}])
    assert resolve_scenario(sc, lib).anchors[0][1] == lib[0].track_id
    # a pool.tracks reference may match ignoring the version, and says so
    r = resolve_scenario(scenario(pool={"tracks": ["Queen - Bohemian Rhapsody (2011 Remaster)"]}), lib)
    assert lib[1].track_id not in r.excluded and lib[0].track_id in r.excluded
    assert any("ignoring version qualifiers" in n for n in r.notes)


def test_conflict_message_names_the_track():
    """Anchored by name, excluded by file path: the clash surfaces at resolve time and the
    message names the track, not its hash id."""
    lib = [mk("A", "One"), mk("B", "Two")]
    sc = scenario(anchors=[{"track": "A - One", "position": 1}], excluded=["A - One.mp3"])
    with pytest.raises(ScenarioError, match="track 'A - One' is both anchored/required and excluded"):
        resolve_scenario(sc, lib)


def test_cli_overrides_are_validated():
    sc = scenario(anchors=[{"track": "A - B", "position": 5}], required=["C - D", "E - F"])
    with pytest.raises(ScenarioError, match="--max-tracks.*beyond"):
        with_overrides(sc, max_tracks=3)
    with pytest.raises(ScenarioError, match="--max-tracks.*anchored / required"):
        with_overrides(scenario(required=["A - B", "C - D", "E - F"]), max_tracks=2)
    with pytest.raises(ScenarioError, match="min_tracks"):
        with_overrides(scenario(min_tracks=5), max_tracks=3)
    with pytest.raises(ScenarioError, match="1440"):
        with_overrides(sc, target_duration_s=25 * 3600)
    out = with_overrides(sc, max_tracks=6, target_duration_s=1800)
    assert out.max_tracks == 6 and out.target_duration_s == 1800 and sc.max_tracks is None


def test_ai_positions_vocabulary_matches_the_ai_schema():
    from veg_mixer.ai.schema import PHASES
    from veg_mixer.scenario import AI_POSITIONS
    assert tuple(PHASES) == AI_POSITIONS


def test_phase_description_is_accepted():
    sc = parse_scenario({**BASE, "phases": [{"name": "a", "share": 1, "energy": 0.5, "description": "calm"}]})
    assert sc.phases[0].description == "calm"
