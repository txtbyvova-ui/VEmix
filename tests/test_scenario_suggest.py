"""VEmix 3.0 AI scenario drafts (spec §5: AI may propose the structure; the plan stays
deterministic). Fake provider only — no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from veg_mixer.ai.providers import AIResult, FakeProvider
from veg_mixer.ai.schema import schema_subset_problems
from veg_mixer.models import TrackFeatures
from veg_mixer.scenario import ScenarioError, load_scenario
from veg_mixer.scenario_suggest import (
    SUGGEST_PROMPT_VERSION,
    SUGGESTION_SCHEMA,
    SYSTEM_PROMPT,
    ascii_identifier,
    draft_to_scenario,
    library_summary,
    suggest_scenario,
    write_draft,
)

GOOD = {
    "scenario": "Spooky Night!", "description": "a spooky evening", "target_duration_min": 45,
    "phases": [
        {"name": "Intro", "share": 1, "energy_start": 0.5, "energy_end": 0.55, "roles": ["opener"],
         "prefer_moods": ["playful"], "avoid_moods": [], "contrast": False, "purpose": "hook"},
        {"name": "build", "share": 2, "energy_start": 0.45, "energy_end": 0.8, "roles": ["tension_builder"],
         "prefer_moods": ["tense"], "avoid_moods": ["romantic"], "contrast": False, "purpose": "rise"},
        {"name": "hush", "share": 1, "energy_start": 0.3, "energy_end": 1.7, "roles": ["breather", "nonsense"],
         "prefer_moods": ["ominous"], "avoid_moods": [], "contrast": True, "purpose": "contrast"},
    ],
    "moods": {"prefer": ["spooky"], "avoid": []},
    "transitions": {"max_bpm_jump": 14, "max_energy_jump": 0.3, "contrast_energy_jump": 0.1},
    "diversity": {"artist_min_gap": 3, "max_tracks_per_artist": 2, "avoid_same_song": True},
    "rationale": "Hook first, build, then a hush.",
}


def track(i: int, energy: float | None, mood: str | None = None) -> TrackFeatures:
    return TrackFeatures(path=Path(f"/lib/Secret Artist {i} - Private Title {i}.mp3"), duration_s=200.0,
                         rel_path=f"Secret Artist {i} - Private Title {i}.mp3", artist=f"Secret Artist {i}",
                         title=f"Private Title {i}", energy=energy, bpm=100 + i, mood=mood)


def test_schema_is_in_the_structured_output_subset():
    assert schema_subset_problems(SUGGESTION_SCHEMA) == []


def test_library_summary_is_anonymous():
    tracks = [track(i, 0.1 * i, "ominous" if i % 2 else None) for i in range(1, 8)] + [track(9, None)]
    s = library_summary(tracks)
    text = json.dumps(s)
    assert "Secret" not in text and "Private" not in text and "/lib" not in text
    assert s["tracks"] == 8 and s["energy_known_share"] == 0.875 and s["moods"]["ominous"] == 4


def test_draft_is_clipped_and_validated(tmp_path):
    provider = FakeProvider(GOOD)
    draft, rationale, problems = suggest_scenario(provider, "a spooky evening", {"tracks": 10}, target_duration_min=50)
    assert problems == [] and rationale.startswith("Hook first")
    assert draft["scenario"] == "Spooky_Night_" and draft["target_duration_min"] == 50
    assert [p["name"] for p in draft["phases"]] == ["intro", "build", "hush"]
    assert draft["phases"][2]["energy"] == [0.3, 1.0] and draft["phases"][2]["roles"] == ["breather"]
    assert draft["transitions"]["contrast_energy_jump"] >= draft["transitions"]["max_energy_jump"]
    assert "anchors" not in draft and "required" not in draft          # tracks are the curator's call
    sent = provider.calls[0]["user"]
    assert "a spooky evening" in sent
    out = tmp_path / "draft.yaml"
    write_draft(out, draft, rationale=rationale, provider_label="fake/fake-1", brief="a spooky evening")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# VEmix 3.0") and "ЧЕРНОВИК" in text
    sc = load_scenario(out)                                             # the normal loader accepts it
    assert sc.name == "Spooky_Night_" and len(sc.phases) == 3
    assert yaml.safe_load(text)["phases"][0]["roles"] == ["opener"]


def test_invalid_answer_is_reasked_then_rejected():
    bad = dict(GOOD, phases=[])                                          # no phases: the loader refuses it
    provider = FakeProvider([bad, GOOD])
    draft, _, problems = suggest_scenario(provider, "brief", {})
    assert len(provider.calls) == 2 and problems and "rejected" in provider.calls[1]["user"]
    assert draft["phases"]
    with pytest.raises(ScenarioError, match="no valid scenario draft"):
        suggest_scenario(FakeProvider([bad, bad]), "brief", {})
    with pytest.raises(ScenarioError, match="brief is empty"):
        suggest_scenario(FakeProvider(GOOD), "  ", {})


def test_draft_to_scenario_never_invents_track_rules():
    d = draft_to_scenario({**GOOD, "anchors": [{"track": "Somebody - Something", "position": 1}]})
    assert "anchors" not in d


# ---------------------------------------------------------------------------
# Review S1: non-ASCII phase names, malformed answers
# ---------------------------------------------------------------------------

def phase(name, **kw) -> dict:
    base = {"name": name, "share": 1, "energy_start": 0.4, "energy_end": 0.6, "roles": [], "prefer_moods": [],
            "avoid_moods": [], "contrast": False, "purpose": "x"}
    return {**base, **kw}


def test_prompt_asks_for_ascii_snake_case_phase_names():
    assert SUGGEST_PROMPT_VERSION == "suggest-v2"
    assert "ASCII snake_case" in SYSTEM_PROMPT and "first_peak" in SYSTEM_PROMPT
    name_schema = SUGGESTION_SCHEMA["properties"]["phases"]["items"]["properties"]["name"]
    assert name_schema["type"] == "string" and "snake_case" in name_schema["description"]


@pytest.mark.parametrize("text, expected", [
    ("Вступление", "vstuplenie"), ("Кульминация", "kulminatsiya"), ("Первый пик!", "pervyy_pik"),
    ("Щедрый ёж", "shchedryy_ezh"), ("Їжак", "yizhak"), ("First-Peak", "first_peak"), ("Café  Noir", "cafe_noir"),
    ("__a__", "a"), ("序章", ""), ("!!!", ""),
])
def test_ascii_identifier(text, expected):
    assert ascii_identifier(text) == expected
    assert ascii_identifier(text) == ascii_identifier(text)                 # deterministic


def test_ascii_identifier_length():
    assert len(ascii_identifier("Длинное " * 20)) <= 40


def test_russian_brief_with_cyrillic_phase_names_is_accepted(tmp_path, caplog):
    answer = {**GOOD, "scenario": "Хэллоуин", "phases": [
        phase("Вступление", roles=["opener"], prefer_moods=["зловещий"]),
        phase("Разгон", energy_start=0.5, energy_end=0.8),
        phase("Первый пик!", energy_start=0.8, energy_end=0.9),
        phase("Пик", energy_start=0.9, energy_end=0.9),
        phase("Пик", energy_start=0.9, energy_end=0.7),
        phase("序章", energy_start=0.6, energy_end=0.4),
    ]}
    provider = FakeProvider(answer)
    repairs: list[str] = []
    draft, _, problems = suggest_scenario(provider, "жуткий вечер на хэллоуин", {"tracks": 10}, warnings=repairs)
    assert len(provider.calls) == 1 and problems == []                  # accepted at once, no re-ask
    assert [p["name"] for p in draft["phases"]] == ["vstuplenie", "razgon", "pervyy_pik", "pik", "pik_2",
                                                    "phase_6"]
    text = "\n".join(repairs)
    assert "'Вступление' written as 'vstuplenie'" in text and "duplicate name 'pik' renamed 'pik_2'" in text
    assert "no ASCII form; named phase_6" in text
    assert any("the AI draft was repaired" in r.message for r in caplog.records)
    out = tmp_path / "draft.yaml"
    write_draft(out, draft, rationale="", provider_label="fake/fake-1", brief="жуткий вечер")
    sc = load_scenario(out)                                             # the normal loader accepts it
    assert [p.name for p in sc.phases][:2] == ["vstuplenie", "razgon"] and sc.name == "Хэллоуин"


def test_phase_names_stay_unique_when_a_fallback_collides():
    raw = {**GOOD, "phases": [phase("phase_2"), phase("序章"), phase("Phase 2"), phase(None), phase(["x"])]}
    notes: list[str] = []
    d = draft_to_scenario(raw, warnings=notes)
    assert [p["name"] for p in d["phases"]] == ["phase_2", "phase_2_2", "phase_2_3", "phase_4", "phase_5"]
    assert any("phases[3].name: missing; named phase_4" in n for n in notes)
    assert any("phases[4].name: expected a string, got a list" in n for n in notes)


@pytest.mark.parametrize("phases", [
    {"intro": phase("intro"), "build": phase("build")},       # an object instead of a list (was KeyError)
    ["intro", "build", "climax"],                             # strings instead of objects (was AttributeError)
    "intro, build, climax",
    3,
    None,
])
def test_malformed_phases_raise_scenario_error_after_one_reask(phases):
    bad = {**GOOD, "phases": phases}
    provider = FakeProvider([bad, bad])
    with pytest.raises(ScenarioError, match="no valid scenario draft") as info:
        suggest_scenario(provider, "brief", {})
    assert len(provider.calls) == 2 and "rejected" in provider.calls[1]["user"]
    message = str(info.value)
    assert "phases" in message
    if phases is not None:
        assert "does not match the schema" in message and "answer.phases" in message
        assert "does not match the schema" in provider.calls[1]["user"]       # the re-ask names the cause


def test_wrong_typed_parts_are_ignored_or_coerced():
    answer = {
        **GOOD,
        "phases": [phase("intro", roles="opener", prefer_moods="playful"),
                   phase("build", avoid_moods=[1, "romantic", None], contrast="yes", roles=["Tension Builder"]),
                   phase("hush", share="2", energy_start="high", energy_end=float("nan")),
                   "not a phase"],
        "moods": ["spooky"], "transitions": [0.3], "diversity": "lots", "description": {"x": 1},
        "target_duration_min": "soon", "rationale": ["not", "text"],
    }
    repairs: list[str] = []
    draft, rationale, problems = suggest_scenario(FakeProvider(answer), "brief", {}, warnings=repairs)
    assert problems == [] and rationale == ""
    p0, p1, p2 = draft["phases"]
    assert p0["roles"] == ["opener"] and p0["moods"]["prefer"] == ["playful"]      # one string -> one-item list
    assert p1["moods"]["avoid"] == ["romantic"] and p1["contrast"] is False and p1["roles"] == ["tension_builder"]
    assert p2["share"] == 2.0 and p2["energy"] == [0.5, 0.5]
    assert draft["moods"] == {"prefer": [], "avoid": []} and "description" not in draft
    assert draft["transitions"] == {"max_bpm_jump": 16.0, "max_energy_jump": 0.3, "contrast_energy_jump": 0.5}
    assert draft["diversity"] == {"artist_min_gap": 2, "max_tracks_per_artist": 3, "avoid_same_song": True}
    assert draft["target_duration_min"] == 60.0
    text = "\n".join(repairs)
    for needle in ("phases[0].roles: a single string instead of a list", "phases[1].avoid_moods[0]: expected a string",
                   "phases[1].contrast: expected true or false", "phases[2].energy_start: 'high' is not a number",
                   "phases[3]: expected an object, got a string; ignored", "moods: expected an object, got a list",
                   "transitions: expected an object", "diversity: expected an object", "phases[2].share: the string"):
        assert needle in text, needle


def test_draft_to_scenario_refuses_a_non_object():
    for data in ([1, 2], "text", None, 3):
        with pytest.raises(ScenarioError, match="expected a JSON object"):
            draft_to_scenario(data)                                     # type: ignore[arg-type]


def test_unparsable_or_non_object_answers_get_one_reask():
    provider = FakeProvider(["this is not json", GOOD])
    draft, _, problems = suggest_scenario(provider, "brief", {})
    assert len(provider.calls) == 2 and draft["phases"]
    assert len(problems) == 1 and "invalid_output" in problems[0]
    provider = FakeProvider(["[1, 2]", GOOD])                         # a JSON array, not an object
    draft, _, problems = suggest_scenario(provider, "brief", {})
    assert len(provider.calls) == 2 and problems == ["the answer is a list, not a JSON object"]
    refusal = AIResult(None, "", "fake", "fake-1", error="refused: no", error_kind="refused")
    provider = FakeProvider([refusal, GOOD])
    with pytest.raises(ScenarioError, match="refused"):
        suggest_scenario(provider, "brief", {})
    assert len(provider.calls) == 1                                     # a refusal is not re-asked


def test_library_summary_sends_exactly_the_documented_fields():
    import veg_mixer.scenario_suggest as mod
    s = library_summary([track(1, 0.4, "tense")])
    assert set(s) == {"tracks", "total_duration_min", "energy", "energy_known_share", "energy_note", "bpm", "moods"}
    assert "role counts" not in (mod.__doc__ or "")


@pytest.mark.parametrize("where", ["target_duration_min", "share"])
def test_huge_integers_never_crash_suggest(where):
    huge = 10 ** 400                       # valid JSON, but no float can hold it
    bad = {**GOOD, "target_duration_min": huge} if where == "target_duration_min" else \
        {**GOOD, "phases": [{**GOOD["phases"][0], "share": huge}, *GOOD["phases"][1:]]}
    provider = FakeProvider([bad, bad])
    try:
        draft, _, _ = suggest_scenario(provider, "brief", {})
    except ScenarioError:
        return                                                   # refused cleanly
    assert isinstance(draft, dict)                               # or repaired (value ignored)


def test_a_provider_that_never_answers_is_a_runtime_failure_not_an_input_error():
    """Network / key / quota failures: exit 5 like interpret; a refusal is an answer (exit 2)."""
    from veg_mixer.scenario_suggest import SuggestProviderError
    down = AIResult(None, "", "fake", "fake-1", error="connection: refused by host", error_kind="connection",
                    retryable=True)
    provider = FakeProvider([down, down])
    with pytest.raises(SuggestProviderError, match="gave no answer"):
        suggest_scenario(provider, "brief", {})
    assert len(provider.calls) == 2
    auth = AIResult(None, "", "fake", "fake-1", error="auth: bad key", error_kind="auth")
    with pytest.raises(SuggestProviderError):
        suggest_scenario(FakeProvider([auth]), "brief", {})
    refusal = AIResult(None, "", "fake", "fake-1", error="refused: no", error_kind="refused")
    with pytest.raises(ScenarioError):
        suggest_scenario(FakeProvider([refusal]), "brief", {})
