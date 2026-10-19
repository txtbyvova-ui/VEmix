"""Spec §6 client curation rules + §7 daypart definitions (profile loading and curate())."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import pytest

from veg_mixer.curation import (
    CYRILLIC_EXCLUDE_REASON,
    CYRILLIC_REVIEW_REASON,
    DEFAULT_DAYPARTS,
    FAMILIARITY_TOKEN,
    NOT_APPROVED_REVIEW_REASON,
    UNKNOWN_LANGUAGE_REVIEW_REASON,
    ProfileError,
    curate,
    daypart_name_problem,
    default_profile,
    editorial_scores,
    get_daypart,
    load_profile,
    normalize_language,
    parse_profile,
    preference_match_tags,
    track_tags,
)
from veg_mixer.metadata import apply_metadata, load_metadata
from veg_mixer.models import (
    ClientProfile,
    DaypartProfile,
    LanguageRules,
    LevelSetting,
    ListRules,
    PreferenceLevel,
    TrackFeatures,
)

REPO = Path(__file__).resolve().parents[1]
CLIENT_YAML = REPO / "profiles" / "client.yaml"

SPEC_EXAMPLE = """\
client: commercial_pop

language:
  allowed:
    - en
    - fr
    - es
    - it
  excluded:
    - ru

preferences:
  recognizable_pop: high
  accessible_melodic_music: high
  experimental_instrumental: low
  aggressive_rap: low

vibe_consistency: high
energy_jumps: low

manual_review_required: true
"""


def T(rel: str, **kw) -> TrackFeatures:
    return TrackFeatures(path=Path("C:/lib") / rel, duration_s=kw.pop("duration_s", 200.0), rel_path=rel, **kw)


def profile(**overrides) -> ClientProfile:
    """Strict-ish base profile built in code (en/fr allowed, ru excluded)."""
    base = dict(
        client="test",
        language=LanguageRules(allowed=["en", "fr"], excluded=["ru"], unknown_policy="review",
                               cyrillic_script_policy="review"),
    )
    base.update(overrides)
    return ClientProfile(**base)


def decision_for(result, rel):
    return next(d for d in result.decisions if d.track.rel_path == rel)


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

def test_sample_client_profile_loads(caplog):
    with caplog.at_level(logging.WARNING, logger="veg_mixer.curation"):
        p = load_profile(CLIENT_YAML)
    assert not caplog.records, [r.getMessage() for r in caplog.records]      # internally consistent
    assert p.client == "commercial_pop" and p.source_path == CLIENT_YAML
    assert p.language.allowed == ["en", "fr", "es", "it"] and p.language.excluded == ["ru"]
    assert p.language.unknown_policy == "review" and p.language.cyrillic_script_policy == "exclude"
    assert p.preferences == {"recognizable_pop": PreferenceLevel.HIGH,
                             "accessible_melodic_music": PreferenceLevel.HIGH,
                             "experimental_instrumental": PreferenceLevel.LOW,
                             "aggressive_rap": PreferenceLevel.LOW}
    assert FAMILIARITY_TOKEN in p.preference_tags["recognizable_pop"]
    assert p.vibe_consistency is LevelSetting.HIGH and p.energy_jumps is LevelSetting.LOW
    assert p.manual_review_required is True and p.require_approved is False
    assert p.blocklist == ListRules() and p.allowlist == ListRules()
    assert set(p.dayparts) == {"morning", "afternoon", "evening"}
    m = p.dayparts["morning"]
    assert m.energy_range == (0.15, 0.60) and m.energy_curve == (0.30, 0.50) and m.opening_max_energy == 0.40
    assert "aggressive" in m.avoid_tags
    assert p.dayparts["evening"].energy_range == (0.40, 0.85)


def test_spec_example_profile_loads_with_safe_defaults(tmp_path):
    p = load_profile(write(tmp_path, "spec.yml", SPEC_EXAMPLE))
    assert p.language.unknown_policy == "review"               # unknown language -> review by default
    assert p.language.cyrillic_script_policy == "review"
    assert p.preference_tags == {}
    effective = preference_match_tags(p)
    assert effective["recognizable_pop"] == ["recognizable_pop", FAMILIARITY_TOKEN]   # curated familiarity
    assert effective["aggressive_rap"] == ["aggressive_rap"]
    assert set(p.dayparts) == set(DEFAULT_DAYPARTS)


def test_json_profile_loads(tmp_path):
    data = {"client": "café", "language": {"allowed": ["EN"], "unknown_policy": "exclude"},
            "preferences": {"groove": "high"}, "preference_tags": {"groove": ["funk", "disco"]},
            "dayparts": {"Evening": {"bpm_range": [100, 125]}}}
    p = load_profile(write(tmp_path, "p.json", json.dumps(data, ensure_ascii=False)))
    assert p.client == "café" and p.language.allowed == ["en"] and p.language.unknown_policy == "exclude"
    assert p.dayparts["evening"].bpm_range == (100.0, 125.0)
    assert p.dayparts["evening"].energy_range == DEFAULT_DAYPARTS["evening"].energy_range


@pytest.mark.parametrize("yaml_text, key, fragment", [
    ("client: x\nvibe_consistancy: high\n", "vibe_consistancy", "did you mean 'vibe_consistency'"),
    ("client: x\nlanguage:\n  unkown_policy: review\n", "language.unkown_policy", "did you mean 'unknown_policy'"),
    ("client: x\ndayparts:\n  morning:\n    energy_rang: [0.1, 0.5]\n", "dayparts.morning.energy_rang",
     "did you mean 'energy_range'"),
    ("client: x\nblocklist:\n  artist: [A]\n", "blocklist.artist", "did you mean 'artists'"),
    ("client: x\npreferences:\n  pop: high\npreference_tags:\n  popp: [pop]\n", "preference_tags.popp",
     "did you mean 'pop'"),
])
def test_typo_keys_raise_profile_error(tmp_path, yaml_text, key, fragment):
    with pytest.raises(ProfileError, match=fragment) as exc:
        load_profile(write(tmp_path, "typo.yaml", yaml_text))
    assert key in str(exc.value) and "typo.yaml" in str(exc.value)


@pytest.mark.parametrize("yaml_text, key, fragment", [
    ("client: x\npreferences:\n  pop: hgih\n", "preferences.pop", "not one of: avoid, low, medium, high"),
    ("client: x\nvibe_consistency: extreme\n", "vibe_consistency", "not one of"),
    ("client: x\nenergy_jumps: no\n", "energy_jumps", "boolean"),
    ("client: x\nlanguage:\n  unknown_policy: maybe\n", "language.unknown_policy", "not one of"),
    ("client: x\nlanguage:\n  cyrillic_script_policy: drop\n", "language.cyrillic_script_policy", "not one of"),
    ("client: x\nlanguage:\n  allowed: [eng]\n", "language.allowed[0]", "ISO 639-1"),
    ("client: x\nlanguage:\n  allowed: [en, no]\n", "language.allowed[1]", "quote the code"),
    ("client: x\nlanguage:\n  allowed: [en]\n  excluded: [en]\n", "language", "both allowed and excluded"),
    ("client: x\nmanual_review_required: 'yes'\n", "manual_review_required", "true or false"),
    ("client: x\nallowlist:\n  only: true\n", "allowlist.only", "allowlist is empty"),
    ("client: x\nblocklist:\n  only: true\n", "blocklist.only", "allowlist only"),
    ("client: x\ndayparts:\n  morning:\n    energy_range: [0.8, 0.2]\n", "dayparts.morning.energy_range",
     "greater than upper"),
    ("client: x\ndayparts:\n  morning:\n    energy_range: [0.1, 1.5]\n", "dayparts.morning.energy_range[1]",
     "out of range"),
    ("client: x\ndayparts:\n  morning:\n    opening_tracks: 2.5\n", "dayparts.morning.opening_tracks",
     "whole number"),
    ("client: x\ndayparts:\n  night:\n    max_tracks: 0\n", "dayparts.night.max_tracks", "out of range"),
    ("client: x\ndayparts:\n  night:\n    preferred_tags: {a: b}\n", "dayparts.night.preferred_tags", "list"),
    ("language:\n  allowed: [en]\n", "client", "required"),
    ("- client: x\n", "<top level>", "expected a mapping"),
    ("client: x\nlanguage: [en]\n", "language", "expected a mapping"),
])
def test_bad_values_raise_profile_error(tmp_path, yaml_text, key, fragment):
    with pytest.raises(ProfileError, match=fragment) as exc:
        load_profile(write(tmp_path, "bad.yaml", yaml_text))
    assert f": {key}:" in str(exc.value)


def test_invalid_yaml_and_formats(tmp_path):
    with pytest.raises(ProfileError, match="invalid YAML at line 3: .*cannot start any token"):
        load_profile(write(tmp_path, "broken.yaml", "client: x\nlanguage:\n\tallowed: [en]\n"))
    with pytest.raises(ProfileError, match="invalid YAML"):
        load_profile(write(tmp_path, "unclosed.yaml", "client: x\nlanguage:\n  allowed: [en\n"))
    with pytest.raises(ProfileError, match="unsupported profile format"):
        load_profile(write(tmp_path, "p.toml", "client = 'x'"))
    with pytest.raises(ProfileError, match="empty"):
        load_profile(write(tmp_path, "empty.yaml", "# nothing\n"))
    with pytest.raises(ProfileError, match="cannot read"):
        load_profile(tmp_path / "missing.yaml")


def test_duplicate_keys_are_rejected_not_silently_overwritten(tmp_path):
    text = ("client: x\n"
            "preferences:\n  pop: high\n"
            "dayparts:\n  morning:\n    energy_range: [0.1, 0.4]\n"
            "dayparts:\n  evening:\n    energy_range: [0.5, 0.9]\n")
    with pytest.raises(ProfileError, match=r"line 7: .*duplicate key 'dayparts' \(first defined at line 4\)"):
        load_profile(write(tmp_path, "dup.yaml", text))
    with pytest.raises(ProfileError, match="duplicate key 'client'"):
        load_profile(write(tmp_path, "dup.json", '{"client": "a", "client": "b"}'))


def test_yaml_python_tags_are_refused(tmp_path):
    """The strict loader is a SafeLoader: no object construction from profile files."""
    with pytest.raises(ProfileError, match="invalid YAML"):
        load_profile(write(tmp_path, "evil.yaml", "client: !!python/object/apply:os.getcwd []\n"))


def test_free_text_notes_allowed_in_named_sections(tmp_path):
    p = load_profile(write(tmp_path, "notes.yaml", (
        "client: x\nnotes: top\n"
        "language:\n  description: languages\n  allowed: [en]\n"
        "preferences:\n  description: what the client likes\n  pop: high\n"
        "dayparts:\n  notes: tuned in May\n  morning:\n    description: soft start\n")))
    assert p.preferences == {"pop": PreferenceLevel.HIGH}
    assert set(p.dayparts) == set(DEFAULT_DAYPARTS)


def test_hand_built_profile_with_invalid_policy_is_rejected():
    bad = profile(language=LanguageRules(unknown_policy="Review"))
    with pytest.raises(ProfileError, match="unknown_policy"):
        curate([T("a.mp3")], bad)
    with pytest.raises(ProfileError, match="preferences.pop"):
        curate([T("a.mp3")], profile(preferences={"pop": "hgih"}))


def test_daypart_profile_values_merge_over_defaults(tmp_path):
    snapshot = copy.deepcopy(DEFAULT_DAYPARTS)
    p = load_profile(write(tmp_path, "dp.yaml", (
        "client: x\n"
        "dayparts:\n"
        "  Morning:\n"
        "    energy_range: [0.2, 0.5]\n"
        "    preferred_tags: [Lo-Fi]\n"
        "    notes: free text is fine here\n"
        "  night:\n"
        "    energy_range: [0.05, 0.3]\n"
        "    target_duration_min: 90\n"
        "  evening:\n")))
    morning = p.dayparts["morning"]
    assert morning.energy_range == (0.2, 0.5) and morning.preferred_tags == ["Lo-Fi"]
    assert morning.energy_curve == DEFAULT_DAYPARTS["morning"].energy_curve           # inherited
    assert morning.opening_max_energy == DEFAULT_DAYPARTS["morning"].opening_max_energy
    assert p.dayparts["evening"] == DEFAULT_DAYPARTS["evening"]                         # null -> defaults
    night = p.dayparts["night"]
    assert night.name == "night" and night.target_duration_s == 5400.0 and night.energy_curve is None
    assert DEFAULT_DAYPARTS == snapshot                                                 # defaults untouched

    got = get_daypart(p, "MORNING")
    assert got == morning and got is not morning
    got.preferred_tags.append("mutated")
    assert p.dayparts["morning"].preferred_tags == ["Lo-Fi"]
    with pytest.raises(KeyError, match="available: afternoon, evening, morning, night"):
        get_daypart(p, "brunch")
    # a hand-built profile without dayparts still resolves the defaults
    assert get_daypart(ClientProfile(client="bare"), "afternoon") == DEFAULT_DAYPARTS["afternoon"]


def test_default_dayparts_follow_spec_intent():
    m, a, e = (DEFAULT_DAYPARTS[k] for k in ("morning", "afternoon", "evening"))
    # morning gentle-to-moderate, afternoon moderate, evening moderate-to-moderately-high
    assert m.energy_range[1] < e.energy_range[1] and m.energy_range[0] < a.energy_range[0] < e.energy_range[0]
    assert m.energy_curve[0] < a.energy_curve[0] <= e.energy_curve[0] and e.energy_curve[1] > a.energy_curve[1]
    assert m.opening_max_energy is not None and m.opening_max_energy <= m.energy_curve[1]
    assert "aggressive" in m.avoid_tags and "melodic" in m.preferred_tags
    assert "groove" in e.preferred_tags


# ---------------------------------------------------------------------------
# curate(): hard rules
# ---------------------------------------------------------------------------

def test_excluded_language_removed_even_if_allowlisted():
    tracks = [T("ru.mp3", artist="Some Artist", title="Song", language="ru"),
              T("rus.mp3", artist="Other", title="Song", language="rus"),       # ISO 639-2 from a tag
              T("en.mp3", artist="Some Artist", title="Hit", language="en")]
    prof = profile(allowlist=ListRules(artists=["Some Artist"], tracks=["rus.mp3"]))
    result = curate(tracks, prof)
    ru, rus, en = result.decisions
    assert not ru.eligible and ru.allowlisted
    assert "language 'ru' is excluded" in ru.exclusion_reasons[0]
    assert "allowlist cannot override" in ru.exclusion_reasons[0]
    assert not rus.eligible
    assert en.eligible and en.allowlisted
    assert [t.rel_path for t in result.eligible] == ["en.mp3"]
    assert [d.track.rel_path for d in result.excluded] == ["ru.mp3", "rus.mp3"]


def test_blocklisted_artist_track_and_tag_removed():
    tracks = [
        T("a/Blocked Band - Song.mp3", artist="Blocked Band", title="Song", language="en"),
        T("a/x.mp3", artist="Guest Star feat. Blocked Band", title="Duet", language="en"),
        T("b/Fine Artist - Banned Title.mp3", artist="Fine Artist", title="Banned Title", language="en"),
        T("c/by_path.mp3", language="en"),
        T("d/tagged.mp3", language="en", vibe_tags=["Explicit"]),
        T("d/genre.mp3", language="en", genre="Pop/Hard-Core"),
        T("e/keep.mp3", artist="Fine Artist", title="Keeper", language="en", genre="Pop"),
    ]
    prof = profile(blocklist=ListRules(artists=["blocked band"], tracks=["Fine Artist – Banned Title", "C\\BY_PATH.mp3"],
                                       tags=["explicit", "hard core"]),
                   allowlist=ListRules(tracks=["a/Blocked Band - Song.mp3"]))
    result = curate(tracks, prof)
    reasons = {d.track.rel_path: d.exclusion_reasons for d in result.decisions}
    assert reasons["a/Blocked Band - Song.mp3"][0].startswith("blocklisted artist 'blocked band'")
    assert "blocklist takes precedence" in reasons["a/Blocked Band - Song.mp3"][0]
    assert reasons["a/x.mp3"] == ["blocklisted artist 'blocked band'"]
    assert reasons["b/Fine Artist - Banned Title.mp3"] == ["blocklisted track 'Fine Artist – Banned Title'"]
    assert reasons["c/by_path.mp3"] == ["blocklisted track 'C\\BY_PATH.mp3'"]
    assert reasons["d/tagged.mp3"] == ["blocklisted tag 'explicit'"]
    assert reasons["d/genre.mp3"] == ["blocklisted tag 'hard core'"]
    assert [t.rel_path for t in result.eligible] == ["e/keep.mp3"]


def test_metadata_excluded_is_hard_and_reported():
    tracks, _ = apply_metadata([T("a.mp3", language="en")], {"a.mp3": {"excluded": True, "notes": "bad edit"}})
    d = curate(tracks, profile(allowlist=ListRules(tracks=["a.mp3"]))).decisions[0]
    assert not d.eligible
    assert d.exclusion_reasons == ["excluded in curator metadata (excluded=true) — note: bad edit"]
    assert d.track.exclusion_reasons == d.exclusion_reasons


# ---------------------------------------------------------------------------
# curate(): language soft rules
# ---------------------------------------------------------------------------

def test_unknown_language_flagged_for_review_not_treated_as_english():
    d = curate([T("unknown.mp3", artist="A", title="B")], profile()).decisions[0]
    assert d.eligible
    assert d.review_reasons == ["language unknown — needs curator review (not assumed to be English)"]
    assert d.track.language is None and d.track.requires_review is True
    # 'und' from an ID3 TLAN tag is not a language either
    d = curate([T("und.mp3", language="und")], profile()).decisions[0]
    assert d.eligible and "language unknown" in d.review_reasons[0]


@pytest.mark.parametrize("policy, eligible, n_review", [("review", True, 1), ("exclude", False, 0),
                                                         ("allow", True, 0)])
def test_unknown_language_policy(policy, eligible, n_review):
    prof = profile(language=LanguageRules(allowed=["en"], unknown_policy=policy, cyrillic_script_policy="ignore"))
    d = curate([T("x.mp3")], prof).decisions[0]
    assert d.eligible is eligible and len(d.review_reasons) == n_review
    if not eligible:
        assert "unknown_policy: exclude" in d.exclusion_reasons[0]


@pytest.mark.parametrize("policy", ["exclude", "review", "ignore"])
def test_cyrillic_script_policy(policy):
    prof = profile(language=LanguageRules(allowed=["en", "uk"], excluded=["ru"], unknown_policy="allow",
                                          cyrillic_script_policy=policy))
    tracks = [T("Ночь.mp3", artist="Группа", title="Ночь", script_hint="cyrillic"),
              T("Known.mp3", artist="Гурт", title="Ніч", script_hint="cyrillic", language="uk")]
    unknown, known = curate(tracks, prof).decisions
    assert known.eligible and not known.review_reasons          # language verified -> hint irrelevant
    if policy == "exclude":
        assert not unknown.eligible and unknown.exclusion_reasons == [CYRILLIC_EXCLUDE_REASON]
        assert CYRILLIC_EXCLUDE_REASON == ("Cyrillic script in artist/title; language unverified — "
                                           "excluded pending curator confirmation")
    elif policy == "review":
        assert unknown.eligible and "Cyrillic script" in unknown.review_reasons[0]
    else:
        assert unknown.eligible and unknown.review_reasons == [] and unknown.exclusion_reasons == []


def test_language_not_in_allowed_list_excluded_unless_allowlisted():
    tracks = [T("de.mp3", language="de", artist="Kraft"), T("de2.mp3", language="DE", artist="Allowed Act")]
    prof = profile(allowlist=ListRules(artists=["allowed act"]))
    de, de2 = curate(tracks, prof).decisions
    assert not de.eligible and de.exclusion_reasons == ["language 'de' is not in the allowed list (en, fr)"]
    assert de2.eligible and de2.allowlisted and de2.review_reasons == []


def test_allowlist_overrides_soft_rules():
    prof = profile(language=LanguageRules(allowed=["en"], unknown_policy="exclude", cyrillic_script_policy="exclude"),
                   preferences={"aggressive_rap": PreferenceLevel.AVOID},
                   preference_tags={"aggressive_rap": ["aggressive rap"]},
                   require_approved=True,
                   allowlist=ListRules(tags=["house favourite"]))
    track = T("x.mp3", script_hint="cyrillic", genre="Aggressive Rap", vibe_tags=["House Favourite"])
    plain = T("y.mp3", script_hint="cyrillic", genre="Aggressive Rap")
    allowed, blocked = curate([track, plain], prof).decisions
    assert allowed.eligible and allowed.allowlisted and allowed.exclusion_reasons == []
    assert allowed.editorial_score == -1.0                       # still ranked low, but eligible
    assert not blocked.eligible
    assert blocked.exclusion_reasons == [
        "language unknown — excluded by the client profile (language.unknown_policy: exclude)",
        CYRILLIC_EXCLUDE_REASON,
        "matches avoided preference 'aggressive_rap' (tag 'Aggressive Rap')",
        "not approved by curator (the client profile sets require_approved)",
    ]


def test_allowlist_only():
    prof = profile(allowlist=ListRules(artists=["Chosen"], only=True))
    tracks = [T("a.mp3", artist="Chosen", language="en"), T("b.mp3", artist="Other", language="en"),
              T("c.mp3", artist="Chosen", language="ru")]
    a, b, c = curate(tracks, prof).decisions
    assert a.eligible
    assert not b.eligible and b.exclusion_reasons == ["not on the client allowlist (allowlist.only is set)"]
    assert not c.eligible and "language 'ru'" in c.exclusion_reasons[0]   # hard rule still first


# ---------------------------------------------------------------------------
# curate(): preferences, approval, review flags
# ---------------------------------------------------------------------------

def test_avoid_preference_excludes_case_insensitively():
    prof = profile(preferences={"aggressive_rap": PreferenceLevel.AVOID},
                   preference_tags={"aggressive_rap": ["hip-hop hardcore", "drill"]})
    tracks = [T("a.mp3", language="en", genre="Hip Hop Hardcore"), T("b.mp3", language="en", mood="DRILL"),
              T("c.mp3", language="en", genre="Hip Hop")]
    a, b, c = curate(tracks, prof).decisions
    assert not a.eligible and a.exclusion_reasons == ["matches avoided preference 'aggressive_rap' (tag 'Hip Hop Hardcore')"]
    assert not b.eligible
    assert c.eligible


def test_require_approved_and_manual_review_required():
    tracks = [T("ok.mp3", language="en", approved=True), T("new.mp3", language="en")]
    ok, new = curate(tracks, profile(require_approved=True)).decisions
    assert ok.eligible and not new.eligible
    assert new.exclusion_reasons == ["not approved by curator (the client profile sets require_approved)"]

    ok, new = curate(tracks, profile(manual_review_required=True)).decisions
    assert ok.eligible and ok.review_reasons == []
    assert new.eligible and new.review_reasons == [NOT_APPROVED_REVIEW_REASON]


def test_requires_review_track_stays_eligible_with_reason():
    tracks, _ = apply_metadata([T("a.mp3", language="en"), T("b.mp3", language="en")],
                               {"a.mp3": {"requires_review": True, "notes": "check intro"}})
    tracks[1].requires_review = True
    tracks[1].review_reasons.append("analysis: tempo unstable")
    a, b = curate(tracks, profile(allowlist=ListRules(tracks=["a.mp3"]))).decisions
    assert a.eligible and a.review_reasons == ["flagged for review in curator metadata — note: check intro"]
    assert b.eligible and b.review_reasons == [] and b.track.review_reasons == ["analysis: tempo unstable"]


def test_editorial_score_ordering():
    prof = profile(preferences={"accessible_melodic_music": PreferenceLevel.HIGH,
                                "experimental_instrumental": PreferenceLevel.LOW,
                                "neutral_stuff": PreferenceLevel.MEDIUM},
                   preference_tags={"accessible_melodic_music": ["melodic", "soft pop"],
                                    "experimental_instrumental": ["experimental", "instrumental"],
                                    "neutral_stuff": ["jazz"]})
    tracks = [T("high.mp3", language="en", vibe_tags=["Melodic"]),
              T("neutral.mp3", language="en", genre="Rock"),
              T("medium.mp3", language="en", genre="Jazz"),
              T("low.mp3", language="en", vibe_tags=["experimental"]),
              T("mixed.mp3", language="en", genre="Soft-Pop", mood="Instrumental")]
    scores = {d.track.rel_path: d.editorial_score for d in curate(tracks, prof).decisions}
    assert scores["high.mp3"] == 1.0
    assert scores["neutral.mp3"] == 0.0 and scores["medium.mp3"] == 0.0
    assert scores["low.mp3"] == -0.6
    assert scores["mixed.mp3"] == pytest.approx(0.2)                      # (1 - 0.6) / 2
    assert scores["high.mp3"] > scores["neutral.mp3"] > scores["low.mp3"]
    assert all(-1.0 <= s <= 1.0 for s in scores.values())


def test_unknown_familiarity_never_raises_a_score():
    p = parse_profile({"client": "x", "preferences": {"recognizable_pop": "high"}})   # spec-style, no tags
    tracks = [T("unknown.mp3", language="en", genre="Pop"),
              T("famous.mp3", language="en", familiarity=0.9),
              T("obscure.mp3", language="en", familiarity=0.1),
              T("auto.mp3", language="en", familiarity=0.95, feature_sources={"familiarity": "auto:guess"})]
    result = curate(tracks, p)
    scores = {d.track.rel_path: d.editorial_score for d in result.decisions}
    assert scores["unknown.mp3"] == 0.0                     # unknown stays neutral: never "popular"
    assert scores["famous.mp3"] == pytest.approx(0.8)
    assert scores["obscure.mp3"] == pytest.approx(-0.8)
    assert scores["auto.mp3"] == 0.0                        # an estimate is not curated familiarity
    assert result.decisions[0].track.familiarity is None
    assert editorial_scores(result) == {d.track.track_id: d.editorial_score for d in result.decisions}


def test_default_profile_is_permissive_but_flags_unknown_language():
    """EDIT-9: without --profile nothing is excluded, but spec §6 still applies — an
    unknown language (and a Cyrillic artist/title) is flagged for review, never silently
    accepted. (The old assertion 'no review reasons at all' encoded the defect.)"""
    tracks = [T("a.mp3"), T("b.mp3", script_hint="cyrillic"), T("c.mp3", language="ru"),
              T("d.mp3", language="de")]
    result = curate(tracks, default_profile())
    assert len(result.eligible) == 4 and all(not d.exclusion_reasons for d in result.decisions)
    a, b, c, d = result.decisions
    assert a.review_reasons == [UNKNOWN_LANGUAGE_REVIEW_REASON] and a.track.requires_review
    assert b.review_reasons == [UNKNOWN_LANGUAGE_REVIEW_REASON, CYRILLIC_REVIEW_REASON]
    assert c.review_reasons == [] and d.review_reasons == []        # known languages: no flag
    assert not c.track.requires_review and not d.track.requires_review
    assert default_profile().language.unknown_policy == "review"
    assert default_profile().language.cyrillic_script_policy == "review"


def test_inputs_not_mutated_and_one_decision_per_track():
    prof = load_profile(CLIENT_YAML)
    tracks = [T("a.mp3"), T("b.mp3", language="ru"), T("c.mp3", script_hint="cyrillic", vibe_tags=["melodic"]),
              T("d.mp3", language="en", approved=True, genre="Pop")]
    before = [t.to_dict() for t in tracks]
    result = curate(tracks, prof)
    assert [t.to_dict() for t in tracks] == before
    assert [d.track.rel_path for d in result.decisions] == ["a.mp3", "b.mp3", "c.mp3", "d.mp3"]
    assert all(d.track is not t for d, t in zip(result.decisions, tracks))
    assert all(d.track.vibe_tags is not t.vibe_tags for d, t in zip(result.decisions, tracks))
    assert len(result.eligible) + len(result.excluded) == len(tracks)
    # curating the curated copies again is stable (no duplicated reasons)
    again = curate([d.track for d in result.decisions], prof)
    assert [d.track.review_reasons for d in again.decisions] == [d.track.review_reasons for d in result.decisions]


def test_language_and_tag_helpers():
    assert normalize_language("EN-us") == "en" and normalize_language("rus") == "ru"
    assert normalize_language("und") is None and normalize_language("  ") is None
    assert normalize_language(None) is None
    tags = track_tags(T("x.mp3", vibe_tags=["Feel_Good"], genre="Pop/Rock", mood="Warm"))
    assert {"feel good", "pop/rock", "pop", "rock", "warm"} <= set(tags)


def test_example_metadata_with_sample_profile_end_to_end():
    """The shipped example CSV + client profile: curator values drive the decisions."""
    overrides = load_metadata(REPO / "profiles" / "metadata_example.csv")
    library = [T(rel, artist=rel.split("/")[-1].split(" - ")[0], title=rel.split(" - ")[-1].rsplit(".", 1)[0],
                 script_hint="cyrillic" if any("\u0400" <= ch <= "\u04ff" for ch in rel) else None)
               for rel in overrides]
    tracks, warnings = apply_metadata(library, overrides)
    assert warnings == []
    result = curate(tracks, load_profile(CLIENT_YAML))
    by_name = {d.track.rel_path.split("/")[0] + ":" + (d.track.language or "?"): d for d in result.decisions}
    morning = by_name["Morning:en"]
    assert morning.eligible and morning.review_reasons == [] and morning.editorial_score > 0
    evening = by_name["Evening:fr"]
    assert evening.eligible and any("curator metadata" in r for r in evening.review_reasons)
    ru = by_name["Misc:ru"]
    assert not ru.eligible and "language 'ru' is excluded" in ru.exclusion_reasons[0]
    sketch = by_name["Misc:?"]
    assert not sketch.eligible and sketch.exclusion_reasons[0].startswith("excluded in curator metadata")


# ---------------------------------------------------------------------------
# Review-finding regressions
# ---------------------------------------------------------------------------

EXCLUDED_RU_ONLY = {"client": "x", "language": {"excluded": ["ru"]}}


@pytest.mark.parametrize("raw", ["ru;en", "rus;eng", "rus/eng", "ru, en", "en;ru", "русский", "Russian"])
def test_multi_language_tag_with_excluded_language_is_hard_excluded(raw):
    """EDIT-1 / INT-4 / SPEC-3: 'ru;en' (what analysis stores for TLAN 'rus;eng') and
    language names must hit `excluded: [ru]` — also with no allowed list at all."""
    d = curate([T("A - ru_en.mp3", artist="A", title="Song", language=raw)],
               parse_profile(EXCLUDED_RU_ONLY)).decisions[0]
    assert not d.eligible
    assert d.exclusion_reasons[0].startswith("language 'ru' is excluded by the client profile")


def test_allowlist_cannot_rescue_a_multi_language_track_with_an_excluded_language():
    """EDIT-1 / SPEC-3: under the sample profile an allowlisted 'ru;en' track used to slip
    through (only the soft allowed-list rule caught it, and the allowlist overrides that)."""
    prof = load_profile(CLIENT_YAML)
    prof.allowlist = ListRules(artists=["A"])
    d = curate([T("A - ru_en.mp3", artist="A", title="Song", language="ru;en", approved=True)], prof).decisions[0]
    assert not d.eligible and d.allowlisted
    assert d.exclusion_reasons == ["language 'ru' is excluded by the client profile (track language 'ru;en') "
                                   "(an allowlist cannot override an excluded language)"]


def test_multi_language_track_is_allowed_only_when_every_language_is_allowed():
    """EDIT-1 / INT-4: 'en;fr' with both languages allowed was wrongly excluded; 'en;de'
    (de not allowed) stays excluded by the soft rule."""
    prof = load_profile(CLIENT_YAML)
    en_fr, en_de = curate([T("B - en_fr.mp3", language="en;fr", approved=True),
                           T("C - en_de.mp3", language="eng;deu", approved=True)], prof).decisions
    assert en_fr.eligible and en_fr.review_reasons == []
    assert not en_de.eligible
    assert en_de.exclusion_reasons[0].startswith("language 'de' (track language 'en;de') is not in the allowed list")


def test_iso639_2_codes_are_resolved_for_the_allowed_list():
    """INT-9: 'por' from a TLAN tag is Portuguese — allowed by `allowed: [en, pt]`."""
    prof = parse_profile({"client": "x", "language": {"allowed": ["en", "pt"]}})
    d = curate([T("Charlie - Three.mp3", language="por")], prof).decisions[0]
    assert d.eligible and d.review_reasons == []


def test_unresolved_language_values_count_as_unknown():
    """EDIT-1: an unrecognised tag value is not a known language — unknown_policy and the
    Cyrillic rule apply (it used to pass as if it were a language code)."""
    prof = load_profile(CLIENT_YAML)          # unknown_policy: review, cyrillic: exclude
    klingon, partial, cyr = curate([
        T("k.mp3", language="klingon", approved=True),
        T("p.mp3", language="en;klingon", approved=True),
        T("Ночь.mp3", artist="Группа", title="Ночь", language="tatarish", script_hint="cyrillic", approved=True),
    ], prof).decisions
    assert klingon.eligible and klingon.review_reasons[0].startswith(UNKNOWN_LANGUAGE_REVIEW_REASON)
    assert "'klingon'" in klingon.review_reasons[0] and klingon.track.requires_review
    assert partial.eligible and partial.review_reasons[0].startswith(UNKNOWN_LANGUAGE_REVIEW_REASON)
    assert not cyr.eligible and CYRILLIC_EXCLUDE_REASON in cyr.exclusion_reasons


def test_curated_familiarity_is_monotonic_against_unknown():
    """EDIT-2: a curated familiarity >= 0.5 never scores below an unknown familiarity for
    otherwise identical tracks (it used to: f=0.8 -> 0.8, unknown -> 1.0), and a curator
    'hit' tag is never pulled below an untagged unknown track."""
    prof = load_profile(CLIENT_YAML)
    tracks = [T(f"{name}.mp3", language="en", vibe_tags=["melodic"] + extra, familiarity=f, approved=True)
              for name, f, extra in [("unknown", None, []), ("f10", 1.0, []), ("f08", 0.8, []), ("f06", 0.6, []),
                                     ("f05", 0.5, []), ("f03", 0.3, []), ("hit_unknown", None, ["hit"]),
                                     ("hit_f03", 0.3, ["hit"]), ("hit_f08", 0.8, ["hit"])]]
    s = {d.track.rel_path[:-4]: d.editorial_score for d in curate(tracks, prof).decisions}
    assert s["f10"] > s["f08"] > s["f06"] > s["unknown"] > s["f03"]
    assert s["f05"] == s["unknown"]                                   # 0.5 is neutral, like unknown
    assert s["hit_unknown"] == s["hit_f08"] == s["hit_f03"] == 1.0     # the curator tag counts in full
    assert s["hit_unknown"] >= s["f10"] > s["unknown"]
    for f in (0.5, 0.6, 0.8, 0.95, 1.0):                              # the invariant itself
        one = curate([T("x.mp3", language="en", vibe_tags=["melodic"], familiarity=f)], prof).decisions[0]
        assert one.editorial_score >= s["unknown"]


def test_allowlist_restores_eligibility_but_keeps_review_flags():
    """EDIT-7: allowlisting an artist must not hide unknown language, the Cyrillic hint or
    the missing approval — the track is eligible, flagged and requires_review is set."""
    prof = load_profile(CLIENT_YAML)        # cyrillic: exclude, unknown: review, manual_review_required
    prof.allowlist = ListRules(artists=["Земфира"])
    d = curate([T("Земфира - Хочешь.mp3", artist="Земфира", title="Хочешь", script_hint="cyrillic")],
               prof).decisions[0]
    assert d.eligible and d.allowlisted and d.exclusion_reasons == []
    assert d.review_reasons == [UNKNOWN_LANGUAGE_REVIEW_REASON, CYRILLIC_REVIEW_REASON, NOT_APPROVED_REVIEW_REASON]
    assert d.track.requires_review is True
    # require_approved / unknown_policy: exclude become review flags for an allowlisted track ...
    strict = profile(language=LanguageRules(allowed=["en"], unknown_policy="exclude"), require_approved=True,
                     allowlist=ListRules(artists=["Fav"]))
    s = curate([T("s.mp3", artist="Fav")], strict).decisions[0]
    assert s.eligible and s.review_reasons == [UNKNOWN_LANGUAGE_REVIEW_REASON, NOT_APPROVED_REVIEW_REASON]
    # ... while a client policy the allowlist exists to override stays silent
    de = curate([T("de.mp3", artist="Fav", language="de", approved=True)], strict).decisions[0]
    assert de.eligible and de.review_reasons == []


def test_analysis_review_reasons_set_requires_review():
    """EDIT-8: reasons left by analysis (Cyrillic hint, truncated file ...) used to stay
    on an eligible track with requires_review=False, so playlist/QC never saw them."""
    truncated = T("B - Two.mp3", language="en", approved=True,
                  review_reasons=["container duration 40.0 s differs from the decoded duration 22.0 s "
                                  "(truncated or damaged file?)"])
    d = curate([truncated], parse_profile(EXCLUDED_RU_ONLY)).decisions[0]
    assert d.eligible and d.review_reasons == [] and d.track.requires_review is True
    # an explicit curator sign-off (requires_review: false in the CSV) is honoured for analysis reasons
    signed, _ = apply_metadata([truncated], {"B - Two.mp3": {"requires_review": False}})
    assert curate(signed, parse_profile(EXCLUDED_RU_ONLY)).decisions[0].track.requires_review is False


def test_cyrillic_title_with_tag_language_that_is_not_cyrillic_is_a_conflict():
    """EDIT-8: Cyrillic artist/title + an automatic (ID3) language 'en' is not proof that
    the song is English; a curator-set 'en' or a Cyrillic-script language is."""
    kw = dict(artist="Кино", title="Группа крови", script_hint="cyrillic", approved=True,
              review_reasons=["Cyrillic script in artist/title — language not verified"])
    id3_en = T("Кино - Группа крови.mp3", language="en", feature_sources={"language": "id3"}, **kw)
    id3_uk = T("uk.mp3", language="uk", feature_sources={"language": "id3"}, **kw)
    manual, _ = apply_metadata([T("manual.mp3", **kw)], {"manual.mp3": {"language": "en"}})
    for prof in (load_profile(CLIENT_YAML), parse_profile(EXCLUDED_RU_ONLY)):
        a, b, c = curate([id3_en, id3_uk, manual[0]], prof).decisions
        assert a.eligible and a.track.requires_review
        assert any("file tag says language 'en'" in r for r in a.review_reasons)
        assert not any("file tag says" in r for r in b.track.review_reasons)   # 'uk' is Cyrillic-script
        assert c.eligible and c.review_reasons == []
        # the curator confirmed the language: the analysis hint is resolved, nothing left open
        assert c.track.review_reasons == [] and c.track.requires_review is False
    ignore = parse_profile({"client": "x", "language": {"cyrillic_script_policy": "ignore"}})
    assert curate([id3_en], ignore).decisions[0].review_reasons == []


@pytest.mark.parametrize("artist", ["Bravo/Zulu", "Bravo / Zulu", "Bravo & Zulu", "Bravo x Zulu",
                                    "Bravo feat. Zulu", "Bravo (feat. Zulu)", "ZULU"])
def test_blocklisted_co_artist_is_excluded_at_curation(artist):
    """INT-6: '/' (the ID3v2.3 multi-artist separator) and the other credit separators
    are split exactly as QC splits them, so the blocklist is enforced before export."""
    prof = profile(blocklist=ListRules(artists=["Zulu"]))
    d = curate([T("Bravo - Two.mp3", artist=artist, title="Two", language="en")], prof).decisions[0]
    assert not d.eligible and d.exclusion_reasons == ["blocklisted artist 'Zulu'"]


def test_full_artist_credit_with_separator_still_matches():
    """INT-6: names that contain a separator ('AC/DC') still match by their full credit."""
    prof = profile(blocklist=ListRules(artists=["AC/DC"]))
    acdc, ac = curate([T("x.mp3", artist="AC/DC", language="en"), T("y.mp3", artist="AC", language="en")],
                      prof).decisions
    assert not acdc.eligible and ac.eligible


@pytest.mark.parametrize("name", ["../../escaped", "10:00-14:00", "a/b", "a\\b", "..", ".", "con", "lpt1",
                                  "night.", "what?", "x" * 65])
def test_unsafe_daypart_names_are_rejected(tmp_path, name):
    """SAFE-13: a daypart name becomes an output folder / file-name suffix — path
    separators, ':' and other Windows-invalid names must fail at profile load."""
    text = f"client: x\ndayparts:\n  {json.dumps(name)}:\n    energy_range: [0.1, 0.5]\n"
    with pytest.raises(ProfileError, match="dayparts") as exc:
        load_profile(write(tmp_path, "dp.yaml", text))
    assert any(w in str(exc.value) for w in ("output folder", "reserved device", "longer than"))


def test_safe_daypart_names_are_accepted_and_hand_built_profiles_are_checked(tmp_path):
    """SAFE-13: Cyrillic / spaced names are fine; profiles built in code are checked too."""
    p = load_profile(write(tmp_path, "dp.yaml", "client: x\ndayparts:\n  утро:\n  late night:\n  10-14:\n"))
    assert {"утро", "late night", "10-14"} <= set(p.dayparts)
    assert daypart_name_problem("evening") is None
    bad = ClientProfile(client="code", dayparts={"../x": DaypartProfile(name="../x")})
    with pytest.raises(ProfileError, match="output folder"):
        get_daypart(bad, "../x")
