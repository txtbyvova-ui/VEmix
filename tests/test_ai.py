"""VEmix 3.0 AI track understanding (spec §4): strict schema + evidence validator, payload
without paths, content-addressed cache, re-ask / negative caching / circuit breaker, and the
Anthropic / OpenAI-compatible / fake providers. No network and no API key: the Anthropic SDK
client and the HTTP transport are fakes."""

from __future__ import annotations

import copy
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from veg_mixer import scenario
from veg_mixer.ai import (
    AI_CACHE_NAME,
    AI_MODES,
    INTERPRETATION_SCHEMA,
    PHASES,
    PROMPT_VERSION,
    ROLES,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    AICache,
    AIConfigError,
    AIResult,
    AIUnavailableError,
    AnthropicProvider,
    FakeProvider,
    OpenAICompatibleProvider,
    build_payload,
    cache_key,
    get_provider,
    input_digest,
    interpret_tracks,
    schema_subset_problems,
    validate_interpretation,
)
from veg_mixer.ai import interpret as interpret_mod
from veg_mixer.ai.providers import FALLBACK_BETA, parse_json_text
from veg_mixer.models import TrackFeatures
from veg_mixer.moods import MOOD_COORDINATES
from veg_mixer.net import HttpClient, HttpResponse, NetTimeout

NOW = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)
SECRET = "sk-test-SECRET-0123456789"

# Shape of an ``audio_intel.analyze_segments`` value (summary nested, dict-valued summaries).
SEGMENTS = {
    "version": "segments-v1", "duration_s": 201.0, "analysed_s": 201.0, "segment_s": 10.0,
    "times": [0.0, 10.0, 20.0], "intensity": [0.3, 0.5, 0.8], "rms_db": [-6.0, 0.0, 1.0],
    "summary": {
        "energy_in": 0.3, "energy_out": 0.8, "energy_mean": 0.55, "energy_peak": 0.85,
        "trajectory": {"value": "rising", "confidence": 0.7, "slope_per_min": 0.12, "oscillation": 0.05},
        "density": {"value": "dense", "score": 0.7, "confidence": 0.6},
        "percussive_density": 0.6, "bass_density": 0.5, "dynamic_range_db": 9.5,
        "tempo": {"bpm": 120.0, "confidence": 0.8, "half_double_ambiguous": False,
                  "candidates": [{"bpm": 120.0, "strength": 0.9, "score": 0.9}]},
        "vocal_presence_proxy": {"value": 0.4, "confidence": 0.3, "method": "harmonic-band-proxy-v1"},
        "timbre": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    },
    "sections": [{"start_s": 0.0, "end_s": 20.0, "label": "intro", "intensity": 0.3, "repeat_group": None,
                  "confidence": 0.4},
                 {"start_s": 60.0, "end_s": 90.0, "label": "peak", "intensity": 0.85, "repeat_group": 1,
                  "confidence": 0.4}],
    "notes": ["only the first 600 s were analysed"],
    "path": "D:/Music/Secret Folder/Artist A - Song One.mp3",        # must never reach the payload
    "source_file": "Secret Folder/Artist A - Song One.mp3",
}


def segments_with(**summary) -> dict:
    seg = copy.deepcopy(SEGMENTS)
    seg["summary"].update(summary)
    return seg


def make_track(n: int = 1, *, folder: str = "Secret Folder", mood: str | None = "tense", bpm: float | None = 120.0,
               energy: float | None = 0.6, manual_energy: bool = False, genre: str | None = "Pop") -> TrackFeatures:
    rel = f"{folder}/Artist {n} - Song {n}.mp3"
    sources = {"artist": "id3", "title": "id3", "genre": "id3"}
    if bpm is not None:
        sources["bpm"] = "auto:bpm-v1"
    if energy is not None:
        sources["energy"] = "manual" if manual_energy else "auto:energy-heuristic-v1"
    t = TrackFeatures(path=Path("D:/Music") / rel, duration_s=200.0 + n, bpm=bpm, camelot="8A", key_confidence=0.7,
                      energy=energy, energy_confidence=1.0 if manual_energy else 0.5, rel_path=rel,
                      artist=f"Artist {n}", title=f"Song {n}", genre=genre, mood=mood,
                      feature_sources={**sources, "camelot": "auto:key-v1", "key_confidence": "auto:key-v1"})
    if mood is not None:
        t.feature_sources["mood"] = "manual"
    return t


def payload_for(track: TrackFeatures | None = None, **kw) -> dict:
    return build_payload(track or make_track(), segments=kw.pop("segments", SEGMENTS), **kw)


def good_answer() -> dict:
    return {
        "moods": [{"label": "Tense", "confidence": 0.7,
                   "evidence": ["measured.segments.trajectory", "curator.mood"]}],
        "energy_trajectory": {"value": "rising", "confidence": 0.8,
                              "evidence": ["measured.segments.energy_in", "measured.segments.energy_out"]},
        "danceability": {"value": 0.6, "confidence": 0.5, "evidence": ["measured.segments.percussive_density"]},
        "groove": {"value": None, "confidence": 0.0, "evidence": []},
        "sonic_density": {"value": "dense", "confidence": 0.6, "evidence": ["measured.segments.density"]},
        "roles": [{"role": "tension_builder", "confidence": 0.7,
                   "evidence": ["measured.segments.trajectory", "measured.bpm"]}],
        "positions": [{"phase": "build", "confidence": 0.6, "evidence": ["measured.segments.energy_out"]}],
        "risky_pairings": [{"with": "a sparse ambient breather", "reason": "density would collapse",
                            "confidence": 0.5, "evidence": ["measured.segments.density"]}],
        "notes": "Segment analysis available; no catalogue facts.",
    }


def walk(node, path="$"):
    yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")


def json_native(obj) -> None:
    json.dumps(obj, allow_nan=False)
    for _, v in walk(obj):
        assert v is None or isinstance(v, (str, int, float, bool, list, dict)), type(v)
        assert not isinstance(v, float) or math.isfinite(v)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_uses_only_the_supported_subset():
    assert schema_subset_problems(INTERPRETATION_SCHEMA) == []
    forbidden = {"minimum", "maximum", "minLength", "maxLength", "exclusiveMinimum", "exclusiveMaximum",
                 "multipleOf", "pattern", "maxItems"}
    objects = 0
    for path, node in walk(INTERPRETATION_SCHEMA):
        if isinstance(node, dict) and not path.endswith(".properties"):
            assert not forbidden & set(node), path
            assert node.get("minItems", 0) <= 1, path
            if node.get("type") == "object":
                objects += 1
                assert node["additionalProperties"] is False, path
                assert node["required"] == list(node["properties"]), path
    assert objects >= 9
    # the checker itself notices violations
    bad = {"type": "object", "properties": {"a": {"type": "number", "minimum": 0}}, "required": []}
    problems = schema_subset_problems(bad)
    assert any("additionalProperties" in p for p in problems)
    assert any("required" in p for p in problems)
    assert any("minimum" in p for p in problems)


def test_schema_vocabularies():
    props = INTERPRETATION_SCHEMA["properties"]
    assert ROLES is scenario.ROLES
    assert props["roles"]["items"]["properties"]["role"]["enum"] == list(scenario.ROLES)
    assert props["positions"]["items"]["properties"]["phase"]["enum"] == list(PHASES)
    assert PHASES == ("intro", "build", "first_peak", "breather", "second_build", "climax", "descent", "closer")
    assert props["energy_trajectory"]["properties"]["value"]["enum"] == ["rising", "falling", "stable", "wave",
                                                                        "unknown"]
    assert props["sonic_density"]["properties"]["value"]["enum"] == ["sparse", "moderate", "dense", "unknown"]
    assert set(INTERPRETATION_SCHEMA["required"]) == {"moods", "energy_trajectory", "danceability", "groove",
                                                      "sonic_density", "roles", "positions", "risky_pairings", "notes"}
    # v2: no identity in the payload, per-field evidence whitelists (old cache entries not reused)
    assert PROMPT_VERSION == "interpret-v2" and SCHEMA_VERSION == "interpretation-v2"
    assert AI_CACHE_NAME == "veg_ai_cache.json" and AI_MODES == ("off", "cache", "online")


def test_system_prompt_rules_and_vocabularies():
    for word in list(ROLES) + list(PHASES):
        assert word in SYSTEM_PROMPT
    assert set(interpret_mod.ROLE_DEFINITIONS) >= set(ROLES), "every role needs a definition"
    for mood in ("ominous", "playful", "tense", "euphoric", "melancholic", "sensual", "mysterious"):
        assert mood in MOOD_COORDINATES and mood in SYSTEM_PROMPT
    assert "memory" in SYSTEM_PROMPT and "evidence" in SYSTEM_PROMPT and "unknown" in SYSTEM_PROMPT
    # the identity is not an input any more: the prompt says so and describes no "track" section
    assert '"track"' not in SYSTEM_PROMPT and "track.*" not in SYSTEM_PROMPT
    assert "deliberately NOT given" in SYSTEM_PROMPT
    assert "rhythm evidence" in SYSTEM_PROMPT and "never support a mood" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def test_valid_answer_passes_cleanly():
    payload = payload_for()
    clean, errors = validate_interpretation(good_answer(), payload)
    assert errors == []
    assert clean["dropped"] == []
    assert clean["moods"] == [{"label": "tense", "confidence": 0.7,
                               "evidence": ["measured.segments.trajectory", "curator.mood"]}]
    assert clean["energy_trajectory"]["value"] == "rising"
    assert clean["danceability"] == {"value": 0.6, "confidence": 0.5,
                                     "evidence": ["measured.segments.percussive_density"]}
    assert clean["groove"] == {"value": None, "confidence": 0.0, "evidence": []}
    assert clean["roles"][0]["role"] == "tension_builder"
    assert clean["risky_pairings"][0]["with"] == "a sparse ambient breather"
    json_native(clean)


def test_missing_and_extra_keys_are_errors_but_the_rest_is_kept():
    answer = good_answer()
    del answer["groove"]
    answer["bpm"] = 128
    clean, errors = validate_interpretation(answer, payload_for())
    assert "missing key 'groove'" in errors
    assert any("unexpected key 'bpm'" in e for e in errors)
    assert clean["groove"] == {"value": None, "confidence": 0.0, "evidence": []}
    assert clean["moods"] and "bpm" not in clean


def test_wrong_types_and_enum_values():
    answer = good_answer()
    answer["energy_trajectory"]["value"] = "exploding"
    answer["roles"].append({"role": "hero", "confidence": 0.9, "evidence": ["measured.segments.energy_mean"]})
    answer["moods"].append({"label": "dark", "confidence": "high", "evidence": ["curator.mood"]})
    answer["moods"].append({"label": "dark", "confidence": 0.5, "evidence": "curator.mood"})
    answer["positions"].append({"phase": "build", "confidence": 0.5, "evidence": [], "extra": 1})
    answer["notes"] = 5
    clean, errors = validate_interpretation(answer, payload_for())
    text = "\n".join(errors)
    assert "energy_trajectory.value: 'exploding' is not one of" in text
    assert "roles[1].role: 'hero' is not one of" in text
    assert "moods[1].confidence: expected number, got string" in text
    assert "moods[2].evidence: expected array, got string" in text
    assert "positions[1]: unexpected key 'extra'" in text
    assert "notes: expected string" in text
    assert clean["energy_trajectory"]["value"] == "unknown"
    assert [r["role"] for r in clean["roles"]] == ["tension_builder"]
    assert [m["label"] for m in clean["moods"]] == ["tense"]


@pytest.mark.parametrize("value, ok, stored", [
    (1.0000004, True, 1.0), (-0.0000004, True, 0.0), (0.4567, True, 0.457),
    (1.5, False, None), (-0.2, False, None), (85, False, None), (float("nan"), False, None),
    (float("inf"), False, None), (True, False, None),
])
def test_confidence_range(value, ok, stored):
    answer = good_answer()
    answer["moods"][0]["confidence"] = value
    clean, errors = validate_interpretation(answer, payload_for())
    if ok:
        assert errors == [] and clean["moods"][0]["confidence"] == stored
    else:
        assert any(e.startswith("moods[0].confidence") for e in errors) and clean["moods"] == []


def test_fabricated_evidence_drops_the_claim():
    payload = payload_for()
    assert "facts" not in payload
    answer = good_answer()
    answer["moods"] = [
        {"label": "melancholic", "confidence": 0.9, "evidence": ["measured.segments.lyrics_sentiment"]},
        {"label": "nostalgic", "confidence": 0.8, "evidence": ["facts.first_release_date"]},
        {"label": "tense", "confidence": 0.6, "evidence": ["curator.mood", "facts.chart_position"]},
        {"label": "dark", "confidence": 0.6, "evidence": []},
        {"label": "playful", "confidence": 0.5, "evidence": ["curator.mood"]},
    ]
    clean, errors = validate_interpretation(answer, payload)
    assert errors == []                                   # evidence problems are drops, not structure errors
    assert [m["label"] for m in clean["moods"]] == ["playful"]
    reasons = {d["claim"]["label"]: d["reason"] for d in clean["dropped"]}
    assert "not a key path of the input" in reasons["melancholic"]
    assert "not a key path of the input" in reasons["nostalgic"]
    assert "not a key path of the input" in reasons["tense"]      # one invented path is enough
    assert reasons["dark"] == "no evidence"
    json_native(clean["dropped"])


def test_identity_fields_are_not_evidence():
    # interpret-v2 payloads carry no identity: citing artist / title cites data that does not exist
    answer = good_answer()
    answer["moods"] = [{"label": "ominous", "confidence": 0.9, "evidence": ["track.title"]},
                       {"label": "spooky", "confidence": 0.6, "evidence": ["track.artist", "curator.mood"]},
                       {"label": "eerie", "confidence": 0.6, "evidence": ["measured"]}]
    clean, _ = validate_interpretation(answer, payload_for())
    assert clean["moods"] == []
    reasons = {d["claim"]["label"]: d["reason"] for d in clean["dropped"]}
    assert "not a key path" in reasons["ominous"] and "not a key path" in reasons["spooky"]
    assert "admissible" in reasons["eerie"]                       # a whole section is not evidence
    # a (hand-built, legacy) payload that still has a track block: identity is never evidence
    legacy = {**payload_for(), "track": {"artist": "Artist 1", "title": "Song 1"}}
    clean, _ = validate_interpretation(answer, legacy)
    assert clean["moods"] == [{"label": "spooky", "confidence": 0.6, "evidence": ["curator.mood"]}]
    assert {d["claim"]["label"] for d in clean["dropped"]} == {"ominous", "eerie"}
    assert all("admissible" in d["reason"] for d in clean["dropped"])


def test_trajectory_and_density_need_segment_evidence():
    no_segments = build_payload(make_track())
    assert "segments" not in no_segments["measured"]
    answer = good_answer()
    answer["energy_trajectory"] = {"value": "stable", "confidence": 0.7, "evidence": ["measured.energy"]}
    answer["sonic_density"] = {"value": "dense", "confidence": 0.7, "evidence": ["measured.energy", "curator.mood"]}
    clean, errors = validate_interpretation(answer, no_segments)
    assert clean["energy_trajectory"] == {"value": "unknown", "confidence": 0.0, "evidence": []}
    assert clean["sonic_density"]["value"] == "unknown"
    assert {d["field"] for d in clean["dropped"]} >= {"energy_trajectory", "sonic_density"}
    # the segment tempo block and the analyzer's notes are not segment ENERGY evidence
    for path in ("measured.segments.tempo.bpm", "measured.segments.notes.0"):
        answer = good_answer()
        answer["energy_trajectory"]["evidence"] = [path]
        answer["roles"][0]["evidence"] = [path]
        clean, _ = validate_interpretation(answer, payload_for())
        assert clean["energy_trajectory"]["value"] == "unknown" and clean["roles"] == [], path
    # 'unknown' asserts nothing and needs no evidence
    answer["energy_trajectory"] = {"value": "unknown", "confidence": 0.3, "evidence": []}
    clean, errors = validate_interpretation(answer, payload_for())
    assert errors == [] and clean["energy_trajectory"] == {"value": "unknown", "confidence": 0.0, "evidence": []}


def test_danceability_needs_audio_or_curator_evidence():
    track = make_track()
    research = {"status": "matched", "facts": {"genre": "dance-pop", "first_release_date": "1982"},
                "source": "musicbrainz"}
    payload = payload_for(track, research=research)
    assert "genre" not in payload["facts"]               # a genre label is no descriptive fact (allowlist)
    answer = good_answer()
    answer["danceability"] = {"value": 0.9, "confidence": 0.8, "evidence": ["facts.genre"]}
    answer["groove"] = {"value": 0.7, "confidence": 0.6, "evidence": ["measured.segments.percussive_density",
                                                                      "facts.first_release_date"]}
    clean, errors = validate_interpretation(answer, payload)
    assert errors == []
    assert clean["danceability"]["value"] is None
    assert clean["groove"]["value"] == 0.7
    assert clean["dropped"][0]["field"] == "danceability"
    answer["danceability"]["value"] = 1.4
    _, errors = validate_interpretation(answer, payload)
    assert any("danceability.value: 1.4 is outside 0..1" in e for e in errors)


def test_roles_and_phases_need_more_than_tempo_genre_or_tags():
    track = make_track()
    track.genre, track.feature_sources["genre"] = "Horror Disco", "manual"
    research = {"status": "matched", "facts": {"first_release_date": "1982-11-30"}, "source": "musicbrainz",
                "tags": [{"name": "halloween", "count": 9}]}
    payload = payload_for(track, research=research, use_community_tags=True)
    answer = good_answer()
    answer["roles"] = [
        {"role": "peak", "confidence": 0.9, "evidence": ["measured.bpm", "curator.genre"]},
        {"role": "opener", "confidence": 0.9, "evidence": ["facts.first_release_date"]},
        {"role": "release", "confidence": 0.6, "evidence": ["measured.key", "measured.duration_s",
                                                            "measured.energy_source"]},
        {"role": "tension_builder", "confidence": 0.7, "evidence": ["measured.bpm", "measured.segments.energy_out"]},
    ]
    answer["positions"] = [{"phase": "climax", "confidence": 0.8, "evidence": ["community_tags.tags.0"]},
                           {"phase": "build", "confidence": 0.6, "evidence": ["curator.mood"]}]
    clean, errors = validate_interpretation(answer, payload)
    assert errors == []
    assert [r["role"] for r in clean["roles"]] == ["tension_builder"]
    assert [p["phase"] for p in clean["positions"]] == ["build"]
    dropped = {d["claim"].get("role") or d["claim"].get("phase"): d["reason"] for d in clean["dropped"]}
    assert set(dropped) == {"peak", "opener", "release", "climax"}
    assert all("not sufficient" in r for r in dropped.values())


def test_mood_labels_normalised_and_duplicates_merged():
    answer = good_answer()
    answer["moods"] = [{"label": "Feel-Good", "confidence": 0.4, "evidence": ["curator.mood"]},
                       {"label": "feel good ", "confidence": 0.7, "evidence": ["measured.segments.energy_mean"]},
                       {"label": "TENSE", "confidence": 0.7, "evidence": ["curator.mood"]}]
    clean, errors = validate_interpretation(answer, payload_for())
    assert errors == []
    assert clean["moods"] == [
        {"label": "feel good", "confidence": 0.7, "evidence": ["curator.mood", "measured.segments.energy_mean"]},
        {"label": "tense", "confidence": 0.7, "evidence": ["curator.mood"]},
    ]


def test_evidence_path_spellings_are_normalised():
    answer = good_answer()
    answer["positions"] = [{"phase": "intro", "confidence": 0.5,
                            "evidence": ["$.measured.segments.sections[0].label", "input.curator.mood",
                                         "measured.segments.sections.1.end"]}]
    clean, errors = validate_interpretation(answer, payload_for())
    assert errors == []
    assert clean["positions"][0]["evidence"] == ["measured.segments.sections.0.label", "curator.mood",
                                                 "measured.segments.sections.1.end"]
    answer["positions"][0]["evidence"] = ["measured.segments.sections[7].label"]      # index out of range
    clean, _ = validate_interpretation(answer, payload_for())
    assert clean["positions"] == []


@pytest.mark.parametrize("data", [None, [], "text", 3, {"mood": "dark"}, {}])
def test_unusable_top_level(data):
    clean, errors = validate_interpretation(data, payload_for())
    assert clean is None and errors


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def research_record(status="matched") -> dict:
    """Shape of ``research.models.ResearchRecord.to_dict()``."""
    mb = {"source": "musicbrainz", "url": "https://musicbrainz.org/recording/1f9c-abc", "license": "CC0",
          "retrieved_at": "2026-10-20T10:00:00+00:00", "fields_used": ["id"]}
    return {
        "track_id": "abc123", "rel_path": "Secret Folder/Client X/Artist 1 - Song 1.mp3",
        "query": {"artist": "Artist 1", "title": "Song 1"}, "status": status, "match_confidence": 0.97,
        "facts": {"recording_mbid": {**mb, "value": "1f9c-abc"}, "isrc": {**mb, "value": ["USSM19902991"]},
                  "artist_mbids": {**mb, "value": ["aa-1", "bb-2"]},
                  "first_release_date": {**mb, "value": "1982-11-30"}, "year": {**mb, "value": 1982},
                  "release_group_type": {**mb, "value": "Album"},
                  "genre_wikidata": {"value": ["disco"], "source": "wikidata", "url": "https://www.wikidata.org/x"},
                  "publication_date_wikidata": {"value": "1983-01-23", "source": "wikidata"},
                  "local_file": {"value": "D:\\Music\\x.mp3", "source": "musicbrainz"}},
        "user_tags": [{"name": "spooky", "count": 3, "kind": "tag", "source": "musicbrainz",
                       "url": "https://musicbrainz.org/tag/spooky", "license": "CC BY-NC-SA 3.0",
                       "non_commercial": True},
                      {"name": "halloween", "count": 9, "kind": "tag", "source": "musicbrainz",
                       "license": "CC BY-NC-SA 3.0", "non_commercial": True},
                      {"name": "Spooky", "count": 1, "kind": "genre", "source": "musicbrainz"}],
        "editorial": [{"field": "style", "value": "Disco", "source": "discogs"}],
        "conflicts": [{"field": "publication_date_wikidata", "values": [{"source": "wikidata", "value": "1983"}]}],
        "candidates": [{"title": "Song 1", "url": "https://musicbrainz.org/recording/zz"}],
        "errors": [], "notes": [], "retrieved_at": "2026-10-20T10:00:00+00:00",
    }


def test_payload_contains_no_paths_ids_or_urls():
    track = make_track(folder="Secret Folder/Client X")
    payload = build_payload(track, segments=SEGMENTS, research=research_record(), use_community_tags=True)
    text = json.dumps(payload)
    for needle in ("Secret Folder", "Client X", "D:", ".mp3", "Music", "1f9c-abc", "USSM19902991", "aa-1",
                   "https://", "abc123", track.rel_path, track.track_id):
        assert needle not in text, needle
    # a Wikidata genre (crowd-edited) is never a verified fact, even in an old-style record
    assert payload["facts"] == {"first_release_date": "1982-11-30", "year": 1982, "release_group_type": "Album",
                                "facts_source": "musicbrainz"}
    seg = payload["measured"]["segments"]
    assert seg["sections"] == [{"label": "intro", "start": 0, "end": 20}, {"label": "peak", "start": 60, "end": 90}]
    assert seg["trajectory"] == {"value": "rising", "confidence": 0.7, "slope_per_min": 0.12, "oscillation": 0.05}
    assert seg["tempo"] == {"bpm": 120, "confidence": 0.8, "half_double_ambiguous": False}
    assert seg["notes"] == ["only the first 600 s were analysed"]
    assert set(seg) == {"energy_in", "energy_out", "energy_mean", "energy_peak", "trajectory", "density",
                        "percussive_density", "bass_density", "dynamic_range_db", "tempo", "vocal_presence_proxy",
                        "sections", "notes"}                     # per-segment curves / timbre are not sent


def test_payload_accepts_a_flat_segment_summary_and_ignores_failed_analysis():
    flat = {"energy_in": 0.2, "energy_out": 0.4, "trajectory": "stable", "density": 0.3,
            "sections": [{"label": "intro", "start": 0.0, "end": 10.0}]}
    seg = build_payload(make_track(), segments=flat)["measured"]["segments"]
    assert seg == {"energy_in": 0.2, "energy_out": 0.4, "trajectory": "stable", "density": 0.3,
                   "sections": [{"label": "intro", "start": 0, "end": 10}]}
    failed = {"version": "segments-v1", "error": "decode failed", "error_kind": "decode", "notes": ["x"]}
    assert "segments" not in build_payload(make_track(), segments=failed)["measured"]


def test_payload_keeps_information_categories_apart():
    track = make_track(manual_energy=True)
    track.danceability, track.feature_sources["danceability"] = 0.8, "manual"
    track.vibe_tags, track.feature_sources["vibe_tags"] = ["dark", "campy"], "manual"
    payload = build_payload(track)
    assert "track" not in payload                              # identity is never sent (interpret-v2)
    measured = payload["measured"]
    assert measured == {"bpm": 120, "bpm_source": "auto:bpm-v1", "key": "8A", "key_source": "auto:key-v1",
                        "key_confidence": 0.7, "duration_s": 201}
    assert "energy" not in measured                            # curator override: not a measurement
    assert payload["curator"] == {"mood": "tense", "energy": 0.6, "danceability": 0.8,
                                  "vibe_tags": ["dark", "campy"]}
    assert "genre" not in json.dumps(payload)                  # an ID3 genre is an unverified file tag
    assert "facts" not in payload and "community_tags" not in payload
    for _, v in walk(payload):
        assert v is not None                                    # unknown = omitted, never null
    auto = build_payload(make_track(mood=None))
    assert auto["measured"]["energy"] == 0.6
    assert auto["measured"]["energy_source"] == "auto:energy-heuristic-v1"
    assert auto["measured"]["energy_confidence"] == 0.5
    assert "curator" not in auto


def test_payload_uses_only_matched_research_and_labels_user_tags():
    track = make_track()
    for status in ("ambiguous", "not_found", "error", "not_cached", "skipped"):
        payload = build_payload(track, research=research_record(status), use_community_tags=True)
        assert "facts" not in payload and "community_tags" not in payload, status
    with_facts = build_payload(track, research=research_record())
    assert with_facts["facts"]["first_release_date"] == "1982-11-30"
    assert "publication_date_wikidata" not in with_facts["facts"]       # sources disagree: not verified
    assert "community_tags" not in with_facts                           # opt-in only
    assert "Disco" not in json.dumps(with_facts)                        # editorial data is not a fact
    tagged = build_payload(track, research=research_record(), use_community_tags=True)
    block = tagged["community_tags"]
    assert block["tags"] == ["halloween", "spooky"] and block["verified"] is False
    assert block["kind"] == "user_tags" and block["source"] == "musicbrainz"
    assert "non-commercial" in block["license_note"] and "CC BY-NC-SA 3.0" in block["license_note"]
    assert "user_tags" not in tagged["facts"]
    simple = build_payload(track, research={"match": {"status": "matched"}, "facts": {"year": 1982},
                                            "source": "musicbrainz", "tags": ["halloween"]},
                           use_community_tags=True)
    assert simple["facts"] == {"year": 1982, "facts_source": "musicbrainz"}
    assert simple["community_tags"]["tags"] == ["halloween"]


def test_payload_accepts_a_research_record_object():
    models = pytest.importorskip("veg_mixer.research.models")
    record = models.ResearchRecord.from_dict(research_record())
    payload = build_payload(make_track(), research=record, use_community_tags=True)
    # compared with the record's own dict: from_dict may migrate legacy fields (research's business)
    assert payload == build_payload(make_track(), research=record.to_dict(), use_community_tags=True)
    assert payload["facts"]["first_release_date"] == "1982-11-30" and payload["community_tags"]["tags"]


def test_payload_is_json_native_and_canonical():
    seg = segments_with(energy_in=np.float32(0.30000001), energy_out=float("nan"), density=np.float64(0.7),
                        dynamic_range_db=np.array([9.5])[0], percussive_density=None)
    seg["sections"] = np.array([])
    payload = build_payload(make_track(), segments=seg)
    json_native(payload)
    s = payload["measured"]["segments"]
    assert s["energy_in"] == 0.3 and s["density"] == 0.7 and s["dynamic_range_db"] == 9.5
    assert "energy_out" not in s and "percussive_density" not in s and "sections" not in s
    a = build_payload(make_track(bpm=120.0001), segments=SEGMENTS)
    b = build_payload(make_track(bpm=120.0), segments=SEGMENTS)
    assert a == b
    assert cache_key(a, provider="fake", model="m") == cache_key(b, provider="fake", model="m")
    assert cache_key(a, provider="fake", model="m") != cache_key(a, provider="fake", model="m2")
    assert cache_key(a, provider="fake", model="m") != cache_key(a, provider="ollama", model="m")
    assert input_digest(a) == input_digest(b)
    c = build_payload(make_track(), segments=segments_with(trajectory={"value": "falling", "confidence": 0.7}))
    assert input_digest(c) != input_digest(a)


# ---------------------------------------------------------------------------
# Interpretation pipeline (fake provider)
# ---------------------------------------------------------------------------

def run(tracks, provider, cache, **kw):
    kw.setdefault("segments", {t.track_id: SEGMENTS for t in tracks})
    return interpret_tracks(tracks, provider=provider, cache=cache, now=lambda: NOW, **kw)


def test_fake_provider_end_to_end(tmp_path):
    tracks = [make_track(1), make_track(2, mood="playful; campy")]
    cache = AICache(tmp_path / AI_CACHE_NAME)
    fake = FakeProvider()
    calls = []
    res = run(tracks, fake, cache, mode="online", progress=lambda i, n, label: calls.append((i, n, label)))
    assert calls == [(1, 2, "Artist 1 - Song 1"), (2, 2, "Artist 2 - Song 2")]
    assert res.stats["requests"] == 2 and res.stats["valid"] == 2 and res.stats["errors"] == 0
    assert set(res.interpretations) == {t.track_id for t in tracks}
    it = res.interpretations[tracks[1].track_id]
    assert it["source"] == "ai:fake/fake-1"
    assert it["prompt_version"] == PROMPT_VERSION and it["created_at"] == "2026-10-22T12:00:00+00:00"
    assert it["from_cache"] is False
    assert [m["label"] for m in it["moods"]] == ["campy", "playful"]
    assert it["energy_trajectory"] == {"value": "rising", "confidence": 0.5,
                                       "evidence": ["measured.segments.trajectory.value"]}
    assert it["sonic_density"] == {"value": "dense", "confidence": 0.4, "evidence": ["measured.segments.density.value"]}
    assert it["dropped"] == []
    json_native(res.interpretations)
    # what the provider saw: the payload only, no paths
    for call in fake.calls:
        assert call["system"] == SYSTEM_PROMPT and call["schema"] == INTERPRETATION_SCHEMA
        assert "Secret Folder" not in call["user"] and ".mp3" not in call["user"]
        assert FakeProvider.payload_of(call["user"])["measured"]["segments"]["trajectory"]["value"] == "rising"
    data = json.loads((tmp_path / AI_CACHE_NAME).read_text(encoding="utf-8"),
                      parse_constant=lambda c: pytest.fail(f"non-finite {c} in cache"))
    assert data["schema"] == "vegroove.ai_cache" and len(data["entries"]) == 2


def test_second_run_is_served_from_cache_without_provider_calls(tmp_path):
    tracks = [make_track(1), make_track(2)]
    path = tmp_path / AI_CACHE_NAME
    first = run(tracks, FakeProvider(), AICache(path), mode="online")
    fake = FakeProvider()
    second = run(tracks, fake, AICache(path), mode="online")
    assert fake.calls == []
    assert second.stats["cache_hits"] == 2 and second.stats["requests"] == 0 and second.stats["valid"] == 2
    for tid, it in second.interpretations.items():
        assert it["from_cache"] is True
        assert {k: v for k, v in it.items() if k != "from_cache"} == \
            {k: v for k, v in first.interpretations[tid].items() if k != "from_cache"}


def test_cache_mode_never_calls_the_provider(tmp_path):
    tracks = [make_track(1), make_track(2)]
    path = tmp_path / AI_CACHE_NAME
    fake = FakeProvider()
    res = run(tracks, fake, AICache(path), mode="cache")
    assert fake.calls == [] and res.interpretations == {}
    assert res.stats["skipped"] == 2 and res.stats["requests"] == 0
    assert any("no cached interpretation" in w for w in res.warnings)
    assert not path.exists()
    # after an online run, cache mode WITHOUT a provider finds the answers by input alone
    run(tracks[:1], FakeProvider(model="model-a"), AICache(path), mode="online")
    res = run(tracks, None, AICache(path), mode="cache")
    assert set(res.interpretations) == {tracks[0].track_id}
    assert res.interpretations[tracks[0].track_id]["source"] == "ai:fake/model-a"
    assert res.stats["cache_hits"] == 1 and res.stats["skipped"] == 1
    # ... but with a provider, only that provider/model's answers count
    res = run(tracks, FakeProvider(model="model-b"), AICache(path), mode="cache")
    assert res.interpretations == {}


def test_invalid_json_gets_one_reask_then_a_negative_cache_entry(tmp_path):
    track = make_track()
    path = tmp_path / AI_CACHE_NAME
    fake = FakeProvider(["this is not json", '{"moods": "nope"}'])
    res = run([track], fake, AICache(path), mode="online")
    assert len(fake.calls) == 2
    assert "REJECTED" not in fake.calls[0]["user"]
    assert "REJECTED" in fake.calls[1]["user"] and "not valid JSON" in fake.calls[1]["user"]
    assert res.stats == {"cache_hits": 0, "requests": 2, "valid": 0, "invalid": 1, "errors": 0, "skipped": 0,
                         "reasks": 1, "dropped_claims": 0}
    assert res.interpretations == {}
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    (entry,) = entries.values()
    assert entry["valid"] is False and entry["errors"]
    # the negative result is not paid for again ...
    again = FakeProvider()
    res = run([track], again, AICache(path), mode="online")
    assert again.calls == [] and res.stats["invalid"] == 1 and res.stats["cache_hits"] == 1
    assert any("invalid" in w for w in res.warnings)
    # ... unless explicitly retried
    res = run([track], again, AICache(path), mode="online", retry_invalid=True)
    assert len(again.calls) == 1 and res.stats["valid"] == 1
    assert json.loads(path.read_text(encoding="utf-8"))["entries"][next(iter(entries))]["valid"] is True


def test_reask_quotes_the_validator_errors_and_can_recover(tmp_path):
    bad = good_answer()
    bad["moods"][0]["confidence"] = 7
    fake = FakeProvider([bad, good_answer()])
    track = make_track()
    res = run([track], fake, AICache(tmp_path / AI_CACHE_NAME), mode="online")
    assert res.stats["valid"] == 1 and res.stats["requests"] == 2 and res.stats["reasks"] == 1
    assert "moods[0].confidence: 7 is outside 0..1" in fake.calls[1]["user"]
    assert FakeProvider.payload_of(fake.calls[1]["user"]) == FakeProvider.payload_of(fake.calls[0]["user"])
    assert res.interpretations[track.track_id]["moods"][0]["label"] == "tense"


def test_dropped_claims_do_not_trigger_a_reask(tmp_path):
    answer = good_answer()
    answer["roles"] = [{"role": "peak", "confidence": 0.9, "evidence": ["measured.bpm"]}]
    fake = FakeProvider([answer])
    track = make_track()
    res = run([track], fake, AICache(tmp_path / AI_CACHE_NAME), mode="online")
    assert len(fake.calls) == 1 and res.stats["valid"] == 1 and res.stats["dropped_claims"] == 1
    it = res.interpretations[track.track_id]
    assert it["roles"] == [] and it["dropped"][0]["claim"]["role"] == "peak"


def connection_error(*_):
    return AIResult(None, "", "fake", "fake-1", error="connection: server down", retryable=True,
                    error_kind="connection")


def test_circuit_breaker_after_consecutive_failures(tmp_path):
    tracks = [make_track(i) for i in range(1, 9)]
    path = tmp_path / AI_CACHE_NAME
    fake = FakeProvider(connection_error)
    res = run(tracks, fake, AICache(path), mode="online", max_consecutive_failures=3)
    assert len(fake.calls) == 3
    assert res.stats["errors"] == 3 and res.stats["skipped"] == 5 and res.stats["requests"] == 3
    assert any("stopped calling fake/fake-1 after 3 consecutive provider errors; 5 remaining" in w
               for w in res.warnings)
    assert not path.exists()                                    # provider errors are never cached


def test_breaker_counts_only_consecutive_failures(tmp_path):
    answers = [connection_error(), connection_error(), good_answer(), connection_error(), connection_error(),
               good_answer()]
    fake = FakeProvider(answers)
    tracks = [make_track(i) for i in range(1, 7)]
    res = run(tracks, fake, AICache(tmp_path / AI_CACHE_NAME), mode="online", max_consecutive_failures=3)
    assert len(fake.calls) == 6 and res.stats["errors"] == 4 and res.stats["valid"] == 2


def test_fatal_provider_error_stops_at_once(tmp_path):
    def auth(*_):
        return AIResult(None, "", "fake", "fake-1", error="auth: invalid key", error_kind="auth")
    fake = FakeProvider(auth)
    res = run([make_track(i) for i in range(1, 5)], fake, AICache(tmp_path / AI_CACHE_NAME), mode="online")
    assert len(fake.calls) == 1 and res.stats["errors"] == 1 and res.stats["skipped"] == 3
    assert any("non-recoverable provider error (auth)" in w for w in res.warnings)


def test_a_raising_provider_never_aborts_the_run(tmp_path):
    fake = FakeProvider([RuntimeError("boom"), ValueError("bad"), OSError("disk"), good_answer()])
    tracks = [make_track(i) for i in range(1, 5)]
    res = run(tracks, fake, AICache(tmp_path / AI_CACHE_NAME), mode="online")
    assert res.stats["errors"] == 3 and res.stats["valid"] == 1
    assert any("provider raised RuntimeError: boom" in w for w in res.warnings)


def test_tracks_with_only_tempo_key_duration_are_not_sent(tmp_path):
    track = make_track(mood=None, energy=None)
    fake = FakeProvider()
    res = run([track], fake, AICache(tmp_path / AI_CACHE_NAME), mode="online", segments={})
    assert fake.calls == [] and res.stats["skipped"] == 1


def test_modes_and_configuration_errors(tmp_path):
    cache = AICache(tmp_path / AI_CACHE_NAME)
    fake = FakeProvider()
    res = run([make_track()], fake, cache, mode="off")
    assert res.interpretations == {} and fake.calls == [] and not any(res.stats.values())
    with pytest.raises(AIConfigError, match="unknown AI mode"):
        run([make_track()], fake, cache, mode="sometimes")
    with pytest.raises(AIConfigError, match="needs a provider"):
        run([make_track()], None, cache, mode="online")
    assert issubclass(AIConfigError, ValueError)


def test_interpretation_never_touches_track_features(tmp_path):
    tracks = [make_track(1), make_track(2)]
    before = [copy.deepcopy(t.to_dict()) for t in tracks]
    run(tracks, FakeProvider(), AICache(tmp_path / AI_CACHE_NAME), mode="online")
    assert [t.to_dict() for t in tracks] == before


def test_read_only_cache_is_never_written(tmp_path):
    path = tmp_path / AI_CACHE_NAME
    res = run([make_track()], FakeProvider(), AICache(path, read_only=True), mode="online")
    assert res.stats["valid"] == 1 and not path.exists()


def test_cache_saves_are_throttled_but_nothing_paid_for_is_lost(tmp_path, monkeypatch):
    monkeypatch.setattr(interpret_mod, "SAVE_EVERY_N", 2)
    monkeypatch.setattr(interpret_mod, "SAVE_EVERY_S", 1e9)
    path = tmp_path / AI_CACHE_NAME
    cache = AICache(path)
    writes = []
    real_save = cache.save
    monkeypatch.setattr(cache, "save", lambda: (writes.append(cache.dirty), real_save()))
    run([make_track(i) for i in range(1, 6)], FakeProvider(), cache, mode="online")
    assert writes == [True, True, True]                         # after answers 2 and 4, then the final one
    assert len(json.loads(path.read_text(encoding="utf-8"))["entries"]) == 5

    # Ctrl+C in the middle of a run: the answers received so far are saved
    path2 = tmp_path / "interrupted.json"

    def progress(i, n, label):
        if i == 3:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run([make_track(i) for i in range(1, 6)], FakeProvider(), AICache(path2), mode="online", progress=progress)
    assert len(json.loads(path2.read_text(encoding="utf-8"))["entries"]) == 2


def test_unsaveable_cache_degrades_to_a_warning(tmp_path, monkeypatch):
    cache = AICache(tmp_path / AI_CACHE_NAME)

    def broken_save():
        raise PermissionError("read-only medium")
    monkeypatch.setattr(cache, "save", broken_save)
    res = run([make_track()], FakeProvider(), cache, mode="online")
    assert res.stats["valid"] == 1
    assert any("could not be saved" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# AI cache file
# ---------------------------------------------------------------------------

def clean_output() -> dict:
    """A validated interpretation as it is stored in the cache (``dropped`` kept apart)."""
    clean, errors = validate_interpretation(good_answer(), payload_for())
    assert errors == []
    return {k: v for k, v in clean.items() if k != "dropped"}


def entry(valid=True):
    return {"output": clean_output() if valid else None, "valid": valid, "errors": [] if valid else ["x"],
            "dropped": [], "provider": "fake", "model": "m", "request_id": None, "usage": {},
            "created_at": "2026-10-22T12:00:00+00:00", "input_digest": "a" * 64}


def test_cache_roundtrip_and_noop_save(tmp_path):
    path = tmp_path / "sub" / AI_CACHE_NAME
    c = AICache(path)
    c.put("b" * 64, entry())
    c.save()
    mtime = path.stat().st_mtime_ns
    c.save()                                                    # unchanged: no write
    assert path.stat().st_mtime_ns == mtime
    c2 = AICache(path)
    assert len(c2) == 1 and c2.get("b" * 64) == entry() and c2.warnings == []
    got = c2.get("b" * 64)
    got["valid"] = False                                        # a private copy
    assert c2.get("b" * 64)["valid"] is True
    with pytest.raises(ValueError):
        c2.put("not-a-key", entry())
    with pytest.raises(ValueError):
        c2.put("c" * 64, {**entry(), "valid": "yes"})
    with pytest.raises(ValueError):
        c2.put("c" * 64, {**entry(), "usage": {"x": float("nan")}})


def test_cache_find_prefers_valid_then_newest(tmp_path):
    c = AICache(tmp_path / AI_CACHE_NAME)
    c.put("1" * 64, {**entry(), "created_at": "2026-01-01T00:00:00+00:00", "model": "old"})
    c.put("2" * 64, {**entry(), "created_at": "2026-05-01T00:00:00+00:00", "model": "new"})
    c.put("3" * 64, {**entry(valid=False), "created_at": "2026-09-01T00:00:00+00:00"})
    key, found = c.find("a" * 64)
    assert key == "2" * 64 and found["model"] == "new"
    assert c.find("f" * 64) is None


def test_damaged_own_cache_is_moved_aside(tmp_path):
    path = tmp_path / AI_CACHE_NAME
    path.write_text('{"schema": "vegroove.ai_cache", "schema_version": 1, "entr', encoding="utf-8")
    c = AICache(path)
    assert len(c) == 0 and not path.exists()
    assert (tmp_path / f"{AI_CACHE_NAME}.corrupt-1").exists()
    assert any("kept as" in w for w in c.warnings)
    c.put("b" * 64, entry())
    c.save()
    assert AICache(path).get("b" * 64) is not None


def test_damaged_cache_in_read_only_mode_is_left_in_place(tmp_path):
    path = tmp_path / AI_CACHE_NAME
    path.write_text('{"schema": "vegroove.ai_cache", "schema_version": 99, "entries": {}}', encoding="utf-8")
    c = AICache(path, read_only=True)
    assert path.exists() and not (tmp_path / f"{AI_CACHE_NAME}.corrupt-1").exists()
    assert any("would be moved aside" in w for w in c.warnings)


@pytest.mark.parametrize("content", ['{"some": "other program"}', '[1, 2]',
                                     '{"schema": "vegroove.analysis_cache", "schema_version": 1, "entries": {}}',
                                     'not json at all'])
def test_foreign_file_is_never_overwritten(tmp_path, content):
    path = tmp_path / AI_CACHE_NAME
    path.write_text(content, encoding="utf-8")
    c = AICache(path)
    assert any("not a VEgroove AI cache" in w for w in c.warnings)
    c.put("b" * 64, entry())
    c.save()
    assert path.read_text(encoding="utf-8") == content
    assert not list(tmp_path.glob("*.corrupt-*"))


def test_malformed_entries_are_dropped_individually(tmp_path):
    path = tmp_path / AI_CACHE_NAME
    good = entry()
    path.write_text(json.dumps({"schema": "vegroove.ai_cache", "schema_version": 1, "entries": {
        "b" * 64: good, "c" * 64: {**good, "valid": "maybe"}, "short": good,
        "d" * 64: {**good, "output": None},
        "e" * 64: {**good, "output": {"moods": "not a list"}}}}), encoding="utf-8")   # served as is: checked
    c = AICache(path)
    assert len(c) == 1 and c.get("b" * 64) == good
    assert len([w for w in c.warnings if "malformed entry" in w]) == 4
    assert any("interpretation schema" in w for w in c.warnings)
    assert c.dirty


@pytest.mark.parametrize("name", ["track.mp3", "track.FLAC", "notes.txt", "cache"])
def test_cache_refuses_paths_that_cannot_be_ours(tmp_path, name):
    with pytest.raises(ValueError):
        AICache(tmp_path / name)


# ---------------------------------------------------------------------------
# Anthropic provider (fake SDK client; no network, no key)
# ---------------------------------------------------------------------------

class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


class FakeAnthropicClient:
    def __init__(self, *responses):
        self.beta = SimpleNamespace(messages=FakeMessages(responses))


def sdk_message(text=None, *, stop="end_turn", blocks=None, stop_details=None, model="claude-opus-5-5"):
    content = blocks if blocks is not None else [SimpleNamespace(type="text", text=text)]
    usage = SimpleNamespace(input_tokens=1200, output_tokens=300, cache_read_input_tokens=0,
                            cache_creation_input_tokens=None)
    return SimpleNamespace(content=content, stop_reason=stop, stop_details=stop_details, model=model, usage=usage,
                           _request_id="req_test123")


def anthropic_provider(*responses, env=None, **kw):
    client = FakeAnthropicClient(*responses)
    return AnthropicProvider(client=client, env=env or {}, **kw), client.beta.messages


def test_anthropic_request_kwargs_are_exact():
    p, messages = anthropic_provider(sdk_message(json.dumps(good_answer())))
    res = p.complete_json(system="SYS", user='{"measured":{}}', schema=INTERPRETATION_SCHEMA, max_tokens=4000)
    (kwargs,) = messages.calls
    assert kwargs == {
        "model": "claude-opus-5-5",
        "max_tokens": 4000,
        "system": "SYS",
        "messages": [{"role": "user", "content": '{"measured":{}}'}],
        "output_config": {"format": {"type": "json_schema", "schema": INTERPRETATION_SCHEMA}, "effort": "medium"},
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
    }
    assert FALLBACK_BETA == "server-side-fallback-2026-07-01"
    for banned in ("temperature", "top_p", "top_k", "thinking", "tool_choice", "tools"):
        assert banned not in kwargs
    assert res.ok and res.data == good_answer()
    assert res.provider == "anthropic" and res.model == "claude-opus-5-5" and res.request_id == "req_test123"
    assert res.usage == {"input_tokens": 1200, "output_tokens": 300, "cache_read_input_tokens": 0}
    assert p.timeout_s == 120.0 and p.max_retries == 2


def test_anthropic_settings_from_arguments_and_environment():
    p, messages = anthropic_provider(sdk_message("{}"), env={"VEG_AI_MODEL": "claude-sonnet-5-5",
                                                             "VEG_AI_EFFORT": "high", "VEG_AI_FALLBACKS": "off",
                                                             "VEG_AI_TIMEOUT_S": "30"})
    p.complete_json(system="S", user="U", schema={"type": "object"})
    kwargs = messages.calls[0]
    assert kwargs["model"] == "claude-sonnet-5-5" and kwargs["output_config"]["effort"] == "high"
    assert "betas" not in kwargs and "fallbacks" not in kwargs and kwargs["max_tokens"] == 4000
    assert p.timeout_s == 30.0
    p, _ = anthropic_provider(env={"VEG_AI_MODEL": "x"}, model="claude-opus-5-5")
    assert p.model == "claude-opus-5-5"
    with pytest.raises(AIConfigError, match="effort"):
        anthropic_provider(env={"VEG_AI_EFFORT": "extreme"})
    with pytest.raises(AIConfigError, match="VEG_AI_FALLBACKS"):
        anthropic_provider(env={"VEG_AI_FALLBACKS": "sometimes"})
    with pytest.raises(AIConfigError, match="timeout"):
        anthropic_provider(env={"VEG_AI_TIMEOUT_S": "soon"})


def test_anthropic_stop_reason_is_checked_before_content():
    refusal = sdk_message(json.dumps(good_answer()), stop="refusal",
                          stop_details=SimpleNamespace(type="refusal", category="cyber", explanation="x"))
    truncated = sdk_message('{"moods": [', stop="max_tokens")
    p, _ = anthropic_provider(refusal, truncated)
    res = p.complete_json(system="S", user="U", schema={})
    assert res.data is None and res.error_kind == "refused" and "cyber" in res.error
    assert res.stop_reason == "refusal" and res.request_id == "req_test123"
    res = p.complete_json(system="S", user="U", schema={})
    assert res.data is None and res.error_kind == "incomplete" and "max_tokens" in res.error


def test_anthropic_reads_the_first_text_block_and_reports_the_serving_model():
    blocks = [SimpleNamespace(type="thinking", thinking=""),
              SimpleNamespace(type="fallback"),
              SimpleNamespace(type="text", text="```json\n" + json.dumps(good_answer()) + "\n```")]
    p, _ = anthropic_provider(sdk_message(blocks=blocks, model="claude-opus-5"),
                              sdk_message(blocks=[SimpleNamespace(type="thinking", thinking="")]),
                              sdk_message("I cannot produce JSON"))
    res = p.complete_json(system="S", user="U", schema={})
    assert res.ok and res.data == good_answer() and res.model == "claude-opus-5"
    res = p.complete_json(system="S", user="U", schema={})
    assert res.error_kind == "bad_response"
    res = p.complete_json(system="S", user="U", schema={})
    assert res.error_kind == "invalid_output" and res.raw_text == "I cannot produce JSON"


def test_anthropic_sdk_exceptions_are_mapped():
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def status(cls, code):
        return cls("api says no", response=httpx.Response(code, request=req, headers={"request-id": "req_err"}),
                   body={"error": {"type": "x", "message": "api says no"}})

    cases = [
        (status(anthropic.AuthenticationError, 401), "auth", False),
        (status(anthropic.PermissionDeniedError, 403), "permission", False),
        (status(anthropic.NotFoundError, 404), "not_found", False),
        (status(anthropic.BadRequestError, 400), "bad_request", False),
        (status(anthropic.RateLimitError, 429), "rate_limit", True),
        (status(anthropic.InternalServerError, 500), "server", True),
        (status(anthropic.APIStatusError, 529), "server", True),
        (status(anthropic.UnprocessableEntityError, 422), "http", False),
        (anthropic.APITimeoutError(request=req), "timeout", True),
        (anthropic.APIConnectionError(request=req), "connection", True),
        (TypeError('"Could not resolve authentication method. Expected one of api_key, auth_token"'), "auth", False),
        (TypeError("create() got an unexpected keyword argument 'fallbacks'"), "config", False),
        (anthropic.AnthropicError("something else"), "sdk", False),
    ]
    p, _ = anthropic_provider(*[c[0] for c in cases])
    for exc, kind, retryable in cases:
        res = p.complete_json(system="S", user="U", schema={})
        assert (res.error_kind, res.retryable) == (kind, retryable), exc
        assert res.data is None and res.error.startswith(f"{kind}: ")
        if isinstance(exc, anthropic.APIStatusError):
            assert res.request_id == "req_err"
    assert "ANTHROPIC_API_KEY" in p.complete_json.__self__._error(cases[10][0]).error


def test_anthropic_without_the_sdk_gives_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)          # import anthropic -> ImportError
    with pytest.raises(AIUnavailableError, match="requirements-ai.txt"):
        AnthropicProvider(env={})
    with pytest.raises(AIUnavailableError):
        get_provider("anthropic", env={})
    # an injected client still works without the package
    p = AnthropicProvider(client=FakeAnthropicClient(sdk_message("{}")), env={})
    assert p.complete_json(system="S", user="U", schema={}).data == {}


def test_anthropic_real_client_is_created_without_network(monkeypatch):
    pytest.importorskip("anthropic")
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_PROFILE",
                 "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)      # a developer's own profile must not matter here
    p = AnthropicProvider(env={}, timeout_s=45, max_retries=4)
    assert p._client.max_retries == 4
    assert p.model == "claude-opus-5-5" and p.effort == "medium" and p.fallbacks is True


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (fake HTTP transport)
# ---------------------------------------------------------------------------

def chat_reply(content, *, finish="stop", refusal=None, model="llama3.1"):
    return {"id": "chatcmpl-1", "model": model,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content, "refusal": refusal}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 250, "total_tokens": 1150}}


def fake_http(*responses):
    sent: list[dict] = []
    queue = list(responses)

    def transport(method, url, headers, body, timeout):
        sent.append({"method": method, "url": url, "headers": dict(headers),
                     "body": json.loads(body.decode("utf-8")) if body else None, "timeout": timeout})
        r = queue.pop(0)
        if isinstance(r, BaseException):
            raise r
        code, payload = r
        return HttpResponse(code, {"x-request-id": "rid-7", "content-type": "application/json"},
                            json.dumps(payload).encode("utf-8"), url)

    return HttpClient(transport=transport, sleep=lambda s: None, max_retries=0), sent


def openai_provider(*responses, env=None, **kw):
    http, sent = fake_http(*responses)
    env = {"VEG_AI_MODEL": "llama3.1", "VEG_AI_API_KEY": SECRET} if env is None else env
    return OpenAICompatibleProvider(http=http, env=env, **{"alias": "ollama", **kw}), sent


def test_openai_compatible_request_and_redacted_logging(caplog):
    caplog.set_level(logging.DEBUG)
    p, sent = openai_provider((200, chat_reply(json.dumps(good_answer()))))
    res = p.complete_json(system="SYS", user="USER", schema=INTERPRETATION_SCHEMA, max_tokens=3000)
    (req,) = sent
    assert req["method"] == "POST" and req["url"] == "http://localhost:11434/v1/chat/completions"
    assert req["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert req["headers"]["Accept-Encoding"] == "identity"        # error bodies stay scrubbable
    body = req["body"]
    assert body["model"] == "llama3.1" and body["max_tokens"] == 3000
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert body["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "vemix_track_interpretation", "strict": True, "schema": INTERPRETATION_SCHEMA}}
    assert body["temperature"] == 0 and body["seed"] == 0
    assert res.ok and res.data == good_answer() and res.request_id == "rid-7" and res.model == "llama3.1"
    assert res.usage == {"input_tokens": 900, "output_tokens": 250} and res.provider == "ollama"
    assert SECRET not in caplog.text
    assert "<redacted>" in caplog.text


def test_openai_compatible_falls_back_to_json_object_after_a_400():
    rejection = (400, {"error": {"message": "response_format json_schema is not supported by this server"}})
    p, sent = openai_provider(rejection, (200, chat_reply(json.dumps(good_answer()))),
                              (200, chat_reply(json.dumps(good_answer()))))
    res = p.complete_json(system="SYS", user="USER", schema=INTERPRETATION_SCHEMA)
    assert res.ok
    assert sent[0]["body"]["response_format"]["type"] == "json_schema"
    assert sent[1]["body"]["response_format"] == {"type": "json_object"}
    system = sent[1]["body"]["messages"][0]["content"]
    assert system.startswith("SYS") and "JSON schema" in system and '"risky_pairings"' in system
    p.complete_json(system="SYS", user="USER", schema=INTERPRETATION_SCHEMA)   # remembered: no 400 round trip
    assert len(sent) == 3 and sent[2]["body"]["response_format"] == {"type": "json_object"}
    # any other 400 is an error, not a format fallback
    p, sent = openai_provider((400, {"error": {"message": "context length exceeded"}}))
    res = p.complete_json(system="S", user="U", schema={})
    assert res.error_kind == "bad_request" and len(sent) == 1


def test_openai_compatible_answer_problems():
    p, _ = openai_provider((200, chat_reply('{"moods": [', finish="length")),
                           (200, chat_reply(None, refusal="I can't help with that")),
                           (200, chat_reply("not json")),
                           (200, {"choices": []}),
                           (200, ["not", "an", "object"]))
    kinds = [p.complete_json(system="S", user="U", schema={}).error_kind for _ in range(5)]
    assert kinds == ["incomplete", "refused", "invalid_output", "bad_response", "bad_response"]


def test_openai_compatible_network_errors_and_key_scrubbing(caplog):
    caplog.set_level(logging.DEBUG)
    p, _ = openai_provider(NetTimeout("timed out after 120 s", retryable=True),
                           (401, {"error": {"message": f"Incorrect API key provided: {SECRET}"}}),
                           (404, {"error": {"message": "model 'llama3.1' not found"}}),
                           (503, {"error": {"message": "overloaded"}}),
                           (429, {"error": {"message": "slow down"}}))
    results = [p.complete_json(system="S", user="U", schema={}) for _ in range(5)]
    assert [(r.error_kind, r.retryable) for r in results] == [
        ("timeout", True), ("auth", False), ("not_found", False), ("server", True), ("rate_limit", True)]
    assert SECRET not in results[1].error and "<redacted>" in results[1].error
    assert SECRET not in caplog.text


def test_openai_compatible_configuration():
    with pytest.raises(AIConfigError, match="VEG_AI_MODEL"):
        OpenAICompatibleProvider(alias="ollama", env={})
    with pytest.raises(AIConfigError, match="VEG_AI_BASE_URL"):
        OpenAICompatibleProvider("m", env={})                   # plain openai-compatible: no default URL
    with pytest.raises(AIConfigError, match="http"):
        OpenAICompatibleProvider("m", base_url="ftp://host/v1", env={})
    base = lambda alias, env, url=None: OpenAICompatibleProvider("m", alias=alias, base_url=url, env=env).base_url
    assert base("ollama", {}) == "http://localhost:11434/v1"
    assert base("ollama", {"OLLAMA_API_BASE": "http://gpu-box:11434"}) == "http://gpu-box:11434/v1"
    assert base("ollama", {"OLLAMA_API_BASE": "http://gpu-box:11434/v1/"}) == "http://gpu-box:11434/v1"
    assert base("lmstudio", {}) == "http://localhost:1234/v1"
    assert base("llamacpp", {}) == "http://localhost:8080/v1"
    assert base("lmstudio", {"VEG_AI_BASE_URL": "http://h:9/v1"}) == "http://h:9/v1"
    assert base("ollama", {"VEG_AI_BASE_URL": "http://h:9/v1"}, "http://arg:1/v1/") == "http://arg:1/v1"
    no_key = OpenAICompatibleProvider("m", alias="ollama", env={})
    assert "Authorization" not in no_key._headers()
    with_key = OpenAICompatibleProvider("m", alias="ollama", env={"VEG_AI_API_KEY": SECRET})
    assert with_key._headers()["Authorization"] == f"Bearer {SECRET}"


@pytest.mark.parametrize("alias, base_url, sent", [
    ("ollama", None, False),                                       # local server
    ("ollama", "http://gpu-box:11434/v1", False),                  # LAN box over plain HTTP
    ("openai-compatible", "https://api.together.example/v1", False),   # a third-party cloud
    ("openai-compatible", "https://api.openai.com/v1", True),      # OpenAI's own API
])
def test_openai_api_key_is_only_sent_to_openai(alias, base_url, sent):
    """OPENAI_API_KEY is often set globally for other tools: it must never be handed to a local,
    LAN or third-party server as a Bearer token (VEG_AI_API_KEY is the key for those)."""
    p = OpenAICompatibleProvider("m", alias=alias, base_url=base_url, env={"OPENAI_API_KEY": SECRET})
    assert ("Authorization" in p._headers()) is sent


def test_openai_compatible_end_to_end_keeps_the_key_out_of_the_cache(tmp_path):
    p, sent = openai_provider((200, chat_reply(json.dumps(good_answer()))))
    track = make_track()
    res = run([track], p, AICache(tmp_path / AI_CACHE_NAME), mode="online")
    assert res.stats["valid"] == 1 and len(sent) == 1
    assert res.interpretations[track.track_id]["source"] == "ai:ollama/llama3.1"
    text = (tmp_path / AI_CACHE_NAME).read_text(encoding="utf-8")
    assert SECRET not in text and "Secret Folder" not in text
    assert "Secret Folder" not in json.dumps(sent[0]["body"])


# ---------------------------------------------------------------------------
# Registry / helpers
# ---------------------------------------------------------------------------

def test_get_provider_registry():
    with pytest.raises(AIConfigError, match="no AI provider selected"):
        get_provider(env={})
    with pytest.raises(AIConfigError, match="unknown AI provider 'gpt'.*anthropic.*openai-compatible.*fake"):
        get_provider("gpt", env={})
    assert isinstance(get_provider("fake"), FakeProvider)
    assert get_provider(None, env={"VEG_AI_PROVIDER": "fake"}).name == "fake"
    assert get_provider("Fake", model="demo", max_retries=3).model == "demo"
    p = get_provider("LM-Studio", model="qwen", env={})
    assert isinstance(p, OpenAICompatibleProvider) and p.name == "lmstudio"
    assert get_provider("llama.cpp", model="q", env={}).base_url == "http://localhost:8080/v1"
    a = get_provider("claude", env={}, client=FakeAnthropicClient(), effort=None, http=None)
    assert isinstance(a, AnthropicProvider) and a.model == "claude-opus-5-5"
    with pytest.raises(AIConfigError, match="does not apply"):
        get_provider("ollama", model="q", env={}, effort="high")
    with pytest.raises(AIConfigError, match="VEG_AI_MODEL"):
        get_provider("ollama", env={})


@pytest.mark.parametrize("text, expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Here you go: {"a": 1} hope it helps', {"a": 1}),
    ("\ufeff  [1]", [1]),
])
def test_parse_json_text_accepts(text, expected):
    assert parse_json_text(text) == (expected, None)


@pytest.mark.parametrize("text", ["", "   ", "nope", '{"a": NaN}', '{"a": Infinity}', '{"a": 1'])
def test_parse_json_text_rejects(text):
    data, err = parse_json_text(text)
    assert data is None and err


def test_cache_key_covers_prompt_and_schema(monkeypatch):
    payload = payload_for()
    key = cache_key(payload, provider="fake", model="m")
    monkeypatch.setattr(interpret_mod, "PROMPT_DIGEST", "0" * 16)
    assert cache_key(payload, provider="fake", model="m") != key


# ---------------------------------------------------------------------------
# Review regressions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["measured.segments.sections.²", "measured.segments.sections[²].label",
                                  "measured.segments.sections.٣", "curator.mood.¹"])
def test_unicode_digit_evidence_never_raises(path):
    """'²'.isdigit() is True but int('²') raises: model output must not crash the validator."""
    answer = good_answer()
    answer["moods"][0]["evidence"] = [path]
    clean, errors = validate_interpretation(answer, payload_for())
    assert errors == [] and clean["moods"] == []
    assert "not a key path" in clean["dropped"][0]["reason"]


def test_deeply_nested_answers_never_raise(tmp_path):
    data, err = parse_json_text("[" * 100_000)
    assert data is None and "nested too deeply" in err
    data, err = parse_json_text('{"moods": ' + "[" * 100_000)
    assert data is None and err
    deep: list = []
    node = deep
    for _ in range(sys.getrecursionlimit() - 60):
        node.append([])
        node = node[0]
    answer = good_answer()
    answer["danceability"]["value"] = deep
    clean, errors = validate_interpretation(answer, payload_for())
    assert errors                                               # invalid, but no RecursionError
    # end to end: the provider's unparsable answer is an invalid answer, not a provider crash
    fake = FakeProvider(["[" * 100_000, "[" * 100_000])
    res = run([make_track()], fake, AICache(tmp_path / AI_CACHE_NAME), mode="online")
    assert res.stats["invalid"] == 1 and res.stats["errors"] == 0 and len(fake.calls) == 2


def test_a_crashing_validator_degrades_to_an_invalid_answer(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise ValueError("validator bug")
    monkeypatch.setattr(interpret_mod, "validate_interpretation", boom)
    res = run([make_track()], FakeProvider(), AICache(tmp_path / AI_CACHE_NAME), mode="online")
    assert res.stats["invalid"] == 1 and res.interpretations == {}


def test_curator_tempo_or_key_alone_is_not_a_dramaturgical_basis():
    track = make_track()
    track.bpm, track.feature_sources["bpm"] = 128.0, "manual"
    track.camelot, track.feature_sources["camelot"] = "9A", "manual"
    payload = payload_for(track)
    assert payload["curator"]["bpm"] == 128 and payload["curator"]["key"] == "9A"
    answer = good_answer()
    answer["roles"] = [{"role": "peak", "confidence": 0.9, "evidence": ["curator.bpm"]},
                       {"role": "opener", "confidence": 0.9, "evidence": ["curator.key", "measured.duration_s"]},
                       {"role": "breather", "confidence": 0.5, "evidence": ["curator.bpm", "curator.mood"]}]
    clean, errors = validate_interpretation(answer, payload)
    assert errors == []
    assert [r["role"] for r in clean["roles"]] == ["breather"]
    assert {d["claim"]["role"] for d in clean["dropped"]} == {"peak", "opener"}


def test_provenance_fields_alone_are_not_evidence():
    seg = segments_with(trajectory={"value": None, "confidence": 0.0, "slope_per_min": None, "oscillation": None})
    payload = payload_for(segments=seg)
    # an undetermined block is not sent at all: nothing citable stands behind it
    assert "trajectory" not in payload["measured"]["segments"]
    answer = good_answer()
    answer["energy_trajectory"] = {"value": "rising", "confidence": 0.8,
                                   "evidence": ["measured.segments.density.confidence"]}
    answer["moods"] = [{"label": "tense", "confidence": 0.6, "evidence": ["measured.energy_source"]},
                       {"label": "dark", "confidence": 0.6,
                        "evidence": ["measured.segments.vocal_presence_proxy.method"]},
                       {"label": "eerie", "confidence": 0.5,
                        "evidence": ["measured.energy_confidence", "measured.energy"]}]
    answer["sonic_density"] = {"value": "dense", "confidence": 0.6,
                               "evidence": ["measured.segments.density.confidence"]}
    clean, errors = validate_interpretation(answer, payload)
    assert errors == []
    assert clean["energy_trajectory"]["value"] == "unknown"
    assert clean["sonic_density"]["value"] == "unknown"
    assert [m["label"] for m in clean["moods"]] == ["eerie"]          # provenance + the value itself: kept
    assert clean["moods"][0]["evidence"] == ["measured.energy_confidence", "measured.energy"]
    reasons = [d["reason"] for d in clean["dropped"]]
    assert sum("provenance" in r for r in reasons) == 4


def test_identity_facts_and_wikidata_items_are_not_sent():
    record = research_record()
    record["facts"].update({
        "recording_title": {"value": "Song 1", "source": "musicbrainz"},
        "artist_credit": {"value": "Artist 1", "source": "musicbrainz"},
        "release_title": {"value": "Album One", "source": "musicbrainz"},
        "release_group": {"value": "Album One", "source": "musicbrainz"},
        "work_title": {"value": "Song 1", "source": "musicbrainz"},
        "wikidata_item": {"value": "Q4567", "source": "wikidata"},
        "genre_wikidata": {"value": ["disco", "Q11399"], "source": "wikidata"},
        "homepage": {"value": "https://artist.example", "source": "wikidata"},
    })
    record["conflicts"] = []
    facts = build_payload(make_track(), research=record)["facts"]
    assert facts == {"first_release_date": "1982-11-30", "year": 1982, "release_group_type": "Album",
                     "publication_date_wikidata": "1983-01-23", "facts_source": "musicbrainz, wikidata"}


def test_a_year_conflict_withholds_every_date_fact():
    """research reports a 'year' conflict when the MusicBrainz year and the Wikidata publication
    date disagree: both dates are disputed, not only the 'year' field."""
    record = research_record()
    record["conflicts"] = [{"field": "year", "values": [{"source": "musicbrainz", "value": 1982},
                                                       {"source": "wikidata", "value": 1983}]}]
    facts = build_payload(make_track(), research=record)["facts"]
    for name in ("year", "first_release_date", "publication_date_wikidata"):
        assert name not in facts, name
    assert facts["release_group_type"] == "Album"


def test_measured_provenance_is_sent_only_with_its_value():
    track = make_track()
    track.bpm = float("nan")
    measured = build_payload(track)["measured"]
    assert "bpm" not in measured and "bpm_source" not in measured
    track = make_track(energy=None)
    track.energy_confidence = 0.5
    measured = build_payload(track)["measured"]
    assert "energy" not in measured and "energy_confidence" not in measured


def test_anthropic_fallback_answer_follows_the_last_fallback_block():
    """content = [declining model's partial text, fallback block, serving model's answer]."""
    partial = SimpleNamespace(type="text", text='{"moods": [{"label": "dar')
    block = SimpleNamespace(type="fallback", to=SimpleNamespace(model="claude-opus-5"),
                            trigger=SimpleNamespace(type="refusal"))
    answer = SimpleNamespace(type="text", text=json.dumps(good_answer()))
    msg = sdk_message(blocks=[partial, block, answer], model="claude-opus-5-5")
    p, _ = anthropic_provider(msg)
    res = p.complete_json(system="S", user="U", schema={})
    assert res.ok and res.data == good_answer() and res.model == "claude-opus-5"
    # usage.iterations names the serving model even without a block
    msg = sdk_message(json.dumps(good_answer()), model="claude-opus-5-5")
    msg.usage.iterations = [SimpleNamespace(type="message", model="claude-opus-5-5"),
                            SimpleNamespace(type="fallback_message", model="claude-sonnet-5")]
    p, _ = anthropic_provider(msg)
    assert p.complete_json(system="S", user="U", schema={}).model == "claude-sonnet-5"


def test_anthropic_context_window_stop_is_incomplete():
    p, _ = anthropic_provider(sdk_message('{"moods": [', stop="model_context_window_exceeded"))
    res = p.complete_json(system="S", user="U", schema={})
    assert res.data is None and res.error_kind == "incomplete"


def test_get_provider_treats_none_options_as_not_given():
    p = get_provider("ollama", model="m", env={}, temperature=None, seed=None, max_retries=None, http=None,
                     effort=None, fallbacks=None, client=None, responses=None)
    assert isinstance(p, OpenAICompatibleProvider) and p.temperature == 0.0 and p.seed == 0
    a = get_provider("anthropic", env={}, client=FakeAnthropicClient(), max_retries=None, effort=None, http=None,
                     temperature=None)
    assert a.max_retries == 2
    for bad in ({"seed": "x"}, {"temperature": "warm"}, {"temperature": float("nan")}):
        with pytest.raises(AIConfigError):
            get_provider("ollama", model="m", env={}, **bad)


def test_parse_json_text_one_line_fence():
    assert parse_json_text('```json {"a": 1}```') == ({"a": 1}, None)


def test_gzip_error_body_is_scrubbed_before_logging(caplog):
    import gzip
    caplog.set_level(logging.DEBUG)

    def transport(method, url, headers, body, timeout):
        payload = json.dumps({"error": {"message": f"bad key {SECRET}"}}).encode("utf-8")
        return HttpResponse(401, {"content-encoding": "gzip"}, gzip.compress(payload), url)

    http = HttpClient(transport=transport, sleep=lambda s: None, max_retries=0)
    p = OpenAICompatibleProvider(http=http, alias="ollama", env={"VEG_AI_MODEL": "m", "VEG_AI_API_KEY": SECRET})
    res = p.complete_json(system="S", user="U", schema={})
    assert res.error_kind == "auth" and SECRET not in res.error and "<redacted>" in res.error
    assert SECRET not in caplog.text


def test_deeply_nested_cache_file_never_raises(tmp_path):
    path = tmp_path / AI_CACHE_NAME
    path.write_text("[" * 100_000, encoding="utf-8")
    c = AICache(path)
    assert len(c) == 0 and any("nested too deeply" in w for w in c.warnings)
    assert path.read_text(encoding="utf-8") == "[" * 100_000              # not ours: untouched
    own = tmp_path / "own.json"
    own.write_text('{"schema": "vegroove.ai_cache", "entries": ' + "[" * 100_000, encoding="utf-8")
    assert len(AICache(own)) == 0 and (tmp_path / "own.json.corrupt-1").exists()


def test_interpretations_name_their_inputs_and_the_user_tag_licence(tmp_path):
    tracks = [make_track(1), make_track(2)]
    research = {tracks[1].track_id: research_record()}
    path = tmp_path / AI_CACHE_NAME
    res = run(tracks, FakeProvider(), AICache(path), mode="online", research=research, use_community_tags=True)
    plain, tagged = (res.interpretations[t.track_id] for t in tracks)
    assert plain["inputs"] == ["curator", "measured"] and "license_note" not in plain
    assert tagged["inputs"] == ["community_tags", "curator", "facts", "measured"]
    assert "non-commercial" in tagged["license_note"]
    again = run(tracks, None, AICache(path), mode="cache", research=research, use_community_tags=True)
    assert again.interpretations[tracks[1].track_id]["license_note"] == tagged["license_note"]


def test_max_tokens_env_is_only_checked_when_the_provider_is_called(tmp_path, monkeypatch):
    monkeypatch.setenv("VEG_AI_MAX_TOKENS", "lots")
    run([make_track()], None, AICache(tmp_path / AI_CACHE_NAME), mode="cache")          # no error
    with pytest.raises(AIConfigError, match="VEG_AI_MAX_TOKENS"):
        run([make_track()], FakeProvider(), AICache(tmp_path / AI_CACHE_NAME), mode="online")
    monkeypatch.setenv("VEG_AI_MAX_TOKENS", "3000")
    fake = FakeProvider()
    run([make_track()], fake, AICache(tmp_path / AI_CACHE_NAME), mode="online")
    assert fake.calls[0]["max_tokens"] == 3000


def test_find_index_follows_overwrites(tmp_path):
    c = AICache(tmp_path / AI_CACHE_NAME)
    c.put("1" * 64, entry())
    c.put("1" * 64, {**entry(), "input_digest": "b" * 64})       # same key, new input digest
    assert c.find("a" * 64) is None
    assert c.find("b" * 64)[0] == "1" * 64
    c.save()
    assert AICache(tmp_path / AI_CACHE_NAME).find("b" * 64)[0] == "1" * 64


# ---------------------------------------------------------------------------
# Review A1 / A2: no identity in the payload, per-field evidence whitelists, privacy note
# ---------------------------------------------------------------------------

def test_payload_never_carries_artist_title_or_file_name(tmp_path):
    """Without tags the title is the file name stem: neither it nor the artist reaches a provider."""
    track = make_track(folder="Secret Folder")
    track.artist, track.title = "Stem Artist Zq", "Stem Title Zq"
    track.feature_sources.update(artist="filename", title="filename")
    payload = build_payload(track, segments=SEGMENTS, research=research_record(), use_community_tags=True)
    assert "track" not in payload
    text = json.dumps(payload)
    for needle in ("Stem Artist Zq", "Stem Title Zq", "Artist 1", "Song 1", "filename", "identity_source"):
        assert needle not in text, needle
    fake = FakeProvider()
    run([track], fake, AICache(tmp_path / AI_CACHE_NAME), mode="online")
    (call,) = fake.calls
    for needle in ("Stem Artist Zq", "Stem Title Zq", "Secret Folder", ".mp3"):
        assert needle not in call["user"] and needle not in call["system"], needle


def full_payloads() -> tuple[dict, dict]:
    """(measured payload with every segment key, all-manual curator payload): every whitelisted path."""
    seg = segments_with(harmony={"key_estimate": "8A", "clarity": 0.6, "stability": 0.8, "modulations": 0})
    measured = build_payload(make_track(mood=None), segments=seg)
    track = make_track(manual_energy=True)
    track.bpm, track.camelot = 124.0, "9B"
    track.genre, track.vibe_tags, track.danceability, track.vocal_presence = "Disco", ["campy"], 0.8, 0.3
    for name in ("bpm", "camelot", "genre", "vibe_tags", "danceability", "vocal_presence"):
        track.feature_sources[name] = "manual"
    curated = build_payload(track)
    return measured, curated


def test_evidence_whitelists_name_real_payload_paths():
    from veg_mixer.ai import EVIDENCE_WHITELISTS
    from veg_mixer.ai.schema import path_exists, under_prefix
    assert set(EVIDENCE_WHITELISTS) == {"moods", "danceability", "groove"}
    measured, curated = full_payloads()
    for field, prefixes in EVIDENCE_WHITELISTS.items():
        for prefix in prefixes:
            assert prefix.split(".")[0] in ("measured", "curator"), (field, prefix)   # never facts / tags / track
            assert path_exists(measured, prefix) or path_exists(curated, prefix), (field, prefix)   # no typos
    assert under_prefix("measured.segments.tempo.bpm", ("measured.segments.tempo",))
    assert not under_prefix("measured.energy_source", ("measured.energy",))           # component-wise


@pytest.mark.parametrize("evidence, kept", [
    (["facts.first_release_date"], False),                  # catalogue facts never support a mood
    (["facts.release_group_type"], False),
    (["facts.year", "facts.release_group_type"], False),
    (["community_tags.tags.0"], False),                     # unverified user tags
    (["measured.duration_s"], False),
    (["measured.segments.sections.1.label"], False),        # heuristic section labels
    (["measured.segments.notes.0"], False),
    (["measured.energy"], True),
    (["measured.key"], True),                               # mode: Camelot A = minor
    (["measured.bpm"], True),
    (["measured.segments.tempo.bpm"], True),
    (["measured.segments.dynamic_range_db"], True),
    (["measured.segments.vocal_presence_proxy.value"], True),
    (["measured.segments.harmony.clarity"], True),
    (["facts.first_release_date", "measured.segments.energy_mean"], True),   # at least one mood path
])
def test_moods_need_mood_relevant_evidence(evidence, kept):
    record = research_record()
    record["conflicts"] = []
    seg = segments_with(harmony={"key_estimate": "8A", "clarity": 0.6, "stability": 0.8, "modulations": 0})
    payload = build_payload(make_track(mood=None), segments=seg, research=record, use_community_tags=True)
    answer = good_answer()
    answer["moods"] = [{"label": "nostalgic", "confidence": 0.8, "evidence": evidence}]
    clean, errors = validate_interpretation(answer, payload)
    assert errors == []
    if kept:
        assert [m["label"] for m in clean["moods"]] == ["nostalgic"]
        assert clean["moods"][0]["evidence"] == evidence
    else:
        assert clean["moods"] == []
        (dropped,) = [d for d in clean["dropped"] if d["field"].startswith("moods")]
        assert "mood-relevant evidence" in dropped["reason"]


@pytest.mark.parametrize("path", ["curator.mood", "curator.vibe_tags.0", "curator.genre", "curator.energy",
                                  "curator.bpm", "curator.key", "curator.vocal_presence"])
def test_curator_labels_support_a_mood(path):
    payload = full_payloads()[1]
    answer = good_answer()
    answer["moods"] = [{"label": "campy", "confidence": 0.7, "evidence": [path]}]
    clean, errors = validate_interpretation(answer, payload)
    assert errors == [] and [m["label"] for m in clean["moods"]] == ["campy"], path


@pytest.mark.parametrize("evidence, kept", [
    (["measured.key"], False),                              # key / mode say nothing about rhythm
    (["measured.energy"], False),
    (["measured.segments.energy_mean"], False),
    (["measured.segments.dynamic_range_db"], False),        # loudness / dynamics
    (["measured.segments.harmony.key_estimate"], False),
    (["measured.segments.density.value"], False),           # arrangement density, not rhythm
    (["curator.mood"], False),
    (["measured.segments.tempo.confidence"], False),        # provenance only
    (["measured.bpm"], True),
    (["measured.segments.tempo.bpm"], True),
    (["measured.segments.tempo.half_double_ambiguous"], False),   # qualifies a BPM, is not one
    (["measured.segments.percussive_density"], True),
    (["measured.key", "measured.bpm"], True),               # at least one rhythm path
])
@pytest.mark.parametrize("field", ["danceability", "groove"])
def test_danceability_and_groove_need_rhythm_evidence(field, evidence, kept):
    seg = segments_with(harmony={"key_estimate": "8A", "clarity": 0.6, "stability": 0.8, "modulations": 0})
    payload = build_payload(make_track(mood=None), segments=seg)
    payload["curator"] = {"mood": "tense"}                  # a curator label that is not about rhythm
    answer = good_answer()
    answer[field] = {"value": 0.7, "confidence": 0.6, "evidence": evidence}
    clean, errors = validate_interpretation(answer, payload)
    assert errors == []
    if kept:
        assert clean[field] == {"value": 0.7, "confidence": 0.6, "evidence": evidence}
    else:
        assert clean[field] == {"value": None, "confidence": 0.0, "evidence": []}
        (dropped,) = [d for d in clean["dropped"] if d["field"] == field]
        assert "rhythm evidence" in dropped["reason"] or "provenance" in dropped["reason"]


def test_curator_danceability_and_tempo_are_rhythm_evidence():
    payload = full_payloads()[1]
    for path in ("curator.danceability", "curator.bpm"):
        answer = good_answer()
        answer["groove"] = {"value": 0.8, "confidence": 0.7, "evidence": [path]}
        clean, errors = validate_interpretation(answer, payload)
        assert errors == [] and clean["groove"]["value"] == 0.8, path
    answer = good_answer()
    answer["groove"] = {"value": 0.8, "confidence": 0.7, "evidence": ["curator.key", "curator.energy"]}
    clean, _ = validate_interpretation(answer, payload)
    assert clean["groove"]["value"] is None


def test_whitelists_are_read_at_validation_time(monkeypatch):
    from veg_mixer.ai import schema as schema_mod
    answer = good_answer()
    answer["moods"] = [{"label": "tense", "confidence": 0.7, "evidence": ["measured.segments.trajectory"]}]
    assert validate_interpretation(answer, payload_for())[0]["moods"]
    monkeypatch.setitem(schema_mod.EVIDENCE_WHITELISTS, "moods", ("curator.mood",))
    clean, _ = validate_interpretation(answer, payload_for())
    assert clean["moods"] == [] and "mood-relevant" in clean["dropped"][0]["reason"]


def test_old_interpret_v1_cache_entries_are_not_served(tmp_path):
    """An answer cached for a v1 payload (identity included, v1 prompt / schema) is never reused."""
    from veg_mixer.ai.cache import make_key
    track = make_track()
    payload = payload_for(track)
    legacy_payload = {**payload, "track": {"artist": "Artist 1", "title": "Song 1"}}
    old = {"prompt_version": "interpret-v1", "prompt_digest": "0" * 16, "schema_version": "interpretation-v1"}
    path = tmp_path / AI_CACHE_NAME
    cache = AICache(path)
    for p in (legacy_payload, payload):                 # neither the old shape nor the old versions match
        cache.put(make_key({**old, "provider": "fake", "model": "fake-1", "input": p}),
                  {**entry(), "input_digest": make_key({**old, "input": p}), "provider": "fake", "model": "fake-1"})
    cache.save()
    res = run([track], None, AICache(path), mode="cache")
    assert res.interpretations == {} and res.stats["cache_hits"] == 0
    fake = FakeProvider()
    res = run([track], fake, AICache(path), mode="online")
    assert len(fake.calls) == 1 and res.stats["cache_hits"] == 0


def test_privacy_note_states_what_is_sent():
    from veg_mixer.ai import PRIVACY_NOTE
    assert PRIVACY_NOTE is interpret_mod.PRIVACY_NOTE
    for words in ("measured features", "verified facts", "curator labels", "never audio", "file paths",
                  "file names", "artist", "title"):
        assert words in PRIVACY_NOTE, words


def test_integer_schema_type_accepts_integral_floats():
    from veg_mixer.ai import check_instance
    assert check_instance(2, {"type": "integer"}) == [] and check_instance(2.0, {"type": "integer"}) == []
    for bad in (2.5, True, float("inf"), "2"):
        assert check_instance(bad, {"type": "integer"}), bad


def test_undetermined_segment_blocks_are_not_sent_and_empty_blocks_are_no_evidence():
    """The analyzer's 'unknown' shapes (no pulse, no clear key, no trajectory, no vocal
    estimate) leave no citable path; a cited block holding only provenance is no evidence."""
    seg = segments_with(tempo={"bpm": None, "confidence": 0.0, "candidates": [], "half_double_ambiguous": False},
                        trajectory={"value": None, "confidence": 0.0},
                        vocal_presence_proxy={"value": None, "confidence": 0.0, "method": "x"},
                        harmony={"key_estimate": None, "clarity": 0.0, "stability": 0.0, "modulations": 0})
    payload = payload_for(segments=seg)
    for key in ("tempo", "trajectory", "vocal_presence_proxy", "harmony"):
        assert key not in payload["measured"]["segments"], key
    payload["measured"]["segments"]["tempo"] = {"confidence": 0.0, "half_double_ambiguous": False}
    answer = good_answer()
    answer["danceability"] = {"value": 0.7, "confidence": 0.6, "evidence": ["measured.segments.tempo"]}
    clean, errors = validate_interpretation(answer, payload)
    assert errors == [] and clean["danceability"]["value"] is None
    assert any("undetermined" in d["reason"] for d in clean["dropped"])


def test_overflowing_numbers_are_schema_errors_not_crashes():
    from veg_mixer.ai.schema import check_instance
    errors = check_instance({"x": 10 ** 400}, {"type": "object", "properties": {"x": {"type": "number"}},
                                               "required": ["x"], "additionalProperties": False}, "answer")
    assert errors and "answer.x" in errors[0]


def test_only_descriptive_facts_and_no_identity_tags_are_sent():
    record = research_record()
    record["conflicts"] = []
    record["facts"].update({"composer": {"value": "Rod Temperton", "source": "musicbrainz"},
                            "label": {"value": "Epic", "source": "musicbrainz"}})
    track = make_track()
    record["user_tags"] = [{"name": track.artist, "count": 50, "source": "musicbrainz"},
                           {"name": "spooky", "count": 3, "source": "musicbrainz"}]
    payload = build_payload(track, research=record, use_community_tags=True)
    assert set(payload["facts"]) - {"facts_source"} <= {"year", "first_release_date", "release_group_type",
                                                         "publication_date_wikidata"}
    assert payload["community_tags"]["tags"] == ["spooky"]


def test_a_discogs_only_dispute_does_not_change_the_payload():
    """MusicBrainz and Wikidata agree, only Discogs (data that expires after 6 h) disagrees:
    the AI gets the same facts before and after the expiry, so the cache key is stable."""
    record = research_record()
    agree = [{"source": "musicbrainz", "value": 1982}, {"source": "wikidata", "value": 1982}]
    record["conflicts"] = [{"field": "year", "values": [*agree, {"source": "discogs", "value": 1980,
                                                                   "attribution": "Data provided by Discogs."}]}]
    before = build_payload(make_track(), research=record)
    record["conflicts"] = []                                       # the Discogs value expired
    after = build_payload(make_track(), research=record)
    assert before == after and before["facts"]["year"] == 1982
    assert input_digest(before) == input_digest(after)
    # a dispute between durable sources still hides the field
    record["conflicts"] = [{"field": "year", "values": [agree[0], {"source": "wikidata", "value": 1983}]}]
    assert "year" not in build_payload(make_track(), research=record)["facts"]
