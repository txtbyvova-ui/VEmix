"""Shared text normalization (one implementation for curation, sequencing and QC)."""

from __future__ import annotations

import pytest

from veg_mixer.textnorm import (
    label_keys,
    normalize_language,
    normalize_rel_path,
    normalize_tag,
    parse_languages,
    primary_artist,
    split_artists,
    split_values,
)


@pytest.mark.parametrize("value, codes, unresolved", [
    ("rus;eng", ("ru", "en"), ()),
    ("ru, en", ("ru", "en"), ()),
    ("rus/eng", ("ru", "en"), ()),
    ("en-US", ("en",), ()),
    ("pt_BR", ("pt",), ()),
    ("Русский", ("ru",), ()),
    ("English and French", ("en", "fr"), ()),
    ("und", (), ()),
    ("zxx", (), ()),
    ("", (), ()),
    (None, (), ()),
    ("klingon", (), ("klingon",)),
    ("por", ("pt",), ()),
    ("nld", ("nl",), ()),
])
def test_parse_languages(value, codes, unresolved):
    assert parse_languages(value) == (codes, unresolved)


def test_normalize_language_canonical_form():
    assert normalize_language("rus;eng") == "ru;en"
    assert normalize_language("ENG") == "en"
    assert normalize_language("und") is None
    assert normalize_language("en;klingon") == "en;klingon"


def test_split_artists_keeps_full_credit_and_parts():
    assert split_artists("AC/DC") == ["ac/dc", "ac", "dc"]
    assert split_artists("Artist A/Artist B")[:3] == ["artist a/artist b", "artist a", "artist b"]
    assert "simon & garfunkel" in split_artists("Simon & Garfunkel")
    assert "b" in split_artists("A feat. B") and "b" in split_artists("A (feat. B)")
    assert split_artists(None) == [] and split_artists("  ") == []
    assert primary_artist("Main Act feat. Guest") == "main act"
    assert primary_artist("Main Act (feat. Guest)") == "main act"


def test_tags_and_values():
    assert normalize_tag("Feel-Good") == normalize_tag("feel_good") == normalize_tag(" FEEL  good ") == "feel good"
    assert split_values("Pop/Rock; dance") == ["Pop", "Rock", "dance"]
    assert label_keys("Pop/Rock") == {"pop/rock", "pop", "rock"}
    assert label_keys(None) == set()


def test_rel_paths():
    assert normalize_rel_path(".\\sub\\Song.mp3") == "sub/Song.mp3"


@pytest.mark.parametrize("value, codes, unresolved", [
    ("rus eng", ("ru", "en"), ()),          # SPEC-3 / VEDIT-4: whitespace-separated
    ("ruseng", ("ru", "en"), ()),           # ID3v2.3 concatenated codes
    ("engfrarus", ("en", "fr", "ru"), ()),
    ("jp", (), ("jp",)),                    # not an ISO 639-1 code -> unknown, not a claim
    ("xx", (), ("xx",)),
    ("banana", (), ("banana",)),            # 6 letters but not valid triplets
])
def test_parse_languages_whitespace_concatenated_and_invalid_codes(value, codes, unresolved):
    assert parse_languages(value) == (codes, unresolved)


@pytest.mark.parametrize("value, codes, unresolved", [
    ("No vocals", (), ()),                         # R2A-1: free text is not 'Norwegian'
    ("No linguistic content", (), ()),             # = ISO 639-2 'zxx'
    ("instrumental", (), ()),
    ("sung in russian", ("ru",), ("sung", "in")),  # names still count; bare 2-letter words do not
    ("ru en", ("ru", "en"), ()),                   # every word is a language -> split
])
def test_parse_languages_free_text_is_not_a_language_claim(value, codes, unresolved):
    assert parse_languages(value) == (codes, unresolved)
