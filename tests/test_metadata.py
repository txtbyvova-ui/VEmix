"""Spec §5: curator metadata import/export (CSV/JSON) and manual-over-auto precedence."""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

import pytest

from veg_mixer.metadata import (
    DERIVED_KEY_CONFIDENCE_SOURCE,
    EXPORT_COLUMNS,
    REFERENCE_COLUMNS,
    MetadataError,
    apply_metadata,
    export_metadata,
    load_metadata,
)
from veg_mixer.models import EDITABLE_FIELDS, TrackFeatures

REPO = Path(__file__).resolve().parents[1]
HEADER = ",".join(EXPORT_COLUMNS)


def make_track(rel: str, **kw) -> TrackFeatures:
    """A track as analysis.py would produce it: auto values with their sources."""
    sources = kw.pop("sources", {})
    t = TrackFeatures(path=Path("C:/library") / rel, duration_s=kw.pop("duration_s", 180.0), rel_path=rel, **kw)
    t.feature_sources.update(sources)
    return t


def library() -> list[TrackFeatures]:
    return [
        make_track("Morning/Alpha Tide - First Light.mp3", artist="Alpha Tide", title="First Light", bpm=98.4,
                   camelot="8B", key_confidence=0.42, energy=0.47, energy_confidence=0.4, genre="Pop",
                   sources={"bpm": "auto:bpm-v1", "camelot": "auto:key-v1", "energy": "auto:energy-heuristic-v1",
                            "genre": "id3", "artist": "id3", "title": "id3"}),
        make_track("Evening/Беta Group - Ночной город.mp3", artist="Беta Group", title="Ночной город", bpm=121.0,
                   energy=0.66, energy_confidence=0.4, script_hint="cyrillic",
                   sources={"bpm": "auto:bpm-v1", "energy": "auto:energy-heuristic-v1", "artist": "filename"}),
        make_track("Misc/Gamma Échelle - Café Été.flac", bpm=None, energy=None),
    ]


def snapshot(tracks: list[TrackFeatures]) -> list[dict]:
    return [t.to_dict() for t in tracks]


def read_csv_dicts(path: Path, delimiter: str = ",") -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


MANUAL = {
    "Morning/Alpha Tide - First Light.mp3": {
        "energy": 0.3, "mood": "calm", "vibe_tags": ["melodic", "acoustic"], "language": "en",
        "familiarity": 0.8, "approved": True, "camelot": "9B",
        "notes": 'Curator note, with comma; "quotes"\nand a second line',
    },
    "Evening/Беta Group - Ночной город.mp3": {
        "language": "uk", "title": "Ночной город (radio edit)", "requires_review": True, "excluded": False,
        "bpm": 120.5, "danceability": 0.75,
    },
}


# ---------------------------------------------------------------------------
# Export / round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".csv", ".json"])
def test_untouched_export_reimport_changes_nothing(tmp_path, suffix):
    tracks = library()
    before = snapshot(tracks)
    out = tmp_path / f"meta{suffix}"
    export_metadata(tracks, out)

    if suffix == ".csv":
        assert out.read_bytes().startswith(b"\xef\xbb\xbf")             # BOM for Excel
        rows = read_csv_dicts(out)
        assert list(rows[0]) == list(EXPORT_COLUMNS)
    else:
        rows = json.loads(out.read_text(encoding="utf-8"))
        assert "Ночной" in out.read_text(encoding="utf-8")              # ensure_ascii=False
    # editable columns are empty: auto estimates never land there ...
    for row in rows:
        assert all(row[f] in ("", None) for f in EDITABLE_FIELDS), row
    # ... they are reference-only
    first = rows[0]
    assert float(first["auto_bpm"]) == 98.4 and first["auto_camelot"] == "8B"
    assert float(first["auto_energy"]) == 0.47 and first["auto_genre"] == "Pop"
    assert "energy=auto:energy-heuristic-v1" in first["sources"]
    assert rows[1]["script_hint"] == "cyrillic"

    loaded = load_metadata(out)
    assert loaded == {}
    applied, warnings = apply_metadata(tracks, loaded)
    assert warnings == []
    assert snapshot(applied) == before


@pytest.mark.parametrize("suffix", [".csv", ".json"])
def test_round_trip_preserves_manual_values(tmp_path, suffix):
    tracks = library()
    out = tmp_path / f"curator{suffix}"
    export_metadata(tracks, out, existing=MANUAL)
    loaded = load_metadata(out)
    assert loaded == MANUAL                                               # identity for manual values

    direct, _ = apply_metadata(tracks, MANUAL)
    via_file, warnings = apply_metadata(tracks, loaded)
    assert warnings == []
    assert snapshot(via_file) == snapshot(direct)

    # export of already-applied tracks (manual sources) reproduces the same manual values
    again = tmp_path / f"again{suffix}"
    export_metadata(via_file, again)
    assert load_metadata(again) == MANUAL


def test_unicode_metadata_file_name_and_folder(tmp_path):
    out = tmp_path / "Кураторская папка" / "метаданные — été.csv"
    export_metadata(library(), out, existing=MANUAL)
    assert load_metadata(out) == MANUAL
    assert [p.name for p in out.parent.iterdir()] == [out.name]


def test_auto_values_of_applied_tracks_are_not_repeated_as_auto(tmp_path):
    applied, _ = apply_metadata(library(), MANUAL)
    out = tmp_path / "m.csv"
    export_metadata(applied, out)
    row = read_csv_dicts(out)[0]
    assert row["energy"] == "0.3" and row["auto_energy"] == ""           # manual value is not "auto"
    assert row["auto_bpm"] == "98.4"                                     # untouched auto value stays reference
    assert "energy=manual" in row["sources"]


def test_export_semicolon_decimal_comma_for_russian_excel(tmp_path):
    out = tmp_path / "ru.csv"
    export_metadata(library(), out, existing=MANUAL, delimiter=";")
    rows = read_csv_dicts(out, delimiter=";")
    assert rows[0]["energy"] == "0,3" and rows[0]["auto_bpm"] == "98,4"
    assert rows[0]["vibe_tags"] == "melodic;acoustic"                    # quoted by the writer
    assert load_metadata(out) == MANUAL


def test_export_keeps_unmatched_curator_rows(tmp_path, caplog):
    existing = {**MANUAL, "Gone/Removed Artist - Old Song.mp3": {"approved": True, "notes": "keep me"}}
    out = tmp_path / "m.csv"
    with caplog.at_level(logging.WARNING, logger="veg_mixer.metadata"):
        export_metadata(library(), out, existing=existing)
    assert "Gone/Removed Artist - Old Song.mp3" in load_metadata(out)
    assert any("matches no track" in r.getMessage() for r in caplog.records)


def test_export_never_writes_a_file_that_cannot_be_reloaded(tmp_path):
    # two curator keys that differ only in case; one claims the track, the other must not
    # be re-emitted as a second row with the same path
    existing = {"Morning/Alpha Tide - First Light.mp3": {"energy": 0.3},
                "MORNING/alpha tide - first light.mp3": {"energy": 0.9},
                "gone/x.mp3": {"approved": True}, "GONE/X.mp3": {"approved": False}}
    out = tmp_path / "m.csv"
    export_metadata(library(), out, existing=existing)
    data = load_metadata(out)                                             # must not raise "duplicate"
    assert data["Morning/Alpha Tide - First Light.mp3"] == {"energy": 0.3}
    assert data["gone/x.mp3"] == {"approved": True} and len(data) == 2


def test_curator_scratch_columns_are_ignored(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("rel_path,energy,_comment\na.mp3,0.4,ask the client\n", encoding="utf-8")
    assert load_metadata(p) == {"a.mp3": {"energy": 0.4}}


def test_json_duplicate_keys_rejected(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('[{"rel_path": "a.mp3", "energy": 0.1, "energy": 0.9}]', encoding="utf-8")
    with pytest.raises(MetadataError, match="duplicate key 'energy'"):
        load_metadata(p)


def test_export_is_atomic_and_refuses_non_metadata_paths(tmp_path):
    out = tmp_path / "sub" / "m.csv"
    export_metadata(library(), out)
    export_metadata(library(), out, existing=MANUAL)                      # overwrite in place
    assert load_metadata(out) == MANUAL
    assert sorted(p.name for p in out.parent.iterdir()) == ["m.csv"]   # no temp leftovers
    song = tmp_path / "song.mp3"
    song.write_bytes(b"ID3fake")
    with pytest.raises(MetadataError, match="csv or .json"):
        export_metadata(library(), song)
    assert song.read_bytes() == b"ID3fake"


# ---------------------------------------------------------------------------
# Import dialects
# ---------------------------------------------------------------------------

def test_load_csv_semicolon_bom_decimal_comma_and_russian_booleans(tmp_path):
    p = tmp_path / "excel_ru.csv"
    text = ("rel_path;energy;bpm;approved;excluded;requires_review;language;camelot;vibe_tags;notes\r\n"
            'Morning\\Alpha Tide - First Light.mp3;0,35;98,5;да;НЕТ;ИСТИНА;EN;08a;Chill, Melodic ;"Привет; мир"\r\n')
    p.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    data = load_metadata(p)
    assert data == {"Morning/Alpha Tide - First Light.mp3": {
        "energy": 0.35, "bpm": 98.5, "approved": True, "excluded": False, "requires_review": True,
        "language": "en", "camelot": "8A", "vibe_tags": ["chill", "melodic"], "notes": "Привет; мир"}}


def test_load_csv_windows_1251_fallback(tmp_path, caplog):
    p = tmp_path / "legacy.csv"
    p.write_bytes("rel_path;mood;language\r\nПесня.mp3;спокойное;uk\r\n".encode("cp1251"))
    with caplog.at_level(logging.WARNING, logger="veg_mixer.metadata"):
        data = load_metadata(p)
    assert data == {"Песня.mp3": {"mood": "спокойное", "language": "uk"}}
    assert any("Windows-1251" in r.getMessage() for r in caplog.records)


def test_load_csv_with_sep_hint_and_tabs(tmp_path):
    p = tmp_path / "hint.csv"
    p.write_text("sep=;\nrel_path;energy\na.mp3;0,5\n", encoding="utf-8")
    assert load_metadata(p) == {"a.mp3": {"energy": 0.5}}
    t = tmp_path / "tabs.csv"
    t.write_text("rel_path\tenergy\tauto_bpm\na.mp3\t0.25\t120\n", encoding="utf-8")
    assert load_metadata(t) == {"a.mp3": {"energy": 0.25}}


def test_load_json_mapping_form_and_unicode(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"Música/Ünïcode – Été.mp3": {"vibe_tags": "dreamy;warm", "familiarity": 1},
                             "x.mp3": {"approved": "yes"}}, ensure_ascii=False), encoding="utf-8")
    assert load_metadata(p) == {"Música/Ünïcode – Été.mp3": {"vibe_tags": ["dreamy", "warm"], "familiarity": 1.0},
                                "x.mp3": {"approved": True}}


def test_example_csv_in_profiles_loads():
    data = load_metadata(REPO / "profiles" / "metadata_example.csv")
    assert 2 <= len(data) <= 4
    assert all(isinstance(v, dict) and v for v in data.values())


# ---------------------------------------------------------------------------
# Validation errors: file + line + column
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("column, value, fragment", [
    ("energy", "1.5", "out of range"),
    ("energy", "-0.1", "out of range"),
    ("bpm", "-3", "out of range"),
    ("bpm", "0", "out of range"),
    ("bpm", "fast", "expected a number"),
    ("key_confidence", "2", "out of range"),
    ("familiarity", "nan", "finite"),
    ("camelot", "13A", "Camelot"),
    ("camelot", "8C", "Camelot"),
    ("language", "eng", "ISO 639-1"),
    ("language", "english", "ISO 639-1"),
    ("approved", "maybe", "boolean"),
    ("excluded", "2", "boolean"),
])
def test_invalid_values_raise_with_line_and_column(tmp_path, column, value, fragment):
    p = tmp_path / "bad.csv"
    p.write_text(f"rel_path,{column}\nok.mp3,\nbad.mp3,{value}\n", encoding="utf-8-sig")
    with pytest.raises(MetadataError, match=fragment) as exc:
        load_metadata(p)
    err = exc.value
    assert err.line == 3 and err.column == column
    assert "line 3" in str(err) and f"column '{column}'" in str(err) and "bad.csv" in str(err)


def test_line_numbers_account_for_multiline_cells(tmp_path):
    p = tmp_path / "multi.csv"
    p.write_text('rel_path,notes,energy\na.mp3,"one\ntwo\nthree",0.5\nb.mp3,,7\n', encoding="utf-8")
    with pytest.raises(MetadataError) as exc:
        load_metadata(p)
    assert exc.value.line == 5


@pytest.mark.parametrize("content, fragment, line", [
    ("rel_path,energie\na.mp3,0.5\n", "did you mean 'energy'", 1),
    ("path,energy\na.mp3,0.5\n", "unknown column 'path' \\(did you mean 'rel_path'", 1),
    ("energy;mood\n0,5;calm\n", "missing required column 'rel_path'", 1),
    ("rel_path,energy,energy\na.mp3,0.5,0.5\n", "duplicate column", 1),
    ("rel_path,energy\na.mp3,0.5\nA.MP3,0.6\n", "duplicate rel_path", 3),
    ("rel_path,energy\n,0.5\n", "empty rel_path", 2),
    ("rel_path,energy\na.mp3,0.5,0.7\n", "cells but the header has", 2),
    ("", "empty", None),
])
def test_structural_csv_errors(tmp_path, content, fragment, line):
    p = tmp_path / "s.csv"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(MetadataError, match=fragment) as exc:
        load_metadata(p)
    assert exc.value.line == line


def test_json_errors_report_line_numbers(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('[\n  {"rel_path": "a.mp3", "energy": 0.5},\n\n  {"rel_path": "b.mp3",\n   "energy": 3}\n]\n',
                 encoding="utf-8")
    with pytest.raises(MetadataError, match="out of range") as exc:
        load_metadata(p)
    assert exc.value.line == 4 and exc.value.column == "energy" and "entry #2" in str(exc.value)

    p.write_text('[\n  {"rel_path": "a.mp3", "energy": 0.5,}\n]', encoding="utf-8")
    with pytest.raises(MetadataError, match="invalid JSON") as exc:
        load_metadata(p)
    assert exc.value.line == 2

    p.write_text('[{"rel_path": "a.mp3", "moood": "calm"}]', encoding="utf-8")
    with pytest.raises(MetadataError, match="did you mean 'mood'"):
        load_metadata(p)

    p.write_text('[{"rel_path": "a.mp3", "bpm": true}]', encoding="utf-8")
    with pytest.raises(MetadataError, match="boolean"):
        load_metadata(p)


def test_unsupported_extension(tmp_path):
    p = tmp_path / "m.xlsx"
    p.write_bytes(b"PK")
    with pytest.raises(MetadataError, match="unsupported"):
        load_metadata(p)


# ---------------------------------------------------------------------------
# apply_metadata
# ---------------------------------------------------------------------------

def test_manual_values_override_auto_and_are_marked_manual():
    tracks = library()
    before = snapshot(tracks)
    applied, warnings = apply_metadata(tracks, MANUAL)
    assert warnings == []
    assert snapshot(tracks) == before                                     # inputs not mutated
    a, b, c = applied
    assert a is not tracks[0] and a.vibe_tags is not tracks[0].vibe_tags
    assert a.energy == 0.3 and a.feature_sources["energy"] == "manual" and a.energy_confidence == 1.0
    assert a.confidence("energy") == 1.0
    assert a.camelot == "9B" and a.key_confidence == 1.0
    assert a.feature_sources["key_confidence"] == DERIVED_KEY_CONFIDENCE_SOURCE
    assert a.bpm == 98.4 and a.feature_sources["bpm"] == "auto:bpm-v1"    # untouched auto value kept
    assert a.genre == "Pop" and a.feature_sources["genre"] == "id3"
    assert a.approved is True and a.feature_sources["approved"] == "manual"
    assert a.familiarity == 0.8 and a.language == "en" and a.notes.startswith("Curator note")
    assert b.language == "uk" and b.title == "Ночной город (radio edit)" and b.requires_review is True
    assert b.excluded is False and b.feature_sources["excluded"] == "manual"
    assert b.bpm == 120.5 and b.feature_sources["bpm"] == "manual"
    assert b.script_hint == "cyrillic"                                    # hint kept, language now known
    assert c.to_dict() == tracks[2].to_dict()                             # no override -> identical copy


def test_matching_case_insensitive_backslashes_and_file_name_fallback():
    tracks = library()
    overrides = {
        "morning\\ALPHA TIDE - first light.MP3": {"energy": 0.2},        # Windows separators, other case
        "Café Été.flac": {"mood": "dreamy"},                               # no match (stem differs)
        "Gamma Échelle - Café Été.flac": {"language": "fr"},              # unique file name
        "OldFolder/Беta Group - Ночной город.mp3": {"approved": True},    # stale folder -> name fallback
    }
    applied, warnings = apply_metadata(tracks, overrides)
    assert applied[0].energy == 0.2
    assert applied[2].language == "fr"
    assert applied[1].approved is True
    assert any("'Café Été.flac' matches no track" in w for w in warnings)
    assert any("OldFolder" in w and "by file name only" in w for w in warnings)
    assert len(warnings) == 2


def test_ambiguous_and_conflicting_keys_warn_and_are_not_applied():
    tracks = [make_track("A/intro.mp3"), make_track("B/intro.mp3"), make_track("C/outro.mp3")]
    applied, warnings = apply_metadata(tracks, {
        "intro.mp3": {"energy": 0.9},
        "C/outro.mp3": {"energy": 0.1},
        "outro.mp3": {"energy": 0.5},
        "nowhere/missing.mp3": {"energy": 0.5},
    })
    assert all(t.energy is None for t in applied[:2])
    assert applied[2].energy == 0.1                                       # exact rel_path wins over name
    assert any("ambiguous" in w and "A/intro.mp3" in w and "B/intro.mp3" in w for w in warnings)
    assert any("'outro.mp3' and 'C/outro.mp3' both match" in w for w in warnings)
    assert any("'nowhere/missing.mp3' matches no track" in w for w in warnings)


def test_apply_validates_programmatic_overrides():
    with pytest.raises(MetadataError, match="out of range") as exc:
        apply_metadata(library(), {"Morning/Alpha Tide - First Light.mp3": {"energy": 4}})
    assert exc.value.column == "energy"
    with pytest.raises(MetadataError, match="unknown field"):
        apply_metadata(library(), {"Morning/Alpha Tide - First Light.mp3": {"popularity": 1.0}})


def test_approved_and_excluded_conflict_warns():
    _, warnings = apply_metadata(library(), {"Morning/Alpha Tide - First Light.mp3": {"approved": True,
                                                                                         "excluded": True}})
    assert any("both approved and excluded" in w for w in warnings)


def test_reference_columns_are_ignored_on_import(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text(HEADER + "\n" + "a.mp3" + "," * (len(EXPORT_COLUMNS) - len(REFERENCE_COLUMNS) - 1)
                 + ",128,8A,0.9,Rock,en,cyrillic,bpm=auto:bpm-v1\n", encoding="utf-8")
    assert load_metadata(p) == {}


# ---------------------------------------------------------------------------
# Review-finding regressions
# ---------------------------------------------------------------------------

def test_curator_can_record_a_multi_language_song(tmp_path):
    """EDIT-1 / INT-4: language is a set — a curator can enter 'en;ru' (or copy the
    'auto_language' value analysis wrote, e.g. 'ru;en'); three-letter codes stay refused."""
    p = tmp_path / "m.csv"
    p.write_text('rel_path,language\na.mp3,"en; RU"\nb.mp3,"fr,fr"\n', encoding="utf-8-sig")
    assert load_metadata(p) == {"a.mp3": {"language": "en;ru"}, "b.mp3": {"language": "fr"}}
    p.write_text("rel_path,language\na.mp3,en;rus\n", encoding="utf-8-sig")
    with pytest.raises(MetadataError, match="ISO 639-1"):
        load_metadata(p)


def test_rel_path_matching_uses_the_shared_unicode_normalization():
    """Shared textnorm keys: a decomposed (NFD, macOS) file name matches the NFC spelling a
    curator typed, case-insensitively — the same keys QC compares."""
    import unicodedata
    nfd = unicodedata.normalize("NFD", "Misc/Gamma Échelle - Café Été.flac")
    applied, warnings = apply_metadata([make_track(nfd)], {"misc/GAMMA ÉCHELLE - café été.flac": {"mood": "dreamy"}})
    assert warnings == [] and applied[0].mood == "dreamy"
