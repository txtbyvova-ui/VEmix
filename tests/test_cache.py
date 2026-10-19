"""P1 analysis cache (spec §9, §13 Cache 1-4): identity, per-section invalidation,
atomic writes, corruption recovery and legacy automix v1 migration."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from conftest import sha256
from veg_mixer import cache as cache_mod
from veg_mixer.cache import ANALYSIS_CACHE_SCHEMA, AnalysisCache, params_hash

BPM_P = {"sample_rate": 22050, "offset_s": 30.0, "duration_s": 90.0, "min_window_s": 30.0}
KEY_P = {"sample_rate": 22050, "offset_s": 30.0, "duration_s": 60.0, "min_window_s": 1.0}
LOUD_A = {"target_lufs": -16.0, "pcm_true_peak_ceiling": -2.0}
LOUD_B = {"target_lufs": -14.0, "pcm_true_peak_ceiling": -2.0}


def make_source(path: Path, payload: bytes = b"RIFF-not-really-audio" * 50) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def fill(c: AnalysisCache, src: Path) -> None:
    c.put(src, "bpm", {"bpm": 121.5}, version="bpm-v1", params=BPM_P)
    c.put(src, "key", {"camelot": "8A", "key_confidence": 0.71}, version="key-v1", params=KEY_P)
    c.put(src, "loudness", {"integrated_lufs": -9.5}, version="loudness-v1", params=LOUD_A)


def bump_mtime(path: Path, seconds: int = 10) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


# ---------------------------------------------------------------------------
# reuse / identity / invalidation
# ---------------------------------------------------------------------------

def test_reuse_across_instances(tmp_path):
    """Spec Cache 1: values survive a save/load cycle and are served as hits."""
    src = make_source(tmp_path / "music" / "a.wav")
    path = tmp_path / "cache" / "analysis_cache.json"
    c = AnalysisCache(path)
    assert c.get(src, "bpm", version="bpm-v1", params=BPM_P) is None
    fill(c, src)
    c.save()
    assert path.is_file() and not c.dirty

    c2 = AnalysisCache(path)
    assert c2.get(src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 121.5}
    assert c2.get(src, "key", version="key-v1", params=KEY_P) == {"camelot": "8A", "key_confidence": 0.71}
    assert c2.stats == {"hits": 2, "misses": 0, "invalidated": 0, "migrated": 0}
    assert c2.warnings == []
    # a relative / unnormalized spelling of the same file hits the same entry
    alias = src.parent / ".." / "music" / src.name
    assert c2.get(alias, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 121.5}


def test_returned_values_are_private_copies(tmp_path):
    src = make_source(tmp_path / "a.wav")
    c = AnalysisCache(tmp_path / "c.json")
    value = {"bpm": 100.0, "nested": {"x": [1, 2]}}
    c.put(src, "bpm", value, version="bpm-v1")
    value["nested"]["x"].append(3)                  # caller mutation after put
    got = c.get(src, "bpm", version="bpm-v1")
    assert got == {"bpm": 100.0, "nested": {"x": [1, 2]}}
    got["bpm"] = 0.0                                 # caller mutation after get
    assert c.get(src, "bpm", version="bpm-v1")["bpm"] == 100.0


def test_params_hash_is_stable():
    assert params_hash({"a": 30, "b": 1.5}) == params_hash({"b": 1.5, "a": 30.0})
    assert params_hash(None) == params_hash({})
    assert params_hash({"a": 30}) != params_hash({"a": 31})


def test_sections_are_independent(tmp_path):
    """Changing the loudness target must not invalidate BPM / key (spec §9)."""
    src = make_source(tmp_path / "a.wav")
    c = AnalysisCache(tmp_path / "c.json")
    fill(c, src)
    assert c.get(src, "loudness", version="loudness-v1", params=LOUD_B) is None
    assert c.stats["invalidated"] == 1
    assert c.get(src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 121.5}
    assert c.get(src, "key", version="key-v1", params=KEY_P)["camelot"] == "8A"
    # a version bump invalidates only that section as well
    assert c.get(src, "key", version="key-v2", params=KEY_P) is None
    assert c.get(src, "bpm", version="bpm-v1", params=BPM_P) is not None
    c.put(src, "loudness", {"integrated_lufs": -9.4}, version="loudness-v1", params=LOUD_B)
    assert c.get(src, "loudness", version="loudness-v1", params=LOUD_B) == {"integrated_lufs": -9.4}
    assert c.get(src, "loudness", version="loudness-v1", params=LOUD_A) is None


def test_source_modification_invalidates_all_sections(tmp_path):
    """Spec Cache 2: a rewritten source (new size + mtime) drops every section."""
    src = make_source(tmp_path / "a.wav")
    path = tmp_path / "c.json"
    c = AnalysisCache(path)
    fill(c, src)
    c.save()
    make_source(src, b"different content, different size")
    c2 = AnalysisCache(path)
    assert c2.get(src, "bpm", version="bpm-v1", params=BPM_P) is None
    assert c2.get(src, "key", version="key-v1", params=KEY_P) is None
    assert c2.get(src, "loudness", version="loudness-v1", params=LOUD_A) is None
    assert c2.stats["invalidated"] == 1 and c2.stats["hits"] == 0
    c2.put(src, "bpm", {"bpm": 90.0}, version="bpm-v1", params=BPM_P)
    c2.save()
    on_disk = json.loads(path.read_text(encoding="utf-8"))["entries"][str(src.resolve())]
    assert on_disk["fingerprint"]["size"] == src.stat().st_size
    assert set(on_disk["sections"]) == {"bpm"}


def test_touch_invalidates_without_content_hash(tmp_path):
    src = make_source(tmp_path / "a.wav")
    c = AnalysisCache(tmp_path / "c.json")
    fill(c, src)
    bump_mtime(src)
    assert c.get(src, "bpm", version="bpm-v1", params=BPM_P) is None


def test_content_hash_survives_touch_but_not_edit(tmp_path):
    src = make_source(tmp_path / "a.wav", b"A" * 4096)
    path = tmp_path / "c.json"
    c = AnalysisCache(path, use_content_hash=True)
    fill(c, src)
    c.save()
    stored = json.loads(path.read_text(encoding="utf-8"))["entries"][str(src.resolve())]["fingerprint"]
    assert stored["sha256"] == sha256(src)

    bump_mtime(src)                                   # touched, content identical
    c2 = AnalysisCache(path, use_content_hash=True)
    assert c2.get(src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 121.5}
    assert c2.dirty                                   # fingerprint refreshed ...
    c2.save()
    refreshed = json.loads(path.read_text(encoding="utf-8"))["entries"][str(src.resolve())]["fingerprint"]
    assert refreshed["mtime_ns"] == src.stat().st_mtime_ns   # ... and persisted

    # same size, same mtime, different bytes: only the content hash can see it
    st = src.stat()
    src.write_bytes(b"B" * 4096)
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns))
    c3 = AnalysisCache(path, use_content_hash=True)
    assert c3.get(src, "bpm", version="bpm-v1", params=BPM_P) is None
    assert c3.stats["invalidated"] == 1


def test_missing_source(tmp_path):
    src = make_source(tmp_path / "a.wav")
    c = AnalysisCache(tmp_path / "c.json")
    fill(c, src)
    src.unlink()
    assert c.get(src, "bpm", version="bpm-v1", params=BPM_P) is None
    with pytest.raises(FileNotFoundError):
        c.put(src, "bpm", {"bpm": 1.0}, version="bpm-v1")


def test_put_rejects_unserializable_values(tmp_path):
    src = make_source(tmp_path / "a.wav")
    c = AnalysisCache(tmp_path / "c.json")
    with pytest.raises(TypeError):
        c.put(src, "bpm", {"bpm": object()}, version="bpm-v1")
    with pytest.raises(TypeError):
        c.put(src, "bpm", [1, 2], version="bpm-v1")          # type: ignore[arg-type]
    with pytest.raises(ValueError):
        c.put(src, "", {"bpm": 1.0}, version="bpm-v1")
    c.put(src, "energy", {"energy": float("nan")}, version="e1")   # non-finite -> None
    assert c.get(src, "energy", version="e1") == {"energy": None}


def test_unicode_paths_round_trip(tmp_path):
    src = make_source(tmp_path / "Музыка" / "Земфира - Хочешь.mp3")
    path = tmp_path / "кэш" / "analysis_cache.json"
    c = AnalysisCache(path)
    c.put(src, "bpm", {"bpm": 97.0, "note": "тест"}, version="bpm-v1")
    c.save()
    text = path.read_text(encoding="utf-8")
    assert "Земфира - Хочешь.mp3" in text and "тест" in text      # ensure_ascii=False
    assert text.startswith('{\n  "schema"')                        # indent=2
    assert AnalysisCache(path).get(src, "bpm", version="bpm-v1") == {"bpm": 97.0, "note": "тест"}


# ---------------------------------------------------------------------------
# atomic save
# ---------------------------------------------------------------------------

def test_save_is_atomic_and_noop_when_clean(tmp_path, monkeypatch):
    src = make_source(tmp_path / "a.wav")
    path = tmp_path / "c.json"
    c = AnalysisCache(path)
    fill(c, src)
    c.save()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.wav", "c.json"]   # no temp leftovers
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == ANALYSIS_CACHE_SCHEMA and data["schema_version"] == 1
    sec = data["entries"][str(src.resolve())]["sections"]["bpm"]
    assert sec == {"version": "bpm-v1", "params": params_hash(BPM_P), "value": {"bpm": 121.5}}

    calls = []
    monkeypatch.setattr(cache_mod.os, "replace", lambda *a: calls.append(a))
    AnalysisCache(path).save()                     # nothing changed -> no write at all
    c.save()
    assert calls == []


def test_failed_save_keeps_previous_file(tmp_path, monkeypatch):
    src = make_source(tmp_path / "a.wav")
    path = tmp_path / "c.json"
    c = AnalysisCache(path)
    fill(c, src)
    c.save()
    good = path.read_bytes()

    c.put(src, "bpm", {"bpm": 60.0}, version="bpm-v1", params=BPM_P)

    def exploding_fsync(fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cache_mod.os, "fsync", exploding_fsync)
    with pytest.raises(OSError, match="No space"):
        c.save()
    assert path.read_bytes() == good                                   # previous file intact
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.wav", "c.json"]   # temp removed
    assert c.dirty
    monkeypatch.undo()
    c.save()                                                           # retry succeeds
    assert AnalysisCache(path).get(src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 60.0}


def test_failed_serialization_mid_write_keeps_previous_file(tmp_path, monkeypatch):
    src = make_source(tmp_path / "a.wav")
    path = tmp_path / "c.json"
    c = AnalysisCache(path)
    fill(c, src)
    c.save()
    good = path.read_bytes()
    c.put(src, "key", {"camelot": "9A", "key_confidence": 0.5}, version="key-v1", params=KEY_P)

    real_dump = json.dump

    def half_dump(obj, fp, **kw):          # write part of the document, then die
        fp.write('{"schema": "vegroove.analysis_cache", "entr')
        raise KeyboardInterrupt

    monkeypatch.setattr(cache_mod.json, "dump", half_dump)
    with pytest.raises(KeyboardInterrupt):
        c.save()
    monkeypatch.setattr(cache_mod.json, "dump", real_dump)
    assert path.read_bytes() == good
    assert not list(tmp_path.glob(".*.tmp"))


# ---------------------------------------------------------------------------
# recovery from invalid content (spec Cache 4)
# ---------------------------------------------------------------------------

def _good_doc(src: Path) -> dict:
    return {"schema": ANALYSIS_CACHE_SCHEMA, "schema_version": 1, "entries": {
        str(src.resolve()): {"fingerprint": {"size": src.stat().st_size, "mtime_ns": src.stat().st_mtime_ns},
                             "sections": {"bpm": {"version": "bpm-v1", "params": params_hash(BPM_P),
                                                  "value": {"bpm": 120.0}}}}}}


def _bad_cache_bytes(kind: str, src: Path) -> bytes:
    full = json.dumps(_good_doc(src), indent=2)
    return {
        "garbage": b"\x00\xff\xfe\x81 not json at all \x9c",
        "truncated": full[: len(full) // 2].encode(),
        "empty": b"",
        "zeroed": b"\x00" * 64,                      # what a crash can leave on some file systems
        "wrong_schema": json.dumps({**_good_doc(src), "schema": "something.else"}).encode(),
        "wrong_version": json.dumps({**_good_doc(src), "schema_version": 99}).encode(),
        "top_level_list": b"[1, 2, 3]",
        "entries_list": json.dumps({**_good_doc(src), "entries": [1, 2]}).encode(),
    }[kind]


# Demonstrably a damaged VEgroove cache: moved aside (never deleted), work continues.
@pytest.mark.parametrize("kind", ["truncated", "empty", "zeroed", "wrong_version", "entries_list"])
def test_recovers_from_invalid_cache_file(tmp_path, kind):
    src = make_source(tmp_path / "a.wav")
    path = tmp_path / "c.json"
    bad = _bad_cache_bytes(kind, src)
    path.write_bytes(bad)

    c = AnalysisCache(path)                                   # must not raise
    assert len(c.warnings) == 1 and "corrupt-1" in c.warnings[0]
    kept = tmp_path / "c.json.corrupt-1"
    assert kept.read_bytes() == bad                           # bad file preserved, not deleted
    assert not path.exists()
    assert c.get(src, "bpm", version="bpm-v1", params=BPM_P) is None
    c.put(src, "bpm", {"bpm": 99.0}, version="bpm-v1", params=BPM_P)
    c.save()
    assert AnalysisCache(path).get(src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 99.0}

    path.write_bytes(bad)                                     # second incident -> next free name
    AnalysisCache(path)
    assert (tmp_path / "c.json.corrupt-2").read_bytes() == bad and kept.read_bytes() == bad


@pytest.mark.parametrize("kind", ["garbage", "wrong_schema", "top_level_list"])
def test_foreign_file_at_cache_path_is_never_renamed_or_overwritten(tmp_path, kind):
    """SAFE-6: these kinds used to be quarantined (renamed) and replaced by cache JSON. A
    file that is not demonstrably OUR damaged cache is someone's data: it stays where it
    is, byte-identical, and the cache simply does not save (the old expectation of a
    '.corrupt-1' rename for them was the defect)."""
    src = make_source(tmp_path / "a.wav")
    path = tmp_path / "c.json"
    bad = _bad_cache_bytes(kind, src)
    path.write_bytes(bad)
    c = AnalysisCache(path)                                   # must not raise
    assert len(c.warnings) == 1 and "left untouched" in c.warnings[0]
    assert c.get(src, "bpm", version="bpm-v1", params=BPM_P) is None
    c.put(src, "bpm", {"bpm": 99.0}, version="bpm-v1", params=BPM_P)     # works in memory ...
    assert c.get(src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 99.0}
    c.save()                                                  # ... but is never written over the file
    assert path.read_bytes() == bad
    assert sorted(x.name for x in tmp_path.iterdir()) == ["a.wav", "c.json"]


def test_malformed_entries_are_dropped_good_ones_kept(tmp_path):
    good_src = make_source(tmp_path / "good.wav")
    other = make_source(tmp_path / "other.wav", b"x" * 10)
    path = tmp_path / "c.json"
    doc = _good_doc(good_src)
    fp_other = {"size": 10, "mtime_ns": other.stat().st_mtime_ns}
    doc["entries"].update({
        str(tmp_path / "no_fingerprint.wav"): {"sections": {}},
        str(tmp_path / "string_size.wav"): {"fingerprint": {"size": "10", "mtime_ns": 1}, "sections": {}},
        str(tmp_path / "not_a_dict.wav"): [1, 2],
        str(other.resolve()): {"fingerprint": fp_other, "sections": {
            "bpm": {"version": "bpm-v1", "params": params_hash(BPM_P), "value": "120"},   # bad value type
            "key": {"version": "key-v1", "params": params_hash(KEY_P),
                    "value": {"camelot": "5A", "key_confidence": 0.4}},
        }},
    })
    path.write_text(json.dumps(doc), encoding="utf-8")

    c = AnalysisCache(path)
    assert not list(tmp_path.glob("*.corrupt-*"))             # file-level recovery not needed
    assert len(c) == 2
    assert sum("dropped malformed entry" in w for w in c.warnings) == 3
    assert any("dropped malformed section" in w and "[bpm]" in w for w in c.warnings)
    assert c.get(good_src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 120.0}
    assert c.get(other, "key", version="key-v1", params=KEY_P)["camelot"] == "5A"
    assert c.get(other, "bpm", version="bpm-v1", params=BPM_P) is None
    c.save()                                                  # the cleaned document is written back
    assert len(json.loads(path.read_text(encoding="utf-8"))["entries"]) == 2


def test_utf8_bom_is_tolerated(tmp_path):
    src = make_source(tmp_path / "a.wav")
    path = tmp_path / "c.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(_good_doc(src)).encode("utf-8"))
    c = AnalysisCache(path)
    assert c.warnings == [] and c.get(src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 120.0}


# ---------------------------------------------------------------------------
# legacy automix v1 migration (spec Cache 3)
# ---------------------------------------------------------------------------

def _legacy(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "music" / "bpm_cache.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:            # exactly how automix v1 wrote it
        json.dump(data, f, indent=2, ensure_ascii=False)
    return p


def test_legacy_float_and_dict_formats_are_migrated(tmp_path):
    a = make_source(tmp_path / "music" / "a.mp3")
    b = make_source(tmp_path / "music" / "Би-2 - Полковнику.mp3", b"y" * 300)
    legacy = _legacy(tmp_path, {
        str(a): 124.0,                                                          # oldest format
        str(b): {"bpm": 98.5, "camelot": "11A", "key_confidence": 0.734},       # v1 format
        str(tmp_path / "music" / "gone.mp3"): {"bpm": 1.0},                     # source no longer exists
    })
    before = sha256(legacy)
    path = tmp_path / "analysis_cache.json"

    c = AnalysisCache(path, legacy_path=legacy)
    assert c.get(a, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 124.0}
    assert c.get(a, "key", version="key-v1", params=KEY_P) is None               # v1 float had no key
    assert c.get(b, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 98.5}
    assert c.get(b, "key", version="key-v1", params=KEY_P) == {"camelot": "11A", "key_confidence": 0.734}
    assert c.stats["migrated"] == 3
    assert c.migrated_from(b, "key") == "bpm_cache.json"
    c.save()
    assert sha256(legacy) == before                                              # legacy untouched

    # migrated sections are now regular entries (accepted with the current version/params)
    c2 = AnalysisCache(path)
    assert c2.get(b, "key", version="key-v1", params=KEY_P) == {"camelot": "11A", "key_confidence": 0.734}
    assert c2.stats == {"hits": 1, "misses": 0, "invalidated": 0, "migrated": 0}
    sec = json.loads(path.read_text(encoding="utf-8"))["entries"][str(b.resolve())]["sections"]["key"]
    assert sec["migrated_from"] == "bpm_cache.json" and sec["version"] == "key-v1"
    assert sha256(legacy) == before


def test_legacy_pending_values_survive_a_partial_run(tmp_path):
    src = make_source(tmp_path / "music" / "a.mp3")
    legacy = _legacy(tmp_path, {str(src): {"bpm": 100.0, "camelot": "3B", "key_confidence": 0.6}})
    path = tmp_path / "c.json"
    c = AnalysisCache(path, legacy_path=legacy)
    assert c.get(src, "probe", version="probe-v1") is None      # entry created, legacy attached
    assert c.get(src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 100.0}
    c.save()                                                    # key not requested in this run
    c2 = AnalysisCache(path)                                    # legacy file not even given now
    assert c2.get(src, "key", version="key-v1", params=KEY_P) == {"camelot": "3B", "key_confidence": 0.6}
    assert c2.stats["migrated"] == 1


def test_legacy_not_reimported_after_source_change(tmp_path):
    src = make_source(tmp_path / "music" / "a.mp3")
    legacy = _legacy(tmp_path, {str(src): {"bpm": 100.0, "camelot": "3B", "key_confidence": 0.6}})
    path = tmp_path / "c.json"
    c = AnalysisCache(path, legacy_path=legacy)
    assert c.get(src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 100.0}
    c.save()
    make_source(src, b"re-encoded master with different bytes")
    c2 = AnalysisCache(path, legacy_path=legacy)
    assert c2.get(src, "bpm", version="bpm-v1", params=BPM_P) is None          # stale legacy value
    assert c2.get(src, "key", version="key-v1", params=KEY_P) is None          # pending dropped too
    assert c2.stats["migrated"] == 0


def test_existing_entry_wins_over_legacy(tmp_path):
    src = make_source(tmp_path / "music" / "a.mp3")
    path = tmp_path / "c.json"
    c = AnalysisCache(path)
    c.put(src, "bpm", {"bpm": 128.0}, version="bpm-v1", params=BPM_P)
    c.save()
    legacy = _legacy(tmp_path, {str(src): {"bpm": 64.0, "camelot": "1A"}})
    c2 = AnalysisCache(path, legacy_path=legacy)
    assert c2.get(src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 128.0}
    assert c2.get(src, "key", version="key-v1", params=KEY_P) is None
    assert c2.stats["migrated"] == 0


def test_fresh_put_supersedes_pending_legacy_value(tmp_path):
    src = make_source(tmp_path / "music" / "a.mp3")
    legacy = _legacy(tmp_path, {str(src): {"bpm": 64.0, "camelot": "1A", "key_confidence": 0.3}})
    c = AnalysisCache(tmp_path / "c.json", legacy_path=legacy)
    c.put(src, "bpm", {"bpm": 128.0}, version="bpm-v1", params=BPM_P)
    assert c.get(src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 128.0}
    assert c.get(src, "key", version="key-v1", params=KEY_P) == {"camelot": "1A", "key_confidence": 0.3}


def test_legacy_key_without_confidence_stays_unknown(tmp_path):
    src = make_source(tmp_path / "music" / "a.mp3")
    legacy = _legacy(tmp_path, {str(src): {"bpm": 90.0, "camelot": "7B"}})
    c = AnalysisCache(tmp_path / "c.json", legacy_path=legacy)
    assert c.get(src, "key", version="key-v1", params=KEY_P) == {"camelot": "7B", "key_confidence": None}


def test_malformed_legacy_entries_are_skipped_with_warning(tmp_path):
    ok = make_source(tmp_path / "music" / "ok.mp3")
    half = make_source(tmp_path / "music" / "half.mp3")
    legacy = _legacy(tmp_path, {
        str(ok): 120.0,
        str(half): {"bpm": "fast", "camelot": "4A", "key_confidence": 0.5},   # bad bpm, good key
        str(tmp_path / "music" / "x.mp3"): {"camelot": "13C"},
        str(tmp_path / "music" / "y.mp3"): [1, 2],
        str(tmp_path / "music" / "z.mp3"): True,
    })
    before = sha256(legacy)
    c = AnalysisCache(tmp_path / "c.json", legacy_path=legacy)
    assert len(c.warnings) == 4 and all("legacy cache" in w for w in c.warnings)
    assert c.get(ok, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 120.0}
    assert c.get(half, "bpm", version="bpm-v1", params=BPM_P) is None
    assert c.get(half, "key", version="key-v1", params=KEY_P)["camelot"] == "4A"
    assert sha256(legacy) == before


@pytest.mark.parametrize("content", [b"{not json", b"[1, 2]", b"\xff\xfe\x00"])
def test_corrupt_legacy_file_is_left_alone(tmp_path, content):
    src = make_source(tmp_path / "music" / "a.mp3")
    legacy = tmp_path / "music" / "bpm_cache.json"
    legacy.write_bytes(content)
    c = AnalysisCache(tmp_path / "c.json", legacy_path=legacy)
    assert len(c.warnings) == 1 and "nothing migrated" in c.warnings[0]
    assert legacy.read_bytes() == content and not list(legacy.parent.glob("*.corrupt-*"))
    assert c.get(src, "bpm", version="bpm-v1", params=BPM_P) is None


def test_legacy_file_at_the_cache_path_is_never_renamed_or_overwritten(tmp_path):
    """Misconfiguration guard: pointing the new cache at the old bpm_cache.json must not
    quarantine (rename) or overwrite it; it is used read-only for migration instead."""
    src = make_source(tmp_path / "music" / "a.mp3")
    legacy = _legacy(tmp_path, {str(src): {"bpm": 101.0, "camelot": "6A", "key_confidence": 0.9}})
    before = sha256(legacy)
    c = AnalysisCache(legacy)
    assert len(c.warnings) == 1 and "legacy automix v1 cache" in c.warnings[0]
    assert c.get(src, "key", version="key-v1", params=KEY_P) == {"camelot": "6A", "key_confidence": 0.9}
    c.put(src, "probe", {"codec": "mp3"}, version="probe-v1")
    c.save()                                                    # refuses to write over the legacy file
    assert sha256(legacy) == before and not list(legacy.parent.glob("*.corrupt-*"))
    assert not list(legacy.parent.glob(".*.tmp"))
    with pytest.raises(ValueError, match="legacy_path must differ"):
        AnalysisCache(legacy, legacy_path=legacy)


def test_missing_legacy_file_is_fine(tmp_path):
    c = AnalysisCache(tmp_path / "c.json", legacy_path=tmp_path / "nope.json")
    assert c.warnings == [] and len(c) == 0


# ---------------------------------------------------------------------------
# thread safety
# ---------------------------------------------------------------------------

def test_concurrent_puts_and_gets(tmp_path):
    sources = [make_source(tmp_path / "music" / f"t{i:02d}.wav", bytes([i]) * (100 + i)) for i in range(24)]
    path = tmp_path / "c.json"
    c = AnalysisCache(path, use_content_hash=True)
    errors: list[BaseException] = []

    def worker(chunk: list[Path]) -> None:
        try:
            for _ in range(5):
                for s in chunk:
                    c.put(s, "bpm", {"bpm": float(s.stat().st_size)}, version="bpm-v1", params=BPM_P)
                    assert c.get(s, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": float(s.stat().st_size)}
                c.save()
        except BaseException as e:  # pragma: no cover - only on failure
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(sources[i::4],)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    c.save()
    c2 = AnalysisCache(path, use_content_hash=True)
    assert all(c2.get(s, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": float(s.stat().st_size)}
               for s in sources)
    assert not list(tmp_path.glob(".*.tmp"))


# ---------------------------------------------------------------------------
# Review-finding regressions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["Demo Coral - Neon Avenue.mp3", "x.FLAC", "curator.csv", "profile.yaml", "cache"])
def test_cache_path_that_cannot_be_ours_is_refused(tmp_path, name):
    """SAFE-6: `--cache "music/<track>.mp3"` used to rename the track to *.corrupt-1 and
    write cache JSON under its name; a curator CSV met the same fate."""
    victim = make_source(tmp_path / name, b"ID3 precious bytes")
    with pytest.raises(ValueError, match=r"\.json"):
        AnalysisCache(victim)
    assert victim.read_bytes() == b"ID3 precious bytes"
    assert sorted(x.name for x in tmp_path.iterdir()) == [name]


@pytest.mark.parametrize("content", [
    '[{"rel_path": "a.mp3", "energy": 0.5}]',                                   # curator metadata (list form)
    '{"a.mp3": {"approved": true, "mood": "calm"}}',                             # curator metadata (dict form)
    '{"schema": "vegroove.manifest", "mode": "playlist", "tracks": []}',         # a VEgroove manifest
    '{"client": "x", "language": {"allowed": ["en"]}}',                          # a JSON client profile
])
def test_other_json_documents_at_the_cache_path_are_left_alone(tmp_path, content):
    """SAFE-6: a .json path that holds another document is read-only input, never renamed."""
    src = make_source(tmp_path / "a.wav")
    path = tmp_path / "other.json"
    path.write_text(content, encoding="utf-8")
    c = AnalysisCache(path)
    c.put(src, "bpm", {"bpm": 120.0}, version="bpm-v1", params=BPM_P)
    c.save()
    assert path.read_text(encoding="utf-8") == content
    assert not list(tmp_path.glob("*.corrupt-*")) and not list(tmp_path.glob(".*.tmp"))
    assert any("left untouched" in w for w in c.warnings)


def _truncated_legacy(tmp_path: Path, a: Path, b: Path) -> tuple[Path, bytes]:
    p = tmp_path / "bpm_cache.json"
    full = json.dumps({str(a): {"bpm": 101.0, "camelot": "6A", "key_confidence": 0.9},
                       str(b): {"bpm": 128.5, "camelot": "9B", "key_confidence": 0.7}}, indent=2)
    cut = full[: full.index("128.5") + 2]                     # v1 died mid-write: '... "bpm": 12'
    p.write_text(cut, encoding="utf-8")
    return p, p.read_bytes()


def test_truncated_legacy_file_given_as_cache_is_never_renamed_and_is_salvaged(tmp_path):
    """SAFE-11: a truncated automix v1 bpm_cache.json given via --cache used to be renamed
    to bpm_cache.json.corrupt-1 and overwritten with v2 JSON (losing the legacy keys)."""
    a = make_source(tmp_path / "music" / "a.mp3")
    b = make_source(tmp_path / "music" / "b.mp3", b"y" * 300)
    legacy, before = _truncated_legacy(tmp_path, a, b)
    c = AnalysisCache(legacy)
    assert any("read-only legacy" in w for w in c.warnings)
    assert any("salvaged 1 complete entry" in w for w in c.warnings)
    assert c.get(a, "key", version="key-v1", params=KEY_P) == {"camelot": "6A", "key_confidence": 0.9}
    assert c.get(b, "bpm", version="bpm-v1", params=BPM_P) is None     # '12' of '128.5' is not a BPM
    c.put(b, "bpm", {"bpm": 128.4}, version="bpm-v1", params=BPM_P)
    c.save()
    assert legacy.read_bytes() == before
    assert sorted(x.name for x in tmp_path.iterdir()) == ["bpm_cache.json", "music"]
    # the same file given as the legacy source of a separate v2 cache is salvaged as well
    c2 = AnalysisCache(tmp_path / "v2.json", legacy_path=legacy)
    assert c2.get(a, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 101.0}
    assert legacy.read_bytes() == before


def test_legacy_named_file_with_malformed_entries_is_used_read_only(tmp_path):
    """SAFE-11: a v1 file with a few malformed entries is still the v1 cache: never moved."""
    a = make_source(tmp_path / "music" / "a.mp3")
    legacy = tmp_path / "bpm_cache.json"
    legacy.write_text(json.dumps({str(a): 120.0, "x.mp3": "fast"}), encoding="utf-8")
    before = legacy.read_bytes()
    c = AnalysisCache(legacy)
    assert c.get(a, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 120.0}
    c.save()
    assert legacy.read_bytes() == before and not list(tmp_path.glob("*.corrupt-*"))


def test_read_only_cache_never_writes_or_renames(tmp_path):
    """SAFE-7: --dry-run used to quarantine (rename) a damaged cache. Read-only mode only
    reports it, and save() writes nothing even when there are new results."""
    src = make_source(tmp_path / "a.wav")
    path = tmp_path / "c.json"
    bad = _bad_cache_bytes("truncated", src)
    path.write_bytes(bad)
    c = AnalysisCache(path, read_only=True)
    assert len(c.warnings) == 1 and "would be moved aside to c.json.corrupt-1" in c.warnings[0]
    c.put(src, "bpm", {"bpm": 99.0}, version="bpm-v1", params=BPM_P)
    c.save()
    assert path.read_bytes() == bad
    assert sorted(x.name for x in tmp_path.iterdir()) == ["a.wav", "c.json"]

    good = tmp_path / "good.json"
    w = AnalysisCache(good)
    fill(w, src)
    w.save()
    saved = good.read_bytes()
    ro = AnalysisCache(good, read_only=True)
    assert ro.get(src, "bpm", version="bpm-v1", params=BPM_P) == {"bpm": 121.5}     # reads work
    ro.put(src, "bpm", {"bpm": 1.0}, version="bpm-v1", params=BPM_P)
    assert ro.dirty
    ro.save()
    assert good.read_bytes() == saved and not list(tmp_path.glob(".*.tmp"))
    assert not (tmp_path / "missing.json").exists()
    AnalysisCache(tmp_path / "missing.json", read_only=True).save()
    assert not (tmp_path / "missing.json").exists()


@pytest.mark.parametrize("content_hash", [False, True])
def test_result_computed_before_a_source_change_is_not_cached(tmp_path, content_hash):
    """SAFE-10: put() used to fingerprint the file AFTER the (long) analysis, so values of
    the old content were stored under the new file's fingerprint and reused forever."""
    src = make_source(tmp_path / "a.wav", b"A" * 4096)
    c = AnalysisCache(tmp_path / "c.json", use_content_hash=content_hash)
    fp0 = c.fingerprint(src)
    assert fp0 is not None and c.same_state(fp0, c.fingerprint(src))
    make_source(src, b"B" * 5000)                             # replaced while it was analysed
    assert c.put(src, "loudness", {"integrated_lufs": -18.3}, version="loudness-v1", params=LOUD_A,
                 expected_fingerprint=fp0) is False
    assert c.get(src, "loudness", version="loudness-v1", params=LOUD_A) is None
    fp1 = c.fingerprint(src)
    assert c.put(src, "loudness", {"integrated_lufs": -10.0}, version="loudness-v1", params=LOUD_A,
                 expected_fingerprint=fp1) is True
    assert c.get(src, "loudness", version="loudness-v1", params=LOUD_A) == {"integrated_lufs": -10.0}
    if content_hash:                                          # a touch with identical bytes is the same state
        bump_mtime(src)
        assert c.put(src, "bpm", {"bpm": 1.0}, version="bpm-v1", params=BPM_P, expected_fingerprint=fp1)


def test_salvage_never_takes_a_half_written_number():
    """SAFE-11: v1's oldest format is {path: bpm}; a value cut mid-number is dropped."""
    text = '{\n  "a.mp3": 120.0,\n  "b.mp3": {"bpm": 99.0, "camelot": "4A"},\n  "c.mp3": 12'
    assert cache_mod._salvage_legacy_entries(text) == {"a.mp3": 120.0, "b.mp3": {"bpm": 99.0, "camelot": "4A"}}
    assert cache_mod._salvage_legacy_entries('{"a.mp3": 120.0} trailing') == {"a.mp3": 120.0}
    assert cache_mod._salvage_legacy_entries("[1, 2") == {} and cache_mod._salvage_legacy_entries("") == {}
