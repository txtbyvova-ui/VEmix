"""Mode A playlist export (spec §4, §13 Playlist 1-5).

Real-FFmpeg integration (LoudnessNormalizer on short synthetic sources) for M3U8
syntax, relative paths, Unicode names, manifest ordering, untouched sources, failed
tracks and re-runs; a fake normalizer (no FFmpeg) for publishing-transaction and
edge-case logic that does not depend on audio.
"""

from __future__ import annotations

import codecs
import dataclasses
import json
import os
import re
import shutil
import threading
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import TOOLS, encode_mp3, music_like, requires_ffmpeg, sha256, write_wav
from veg_mixer import GENERATOR, playlist
from veg_mixer.config import LoudnessConfig
from veg_mixer.ffmpeg_tools import FFmpegNotFoundError, FFmpegTools, probe_audio
from veg_mixer.loudness import LoudnessNormalizer
from veg_mixer.models import (
    AudioProbe,
    DecodeCheck,
    FlaggedTransition,
    LoudnessMeasurement,
    NormalizationResult,
    NormalizationStatus,
    PlaylistManifest,
    SequencedTrack,
    SequenceResult,
    ShortTermStats,
    TrackFeatures,
    TransitionScore,
)
from veg_mixer.playlist import (
    PlaylistExportError,
    export_playlist,
    read_m3u8,
    render_m3u8,
    safe_filename,
    sanitize_component,
    write_m3u8,
)

CFG = LoudnessConfig()
TP_LIMIT = CFG.target_true_peak_dbtp + CFG.true_peak_tolerance_db
SECONDS = 5.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def feat(path: Path, artist: str | None, title: str | None, **kw) -> TrackFeatures:
    return TrackFeatures(path=Path(path), duration_s=kw.pop("duration_s", SECONDS), artist=artist, title=title,
                         rel_path=Path(path).name, **kw)


def make_sequence(tracks: list[TrackFeatures], *, daypart: str | None = "morning",
                  flagged: list[FlaggedTransition] | None = None) -> SequenceResult:
    seq = [SequencedTrack(position=i, track=t, daypart_fit=0.7,
                          transition_in=None if i == 1 else TransitionScore(
                              total=0.8, components={"energy_continuity": 0.9, "bpm_continuity": None},
                              coverage=0.75))
           for i, t in enumerate(tracks, 1)]
    return SequenceResult(daypart=daypart, tracks=seq, flagged=list(flagged or []),
                          total_duration_s=sum(t.duration_s for t in tracks), seed=11)


def snapshot(folder: Path) -> dict[str, tuple[str, int]]:
    """relative path -> (sha256, mtime_ns) of every file below ``folder``."""
    if not folder.exists():
        return {}
    return {p.relative_to(folder).as_posix(): (sha256(p), p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def load_manifest(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def make_source(tools, folder: Path, name: str, *, seed: int, peak_dbfs: float = -6.0,
                tags: dict | None = None) -> Path:
    """Synthetic WAV or MP3 source with a (possibly Unicode) file name."""
    folder.mkdir(parents=True, exist_ok=True)
    data = music_like(SECONDS, peak_dbfs=peak_dbfs, seed=seed, bpm=100 + 5 * seed, tone_hz=180 + 20 * seed)
    tmp_wav = write_wav(folder / "_fixture_tmp" / f"src{seed}.wav", data)
    dest = folder / name
    if dest.suffix.lower() == ".wav":
        os.replace(tmp_wav, dest)
    else:
        encode_mp3(tools, tmp_wav, dest, tags=tags)
        tmp_wav.unlink()
    return dest


def _cleanup_fixture_tmp(folder: Path) -> None:
    shutil.rmtree(folder / "_fixture_tmp", ignore_errors=True)


# ---------------------------------------------------------------------------
# Fake normalizer: real LoudnessNormalizer API, no FFmpeg
# ---------------------------------------------------------------------------

FAKE_TOOLS = FFmpegTools(ffmpeg="ffmpeg", ffprobe=None, version="fake", has_soxr=False, has_libmp3lame=True)
STEADY = ShortTermStats(p10=-20, p50=-17, p95=-14, max=-13, head_lufs=-17, tail_lufs=-17, frames=50)


class FakeNormalizer(LoudnessNormalizer):
    """Writes a small placeholder file instead of running FFmpeg. Per source file name:
    ``fail`` -> FAILED, ``review`` -> REVIEW, ``raise_for`` -> raise ``exc``,
    ``short_term`` -> output short-term stats, ``delay`` -> sleep seconds."""

    def __init__(self, *, fail=(), review=(), raise_for=(), exc: BaseException | None = None,
                 short_term: dict | None = None, delay: dict | None = None, forbid: bool = False) -> None:
        super().__init__(FAKE_TOOLS)
        self.fail, self.review, self.raise_for = set(fail), set(review), set(raise_for)
        self.exc = exc or RuntimeError("boom")
        self.short_term = short_term or {}
        self.delay = delay or {}
        self.forbid = forbid
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def normalize_to_mp3(self, source, dest, *, input_measurement=None, source_probe=None, metadata=None):
        if self.forbid:
            raise AssertionError("the normalizer must not be called")
        source, dest = Path(source), Path(dest)
        name = source.name
        with self._lock:
            self.calls.append({"source": source, "dest": dest, "metadata": dict(metadata or {}),
                               "input_measurement": input_measurement, "source_probe": source_probe})
        time.sleep(self.delay.get(name, 0.0))
        if name in self.raise_for:
            raise self.exc
        base = dict(source=source.resolve(), target_lufs=-16.0, target_true_peak_dbtp=-1.5)
        if name in self.fail or not source.is_file():
            return NormalizationResult(output=None, status=NormalizationStatus.FAILED, requires_review=True,
                                       errors=[f"FFmpegError: cannot decode {name}"], **base)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake mp3 of " + name.encode("utf-8"))
        m = LoudnessMeasurement(integrated_lufs=-16.1, true_peak_dbtp=-2.0, lra_lu=5.0, threshold_lufs=-26.0,
                                short_term=self.short_term.get(name, STEADY), duration_s=SECONDS)
        review = name in self.review
        return NormalizationResult(
            output=dest.resolve(), status=NormalizationStatus.REVIEW if review else NormalizationStatus.PASSED,
            requires_review=review, output_measurement=m, decode_check=DecodeCheck(ok=True, decoded_duration_s=SECONDS),
            review_reasons=["integrated loudness 19.0 LUFS below target: peak-bound"] if review else [], **base)


def fake_library(folder: Path, names: list[str]) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in names:
        p = folder / n
        p.write_bytes(b"source " + n.encode("utf-8"))
        paths.append(p)
    return paths


def fake_tracks(paths: list[Path]) -> list[TrackFeatures]:
    out = []
    for p in paths:
        artist, sep, title = p.stem.partition(" - ")
        out.append(feat(p, artist or None, title or None) if sep else feat(p, None, None))
    return out


# ---------------------------------------------------------------------------
# safe_filename / sanitize_component (pure)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("artist, title, fallback, expected", [
    ("Artist One", "Song A", "x", "001_Artist_One_Song_A.mp3"),
    (None, "Song A", "x", "001_Song_A.mp3"),
    ("AC/DC", 'What? <Live> "Edit" | a*b:c\\d', "x", "001_AC_DC_What_Live_Edit_a_b_c_d.mp3"),
    ("Artist", "Remix - Club   Edit", "x", "001_Artist_Remix-Club_Edit.mp3"),
    ("Line\nBreak\tTab", "Ctrl\x00\x07Chars", "x", "001_Line_Break_Tab_Ctrl_Chars.mp3"),
    ("Hidden\u200bZero\u202eWidth", "Title...", "x", "001_HiddenZeroWidth_Title.mp3"),
    ("Кино", "Группа крови", "x", "001_Кино_Группа_крови.mp3"),
    ("東京事変", "群青日和", "x", "001_東京事変_群青日和.mp3"),
    ("Hash#Tag", "100% Pure 🔥", "x", "001_Hash_Tag_100_Pure.mp3"),
    (None, None, "Some Artist - Some Title", "001_Some_Artist-Some_Title.mp3"),
    ("Artist", None, "stem only", "001_stem_only.mp3"),
    ("???", "***", "", "001_track.mp3"),
    ("", "   ", "...", "001_track.mp3"),
])
def test_safe_filename_cases(artist, title, fallback, expected):
    assert safe_filename(1, artist, title, fallback) == expected


def test_safe_filename_nfc_width_and_index_validation():
    nfd = unicodedata.normalize("NFD", "Beyoncé")
    name = safe_filename(7, nfd, "Déjà vu", "x")
    assert name == "007_Beyoncé_Déjà_vu.mp3" and unicodedata.is_normalized("NFC", name)
    assert safe_filename(1234, "A", "B", "x", width=4) == "1234_A_B.mp3"
    assert safe_filename(5, "A", "B", "x", width=4) == "0005_A_B.mp3"
    with pytest.raises(ValueError):
        safe_filename(0, "A", "B", "x")


def test_safe_filename_reserved_device_names():
    for dev in ("CON", "prn", "Aux", "NUL", "COM1", "lpt9", "CON.txt"):
        comp = sanitize_component(dev)
        assert comp.split(".")[0].upper() not in {"CON", "PRN", "AUX", "NUL", "COM1", "LPT9"}, comp
        assert comp.startswith(dev.split(".")[0])          # still readable
    assert sanitize_component("CON.txt") == "CON_.txt"
    # in a full name the index prefix already keeps the base away from device names
    for dev in ("CON", "nul", "COM1", "LPT1"):
        name = safe_filename(1, None, dev, "x")
        assert name == f"001_{dev}.mp3"
        assert name.split(".")[0].upper() not in playlist._RESERVED_NAMES


def test_safe_filename_trailing_dots_and_spaces_are_stripped():
    assert sanitize_component("Title. . .  ") == "Title"
    assert sanitize_component("  --Leading dash") == "Leading_dash"
    assert not safe_filename(1, "Artist.", "Title. ", "x").replace(".mp3", "").endswith((".", " "))


@pytest.mark.parametrize("artist, title", [
    ("A" * 300, "B" * 300),
    ("Очень длинное имя исполнителя " * 10, "Очень длинное название песни " * 10),
    ("青" * 200, "夜" * 200),
])
def test_safe_filename_very_long_is_capped(artist, title):
    name = safe_filename(12, artist, title, "x")
    assert len(name) <= 120 and len(name.encode("utf-8")) <= playlist.MAX_NAME_BYTES
    assert name.startswith("012_") and name.endswith(".mp3")
    stem = name[len("012_"):-len(".mp3")]
    assert "_" in stem and stem.split("_", 1)[1]           # the title keeps a share of the space
    assert not stem.endswith(("_", ".", "-", " "))
    assert len(safe_filename(1, "A" * 50, "B" * 50, "x", max_len=40)) <= 40


def test_plan_names_resolves_case_insensitive_collisions_and_foreign_files():
    tracks = [feat(Path(f"{i}.mp3"), a, t) for i, (a, t) in
              enumerate([("Artist", "Song"), ("Artist", "Song"), ("ARTIST", "song"), ("Other", "Song")])]
    names = playlist._plan_names([(k, t) for k, t in enumerate(tracks, 1)], 3, foreign={"004_other_song.mp3"})
    assert names == ["001_Artist_Song.mp3", "002_Artist_Song_2.mp3", "003_ARTIST_song_3.mp3",
                     "004_Other_Song_2.mp3"]


# ---------------------------------------------------------------------------
# M3U8 read/write (pure)
# ---------------------------------------------------------------------------

def _mt(index, file, title, artist=None, duration=5.0):
    from veg_mixer.models import ManifestTrack
    return ManifestTrack(index=index, file=file, title=title, artist=artist, duration_s=duration,
                         source_rel="s", source_path="s", track_id="t")


def test_write_and_read_m3u8_roundtrip(tmp_path):
    entries = [_mt(1, "tracks/001_a.mp3", "Song\nWith\r\nBreaks", "Art\u2028ist", 184.5),
               _mt(2, "tracks/002_b.mp3", "Été, à Paris", None, 59.4),
               _mt(3, "tracks/003_c.mp3", "Unknown length", None, None)]
    path = tmp_path / "p.m3u8"
    write_m3u8(path, entries, title="My\nList")
    raw = path.read_bytes()
    assert not raw.startswith(codecs.BOM_UTF8) and b"\r" not in raw
    lines = raw.decode("utf-8").split("\n")
    assert lines[:3] == ["#EXTM3U", "#PLAYLIST:My List", "#EXTINF:185,Art ist - Song With Breaks"]
    assert read_m3u8(path) == [(185.0, "Art ist - Song With Breaks", "tracks/001_a.mp3"),
                               (59.0, "Été, à Paris", "tracks/002_b.mp3"),
                               (None, "Unknown length", "tracks/003_c.mp3")]
    assert not list(tmp_path.glob(".*"))          # atomic write left no temp file


@pytest.mark.parametrize("bad", ["C:/music/a.mp3", "/abs/a.mp3", "\\\\server\\share\\a.mp3", "../a.mp3",
                                 "tracks/../../a.mp3", "http://x/a.mp3", "tracks/a\n.mp3"])
def test_write_m3u8_rejects_non_relative_paths(tmp_path, bad):
    with pytest.raises(ValueError):
        write_m3u8(tmp_path / "p.m3u8", [_mt(1, bad, "t")])
    assert not (tmp_path / "p.m3u8").exists()


def test_render_m3u8_converts_backslashes():
    assert "tracks/001_a.mp3" in render_m3u8([_mt(1, "tracks\\001_a.mp3", "t")])


@pytest.mark.parametrize("text, fragment", [
    ("", "missing #EXTM3U"),
    ("tracks/a.mp3\n", "missing #EXTM3U"),
    ("#EXTM3U\n#EXTINF:5,A\n", "without a path"),
    ("#EXTM3U\n#EXTINF:5,A\n#EXTINF:6,B\ntracks/b.mp3\n", "without a path"),
    ("#EXTM3U\n#EXTINF:abc,A\ntracks/a.mp3\n", "bad #EXTINF duration"),
    ("#EXTM3U\n#EXTINF:inf,A\ntracks/a.mp3\n", "bad #EXTINF duration"),
    ("#EXTM3U\n#EXTINF:5 A\ntracks/a.mp3\n", "malformed #EXTINF"),
])
def test_read_m3u8_rejects_syntax_errors(tmp_path, text, fragment):
    p = tmp_path / "bad.m3u8"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=fragment):
        read_m3u8(p)


def test_read_m3u8_rejects_invalid_utf8_and_accepts_plain_entries(tmp_path):
    p = tmp_path / "latin1.m3u8"
    p.write_bytes("#EXTM3U\n#EXTINF:5,Été\ntracks/a.mp3\n".encode("latin-1"))
    with pytest.raises(ValueError, match="UTF-8"):
        read_m3u8(p)
    q = tmp_path / "plain.m3u8"
    q.write_text("#EXTM3U\r\n# comment\r\ntracks/a.mp3\r\n#EXTINF:-1 tvg-id=\"x\",Live\r\ntracks/b.mp3\r\n",
                 encoding="utf-8")
    assert read_m3u8(q) == [(None, "", "tracks/a.mp3"), (None, "Live", "tracks/b.mp3")]


# ---------------------------------------------------------------------------
# ID3 track-number patch (pure, synthetic tags)
# ---------------------------------------------------------------------------

def _id3(major: int, frames: list[tuple[bytes, bytes]], padding: int = 10) -> bytes:
    def size(n):
        return playlist._to_synchsafe(n) if major == 4 else n.to_bytes(4, "big")
    body = b"".join(fid + size(len(p)) + b"\x00\x00" + p for fid, p in frames) + b"\x00" * padding
    return b"ID3" + bytes([major, 0, 0]) + playlist._to_synchsafe(len(body)) + body


@pytest.mark.parametrize("major", [3, 4])
def test_patch_id3_track_number_in_place(tmp_path, major):
    audio = bytes(range(256)) * 4
    tag = _id3(major, [(b"TIT2", b"\x01\xff\xfeA\x00"), (b"TRCK", b"\x0010/12\x00"), (b"TALB", b"\x00Album\x00")])
    p = tmp_path / "t.mp3"
    p.write_bytes(tag + audio)
    assert playlist._read_id3_text(p, "TRCK") == "10/12"
    assert playlist._patch_id3_text(p, "TRCK", "9/11")
    data = p.read_bytes()
    assert len(data) == len(tag) + len(audio) and data.endswith(audio)     # tag size kept, audio untouched
    assert playlist._read_id3_text(p, "TRCK") == "9/11"
    assert playlist._read_id3_text(p, "TALB") == "Album" and playlist._read_id3_text(p, "TIT2") == "A"
    assert not playlist._patch_id3_text(p, "TRCK", "123456/1234567")       # longer value: refused
    q = tmp_path / "no_tag.mp3"
    q.write_bytes(audio)
    assert not playlist._patch_id3_text(q, "TRCK", "1/2")


# ---------------------------------------------------------------------------
# Real FFmpeg: one export shared by the format/ordering/Unicode tests
# ---------------------------------------------------------------------------

LIBRARY = [  # (file name, artist, title, seed, peak dBFS)
    ("Artist One - Song A.mp3", "Artist One", "Song A", 1, -6.0),
    ("Ночной Город - Песня о ветре.wav", "Ночной Город", "Песня о ветре", 2, -14.0),
    ("Zoë Måns - Été à Paris.mp3", "Zoë Måns", "Été à Paris", 3, -3.0),
    ("青い鳥 - 夜明けの歌.mp3", "青い鳥", "夜明けの歌", 4, -10.0),
    ("dup copy.mp3", "Artist One", "Song A", 5, -6.0),
    ("dup case.mp3", "ARTIST ONE", "song a", 6, -8.0),
]


@pytest.fixture(scope="module")
def exported(ffmpeg_tools, tmp_path_factory):
    lib = tmp_path_factory.mktemp("library")
    out = tmp_path_factory.mktemp("export") / "Выход ünï"          # Unicode output folder
    tracks = []
    for name, artist, title, seed, peak in LIBRARY:
        src = make_source(ffmpeg_tools, lib, name, seed=seed, peak_dbfs=peak)
        tracks.append(feat(src, artist, title, bpm=100.0 + seed, camelot="8A", key_confidence=0.6,
                           feature_sources={"bpm": "auto:bpm-v1", "camelot": "auto:key-v1", "artist": "filename"}))
    _cleanup_fixture_tmp(lib)
    seq = make_sequence(tracks)
    # cached pass-1 probes, as the pipeline passes them (their tags decide whether album may be set)
    probes = {t.track_id: probe_audio(ffmpeg_tools, t.path) for t in tracks}
    before = snapshot(lib)
    calls: list[tuple[int, int, str]] = []
    result = export_playlist(seq, out, LoudnessNormalizer(ffmpeg_tools), profile_name="test_profile",
                             excluded=[], source_probes=probes, jobs=3,
                             progress=lambda d, t, m: calls.append((d, t, m)))
    return SimpleNamespace(lib=lib, out=out, seq=seq, result=result, before=before, progress=calls,
                           manifest=load_manifest(out), m3u8=read_m3u8(out / "playlist.m3u8"))


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_export_m3u8_syntax_is_valid_extended_m3u(exported):
    """Spec Playlist 1."""
    raw = (exported.out / "playlist.m3u8").read_bytes()
    assert not raw.startswith(codecs.BOM_UTF8) and b"\r" not in raw
    lines = raw.decode("utf-8").split("\n")
    assert lines[0] == "#EXTM3U"
    assert lines[1] == "#PLAYLIST:VEgroove - test_profile - morning"
    assert lines[-1] == ""                                    # trailing newline, nothing after
    body = lines[2:-1]
    assert len(body) == 2 * len(LIBRARY)
    for k in range(0, len(body), 2):
        extinf, path = body[k], body[k + 1]
        m = re.fullmatch(r"#EXTINF:(\d+),(.+)", extinf)
        assert m, extinf
        track = exported.manifest["tracks"][k // 2]
        assert int(m.group(1)) == round(track["duration_s"]) == round(SECONDS)
        assert m.group(2) == f"{track['artist']} - {track['title']}"
        assert path and not path.startswith("#") and path == track["file"]
    assert [(d, t) for d, t, _ in exported.m3u8] == [(float(round(SECONDS)), f"{a} - {t}")
                                                     for _, a, t, _, _ in LIBRARY]


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_export_paths_are_relative_and_resolve_inside_output(exported):
    """Spec Playlist 2."""
    root = exported.out.resolve()
    for _, _, rel in exported.m3u8:
        assert not os.path.isabs(rel) and not rel.startswith("/") and "\\" not in rel
        assert not re.match(r"^[A-Za-z]:", rel) and "%" not in rel     # no drive, no URL encoding
        assert rel.startswith("tracks/") and ".." not in rel.split("/")
        p = (exported.out / rel).resolve()
        assert p.is_relative_to(root) and p.is_file() and p.stat().st_size > 1000


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_export_unicode_filenames(exported):
    """Spec Playlist 3: Cyrillic / accents / CJK survive in file names, M3U8 and manifest."""
    names = sorted(os.listdir(exported.out / "tracks"))
    for fragment in ("Ночной_Город_Песня_о_ветре", "Zoë_Måns_Été_à_Paris", "青い鳥_夜明けの歌"):
        assert any(fragment in n for n in names), (fragment, names)
    assert all(unicodedata.is_normalized("NFC", n) for n in names)
    text = (exported.out / "playlist.m3u8").read_bytes().decode("utf-8", errors="strict")
    assert "Ночной Город - Песня о ветре" in text and "青い鳥 - 夜明けの歌" in text
    assert [rel for _, _, rel in exported.m3u8] == [t["file"] for t in exported.manifest["tracks"]]
    assert all((exported.out / rel).is_file() for _, _, rel in exported.m3u8)
    tags = probe_audio(TOOLS, exported.out / exported.manifest["tracks"][1]["file"]).tags
    assert tags.get("artist") == "Ночной Город" and tags.get("title") == "Песня о ветре"


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_export_manifest_ordering(exported):
    """Spec Playlist 4: indices 1..N == sequence order == M3U8 order == file numbering."""
    m = exported.manifest
    n = len(LIBRARY)
    assert [t["index"] for t in m["tracks"]] == list(range(1, n + 1))
    assert [t["track_id"] for t in m["tracks"]] == [st.track.track_id for st in exported.seq.tracks]
    assert [t["source_rel"] for t in m["tracks"]] == [name for name, *_ in LIBRARY]
    assert [rel for _, _, rel in exported.m3u8] == [t["file"] for t in m["tracks"]]
    for t in m["tracks"]:
        assert Path(t["file"]).name.startswith(f"{t['index']:03d}_")
    parsed = PlaylistManifest.from_dict(m)
    assert parsed.mode == "playlist" and parsed.generator == GENERATOR
    assert parsed.profile == "test_profile" and parsed.daypart == "morning"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", parsed.generated_at)
    assert m["loudness_targets"]["target_lufs"] == CFG.target_lufs
    assert m["loudness_targets"]["sample_rate"] == 44100 and m["loudness_targets"]["bitrate"] == "320k"
    assert m["excluded"] == [] and exported.result.manifest.tracks == parsed.tracks


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_export_sources_untouched(exported):
    """Spec Playlist 5: sha256 and mtime of every source unchanged, nothing added next to them."""
    assert snapshot(exported.lib) == exported.before


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_export_identical_artist_title_get_distinct_files(exported):
    files = [t["file"] for t in exported.manifest["tracks"]]
    assert len({f.casefold() for f in files}) == len(files)
    slugs = [Path(f).name.split("_", 1)[1].casefold() for f in files]
    assert len(set(slugs)) == len(slugs)                      # unique even without the index prefix
    assert Path(files[0]).name == "001_Artist_One_Song_A.mp3"
    assert Path(files[4]).name == "005_Artist_One_Song_A_2.mp3"
    assert Path(files[5]).name == "006_ARTIST_ONE_song_a_3.mp3"


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_export_loudness_metadata_features_and_tags(exported):
    n = len(LIBRARY)
    for k, t in enumerate(exported.manifest["tracks"]):
        loud = t["loudness"]
        assert loud["normalization_status"] == "passed", loud
        assert abs(loud["measured_lufs"] - CFG.target_lufs) <= CFG.lufs_tolerance
        assert loud["true_peak_dbtp"] <= TP_LIMIT and loud["original_lufs"] is not None
        assert t["duration_s"] == pytest.approx(SECONDS, abs=0.25)
        assert t["flags"] == []
        f = t["features"]
        assert f["energy"] is None and f["mood"] is None and f["language"] is None   # unknown stays unknown
        assert f["feature_sources"]["bpm"] == "auto:bpm-v1" and f["sequence_position"] == k + 1
        assert (t["transition_in"] is None) == (k == 0)
        if k:
            assert t["transition_in"]["components"]["bpm_continuity"] is None
        tags = probe_audio(TOOLS, exported.out / t["file"]).tags
        assert tags.get("track") == f"{k + 1}/{n}"
        assert tags.get("vegroove_playlist") == "VEgroove - test_profile - morning"
        assert tags.get("album") == "VEgroove - test_profile - morning"     # the sources have no album
    r = exported.result
    assert r.manifest_path == exported.out / "manifest.json" and not r.dry_run and not r.failed
    assert [x.output for x in r.results] == [exported.out / t["file"] for t in exported.manifest["tracks"]]
    text = (exported.out / "tracklist.txt").read_text(encoding="utf-8")
    assert "VEgroove playlist: VEgroove - test_profile - morning" in text
    assert "0:00:05" in text and "Ночной Город - Песня о ветре" in text and "passed" in text
    assert "skipping, repeat and seeking are" in text


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_export_leaves_no_temp_or_staging_files(exported):
    assert not [p for p in exported.out.rglob("*") if p.name.startswith(".")]
    assert len(os.listdir(exported.out / "tracks")) == len(LIBRARY)
    assert sorted(os.listdir(exported.out)) == ["manifest.json", "playlist.m3u8", "tracklist.txt", "tracks"]


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_export_progress_is_reported(exported):
    n = len(LIBRARY)
    assert exported.progress[0][:2] == (0, n)
    assert [d for d, _, _ in exported.progress[1:]] == list(range(1, n + 1))
    assert all(t == n for _, t, _ in exported.progress)


# ---------------------------------------------------------------------------
# Real FFmpeg: failed track, re-run
# ---------------------------------------------------------------------------

@requires_ffmpeg
@pytest.mark.ffmpeg
def test_failed_track_is_excluded_and_indices_renumbered(ffmpeg_tools, tmp_path):
    lib, out = tmp_path / "lib", tmp_path / "out"
    good1 = make_source(ffmpeg_tools, lib, "Good One - First.mp3", seed=11)
    corrupt = lib / "Broken - Signal.mp3"
    corrupt.write_bytes(b"ID3" + bytes(range(256)) * 40)
    good2 = make_source(ffmpeg_tools, lib, "Good Two - Second.mp3", seed=12)
    good3 = make_source(ffmpeg_tools, lib, "Good Three - Third.wav", seed=13)
    _cleanup_fixture_tmp(lib)
    tracks = fake_tracks([good1, corrupt, good2, good3])
    seq = make_sequence(tracks, flagged=[
        FlaggedTransition(1, 2, "Good One - First", "Broken - Signal", ["energy jump +0.45"], 0.3),
        FlaggedTransition(3, 4, "Good Two - Second", "Good Three - Third", ["genre change pop -> rock"], 0.4)])
    before = snapshot(lib)

    res = export_playlist(seq, out, LoudnessNormalizer(ffmpeg_tools), profile_name="p", jobs=2,
                          excluded=[{"source_rel": "banned.mp3", "reasons": ["blocklisted artist"]}])

    assert [r.status for r in res.results].count(NormalizationStatus.FAILED) == 1
    assert res.results[1].status is NormalizationStatus.FAILED and res.results[1].output is None
    entries = read_m3u8(out / "playlist.m3u8")
    m = load_manifest(out)
    assert len(entries) == 3 and [t["index"] for t in m["tracks"]] == [1, 2, 3]
    assert [t["source_rel"] for t in m["tracks"]] == [good1.name, good2.name, good3.name]
    assert [Path(rel).name for _, _, rel in entries] == ["001_Good_One_First.mp3", "002_Good_Two_Second.mp3",
                                                         "003_Good_Three_Third.mp3"]
    assert not any("Broken" in rel or "Signal" in rel for _, _, rel in entries)
    assert sorted(os.listdir(out / "tracks")) == sorted(Path(rel).name for _, _, rel in entries)

    stages = {e["source_rel"]: e for e in m["excluded"]}
    assert stages["banned.mp3"]["stage"] == "curation"
    failed = stages[corrupt.name]
    assert failed["stage"] == "normalization" and failed["sequence_position"] == 2
    assert failed["reasons"] and all(r.startswith("normalization failed: ") for r in failed["reasons"])
    assert any("Broken - Signal" in w and "normalization failed" in w for w in m["warnings"])

    kinds = {(f["kind"], f["from_index"], f["to_index"]) for f in m["flagged_transitions"]}
    assert ("sequencing", 2, 3) in kinds and ("unscored", 1, 2) in kinds
    assert not any(f["kind"] == "sequencing" and f["to_index"] == 2 for f in m["flagged_transitions"])
    second = m["tracks"][1]
    assert "listen_transition" in second["flags"] and second["transition_in"] is None
    assert m["tracks"][2]["transition_in"] is not None

    for k, (_, _, rel) in enumerate(entries, 1):           # ID3 numbers follow the final order
        assert probe_audio(ffmpeg_tools, out / rel).tags.get("track") == f"{k}/3"
    text = (out / "tracklist.txt").read_text(encoding="utf-8")
    assert corrupt.name in text and "normalization failed" in text and "[unscored]" in text
    assert snapshot(lib) == before


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_rerun_removes_stale_files_but_keeps_user_files(ffmpeg_tools, tmp_path):
    lib, out = tmp_path / "lib", tmp_path / "out"
    a = make_source(ffmpeg_tools, lib, "Alpha - One.mp3", seed=21)
    b = make_source(ffmpeg_tools, lib, "Beta - Two.mp3", seed=22)
    c = make_source(ffmpeg_tools, lib, "Gamma - Three.mp3", seed=23)
    _cleanup_fixture_tmp(lib)
    ta, tb, tc = fake_tracks([a, b, c])
    norm = LoudnessNormalizer(ffmpeg_tools)
    export_playlist(make_sequence([ta, tb, tc]), out, norm, profile_name="p", excluded=[], jobs=3)
    assert sorted(os.listdir(out / "tracks")) == ["001_Alpha_One.mp3", "002_Beta_Two.mp3", "003_Gamma_Three.mp3"]
    old_first = sha256(out / "tracks" / "001_Alpha_One.mp3")

    (out / "tracks" / "my_notes.txt").write_text("keep me", encoding="utf-8")
    (out / "tracks" / "my favourite.mp3").write_bytes(b"user audio")
    (out / "readme.txt").write_text("user readme", encoding="utf-8")
    user_before = {p: sha256(p) for p in (out / "tracks" / "my_notes.txt", out / "tracks" / "my favourite.mp3",
                                          out / "readme.txt")}

    res = export_playlist(make_sequence([ta, tc]), out, norm, profile_name="p", excluded=[], jobs=2)

    assert sorted(os.listdir(out / "tracks")) == ["001_Alpha_One.mp3", "002_Gamma_Three.mp3",
                                                  "my favourite.mp3", "my_notes.txt"]
    assert {p: sha256(p) for p in user_before} == user_before
    assert sha256(out / "tracks" / "001_Alpha_One.mp3") != old_first        # replaced by the new run
    assert [rel for _, _, rel in read_m3u8(out / "playlist.m3u8")] == ["tracks/001_Alpha_One.mp3",
                                                                        "tracks/002_Gamma_Three.mp3"]
    assert [t["source_rel"] for t in load_manifest(out)["tracks"]] == [a.name, c.name]
    assert load_manifest(out)["leftover_files"] == []        # shared-model field, always present now
    assert res.manifest_path is not None and not [p for p in out.rglob(".*")]


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_spec9_source_album_is_kept_and_playlist_name_goes_to_txxx(ffmpeg_tools, tmp_path):
    """SPEC-9: the export must not overwrite a source album tag with the playlist name."""
    lib, out = tmp_path / "lib", tmp_path / "out"
    tagged = make_source(ffmpeg_tools, lib, "Zed - Song.mp3", seed=31,
                         tags={"artist": "Zed", "title": "Song", "album": "Original Album 1999",
                               "date": "1999", "track": "7/12"})
    plain = make_source(ffmpeg_tools, lib, "Nobody - Untagged.mp3", seed=32)
    _cleanup_fixture_tmp(lib)
    tracks = fake_tracks([tagged, plain])
    probes = {t.track_id: probe_audio(ffmpeg_tools, t.path) for t in tracks}
    assert probes[tracks[0].track_id].tags.get("album") == "Original Album 1999"

    export_playlist(make_sequence(tracks), out, LoudnessNormalizer(ffmpeg_tools), profile_name="p",
                    excluded=[], source_probes=probes)

    m = load_manifest(out)
    first = probe_audio(ffmpeg_tools, out / m["tracks"][0]["file"]).tags
    assert first.get("album") == "Original Album 1999"                 # useful metadata preserved
    assert first.get("vegroove_playlist") == "VEgroove - p - morning"  # playlist name in its own frame
    assert first.get("date") == "1999" and first.get("artist") == "Zed" and first.get("title") == "Song"
    assert first.get("track") == "1/2"                                 # play order, by design
    second = probe_audio(ffmpeg_tools, out / m["tracks"][1]["file"]).tags
    assert second.get("album") == "VEgroove - p - morning"             # no source album: grouped by playlist
    assert second.get("vegroove_playlist") == "VEgroove - p - morning"


# ---------------------------------------------------------------------------
# Fake normalizer: transaction and edge cases (no FFmpeg)
# ---------------------------------------------------------------------------

def test_adjacent_loudness_jump_and_review_are_flagged_not_fixed(tmp_path):
    paths = fake_library(tmp_path / "lib", ["A - One.wav", "B - Two.wav", "C - Three.wav"])
    loud = ShortTermStats(p10=-12, p50=-9, p95=-6, max=-5, head_lufs=-7, tail_lufs=-7, frames=50)
    norm = FakeNormalizer(short_term={"B - Two.wav": loud}, review={"C - Three.wav"})
    res = export_playlist(make_sequence(fake_tracks(paths)), tmp_path / "out", norm, excluded=[])
    m = res.manifest
    loudness = [f for f in m.flagged_transitions if f["kind"] == "loudness"]
    assert {(f["from_index"], f["to_index"]) for f in loudness} == {(1, 2), (2, 3)}
    assert all("short-term loudness" in r or "boundary" in r for f in loudness for r in f["reasons"])
    assert m.tracks[1].flags == ["listen_transition"]
    assert m.tracks[2].flags == ["loudness_review", "listen_transition"]
    assert m.tracks[2].loudness["normalization_status"] == "review"
    assert any(w.startswith("track 3 (C - Three): loudness review") for w in m.warnings)
    assert len(read_m3u8(tmp_path / "out" / "playlist.m3u8")) == 3        # REVIEW tracks are included


def _probe(path: Path, **tags: str) -> AudioProbe:
    return AudioProbe(path=Path(path), duration_s=SECONDS, sample_rate=44100, channels=2, codec="mp3", tags=tags)


def test_metadata_passed_to_normalizer(tmp_path):
    # SPEC-9: the playlist name goes to TXXX:VEGROOVE_PLAYLIST; album only when the source has none
    # (the old expectation album="VEgroove - ..." for every track was the bug).
    paths = fake_library(tmp_path / "lib", ["A - One.wav", "untitled.wav", "C - Three.mp3", "D - Four.mp3",
                                            "E - Five.mp3"])
    tracks = fake_tracks(paths)
    m_in = LoudnessMeasurement(integrated_lufs=-20.0, true_peak_dbtp=-6.0, lra_lu=5.0, threshold_lufs=-30.0)
    probes = {tracks[1].track_id: _probe(paths[1]),                                   # no album in the source
              tracks[2].track_id: _probe(paths[2], album="Original Album 1999"),      # useful album: kept
              tracks[3].track_id: _probe(paths[3], album="VEgroove - old - morning",  # an earlier VEgroove export
                                         vegroove_playlist="VEgroove - old - morning"),
              tracks[4].track_id: _probe(paths[4], album="  ")}                       # blank album
    norm = FakeNormalizer()
    res = export_playlist(make_sequence(tracks, daypart="evening"), tmp_path / "out", norm, profile_name="club",
                          excluded=[], input_measurements={tracks[0].track_id: m_in}, source_probes=probes)
    name = "VEgroove - club - evening"
    calls = {c["source"].name: c for c in norm.calls}
    # no probe: the source tags are unknown, so its album is left alone
    assert calls["A - One.wav"]["metadata"] == {"track": "1/5", "VEGROOVE_PLAYLIST": name,
                                                "title": "One", "artist": "A"}
    assert calls["A - One.wav"]["input_measurement"] is m_in and calls["A - One.wav"]["source_probe"] is None
    assert calls["untitled.wav"]["metadata"] == {"track": "2/5", "VEGROOVE_PLAYLIST": name, "album": name}
    assert calls["untitled.wav"]["input_measurement"] is None
    assert calls["untitled.wav"]["source_probe"] is probes[tracks[1].track_id]
    assert "album" not in calls["C - Three.mp3"]["metadata"]
    assert calls["C - Three.mp3"]["metadata"]["VEGROOVE_PLAYLIST"] == name
    assert calls["D - Four.mp3"]["metadata"]["album"] == name
    assert calls["E - Five.mp3"]["metadata"]["album"] == name

    # source tags not copied at all: the output has no album to lose, so the playlist name is used
    norm2 = FakeNormalizer()
    norm2.encode = dataclasses.replace(norm2.encode, copy_tags=False)
    export_playlist(make_sequence(tracks[2:3]), tmp_path / "out2", norm2, profile_name="club", excluded=[],
                    source_probes=probes)
    assert norm2.calls[0]["metadata"]["album"] == "VEgroove - club - morning"
    t2 = res.manifest.tracks[1]
    assert t2.artist is None and t2.title == "untitled" and t2.file == "tracks/002_untitled.mp3"
    assert read_m3u8(tmp_path / "out" / "playlist.m3u8")[1][1] == "untitled"


def test_parallel_jobs_keep_sequence_order(tmp_path):
    names = [f"Artist{i} - Song{i}.wav" for i in range(6)]
    paths = fake_library(tmp_path / "lib", names)
    norm = FakeNormalizer(delay={n: 0.02 * (6 - i) for i, n in enumerate(names)})   # later tracks finish first
    res = export_playlist(make_sequence(fake_tracks(paths)), tmp_path / "out", norm, excluded=[], jobs=4)
    assert [t.source_rel for t in res.manifest.tracks] == names
    assert [r.source.name for r in res.results] == names
    for t in res.manifest.tracks:
        content = (tmp_path / "out" / t.file).read_bytes()
        assert content == b"fake mp3 of " + t.source_rel.encode("utf-8")   # file matches its index


def test_dry_run_writes_nothing(tmp_path):
    paths = fake_library(tmp_path / "lib", ["A - One.wav", "B - Two.wav"])
    seq = make_sequence(fake_tracks(paths))
    out = tmp_path / "out"
    res = export_playlist(seq, out, FakeNormalizer(forbid=True), profile_name="p", excluded=[], dry_run=True)
    assert not out.exists()
    assert res.dry_run and res.results == [] and res.manifest_path is None and res.m3u8_path is None
    assert [t.file for t in res.manifest.tracks] == ["tracks/001_A_One.mp3", "tracks/002_B_Two.mp3"]
    assert res.manifest.tracks[0].duration_s is None and res.manifest.tracks[0].loudness == {}
    assert "dry run: nothing was written" in res.manifest.warnings
    # normalizer may be None for a dry run
    assert export_playlist(seq, out, None, excluded=[], dry_run=True).dry_run and not out.exists()

    # an existing export stays byte-identical; the stale-file preview is reported
    export_playlist(make_sequence(fake_tracks(paths + fake_library(tmp_path / "lib", ["C - Three.wav"]))),
                    out, FakeNormalizer(), excluded=[])
    before = snapshot(out)
    res2 = export_playlist(seq, out, FakeNormalizer(forbid=True), excluded=[], dry_run=True)
    assert snapshot(out) == before
    assert any("would be removed: tracks/003_C_Three.mp3" in w for w in res2.manifest.warnings)


def test_source_inside_output_tracks_is_refused_before_processing(tmp_path):
    out = tmp_path / "out"
    lib_paths = fake_library(tmp_path / "lib", ["A - One.wav"])
    inside = fake_library(out / "tracks", ["001_B_Two.mp3"])
    before = snapshot(out)
    norm = FakeNormalizer(forbid=True)
    with pytest.raises(PlaylistExportError, match="inside the output tracks folder"):
        export_playlist(make_sequence(fake_tracks(lib_paths + inside)), out, norm, excluded=[])
    assert snapshot(out) == before and not norm.calls


@pytest.mark.parametrize("content", ['{"name": "my web app"}', "{not json", '{"schema": "vegroove.manifest", '
                                                                            '"mode": "mix", "tracks": []}'])
def test_foreign_manifest_is_never_overwritten(tmp_path, content):
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.json").write_text(content, encoding="utf-8")
    paths = fake_library(tmp_path / "lib", ["A - One.wav"])
    with pytest.raises(PlaylistExportError, match="refusing to overwrite"):
        export_playlist(make_sequence(fake_tracks(paths)), out, FakeNormalizer(forbid=True), excluded=[])
    assert (out / "manifest.json").read_text(encoding="utf-8") == content and os.listdir(out) == ["manifest.json"]


def test_user_file_with_planned_name_is_not_overwritten(tmp_path):
    out = tmp_path / "out"
    user = out / "tracks" / "001_a_one.mp3"            # differs only in case from the planned name
    user.parent.mkdir(parents=True)
    user.write_bytes(b"user's own file")
    paths = fake_library(tmp_path / "lib", ["A - One.wav"])
    res = export_playlist(make_sequence(fake_tracks(paths)), out, FakeNormalizer(), excluded=[])
    assert user.read_bytes() == b"user's own file"
    assert res.manifest.tracks[0].file == "tracks/001_A_One_2.mp3"
    assert (out / res.manifest.tracks[0].file).read_bytes().startswith(b"fake mp3")


def test_all_failed_publishes_nothing_and_keeps_previous_output(tmp_path):
    paths = fake_library(tmp_path / "lib", ["A - One.wav", "B - Two.wav"])
    seq = make_sequence(fake_tracks(paths))
    fresh = tmp_path / "fresh"
    res = export_playlist(seq, fresh, FakeNormalizer(fail={"A - One.wav", "B - Two.wav"}), excluded=[])
    assert not fresh.exists()                            # no empty folders left behind
    assert res.manifest_path is None and len(res.failed) == 2
    assert [e["stage"] for e in res.manifest.excluded] == ["normalization", "normalization"]
    assert any("nothing was published" in w for w in res.manifest.warnings)

    out = tmp_path / "out"
    export_playlist(seq, out, FakeNormalizer(), excluded=[])
    before = snapshot(out)
    res2 = export_playlist(seq, out, FakeNormalizer(fail={"A - One.wav", "B - Two.wav"}), excluded=[])
    assert res2.m3u8_path is None and snapshot(out) == before


def test_unexpected_normalizer_errors_are_reported_and_ffmpeg_missing_propagates(tmp_path):
    paths = fake_library(tmp_path / "lib", ["A - One.wav", "B - Two.wav", "C - Three.wav"])
    seq = make_sequence(fake_tracks(paths))
    out = tmp_path / "out"
    res = export_playlist(seq, out, FakeNormalizer(raise_for={"B - Two.wav"}), excluded=[])
    failed = [e for e in res.manifest.excluded if e["stage"] == "normalization"]
    assert failed[0]["source_rel"] == "B - Two.wav"
    assert failed[0]["reasons"] == ["normalization failed: unexpected RuntimeError: boom"]
    assert [t.index for t in res.manifest.tracks] == [1, 2]

    before = snapshot(out)
    with pytest.raises(FFmpegNotFoundError):
        export_playlist(seq, out, FakeNormalizer(raise_for={"A - One.wav"}, exc=FFmpegNotFoundError("no ffmpeg")),
                        excluded=[])
    assert snapshot(out) == before and not list((out / "tracks").glob(".veg-staging-*"))


def test_publish_failure_rolls_back_to_previous_output(tmp_path, monkeypatch):
    paths = fake_library(tmp_path / "lib", ["A - One.wav", "B - Two.wav", "C - Three.wav"])
    tracks = fake_tracks(paths)
    out = tmp_path / "out"
    export_playlist(make_sequence(tracks), out, FakeNormalizer(), excluded=[])
    before = snapshot(out)

    real_write_temp = playlist._write_temp

    def failing_write_temp(directory, name, data):
        if name == "playlist.m3u8":                       # manifest already moved in at this point
            raise OSError("disk full")
        return real_write_temp(directory, name, data)

    monkeypatch.setattr(playlist, "_write_temp", failing_write_temp)
    with pytest.raises(OSError, match="disk full"):
        export_playlist(make_sequence([tracks[2], tracks[0]]), out, FakeNormalizer(), excluded=[])
    assert snapshot(out) == before                         # bytes and mtimes of the previous run
    assert not [p for p in out.rglob(".*")]


def test_stale_file_that_cannot_be_removed_stays_owned(tmp_path, monkeypatch):
    paths = fake_library(tmp_path / "lib", ["A - One.wav", "B - Two.wav", "C - Three.wav"])
    tracks = fake_tracks(paths)
    out = tmp_path / "out"
    export_playlist(make_sequence(tracks), out, FakeNormalizer(), excluded=[])

    real_move = playlist._Journal.move

    def locked(self, src, dst):
        if Path(src).name == "003_C_Three.mp3":
            raise PermissionError("in use by a player")
        return real_move(self, src, dst)

    monkeypatch.setattr(playlist._Journal, "move", locked)
    res = export_playlist(make_sequence(tracks[:2]), out, FakeNormalizer(), excluded=[])
    m = load_manifest(out)
    assert m["leftover_files"] == ["tracks/003_C_Three.mp3"]
    assert any("003_C_Three.mp3" in w and "next export" in w for w in res.manifest.warnings)
    assert (out / "tracks" / "003_C_Three.mp3").exists() and len(m["tracks"]) == 2

    monkeypatch.setattr(playlist._Journal, "move", real_move)
    export_playlist(make_sequence(tracks[:1]), out, FakeNormalizer(), excluded=[])
    assert sorted(os.listdir(out / "tracks")) == ["001_A_One.mp3"]
    assert load_manifest(out)["leftover_files"] == []


def test_leftover_files_of_an_older_manifest_are_still_owned(tmp_path):
    """Backward compatibility: ``leftover_files`` written before it was a model field."""
    paths = fake_library(tmp_path / "lib", ["A - One.wav", "B - Two.wav"])
    tracks = fake_tracks(paths)
    out = tmp_path / "out"
    export_playlist(make_sequence(tracks[:1]), out, FakeNormalizer(), excluded=[])
    (out / "tracks" / "009_Old_Locked.mp3").write_bytes(b"old run")
    data = load_manifest(out)
    del data["leftover_files"], data["not_selected"]                  # an older manifest layout ...
    data["leftover_files"] = ["tracks/009_Old_Locked.mp3"]            # ... with the ad-hoc key
    (out / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    export_playlist(make_sequence(tracks), out, FakeNormalizer(), excluded=[])
    assert sorted(os.listdir(out / "tracks")) == ["001_A_One.mp3", "002_B_Two.mp3"]


def test_bad_jobs_are_rejected(tmp_path):
    paths = fake_library(tmp_path / "lib", ["A - One.wav"])
    with pytest.raises(ValueError, match="jobs"):
        export_playlist(make_sequence(fake_tracks(paths)), tmp_path / "out", FakeNormalizer(), excluded=[], jobs=0)
    with pytest.raises(ValueError, match="jobs"):
        export_playlist(SequenceResult(daypart=None, tracks=[]), tmp_path / "out", FakeNormalizer(), excluded=[],
                        jobs=0)
    assert not (tmp_path / "out").exists()


def test_int12_empty_sequence_returns_an_unpublished_result_with_a_reason(tmp_path):
    """INT-12: an empty daypart must not raise (it aborted every later daypart); nothing is
    written, the result has no paths and its manifest says why. (Replaces the old
    'empty sequence is rejected with ValueError' expectation, which the finding proved wrong.)"""
    out = tmp_path / "out"
    empty = SequenceResult(daypart="evening", tracks=[])
    ns = [{"rel_path": "Drone Study.mp3", "track_id": "t1", "reason": "poor fit for daypart 'evening'"}]
    for dry in (False, True):
        res = export_playlist(empty, out, FakeNormalizer(forbid=True), profile_name="p", dry_run=dry,
                              excluded=[{"source_rel": "banned.mp3", "reasons": ["blocklisted"]}], not_selected=ns)
        assert not out.exists()
        assert res.dry_run is dry and res.results == [] and not res.failed
        assert res.manifest_path is None and res.m3u8_path is None and res.tracklist_path is None
        m = res.manifest
        assert m.tracks == [] and m.daypart == "evening" and m.profile == "p"
        assert any("evening" in w and "contains no tracks" in w and "nothing was exported" in w for w in m.warnings)
        assert m.excluded == [{"source_rel": "banned.mp3", "reasons": ["blocklisted"], "stage": "curation"}]
        assert m.not_selected[0]["source_rel"] == "Drone Study.mp3"
    assert export_playlist(empty, out, None, excluded=[], dry_run=True).manifest_path is None

    # an earlier output for that daypart stays byte-identical and is reported as stale;
    # nothing is checked either (a folder the exporter would refuse does not raise)
    paths = fake_library(tmp_path / "lib", ["A - One.wav"])
    export_playlist(make_sequence(fake_tracks(paths), daypart="evening"), out, FakeNormalizer(), excluded=[])
    before = snapshot(out)
    res = export_playlist(empty, out, FakeNormalizer(forbid=True), excluded=[])
    assert snapshot(out) == before
    assert any("earlier run" in w and "does NOT describe this run" in w for w in res.manifest.warnings)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "manifest.json").write_text('{"name": "my web app"}', encoding="utf-8")
    assert export_playlist(empty, foreign, FakeNormalizer(forbid=True), excluded=[]).manifest_path is None
    assert (foreign / "manifest.json").read_text(encoding="utf-8") == '{"name": "my web app"}'


def test_spec4_int15_not_selected_tracks_are_recorded_in_manifest_and_tracklist(tmp_path):
    """SPEC-4 / INT-15: eligible tracks the sequencer did not place must not vanish."""
    paths = fake_library(tmp_path / "lib", ["A - One.wav", "B - Two.wav"])
    seq = make_sequence(fake_tracks(paths))
    ns = [{"rel_path": "Demo Field - Drone Study No. 3.mp3", "track_id": "abc",
           "reason": "poor fit for daypart 'morning': selection score 0.10"},
          {"source_rel": "Late - Night.mp3", "track_id": "def", "reason": "target duration 1000 s reached"}]
    original = json.loads(json.dumps(ns))
    out = tmp_path / "out"
    res = export_playlist(seq, out, FakeNormalizer(), excluded=[], not_selected=ns)
    m = load_manifest(out)
    assert m["not_selected"] == [
        {"rel_path": "Demo Field - Drone Study No. 3.mp3", "source_rel": "Demo Field - Drone Study No. 3.mp3",
         "track_id": "abc", "reason": "poor fit for daypart 'morning': selection score 0.10"},
        {"source_rel": "Late - Night.mp3", "track_id": "def", "reason": "target duration 1000 s reached"}]
    assert PlaylistManifest.from_dict(m).not_selected == m["not_selected"] == res.manifest.not_selected
    assert ns == original                                     # the caller's list is not modified
    text = (out / "tracklist.txt").read_text(encoding="utf-8")
    assert "2 in playlist, 0 failed normalization, 0 excluded, 2 not selected" in text   # VSPEC-3 wording
    assert "=== NOT SELECTED" in text
    assert "- Demo Field - Drone Study No. 3.mp3: poor fit for daypart 'morning'" in text
    assert "- Late - Night.mp3: target duration 1000 s reached" in text

    preview = export_playlist(seq, out, None, excluded=[], not_selected=ns, dry_run=True).manifest
    assert [x["source_rel"] for x in preview.not_selected] == ["Demo Field - Drone Study No. 3.mp3",
                                                               "Late - Night.mp3"]
    failed = export_playlist(seq, tmp_path / "out_failed", FakeNormalizer(fail={"A - One.wav", "B - Two.wav"}),
                             excluded=[], not_selected=ns)
    assert failed.manifest_path is None and len(failed.manifest.not_selected) == 2

    export_playlist(seq, out, FakeNormalizer(), excluded=[])          # no list given: nothing recorded
    assert load_manifest(out)["not_selected"] == []
    assert "=== NOT SELECTED (eligible, not placed in this playlist) ===\n(none)" in \
        (out / "tracklist.txt").read_text(encoding="utf-8")


def test_every_excluded_entry_carries_a_stage(tmp_path):
    paths = fake_library(tmp_path / "lib", ["A - One.wav", "B - Two.wav"])
    excluded = [{"source_rel": "banned.mp3", "reasons": ["blocklisted artist"]},
                {"source_rel": "broken.mp3", "reasons": ["cannot decode"], "stage": "analysis"},
                {"source_rel": "blank.mp3", "reasons": ["x"], "stage": None},
                {"source_rel": "empty.mp3", "reasons": ["y"], "stage": "  "},
                "free-text exclusion"]
    out = tmp_path / "out"
    export_playlist(make_sequence(fake_tracks(paths)), out, FakeNormalizer(fail={"B - Two.wav"}), excluded=excluded)
    entries = load_manifest(out)["excluded"]
    assert [(e.get("source_rel"), e["stage"]) for e in entries] == [
        ("banned.mp3", "curation"), ("broken.mp3", "analysis"), ("blank.mp3", "curation"),
        ("empty.mp3", "curation"), (None, "curation"), ("B - Two.wav", "normalization")]
    assert entries[4]["reasons"] == ["free-text exclusion"]
    assert entries[5]["loudness"]["normalization_status"] == "failed"
    assert excluded[2]["stage"] is None                        # caller's entries are copied, not modified


def test_only_abandoned_staging_folders_are_removed(tmp_path):
    out = tmp_path / "out"
    old = out / "tracks" / ".veg-staging-crashed"
    fresh = out / "tracks" / ".veg-staging-running"
    for d in (old, fresh):
        d.mkdir(parents=True)
        (d / "000001.mp3").write_bytes(b"partial")
    age = playlist.STALE_STAGING_AGE_S + 60
    os.utime(old, (time.time() - age, time.time() - age))
    paths = fake_library(tmp_path / "lib", ["A - One.wav"])
    export_playlist(make_sequence(fake_tracks(paths)), out, FakeNormalizer(), excluded=[])
    assert not old.exists()
    assert (fresh / "000001.mp3").read_bytes() == b"partial"     # may belong to a running export


# ---------------------------------------------------------------------------
# SAFE-5: foreign playlist.m3u8 / tracklist.txt are never replaced
# ---------------------------------------------------------------------------

HAND_MADE_M3U = "#EXTM3U\n#EXTINF:300,Client - Their hand-made list\nmusic/a.mp3\n"
CLIENT_NOTES = "client notes, do not lose\n"


@pytest.mark.parametrize("files", [{"playlist.m3u8": HAND_MADE_M3U}, {"tracklist.txt": CLIENT_NOTES},
                                   {"playlist.m3u8": HAND_MADE_M3U, "tracklist.txt": CLIENT_NOTES}])
def test_safe5_foreign_playlist_files_are_never_overwritten(tmp_path, files):
    """SAFE-5: without a VEgroove playlist manifest, a client's own playlist.m3u8 /
    tracklist.txt used to be moved into staging and deleted with it."""
    out = tmp_path / "player"
    out.mkdir()
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    before = snapshot(out)
    paths = fake_library(tmp_path / "lib", ["A - One.wav"])
    norm = FakeNormalizer(forbid=True)                       # refused before any processing
    for dry in (True, False):
        with pytest.raises(PlaylistExportError, match="not written by VEgroove.*refusing to overwrite"):
            export_playlist(make_sequence(fake_tracks(paths)), out, norm, excluded=[], dry_run=dry)
    assert snapshot(out) == before and sorted(os.listdir(out)) == sorted(files)


def test_safe5_folder_named_like_an_output_file_is_never_moved(tmp_path):
    paths = fake_library(tmp_path / "lib", ["A - One.wav"])
    out = tmp_path / "out"
    export_playlist(make_sequence(fake_tracks(paths)), out, FakeNormalizer(), excluded=[])
    (out / "tracklist.txt").unlink()
    (out / "tracklist.txt").mkdir()                          # the user put a folder under that name
    (out / "tracklist.txt" / "keep.txt").write_text("user data", encoding="utf-8")
    before = snapshot(out)
    with pytest.raises(PlaylistExportError, match="tracklist.txt"):
        export_playlist(make_sequence(fake_tracks(paths)), out, FakeNormalizer(forbid=True), excluded=[])
    assert snapshot(out) == before


def test_safe5_vegroove_files_of_an_interrupted_publish_are_replaced(tmp_path):
    """No manifest, but playlist.m3u8 / tracklist.txt carry the VEgroove header (an earlier
    publish was interrupted after moving the manifest aside): they are the exporter's own."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "playlist.m3u8").write_text(render_m3u8([], "VEgroove - p - morning"), encoding="utf-8")
    (out / "tracklist.txt").write_text("VEgroove playlist: VEgroove - p - morning\nGenerated: ...\n",
                                       encoding="utf-8")
    paths = fake_library(tmp_path / "lib", ["A - One.wav"])
    res = export_playlist(make_sequence(fake_tracks(paths)), out, FakeNormalizer(), profile_name="p", excluded=[])
    assert res.manifest_path is not None
    assert [rel for _, _, rel in read_m3u8(out / "playlist.m3u8")] == ["tracks/001_A_One.mp3"]
    text = (out / "tracklist.txt").read_text(encoding="utf-8")
    assert "Generated: ..." not in text and "A - One" in text


def test_safe5_foreign_playlist_file_appearing_during_the_export_is_not_overwritten(tmp_path):
    out = tmp_path / "out"

    class UserDropsAPlaylist(FakeNormalizer):
        def normalize_to_mp3(self, source, dest, **kw):
            out.mkdir(exist_ok=True)
            (out / "playlist.m3u8").write_text(HAND_MADE_M3U, encoding="utf-8")
            return super().normalize_to_mp3(source, dest, **kw)

    paths = fake_library(tmp_path / "lib", ["A - One.wav"])
    with pytest.raises(PlaylistExportError, match="playlist.m3u8.*refusing to overwrite"):
        export_playlist(make_sequence(fake_tracks(paths)), out, UserDropsAPlaylist(), excluded=[])
    assert (out / "playlist.m3u8").read_text(encoding="utf-8") == HAND_MADE_M3U
    assert os.listdir(out) == ["playlist.m3u8"]               # nothing published, no staging left


# ---------------------------------------------------------------------------
# SAFE-16: a folder kept for manual recovery is never auto-deleted
# ---------------------------------------------------------------------------

def test_safe16_incomplete_rollback_keeps_a_recovery_folder_that_is_never_auto_deleted(tmp_path, monkeypatch):
    paths = fake_library(tmp_path / "lib", ["A - One.wav", "B - Two.wav", "C - Three.wav"])
    tracks = fake_tracks(paths)
    out = tmp_path / "out"
    export_playlist(make_sequence(tracks), out, FakeNormalizer(), excluded=[])
    old_first = (out / "tracks" / "001_A_One.mp3").read_bytes()

    real_write_temp, real_replace = playlist._write_temp, os.replace

    def failing_write_temp(directory, name, data):
        if name == "playlist.m3u8":
            raise OSError("disk full")
        return real_write_temp(directory, name, data)

    def locked_on_undo(src, dst):
        # the undo of "tracks/001_A_One.mp3 -> <staging>/previous/00000.mp3" fails (file locked)
        if Path(src).parent.name == "previous" and Path(src).name == "00000.mp3":
            raise PermissionError("locked by a player")
        return real_replace(src, dst)

    monkeypatch.setattr(playlist, "_write_temp", failing_write_temp)
    monkeypatch.setattr(playlist.os, "replace", locked_on_undo)
    with pytest.raises(PlaylistExportError, match=r"could not be fully undone.*\.veg-recovery-") as info:
        export_playlist(make_sequence([tracks[2], tracks[0]]), out, FakeNormalizer(), excluded=[])
    monkeypatch.undo()

    names = os.listdir(out / "tracks")
    assert not [n for n in names if n.startswith(playlist.STAGING_PREFIX)]
    kept = [n for n in names if n.startswith(playlist.RECOVERY_PREFIX)]
    assert len(kept) == 1
    recovery = out / "tracks" / kept[0]
    assert info.value.recovery_dir == recovery and "never deleted automatically" in str(info.value)
    assert (recovery / "previous" / "00000.mp3").read_bytes() == old_first     # the only copy of that file
    note = (recovery / playlist.RECOVERY_NOTE).read_text(encoding="utf-8")
    assert "previous/00000.mp3" in note and "001_A_One.mp3" in note and "disk full" in note

    age = playlist.STALE_STAGING_AGE_S + 3600                # long idle: still not removed
    os.utime(recovery, (time.time() - age, time.time() - age))
    res = export_playlist(make_sequence(tracks), out, FakeNormalizer(), excluded=[])
    assert (recovery / "previous" / "00000.mp3").read_bytes() == old_first
    assert any(kept[0] in w and "manual recovery" in w for w in res.manifest.warnings)
    assert any(kept[0] in w for w in load_manifest(out)["warnings"])
    dry = export_playlist(make_sequence(tracks), out, None, excluded=[], dry_run=True)
    assert any(kept[0] in w for w in dry.manifest.warnings)


def test_safe16_staging_folder_marked_for_recovery_is_not_removed(tmp_path):
    """Fallback when the rename to .veg-recovery-* failed: RECOVERY.txt alone protects it."""
    out = tmp_path / "out"
    marked = out / "tracks" / ".veg-staging-failedrename"
    (marked / "previous").mkdir(parents=True)
    (marked / "previous" / "00000.mp3").write_bytes(b"only copy")
    (marked / playlist.RECOVERY_NOTE).write_text("recovery note", encoding="utf-8")
    age = playlist.STALE_STAGING_AGE_S + 60
    os.utime(marked, (time.time() - age, time.time() - age))
    paths = fake_library(tmp_path / "lib", ["A - One.wav"])
    res = export_playlist(make_sequence(fake_tracks(paths)), out, FakeNormalizer(), excluded=[])
    assert (marked / "previous" / "00000.mp3").read_bytes() == b"only copy"
    assert any(".veg-staging-failedrename" in w and "manual recovery" in w for w in res.manifest.warnings)


def test_previous_manifest_entry_that_is_now_a_folder_is_left_untouched(tmp_path):
    paths = fake_library(tmp_path / "lib", ["A - One.wav", "B - Two.wav"])
    tracks = fake_tracks(paths)
    out = tmp_path / "out"
    export_playlist(make_sequence(tracks), out, FakeNormalizer(), excluded=[])
    stale = out / "tracks" / "002_B_Two.mp3"
    stale.unlink()
    stale.mkdir()                                     # the user put a folder under that name
    (stale / "keep.txt").write_text("user data", encoding="utf-8")
    res = export_playlist(make_sequence(tracks[:1]), out, FakeNormalizer(), excluded=[])
    assert (stale / "keep.txt").read_text(encoding="utf-8") == "user data"
    assert any("is a folder; it is left untouched" in w for w in res.manifest.warnings)


def test_numpy_feature_values_are_serialized(tmp_path):
    paths = fake_library(tmp_path / "lib", ["A - One.wav"])
    t = feat(paths[0], "A", "One", bpm=np.float32(121.5), energy=np.float64("nan"),
             key_confidence=np.float32(0.25))
    export_playlist(make_sequence([t]), tmp_path / "out", FakeNormalizer(), excluded=[])
    f = load_manifest(tmp_path / "out")["tracks"][0]["features"]
    assert f["bpm"] == 121.5 and f["key_confidence"] == 0.25 and f["energy"] is None
