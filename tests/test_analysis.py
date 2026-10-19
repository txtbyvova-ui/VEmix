"""Auto-analysis (spec §5, §9): BPM/key regression against automix v1, cache reuse
and invalidation through the analysis functions, legacy migration, the energy
heuristic, tag/file-name metadata and broken-input handling."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf

from conftest import SR, encode_mp3, music_like, requires_ffmpeg, sha256, write_wav
from veg_mixer import analysis
from veg_mixer.analysis import (
    CHANGED_DURING_ANALYSIS_REASON,
    CYRILLIC_REVIEW_REASON,
    ENERGY_ALGO_VERSION,
    TrackAnalyzer,
    detect_bpm,
    detect_key,
    detect_script_hint,
    estimate_energy,
    library_rel_path,
    normalize_language,
    parse_artist_title,
    scan_library,
)
from veg_mixer.cache import AnalysisCache
from veg_mixer.config import AnalysisConfig, LoudnessConfig
from veg_mixer.ffmpeg_tools import FFmpegNotFoundError, FFmpegTools
from veg_mixer.loudness import LoudnessNormalizer
from veg_mixer.models import AudioProbe

LEGACY_FILE = Path(__file__).resolve().parents[1] / "legacy" / "automix_v1.py"
CFG = AnalysisConfig()


# ---------------------------------------------------------------------------
# synthetic material (no copyrighted music)
# ---------------------------------------------------------------------------

def chord_clicks(seconds: float, *, bpm: float = 120.0, root_hz: float = 220.0, minor: bool = False,
                 seed: int = 0) -> np.ndarray:
    """Sustained triad (+ octave) with a click/noise burst on every beat."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    third = 2 ** ((3 if minor else 4) / 12)
    tones = sum(np.sin(2 * np.pi * f * t) for f in (root_hz, root_hz * third, root_hz * 2 ** (7 / 12), root_hz * 2))
    click = np.exp(-(t % (60.0 / bpm)) / 0.01) * (0.6 * np.sin(2 * np.pi * 1000 * t) + 0.4 * rng.standard_normal(n))
    sig = 0.12 * tones + 0.5 * click
    sig = sig / np.max(np.abs(sig)) * 0.8
    return np.stack([sig, sig], 1).astype(np.float32)


def dense_bright(seconds: float, *, bpm: float = 128.0, seed: int = 0) -> np.ndarray:
    """Four-on-the-floor kick, 16th-note hats and a gated bright saw chord."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    beat = 60.0 / bpm
    kick = np.sin(2 * np.pi * 55 * t) * np.exp(-(t % beat) / 0.06)
    hat = np.diff(rng.standard_normal(n) * np.exp(-(t % (beat / 4)) / 0.015), prepend=0.0)
    saw = sum(np.sin(2 * np.pi * k * f * t) / k for f in (220.0, 277.18, 329.63) for k in range(1, 30))
    gate = np.exp(-((t + beat / 2) % (beat / 2)) / 0.12)
    sig = 0.9 * kick + 0.35 * hat + 0.08 * saw * gate
    sig = sig / np.max(np.abs(sig)) * 0.9
    return np.stack([sig, sig], 1).astype(np.float32)


def soft_sparse(seconds: float) -> np.ndarray:
    """Slow-attack low sine pad, one note every 3 s."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ph = t % 3.0
    env = (1 - np.exp(-ph / 0.8)) * np.exp(-ph / 2.5)
    f = np.array([196.0, 174.61, 220.0, 164.81])[(t // 3.0).astype(int) % 4]
    sig = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * env
    sig = sig / np.max(np.abs(sig)) * 0.5
    return np.stack([sig, sig], 1).astype(np.float32)


# ---------------------------------------------------------------------------
# fixtures / spies
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def legacy():
    """legacy/automix_v1.py imported by path (it is kept for exactly this test)."""
    spec = importlib.util.spec_from_file_location("automix_v1_legacy", LEGACY_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses need the module registered
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


@pytest.fixture
def librosa_calls(monkeypatch):
    """Counts every librosa decode / analysis entry point the analysis code uses."""
    calls = {"load": 0, "beat_track": 0, "chroma_cqt": 0, "stft": 0}

    def wrap(owner, name):
        real = getattr(owner, name)

        def counted(*a, **k):
            calls[name] += 1
            return real(*a, **k)
        monkeypatch.setattr(owner, name, counted)

    wrap(librosa, "load")
    wrap(librosa.beat, "beat_track")
    wrap(librosa.feature, "chroma_cqt")
    wrap(librosa, "stft")
    return calls


@pytest.fixture
def window_offsets(monkeypatch):
    """Records the offset of every analysis-window load (30.0 = offset window, 0.0 = fallback)."""
    offsets: list[float] = []
    real = analysis._load_window

    def spy(path, sr, offset, duration, tools, **kw):
        offsets.append(float(offset))
        return real(path, sr, offset, duration, tools, **kw)
    monkeypatch.setattr(analysis, "_load_window", spy)
    return offsets


# ---------------------------------------------------------------------------
# REGRESSION: BPM / key identical to automix v1
# ---------------------------------------------------------------------------

# seconds, bpm, root, minor, expected camelot, BPM window path, key window path
REGRESSION_CLIPS = [
    (30.5, 100.0, 261.63, True, "5A", "fallback", "fallback"),   # C minor; < 1 s after the offset
    (45.0, 128.0, 196.00, False, "9B", "fallback", "offset"),    # G major; 15 s after the offset
    (70.0, 110.0, 293.66, True, "7A", "offset", "offset"),       # D minor; 40 s after the offset
]
PATHS = {"offset": [30.0], "fallback": [30.0, 0.0]}


@pytest.mark.parametrize("seconds, bpm, root, minor, camelot, bpm_path, key_path", REGRESSION_CLIPS)
def test_bpm_and_key_identical_to_legacy(tmp_path, legacy, window_offsets, seconds, bpm, root, minor,
                                         camelot, bpm_path, key_path):
    src = write_wav(tmp_path / "clip.wav", chord_clicks(seconds, bpm=bpm, root_hz=root, minor=minor))
    cache = AnalysisCache(tmp_path / "analysis_cache.json")

    ours_bpm = detect_bpm(src, cache, CFG)
    bpm_offsets = list(window_offsets)
    window_offsets.clear()
    ours_key = detect_key(src, cache, CFG)
    key_offsets = list(window_offsets)

    lcache = legacy.BPMCache(tmp_path / "legacy_bpm_cache.json")
    assert ours_bpm == legacy.detect_bpm(src, lcache)            # bit-identical
    assert ours_key == legacy.detect_key(src, lcache)            # (camelot, unrounded correlation)
    assert bpm_offsets == PATHS[bpm_path] and key_offsets == PATHS[key_path]
    # ... and the shared algorithm is musically sane on the fixture
    assert ours_key[0] == camelot
    assert abs(ours_bpm - bpm) / bpm < 0.05
    # cached values are returned unchanged (full precision, not v1's 3-decimal rounding)
    assert detect_bpm(src, cache, CFG) == ours_bpm and detect_key(src, cache, CFG) == ours_key


def test_clip_shorter_than_offset_uses_legacy_fallback(tmp_path, legacy, monkeypatch, window_offsets):
    """A track shorter than the 30 s offset: v1 (audioread era) got an empty window and
    retried from the start. libsndfile raises on such a seek, so the legacy code is run
    with an audioread-like loader to obtain its intended result."""
    src = write_wav(tmp_path / "short.wav", chord_clicks(25.0, bpm=120.0, root_hz=220.0))
    cache = AnalysisCache(tmp_path / "c.json")
    ours = (detect_bpm(src, cache, CFG), detect_key(src, cache, CFG))
    assert window_offsets == [30.0, 0.0, 30.0, 0.0]

    real_load = librosa.load

    def audioread_like(path, *, sr=22050, mono=True, offset=0.0, duration=None, **kw):
        if offset and offset >= sf.info(path).duration:
            return np.zeros(0, dtype=np.float32), sr
        return real_load(path, sr=sr, mono=mono, offset=offset, duration=duration, **kw)

    monkeypatch.setattr(librosa, "load", audioread_like)
    lcache = legacy.BPMCache(tmp_path / "legacy.json")
    assert ours == (legacy.detect_bpm(src, lcache), legacy.detect_key(src, lcache))
    assert ours[1][0] == "11B"                                    # A major


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_mp3_bpm_and_key_identical_to_legacy(ffmpeg_tools, tmp_path, legacy):
    wav = write_wav(tmp_path / "fx" / "clip.wav", chord_clicks(40.0, bpm=124.0, root_hz=246.94, seed=3))
    mp3 = encode_mp3(ffmpeg_tools, wav, tmp_path / "fx" / "clip.mp3")
    cache = AnalysisCache(tmp_path / "c.json")
    lcache = legacy.BPMCache(tmp_path / "legacy.json")
    assert detect_bpm(mp3, cache, CFG) == legacy.detect_bpm(mp3, lcache)
    assert detect_key(mp3, cache, CFG) == legacy.detect_key(mp3, lcache)


def test_tables_moved_verbatim(legacy):
    assert analysis.CAMELOT_MAP == legacy.CAMELOT_MAP
    assert np.array_equal(analysis.MAJOR_PROFILE, legacy.MAJOR_PROFILE)
    assert np.array_equal(analysis.MINOR_PROFILE, legacy.MINOR_PROFILE)
    # the conservative key window is part of the contract (project memory rule)
    assert (CFG.key_offset_s, CFG.key_duration_s, CFG.key_min_window_s) == (30.0, 60.0, 1.0)


# ---------------------------------------------------------------------------
# cache reuse / invalidation / legacy migration through the analysis API
# ---------------------------------------------------------------------------

def test_second_analysis_is_served_from_cache(tmp_path, librosa_calls):
    """Spec Cache 1 through the analysis API: the second run never touches librosa."""
    src = write_wav(tmp_path / "a.wav", chord_clicks(32.0, bpm=118.0, root_hz=220.0))
    path = tmp_path / "analysis_cache.json"
    c1 = AnalysisCache(path)
    first = (detect_bpm(src, c1, CFG), detect_key(src, c1, CFG), estimate_energy(src, c1, CFG))
    c1.save()
    after_first = dict(librosa_calls)
    assert after_first["beat_track"] == 1 and after_first["chroma_cqt"] == 1 and after_first["stft"] >= 1

    c2 = AnalysisCache(path)
    second = (detect_bpm(src, c2, CFG), detect_key(src, c2, CFG), estimate_energy(src, c2, CFG))
    assert second == first
    assert librosa_calls == after_first                       # no decode, no analysis
    assert c2.stats["hits"] == 3 and c2.stats["misses"] == 0


def test_modified_source_is_reanalysed(tmp_path, librosa_calls):
    src = tmp_path / "a.wav"
    write_wav(src, chord_clicks(32.0, bpm=100.0, root_hz=220.0))
    c = AnalysisCache(tmp_path / "c.json")
    b1, k1 = detect_bpm(src, c, CFG), detect_key(src, c, CFG)
    assert librosa_calls["beat_track"] == 1 and librosa_calls["chroma_cqt"] == 1
    write_wav(src, chord_clicks(40.0, bpm=140.0, root_hz=261.63, minor=True))   # new master: other size
    b2, k2 = detect_bpm(src, c, CFG), detect_key(src, c, CFG)
    assert librosa_calls["beat_track"] == 2 and librosa_calls["chroma_cqt"] == 2   # recomputed
    assert abs(b2 - 140.0) < 7 and b2 != b1
    assert k2[0] == "5A" and k1[0] == "11B"
    assert c.stats["invalidated"] == 1


def test_analysis_params_change_invalidates_only_that_section(tmp_path, librosa_calls):
    src = write_wav(tmp_path / "a.wav", chord_clicks(32.0))
    c = AnalysisCache(tmp_path / "c.json")
    detect_bpm(src, c, CFG)
    detect_key(src, c, CFG)
    wider_bpm = AnalysisConfig(bpm_duration_s=120.0)
    detect_key(src, c, wider_bpm)                      # key params unchanged -> cached
    assert librosa_calls["chroma_cqt"] == 1
    detect_bpm(src, c, wider_bpm)                      # bpm params changed -> recomputed
    assert librosa_calls["beat_track"] == 2


def test_legacy_values_are_used_without_recomputation(tmp_path, librosa_calls):
    music = tmp_path / "music"
    a = write_wav(music / "a.wav", chord_clicks(31.0, bpm=100.0))
    b = write_wav(music / "b.wav", chord_clicks(31.0, bpm=120.0))
    legacy_file = music / "bpm_cache.json"
    with legacy_file.open("w", encoding="utf-8") as f:        # automix v1 wrote keys as str(path)
        json.dump({str(a): 123.4,
                   str(b): {"bpm": 87.5, "camelot": "4A", "key_confidence": 0.555}}, f, indent=2)
    before = sha256(legacy_file)

    cache = AnalysisCache(tmp_path / "analysis_cache.json", legacy_path=legacy_file)
    assert detect_bpm(a, cache, CFG) == 123.4
    assert detect_bpm(b, cache, CFG) == 87.5
    # the legacy key is preserved although a fresh detection would say 11B
    assert detect_key(b, cache, CFG) == ("4A", 0.555)
    assert librosa_calls == {"load": 0, "beat_track": 0, "chroma_cqt": 0, "stft": 0}
    assert cache.stats["migrated"] == 3

    cam, conf = detect_key(a, cache, CFG)                     # v1 float format had no key -> computed
    assert librosa_calls["chroma_cqt"] == 1 and cam == "11B" and conf > 0.5
    cache.save()
    assert sha256(legacy_file) == before                      # legacy file never modified
    reopened = AnalysisCache(tmp_path / "analysis_cache.json")
    assert detect_key(b, reopened, CFG) == ("4A", 0.555) and librosa_calls["chroma_cqt"] == 1


# ---------------------------------------------------------------------------
# energy heuristic
# ---------------------------------------------------------------------------

def test_energy_ranks_dense_above_sparse(tmp_path):
    c = AnalysisCache(tmp_path / "c.json")
    e_dense = estimate_energy(write_wav(tmp_path / "dense.wav", dense_bright(25.0)), c, CFG)
    e_soft = estimate_energy(write_wav(tmp_path / "soft.wav", soft_sparse(25.0)), c, CFG)
    e_mid = estimate_energy(write_wav(tmp_path / "mid.wav", music_like(25.0, seed=3)), c, CFG)
    assert 0.0 <= e_soft < e_mid < e_dense <= 1.0
    assert e_dense - e_soft > 0.4


@pytest.mark.parametrize("make", [dense_bright, soft_sparse, music_like], ids=["dense", "soft", "music_like"])
def test_energy_does_not_depend_on_level(tmp_path, make):
    c = AnalysisCache(tmp_path / "c.json")
    x = make(25.0)
    loud = estimate_energy(write_wav(tmp_path / "loud.wav", x), c, CFG)
    quiet = estimate_energy(write_wav(tmp_path / "quiet.wav", x * 10 ** (-20 / 20), subtype="PCM_16"), c, CFG)
    assert loud is not None and abs(loud - quiet) <= 0.03


def test_energy_unknown_for_short_or_silent_audio(tmp_path, librosa_calls):
    c = AnalysisCache(tmp_path / "c.json")
    short = write_wav(tmp_path / "short.wav", dense_bright(3.0))
    silent = write_wav(tmp_path / "silent.wav", np.zeros((25 * SR, 2), np.float32))
    assert estimate_energy(short, c, CFG) is None
    assert estimate_energy(silent, c, CFG) is None
    loads = librosa_calls["load"]
    assert estimate_energy(silent, c, CFG) is None            # the "unknown" result is cached too
    assert librosa_calls["load"] == loads


# ---------------------------------------------------------------------------
# metadata helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem, expected", [
    ("Artist - Title", ("Artist", "Title")),
    ("Artist - Title - Extended Mix", ("Artist", "Title - Extended Mix")),   # first separator only
    ("Artist – Title", ("Artist", "Title")),                                 # en dash
    ("50 Cent - Candy Shop", ("50 Cent", "Candy Shop")),
    ("01 - Artist - Title", ("Artist", "Title")),
    ("01. Title", (None, "Title")),
    ("Just A Title", (None, "Just A Title")),
    ("Земфира - Хочешь", ("Земфира", "Хочешь")),
    ("AC-DC - T.N.T.", ("AC-DC", "T.N.T.")),
    ("  ", (None, None)),
])
def test_parse_artist_title(stem, expected):
    assert parse_artist_title(stem) == expected


def test_detect_script_hint():
    assert detect_script_hint("Земфира", None) == "cyrillic"
    assert detect_script_hint(None, "Title", "Би-2 - Полковнику") == "cyrillic"
    assert detect_script_hint("Beyoncé", "Déjà vu", "Ελληνικά") is None   # Latin / Greek are not Cyrillic
    assert detect_script_hint(None, "") is None


@pytest.mark.parametrize("raw, expected", [
    ("eng", "en"), ("ENG", "en"), ("rus", "ru"), ("fre", "fr"), ("fra", "fr"), ("spa", "es"),
    ("ita", "it"), ("ger", "de"), ("deu", "de"), ("English", "en"), ("en-US", "en"), ("pt", "pt"),
    # INT-9: 'por' used to be stored as-is ('por') while curation read it as 'pt'; the
    # shared textnorm table now resolves every ISO 639-2 code at analysis time.
    ("por", "pt"), ("nld", "nl"), ("ukr", "uk"), ("eng;rus", "en;ru"), ("rus/eng", "ru;en"),
    ("Русский", "ru"), ("klingon", "klingon"), ("und", None), ("xxx", None), ("", None), (None, None),
])
def test_normalize_language(raw, expected):
    assert normalize_language(raw) == expected


def test_library_rel_path(tmp_path):
    root = tmp_path / "lib"
    assert library_rel_path(root / "sub" / "Трек.mp3", root) == "sub/Трек.mp3"
    assert library_rel_path(tmp_path / "other" / "x.mp3", root) == "../other/x.mp3"


def test_scan_library(tmp_path):
    lib = tmp_path / "lib"
    for name in ["b.mp3", "A.MP3", "Земфира - Хочешь.mp3", "c.FlAc", ".hidden.mp3",
                 ".track.veg-1a2b3c4d.mp3", "notes.txt", "cover.jpg", "sub/d.wav", ".tmp/e.mp3"]:
        p = lib / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    (lib / "folder.mp3").mkdir()                                   # a directory, not a file
    flat = scan_library(lib)
    assert [p.name for p in flat] == ["A.MP3", "b.mp3", "c.FlAc", "Земфира - Хочешь.mp3"]
    assert all(p.is_absolute() for p in flat)
    deep = scan_library(lib, recursive=True)
    assert [p.relative_to(lib.resolve()).as_posix() for p in deep] == [
        "A.MP3", "b.mp3", "c.FlAc", "sub/d.wav", "Земфира - Хочешь.mp3"]
    assert [p.name for p in scan_library(lib, extensions=(".wav",), recursive=True)] == ["d.wav"]
    with pytest.raises(FileNotFoundError):
        scan_library(tmp_path / "missing")
    with pytest.raises(NotADirectoryError):
        scan_library(lib / "b.mp3")


# ---------------------------------------------------------------------------
# TrackAnalyzer
# ---------------------------------------------------------------------------

def test_ffmpeg_missing_propagates(tmp_path, monkeypatch):
    src = write_wav(tmp_path / "a.wav", chord_clicks(2.0))
    fake = FFmpegTools(ffmpeg="ffmpeg", ffprobe=None, version="fake", has_soxr=False, has_libmp3lame=True)

    def gone(tools, path):
        raise FFmpegNotFoundError("ffmpeg was not found")

    monkeypatch.setattr(analysis, "probe_audio", gone)
    with pytest.raises(FFmpegNotFoundError):
        TrackAnalyzer(fake, AnalysisCache(tmp_path / "c.json"), CFG).analyze(src, tmp_path)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_analyze_tagged_mp3_and_reuse_cache(ffmpeg_tools, tmp_path, monkeypatch, librosa_calls):
    lib = tmp_path / "lib"
    lib.mkdir()
    wav = write_wav(tmp_path / "fx" / "src.wav", music_like(35.0, seed=11, bpm=124.0))
    src = encode_mp3(ffmpeg_tools, wav, lib / "Filename Artist - Filename Title.mp3",
                     tags={"artist": "Tag Artist", "title": "Tag Title", "genre": "Pop", "language": "eng"})
    before = sha256(src)
    cache_path = tmp_path / "analysis_cache.json"
    norm = LoudnessNormalizer(ffmpeg_tools)

    cache = AnalysisCache(cache_path)
    at = TrackAnalyzer(ffmpeg_tools, cache, CFG, normalizer=norm).analyze(src, lib)
    assert at.ok, at.errors
    f = at.features
    assert (f.artist, f.title, f.genre, f.language) == ("Tag Artist", "Tag Title", "Pop", "en")
    assert {k: f.feature_sources[k] for k in ("artist", "title", "genre", "language")} == {
        "artist": "id3", "title": "id3", "genre": "id3", "language": "id3"}
    assert f.rel_path == "Filename Artist - Filename Title.mp3" and f.track_id
    assert f.bpm is not None and abs(f.bpm - 124.0) / 124.0 < 0.05
    assert f.camelot is not None and f.key_confidence is not None and f.key_confidence > 0
    assert f.feature_sources["bpm"] == "auto:bpm-v1" and f.feature_sources["camelot"] == "auto:key-v1"
    assert f.energy is not None and 0.0 <= f.energy <= 1.0
    assert f.energy_confidence == CFG.energy_confidence
    assert f.feature_sources["energy"] == f"auto:{ENERGY_ALGO_VERSION}"
    # never estimated, never fabricated
    assert (f.danceability, f.vocal_presence, f.mood, f.familiarity) == (None, None, None, None)
    assert f.script_hint is None and f.review_reasons == []
    assert f.duration_s == pytest.approx(35.0, abs=0.1)
    assert at.probe.codec == "mp3" and at.loudness is not None and at.loudness.integrated_lufs < 0
    assert at.from_cache == {"probe": False, "bpm": False, "key": False, "energy": False, "loudness": False}
    cache.save()

    # second run from the saved cache: everything served, nothing decoded or measured again
    measured: list[Path] = []
    real_measure = LoudnessNormalizer.measure

    def counting_measure(self, path, **kw):
        measured.append(Path(path))
        return real_measure(self, path, **kw)

    monkeypatch.setattr(LoudnessNormalizer, "measure", counting_measure)
    after_first = dict(librosa_calls)
    cache2 = AnalysisCache(cache_path)
    at2 = TrackAnalyzer(ffmpeg_tools, cache2, CFG, normalizer=norm).analyze(src, lib)
    assert at2.from_cache == {"probe": True, "bpm": True, "key": True, "energy": True, "loudness": True}
    assert librosa_calls == after_first and measured == []
    assert at2.features.to_dict() == f.to_dict()
    assert at2.loudness == at.loudness

    # a new normalization target re-measures loudness only; BPM / key / energy stay cached
    norm_b = LoudnessNormalizer(ffmpeg_tools, LoudnessConfig(target_lufs=-14.0))
    at3 = TrackAnalyzer(ffmpeg_tools, cache2, CFG, normalizer=norm_b).analyze(src, lib)
    assert at3.from_cache == {"probe": True, "bpm": True, "key": True, "energy": True, "loudness": False}
    assert len(measured) == 1 and librosa_calls == after_first
    assert sha256(src) == before                                     # the source is never written


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_analyze_filename_fallback_cyrillic_hint_and_legacy_key(ffmpeg_tools, tmp_path):
    lib = tmp_path / "lib"
    src = write_wav(lib / "рус" / "Земфира - Хочешь.wav", chord_clicks(31.0, bpm=96.0, root_hz=220.0))
    legacy_file = lib / "bpm_cache.json"
    legacy_file.write_text(json.dumps({str(src): {"bpm": 95.9, "camelot": "8A", "key_confidence": 0.81}},
                                      ensure_ascii=False), encoding="utf-8")
    cache = AnalysisCache(tmp_path / "c.json", legacy_path=legacy_file)
    at = TrackAnalyzer(ffmpeg_tools, cache, CFG).analyze(src, lib, measure_loudness=True)   # no normalizer
    assert at.ok, at.errors
    f = at.features
    assert (f.artist, f.title) == ("Земфира", "Хочешь")
    assert f.feature_sources["artist"] == "filename" and f.feature_sources["title"] == "filename"
    assert f.rel_path == "рус/Земфира - Хочешь.wav"
    assert f.script_hint == "cyrillic" and CYRILLIC_REVIEW_REASON in f.review_reasons
    assert f.language is None and "language" not in f.feature_sources     # a hint, not a language claim
    assert f.genre is None
    assert (f.bpm, f.camelot, f.key_confidence) == (95.9, "8A", 0.81)      # legacy values preserved
    assert f.feature_sources["camelot"] == "auto:key-v1:legacy-cache"
    assert at.from_cache["bpm"] is True and at.from_cache["key"] is True
    assert at.loudness is None and "loudness" not in at.from_cache


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_analyze_broken_inputs_report_errors(ffmpeg_tools, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    corrupt = lib / "Broken Artist - Broken Song.mp3"
    corrupt.write_bytes(b"ID3" + bytes(range(256)) * 40)
    empty = lib / "empty.wav"
    empty.write_bytes(b"")
    silent = write_wav(lib / "silent.wav", np.zeros((12 * SR, 2), np.float32))
    analyzer = TrackAnalyzer(ffmpeg_tools, AnalysisCache(tmp_path / "c.json"), CFG,
                             normalizer=LoudnessNormalizer(ffmpeg_tools))

    for src in (corrupt, empty):
        at = analyzer.analyze(src, lib)                           # must not raise
        assert not at.ok and at.errors and at.probe is None, src
        assert at.features.rel_path == src.name
    assert analyzer.analyze(corrupt, lib).features.artist == "Broken Artist"

    at = analyzer.analyze(silent, lib)
    assert not at.ok and any("loudness" in e for e in at.errors)
    f = at.features
    assert f.bpm is None and f.camelot is None and f.energy is None    # unknown, not 0 / '8B'
    assert any("tempo" in r for r in f.review_reasons) and any("key" in r for r in f.review_reasons)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_formats_libsndfile_cannot_read_are_decoded_by_ffmpeg(ffmpeg_tools, tmp_path):
    wav = write_wav(tmp_path / "a.wav", chord_clicks(32.0, bpm=110.0, root_hz=196.0))
    m4a = tmp_path / "a.m4a"
    subprocess.run([ffmpeg_tools.ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", str(wav), "-c:a", "aac",
                    "-b:a", "192k", str(m4a)], check=True, capture_output=True)
    with pytest.raises(sf.LibsndfileError):
        sf.info(str(m4a))                                          # precondition of this test
    c = AnalysisCache(tmp_path / "c.json")
    bpm_wav, key_wav = detect_bpm(wav, c, CFG), detect_key(wav, c, CFG)
    bpm_m4a = detect_bpm(m4a, c, CFG, tools=ffmpeg_tools)
    key_m4a = detect_key(m4a, c, CFG, tools=ffmpeg_tools)
    assert abs(bpm_m4a - bpm_wav) / bpm_wav < 0.03 and key_m4a[0] == key_wav[0] == "9B"
    assert not list(tmp_path.glob("veg_dec_*"))


# ---------------------------------------------------------------------------
# Review-finding regressions
# ---------------------------------------------------------------------------

def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _manifest(path: Path, schema: str = "vegroove.manifest") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": schema, "mode": "mix"}), encoding="utf-8")
    return path


def test_scan_library_skips_vegroove_outputs(tmp_path):
    """SAFE-9: a mix written into the library (music/mix.mp3 + mix.manifest.json) used to
    become a 'track' of the next mix; a playlist export inside a recursively scanned
    library was re-ingested from its tracks/ folder."""
    lib = tmp_path / "music"
    for name in ["Demo Aurora - Morning Glow.mp3", "Demo Coral - Neon Avenue.mp3", "mix.mp3", "other.mp3",
                 "playlist_out/tracks/001_Demo_Aurora.mp3", "playlist_out/evening/tracks/001_x.mp3",
                 "sub/keep.wav"]:
        _touch(lib / name)
    _manifest(lib / "mix.manifest.json")                                   # ours -> mix.mp3 is an output
    _manifest(lib / "other.manifest.json", schema="someone.else")          # not ours -> other.mp3 stays
    _manifest(lib / "playlist_out" / "evening" / "manifest.json")
    _manifest(lib / "playlist_out" / "manifest.json")

    flat = [p.name for p in scan_library(lib)]
    assert flat == ["Demo Aurora - Morning Glow.mp3", "Demo Coral - Neon Avenue.mp3", "other.mp3"]
    deep = [p.relative_to(lib.resolve()).as_posix() for p in scan_library(lib, recursive=True)]
    assert deep == ["Demo Aurora - Morning Glow.mp3", "Demo Coral - Neon Avenue.mp3", "other.mp3", "sub/keep.wav"]
    # a playlist export scanned recursively as the library root is an output as a whole
    assert scan_library(lib / "playlist_out", recursive=True) == []


def test_scan_library_exclude_paths(tmp_path):
    """SAFE-9: the caller's output locations (normalize --output inside the library, the
    planned mix file) are excluded, whatever their spelling."""
    lib = tmp_path / "lib"
    for name in ["a.mp3", "b.mp3", "normalized/a.mp3", "normalized/deep/b.mp3", "keep/c.mp3"]:
        _touch(lib / name)
    alias = lib / "keep" / ".." / "normalized"                          # other spelling of lib/normalized
    got = scan_library(lib, recursive=True, exclude=[alias, lib / "b.mp3"])
    assert [p.relative_to(lib.resolve()).as_posix() for p in got] == ["a.mp3", "keep/c.mp3"]
    assert [p.name for p in scan_library(lib, exclude=[lib / "a.mp3"])] == ["b.mp3"]
    assert [p.name for p in scan_library(lib, recursive=True, exclude=[tmp_path / "elsewhere"])][:2] == ["a.mp3", "b.mp3"]
    assert scan_library(lib, recursive=True, exclude=[lib]) == []


def test_file_changed_during_analysis_is_flagged_and_not_cached(tmp_path, monkeypatch):
    """SAFE-10: a file replaced while it was being analysed had the OLD results cached
    under the NEW fingerprint and reused on every later run."""
    lib = tmp_path / "lib"
    src = write_wav(lib / "Artist - Song.wav", chord_clicks(2.0))
    fake = FFmpegTools(ffmpeg="ffmpeg", ffprobe="ffprobe", version="fake", has_soxr=False, has_libmp3lame=True)
    monkeypatch.setattr(analysis, "probe_audio", lambda tools, path: AudioProbe(
        path=Path(path), duration_s=2.0, sample_rate=SR, channels=2, codec="pcm_f32le"))

    def bpm_then_replace(path, cfg, *, tools=None):
        write_wav(src, chord_clicks(3.0, seed=5))                      # a sync client swaps the file
        return 120.0

    monkeypatch.setattr(analysis, "compute_bpm", bpm_then_replace)
    monkeypatch.setattr(analysis, "compute_key", lambda path, cfg, *, tools=None: ("8A", 0.5))
    cfg = AnalysisConfig(estimate_energy=False)
    cache = AnalysisCache(tmp_path / "c.json")
    at = TrackAnalyzer(fake, cache, cfg).analyze(src, lib)
    assert CHANGED_DURING_ANALYSIS_REASON in at.features.review_reasons
    assert cache.get(src, "bpm", version=analysis.BPM_ALGO_VERSION, params=analysis.bpm_params(cfg)) is None
    assert cache.get(src, "key", version=analysis.KEY_ALGO_VERSION, params=analysis.key_params(cfg)) is None

    # an unchanged file is analysed and cached normally, without the flag
    monkeypatch.setattr(analysis, "compute_bpm", lambda path, cfg, *, tools=None: 121.0)
    at2 = TrackAnalyzer(fake, cache, cfg).analyze(src, lib)
    assert CHANGED_DURING_ANALYSIS_REASON not in at2.features.review_reasons
    assert cache.get(src, "bpm", version=analysis.BPM_ALGO_VERSION, params=analysis.bpm_params(cfg)) == {"bpm": 121.0}


def _id3v24_tlan_mp3(path, values: list[str]):
    """MP3-like file: an ID3v2.4 tag whose TLAN frame holds NUL-separated values."""
    def syncsafe(n):
        return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])
    body = b"\x03" + "\x00".join(values).encode("utf-8")
    frame = b"TLAN" + syncsafe(len(body)) + b"\x00\x00" + body
    tag = b"ID3\x04\x00\x00" + syncsafe(len(frame)) + frame
    path.write_bytes(tag + b"\xff\xfb\x90\x00" * 16)
    return path


def test_id3v24_multi_value_tlan_keeps_every_language(tmp_path):
    """SPEC-3 (round 2): ffprobe returns only the first value of 'eng\0rus'; the reader
    returns both, so the excluded-language rule cannot be bypassed."""
    from veg_mixer.id3 import read_text_frames
    from veg_mixer.textnorm import normalize_language
    f = _id3v24_tlan_mp3(tmp_path / "duet.mp3", ["eng", "rus"])
    values = read_text_frames(f, ("TLAN",))["TLAN"]
    assert values == ["eng", "rus"]
    assert normalize_language(";".join(values)) == "en;ru"
    assert read_text_frames(tmp_path / "missing.mp3") == {}
    (tmp_path / "plain.mp3").write_bytes(b"\xff\xfb\x90\x00" * 16)
    assert read_text_frames(tmp_path / "plain.mp3") == {}
