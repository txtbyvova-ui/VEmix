"""QC + reporting (spec §10/§11, §3 "Additional QC"): pure units (strict M3U8
parser, silence classification, status logic, reports) and real-FFmpeg
integration on synthetic playlists and continuous mixes."""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from conftest import SR, encode_mp3, music_like, pinkish_noise, requires_ffmpeg, sha256, spiky, write_wav
from veg_mixer.config import LoudnessConfig, QCConfig
from veg_mixer.ffmpeg_tools import FFmpegError
from veg_mixer.loudness import LoudnessNormalizer
from veg_mixer.models import (
    ClientProfile,
    LanguageRules,
    ListRules,
    ManifestTrack,
    MixTrackEntry,
    NormalizationStatus,
    OverallStatus,
    PlaylistManifest,
    PreferenceLevel,
    QCIssue,
    QCReport,
    Severity,
    TrackFeatures,
    TrackQC,
    TransitionReport,
    make_track_id,
    to_jsonable,
)
from veg_mixer.quality_control import (
    detect_silence_gaps,
    is_critical_check,
    overall_status,
    parse_m3u8,
    parse_tracklist,
    qc_mix,
    qc_playlist,
    scan_pcm,
    scan_silence,
)
from veg_mixer.reports import (
    NOT_SELECTED_CODE,
    SUMMARY_TITLE,
    compute_summary,
    excluded_stage,
    format_hms,
    qc_report_from_dict,
    render_text_report,
    write_reports,
)

CFG = LoudnessConfig()
QC = QCConfig()
TP_LIMIT = CFG.target_true_peak_dbtp + CFG.true_peak_tolerance_db
PLAYLIST_CRITICAL = {"manifest", "m3u8_syntax", "files_exist", "paths_inside_output", "files_decodable",
                     "nonzero_duration", "loudness", "true_peak", "track_order", "duplicates", "excluded_tracks"}
MIX_CRITICAL = {"manifest", "mix_file", "mix_loudness", "mix_true_peak", "mix_duration", "silence", "clipping",
                "transition_timing", "tracklist_timestamps"}


def issue(sev: Severity, code: str = "x") -> QCIssue:
    return QCIssue(severity=sev, code=code, message=code)


def codes(report: QCReport, severity: Severity | None = None) -> set[str]:
    return {i.code for i in report.issues if severity is None or i.severity is severity}


def issues_of(report: QCReport, code: str) -> list[QCIssue]:
    return [i for i in report.issues if i.code == code]


# ---------------------------------------------------------------------------
# overall_status (spec §11: never PASS when critical checks were skipped)
# ---------------------------------------------------------------------------

def test_overall_status_logic():
    assert overall_status([], []) is OverallStatus.PASS
    assert overall_status([issue(Severity.INFO)], []) is OverallStatus.PASS
    assert overall_status([issue(Severity.INFO), issue(Severity.WARNING)], []) is OverallStatus.WARNING
    assert overall_status([issue(Severity.WARNING), issue(Severity.FAIL)], []) is OverallStatus.FAIL
    # skipped critical check -> never PASS
    assert overall_status([], ["track_order: playlist.m3u8 unavailable"]) is OverallStatus.WARNING
    assert overall_status([], ["mix_duration: manifest has no expected_duration_s"]) is OverallStatus.WARNING
    # an unknown check name is treated as critical (conservative)
    assert overall_status([], ["something_new: not implemented"]) is OverallStatus.WARNING
    # explicitly non-critical checks may be skipped without losing PASS
    assert overall_status([], ["profile_rules: no client profile supplied"]) is OverallStatus.PASS
    # FAIL still dominates skipped checks
    assert overall_status([issue(Severity.FAIL)], ["track_order: x"]) is OverallStatus.FAIL
    # severities loaded from JSON as plain strings work too
    assert overall_status([QCIssue(severity="FAIL", code="c", message="m")], []) is OverallStatus.FAIL


# ---------------------------------------------------------------------------
# Strict M3U8 parser
# ---------------------------------------------------------------------------

def test_parse_m3u8_valid_unicode(tmp_path):
    p = tmp_path / "playlist.m3u8"
    p.write_text("#EXTM3U\n#PLAYLIST:Утро\n# a comment\n\n#EXTINF:183,Café Ensemble - Été à Paris\n"
                 "tracks/001_Café_Été.mp3\n#EXTINF:-1,Unknown length\r\ntracks/002 with space.mp3\r\n",
                 encoding="utf-8")
    r = parse_m3u8(p)
    assert r.errors == [] and r.warnings == []
    assert r.playlist_title == "Утро"
    assert [(e.duration_s, e.title, e.uri) for e in r.entries] == [
        (183.0, "Café Ensemble - Été à Paris", "tracks/001_Café_Été.mp3"),
        (None, "Unknown length", "tracks/002 with space.mp3")]


@pytest.mark.parametrize("text, fragment", [
    ("#EXTINF:10,A\na.mp3\n", "must start with '#EXTM3U'"),
    ("", "must start with '#EXTM3U'"),
    ("#EXTM3U\n", "no entries"),
    ("#EXTM3U\n#EXTINF:10,A\n#EXTINF:11,B\nb.mp3\n", "not followed by a file path"),
    ("#EXTM3U\n#EXTINF:10,A\n", "not followed by a file path"),
    ("#EXTM3U\na.mp3\n", "no preceding #EXTINF"),
    ("#EXTM3U\n#EXTINF:abc,A\na.mp3\n", "malformed #EXTINF"),
    ("#EXTM3U\n#EXTINF:10 A\na.mp3\n", "malformed #EXTINF"),
    ("#EXTM3U\n#EXTINF:10,\na.mp3\n", "empty title"),
    ("#EXTM3U\n#EXTINF:-5,A\na.mp3\n", "negative"),
    ("#EXTM3U\n#EXTINF:10,A\nC:/music/a.mp3\n", "absolute path"),
    ("#EXTM3U\n#EXTINF:10,A\n/music/a.mp3\n", "absolute path"),
    ("#EXTM3U\n#EXTINF:10,A\ntracks\\a.mp3\n", "backslash"),
    ("#EXTM3U\n#EXTINF:10,A\nhttp://example.com/a.mp3\n", "URL"),
    ("#EXTM3U\n#EXTM3U\n#EXTINF:10,A\na.mp3\n", "duplicate"),
    ("#EXTM3U\n#EXTINF:10,A\n a.mp3\n", "whitespace"),
])
def test_parse_m3u8_rejects_invalid(tmp_path, text, fragment):
    p = tmp_path / "bad.m3u8"
    p.write_text(text, encoding="utf-8")
    r = parse_m3u8(p)
    assert any(fragment in e for e in r.errors), r.errors


def test_parse_m3u8_encoding_problems(tmp_path):
    p = tmp_path / "latin1.m3u8"
    p.write_bytes("#EXTM3U\n#EXTINF:10,Café\na.mp3\n".encode("latin-1"))
    assert any("UTF-8" in e for e in parse_m3u8(p).errors)
    q = tmp_path / "bom.m3u8"
    q.write_bytes(b"\xef\xbb\xbf#EXTM3U\n#EXTINF:10,A\na.mp3\n")
    r = parse_m3u8(q)
    assert r.errors == [] and any("byte-order mark" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# Silence classification (pure)
# ---------------------------------------------------------------------------

def _levels(total_s: float, segments: list[tuple[float, float, float]], base: float = -18.0) -> np.ndarray:
    """0.5 s windows at ``base`` dBFS with [start, end) segments set to a level."""
    lv = np.full(int(total_s / QC.silence_window_s), base)
    for a, b, level in segments:
        lv[int(a / QC.silence_window_s):int(b / QC.silence_window_s)] = level
    return lv


def test_detect_silence_gaps_classification():
    lv = _levels(120, [
        (0.0, 2.0, -70.0),        # inside the planned 3 s fade-in        -> edge_fade (INFO)
        (20.0, 23.0, -200.0),     # digital zero, far from any boundary  -> digital_silence
        (40.0, 41.0, -200.0),     # 1 s only: below the minimum duration -> ignored
        (58.0, 60.5, -62.0),      # quiet, 0.5 s from a boundary         -> transition_gap
        (80.0, 84.0, -58.0),      # quiet mid-track                      -> quiet_passage (INFO)
        (116.0, 120.0, -75.0),    # inside the planned 5 s fade-out      -> edge_fade
    ])
    gaps = detect_silence_gaps(lv, QC.silence_window_s, QC, total_duration_s=120.0,
                               boundaries_s=[30.0, 61.0, 100.0], fade_in_s=3.0, fade_out_s=5.0)
    got = [(g["start_s"], g["end_s"], g["classification"], g["severity"], g["suspicious"]) for g in gaps]
    assert got == [
        (0.0, 2.0, "edge_fade", "INFO", False),
        (20.0, 23.0, "digital_silence", "WARNING", True),
        (58.0, 60.5, "transition_gap", "WARNING", True),
        (80.0, 84.0, "quiet_passage", "INFO", False),
        (116.0, 120.0, "edge_fade", "INFO", False),
    ]
    tg = gaps[2]
    assert tg["near_boundary"] and tg["boundary_s"] == 61.0 and not tg["digital_silence"]
    assert gaps[1]["digital_silence"] and gaps[1]["min_level_dbfs"] == -200.0
    for g in gaps:   # every entry carries seconds + H:MM:SS
        assert g["start_hms"] == format_hms(g["start_s"]) and g["end_hms"] == format_hms(g["end_s"])
        assert g["start_hms"] in g["message"]
    # without planned fades the same edge gaps are judged on their own merits
    plain = detect_silence_gaps(lv, QC.silence_window_s, QC, total_duration_s=120.0, boundaries_s=[])
    assert plain[0]["classification"] == "quiet_passage" and plain[-1]["classification"] == "quiet_passage"
    # an edge gap longer than the planned fade is NOT excused
    long_head = detect_silence_gaps(_levels(60, [(0, 10, -200.0)]), QC.silence_window_s, QC,
                                    total_duration_s=60.0, fade_in_s=3.0)
    assert long_head[0]["classification"] == "digital_silence"
    assert detect_silence_gaps(np.empty(0), 0.5, QC) == []


def test_format_hms_and_tracklist_parser(tmp_path):
    assert format_hms(0) == "0:00:00" and format_hms(59.999) == "0:00:59"
    assert format_hms(3725.4) == "1:02:05" and format_hms(None) == "n/a" and format_hms(float("nan")) == "n/a"
    p = tmp_path / "mix_tracklist.txt"
    p.write_text("TRACKLIST: mix.mp3\n#    | Time       | Track\n-----|------------|------\n"
                 "1    | 0:00:00    | A\n2    | 0:01:05    | B\n10   | 1:00:00    | C\n", encoding="utf-8")
    rows, errors = parse_tracklist(p)
    assert errors == [] and rows == {1: ["0:00:00"], 2: ["0:01:05"], 10: ["1:00:00"]}


# ---------------------------------------------------------------------------
# Reports (pure)
# ---------------------------------------------------------------------------

def _sample_report() -> QCReport:
    r = QCReport(mode="playlist", generated_at="2026-10-19T12:00:00+00:00", status=OverallStatus.WARNING,
                 target_lufs=-16.0, target_true_peak_dbtp=-1.5)
    r.tracks = [
        TrackQC(1, "tracks/001_Café_Été.mp3", "Café - Été.flac", -21.3, -16.4, -3.1, -1.62, 183.2, "passed"),
        TrackQC(2, "tracks/002_b.mp3", "b.mp3", -9.8, -15.8, 0.4, -1.55, 201.0, "review", ["WARNING: listen"]),
        TrackQC(3, "tracks/003_c.mp3", "c.mp3", None, None, None, None, None, "failed", ["FAIL: missing"]),
    ]
    r.issues = [QCIssue(Severity.FAIL, "file_missing", "missing", track_position=3),
                QCIssue(Severity.WARNING, "silence_transition_gap", "gap", track_position=2, timestamp_s=75.5)]
    r.flagged_transitions = [
        {"from_index": 1, "to_index": 2, "reasons": ["energy jump 0.45"], "kind": "sequencing", "source": "manifest"},
        {"from_index": 2, "to_index": 3, "reasons": ["bpm jump"], "kind": "sequencing", "source": "manifest"},
        {"from_index": 2, "to_index": 3, "reasons": ["Energy contrast"], "kind": "loudness", "source": "qc"},
    ]
    r.silence_warnings = [{"start_s": 75.5, "end_s": 78.0, "duration_s": 2.5, "start_hms": "0:01:15",
                           "end_hms": "0:01:18", "classification": "transition_gap", "severity": "WARNING",
                           "suspicious": True, "message": "gap"},
                          {"start_s": 90.0, "end_s": 92.0, "duration_s": 2.0, "start_hms": "0:01:30",
                           "end_hms": "0:01:32", "classification": "quiet_passage", "severity": "INFO",
                           "suspicious": False, "message": "quiet"}]
    r.excluded = [{"source_rel": "ru/song.mp3", "reasons": ["language ru excluded"]}]
    r.unknown_metadata = 2
    r.checks_run = ["manifest", "loudness"]
    r.checks_skipped = ["profile_rules: no client profile supplied"]
    r.summary = compute_summary(r)
    return r


def test_text_report_starts_with_exact_summary_block():
    text = render_text_report(_sample_report())
    assert text.splitlines()[:17] == [
        "VEgroove Audio QC",
        "",
        "Tracks processed: 3",
        "Tracks passed: 1",
        "Tracks failed: 1",
        "Tracks requiring review: 1",
        "",
        "Target LUFS: -16.0",
        "Measured LUFS range: -16.4 .. -15.8 LUFS",
        "Maximum measured True Peak: -1.55 dBTP",
        "",
        "Excluded tracks: 1",
        "Unknown metadata: 2",
        "Suspicious energy transitions: 1",     # only sequencing flags mentioning energy
        "Silence warnings: 1",                  # suspicious gaps only; the quiet passage is INFO
        "",
        "Overall status: WARNING",
    ]
    for fragment in ("energy jump 0.45", "bpm jump", "0:01:15", "(75.5 s)", "quiet_passage",
                     "Café", "Checks skipped: profile_rules: no client profile supplied", "ru/song.mp3"):
        assert fragment in text


def test_text_report_unknown_values_are_na():
    r = QCReport(mode="playlist", generated_at="x", status=OverallStatus.FAIL, target_lufs=-16.0,
                 target_true_peak_dbtp=-1.5)
    lines = render_text_report(r).splitlines()
    assert "Measured LUFS range: n/a" in lines and "Maximum measured True Peak: n/a" in lines
    assert "Overall status: FAIL" in lines


def test_reports_json_round_trip_and_atomic_write(tmp_path):
    report = _sample_report()
    out = tmp_path / "отчёт"
    jp, tp = write_reports(report, out, stem="qc_report")
    assert (jp, tp) == (out / "qc_report.json", out / "qc_report.txt")
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert "Café" in jp.read_text(encoding="utf-8")          # ensure_ascii=False
    assert data == report.to_dict()
    back = qc_report_from_dict(data)
    assert back.to_dict() == report.to_dict()
    assert back.summary == compute_summary(back)
    assert tp.read_text(encoding="utf-8") == render_text_report(report)
    assert sorted(p.name for p in out.iterdir()) == ["qc_report.json", "qc_report.txt"]   # no temp leftovers


# ---------------------------------------------------------------------------
# Real FFmpeg: playlist fixture
# ---------------------------------------------------------------------------

PLAYLIST_SPECS = [   # artist, title, seed, language, energy, source peak dBFS, vibe tags
    ("Alpha Artist", "First Song", 1, "en", 0.40, -6.0, ["warm", "bright"]),
    ("Café Ensemble", "Été à Paris", 2, "fr", 0.45, -14.0, ["warm", "melodic"]),
    ("Gamma", "Third Song", 3, "es", 0.50, -3.0, ["warm", "groovy"]),
]


def _write_manifest(out: Path, manifest: PlaylistManifest | dict) -> None:
    data = manifest.to_dict() if isinstance(manifest, PlaylistManifest) else manifest
    (out / "manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_m3u8(out: Path, rows: list[tuple[float | None, str, str]]) -> None:
    lines = ["#EXTM3U"]
    for dur, title, rel in rows:
        lines += [f"#EXTINF:{-1 if dur is None else int(round(dur))},{title}", rel]
    (out / "playlist.m3u8").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest_json(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def _m3u8_rows(data: dict) -> list[tuple[float | None, str, str]]:
    return [(t["duration_s"], f"{t['artist']} - {t['title']}", t["file"]) for t in data["tracks"]]


@pytest.fixture(scope="module")
def golden_playlist(tmp_path_factory):
    """3 synthetic tracks normalized by the real normalizer into a playlist folder
    (Unicode path + file names), hand-written manifest / m3u8 / tracklist."""
    from conftest import TOOLS
    if TOOLS is None:
        pytest.skip("FFmpeg not available")
    root = tmp_path_factory.mktemp("qc_playlist")
    out = root / "плейлист_out"
    norm = LoudnessNormalizer(TOOLS)
    tracks = []
    for i, (artist, title, seed, lang, energy, peak, tags) in enumerate(PLAYLIST_SPECS, start=1):
        src = write_wav(root / "library" / f"{artist} - {title}.wav", music_like(8, peak_dbfs=peak, seed=seed))
        fname = f"{i:03d}_{artist.replace(' ', '_')}_{title.replace(' ', '_')}.mp3"
        r = norm.normalize_to_mp3(src, out / "tracks" / fname, metadata={"artist": artist, "title": title})
        assert r.status is NormalizationStatus.PASSED, (r.errors, r.review_reasons)
        meta = r.to_metadata()
        feats = TrackFeatures(path=src, duration_s=8.0, energy=energy, mood="uplifting", genre="pop",
                              vibe_tags=tags, language=lang, artist=artist, title=title,
                              rel_path=src.name, feature_sources={"energy": "manual"})
        tracks.append(ManifestTrack(index=i, file=f"tracks/{fname}", title=title, artist=artist,
                                    duration_s=meta["duration_s"], source_rel=src.name, source_path=str(src),
                                    track_id=make_track_id(src.name), loudness=meta, features=feats.to_dict()))
    manifest = PlaylistManifest(
        generator="test", generated_at=datetime.now(timezone.utc).isoformat(), mode="playlist",
        profile="commercial_pop", daypart="afternoon",
        loudness_targets={"target_lufs": CFG.target_lufs, "target_true_peak_dbtp": CFG.target_true_peak_dbtp},
        tracks=tracks, excluded=[{"source_rel": "Иван - Песня.mp3", "reasons": ["language 'ru' excluded"]}])
    _write_manifest(out, manifest)
    _write_m3u8(out, _m3u8_rows(manifest.to_dict()))
    (out / "tracklist.txt").write_text("".join(f"{t.index:03d}. {t.artist} - {t.title}\n" for t in tracks),
                                       encoding="utf-8")
    return out


@pytest.fixture
def playlist_dir(golden_playlist, tmp_path):
    dest = tmp_path / golden_playlist.name
    shutil.copytree(golden_playlist, dest)
    return dest


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_valid_playlist_passes(ffmpeg_tools, playlist_dir):
    before = {p.relative_to(playlist_dir).as_posix(): sha256(p) for p in playlist_dir.rglob("*") if p.is_file()}
    norm = LoudnessNormalizer(ffmpeg_tools)
    profile = ClientProfile(client="commercial_pop",
                            language=LanguageRules(allowed=["en", "fr", "es", "it"], excluded=["ru"]),
                            blocklist=ListRules(artists=["Banned Artist"], tags=["aggressive"]))
    report = qc_playlist(playlist_dir, norm, profile=profile)
    assert report.status is OverallStatus.PASS, [(i.code, i.message) for i in report.issues]
    assert PLAYLIST_CRITICAL <= set(report.checks_run)
    assert "profile_rules" in report.checks_run and "adjacent_short_term" in report.checks_run
    assert report.checks_skipped == []
    assert [t.status for t in report.tracks] == ["passed"] * 3
    for t in report.tracks:
        # QC re-measured the published MP3 itself
        again = norm.measure(playlist_dir / t.file)
        assert math.isclose(t.final_lufs, again.integrated_lufs, abs_tol=0.05)
        assert math.isclose(t.final_true_peak_dbtp, again.true_peak_dbtp, abs_tol=0.05)
        assert abs(t.final_lufs - CFG.target_lufs) <= CFG.lufs_tolerance and t.final_true_peak_dbtp <= TP_LIMIT
        assert t.original_lufs is not None and t.duration_s == pytest.approx(8.0, abs=0.1)
    s = report.summary
    assert (s["tracks_processed"], s["tracks_passed"], s["tracks_failed"], s["tracks_requiring_review"]) == (3, 3, 0, 0)
    assert s["excluded_tracks"] == 1 and s["unknown_metadata"] == 0 and s["overall_status"] == "PASS"
    assert s["measured_lufs_min"] <= s["measured_lufs_max"] and s["max_true_peak_dbtp"] <= TP_LIMIT
    jp, tp = write_reports(report, playlist_dir)
    text = tp.read_text(encoding="utf-8")
    assert text.startswith(SUMMARY_TITLE + "\n\nTracks processed: 3\n") and "Overall status: PASS" in text
    # QC never modifies the delivery (only the two report files were added)
    after = {p.relative_to(playlist_dir).as_posix(): sha256(p) for p in playlist_dir.rglob("*") if p.is_file()}
    assert {k: v for k, v in after.items() if not k.startswith("qc_report.")} == before


def _mut_missing(d, tools):
    (d / _manifest_json(d)["tracks"][1]["file"]).unlink()


def _mut_corrupt(d, tools):
    (d / _manifest_json(d)["tracks"][1]["file"]).write_bytes(b"ID3" + bytes(range(256)) * 40)


def _mut_zero_duration(d, tools):
    wav = write_wav(d.parent / "fx" / "empty.wav", np.zeros((0, 2), np.float32))
    encode_mp3(tools, wav, d / _manifest_json(d)["tracks"][1]["file"])


def _mut_m3u8_order(d, tools):
    rows = _m3u8_rows(_manifest_json(d))
    rows[0], rows[1] = rows[1], rows[0]
    _write_m3u8(d, rows)


def _mut_index_sequence(d, tools):
    data = _manifest_json(d)
    data["tracks"][2]["index"] = 4
    _write_manifest(d, data)


def _mut_duplicate(d, tools):
    data = _manifest_json(d)
    for k in ("source_rel", "source_path", "track_id"):
        data["tracks"][2][k] = data["tracks"][0][k]
    _write_manifest(d, data)


def _mut_excluded_in_manifest(d, tools):
    data = _manifest_json(d)
    data["excluded"].append({"source_rel": data["tracks"][1]["source_rel"], "reasons": ["blocklisted artist"]})
    _write_manifest(d, data)


def _mut_excluded_flag(d, tools):
    data = _manifest_json(d)
    data["tracks"][1]["features"]["excluded"] = True
    _write_manifest(d, data)


def _mut_hot_file(d, tools):
    """Reference an un-normalized hot MP3 from manifest + m3u8."""
    wav = write_wav(d.parent / "fx" / "hot.wav", music_like(8, peak_dbfs=-0.1, seed=9, drive=6))
    encode_mp3(tools, wav, d / "tracks" / "003_hot.mp3")
    data = _manifest_json(d)
    data["tracks"][2]["file"] = "tracks/003_hot.mp3"
    _write_manifest(d, data)
    _write_m3u8(d, _m3u8_rows(data))


def _mut_outside(d, tools):
    shutil.copyfile(d / _manifest_json(d)["tracks"][2]["file"], d.parent / "outside.mp3")
    data = _manifest_json(d)
    data["tracks"][2]["file"] = "../outside.mp3"
    _write_manifest(d, data)
    _write_m3u8(d, _m3u8_rows(data))


def _mut_m3u8_missing(d, tools):
    (d / "playlist.m3u8").unlink()


def _mut_m3u8_syntax(d, tools):
    text = (d / "playlist.m3u8").read_text(encoding="utf-8")
    (d / "playlist.m3u8").write_text(text.replace("#EXTM3U\n", "", 1), encoding="utf-8")


def _mut_status_failed(d, tools):
    data = _manifest_json(d)
    data["tracks"][0]["loudness"]["normalization_status"] = "failed"
    _write_manifest(d, data)


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.parametrize("mutate, code, position", [
    (_mut_missing, "file_missing", 2),
    (_mut_corrupt, "file_not_decodable", 2),
    (_mut_zero_duration, "zero_duration", 2),
    (_mut_m3u8_order, "track_order_mismatch", None),
    (_mut_index_sequence, "track_order_mismatch", None),
    (_mut_duplicate, "duplicate_source", 3),
    (_mut_excluded_in_manifest, "excluded_track_present", 2),
    (_mut_excluded_flag, "excluded_track_present", 2),
    (_mut_hot_file, "true_peak_exceeded", 3),
    (_mut_outside, "path_outside_output", 3),
    (_mut_m3u8_missing, "m3u8_missing", None),
    (_mut_m3u8_syntax, "m3u8_syntax", None),
    (_mut_status_failed, "failed_track_published", 1),
], ids=lambda v: getattr(v, "__name__", str(v)))
def test_playlist_mutations_fail(ffmpeg_tools, playlist_dir, mutate, code, position):
    mutate(playlist_dir, ffmpeg_tools)
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools))
    assert report.status is OverallStatus.FAIL
    hits = [i for i in report.issues if i.code == code and i.severity is Severity.FAIL]
    assert hits, [(i.severity.value, i.code, i.message) for i in report.issues]
    if position is not None:
        assert any(i.track_position == position for i in hits)
        assert next(t for t in report.tracks if t.position == position).status == "failed"
    assert report.summary["overall_status"] == "FAIL" and report.summary["tracks_failed"] >= (1 if position else 0)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_hot_file_is_remeasured_not_trusted(ffmpeg_tools, playlist_dir):
    """The manifest still claims 'passed' at ~-16 LUFS; QC measures the real file."""
    _mut_hot_file(playlist_dir, ffmpeg_tools)
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools))
    row = report.tracks[2]
    assert row.final_true_peak_dbtp > TP_LIMIT and row.final_lufs > CFG.target_lufs + CFG.lufs_tolerance
    assert {"true_peak_exceeded", "loudness_out_of_tolerance", "manifest_loudness_mismatch"} <= codes(report)


AVOID_RAP = {"preferences": {"aggressive_rap": PreferenceLevel.AVOID},
             "preference_tags": {"aggressive_rap": ["melodic"]}}   # track 2 carries the 'melodic' tag


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.parametrize("profile, language, code", [
    (ClientProfile(client="c", blocklist=ListRules(artists=["café ensemble"])), "fr", "blocklisted_track"),
    (ClientProfile(client="c", blocklist=ListRules(tracks=["Café Ensemble - Été à Paris"]),
                   allowlist=ListRules(artists=["Café Ensemble"])), "fr", "blocklisted_track"),  # allowlist can't override
    (ClientProfile(client="c", blocklist=ListRules(tags=["MELODIC"])), "fr", "blocklisted_track"),
    (ClientProfile(client="c", language=LanguageRules(excluded=["ru"])), "ru", "excluded_language"),
    (ClientProfile(client="c", language=LanguageRules(excluded=["ru"]),
                   allowlist=ListRules(artists=["Café Ensemble"])), "ru", "excluded_language"),  # not overridable
    (ClientProfile(client="c", language=LanguageRules(unknown_policy="exclude")), None, "excluded_language"),
    (ClientProfile(client="c", **AVOID_RAP), "fr", "avoided_preference"),
    # soft rules (avoid preferences, unknown-language policy) ARE overridden by the allowlist
    (ClientProfile(client="c", allowlist=ListRules(artists=["Café Ensemble"]), **AVOID_RAP), "fr", None),
    (ClientProfile(client="c", language=LanguageRules(unknown_policy="exclude"),
                   allowlist=ListRules(tracks=["Café Ensemble - Été à Paris.wav"])), None, None),
], ids=["artist", "track-over-allowlist", "tag", "ru-language", "ru-over-allowlist", "unknown-language-excluded",
        "avoid-preference", "avoid-allowlisted", "unknown-language-allowlisted"])
def test_profile_exclusion_rules(ffmpeg_tools, playlist_dir, profile, language, code):
    if language != "fr":
        data = _manifest_json(playlist_dir)
        data["tracks"][1]["features"]["language"] = language
        _write_manifest(playlist_dir, data)
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools), profile=profile)
    if code is None:
        assert not codes(report, Severity.FAIL), [(i.code, i.message) for i in report.issues]
        assert report.tracks[1].status != "failed"
        return
    assert report.status is OverallStatus.FAIL
    hits = issues_of(report, code)
    assert hits and all(i.severity is Severity.FAIL and i.track_position == 2 for i in hits), \
        [(i.code, i.message) for i in report.issues]
    assert [t.status for t in report.tracks] == ["passed", "failed", "passed"]


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_manifest_missing_is_a_fail_report_not_an_exception(ffmpeg_tools, playlist_dir):
    (playlist_dir / "manifest.json").unlink()
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools))
    assert report.status is OverallStatus.FAIL and "manifest_missing" in codes(report, Severity.FAIL)
    skipped = {s.split(":")[0] for s in report.checks_skipped}
    assert {"track_order", "duplicates", "excluded_tracks"} <= skipped
    # the files referenced by the playlist itself were still verified
    assert len(report.tracks) == 3 and all(t.final_lufs is not None for t in report.tracks)
    (playlist_dir / "manifest.json").write_text("{ not json", encoding="utf-8")
    assert "manifest_invalid" in codes(qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools)))
    (playlist_dir / "manifest.json").write_text('{"schema": "other"}', encoding="utf-8")
    assert "manifest_invalid" in codes(qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools)))
    empty = playlist_dir.parent / "empty"
    empty.mkdir()
    r = qc_playlist(empty, LoudnessNormalizer(ffmpeg_tools))
    assert r.status is OverallStatus.FAIL and {"manifest_missing", "m3u8_missing"} <= codes(r)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_hand_edited_manifest_shapes(ffmpeg_tools, playlist_dir):
    """null features/loudness = unknown (no crash); structurally wrong fields = manifest_invalid."""
    data = _manifest_json(playlist_dir)
    data["tracks"][0]["features"] = None
    data["tracks"][0]["loudness"] = None
    _write_manifest(playlist_dir, data)
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools))
    assert report.status is OverallStatus.PASS, [(i.code, i.message) for i in report.issues]
    assert report.unknown_metadata == 1 and report.tracks[0].original_lufs is None
    assert report.tracks[0].final_lufs is not None               # still measured by QC
    for broken in ({"excluded": ["Иван - Песня.mp3"]}, {"flagged_transitions": "none"}):
        _write_manifest(playlist_dir, {**data, **broken})
        r = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools))
        assert r.status is OverallStatus.FAIL and "manifest_invalid" in codes(r), broken
    data["tracks"][1]["features"] = "pop"
    _write_manifest(playlist_dir, data)
    assert "manifest_invalid" in codes(qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools)))


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_review_track_off_target_is_warning_and_adjacent_jump_flagged(ffmpeg_tools, playlist_dir):
    """A peak-limited REVIEW track below target -> WARNING 'requires_review' (not FAIL);
    its short-term loudness differs from its neighbours -> adjacent-transition warnings."""
    src = write_wav(playlist_dir.parent / "fx" / "spiky.wav", spiky(8, seed=5))
    r = LoudnessNormalizer(ffmpeg_tools).normalize_to_mp3(src, playlist_dir / "tracks" / "002_spiky.mp3")
    assert r.status is NormalizationStatus.REVIEW
    data = _manifest_json(playlist_dir)
    data["tracks"][1].update(file="tracks/002_spiky.mp3", loudness=r.to_metadata(),
                             duration_s=r.to_metadata()["duration_s"])
    _write_manifest(playlist_dir, data)
    _write_m3u8(playlist_dir, _m3u8_rows(data))
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools))
    assert not codes(report, Severity.FAIL), [(i.code, i.message) for i in report.issues]
    assert report.status is OverallStatus.WARNING
    review = issues_of(report, "requires_review")
    assert len(review) == 1 and review[0].severity is Severity.WARNING and review[0].track_position == 2
    assert "requires review" in review[0].message
    assert report.tracks[1].status == "review" and report.summary["tracks_requiring_review"] == 1
    jumps = issues_of(report, "adjacent_loudness_jump")
    assert {i.message.split(":")[0] for i in jumps} == {"transition #1 -> #2", "transition #2 -> #3"}
    qc_flags = [f for f in report.flagged_transitions if f["source"] == "qc"]
    assert {(f["from_index"], f["to_index"]) for f in qc_flags} == {(1, 2), (2, 3)}
    assert all(f["reasons"] for f in qc_flags)
    # the old 002 file is no longer referenced -> orphan warning
    assert "orphan_file" in codes(report, Severity.WARNING)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_flagged_sequencing_transitions_and_unknown_metadata(ffmpeg_tools, playlist_dir):
    data = _manifest_json(playlist_dir)
    data["flagged_transitions"] = [
        {"from_index": 1, "to_index": 2, "reasons": ["energy jump 0.45 > 0.25"], "kind": "sequencing"},
        {"from_index": 2, "to_index": 3, "reasons": ["unknown mood: transition not verified",
                                                      "unknown energy on both sides — energy continuity not verified"],
         "kind": "sequencing"},
        {"from_index": 2, "to_index": 3, "reasons": ["short-term p95 +5 LU"], "kind": "loudness"},
        {"from_index": 1, "to_index": 3, "reasons": ["a failed track was removed; never scored"], "kind": "unscored"},
    ]
    data["tracks"][2]["features"]["mood"] = None
    _write_manifest(playlist_dir, data)
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools))
    assert report.status is OverallStatus.WARNING
    assert report.summary["suspicious_energy_transitions"] == 1 and report.unknown_metadata == 1
    assert [i.track_position for i in issues_of(report, "suspicious_energy_transition")] == [2]
    assert issues_of(report, "flagged_transition")
    listed = [(f["from_index"], f["to_index"], f["kind"]) for f in report.flagged_transitions]
    assert listed[:4] == [(1, 2, "sequencing"), (2, 3, "sequencing"), (2, 3, "loudness"), (1, 3, "unscored")]
    sev = {i.message.split(" flagged")[0]: i.severity for i in issues_of(report, "flagged_transition")}
    assert sev == {"transition #2 -> #3": Severity.WARNING,                 # sequencer reason (not energy)
                   "transition #2 -> #3 (loudness)": Severity.INFO,         # re-checked by QC itself
                   "transition #1 -> #3 (unscored)": Severity.WARNING}
    text = render_text_report(report)
    assert "energy jump 0.45 > 0.25" in text and "Suspicious energy transitions: 1" in text


# ---------------------------------------------------------------------------
# Real FFmpeg: continuous mix
# ---------------------------------------------------------------------------

def _equal_power_join(a: np.ndarray, b: np.ndarray, xfade_s: float) -> np.ndarray:
    n = int(round(xfade_s * SR))
    if n == 0:
        return np.concatenate([a, b])
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    over = a[-n:] * np.cos(t * np.pi / 2) + b[:n] * np.sin(t * np.pi / 2)
    return np.concatenate([a[:-n], over, b[n:]]).astype(np.float32)


def _fades(x: np.ndarray, fade_in_s: float, fade_out_s: float) -> np.ndarray:
    x = x.copy()
    ni, no = int(fade_in_s * SR), int(fade_out_s * SR)
    x[:ni] *= np.linspace(0, 1, ni, dtype=np.float32)[:, None]
    x[len(x) - no:] *= np.linspace(1, 0, no, dtype=np.float32)[:, None]
    return x


def _mix_manifest(out: Path, stem: str, layout: list[tuple[str, float, float]], *, expected: float,
                  loudness: dict, fade_in_s: float, fade_out_s: float) -> Path:
    """layout: (name, duration_s, crossfade_in_s) -> consistent timeline + tracklist."""
    entries, transitions, t = [], [], 0.0
    for pos, (name, dur, xf) in enumerate(layout, start=1):
        start = 0.0 if pos == 1 else t - xf
        entries.append(MixTrackEntry(position=pos, name=name, source_rel=f"{name}.wav", start_s=start,
                                     end_s=start + dur, duration_s=dur, crossfade_in_s=xf,
                                     fx_in="—" if pos == 1 else "clean"))
        if pos > 1:
            transitions.append(TransitionReport(position=pos, crossfade_s=xf, fx="clean", curve="equal_power",
                                                overlap_start_s=start, overlap_peak_dbfs=-3.0,
                                                overlap_true_peak_dbtp=-2.0))
        t = start + dur
    rows = ["TRACKLIST: mix", "#    | Time       | Xfade  | Track"]
    rows += [f"{e.position:<4} | {format_hms(e.start_s):<10} | {e.crossfade_in_s:<6.1f} | {e.name}" for e in entries]
    (out / f"{stem}_tracklist.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = PlaylistManifest(
        generator="test", generated_at="2026-10-19T00:00:00+00:00", mode="mix", profile=None, daypart=None,
        loudness_targets={"target_lufs": CFG.target_lufs}, tracks=[],
        mix={"file": f"{stem}.mp3", "expected_duration_s": expected, "sample_rate": SR,
             "tracks": to_jsonable(entries), "transitions": to_jsonable(transitions), "loudness": loudness,
             "tracklist": f"{stem}_tracklist.txt", "seed": 0, "clipped_samples": 0,
             "fade_in_s": fade_in_s, "fade_out_s": fade_out_s})
    path = out / f"{stem}.manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


DEFECT_LAYOUT = [("Track One", 15.0, 0.0), ("Track Two", 17.0, 0.0), ("Track Three", 14.0, 2.0)]
CLEAN_LAYOUT = [("Ein", 14.0, 0.0), ("Zwei", 14.0, 2.0), ("Drei", 14.0, 2.0)]


@pytest.fixture(scope="module")
def mixes(tmp_path_factory):
    """defect mix (44 s): (a) 3 s digital silence 12-15 s ending exactly at the #1/#2
    boundary, (b) soft -45 dBFS-peak passage 21-24.5 s inside #2, (c) a hard-clipped
    burst at 38.0 s in #3. clean mix (38 s): rendered + normalized like the pipeline."""
    from conftest import TOOLS
    if TOOLS is None:
        pytest.skip("FFmpeg not available")
    out = tmp_path_factory.mktemp("qc_mix") / "микс"
    out.mkdir()
    rng = np.random.default_rng(7)
    # --- defect mix ---
    t1 = music_like(15, peak_dbfs=-7, seed=11)
    t1[12 * SR:] = 0.0                                             # (a) untrimmed digital-silence tail
    t2 = music_like(17, peak_dbfs=-7, seed=12)
    soft = pinkish_noise(int(3.5 * SR), rng)
    t2[6 * SR:int(9.5 * SR)] = (soft / np.max(np.abs(soft)) * 10 ** (-45 / 20)).astype(np.float32)   # (b)
    t3 = music_like(14, peak_dbfs=-7, seed=13)
    k = 8 * SR
    burst = 4.0 * np.sin(2 * np.pi * 150 * np.arange(int(0.04 * SR)) / SR)
    t3[k:k + len(burst)] = np.clip(t3[k:k + len(burst)] + burst[:, None], -1.0, 1.0)   # (c) clipped samples
    defect = _equal_power_join(np.concatenate([t1, t2]), t3, 2.0)
    defect = _fades(defect, 1.0, 2.0)
    assert len(defect) == 44 * SR
    wav = write_wav(out.parent / "src" / "defect.wav", defect)
    encode_mp3(TOOLS, wav, out / "defect.mp3", bitrate="320k")
    defect_manifest = _mix_manifest(out, "defect", DEFECT_LAYOUT, expected=44.0, loudness={},
                                    fade_in_s=1.0, fade_out_s=2.0)
    # --- clean mix ---
    a, b, c = (music_like(14, peak_dbfs=-3, seed=s) for s in (21, 22, 23))
    clean = _fades(_equal_power_join(_equal_power_join(a, b, 2.0), c, 2.0), 1.0, 2.0)
    assert len(clean) == 38 * SR
    wav = write_wav(out.parent / "src" / "clean.wav", clean)
    r = LoudnessNormalizer(TOOLS).normalize_to_mp3(wav, out / "clean.mp3")
    assert r.status is NormalizationStatus.PASSED, (r.errors, r.review_reasons)
    clean_manifest = _mix_manifest(out, "clean", CLEAN_LAYOUT, expected=38.0, loudness=r.to_metadata(),
                                   fade_in_s=1.0, fade_out_s=2.0)
    return {"dir": out, "defect": defect_manifest, "clean": clean_manifest}


def _copy_mix(mixes, tmp_path, which: str) -> Path:
    dest = tmp_path / mixes["dir"].name
    shutil.copytree(mixes["dir"], dest)
    return dest / mixes[which].name


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_mix_defects_detected_with_timestamps(ffmpeg_tools, mixes):
    norm = LoudnessNormalizer(ffmpeg_tools)
    report = qc_mix(mixes["defect"], norm)
    assert report.status is OverallStatus.FAIL
    assert MIX_CRITICAL <= set(report.checks_run)
    assert not [s for s in report.checks_skipped if is_critical_check(s)]
    # the manifest itself is consistent: no false timing / tracklist / duration alarms
    assert not codes(report) & {"transition_timing", "tracklist_timestamp_mismatch", "tracklist_missing",
                                "mix_duration_mismatch", "file_not_decodable"}
    gaps = report.silence_warnings
    digital = [g for g in gaps if g["digital_silence"]]
    assert len(digital) == 1
    g = digital[0]
    assert g["start_s"] == pytest.approx(12.0, abs=0.5) and g["end_s"] == pytest.approx(15.0, abs=0.5)
    assert g["classification"] == "digital_silence" and g["suspicious"] and g["severity"] == "WARNING"
    assert g["near_boundary"] and g["boundary_s"] == pytest.approx(15.0) and g["positions"][0] == 1
    quiet = [x for x in gaps if x["classification"] == "quiet_passage"]
    assert len(quiet) == 1
    q = quiet[0]
    assert q["start_s"] == pytest.approx(21.0, abs=0.5) and q["end_s"] == pytest.approx(24.5, abs=0.5)
    assert q["severity"] == "INFO" and not q["suspicious"] and not q["digital_silence"]
    assert -90 < q["min_level_dbfs"] < QC.silence_threshold_dbfs and q["positions"] == [2]
    assert len(gaps) == 2 and report.summary["silence_warnings"] == 1
    sil = issues_of(report, "silence_digital_silence")
    assert sil[0].severity is Severity.WARNING and sil[0].timestamp_s == pytest.approx(12.0, abs=0.5)
    assert "0:00:12" in sil[0].message and "s)" in sil[0].message
    # clipping, with first timestamps
    clips = issues_of(report, "clipping_detected")
    assert clips and all(i.severity is Severity.FAIL for i in clips)
    assert clips[0].timestamp_s == pytest.approx(38.0, abs=0.1) and clips[0].track_position == 3
    assert report.mix["clipped_samples"] > 0 and "0:00:38" in clips[0].message
    # final loudness / true peak measured on the mix MP3 itself
    m = norm.measure(mixes["dir"] / "defect.mp3")
    assert report.mix["integrated_lufs"] == pytest.approx(m.integrated_lufs, abs=0.05)
    assert report.mix["true_peak_dbtp"] == pytest.approx(m.true_peak_dbtp, abs=0.05)
    assert report.mix["true_peak_dbtp"] > TP_LIMIT and "true_peak_exceeded" in codes(report, Severity.FAIL)
    assert report.mix["decoded_duration_s"] == pytest.approx(44.0, abs=0.1)
    assert len(report.mix["segments"]) == 3 and len(report.tracks) == 3
    text = render_text_report(report)
    for label in ("Tracks processed:", "Tracks passed:", "Tracks failed:", "Tracks requiring review:",
                  "Target LUFS:", "Measured LUFS range:", "Maximum measured True Peak:", "Excluded tracks:",
                  "Unknown metadata:", "Suspicious energy transitions:", "Silence warnings: 1", "Overall status: FAIL"):
        assert label in text
    assert "0:00:12 - 0:00:15" in text and "0:00:38" in text


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_scan_silence_public_api_streams_and_classifies(ffmpeg_tools, mixes):
    path = mixes["dir"] / "defect.mp3"
    gaps = scan_silence(ffmpeg_tools, path, QC, boundaries_s=[15.0, 30.0, 32.0], fade_in_s=1.0, fade_out_s=2.0)
    assert [g["classification"] for g in gaps] == ["digital_silence", "quiet_passage"]
    # no boundary information -> the digital gap is still digital, the quiet passage still quiet
    assert [g["near_boundary"] for g in scan_silence(ffmpeg_tools, path, QC)] == [False, False]
    bad = mixes["dir"].parent / "garbage.mp3"
    bad.write_bytes(b"\x00" * 1000)
    with pytest.raises(FFmpegError):             # undecodable is never reported as "no silence"
        scan_silence(ffmpeg_tools, bad, QC)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_streaming_scan_is_chunk_independent(ffmpeg_tools, mixes):
    """The mix is processed in bounded chunks; results must not depend on where the
    chunk boundaries fall (window carry-over, clip events spanning chunks)."""
    path = mixes["dir"] / "defect.mp3"
    small = scan_pcm(ffmpeg_tools, path, QC, chunk_s=0.5)       # one RMS window per read
    large = scan_pcm(ffmpeg_tools, path, QC, chunk_s=120.0)     # whole file in one read
    assert small.errors == [] and large.errors == []
    assert small.decoded_duration_s == large.decoded_duration_s == pytest.approx(44.0, abs=0.05)
    assert len(small.levels_dbfs) == len(large.levels_dbfs) == 88
    np.testing.assert_allclose(small.levels_dbfs, large.levels_dbfs, atol=1e-6)
    assert small.clipped_samples == large.clipped_samples > 0
    assert small.clip_events == large.clip_events and small.clip_event_count == 1
    assert small.peak_dbfs == large.peak_dbfs


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_clean_mix_passes(ffmpeg_tools, mixes):
    report = qc_mix(mixes["clean"], LoudnessNormalizer(ffmpeg_tools))
    assert report.status is OverallStatus.PASS, [(i.severity.value, i.code, i.message) for i in report.issues]
    assert MIX_CRITICAL <= set(report.checks_run)
    assert not [s for s in report.checks_skipped if is_critical_check(s)]
    assert abs(report.mix["integrated_lufs"] - CFG.target_lufs) <= CFG.lufs_tolerance
    assert report.mix["true_peak_dbtp"] <= TP_LIMIT and report.mix["clipped_samples"] == 0
    assert report.silence_warnings == []
    assert [t.status for t in report.tracks] == ["passed"] * 3
    assert all(t.final_lufs is not None for t in report.tracks)
    assert report.summary["tracks_processed"] == 3 and report.summary["overall_status"] == "PASS"


def _mix_edit(path: Path, fn) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _wrong_duration(p: Path):
    _mix_edit(p, lambda d: d["mix"].update(expected_duration_s=50.0))


def _bad_start(p: Path):
    def fn(d):
        d["mix"]["tracks"][2]["start_s"] += 0.5
        d["mix"]["tracks"][2]["end_s"] += 0.5
    _mix_edit(p, fn)


def _bad_tracklist(p: Path):
    tl = p.parent / "clean_tracklist.txt"
    tl.write_text(tl.read_text(encoding="utf-8").replace("0:00:24", "0:00:26"), encoding="utf-8")


def _no_tracklist_file(p: Path):
    (p.parent / "clean_tracklist.txt").unlink()


def _mix_file_missing(p: Path):
    (p.parent / "clean.mp3").unlink()


def _nan_start(p: Path):
    """NaN would slip through every tolerance comparison -> must be rejected."""
    text = p.read_text(encoding="utf-8")
    data = json.loads(text)
    data["mix"]["tracks"][1]["start_s"] = "NaN"
    p.write_text(json.dumps(data), encoding="utf-8")


def _missing_field(p: Path):
    _mix_edit(p, lambda d: d["mix"]["tracks"][2].pop("end_s"))


def _peak_protection(p: Path):
    def fn(d):
        d["mix"]["transitions"][1].update(gain_reduction_db=1.5, uncertain=True, flags=["tempo mismatch 9%"])
        # pipeline also copies renderer flags into the manifest as kind "transition"
        d["flagged_transitions"] = [{"from_index": 2, "to_index": 3, "reasons": ["tempo mismatch 9%"],
                                     "kind": "transition"}]
    _mix_edit(p, fn)


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.parametrize("mutate, code, severity", [
    (_wrong_duration, "mix_duration_mismatch", Severity.FAIL),
    (_bad_start, "transition_timing", Severity.FAIL),
    (_bad_tracklist, "tracklist_timestamp_mismatch", Severity.FAIL),
    (_no_tracklist_file, "tracklist_missing", Severity.FAIL),
    (_mix_file_missing, "file_missing", Severity.FAIL),
    (_nan_start, "manifest_invalid", Severity.FAIL),
    (_missing_field, "manifest_invalid", Severity.FAIL),
    (_peak_protection, "peak_protection_applied", Severity.WARNING),
], ids=lambda v: getattr(v, "__name__", str(v)))
def test_mix_manifest_mutations(ffmpeg_tools, mixes, tmp_path, mutate, code, severity):
    manifest = _copy_mix(mixes, tmp_path, "clean")
    mutate(manifest)
    report = qc_mix(manifest, LoudnessNormalizer(ffmpeg_tools))
    hits = issues_of(report, code)
    assert hits and all(i.severity is severity for i in hits), [(i.code, i.message) for i in report.issues]
    expected_status = OverallStatus.FAIL if severity is Severity.FAIL else OverallStatus.WARNING
    assert report.status is expected_status
    if code == "transition_timing":
        assert hits[0].track_position == 3 and "off by +0.500 s" in hits[0].message
    if code == "tracklist_timestamp_mismatch":
        assert hits[0].track_position == 3 and "0:00:26" in hits[0].message and "0:00:24" in hits[0].message
    if code == "peak_protection_applied":
        assert hits[0].timestamp_s == pytest.approx(24.0) and "0:00:24" in hits[0].message
        assert issues_of(report, "uncertain_transition")[0].severity is Severity.WARNING
        pair = [f for f in report.flagged_transitions if f["to_index"] == 3]
        assert len(pair) == 1 and pair[0]["source"] == "renderer"            # not listed twice
        assert "peak protection reduced the overlap by 1.5 dB" in pair[0]["reasons"]
    if code == "file_missing":
        assert {"silence: mix file could not be decoded", "clipping: mix file could not be decoded"} <= \
            set(report.checks_skipped)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_mix_skipped_critical_check_is_never_pass(ffmpeg_tools, mixes, tmp_path):
    manifest = _copy_mix(mixes, tmp_path, "clean")

    def drop(d):
        del d["mix"]["expected_duration_s"]
        d["mix"]["tracklist"] = None
    _mix_edit(manifest, drop)
    report = qc_mix(manifest, LoudnessNormalizer(ffmpeg_tools))
    assert not codes(report, Severity.FAIL) and not codes(report, Severity.WARNING)
    assert {"mix_duration", "tracklist_timestamps"} <= {s.split(":")[0] for s in report.checks_skipped}
    assert report.status is OverallStatus.WARNING          # all measured checks passed, still not PASS
    assert "Overall status: WARNING" in render_text_report(report)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_mix_manifest_missing_or_wrong_mode(ffmpeg_tools, golden_playlist, tmp_path):
    norm = LoudnessNormalizer(ffmpeg_tools)
    r = qc_mix(tmp_path / "nope.manifest.json", norm)
    assert r.status is OverallStatus.FAIL and codes(r) == {"manifest_missing"}
    r = qc_mix(golden_playlist / "manifest.json", norm)
    assert r.status is OverallStatus.FAIL and "manifest_wrong_mode" in codes(r)


# ---------------------------------------------------------------------------
# Regression tests for review findings (the finding id is named in each test)
# ---------------------------------------------------------------------------

def _fine_levels(total_s: float, segments: list[tuple[float, float, float]], base: float = -18.0,
                 window_s: float = 0.05) -> np.ndarray:
    """Levels on a ``window_s`` grid at ``base`` dBFS with [start, end) segments set to a level."""
    lv = np.full(int(round(total_s / window_s)), base)
    for a, b, level in segments:
        lv[int(round(a / window_s)):int(round(b / window_s))] = level
    return lv


def _click_bed(lv: np.ndarray, start_s: float, end_s: float, window_s: float, every_s: float = 2.0,
               click_dbfs: float = -25.0) -> np.ndarray:
    """A short transient (one window) every ``every_s`` inside [start_s, end_s)."""
    t = start_s + every_s / 2
    while t < end_s:
        lv[int(round(t / window_s))] = click_dbfs
        t += every_s
    return lv


@pytest.mark.parametrize("window_s", [0.05, 0.5])
def test_spec15_quiet_bed_with_clicks_is_one_quiet_passage(window_s):
    """SPEC-15: one quiet track (bed at -55 dBFS, a click every 2 s) right after a
    transition was split at every click into WARNING 'transition gaps' near the
    boundaries plus dozens of INFO fragments. Now: one merged quiet passage (INFO)."""
    lv = _fine_levels(120, [(30.0, 90.0, -55.0)], window_s=window_s)
    _click_bed(lv, 30.0, 90.0, window_s)          # clicks at 31, 33, ... 89 s: the last run is < 1 s
    gaps = detect_silence_gaps(lv, window_s, QC, total_duration_s=120.0, boundaries_s=[29.0, 30.0, 90.0, 91.0])
    assert len(gaps) == 1, [(g["start_s"], g["end_s"], g["classification"]) for g in gaps]
    g = gaps[0]
    assert g["classification"] == "quiet_passage" and g["severity"] == "INFO" and not g["suspicious"]
    assert g["start_s"] == pytest.approx(30.0, abs=1e-6) and g["end_s"] == pytest.approx(90.0, abs=1e-6)
    assert g["near_boundary"] and g["merged_runs"] == 31 and g["silent_share"] >= 0.7
    assert "extends into the track body" in g["message"]


def test_spec15_merging_keeps_real_gaps_and_ignores_rhythmic_rests():
    """SPEC-15: merging must not hide real gaps nor invent them."""
    w = 0.05
    # digital silence interrupted by one click is ONE digital gap
    lv = _fine_levels(60, [(20.0, 30.0, -200.0)])
    lv[int(round(25.0 / w))] = -20.0
    gaps = detect_silence_gaps(lv, w, QC, total_duration_s=60.0)
    assert [(g["classification"], g["merged_runs"], g["start_s"], g["end_s"]) for g in gaps] == \
        [("digital_silence", 2, 20.0, 30.0)]
    # a rhythmic passage with 0.15 s rests is not merged into a "gap"
    lv = _fine_levels(60, [])
    for k in range(int(20 / w), int(40 / w), 7):
        lv[k:k + 3] = -200.0
    assert detect_silence_gaps(lv, w, QC, total_duration_s=60.0) == []
    # a short quiet gap straddling a boundary is still a suspicious transition gap
    lv = _fine_levels(120, [(58.0, 60.5, -62.0), (80.0, 84.0, -58.0)])
    gaps = detect_silence_gaps(lv, w, QC, total_duration_s=120.0, boundaries_s=[61.0])
    assert [(g["classification"], g["severity"]) for g in gaps] == [("transition_gap", "WARNING"),
                                                                   ("quiet_passage", "INFO")]


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_spec15_decoded_quiet_track_with_clicks_is_one_info_region(ffmpeg_tools, tmp_path):
    """SPEC-15 on real decoded audio: a quiet bed (-62 dBFS peak) with a click every
    2 s between two normal tracks is ONE quiet passage, not warnings + fragments."""
    x = np.concatenate([music_like(30, peak_dbfs=-3, seed=1), spiky(60, bed_peak_dbfs=-62, spike=0.3, seed=2),
                        music_like(30, peak_dbfs=-3, seed=3)])
    mp3 = encode_mp3(ffmpeg_tools, write_wav(tmp_path / "bed.wav", x), tmp_path / "bed.mp3")
    gaps = scan_silence(ffmpeg_tools, mp3, QC, boundaries_s=[30.0, 90.0])
    assert [(g["classification"], g["severity"]) for g in gaps] == [("quiet_passage", "INFO")], gaps
    assert gaps[0]["start_s"] == pytest.approx(30.0, abs=0.1) and gaps[0]["end_s"] == pytest.approx(90.0, abs=0.1)
    assert gaps[0]["merged_runs"] >= 25


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_dsp5_silence_edges_at_subwindow_resolution(ffmpeg_tools, tmp_path):
    """DSP-5: a 1.70 s digital-zero gap straddling 0.5 s window borders kept only two
    whole coarse windows (1.0 s < 1.5 s) and was missed. The sub-window grid finds it,
    with edges accurate to ~50 ms."""
    x = music_like(30, peak_dbfs=-3, seed=5)
    truth = [(10.27, 11.97), (20.0, 21.7)]
    for a, b in truth:
        x[int(a * SR):int(b * SR)] = 0.0
    wav = write_wav(tmp_path / "gaps.wav", x)
    scan = scan_pcm(ffmpeg_tools, wav, QC)
    coarse = detect_silence_gaps(scan.levels_dbfs, scan.window_s, QC, total_duration_s=scan.decoded_duration_s)
    assert len(coarse) == 1                       # the old 0.5 s grid alone misses the gap at 10.27 s
    gaps = scan_silence(ffmpeg_tools, wav, QC)
    assert [g["classification"] for g in gaps] == ["digital_silence", "digital_silence"]
    for g, (a, b) in zip(gaps, truth):
        assert g["start_s"] == pytest.approx(a, abs=0.06) and g["end_s"] == pytest.approx(b, abs=0.06)
    assert scan.sub_window_s == pytest.approx(0.05) and len(scan.sub_levels_dbfs) == 600
    small = scan_pcm(ffmpeg_tools, wav, QC, chunk_s=0.5)      # the fine grid is chunk-independent too
    np.testing.assert_allclose(small.sub_levels_dbfs, scan.sub_levels_dbfs, atol=1e-6)


def test_spec1_excluded_stage_and_summary_counts():
    """SPEC-1 / INT-16 (report side): normalization failures are 'Tracks failed' rows,
    not 'Excluded tracks'; analysis failures and curation exclusions are exclusions.
    Older manifests without 'stage' are recognised by the reason prefix."""
    assert excluded_stage({"source_rel": "a", "reasons": ["blocklisted artist 'x'"]}) == "curation"
    assert excluded_stage({"stage": "analysis", "reasons": ["analysis failed: probe"]}) == "analysis"
    assert excluded_stage({"reasons": ["analysis failed: probe failed"]}) == "analysis"
    assert excluded_stage({"reasons": ["normalization failed: near-silent"]}) == "normalization"
    assert excluded_stage({"stage": "curation", "loudness": {"normalization_status": "failed"}}) == "normalization"
    r = _sample_report()
    r.excluded = r.excluded + [
        {"source_rel": "x.mp3", "stage": "normalization", "reasons": ["normalization failed: y"]},
        {"source_rel": "broken.mp3", "stage": "analysis", "reasons": ["analysis failed: z"]}]
    r.tracks = r.tracks + [TrackQC(0, "", "x.mp3", -40.0, None, -20.0, None, None, "failed",
                                   ["FAIL: not published — y"])]
    s = compute_summary(r)
    assert (s["excluded_tracks"], s["tracks_failed"], s["tracks_processed"]) == (2, 2, 4)
    text = render_text_report(r)
    assert "Excluded tracks: 2" in text and "Tracks failed: 2" in text
    assert "(not published) x.mp3" in text
    excluded_part = text.split("Excluded tracks\n---------------\n")[1].split("\n\n")[0]
    assert "broken.mp3 [analysis]" in excluded_part and "x.mp3" not in excluded_part


def _add_excluded(d: Path, entry: dict) -> None:
    data = _manifest_json(d)
    data["excluded"].append(entry)
    _write_manifest(d, data)


FAILED_NORMALIZATION = {"source_rel": "Delta - Whisper.wav", "stage": "normalization", "sequence_position": 2,
                        "reasons": ["normalization failed: input is near-silent (-67.0 LUFS)"],
                        "loudness": {"normalization_status": "failed", "original_lufs": -67.0,
                                     "original_true_peak_dbtp": -50.0}}


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_spec1_failed_normalization_fails_playlist_qc(ffmpeg_tools, playlist_dir):
    """SPEC-1: a selected track that FAILED normalization was only listed in
    manifest.excluded -> 'Tracks failed: 0', Overall PASS, exit 0."""
    _add_excluded(playlist_dir, FAILED_NORMALIZATION)
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools))
    assert report.status is OverallStatus.FAIL
    hits = issues_of(report, "normalization_failed")
    assert len(hits) == 1 and hits[0].severity is Severity.FAIL and hits[0].track_position is None
    assert "near-silent" in hits[0].message and "planned position 2" in hits[0].message
    failed = [t for t in report.tracks if t.status == "failed"]
    assert [(t.position, t.source_rel, t.original_lufs) for t in failed] == [(0, "Delta - Whisper.wav", -67.0)]
    assert [t.status for t in report.tracks[:3]] == ["passed"] * 3        # published tracks unaffected
    s = report.summary
    assert (s["tracks_processed"], s["tracks_passed"], s["tracks_failed"]) == (4, 3, 1)
    assert s["excluded_tracks"] == 1                                       # the curation exclusion only
    text = render_text_report(report)
    assert "Tracks failed: 1" in text and "(not published) Delta - Whisper.wav" in text


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_int16_undecodable_source_is_a_warning(ffmpeg_tools, playlist_dir):
    """INT-16: an undecodable library file (stage 'analysis') gave a clean PASS / exit 0."""
    _add_excluded(playlist_dir, {"source_rel": "Demo Broken - Corrupt File.mp3", "stage": "analysis",
                                 "reasons": ["analysis failed: probe failed: invalid data"]})
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools))
    assert report.status is OverallStatus.WARNING
    hits = issues_of(report, "source_analysis_failed")
    assert len(hits) == 1 and hits[0].severity is Severity.WARNING and "Corrupt File" in hits[0].message
    assert report.summary["excluded_tracks"] == 2 and report.summary["tracks_failed"] == 0
    assert "Demo Broken - Corrupt File.mp3 [analysis]" in render_text_report(report)


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.parametrize("language, failed", [("en", False), (None, True)])
def test_int3_spec2_cyrillic_rule_only_for_unknown_language(ffmpeg_tools, playlist_dir, language, failed):
    """INT-3 / SPEC-2: curation lifts the Cyrillic-script rule once the curator sets a
    language (metadata CSV -> source 'manual'); QC FAILed the curator-confirmed track
    anyway (excluded_script)."""
    data = _manifest_json(playlist_dir)
    feats = data["tracks"][1]["features"]
    feats.update(script_hint="cyrillic", language=language)
    if language is not None:
        feats["feature_sources"]["language"] = "manual"
    _write_manifest(playlist_dir, data)
    profile = ClientProfile(client="c", language=LanguageRules(unknown_policy="allow",
                                                               cyrillic_script_policy="exclude"))
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools), profile=profile)
    hits = issues_of(report, "excluded_script")
    if not failed:
        assert not hits and report.status is OverallStatus.PASS, [(i.code, i.message) for i in report.issues]
        return
    assert hits and all(i.severity is Severity.FAIL and i.track_position == 2 for i in hits)
    assert "Cyrillic" in hits[0].message and report.status is OverallStatus.FAIL


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_int5_allowlist_matches_like_curation(ffmpeg_tools, playlist_dir):
    """INT-5: QC's own matcher missed en-dash 'Artist – Title' entries and normalized /
    multi-valued tags ('dance pop' vs 'Dance-Pop/Rock'), so tracks curation allowlisted
    FAILed 'not_allowlisted'."""
    data = _manifest_json(playlist_dir)
    data["tracks"][2]["features"]["genre"] = "Dance-Pop/Rock"
    _write_manifest(playlist_dir, data)
    profile = ClientProfile(client="c", allowlist=ListRules(tracks=["Café Ensemble – Été à Paris"],   # en dash
                                                           tags=["dance pop"], only=True))
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools), profile=profile)
    hits = issues_of(report, "not_allowlisted")
    assert [i.track_position for i in hits] == [1], [(i.code, i.message) for i in report.issues]
    assert [t.status for t in report.tracks] == ["failed", "passed", "passed"]


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.parametrize("language, code", [("por", None), ("deu", "language_not_allowed")])
def test_int9_iso639_2_languages_like_curation(ffmpeg_tools, playlist_dir, language, code):
    """INT-9: 'por' (ID3 TLAN, ISO 639-2) is 'pt' for curation but was compared raw by QC
    (false WARNING). A language curation rejects is now a FAIL (it would be excluded)."""
    data = _manifest_json(playlist_dir)
    data["tracks"][1]["features"]["language"] = language
    _write_manifest(playlist_dir, data)
    profile = ClientProfile(client="c", language=LanguageRules(allowed=["en", "fr", "es", "pt"]))
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools), profile=profile)
    if code is None:
        assert report.status is OverallStatus.PASS, [(i.code, i.message) for i in report.issues]
        return
    hits = issues_of(report, code)
    assert hits and all(i.severity is Severity.FAIL and i.track_position == 2 for i in hits)
    assert "'de'" in hits[0].message


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_spec4_int15_not_selected_tracks_listed_as_info(ffmpeg_tools, playlist_dir):
    """SPEC-4 / INT-15 (report side): eligible tracks the sequencer dropped vanished from
    every report; manifest.not_selected is now listed (INFO, does not change the status)."""
    data = _manifest_json(playlist_dir)
    data["not_selected"] = [{"source_rel": "Demo Field - Drone Study No. 3.mp3", "track_id": "abc",
                             "reason": "poor fit for every daypart (energy 0.10 below 0.15)"}]
    _write_manifest(playlist_dir, data)
    report = qc_playlist(playlist_dir, LoudnessNormalizer(ffmpeg_tools))
    assert report.status is OverallStatus.PASS, [(i.code, i.message) for i in report.issues]
    hits = issues_of(report, NOT_SELECTED_CODE)
    assert len(hits) == 1 and hits[0].severity is Severity.INFO and "poor fit" in hits[0].message
    assert hits[0].file == "Demo Field - Drone Study No. 3.mp3"
    assert report.summary["not_selected"] == 1 and report.summary["excluded_tracks"] == 1
    text = render_text_report(report)
    section = text.split("Not selected (eligible, not placed by the sequencer)\n")[1]
    assert "Drone Study No. 3" in section.split("\n\n")[0] and "poor fit" in section


def _mix_with_tracks(mixes, tmp_path, features: dict[int, dict]) -> Path:
    """Clean-mix copy whose manifest carries per-track entries like the pipeline writes
    them (``features`` = TrackFeatures.to_dict() without 'path'), overridden per position."""
    manifest = _copy_mix(mixes, tmp_path, "clean")

    def fn(d):
        tracks = []
        for e in d["mix"]["tracks"]:
            name, rel = e["name"], e["source_rel"]
            feats = TrackFeatures(path=Path(rel), duration_s=e["duration_s"], energy=0.5, mood="calm",
                                  genre="pop", language="en", artist=f"{name} Artist", title=name,
                                  rel_path=rel, approved=True).to_dict()
            feats.pop("path")
            feats.update(features.get(e["position"], {}))
            tracks.append({"index": e["position"], "file": "", "title": name, "artist": f"{name} Artist",
                           "duration_s": e["duration_s"], "source_rel": rel,
                           "source_path": str(manifest.parent / rel), "track_id": make_track_id(rel),
                           "loudness": {"normalization_status": "passed"}, "features": feats})
        d["tracks"] = tracks
    _mix_edit(manifest, fn)
    return manifest


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.parametrize("edit", [
    {"published": False, "errors": ["PermissionError: [WinError 5] mix.mp3 is open in a player"]},
    {"errors": ["final mix true peak over after 3 attempts"]},           # older manifest: no 'published'
], ids=["published-false", "final-normalization-failed"])
def test_int2_safe4_dsp3_unpublished_mix_fails(ffmpeg_tools, mixes, tmp_path, edit):
    """INT-2 / SAFE-4 / DSP-3: when the final encode failed, the manifest said so
    (published=false, status 'failed') but QC measured the previous run's mix.mp3 and
    reported PASS. Now: FAIL, and the stale file is not measured."""
    manifest = _copy_mix(mixes, tmp_path, "clean")

    def fn(d):
        if "published" in edit:
            d["mix"]["published"] = edit["published"]
        d["mix"]["loudness"].update(normalization_status="failed", errors=edit["errors"])
        d["warnings"] = [f"final mix: {edit['errors'][0]}"]
    _mix_edit(manifest, fn)
    report = qc_mix(manifest, LoudnessNormalizer(ffmpeg_tools))
    assert report.status is OverallStatus.FAIL
    hits = issues_of(report, "mix_not_published")
    assert len(hits) == 1 and hits[0].severity is Severity.FAIL and edit["errors"][0] in hits[0].message
    assert report.mix.get("integrated_lufs") is None and report.silence_warnings == []   # not measured
    assert "mix_loudness: the mix was not published" in report.checks_skipped
    assert [t.status for t in report.tracks] == ["failed"] * 3
    assert (report.summary["tracks_passed"], report.summary["tracks_failed"]) == (0, 3)
    text = render_text_report(report)
    assert "Published: NO" in text and f"Export warning: final mix: {edit['errors'][0]}" in text


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_safe4_published_mix_hash_and_size_verified(ffmpeg_tools, mixes, tmp_path):
    """SAFE-4: the manifest records sha256 + size of the published MP3; a stale or
    replaced file at the same path must never pass."""
    norm = LoudnessNormalizer(ffmpeg_tools)
    manifest = _copy_mix(mixes, tmp_path, "clean")
    mp3 = manifest.parent / "clean.mp3"
    _mix_edit(manifest, lambda d: d["mix"].update(published=True, sha256=sha256(mp3), size_bytes=mp3.stat().st_size))
    ok = qc_mix(manifest, norm)
    assert ok.status is OverallStatus.PASS, [(i.code, i.message) for i in ok.issues]
    assert "mix_file_integrity" in ok.checks_run
    raw = bytearray(mp3.read_bytes())
    raw[len(raw) // 2] ^= 0xFF                      # same size, different content
    mp3.write_bytes(bytes(raw))
    bad = qc_mix(manifest, norm)
    hits = issues_of(bad, "mix_file_mismatch")
    assert bad.status is OverallStatus.FAIL and hits and "differ" in hits[0].message
    assert bad.mix.get("integrated_lufs") is None and [t.status for t in bad.tracks] == ["failed"] * 3
    mp3.write_bytes(bytes(raw) + b"\x00" * 10)      # different size
    hits = issues_of(qc_mix(manifest, norm), "mix_file_mismatch")
    assert hits and "bytes" in hits[0].message


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_vsafe3_tag_only_edit_of_a_delivered_mix_still_passes(ffmpeg_tools, mixes, tmp_path):
    """VSAFE-3: current manifests hash the MP3 AUDIO payload (ID3/APE excluded): a curator's
    tag-only edit passes, a changed audio payload still FAILs."""
    import subprocess
    from veg_mixer.fsutil import mp3_audio_sha256
    norm = LoudnessNormalizer(ffmpeg_tools)
    manifest = _copy_mix(mixes, tmp_path, "clean")
    mp3 = manifest.parent / "clean.mp3"
    audio_sha, audio_size = mp3_audio_sha256(mp3)
    _mix_edit(manifest, lambda d: d["mix"].update(published=True, audio_sha256=audio_sha,
                                                  audio_size_bytes=audio_size))
    retagged = mp3.with_name("retagged.mp3")
    subprocess.run([ffmpeg_tools.ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", str(mp3), "-map", "0:a",
                    "-c", "copy", "-metadata", "title=Edited by curator", "-id3v2_version", "3", str(retagged)],
                   check=True, capture_output=True)
    assert mp3_audio_sha256(retagged) == (audio_sha, audio_size)       # only the tag changed
    retagged.replace(mp3)
    ok = qc_mix(manifest, norm)
    assert not issues_of(ok, "mix_file_mismatch"), [(i.code, i.message) for i in ok.issues]
    raw = bytearray(mp3.read_bytes())
    raw[-2000] ^= 0xFF                                                  # audio payload changed
    mp3.write_bytes(bytes(raw))
    hits = issues_of(qc_mix(manifest, norm), "mix_file_mismatch")
    assert hits and "audio payload" in hits[0].message


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_spec1_failed_track_in_mix_counts_as_failed(ffmpeg_tools, mixes, tmp_path):
    """SPEC-1 (mix): a track that failed normalization and was left out of the mix
    gave 'Tracks failed: 0' / PASS. Also lists manifest.not_selected (SPEC-4)."""
    manifest = _copy_mix(mixes, tmp_path, "clean")

    def fn(d):
        d["excluded"] = [{"source_rel": "Vier.wav", "stage": "normalization",
                          "reasons": ["normalization failed: decode error at 0:01:02"],
                          "loudness": {"normalization_status": "failed", "original_lufs": -30.0}}]
        d["not_selected"] = [{"source_rel": "Fünf.wav", "reason": "duration limit reached"}]
    _mix_edit(manifest, fn)
    report = qc_mix(manifest, LoudnessNormalizer(ffmpeg_tools))
    assert report.status is OverallStatus.FAIL
    assert [i.severity for i in issues_of(report, "normalization_failed")] == [Severity.FAIL]
    s = report.summary
    assert (s["tracks_processed"], s["tracks_passed"], s["tracks_failed"], s["excluded_tracks"]) == (4, 3, 1, 0)
    assert [t.status for t in report.tracks] == ["passed", "passed", "passed", "failed"]
    assert len(issues_of(report, NOT_SELECTED_CODE)) == 1 and s["not_selected"] == 1


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_spec5_mix_qc_reports_editorial_review_flags(ffmpeg_tools, mixes, tmp_path):
    """SPEC-5: mix QC only looked at loudness 'review'; editorial review flags (unknown
    language, not approved) were reported as 'passed'."""
    manifest = _mix_with_tracks(mixes, tmp_path, {2: {
        "language": None, "requires_review": True,
        "review_reasons": ["language unknown — needs curator review (not assumed to be English)"]}})
    report = qc_mix(manifest, LoudnessNormalizer(ffmpeg_tools))
    hits = issues_of(report, "requires_review")
    assert len(hits) == 1 and hits[0].severity is Severity.WARNING and hits[0].track_position == 2
    assert hits[0].timestamp_s == pytest.approx(12.0) and "language unknown" in hits[0].message
    assert [t.status for t in report.tracks] == ["passed", "review", "passed"]
    assert report.summary["tracks_requiring_review"] == 1 and report.status is OverallStatus.WARNING


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_int17_spec11_mix_qc_applies_the_client_profile(ffmpeg_tools, mixes, tmp_path):
    """INT-17 / SPEC-11: qc_mix had no profile parameter ('profile_rules: no client
    profile supplied' even with --profile)."""
    norm = LoudnessNormalizer(ffmpeg_tools)
    manifest = _mix_with_tracks(mixes, tmp_path, {3: {"genre": "Hard-Rock"}})
    profile = ClientProfile(client="c", blocklist=ListRules(tags=["hard rock"]))
    report = qc_mix(manifest, norm, profile=profile)
    assert "profile_rules" in report.checks_run
    hits = issues_of(report, "blocklisted_track")
    assert hits and all(i.severity is Severity.FAIL and i.track_position == 3 for i in hits)
    assert [t.status for t in report.tracks] == ["passed", "passed", "failed"]
    assert report.status is OverallStatus.FAIL
    # without per-track features the rules cannot be verified: listed as skipped, never "passed"
    plain = qc_mix(mixes["clean"], norm, profile=profile)
    assert "profile_rules: mix manifest carries no per-track features" in plain.checks_skipped


def test_vspec1_dead_air_ending_at_a_transition_stays_a_warning():
    """VSPEC-1: 12 s of dead air (-60 dBFS, not digital) that ends where the next track
    starts is a suspicious transition gap however long it is."""
    from veg_mixer.quality_control import SILENCE_SUBWINDOW_S, detect_silence_gaps
    cfg = QCConfig()
    w = SILENCE_SUBWINDOW_S
    levels = np.full(int(60 / w), -20.0)
    levels[int(30 / w):int(42 / w)] = -60.0            # dead air 30..42 s, next track at 42 s
    gaps = detect_silence_gaps(levels, w, cfg, total_duration_s=60.0, boundaries_s=[42.0])
    assert len(gaps) == 1 and gaps[0]["classification"] == "transition_gap" and gaps[0]["suspicious"]


def test_spec15_quiet_bed_with_clicks_is_one_quiet_passage():
    """SPEC-15 (re-verification): a 60 s quiet bed with clicks every ~1 s that runs into the
    transition is ONE region and the track's own material (INFO), not fragments or a
    'suspicious gap'."""
    from veg_mixer.quality_control import SILENCE_SUBWINDOW_S, detect_silence_gaps
    cfg = QCConfig()
    w = SILENCE_SUBWINDOW_S
    levels = np.full(int(80 / w), -20.0)
    bed = slice(int(10 / w), int(70 / w))
    levels[bed] = -55.0
    for k in range(int(10 / w), int(70 / w), int(1.0 / w)):
        levels[k] = -25.0                              # one 50 ms click per second
    gaps = detect_silence_gaps(levels, w, cfg, total_duration_s=80.0, boundaries_s=[70.0])
    assert len(gaps) == 1
    assert gaps[0]["classification"] == "quiet_passage" and not gaps[0]["suspicious"]
    assert gaps[0]["duration_s"] >= 55


@pytest.mark.parametrize("dead_air_s", [3, 12, 24, 30, 60])
def test_vspec1_dead_air_at_a_tracks_tail_is_a_warning_at_any_length(dead_air_s):
    """VSPEC-1 (round 2): longer dead air at a transition must never look LESS suspicious;
    only a region covering most of its own track (a quiet track) is INFO."""
    from veg_mixer.quality_control import SILENCE_SUBWINDOW_S, detect_silence_gaps
    w = SILENCE_SUBWINDOW_S
    levels = np.full(int(400 / w), -20.0)              # tracks: 0-200 s and 200-400 s
    levels[int((200 - dead_air_s) / w):int(200 / w)] = -62.0
    gaps = detect_silence_gaps(levels, w, QCConfig(), total_duration_s=400.0, boundaries_s=[200.0])
    assert len(gaps) == 1 and gaps[0]["classification"] == "transition_gap", gaps


def test_r2b1_distinct_files_whose_names_casefold_alike_are_not_duplicates():
    """R2B-1: 'Straße.mp3' and 'Strasse.mp3' are two files (track ids differ since the
    VSPEC-2 fix); QC must not report them as one source."""
    from veg_mixer.models import ManifestTrack
    from veg_mixer.quality_control import _Builder, _check_duplicates
    tracks = [ManifestTrack(index=i + 1, file=f"tracks/00{i + 1}.mp3", title=n, artist="A", duration_s=10.0,
                            source_rel=n, source_path=f"/lib/{n}", track_id=make_track_id(n))
              for i, n in enumerate(["Strasse.mp3", "Straße.mp3"])]
    b = _Builder()
    _check_duplicates(b, tracks)
    assert not [i for i in b.issues if i.code == "duplicate_source"]
    same = [tracks[0], ManifestTrack(index=2, file="tracks/002.mp3", title="x", artist="A", duration_s=10.0,
                                     source_rel="Strasse.mp3", source_path="/lib/Strasse.mp3",
                                     track_id=make_track_id("Strasse.mp3"))]
    b2 = _Builder()
    _check_duplicates(b2, same)
    assert [i for i in b2.issues if i.code == "duplicate_source"]


def test_recorded_targets_are_validated_field_by_field():
    """VSPEC-4 / R2A-2: one invalid recorded target is reported and skipped; the valid
    ones (e.g. a 48 kHz sample rate) still apply."""
    from veg_mixer.config import EncodeConfig, apply_recorded_targets
    lc, enc, problems = apply_recorded_targets(
        {"target_lufs": "loud", "sample_rate": 48000, "true_peak_tolerance_db": True},
        LoudnessConfig(), EncodeConfig())
    assert lc.target_lufs == LoudnessConfig().target_lufs and enc.sample_rate == 48000
    assert len(problems) == 2 and any("target_lufs" in p for p in problems)
    _, enc2, problems2 = apply_recorded_targets({"sample_rate": "48k"}, LoudnessConfig(), EncodeConfig())
    assert enc2.sample_rate == 44100 and problems2 and "sample_rate" in problems2[0]
    _, _, problems3 = apply_recorded_targets(["not", "a", "dict"], LoudnessConfig(), EncodeConfig())
    assert problems3
