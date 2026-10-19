"""CLI + end-to-end pipeline tests (real FFmpeg, synthetic library)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import encode_mp3, music_like, requires_ffmpeg, sha256, write_wav
from veg_mixer import cli
from veg_mixer.pipeline import EXIT_ENV, EXIT_FAIL, EXIT_OK, EXIT_USAGE, EXIT_WARNING, resolve_cache_paths


# ---------------------------------------------------------------------------
# argument handling (no FFmpeg needed)
# ---------------------------------------------------------------------------

def test_legacy_invocation_maps_to_mix():
    assert cli._legacy_argv(["--folder", "x", "--parts", "2"]) == ["mix", "--folder", "x", "--parts", "2"]
    assert cli._legacy_argv(["playlist", "--folder", "x"]) == ["playlist", "--folder", "x"]
    assert cli._legacy_argv(["--help"]) == ["--help"]
    args = cli.build_parser().parse_args(cli._legacy_argv(["--folder", "m", "--bitrate", "256", "--cache", "c.json"]))
    assert args.command == "mix" and args.parts == 1 and args.cache == "c.json"
    cfg, _ = cli.config_from_args(args)
    assert cfg.encode.bitrate == "256k"
    assert cfg.transitions.fx_mode == "clean"          # random FX is opt-in now


def test_config_overrides_and_validation():
    args = cli.build_parser().parse_args(["normalize", "--folder", "m", "--output", "o",
                                          "--target-lufs", "-14", "--true-peak", "-1"])
    cfg, overridden = cli.config_from_args(args)
    assert cfg.loudness.target_lufs == -14 and cfg.loudness.target_true_peak_dbtp == -1 and overridden
    assert cli.main(["normalize", "--folder", "m", "--output", "o", "--target-lufs", "-2"]) == EXIT_USAGE


def test_missing_ffmpeg_is_a_clear_environment_error(tmp_path, monkeypatch, capsys):
    from veg_mixer import ffmpeg_tools
    ffmpeg_tools.find_ffmpeg.cache_clear()
    monkeypatch.setenv("VEG_FFMPEG", str(tmp_path / "no-such-ffmpeg.exe"))
    try:
        code = cli.main(["analyze", "--folder", str(tmp_path)])
    finally:
        ffmpeg_tools.find_ffmpeg.cache_clear()
    assert code == EXIT_ENV
    err = capsys.readouterr().err
    assert "ffmpeg" in err.lower() and "winget" in err


def test_legacy_cache_passed_via_cache_flag_is_never_rewritten(tmp_path):
    legacy = tmp_path / "bpm_cache.json"
    legacy.write_text(json.dumps({"C:/x.mp3": 120.0}), encoding="utf-8")
    new, old = resolve_cache_paths(tmp_path, str(legacy))
    assert old == legacy.resolve() and new != legacy.resolve()
    new2, old2 = resolve_cache_paths(tmp_path, None)
    assert new2.name == "veg_analysis_cache.json" and old2 == legacy


# ---------------------------------------------------------------------------
# end-to-end with real FFmpeg
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def library(tmp_path_factory, ffmpeg_tools):
    root = tmp_path_factory.mktemp("lib")
    music = root / "music"
    music.mkdir()
    specs = [  # name, bpm, peak, seed, tags
        ("Demo A - Soft Start", 92, -12.0, 1, {"artist": "Demo A", "title": "Soft Start", "language": "en", "genre": "pop"}),
        ("Demo B - Middle", 110, -3.0, 2, {"artist": "Demo B", "title": "Middle", "language": "en", "genre": "pop"}),
        ("Demo C - Groove", 124, -0.5, 3, {"artist": "Demo C", "title": "Groove", "language": "fr", "genre": "pop"}),
        ("Демо Д - Песня", 100, -2.0, 4, {"artist": "Демо Д", "title": "Песня"}),
        ("Demo E - Russian Tag", 105, -2.0, 5, {"artist": "Demo E", "title": "Russian Tag", "language": "ru"}),
    ]
    for name, bpm, peak, seed, tags in specs:
        wav = write_wav(root / "wav" / f"{seed}.wav", music_like(24, peak_dbfs=peak, seed=seed, bpm=bpm))
        encode_mp3(ffmpeg_tools, wav, music / f"{name}.mp3", tags=tags)
    (music / "Demo Z - Broken.mp3").write_bytes(b"ID3" + bytes(range(256)) * 20)
    profile = root / "client.yaml"
    profile.write_text(
        "client: test_client\n"
        "language:\n  allowed: [en, fr]\n  excluded: [ru]\n  unknown_policy: review\n  cyrillic_script_policy: exclude\n",
        encoding="utf-8")
    hashes = {p.name: sha256(p) for p in music.iterdir()}
    return {"root": root, "music": music, "profile": profile, "hashes": hashes}


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_playlist_end_to_end(library):
    out = library["root"] / "out_playlist"
    code = cli.main(["playlist", "--folder", str(library["music"]), "--profile", str(library["profile"]),
                     "--output", str(out), "--jobs", "2"])
    assert code in (EXIT_OK, EXIT_WARNING, EXIT_FAIL)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    files = [t["file"] for t in manifest["tracks"]]
    assert len(files) == 3                                  # ru tag + cyrillic + broken left out
    assert all((out / f).is_file() for f in files)
    reasons = " ".join(" ".join(e["reasons"]) for e in manifest["excluded"])
    assert "ru" in reasons.lower() and "cyrillic" in reasons.lower()
    assert any("Broken" in e["source_rel"] for e in manifest["excluded"])
    m3u = (out / "playlist.m3u8").read_text(encoding="utf-8").splitlines()
    assert m3u[0] == "#EXTM3U" and [ln for ln in m3u if not ln.startswith("#") and ln] == files
    assert (out / "qc_report.json").is_file() and (out / "qc_report.txt").is_file()
    text = (out / "qc_report.txt").read_text(encoding="utf-8")
    assert text.startswith("VEgroove Audio QC") and "Overall status:" in text
    # QC re-run from the CLI agrees
    assert cli.main(["qc", "--folder", str(out)]) == code or code == EXIT_FAIL
    assert {p.name: sha256(p) for p in library["music"].iterdir() if p.suffix == ".mp3"} == \
        {k: v for k, v in library["hashes"].items() if k.endswith(".mp3")}


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_mix_end_to_end_and_legacy_syntax(library):
    out = library["root"] / "out_mix" / "mix.mp3"
    code = cli.main(["mix", "--folder", str(library["music"]), "--profile", str(library["profile"]),
                     "--output", str(out), "--crossfade", "3"])
    assert code in (EXIT_OK, EXIT_WARNING)
    assert out.is_file()
    manifest = json.loads(out.with_name("mix.manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "mix" and manifest["mix"]["file"] == "mix.mp3"
    assert manifest["mix"]["loudness"]["normalization_status"] in ("passed", "review")
    assert out.with_name("mix_tracklist.txt").is_file()
    assert out.with_name("mix_qc_report.txt").is_file()
    # legacy invocation without a profile still produces a mix
    legacy_out = library["root"] / "out_legacy" / "legacy.mp3"
    code2 = cli.main(["--folder", str(library["music"]), "--output", str(legacy_out), "--bitrate", "192k"])
    assert code2 in (EXIT_OK, EXIT_WARNING, EXIT_FAIL) and legacy_out.is_file()


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_dry_run_writes_nothing(library):
    out = library["root"] / "out_dry"
    before = sorted(p.name for p in library["music"].iterdir())
    code = cli.main(["playlist", "--folder", str(library["music"]), "--output", str(out), "--dry-run"])
    assert code == EXIT_OK and not out.exists()
    assert sorted(p.name for p in library["music"].iterdir()) == before
    code = cli.main(["mix", "--folder", str(library["music"]), "--output", str(out / "m.mp3"), "--dry-run"])
    assert code == EXIT_OK and not out.exists()


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_mix_output_may_not_overwrite_a_library_source(library):
    """INT-1 / SAFE-1: `mix --output` pointing at a source file is refused before any work."""
    victim = library["music"] / "Demo B - Middle.mp3"
    before = sha256(victim)
    code = cli.main(["mix", "--folder", str(library["music"]), "--output", str(victim)])
    assert code == EXIT_USAGE and sha256(victim) == before


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_normalize_refuses_colliding_or_overwriting_outputs(tmp_path, ffmpeg_tools):
    """SAFE-2 / SAFE-3 / INT-13: song.wav + song.flac -> one song.mp3 is refused; so is
    an output folder that is the library itself."""
    lib = tmp_path / "lib"
    write_wav(lib / "song.wav", music_like(3, seed=1))
    write_wav(lib / "song.flac", music_like(3, seed=2), subtype="PCM_16")
    assert cli.main(["normalize", "--folder", str(lib), "--output", str(tmp_path / "out")]) == EXIT_USAGE
    assert not (tmp_path / "out").exists()
    assert cli.main(["normalize", "--folder", str(lib), "--output", str(lib)]) == EXIT_USAGE


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_dry_run_never_renames_a_corrupt_cache_or_creates_work_dirs(library, tmp_path):
    """SAFE-7: --dry-run writes nothing — not even a .corrupt-* rename or --work-dir."""
    music = library["root"] / "music_copy_dry"
    music.mkdir()
    for p in library["music"].glob("Demo A*.mp3"):
        (music / p.name).write_bytes(p.read_bytes())
    cache = music / "veg_analysis_cache.json"
    cache.write_text("{ truncated", encoding="utf-8")
    work = tmp_path / "work_should_not_exist"
    code = cli.main(["playlist", "--folder", str(music), "--output", str(tmp_path / "o"), "--dry-run",
                     "--work-dir", str(work)])
    assert code == EXIT_OK
    assert cache.read_text(encoding="utf-8") == "{ truncated"
    assert not list(music.glob("*.corrupt-*")) and not work.exists() and not (tmp_path / "o").exists()


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_qc_uses_manifest_sample_rate_and_targets(library):
    """INT-7 / SPEC-6 / INT-8: a 48 kHz export re-checked by `qc` uses the recorded targets;
    one explicit option overrides only that field."""
    out = library["root"] / "out_48k"
    code = cli.main(["playlist", "--folder", str(library["music"]), "--profile", str(library["profile"]),
                     "--output", str(out), "--sample-rate", "48000", "--max-tracks", "2"])
    assert code in (EXIT_OK, EXIT_WARNING)
    assert cli.main(["qc", "--folder", str(out), "--warnings-ok"]) == EXIT_OK
    assert cli.main(["qc", "--folder", str(out), "--lufs-tolerance", "1.5", "--warnings-ok"]) == EXIT_OK


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_failed_selected_track_never_reports_pass(tmp_path, ffmpeg_tools):
    """SPEC-1: a selected track that fails normalization makes the run FAIL (exit 1)."""
    lib = tmp_path / "lib"
    lib.mkdir()
    for i, peak in enumerate((-1.0, -3.0)):
        wav = write_wav(tmp_path / "w" / f"{i}.wav", music_like(12, peak_dbfs=peak, seed=i))
        encode_mp3(ffmpeg_tools, wav, lib / f"Demo {i} - Song {i}.mp3")
    # ~-66 LUFS: measurable (above the -70 LUFS gate) but below min_input_lufs -> normalization FAILS
    whisper = write_wav(tmp_path / "w" / "whisper.wav", music_like(12, peak_dbfs=-57, seed=9))
    encode_mp3(ffmpeg_tools, whisper, lib / "Demo W - Whisper.mp3")
    assert cli.main(["playlist", "--folder", str(lib), "--output", str(tmp_path / "pl")]) == EXIT_FAIL
    report = json.loads((tmp_path / "pl" / "qc_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL" and report["summary"]["tracks_failed"] >= 1
    assert cli.main(["mix", "--folder", str(lib), "--output", str(tmp_path / "mx" / "m.mp3")]) == EXIT_FAIL


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_empty_daypart_fails_without_aborting_the_others(library, tmp_path):
    """INT-11 / INT-12: a daypart nothing fits is reported as FAIL; the others are exported."""
    profile = tmp_path / "p.yaml"
    profile.write_text(
        "client: t\nlanguage:\n  excluded: [ru]\n"
        "dayparts:\n  night:\n    energy_range: [0.99, 1.0]\n    energy_curve: [0.99, 1.0]\n",
        encoding="utf-8")
    out = tmp_path / "dp"
    code = cli.main(["playlist", "--folder", str(library["music"]), "--profile", str(profile),
                     "--output", str(out), "--daypart", "all"])
    assert code == EXIT_FAIL                                  # 'night' produced nothing
    produced = [p.name for p in out.iterdir() if (p / "manifest.json").is_file()]
    assert produced                                            # other dayparts still exported


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_normalize_command_reports_failures(library):
    out = library["root"] / "out_norm"
    code = cli.main(["normalize", "--folder", str(library["music"]), "--output", str(out)])
    assert code == EXIT_FAIL                               # the corrupt file must fail loudly
    report = json.loads((out / "normalize_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert any("Broken" in (t["file"] or "") or "Broken" in (t["source_rel"] or "") for t in report["tracks"]
               if t["status"] == "failed")
    assert len(list(out.glob("*.mp3"))) == 6 - 1


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_playlist_with_library_inside_output_folder_still_finds_tracks(tmp_path, ffmpeg_tools):
    """VSAFE-1 / VINT-2: --output that CONTAINS the library is a safe layout (outputs go to
    <output>/tracks); the library must still be scanned. --output == library is refused."""
    root = tmp_path / "radio"
    music = root / "music"
    music.mkdir(parents=True)
    for i in range(2):
        wav = write_wav(tmp_path / "w" / f"{i}.wav", music_like(10, peak_dbfs=-3, seed=i))
        encode_mp3(ffmpeg_tools, wav, music / f"Demo {i} - Song {i}.mp3",
                   tags={"artist": f"Demo {i}", "title": f"Song {i}", "language": "en"})
    code = cli.main(["playlist", "--folder", str(music), "--output", str(root)])
    assert code in (EXIT_OK, EXIT_WARNING)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tracks"]) == 2
    assert cli.main(["playlist", "--folder", str(music), "--output", str(music)]) == EXIT_USAGE
