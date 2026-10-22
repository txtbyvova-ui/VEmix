"""VEmix 3.0 end-to-end through the CLI with the real FFmpeg (synthetic audio only):
analyze --segments -> plan (+ exports, dry run) -> compare -> render (playlist / mix) -> qc,
a stale plan, and the 2.0 commands' outputs staying as before."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from veg_mixer.cli import main

from conftest import encode_mp3, music_like, requires_ffmpeg, sha256, write_wav

pytestmark = [pytest.mark.ffmpeg, pytest.mark.slow, requires_ffmpeg]

# artist, title, bpm, curated energy, mood, am depth, drive
TRACKS = [
    ("Test Anchor", "Opening Night", 118, 0.62, "playful", 0.4, 2.0),
    ("Test Low", "Fog", 90, 0.22, "ominous", 0.2, 0.5),
    ("Test Mid", "Steps", 110, 0.48, "mysterious", 0.4, 1.5),
    ("Test Mid", "Stairs", 112, 0.55, "tense", 0.5, 1.8),
    ("Test High", "Rush", 124, 0.82, "energetic", 0.6, 3.0),
    ("Test High", "Peak Hour", 126, 0.90, "euphoric", 0.6, 3.5),
    ("Test Calm", "Last Light", 88, 0.25, "eerie", 0.2, 0.8),
    ("Test Groove", "Shuffle", 116, 0.66, "groovy", 0.5, 2.2),
]
SCENARIO = """
scenario: cli_test
target_duration_min: 4.5
duration_tolerance_min: 1
alternatives: 2
anchors:
  - track: "Test Anchor - Opening Night"
    position: 1
phases:
  - {name: intro, share: 0.2, energy: [0.6, 0.6], roles: [opener]}
  - {name: build, share: 0.3, energy: [0.45, 0.8]}
  - {name: climax, share: 0.3, energy: [0.85, 0.9], roles: [peak]}
  - {name: closer, share: 0.2, energy: [0.4, 0.25], contrast: true, moods: {prefer: [ominous, eerie]}}
transitions: {max_bpm_jump: 40, max_energy_jump: 0.45, contrast_energy_jump: 0.7}
diversity: {artist_min_gap: 1}
"""


@pytest.fixture(scope="module")
def env(tmp_path_factory, ffmpeg_tools):
    root = tmp_path_factory.mktemp("v3")
    lib = root / "lib"
    lib.mkdir()
    rows = ["rel_path,energy,mood,approved"]
    for i, (artist, title, bpm, energy, mood, am, drive) in enumerate(TRACKS):
        wav = write_wav(root / "wav" / f"{i}.wav", music_like(40.0, seed=i, bpm=bpm, am_depth=am, drive=drive,
                                                               tone_hz=180.0 + 20 * i))
        name = f"{artist} - {title}.mp3"
        encode_mp3(ffmpeg_tools, wav, lib / name, tags={"artist": artist, "title": title, "language": "en"})
        rows.append(f"{name},{energy},{mood},true")
    meta = root / "meta.csv"
    meta.write_text("\n".join(rows) + "\n", encoding="utf-8")
    scenario = root / "scenario.yaml"
    scenario.write_text(SCENARIO, encoding="utf-8")
    hashes = {p.name: sha256(p) for p in lib.iterdir()}
    return {"root": root, "lib": lib, "meta": meta, "scenario": scenario, "hashes": hashes}


def _plan_args(env, *extra):
    return ["plan", "--folder", str(env["lib"]), "--scenario", str(env["scenario"]), "--metadata", str(env["meta"]),
            "--jobs", "2", *extra]


def test_end_to_end(env, capsys):
    lib, root = env["lib"], env["root"]
    # 1) segment analysis is opt-in and cached in its own section
    assert main(["analyze", "--folder", str(lib), "--segments", "--jobs", "2"]) == 0
    cache = json.loads((lib / "veg_analysis_cache.json").read_text(encoding="utf-8"))
    entries = cache["entries"].values()
    assert all("segments" in e["sections"] for e in entries)
    assert cache["schema_version"] == 1

    # 2) dry run: full report printed, nothing written (no plan file, cache untouched)
    before = (lib / "veg_analysis_cache.json").stat().st_mtime_ns
    dry = root / "dry_plan.json"
    code = main(_plan_args(env, "--dry-run", "--save", str(dry)))
    out = capsys.readouterr().out
    assert code in (0, 4)
    assert "Test Anchor - Opening Night" in out and "Why each track is here" in out
    assert "[dry-run] would write" in out and not dry.exists()
    assert (lib / "veg_analysis_cache.json").stat().st_mtime_ns == before

    # 3) plan with exports
    plan = root / "plan.json"
    code = main(_plan_args(env, "--save", str(plan), "--m3u", str(root / "plan.m3u8"), "--csv",
                           str(root / "plan.csv"), "--report", str(root / "plan.txt")))
    out = capsys.readouterr().out
    assert code in (0, 4), out
    doc = json.loads(plan.read_text(encoding="utf-8"))
    assert doc["schema"] == "vegroove.plan" and len(doc["result"]["variants"]) == 2
    a = doc["result"]["variants"][0]
    assert a["entries"][0]["display"] == "Test Anchor - Opening Night"
    assert doc["result"]["inputs"]["with_segments"] == len(TRACKS)
    m3u = (root / "plan.m3u8").read_text(encoding="utf-8").splitlines()
    assert m3u[0] == "#EXTM3U" and m3u[3].endswith("Test Anchor - Opening Night.mp3")
    assert (root / "plan.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert "VEmix plan comparison" in (root / "plan.txt").read_text(encoding="utf-8")

    # 4) compare reads the plan file only
    assert main(["compare", "--plan", str(plan)]) == 0
    assert "variant" in capsys.readouterr().out.lower()

    # 5) render variant A as a playlist: planned order, plan section, plan QC checks
    out_dir = root / "out"
    code = main(["render", "--plan", str(plan), "--output", str(out_dir), "--jobs", "2"])
    capsys.readouterr()
    assert code in (0, 4)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["plan"]["variant"] == "A" and manifest["plan"]["scenario"] == "cli_test"
    assert [t["track_id"] for t in manifest["tracks"]] == a["order"]
    qc = json.loads((out_dir / "qc_report.json").read_text(encoding="utf-8"))
    assert {"plan_order", "plan_anchors"} <= set(qc["checks_run"])
    assert not [i for i in qc["issues"] if i["code"] in ("plan_order_mismatch", "plan_anchor_moved")]
    # qc re-run on the delivery agrees
    assert main(["qc", "--folder", str(out_dir)]) in (0, 4)
    capsys.readouterr()

    # 6) render variant B as a continuous mix
    mix = root / "mix" / "plan_mix.mp3"
    code = main(["render", "--plan", str(plan), "--variant", "B", "--mode", "mix", "--output", str(mix),
                 "--jobs", "2"])
    capsys.readouterr()
    assert code in (0, 4) and mix.is_file()
    mm = json.loads(mix.with_name("plan_mix.manifest.json").read_text(encoding="utf-8"))
    assert mm["plan"]["variant"] == "B" and mm["mix"]["published"] is True

    # 7) sources never modified
    assert {p.name: sha256(p) for p in lib.iterdir() if p.suffix == ".mp3"} == env["hashes"]


def test_two_point_zero_outputs_have_no_plan_section(env, capsys):
    out_dir = env["root"] / "legacy_out"
    code = main(["playlist", "--folder", str(env["lib"]), "--output", str(out_dir), "--max-tracks", "3",
                 "--jobs", "2"])
    capsys.readouterr()
    assert code in (0, 4)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "plan" not in manifest
    qc = json.loads((out_dir / "qc_report.json").read_text(encoding="utf-8"))
    assert "plan_order" not in qc["checks_run"]


def test_stale_plan_is_refused(env, capsys):
    root = env["root"]
    plan = root / "stale_plan.json"
    assert main(_plan_args(env, "--save", str(plan), "--alternatives", "1")) in (0, 4)
    doc = json.loads(plan.read_text(encoding="utf-8"))
    rel = doc["library"]["tracks"][doc["result"]["variants"][0]["order"][1]]["rel_path"]
    victim = env["lib"] / rel
    st = victim.stat()
    os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))      # "edited" after planning
    try:
        code = main(["render", "--plan", str(plan), "--output", str(root / "stale_out")])
        err = capsys.readouterr().err
        assert code == 2 and "stale" in err
        assert not (root / "stale_out").exists()
    finally:
        os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_missing_anchor_is_a_usage_error(env, capsys, tmp_path):
    sc = tmp_path / "bad.yaml"
    sc.write_text(SCENARIO.replace("Test Anchor - Opening Night", "Nobody - Nothing"), encoding="utf-8")
    code = main(["plan", "--folder", str(env["lib"]), "--scenario", str(sc)])
    err = capsys.readouterr().err
    assert code == 2 and "matches no eligible track" in err


def test_enrich_cache_mode_and_fake_interpretation_feed_the_plan(env, capsys):
    lib = env["lib"]
    # cache-only research: no network, nothing cached yet -> every track 'not_cached', no file written
    code = main(["enrich", "--folder", str(lib), "--mode", "cache", "--jobs", "2"])
    out = capsys.readouterr().out
    assert code == 0 and "not_cached" in out
    assert not (lib / "veg_research_cache.json").exists()
    # the fake provider echoes curator data; its interpretations are cached and used by plan
    code = main(["interpret", "--folder", str(lib), "--metadata", str(env["meta"]), "--provider", "fake",
                 "--jobs", "2"])
    capsys.readouterr()
    assert code == 0 and (lib / "veg_ai_cache.json").is_file()
    code = main(_plan_args(env, "--alternatives", "1", "--no-reasons"))
    out = capsys.readouterr().out
    assert code in (0, 4)
    assert "'fake' demo provider" in out
    assert "AI interpretation for 8" in out
    # --ai off ignores them
    code = main(_plan_args(env, "--alternatives", "1", "--no-reasons", "--ai", "off"))
    out = capsys.readouterr().out
    assert "AI interpretation for 0" in out


def test_outputs_never_overwrite_the_commands_inputs(env, capsys, tmp_path):
    lib, meta, scenario = env["lib"], env["meta"], env["scenario"]
    before = {p: sha256(p) for p in (meta, scenario)}
    code = main(_plan_args(env, "--csv", str(meta)))
    assert code == 2 and "refusing to write" in capsys.readouterr().err
    code = main(_plan_args(env, "--report", str(scenario)))
    assert code == 2 and "--scenario" in capsys.readouterr().err
    ai_cache = tmp_path / "ai.json"
    code = main(["interpret", "--folder", str(lib), "--provider", "fake", "--ai-cache", str(ai_cache),
                 "--report", str(ai_cache)])
    assert code == 2 and "--ai-cache" in capsys.readouterr().err
    code = main(["enrich", "--folder", str(lib), "--mode", "cache", "--research-cache", str(tmp_path / "r.json"),
                 "--report", str(tmp_path / "r.json")])
    assert code == 2 and "--research-cache" in capsys.readouterr().err
    assert {p: sha256(p) for p in (meta, scenario)} == before


def test_research_and_ai_caches_follow_a_moved_analysis_cache(env, capsys, tmp_path):
    caches = tmp_path / "caches"
    caches.mkdir()
    code = main(["interpret", "--folder", str(env["lib"]), "--metadata", str(env["meta"]), "--provider", "fake",
                 "--cache", str(caches / "analysis.json"), "--jobs", "2"])
    capsys.readouterr()
    assert code == 0
    assert (caches / "veg_ai_cache.json").is_file()


def test_missing_ai_sdk_is_an_environment_error(env, capsys, monkeypatch):
    import veg_mixer.ai as ai_pkg
    from veg_mixer.ai import AIUnavailableError

    def unavailable(*a, **k):
        raise AIUnavailableError("the 'anthropic' package is not installed (pip install -r requirements-ai.txt)")

    monkeypatch.setattr(ai_pkg, "get_provider", unavailable)
    code = main(["interpret", "--folder", str(env["lib"]), "--provider", "anthropic", "--jobs", "2"])
    assert code == 3 and "anthropic" in capsys.readouterr().err


def test_render_keeps_a_plan_seed_of_zero(env, capsys, tmp_path, monkeypatch):
    import veg_mixer.plan_pipeline as pp
    from veg_mixer.models import OverallStatus
    plan = tmp_path / "seed0.json"
    assert main(_plan_args(env, "--alternatives", "1", "--no-reasons", "--seed", "0", "--save", str(plan))) in (0, 4)
    seen = {}

    def fake_mix(ctx, sel, seq, out, seed, plan=None):
        seen["seed"] = seed
        return OverallStatus.PASS

    monkeypatch.setattr(pp, "_render_one_mix", fake_mix)
    assert main(["render", "--plan", str(plan), "--mode", "mix", "--output", str(tmp_path / "m.mp3")]) == 0
    capsys.readouterr()
    assert seen["seed"] == 0


def test_interpret_retry_invalid_option_exists():
    from veg_mixer.cli import build_parser
    args = build_parser().parse_args(["interpret", "--folder", "x", "--retry-invalid"])
    assert args.retry_invalid is True
