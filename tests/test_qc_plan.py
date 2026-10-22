"""Delivery QC of VEmix 3.0 plan renders (``manifest.plan``) and 2.0 sequencer backward
compatibility with 3.0 curator role tags.

* plan hard-constraint violations reach the QC report (never a silent PASS);
* anchors are checked from the right end of the PUBLISHED order (``position -1`` = last),
  phase anchors are not positional;
* in a mix, a transition into a planned contrast that the 2.0 engine marked uncertain for
  the energy difference alone is INFO, any tempo / beat cause still warns;
* ``role:*`` vibe tags do not enter the 2.0 vibe similarity (scalar and vectorized scorer).

Manifests are built directly; nothing here decodes audio, so no FFmpeg is needed."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from veg_mixer.config import SequencingConfig
from veg_mixer.ffmpeg_tools import FFmpegTools
from veg_mixer.loudness import LoudnessNormalizer
from veg_mixer.models import (
    ManifestTrack,
    MixTrackEntry,
    OverallStatus,
    PlaylistManifest,
    Severity,
    TrackFeatures,
    TransitionReport,
    to_jsonable,
)
from veg_mixer.quality_control import (
    _anchor_slot,
    _Builder,
    _check_plan,
    _energy_only_uncertainty,
    _planned_contrast_positions,
    overall_status,
    qc_mix,
)
from veg_mixer.sequencing import _FeatureMatrix, build_sequence, transition_score

CFG = SequencingConfig()
ENERGY_FLAG = "large energy difference 0.80 → 0.30 (|ΔE| 0.50 > 0.30) — overlap shortened to 3.0 s"
TEMPO_FLAG = "no beat/phrase alignment; tempo 172.3 vs 123 BPM (29% apart) — shortened to 3.0 s, listen"
UNKNOWN_BPM_FLAG = "no beat/phrase alignment; BPM unknown (incoming) — shortened to 3.0 s, listen"
ENERGY_MSG = "energy jump +0.50 (0.30 -> 0.80) exceeds the scenario limit 0.30"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def plan_section(names: list[str], *, anchors: dict[str, str] | None = None, contrast: tuple[str, ...] = (),
                 violations: list | None = None, with_ids: bool = True, positions: dict[str, int] | None = None,
                 contrast_field: bool = True) -> dict:
    """The ``plan`` section ``plan_pipeline._plan_section`` writes (planned order = ``names``)."""
    anchors, positions = anchors or {}, positions or {}
    order = []
    for k, n in enumerate(names, start=1):
        e = {"position": positions.get(n, k), "track_id": f"id-{n}" if with_ids else None, "rel_path": f"{n}.mp3",
             "display": f"Artist - {n}", "phase": "build", "target_energy": 0.5, "energy": 0.5, "deviation": 0.0,
             "anchored": anchors.get(n)}
        if contrast_field:
            e["planned_contrast"] = n in contrast
        order.append(e)
    d = {"schema": "vemix.plan", "file": "plan.json", "scenario": "halloween_special", "scenario_sha256": "0" * 64,
         "variant": "A", "preset": "balanced", "seed": 7, "objective": 0.1, "order": order}
    if violations is not None:
        d["violations"] = violations
    return d


def manifest(published: list[str], *, plan: dict | None, flagged: list[dict] = (), mode: str = "playlist",
             mix: dict | None = None) -> PlaylistManifest:
    tracks = [ManifestTrack(index=i, file=f"tracks/{i:03d}.mp3", title=n, artist="Artist", duration_s=150.0,
                            source_rel=f"{n}.mp3", source_path=f"/lib/{n}.mp3", track_id=f"id-{n}")
              for i, n in enumerate(published, start=1)]
    return PlaylistManifest(generator="test", generated_at="2026-10-22T00:00:00+00:00", mode=mode, profile=None,
                            daypart=None, loudness_targets={}, tracks=tracks, flagged_transitions=list(flagged),
                            plan=plan, mix=mix)


def run_check_plan(m: PlaylistManifest) -> _Builder:
    b = _Builder()
    _check_plan(b, m)
    return b


def of(b_or_report, code: str) -> list:
    return [i for i in b_or_report.issues if i.code == code]


# ---------------------------------------------------------------------------
# Q1: plan hard-constraint violations are reported
# ---------------------------------------------------------------------------

NAMES = ["t1", "t2", "t3", "t4", "t5"]


def test_non_transition_plan_violations_are_warnings_and_prevent_pass():
    violations = [
        {"kind": "anchor", "severity": 10.0, "position": 1, "message": "anchor 'Artist - t9' is not at position 1"},
        {"kind": "required", "severity": 10.0, "position": None, "message": "required track 'Artist - t8' is missing"},
        {"kind": "phase_anchor", "severity": 10.0, "position": 4,
         "message": "'Artist - t4' should play in phase 'climax' but is in 'build'"},
        {"kind": "duration", "severity": 1.5, "position": None,
         "message": "duration 50.0 min is 6.0 min under the target 60.0 ± 4.0 min"},
        {"kind": "artist_count", "severity": 1.0, "position": None, "message": "4 tracks by 'Artist' (maximum 3)"},
        {"kind": "same_song", "severity": 1.0, "position": None,
         "message": "several versions of one song: Artist - t2, Artist - t3"},
        {"kind": "track_count", "severity": 2.0, "position": None, "message": "5 tracks > max_tracks 3"},
    ]
    b = run_check_plan(manifest(NAMES, plan=plan_section(NAMES, violations=violations)))
    found = of(b, "plan_violation")
    assert len(found) == len(violations)
    assert all(i.severity is Severity.WARNING for i in found)
    for v, i in zip(violations, found):
        assert v["kind"] in i.message and v["message"] in i.message
    assert "at planned #1" in found[0].message and "at planned #" not in found[1].message
    assert "plan_constraints" in b.checks_run
    assert overall_status(b.issues, b.checks_skipped) is OverallStatus.WARNING


def test_plan_without_violations_still_passes_and_missing_key_is_empty():
    for plan in (plan_section(NAMES, violations=[]), plan_section(NAMES)):     # older plan: no 'violations'
        b = run_check_plan(manifest(NAMES, plan=plan))
        assert not of(b, "plan_violation") and not of(b, "plan_invalid")
        assert overall_status(b.issues, b.checks_skipped) is OverallStatus.PASS


def test_transition_violations_surfaced_as_flags_are_not_reported_twice():
    flagged = [{"from_index": 2, "to_index": 3, "kind": "sequencing",
                "reasons": [f"plan constraint not met: {ENERGY_MSG}"]}]
    violations = [
        {"kind": "energy_jump", "severity": 1.6, "position": 3, "message": ENERGY_MSG},          # surfaced
        {"kind": "tempo_jump", "severity": 1.2, "position": None,                               # dropped upstream
         "message": "tempo jump 90 -> 140 BPM (folded difference 50.0 > 12)"},
    ]
    b = run_check_plan(manifest(NAMES, plan=plan_section(NAMES, violations=violations), flagged=flagged))
    found = of(b, "plan_violation")
    assert [i.message for i in found] == [
        "the plan breaks a hard constraint of its scenario (tempo_jump): tempo jump 90 -> 140 BPM "
        "(folded difference 50.0 > 12)"]


def test_malformed_violations_are_not_silently_ignored():
    b = run_check_plan(manifest(NAMES, plan=plan_section(NAMES, violations="broken")))
    assert [i.severity for i in of(b, "plan_invalid")] == [Severity.WARNING]
    b = run_check_plan(manifest(NAMES, plan=plan_section(NAMES, violations=["anchor lost"])))
    assert [i.severity for i in of(b, "plan_violation")] == [Severity.WARNING]
    assert "anchor lost" in of(b, "plan_violation")[0].message


def test_two_point_zero_manifest_has_no_plan_checks():
    b = run_check_plan(manifest(NAMES, plan=None))
    assert b.issues == [] and b.checks_run == []


# ---------------------------------------------------------------------------
# Q2: anchors are checked from the right end of the PUBLISHED order
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anchored, planned, length, want", [
    ("position 1", 1, 5, 1),
    ("position 3", 3, 4, 3),
    ("position -1", 5, 5, 5),
    ("position -1", 5, 4, 4),        # last stays last after a removal
    ("position -2", 4, 4, 3),
    ("Position  -1", 5, 4, 4),
    ("phase climax", 3, 4, None),    # not positional
    ("position 0", 1, 5, None),
    ("legacy label", 2, 5, 2),       # unknown label: the planned position
])
def test_anchor_slot(anchored, planned, length, want):
    assert _anchor_slot(anchored, planned, length) == want


def test_last_position_anchor_not_reported_after_earlier_track_failed():
    plan = plan_section(NAMES, anchors={"t1": "position 1", "t3": "phase climax", "t5": "position -1"})
    b = run_check_plan(manifest(["t1", "t3", "t4", "t5"], plan=plan))          # t2 failed normalization
    assert [i.code for i in b.issues if i.severity is not Severity.INFO] == ["plan_track_missing"]
    assert not of(b, "plan_anchor_moved") and not of(b, "plan_order_mismatch")


def test_anchor_that_really_moved_is_reported():
    plan = plan_section(NAMES, anchors={"t3": "position 3"})
    b = run_check_plan(manifest(["t1", "t3", "t4", "t5"], plan=plan))          # t2 lost: t3 now plays at #2
    moved = of(b, "plan_anchor_moved")
    assert len(moved) == 1 and moved[0].severity is Severity.WARNING
    assert "plays at #2 instead of #3" in moved[0].message and moved[0].track_position == 2

    plan = plan_section(NAMES, anchors={"t4": "position -2"})
    b = run_check_plan(manifest(["t1", "t2", "t3", "t4"], plan=plan))          # t5 lost: t4 is now LAST
    moved = of(b, "plan_anchor_moved")
    assert len(moved) == 1 and "plays at #4 instead of #3" in moved[0].message


def test_anchor_the_plan_itself_missed_names_the_plan_position():
    names = ["t2", "t1", "t3"]                                                  # anchor 'position 1' planned at #2
    plan = plan_section(names, anchors={"t1": "position 1"},
                        violations=[{"kind": "anchor", "severity": 10.0, "position": 1,
                                     "message": "anchor 'Artist - t1' is not at position 1"}])
    b = run_check_plan(manifest(names, plan=plan))
    moved = of(b, "plan_anchor_moved")
    assert len(moved) == 1 and "the plan itself placed it at #2" in moved[0].message
    assert len(of(b, "plan_violation")) == 1


# ---------------------------------------------------------------------------
# Q3: planned contrast in a mix render
# ---------------------------------------------------------------------------

def test_energy_only_uncertainty():
    assert _energy_only_uncertainty([ENERGY_FLAG])
    assert _energy_only_uncertainty([ENERGY_FLAG, "FX echo (random mode, seed-controlled)"])
    assert not _energy_only_uncertainty([ENERGY_FLAG, TEMPO_FLAG])
    assert not _energy_only_uncertainty([ENERGY_FLAG, UNKNOWN_BPM_FLAG])
    assert not _energy_only_uncertainty([TEMPO_FLAG])
    assert not _energy_only_uncertainty([])


def mix_entries(names: list[str]) -> list[MixTrackEntry]:
    out, t = [], 0.0
    for pos, n in enumerate(names, start=1):
        xf = 0.0 if pos == 1 else 3.0
        start = 0.0 if pos == 1 else t - xf
        out.append(MixTrackEntry(position=pos, name=f"Artist - {n}", source_rel=f"{n}.mp3", start_s=start,
                                 end_s=start + 150.0, duration_s=150.0, crossfade_in_s=xf,
                                 fx_in="—" if pos == 1 else "clean"))
        t = start + 150.0
    return out


def test_planned_contrast_positions_match_by_id_or_rel_path_and_need_the_planned_pair():
    names = ["t1", "t2", "t3", "t4"]
    entries = mix_entries(names)
    m = manifest(names, plan=plan_section(names, contrast=("t3",)), mode="mix")
    assert _planned_contrast_positions(m, entries) == {3}
    m = manifest(names, plan=plan_section(names, contrast=("t3",), with_ids=False), mode="mix")   # rel_path only
    assert _planned_contrast_positions(m, entries) == {3}
    m.tracks = []                                                               # mix-entry source_rel is enough
    assert _planned_contrast_positions(m, entries) == {3}
    # the planned predecessor t2 failed normalization: t1 -> t3 is not the planned contrast
    m = manifest(["t1", "t3", "t4"], plan=plan_section(names, contrast=("t3",)), mode="mix")
    assert _planned_contrast_positions(m, mix_entries(["t1", "t3", "t4"])) == set()
    # no plan / older plan without the field
    assert _planned_contrast_positions(manifest(names, plan=None, mode="mix"), entries) == set()
    m = manifest(names, plan=plan_section(names, contrast_field=False), mode="mix")
    assert _planned_contrast_positions(m, entries) == set()


def _normalizer() -> LoudnessNormalizer:
    # the mixes below are unpublished: qc_mix never measures audio, the tools are never run
    return LoudnessNormalizer(FFmpegTools(ffmpeg="ffmpeg-not-used", ffprobe=None, version="test",
                                          has_soxr=False, has_libmp3lame=False))


def _write_mix_manifest(tmp_path: Path, names: list[str], flags: dict[int, list[str]], *,
                        plan: dict | None, flagged: list[dict] = ()) -> Path:
    entries = mix_entries(names)
    transitions = [TransitionReport(position=e.position, crossfade_s=3.0, fx="clean", curve="equal_power",
                                    overlap_start_s=e.start_s, overlap_peak_dbfs=-3.0, overlap_true_peak_dbtp=-2.0,
                                    uncertain=e.position in flags, flags=flags.get(e.position, []))
                   for e in entries[1:]]
    mix = {"file": "mix.mp3", "published": False, "expected_duration_s": entries[-1].end_s, "sample_rate": 44100,
           "tracks": to_jsonable(entries), "transitions": to_jsonable(transitions), "loudness": {},
           "tracklist": None, "seed": 0, "clipped_samples": 0, "fade_in_s": 1.0, "fade_out_s": 2.0}
    m = manifest(names, plan=plan, flagged=list(flagged), mode="mix", mix=mix)
    path = tmp_path / "mix.manifest.json"
    path.write_text(json.dumps(m.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_mix_planned_contrast_energy_only_is_info_tempo_still_warns(tmp_path):
    names = ["t1", "t2", "t3", "t4", "t5"]
    flags = {2: [TEMPO_FLAG],                       # ordinary tempo-uncertain transition
             3: [ENERGY_FLAG],                      # planned contrast, energy only
             5: [ENERGY_FLAG, UNKNOWN_BPM_FLAG]}    # planned contrast, but the tempo is unknown
    # what the export wrote for uncertain transitions (pipeline: kind "transition")
    flagged = [{"from_index": p - 1, "to_index": p, "kind": "transition", "reasons": f} for p, f in flags.items()]
    plan = plan_section(names, contrast=("t3", "t5"),
                        violations=[{"kind": "duration", "severity": 1.2, "position": None,
                                     "message": "duration 12.5 min is 47.5 min under the target 60.0 ± 4.0 min"}])
    path = _write_mix_manifest(tmp_path, names, flags, plan=plan, flagged=flagged)
    report = qc_mix(path, _normalizer())

    assert {i.track_position for i in of(report, "uncertain_transition")} == {2, 5}
    assert all(i.severity is Severity.WARNING for i in of(report, "uncertain_transition"))
    warn5 = next(i for i in of(report, "uncertain_transition") if i.track_position == 5)
    assert "planned contrast of the scenario" in warn5.message
    info = of(report, "planned_contrast_transition")
    assert len(info) == 1 and info[0].severity is Severity.INFO and info[0].track_position == 3
    assert "planned contrast of the scenario" in info[0].message and ENERGY_FLAG in info[0].message
    # the renderer entry for #3 is listed once (the export's duplicate is not re-added)
    to3 = [f for f in report.flagged_transitions if f.get("to_index") == 3]
    assert len(to3) == 1 and to3[0]["source"] == "renderer"
    assert "planned contrast of the scenario (energy change intended)" in to3[0]["reasons"]
    # Q1 wiring in a real report
    assert [i.severity for i in of(report, "plan_violation")] == [Severity.WARNING]


def test_mix_without_plan_keeps_energy_only_uncertainty_a_warning(tmp_path):
    names = ["t1", "t2", "t3"]
    path = _write_mix_manifest(tmp_path, names, {3: [ENERGY_FLAG]}, plan=None)
    report = qc_mix(path, _normalizer())
    assert [i.severity for i in of(report, "uncertain_transition")] == [Severity.WARNING]
    assert not of(report, "planned_contrast_transition")
    # a plan whose transition into #3 is NOT a planned contrast: still a warning
    path = _write_mix_manifest(tmp_path, names, {3: [ENERGY_FLAG]}, plan=plan_section(names, contrast=("t2",)))
    report = qc_mix(path, _normalizer())
    assert [i.severity for i in of(report, "uncertain_transition")] == [Severity.WARNING]


# ---------------------------------------------------------------------------
# Q4: 3.0 role tags do not change the 2.0 vibe similarity / order
# ---------------------------------------------------------------------------

def mk(name: str, *, energy: float | None = None, bpm: float | None = None, mood: str | None = None,
       tags: list[str] = (), genre: str | None = None, artist: str = "A") -> TrackFeatures:
    sources = {k: "manual" for k, v in dict(energy=energy, bpm=bpm, mood=mood, genre=genre).items() if v is not None}
    if tags:
        sources["vibe_tags"] = "manual"
    return TrackFeatures(path=Path(f"{name}.mp3"), duration_s=200.0, bpm=bpm, energy=energy, mood=mood, genre=genre,
                         vibe_tags=list(tags), artist=artist, title=name, feature_sources=sources,
                         rel_path=f"{name}.mp3")


ROLE_TAGS = ["role:opener", "ROLE:Closer", " Role: peak_time ", "role:bridge"]


def _with_roles(t: TrackFeatures, k: int) -> TrackFeatures:
    """The same track with one or two 3.0 role tags added to its curator vibe tags."""
    extra = ROLE_TAGS[k % len(ROLE_TAGS): k % len(ROLE_TAGS) + 1 + k % 2]
    return dataclasses.replace(t, vibe_tags=list(t.vibe_tags) + extra)


def _pool(n: int, seed: int) -> list[TrackFeatures]:
    rng = np.random.default_rng(seed)
    moods = ["calm", "happy", "groovy", "dark", None]
    tags = ["summer", "night", "groovy", "soft", "dance"]
    out = []
    for i in range(n):
        picked = [tags[int(x)] for x in rng.choice(len(tags), size=int(rng.integers(0, 3)), replace=False)]
        out.append(mk(f"p{i:03d}", energy=float(rng.uniform(0.1, 0.9)), bpm=float(rng.uniform(80, 140)),
                      mood=moods[int(rng.integers(0, len(moods)))], tags=picked, genre="pop",
                      artist=f"Artist {i % 7}"))
    return out


def test_scalar_vibe_similarity_ignores_role_tags():
    a, b = mk("a", mood="calm", tags=["chill", "night"]), mk("b", mood="calm", tags=["chill"])
    base = transition_score(a, b, CFG)
    a_roles = mk("a", mood="calm", tags=["chill", "night", "role:opener", "ROLE:Closer"])
    b_roles = mk("b", mood="calm", tags=["chill", " Role: opener"])            # shared role tag: no extra overlap
    assert transition_score(a_roles, b_roles, CFG).components == base.components
    assert transition_score(a_roles, b_roles, CFG).total == base.total
    # a track whose only tags are roles has no vibe tags at all (mood decides, as before 3.0)
    only_roles = mk("c", mood="calm", tags=["role:closer"])
    assert (transition_score(only_roles, b, CFG).components["vibe_similarity"]
            == transition_score(mk("c", mood="calm"), b, CFG).components["vibe_similarity"])


def test_vectorized_rows_ignore_role_tags_and_match_scalar():
    pool = _pool(30, seed=5)
    roles = [_with_roles(t, k) if k % 3 else t for k, t in enumerate(pool)]
    assert any(any(str(x).strip().lower().startswith("role:") for x in t.vibe_tags) for t in roles)
    fm, fm_roles = _FeatureMatrix(pool, CFG), _FeatureMatrix(roles, CFG)
    for i in range(len(pool)):
        row, jump, sim = fm.row(i)
        row_r, jump_r, sim_r = fm_roles.row(i)
        np.testing.assert_array_equal(row, row_r)
        np.testing.assert_array_equal(sim, sim_r)
        for j in range(len(pool)):
            ts = transition_score(roles[i], roles[j], CFG)
            assert row_r[j] == pytest.approx(ts.total, abs=1e-12), (i, j)
            assert ts.total == transition_score(pool[i], pool[j], CFG).total


def test_role_tags_do_not_change_the_two_point_zero_order():
    pool = _pool(40, seed=11)
    roles = [_with_roles(t, k) for k, t in enumerate(pool)]
    plain = build_sequence(pool, CFG, seed=3)
    tagged = build_sequence(roles, CFG, seed=3)
    assert [s.track.rel_path for s in tagged.tracks] == [s.track.rel_path for s in plain.tracks]


def test_identical_transition_messages_each_need_their_own_flag():
    """Two violations with the same text (same artist, same lag): one flag accounts for one."""
    msg = "same artist 'A' with only 0 track(s) in between (minimum 1)"
    flagged = [{"from_index": 1, "to_index": 2, "kind": "sequencing", "reasons": [f"plan constraint not met: {msg}"]}]
    violations = [{"kind": "artist_gap", "severity": 1.0, "position": 2, "message": msg},
                  {"kind": "artist_gap", "severity": 1.0, "position": 5, "message": msg}]
    b = run_check_plan(manifest(NAMES, plan=plan_section(NAMES, violations=violations), flagged=flagged))
    assert len(of(b, "plan_violation")) == 1


def test_role_tags_do_not_change_the_daypart_fit_of_two_point_zero():
    """The daypart preference fit reads curator labels too: a role tag on a track without
    other labels must not turn 'preferences unknown' (neutral) into 'no preferred label'."""
    from veg_mixer.models import DaypartProfile
    from veg_mixer.sequencing import daypart_fit
    daypart = DaypartProfile(name="evening", energy_range=(0.2, 0.8), preferred_tags=["chill"], max_tracks=8)
    assert daypart_fit(mk("x", energy=0.5, bpm=120.0, tags=["role:opener"]), daypart).fit ==         daypart_fit(mk("x", energy=0.5, bpm=120.0), daypart).fit
    rng = np.random.default_rng(1)
    pool = [mk(f"p{i:02d}", tags=(["chill"] if i % 4 == 0 else []), genre=("pop" if i % 2 == 0 else None),
               energy=float(rng.uniform(0.2, 0.8)), bpm=float(rng.uniform(95, 130)), artist=f"Art {i}")
            for i in range(14)]
    roles = [t if t.genre else dataclasses.replace(t, vibe_tags=[*t.vibe_tags, "role:opener"]) for t in pool]
    plain = build_sequence(pool, CFG, daypart=daypart, seed=3)
    tagged = build_sequence(roles, CFG, daypart=daypart, seed=3)
    assert [s.track.rel_path for s in tagged.tracks] == [s.track.rel_path for s in plain.tracks]
