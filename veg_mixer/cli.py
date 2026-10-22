"""Command-line interface.

    python automix.py analyze   --folder ./music [--export-metadata curator.csv]
    python automix.py normalize --folder ./music --output ./normalized --target-lufs -16 --true-peak -1.5
    python automix.py playlist  --folder ./music --profile ./profiles/client.yaml --output ./output [--daypart all]
    python automix.py mix       --folder ./music --profile ./profiles/client.yaml --output ./output/mix.mp3
    python automix.py qc        --folder ./output

VEmix 3.0 (scenario planning; see docs/ARCHITECTURE.md):

    python automix.py analyze   --folder ./music --segments
    python automix.py enrich    --folder ./music [--mode online|cache|off]
    python automix.py interpret --folder ./music --provider anthropic|ollama|... [--mode online|cache|off]
    python automix.py plan      --folder ./music --scenario ./scenarios/halloween_special.yaml --save plan.json
    python automix.py suggest   --folder ./music --brief "..." --output draft.yaml --provider anthropic
    python automix.py compare   --plan plan.json
    python automix.py render    --plan plan.json --variant A --mode playlist --output ./output

Backward compatible: ``python automix.py --folder ./music [--output X.mp3] [--parts N]
[--bitrate 320k] [--cache bpm_cache.json]`` (no sub-command) runs ``mix`` like automix v1,
now with the improved transition engine, loudness normalization and QC.

Exit codes: 0 PASS, 1 FAIL, 2 usage/input error, 3 environment error (FFmpeg missing or
unusable, or an optional package such as the anthropic SDK not installed), 4 WARNING (use
--warnings-ok to map WARNING to 0), 5 runtime error (output folder locked, export rolled
back, disk or network errors; enrich/interpret: every request failed; suggest: the AI
provider gave no answer), 130 interrupted (Ctrl+C).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import traceback
import warnings
from pathlib import Path

from . import __version__
from .config import AppConfig, ConfigError
from .ffmpeg_tools import FFmpegNotFoundError, find_ffmpeg
from .net import NetError
from .playlist import PlaylistExportError
from .pipeline import (
    ALL_DAYPARTS,
    EXIT_ENV,
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_USAGE,
    EXIT_WARNING,
    OptionalDependencyError,
    RunContext,
    run_analyze,
    run_mix,
    run_normalize,
    run_playlist,
    run_qc,
)

COMMANDS = ("analyze", "normalize", "playlist", "mix", "qc", "enrich", "interpret", "plan", "compare", "render",
            "suggest")
# 3.0 sub-commands (their own argument sets; options of the 2.0 commands are unchanged)
V3_COMMANDS = ("enrich", "interpret", "plan", "compare", "render", "suggest")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def _common(p: argparse.ArgumentParser, *, folder_help: str) -> None:
    p.add_argument("--folder", required=True, help=folder_help)
    _runtime(p)


def _runtime(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("runtime")
    g.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)),
                   help="parallel FFmpeg jobs (default: half the CPUs, max 4)")
    g.add_argument("--dry-run", action="store_true", help="plan and print, write nothing")
    g.add_argument("--work-dir", help="directory for temporary lossless intermediates")
    g.add_argument("--ffmpeg", help="path to ffmpeg (default: VEG_FFMPEG or PATH)")
    g.add_argument("--ffprobe", help="path to ffprobe (default: VEG_FFPROBE, next to ffmpeg, or PATH)")
    g.add_argument("--warnings-ok", action="store_true",
                   help="exit 0 instead of 4 for any WARNING result (QC status WARNING; enrich: ambiguous matches or "
                        "failed lookups; interpret: failed or invalid AI answers; plan: hard-constraint violations "
                        "in the shown variant)")
    g.add_argument("-v", "--verbose", action="store_true")


def _library(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("library")
    g.add_argument("--cache", help="analysis cache JSON (default: <folder>/veg_analysis_cache.json; "
                                   "a legacy bpm_cache.json is migrated read-only)")
    g.add_argument("--metadata", help="curator metadata CSV/JSON (manual values override automatic ones)")
    g.add_argument("--recursive", action="store_true", help="scan sub-folders too")


def _loudness(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("loudness (EBU R128)")
    g.add_argument("--target-lufs", type=float, help="integrated loudness target (default -16)")
    g.add_argument("--true-peak", type=float, help="true-peak ceiling in dBTP (default -1.5)")
    g.add_argument("--lra", type=float, help="loudness-range processing target in LU (default 11)")
    g.add_argument("--lufs-tolerance", type=float, help="accepted deviation in LU (default 1.0)")
    g.add_argument("--true-peak-tolerance", type=float, help="accepted true-peak overshoot in dB (default 0.2)")
    g.add_argument("--max-dynamic-gain", type=float,
                   help="opt in to loudnorm dynamic mode (gain riding/AGC + true-peak limiting) for shortfalls up to this many dB; always flagged for review (default 0 = never)")
    g.add_argument("--bitrate", help="MP3 bitrate (default 320k)")
    g.add_argument("--sample-rate", type=int, choices=(44100, 48000), help="output sample rate (default 44100)")


def _selection(p: argparse.ArgumentParser, *, daypart_all: bool) -> None:
    g = p.add_argument_group("curation and sequencing")
    g.add_argument("--profile", help="client profile YAML/JSON (default: permissive built-in profile)")
    choices = list(ALL_DAYPARTS) + (["all"] if daypart_all else [])
    g.add_argument("--daypart", help=f"daypart profile: {', '.join(choices)} or a custom name from the profile"
                                     + (" ('all' also includes custom dayparts)" if daypart_all else ""))
    g.add_argument("--seed", type=int, help="seed for controlled variation (default: derived from names)")
    g.add_argument("--duration-min", type=float, help="target duration per playlist/mix in minutes")
    g.add_argument("--max-tracks", type=int, help="maximum tracks per playlist/mix")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automix.py", description="VEgroove Automix / VEmix 3.0 — loudness-normalized playlists, continuous mixes and scenario planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 PASS, 1 FAIL, 2 usage error, 3 FFmpeg (or an optional package such as the AI SDK) "
               "missing, 4 WARNING,\n5 runtime error (incl. network; enrich/interpret: every request failed; "
               "suggest: no answer from the AI provider), 130 interrupted (Ctrl+C).")
    parser.add_argument("--version", action="version", version=f"veg_mixer {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    a = sub.add_parser("analyze", help="analyse BPM/key/energy/loudness (cached); export curator metadata")
    _common(a, folder_help="music library folder")
    _library(a)
    a.add_argument("--export-metadata", help="write a curator CSV/JSON template with current values")
    a.add_argument("--no-loudness", action="store_true", help="skip loudness measurement")
    a.add_argument("--segments", action="store_true",
                   help="VEmix 3.0: also compute segment-level analysis (energy over time, density, structure; cached)")

    n = sub.add_parser("normalize", help="normalize every file to verified MP3s (no sequencing)")
    _common(n, folder_help="music library folder")
    n.add_argument("--output", required=True, help="output folder (mirrors the library layout)")
    n.add_argument("--recursive", action="store_true")
    _loudness(n)

    pl = sub.add_parser("playlist", help="Mode A: separate normalized tracks + M3U8 + manifest + QC")
    _common(pl, folder_help="music library folder")
    pl.add_argument("--output", required=True, help="output folder")
    _library(pl)
    _selection(pl, daypart_all=True)
    _loudness(pl)

    m = sub.add_parser("mix", help="Mode B: one continuous MP3 (improved transitions + QC)")
    _common(m, folder_help="music library folder")
    m.add_argument("--output", help="output MP3 (default: <folder>/../<folder>_mix.mp3 like automix v1)")
    m.add_argument("--parts", type=int, default=1, help="split the sequenced mix into N files (default 1)")
    _library(m)
    _selection(m, daypart_all=True)
    _loudness(m)
    t = m.add_argument_group("transitions")
    t.add_argument("--crossfade", type=float, help="default crossfade in seconds (default 6)")
    t.add_argument("--max-overlap", type=float, help="maximum overlap in seconds (default 10)")
    t.add_argument("--no-crossfade", action="store_true", help="disable crossfades (de-clicked butt splices)")
    t.add_argument("--curve", choices=("equal_power", "linear", "s_curve"), help="fade curve (default equal_power)")
    t.add_argument("--fx", choices=("clean", "filter_sweep", "echo_tail", "filter+echo", "random"),
                   help="transition FX (default clean; random is opt-in)")
    t.add_argument("--no-trim-silence", action="store_true", help="keep leading/trailing silence")

    q = sub.add_parser("qc", help="re-verify an output folder (playlist or mix) and write QC reports")
    _common(q, folder_help="output folder produced by playlist/mix (daypart sub-folders are found too)")
    q.add_argument("--profile", help="client profile for exclusion checks")
    _loudness(q)
    _v3_parsers(sub)
    return parser


def _v3_parsers(sub) -> None:
    """VEmix 3.0 sub-commands. ``allow_abbrev=False``: their options must never become
    prefixes that change how 2.0 invocations are parsed."""
    modes = ("online", "cache", "off")

    e = sub.add_parser("enrich", allow_abbrev=False,
                       help="look up sourced facts about tracks (MusicBrainz, Wikidata; Discogs with a token)")
    _common(e, folder_help="music library folder")
    _library(e)
    e.add_argument("--mode", choices=modes, default="online",
                   help="online: cache first, network for misses (default); cache: cache only; off: nothing")
    e.add_argument("--research-cache", help="research cache JSON (default: veg_research_cache.json next to the "
                                            "analysis cache, i.e. in <folder> unless --cache moves it)")
    e.add_argument("--sources",
                   help="comma-separated: musicbrainz (required), wikidata, discogs (default: --mode online - "
                        "musicbrainz,wikidata, plus discogs when VEG_DISCOGS_TOKEN is set; --mode cache or --dry-run - "
                        "every cached source, Discogs included, no token needed)")
    e.add_argument("--contact", help="contact (e-mail or URL) for the User-Agent required by MusicBrainz "
                                     "(default: VEG_CONTACT or the project URL)")
    e.add_argument("--review-csv", help="write ambiguous/matched candidates for curator review (CSV)")
    e.add_argument("--report", help="write all research records as JSON")

    i = sub.add_parser("interpret", allow_abbrev=False,
                       help="structured AI interpretation of tracks (moods, roles; strict JSON schema)")
    _common(i, folder_help="music library folder")
    _library(i)
    i.add_argument("--mode", choices=modes, default="online",
                   help="online: cache first, then the provider (default); cache: cache only; off: nothing")
    i.add_argument("--provider", help="anthropic | openai-compatible | ollama | lmstudio | llamacpp | fake (demo echo) "
                                      "(default: VEG_AI_PROVIDER)")
    i.add_argument("--model", help="model id (default: VEG_AI_MODEL; anthropic: claude-opus-5-5)")
    i.add_argument("--base-url", help="OpenAI-compatible server URL (default: VEG_AI_BASE_URL)")
    i.add_argument("--ai-cache", help="AI cache JSON (default: veg_ai_cache.json next to the analysis cache)")
    i.add_argument("--research", choices=("cache", "off"), default="cache",
                   help="use verified facts from the research cache (default) or not")
    i.add_argument("--research-cache", help="research cache JSON")
    i.add_argument("--segments", choices=("cache", "compute", "off"), default="cache",
                   help="segment analysis input: cached only (default), compute missing, or off")
    i.add_argument("--use-community-tags", action="store_true",
                   help="also send community tags as unverified hints: MusicBrainz tags/genres (CC BY-NC-SA, "
                        "non-commercial) and Wikidata genres (crowd-edited)")
    i.add_argument("--retry-invalid", action="store_true",
                   help="online: ask the provider again for tracks whose cached answer is invalid")
    i.add_argument("--report", help="write interpretations as JSON")

    pl = sub.add_parser("plan", allow_abbrev=False,
                        help="plan a scenario (dry-run report of the full order, alternatives, exports)")
    _common(pl, folder_help="music library folder")
    _library(pl)
    g = pl.add_argument_group("scenario")
    g.add_argument("--scenario", required=True, help="scenario YAML/JSON (phases, anchors, curve, rules)")
    g.add_argument("--profile", help="client profile YAML/JSON (curation rules; default: permissive)")
    g.add_argument("--seed", type=int, help="seed (default: scenario seed or derived from its name)")
    g.add_argument("--alternatives", type=int, help="number of variants (1-6; default: scenario)")
    g.add_argument("--duration-min", type=float,
                   help="override the scenario target duration (minutes, > 0 and at most 1440; the scenario's "
                        "duration_tolerance_min is dropped for the default, and use_all_tracks is turned off)")
    g.add_argument("--max-tracks", type=int,
                   help="override the scenario maximum track count (checked against its anchors, required tracks "
                        "and min_tracks; use_all_tracks is turned off)")
    g.add_argument("--variant", help="variant shown in detail and exported to --m3u (default A)")
    g.add_argument("--research", choices=("cache", "off"), default="cache", help="use cached research facts")
    g.add_argument("--ai", choices=("cache", "off"), default="cache", help="use cached AI interpretations")
    g.add_argument("--segments", choices=("cache", "compute", "off"), default="cache",
                   help="segment analysis: cached only (default), compute missing, or off")
    g.add_argument("--research-cache", help="research cache JSON")
    g.add_argument("--ai-cache", help="AI cache JSON")
    g.add_argument("--use-community-tags", action="store_true",
                   help="let community tags (MusicBrainz tags/genres, CC BY-NC-SA; Wikidata genres, crowd-edited) "
                        "count as low-confidence mood hints, and use the cached AI answers made with them "
                        "(interpret --use-community-tags)")
    o = pl.add_argument_group("outputs (nothing is rendered)")
    o.add_argument("--save", help="write the plan file (input of `render` and `compare`)")
    o.add_argument("--m3u", help="write the shown variant as M3U8 pointing at the library files")
    o.add_argument("--csv", help="write every variant position by position (CSV)")
    o.add_argument("--report", help="write the full text report")
    o.add_argument("--no-reasons", action="store_true", help="omit the per-position reasons from the console")

    sg = sub.add_parser("suggest", allow_abbrev=False,
                        help="AI drafts a scenario (phases, curve, moods) from a brief; you review and plan with it")
    _common(sg, folder_help="music library folder (an anonymous summary of it is sent, never names or audio)")
    _library(sg)
    sg.add_argument("--brief", required=True, help="what the playlist is for, in your own words")
    sg.add_argument("--output", required=True, help="scenario draft to write (.yaml)")
    sg.add_argument("--duration-min", type=float, help="target duration in minutes")
    sg.add_argument("--provider", help="AI provider (default: VEG_AI_PROVIDER)")
    sg.add_argument("--model", help="model id (default: VEG_AI_MODEL; anthropic: claude-opus-5-5)")
    sg.add_argument("--base-url", help="OpenAI-compatible server URL")

    c = sub.add_parser("compare", allow_abbrev=False, help="compare the variants stored in a plan file")
    c.add_argument("--plan", required=True, help="plan file written by `plan --save`")
    c.add_argument("--variant", help="also print this variant in detail")
    c.add_argument("--reasons", action="store_true", help="with --variant: include per-position reasons")
    c.add_argument("-v", "--verbose", action="store_true")

    r = sub.add_parser("render", allow_abbrev=False,
                       help="render one variant of a plan file as a playlist or a continuous mix (+ QC)")
    r.add_argument("--plan", required=True, help="plan file written by `plan --save`")
    r.add_argument("--output", required=True, help="output folder (playlist) or .mp3 file (mix)")
    r.add_argument("--mode", choices=("playlist", "mix"), default="playlist")
    r.add_argument("--variant", help="variant label or preset (default: the variant `plan` showed and exported to "
                                     "--m3u, stored in the plan file as settings.variant_shown; the first variant, "
                                     "A, for plan files without it)")
    r.add_argument("--cache", help="analysis cache JSON (default: the cache the plan was made with, if `plan` got "
                                   "--cache and that file still exists; otherwise <library>/veg_analysis_cache.json)")
    r.add_argument("--profile", help="override the plan's client profile (QC exclusion checks)")
    r.add_argument("--metadata", help="override the plan's curator metadata")
    r.add_argument("--seed", type=int, help="seed for the mix renderer (default: the plan seed)")
    _runtime(r)
    _loudness(r)
    t = r.add_argument_group("transitions (mix)")
    t.add_argument("--crossfade", type=float, help="default crossfade in seconds (default 6)")
    t.add_argument("--max-overlap", type=float, help="maximum overlap in seconds (default 10)")
    t.add_argument("--no-crossfade", action="store_true", help="disable crossfades")
    t.add_argument("--curve", choices=("equal_power", "linear", "s_curve"), help="fade curve")
    t.add_argument("--fx", choices=("clean", "filter_sweep", "echo_tail", "filter+echo", "random"),
                   help="transition FX (default clean)")
    t.add_argument("--no-trim-silence", action="store_true", help="keep leading/trailing silence")


def _legacy_argv(argv: list[str]) -> list[str]:
    """automix v1 had no sub-commands: `automix.py --folder X ...` means `mix`."""
    if argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help", "--version") and argv[0].startswith("-"):
        return ["mix", *argv]
    return argv


# ---------------------------------------------------------------------------
# config from args
# ---------------------------------------------------------------------------

def config_from_args(args: argparse.Namespace) -> tuple[AppConfig, frozenset[str]]:
    """Engine config from the command line + the set of fields given explicitly
    ("loudness.<field>" / "encode.<field>"), so qc can keep every other manifest target."""
    cfg = AppConfig()
    lc, enc, tc, sc = cfg.loudness, cfg.encode, cfg.transitions, cfg.sequencing
    overrides: set[str] = set()
    mapping = {"target_lufs": "target_lufs", "true_peak": "target_true_peak_dbtp", "lra": "target_lra",
               "lufs_tolerance": "lufs_tolerance", "true_peak_tolerance": "true_peak_tolerance_db",
               "max_dynamic_gain": "max_dynamic_gain_db"}
    for arg, fld in mapping.items():
        v = getattr(args, arg, None)
        if v is not None:
            lc = dataclasses.replace(lc, **{fld: v})
            overrides.add(f"loudness.{fld}")
    if getattr(args, "bitrate", None):
        b = args.bitrate if args.bitrate.lower().endswith("k") else args.bitrate + "k"
        enc = dataclasses.replace(enc, bitrate=b)
        overrides.add("encode.bitrate")
    if getattr(args, "sample_rate", None):
        enc = dataclasses.replace(enc, sample_rate=args.sample_rate)
        overrides.add("encode.sample_rate")
    if getattr(args, "crossfade", None) is not None:
        tc = dataclasses.replace(tc, default_crossfade_s=args.crossfade)
    if getattr(args, "max_overlap", None) is not None:
        tc = dataclasses.replace(tc, max_overlap_s=args.max_overlap)
    if getattr(args, "no_crossfade", False):
        tc = dataclasses.replace(tc, enabled=False)
    if getattr(args, "curve", None):
        tc = dataclasses.replace(tc, curve=args.curve)
    if getattr(args, "fx", None):
        tc = dataclasses.replace(tc, fx_mode=args.fx)
    if getattr(args, "no_trim_silence", False):
        tc = dataclasses.replace(tc, trim_silence=False)
    cfg = dataclasses.replace(cfg, loudness=lc, encode=enc, transitions=tc, sequencing=sc)
    cfg.validate()
    return cfg, frozenset(overrides)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _out(msg: str) -> None:
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    _setup_console()
    argv = _legacy_argv(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_USAGE
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.command == "compare":          # reads a plan file only: no FFmpeg, no audio
        from .plan_pipeline import run_compare
        try:
            return run_compare(RunContext(tools=None, config=AppConfig(), out=_out),  # type: ignore[arg-type]
                               Path(args.plan), variant=args.variant, reasons=args.reasons)
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return EXIT_USAGE
    if not args.verbose:
        # numba's harmless re-compilation notice when librosa runs in several threads, and
        # librosa's short-window notices, would drown the progress output
        warnings.filterwarnings("ignore", message=r"Compilation requested for previously compiled")
        warnings.filterwarnings("ignore", category=UserWarning, module=r"librosa\..*")
    try:
        cfg, overrides = config_from_args(args)
    except ConfigError as e:
        print(f"ERROR: invalid setting: {e}", file=sys.stderr)
        return EXIT_USAGE
    try:
        tools = find_ffmpeg(args.ffmpeg, args.ffprobe)
    except FFmpegNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_ENV
    if args.verbose:
        _out(f"Using {tools.version} ({tools.ffmpeg}); ffprobe: {tools.ffprobe or 'not found'}")
    ctx = RunContext(tools=tools, config=cfg, jobs=max(1, args.jobs), dry_run=args.dry_run,
                     work_dir=Path(args.work_dir).resolve() if args.work_dir else None, out=_out,
                     overrides=overrides)
    if ctx.work_dir and not ctx.dry_run:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
    elif ctx.work_dir and not ctx.work_dir.is_dir():
        ctx.work_dir = None          # dry run: never create directories

    try:
        code = _dispatch(args, ctx)
    except FFmpegNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_ENV
    except (FileNotFoundError, ValueError, KeyError) as e:   # incl. ProfileError, MetadataError, ScenarioError
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_USAGE
    except OptionalDependencyError as e:    # an optional package (e.g. the anthropic SDK) is not installed
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_ENV
    except NetError as e:               # research/AI modules degrade themselves; this is a last resort
        print(f"ERROR: network: {e}", file=sys.stderr)
        return EXIT_RUNTIME
    except (PlaylistExportError, RuntimeError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return EXIT_RUNTIME
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    if code == EXIT_WARNING and args.warnings_ok:
        return EXIT_OK
    return code


def _dispatch(args: argparse.Namespace, ctx: RunContext) -> int:
    if args.command in V3_COMMANDS:
        return _dispatch_v3(args, ctx)
    folder = Path(args.folder)
    duration_s = args.duration_min * 60 if getattr(args, "duration_min", None) else None
    if args.command == "analyze":
        return run_analyze(ctx, folder, cache_arg=args.cache, recursive=args.recursive,
                           metadata_path=args.metadata, export_path=args.export_metadata,
                           measure_loudness=not args.no_loudness, segments=args.segments)
    if args.command == "normalize":
        return run_normalize(ctx, folder, Path(args.output), recursive=args.recursive)
    if args.command == "playlist":
        return run_playlist(ctx, folder, Path(args.output), profile_path=args.profile, metadata_path=args.metadata,
                            cache_arg=args.cache, recursive=args.recursive, daypart=args.daypart, seed=args.seed,
                            target_duration_s=duration_s, max_tracks=args.max_tracks)
    if args.command == "mix":
        src = folder.resolve()
        output = Path(args.output) if args.output else src.parent / f"{src.name}_mix.mp3"
        return run_mix(ctx, folder, output, profile_path=args.profile, metadata_path=args.metadata,
                       cache_arg=args.cache, recursive=args.recursive, daypart=args.daypart, parts=args.parts,
                       seed=args.seed, target_duration_s=duration_s, max_tracks=args.max_tracks)
    if args.command == "qc":
        return run_qc(ctx, folder, profile_path=args.profile)
    raise ValueError(f"unknown command {args.command}")


def _dispatch_v3(args: argparse.Namespace, ctx: RunContext) -> int:
    from .plan_pipeline import run_enrich, run_interpret, run_plan, run_render, run_suggest
    if args.command == "enrich":
        sources = tuple(s.strip().lower() for s in args.sources.split(",") if s.strip()) if args.sources else None
        return run_enrich(ctx, Path(args.folder), cache_arg=args.cache, recursive=args.recursive,
                          metadata_path=args.metadata, mode=args.mode, research_cache=args.research_cache,
                          sources=sources, contact=args.contact, review_csv=args.review_csv, report_json=args.report)
    if args.command == "interpret":
        return run_interpret(ctx, Path(args.folder), cache_arg=args.cache, recursive=args.recursive,
                             metadata_path=args.metadata, mode=args.mode, provider=args.provider, model=args.model,
                             base_url=args.base_url, ai_cache=args.ai_cache, research_cache=args.research_cache,
                             research=args.research, segments=args.segments,
                             use_community_tags=args.use_community_tags, report_json=args.report,
                             retry_invalid=args.retry_invalid)
    if args.command == "plan":
        return run_plan(ctx, Path(args.folder), scenario_path=args.scenario, profile_path=args.profile,
                        metadata_path=args.metadata, cache_arg=args.cache, recursive=args.recursive, seed=args.seed,
                        alternatives=args.alternatives,
                        target_duration_s=args.duration_min * 60 if args.duration_min is not None else None,
                        max_tracks=args.max_tracks, variant=args.variant, research=args.research, ai=args.ai,
                        segments=args.segments, research_cache=args.research_cache, ai_cache=args.ai_cache,
                        use_community_tags=args.use_community_tags, save_path=args.save, m3u_path=args.m3u,
                        csv_path=args.csv, report_path=args.report, reasons=not args.no_reasons)
    if args.command == "suggest":
        return run_suggest(ctx, Path(args.folder), brief=args.brief, output=args.output, cache_arg=args.cache,
                           recursive=args.recursive, metadata_path=args.metadata,
                           duration_min=args.duration_min, provider=args.provider, model=args.model,
                           base_url=args.base_url)
    if args.command == "render":
        return run_render(ctx, Path(args.plan), output=args.output, mode=args.mode, variant=args.variant,
                          cache_arg=args.cache, profile_path=args.profile, metadata_path=args.metadata,
                          seed=args.seed)
    raise ValueError(f"unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
