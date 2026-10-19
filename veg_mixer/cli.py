"""Command-line interface.

    python automix.py analyze   --folder ./music [--export-metadata curator.csv]
    python automix.py normalize --folder ./music --output ./normalized --target-lufs -16 --true-peak -1.5
    python automix.py playlist  --folder ./music --profile ./profiles/client.yaml --output ./output [--daypart all]
    python automix.py mix       --folder ./music --profile ./profiles/client.yaml --output ./output/mix.mp3
    python automix.py qc        --folder ./output

Backward compatible: ``python automix.py --folder ./music [--output X.mp3] [--parts N]
[--bitrate 320k] [--cache bpm_cache.json]`` (no sub-command) runs ``mix`` like automix v1,
now with the improved transition engine, loudness normalization and QC.

Exit codes: 0 PASS, 1 FAIL, 2 usage/input error, 3 environment error (FFmpeg missing),
4 WARNING (use --warnings-ok to map WARNING to 0), 5 runtime error (output folder locked,
export rolled back, disk errors).
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
from .playlist import PlaylistExportError
from .pipeline import (
    ALL_DAYPARTS,
    EXIT_ENV,
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_USAGE,
    EXIT_WARNING,
    RunContext,
    run_analyze,
    run_mix,
    run_normalize,
    run_playlist,
    run_qc,
)

COMMANDS = ("analyze", "normalize", "playlist", "mix", "qc")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def _common(p: argparse.ArgumentParser, *, folder_help: str) -> None:
    p.add_argument("--folder", required=True, help=folder_help)
    g = p.add_argument_group("runtime")
    g.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)),
                   help="parallel FFmpeg jobs (default: half the CPUs, max 4)")
    g.add_argument("--dry-run", action="store_true", help="plan and print, write nothing")
    g.add_argument("--work-dir", help="directory for temporary lossless intermediates")
    g.add_argument("--ffmpeg", help="path to ffmpeg (default: VEG_FFMPEG or PATH)")
    g.add_argument("--ffprobe", help="path to ffprobe (default: VEG_FFPROBE, next to ffmpeg, or PATH)")
    g.add_argument("--warnings-ok", action="store_true", help="exit 0 instead of 4 when QC status is WARNING")
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
        prog="automix.py", description="VEgroove Automix 2.0 — loudness-normalized playlists and continuous mixes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 PASS, 1 FAIL, 2 usage error, 3 FFmpeg missing, 4 WARNING, 5 runtime error.")
    parser.add_argument("--version", action="version", version=f"veg_mixer {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    a = sub.add_parser("analyze", help="analyse BPM/key/energy/loudness (cached); export curator metadata")
    _common(a, folder_help="music library folder")
    _library(a)
    a.add_argument("--export-metadata", help="write a curator CSV/JSON template with current values")
    a.add_argument("--no-loudness", action="store_true", help="skip loudness measurement")

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
    return parser


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
    except (FileNotFoundError, ValueError, KeyError) as e:   # incl. ProfileError, MetadataError
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_USAGE
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
    folder = Path(args.folder)
    duration_s = args.duration_min * 60 if getattr(args, "duration_min", None) else None
    if args.command == "analyze":
        return run_analyze(ctx, folder, cache_arg=args.cache, recursive=args.recursive,
                           metadata_path=args.metadata, export_path=args.export_metadata,
                           measure_loudness=not args.no_loudness)
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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
