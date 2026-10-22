"""Mode A (primary product): individually playable, loudness-normalized tracks plus
playlist files.

Output layout (``output_dir``)::

    playlist.m3u8                       extended M3U, UTF-8 (no BOM), relative POSIX paths
    manifest.json                       machine-readable description (schema below)
    tracklist.txt                       human-readable tracklist with loudness numbers
    tracks/NNN_<artist>_<title>.mp3     NNN = 1-based play order, zero-padded to >= 3 digits

An M3U8 file is a playlist, not a player. It hands any player the tracks in order as
separate files; skipping, repeat and seeking are features of the player that opens
it. Nothing in this module implements (or claims to implement) playback controls.

Publishing
----------
Every track is normalized (``LoudnessNormalizer.normalize_to_mp3``, the only lossy
step) into a private staging folder ``tracks/.veg-staging-*``. Only when the whole
sequence has been processed are the final names assigned (failed tracks removed,
indices renumbered contiguously, ID3 track numbers updated in place) and everything
is published in one short rename transaction:

1. files the PREVIOUS manifest listed under ``tracks/`` and the three playlist files
   are moved aside into the staging folder;
2. the new tracks are moved in under their final names;
3. ``manifest.json``, ``playlist.m3u8`` and ``tracklist.txt`` are written (temp file in
   the same folder + ``os.replace``).

Any error undoes every rename, so the folder never mixes two runs. Files the previous
manifest did not list (user files) are never moved, overwritten or deleted; a new name
that would hit one gets a ``_2`` / ``_3`` suffix instead. Sources must not live in
``tracks/`` — that is refused before any processing.

The three playlist files are only replaced when they belong to the exporter: a
VEgroove *playlist* ``manifest.json`` owns the folder, or (no manifest — an interrupted
earlier publish) the file carries the VEgroove header (``#PLAYLIST:VEgroove`` /
``VEgroove playlist:``). A foreign ``manifest.json``, ``playlist.m3u8`` or
``tracklist.txt`` (a client's hand-made list, a web app's manifest, a folder with one of
those names) makes the export refuse before any processing — it would be overwritten.

If an undo itself fails (a file locked mid-rollback), the staging folder still holds
files of the previous run. It is renamed to ``tracks/.veg-recovery-<UTC time>-<id>``
with a ``RECOVERY.txt`` note (what belongs where) and is never deleted automatically:
every later export lists it in ``manifest.warnings`` until the user removes it. Other
``.veg-staging-*`` folders idle for ``STALE_STAGING_AGE_S`` are abandoned work and are
removed (unless they carry ``RECOVERY.txt`` — the fallback when the rename failed).

An empty sequence (e.g. a daypart no eligible track fits) is not an error: nothing is
written, the previous output stays untouched, and the result has no paths and an
unpublished manifest whose ``warnings`` say why.

ID3 tags: the source tags are copied by the normalizer; the exporter adds
``track = i/N`` (play order) and the playlist name in ``TXXX:VEGROOVE_PLAYLIST``.
``album`` is set to the playlist name only when the output would otherwise have none
(source without album, or source tags not copied) — a source album is never replaced.

manifest.json (``schema = "vegroove.manifest"``, ``mode = "playlist"``)
-----------------------------------------------------------------------
Intended for QC and a future web player. ``null`` always means *unknown*.

* ``schema``, ``schema_version``, ``generator``, ``generated_at`` (UTC, ISO-8601 ``Z``),
  ``mode``, ``profile``, ``daypart``.
* ``loudness_targets``: ``target_lufs``, ``target_true_peak_dbtp``, ``target_lra``,
  ``lufs_tolerance``, ``true_peak_tolerance_db``, ``codec``, ``encoder``, ``bitrate``,
  ``sample_rate``, ``channels``.
* ``tracks[]`` in play order (same order as ``playlist.m3u8``):

  - ``index`` — 1-based play order; ``file`` — path relative to the manifest folder
    (POSIX separators; URL-encode each segment when building URLs);
  - ``title``, ``artist`` (``null`` = unknown; ``title`` falls back to the file stem);
  - ``duration_s`` — decoded duration of the published MP3;
  - ``source_rel``, ``source_path``, ``track_id``;
  - ``loudness`` — ``NormalizationResult.to_metadata()``: ``target_lufs``,
    ``measured_lufs``, ``true_peak_dbtp``, ``original_lufs``, ``original_true_peak_dbtp``,
    ``normalization_status`` (``passed`` | ``review``), ``requires_review``,
    ``processing_mode``, ``warnings``, ``review_reasons``, ...;
  - ``features`` — ``bpm``, ``camelot``, ``key_confidence``, ``energy``,
    ``energy_confidence``, ``danceability``, ``vocal_presence``, ``genre``, ``mood``,
    ``vibe_tags``, ``language``, ``familiarity``, ``approved``, ``requires_review``,
    ``review_reasons``, ``script_hint``, ``feature_sources`` (field -> ``manual`` |
    ``id3`` | ``filename`` | ``auto:<algo>``), ``source_duration_s``,
    ``sequence_position``, ``daypart_fit``, ``sequence_notes``;
  - ``transition_in`` — the sequencer's score of the transition from the previous
    track (``total``, ``components``, ``coverage``, ``flags``); ``null`` for the first
    track and when the previous track changed because a track failed;
  - ``flags`` — subset of ``loudness_review`` (status ``review`` or review reasons),
    ``metadata_review`` (curation asked for review), ``listen_transition`` (incoming
    transition listed in ``flagged_transitions``).

* ``excluded[]`` — exclusions as passed in by the caller (their ``stage``, default
  ``"curation"``; ``"analysis"`` for unreadable files) plus tracks that failed
  normalization (``stage = "normalization"``, ``reasons`` =
  ``["normalization failed: ..."]``, ``sequence_position``, ``loudness``). Every entry
  carries ``stage``.
* ``not_selected[]`` — eligible tracks the sequencer did not place in this playlist, as
  passed in by the caller: ``source_rel`` (copied from the sequencer's ``rel_path``,
  which is kept too), ``track_id``, ``reason``. Listed in ``tracklist.txt`` as well.
* ``flagged_transitions[]`` — ``from_index``, ``to_index``, ``kind`` and ``reasons``.
  ``kind``: ``sequencing`` (from the sequencer; adds ``from_track``, ``to_track``,
  ``score``), ``loudness`` (adjacent short-term loudness differs a lot — detection
  only, nothing is compressed), ``unscored`` (tracks became adjacent because a track
  between them failed; the sequencer never scored that transition).
* ``warnings[]`` — human-readable; failures are never dropped silently.
* ``leftover_files[]`` — files of an earlier run that could not be removed (e.g. open
  in a player); still owned by the exporter and removed by the next run (``[]`` when
  there are none; older manifests without the key are read the same way).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
import unicodedata
import uuid
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import GENERATOR
from .config import EncodeConfig, LoudnessConfig
from .ffmpeg_tools import FFmpegNotFoundError
from .loudness import LoudnessNormalizer, compare_adjacent_short_term
from .models import (
    AudioProbe,
    LoudnessMeasurement,
    ManifestTrack,
    NormalizationResult,
    NormalizationStatus,
    PlaylistExportResult,
    PlaylistManifest,
    SequencedTrack,
    SequenceResult,
    TrackFeatures,
    to_jsonable,
)

log = logging.getLogger(__name__)

PLAYLIST_FILE = "playlist.m3u8"
MANIFEST_FILE = "manifest.json"
TRACKLIST_FILE = "tracklist.txt"
TRACKS_DIR = "tracks"
OUTPUT_FILES = (MANIFEST_FILE, PLAYLIST_FILE, TRACKLIST_FILE)
STAGING_PREFIX = ".veg-staging-"
STALE_STAGING_AGE_S = 6 * 3600
# A staging folder that still holds previous-run files after a failed undo; never auto-deleted.
RECOVERY_PREFIX = ".veg-recovery-"
RECOVERY_NOTE = "RECOVERY.txt"
# ID3 TXXX frame carrying the playlist name (album is only set when the source has none).
PLAYLIST_TAG = "VEGROOVE_PLAYLIST"
# Header lines that identify a playlist file written by this module.
_OUTPUT_SIGNATURES = {PLAYLIST_FILE: "#PLAYLIST:VEgroove", TRACKLIST_FILE: "VEgroove playlist:"}
TRACK_EXT = ".mp3"
MIN_INDEX_WIDTH = 3
# Portable file-name limit: NTFS allows 255 UTF-16 units, ext4 255 bytes.
MAX_NAME_BYTES = 240

ProgressCallback = Callable[[int, int, str], None]

# Flags on ManifestTrack.flags
FLAG_LOUDNESS_REVIEW = "loudness_review"
FLAG_METADATA_REVIEW = "metadata_review"
FLAG_LISTEN_TRANSITION = "listen_transition"
_FLAG_ORDER = (FLAG_LOUDNESS_REVIEW, FLAG_METADATA_REVIEW, FLAG_LISTEN_TRANSITION)

_FEATURE_FIELDS = (
    "bpm", "camelot", "key_confidence", "energy", "energy_confidence", "danceability",
    "vocal_presence", "genre", "mood", "vibe_tags", "language", "familiarity",
    "approved", "requires_review", "review_reasons", "script_hint", "feature_sources",
)


class PlaylistExportError(RuntimeError):
    """The playlist cannot be exported safely (nothing, or nothing new, was published)."""


class _RollbackIncomplete(PlaylistExportError):
    """Publishing failed and not every rename could be undone; the staging folder is
    kept for manual recovery at ``recovery_dir``."""

    def __init__(self, message: str, recovery_dir: Path) -> None:
        super().__init__(message)
        self.recovery_dir = recovery_dir


# ---------------------------------------------------------------------------
# File names (pure)
# ---------------------------------------------------------------------------

_WIN_FORBIDDEN = frozenset('<>:"/\\|?*')
# Break URLs of a web player unless encoded; replaced as well.
_URL_HOSTILE = frozenset("#%")
# Punctuation that is safe on every file system and readable in a file name.
_KEEP_PUNCT = frozenset("-.,()[]'!&+")
_EDGE_JUNK = "_.-, "
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{dev}{n}" for dev in ("COM", "LPT") for n in "0123456789¹²³"})


def sanitize_component(text: str | None) -> str:
    """Turn free text into one safe, readable file-name component.

    NFC-normalized; Unicode letters, digits and marks are kept (Cyrillic, accents,
    CJK, ...). Windows-forbidden characters ``<>:"/\\|?*``, ``#``/``%``, control
    characters, whitespace and other symbols become ``_`` (runs collapsed); invisible
    format characters (zero-width, bidi overrides) are dropped. Leading/trailing
    ``_ . - ,`` and spaces are stripped and Windows device names (``CON``, ``COM1``,
    ...) get a trailing ``_``. Returns ``""`` when nothing usable is left.
    """
    if not text:
        return ""
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", str(text)):
        cat = unicodedata.category(ch)
        if ch in _WIN_FORBIDDEN or ch in _URL_HOSTILE or ch.isspace() or cat == "Cc":
            out.append("_")
        elif cat in ("Cf", "Cs", "Co", "Cn"):
            continue
        elif cat[0] in "LNM" or ch in _KEEP_PUNCT:
            out.append(ch)
        else:
            out.append("_")       # symbols, emoji, other punctuation
    s = "".join(out)
    s = re.sub(r"_*-_*", "-", s)  # "Artist - Title" -> "Artist-Title"
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"\.{2,}", ".", s)
    s = s.strip(_EDGE_JUNK)
    base, dot, rest = s.partition(".")
    if base.upper() in _RESERVED_NAMES:     # Windows checks the part before the first dot
        s = f"{base}_{dot}{rest}"
    return s


def _trim(s: str, n: int) -> str:
    return s[:max(n, 0)].rstrip(_EDGE_JUNK)


def _fit_parts(parts: list[str], budget: int) -> str:
    """Join ``[artist, title]`` (or ``[stem]``) with ``_`` within ``budget`` characters;
    the title keeps at least two thirds of the space."""
    if len(parts) == 1:
        return _trim(parts[0], budget)
    artist, title = parts
    if len(artist) + 1 + len(title) <= budget:
        return f"{artist}_{title}"
    title_t = _trim(title, max(budget - len(artist) - 1, (2 * budget) // 3))
    artist_t = _trim(artist, budget - len(title_t) - 1)
    return f"{artist_t}_{title_t}" if artist_t else title_t


def _compose_name(index: int, artist: str | None, title: str | None, fallback: str, *,
                  max_len: int, width: int, ext: str, suffix: str) -> str:
    if index < 1:
        raise ValueError(f"track index is 1-based, got {index}")
    prefix = f"{index:0{max(width, 1)}d}_"
    title_s = sanitize_component(title)
    if title_s:
        parts = [p for p in (sanitize_component(artist), title_s) if p]
    else:   # unknown title: the source file stem usually carries "Artist - Title"
        parts = [sanitize_component(fallback) or "track"]
    budget = max_len - len(prefix) - len(suffix) - len(ext)
    if budget < 1:
        raise ValueError(f"max_len={max_len} leaves no room for a name after {prefix!r}")
    while True:
        name = f"{prefix}{_fit_parts(parts, budget)}{suffix}{ext}"
        if len(name.encode("utf-8")) <= MAX_NAME_BYTES or budget <= 1:
            return name
        budget -= 1


def safe_filename(index: int, artist: str | None, title: str | None, fallback: str, *,
                  max_len: int = 120, width: int = MIN_INDEX_WIDTH, ext: str = TRACK_EXT) -> str:
    """``NNN_<artist>_<title>.mp3`` — a portable file name for play position ``index``.

    ``NNN`` is zero-padded to ``width`` (>= 3) digits. Unknown title -> the sanitized
    ``fallback`` (the source file stem), then ``"track"``. The result is at most
    ``max_len`` characters and ``MAX_NAME_BYTES`` UTF-8 bytes. Collisions between
    tracks are resolved by the exporter (``_2``, ``_3`` ... suffixes), not here.
    """
    return _compose_name(index, artist, title, fallback, max_len=max_len, width=width, ext=ext, suffix="")


def _slug_key(name: str) -> str:
    """Collision key of a planned name: everything after the ``NNN_`` prefix, case-folded."""
    return name.split("_", 1)[1].casefold()


def _plan_names(entries: list[tuple[int, TrackFeatures]], width: int, foreign: set[str]) -> list[str]:
    """Final names for ``(index, track)`` pairs. A name whose artist/title part repeats
    an earlier one (case-insensitively) or that would hit a foreign file gets ``_2``,
    ``_3`` ..."""
    used: set[str] = set()
    names: list[str] = []
    for index, t in entries:
        n = 1
        while True:
            name = _compose_name(index, t.artist, t.title, Path(t.path).stem, max_len=120, width=width,
                                 ext=TRACK_EXT, suffix="" if n == 1 else f"_{n}")
            key = _slug_key(name)
            if key not in used and name.casefold() not in foreign:
                break
            n += 1
        used.add(key)
        names.append(name)
    return names


def index_width(count: int) -> int:
    return max(MIN_INDEX_WIDTH, len(str(max(count, 1))))


# ---------------------------------------------------------------------------
# M3U8
# ---------------------------------------------------------------------------

def _clean_text(text: object) -> str:
    """Single-line text: control characters and line separators -> space, whitespace collapsed."""
    if text is None:
        return ""
    s = "".join(" " if unicodedata.category(c) in ("Cc", "Zl", "Zp") else c
                for c in unicodedata.normalize("NFC", str(text)))
    return re.sub(r"\s+", " ", s).strip()


def _relative_posix(file: str) -> str:
    rel = str(file).replace("\\", "/")
    if (not rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel)
            or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", rel)):
        raise ValueError(f"playlist entries must be relative paths, got {file!r}")
    if any(part in ("", ".", "..") for part in rel.split("/")):
        raise ValueError(f"playlist entry {file!r} must stay inside the playlist folder")
    if any(unicodedata.category(c) in ("Cc", "Zl", "Zp") for c in rel):
        raise ValueError(f"playlist entry {file!r} contains control characters")
    return rel


def _entry_label(entry: ManifestTrack) -> str:
    label = f"{entry.artist} - {entry.title}" if entry.artist else entry.title
    return _clean_text(label) or Path(entry.file).stem


def render_m3u8(entries: list[ManifestTrack], title: str | None = None) -> str:
    """Extended M3U text: ``#EXTM3U``, optional ``#PLAYLIST:``, then per entry
    ``#EXTINF:<seconds>,<Artist - Title>`` and the relative POSIX path. Unknown
    durations are written as ``-1`` (the M3U convention)."""
    lines = ["#EXTM3U"]
    playlist_title = _clean_text(title)
    if playlist_title:
        lines.append(f"#PLAYLIST:{playlist_title}")
    for entry in entries:
        rel = _relative_posix(entry.file)
        d = entry.duration_s
        seconds = int(math.floor(d + 0.5)) if d is not None and math.isfinite(d) and d >= 0 else -1
        lines.append(f"#EXTINF:{seconds},{_entry_label(entry)}")
        lines.append(f"./{rel}" if rel.startswith("#") else rel)   # a leading '#' would read as a comment
    return "\n".join(lines) + "\n"


def write_m3u8(path: Path, entries: list[ManifestTrack], title: str | None = None) -> None:
    """Write an extended M3U playlist (UTF-8 without BOM, LF) atomically."""
    _atomic_write_bytes(Path(path), render_m3u8(entries, title).encode("utf-8"))


def _parse_extinf(line: str, where: str) -> tuple[float | None, str]:
    body = line[len("#EXTINF:"):]
    if "," not in body:
        raise ValueError(f"{where}: malformed #EXTINF (expected '<duration>,<title>'): {line!r}")
    head, title = body.split(",", 1)
    token = head.strip().split()[0] if head.strip() else ""
    try:
        duration = float(token)
    except ValueError:
        raise ValueError(f"{where}: bad #EXTINF duration {token!r}") from None
    if not math.isfinite(duration):
        raise ValueError(f"{where}: bad #EXTINF duration {token!r}")
    return (None if duration < 0 else duration), title.strip()


def read_m3u8(path: Path) -> list[tuple[float | None, str, str]]:
    """Parse an extended M3U file into ``(duration_s | None, title, relpath)`` triples.

    Raises ValueError for invalid UTF-8, a missing ``#EXTM3U`` header, an ``#EXTINF``
    not followed by a path, or a malformed duration. A path without ``#EXTINF`` is
    legal M3U and yields ``(None, "", path)``.
    """
    path = Path(path)
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(f"{path}: not valid UTF-8 ({e})") from e
    entries: list[tuple[float | None, str, str]] = []
    header_seen = False
    pending: tuple[float | None, str] | None = None
    pending_line = 0
    for no, raw in enumerate(re.split(r"\r\n|\r|\n", text), 1):
        line = raw.strip()
        if not line:
            continue
        if not header_seen:
            if line != "#EXTM3U":
                raise ValueError(f"{path}:{no}: missing #EXTM3U header")
            header_seen = True
            continue
        if line.startswith("#EXTINF:"):
            if pending is not None:
                raise ValueError(f"{path}:{pending_line}: #EXTINF without a path")
            pending, pending_line = _parse_extinf(line, f"{path}:{no}"), no
        elif line.startswith("#"):
            continue          # other directives (#PLAYLIST, #EXTGRP, ...) and comments
        elif pending is None:
            entries.append((None, "", line))
        else:
            entries.append((pending[0], pending[1], line))
            pending = None
    if not header_seen:
        raise ValueError(f"{path}: empty file, missing #EXTM3U header")
    if pending is not None:
        raise ValueError(f"{path}:{pending_line}: #EXTINF without a path")
    return entries


# ---------------------------------------------------------------------------
# tracklist.txt
# ---------------------------------------------------------------------------

def _hms(seconds: float) -> str:
    s = int(math.floor(seconds + 0.5))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _mss(seconds: float) -> str:
    s = int(math.floor(seconds + 0.5))
    return f"{s // 60}:{s % 60:02d}"


def _num(v: object, fmt: str = "{:.1f}") -> str:
    return fmt.format(v) if isinstance(v, (int, float)) and math.isfinite(v) else "—"


def render_tracklist(manifest: PlaylistManifest, *, title: str | None = None) -> str:
    """Human-readable tracklist: header, one row per track (start offset, duration,
    LUFS, true peak, status, flags), flagged transitions, exclusions, warnings."""
    lt = manifest.loudness_targets or {}
    tracks = manifest.tracks
    known = [t.duration_s for t in tracks if t.duration_s is not None]
    total_txt = _hms(sum(known)) + ("" if len(known) == len(tracks) else " (+ unknown)")
    lines = [
        f"VEgroove playlist: {_clean_text(title) or playlist_title(manifest.profile, manifest.daypart)}",
        f"Generated:  {manifest.generated_at} ({manifest.generator})",
        f"Profile:    {manifest.profile or '—'}",
        f"Daypart:    {manifest.daypart or '—'}",
        f"Tracks:     {len(tracks)} in playlist, "
        f"{sum(1 for e in manifest.excluded if e.get('stage') == 'normalization')} failed normalization, "
        f"{sum(1 for e in manifest.excluded if e.get('stage') != 'normalization')} excluded"
        + (f", {len(manifest.not_selected)} not selected" if manifest.not_selected else ""),
        f"Duration:   {total_txt}",
        (f"Targets:    {_num(lt.get('target_lufs'))} LUFS ±{_num(lt.get('lufs_tolerance'))} LU, "
         f"true peak <= {_num(lt.get('target_true_peak_dbtp'))} dBTP "
         f"(+{_num(lt.get('true_peak_tolerance_db'))} dB tolerance); "
         f"{lt.get('codec', '?')} {lt.get('bitrate', '?')}, {lt.get('sample_rate', '?')} Hz"),
        "Note:       playlist.m3u8 is a playlist file; skipping, repeat and seeking are",
        "            provided by the player that opens it.",
        "",
    ]
    labels = [_entry_label(t) for t in tracks]
    w = min(max([len(x) for x in labels] + [len("Artist - Title")]), 60)
    lines.append(f"{'#':>4} | {'Start':>8} | {'Length':>6} | {'Artist - Title':<{w}} | "
                 f"{'LUFS':>6} | {'TP':>6} | {'Status':<7} | Flags")
    lines.append("-" * (4 + 3 + 8 + 3 + 6 + 3 + w + 3 + 6 + 3 + 6 + 3 + 7 + 3 + 5))
    start: float | None = 0.0
    for t, label in zip(tracks, labels):
        if len(label) > w:
            label = label[:w - 1] + "…"
        loud = t.loudness or {}
        lines.append(
            f"{t.index:>4} | {_hms(start) if start is not None else '?':>8} | "
            f"{_mss(t.duration_s) if t.duration_s is not None else '?':>6} | {label:<{w}} | "
            f"{_num(loud.get('measured_lufs')):>6} | {_num(loud.get('true_peak_dbtp')):>6} | "
            f"{loud.get('normalization_status') or 'planned':<7} | {', '.join(t.flags)}")
        start = start + t.duration_s if start is not None and t.duration_s is not None else None

    lines += ["", "=== FLAGGED TRANSITIONS (listening review) ==="]
    if not manifest.flagged_transitions:
        lines.append("(none)")
    for ft in manifest.flagged_transitions:
        lines.append(f"{ft.get('from_index')} -> {ft.get('to_index')} [{ft.get('kind', '?')}]: "
                     + "; ".join(str(r) for r in ft.get("reasons") or []))
    lines += ["", "=== EXCLUDED ==="]
    if not manifest.excluded:
        lines.append("(none)")
    for ex in manifest.excluded:
        name = ex.get("source_rel") or ex.get("source_path") or ex.get("track_id") or "?"
        reasons = ex.get("reasons") or []
        if isinstance(reasons, str):
            reasons = [reasons]
        lines.append(f"- {name} [{ex.get('stage', '?')}]: " + "; ".join(str(r) for r in reasons))
    lines += ["", "=== NOT SELECTED (eligible, not placed in this playlist) ==="]
    if not manifest.not_selected:
        lines.append("(none)")
    for ns in manifest.not_selected:
        name = ns.get("source_rel") or ns.get("rel_path") or ns.get("track_id") or "?"
        reason = ns.get("reason") or ns.get("reasons") or "no reason given"
        if not isinstance(reason, str):
            reason = "; ".join(str(r) for r in reason)
        lines.append(f"- {name}: {reason}")
    lines += ["", "=== WARNINGS ==="]
    lines += [f"- {w_}" for w_ in manifest.warnings] or ["(none)"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# ID3 track number patch (needed when failed tracks shift the play order)
# ---------------------------------------------------------------------------

def _synchsafe(b: bytes) -> int:
    return (b[0] & 0x7F) << 21 | (b[1] & 0x7F) << 14 | (b[2] & 0x7F) << 7 | (b[3] & 0x7F)


def _to_synchsafe(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def _read_id3_tag(path: Path) -> tuple[int, bytearray] | None:
    """(major version, tag body) of a plain ID3v2.3/2.4 tag; None when absent or when
    it uses unsynchronisation / an extended header (not handled)."""
    with open(path, "rb") as f:
        header = f.read(10)
        if len(header) < 10 or header[:3] != b"ID3":
            return None
        major, flags = header[3], header[5]
        if major not in (3, 4) or flags & 0xC0:
            return None
        size = _synchsafe(header[6:10])
        body = bytearray(f.read(size))
    return (major, body) if len(body) == size else None


def _id3_frames(body: bytearray, major: int):
    """Yield ``(offset, frame_id, size, flags)``; stops at padding or garbage."""
    pos = 0
    while pos + 10 <= len(body):
        fid = bytes(body[pos:pos + 4])
        if not re.fullmatch(rb"[A-Z0-9]{4}", fid):
            return
        raw = bytes(body[pos + 4:pos + 8])
        size = _synchsafe(raw) if major == 4 else int.from_bytes(raw, "big")
        if pos + 10 + size > len(body):
            return
        yield pos, fid, size, bytes(body[pos + 8:pos + 10])
        pos += 10 + size


_ID3_CODECS = {0: "latin-1", 1: "utf-16", 2: "utf-16-be", 3: "utf-8"}


def _read_id3_text(path: Path, frame_id: str) -> str | None:
    tag = _read_id3_tag(path)
    if tag is None:
        return None
    major, body = tag
    for pos, fid, size, _flags in _id3_frames(body, major):
        if fid == frame_id.encode("ascii") and size >= 1:
            payload = bytes(body[pos + 10:pos + 10 + size])
            codec = _ID3_CODECS.get(payload[0])
            return payload[1:].decode(codec, errors="replace").rstrip("\x00") if codec else None
    return None


def _patch_id3_text(path: Path, frame_id: str, value: str) -> bool:
    """Replace a text frame in place when the new value is not longer than the old one.
    The tag keeps its size (freed bytes become padding), so the audio frames are not
    touched. Returns False when the tag layout is not one we can edit safely."""
    tag = _read_id3_tag(path)
    if tag is None:
        return False
    major, body = tag
    for pos, fid, size, flags in _id3_frames(body, major):
        if fid != frame_id.encode("ascii"):
            continue
        if flags != b"\x00\x00" or size < 1:
            return False
        payload = bytes(body[pos + 10:pos + 10 + size])
        enc = payload[0]
        term = b"\x00\x00" if enc in (1, 2) else b"\x00"
        terminated = payload[1:].endswith(term)
        try:
            raw = {0: lambda: value.encode("latin-1"), 1: lambda: b"\xff\xfe" + value.encode("utf-16-le"),
                   2: lambda: value.encode("utf-16-be"), 3: lambda: value.encode("utf-8")}[enc]()
        except (KeyError, UnicodeEncodeError):
            return False
        new = bytes([enc]) + raw + (term if terminated else b"")
        if len(new) > size:
            return False
        size_bytes = _to_synchsafe(len(new)) if major == 4 else len(new).to_bytes(4, "big")
        rebuilt = body[:pos] + fid + size_bytes + flags + new + body[pos + 10 + size:]
        rebuilt += b"\x00" * (len(body) - len(rebuilt))
        with open(path, "r+b") as f:
            f.seek(10)
            f.write(rebuilt)
        return True
    return False


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def _write_temp(directory: Path, name: str, data: bytes) -> Path:
    tmp = directory / f".{name}.veg-{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _write_temp(path.parent, path.name, data)
    try:
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _norm(p: Path | str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))


def _is_within(child: str, parent: str) -> bool:
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:      # different drives
        return False


class _Journal:
    """Rename journal: every move is recorded so a failed publish can be undone."""

    def __init__(self) -> None:
        self.moves: list[tuple[Path, Path]] = []

    def move(self, src: Path, dst: Path) -> None:
        os.replace(src, dst)
        self.moves.append((src, dst))

    def rollback(self) -> list[tuple[Path, Path, str]]:
        """Undo every move (newest first); returns ``(src, dst, error)`` of the moves that
        could not be undone — the file is still at ``dst`` but belongs at ``src``."""
        problems: list[tuple[Path, Path, str]] = []
        for src, dst in reversed(self.moves):
            try:
                os.replace(dst, src)
            except OSError as e:
                problems.append((src, dst, str(e)))
        self.moves.clear()
        return problems


# ---------------------------------------------------------------------------
# Previous run / safety checks
# ---------------------------------------------------------------------------

def _previous_owned_files(manifest_path: Path) -> tuple[list[str], list[str]]:
    """Names under ``tracks/`` that the previous manifest listed (they belong to the
    exporter), plus warnings. Refuses to continue when ``manifest.json`` exists but is
    not a VEgroove playlist manifest — it would be overwritten."""
    if not manifest_path.exists():
        return [], []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PlaylistExportError(
            f"{manifest_path} exists but is not readable JSON ({e}); refusing to overwrite it — "
            f"move it away or choose another output folder") from e
    if not isinstance(data, dict) or data.get("schema") != "vegroove.manifest":
        raise PlaylistExportError(
            f"{manifest_path} is not a VEgroove manifest; refusing to overwrite it — "
            f"choose another output folder")
    if data.get("mode") != "playlist":
        raise PlaylistExportError(
            f"{manifest_path} is a VEgroove {data.get('mode')!r} manifest, not a playlist; "
            f"refusing to overwrite it — choose another output folder")

    files = [t.get("file") for t in data.get("tracks") or [] if isinstance(t, dict)]
    files += list(data.get("leftover_files") or [])
    tracks_dir = manifest_path.parent / TRACKS_DIR
    owned: list[str] = []
    warnings: list[str] = []
    for rel in files:
        parts = str(rel).replace("\\", "/").split("/") if isinstance(rel, str) else []
        name = parts[1] if len(parts) == 2 and parts[0] == TRACKS_DIR else ""
        if (not name or name in (".", "..") or name.startswith((STAGING_PREFIX, RECOVERY_PREFIX))
                or any(c in name for c in '<>:"|?*') or name != name.strip()):
            warnings.append(f"previous manifest lists an unexpected path {rel!r}; it is left untouched")
            continue
        p = tracks_dir / name
        if p.is_dir() and not p.is_symlink():     # we only ever write files there
            warnings.append(f"{TRACKS_DIR}/{name} listed by the previous manifest is a folder; it is left untouched")
            continue
        if name not in owned:
            owned.append(name)
    return owned, warnings


def _carries_signature(path: Path, signature: str) -> bool:
    """True when one of the first lines of ``path`` starts with ``signature``."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096).decode("utf-8-sig", errors="replace")
    except OSError:
        return False
    return any(line.strip().startswith(signature) for line in head.splitlines()[:3])


def _foreign_output_files(output_dir: Path, *, manifest_owned: bool) -> list[str]:
    """Names of ``OUTPUT_FILES`` that exist in ``output_dir`` but are not the exporter's
    to replace. They are when a VEgroove playlist ``manifest.json`` owned the folder at
    the start of the export (``manifest_owned``) or — ``playlist.m3u8`` /
    ``tracklist.txt`` of an interrupted earlier publish — when the file carries the
    VEgroove header. A folder under one of those names is never ours."""
    foreign: list[str] = []
    for fname in OUTPUT_FILES:
        p = output_dir / fname
        if not os.path.lexists(p):
            continue
        if p.is_dir() and not p.is_symlink():
            foreign.append(fname)
        elif not manifest_owned:
            signature = _OUTPUT_SIGNATURES.get(fname)
            if signature is None or not _carries_signature(p, signature):
                foreign.append(fname)
    return foreign


def _refuse_foreign_outputs(output_dir: Path, foreign: list[str]) -> PlaylistExportError:
    names = ", ".join(foreign)
    return PlaylistExportError(
        f"{names} in {output_dir} {'was' if len(foreign) == 1 else 'were'} not written by VEgroove "
        f"(no VEgroove playlist manifest owns this folder); refusing to overwrite "
        f"{'it' if len(foreign) == 1 else 'them'} — move {'it' if len(foreign) == 1 else 'them'} away "
        f"or choose another output folder")


def _check_sources(items: list[SequencedTrack], output_dir: Path, tracks_dir: Path) -> None:
    """Refuse before any processing when a destination could resolve to a source."""
    tracks_n = _norm(tracks_dir)
    protected = {_norm(output_dir / f): f for f in OUTPUT_FILES}
    for st in items:
        src = _norm(st.track.path)
        if _is_within(src, tracks_n):
            raise PlaylistExportError(
                f"source {st.track.path} is inside the output tracks folder {tracks_dir}; files there are "
                f"replaced or removed by the export — move the sources or choose another output folder")
        if src in protected:
            raise PlaylistExportError(f"source {st.track.path} would be overwritten by {protected[src]}")


def _foreign_names(tracks_dir: Path, owned: list[str]) -> set[str]:
    """Case-folded names in ``tracks/`` that the previous manifest did not list."""
    try:
        entries = os.listdir(tracks_dir)
    except (FileNotFoundError, NotADirectoryError):
        return set()
    owned_set = set(owned)
    return {unicodedata.normalize("NFC", e).casefold() for e in entries
            if e not in owned_set and not e.startswith(STAGING_PREFIX)}


def _remove_stale_staging(tracks_dir: Path, *, now: float | None = None) -> None:
    """Remove staging folders of interrupted exports. A running export touches its
    staging folder with every finished track, so only folders idle for
    ``STALE_STAGING_AGE_S`` are treated as abandoned (concurrent exports into one
    folder are not supported, but must not destroy each other's work). A folder kept
    for manual recovery (``RECOVERY.txt`` inside, or renamed to ``.veg-recovery-*``) is
    never removed."""
    now = time.time() if now is None else now
    for p in tracks_dir.glob(STAGING_PREFIX + "*"):
        try:
            idle = now - p.stat().st_mtime
        except OSError:
            continue
        if not p.is_dir() or p.is_symlink() or idle <= STALE_STAGING_AGE_S:
            continue
        if os.path.lexists(p / RECOVERY_NOTE):
            log.warning("staging folder %s is kept for manual recovery (%s); not removing it", p, RECOVERY_NOTE)
            continue
        log.warning("removing staging folder left by an interrupted export: %s", p)
        shutil.rmtree(p, ignore_errors=True)


def _recovery_folders(tracks_dir: Path) -> list[str]:
    """Names of folders in ``tracks/`` kept for manual recovery by an earlier export."""
    try:
        entries = sorted(os.listdir(tracks_dir))
    except (FileNotFoundError, NotADirectoryError):
        return []
    found = []
    for name in entries:
        p = tracks_dir / name
        if not p.is_dir() or p.is_symlink():
            continue
        if name.startswith(RECOVERY_PREFIX) or (name.startswith(STAGING_PREFIX)
                                                and os.path.lexists(p / RECOVERY_NOTE)):
            found.append(name)
    return found


def _recovery_warnings(tracks_dir: Path) -> list[str]:
    return [f"{TRACKS_DIR}/{name} holds files of an earlier export whose publishing could not be undone; "
            f"it is kept for manual recovery (see {RECOVERY_NOTE} inside) and is never deleted "
            f"automatically — remove it once it is no longer needed" for name in _recovery_folders(tracks_dir)]


def _keep_for_recovery(staging: Path, output_dir: Path, cause: BaseException,
                       failures: list[tuple[Path, Path, str]]) -> Path:
    """Protect a staging folder that still holds files after an incomplete rollback:
    write ``RECOVERY.txt`` (what belongs where) and rename the folder to
    ``.veg-recovery-<UTC time>-<id>``, a name ``_remove_stale_staging`` never matches.
    When the rename fails the note alone protects it. Returns the folder's final path."""
    def where(p: Path) -> str:
        try:
            return p.relative_to(staging).as_posix()      # inside this folder: survives the rename
        except ValueError:
            return str(p)

    lines = [
        "VEgroove playlist export: publishing failed and could not be fully undone.",
        f"Cause: {cause}",
        f"Output folder: {output_dir}",
        "",
        "Files that could not be moved back (paths relative to this folder, or absolute):",
        *[f"- {where(dst)}  belongs at  {where(src)}  ({err})" for src, dst, err in failures],
        "",
        ("manifest.json is among them: move it back FIRST — without it the next export treats the "
         "restored tracks as foreign files and duplicates them. Then move the rest back and delete this folder."
         if any(Path(str(src)).name == MANIFEST_FILE for src, _dst, _err in failures)
         else "Move them back by hand (or simply run the export again), then delete this folder."),
        "VEgroove never deletes it automatically; every export lists it in its manifest warnings.",
    ]
    try:
        _atomic_write_bytes(staging / RECOVERY_NOTE, ("\n".join(lines) + "\n").encode("utf-8"))
    except OSError as e:
        log.error("could not write %s: %s", staging / RECOVERY_NOTE, e)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = staging.with_name(f"{RECOVERY_PREFIX}{stamp}-{uuid.uuid4().hex[:6]}")
    try:
        os.replace(staging, target)
    except OSError as e:
        log.error("could not rename %s to %s: %s", staging, target.name, e)
        return staging
    return target


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    seq_index: int                       # 0-based index in sequence.tracks
    st: SequencedTrack
    result: NormalizationResult | None = None
    index: int = 0                       # final 1-based play order
    name: str = ""
    flags: set[str] = field(default_factory=set)
    transition_in: dict | None = None

    @property
    def track(self) -> TrackFeatures:
        return self.st.track


def playlist_title(profile_name: str | None, daypart: str | None) -> str:
    """Playlist / album name, e.g. ``VEgroove - commercial_pop - morning``."""
    return " - ".join(["VEgroove", *(p for p in (profile_name, daypart) if p)])


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _loudness_targets(normalizer: LoudnessNormalizer | None) -> dict:
    cfg = normalizer.cfg if normalizer is not None else LoudnessConfig()
    enc = normalizer.encode if normalizer is not None else EncodeConfig()
    return {
        "target_lufs": cfg.target_lufs,
        "target_true_peak_dbtp": cfg.target_true_peak_dbtp,
        "target_lra": cfg.target_lra,
        "lufs_tolerance": cfg.lufs_tolerance,
        "true_peak_tolerance_db": cfg.true_peak_tolerance_db,
        "codec": "mp3",
        "encoder": enc.codec,
        "bitrate": enc.bitrate,
        "sample_rate": enc.sample_rate,
        "channels": enc.channels,
    }


def _tag_metadata(t: TrackFeatures, index: int, total: int, playlist_name: str, *,
                  source_tags: dict[str, str] | None, tags_copied: bool = True) -> dict[str, str]:
    """Tags written over the copied source tags (``-metadata``).

    ``track`` is the play order ``i/N``; the playlist name always goes into the
    dedicated ``TXXX:VEGROOVE_PLAYLIST`` frame. ``album`` is set to the playlist name only
    when the output would otherwise have none: the source tags are not copied, or the
    source (``source_tags`` from its probe) has no album — or carries the name an
    earlier VEgroove export wrote. Unknown source tags (no probe): album left alone, so
    a useful source album is never replaced.
    """
    md = {"track": f"{index}/{total}", PLAYLIST_TAG: playlist_name}
    if not tags_copied:
        md["album"] = playlist_name
    elif source_tags is not None:
        tags = {str(k).lower(): v for k, v in source_tags.items()}
        album = _clean_text(tags.get("album"))
        if not album or album == _clean_text(tags.get(PLAYLIST_TAG.lower())):
            md["album"] = playlist_name
    for key, value in (("title", t.title), ("artist", t.artist)):
        if _clean_text(value):          # unknown -> keep the source tag, never blank it
            md[key] = _clean_text(value)
    return md


def _features(st: SequencedTrack) -> dict:
    t = st.track
    d = {name: to_jsonable(getattr(t, name)) for name in _FEATURE_FIELDS}
    d["source_duration_s"] = t.duration_s
    d["sequence_position"] = st.position
    d["daypart_fit"] = st.daypart_fit
    d["sequence_notes"] = list(st.notes)
    return d


def _output_duration(r: NormalizationResult | None) -> float | None:
    if r is None:
        return None
    for v in (r.decode_check.decoded_duration_s if r.decode_check else None,
              r.output_measurement.duration_s if r.output_measurement else None,
              r.output_probe.duration_s if r.output_probe else None):
        if v is not None and math.isfinite(v):
            return v
    return None


def _excluded_entry(item: object) -> dict:
    """Copy of a caller-provided exclusion. Every entry carries ``stage``: the caller's
    (``curation`` | ``analysis`` | ...), ``curation`` when missing or blank."""
    d = to_jsonable(item)
    d = dict(d) if isinstance(d, dict) else {"reasons": [str(d)]}
    stage = d.get("stage")
    if not isinstance(stage, str) or not stage.strip():
        d["stage"] = "curation"
    return d


def _not_selected_entry(item: object) -> dict:
    """Copy of a sequencer ``not_selected`` entry in manifest form: ``source_rel`` is
    added from the sequencer's ``rel_path`` (which is kept); a non-dict becomes ``reason``."""
    d = to_jsonable(item)
    d = dict(d) if isinstance(d, dict) else {"reason": str(d)}
    if d.get("source_rel") is None and d.get("rel_path") is not None:
        d["source_rel"] = d["rel_path"]
    return d


def _failed_entry(e: _Entry) -> dict:
    t, r = e.track, e.result
    errors = (r.errors if r else []) or ["no details were reported"]
    return {
        "source_rel": t.rel_path,
        "source_path": str(Path(t.path).absolute()),
        "track_id": t.track_id,
        "title": t.title,
        "artist": t.artist,
        "sequence_position": e.st.position,
        "stage": "normalization",
        "reasons": [f"normalization failed: {msg}" for msg in errors],
        "loudness": r.to_metadata() if r else {},
    }


def _assemble(survivors: list[_Entry], sequence: SequenceResult, normalizer: LoudnessNormalizer | None,
              *, check_loudness: bool) -> tuple[list[dict], list[str]]:
    """Fill ``flags`` / ``transition_in`` of the survivors (final indices already set)
    and return (flagged_transitions, warnings)."""
    flagged: list[dict] = []
    warnings: list[str] = []
    by_position: dict[int, _Entry] = {}
    for e in survivors:
        by_position.setdefault(e.st.position, e)

    for k, e in enumerate(survivors):
        prev = survivors[k - 1] if k else None
        kept = (prev is not None and prev.seq_index == e.seq_index - 1) or (prev is None and e.seq_index == 0)
        e.transition_in = to_jsonable(e.st.transition_in) if kept and e.st.transition_in is not None else None
        r = e.result
        if r is not None and (r.status is NormalizationStatus.REVIEW or r.requires_review):
            e.flags.add(FLAG_LOUDNESS_REVIEW)
            why = "; ".join(r.review_reasons) or "normalization status is 'review'"
            warnings.append(f"track {e.index} ({e.track.display_name}): loudness review — {why}")
        if e.track.requires_review:
            e.flags.add(FLAG_METADATA_REVIEW)

    # sequencer flags, re-indexed; those touching a failed track no longer exist
    for ft in sequence.flagged:
        a, b = by_position.get(ft.from_position), by_position.get(ft.to_position)
        if a is None or b is None:
            log.info("flagged transition %s -> %s dropped: one side is not in the playlist",
                     ft.from_position, ft.to_position)
            continue
        flagged.append({"kind": "sequencing", "from_index": a.index, "to_index": b.index,
                        "from_track": ft.from_track, "to_track": ft.to_track,
                        "reasons": list(ft.reasons), "score": ft.score})
        b.flags.add(FLAG_LISTEN_TRANSITION)

    for prev, nxt in zip(survivors, survivors[1:]):
        if nxt.seq_index != prev.seq_index + 1:
            removed = [str(st.position) for st in sequence.tracks[prev.seq_index + 1:nxt.seq_index]] or ["?"]
            flagged.append({"kind": "unscored", "from_index": prev.index, "to_index": nxt.index,
                            "reasons": [f"track(s) at sequence position(s) {', '.join(removed)} failed "
                                        f"normalization and were removed; the sequencer never scored "
                                        f"this transition"]})
            nxt.flags.add(FLAG_LISTEN_TRANSITION)
        if not check_loudness or normalizer is None:
            continue
        a = prev.result.output_measurement if prev.result else None
        b = nxt.result.output_measurement if nxt.result else None
        if a is None or b is None or a.short_term is None or b.short_term is None:
            warnings.append(f"short-term loudness of track {prev.index} or {nxt.index} is unavailable; "
                            f"adjacent loudness check skipped for {prev.index} -> {nxt.index}")
            continue
        reasons = compare_adjacent_short_term(a.short_term, b.short_term, normalizer.qc)
        if reasons:
            flagged.append({"kind": "loudness", "from_index": prev.index, "to_index": nxt.index,
                            "reasons": reasons})
            nxt.flags.add(FLAG_LISTEN_TRANSITION)

    flagged.sort(key=lambda f: (f["to_index"], f["from_index"], f["kind"]))
    return flagged, warnings


def _manifest_track(e: _Entry) -> ManifestTrack:
    t, r = e.track, e.result
    duration = _output_duration(r)
    return ManifestTrack(
        index=e.index,
        file=f"{TRACKS_DIR}/{e.name}",
        title=t.title or Path(t.path).stem,
        artist=t.artist,
        duration_s=round(duration, 3) if duration is not None else None,
        source_rel=t.rel_path,
        source_path=str(Path(t.path).absolute()),
        track_id=t.track_id,
        loudness=r.to_metadata() if r is not None else {},
        features=_features(e.st),
        transition_in=e.transition_in,
        flags=[f for f in _FLAG_ORDER if f in e.flags],
    )


def _json_default(obj: object) -> object:
    """NumPy scalars (analysis output) -> Python numbers; non-finite -> null."""
    item = getattr(obj, "item", None)
    if callable(item):
        v = item()
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def _render_manifest(manifest: PlaylistManifest) -> bytes:
    return (json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
            + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Normalization stage
# ---------------------------------------------------------------------------

def _failed_result(source: Path, normalizer: LoudnessNormalizer, message: str) -> NormalizationResult:
    return NormalizationResult(
        source=Path(source), output=None, status=NormalizationStatus.FAILED, requires_review=True,
        target_lufs=normalizer.cfg.target_lufs, target_true_peak_dbtp=normalizer.cfg.target_true_peak_dbtp,
        errors=[message])


def _normalize_all(entries: list[_Entry], staging: Path, normalizer: LoudnessNormalizer, *, playlist_name: str,
                   input_measurements: dict[str, LoudnessMeasurement], source_probes: dict[str, AudioProbe],
                   jobs: int, progress: ProgressCallback | None) -> None:
    """Normalize every entry into ``staging`` (parallel, results kept in sequence order).
    Tags carry the provisional ``i/N``; they are corrected after failures."""
    total = len(entries)
    tags_copied = bool(getattr(getattr(normalizer, "encode", None), "copy_tags", True))

    def work(e: _Entry) -> NormalizationResult:
        t = e.track
        dest = staging / f"{e.seq_index + 1:06d}{TRACK_EXT}"
        probe = source_probes.get(t.track_id)
        try:
            return normalizer.normalize_to_mp3(
                Path(t.path), dest,
                input_measurement=input_measurements.get(t.track_id),
                source_probe=probe,
                metadata=_tag_metadata(t, e.seq_index + 1, total, playlist_name,
                                       source_tags=probe.tags if probe is not None else None,
                                       tags_copied=tags_copied))
        except FFmpegNotFoundError:
            raise
        except Exception as exc:   # report, never drop: the track lands in manifest.excluded
            log.exception("unexpected error while normalizing %s", t.path)
            return _failed_result(Path(t.path), normalizer, f"unexpected {type(exc).__name__}: {exc}")

    if progress:
        progress(0, total, f"normalizing {total} track(s)")
    pool = ThreadPoolExecutor(max_workers=max(1, min(jobs, total)), thread_name_prefix="veg-playlist")
    try:
        futures = {pool.submit(work, e): e for e in entries}
        for done, fut in enumerate(as_completed(futures), 1):
            e = futures[fut]
            e.result = fut.result()
            if e.result.status is NormalizationStatus.FAILED:
                log.warning("normalization failed for %s: %s", e.track.path, "; ".join(e.result.errors))
            if progress:
                progress(done, total, f"{e.track.display_name}: {e.result.status.value}")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _renumber_tags(survivors: list[_Entry], total_before: int) -> list[str]:
    """Correct ``track=i/N`` in the staged MP3s after failures changed the play order.
    In-place ID3 edit of an unpublished file; audio frames are not touched."""
    warnings: list[str] = []
    total = len(survivors)
    for e in survivors:
        if (e.index, total) == (e.seq_index + 1, total_before):
            continue
        staged = Path(e.result.output)
        value = f"{e.index}/{total}"
        try:
            ok = _patch_id3_text(staged, "TRCK", value) and _read_id3_text(staged, "TRCK") == value
        except OSError as exc:
            ok = False
            log.warning("could not patch the track tag of %s: %s", staged, exc)
        if not ok:
            warnings.append(f"track {e.index} ({e.track.display_name}): could not update the ID3 track number "
                            f"(file still says {e.seq_index + 1}/{total_before})")
    return warnings


# ---------------------------------------------------------------------------
# Publishing transaction
# ---------------------------------------------------------------------------

def _publish(*, output_dir: Path, tracks_dir: Path, staging: Path, placements: list[tuple[Path, str]],
             owned: list[str], manifest_owned: bool, render: Callable[[list[str]], dict[str, bytes]]) -> list[str]:
    """Replace the previous run's files with the staged ones; all-or-nothing.
    Returns the previous run's files that could not be removed (still owned).

    When the undo of a failed publish is itself incomplete, the staging folder is kept
    for manual recovery (``_keep_for_recovery``) and ``_RollbackIncomplete`` names it."""
    # a foreign playlist file may have appeared while the tracks were being normalized
    foreign = _foreign_output_files(output_dir, manifest_owned=manifest_owned)
    if foreign:
        raise _refuse_foreign_outputs(output_dir, foreign)
    backup = staging / "previous"
    backup.mkdir()
    needed = {name.casefold() for _, name in placements}
    journal = _Journal()
    leftovers: list[str] = []
    temps: list[Path] = []
    try:
        # 1. move everything the previous manifest owned out of the way
        for k, name in enumerate(owned):
            src = tracks_dir / name
            if not os.path.lexists(src):
                continue
            try:
                journal.move(src, backup / f"{k:05d}{TRACK_EXT}")
            except OSError as e:
                if name.casefold() in needed:
                    raise PlaylistExportError(f"cannot replace {src}: {e} — is it open in a player?") from e
                log.warning("could not remove stale file %s: %s", src, e)
                leftovers.append(name)
        for _, name in placements:
            if os.path.lexists(tracks_dir / name):
                raise PlaylistExportError(f"{tracks_dir / name} appeared during the export; not overwriting it")
        for fname in OUTPUT_FILES:
            p = output_dir / fname
            if os.path.lexists(p):
                try:
                    journal.move(p, backup / fname)
                except OSError as e:
                    raise PlaylistExportError(f"cannot replace {p}: {e} — is it open in another program?") from e
        # 2. new tracks under their final names
        for staged, name in placements:
            journal.move(staged, tracks_dir / name)
        # 3. playlist files last, so they only ever describe a complete set of tracks
        for fname, data in render(leftovers).items():
            tmp = _write_temp(output_dir, fname, data)
            temps.append(tmp)
            journal.move(tmp, output_dir / fname)
    except BaseException as exc:
        problems = journal.rollback()
        for tmp in temps:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        if problems:
            kept = _keep_for_recovery(staging, output_dir, exc, problems)

            def moved(p: Path) -> Path:         # paths inside the renamed staging folder
                try:
                    return kept / p.relative_to(staging)
                except ValueError:
                    return p

            details = "; ".join(f"{moved(dst)} -> {moved(src)}: {err}" for src, dst, err in problems)
            raise _RollbackIncomplete(
                f"publishing failed ({exc}) and could not be fully undone: {details}. Files of the "
                f"previous run are kept in {kept / backup.name} (see {kept / RECOVERY_NOTE}); this folder "
                f"is never deleted automatically", recovery_dir=kept) from exc
        raise
    return leftovers


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_playlist(sequence: SequenceResult, output_dir: Path, normalizer: LoudnessNormalizer | None, *,
                    profile_name: str | None = None, excluded: list[dict] | None = None,
                    input_measurements: dict[str, LoudnessMeasurement] | None = None,
                    source_probes: dict[str, AudioProbe] | None = None, jobs: int = 1,
                    dry_run: bool = False, progress: ProgressCallback | None = None,
                    not_selected: list[dict] | None = None, plan: dict | None = None) -> PlaylistExportResult:
    """Export ``sequence`` as individually playable normalized MP3s + playlist files.

    ``input_measurements`` / ``source_probes`` are keyed by ``track_id`` (cached pass-1
    data; missing entries are measured; a probe's tags decide whether ``album`` may be
    set). ``excluded`` are the exclusions to list in the manifest (each keeps its
    ``stage``, default ``curation``); ``not_selected`` are the eligible tracks the
    sequencer did not place in this playlist (``{"rel_path" | "source_rel", "track_id",
    "reason"}``), copied into ``manifest.not_selected`` and the tracklist.
    ``progress(done, total, message)`` is called from the calling thread. Tracks whose
    normalization FAILED are not published; they are listed in ``manifest.excluded``
    (``stage = "normalization"``) and ``manifest.warnings`` and the remaining tracks are
    renumbered contiguously. If no track survives, nothing is published (the previous
    output stays untouched) and the paths in the result are ``None``.

    An empty ``sequence`` is not an error: nothing is written or checked, the paths in
    the result are ``None`` and ``result.manifest`` is an unpublished manifest without
    tracks whose ``warnings`` give the reason (and mention an earlier output left in
    ``output_dir``, which does not describe this run).

    ``dry_run`` writes nothing (no folders either) and returns a manifest preview with
    the planned file names; ``normalizer`` may be ``None`` then.

    ``plan`` (VEmix 3.0 ``render``) is copied into ``manifest.plan`` so ``qc`` can check the
    delivery against the dramaturgy plan; without it the manifest is exactly as in 2.0.

    Raises PlaylistExportError when the export cannot be done safely (a source inside
    ``tracks/``, a foreign ``manifest.json`` / ``playlist.m3u8`` / ``tracklist.txt``, a
    locked file during publishing) and FFmpegNotFoundError when FFmpeg is missing.
    """
    output_dir = Path(output_dir).expanduser().absolute()
    tracks_dir = output_dir / TRACKS_DIR
    items = list(sequence.tracks)
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    if output_dir.exists() and not output_dir.is_dir():
        raise PlaylistExportError(f"output path {output_dir} exists and is not a folder")
    if normalizer is None and not dry_run:
        raise ValueError("a LoudnessNormalizer is required unless dry_run=True")

    title = playlist_title(profile_name, sequence.daypart)
    generated_at = _utc_now()
    excluded_entries = [_excluded_entry(x) for x in excluded or []]
    not_selected_entries = [_not_selected_entry(x) for x in not_selected or []]
    warnings: list[str] = []

    def manifest_for(survivors: list[_Entry], extra_warnings: list[str], excluded_all: list[dict],
                     flagged: list[dict]) -> PlaylistManifest:
        return PlaylistManifest(
            generator=GENERATOR, generated_at=generated_at, mode="playlist", profile=profile_name,
            daypart=sequence.daypart, loudness_targets=_loudness_targets(normalizer),
            tracks=[_manifest_track(e) for e in survivors], excluded=excluded_all,
            flagged_transitions=flagged, warnings=[*extra_warnings, *warnings],
            not_selected=list(not_selected_entries), plan=plan)

    if not items:
        where = f" for daypart '{sequence.daypart}'" if sequence.daypart else ""
        reason = f"the sequence{where} contains no tracks (no eligible track was placed): nothing was exported"
        notes = (["dry run: nothing was written"] if dry_run else []) + [reason]
        if os.path.lexists(output_dir / MANIFEST_FILE):
            notes.append(f"{output_dir} still holds the output of an earlier run; it was left untouched "
                         f"and does NOT describe this run")
        log.warning("playlist export to %s skipped: %s", output_dir, reason)
        return PlaylistExportResult(output_dir=output_dir, manifest_path=None, m3u8_path=None,
                                    tracklist_path=None, manifest=manifest_for([], notes, excluded_entries, []),
                                    results=[], dry_run=dry_run)

    _check_sources(items, output_dir, tracks_dir)
    owned, owned_warnings = _previous_owned_files(output_dir / MANIFEST_FILE)
    warnings += owned_warnings
    # _previous_owned_files refused a foreign manifest; an existing one owns the folder
    manifest_owned = (output_dir / MANIFEST_FILE).exists()
    foreign_outputs = _foreign_output_files(output_dir, manifest_owned=manifest_owned)
    if foreign_outputs:
        raise _refuse_foreign_outputs(output_dir, foreign_outputs)
    warnings += _recovery_warnings(tracks_dir)
    entries = [_Entry(seq_index=i, st=st) for i, st in enumerate(items)]
    for path, count in Counter(_norm(st.track.path) for st in items).items():
        if count > 1:
            warnings.append(f"source {path} appears {count} times in the sequence")

    if dry_run:
        width = index_width(len(entries))
        names = _plan_names([(e.seq_index + 1, e.track) for e in entries], width, _foreign_names(tracks_dir, owned))
        for e, name in zip(entries, names):
            e.index, e.name = e.seq_index + 1, name
        notes = ["dry run: nothing was written"]
        notes += [f"source not found: {e.track.path} (normalization would fail)"
                  for e in entries if not Path(e.track.path).is_file()]
        stale = [n for n in owned if n not in set(names) and os.path.lexists(tracks_dir / n)]
        if stale:
            notes.append(f"dry run: {len(stale)} file(s) of the previous run would be removed: "
                         + ", ".join(f"{TRACKS_DIR}/{n}" for n in stale))
        flagged, assemble_warnings = _assemble(entries, sequence, normalizer, check_loudness=False)
        preview = manifest_for(entries, notes + assemble_warnings, excluded_entries, flagged)
        return PlaylistExportResult(output_dir=output_dir, manifest_path=None, m3u8_path=None,
                                    tracklist_path=None, manifest=preview, results=[], dry_run=True)

    assert normalizer is not None
    created_dirs = [p for p in (output_dir, tracks_dir) if not p.exists()]
    tracks_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_staging(tracks_dir)
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=tracks_dir))
    keep_staging = published = False
    try:
        _normalize_all(entries, staging, normalizer, playlist_name=title,
                       input_measurements=dict(input_measurements or {}),
                       source_probes=dict(source_probes or {}), jobs=jobs, progress=progress)
        survivors: list[_Entry] = []
        failed: list[_Entry] = []
        for e in entries:
            r = e.result
            assert r is not None
            if r.status is not NormalizationStatus.FAILED and (r.output is None or not Path(r.output).is_file()):
                r.errors.append("the normalizer reported success but produced no output file")
                r.status, r.requires_review, r.output = NormalizationStatus.FAILED, True, None
            (failed if r.status is NormalizationStatus.FAILED else survivors).append(e)

        failure_warnings = []
        for e in failed:
            errs = e.result.errors or ["no details were reported"]
            more = f" (+{len(errs) - 1} more)" if len(errs) > 1 else ""
            failure_warnings.append(f"track at sequence position {e.st.position} ({e.track.display_name}) "
                                    f"excluded from the playlist — normalization failed: {errs[0]}{more}")
        excluded_all = excluded_entries + [_failed_entry(e) for e in failed]

        if not survivors:
            for e in failed:
                e.result.output = None
            manifest = manifest_for([], failure_warnings + [
                "no track passed normalization: nothing was published; the previous output (if any) "
                "is left untouched"], excluded_all, [])
            log.error("playlist export: all %d track(s) failed normalization; nothing published", len(entries))
            return PlaylistExportResult(output_dir=output_dir, manifest_path=None, m3u8_path=None,
                                        tracklist_path=None, manifest=manifest,
                                        results=[e.result for e in entries])

        width = index_width(len(survivors))
        names = _plan_names([(k, e.track) for k, e in enumerate(survivors, 1)], width,
                            _foreign_names(tracks_dir, owned))
        for k, (e, name) in enumerate(zip(survivors, names), 1):
            e.index, e.name = k, name
        tag_warnings = _renumber_tags(survivors, len(entries)) if failed else []
        flagged, assemble_warnings = _assemble(survivors, sequence, normalizer, check_loudness=True)
        manifest = manifest_for(survivors, failure_warnings + tag_warnings + assemble_warnings,
                                excluded_all, flagged)
        _render_manifest(manifest)      # fail on unserializable data before anything is moved

        def render(leftovers: list[str]) -> dict[str, bytes]:
            manifest.leftover_files = [f"{TRACKS_DIR}/{n}" for n in leftovers]
            manifest.warnings += [f"could not remove {TRACKS_DIR}/{n} from the previous run (in use?); "
                                  f"it will be removed by the next export" for n in leftovers]
            # manifest first: it is the ownership record the next run cleans up from
            return {
                MANIFEST_FILE: _render_manifest(manifest),
                PLAYLIST_FILE: render_m3u8(manifest.tracks, title).encode("utf-8"),
                TRACKLIST_FILE: render_tracklist(manifest, title=title).encode("utf-8"),
            }

        try:
            _publish(output_dir=output_dir, tracks_dir=tracks_dir, staging=staging,
                     placements=[(Path(e.result.output), e.name) for e in survivors], owned=owned,
                     manifest_owned=manifest_owned, render=render)
        except _RollbackIncomplete as exc:
            keep_staging, staging = True, exc.recovery_dir
            raise
        published = True
        for e in survivors:
            e.result.output = tracks_dir / e.name
        log.info("playlist published: %d track(s), %d failed, to %s", len(survivors), len(failed), output_dir)
        return PlaylistExportResult(
            output_dir=output_dir, manifest_path=output_dir / MANIFEST_FILE, m3u8_path=output_dir / PLAYLIST_FILE,
            tracklist_path=output_dir / TRACKLIST_FILE, manifest=manifest, results=[e.result for e in entries])
    finally:
        if keep_staging:
            log.error("staging folder kept for manual recovery: %s", staging)
        else:
            shutil.rmtree(staging, ignore_errors=True)
            if staging.exists():
                log.warning("could not fully remove the staging folder %s; the next export removes it", staging)
        if not published:
            for p in reversed(created_dirs):    # leave no empty folders behind a failed export
                try:
                    p.rmdir()
                except OSError:
                    pass


__all__ = [
    "MANIFEST_FILE", "PLAYLIST_FILE", "TRACKLIST_FILE", "TRACKS_DIR", "PlaylistExportError",
    "export_playlist", "index_width", "playlist_title", "read_m3u8", "render_m3u8", "render_tracklist",
    "safe_filename", "sanitize_component", "write_m3u8",
]
