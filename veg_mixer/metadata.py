"""Curator (editorial) metadata: export to CSV/JSON → human edit → import → apply.

Spec §5: manual editorial metadata takes precedence over automatic estimates and
unknown values stay unknown. The round trip is designed so that

    export (untouched) -> load -> apply   ==   no change

Editable columns are pre-filled ONLY with values a curator already set (the
``existing`` overrides, or fields whose ``feature_sources`` say ``manual``).
Automatic estimates go to read-only ``auto_*`` reference columns, which the importer
ignores — so an auto estimate can never be promoted to "manual" by accident.

CSV dialect: UTF-8 with BOM (Excel), ``,`` delimiter by default. The importer also
accepts what a Russian-locale Excel writes back: ``;`` delimiter, decimal comma,
``да/нет`` / ``ИСТИНА/ЛОЖЬ`` booleans, and Windows-1251 when the file is not valid
UTF-8 (logged as a warning). Empty cell = "do not override". Unknown columns are an
error (typo protection), except ``auto_*`` reference columns and curator scratch
columns whose name starts with ``_`` (e.g. ``_comment``), which are ignored.
"""

from __future__ import annotations

import copy
import csv
import difflib
import io
import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .models import EDITABLE_FIELDS, TrackFeatures
from .textnorm import normalize_rel_path, rel_path_key, split_values

log = logging.getLogger(__name__)

# Read-only reference columns written after the editable ones (ignored on import).
REFERENCE_COLUMNS = ("auto_bpm", "auto_camelot", "auto_energy", "auto_genre", "auto_language",
                     "script_hint", "sources")
EXPORT_COLUMNS = ("rel_path", *EDITABLE_FIELDS, *REFERENCE_COLUMNS)
METADATA_SUFFIXES = (".csv", ".json")
CSV_DELIMITERS = (",", ";", "\t")

MANUAL = "manual"
# A manually set Camelot key is curator-verified: its key_confidence becomes 1.0 unless
# the curator also gave key_confidence explicitly. The source label says where it came from.
DERIVED_KEY_CONFIDENCE_SOURCE = "derived:manual-camelot"

# field -> (low, high, low_exclusive)
FLOAT_RANGES: dict[str, tuple[float, float, bool]] = {
    "bpm": (0.0, 400.0, True),
    "key_confidence": (0.0, 1.0, False),
    "energy": (0.0, 1.0, False),
    "danceability": (0.0, 1.0, False),
    "vocal_presence": (0.0, 1.0, False),
    "familiarity": (0.0, 1.0, False),
}
BOOL_FIELDS = ("approved", "excluded", "requires_review")
TEXT_FIELDS = ("artist", "title", "genre", "mood", "notes")
# casefolded; "истина/ложь" is what a Russian-locale Excel writes for TRUE/FALSE
TRUE_WORDS = frozenset({"true", "yes", "y", "1", "да", "истина"})
FALSE_WORDS = frozenset({"false", "no", "n", "0", "нет", "ложь"})

CAMELOT_RE = re.compile(r"^0?(1[0-2]|[1-9])([AB])$", re.IGNORECASE)
LANGUAGE_RE = re.compile(r"^[a-z]{2}$")
TAG_SPLIT_RE = re.compile(r"[;,]")


class MetadataError(ValueError):
    """Invalid curator metadata. The message names the file, line and column."""

    def __init__(self, message: str, *, path: Path | str | None = None, line: int | None = None,
                 column: str | None = None, entry: str | None = None) -> None:
        where = [str(p) for p in (path,) if p]
        if line:
            where.append(f"line {line}")
        if entry:
            where.append(entry)
        if column:
            where.append(f"column '{column}'")
        super().__init__(f"{', '.join(where)}: {message}" if where else message)
        self.path = Path(path) if path else None
        self.line = line
        self.column = column
        self.reason = message


# ---------------------------------------------------------------------------
# Value validation (shared by CSV, JSON and programmatic overrides)
# ---------------------------------------------------------------------------

# ``normalize_rel_path`` (library-relative POSIX path key, case preserved, NFC) is the
# shared textnorm implementation, re-exported here for existing callers; matching uses
# ``rel_path_key`` (case-insensitive) — the same keys QC uses.


def coerce_value(field: str, value: Any) -> Any:
    """Validate and convert one editable value — a CSV string or a typed JSON value.

    Returns ``None`` for "no override" (None / empty / empty tag list). Raises
    ``ValueError`` with a human-readable message on invalid input.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    if field in FLOAT_RANGES:
        return _coerce_float(field, value)
    if field in BOOL_FIELDS:
        return _coerce_bool(value)
    if field == "camelot":
        if not isinstance(value, str) or not CAMELOT_RE.match(value):
            raise ValueError(f"expected a Camelot key 1A..12B (e.g. '8A'), got {value!r}")
        m = CAMELOT_RE.match(value)
        return f"{int(m.group(1))}{m.group(2).upper()}"
    if field == "language":
        return _coerce_language(value)
    if field == "vibe_tags":
        return _coerce_tags(value)
    if field in TEXT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(f"expected text, got {type(value).__name__}")
        return str(value).strip() or None
    raise ValueError(f"'{field}' is not an editable metadata field")


def _coerce_language(value: Any) -> str:
    """ISO 639-1 code, or several for a multi-language song ('en;ru', 'en, ru' -> 'en;ru').
    Only two-letter codes are accepted from a curator (typo protection: 'eng' is refused)."""
    codes = split_values(value) if isinstance(value, str) else []
    if not codes or not all(LANGUAGE_RE.match(c.lower()) for c in codes):
        raise ValueError(f"expected an ISO 639-1 two-letter language code such as 'en' "
                         f"(several: 'en;fr'), got {value!r}")
    out: list[str] = []
    for c in codes:
        if c.lower() not in out:
            out.append(c.lower())
    return ";".join(out)


def _coerce_float(field: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"expected a number, got the boolean {value}")
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        s = value.replace(" ", "").replace(" ", "")
        if s.count(",") == 1 and "." not in s:      # decimal comma (Russian Excel)
            s = s.replace(",", ".")
        try:
            f = float(s)
        except ValueError:
            raise ValueError(f"expected a number, got {value!r}") from None
    else:
        raise ValueError(f"expected a number, got {type(value).__name__}")
    if not math.isfinite(f):
        raise ValueError(f"expected a finite number, got {value!r}")
    lo, hi, lo_exclusive = FLOAT_RANGES[field]
    if (f <= lo if lo_exclusive else f < lo) or f > hi:
        bound = f"> {lo:g} and <= {hi:g}" if lo_exclusive else f"within {lo:g}..{hi:g}"
        raise ValueError(f"{f:g} is out of range: {field} must be {bound}")
    return f


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().casefold()
        if word in TRUE_WORDS:
            return True
        if word in FALSE_WORDS:
            return False
    raise ValueError(f"expected a boolean (true/false, yes/no, 1/0, да/нет), got {value!r}")


def _coerce_tags(value: Any) -> list[str] | None:
    if isinstance(value, str):
        parts: list[Any] = TAG_SPLIT_RE.split(value)
    elif isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (str, int, float)):
                raise ValueError(f"vibe_tags must be text, got {item!r}")
            parts.extend(TAG_SPLIT_RE.split(str(item)))
    else:
        raise ValueError(f"expected tags separated by ';', got {type(value).__name__}")
    tags: list[str] = []
    for p in parts:
        tag = " ".join(str(p).split()).lower()
        if tag and tag not in tags:
            tags.append(tag)
    return tags or None


# ---------------------------------------------------------------------------
# Matching override keys to tracks
# ---------------------------------------------------------------------------

def match_overrides(tracks: list[TrackFeatures], keys: Iterable[str]) -> tuple[dict[int, str], list[str]]:
    """Map override keys to track indices.

    1. rel_path, case-insensitive (POSIX separators);
    2. otherwise the file name, if it is unique in the library.
    Returns ``({track_index: key}, warnings)`` — unmatched, ambiguous or conflicting
    keys produce warnings and are not applied.
    """
    by_rel: dict[str, list[int]] = {}
    by_name: dict[str, list[int]] = {}
    for i, t in enumerate(tracks):
        rel = rel_path_key(t.rel_path or t.path.name)
        by_rel.setdefault(rel, []).append(i)
        by_name.setdefault(PurePosixPath(rel).name, []).append(i)

    assigned: dict[int, str] = {}
    warnings: list[str] = []

    def claim(index: int, key: str) -> None:
        if index in assigned:
            warnings.append(f"metadata: '{key}' and '{assigned[index]}' both match "
                            f"'{tracks[index].rel_path}'; '{key}' ignored")
        else:
            assigned[index] = key

    pending: list[str] = []
    for key in keys:
        hits = by_rel.get(rel_path_key(key), [])
        if len(hits) == 1:
            claim(hits[0], key)
        elif len(hits) > 1:
            warnings.append(f"metadata: '{key}' is ambiguous — it matches {len(hits)} tracks whose paths "
                            f"differ only in letter case; not applied")
        else:
            pending.append(key)

    for key in pending:            # file-name fallback only after all exact matches
        rel = normalize_rel_path(key)
        name = PurePosixPath(rel).name if rel else ""
        hits = by_name.get(name.casefold(), []) if name else []
        if len(hits) == 1:
            if "/" in rel:     # a stale folder in the curator file is worth knowing about
                warnings.append(f"metadata: '{key}' matched '{tracks[hits[0]].rel_path}' by file name only")
            claim(hits[0], key)
        elif len(hits) > 1:
            listed = ", ".join(tracks[i].rel_path for i in hits[:5])
            warnings.append(f"metadata: '{key}' is ambiguous — the file name matches {len(hits)} tracks "
                            f"({listed}); use the full rel_path. Not applied")
        else:
            warnings.append(f"metadata: '{key}' matches no track in the library; not applied")
    return assigned, warnings


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_metadata(tracks: list[TrackFeatures],
                   overrides: dict[str, dict]) -> tuple[list[TrackFeatures], list[str]]:
    """Return copies of ``tracks`` with curator values applied (inputs are not mutated).

    Manual values win over automatic ones and are marked ``feature_sources[f] = "manual"``.
    Values are validated again here (overrides may come from code, not only from a file).
    """
    out = [copy.deepcopy(t) for t in tracks]
    matches, warnings = match_overrides(out, list(overrides))
    for index, key in sorted(matches.items()):
        track = out[index]
        values = _validated_values(overrides[key], entry=f"override '{key}'")
        for field, value in values.items():
            setattr(track, field, list(value) if isinstance(value, list) else value)
            track.feature_sources[field] = MANUAL
        if "energy" in values:
            track.energy_confidence = 1.0       # same as TrackFeatures.confidence() for manual values
        if "camelot" in values and "key_confidence" not in values:
            track.key_confidence = 1.0
            track.feature_sources["key_confidence"] = DERIVED_KEY_CONFIDENCE_SOURCE
        if track.approved and track.excluded and {"approved", "excluded"} & set(values):
            warnings.append(f"metadata: '{key}' is marked both approved and excluded; "
                            f"exclusion takes precedence")
    for w in warnings:
        log.warning(w)
    return out, warnings


def _validated_values(raw: dict, *, path: Path | None = None, line: int | None = None,
                      entry: str | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise MetadataError(f"expected a mapping of fields, got {type(raw).__name__}",
                            path=path, line=line, entry=entry)
    values: dict[str, Any] = {}
    for field, value in raw.items():
        if field not in EDITABLE_FIELDS:
            if _is_ignored_column(field):
                continue
            raise MetadataError(f"unknown field '{field}'{_suggest(field)}", path=path, line=line,
                                entry=entry, column=str(field))
        try:
            v = coerce_value(field, value)
        except ValueError as e:
            raise MetadataError(str(e), path=path, line=line, entry=entry, column=field) from None
        if v is not None:
            values[field] = v
    return values


def _is_ignored_column(name: Any) -> bool:
    return isinstance(name, str) and (name == "rel_path" or name in REFERENCE_COLUMNS
                                      or name.startswith(("auto_", "_")))


def _suggest(name: Any) -> str:
    close = difflib.get_close_matches(str(name), EXPORT_COLUMNS, n=1)
    return f" (did you mean '{close[0]}'?)" if close else ""


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_metadata(path: Path) -> dict[str, dict]:
    """Read a curator file (.csv or .json) into ``{rel_path: {field: value}}``.

    Only rows that override at least one field are returned; values are typed
    (floats, bools, ``list[str]`` tags). Errors raise ``MetadataError`` naming the
    file, line and column.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in METADATA_SUFFIXES:
        raise MetadataError(f"unsupported metadata format '{suffix}' (use .csv or .json)", path=path)
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise MetadataError(f"cannot read file: {e}", path=path) from e
    text = _decode(raw, path, allow_cp1251=(suffix == ".csv"))
    rows = _csv_rows(text, path) if suffix == ".csv" else _json_rows(text, path)

    result: dict[str, dict] = {}
    seen: dict[str, int | str | None] = {}
    for line, entry, raw_row in rows:
        rel_raw = raw_row.get("rel_path")
        if rel_raw is not None and not isinstance(rel_raw, str):
            raise MetadataError(f"rel_path must be text, got {rel_raw!r}", path=path, line=line,
                                entry=entry, column="rel_path")
        key = normalize_rel_path(rel_raw)
        values = _validated_values(raw_row, path=path, line=line, entry=entry)
        if not key:
            if values:
                raise MetadataError("row has values but an empty rel_path", path=path, line=line,
                                    entry=entry, column="rel_path")
            continue
        if not values:
            continue          # nothing to override (e.g. an untouched exported row)
        folded = key.casefold()
        if folded in seen:
            first = seen[folded]
            raise MetadataError(f"duplicate rel_path '{key}' (first defined at "
                                f"{'line ' + str(first) if isinstance(first, int) else first})",
                                path=path, line=line, entry=entry, column="rel_path")
        seen[folded] = line if line is not None else entry
        result[key] = values
    log.info("metadata: %d curated row(s) loaded from %s", len(result), path)
    return result


def _decode(raw: bytes, path: Path, *, allow_cp1251: bool) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):      # Excel "Unicode text"
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError as e:
            raise MetadataError(f"invalid UTF-16 text: {e.reason}", path=path) from None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        if not allow_cp1251:
            raise MetadataError(f"file is not valid UTF-8 (byte offset {e.start})", path=path) from None
        try:
            text = raw.decode("cp1251")
        except UnicodeDecodeError:
            raise MetadataError(f"file is neither UTF-8 nor Windows-1251 (byte offset {e.start}); "
                                f"save it as 'CSV UTF-8'", path=path) from None
        log.warning("%s is not valid UTF-8; decoded as Windows-1251 (legacy Excel CSV). "
                    "Save as 'CSV UTF-8' to avoid ambiguity.", path)
        return text


RawRow = tuple[int | None, str | None, dict[str, Any]]


def _csv_rows(text: str, path: Path) -> list[RawRow]:
    lines = text.splitlines(keepends=True)
    offset = 0
    # skip leading blank lines; honour an Excel "sep=;" hint line
    while offset < len(lines) and not lines[offset].strip():
        offset += 1
    if offset >= len(lines):
        raise MetadataError("file is empty (no header row)", path=path)
    first = lines[offset].strip()
    if first.lower().startswith("sep=") and len(first) == 5:
        delimiter = first[4]
        offset += 1
    else:
        counts = {d: first.count(d) for d in CSV_DELIMITERS}
        delimiter = max(CSV_DELIMITERS, key=lambda d: counts[d]) if any(counts.values()) else ","

    reader = csv.reader(io.StringIO("".join(lines[offset:])), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        raise MetadataError("file is empty (no header row)", path=path) from None
    except csv.Error as e:
        raise MetadataError(f"malformed CSV header: {e}", path=path, line=offset + 1) from None
    header_line = offset + reader.line_num
    columns = [c.strip().lstrip("﻿").lower() for c in header]
    for c in columns:
        if c and c not in EXPORT_COLUMNS and not _is_ignored_column(c):
            raise MetadataError(f"unknown column '{c}'{_suggest(c)}", path=path, line=header_line, column=c)
    dupes = sorted({c for c in columns if c and columns.count(c) > 1})
    if dupes:
        raise MetadataError(f"duplicate column(s): {', '.join(dupes)}", path=path, line=header_line,
                            column=dupes[0])
    if "rel_path" not in columns:
        raise MetadataError(f"missing required column 'rel_path' (delimiter detected: {delimiter!r}; "
                            f"columns found: {', '.join(c for c in columns if c) or 'none'})",
                            path=path, line=header_line)

    rows: list[RawRow] = []
    iterator = iter(reader)
    while True:
        line = offset + reader.line_num + 1        # first physical line of the next record
        try:
            cells = next(iterator)
        except StopIteration:
            break
        except csv.Error as e:
            raise MetadataError(f"malformed CSV: {e}", path=path, line=offset + reader.line_num) from None
        if not any(c.strip() for c in cells):
            continue
        extra = cells[len(columns):]
        if any(c.strip() for c in extra):
            raise MetadataError(f"row has {len(cells)} cells but the header has {len(columns)} "
                                f"(check the delimiter / quoting)", path=path, line=line)
        row: dict[str, Any] = {}
        for i, col in enumerate(columns):
            cell = cells[i] if i < len(cells) else ""
            if not col:
                if cell.strip():
                    raise MetadataError(f"value {cell.strip()!r} in a column without a header",
                                        path=path, line=line)
                continue
            if col == "rel_path" or col in EDITABLE_FIELDS:
                row[col] = cell
        rows.append((line, None, row))
    return rows


class _DuplicateKey(ValueError):
    pass


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey(key)
        out[key] = value
    return out


def _json_rows(text: str, path: Path) -> list[RawRow]:
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as e:
        raise MetadataError(f"invalid JSON: {e.msg} (character {e.colno})", path=path, line=e.lineno) from None
    except _DuplicateKey as e:      # json would silently keep the last value
        raise MetadataError(f"invalid JSON: duplicate key '{e}' in an object", path=path, column=str(e)) from None
    rows: list[RawRow] = []
    if isinstance(data, list):
        item_lines = _json_array_item_lines(text)
        for i, item in enumerate(data):
            line = item_lines[i] if i < len(item_lines) else None
            entry = f"entry #{i + 1}"
            if not isinstance(item, dict):
                raise MetadataError(f"expected an object, got {type(item).__name__}", path=path,
                                    line=line, entry=entry)
            rows.append((line, entry, item))
    elif isinstance(data, dict):        # {rel_path: {field: value}}
        for key, item in data.items():
            entry = f"entry '{key}'"
            if not isinstance(item, dict):
                raise MetadataError(f"expected an object, got {type(item).__name__}", path=path, entry=entry)
            inner = item.get("rel_path")
            if inner not in (None, "") and rel_path_key(str(inner)) != rel_path_key(key):
                raise MetadataError(f"rel_path {inner!r} contradicts the entry key", path=path,
                                    entry=entry, column="rel_path")
            rows.append((None, entry, {**item, "rel_path": key}))
    else:
        raise MetadataError("expected a JSON list of objects (or an object keyed by rel_path)", path=path)
    return rows


def _json_array_item_lines(text: str) -> list[int]:
    """Line number where each top-level array element starts (best effort; the
    document has already been parsed successfully)."""
    decoder = json.JSONDecoder()
    ws = re.compile(r"[ \t\r\n]*")
    idx = ws.match(text, 0).end()
    if idx >= len(text) or text[idx] != "[":
        return []
    idx = ws.match(text, idx + 1).end()
    lines: list[int] = []
    try:
        while idx < len(text) and text[idx] != "]":
            _, end = decoder.raw_decode(text, idx)
            lines.append(text.count("\n", 0, idx) + 1)
            idx = ws.match(text, end).end()
            if idx < len(text) and text[idx] == ",":
                idx = ws.match(text, idx + 1).end()
    except json.JSONDecodeError:
        pass
    return lines


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_metadata(tracks: list[TrackFeatures], path: Path, *, existing: dict | None = None,
                    delimiter: str = ",", decimal_comma: bool | None = None,
                    keep_unmatched: bool = True) -> None:
    """Write a curator sheet (.csv or .json) for ``tracks`` (atomic write).

    Editable columns hold only manual values: ``existing`` overrides (e.g. a previously
    loaded curator file) or track fields whose source is ``manual``. Automatic values
    appear only in the read-only ``auto_*`` / ``sources`` columns. Rows of ``existing``
    that match no track are kept (``keep_unmatched``) so curator work is never lost.

    CSV options: ``delimiter`` (',' default; ';' suits Russian-locale Excel) and
    ``decimal_comma`` (default: on when the delimiter is ';').
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in METADATA_SUFFIXES:
        raise MetadataError(f"refusing to write metadata to '{path.name}': use a .csv or .json file", path=path)
    if delimiter not in CSV_DELIMITERS:
        raise ValueError(f"unsupported CSV delimiter {delimiter!r}; use one of {CSV_DELIMITERS}")
    if decimal_comma is None:
        decimal_comma = delimiter == ";"
    if decimal_comma and delimiter == ",":
        raise ValueError("decimal_comma requires a ';' or tab delimiter")

    existing = existing or {}
    matches, warnings = match_overrides(tracks, list(existing))
    for w in warnings:
        log.warning("export: %s", w)

    rows: list[dict[str, Any]] = []
    for i, t in enumerate(tracks):
        manual = _validated_values(existing[matches[i]], entry=f"existing '{matches[i]}'") if i in matches else {}
        row: dict[str, Any] = {"rel_path": normalize_rel_path(t.rel_path or t.path.name)}
        for field in EDITABLE_FIELDS:
            value = manual.get(field)
            if value is None and t.feature_sources.get(field) == MANUAL:
                value = getattr(t, field)
                if value in ("", []):
                    value = None
            row[field] = list(value) if isinstance(value, list) else value
        row["auto_bpm"] = _auto(t, "bpm", 2)
        row["auto_camelot"] = _auto(t, "camelot")
        row["auto_energy"] = _auto(t, "energy", 3)
        row["auto_genre"] = _auto(t, "genre")
        row["auto_language"] = _auto(t, "language")
        row["script_hint"] = t.script_hint
        row["sources"] = ";".join(f"{k}={v}" for k, v in sorted(t.feature_sources.items())) or None
        rows.append(row)

    if keep_unmatched:
        used = set(matches.values())
        written = {rel_path_key(row["rel_path"]) for row in rows}
        for key, values in existing.items():
            if key in used:
                continue
            if rel_path_key(key) in written:
                # same path as a row already written (differs only in case): writing it would
                # produce a file that load_metadata rejects as a duplicate
                log.warning("export: dropping curator row '%s': it conflicts with row '%s'", key,
                            normalize_rel_path(key))
                continue
            written.add(rel_path_key(key))
            log.warning("export: keeping curator row '%s' although it matches no track", key)
            row = {c: None for c in EXPORT_COLUMNS}
            row["rel_path"] = normalize_rel_path(key)
            row.update(_validated_values(values, entry=f"existing '{key}'"))
            rows.append(row)

    if suffix == ".csv":
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=delimiter, lineterminator="\r\n")
        writer.writerow(EXPORT_COLUMNS)
        for row in rows:
            writer.writerow([_cell(row[c], decimal_comma) for c in EXPORT_COLUMNS])
        _atomic_write_text(path, buf.getvalue(), encoding="utf-8-sig")   # BOM: Excel detects UTF-8
    else:
        payload = [{c: row[c] for c in EXPORT_COLUMNS} for row in rows]
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("metadata: exported %d row(s) to %s", len(rows), path)


def _auto(t: TrackFeatures, field: str, digits: int | None = None) -> Any:
    """Current non-manual value of ``field`` (for reference only)."""
    if t.feature_sources.get(field) == MANUAL:
        return None
    value = getattr(t, field)
    if value in (None, ""):
        return None
    if digits is not None and isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    return value


def _format_float(value: float, decimal_comma: bool) -> str:
    text = repr(float(value))             # repr round-trips exactly
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace(".", ",") if decimal_comma else text


def _cell(value: Any, decimal_comma: bool) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _format_float(value, decimal_comma)
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return str(value)


def _atomic_write_text(path: Path, text: str, *, encoding: str) -> None:
    """Temp file in the destination folder + os.replace (never a half-written file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            log.warning("could not remove temporary file %s", tmp)
        raise
