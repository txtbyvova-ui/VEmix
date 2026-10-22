"""Persistent cache of external music-catalogue data (VEmix 3.0, spec §3).

Online research is slow by design (MusicBrainz allows about one request per second), so
every useful answer is cached and re-used: a second run over the same library makes no
requests at all, and the ``cache`` research mode works fully offline.

File format (UTF-8 JSON, ``indent=2``)::

    {
      "schema": "vegroove.research_cache",
      "schema_version": 1,
      "entries": {
        "mb:search:michael jackson|thriller|358": {
          "source": "musicbrainz",
          "stored_at": "2026-10-22T10:00:00+00:00",
          "expires_at": "2027-01-20T10:00:00+00:00",
          "value": {...}
        }
      }
    }

Rules:

* Keys describe a RECORDING IDENTITY or a QUERY (normalised artist | title | rounded
  duration, an ISRC or an MBID), never an audio-file fingerprint: two copies of a song
  share the answer, and re-encoding a file does not trigger new requests.
* Every entry has a per-source time-to-live (``DEFAULT_TTL_S``); an expired entry reads
  as missing (``get`` returns None). ``get_entry(..., allow_expired=True)`` still returns
  an expired MusicBrainz / Wikidata entry, flagged ``expired`` — cache-only runs use it and
  mark the record stale, so a plan stays reproducible after the TTL. Such entries are kept
  in the file until a fresh answer replaces them (or ``purge_expired`` drops them).
* Discogs data is never kept longer than 6 hours — its API terms forbid showing data
  older than that — whatever ``ttl_s`` says: an expired Discogs entry is never returned
  (not even with ``allow_expired``) and is purged from the file on load / save.
* Only a file that is demonstrably a damaged research cache (our schema with a bad
  version / entries, a truncated document that starts like ours, an empty file) is moved
  aside to ``<name>.corrupt-<n>`` (never deleted). Anything else at the path — another
  program's JSON, the analysis cache, a curator sheet — is never renamed or overwritten:
  it is left untouched and saving is disabled (warning). Non-``.json`` paths are refused.
* ``read_only`` (dry run / cache-only inspection): nothing is ever written or renamed.
* ``save()`` is atomic (``fsutil.atomic_write_json``) and a no-op when nothing changed.
  Values must be JSON-native and UTF-8 encodable (checked on ``put``; NaN and lone
  surrogates are rejected, and such entries in a loaded file are dropped). Thread-safe.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import AUDIO_EXTENSIONS, LOSSLESS_EXTENSIONS
from ..fsutil import atomic_write_json

log = logging.getLogger(__name__)

RESEARCH_CACHE_SCHEMA = "vegroove.research_cache"
RESEARCH_CACHE_SCHEMA_VERSION = 1
_OWN_CACHE_PREFIX_RE = re.compile(r'\A\s*\{\s*"schema"\s*:\s*"' + re.escape(RESEARCH_CACHE_SCHEMA) + '"')

_DAY = 24 * 3600.0
# Time-to-live of POSITIVE answers per source. Catalogue facts change rarely; a quarter
# keeps cache-only runs useful offline while still picking up corrections.
DEFAULT_TTL_S: dict[str, float] = {
    "musicbrainz": 90 * _DAY,
    "wikidata": 90 * _DAY,
    "discogs": 6 * 3600.0,
}
FALLBACK_TTL_S = 30 * _DAY
# "Not found" answers expire sooner: the catalogue may gain the recording.
NEGATIVE_TTL_S = 14 * _DAY
# Discogs API Terms of Use: data older than 6 hours must not be shown / kept.
DISCOGS_MAX_AGE_S = 6 * 3600.0
# Expired MusicBrainz / Wikidata answers stay as a stale fallback for cache-only runs, but not
# forever: an answer nobody refreshed for a year past its TTL (renamed / deleted tracks, old
# "not found" answers) is dropped on load / save, so the file does not grow without bound.
STALE_RETENTION_S = 365 * _DAY
_MAX_DETAILED_WARNINGS = 10


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    """ISO 8601 UTC timestamp with seconds precision (the project's convention)."""
    return _aware(dt).astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(text: Any) -> datetime | None:
    """Aware datetime from an ISO 8601 string (naive = UTC); None when unparsable."""
    if not isinstance(text, str) or not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _aware(dt)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class ResearchCache:
    """Key/value cache of research data with per-source expiry."""

    def __init__(self, path: Path, *, read_only: bool = False,
                 clock: Callable[[], datetime] | None = None) -> None:
        """``clock`` returns the current (aware) time; ``get``/``put`` also accept an
        explicit ``now``. Raises ValueError for a path that cannot be a research cache
        (an audio file, a non-``.json`` file)."""
        self.path = Path(path)
        _check_cache_path(self.path)
        self.read_only = bool(read_only)
        self._clock = clock or utc_now
        self.stats: dict[str, int] = {"hits": 0, "misses": 0, "expired": 0, "stale": 0, "stored": 0, "purged": 0}
        self.warnings: list[str] = []
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._save_disabled = False
        self._save_warned = False
        self._load()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self._dirty

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._entries

    def get(self, key: str, *, now: datetime | None = None) -> dict | None:
        """Cached value for ``key`` (a private deep copy), or None when missing or expired."""
        hit = self.get_entry(key, now=now)
        return None if hit is None else hit["value"]

    def get_entry(self, key: str, *, now: datetime | None = None, allow_expired: bool = False) -> dict | None:
        """``{value, source, stored_at, expires_at, expired}`` for ``key`` (``value`` a private
        deep copy), or None when missing — or expired, unless ``allow_expired`` (cache-only
        runs: the caller marks what it builds from it as stale). Discogs data past its
        6 hours is never returned, whatever ``allow_expired`` says."""
        _check_key(key)
        at = _aware(now) if now is not None else self._now()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.stats["misses"] += 1
                return None
            expired = _expired(entry, at)
            if expired and (not allow_expired or _must_drop(entry, at)):
                self.stats["expired"] += 1
                self.stats["misses"] += 1
                return None
            self.stats["hits"] += 1
            if expired:
                self.stats["stale"] += 1
            return {"value": copy.deepcopy(entry["value"]), "source": entry["source"],
                    "stored_at": entry["stored_at"], "expires_at": entry["expires_at"], "expired": expired}

    def entry_info(self, key: str) -> dict | None:
        """``{source, stored_at, expires_at}`` of an entry (expired or not), None if absent."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            return {k: entry[k] for k in ("source", "stored_at", "expires_at")}

    def put(self, key: str, value: dict, *, source: str, ttl_s: float | None = None,
            now: datetime | None = None) -> None:
        """Store a JSON-native dict under ``key``. ``ttl_s`` defaults to the source's TTL;
        Discogs data is capped at 6 hours. Raises TypeError / ValueError for a value that is
        not a JSON-serializable dict or a non-positive TTL (programming errors)."""
        _check_key(key)
        if not isinstance(source, str) or not source:
            raise ValueError(f"research cache source must be a non-empty string, got {source!r}")
        if not isinstance(value, dict):
            raise TypeError(f"research cache value must be a dict, got {type(value).__name__}")
        ttl = DEFAULT_TTL_S.get(source, FALLBACK_TTL_S) if ttl_s is None else float(ttl_s)
        if not (ttl > 0 and math.isfinite(ttl)):       # inf would overflow the expiry timestamp
            raise ValueError(f"research cache ttl_s must be a finite number > 0, got {ttl_s!r}")
        if source == "discogs":
            ttl = min(ttl, DISCOGS_MAX_AGE_S)
        # deep, validated copy: later caller mutations cannot leak in; NaN (ValueError),
        # non-JSON objects such as numpy integers (TypeError) and lone surrogates, which
        # could never be saved as UTF-8 (UnicodeEncodeError, a ValueError), are refused
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        text.encode("utf-8")
        stored = json.loads(text)
        at = _aware(now) if now is not None else self._now()
        entry = {"source": source, "stored_at": iso_utc(at),
                 "expires_at": iso_utc(at + timedelta(seconds=ttl)), "value": stored}
        with self._lock:
            self._entries[key] = entry
            self.stats["stored"] += 1
            self._dirty = True

    def purge_expired(self, *, now: datetime | None = None) -> int:
        """Drop ALL expired entries from memory (written on the next ``save``), including the
        MusicBrainz / Wikidata entries kept for stale cache-only use. Returns the count."""
        at = _aware(now) if now is not None else self._now()
        with self._lock:
            return self._purge_locked(at, keep_stale=False)

    def save(self) -> None:
        """Atomically write the cache; no-op when nothing changed, in read-only mode, or
        when the file at the path is not ours. Data that must not be kept (Discogs past
        6 hours) is purged first; other expired entries stay as stale fallback. On failure
        the previous file is left untouched and the OSError propagates."""
        with self._lock:
            if self.read_only:
                if self._dirty:
                    log.debug("research cache: read-only; not saving %s", self.path)
                return
            self._purge_locked(self._now())
            if not self._dirty:
                return
            if self._save_disabled:
                (log.debug if self._save_warned else log.warning)(
                    "research cache: not saving %s (the existing file is not a research cache or could not "
                    "be read)", self.path)
                self._save_warned = True
                return
            data = {
                "schema": RESEARCH_CACHE_SCHEMA,
                "schema_version": RESEARCH_CACHE_SCHEMA_VERSION,
                "entries": {k: self._entries[k] for k in sorted(self._entries)},
            }
            atomic_write_json(self.path, data)
            self._dirty = False
            log.debug("research cache: saved %d entries to %s", len(self._entries), self.path)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _now(self) -> datetime:
        return _aware(self._clock())

    def _purge_locked(self, at: datetime, *, keep_stale: bool = True) -> int:
        """Drop what must not be kept (expired Discogs data); with ``keep_stale`` False every
        expired entry."""
        expired = [k for k, e in self._entries.items() if (_must_drop(e, at) if keep_stale else _expired(e, at))]
        for k in expired:
            del self._entries[k]
        if expired:
            self.stats["purged"] += len(expired)
            self._dirty = True
        return len(expired)

    def _load(self) -> None:
        p = self.path
        if not p.exists():
            return
        try:
            raw_bytes = p.read_bytes()
        except OSError as e:
            self._save_disabled = True
            self._warn(f"research cache {p} could not be read ({type(e).__name__}: {e}); continuing with an "
                       f"empty cache and NOT saving, so the existing file is not overwritten")
            return
        if not raw_bytes.removeprefix(b"\xef\xbb\xbf").strip(b"\x00 \t\r\n"):
            self._unusable("the file is empty", ours=True)
            return
        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            self._unusable(f"not valid UTF-8 ({e.reason} at byte {e.start})", ours=False)
            return
        try:
            data = json.loads(text, parse_constant=lambda _c: None)
        except (json.JSONDecodeError, RecursionError) as e:
            where = f" at line {e.lineno} column {e.colno}" if isinstance(e, json.JSONDecodeError) else ""
            self._unusable(f"invalid or truncated JSON{where}", ours=bool(_OWN_CACHE_PREFIX_RE.match(text)))
            return
        if not isinstance(data, dict):
            self._unusable(f"top level is {type(data).__name__}, expected an object", ours=False)
            return
        if data.get("schema") != RESEARCH_CACHE_SCHEMA:
            self._unusable(f"schema is {data.get('schema')!r}, expected {RESEARCH_CACHE_SCHEMA!r}", ours=False)
            return
        if data.get("schema_version") != RESEARCH_CACHE_SCHEMA_VERSION:
            self._unusable(f"schema_version is {data.get('schema_version')!r}, expected "
                           f"{RESEARCH_CACHE_SCHEMA_VERSION}", ours=True)
            return
        entries = data.get("entries")
        if not isinstance(entries, dict):
            self._unusable(f"'entries' is {type(entries).__name__}, expected an object", ours=True)
            return
        bad: list[str] = []
        for key, raw in entries.items():
            entry = _parse_entry(raw)
            if entry is None or not key or not _utf8_ok([key, entry]):
                bad.append(str(key).encode("utf-8", "backslashreplace").decode("utf-8"))
                continue
            self._entries[str(key)] = entry
        if bad:
            self._warn_many(f"research cache {p.name}: dropped malformed entry", bad)
            self._dirty = True
        # data that must not be kept (Discogs older than 6 h) goes now; other expired entries
        # stay for cache-only runs (used as stale data) until a fresh answer replaces them
        self._purge_locked(self._now())

    def _unusable(self, reason: str, *, ours: bool) -> None:
        """Move a demonstrably damaged research cache aside; leave anything else untouched
        and disable saving (it is someone else's file)."""
        p = self.path
        if not ours:
            self._save_disabled = True
            self._warn(f"{p} is not a VEmix research cache ({reason}); it was left untouched and research "
                       f"results will NOT be saved — pass a different research cache path")
            return
        n = 1
        while True:
            target = p.with_name(f"{p.name}.corrupt-{n}")
            if not target.exists():
                break
            n += 1
        if self.read_only:
            self._save_disabled = True
            self._warn(f"research cache {p} is unusable ({reason}); it would be moved aside to {target.name} "
                       f"(read-only: left in place), continuing with an empty cache")
            return
        try:
            p.replace(target)
        except OSError as e:
            self._save_disabled = True
            self._warn(f"research cache {p} is unusable ({reason}) and could not be moved aside "
                       f"({type(e).__name__}: {e}); continuing with an empty cache and NOT saving")
            return
        self._warn(f"research cache {p} is unusable ({reason}); kept as {target.name}, starting with an "
                   f"empty cache")

    def _warn(self, message: str) -> None:
        log.warning("%s", message)
        self.warnings.append(message)

    def _warn_many(self, prefix: str, items: list[str]) -> None:
        for item in items[:_MAX_DETAILED_WARNINGS]:
            self._warn(f"{prefix}: {item}")
        if len(items) > _MAX_DETAILED_WARNINGS:
            self._warn(f"{prefix}: ... and {len(items) - _MAX_DETAILED_WARNINGS} more")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check_key(key: Any) -> None:
    if not isinstance(key, str) or not key:
        raise ValueError(f"research cache key must be a non-empty string, got {key!r}")
    key.encode("utf-8")         # UnicodeEncodeError (a ValueError) for a lone surrogate


def _utf8_ok(obj: Any) -> bool:
    """True when ``obj`` can be saved as UTF-8 JSON (a hand-edited file may carry an
    unpaired '\\ud800' escape, which would make every later save fail)."""
    try:
        json.dumps(obj, ensure_ascii=False).encode("utf-8")
    except (UnicodeEncodeError, TypeError, ValueError):
        return False
    return True


def _expired(entry: dict[str, Any], at: datetime) -> bool:
    stored = parse_iso(entry.get("stored_at"))
    expires = parse_iso(entry.get("expires_at"))
    if expires is not None and at >= expires:
        return True
    if entry.get("source") == "discogs":
        # hard rule, independent of what the file claims: unknown or too old age = expired
        if stored is None or (at - stored).total_seconds() >= DISCOGS_MAX_AGE_S \
                or (stored - at).total_seconds() > 300:
            return True
    return False


def _must_drop(entry: dict[str, Any], at: datetime) -> bool:
    """Expired data that must neither be used nor kept: Discogs (API terms, 6 hours), and any
    other answer more than ``STALE_RETENTION_S`` past its expiry."""
    if entry.get("source") == "discogs":
        return _expired(entry, at)
    expires = parse_iso(entry.get("expires_at"))
    return expires is not None and (at - expires).total_seconds() > STALE_RETENTION_S


def _parse_entry(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    source, stored_at, expires_at, value = (raw.get("source"), raw.get("stored_at"), raw.get("expires_at"),
                                            raw.get("value"))
    if not (isinstance(source, str) and source and isinstance(value, dict)):
        return None
    if parse_iso(stored_at) is None:
        return None
    if expires_at is not None and parse_iso(expires_at) is None:
        return None
    return {"source": source, "stored_at": stored_at, "expires_at": expires_at, "value": value}


def _check_cache_path(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS or suffix in LOSSLESS_EXTENSIONS:
        raise ValueError(f"refusing to use the audio file {path} as the research cache; "
                         f"pass a .json path (e.g. veg_research_cache.json)")
    if suffix != ".json":
        raise ValueError(f"the research cache must be a .json file, got {path.name!r}; "
                         f"pass a .json path (e.g. veg_research_cache.json)")


__all__ = ["RESEARCH_CACHE_SCHEMA", "RESEARCH_CACHE_SCHEMA_VERSION", "DEFAULT_TTL_S", "FALLBACK_TTL_S",
           "NEGATIVE_TTL_S", "DISCOGS_MAX_AGE_S", "STALE_RETENTION_S", "ResearchCache", "utc_now", "iso_utc", "parse_iso"]
