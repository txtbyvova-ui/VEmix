"""P1: analysis cache with real file identity, per-section versioning and
legacy (automix v1 ``bpm_cache.json``) migration.

File format (UTF-8 JSON, ``indent=2``)::

    {
      "schema": "vegroove.analysis_cache",
      "schema_version": 1,
      "entries": {
        "<absolute resolved source path>": {
          "fingerprint": {"size": 123, "mtime_ns": 456, "sha256": "..."},   # sha256 only in content-hash mode
          "sections": {
            "<section>": {"version": "bpm-v1", "params": "<sha1 of params>",
                          "value": {...}, "migrated_from": "bpm_cache.json"}   # migrated_from optional
          },
          "legacy_pending": {"source": "bpm_cache.json", "values": {"key": {...}}}   # optional
        }
      }
    }

Rules:

* Identity = absolute resolved path + size + ``mtime_ns`` (+ sha256 when
  ``use_content_hash``). A fingerprint change invalidates ALL sections of the file.
  In content-hash mode the hash is decisive: a touched-but-identical file stays valid
  and its fingerprint is refreshed. Every file is hashed at most once per process and
  file state (the hash is memoized by size + mtime).
* Sections (``probe``, ``bpm``, ``key``, ``energy``, ``loudness``) are independent: each
  stores its algorithm ``version`` and a stable hash of its ``params``. Changing the
  normalization target only invalidates ``loudness``.
* Only a file that is DEMONSTRABLY a damaged VEgroove cache (our schema with a bad
  version / entries, a truncated document that starts like ours, or an empty file) is
  moved aside to ``<name>.corrupt-<n>`` (never deleted); work continues with an empty
  cache plus a warning. Anything else at the cache path — another program's JSON, a
  binary file, an automix v1 ``bpm_cache.json`` (even a truncated one) — is never
  renamed or overwritten: it is left untouched and saving is disabled (a legacy file
  is still read for migration). Audio files and non-``.json`` paths are refused
  outright (ValueError). Individual malformed entries / sections are dropped with a
  warning; good ones are kept.
* ``read_only`` (dry run): nothing is ever written or renamed — ``save()`` is a no-op
  and a damaged cache is only reported as "would be moved aside".
* A result computed while the source changed is not stored: callers take
  ``fingerprint()`` BEFORE reading the file and pass it to ``put(expected_fingerprint=)``.
* Legacy migration: the old ``bpm_cache.json`` is only READ, never modified (complete
  entries of a truncated file are salvaged). Its
  values are attached to a new entry as ``legacy_pending`` (fingerprint taken at that
  moment) and materialized lazily by ``get()`` with the CURRENT section version/params,
  so the analysis code accepts them. Legacy values are imported only for paths that
  have no entry in the new cache; once an entry exists they are never re-imported
  (a later source modification therefore cannot resurrect stale legacy values).
  Preserving legacy keys matters: re-detected keys produced very bad mixes in the past.
* ``save()`` is atomic (temp file in the same directory, flush + fsync, ``os.replace``)
  and a no-op when nothing changed. All public methods are thread-safe.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import stat
import threading
import uuid
from pathlib import Path
from typing import Any

from .config import AUDIO_EXTENSIONS, LOSSLESS_EXTENSIONS
from .models import to_jsonable

log = logging.getLogger(__name__)

ANALYSIS_CACHE_SCHEMA = "vegroove.analysis_cache"
ANALYSIS_CACHE_SCHEMA_VERSION = 1
# automix v1 cache name: a file with this name is never renamed or overwritten.
LEGACY_CACHE_NAME = "bpm_cache.json"
# save() always writes this header first: a truncated document starting like this is ours.
_OWN_CACHE_PREFIX_RE = re.compile(r'\A\s*\{\s*"schema"\s*:\s*"' + re.escape(ANALYSIS_CACHE_SCHEMA) + '"')

# Sections that legacy bpm_cache.json values are migrated into.
LEGACY_SECTIONS = ("bpm", "key")
_CAMELOT_RE = re.compile(r"^(?:[1-9]|1[0-2])[AB]$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK = 1 << 20
# Detailed warnings per load; further malformed items are summarized in one line.
_MAX_DETAILED_WARNINGS = 10


def params_hash(params: dict | None) -> str:
    """Stable hash of an analysis-parameter dict.

    Keys are sorted; integers are folded to floats so ``30`` and ``30.0`` hash the
    same; paths/enums/dataclasses go through ``to_jsonable``.
    """
    canonical = json.dumps(_fold_numbers(to_jsonable(params or {})), sort_keys=True,
                           separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _fold_numbers(obj: Any) -> Any:
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _fold_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fold_numbers(v) for v in obj]
    return obj


def _source_key(source: Path) -> str:
    """Cache key: absolute resolved path (symlinks resolved, ``..`` removed)."""
    p = Path(source)
    try:
        return str(p.resolve())
    except OSError:
        return os.path.abspath(p)


def _legacy_norm(path_text: str, base: Path) -> str:
    """Normalized form used to match legacy keys against sources (case-insensitive on Windows)."""
    p = Path(path_text)
    if not p.is_absolute():
        p = base / p
    try:
        p = p.resolve()
    except OSError:
        p = Path(os.path.abspath(p))
    return os.path.normcase(str(p))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class AnalysisCache:
    """Persistent per-file, per-section cache of audio-analysis results."""

    def __init__(self, path: Path, *, legacy_path: Path | None = None,
                 use_content_hash: bool = False, read_only: bool = False) -> None:
        """``read_only``: never write or rename anything (dry run) — ``save()`` is a no-op.

        Raises ValueError for a cache path that cannot be ours (an audio file, a
        non-``.json`` file) and when ``legacy_path`` is the cache path itself."""
        self.path = Path(path)
        _check_cache_path(self.path)
        self.legacy_path = Path(legacy_path) if legacy_path else None
        if self.legacy_path is not None and _same_file(self.path, self.legacy_path):
            raise ValueError(f"legacy_path must differ from the cache path ({self.path}): "
                             f"the legacy file is read-only")
        self.read_only = bool(read_only)
        self.use_content_hash = bool(use_content_hash)
        self.stats: dict[str, int] = {"hits": 0, "misses": 0, "invalidated": 0, "migrated": 0}
        self.warnings: list[str] = []
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._dirty = False
        self._save_disabled = False
        # normcase(abs path) -> {"bpm": {...}, "key": {...}} from the legacy file
        self._legacy: dict[str, dict[str, dict]] = {}
        self._legacy_name = self.legacy_path.name if self.legacy_path else ""
        # cache key -> (size, mtime_ns, sha256): every file state is hashed once
        self._hash_memo: dict[str, tuple[int, int, str]] = {}
        self._load()
        if self.legacy_path is not None:
            self._load_legacy()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self._dirty

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, source: Path, section: str, *, version: str, params: dict | None = None) -> dict | None:
        """Cached value of ``section`` for ``source`` or None (missing source, no entry,
        changed file, other version/params). The returned dict is a private copy."""
        _check_section_args(section, version)
        key = _source_key(source)
        fp = self._fingerprint(Path(key), key)
        ph = params_hash(params)
        with self._lock:
            if fp is None:   # the source vanished: nothing can be trusted
                self.stats["misses"] += 1
                return None
            entry = self._entries.get(key)
            if entry is None:
                entry = self._create_entry_from_legacy(key, fp)
                if entry is None:
                    self.stats["misses"] += 1
                    return None
            elif not self._validate_fingerprint(key, fp):
                self.stats["misses"] += 1
                return None
            sec = entry["sections"].get(section)
            if sec is None:
                value = self._materialize_legacy(entry, section, version, ph)
                if value is None:
                    self.stats["misses"] += 1
                    return None
                self.stats["hits"] += 1
                return copy.deepcopy(value)
            if sec["version"] != version or sec["params"] != ph:
                log.debug("cache: %s/%s invalidated (version %s -> %s or params changed)",
                          key, section, sec["version"], version)
                self.stats["invalidated"] += 1
                self.stats["misses"] += 1
                return None
            self.stats["hits"] += 1
            return copy.deepcopy(sec["value"])

    def put(self, source: Path, section: str, value: dict, *, version: str, params: dict | None = None,
            expected_fingerprint: dict | None = None) -> bool:
        """Store ``value`` (a JSON-serializable dict) for ``source``/``section``.

        The fingerprint is taken now. With ``expected_fingerprint`` (``fingerprint()``
        taken BEFORE the source was read) nothing is stored when the file has changed
        since — the value may describe the old content — and False is returned. Raises
        FileNotFoundError when the source does not exist and ValueError/TypeError when
        the value is not JSON-serializable. Returns True when stored.
        """
        _check_section_args(section, version)
        if not isinstance(value, dict):
            raise TypeError(f"cache value must be a dict, got {type(value).__name__}")
        # Deep, JSON-validated copy: later caller mutations cannot leak into the cache,
        # and an unserializable value fails here instead of breaking save().
        stored = json.loads(json.dumps(to_jsonable(value), ensure_ascii=False, allow_nan=False))
        key = _source_key(source)
        fp = self._fingerprint(Path(key), key)
        if fp is None:
            raise FileNotFoundError(f"cannot cache analysis of a missing file: {key}")
        if expected_fingerprint is not None and not self.same_state(expected_fingerprint, fp):
            log.warning("cache: %s changed while it was being analysed; %s result not cached", key, section)
            return False
        ph = params_hash(params)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = self._new_entry(key, fp)
            else:
                self._validate_fingerprint(key, fp)
                entry = self._entries[key]   # may have been reset
            entry["sections"][section] = {"version": version, "params": ph, "value": stored}
            pending = entry.get("legacy_pending")
            if pending:
                pending["values"].pop(section, None)   # a fresh result supersedes legacy data
                if not pending["values"]:
                    entry.pop("legacy_pending", None)
            self._dirty = True
        return True

    def fingerprint(self, source: Path) -> dict | None:
        """Current identity of ``source`` (``{size, mtime_ns[, sha256]}``) or None when it is
        not a regular file. Take it before reading a file; see ``put(expected_fingerprint=)``."""
        key = _source_key(source)
        return self._fingerprint(Path(key), key)

    def same_state(self, a: dict | None, b: dict | None) -> bool:
        """True when two fingerprints describe the same file state (by content hash in
        content-hash mode, else by size + mtime_ns)."""
        if a is None or b is None:
            return a is None and b is None
        if a.get("size") != b.get("size"):
            return False
        if self.use_content_hash and "sha256" in a and "sha256" in b:
            return a["sha256"] == b["sha256"]
        return a.get("mtime_ns") == b.get("mtime_ns")

    def migrated_from(self, source: Path, section: str) -> str | None:
        """Name of the legacy file a cached section was imported from (None otherwise).
        Call right after a successful ``get()``; it does not re-check the fingerprint."""
        key = _source_key(source)
        with self._lock:
            entry = self._entries.get(key)
            sec = entry["sections"].get(section) if entry else None
            return sec.get("migrated_from") if sec else None

    def save(self) -> None:
        """Atomically write the cache file; no-op when nothing changed.

        On failure the previous file is left untouched, the temp file is removed and
        the exception propagates (the in-memory cache stays dirty)."""
        with self._lock:
            if not self._dirty:
                return
            if self.read_only:
                log.debug("cache: read-only (dry run); not saving %s", self.path)
                return
            if self._save_disabled:
                log.warning("cache: not saving %s (the existing file could not be read or moved aside)",
                            self.path)
                return
            data = {
                "schema": ANALYSIS_CACHE_SCHEMA,
                "schema_version": ANALYSIS_CACHE_SCHEMA_VERSION,
                "entries": {k: _ordered_entry(self._entries[k]) for k in sorted(self._entries)},
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex[:8]}.tmp")
            try:
                with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    log.warning("cache: could not remove temporary file %s", tmp)
                raise
            self._dirty = False
            log.debug("cache: saved %d entries to %s", len(self._entries), self.path)

    # ------------------------------------------------------------------
    # fingerprints
    # ------------------------------------------------------------------

    def _fingerprint(self, path: Path, key: str) -> dict | None:
        """Current fingerprint of the source or None when it is not a regular file.
        Computed outside the lock (hashing can be slow)."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        if not stat.S_ISREG(st.st_mode):
            return None
        fp: dict[str, Any] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        if self.use_content_hash:
            with self._lock:
                memo = self._hash_memo.get(key)
            if memo and memo[0] == st.st_size and memo[1] == st.st_mtime_ns:
                fp["sha256"] = memo[2]
            else:
                try:
                    digest = _sha256_file(path)
                except OSError:
                    return None
                with self._lock:
                    self._hash_memo[key] = (st.st_size, st.st_mtime_ns, digest)
                fp["sha256"] = digest
        return fp

    def _validate_fingerprint(self, key: str, fp: dict) -> bool:
        """True when the entry for ``key`` still describes the file. A changed file resets
        the entry (all sections and pending legacy values dropped). Lock must be held."""
        entry = self._entries[key]
        stored = entry["fingerprint"]
        valid = stored["size"] == fp["size"]
        if valid:
            if self.use_content_hash:
                if "sha256" in stored:
                    valid = stored["sha256"] == fp["sha256"]
                else:   # entry written without hashing: trust it only if untouched
                    valid = stored["mtime_ns"] == fp["mtime_ns"]
            else:
                valid = stored["mtime_ns"] == fp["mtime_ns"]
        if not valid:
            log.debug("cache: %s changed on disk; dropping %d cached section(s)", key, len(entry["sections"]))
            self.stats["invalidated"] += 1
            self._entries[key] = {"fingerprint": dict(fp), "sections": {}}
            self._dirty = True
            return False
        if stored != fp and (self.use_content_hash or "sha256" not in stored):
            # identical content (or a new hash for an untouched file): refresh in place
            entry["fingerprint"] = dict(fp)
            self._dirty = True
        return True

    # ------------------------------------------------------------------
    # legacy migration
    # ------------------------------------------------------------------

    def _new_entry(self, key: str, fp: dict) -> dict:
        """Create an entry; attach legacy values for this path (consumed once). Lock held."""
        entry: dict[str, Any] = {"fingerprint": dict(fp), "sections": {}}
        legacy = self._legacy.pop(os.path.normcase(key), None)
        if legacy:
            entry["legacy_pending"] = {"source": self._legacy_name, "values": legacy}
        self._entries[key] = entry
        self._dirty = True
        return entry

    def _create_entry_from_legacy(self, key: str, fp: dict) -> dict | None:
        if os.path.normcase(key) not in self._legacy:
            return None
        return self._new_entry(key, fp)

    def _materialize_legacy(self, entry: dict, section: str, version: str, ph: str) -> dict | None:
        """Turn a pending legacy value into a regular section with the CURRENT version and
        params (the analysis code then accepts it). Lock held."""
        pending = entry.get("legacy_pending")
        if not pending or section not in pending["values"]:
            return None
        value = pending["values"].pop(section)
        if not pending["values"]:
            entry.pop("legacy_pending", None)
        entry["sections"][section] = {"version": version, "params": ph, "value": value,
                                      "migrated_from": pending["source"]}
        self.stats["migrated"] += 1
        self._dirty = True
        log.info("cache: migrated legacy %s value from %s", section, pending["source"])
        return value

    def _load_legacy(self) -> None:
        """Read the automix v1 cache. Read only: the file is never modified or moved.
        Complete entries before the damage of a truncated file are salvaged (v1 saved it
        non-atomically, and its BPM/keys are worth keeping)."""
        lp = self.legacy_path
        assert lp is not None
        if not lp.is_file():
            log.debug("cache: legacy cache %s not found; nothing to migrate", lp)
            return
        try:
            text = lp.read_bytes().decode("utf-8-sig")
        except (OSError, UnicodeDecodeError) as e:
            self._warn(f"legacy cache {lp} is unreadable ({type(e).__name__}: {e}); nothing migrated "
                       f"(the file was left untouched)")
            return
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raw = _salvage_legacy_entries(text)
            if not raw:
                self._warn(f"legacy cache {lp} is unreadable ({type(e).__name__}: {e}); nothing migrated "
                           f"(the file was left untouched)")
                return
            self._warn(f"legacy cache {lp} is truncated or damaged ({e.msg} at line {e.lineno}); salvaged "
                       f"{len(raw)} complete entr{'y' if len(raw) == 1 else 'ies'} before the damage "
                       f"(the file was left untouched)")
        if not isinstance(raw, dict):
            self._warn(f"legacy cache {lp} is not a JSON object; nothing migrated (file left untouched)")
            return
        base = lp.parent.resolve()
        bad: list[str] = []
        for path_text, val in raw.items():
            values, problem = _parse_legacy_value(val)
            if problem:
                bad.append(f"{path_text} ({problem})")
            if values:
                self._legacy[_legacy_norm(str(path_text), base)] = values
        if bad:
            self._warn_many(f"legacy cache {lp.name}: ignored malformed entry", bad)
        if self._legacy:
            # sources already present in the new cache keep their entries (no re-import)
            with self._lock:
                existing = {os.path.normcase(k) for k in self._entries}
            for k in list(self._legacy):
                if k in existing:
                    del self._legacy[k]
            log.info("cache: %d legacy entr%s available for migration from %s",
                     len(self._legacy), "y" if len(self._legacy) == 1 else "ies", lp)

    # ------------------------------------------------------------------
    # loading / recovery
    # ------------------------------------------------------------------

    def _load(self) -> None:
        p = self.path
        if not p.exists():
            return
        try:
            raw_bytes = p.read_bytes()
        except OSError as e:
            self._save_disabled = True
            self._warn(f"analysis cache {p} could not be read ({type(e).__name__}: {e}); continuing with an "
                       f"empty cache and NOT saving, so the existing file is not overwritten")
            return
        if not raw_bytes.removeprefix(b"\xef\xbb\xbf").strip(b"\x00 \t\r\n"):
            self._unusable("the file is empty", ours=True)     # nothing in it to protect
            return
        try:
            text = raw_bytes.decode("utf-8-sig")   # tolerate a BOM added by an editor
        except UnicodeDecodeError as e:
            self._unusable(f"not valid UTF-8 ({e.reason} at byte {e.start})", ours=False)
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            self._unusable(f"invalid or truncated JSON ({e.msg} at line {e.lineno} column {e.colno})",
                           ours=bool(_OWN_CACHE_PREFIX_RE.match(text)))
            return
        if not isinstance(data, dict):
            self._unusable(f"top level is {type(data).__name__}, expected an object", ours=False)
            return
        if "schema" not in data and (_looks_like_legacy(data) or self._legacy_named):
            # The old automix bpm_cache.json at our path: never rename or overwrite it.
            self._save_disabled = True
            self._warn(f"{p} is a legacy automix v1 cache: it is only read for migration and will NOT be "
                       f"overwritten; use a different path for the analysis cache to keep new results")
            if self.legacy_path is None:
                self.legacy_path, self._legacy_name = p, p.name
            return
        if data.get("schema") != ANALYSIS_CACHE_SCHEMA:
            self._unusable(f"schema is {data.get('schema')!r}, expected {ANALYSIS_CACHE_SCHEMA!r}", ours=False)
            return
        if data.get("schema_version") != ANALYSIS_CACHE_SCHEMA_VERSION:
            self._unusable(f"schema_version is {data.get('schema_version')!r}, "
                           f"expected {ANALYSIS_CACHE_SCHEMA_VERSION}", ours=True)
            return
        entries = data.get("entries")
        if not isinstance(entries, dict):
            self._unusable(f"'entries' is {type(entries).__name__}, expected an object", ours=True)
            return
        bad_entries: list[str] = []
        bad_sections: list[str] = []
        for key, raw in entries.items():
            entry = _parse_entry(raw, bad_sections, str(key))
            if entry is None:
                bad_entries.append(str(key))
                continue
            self._entries[str(key)] = entry
        if bad_entries:
            self._warn_many(f"analysis cache {p.name}: dropped malformed entry", bad_entries)
        if bad_sections:
            self._warn_many(f"analysis cache {p.name}: dropped malformed section", bad_sections)
        if bad_entries or bad_sections:
            self._dirty = True   # the cleaned version is written on the next save

    @property
    def _legacy_named(self) -> bool:
        return self.path.name.casefold() == LEGACY_CACHE_NAME

    def _unusable(self, reason: str, *, ours: bool) -> None:
        """The file at the cache path cannot be loaded. Only a demonstrably damaged VEgroove
        cache (``ours``) is moved aside; anything else is someone's data and is never
        renamed or overwritten — saving is disabled instead. A file named like the automix
        v1 cache is never moved, and it is still read for migration (complete entries of a
        truncated file are salvaged)."""
        if ours and not self._legacy_named:
            self._quarantine(reason)
            return
        p = self.path
        self._save_disabled = True
        if self._legacy_named and self.legacy_path is None:
            self.legacy_path, self._legacy_name = p, p.name
            self._warn(f"{p} is not a valid analysis cache ({reason}); it is treated as a read-only legacy "
                       f"automix v1 cache and will NOT be modified; use a different --cache path to keep "
                       f"new results")
            return
        self._warn(f"{p} is not a VEgroove analysis cache ({reason}); it was left untouched and new analysis "
                   f"results will NOT be saved — pass a different --cache path")

    def _quarantine(self, reason: str) -> None:
        """Move a damaged cache file aside to ``<name>.corrupt-<n>`` (never delete it).
        In read-only mode only report what would happen."""
        p = self.path
        n = 1
        while True:
            target = p.with_name(f"{p.name}.corrupt-{n}")
            if not target.exists():
                break
            n += 1
        if self.read_only:
            self._save_disabled = True
            self._warn(f"analysis cache {p} is unusable ({reason}); it would be moved aside to {target.name} "
                       f"(read-only / dry run: left in place), continuing with an empty cache")
            return
        try:
            os.replace(p, target)
        except OSError as e:
            self._save_disabled = True
            self._warn(f"analysis cache {p} is unusable ({reason}) and could not be moved aside "
                       f"({type(e).__name__}: {e}); continuing with an empty cache and NOT saving")
            return
        self._warn(f"analysis cache {p} is unusable ({reason}); kept as {target.name}, "
                   f"starting with an empty cache")

    def _warn(self, message: str) -> None:
        log.warning("%s", message)
        self.warnings.append(message)

    def _warn_many(self, prefix: str, items: list[str]) -> None:
        for item in items[:_MAX_DETAILED_WARNINGS]:
            self._warn(f"{prefix}: {item}")
        if len(items) > _MAX_DETAILED_WARNINGS:
            self._warn(f"{prefix}: ... and {len(items) - _MAX_DETAILED_WARNINGS} more")


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

def _check_section_args(section: str, version: str) -> None:
    if not isinstance(section, str) or not section:
        raise ValueError(f"cache section must be a non-empty string, got {section!r}")
    if not isinstance(version, str) or not version:
        raise ValueError(f"cache version must be a non-empty string, got {version!r}")


def _parse_fingerprint(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    size, mtime = raw.get("size"), raw.get("mtime_ns")
    if not (isinstance(size, int) and not isinstance(size, bool) and size >= 0):
        return None
    if not (isinstance(mtime, int) and not isinstance(mtime, bool)):
        return None
    fp: dict[str, Any] = {"size": size, "mtime_ns": mtime}
    if "sha256" in raw:
        if not (isinstance(raw["sha256"], str) and _SHA256_RE.match(raw["sha256"])):
            return None
        fp["sha256"] = raw["sha256"]
    return fp


def _parse_section(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    version, ph, value = raw.get("version"), raw.get("params"), raw.get("value")
    if not (isinstance(version, str) and version and isinstance(ph, str) and isinstance(value, dict)):
        return None
    sec: dict[str, Any] = {"version": version, "params": ph, "value": value}
    if "migrated_from" in raw:
        if not isinstance(raw["migrated_from"], str):
            return None
        sec["migrated_from"] = raw["migrated_from"]
    return sec


def _parse_entry(raw: Any, bad_sections: list[str], key: str) -> dict | None:
    """Validated entry, or None when the entry itself is unusable. Malformed sections
    are dropped individually (recorded in ``bad_sections``)."""
    if not isinstance(raw, dict) or not key:
        return None
    fp = _parse_fingerprint(raw.get("fingerprint"))
    sections_raw = raw.get("sections", {})
    if fp is None or not isinstance(sections_raw, dict):
        return None
    sections: dict[str, dict] = {}
    for name, sraw in sections_raw.items():
        sec = _parse_section(sraw)
        if sec is None or not name:
            bad_sections.append(f"{key} [{name}]")
            continue
        sections[str(name)] = sec
    entry: dict[str, Any] = {"fingerprint": fp, "sections": sections}
    pending = raw.get("legacy_pending")
    if pending is not None:
        ok = (isinstance(pending, dict) and isinstance(pending.get("source"), str)
              and isinstance(pending.get("values"), dict)
              and all(s in LEGACY_SECTIONS and isinstance(v, dict) for s, v in pending["values"].items()))
        if ok and pending["values"]:
            entry["legacy_pending"] = {"source": pending["source"], "values": dict(pending["values"])}
        elif not ok:
            bad_sections.append(f"{key} [legacy_pending]")
    return entry


def _parse_legacy_value(val: Any) -> tuple[dict[str, dict], str | None]:
    """automix v1 entry -> ({"bpm": {...}, "key": {...}}, problem).

    Accepted: a number (BPM only) or ``{"bpm": float, "camelot": "8A", "key_confidence": float}``
    with any of the fields present. Valid fields are kept even when a sibling field is
    malformed; ``problem`` describes what was ignored (None when everything was usable).
    """
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if _is_number(val) and val >= 0:
            return {"bpm": {"bpm": float(val)}}, None
        return {}, f"invalid bpm {val!r}"
    if not isinstance(val, dict):
        return {}, f"unsupported value of type {type(val).__name__}"
    out: dict[str, dict] = {}
    problems: list[str] = []
    if "bpm" in val:
        bpm = val["bpm"]
        if _is_number(bpm) and bpm >= 0:
            out["bpm"] = {"bpm": float(bpm)}
        else:
            problems.append(f"invalid bpm {bpm!r}")
    if "camelot" in val:
        cam = val["camelot"]
        if isinstance(cam, str) and _CAMELOT_RE.match(cam):
            conf = val.get("key_confidence")
            out["key"] = {"camelot": cam, "key_confidence": float(conf) if _is_number(conf) else None}
        else:
            problems.append(f"invalid camelot {cam!r}")
    return out, "; ".join(problems) or None


def _check_cache_path(path: Path) -> None:
    """Refuse cache paths that cannot be ours: renaming / rewriting a library track or a
    curator sheet given to ``--cache`` by mistake would destroy it."""
    suffix = path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS or suffix in LOSSLESS_EXTENSIONS:
        raise ValueError(f"refusing to use the audio file {path} as the analysis cache; "
                         f"pass a .json path (e.g. veg_analysis_cache.json)")
    if suffix != ".json":
        raise ValueError(f"the analysis cache must be a .json file, got {path.name!r}; "
                         f"pass a .json path (e.g. veg_analysis_cache.json)")


def _salvage_legacy_entries(text: str) -> dict:
    """Complete ``"path": value`` pairs at the start of a damaged automix v1 object. A value
    counts only when a ',' or '}' follows it, so a number cut in half ('12' of '128.5')
    is never taken for a BPM."""
    decoder = json.JSONDecoder()
    ws = re.compile(r"\s*")
    i = ws.match(text, 0).end()
    if i >= len(text) or text[i] != "{":
        return {}
    out: dict = {}
    i = ws.match(text, i + 1).end()
    try:
        while i < len(text) and text[i] != "}":
            key, i = decoder.raw_decode(text, i)
            i = ws.match(text, i).end()
            if not isinstance(key, str) or i >= len(text) or text[i] != ":":
                break
            value, i = decoder.raw_decode(text, ws.match(text, i + 1).end())
            i = ws.match(text, i).end()
            if i >= len(text) or text[i] not in ",}":
                break
            out[key] = value
            if text[i] == ",":
                i = ws.match(text, i + 1).end()
    except json.JSONDecodeError:
        pass
    return out


def _looks_like_legacy(data: dict) -> bool:
    """automix v1 ``bpm_cache.json``: {path: float} / {path: {"bpm", "camelot", "key_confidence"}}."""
    legacy_fields = {"bpm", "camelot", "key_confidence"}
    return bool(data) and all((isinstance(v, (int, float)) and not isinstance(v, bool))
               or (isinstance(v, dict) and set(v) <= legacy_fields)
               for v in data.values())


def _same_file(a: Path, b: Path) -> bool:
    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
    except OSError:
        pass
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _ordered_entry(entry: dict) -> dict:
    out: dict[str, Any] = {"fingerprint": entry["fingerprint"],
                           "sections": {k: entry["sections"][k] for k in sorted(entry["sections"])}}
    if entry.get("legacy_pending"):
        out["legacy_pending"] = entry["legacy_pending"]
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()
