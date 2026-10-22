"""Cache of AI interpretations (VEmix 3.0, spec §4/§9): one paid answer per distinct input.

File format (UTF-8 JSON, ``indent=2``)::

    {
      "schema": "vegroove.ai_cache",
      "schema_version": 1,
      "entries": {
        "<sha256 key>": {
          "output": {...} | null,       # validated interpretation (dropped claims excluded)
          "valid": true,                # false = negative result: invalid even after one re-ask
          "errors": [...], "dropped": [...],
          "provider": "anthropic", "model": "claude-opus-5-5",   # model that produced the answer
          "request_id": "req_..." | null, "usage": {"input_tokens": 1234, ...},
          "created_at": "2026-10-22T10:00:00+00:00",
          "input_digest": "<sha256>",   # provider-independent digest of the same input
          "inputs": ["curator", "measured"]   # payload sections the answer was built from
        }
      }
    }

Rules:

* The key is CONTENT-ADDRESSED: a hash of the prompt version, schema version, provider, model
  and the canonical input payload (``interpret.cache_key``) — not of a file path or a file
  fingerprint. Renaming the library, moving it to another disk or editing an ID3 tag that is
  not part of the payload keeps the answer; any change of the data the model saw (a new
  segment analysis, a curator's mood, a matched fact) asks again.
* ``input_digest`` lets a cache-only run without a configured provider find an answer by
  input alone (``find``).
* A negative result (``valid: false``) is cached too, so an input that produced invalid output
  twice is not paid for on every run; provider errors (network, refusal, rate limit) are NOT
  cached.
* Damaged-file policy as in ``veg_mixer.cache``: only a file that is demonstrably a damaged
  VEgroove AI cache (our schema with a bad version / entries, a truncated document that starts
  like ours, an empty file) is moved aside to ``<name>.corrupt-<n>`` (never deleted); anything
  else at the path is never renamed or overwritten — it is left untouched, a warning is
  recorded and saving is disabled. Audio files and non-``.json`` paths are refused (ValueError).
  Malformed entries — including a ``valid`` entry whose output does not have the shape of a
  validated interpretation — are dropped one by one with a warning.
* ``read_only`` (dry run): nothing is written or renamed. ``save()`` is atomic
  (``fsutil.atomic_write_json``) and a no-op when nothing changed. Thread-safe.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import threading
from pathlib import Path
from typing import Any

from ..config import AUDIO_EXTENSIONS, LOSSLESS_EXTENSIONS
from ..fsutil import atomic_write_json
from .schema import INTERPRETATION_SCHEMA, check_instance

log = logging.getLogger(__name__)

AI_CACHE_SCHEMA = "vegroove.ai_cache"
AI_CACHE_SCHEMA_VERSION = 1
AI_CACHE_NAME = "veg_ai_cache.json"
_OWN_PREFIX_RE = re.compile(r'\A\s*\{\s*"schema"\s*:\s*"' + re.escape(AI_CACHE_SCHEMA) + '"')
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_DETAILED_WARNINGS = 10


# ---------------------------------------------------------------------------
# Canonical JSON / keys
# ---------------------------------------------------------------------------

def _canonical_value(obj: Any) -> Any:
    """Floats rounded to 3 decimals (integral values become ints, so 120 == 120.0 == 120.0004);
    tuples -> lists; keys -> str. Non-finite floats and other types are an error."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("non-finite number in AI payload")
        r = round(obj, 3)
        return int(r) if r == int(r) and abs(r) < 1e15 else r
    if isinstance(obj, dict):
        return {str(k): _canonical_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical_value(v) for v in obj]
    raise TypeError(f"value of type {type(obj).__name__} is not JSON-native")


def canonical_json(obj: Any) -> str:
    """Deterministic compact JSON (sorted keys, rounded floats): the cache-key material and the
    exact text sent to a provider."""
    return json.dumps(_canonical_value(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def make_key(material: Any) -> str:
    """sha256 of the canonical JSON of ``material``."""
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class AICache:
    """Persistent store of interpretation results keyed by ``interpret.cache_key``."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        _check_path(self.path)
        self.read_only = bool(read_only)
        self.stats: dict[str, int] = {"hits": 0, "misses": 0, "stored": 0}
        self.warnings: list[str] = []
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._by_digest: dict[str, set[str]] = {}       # input_digest -> keys (``find``)
        self._dirty = False
        self._save_disabled = False
        self._load()

    # --- public --------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self._dirty

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, key: str) -> dict | None:
        """A private copy of the entry for ``key`` or None."""
        with self._lock:
            entry = self._entries.get(key)
            self.stats["hits" if entry is not None else "misses"] += 1
            return copy.deepcopy(entry) if entry is not None else None

    def find(self, input_digest: str) -> tuple[str, dict] | None:
        """Best entry for an input regardless of provider / model: valid before invalid, then the
        newest, then the smallest key (deterministic). Counts as a hit or a miss. Indexed by
        digest: a cache-only run over N tracks is O(N), not O(N x entries)."""
        with self._lock:
            matches = [(k, self._entries[k]) for k in self._by_digest.get(input_digest, ())]
            if not matches:
                self.stats["misses"] += 1
                return None
            matches.sort(key=lambda ke: ke[0])                                   # ties: smallest key
            matches.sort(key=lambda ke: ke[1]["created_at"], reverse=True)       # newest (stable sort)
            matches.sort(key=lambda ke: not ke[1]["valid"])                      # valid first
            self.stats["hits"] += 1
            key, entry = matches[0]
            return key, copy.deepcopy(entry)

    def put(self, key: str, value: dict) -> None:
        """Store an entry (validated and deep-copied: later caller mutations cannot leak in).
        Raises ValueError / TypeError for a malformed key or a value that is not JSON-native."""
        if not isinstance(key, str) or not _KEY_RE.match(key):
            raise ValueError(f"AI cache key must be a sha256 hex digest, got {key!r}")
        stored = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        problem = _entry_problem(stored)
        if problem:
            raise ValueError(f"invalid AI cache entry: {problem}")
        with self._lock:
            self._store(key, stored)
            self.stats["stored"] += 1
            self._dirty = True

    def _store(self, key: str, entry: dict) -> None:
        old = self._entries.get(key)
        if old is not None and old.get("input_digest") in self._by_digest:
            self._by_digest[old["input_digest"]].discard(key)
        self._entries[key] = entry
        digest = entry.get("input_digest")
        if isinstance(digest, str):
            self._by_digest.setdefault(digest, set()).add(key)

    def save(self) -> None:
        """Atomically write the cache; no-op when unchanged, read-only or disabled. On failure the
        previous file is untouched and the exception (OSError) propagates."""
        with self._lock:
            if not self._dirty:
                return
            if self.read_only:
                log.debug("AI cache: read-only (dry run); not saving %s", self.path)
                return
            if self._save_disabled:
                log.warning("AI cache: not saving %s (the existing file is not a VEgroove AI cache or could "
                            "not be moved aside)", self.path)
                return
            data = {"schema": AI_CACHE_SCHEMA, "schema_version": AI_CACHE_SCHEMA_VERSION,
                    "entries": {k: self._entries[k] for k in sorted(self._entries)}}
            json.dumps(data, allow_nan=False)        # JSON-native by construction; fail before writing
            atomic_write_json(self.path, data)
            self._dirty = False
            log.debug("AI cache: saved %d entries to %s", len(self._entries), self.path)

    # --- loading / recovery --------------------------------------------------

    def _load(self) -> None:
        p = self.path
        if not p.exists():
            return
        try:
            raw = p.read_bytes()
        except OSError as e:
            self._save_disabled = True
            self._warn(f"AI cache {p} could not be read ({type(e).__name__}: {e}); continuing with an empty "
                       f"cache and NOT saving, so the existing file is not overwritten")
            return
        if not raw.removeprefix(b"\xef\xbb\xbf").strip(b"\x00 \t\r\n"):
            self._unusable("the file is empty", ours=True)
            return
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            self._unusable(f"not valid UTF-8 ({e.reason} at byte {e.start})", ours=False)
            return
        try:
            data = json.loads(text, parse_constant=_reject_constant)
        except (ValueError, RecursionError) as e:
            reason = "nested too deeply" if isinstance(e, RecursionError) else str(e)
            self._unusable(f"invalid or truncated JSON ({reason})", ours=bool(_OWN_PREFIX_RE.match(text)))
            return
        if not isinstance(data, dict):
            self._unusable(f"top level is {type(data).__name__}, expected an object", ours=False)
            return
        if data.get("schema") != AI_CACHE_SCHEMA:
            self._unusable(f"schema is {data.get('schema')!r}, expected {AI_CACHE_SCHEMA!r}", ours=False)
            return
        if data.get("schema_version") != AI_CACHE_SCHEMA_VERSION:
            self._unusable(f"schema_version is {data.get('schema_version')!r}, expected {AI_CACHE_SCHEMA_VERSION}",
                           ours=True)
            return
        entries = data.get("entries")
        if not isinstance(entries, dict):
            self._unusable(f"'entries' is {type(entries).__name__}, expected an object", ours=True)
            return
        bad: list[str] = []
        for key, entry in entries.items():
            problem = "key is not a sha256 digest" if not _KEY_RE.match(str(key)) else _entry_problem(entry)
            if problem:
                bad.append(f"{str(key)[:16]}... ({problem})")
                continue
            self._store(str(key), entry)
        if bad:
            for item in bad[:_MAX_DETAILED_WARNINGS]:
                self._warn(f"AI cache {p.name}: dropped malformed entry {item}")
            if len(bad) > _MAX_DETAILED_WARNINGS:
                self._warn(f"AI cache {p.name}: ... and {len(bad) - _MAX_DETAILED_WARNINGS} more malformed entries")
            self._dirty = True         # the cleaned version is written on the next save

    def _unusable(self, reason: str, *, ours: bool) -> None:
        if ours:
            self._quarantine(reason)
            return
        self._save_disabled = True
        self._warn(f"{self.path} is not a VEgroove AI cache ({reason}); it was left untouched and new AI results "
                   f"will NOT be saved — pass a different AI cache path")

    def _quarantine(self, reason: str) -> None:
        p = self.path
        n = 1
        while p.with_name(f"{p.name}.corrupt-{n}").exists():
            n += 1
        target = p.with_name(f"{p.name}.corrupt-{n}")
        if self.read_only:
            self._save_disabled = True
            self._warn(f"AI cache {p} is unusable ({reason}); it would be moved aside to {target.name} "
                       f"(read-only / dry run: left in place), continuing with an empty cache")
            return
        try:
            os.replace(p, target)
        except OSError as e:
            self._save_disabled = True
            self._warn(f"AI cache {p} is unusable ({reason}) and could not be moved aside "
                       f"({type(e).__name__}: {e}); continuing with an empty cache and NOT saving")
            return
        self._warn(f"AI cache {p} is unusable ({reason}); kept as {target.name}, starting with an empty cache")

    def _warn(self, message: str) -> None:
        log.warning("%s", message)
        self.warnings.append(message)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name}")


def _is_str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _entry_problem(e: Any) -> str | None:
    """Why a cache entry is malformed (None when it is fine)."""
    if not isinstance(e, dict):
        return f"entry is {type(e).__name__}, expected an object"
    if not isinstance(e.get("valid"), bool):
        return "'valid' must be true or false"
    output = e.get("output")
    if output is not None and not isinstance(output, dict):
        return "'output' must be an object or null"
    if e["valid"] and output is None:
        return "a valid entry needs an output"
    if e["valid"]:
        # a valid output is served as is: it must have the validated interpretation's shape
        shape = check_instance(output, INTERPRETATION_SCHEMA, "output")
        if shape:
            return f"'output' does not match the interpretation schema ({shape[0]})"
    if not _is_str_list(e.get("errors")):
        return "'errors' must be a list of strings"
    if not isinstance(e.get("dropped"), list):
        return "'dropped' must be a list"
    for k in ("provider", "model", "created_at"):
        if not isinstance(e.get(k), str) or not e[k]:
            return f"'{k}' must be a non-empty string"
    if e.get("request_id") is not None and not isinstance(e["request_id"], str):
        return "'request_id' must be a string or null"
    if not isinstance(e.get("usage", {}), dict):
        return "'usage' must be an object"
    if e.get("input_digest") is not None and not (isinstance(e["input_digest"], str) and _KEY_RE.match(e["input_digest"])):
        return "'input_digest' must be a sha256 digest"
    if e.get("inputs") is not None and not _is_str_list(e["inputs"]):
        return "'inputs' must be a list of strings"
    return None


def _check_path(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS or suffix in LOSSLESS_EXTENSIONS:
        raise ValueError(f"refusing to use the audio file {path} as the AI cache; pass a .json path "
                         f"(e.g. {AI_CACHE_NAME})")
    if suffix != ".json":
        raise ValueError(f"the AI cache must be a .json file, got {path.name!r}; pass a .json path "
                         f"(e.g. {AI_CACHE_NAME})")


__all__ = ["AI_CACHE_SCHEMA", "AI_CACHE_SCHEMA_VERSION", "AI_CACHE_NAME", "AICache", "canonical_json", "make_key"]
