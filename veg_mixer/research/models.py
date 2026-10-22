"""Data contracts of the Music Research Agent (VEmix 3.0, spec §3).

A :class:`ResearchRecord` is everything research found about ONE library track, kept
strictly apart from :class:`~veg_mixer.models.TrackFeatures`: external statements are a
category of their own (spec §1) and never overwrite measured audio features, file tags
or curator data. Each part of a record says what kind of information it is:

* ``facts`` — verifiable catalogue data (MusicBrainz core data, Wikidata statements; both
  CC0): ``{field: {value, source, url, retrieved_at, fields_used, license}}``. Only a
  ``matched`` record has facts — an ambiguous match is NEVER applied.
* ``user_tags`` — classifications by catalogue users, never facts: folksonomy tags and
  genres voted by MusicBrainz users (*supplementary* data under CC BY-NC-SA 3.0, flagged
  ``non_commercial``) and Wikidata genre statements (CC0, but crowd-edited; ``scope``
  'work' = the genre of the SONG, e.g. of the original of a cover). Consumers use them only
  when community tags are explicitly allowed.
* ``editorial`` — community/editorial descriptions from other catalogues (Discogs genre,
  style, year): used only to cross-check, each value with its attribution and expiry.
* ``conflicts`` — the same field with different values in different sources (e.g. the
  MusicBrainz first-release year vs the Wikidata publication year). Nothing is resolved
  silently: the curator sees both values.
* ``candidates`` — the scored catalogue entries the matcher considered, with the reasons
  for each score, so an ambiguous match can be resolved by a person.
* ``stale`` — True when a cache-only run built the record from cache entries past their
  time-to-live (see ``notes`` for their age); serialized only when True.

Every Discogs value (editorial items, conflict values) carries ``attribution`` ("Data
provided by Discogs.") and ``expires_at``: it must not be shown after that time
(``prune_expired``; ``enrich.report_document`` writes reports without expired values).

Popularity / familiarity is never derived from any source, and BPM / key never come from
external data (they are measured locally).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

RESEARCH_CACHE_NAME = "veg_research_cache.json"
RESEARCH_MODES = ("off", "cache", "online")
# Identity resolution needs MusicBrainz; the others only cross-check a matched identity.
KNOWN_SOURCES = ("musicbrainz", "wikidata", "discogs")
DEFAULT_SOURCES = ("musicbrainz", "wikidata")

STATUS_MATCHED = "matched"          # identity confirmed, facts applied
STATUS_AMBIGUOUS = "ambiguous"      # plausible candidates, none certain: NOTHING applied
STATUS_NOT_FOUND = "not_found"      # no plausible candidate in the catalogue
STATUS_ERROR = "error"              # network / response failure (see ``errors``)
STATUS_SKIPPED = "skipped"          # not enough identity (no artist/title, no ISRC/MBID)
STATUS_NOT_CACHED = "not_cached"    # cache-only mode and nothing cached for this query
STATUSES = (STATUS_MATCHED, STATUS_AMBIGUOUS, STATUS_NOT_FOUND, STATUS_ERROR, STATUS_SKIPPED,
            STATUS_NOT_CACHED)

# Licences / attributions stored with every value.
LICENSE_CC0 = "CC0"
LICENSE_MB_SUPPLEMENTARY = "CC BY-NC-SA 3.0"    # MusicBrainz tags, genres, ratings
LICENSE_DISCOGS = "Discogs API Terms of Use"
DISCOGS_ATTRIBUTION = "Data provided by Discogs."
DISCOGS_SOURCE = "discogs"

# album / year (from the file's tags) only re-rank candidates; they are not searched for
QUERY_KEYS = ("artist", "title", "duration_s", "isrc", "mbid", "album", "year")
# facts written by earlier versions that are not facts (migrated by ``from_dict``)
LEGACY_TAG_FACTS = ("genre_wikidata",)


# lone UTF-16 surrogates: from undecodable file names / tags ('surrogateescape') or an
# unpaired '\ud800' JSON escape. They cannot be encoded as UTF-8, so a single one would
# make every later save of the cache or the report fail.
_LONE_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def clean_text(text: str) -> str:
    """``text`` with lone surrogates replaced by U+FFFD (always UTF-8 encodable)."""
    return _LONE_SURROGATE_RE.sub("�", text) if _LONE_SURROGATE_RE.search(text) else text


def _finite(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except OverflowError:            # an integer too large for a float
        return None
    return f if math.isfinite(f) else None


def jsonable(obj: Any) -> Any:
    """Deep copy restricted to JSON-native types (dict / list / str / int / float / bool /
    None); non-finite floats become None, numpy scalars plain numbers, lone surrogates
    U+FFFD and unknown objects their ``str()``. Research data comes from parsed JSON, so
    this only guards against accidents."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return clean_text(str(obj)) if isinstance(obj, str) else bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):          # numpy.float64 is a float subclass: plain float out
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {clean_text(str(k)): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):        # numpy scalars
        try:
            return jsonable(obj.item())
        except (TypeError, ValueError):
            pass
    return str(obj)


def value_expired(item: Any, now: datetime) -> bool:
    """True when a value must no longer be shown: its ``expires_at`` has passed or cannot be
    read, or it is Discogs data without one (unknown age: the Discogs API terms forbid
    showing data older than 6 hours)."""
    if not isinstance(item, dict):
        return False
    at = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    exp = item.get("expires_at")
    if not isinstance(exp, str):
        return item.get("source") == DISCOGS_SOURCE
    try:
        dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
    except ValueError:
        return True          # unknown age: treat as expired
    return at >= (dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc))


@dataclass
class ResearchRecord:
    """Research result for one track (see the module docstring for the categories).

    Shapes (all JSON-native):

    * ``query`` — ``{artist, title, duration_s, isrc, mbid, album, year}`` as used for
      matching (album / year from the file's tags only re-rank candidates).
    * ``chosen`` — ``{source, id, url}`` of the matched entry, None unless ``matched``.
    * ``candidates`` — ``[{source, id, url, title, artist, date, length_s,
      disambiguation, score, relevance, reasons}]`` best first; ``score`` is OUR 0..1
      match score, ``relevance`` the catalogue's own ranking value (MusicBrainz 0..100 is
      relevance within one query, not a probability).
    * ``facts`` — ``{field: {value, source, url, retrieved_at, fields_used, license}}``.
    * ``user_tags`` — ``[{name, count, kind, source, url, license, non_commercial,
      retrieved_at[, scope, method, id]}]`` (``kind`` = 'tag' | 'genre'; Wikidata genres:
      ``source`` 'wikidata', ``count`` None, ``scope`` 'recording' | 'work').
    * ``editorial`` — ``[{field, value, source, url, attribution, license, retrieved_at,
      expires_at}]``.
    * ``conflicts`` — ``[{field, values: [{source, value, url[, scope][, attribution,
      expires_at]}]}]`` (Discogs values carry ``attribution`` and ``expires_at``).
    * ``notes`` — remarks for the curator (why a match is ambiguous, what was not cached,
      how old stale data is).
    * ``stale`` — built by a cache-only run from entries past their time-to-live.
    """
    track_id: str
    rel_path: str
    query: dict[str, Any] = field(default_factory=dict)
    status: str = STATUS_SKIPPED
    match_confidence: float | None = None
    chosen: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    user_tags: list[dict[str, Any]] = field(default_factory=list)
    editorial: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)       # non-fatal remarks for the curator
    retrieved_at: str | None = None                      # newest underlying retrieval (ISO UTC)
    stale: bool = False                                  # cache-only run used data past its TTL

    @property
    def applied(self) -> bool:
        """True when the record carries applied data (only a confirmed match does)."""
        return self.status == STATUS_MATCHED

    def fact(self, name: str) -> Any:
        """Value of a fact, or None when unknown."""
        entry = self.facts.get(name)
        return entry.get("value") if isinstance(entry, dict) else None

    def prune_expired(self, now: datetime) -> int:
        """Drop values that must no longer be shown (``value_expired``: ``expires_at`` passed,
        or Discogs data of unknown age — Discogs data must not be shown after 6 hours): facts,
        user tags, editorial items and conflict values; a conflict left with fewer than two
        distinct values disappears. Called by enrichment and by ``enrich.report_document``;
        use it on records loaded from a saved report. Returns the number of removed values."""
        at = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)

        def expired(item: dict[str, Any]) -> bool:
            return value_expired(item, at)

        removed = 0
        kept_facts = {k: v for k, v in self.facts.items() if not expired(v)}
        removed += len(self.facts) - len(kept_facts)
        self.facts = kept_facts
        kept_tags = [t for t in self.user_tags if isinstance(t, dict) and not expired(t)]
        removed += len(self.user_tags) - len(kept_tags)
        self.user_tags = kept_tags
        kept_editorial = [e for e in self.editorial if isinstance(e, dict) and not expired(e)]
        removed += len(self.editorial) - len(kept_editorial)
        self.editorial = kept_editorial
        conflicts = []
        for c in self.conflicts:
            raw_values = c.get("values") if isinstance(c, dict) else None     # tolerant: saved reports
            values = [v for v in raw_values if isinstance(v, dict)] if isinstance(raw_values, list) else []
            kept = [v for v in values if not expired(v)]
            removed += len(values) - len(kept)
            if len({json.dumps(v.get("value"), sort_keys=True) for v in kept}) > 1:
                conflicts.append({**c, "values": kept})
        self.conflicts = conflicts
        return removed

    def to_dict(self) -> dict[str, Any]:
        out = {
            "track_id": self.track_id,
            "rel_path": self.rel_path,
            "query": {k: self.query.get(k) for k in QUERY_KEYS},
            "status": self.status,
            "match_confidence": (None if self.match_confidence is None
                                 else round(float(self.match_confidence), 4)),
            "chosen": self.chosen,
            "candidates": self.candidates,
            "facts": self.facts,
            "user_tags": self.user_tags,
            "editorial": self.editorial,
            "conflicts": self.conflicts,
            "errors": self.errors,
            "notes": self.notes,
            "retrieved_at": self.retrieved_at,
        }
        if self.stale:
            out["stale"] = True
        return jsonable(out)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResearchRecord":
        """Tolerant reader (unknown keys ignored, wrong types replaced by empty values);
        raises ValueError only when ``d`` is not a dict or the status is unknown. Facts of
        earlier versions that are not facts (``facts.genre_wikidata``: crowd-edited Wikidata
        genres) are moved to ``user_tags``, never read as facts."""
        if not isinstance(d, dict):
            raise ValueError(f"research record must be an object, got {type(d).__name__}")
        status = d.get("status", STATUS_SKIPPED)
        if status not in STATUSES:
            raise ValueError(f"unknown research status {status!r}")
        query = d.get("query") if isinstance(d.get("query"), dict) else {}
        chosen = d.get("chosen") if isinstance(d.get("chosen"), dict) else None

        def _list_of_dicts(key: str) -> list[dict[str, Any]]:
            v = d.get(key)
            return [jsonable(x) for x in v if isinstance(x, dict)] if isinstance(v, list) else []

        def _strings(key: str) -> list[str]:
            v = d.get(key)
            return [str(x) for x in v if isinstance(x, str)] if isinstance(v, list) else []

        facts_raw = d.get("facts") if isinstance(d.get("facts"), dict) else {}
        facts = {str(k): jsonable(v) for k, v in facts_raw.items() if isinstance(v, dict)}
        user_tags = _list_of_dicts("user_tags")
        for name in LEGACY_TAG_FACTS:
            legacy = facts.pop(name, None)
            if legacy is not None:
                from .wikidata import legacy_genre_tags     # lazy: wikidata imports this module
                have = {(t.get("source"), str(t.get("name")).casefold()) for t in user_tags}
                user_tags.extend(t for t in legacy_genre_tags(legacy)
                                 if (t["source"], t["name"].casefold()) not in have)
        return cls(
            track_id=str(d.get("track_id") or ""),
            rel_path=str(d.get("rel_path") or ""),
            query={k: jsonable(query.get(k)) for k in QUERY_KEYS},
            status=status,
            match_confidence=_finite(d.get("match_confidence")),
            chosen=jsonable(chosen) if chosen is not None else None,
            candidates=_list_of_dicts("candidates"),
            facts=facts,
            user_tags=user_tags,
            editorial=_list_of_dicts("editorial"),
            conflicts=_list_of_dicts("conflicts"),
            errors=_strings("errors"),
            notes=_strings("notes"),
            retrieved_at=d.get("retrieved_at") if isinstance(d.get("retrieved_at"), str) else None,
            stale=d.get("stale") is True,
        )


__all__ = [
    "RESEARCH_CACHE_NAME", "RESEARCH_MODES", "KNOWN_SOURCES", "DEFAULT_SOURCES", "STATUSES",
    "STATUS_MATCHED", "STATUS_AMBIGUOUS", "STATUS_NOT_FOUND", "STATUS_ERROR", "STATUS_SKIPPED",
    "STATUS_NOT_CACHED", "LICENSE_CC0", "LICENSE_MB_SUPPLEMENTARY", "LICENSE_DISCOGS",
    "DISCOGS_ATTRIBUTION", "DISCOGS_SOURCE", "QUERY_KEYS", "LEGACY_TAG_FACTS", "ResearchRecord",
    "clean_text", "jsonable", "value_expired",
]
