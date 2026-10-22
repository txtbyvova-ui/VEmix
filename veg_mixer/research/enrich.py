"""Music Research Agent: online / cache-only enrichment of library tracks (spec §3).

For each track the agent establishes a catalogue identity and collects sourced
statements about it, without ever touching the audio file or ``TrackFeatures``:

1. identity — in order of evidence: a MusicBrainz recording ID in the file's tags
   (Picard), an ISRC tag, else a recording search by artist + title + duration
   (``musicbrainz``), re-ranked and gated by ``matching`` (the file's album / year tags
   break ties between plausible candidates) — an ambiguous match is kept for the curator
   and NOT applied;
2. facts of a confirmed match — MusicBrainz lookup (CC0 core data) and a Wikidata
   cross-check by MBID (CC0); MusicBrainz tags/genres and Wikidata genres go to
   ``user_tags`` (classifications, not facts; MusicBrainz's are CC BY-NC-SA 3.0);
3. optional Discogs cross-check (only with ``VEG_DISCOGS_TOKEN``) → ``editorial``, every
   value with "Data provided by Discogs." and ``expires_at`` (6 hours);
4. conflicts between sources (first-release year vs Wikidata publication year vs
   Discogs year; file ISRC vs catalogue ISRCs) are recorded, never resolved silently.

Modes (``RESEARCH_MODES``): ``off`` does nothing; ``cache`` answers from the research
cache only and never opens a connection (misses → ``not_cached``); ``online`` uses the
cache first and the network for misses / expired entries, rate-limited per host
(``default_client``: MusicBrainz 1.1 s, Wikidata 1.0 s, Discogs 1.1 s between requests).

Sources: without an explicit ``sources``, ``online`` uses ``default_sources()``
(MusicBrainz + Wikidata, + Discogs when a token is configured) and ``cache`` consults every
source that may have cached answers (Discogs too: reading the cache needs no token).

Cache-only runs and the time-to-live: a MusicBrainz / Wikidata cache entry past its TTL is
still used in ``cache`` mode — the plan stays reproducible — but the record is marked
``stale`` (with a note naming the age; ``stats['stale']`` counts such records). Discogs
data past 6 hours is never used (API terms). ``online`` refreshes expired entries.

Reports: ``report_document`` builds the JSON report (expired Discogs values pruned,
attribution next to every Discogs value); ``format_conflict`` renders a conflict for the
console with the Discogs attribution next to its value.

Failures degrade, they never abort the run: a network or response error becomes the
track's ``error`` status (or an ``errors`` note when only a cross-check failed). After
``max_consecutive_errors`` consecutive failures of one source a circuit breaker stops
requests to it for the rest of the run (MusicBrainz → no requests at all; the remaining
tracks still get cached answers, the others ``error`` with the reason). Nothing raises
``OSError`` / ``ValueError`` for network or parse problems.

Not implemented on purpose: Bandcamp and artist/label sites (no public API — HTML
scraping is fragile and often against the site's terms), Last.fm (API terms restrict
commercial use; its tags are popularity-weighted folksonomy), AcoustID (needs an audio
fingerprint computed from the file and sent to a service: requires the user's explicit
opt-in, spec §2). Popularity / familiarity is never derived from any source; BPM and key
never come from external data.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..fsutil import atomic_write_text
from ..models import AudioProbe, TrackFeatures
from ..net import PROJECT_URL, HttpClient, NetError, user_agent
from . import discogs as dg
from . import musicbrainz as mb
from . import wikidata as wd
from .cache import DISCOGS_MAX_AGE_S, NEGATIVE_TTL_S, ResearchCache, iso_utc, parse_iso, utc_now
from .matching import (
    DEFAULT_MATCH_CONFIG,
    LocalIdentity,
    MatchConfig,
    MatchDecision,
    decide,
    decide_identified,
    score_recordings,
)
from .models import (
    DEFAULT_SOURCES,
    DISCOGS_ATTRIBUTION,
    DISCOGS_SOURCE,
    KNOWN_SOURCES,
    RESEARCH_MODES,
    STATUS_AMBIGUOUS,
    STATUS_ERROR,
    STATUS_MATCHED,
    STATUS_NOT_CACHED,
    STATUS_NOT_FOUND,
    STATUS_SKIPPED,
    ResearchRecord,
    clean_text,
    jsonable,
    value_expired,
)

log = logging.getLogger(__name__)

CONTACT_ENV = "VEG_CONTACT"
# host -> minimum seconds between two requests (MusicBrainz: ~1 request/s per client)
DEFAULT_MIN_INTERVAL_S: dict[str, float] = {
    "musicbrainz.org": 1.1,
    "www.wikidata.org": 1.0,
    "api.discogs.com": 1.1,
}
DEFAULT_CHECKPOINT_EVERY = 25
# 'stale': records a cache-only run built from cache entries past their time-to-live
STAT_KEYS = ("matched", "ambiguous", "not_found", "errors", "skipped", "not_cached", "cache_hits",
             "network_requests", "stale")
# ffprobe tag keys (lower-cased) of the album and the release date, best first: the file's
# own release date, then the original release date (Picard's 'originaldate' / ID3 TDOR)
ALBUM_TAG_KEYS = ("album",)
YEAR_TAG_KEYS = ("date", "year", "tdrc", "tyer", "originaldate", "originalyear", "tdor")
_LEADING_YEAR_RE = re.compile(r"\s*(\d{4})", re.ASCII)
_ANY_YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)", re.ASCII)
REPORT_SCHEMA = "vegroove.research_report"
DISCOGS_NOTICE = (f"Values with source 'discogs' are Discogs data ({DISCOGS_ATTRIBUTION}): each carries that "
                  f"attribution and 'expires_at' — it must not be shown after that time "
                  f"({DISCOGS_MAX_AGE_S / 3600:g} hours after retrieval, Discogs API Terms of Use). Values already "
                  f"expired when this report was written were removed.")
_STATUS_STAT = {STATUS_MATCHED: "matched", STATUS_AMBIGUOUS: "ambiguous", STATUS_NOT_FOUND: "not_found",
                STATUS_ERROR: "errors", STATUS_SKIPPED: "skipped", STATUS_NOT_CACHED: "not_cached"}
REVIEW_CSV_COLUMNS = ("rel_path", "status", "match_confidence", "candidate_rank", "title", "artist", "date",
                      "length", "disambiguation", "url", "chosen", "score", "reasons")


def _clean_contact(text: Any) -> str:
    """Contact for the User-Agent: printable ASCII only. http.client sends header values
    as Latin-1 and refuses line breaks, so a contact such as 'Роман <me@example.org>'
    would make EVERY request fail; the non-ASCII part is dropped instead."""
    s = " ".join(str(text or "").split())
    cleaned = " ".join("".join(ch for ch in s if " " <= ch <= "~").split())
    if cleaned != s:
        log.warning("research: the contact for the User-Agent was reduced to printable ASCII")
    return cleaned


def default_client(contact: str | None = None, *, timeout_s: float = 20.0) -> HttpClient:
    """HTTP client for research: User-Agent ``VEmix/<ver> ( <contact> )`` (contact from the
    argument, else ``VEG_CONTACT``, else the project URL) and per-host rate limits."""
    contact = _clean_contact(contact or os.environ.get(CONTACT_ENV, "")) or PROJECT_URL
    return HttpClient(user_agent=user_agent(contact), timeout_s=timeout_s,
                      min_interval_s=dict(DEFAULT_MIN_INTERVAL_S))


def _enforce_rate_limits(client: HttpClient) -> None:
    """Raise (never lower) the client's per-host intervals to the research minimums: a
    caller-supplied client without them would hammer MusicBrainz, which then refuses
    every request from this address."""
    intervals = getattr(client, "min_interval_s", None)
    if not isinstance(intervals, dict):
        return
    for host, interval in DEFAULT_MIN_INTERVAL_S.items():
        if not intervals.get(host, 0.0) >= interval:
            intervals[host] = interval


def default_sources(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """MusicBrainz + Wikidata, plus Discogs when ``VEG_DISCOGS_TOKEN`` is set."""
    return DEFAULT_SOURCES + (("discogs",) if dg.token_from_env(env) else ())


@dataclass
class EnrichResult:
    records: dict[str, ResearchRecord] = field(default_factory=dict)     # by track_id
    stats: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STAT_KEYS, 0))
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable({"stats": self.stats, "warnings": self.warnings,
                         "records": {k: r.to_dict() for k, r in self.records.items()}})

    def summary(self) -> str:
        s = self.stats
        stale = f"; {s['stale']} stale (cache entries past their time-to-live)" if s.get("stale") else ""
        return (f"{s['matched']} matched, {s['ambiguous']} ambiguous, {s['not_found']} not found, "
                f"{s['not_cached']} not cached, {s['skipped']} skipped, {s['errors']} errors; "
                f"{s['cache_hits']} from cache, {s['network_requests']} network requests{stale}")


# ---------------------------------------------------------------------------
# session: cache-first fetching, per-source circuit breaker
# ---------------------------------------------------------------------------

# what malformed response / cache data can raise in the parsing and matching code
# (ArithmeticError: e.g. OverflowError from an absurdly large number)
_DATA_ERRORS = (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError, RecursionError)


class _NotCached(Exception):
    """Cache-only mode and the data is not cached."""


class _NetworkDisabled(Exception):
    """The circuit breaker of this source is open."""


class _Session:
    def __init__(self, *, mode: str, cache: ResearchCache, client: HttpClient | None,
                 now: Callable[[], datetime], max_errors: int) -> None:
        self.mode = mode
        self.cache = cache
        self.client = client
        self._now = now
        self.max_errors = max_errors
        self.consecutive: dict[str, int] = {}
        self.tripped: dict[str, str] = {}
        self.track_hits = 0
        self.track_net = 0
        self.track_stale: list[dict[str, Any]] = []      # expired cache entries used (cache mode)

    def now(self) -> datetime:
        dt = self._now()
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=utc_now().tzinfo)

    def disabled_reason(self, source: str) -> str | None:
        return self.tripped.get("musicbrainz") or self.tripped.get(source)

    def fetch(self, key: str, source: str, fn: Callable[[], dict[str, Any]], *,
              negative: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, Any]:
        """Cached value, else (online mode) ``fn()`` stored with ``retrieved_at``. In cache
        mode an entry past its time-to-live is still used (``track_stale``; never Discogs —
        the cache refuses expired Discogs data). Raises _NotCached, _NetworkDisabled or a
        NetError (counted by the breaker)."""
        hit = self.cache.get_entry(key, now=self.now(), allow_expired=self.mode == "cache")
        if hit is not None:
            self.track_hits += 1
            if hit["expired"]:
                self.track_stale.append({k: hit[k] for k in ("source", "stored_at", "expires_at")})
            return hit["value"]
        if self.mode != "online" or self.client is None:
            raise _NotCached(key)
        reason = self.disabled_reason(source)
        if reason:
            raise _NetworkDisabled(reason)
        self.track_net += 1
        try:
            value = fn()
        except NetError as e:
            self._failed(source, e)
            raise
        except _DATA_ERRORS as e:
            err = mb.ResponseShapeError(f"malformed {source} response ({type(e).__name__}: {e})")
            self._failed(source, err)
            raise err from e
        self.consecutive[source] = 0
        value = dict(value)
        value["retrieved_at"] = iso_utc(self.now())
        ttl = NEGATIVE_TTL_S if (negative is not None and negative(value)) else None
        self.cache.put(key, value, source=source, ttl_s=ttl, now=self.now())
        return value

    def _failed(self, source: str, err: Exception) -> None:
        n = self.consecutive.get(source, 0) + 1
        self.consecutive[source] = n
        if n >= self.max_errors and source not in self.tripped:
            scope = "no further network requests" if source == "musicbrainz" else f"no further {source} requests"
            self.tripped[source] = (f"{source}: {n} consecutive network failures (last: {err}); "
                                    f"circuit breaker open — {scope} in this run")
            log.warning("research: %s", self.tripped[source])


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def enrich_tracks(tracks: Sequence[TrackFeatures], *, probes: Mapping[str, AudioProbe] | None = None,
                  mode: str = "cache", cache: ResearchCache, client: HttpClient | None = None,
                  sources: Sequence[str] | None = None,
                  progress: Callable[[int, int, str], None] | None = None,
                  now: Callable[[], datetime] | None = None, max_consecutive_errors: int = 5,
                  match_config: MatchConfig | None = None,
                  checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
                  discogs_token: str | None = None) -> EnrichResult:
    """Research every track (see the module docstring). ``probes`` (by ``track_id``)
    provide ISRC / MusicBrainz-ID / album / date tags and the duration when a track has
    none. ``sources`` None = the mode's default: ``online`` → ``default_sources()`` (plus
    Discogs when ``discogs_token`` is given); ``cache`` → every known source (cached Discogs
    answers are read without a token; a Discogs miss is then silent).
    ``client`` defaults to ``default_client()`` in online mode; a caller's client gets its
    per-host intervals raised to the research minimums (never lowered). A Discogs token
    with spaces / line breaks is refused with a warning (never echoed); cache mode needs no
    token. ``checkpoint_every`` saves the cache after that many tracks that needed the
    network (0 = never; a save failure is a warning) — call ``cache.save()`` after the run
    as well. In cache mode, records built from entries past their time-to-live are marked
    ``stale`` (``stats['stale']``, a note and a warning naming the age).
    Raises ValueError only for invalid arguments (unknown mode / source)."""
    if mode not in RESEARCH_MODES:
        raise ValueError(f"unknown research mode {mode!r}; expected one of {', '.join(RESEARCH_MODES)}")
    implicit: frozenset[str] = frozenset()
    if sources is None:
        if mode == "cache":
            sources = KNOWN_SOURCES
            implicit = frozenset(s for s in KNOWN_SOURCES if s not in DEFAULT_SOURCES)
        else:
            sources = default_sources() + (("discogs",) if (discogs_token or "").strip() else ())
    srcs = tuple(dict.fromkeys(str(s).strip().casefold() for s in sources))
    unknown = [s for s in srcs if s not in KNOWN_SOURCES]
    if unknown:
        raise ValueError(f"unknown research source(s) {', '.join(unknown)}; known: {', '.join(KNOWN_SOURCES)}")
    if "musicbrainz" not in srcs:
        raise ValueError("research needs the 'musicbrainz' source: the other sources only cross-check a "
                         "MusicBrainz identity")
    if max_consecutive_errors < 1:
        raise ValueError(f"max_consecutive_errors must be >= 1, got {max_consecutive_errors}")
    cfg = match_config or DEFAULT_MATCH_CONFIG
    result = EnrichResult()
    if mode == "off":
        return result
    if mode == "online" and client is None:
        client = default_client()
    if mode == "online" and client is not None:
        _enforce_rate_limits(client)
    token: str | None = None
    if "discogs" in srcs and mode == "online":
        # only requests need the token: cache mode reads cached Discogs answers without one
        token = (discogs_token or "").strip() or dg.token_from_env()
        if not token:
            result.warnings.append(f"discogs: skipped — set {dg.TOKEN_ENV} to a Discogs personal access token "
                                   f"to enable the Discogs cross-check")
        elif not dg.valid_token(token):
            # the value is never echoed: it is a credential
            result.warnings.append(f"discogs: skipped — the Discogs token ({dg.TOKEN_ENV}) contains spaces, line "
                                   f"breaks or non-ASCII characters (value not shown)")
            token = None
        if not token:
            srcs = tuple(s for s in srcs if s != "discogs")
    sess = _Session(mode=mode, cache=cache, client=client if mode == "online" else None,
                    now=now or utc_now, max_errors=max_consecutive_errors)
    requests_before = client.stats.get("requests", 0) if (client is not None and mode == "online") else 0
    items = list(tracks)
    total = len(items)
    since_checkpoint = 0
    stale_ages: list[int] = []
    for i, track in enumerate(items, 1):
        sess.track_hits = sess.track_net = 0
        sess.track_stale = []
        probe = (probes or {}).get(track.track_id)
        try:
            rec = _research_track(sess, track, probe, srcs, token, cfg, implicit=implicit)
        except _DATA_ERRORS as e:
            # defensive: malformed cached data must not abort a long run
            log.debug("research: malformed data for %s", track.rel_path, exc_info=True)
            rec = ResearchRecord(track_id=track.track_id, rel_path=track.rel_path, status=STATUS_ERROR,
                                 errors=[f"malformed research data ({type(e).__name__}: {e})"])
        if sess.track_stale and rec.status != STATUS_SKIPPED:
            stale_ages.append(_mark_stale(rec, sess.track_stale, sess.now()))
            result.stats["stale"] += 1
        if track.track_id in result.records:
            result.warnings.append(f"duplicate track id {track.track_id} ({track.rel_path}); later record kept")
        result.records[track.track_id] = rec
        result.stats[_STATUS_STAT[rec.status]] += 1
        if rec.status != STATUS_SKIPPED and sess.track_net == 0 and sess.track_hits > 0:
            result.stats["cache_hits"] += 1
        if sess.track_net:
            since_checkpoint += 1
            if checkpoint_every and since_checkpoint >= checkpoint_every:
                since_checkpoint = 0
                _checkpoint(cache, result)
        log.debug("research: %s -> %s", track.rel_path, rec.status)
        if progress is not None:
            progress(i, total, f"{track.display_name}: {rec.status}")
    for reason in sess.tripped.values():
        result.warnings.append(reason)
    if stale_ages:
        result.warnings.append(f"{len(stale_ages)} research record(s) were built from cache entries past their "
                               f"time-to-live (oldest retrieved {max(stale_ages)} day(s) ago) and are marked stale; "
                               f"run online enrichment to refresh them")
    if client is not None and mode == "online":
        result.stats["network_requests"] = max(0, client.stats.get("requests", 0) - requests_before)
    return result


def _mark_stale(rec: ResearchRecord, hits: list[dict[str, Any]], now: datetime) -> int:
    """Flag a record built from expired cache entries; returns the oldest entry's age in days."""
    rec.stale = True
    ages: list[tuple[int, int | None]] = []
    for h in hits:
        stored, expires = parse_iso(h.get("stored_at")), parse_iso(h.get("expires_at"))
        if stored is None:
            continue
        ttl = int((expires - stored).total_seconds() // 86400) if expires is not None else None
        ages.append((max(0, int((now - stored).total_seconds() // 86400)), ttl))
    oldest, ttl = max(ages, key=lambda a: a[0]) if ages else (0, None)
    names = ", ".join(sorted({str(h.get("source")) for h in hits}))
    ttl_text = f"their {ttl}-day time-to-live" if ttl is not None else "their time-to-live"
    rec.notes.append(f"stale research data: {names} cache entries retrieved up to {oldest} day(s) ago are past "
                     f"{ttl_text}; used because the run is cache-only — run online enrichment to refresh them")
    return oldest


def _checkpoint(cache: ResearchCache, result: EnrichResult) -> None:
    try:
        cache.save()
    except OSError as e:
        msg = f"research cache {cache.path} could not be saved ({type(e).__name__}: {e})"
        if msg not in result.warnings:
            result.warnings.append(msg)
        log.warning("research: %s", msg)


# ---------------------------------------------------------------------------
# one track
# ---------------------------------------------------------------------------

def _clean(text: Any) -> str | None:
    """Identity text as searched / used in cache keys: whitespace collapsed and lone
    surrogates (an undecodable tag) replaced — they could be neither URL-encoded nor
    saved, and the failure would count against the MusicBrainz circuit breaker."""
    if text is None:
        return None
    s = clean_text(" ".join(str(text).split()))
    return s or None


def _positive_finite(value: Any) -> float | None:
    """A usable duration (plain float; NaN / inf / numpy scalars never reach the matcher
    or the JSON record)."""
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def tag_album(tags: Mapping[str, Any] | None) -> str | None:
    """Album title from the file's tags (whitespace collapsed), None when absent."""
    for key in ALBUM_TAG_KEYS:
        album = _clean((tags or {}).get(key))
        if album:
            return album
    return None


def tag_year(tags: Mapping[str, Any] | None) -> int | None:
    """Release year from the file's tags ('1982', '1982-11-30', '19821130', '30.11.1982'):
    the first plausible year (1860..2100) of ``YEAR_TAG_KEYS`` in order, else None."""
    for key in YEAR_TAG_KEYS:
        raw = (tags or {}).get(key)
        if raw is None:
            continue
        s = str(raw)
        m = _LEADING_YEAR_RE.match(s) or _ANY_YEAR_RE.search(s)
        if m and 1860 <= int(m.group(1)) <= 2100:
            return int(m.group(1))
    return None


def _local_identity(track: TrackFeatures, probe: AudioProbe | None,
                    cfg: MatchConfig) -> tuple[dict[str, Any], LocalIdentity]:
    artist, title = _clean(track.artist), _clean(track.title)
    duration = _positive_finite(track.duration_s)
    if duration is None and probe is not None:
        duration = _positive_finite(probe.duration_s)
    tags = probe.tags if probe is not None else {}
    isrc, mbid = mb.isrc_from_tags(tags), mb.mbid_from_tags(tags)
    album, year = tag_album(tags), tag_year(tags)
    from_filename = [f for f in ("artist", "title") if track.feature_sources.get(f) == "filename"]
    factor = cfg.filename_identity_factor if from_filename else 1.0
    note = f"{' and '.join(from_filename)} parsed from the file name" if from_filename else None
    query = {"artist": artist, "title": title, "duration_s": round(duration, 3) if duration else None,
             "isrc": isrc, "mbid": mbid, "album": album, "year": year}
    return query, LocalIdentity(artist=artist, title=title, duration_s=duration, isrc=isrc,
                                identity_factor=factor, identity_note=note, album=album, year=year)


def _with_time(recordings: Any, retrieved_at: Any) -> list[dict[str, Any]]:
    if not isinstance(recordings, list):
        return []
    return [dict(r, retrieved_at=retrieved_at) for r in recordings if isinstance(r, dict) and r.get("id")]


def _research_track(sess: _Session, track: TrackFeatures, probe: AudioProbe | None, sources: tuple[str, ...],
                    token: str | None, cfg: MatchConfig, *, implicit: frozenset[str] = frozenset()) -> ResearchRecord:
    """``implicit``: sources consulted only because they may be cached (cache mode without
    explicit sources) — a cache miss there is not worth a note."""
    query, local = _local_identity(track, probe, cfg)
    rec = ResearchRecord(track_id=track.track_id, rel_path=track.rel_path, query=query)
    if not ((local.artist and local.title) or local.isrc or query["mbid"]):
        rec.status = STATUS_SKIPPED
        rec.notes.append("no artist/title and no ISRC/MusicBrainz ID tag: nothing to look up")
        return rec
    used: list[Any] = []
    try:
        decision, details = _identify(sess, local, query, rec, used, cfg)
    except _NotCached:
        rec.status = STATUS_NOT_CACHED
        rec.notes.append("not in the research cache (cache-only mode): run online enrichment to look it up")
        return rec
    except _NetworkDisabled as e:
        rec.status = STATUS_ERROR
        rec.errors.append(str(e))
        return rec
    except NetError as e:
        rec.status = STATUS_ERROR
        rec.errors.append(f"musicbrainz: {e}")
        return rec
    rec.candidates = [c.to_dict() for c in decision.candidates]
    rec.match_confidence = None if decision.confidence is None else round(decision.confidence, 4)
    if decision.status != STATUS_MATCHED or decision.chosen is None:
        rec.status = decision.status
        rec.notes.extend(decision.reasons)
        rec.retrieved_at = _latest(used)
        return rec

    chosen = decision.chosen
    full = details if details is not None else _lookup_details(sess, chosen.id, chosen.data, rec, used)
    if details is None:
        if full is not chosen.data:
            problems = _verify(local, full, cfg)
        elif local.artist and chosen.components.get("artist") is None:
            # ISRC route without the recording details: the artist was never compared
            problems = ["the artist could not be verified (recording details unavailable): not applied"]
        else:
            problems = []
        if problems:
            # the complete record contradicts the file: keep it for the curator, apply nothing
            rec.status = STATUS_AMBIGUOUS
            rec.notes.extend(problems)
            rec.retrieved_at = _latest(used)
            return rec
    rec.status = STATUS_MATCHED
    rid = full["id"]
    if rid != chosen.id:
        rec.notes.append(f"MusicBrainz recording {chosen.id} was merged into {rid}")
    rec.chosen = {"source": mb.SOURCE, "id": rid, "url": mb.recording_url(rid)}
    rec.notes.extend(decision.reasons)
    retrieved_at = full.get("retrieved_at")
    rec.facts = mb.recording_facts(full, retrieved_at=retrieved_at)
    rec.user_tags = mb.recording_user_tags(full, retrieved_at=retrieved_at)
    if local.isrc and full.get("isrcs") and local.isrc not in full["isrcs"]:
        rec.conflicts.append({"field": "isrc", "values": [
            {"source": "file_tag", "value": local.isrc, "url": None},
            {"source": mb.SOURCE, "value": list(full["isrcs"]), "url": mb.recording_url(rid)}]})
    if "wikidata" in sources:
        _wikidata(sess, rec, full, used)
    if "discogs" in sources and (token or sess.mode == "cache"):
        _discogs(sess, rec, local, full, token or "", used, quiet_miss="discogs" in implicit)
    _year_conflicts(rec)
    rec.prune_expired(sess.now())         # nothing that must no longer be shown leaves this function
    rec.retrieved_at = _latest(used)
    return rec


def _identify(sess: _Session, local: LocalIdentity, query: dict[str, Any], rec: ResearchRecord,
              used: list[Any], cfg: MatchConfig) -> tuple[MatchDecision, dict[str, Any] | None]:
    """(decision, full lookup data when the route already fetched it)."""
    client = sess.client
    mbid = query.get("mbid")
    if mbid:
        full = sess.fetch(mb.recording_key(mbid), mb.SOURCE, lambda: mb.lookup_recording(client, mbid),
                          negative=lambda v: bool(v.get("not_found")))
        if not full.get("not_found") and full.get("id"):
            used.append(full.get("retrieved_at"))
            cand = score_recordings(local, [full], url_for=mb.recording_url, cfg=cfg)[0]
            return decide_identified(local, cand, evidence="MusicBrainz recording ID tag", cfg=cfg), full
        rec.notes.append(f"MusicBrainz has no recording {mbid} (from the file's tags); identifying otherwise")
    if local.isrc:
        isrc = local.isrc
        data = sess.fetch(mb.isrc_key(isrc), mb.SOURCE, lambda: mb.lookup_isrc(client, isrc),
                          negative=lambda v: not v.get("recordings"))
        used.append(data.get("retrieved_at"))
        recs = _with_time(data.get("recordings"), data.get("retrieved_at"))
        if recs:
            return decide(local, score_recordings(local, recs, url_for=mb.recording_url, cfg=cfg), cfg=cfg), None
        rec.notes.append(f"ISRC {isrc} is not in MusicBrainz; searching by artist and title")
    if not (local.artist and local.title):
        return MatchDecision(STATUS_NOT_FOUND, None, None, [], ["no artist/title to search for"]), None
    artist, title, duration = local.artist, local.title, local.duration_s
    data = sess.fetch(mb.search_key(artist, title, duration), mb.SOURCE,
                      lambda: mb.search_recordings(client, artist, title, duration),
                      negative=lambda v: not v.get("recordings"))
    used.append(data.get("retrieved_at"))
    if data.get("fallback_query"):
        rec.notes.append("no result within the duration window; candidates come from a search without it")
    recs = _with_time(data.get("recordings"), data.get("retrieved_at"))
    return decide(local, score_recordings(local, recs, url_for=mb.recording_url, cfg=cfg), cfg=cfg), None


def _lookup_details(sess: _Session, mbid: str, fallback: dict[str, Any], rec: ResearchRecord,
                    used: list[Any]) -> dict[str, Any]:
    """Full recording data for the facts; the search result itself when unavailable."""
    try:
        full = sess.fetch(mb.recording_key(mbid), mb.SOURCE, lambda: mb.lookup_recording(sess.client, mbid),
                          negative=lambda v: bool(v.get("not_found")))
    except _NotCached:
        rec.notes.append("recording details not cached: facts limited to the search result")
        return fallback
    except (_NetworkDisabled, NetError) as e:
        rec.errors.append(f"musicbrainz lookup: {e}")
        rec.notes.append("facts limited to the search result")
        return fallback
    if full.get("not_found") or not full.get("id"):
        rec.notes.append("recording lookup found nothing: facts limited to the search result")
        return fallback
    used.append(full.get("retrieved_at"))
    return full


def _verify(local: LocalIdentity, full: dict[str, Any], cfg: MatchConfig) -> list[str]:
    """Re-check the identity against the complete recording (an ISRC result has no
    artist credit before the lookup)."""
    cand = score_recordings(local, [full], cfg=cfg)[0]
    problems = []
    a, t = cand.components.get("artist"), cand.components.get("title")
    if a is not None and a < cfg.min_artist_similarity:
        problems.append(f"the recording is credited to '{full.get('artist_phrase')}' (artist similarity "
                        f"{a:.2f} < {cfg.min_artist_similarity:.2f}): not applied")
    if t is not None and t < cfg.min_title_similarity:
        problems.append(f"the recording is titled '{full.get('title')}' (title similarity {t:.2f}): not applied")
    if cand.duration_diff_s is not None and cand.duration_diff_s > cfg.duration_tolerance_s:
        problems.append(f"the recording's duration differs by {cand.duration_diff_s:.1f} s: not applied")
    return problems


def _wikidata(sess: _Session, rec: ResearchRecord, full: dict[str, Any], used: list[Any]) -> None:
    rid = full["id"]
    work = rec.fact("work_mbid")
    try:
        data = sess.fetch(wd.crosscheck_key(rid, work), wd.SOURCE, lambda: wd.crosscheck(sess.client, rid, work),
                          negative=lambda v: not v.get("qid"))
    except _NotCached:
        rec.notes.append("wikidata cross-check not cached")
        return
    except (_NetworkDisabled, NetError) as e:
        rec.errors.append(f"wikidata: {e}")
        return
    used.append(data.get("retrieved_at"))
    if data.get("multiple"):
        rec.notes.append(f"wikidata: several items carry this MusicBrainz ID ({', '.join(map(str, data['multiple']))}); "
                         f"none used")
    elif data.get("rejected"):
        rec.notes.append(f"wikidata: item {data['rejected']} not used ({data.get('rejected_reason') or 'unverified'})")
    elif not data.get("qid"):
        rec.notes.append("wikidata: no item with this MusicBrainz recording/work ID")
    rec.facts.update(wd.crosscheck_facts(data))
    # genres are crowd-edited classifications (of the SONG for a work-scope item): user tags
    rec.user_tags.extend(wd.crosscheck_user_tags(data))


def _discogs(sess: _Session, rec: ResearchRecord, local: LocalIdentity, full: dict[str, Any], token: str,
             used: list[Any], *, quiet_miss: bool = False) -> None:
    credit = full.get("artist_credit") or []
    artist = (credit[0].get("name") if credit and isinstance(credit[0], dict) else None) or local.artist
    title = full.get("title") or local.title
    if not artist or not title:
        return
    try:
        data = sess.fetch(dg.search_key(artist, title), dg.SOURCE,
                          lambda: dg.search(sess.client, token, artist, title),
                          negative=lambda v: not v.get("results"))
    except _NotCached:
        if not quiet_miss:
            rec.notes.append("discogs cross-check not cached (or older than 6 hours)")
        return
    except (_NetworkDisabled, NetError) as e:
        rec.errors.append(f"discogs: {e}")
        return
    used.append(data.get("retrieved_at"))
    items = dg.editorial(data, artist)
    if not items:
        rec.notes.append("discogs: no master by this artist with this track")
    rec.editorial.extend(items)


def _year_conflicts(rec: ResearchRecord) -> None:
    values: list[dict[str, Any]] = []
    mb_year = rec.facts.get("year")
    if mb_year and isinstance(mb_year.get("value"), int):
        values.append({"source": mb.SOURCE, "value": mb_year["value"], "url": mb_year.get("url")})
    pub = rec.facts.get("publication_date_wikidata")
    if pub and isinstance(pub.get("value"), str) and pub["value"][:4].isascii() and pub["value"][:4].isdigit():
        value = {"source": wd.SOURCE, "value": int(pub["value"][:4]), "url": pub.get("url")}
        if pub.get("scope") == "work":
            # the SONG's publication (found via the work MBID): for a cover or a re-recording
            # a different year is expected — the curator must see that it is not this recording
            value["scope"] = "work"
        values.append(value)
    for item in rec.editorial:
        if item.get("field") == "year" and isinstance(item.get("value"), int):
            value = {"source": item.get("source"), "value": item["value"], "url": item.get("url")}
            if item.get("source") == DISCOGS_SOURCE:
                # the attribution travels with the value; it must not be shown after 6 h
                value["attribution"] = item.get("attribution") or DISCOGS_ATTRIBUTION
            if item.get("expires_at") or item.get("source") == DISCOGS_SOURCE:
                value["expires_at"] = item.get("expires_at")
            values.append(value)
    if len({v["value"] for v in values}) > 1:
        rec.conflicts.append({"field": "year", "values": values})


def _latest(times: list[Any]) -> str | None:
    vals = sorted(t for t in times if isinstance(t, str) and t)
    return vals[-1] if vals else None


# ---------------------------------------------------------------------------
# report / console rendering (Discogs: attribution next to the value, 6 h lifetime)
# ---------------------------------------------------------------------------

def _discogs_values(d: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Every Discogs-sourced value dict of a record dict (facts, user tags, editorial,
    conflict values)."""
    facts = d.get("facts") if isinstance(d.get("facts"), dict) else {}
    pools: list[Any] = [list(facts.values()), d.get("user_tags"), d.get("editorial")]
    pools += [c.get("values") for c in d.get("conflicts") or [] if isinstance(c, dict)]
    for pool in pools:
        for item in pool if isinstance(pool, list) else []:
            if isinstance(item, dict) and item.get("source") == DISCOGS_SOURCE:
                yield item


def report_document(records: EnrichResult | Mapping[str, ResearchRecord | Mapping[str, Any]]
                    | Iterable[ResearchRecord], *, now: datetime | None = None,
                    stats: Mapping[str, int] | None = None,
                    warnings: Sequence[str] | None = None) -> dict[str, Any]:
    """JSON-ready research report (for ``enrich --report``)::

        {"schema": "vegroove.research_report", "generated_at": <ISO UTC>,
         "records": {track_id: record dict}, "stats": {...}, "warnings": [...],
         "discogs": {"notice", "attribution", "max_age_hours", "values", "expires_at",
                     "removed_expired"}}

    ``records`` is an ``EnrichResult`` (its stats / warnings are included unless given),
    a mapping track_id -> ``ResearchRecord`` (or its dict) or an iterable of records. Values
    that must no longer be shown at ``now`` (default: the current time) are removed
    (``ResearchRecord.prune_expired``: Discogs data older than 6 hours or of unknown age);
    every remaining Discogs value carries ``attribution`` "Data provided by Discogs." and
    ``expires_at``. ``discogs.expires_at`` is the earliest expiry in the report: the report
    must not be shown after it without being rebuilt. The input records are not modified.
    Raises ValueError for a record dict with an unknown status."""
    if isinstance(records, EnrichResult):
        stats = records.stats if stats is None else stats
        warnings = records.warnings if warnings is None else warnings
        records = records.records
    at = now or utc_now()
    at = at if at.tzinfo is not None else at.replace(tzinfo=utc_now().tzinfo)
    pairs = records.items() if isinstance(records, Mapping) else ((r.track_id, r) for r in records)
    out: dict[str, Any] = {}
    removed = n_discogs = 0
    expiries: list[str] = []
    for tid, rec in pairs:
        snapshot = ResearchRecord.from_dict(rec.to_dict() if isinstance(rec, ResearchRecord) else dict(rec))
        removed += snapshot.prune_expired(at)
        d = snapshot.to_dict()
        for item in _discogs_values(d):
            item["attribution"] = item.get("attribution") or DISCOGS_ATTRIBUTION
            n_discogs += 1
            if isinstance(item.get("expires_at"), str):
                expiries.append(item["expires_at"])
        out[str(tid)] = d
    doc: dict[str, Any] = {"schema": REPORT_SCHEMA, "generated_at": iso_utc(at), "records": out}
    if stats is not None:
        doc["stats"] = dict(stats)
    if warnings is not None:
        doc["warnings"] = list(warnings)
    parsed = [p for p in (parse_iso(e) for e in expiries) if p is not None]
    doc["discogs"] = {"notice": DISCOGS_NOTICE, "attribution": DISCOGS_ATTRIBUTION,
                      "max_age_hours": DISCOGS_MAX_AGE_S / 3600, "values": n_discogs,
                      "expires_at": iso_utc(min(parsed)) if parsed else None, "removed_expired": removed}
    return jsonable(doc)


def _fmt_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt_value(v) for v in value)
    return clean_text(str(value))


def format_conflict(conflict: Mapping[str, Any], *, now: datetime | None = None) -> str:
    """One console line for a conflict, e.g.
    ``year: musicbrainz=1982; wikidata=1983 (work); discogs=1982 (Data provided by Discogs.)``
    — the Discogs attribution right next to its value. ``(work)``: the Wikidata date is the
    SONG's (found via the work), not this recording's. With ``now``, values that must no
    longer be shown (expired Discogs data) are left out."""
    field_name = _fmt_value(conflict.get("field")) if conflict.get("field") is not None else "?"
    parts = []
    for v in conflict.get("values") or []:
        if not isinstance(v, Mapping) or (now is not None and value_expired(dict(v), now)):
            continue
        source = _fmt_value(v.get("source") or "unknown")
        text = f"{source}={_fmt_value(v.get('value'))}"
        if v.get("scope") == "work":
            text += " (work)"
        if v.get("source") == DISCOGS_SOURCE:
            text += f" ({_fmt_value(v.get('attribution') or DISCOGS_ATTRIBUTION)})"
        parts.append(text)
    return f"{field_name}: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# curator review sheet
# ---------------------------------------------------------------------------

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_cell(value: Any) -> str:
    """Text for a spreadsheet cell; catalogue text starting like a formula is prefixed
    with an apostrophe (catalogue data is user-editable — no formula injection)."""
    if value is None:
        return ""
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    elif isinstance(value, (list, tuple)):
        text = "; ".join(str(v) for v in value)
    else:
        text = str(value)
    text = clean_text(text)          # a file name may carry a lone surrogate: unwritable as UTF-8
    return "'" + text if text.startswith(_FORMULA_PREFIXES) else text


def _fmt_length(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
        return ""
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def export_review_csv(result: EnrichResult, path: Path) -> None:
    """Curator sheet of every candidate of ambiguous and matched tracks (UTF-8 with BOM so
    Excel detects the encoding; atomic write). One row per candidate, best first; the
    ``chosen`` column marks the applied candidate of a match. Raises ValueError for a
    non-``.csv`` path."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"the research review sheet must be a .csv file, got {path.name!r}")
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerow(REVIEW_CSV_COLUMNS)
    records = sorted(result.records.values(), key=lambda r: (r.rel_path.casefold(), r.track_id))
    for rec in records:
        if rec.status not in (STATUS_AMBIGUOUS, STATUS_MATCHED):
            continue
        chosen_id = (rec.chosen or {}).get("id")
        cands = rec.candidates or [{}]
        for rank, cand in enumerate(cands, 1):
            writer.writerow([_csv_cell(v) for v in (
                rec.rel_path, rec.status, rec.match_confidence, rank if cand else None, cand.get("title"),
                cand.get("artist"), cand.get("date"), _fmt_length(cand.get("length_s")), cand.get("disambiguation"),
                cand.get("url"), ("yes" if cand and cand.get("id") == chosen_id else "no") if cand else None,
                cand.get("score"), cand.get("reasons"))])
    atomic_write_text(path, buf.getvalue(), encoding="utf-8-sig")


__all__ = ["CONTACT_ENV", "DEFAULT_MIN_INTERVAL_S", "STAT_KEYS", "REVIEW_CSV_COLUMNS", "ALBUM_TAG_KEYS",
           "YEAR_TAG_KEYS", "REPORT_SCHEMA", "DISCOGS_NOTICE", "EnrichResult", "default_client", "default_sources",
           "enrich_tracks", "export_review_csv", "format_conflict", "report_document", "tag_album", "tag_year"]
