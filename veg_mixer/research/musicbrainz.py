"""MusicBrainz Web Service v2 (https://musicbrainz.org/doc/MusicBrainz_API) — the primary
identity source of the research agent.

Endpoints used (``fmt=json``, one request per ~1.1 s per client, descriptive
User-Agent — enforced by ``net.HttpClient`` / ``enrich.default_client``):

* recording search ``/ws/2/recording?query=<Lucene>&limit=10`` with escaped phrases
  ``recording:"…" AND artist:"…"`` and, when the duration is known,
  ``dur:[ms-7000 TO ms+7000]``. A duration-filtered search with no result is repeated
  once without the filter, so the curator still sees candidates (they then fail the
  duration gate). The ``score`` field ranks results within ONE query; it is kept as
  ``relevance`` and never used as a match probability (``matching`` re-ranks);
* recording lookup ``/ws/2/recording/<mbid>?inc=artist-credits+isrcs+releases+
  release-groups+work-rels+genres+tags``;
* ISRC lookup ``/ws/2/isrc/<isrc>`` when the file carries an ISRC (TSRC) tag.

Licensing: MusicBrainz *core data* (recordings, artists, releases, release groups,
works, ISRCs, dates, MBIDs) is CC0 and becomes ``facts``. Tags and genres are
*supplementary data* under CC BY-NC-SA 3.0: they are opinions voted by users, stored as
``user_tags`` with that licence and a ``non_commercial`` flag, never as facts. Ratings
and other supplementary data are not requested.

Everything here works on a compact, validated form of the responses
(``compact_recording``); a response of the wrong shape raises
:class:`ResponseShapeError` (a ``NetDecodeError``), never a bare KeyError/ValueError.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Mapping

from ..net import HttpClient, NetDecodeError, NetHTTPError
from .models import LICENSE_CC0, LICENSE_MB_SUPPLEMENTARY

SOURCE = "musicbrainz"
MB_WS = "https://musicbrainz.org/ws/2"
MB_SITE = "https://musicbrainz.org"
LOOKUP_INC = "artist-credits+isrcs+releases+release-groups+work-rels+genres+tags"
SEARCH_LIMIT = 10
DURATION_WINDOW_MS = 7000
MAX_RELEASES = 25          # kept per recording (lookups return at most 25 anyway)
MAX_USER_TAGS = 25

MBID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
ISRC_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{3}[0-9]{7}$")
# ffprobe tag keys (lower-cased) holding an ISRC
ISRC_TAG_KEYS = ("isrc", "tsrc", "txxx:isrc")
# ffprobe tag keys holding a MusicBrainz RECORDING id: Vorbis/FLAC 'MUSICBRAINZ_TRACKID'
# (Picard's historical name for the recording id), MP4 freeform 'MusicBrainz Track Id'
# and explicit recording-id spellings. Release-TRACK ids ('musicbrainz_releasetrackid',
# 'musicbrainz release track id') identify a track on a release, not a recording, and
# are deliberately not used.
MBID_TAG_KEYS = ("musicbrainz_recordingid", "musicbrainz_recording_id", "musicbrainz recording id",
                 "musicbrainz_trackid", "musicbrainz track id")

_BRACKETS_RE = re.compile(r"\s*[(\[][^)\]]*[)\]]")
_VERSION_TAIL_RE = re.compile(r"\s+[-–—]\s+(?:\d{4}\s+)?(?:remaster(?:ed)?|single|radio|album|edit|mono|stereo)"
                              r"\b.*$", re.IGNORECASE)
_FEAT_RE = re.compile(r"\s*[(\[]\s*(?:feat\.?|ft\.?|featuring)\s.*$|\s+(?:feat\.?|ft\.?|featuring)\s.*$",
                      re.IGNORECASE)
_TAG_SPLIT_RE = re.compile(r"[;/,\s]+")
# ASCII digits only: a str regex's \d also matches e.g. Arabic-Indic digits
_DATE_RE = re.compile(r"^\d{4}(?:-\d{2}(?:-\d{2})?)?$", re.ASCII)


class ResponseShapeError(NetDecodeError):
    """A syntactically valid JSON response whose structure is not what the API documents."""


# ---------------------------------------------------------------------------
# URLs and identifiers
# ---------------------------------------------------------------------------

def recording_url(mbid: str) -> str:
    return f"{MB_SITE}/recording/{mbid}"


def artist_url(mbid: str) -> str:
    return f"{MB_SITE}/artist/{mbid}"


def release_url(mbid: str) -> str:
    return f"{MB_SITE}/release/{mbid}"


def release_group_url(mbid: str) -> str:
    return f"{MB_SITE}/release-group/{mbid}"


def work_url(mbid: str) -> str:
    return f"{MB_SITE}/work/{mbid}"


def normalize_mbid(text: Any) -> str | None:
    s = str(text or "").strip().lower()
    return s if MBID_RE.match(s) else None


def normalize_isrc(text: Any) -> str | None:
    s = re.sub(r"[\s\-]", "", str(text or "")).upper()
    return s if ISRC_RE.match(s) else None


def isrc_from_tags(tags: Mapping[str, Any] | None) -> str | None:
    """First valid ISRC in the file's tags (several values may be separated by ; / , )."""
    for key in ISRC_TAG_KEYS:
        raw = (tags or {}).get(key)
        if raw is None:
            continue
        for part in re.split(r"[;/,]+", str(raw)):
            isrc = normalize_isrc(part)
            if isrc:
                return isrc
    return None


def mbid_from_tags(tags: Mapping[str, Any] | None) -> str | None:
    """MusicBrainz recording MBID from Picard-style tags; None when absent, invalid or
    when the tags name several different recordings."""
    found: list[str] = []
    for key in MBID_TAG_KEYS:
        raw = (tags or {}).get(key)
        if raw is None:
            continue
        for part in _TAG_SPLIT_RE.split(str(raw)):
            mbid = normalize_mbid(part)
            if mbid and mbid not in found:
                found.append(mbid)
    return found[0] if len(found) == 1 else None


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------

def lucene_phrase(text: str) -> str:
    """A Lucene phrase query: only ``\\`` and ``"`` need escaping inside quotes."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def query_title(title: str) -> str:
    """Title as searched: bracketed qualifiers and '- 2008 Remaster' tails removed
    (they rarely appear in the catalogue's title); case and punctuation kept."""
    s = unicodedata.normalize("NFC", str(title)).strip()
    cleaned = " ".join(_VERSION_TAIL_RE.sub("", _BRACKETS_RE.sub("", s)).split())
    return cleaned or " ".join(s.split())


def query_artist(artist: str) -> str:
    """Main credited artist as searched (featured guests dropped)."""
    s = unicodedata.normalize("NFC", str(artist)).strip()
    cleaned = " ".join(_FEAT_RE.sub("", s).split())
    return cleaned or " ".join(s.split())


def search_duration_s(duration_s: float | None) -> int | None:
    """Duration used by the search (whole seconds), None when unknown/implausible."""
    if duration_s is None or not duration_s > 0 or duration_s > 6 * 3600:
        return None
    return int(round(duration_s))


def build_search_query(artist: str, title: str, duration_s: float | None, *, with_duration: bool = True) -> str:
    parts = [f"recording:{lucene_phrase(query_title(title))}", f"artist:{lucene_phrase(query_artist(artist))}"]
    secs = search_duration_s(duration_s) if with_duration else None
    if secs is not None:
        ms = secs * 1000
        parts.append(f"dur:[{max(0, ms - DURATION_WINDOW_MS)} TO {ms + DURATION_WINDOW_MS}]")
    return " AND ".join(parts)


def search_key(artist: str, title: str, duration_s: float | None) -> str:
    """Cache key of a recording search: normalised artist | title | rounded duration —
    exactly the inputs of ``build_search_query``."""
    secs = search_duration_s(duration_s)
    a = query_artist(artist).casefold()
    t = query_title(title).casefold()
    return f"mb:search:{a}|{t}|{secs if secs is not None else '-'}"


def recording_key(mbid: str) -> str:
    return f"mb:recording:{mbid}"


def isrc_key(isrc: str) -> str:
    return f"mb:isrc:{isrc}"


# ---------------------------------------------------------------------------
# response parsing
# ---------------------------------------------------------------------------

def _s(v: Any) -> str | None:
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return None


def _date(v: Any) -> str | None:
    s = _s(v)
    return s if s and _DATE_RE.match(s) else None


def _list(v: Any) -> list:
    return v if isinstance(v, list) else []


def compact_release(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not normalize_mbid(raw.get("id")):
        return None
    rg = raw.get("release-group") if isinstance(raw.get("release-group"), dict) else {}
    return {
        "id": normalize_mbid(raw.get("id")),
        "title": _s(raw.get("title")),
        "status": _s(raw.get("status")),
        "date": _date(raw.get("date")),
        "country": _s(raw.get("country")),
        "rg_id": normalize_mbid(rg.get("id")),
        "rg_title": _s(rg.get("title")),
        "primary_type": _s(rg.get("primary-type")),
        "secondary_types": [t for t in (_s(x) for x in _list(rg.get("secondary-types"))) if t],
    }


def compact_recording(raw: Any) -> dict[str, Any] | None:
    """Validated compact form of a MusicBrainz recording (search result or lookup);
    None when it has no valid id. Unknown / mistyped fields become None / []."""
    if not isinstance(raw, dict):
        return None
    rid = normalize_mbid(raw.get("id"))
    if not rid:
        return None
    credit = []
    for c in _list(raw.get("artist-credit")):
        if not isinstance(c, dict):
            continue
        artist = c.get("artist") if isinstance(c.get("artist"), dict) else {}
        name = _s(c.get("name")) or _s(artist.get("name"))
        if not name:
            continue
        credit.append({"name": name, "id": normalize_mbid(artist.get("id")),
                       "joinphrase": c.get("joinphrase") if isinstance(c.get("joinphrase"), str) else ""})
    phrase = "".join(c["name"] + c["joinphrase"] for c in credit).strip() or None
    length = raw.get("length")
    length_ms = int(length) if isinstance(length, (int, float)) and not isinstance(length, bool) \
        and 0 < length < 24 * 3600 * 1000 else None
    score = raw.get("score")
    if isinstance(score, str) and score.isascii() and score.isdigit() and len(score) <= 6:
        score = int(score)
    # MusicBrainz documents 0..100; anything else (a 400-digit integer would overflow every
    # float conversion downstream) is dropped rather than trusted
    if not (isinstance(score, (int, float)) and not isinstance(score, bool) and -1e6 < score < 1e6):
        score = None
    rec: dict[str, Any] = {
        "id": rid,
        "title": _s(raw.get("title")),
        "length_ms": length_ms,
        "disambiguation": _s(raw.get("disambiguation")),
        "video": raw.get("video") is True,
        "first_release_date": _date(raw.get("first-release-date")),
        "artist_credit": credit,
        "artist_phrase": phrase,
        "isrcs": sorted({i for i in (normalize_isrc(x) for x in _list(raw.get("isrcs"))) if i}),
        "releases": [r for r in (compact_release(x) for x in _list(raw.get("releases"))) if r][:MAX_RELEASES],
    }
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        rec["relevance"] = score
    works = []
    for rel in _list(raw.get("relations")):
        if not isinstance(rel, dict) or rel.get("target-type") != "work" or not isinstance(rel.get("work"), dict):
            continue
        wid = normalize_mbid(rel["work"].get("id"))
        if wid and rel.get("type") in (None, "performance"):
            works.append({"id": wid, "title": _s(rel["work"].get("title"))})
    if works or "relations" in raw:
        rec["works"] = works
    for key in ("tags", "genres"):
        if key in raw:
            items = []
            for t in _list(raw.get(key)):
                if isinstance(t, dict) and _s(t.get("name")):
                    count = t.get("count")
                    items.append({"name": _s(t.get("name")),
                                  "count": int(count) if isinstance(count, int) and not isinstance(count, bool)
                                  else None})
            rec[key] = items
    return rec


def parse_search(data: Any) -> tuple[list[dict[str, Any]], int]:
    """(compact recordings, total count) of a recording-search response."""
    if not isinstance(data, dict) or not isinstance(data.get("recordings"), list):
        raise ResponseShapeError("MusicBrainz search response has no 'recordings' list")
    recs = [r for r in (compact_recording(x) for x in data["recordings"]) if r]
    total = data.get("count")
    return recs, total if isinstance(total, int) and not isinstance(total, bool) else len(recs)


# ---------------------------------------------------------------------------
# requests (raise net.NetError subclasses; 404 = an answer, not a failure)
# ---------------------------------------------------------------------------

def search_recordings(client: HttpClient, artist: str, title: str, duration_s: float | None) -> dict[str, Any]:
    query = build_search_query(artist, title, duration_s)
    data, _resp = client.get_json(f"{MB_WS}/recording", params={"query": query, "limit": SEARCH_LIMIT,
                                                                 "fmt": "json"})
    recs, total = parse_search(data)
    fallback = None
    if not recs and search_duration_s(duration_s) is not None:
        fallback = build_search_query(artist, title, duration_s, with_duration=False)
        data, _resp = client.get_json(f"{MB_WS}/recording", params={"query": fallback, "limit": SEARCH_LIMIT,
                                                                     "fmt": "json"})
        recs, total = parse_search(data)
    return {"query": query, "fallback_query": fallback, "total": total, "recordings": recs}


def lookup_recording(client: HttpClient, mbid: str) -> dict[str, Any]:
    """Compact recording with releases, ISRCs, works, tags and genres;
    ``{"id": mbid, "not_found": True}`` when MusicBrainz has no such recording."""
    # '+' separators are written literally: that is the documented form of ``inc``
    url = f"{MB_WS}/recording/{mbid}?inc={LOOKUP_INC}&fmt=json"
    try:
        data, _resp = client.get_json(url)
    except NetHTTPError as e:
        if e.status == 404:
            return {"id": mbid, "not_found": True}
        raise
    rec = compact_recording(data)      # a merged MBID resolves to the surviving recording
    if rec is None:
        raise ResponseShapeError("MusicBrainz recording lookup returned no valid recording")
    rec.setdefault("works", [])
    rec.setdefault("tags", [])
    rec.setdefault("genres", [])
    return rec


def lookup_isrc(client: HttpClient, isrc: str) -> dict[str, Any]:
    """Recordings carrying ``isrc`` (each marked with it); empty list when unknown."""
    try:
        data, _resp = client.get_json(f"{MB_WS}/isrc/{isrc}", params={"fmt": "json"})
    except NetHTTPError as e:
        if e.status == 404:
            return {"isrc": isrc, "recordings": []}
        raise
    if not isinstance(data, dict) or not isinstance(data.get("recordings"), list):
        raise ResponseShapeError("MusicBrainz ISRC response has no 'recordings' list")
    recs = [r for r in (compact_recording(x) for x in data["recordings"]) if r]
    for r in recs:
        if isrc not in r["isrcs"]:
            r["isrcs"] = sorted({*r["isrcs"], isrc})
    return {"isrc": isrc, "recordings": recs}


# ---------------------------------------------------------------------------
# facts / user tags
# ---------------------------------------------------------------------------

_PRIMARY_RANK = {"album": 0, "ep": 1, "single": 2}


def representative_release(releases: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """The release that best stands for the recording: official first, then without
    secondary types (not a compilation / live / soundtrack), then the earliest date, then
    album > EP > single. A documented heuristic — see ``fields_used`` of the facts."""
    rels = [r for r in releases if isinstance(r, dict) and r.get("id")]
    if not rels:
        return None

    def key(r: dict[str, Any]) -> tuple:
        return ((r.get("status") or "").casefold() != "official",
                bool(r.get("secondary_types")),
                r.get("date") or "9999",
                _PRIMARY_RANK.get((r.get("primary_type") or "").casefold(), 3),
                r.get("title") or "", r["id"])

    return min(rels, key=key)


def _fact(value: Any, *, url: str, retrieved_at: str | None, fields_used: list[str],
          method: str | None = None) -> dict[str, Any]:
    out = {"value": value, "source": SOURCE, "url": url, "retrieved_at": retrieved_at,
           "fields_used": fields_used, "license": LICENSE_CC0}
    if method:
        out["method"] = method
    return out


def recording_facts(rec: dict[str, Any], *, retrieved_at: str | None) -> dict[str, dict[str, Any]]:
    """CC0 facts of a matched recording (compact form, ideally from a lookup)."""
    rid = rec["id"]
    url = recording_url(rid)
    facts: dict[str, dict[str, Any]] = {
        "recording_mbid": _fact(rid, url=url, retrieved_at=retrieved_at, fields_used=["id"]),
    }
    if rec.get("title"):
        facts["recording_title"] = _fact(rec["title"], url=url, retrieved_at=retrieved_at, fields_used=["title"])
    if rec.get("artist_phrase"):
        facts["artist_credit"] = _fact(rec["artist_phrase"], url=url, retrieved_at=retrieved_at,
                                       fields_used=["artist-credit[].name", "artist-credit[].joinphrase"])
    artist_ids = [c["id"] for c in rec.get("artist_credit") or [] if c.get("id")]
    if artist_ids:
        facts["artist_mbid"] = _fact(artist_ids[0], url=artist_url(artist_ids[0]), retrieved_at=retrieved_at,
                                     fields_used=["artist-credit[0].artist.id"])
        if len(artist_ids) > 1:
            facts["artist_mbids"] = _fact(artist_ids, url=url, retrieved_at=retrieved_at,
                                          fields_used=["artist-credit[].artist.id"])
    if rec.get("isrcs"):
        facts["isrc"] = _fact(list(rec["isrcs"]), url=url, retrieved_at=retrieved_at, fields_used=["isrcs"])
    frd = rec.get("first_release_date")
    if frd:
        facts["first_release_date"] = _fact(frd, url=url, retrieved_at=retrieved_at,
                                            fields_used=["first-release-date"])
        facts["year"] = _fact(int(frd[:4]), url=url, retrieved_at=retrieved_at, fields_used=["first-release-date"])
    rep = representative_release(rec.get("releases") or [])
    if rep:
        method = "earliest official release without secondary types (album > EP > single)"
        used = ["releases[].status", "releases[].date", "releases[].release-group.primary-type",
                "releases[].release-group.secondary-types"]
        if rep.get("title"):
            facts["release_title"] = _fact(rep["title"], url=release_url(rep["id"]), retrieved_at=retrieved_at,
                                           fields_used=used + ["releases[].title"], method=method)
        facts["release_mbid"] = _fact(rep["id"], url=release_url(rep["id"]), retrieved_at=retrieved_at,
                                      fields_used=used + ["releases[].id"], method=method)
        if rep.get("rg_id"):
            rg_url = release_group_url(rep["rg_id"])
            if rep.get("rg_title"):
                facts["release_group"] = _fact(rep["rg_title"], url=rg_url, retrieved_at=retrieved_at,
                                               fields_used=used + ["releases[].release-group.title"], method=method)
            facts["release_group_mbid"] = _fact(rep["rg_id"], url=rg_url, retrieved_at=retrieved_at,
                                                fields_used=used + ["releases[].release-group.id"], method=method)
            if rep.get("primary_type"):
                facts["release_group_type"] = _fact(rep["primary_type"], url=rg_url, retrieved_at=retrieved_at,
                                                    fields_used=["releases[].release-group.primary-type"],
                                                    method=method)
    works = rec.get("works") or []
    if works:
        w = works[0]
        facts["work_mbid"] = _fact(w["id"], url=work_url(w["id"]), retrieved_at=retrieved_at,
                                   fields_used=["relations[type=performance].work.id"])
        if w.get("title"):
            facts["work_title"] = _fact(w["title"], url=work_url(w["id"]), retrieved_at=retrieved_at,
                                        fields_used=["relations[type=performance].work.title"])
        if len(works) > 1:   # medley / several songs
            facts["work_mbids"] = _fact([x["id"] for x in works], url=url, retrieved_at=retrieved_at,
                                        fields_used=["relations[type=performance].work.id"])
    return facts


def recording_user_tags(rec: dict[str, Any], *, retrieved_at: str | None) -> list[dict[str, Any]]:
    """MusicBrainz genres and folksonomy tags of a recording: SUPPLEMENTARY data
    (CC BY-NC-SA 3.0, non-commercial), opinions of MusicBrainz users — not facts."""
    url = recording_url(rec["id"])
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for kind, key in (("genre", "genres"), ("tag", "tags")):
        for t in rec.get(key) or []:
            name = t.get("name")
            if not name or (kind, name.casefold()) in seen:
                continue
            seen.add((kind, name.casefold()))
            out.append({"name": name, "count": t.get("count"), "kind": kind, "source": SOURCE, "url": url,
                        "license": LICENSE_MB_SUPPLEMENTARY, "non_commercial": True, "retrieved_at": retrieved_at})
    out.sort(key=lambda x: (x["kind"] != "genre", -(x["count"] or 0), x["name"].casefold()))
    return out[:MAX_USER_TAGS]


__all__ = ["SOURCE", "MB_WS", "MB_SITE", "LOOKUP_INC", "SEARCH_LIMIT", "ISRC_TAG_KEYS", "MBID_TAG_KEYS",
           "ResponseShapeError", "recording_url", "artist_url", "release_url", "release_group_url", "work_url",
           "normalize_mbid", "normalize_isrc", "isrc_from_tags", "mbid_from_tags", "lucene_phrase",
           "query_title", "query_artist", "build_search_query", "search_key", "recording_key", "isrc_key",
           "compact_recording", "compact_release", "parse_search", "search_recordings", "lookup_recording",
           "lookup_isrc", "representative_release", "recording_facts", "recording_user_tags"]
