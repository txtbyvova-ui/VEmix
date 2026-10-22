"""Discogs cross-check (optional; OFF unless a personal access token is configured).

Discogs is used only to CROSS-CHECK a recording MusicBrainz already identified: the
earliest year of a Discogs master containing the track, and that master's genres and
styles. Discogs genres/styles are community/editorial classifications, so everything
taken from Discogs is stored as ``editorial`` (never as a fact), each value with the
attribution "Data provided by Discogs." and a link to the Discogs page.

Terms of use honoured here:

* API access only (``https://api.discogs.com``), authenticated with the user's own token
  from the environment (``VEG_DISCOGS_TOKEN``) sent in the ``Authorization`` header —
  never in the URL, never logged (``net.redact_headers``), never cached; a token with
  spaces or line breaks is refused before any request (``valid_token``), so no HTTP-layer
  error message can quote it;
* data older than 6 hours must not be shown and is not stored longer than necessary:
  the research cache keeps Discogs answers for at most 6 hours (never served once expired,
  not even to cache-only runs), and every Discogs value — editorial items and the Discogs
  values of conflicts — carries ``attribution`` and ``expires_at``; reports are written
  without expired values (``enrich.report_document``) and console conflict lines name the
  attribution next to the value (``enrich.format_conflict``);
* reading cached answers needs no token: a cache-only run consults the Discogs cache
  entries that are still valid;
* at most ~1 request per second (``enrich.default_client``: 1.1 s per host).

Searches use ``/database/search?type=master&artist=…&track=…``. Results whose artist does
not match are ignored; Discogs disambiguation suffixes ("Artist (2)") and name-variation
marks ("Artist*") are stripped before comparing.
"""

from __future__ import annotations

import os
import re
from datetime import timedelta
from typing import Any, Mapping

from ..net import HttpClient
from .cache import DISCOGS_MAX_AGE_S, iso_utc, parse_iso
from .matching import artist_similarity, fold
from .models import DISCOGS_ATTRIBUTION, LICENSE_DISCOGS
from .musicbrainz import ResponseShapeError, query_artist, query_title

SOURCE = "discogs"
DISCOGS_API = "https://api.discogs.com"
DISCOGS_SITE = "https://www.discogs.com"
TOKEN_ENV = "VEG_DISCOGS_TOKEN"
SEARCH_PER_PAGE = 10
MIN_ARTIST_SIMILARITY = 0.85

_ARTIST_SUFFIX_RE = re.compile(r"\s*\(\d+\)\s*$")
# a personal access token is printable ASCII without spaces; a line break pasted into the
# variable would make every request fail (and urllib's error would quote the header)
_TOKEN_RE = re.compile(r"[\x21-\x7e]+")


def token_from_env(env: Mapping[str, str] | None = None) -> str | None:
    token = (os.environ if env is None else env).get(TOKEN_ENV, "") or ""
    token = token.strip()
    return token or None


def valid_token(token: str | None) -> bool:
    """True for a plausible token (printable ASCII, no whitespace). The token itself is
    never echoed in any message."""
    return isinstance(token, str) and bool(_TOKEN_RE.fullmatch(token))


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Discogs token={token}"}


def search_key(artist: str, title: str) -> str:
    return f"discogs:search:{query_artist(artist).casefold()}|{query_title(title).casefold()}"


def _clean_artist(name: str) -> str:
    return _ARTIST_SUFFIX_RE.sub("", name.strip().rstrip("*")).strip()


def compact_result(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("id"), int) or isinstance(raw.get("id"), bool):
        return None
    title = raw.get("title") if isinstance(raw.get("title"), str) else ""
    artist, _sep, release_title = title.partition(" - ")
    year = raw.get("year")
    if isinstance(year, str) and year.strip().isascii() and year.strip().isdigit():    # '²'.isdigit() is True
        year = int(year.strip())
    if not (isinstance(year, int) and not isinstance(year, bool) and 1800 < year < 2200):
        year = None
    uri = raw.get("uri") if isinstance(raw.get("uri"), str) else ""
    kind = raw.get("type") if isinstance(raw.get("type"), str) else None
    url = uri if uri.startswith(DISCOGS_SITE + "/") else (f"{DISCOGS_SITE}{uri}" if uri.startswith("/") else None)
    if url is None and kind in ("master", "release"):
        # the attribution must link to Discogs: the canonical page works without the slug
        url = f"{DISCOGS_SITE}/{kind}/{raw['id']}"
    return {
        "id": raw["id"],
        "type": kind,
        "artist": _clean_artist(artist) if release_title else None,
        "title": release_title or title or None,
        "year": year,
        "genres": [g for g in raw.get("genre") or [] if isinstance(g, str) and g] if isinstance(raw.get("genre"), list) else [],
        "styles": [s for s in raw.get("style") or [] if isinstance(s, str) and s] if isinstance(raw.get("style"), list) else [],
        "url": url,
    }


def search(client: HttpClient, token: str, artist: str, title: str) -> dict[str, Any]:
    """Masters by ``artist`` containing a track titled ``title`` (compact results)."""
    data, _resp = client.get_json(f"{DISCOGS_API}/database/search", params={
        "type": "master", "artist": query_artist(artist), "track": query_title(title),
        "per_page": SEARCH_PER_PAGE}, headers=auth_headers(token))
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ResponseShapeError("Discogs search response has no 'results' list")
    return {"results": [r for r in (compact_result(x) for x in data["results"]) if r]}


def editorial(data: dict[str, Any], artist: str) -> list[dict[str, Any]]:
    """Editorial cross-check values from a cached search: the earliest matching master's
    year, genres and styles — each with attribution, link and expiry."""
    retrieved_at = data.get("retrieved_at")
    stored = parse_iso(retrieved_at)
    expires_at = iso_utc(stored + timedelta(seconds=DISCOGS_MAX_AGE_S)) if stored else None
    matching = [r for r in data.get("results") or [] if isinstance(r, dict) and r.get("artist")
                and (artist_similarity(artist, r["artist"]) or 0.0) >= MIN_ARTIST_SIMILARITY]
    dated = [r for r in matching if r.get("year")]
    if not dated:
        return []
    first = min(dated, key=lambda r: (r["year"], fold(r.get("title")), r["id"]))

    def item(field: str, value: Any) -> dict[str, Any]:
        return {"field": field, "value": value, "source": SOURCE, "url": first.get("url"),
                "attribution": DISCOGS_ATTRIBUTION, "license": LICENSE_DISCOGS, "retrieved_at": retrieved_at,
                "expires_at": expires_at}

    out = [item("year", first["year"])]
    if first.get("genres"):
        out.append(item("genre", list(first["genres"])))
    if first.get("styles"):
        out.append(item("style", list(first["styles"])))
    return out


__all__ = ["SOURCE", "DISCOGS_API", "TOKEN_ENV", "token_from_env", "valid_token", "auth_headers", "search_key", "compact_result",
           "search", "editorial"]
