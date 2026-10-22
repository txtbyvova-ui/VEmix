"""Wikidata cross-check of a MusicBrainz identity (CC0).

Wikidata is used ONLY to cross-check a recording that MusicBrainz already identified —
never to identify a track by name (label search would match covers, homonyms and
films). The item is found by its MusicBrainz identifier statement through the
MediaWiki API (CirrusSearch ``haswbstatement``):

1. ``action=query&list=search&srsearch=haswbstatement:P4404=<recording MBID>``
   (P4404 = MusicBrainz recording ID); when there is no such item and the recording
   performs a work, ``haswbstatement:P435=<work MBID>`` (P435 = MusicBrainz work ID —
   that item describes the SONG, not this recording);
2. ``action=wbgetentities&ids=<Q…>&props=claims|labels&languages=en`` for P577
   (publication date) and P136 (genre);
3. ``wbgetentities`` once more for the English labels of the genre items.

The item found by search is verified before use: it must still exist (a merged item is
followed to its redirect target) and carry the MBID statement it was found by — the
search index can lag behind edits. Statements of deprecated rank are ignored;
preferred-rank statements win over normal ones. Dates keep their Wikidata precision ('1983', '1983-11', '1983-11-12'); coarser
than a year (decade, century) is ignored. Several items carrying the same MBID are a
data error: nothing is taken from either (noted for the curator).

The item and its publication date become facts (``crosscheck_facts``). Genre (P136)
statements do NOT: they are crowd-edited classifications — and for an item found via the
work MBID they describe the song (e.g. the original of a cover), not this recording — so
they are user tags (``crosscheck_user_tags``) with their ``scope``, used only where
community tags are explicitly allowed, like MusicBrainz genres.
"""

from __future__ import annotations

import re
from typing import Any

from ..net import HttpClient
from .models import LICENSE_CC0
from .musicbrainz import ResponseShapeError

SOURCE = "wikidata"
WD_API = "https://www.wikidata.org/w/api.php"
WD_SITE = "https://www.wikidata.org"
P_MB_RECORDING = "P4404"
P_MB_WORK = "P435"
P_PUBLICATION_DATE = "P577"
P_GENRE = "P136"
MAX_GENRES = 20

_QID_RE = re.compile(r"^Q[1-9][0-9]*$")
_TIME_RE = re.compile(r"^\+(\d{4,})-(\d{2})-(\d{2})T", re.ASCII)


def item_url(qid: str) -> str:
    return f"{WD_SITE}/wiki/{qid}"


def crosscheck_key(recording_mbid: str, work_mbid: str | None) -> str:
    return f"wd:mbid:{recording_mbid}|{work_mbid or '-'}"


def find_items(client: HttpClient, prop: str, mbid: str) -> list[str]:
    """QIDs of items with ``prop`` = ``mbid`` (normally zero or one)."""
    data, _resp = client.get_json(WD_API, params={
        "action": "query", "list": "search", "srsearch": f"haswbstatement:{prop}={mbid}",
        "srlimit": 5, "srnamespace": 0, "format": "json"})
    if not isinstance(data, dict) or not isinstance(data.get("query"), dict) \
            or not isinstance(data["query"].get("search"), list):
        raise ResponseShapeError("Wikidata search response has no 'query.search' list")
    out: list[str] = []
    for hit in data["query"]["search"]:
        title = hit.get("title") if isinstance(hit, dict) else None
        if isinstance(title, str) and _QID_RE.match(title) and title not in out:
            out.append(title)
    return out


def get_entities(client: HttpClient, ids: list[str], *, props: str) -> dict[str, dict[str, Any]]:
    data, _resp = client.get_json(WD_API, params={
        "action": "wbgetentities", "ids": "|".join(ids), "props": props, "languages": "en", "format": "json"})
    if not isinstance(data, dict) or not isinstance(data.get("entities"), dict):
        raise ResponseShapeError("Wikidata wbgetentities response has no 'entities' object")
    return {k: v for k, v in data["entities"].items() if isinstance(v, dict)}


def _best_statements(claims: Any, prop: str) -> list[dict[str, Any]]:
    stmts = [s for s in (claims or {}).get(prop) or [] if isinstance(s, dict) and s.get("rank") != "deprecated"] \
        if isinstance(claims, dict) else []
    preferred = [s for s in stmts if s.get("rank") == "preferred"]
    return preferred or stmts


def _datavalue(stmt: dict[str, Any]) -> Any:
    snak = stmt.get("mainsnak")
    if not isinstance(snak, dict) or snak.get("snaktype") != "value":
        return None
    dv = snak.get("datavalue")
    return dv.get("value") if isinstance(dv, dict) else None


def parse_time(value: Any) -> tuple[str, int] | None:
    """('1983-11-12', precision) from a Wikidata time value; None for BCE dates, dates
    coarser than a year and malformed values."""
    if not isinstance(value, dict):
        return None
    m = _TIME_RE.match(str(value.get("time") or ""))
    precision = value.get("precision")
    if not m or not isinstance(precision, int) or precision < 9:
        return None
    year, month, day = m.group(1).lstrip("0") or "0", m.group(2), m.group(3)
    if len(year) != 4:
        return None
    if precision == 9 or month == "00":
        return year, 9
    if precision == 10 or day == "00":
        return f"{year}-{month}", 10
    return f"{year}-{month}-{day}", 11


def _resolve_entity(entities: dict[str, dict[str, Any]], qid: str) -> dict[str, Any] | None:
    """The entity returned for ``qid``: under its own id, or — for a merged item that
    now redirects — the target entity whose ``redirects.from`` names it."""
    entity = entities.get(qid)
    if entity is not None:
        return entity
    for candidate in entities.values():
        red = candidate.get("redirects")
        if isinstance(red, dict) and red.get("from") == qid:
            return candidate
    return None


def has_statement(claims: Any, prop: str, value: str) -> bool:
    """True when a non-deprecated ``prop`` statement has the string value ``value``
    (MBIDs compared case-insensitively)."""
    want = value.casefold()
    for stmt in _best_statements(claims, prop):
        v = _datavalue(stmt)
        if isinstance(v, str) and v.strip().casefold() == want:
            return True
    return False


def crosscheck(client: HttpClient, recording_mbid: str, work_mbid: str | None) -> dict[str, Any]:
    """Cross-check data for a MusicBrainz recording (cacheable dict):
    ``{qid, via, label, publication_dates: [{value, precision}], genres: [{id, label}],
    multiple: [qids], rejected, rejected_reason}`` — ``qid`` None when no (unique, verified)
    item exists.

    The search index can lag behind the data, so the item found is VERIFIED: it must exist
    (not deleted) and carry the MBID statement it was found by; otherwise nothing is taken
    from it (``rejected`` names it, ``rejected_reason`` says why)."""
    out: dict[str, Any] = {"qid": None, "via": None, "label": None, "publication_dates": [], "genres": [],
                           "multiple": [], "rejected": None, "rejected_reason": None}
    routes = [(P_MB_RECORDING, recording_mbid)] + ([(P_MB_WORK, work_mbid)] if work_mbid else [])
    found: tuple[str, str, str] | None = None
    for prop, mbid in routes:
        qids = find_items(client, prop, mbid)
        if len(qids) > 1:
            out["multiple"] = qids
            out["via"] = prop
            return out
        if qids:
            found = (qids[0], prop, mbid)
            break
    if found is None:
        return out
    qid, prop, mbid = found
    entity = _resolve_entity(get_entities(client, [qid], props="claims|labels"), qid)
    if entity is None or "missing" in entity:
        out.update(via=prop, rejected=qid, rejected_reason="the item no longer exists")
        return out
    if not has_statement(entity.get("claims"), prop, mbid):
        out.update(via=prop, rejected=qid,
                   rejected_reason=f"the item does not carry {prop}={mbid} (search index out of date)")
        return out
    eid = entity.get("id")
    out["qid"], out["via"] = (eid if isinstance(eid, str) and _QID_RE.match(eid) else qid), prop
    labels = entity.get("labels") if isinstance(entity.get("labels"), dict) else {}
    en = labels.get("en") if isinstance(labels.get("en"), dict) else {}
    out["label"] = en.get("value") if isinstance(en.get("value"), str) else None
    claims = entity.get("claims")
    dates = []
    for stmt in _best_statements(claims, P_PUBLICATION_DATE):
        parsed = parse_time(_datavalue(stmt))
        if parsed and {"value": parsed[0], "precision": parsed[1]} not in dates:
            dates.append({"value": parsed[0], "precision": parsed[1]})
    out["publication_dates"] = sorted(dates, key=lambda d: d["value"])
    genre_ids: list[str] = []
    for stmt in _best_statements(claims, P_GENRE):
        v = _datavalue(stmt)
        gid = v.get("id") if isinstance(v, dict) else None
        if isinstance(gid, str) and _QID_RE.match(gid) and gid not in genre_ids:
            genre_ids.append(gid)
    genre_ids = genre_ids[:MAX_GENRES]
    if genre_ids:
        ents = get_entities(client, genre_ids, props="labels")
        for gid in genre_ids:
            ent_labels = (_resolve_entity(ents, gid) or {}).get("labels")
            lab = ent_labels.get("en") if isinstance(ent_labels, dict) else None
            label = lab.get("value") if isinstance(lab, dict) and isinstance(lab.get("value"), str) else None
            out["genres"].append({"id": gid, "label": label})
    return out


def _item_scope(data: dict[str, Any]) -> tuple[str, str, str]:
    """(route, method, scope) of a ``crosscheck`` result's item."""
    via = data.get("via") or P_MB_RECORDING
    method = ("item describes the recording" if via == P_MB_RECORDING
              else "item describes the musical work (song), not this particular recording")
    # 'work': a date / genre of the SONG — e.g. the original of a cover — not of this recording
    return f"haswbstatement:{via}", method, ("recording" if via == P_MB_RECORDING else "work")


def crosscheck_facts(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """CC0 facts from a ``crosscheck`` result (empty when no unique item): the item and its
    publication date. Genres are not facts (``crosscheck_user_tags``)."""
    qid = data.get("qid")
    if not isinstance(qid, str) or not _QID_RE.match(qid):
        return {}
    url = item_url(qid)
    retrieved_at = data.get("retrieved_at")
    route, method, scope = _item_scope(data)

    def fact(value: Any, used: list[str]) -> dict[str, Any]:
        return {"value": value, "source": SOURCE, "url": url, "retrieved_at": retrieved_at,
                "fields_used": used, "license": LICENSE_CC0, "method": method, "scope": scope}

    facts = {"wikidata_item": fact(qid, [route])}
    dates = [d for d in data.get("publication_dates") or [] if isinstance(d, dict) and d.get("value")]
    if dates:
        earliest = min(dates, key=lambda d: d["value"])
        facts["publication_date_wikidata"] = fact(earliest["value"], [route, P_PUBLICATION_DATE])
    return facts


def genre_tag(name: str, *, url: str | None, retrieved_at: Any, scope: str, method: str | None = None,
              genre_id: str | None = None) -> dict[str, Any]:
    """One Wikidata genre as a user tag (same shape as ``musicbrainz.recording_user_tags``)."""
    tag: dict[str, Any] = {"name": name, "count": None, "kind": "genre", "source": SOURCE, "url": url,
                           "license": LICENSE_CC0, "non_commercial": False, "retrieved_at": retrieved_at,
                           "scope": scope}
    if method:
        tag["method"] = method
    if genre_id:
        tag["id"] = genre_id
    return tag


def crosscheck_user_tags(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Genre (P136) statements of a ``crosscheck`` result's item as USER TAGS (crowd-edited
    classifications, not facts; empty when no unique item). Genres without an English label
    are skipped: a bare item id says nothing."""
    qid = data.get("qid")
    if not isinstance(qid, str) or not _QID_RE.match(qid):
        return []
    _route, method, scope = _item_scope(data)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for g in data.get("genres") or []:
        label = g.get("label") if isinstance(g, dict) else None
        if not isinstance(label, str) or not label.strip() or label.casefold() in seen:
            continue
        seen.add(label.casefold())
        gid = g.get("id") if isinstance(g.get("id"), str) and _QID_RE.match(g["id"]) else None
        out.append(genre_tag(label, url=item_url(qid), retrieved_at=data.get("retrieved_at"), scope=scope,
                             method=method, genre_id=gid))
    return out


def legacy_genre_tags(fact: Any) -> list[dict[str, Any]]:
    """User tags for a ``genre_wikidata`` FACT written by earlier versions (saved reports /
    records): the genres are migrated, never read as facts. Bare item ids are dropped."""
    if not isinstance(fact, dict):
        return []
    values = fact.get("value")
    values = values if isinstance(values, list) else [values]
    url = fact.get("url") if isinstance(fact.get("url"), str) else None
    scope = fact.get("scope") if fact.get("scope") in ("recording", "work") else "recording"
    method = fact.get("method") if isinstance(fact.get("method"), str) else None
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for v in values:
        if isinstance(v, str) and v.strip() and not _QID_RE.match(v.strip()) and v.casefold() not in seen:
            seen.add(v.casefold())
            out.append(genre_tag(v.strip(), url=url, retrieved_at=fact.get("retrieved_at"), scope=scope,
                                 method=method))
    return out


__all__ = ["SOURCE", "WD_API", "P_MB_RECORDING", "P_MB_WORK", "P_PUBLICATION_DATE", "P_GENRE", "item_url",
           "crosscheck_key", "find_items", "get_entities", "has_statement", "parse_time", "crosscheck",
           "crosscheck_facts", "crosscheck_user_tags", "genre_tag", "legacy_genre_tags"]
