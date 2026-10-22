"""Music Research Agent (VEmix 3.0, spec §3): matching, provenance, licences, conflicts,
cache modes and TTLs, network failure handling, rate limiting and secrets.

No live network: every request goes to ``FakeServer``, a transport answering with canned
JSON shaped like real MusicBrainz WS2 / Wikidata API / Discogs API responses. Time
(rate limiting, Retry-After) and "now" (cache expiry) are injected.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import io
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from veg_mixer.models import AudioProbe, TrackFeatures
from veg_mixer.net import HttpResponse, NetTimeout
from veg_mixer.research import (
    DISCOGS_ATTRIBUTION,
    LICENSE_CC0,
    LICENSE_MB_SUPPLEMENTARY,
    RESEARCH_CACHE_NAME,
    RESEARCH_MODES,
    STAT_KEYS,
    EnrichResult,
    ResearchCache,
    ResearchRecord,
    default_client,
    default_sources,
    enrich_tracks,
    export_review_csv,
    format_conflict,
    report_document,
)
from veg_mixer.research import enrich as research_enrich, matching, musicbrainz as mb, wikidata as wd
from veg_mixer.research.cache import RESEARCH_CACHE_SCHEMA

# ---------------------------------------------------------------------------
# canned catalogue data (fictional MBIDs, real response shapes)
# ---------------------------------------------------------------------------

MJ = "f27ec8db-af05-4f36-916e-3d57f91ecf5e"
OTHER_ARTIST = "0383dadf-2a4e-4d10-a46a-e9e041da8eb3"
REC_ORIG = "11111111-1111-4111-8111-111111111111"
REC_LIVE = "22222222-2222-4222-8222-222222222222"
REC_360 = "33333333-3333-4333-8333-333333333333"
REC_TWIN = "44444444-4444-4444-8444-444444444444"
REC_OTHER = "55555555-5555-4555-8555-555555555555"
REL_ALBUM = "aaaaaaaa-0000-4000-8000-000000000001"
REL_SINGLE = "aaaaaaaa-0000-4000-8000-000000000002"
REL_COMP = "aaaaaaaa-0000-4000-8000-000000000003"
REL_LIVE = "aaaaaaaa-0000-4000-8000-000000000004"
REL_360 = "aaaaaaaa-0000-4000-8000-000000000005"
RG_ALBUM = "bbbbbbbb-0000-4000-8000-000000000001"
RG_SINGLE = "bbbbbbbb-0000-4000-8000-000000000002"
RG_COMP = "bbbbbbbb-0000-4000-8000-000000000003"
RG_LIVE = "bbbbbbbb-0000-4000-8000-000000000004"
RG_360 = "bbbbbbbb-0000-4000-8000-000000000005"
WORK = "cccccccc-0000-4000-8000-000000000001"
ISRC = "USSM19902991"
QID = "Q209283"
T0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)


def mb_release(rid, title, date, rg_id, *, primary="Album", secondary=(), status="Official"):
    return {"id": rid, "status-id": "4e304316-386d-3409-af2e-78857eec5cfe", "count": 1, "title": title,
            "status": status, "date": date, "country": "US",
            "release-group": {"id": rg_id, "type-id": "f529b476-6e62-324f-b0aa-1f3e33d313fc", "title": title,
                              "primary-type": primary, "secondary-types": list(secondary)},
            "track-count": 9}


def mb_recording(rid, title, length_ms, *, artist="Michael Jackson", artist_id=MJ, disambiguation=None,
                 frd=None, releases=(), isrcs=None, score=None, video=None, joinphrase=None, extra=None):
    credit = {"name": artist, "artist": {"id": artist_id, "name": artist, "sort-name": artist}}
    if joinphrase is not None:
        credit["joinphrase"] = joinphrase
    rec = {"id": rid, "title": title, "length": length_ms, "video": video, "artist-credit": [credit],
           "releases": list(releases)}
    if score is not None:
        rec["score"] = score
    if disambiguation is not None:
        rec["disambiguation"] = disambiguation
    if frd is not None:
        rec["first-release-date"] = frd
    if isrcs is not None:
        rec["isrcs"] = list(isrcs)
    rec.update(extra or {})
    return rec


ALBUM = mb_release(REL_ALBUM, "Thriller", "1982-11-30", RG_ALBUM)
COMP = mb_release(REL_COMP, "Number Ones", "2003-11-17", RG_COMP, secondary=("Compilation",))
SINGLE = mb_release(REL_SINGLE, "Thriller", "1983-11-11", RG_SINGLE, primary="Single")

S_ORIG = mb_recording(REC_ORIG, "Thriller", 357266, disambiguation="", frd="1982-11-30", releases=[ALBUM, COMP],
                      isrcs=[ISRC], score=100)
S_LIVE = mb_recording(REC_LIVE, "Thriller", 360500, disambiguation="live, 1992-10-01: Bucharest, Romania",
                      frd="2004-11-16", score=100,
                      releases=[mb_release(REL_LIVE, "Live in Bucharest", "2004-11-16", RG_LIVE, secondary=("Live",))])
S_360 = mb_recording(REC_360, "Thriller", 358000, disambiguation="360 Reality Audio mix", frd="2021-03-12",
                     score=100, releases=[mb_release(REL_360, "Thriller", "2021-03-12", RG_360)])
S_TWIN = mb_recording(REC_TWIN, "Thriller", 357300, disambiguation="", frd="1982-11-30", score=100,
                      releases=[mb_release("aaaaaaaa-0000-4000-8000-000000000009", "Thriller", "1982-11-30",
                                           "bbbbbbbb-0000-4000-8000-000000000009")])

LOOKUP_ORIG = {
    **mb_recording(REC_ORIG, "Thriller", 357266, disambiguation="", frd="1982-11-30",
                   releases=[COMP, SINGLE, ALBUM], isrcs=[ISRC], video=False, joinphrase=""),
    "relations": [{"type": "performance", "type-id": "a3005666-a872-32c3-ad06-98af558e99b0",
                   "target-type": "work", "direction": "forward", "attributes": [],
                   "work": {"id": WORK, "title": "Thriller", "disambiguation": "", "type": "Song",
                            "language": "eng"}}],
    "genres": [{"name": "pop", "count": 4, "id": "911c7bbb-172d-4df8-9478-dbff4296e791", "disambiguation": ""},
               {"name": "funk", "count": 2, "id": "a1cb5c33-12a7-4dfb-9e6c-3b1c0d3c0a2b", "disambiguation": ""}],
    "tags": [{"name": "halloween", "count": 3}, {"name": "pop", "count": 4}],
}


def wd_time(value, precision=11, rank="normal"):
    return {"mainsnak": {"snaktype": "value", "property": "P577",
                         "datavalue": {"value": {"time": value, "timezone": 0, "before": 0, "after": 0,
                                                 "precision": precision,
                                                 "calendarmodel": "http://www.wikidata.org/entity/Q1985727"},
                                       "type": "time"}, "datatype": "time"},
            "type": "statement", "id": f"{QID}$x", "rank": rank}


def wd_item_claim(prop, qid, rank="normal"):
    return {"mainsnak": {"snaktype": "value", "property": prop,
                         "datavalue": {"value": {"entity-type": "item", "numeric-id": int(qid[1:]), "id": qid},
                                       "type": "wikibase-entityid"}, "datatype": "wikibase-item"},
            "type": "statement", "rank": rank}


def wd_external_id(prop, value, rank="normal"):
    return {"mainsnak": {"snaktype": "value", "property": prop,
                         "datavalue": {"value": value, "type": "string"}, "datatype": "external-id"},
            "type": "statement", "rank": rank}


WD_THRILLER = {"type": "item", "id": QID, "labels": {"en": {"language": "en", "value": "Thriller"}},
               "claims": {"P4404": [wd_external_id("P4404", REC_ORIG)],
                          "P577": [wd_time("+1983-11-12T00:00:00Z")],
                          "P136": [wd_item_claim("P136", "Q37073"), wd_item_claim("P136", "Q164444"),
                                   wd_item_claim("P136", "Q11399", rank="deprecated")]}}
WD_GENRES = {"Q37073": {"type": "item", "id": "Q37073", "labels": {"en": {"language": "en", "value": "pop music"}}},
             "Q164444": {"type": "item", "id": "Q164444", "labels": {"en": {"language": "en", "value": "funk"}}},
             "Q11399": {"type": "item", "id": "Q11399", "labels": {"en": {"language": "en", "value": "rock music"}}}}

DISCOGS_RESULTS = [
    {"id": 99999, "type": "master", "master_id": 99999, "uri": "/master/99999-Various-Halloween-Hits",
     "title": "Various - Halloween Hits", "year": "1975", "genre": ["Pop"], "style": ["Novelty"],
     "resource_uri": "https://api.discogs.com/masters/99999"},
    {"id": 8883, "type": "master", "master_id": 8883, "uri": "/master/8883-Michael-Jackson-Thriller",
     "title": "Michael Jackson - Thriller", "country": "US", "year": "1982", "genre": ["Funk / Soul", "Pop"],
     "style": ["Disco", "Synth-pop"], "resource_uri": "https://api.discogs.com/masters/8883"},
    {"id": 51002, "type": "master", "master_id": 51002, "uri": "/master/51002-Michael-Jackson-Number-Ones",
     "title": "Michael Jackson - Number Ones", "year": "2003", "genre": ["Pop"], "style": ["Ballad"]},
]
SECRET = "DiscogsSecretToken-123456"


# ---------------------------------------------------------------------------
# fake transport / clock
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self) -> None:
        self.t = 5000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _json(data, status=200, headers=None, url=""):
    return HttpResponse(status, {"content-type": "application/json", **(headers or {})},
                        json.dumps(data).encode("utf-8"), url)


class FakeServer:
    """Answers like musicbrainz.org/ws/2, www.wikidata.org/w/api.php and api.discogs.com."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[dict] = []
        self.mb_search: list[tuple[str, list[dict]]] = []      # (substring of the Lucene query, recordings)
        self.mb_recordings: dict[str, dict] = {}
        self.mb_isrc: dict[str, list[dict]] = {}
        self.wd_search: dict[str, list[str]] = {}
        self.wd_entities: dict[str, dict] = {}
        self.discogs_results: list[dict] = []
        self.scripted: dict[str, list] = {}       # host -> queue of HttpResponse | Exception
        self.always: dict[str, object] = {}       # host -> HttpResponse | Exception for every call

    def hosts(self) -> list[str]:
        return [c["host"] for c in self.calls]

    def __call__(self, method, url, headers, body, timeout):
        parts = urlsplit(url)
        params = {k: v[0] for k, v in parse_qs(parts.query).items()}
        self.calls.append({"method": method, "url": url, "host": parts.netloc, "path": parts.path,
                           "params": params, "headers": dict(headers), "t": self.clock.t})
        queue = self.scripted.get(parts.netloc)
        if queue:
            return self._emit(queue.pop(0), url)
        if parts.netloc in self.always:
            return self._emit(self.always[parts.netloc], url)
        if parts.netloc == "musicbrainz.org":
            return self._musicbrainz(parts.path, params, url)
        if parts.netloc == "www.wikidata.org":
            return self._wikidata(params, url)
        if parts.netloc == "api.discogs.com":
            if not headers.get("Authorization", "").startswith("Discogs token="):
                return _json({"message": "You must authenticate to access this resource."}, 401, url=url)
            return _json({"pagination": {"page": 1, "pages": 1, "per_page": 10,
                                         "items": len(self.discogs_results), "urls": {}},
                          "results": self.discogs_results}, url=url)
        raise AssertionError(f"unexpected host {parts.netloc}")

    @staticmethod
    def _emit(item, url):
        if isinstance(item, Exception):
            raise item
        return item

    def _musicbrainz(self, path, params, url):
        assert params.get("fmt") == "json"
        if path == "/ws/2/recording":
            q = params["query"]
            dur = re.search(r"dur:\[(\d+) TO (\d+)\]", q)
            for needle, recs in self.mb_search:
                if needle in q:
                    if dur:     # like MusicBrainz: recordings outside the window (or without length) drop out
                        lo, hi = int(dur.group(1)), int(dur.group(2))
                        recs = [r for r in recs if not isinstance(r, dict) or not isinstance(r.get("length"), int)
                                or lo <= r["length"] <= hi]
                    return _json({"created": "2026-10-22T10:00:00.000Z", "count": len(recs), "offset": 0,
                                  "recordings": recs}, url=url)
            return _json({"created": "2026-10-22T10:00:00.000Z", "count": 0, "offset": 0, "recordings": []}, url=url)
        if path.startswith("/ws/2/recording/"):
            rid = path.rsplit("/", 1)[1]
            assert params.get("inc") == "artist-credits isrcs releases release-groups work-rels genres tags"
            if rid in self.mb_recordings:
                return _json(self.mb_recordings[rid], url=url)
            return _json({"error": "Not Found", "help": "For usage, please see: https://musicbrainz.org/development/mmd"},
                         404, url=url)
        if path.startswith("/ws/2/isrc/"):
            isrc = path.rsplit("/", 1)[1]
            if isrc in self.mb_isrc:
                return _json({"isrc": isrc, "recordings": self.mb_isrc[isrc]}, url=url)
            return _json({"error": "Not Found"}, 404, url=url)
        raise AssertionError(f"unexpected MusicBrainz path {path}")

    def _wikidata(self, params, url):
        assert params.get("format") == "json"
        if params.get("action") == "query":
            qids = self.wd_search.get(params["srsearch"], [])
            return _json({"batchcomplete": "", "query": {"searchinfo": {"totalhits": len(qids)},
                                                         "search": [{"ns": 0, "title": q, "pageid": 1000 + i,
                                                                     "size": 5000, "wordcount": 0,
                                                                     "snippet": "", "timestamp": "2026-09-01T00:00:00Z"}
                                                                    for i, q in enumerate(qids)]}}, url=url)
        if params.get("action") == "wbgetentities":
            ents = {}
            for qid in params["ids"].split("|"):
                ents[qid] = self.wd_entities.get(qid, {"id": qid, "missing": ""})
            return _json({"entities": ents, "success": 1}, url=url)
        raise AssertionError(f"unexpected Wikidata action {params}")


class Env:
    """Fake server + client + cache + injectable clock/now for one test."""

    def __init__(self, tmp_path: Path, *, max_retries: int = 3) -> None:
        self.tmp_path = tmp_path
        self.clock = FakeClock()
        self.server = FakeServer(self.clock)
        self.now_value = T0
        self.client = dataclasses.replace(default_client("tests@example.invalid"), transport=self.server,
                                          sleep=self.clock.sleep, clock=self.clock.monotonic,
                                          max_retries=max_retries)
        self.cache_path = tmp_path / RESEARCH_CACHE_NAME
        self.cache = self.new_cache()

    def now(self) -> datetime:
        return self.now_value

    def new_cache(self, **kw) -> ResearchCache:
        return ResearchCache(self.cache_path, clock=self.now, **kw)

    def run(self, tracks, **kw) -> EnrichResult:
        kw.setdefault("mode", "online")
        kw.setdefault("cache", self.cache)
        kw.setdefault("client", self.client)
        kw.setdefault("now", self.now)
        return enrich_tracks(tracks, **kw)

    def thriller_catalogue(self, search=(S_360, S_LIVE, S_ORIG)) -> "Env":
        self.server.mb_search.append(('recording:"Thriller" AND artist:"Michael Jackson"', list(search)))
        self.server.mb_recordings[REC_ORIG] = LOOKUP_ORIG
        self.server.wd_search[f"haswbstatement:P4404={REC_ORIG}"] = [QID]
        self.server.wd_entities.update({QID: WD_THRILLER, **WD_GENRES})
        return self


def make_track(artist="Michael Jackson", title="Thriller", duration=357.6, *, rel="Michael Jackson - Thriller.mp3",
               source="id3") -> TrackFeatures:
    sources = {}
    if artist:
        sources["artist"] = source
    if title:
        sources["title"] = source
    return TrackFeatures(path=Path("C:/library") / rel, duration_s=duration, rel_path=rel, artist=artist, title=title,
                         bpm=118.0, camelot="8A", energy=0.7, feature_sources={**sources, "bpm": "auto:bpm-v1"})


def probe_for(track: TrackFeatures, tags: dict) -> AudioProbe:
    return AudioProbe(path=track.path, duration_s=track.duration_s, sample_rate=44100, channels=2, codec="mp3",
                      tags=dict(tags))


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


@pytest.fixture(autouse=True)
def _no_discogs_token_in_env(monkeypatch):
    # sources=None reads VEG_DISCOGS_TOKEN (default_sources): a developer's real token must not
    # change what the tests request; tests that need it set it themselves
    monkeypatch.delenv("VEG_DISCOGS_TOKEN", raising=False)


# ---------------------------------------------------------------------------
# matching and provenance
# ---------------------------------------------------------------------------

def test_clear_match_applies_facts_with_provenance(env):
    env.thriller_catalogue(search=[S_ORIG])
    t = make_track()
    before = copy.deepcopy(t.to_dict())
    res = env.run([t])
    rec = res.records[t.track_id]
    assert rec.status == "matched" and rec.applied
    assert rec.chosen == {"source": "musicbrainz", "id": REC_ORIG, "url": f"https://musicbrainz.org/recording/{REC_ORIG}"}
    assert rec.match_confidence >= 0.85
    assert rec.query == {"artist": "Michael Jackson", "title": "Thriller", "duration_s": 357.6, "isrc": None,
                         "mbid": None, "album": None, "year": None}
    for name in ("recording_mbid", "artist_mbid", "isrc", "first_release_date", "year", "release_title",
                 "release_group", "release_group_mbid", "work_mbid"):
        f = rec.facts[name]
        assert f["source"] == "musicbrainz" and f["license"] == LICENSE_CC0, name
        assert f["url"].startswith("https://musicbrainz.org/"), name
        assert f["retrieved_at"] == "2026-10-22T10:00:00+00:00", name
        assert f["fields_used"] and all(isinstance(x, str) for x in f["fields_used"]), name
    assert rec.facts["artist_mbid"]["value"] == MJ and rec.facts["artist_mbid"]["url"].endswith(f"/artist/{MJ}")
    assert rec.facts["isrc"]["value"] == [ISRC]
    assert rec.facts["first_release_date"]["value"] == "1982-11-30" and rec.facts["year"]["value"] == 1982
    # representative release: the earliest official original album, not the compilation / single
    assert rec.facts["release_mbid"]["value"] == REL_ALBUM and rec.facts["release_group_type"]["value"] == "Album"
    assert rec.facts["work_mbid"]["value"] == WORK
    # wikidata cross-check facts (CC0)
    assert rec.facts["wikidata_item"]["value"] == QID
    assert rec.facts["publication_date_wikidata"]["value"] == "1983-11-12"
    # Wikidata genres are crowd-edited classifications: user tags, never facts (R2)
    assert "genre_wikidata" not in rec.facts
    wd_genres = [u for u in rec.user_tags if u["source"] == "wikidata"]
    assert [u["name"] for u in wd_genres] == ["pop music", "funk"]                # deprecated rank ignored
    assert all(u["url"] == f"https://www.wikidata.org/wiki/{QID}" and u["kind"] == "genre" for u in wd_genres)
    # the audio features are untouched: research never writes into TrackFeatures
    assert t.to_dict() == before
    assert rec.retrieved_at == "2026-10-22T10:00:00+00:00"
    # JSON-native, no NaN
    json.dumps(res.to_dict(), allow_nan=False)
    assert ResearchRecord.from_dict(json.loads(json.dumps(rec.to_dict()))).to_dict() == rec.to_dict()
    assert res.stats["matched"] == 1 and set(res.stats) == set(STAT_KEYS)
    # requests: search, lookup, wikidata search + entities + genre labels
    assert [c["host"] for c in env.server.calls] == ["musicbrainz.org"] * 2 + ["www.wikidata.org"] * 3
    assert res.stats["network_requests"] == 5


def test_musicbrainz_tags_are_user_tags_not_facts(env):
    env.thriller_catalogue(search=[S_ORIG])
    rec = env.run([make_track()]).records[make_track().track_id]
    mb_tags = [u for u in rec.user_tags if u["source"] == "musicbrainz"]
    names = {(u["kind"], u["name"]) for u in mb_tags}
    assert names == {("genre", "pop"), ("genre", "funk"), ("tag", "halloween"), ("tag", "pop")}
    for u in mb_tags:
        assert u["license"] == LICENSE_MB_SUPPLEMENTARY == "CC BY-NC-SA 3.0"
        assert u["non_commercial"] is True and u["url"]
    assert {u["source"] for u in rec.user_tags} == {"musicbrainz", "wikidata"}
    assert rec.user_tags[0] == {**rec.user_tags[0], "kind": "genre", "name": "pop", "count": 4}
    # no MusicBrainz genre / tag became a fact; no popularity / tempo / key from outside
    assert not any(k in rec.facts for k in ("genre", "mood", "tags", "familiarity", "popularity", "bpm", "camelot"))
    assert all(f["license"] == LICENSE_CC0 for f in rec.facts.values())


def test_conflicting_years_are_recorded(env):
    env.thriller_catalogue(search=[S_ORIG])
    rec = env.run([make_track()]).records[make_track().track_id]
    assert rec.conflicts == [{"field": "year", "values": [
        {"source": "musicbrainz", "value": 1982, "url": f"https://musicbrainz.org/recording/{REC_ORIG}"},
        {"source": "wikidata", "value": 1983, "url": f"https://www.wikidata.org/wiki/{QID}"}]}]
    # both values stay available as facts; nothing was resolved silently
    assert rec.fact("year") == 1982 and rec.fact("publication_date_wikidata") == "1983-11-12"


def test_isrc_conflict_between_file_tag_and_catalogue(env):
    env.thriller_catalogue(search=[S_ORIG])
    env.server.mb_isrc["GBAAA0000001"] = []            # unknown ISRC -> search by name
    t = make_track()
    rec = env.run([t], probes={t.track_id: probe_for(t, {"isrc": "GB-AAA-00-00001"})}).records[t.track_id]
    assert rec.status == "matched"
    assert {"field": "isrc", "values": [{"source": "file_tag", "value": "GBAAA0000001", "url": None},
                                        {"source": "musicbrainz", "value": [ISRC],
                                         "url": f"https://musicbrainz.org/recording/{REC_ORIG}"}]} in rec.conflicts


def test_thriller_like_case_picks_original_not_live_or_360(env):
    env.thriller_catalogue(search=[S_360, S_LIVE, S_ORIG])      # all three: MusicBrainz score 100
    t = make_track()
    rec = env.run([t]).records[t.track_id]
    assert rec.status == "matched" and rec.chosen["id"] == REC_ORIG
    by_id = {c["id"]: c for c in rec.candidates}
    assert set(by_id) == {REC_ORIG, REC_LIVE, REC_360}
    assert rec.candidates[0]["id"] == REC_ORIG
    assert all(c["relevance"] == 100 for c in rec.candidates)
    assert by_id[REC_ORIG]["score"] - max(by_id[REC_LIVE]["score"], by_id[REC_360]["score"]) >= 0.08
    assert any("'live'" in r for r in by_id[REC_LIVE]["reasons"])
    assert any("'360'" in r for r in by_id[REC_360]["reasons"])
    assert any("not a probability" in r for r in by_id[REC_ORIG]["reasons"])


def test_thriller_without_original_never_applies_live_version(env):
    env.thriller_catalogue(search=[S_LIVE, S_360])
    t = make_track()
    rec = env.run([t]).records[t.track_id]
    assert rec.status in ("ambiguous", "not_found")
    assert rec.chosen is None and rec.facts == {} and rec.user_tags == [] and rec.editorial == []
    assert rec.conflicts == []
    # nothing beyond the search was requested for a non-match
    assert [c["path"] for c in env.server.calls] == ["/ws/2/recording"]


def test_local_live_title_prefers_the_live_recording(env):
    env.thriller_catalogue(search=[S_ORIG, S_LIVE])
    env.server.mb_search.insert(0, ('recording:"Thriller" AND artist:"Michael Jackson"', [S_ORIG, S_LIVE]))
    env.server.mb_recordings[REC_LIVE] = {**S_LIVE, "relations": [], "genres": [], "tags": []}
    t = make_track(title="Thriller (Live)", duration=360.2)
    rec = env.run([t], sources=["musicbrainz"]).records[t.track_id]
    assert rec.candidates[0]["id"] == REC_LIVE


def test_ambiguous_match_applies_nothing(env):
    env.thriller_catalogue(search=[S_ORIG, S_TWIN])            # two equally plausible originals
    t = make_track()
    res = env.run([t])
    rec = res.records[t.track_id]
    assert rec.status == "ambiguous" and res.stats["ambiguous"] == 1
    assert rec.chosen is None and not rec.applied
    assert rec.facts == {} and rec.user_tags == [] and rec.editorial == [] and rec.conflicts == []
    assert {c["id"] for c in rec.candidates} == {REC_ORIG, REC_TWIN}
    assert any("margin" in n for n in rec.notes)
    assert rec.match_confidence is not None                     # the best candidate's score, for the curator
    assert [c["path"] for c in env.server.calls] == ["/ws/2/recording"]   # no lookup, no wikidata


def test_duration_outside_tolerance_is_not_applied(env):
    env.thriller_catalogue(search=[S_ORIG])
    t = make_track(duration=311.0)                               # a radio edit, 46 s shorter
    rec = env.run([t]).records[t.track_id]
    assert rec.status in ("ambiguous", "not_found") and rec.facts == {}
    # the search used a duration window, found nothing, then searched without it
    queries = [c["params"]["query"] for c in env.server.calls]
    assert "dur:[304000 TO 318000]" in queries[0] and "dur:" not in queries[1]


def test_filename_identity_lowers_confidence(env):
    env.thriller_catalogue(search=[S_ORIG])
    tag_rec = env.run([make_track()], sources=["musicbrainz"]).records[make_track().track_id]
    fn_track = make_track(rel="other/Michael Jackson - Thriller.mp3", source="filename")
    fn_rec = env.run([fn_track], sources=["musicbrainz"]).records[fn_track.track_id]
    assert fn_rec.match_confidence < tag_rec.match_confidence
    assert fn_rec.match_confidence == pytest.approx(tag_rec.match_confidence * 0.92, abs=1e-3)
    assert any("parsed from the file name" in r for r in fn_rec.candidates[0]["reasons"])


def test_tracks_without_identity_are_skipped(env):
    t = make_track(artist=None, title=None, rel="untitled.mp3")
    res = env.run([t])
    assert res.records[t.track_id].status == "skipped" and res.stats["skipped"] == 1
    assert env.server.calls == []


def test_mode_off_does_nothing(env):
    res = env.run([make_track()], mode="off")
    assert res.records == {} and env.server.calls == [] and all(v == 0 for v in res.stats.values())
    assert RESEARCH_MODES == ("off", "cache", "online")


def test_invalid_arguments(env):
    with pytest.raises(ValueError, match="mode"):
        env.run([make_track()], mode="offline")
    with pytest.raises(ValueError, match="unknown research source"):
        env.run([make_track()], sources=["musicbrainz", "lastfm"])
    with pytest.raises(ValueError, match="musicbrainz"):
        env.run([make_track()], sources=["wikidata"])


# ---------------------------------------------------------------------------
# identifier tags
# ---------------------------------------------------------------------------

def test_isrc_tag_path_uses_isrc_lookup_not_search(env):
    env.thriller_catalogue()
    env.server.mb_isrc[ISRC] = [{"id": REC_ORIG, "title": "Thriller", "length": 357266, "disambiguation": "",
                                 "video": False}]
    t = make_track()
    res = env.run([t], probes={t.track_id: probe_for(t, {"tsrc": "us-sm1-99-02991"})}, sources=["musicbrainz"])
    rec = res.records[t.track_id]
    assert rec.query["isrc"] == ISRC
    assert rec.status == "matched" and rec.chosen["id"] == REC_ORIG
    assert [c["path"] for c in env.server.calls] == [f"/ws/2/isrc/{ISRC}", f"/ws/2/recording/{REC_ORIG}"]
    assert any("ISRC" in r for r in rec.candidates[0]["reasons"])


def test_isrc_route_without_recording_details_is_not_applied(env):
    # the ISRC result has no artist credit; without the lookup the artist is never verified
    env.server.mb_isrc[ISRC] = [{"id": REC_ORIG, "title": "Thriller", "length": 357266, "video": False}]
    t = make_track()
    rec = env.run([t], probes={t.track_id: probe_for(t, {"isrc": ISRC})}, sources=["musicbrainz"]).records[t.track_id]
    assert rec.status == "ambiguous" and rec.facts == {} and rec.chosen is None
    assert any("could not be verified" in n for n in rec.notes)


def test_isrc_pointing_to_another_artist_is_not_applied(env):
    env.server.mb_isrc[ISRC] = [{"id": REC_OTHER, "title": "Thriller", "length": 357000, "video": False}]
    env.server.mb_recordings[REC_OTHER] = {**mb_recording(REC_OTHER, "Thriller", 357000, artist="Tribute Band",
                                                          artist_id=OTHER_ARTIST, frd="2010"),
                                           "relations": [], "genres": [], "tags": []}
    t = make_track()
    rec = env.run([t], probes={t.track_id: probe_for(t, {"isrc": ISRC})}, sources=["musicbrainz"]).records[t.track_id]
    assert rec.status == "ambiguous" and rec.facts == {} and rec.chosen is None
    assert any("Tribute Band" in n for n in rec.notes)


def test_mbid_tag_path_is_a_direct_lookup(env):
    env.thriller_catalogue()
    t = make_track()
    res = env.run([t], probes={t.track_id: probe_for(t, {"musicbrainz_trackid": REC_ORIG.upper(),
                                                         "musicbrainz_releasetrackid": REC_TWIN})})
    rec = res.records[t.track_id]
    assert rec.query["mbid"] == REC_ORIG
    assert rec.status == "matched" and rec.match_confidence == pytest.approx(0.99)
    assert [c["path"] for c in env.server.calls if c["host"] == "musicbrainz.org"] == [f"/ws/2/recording/{REC_ORIG}"]
    assert rec.facts["recording_mbid"]["value"] == REC_ORIG


def test_unknown_mbid_tag_falls_back_to_search(env):
    env.thriller_catalogue(search=[S_ORIG])
    t = make_track()
    rec = env.run([t], probes={t.track_id: probe_for(t, {"musicbrainz recording id": REC_TWIN})},
                  sources=["musicbrainz"]).records[t.track_id]
    assert rec.status == "matched" and rec.chosen["id"] == REC_ORIG
    assert any("has no recording" in n for n in rec.notes)
    assert [c["path"] for c in env.server.calls] == [f"/ws/2/recording/{REC_TWIN}", "/ws/2/recording",
                                                      f"/ws/2/recording/{REC_ORIG}"]


def test_tag_helpers():
    assert mb.isrc_from_tags({"tsrc": "bad; US-SM1-99-02991"}) == ISRC
    assert mb.isrc_from_tags({"isrc": "nonsense"}) is None
    assert mb.mbid_from_tags({"musicbrainz_trackid": f"{REC_ORIG}/{REC_TWIN}"}) is None     # two recordings
    assert mb.mbid_from_tags({"musicbrainz track id": f" {REC_ORIG} "}) == REC_ORIG
    assert mb.mbid_from_tags({"musicbrainz_releasetrackid": REC_ORIG}) is None


# ---------------------------------------------------------------------------
# cache modes and TTL
# ---------------------------------------------------------------------------

def test_cache_only_makes_zero_transport_calls(env):
    env.thriller_catalogue(search=[S_ORIG])
    t = make_track()
    res = env.run([t], mode="cache")
    assert env.server.calls == []
    assert res.records[t.track_id].status == "not_cached" and res.stats["not_cached"] == 1
    # after an online run, the cache-only run reproduces the record without any request
    online = env.run([t])
    env.cache.save()
    n_calls = len(env.server.calls)
    offline = env.run([t], mode="cache", cache=env.new_cache(read_only=True))
    assert len(env.server.calls) == n_calls
    assert offline.records[t.track_id].to_dict() == online.records[t.track_id].to_dict()
    assert offline.stats["cache_hits"] == 1 and offline.stats["network_requests"] == 0


def test_online_second_run_is_all_cache_hits(env):
    env.thriller_catalogue(search=[S_360, S_LIVE, S_ORIG])
    tracks = [make_track(), make_track(rel="dup/copy.mp3"), make_track(artist=None, title=None, rel="x.mp3")]
    first = env.run(tracks)
    env.cache.save()
    assert first.stats["network_requests"] > 0
    calls = len(env.server.calls)
    env.clock.t += 3600
    second = env.run(tracks, cache=env.new_cache())
    assert len(env.server.calls) == calls                      # zero requests
    assert second.stats["network_requests"] == 0
    assert second.stats["cache_hits"] == 2 and second.stats["skipped"] == 1
    for t in tracks:
        assert second.records[t.track_id].to_dict() == first.records[t.track_id].to_dict()
    # the duplicate copy of the same song shared the first answer (keys are by query, not by file)
    assert first.stats["cache_hits"] == 1


def test_checkpoint_saves_during_long_runs(env):
    env.thriller_catalogue(search=[S_ORIG])
    env.run([make_track()], sources=["musicbrainz"], checkpoint_every=1)
    assert env.cache_path.is_file() and not env.cache.dirty           # saved without an explicit save()
    other = Env(env.tmp_path / "b")
    other.thriller_catalogue(search=[S_ORIG])
    other.run([make_track()], sources=["musicbrainz"], checkpoint_every=0)
    assert not other.cache_path.exists() and other.cache.dirty


def test_cache_is_keyed_by_query_not_file(env):
    env.thriller_catalogue(search=[S_ORIG])
    env.run([make_track()], sources=["musicbrainz"])
    keys = set(env.cache._entries)
    assert f"mb:search:michael jackson|thriller|358" in keys and f"mb:recording:{REC_ORIG}" in keys
    assert not any("library" in k or ".mp3" in k for k in keys)


def test_discogs_ttl_six_hours(env):
    env.thriller_catalogue(search=[S_ORIG])
    env.server.discogs_results = DISCOGS_RESULTS
    t = make_track()
    sources = ["musicbrainz", "wikidata", "discogs"]
    rec = env.run([t], sources=sources, discogs_token=SECRET).records[t.track_id]
    ed = {e["field"]: e for e in rec.editorial}
    assert ed["year"]["value"] == 1982 and ed["genre"]["value"] == ["Funk / Soul", "Pop"]
    assert ed["style"]["value"] == ["Disco", "Synth-pop"]
    for e in rec.editorial:
        assert e["source"] == "discogs" and e["attribution"] == DISCOGS_ATTRIBUTION == "Data provided by Discogs."
        assert e["url"] == "https://www.discogs.com/master/8883-Michael-Jackson-Thriller"
        assert e["expires_at"] == "2026-10-22T16:00:00+00:00"
    assert not any(k.startswith("discogs") or "style" in k for k in rec.facts)      # editorial, not facts
    # a Discogs search request carried the token in the header, never in the URL
    dcall = next(c for c in env.server.calls if c["host"] == "api.discogs.com")
    assert dcall["headers"]["Authorization"] == f"Discogs token={SECRET}" and SECRET not in dcall["url"]
    env.cache.save()

    env.now_value = T0 + timedelta(hours=5)
    rec5 = env.run([t], mode="cache", sources=sources, discogs_token=SECRET,
                   cache=env.new_cache(read_only=True)).records[t.track_id]
    assert {e["field"] for e in rec5.editorial} == {"year", "genre", "style"}

    env.now_value = T0 + timedelta(hours=7)
    cache7 = env.new_cache()
    rec7 = env.run([t], mode="cache", sources=sources, discogs_token=SECRET, cache=cache7).records[t.track_id]
    assert rec7.status == "matched" and rec7.editorial == []         # expired Discogs data is not shown
    assert rec7.fact("recording_mbid") == REC_ORIG                    # MusicBrainz facts are still valid
    assert any("discogs" in n for n in rec7.notes)
    cache7.save()                                                     # ... and is not kept on disk
    data = json.loads(env.cache_path.read_text("utf-8"))
    assert not any(k.startswith("discogs:") for k in data["entries"])
    assert any(k.startswith("mb:") for k in data["entries"])


def test_saved_record_prunes_expired_discogs_values(env):
    env.thriller_catalogue(search=[S_ORIG])
    env.server.discogs_results = DISCOGS_RESULTS
    t = make_track()
    rec = env.run([t], sources=["musicbrainz", "wikidata", "discogs"], discogs_token=SECRET).records[t.track_id]
    year = next(c for c in rec.conflicts if c["field"] == "year")
    assert [v["source"] for v in year["values"]] == ["musicbrainz", "wikidata", "discogs"]
    assert year["values"][2]["expires_at"] == "2026-10-22T16:00:00+00:00"
    assert year["values"][2]["attribution"] == DISCOGS_ATTRIBUTION                # R5: next to the value
    saved = ResearchRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert saved.prune_expired(T0 + timedelta(hours=5)) == 0
    assert saved.prune_expired(T0 + timedelta(hours=7)) == 4          # 3 editorial values + 1 conflict value
    assert saved.editorial == []
    assert saved.conflicts == [{"field": "year", "values": year["values"][:2]}]   # MB vs Wikidata stays


def test_discogs_needs_a_token(env, monkeypatch):
    monkeypatch.delenv("VEG_DISCOGS_TOKEN", raising=False)
    env.thriller_catalogue(search=[S_ORIG])
    res = env.run([make_track()], sources=["musicbrainz", "discogs"])
    assert any("VEG_DISCOGS_TOKEN" in w for w in res.warnings)
    assert "api.discogs.com" not in env.server.hosts()
    assert default_sources({}) == ("musicbrainz", "wikidata")
    assert default_sources({"VEG_DISCOGS_TOKEN": "x"}) == ("musicbrainz", "wikidata", "discogs")


# ---------------------------------------------------------------------------
# network failures degrade, never raise
# ---------------------------------------------------------------------------

def test_503_with_retry_after_then_success(env):
    env.thriller_catalogue(search=[S_ORIG])
    env.server.scripted["musicbrainz.org"] = [_json({"error": "Your requests are exceeding the allowable rate limit."},
                                                    503, headers={"retry-after": "3"})]
    t = make_track()
    res = env.run([t], sources=["musicbrainz"])
    assert res.records[t.track_id].status == "matched"
    assert 3.0 in env.clock.sleeps                           # Retry-After honoured with the injected sleep
    assert env.client.stats["retries"] == 1


def test_429_exhaustion_gives_error_status_not_exception(tmp_path):
    env = Env(tmp_path, max_retries=2)
    env.server.always["musicbrainz.org"] = _json({"error": "rate limited"}, 429)
    t = make_track()
    res = env.run([t])
    rec = res.records[t.track_id]
    assert rec.status == "error" and res.stats["errors"] == 1
    assert any("HTTP 429" in e for e in rec.errors)
    assert len(env.server.calls) == 3
    assert not env.cache.dirty                                # errors are never cached


def test_timeout_gives_error_status(tmp_path):
    env = Env(tmp_path, max_retries=1)
    env.server.always["musicbrainz.org"] = NetTimeout("timed out after 20 s", retryable=True)
    t = make_track()
    rec = env.run([t]).records[t.track_id]
    assert rec.status == "error" and any("timed out" in e for e in rec.errors)


@pytest.mark.parametrize("response", [
    HttpResponse(200, {"content-type": "text/html"}, b"<html><body>Bad gateway</body></html>", ""),
    HttpResponse(200, {}, b'{"recordings": "not-a-list"}', ""),
    HttpResponse(200, {}, b'{"created": "x", "count": 1, "recordings": [{"id": 5}]', ""),     # truncated
])
def test_malformed_responses_give_error_status(tmp_path, response):
    env = Env(tmp_path, max_retries=0)
    env.server.always["musicbrainz.org"] = response
    t = make_track()
    rec = env.run([t]).records[t.track_id]
    assert rec.status == "error"
    assert rec.errors and rec.errors[0].startswith("musicbrainz:")


def test_malformed_items_are_ignored_not_fatal(env):
    env.server.mb_search.append(("Thriller", [{"id": "not-a-uuid"}, "junk", None,
                                              {**S_ORIG, "artist-credit": "oops", "releases": "x", "length": "long"}]))
    t = make_track()
    rec = env.run([t], sources=["musicbrainz"]).records[t.track_id]
    assert rec.status in ("ambiguous", "not_found", "matched")
    assert rec.status != "error"
    assert [c["id"] for c in rec.candidates] == [REC_ORIG]


def test_circuit_breaker_stops_requests(tmp_path):
    env = Env(tmp_path, max_retries=0)
    env.server.always["musicbrainz.org"] = _json({"error": "boom"}, 500)
    tracks = [make_track(title=f"Song {i}", rel=f"s{i}.mp3") for i in range(6)]
    res = env.run(tracks, max_consecutive_errors=2)
    assert len(env.server.calls) == 2                       # nothing after the breaker opened
    statuses = [res.records[t.track_id].status for t in tracks]
    assert statuses == ["error"] * 6
    later = res.records[tracks[3].track_id].errors[0]
    assert "circuit breaker" in later and "2 consecutive" in later
    assert any("circuit breaker" in w for w in res.warnings)


def test_breaker_still_serves_cached_tracks(env):
    env.thriller_catalogue(search=[S_ORIG])
    cached = make_track()
    env.run([cached], sources=["musicbrainz"])
    env.server.always["musicbrainz.org"] = _json({"error": "boom"}, 500)
    env.client.max_retries = 0
    tracks = [make_track(title="Song A", rel="a.mp3"), make_track(title="Song B", rel="b.mp3"), cached,
              make_track(title="Song C", rel="c.mp3")]
    res = env.run(tracks, sources=["musicbrainz"], max_consecutive_errors=2)
    assert res.records[cached.track_id].status == "matched"
    assert res.records[tracks[3].track_id].status == "error"


def test_wikidata_failure_keeps_the_musicbrainz_match(tmp_path):
    env = Env(tmp_path, max_retries=0)
    env.thriller_catalogue(search=[S_ORIG])
    env.server.always["www.wikidata.org"] = _json({"error": {"code": "maxlag"}}, 503)
    rec = env.run([make_track()]).records[make_track().track_id]
    assert rec.status == "matched" and rec.fact("recording_mbid") == REC_ORIG
    assert "wikidata_item" not in rec.facts and any(e.startswith("wikidata:") for e in rec.errors)


def test_progress_callback(env):
    env.thriller_catalogue(search=[S_ORIG])
    seen = []
    env.run([make_track(), make_track(artist=None, title=None, rel="n.mp3")], sources=["musicbrainz"],
            progress=lambda i, n, msg: seen.append((i, n, msg)))
    assert seen == [(1, 2, "Michael Jackson - Thriller: matched"), (2, 2, "n: skipped")]


# ---------------------------------------------------------------------------
# rate limiting, User-Agent, secrets
# ---------------------------------------------------------------------------

def test_rate_limiting_spacing_per_host(env):
    env.thriller_catalogue(search=[S_ORIG])
    env.server.mb_search.append(("Billie", [mb_recording(REC_OTHER, "Billie Jean", 294000, frd="1982-11-30",
                                                          releases=[ALBUM], score=100)]))
    env.server.mb_recordings[REC_OTHER] = {**mb_recording(REC_OTHER, "Billie Jean", 294000, frd="1982-11-30",
                                                          releases=[ALBUM]), "relations": [], "genres": [], "tags": []}
    env.run([make_track(), make_track(title="Billie Jean", duration=294.2, rel="bj.mp3")])
    for host, gap in (("musicbrainz.org", 1.1), ("www.wikidata.org", 1.0)):
        times = [c["t"] for c in env.server.calls if c["host"] == host]
        assert len(times) >= 2
        assert all(b - a >= gap - 1e-9 for a, b in zip(times, times[1:])), host
    assert sum(env.clock.sleeps) > 0                         # the injected sleep, never a real one


def test_user_agent_present_and_no_secret_in_logs_or_files(env, caplog, monkeypatch):
    monkeypatch.setenv("VEG_DISCOGS_TOKEN", SECRET)
    env.thriller_catalogue(search=[S_ORIG])
    env.server.discogs_results = DISCOGS_RESULTS
    env.server.scripted["api.discogs.com"] = [_json({"message": "Server error"}, 500)]
    with caplog.at_level(logging.DEBUG):
        res = env.run([make_track()], sources=default_sources())
    env.cache.save()
    for c in env.server.calls:
        ua = c["headers"]["User-Agent"]
        assert ua.startswith("VEmix/") and ua.endswith("( tests@example.invalid )")
        if c["host"] != "api.discogs.com":
            assert "Authorization" not in c["headers"]
    assert any(c["host"] == "api.discogs.com" for c in env.server.calls)
    assert SECRET not in caplog.text
    assert SECRET not in env.cache_path.read_text("utf-8")
    assert SECRET not in json.dumps(res.to_dict())
    assert res.records[make_track().track_id].editorial                 # retried after the 500


def test_default_client_contact_and_intervals(monkeypatch):
    monkeypatch.setenv("VEG_CONTACT", "curator@example.invalid")
    c = default_client()
    assert c.user_agent.endswith("( curator@example.invalid )")
    assert c.min_interval_s["musicbrainz.org"] >= 1.1 and c.min_interval_s["www.wikidata.org"] >= 1.0
    assert c.min_interval_s["api.discogs.com"] >= 1.0
    monkeypatch.delenv("VEG_CONTACT")
    assert "github.com" in default_client().user_agent
    assert default_client("me@example.invalid", timeout_s=5).timeout_s == 5


# ---------------------------------------------------------------------------
# research cache file
# ---------------------------------------------------------------------------

def test_cache_roundtrip_schema_and_noop_save(tmp_path):
    p = tmp_path / RESEARCH_CACHE_NAME
    c = ResearchCache(p, clock=lambda: T0)
    c.save()
    assert not p.exists()                                     # nothing to write
    c.put("mb:recording:x", {"a": 1}, source="musicbrainz")
    c.save()
    data = json.loads(p.read_text("utf-8"))
    assert data["schema"] == RESEARCH_CACHE_SCHEMA == "vegroove.research_cache" and data["schema_version"] == 1
    e = data["entries"]["mb:recording:x"]
    assert e["source"] == "musicbrainz" and e["value"] == {"a": 1} and e["stored_at"] == "2026-10-22T10:00:00+00:00"
    mtime = p.stat().st_mtime_ns
    c2 = ResearchCache(p, clock=lambda: T0)
    assert c2.get("mb:recording:x") == {"a": 1}
    c2.save()
    assert p.stat().st_mtime_ns == mtime                      # unchanged -> no write
    with pytest.raises(ValueError):
        c2.put("k", {"bad": float("nan")}, source="musicbrainz")
    with pytest.raises(TypeError):
        c2.put("k", {"x": object()}, source="musicbrainz")
    with pytest.raises(TypeError):
        c2.put("k", ["not", "a", "dict"], source="musicbrainz")
    assert "k" not in c2
    with pytest.raises(ValueError):
        c2.put("k", {"a": 1}, source="musicbrainz", ttl_s=0)


def test_cache_ttl_expiry_and_discogs_cap(tmp_path):
    p = tmp_path / RESEARCH_CACHE_NAME
    c = ResearchCache(p, clock=lambda: T0)
    c.put("wd:x", {"q": 1}, source="wikidata", ttl_s=60)
    c.put("discogs:x", {"r": 1}, source="discogs", ttl_s=30 * 24 * 3600)     # capped to 6 h
    assert c.get("wd:x", now=T0 + timedelta(seconds=59)) == {"q": 1}
    assert c.get("wd:x", now=T0 + timedelta(seconds=61)) is None
    assert c.get("discogs:x", now=T0 + timedelta(hours=5, minutes=59)) == {"r": 1}
    assert c.get("discogs:x", now=T0 + timedelta(hours=6)) is None
    assert c.entry_info("discogs:x")["expires_at"] == "2026-10-22T16:00:00+00:00"


def test_cache_never_touches_foreign_files(tmp_path):
    p = tmp_path / "veg_analysis_cache.json"
    original = json.dumps({"schema": "vegroove.analysis_cache", "schema_version": 1, "entries": {}})
    p.write_text(original, "utf-8")
    c = ResearchCache(p)
    assert any("not a VEmix research cache" in w for w in c.warnings)
    c.put("k", {"a": 1}, source="musicbrainz")
    c.save()
    assert p.read_text("utf-8") == original and not list(tmp_path.glob("*.corrupt-*"))
    with pytest.raises(ValueError):
        ResearchCache(tmp_path / "song.mp3")
    with pytest.raises(ValueError):
        ResearchCache(tmp_path / "research.csv")


def test_corrupt_own_cache_is_moved_aside(tmp_path):
    p = tmp_path / RESEARCH_CACHE_NAME
    p.write_text('{"schema": "vegroove.research_cache", "schema_version": 1, "entries": {"k": {"sou', "utf-8")
    ro = ResearchCache(p, read_only=True)
    assert p.exists() and any("would be moved aside" in w for w in ro.warnings)
    c = ResearchCache(p)
    assert not p.exists() and (tmp_path / f"{RESEARCH_CACHE_NAME}.corrupt-1").exists()
    assert any("corrupt-1" in w for w in c.warnings)
    c.put("k", {"a": 1}, source="musicbrainz")
    c.save()
    assert json.loads(p.read_text("utf-8"))["entries"]["k"]["value"] == {"a": 1}


def test_malformed_entries_dropped_and_read_only_never_writes(tmp_path):
    p = tmp_path / RESEARCH_CACHE_NAME
    good = {"source": "musicbrainz", "stored_at": "2026-10-22T09:00:00+00:00",
            "expires_at": "2027-01-01T00:00:00+00:00", "value": {"a": 1}}
    p.write_text(json.dumps({"schema": RESEARCH_CACHE_SCHEMA, "schema_version": 1,
                             "entries": {"good": good, "bad": {"source": "x"}, "worse": 3}}), "utf-8")
    before = p.read_text("utf-8")
    ro = ResearchCache(p, read_only=True, clock=lambda: T0)
    assert ro.get("good") == {"a": 1} and ro.get("bad") is None
    assert any("dropped malformed entry" in w for w in ro.warnings)
    ro.put("new", {"b": 2}, source="wikidata")
    ro.save()
    assert p.read_text("utf-8") == before
    rw = ResearchCache(p, clock=lambda: T0)
    rw.save()
    assert set(json.loads(p.read_text("utf-8"))["entries"]) == {"good"}


# ---------------------------------------------------------------------------
# review sheet / record model / matcher units
# ---------------------------------------------------------------------------

def test_export_review_csv(env, tmp_path):
    env.thriller_catalogue(search=[S_ORIG, S_TWIN])                  # ambiguous
    env.server.mb_search.insert(0, ("Billie", [mb_recording(REC_OTHER, "=HYPERLINK(\"x\")", 294000, frd="1982",
                                                             releases=[ALBUM], score=90)]))
    t1, t2 = make_track(), make_track(title="Billie Jean", duration=294.0, rel="bj.mp3")
    res = env.run([t1, t2, make_track(artist=None, title=None, rel="z.mp3")], sources=["musicbrainz"])
    out = tmp_path / "review" / "research_review.csv"
    export_review_csv(res, out)
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in raw
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    assert list(rows[0])[:10] == ["rel_path", "status", "match_confidence", "candidate_rank", "title", "artist",
                                  "date", "length", "disambiguation", "url"]
    amb = [r for r in rows if r["rel_path"] == t1.rel_path]
    assert [r["candidate_rank"] for r in amb] == ["1", "2"] and {r["status"] for r in amb} == {"ambiguous"}
    assert amb[0]["length"] == "5:57" and amb[0]["url"].startswith("https://musicbrainz.org/recording/")
    assert all(r["rel_path"] != "z.mp3" for r in rows)              # skipped tracks are not listed
    titles = [r["title"] for r in rows]
    assert not any(tt.startswith("=") for tt in titles)              # no formula injection
    with pytest.raises(ValueError):
        export_review_csv(res, tmp_path / "review.xlsx")
    # a file name with a lone surrogate (possible on NTFS) does not make the export fail
    rec = res.records[t1.track_id]
    res.records[t1.track_id] = dataclasses.replace(rec, rel_path="bad\udc80name.mp3")
    export_review_csv(res, out)
    assert "bad�name.mp3" in out.read_text("utf-8-sig")


def test_record_from_dict_is_tolerant():
    rec = ResearchRecord.from_dict({"track_id": "t", "rel_path": "a.mp3", "status": "matched",
                                    "match_confidence": float("nan"), "facts": {"year": {"value": 1982}, "x": 3},
                                    "candidates": [{"id": "1"}, "junk"], "errors": ["e", 5],
                                    "conflicts": [{"field": "year", "values": True},
                                                  {"field": "isrc", "values": [{"value": 1}, {"value": 2}]}]})
    assert rec.match_confidence is None and set(rec.facts) == {"year"} and rec.fact("year") == 1982
    assert rec.candidates == [{"id": "1"}] and rec.errors == ["e"]
    assert rec.prune_expired(T0) == 0                           # a mistyped 'values' does not raise
    assert rec.conflicts == [{"field": "isrc", "values": [{"value": 1}, {"value": 2}]}]
    with pytest.raises(ValueError):
        ResearchRecord.from_dict({"status": "certain"})


def test_matcher_units():
    local = matching.LocalIdentity(artist="Michael Jackson", title="Thriller", duration_s=357.6)
    recs = [mb.compact_recording(r) for r in (S_360, S_LIVE, S_ORIG)]
    scored = matching.score_recordings(local, recs)
    assert scored[0].id == REC_ORIG
    decision = matching.decide(local, scored)
    assert decision.status == "matched" and decision.chosen.id == REC_ORIG
    # version markers
    assert matching.marker_counts("360 Reality Audio mix") == {"360": 1, "mix": 1}
    assert matching.marker_counts("Original Mix") == {} and matching.marker_counts("2011 Remaster") == {}
    assert matching.marker_counts("Extended Remix") == {"remix": 1}
    assert matching.core_title("Thriller - Live") == "thriller" == matching.core_title("Thriller (2008 Remaster)")
    # similarities
    assert matching.title_similarity("Don't Stop 'Til You Get Enough", "Don’t Stop ’Til You Get Enough") == 1.0
    assert matching.artist_similarity("Beyonce", "Beyoncé") == 1.0
    assert matching.artist_similarity("Beatles", "The Beatles") == 1.0
    duet = ["Michael Jackson", "Paul McCartney"]
    assert matching.artist_similarity("Michael Jackson", "Michael Jackson & Paul McCartney", duet) == 0.95
    assert matching.artist_similarity("Paul McCartney", "Michael Jackson & Paul McCartney", duet) == 0.85
    assert matching.artist_similarity("Paul McCartney, Michael Jackson", "Michael Jackson & Paul McCartney",
                                      duet) == 0.9
    # a band name is never split: 'Fire' is not 'Earth, Wind & Fire' and vice versa
    assert matching.artist_similarity("Fire", "Earth, Wind & Fire", ["Earth, Wind & Fire"]) < 0.8
    assert matching.artist_similarity("Earth, Wind & Fire", "Fire", ["Fire"]) < 0.8
    assert matching.artist_similarity("Earth Wind and Fire", "Earth, Wind & Fire", ["Earth, Wind & Fire"]) == 1.0
    assert matching.duration_similarity(357.6, 357.3, 7.0) == 1.0
    assert matching.duration_similarity(None, 357.3, 7.0) is None
    # Lucene escaping
    assert mb.build_search_query('AC/DC', 'Say "Hi" \\ (Live)', None) == \
        'recording:"Say \\"Hi\\" \\\\" AND artist:"AC/DC"'


def test_wikidata_parsing_units():
    assert wd.parse_time({"time": "+1983-11-12T00:00:00Z", "precision": 11}) == ("1983-11-12", 11)
    assert wd.parse_time({"time": "+1983-00-00T00:00:00Z", "precision": 9}) == ("1983", 9)
    assert wd.parse_time({"time": "+1983-11-00T00:00:00Z", "precision": 10}) == ("1983-11", 10)
    assert wd.parse_time({"time": "+1980-00-00T00:00:00Z", "precision": 8}) is None     # decade
    assert wd.parse_time({"time": "-0500-00-00T00:00:00Z", "precision": 9}) is None
    claims = {"P577": [wd_time("+1990-01-01T00:00:00Z", rank="deprecated"), wd_time("+1983-11-12T00:00:00Z"),
                       wd_time("+1984-01-01T00:00:00Z", rank="preferred")]}
    assert [wd.parse_time(wd._datavalue(s)) for s in wd._best_statements(claims, "P577")] == [("1984-01-01", 11)]
    assert wd.crosscheck_facts({"qid": None}) == {}


# ---------------------------------------------------------------------------
# regressions found in review
# ---------------------------------------------------------------------------

def test_wikidata_item_is_verified_before_use(tmp_path):
    # (a) the search index names an item that no longer exists -> no wikidata fact at all
    env = Env(tmp_path / "a")
    env.thriller_catalogue(search=[S_ORIG])
    del env.server.wd_entities[QID]
    rec = env.run([make_track()]).records[make_track().track_id]
    assert rec.status == "matched" and not any(f["source"] == "wikidata" for f in rec.facts.values())
    assert any(f"item {QID} not used" in n and "no longer exists" in n for n in rec.notes)
    # (b) the item exists but does not carry P4404=<recording> (stale index) -> not used
    env = Env(tmp_path / "b")
    env.thriller_catalogue(search=[S_ORIG])
    env.server.wd_entities[QID] = {**WD_THRILLER, "claims": {**WD_THRILLER["claims"],
                                                             "P4404": [wd_external_id("P4404", REC_TWIN)]}}
    rec = env.run([make_track()]).records[make_track().track_id]
    assert "wikidata_item" not in rec.facts and "genre_wikidata" not in rec.facts
    assert any("does not carry P4404" in n for n in rec.notes)
    assert not any(c["field"] == "year" for c in rec.conflicts)        # nothing to conflict with
    # (c) a merged item: wbgetentities answers with the redirect target
    env = Env(tmp_path / "c")
    env.thriller_catalogue(search=[S_ORIG])
    target = {**WD_THRILLER, "id": "Q999", "redirects": {"from": QID, "to": "Q999"}}
    orig_wikidata = env.server._wikidata

    def redirecting(params, url):
        if params.get("action") == "wbgetentities" and params["ids"] == QID:
            return _json({"entities": {"Q999": target}, "success": 1}, url=url)
        return orig_wikidata(params, url)
    env.server._wikidata = redirecting
    rec = env.run([make_track()]).records[make_track().track_id]
    assert rec.fact("wikidata_item") == "Q999" and rec.facts["wikidata_item"]["url"].endswith("/Q999")


def test_wikidata_work_route_is_scoped_as_the_song(env):
    # no item for the recording; the SONG's item (P435 = work MBID) is used and labelled as such
    env.thriller_catalogue(search=[S_ORIG])
    del env.server.wd_search[f"haswbstatement:P4404={REC_ORIG}"]
    env.server.wd_search[f"haswbstatement:P435={WORK}"] = [QID]
    env.server.wd_entities[QID] = {**WD_THRILLER, "claims": {**WD_THRILLER["claims"],
                                                             "P4404": [], "P435": [wd_external_id("P435", WORK)]}}
    rec = env.run([make_track()]).records[make_track().track_id]
    assert rec.facts["publication_date_wikidata"]["scope"] == "work"
    assert "musical work" in rec.facts["publication_date_wikidata"]["method"]
    year = next(c for c in rec.conflicts if c["field"] == "year")
    assert {"source": "wikidata", "value": 1983, "url": f"https://www.wikidata.org/wiki/{QID}",
            "scope": "work"} in year["values"]


def test_malformed_discogs_token_is_never_used_or_echoed(env, caplog):
    env.thriller_catalogue(search=[S_ORIG])
    env.server.discogs_results = DISCOGS_RESULTS
    bad = "SECRETTOKEN\nX-Injected: 1"
    with caplog.at_level(logging.DEBUG):
        res = env.run([make_track()], sources=["musicbrainz", "discogs"], discogs_token=bad)
    assert "api.discogs.com" not in env.server.hosts()
    assert any("contains spaces, line breaks" in w for w in res.warnings)
    assert "SECRETTOKEN" not in json.dumps(res.to_dict()) and "SECRETTOKEN" not in caplog.text
    assert res.records[make_track().track_id].status == "matched"


def test_discogs_attribution_link_without_uri():
    from veg_mixer.research import discogs as dg
    r = dg.compact_result({"id": 8883, "type": "master", "title": "Michael Jackson - Thriller", "year": "1982"})
    assert r["url"] == "https://www.discogs.com/master/8883"
    # a foreign absolute link is not taken as the Discogs attribution
    r = dg.compact_result({"id": 1, "type": "release", "title": "A - B", "uri": "https://evil.example/x"})
    assert r["url"] == "https://www.discogs.com/release/1"
    assert dg.compact_result({"id": 2, "title": "A - B", "year": "19²2"})["year"] is None   # no ValueError
    assert dg.valid_token("abcDEF123") and not dg.valid_token("abc def") and not dg.valid_token("a\nb")


def test_caller_client_without_intervals_is_still_rate_limited(env):
    env.thriller_catalogue(search=[S_ORIG])
    env.server.mb_search.append(("Billie", [mb_recording(REC_OTHER, "Billie Jean", 294000, frd="1982-11-30",
                                                          releases=[ALBUM], score=100)]))
    env.client.min_interval_s.clear()                          # e.g. a bare HttpClient(...)
    env.run([make_track(), make_track(title="Billie Jean", duration=294.2, rel="bj.mp3")], sources=["musicbrainz"])
    times = [c["t"] for c in env.server.calls if c["host"] == "musicbrainz.org"]
    assert len(times) >= 3 and all(b - a >= 1.1 - 1e-9 for a, b in zip(times, times[1:]))
    # intervals are only ever raised, never lowered
    env.client.min_interval_s["musicbrainz.org"] = 2.0
    env.run([make_track(title="New", rel="n.mp3")], sources=["musicbrainz"])
    assert env.client.min_interval_s["musicbrainz.org"] == 2.0


def test_non_ascii_contact_does_not_break_the_user_agent(monkeypatch):
    monkeypatch.setenv("VEG_CONTACT", "Роман <curator@example.invalid>\n")
    ua = default_client().user_agent
    assert ua.isascii() and ua.endswith("( <curator@example.invalid> )")
    ua.encode("latin-1")                                       # what http.client requires
    monkeypatch.setenv("VEG_CONTACT", "Роман")
    assert "github.com" in default_client().user_agent         # nothing left -> project URL


def test_non_finite_durations_never_pass_the_duration_gate(env):
    local = matching.LocalIdentity(artist="Michael Jackson", title="Thriller", duration_s=float("nan"))
    scored = matching.score_recordings(local, [mb.compact_recording(S_ORIG)])
    assert scored[0].components["duration"] is None and scored[0].duration_diff_s is None
    assert matching.decide(local, scored).status == "ambiguous"
    assert matching.duration_similarity(float("inf"), 357.0, 7.0) is None
    assert matching.duration_similarity(357.0, float("nan"), 7.0) is None
    env.thriller_catalogue(search=[S_ORIG])
    for i, bad in enumerate((float("nan"), float("inf"))):
        t = make_track(duration=bad, rel=f"bad{i}.mp3")
        rec = env.run([t], sources=["musicbrainz"]).records[t.track_id]
        assert rec.status == "ambiguous" and rec.query["duration_s"] is None and rec.facts == {}
        json.dumps(rec.to_dict(), allow_nan=False)


def test_unicode_digits_are_not_dates():
    assert matching._year("²²²²-01-01") is None          # '²'.isdigit() is True
    assert mb.compact_recording({"id": REC_ORIG, "first-release-date": "١٩٨٢"})[
        "first_release_date"] is None                                       # Arabic-Indic digits
    assert wd.parse_time({"time": "+١٩٨٢-01-01T00:00:00Z", "precision": 11}) is None


def test_records_are_json_native_with_numpy_inputs():
    np = pytest.importorskip("numpy")
    rec = ResearchRecord(track_id="t", rel_path="a.mp3", match_confidence=np.float64(0.9),
                         query={"duration_s": np.float64(3.25)}, candidates=[{"score": np.float32(0.5),
                                                                             "n": np.int64(2), "ok": np.bool_(True)}])
    d = rec.to_dict()
    assert type(d["match_confidence"]) is float and type(d["query"]["duration_s"]) is float
    c = d["candidates"][0]
    assert type(c["score"]) is float and type(c["n"]) is int and type(c["ok"]) is bool


def test_lone_surrogates_never_break_saving(env, tmp_path):
    # (a) an unpaired '\ud800' escape in a response: the run AND the cache save succeed
    env.thriller_catalogue(search=[S_ORIG])
    lookup = json.dumps({**LOOKUP_ORIG, "tags": [{"name": "PLACEHOLDER", "count": 1}]})
    body = lookup.replace("PLACEHOLDER", "bad\\ud800tag").encode("utf-8")
    orig_mb = env.server._musicbrainz

    def mb_with_surrogate(path, params, url):
        if path.startswith("/ws/2/recording/"):
            return HttpResponse(200, {}, body, url)
        return orig_mb(path, params, url)
    env.server._musicbrainz = mb_with_surrogate
    res = env.run([make_track()], sources=["musicbrainz"])
    rec = res.records[make_track().track_id]
    assert rec.status == "matched" and {"bad�tag"} <= {u["name"] for u in rec.user_tags}
    env.cache.save()                                      # used to raise UnicodeEncodeError
    json.dumps(res.to_dict(), ensure_ascii=False).encode("utf-8")
    # (b) an undecodable local tag (surrogateescape): searched, no bogus 'malformed response'
    other = Env(tmp_path / "b")
    other.thriller_catalogue(search=[S_ORIG])
    t = make_track(title="Thriller\udc80", rel="s.mp3")
    rec = other.run([t], sources=["musicbrainz"]).records[t.track_id]
    assert rec.status != "error" and other.server.calls and not rec.errors
    other.cache.save()
    assert other.run([t], sources=["musicbrainz"]).records[t.track_id].status == rec.status   # cache hit
    # (c) the cache refuses what it could not save, and drops it from a hand-edited file
    c = ResearchCache(tmp_path / "c.json", clock=lambda: T0)
    with pytest.raises(ValueError):
        c.put("k", {"a": "x\ud800"}, source="musicbrainz")
    with pytest.raises(ValueError):
        c.put("k\ud800", {"a": 1}, source="musicbrainz")
    entry = '{"source": "musicbrainz", "stored_at": "2026-10-22T09:00:00+00:00", "value": {"a": "x\\ud800"}}'
    (tmp_path / "d.json").write_text('{"schema": "vegroove.research_cache", "schema_version": 1, '
                                     '"entries": {"bad": ' + entry + '}}', "utf-8")
    d = ResearchCache(tmp_path / "d.json", clock=lambda: T0)
    assert "bad" not in d and any("dropped malformed entry" in w for w in d.warnings)
    d.put("good", {"a": 1}, source="musicbrainz")
    d.save()


def test_absurd_numbers_in_responses_do_not_abort_the_run(env):
    # a 400-digit 'score' used to raise OverflowError out of enrich_tracks (float conversion)
    env.thriller_catalogue(search=[{**S_ORIG, "score": 10 ** 400}])
    res = env.run([make_track()], sources=["musicbrainz"])
    rec = res.records[make_track().track_id]
    assert rec.status == "matched" and rec.candidates[0]["relevance"] is None
    env.cache.save()
    json.dumps(res.to_dict(), allow_nan=False)
    assert ResearchRecord.from_dict({"status": "matched", "match_confidence": 10 ** 400}).match_confidence is None
    assert matching.duration_similarity(10 ** 400, 300.0, 7.0) is None


def test_cache_rejects_non_finite_ttl(tmp_path):
    c = ResearchCache(tmp_path / RESEARCH_CACHE_NAME, clock=lambda: T0)
    for ttl in (float("inf"), float("nan"), -1.0):
        with pytest.raises(ValueError):
            c.put("k", {"a": 1}, source="musicbrainz", ttl_s=ttl)
    assert "k" not in c


# ---------------------------------------------------------------------------
# second review round: album / year matching (R1), Wikidata genres (R2), cache-mode
# sources (R3), stale cache entries (R4), Discogs retention / attribution (R5), default
# sources (R6)
# ---------------------------------------------------------------------------

REC_RERECORD = "66666666-6666-4666-8666-666666666666"
S_RERECORD = mb_recording(REC_RERECORD, "Thriller", 357900, disambiguation="", frd="2008-02-11", score=100,
                          releases=[mb_release("aaaaaaaa-0000-4000-8000-000000000006", "Thriller 25", "2008-02-11",
                                               "bbbbbbbb-0000-4000-8000-000000000006")])


def _thriller_with_rerecording(env: Env) -> Env:
    env.thriller_catalogue(search=[S_ORIG, S_RERECORD])
    env.server.mb_recordings[REC_RERECORD] = {**S_RERECORD, "relations": [], "genres": [], "tags": []}
    return env


def test_album_and_year_tags_pick_the_tagged_release_not_the_earliest(env):
    # R1: without album / year the 'earliest first release' prior favours the original
    # (here too weakly to decide: ambiguous); the file's album + year now point to the release
    _thriller_with_rerecording(env)
    t = make_track()
    plain = env.run([t], sources=["musicbrainz"]).records[t.track_id]
    assert plain.status == "ambiguous" and plain.candidates[0]["id"] == REC_ORIG
    assert "earliest_release" in plain.candidates[0]["evidence"]

    tagged = env.run([t], sources=["musicbrainz"],
                     probes={t.track_id: probe_for(t, {"album": "Thriller 25", "date": "2008-02-11"})}
                     ).records[t.track_id]
    assert tagged.query["album"] == "Thriller 25" and tagged.query["year"] == 2008
    assert tagged.status == "matched" and tagged.chosen["id"] == REC_RERECORD
    best, other = tagged.candidates
    assert best["evidence"] == {"official": 0.03, "album": 0.05, "year": 0.04}
    assert "album 'Thriller 25' matches release 'Thriller 25' +0.05" in best["reasons"]
    # the original is on two releases, none of them 'Thriller 25': a small penalty, and no
    # earliest-release bonus once the file names album / year
    assert other["id"] == REC_ORIG and other["evidence"] == {"official": 0.03, "album": -0.03}
    assert "album/year evidence (tie-breaker): album +0.05, year +0.04" in tagged.notes

    original = env.run([t], sources=["musicbrainz"],
                       probes={t.track_id: probe_for(t, {"album": "Thriller", "date": "1982"})}).records[t.track_id]
    assert original.status == "matched" and original.chosen["id"] == REC_ORIG
    assert original.candidates[0]["evidence"] == {"official": 0.03, "album": 0.05, "year": 0.04}


def test_album_and_year_never_confirm_a_weak_identity():
    # R1: the accept gate ignores the album/year bonus: they break ties, never turn a weak
    # artist/title/duration match into 'matched'
    cfg = matching.MatchConfig(accept_score=0.99, official_bonus=0.0)
    off = mb.compact_recording(mb_recording(REC_ORIG, "Thriller", 361600, frd="1982-11-30", releases=[ALBUM]))
    hinted = matching.LocalIdentity(artist="Michael Jackson", title="Thriller", duration_s=357.6,
                                    album="Thriller", year=1982)
    [cand] = matching.score_recordings(hinted, [off], cfg=cfg)
    assert cand.score == pytest.approx(1.0) and cand.identity_score == pytest.approx(0.9714, abs=1e-3)
    decision = matching.decide(hinted, [cand], cfg=cfg)
    assert decision.status == "ambiguous" and decision.chosen is None
    assert any("without the album/year bonus" in r and "only break ties" in r for r in decision.reasons)
    # control: the same identity without the duration gap is accepted
    exact = mb.compact_recording(mb_recording(REC_ORIG, "Thriller", 357600, frd="1982-11-30", releases=[ALBUM]))
    assert matching.decide(hinted, matching.score_recordings(hinted, [exact], cfg=cfg), cfg=cfg).status == "matched"


def test_local_year_replaces_the_earliest_release_prior():
    recs = [mb.compact_recording(r) for r in (S_ORIG, S_RERECORD)]
    no_hints = matching.LocalIdentity(artist="Michael Jackson", title="Thriller", duration_s=357.6)
    assert [c.id for c in matching.score_recordings(no_hints, recs)] == [REC_ORIG, REC_RERECORD]
    year_only = dataclasses.replace(no_hints, year=2008)
    scored = matching.score_recordings(year_only, recs)
    assert [c.id for c in scored] == [REC_RERECORD, REC_ORIG]
    assert scored[0].evidence == {"official": 0.03, "year": 0.04} and "earliest_release" not in scored[1].evidence
    assert scored[0].components["year"] == 1.0 and scored[1].components["year"] == 0.0
    # 0.04 alone is below the margin: still not applied
    assert matching.decide(year_only, scored).status == "ambiguous"


def test_album_penalty_only_for_complete_release_lists():
    local = matching.LocalIdentity(artist="Michael Jackson", title="Thriller", duration_s=357.6,
                                   album="Halloween Party Hits")
    few = mb.compact_recording(S_ORIG)
    many = mb.compact_recording({**S_ORIG, "releases": [
        mb_release(f"aaaaaaaa-0000-4000-8000-{i:012d}", f"Hits {i}", "1990-01-01", f"bbbbbbbb-0000-4000-8000-{i:012d}")
        for i in range(100, 125)]})
    assert len(many["releases"]) == 25
    [c_few] = matching.score_recordings(local, [few])
    [c_many] = matching.score_recordings(local, [many])
    assert c_few.evidence.get("album") == -0.03
    assert "album" not in c_many.evidence and any("may be incomplete" in r for r in c_many.reasons)
    # an album hint is still a hint: no earliest-release prior for either
    assert "earliest_release" not in c_few.evidence and "earliest_release" not in c_many.evidence


def test_album_and_year_tag_parsing():
    assert research_enrich.tag_year({"date": "1982-11-30"}) == 1982
    assert research_enrich.tag_year({"date": "19821130"}) == 1982
    assert research_enrich.tag_year({"originaldate": "30.11.1982"}) == 1982
    assert research_enrich.tag_year({"date": "2008", "originaldate": "1982"}) == 2008      # the file's own release
    assert research_enrich.tag_year({"date": "unknown", "tyer": "1983"}) == 1983
    for bad in ("0000", "²²²²", "١٩٨٢", "", "12"):
        assert research_enrich.tag_year({"date": bad}) is None, bad
    assert research_enrich.tag_album({"album": "  Thriller   25 "}) == "Thriller 25"
    assert research_enrich.tag_album({}) is None and research_enrich.tag_year(None) is None


def test_wikidata_genres_are_user_tags_used_only_with_the_opt_in(env):
    # R2: crowd-edited classifications, not verified facts
    from veg_mixer.ai.interpret import build_payload
    env.thriller_catalogue(search=[S_ORIG])
    t = make_track()
    rec = env.run([t]).records[t.track_id]
    assert "genre_wikidata" not in rec.facts and not any("genre" in k for k in rec.facts)
    wd_tags = [u for u in rec.user_tags if u["source"] == "wikidata"]
    assert [u["name"] for u in wd_tags] == ["pop music", "funk"]
    for u in wd_tags:
        assert u["kind"] == "genre" and u["license"] == LICENSE_CC0 and u["non_commercial"] is False
        assert u["scope"] == "recording" and u["count"] is None and u["retrieved_at"] == "2026-10-22T10:00:00+00:00"
    # the AI payload: not a fact; only in the (unverified) community-tags block when allowed
    plain = build_payload(t, research=rec.to_dict())
    assert "pop music" not in json.dumps(plain) and "community_tags" not in plain
    opted = build_payload(t, research=rec.to_dict(), use_community_tags=True)
    assert "pop music" in opted["community_tags"]["tags"] and opted["community_tags"]["verified"] is False
    assert "pop music" not in json.dumps(opted.get("facts", {}))


def test_wikidata_work_genres_are_scoped_as_the_song(env):
    env.thriller_catalogue(search=[S_ORIG])
    del env.server.wd_search[f"haswbstatement:P4404={REC_ORIG}"]
    env.server.wd_search[f"haswbstatement:P435={WORK}"] = [QID]
    env.server.wd_entities[QID] = {**WD_THRILLER, "claims": {**WD_THRILLER["claims"],
                                                             "P4404": [], "P435": [wd_external_id("P435", WORK)]}}
    rec = env.run([make_track()]).records[make_track().track_id]
    wd_tags = [u for u in rec.user_tags if u["source"] == "wikidata"]
    assert wd_tags and all(u["scope"] == "work" and "musical work" in u["method"] for u in wd_tags)
    assert wd.crosscheck_user_tags({"qid": None, "genres": [{"id": "Q1", "label": "pop"}]}) == []
    # a genre without an English label is a bare item id: dropped
    assert wd.crosscheck_user_tags({"qid": QID, "genres": [{"id": "Q37073", "label": None}]}) == []


def test_old_records_with_a_wikidata_genre_fact_are_migrated():
    old = {"track_id": "t", "rel_path": "a.mp3", "status": "matched",
           "facts": {"year": {"value": 1982, "source": "musicbrainz"},
                     "genre_wikidata": {"value": ["disco", "Q11399"], "source": "wikidata",
                                        "url": f"https://www.wikidata.org/wiki/{QID}", "scope": "work",
                                        "retrieved_at": "2026-01-01T00:00:00+00:00", "license": LICENSE_CC0}},
           "user_tags": [{"name": "pop", "kind": "genre", "source": "musicbrainz"}]}
    rec = ResearchRecord.from_dict(old)
    assert set(rec.facts) == {"year"}                                   # no longer a fact
    assert rec.user_tags[0]["source"] == "musicbrainz"
    [genre] = [u for u in rec.user_tags if u["source"] == "wikidata"]
    assert genre["name"] == "disco" and genre["scope"] == "work" and genre["kind"] == "genre"
    again = ResearchRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert again.to_dict() == rec.to_dict()                             # migrated once, never duplicated


def test_cache_mode_without_sources_keeps_cached_discogs_conflicts(env):
    # R3: plan / interpret call enrich_tracks(mode='cache') without sources; cached Discogs
    # cross-checks must still show up (a disputed year must not look undisputed)
    env.thriller_catalogue(search=[S_ORIG])
    env.server.discogs_results = [dict(r, year="1980") if r["id"] == 8883 else r for r in DISCOGS_RESULTS]
    t = make_track()
    online = env.run([t], sources=["musicbrainz", "wikidata", "discogs"], discogs_token=SECRET).records[t.track_id]
    env.cache.save()
    calls = len(env.server.calls)

    env.now_value = T0 + timedelta(hours=5)
    rec = env.run([t], mode="cache", client=None, cache=env.new_cache(read_only=True)).records[t.track_id]
    assert len(env.server.calls) == calls                                # cache only, no token needed
    assert rec.editorial == online.editorial and rec.conflicts == online.conflicts
    year = next(c for c in rec.conflicts if c["field"] == "year")
    assert [(v["source"], v["value"]) for v in year["values"]] == [("musicbrainz", 1982), ("wikidata", 1983),
                                                                   ("discogs", 1980)]

    env.now_value = T0 + timedelta(hours=7)                             # Discogs data expired
    rec7 = env.run([t], mode="cache", client=None, cache=env.new_cache(read_only=True)).records[t.track_id]
    assert rec7.editorial == [] and not any(v["source"] == "discogs" for c in rec7.conflicts for v in c["values"])
    assert not any("discogs" in n for n in rec7.notes)                  # implicit source: a miss is silent
    assert rec7.status == "matched" and any(c["field"] == "year" for c in rec7.conflicts)   # MB vs Wikidata


def test_cache_mode_uses_expired_entries_but_marks_them_stale(env):
    # R4: after the 90-day TTL a cache-only plan keeps its facts, flagged stale; Discogs
    # data past 6 h is never used; online mode refreshes
    env.thriller_catalogue(search=[S_ORIG])
    env.server.discogs_results = DISCOGS_RESULTS
    t = make_track()
    online = env.run([t], sources=["musicbrainz", "wikidata", "discogs"], discogs_token=SECRET).records[t.track_id]
    env.cache.save()
    assert "stale" not in online.to_dict() and online.stale is False

    env.now_value = T0 + timedelta(days=120)
    calls = len(env.server.calls)
    res = env.run([t], mode="cache", client=None, cache=env.new_cache(read_only=True))
    rec = res.records[t.track_id]
    assert len(env.server.calls) == calls
    assert rec.status == "matched" and rec.stale is True and rec.to_dict()["stale"] is True
    assert rec.facts == online.facts                                    # nothing silently lost
    assert rec.editorial == []                                          # Discogs: never past 6 h
    assert any("stale research data" in n and "120 day(s)" in n and "90-day" in n for n in rec.notes)
    assert res.stats["stale"] == 1 and "1 stale" in res.summary()
    assert any("past their time-to-live" in w and "120 day(s)" in w for w in res.warnings)
    assert ResearchRecord.from_dict(json.loads(json.dumps(rec.to_dict()))).stale is True

    fresh = env.run([t], cache=env.new_cache())                         # online: refreshed, not stale
    assert len(env.server.calls) > calls
    assert fresh.records[t.track_id].stale is False and fresh.stats["stale"] == 0


def test_cache_keeps_expired_entries_for_stale_reads_but_never_discogs(tmp_path):
    p = tmp_path / RESEARCH_CACHE_NAME
    clock = {"now": T0}
    c = ResearchCache(p, clock=lambda: clock["now"])
    c.put("mb:x", {"a": 1}, source="musicbrainz", ttl_s=60)
    c.put("discogs:x", {"r": 1}, source="discogs")
    c.save()
    clock["now"] = T0 + timedelta(hours=7)
    c2 = ResearchCache(p, clock=lambda: clock["now"])
    assert "discogs:x" not in c2 and "mb:x" in c2                      # Discogs purged on load
    assert c2.get("mb:x") is None and c2.get_entry("mb:x") is None     # expired = missing by default
    hit = c2.get_entry("mb:x", allow_expired=True)
    assert hit["value"] == {"a": 1} and hit["expired"] is True and c2.stats["stale"] == 1
    c2.put("discogs:y", {"r": 2}, source="discogs", now=T0)             # stored 7 h ago
    assert c2.get_entry("discogs:y", allow_expired=True) is None       # never served once expired
    c2.save()
    assert set(json.loads(p.read_text("utf-8"))["entries"]) == {"mb:x"}    # kept for cache-only runs
    assert c2.purge_expired() == 1 and "mb:x" not in c2                 # an explicit purge drops it


def test_report_document_prunes_expired_discogs_values_and_keeps_attribution(env):
    # R5 (a): the --report JSON carries no expired Discogs value and names the attribution
    env.thriller_catalogue(search=[S_ORIG])
    env.server.discogs_results = DISCOGS_RESULTS
    t = make_track()
    res = env.run([t], sources=["musicbrainz", "wikidata", "discogs"], discogs_token=SECRET)
    doc = report_document(res, now=T0 + timedelta(hours=5))
    json.dumps(doc, allow_nan=False)
    rec = doc["records"][t.track_id]
    discogs_values = rec["editorial"] + [v for c in rec["conflicts"] for v in c["values"] if v["source"] == "discogs"]
    assert len(discogs_values) == 4
    for v in discogs_values:
        assert v["attribution"] == DISCOGS_ATTRIBUTION and v["expires_at"] == "2026-10-22T16:00:00+00:00"
    assert doc["discogs"]["values"] == 4 and doc["discogs"]["expires_at"] == "2026-10-22T16:00:00+00:00"
    assert "6 hours" in doc["discogs"]["notice"] and DISCOGS_ATTRIBUTION in doc["discogs"]["notice"]
    assert doc["stats"] == res.stats and doc["warnings"] == res.warnings
    assert doc["schema"] == "vegroove.research_report" and doc["generated_at"] == "2026-10-22T15:00:00+00:00"

    doc7 = report_document(res, now=T0 + timedelta(hours=7))
    rec7 = doc7["records"][t.track_id]
    assert rec7["editorial"] == [] and doc7["discogs"]["values"] == 0 and doc7["discogs"]["removed_expired"] == 4
    assert not any(v["source"] == "discogs" for c in rec7["conflicts"] for v in c["values"])
    assert len(res.records[t.track_id].editorial) == 3                  # the run's records are untouched
    # records loaded from an older report: a missing attribution is added, unknown age = removed
    old = {t.track_id: {**res.records[t.track_id].to_dict(), "editorial": [
        {"field": "year", "value": 1982, "source": "discogs", "expires_at": "2026-10-22T16:00:00+00:00"},
        {"field": "genre", "value": ["Pop"], "source": "discogs"}]}}
    doc_old = report_document(old, now=T0 + timedelta(hours=1))
    assert doc_old["records"][t.track_id]["editorial"] == [
        {"field": "year", "value": 1982, "source": "discogs", "expires_at": "2026-10-22T16:00:00+00:00",
         "attribution": DISCOGS_ATTRIBUTION}]
    assert "stats" not in doc_old and "warnings" not in doc_old


def test_format_conflict_names_discogs_next_to_its_value(env):
    # R5 (b): the console line carries the attribution
    env.thriller_catalogue(search=[S_ORIG])
    env.server.discogs_results = DISCOGS_RESULTS
    t = make_track()
    rec = env.run([t], sources=["musicbrainz", "wikidata", "discogs"], discogs_token=SECRET).records[t.track_id]
    year = next(c for c in rec.conflicts if c["field"] == "year")
    assert format_conflict(year) == \
        "year: musicbrainz=1982; wikidata=1983; discogs=1982 (Data provided by Discogs.)"
    assert format_conflict(year, now=T0 + timedelta(hours=7)) == "year: musicbrainz=1982; wikidata=1983"
    isrc = {"field": "isrc", "values": [{"source": "file_tag", "value": "GBAAA0000001", "url": None},
                                        {"source": "musicbrainz", "value": [ISRC, "USSM10000001"]}]}
    assert format_conflict(isrc) == f"isrc: file_tag=GBAAA0000001; musicbrainz={ISRC}, USSM10000001"
    work = {"field": "year", "values": [{"source": "musicbrainz", "value": 1999},
                                        {"source": "wikidata", "value": 1983, "scope": "work"}]}
    assert format_conflict(work) == "year: musicbrainz=1999; wikidata=1983 (work)"


def test_online_default_sources_follow_the_discogs_token(env, monkeypatch):
    # R6: VEG_DISCOGS_TOKEN alone enables Discogs when no sources are given
    env.thriller_catalogue(search=[S_ORIG])
    env.server.discogs_results = DISCOGS_RESULTS
    t = make_track()
    res = env.run([t])
    assert "api.discogs.com" not in env.server.hosts() and not any("discogs" in w for w in res.warnings)
    monkeypatch.setenv("VEG_DISCOGS_TOKEN", SECRET)
    rec = env.run([t]).records[t.track_id]
    dcall = next(c for c in env.server.calls if c["host"] == "api.discogs.com")
    assert dcall["headers"]["Authorization"] == f"Discogs token={SECRET}"
    assert {e["field"] for e in rec.editorial} == {"year", "genre", "style"}
    # an explicit token without the environment variable works the same way
    monkeypatch.delenv("VEG_DISCOGS_TOKEN")
    other = Env(env.tmp_path / "explicit")
    other.thriller_catalogue(search=[S_ORIG])
    other.server.discogs_results = DISCOGS_RESULTS
    other.run([t], discogs_token=SECRET)
    assert "api.discogs.com" in other.server.hosts()


def test_stale_entries_are_dropped_a_year_past_their_ttl(tmp_path):
    from veg_mixer.research.cache import STALE_RETENTION_S
    p = tmp_path / RESEARCH_CACHE_NAME
    clock = {"now": T0}
    c = ResearchCache(p, clock=lambda: clock["now"])
    c.put("mb:old", {"a": 1}, source="musicbrainz", ttl_s=60)
    c.put("mb:fresh", {"a": 2}, source="musicbrainz")
    c.save()
    clock["now"] = T0 + timedelta(seconds=60 + STALE_RETENTION_S + 3600)
    c2 = ResearchCache(p, clock=lambda: clock["now"])
    assert "mb:old" not in c2 and "mb:fresh" in c2                     # 90-day TTL + a year: still kept

