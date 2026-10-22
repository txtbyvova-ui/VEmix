"""AI track understanding (VEmix 3.0, spec §4): structured, evidence-backed interpretations.

For each track the interpreter builds a canonical INPUT PAYLOAD from what is actually known —
measured audio features, verified external facts and the curator's manual metadata — sends it
as JSON text to a pluggable provider (``providers.py``) with a strict output schema
(``schema.py``), validates the answer and caches it (``cache.py``).

Payload (``build_payload``), JSON-native, never with file paths, folder names, file names,
artist or title, unknown values omitted (never null-filled with guesses)::

    {"measured": {"bpm", "bpm_source", "key" (Camelot), "key_source", "key_confidence",
                  "energy", "energy_source", "energy_confidence", "duration_s",
                  "segments": {"energy_in", "energy_out", "energy_mean", "energy_peak",
                               "trajectory" {value, confidence, ...}, "density" {value, score, ...},
                               "percussive_density", "bass_density", "dynamic_range_db",
                               "tempo" {bpm, confidence, half_double_ambiguous},
                               "vocal_presence_proxy" {value, confidence, method},
                               "harmony" {key_estimate, clarity, stability, modulations},
                               "sections": [{"label", "start", "end"}], "notes": [...]}},
     "facts":    {<field>: <value>, ..., "facts_source": "musicbrainz, wikidata"},  # matched only
     "community_tags": {"kind": "user_tags", "verified": false, "license_note": ..., "tags": [...]},
     "curator":  {"mood", "genre", "vibe_tags", "energy", "danceability", "vocal_presence",
                  "bpm", "key"}}                                           # feature_sources == manual

Segments are the ``audio_intel.analyze_segments`` value (summary keys read from its
``summary``; a flat dict with the same keys is accepted; an ``error`` value is ignored). A
block whose main field is not determined (null or "unknown": ``tempo.bpm``,
``harmony.key_estimate``, otherwise ``value``) is left out.

No identity (spec §1, the AI must not invent facts about compositions): the model interprets
only measured features, verified facts and curator labels. Artist and title are not sent — they
describe nothing measurable and only invite recall from memory, and without tags the title is
the file name stem — and neither is anything else derived from a file name. ``PRIVACY_NOTE`` is
the one-line statement of what is sent, for the console.

Categories are not mixed (spec §1): a value the curator overrode appears under ``curator`` only,
an ID3 genre (an unverified file tag) is not sent at all, external facts come only from a
research record whose status is ``matched`` (an ambiguous match is never used), and user tags
(MusicBrainz tags / genres, Wikidata genres) only when explicitly enabled and labelled as
unverified data with a licence note (some sources allow non-commercial use only).

Research records (``research[track_id]``: a ``research.models.ResearchRecord`` or its
``to_dict()``) are read defensively: ``{"status": "matched", "facts": {field: {"value": v,
"source": ...} | v}, "user_tags": [{"name", "count", "license", "source"}], "conflicts":
[{"field", "values": [{"value", "source"}]}]}``. Only the descriptive facts of the allowlist
``DESCRIPTIVE_FACTS`` (``year``, ``first_release_date``, ``release_group_type``,
``publication_date_wikidata``) are sent. Identifiers and URLs (MBIDs, ISRCs, Wikidata items,
links), identity facts (recording / release / work titles, artist credits), tags and every
other research field are left out — they describe nothing measurable and only invite recall
from memory, and a new field (label, composer, performer ...) is never sent before someone
decides it is not identifying. A field that durable sources disagree on is not sent either,
nor the facts carrying the same information (a ``year`` conflict also withholds
``first_release_date`` and ``publication_date_wikidata``): it is not verified. A dispute that
rests only on an ephemeral source (``_EPHEMERAL_SOURCES``: Discogs, whose values expire after
6 h) does not change the payload, so the cache key stays the same across the expiry; the
planner reports such disputes.

Interpretation flow (``interpret_tracks``, sequential, never parallel API calls):

* ``mode="off"`` — nothing; ``"cache"`` — cached answers only, the provider is never called
  (without a provider, any cached answer for the same input is used); ``"online"`` — cache
  first, then the provider.
* A structurally invalid answer gets ONE re-ask that quotes the validation errors; if it is
  still invalid the negative result is cached (``valid: false``) so it is not paid for again
  (``retry_invalid`` — ``interpret --mode online --retry-invalid`` — asks again). Claims
  dropped by the evidence rules alone do not make an answer invalid.
* Provider errors (network, rate limit, refusal, truncated answer) are not cached. After
  ``max_consecutive_failures`` errors in a row — or at once for a fatal error such as bad
  credentials or an unknown model — the provider is not called again for the remaining tracks
  (circuit breaker; they get a warning).
* Nothing raises for provider, network or JSON problems; misconfiguration (unknown mode,
  ``online`` without a provider) raises ``AIConfigError`` (a ValueError).

The result is ADVISORY data with its own provenance (``source = "ai:<provider>/<model>"``,
prompt version, request id, created_at). It is never written into ``TrackFeatures`` fields:
those feed curation and the 2.0 sequencer, and AI output must stay a separate category.

Limits: the model judges only the numbers it is given; it cannot listen. Every claim cites
input paths and carries a confidence, but whether the cited data truly supports the claim is
the model's judgement — the planner must weigh these interpretations below measurements.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from ..models import TrackFeatures
from ..moods import MOOD_COORDINATES
from ..textnorm import normalize_tag, primary_artist
from .cache import AICache, canonical_json, make_key
from .providers import FATAL_ERROR_KINDS, INVALID_OUTPUT, AIConfigError, AIProvider, AIResult
from .schema import INTERPRETATION_SCHEMA, PHASES, ROLES, SCHEMA_VERSION, validate_interpretation

log = logging.getLogger(__name__)

# v2: the payload no longer carries the track's identity (artist / title) and the prompt states
# the per-field evidence rules. Part of every cache key (with PROMPT_DIGEST and SCHEMA_VERSION):
# answers given for a v1 payload are never served for the new payload shape.
PROMPT_VERSION = "interpret-v2"
AI_MODES = ("off", "cache", "online")
# What `interpret` sends, for the console (the CLI prints it before calling a provider).
PRIVACY_NOTE = ("only measured features, verified facts and curator labels are sent (unverified community tags "
                "only when enabled); never audio, file paths, file names, artist or title")
# Claude Opus 5.5 always thinks, and thinking counts towards max_tokens: ~4000 leaves too little
# room for thinking + the JSON answer at medium effort. A higher cap costs nothing unless used.
DEFAULT_MAX_TOKENS = 8000
MAX_DETAILED_WARNINGS = 10
_MAX_REASK_ERRORS = 15
# Cache saves during a run (plus one at the end): bounded rewrites of the whole file.
SAVE_EVERY_S = 30.0
SAVE_EVERY_N = 25

ROLE_DEFINITIONS: dict[str, str] = {
    "opener": "starts the set: sets the scene and mood without spending peak energy",
    "tension_builder": "raises suspense or intensity; energy or density grows",
    "groove_builder": "steady rhythmic drive that keeps the floor moving at a sustained level",
    "release": "resolves built-up tension: a drop, a chorus payoff, a burst of energy",
    "breather": "a lower-energy or sparser moment that rests the listener between peaks",
    "peak": "one of the most intense moments of a set",
    "transition": "bridges two different moods, tempos or energy levels",
    "comedown": "brings energy down gradually after a peak",
    "closer": "ends the set: resolves and lets the listener out",
}
PHASE_DEFINITIONS: dict[str, str] = {
    "intro": "the opening minutes: sets the scene",
    "build": "gradual rise towards the first peak",
    "first_peak": "the first high point",
    "breather": "a planned dip after the first peak",
    "second_build": "renewed rise after the breather",
    "climax": "the highest point of the whole set",
    "descent": "controlled fall in energy after the climax",
    "closer": "the final track(s)",
}


def _system_prompt() -> str:
    roles = "\n".join(f"  - {r}: {ROLE_DEFINITIONS.get(r, '(no definition)')}" for r in ROLES)
    phases = "\n".join(f"  - {p}: {PHASE_DEFINITIONS.get(p, '(no definition)')}" for p in PHASES)
    moods = ", ".join(MOOD_COORDINATES)
    return f"""You describe ONE music track for a playlist planner that builds a dramaturgical arc (intro, build, peaks, breathers, climax, descent). Your description is advisory: the planner weighs it below measured audio features.

INPUT. The user message is one JSON object. The track's identity (artist, title, file name) is deliberately NOT given: describe the data, not a song you might recognise.
- "measured": features measured from the audio file by a local analyzer: BPM, Camelot key (A = minor, B = major), a 0..1 energy estimate, duration; under "segments", time-resolved analysis (energy at the start and end, trajectory, density of percussion, bass and layers, dynamic range, tempo details, harmony, sections, the analyzer's own caveats in "notes"). Each value is only as reliable as its *_source / *_confidence fields say. Section labels are heuristic guesses from relative intensity and position, not a musical analysis.
- "facts": verified catalogue metadata from external sources.
- "community_tags": unverified user-submitted tags (present only when enabled).
- "curator": values entered by a human curator.
Any section may be missing; missing means unknown.

RULES
1. Use ONLY the data in the input. Do not use your own knowledge or memory of any song, artist, album, lyrics, popularity or how a track is usually played.
2. Never guess audio characteristics that are not in the input. If the data is insufficient, answer "unknown", null or an empty list, with low confidence.
3. Every claim lists "evidence": key paths into the input JSON in dot notation with numeric list indices, e.g. "measured.segments.energy_out", "measured.segments.sections.0.label", "curator.mood", "facts.first_release_date". Cite only paths that exist in the input. Fields that only say how reliable or where from a value is (*_source, *_confidence, "confidence", "method") support nothing on their own: cite the value itself. Claims with missing or invented evidence are discarded.
4. "confidence" is a number from 0 to 1: how strongly the cited data supports the claim. Use 0.9 or more only for direct curator statements or unambiguous measurements.
5. energy_trajectory and sonic_density describe how the track evolves: they need "measured.segments" evidence. Without segment data answer "unknown".
6. moods need evidence a mood can be read from: measured energy, dynamics, segment trajectory / density / vocal presence, key / mode or tempo, or the curator's mood, vibe_tags or genre (or the curator's energy / bpm / key / vocal_presence). Catalogue facts, community tags, section labels or duration alone never support a mood.
7. danceability and groove are estimates from 0 to 1 (null when unknown); they need rhythm evidence: measured BPM, the segment tempo details or percussive density, or the curator's danceability / bpm. Key, mode, energy or loudness alone are not rhythm evidence.
8. roles and positions are dramaturgical judgements. Tempo, key, duration, a genre label, user tags, catalogue facts or popularity alone are never sufficient (also when the curator entered them): cite energy, segment or curator mood / energy evidence, or leave the list empty.
9. risky_pairings: kinds of neighbouring tracks that would clash with this one (e.g. "a much faster track", "a euphoric peak track") and why. Describe a kind of track, never a specific song.
10. notes: at most two short sentences about the data (for example what is missing). No facts that are not in the input.

VOCABULARY
roles:
{roles}
positions (phases of a set):
{phases}
moods: prefer these labels: {moods}. Use another one- or two-word label only when none fits.

OUTPUT: exactly one JSON object that matches the provided schema. No text outside the JSON."""


SYSTEM_PROMPT = _system_prompt()
# Any change of the prompt text or the schema changes this digest and therefore every cache key:
# an answer is reused only for exactly the instructions it was produced under.
PROMPT_DIGEST = hashlib.sha256((SYSTEM_PROMPT + "\n" + canonical_json(INTERPRETATION_SCHEMA))
                               .encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

# Summary keys of a segment analysis (``audio_intel``: under "summary"; a flat dict is accepted
# too). Per-segment curves and the unlabelled timbre vector are not sent.
SEGMENT_KEYS = ("energy_in", "energy_out", "energy_mean", "energy_peak", "trajectory", "density",
                "percussive_density", "bass_density", "dynamic_range_db", "tempo", "vocal_presence_proxy",
                "harmony")
TEMPO_KEYS = ("bpm", "confidence", "half_double_ambiguous")
HARMONY_KEYS = ("key_estimate", "clarity", "stability", "modulations")
CURATOR_FIELDS = ("mood", "genre", "vibe_tags", "energy", "danceability", "vocal_presence")
MAX_SECTIONS = 32
MAX_LIST = 64
MAX_TEXT = 200
MAX_NOTES = 5
MAX_COMMUNITY_TAGS = 20
COMMUNITY_TAGS_NOTE = ("user-submitted tags from external catalogues: unverified opinions; licences vary by "
                       "source and some allow non-commercial use only")
_MISSING = object()
_PATHLIKE_KEYS = frozenset({"path", "paths", "file", "files", "filename", "file_name", "folder", "dir", "directory",
                            "rel_path", "abs_path", "source_path", "url", "urls", "uri", "href", "link", "links"})
_PATHLIKE_SUFFIXES = ("_path", "_file", "_dir", "_folder", "_url", "_uri")
_IDENTIFIER_KEYS = frozenset({"id", "ids", "mbid", "mbids", "isrc", "isrcs", "iswc", "qid", "barcode",
                              "catalog_number", "wikidata_item"})
_IDENTIFIER_SUFFIXES = ("_id", "_ids", "_mbid", "_mbids", "_qid", "_isrc", "_isrcs", "_iswc", "_item")
# Facts that only restate WHICH recording this is (titles, credits): they describe nothing the
# planner can use and, like the track's own artist and title (not sent either), would only let
# the model answer from its memory of the song.
_IDENTITY_FACT_KEYS = frozenset({"title", "artist", "artists", "artist_credit", "album", "release",
                                 "release_group"})
_IDENTITY_FACT_SUFFIXES = ("_title", "_titles")
# A conflict on one field disputes every fact that carries the same information (research
# reports a "year" conflict between the MusicBrainz year and the Wikidata publication date).
_CONFLICT_GROUPS = {"year": frozenset({"year", "first_release_date", "publication_date_wikidata",
                                       "release_date", "date"})}
_QID_RE = re.compile(r"^Q[0-9]+$")
# Local paths ('C:\\', '\\\\server', '/home/x/', '~/') and URLs never reach a provider.
_PATH_VALUE_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/[^/\s]+/|~[\\/]|[A-Za-z][A-Za-z0-9+.-]*://)")
# crowd-edited classifications are tags, never verified facts — also in records written before
# research moved Wikidata genres to user_tags
_TAG_FIELDS = ("community_tags", "user_tags", "tags", "folksonomy", "genre_wikidata")


def _pathlike_key(key: str) -> bool:
    k = key.lower()
    return k in _PATHLIKE_KEYS or k.endswith(_PATHLIKE_SUFFIXES)


def _identifier_key(key: str) -> bool:
    k = key.lower()
    return k in _IDENTIFIER_KEYS or k.endswith(_IDENTIFIER_SUFFIXES)


def _identity_fact_key(key: str) -> bool:
    k = key.lower()
    return k in _IDENTITY_FACT_KEYS or k.endswith(_IDENTITY_FACT_SUFFIXES)


def _clean(value: Any, *, depth: int = 0) -> Any:
    """JSON-native, path-free copy: numpy scalars/arrays converted, floats rounded to 3 decimals,
    non-finite numbers / None / empty containers / path-like keys and values omitted."""
    if depth > 6 or value is None:
        return _MISSING
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):      # numpy scalar / array
        value = value.tolist()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 3) if math.isfinite(value) else _MISSING
    if isinstance(value, str):
        s = value.strip()
        if not s or _PATH_VALUE_RE.match(s):
            return _MISSING
        return s[:MAX_TEXT]
    if isinstance(value, (list, tuple)):
        items = [c for c in (_clean(v, depth=depth + 1) for v in list(value)[:MAX_LIST]) if c is not _MISSING]
        return items or _MISSING
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str) or _pathlike_key(k):
                continue
            c = _clean(v, depth=depth + 1)
            if c is not _MISSING:
                out[k] = c
        return out or _MISSING
    return _MISSING


def _put(target: dict, key: str, value: Any) -> None:
    c = _clean(value)
    if c is not _MISSING:
        target[key] = c


# segment-summary blocks whose main field says "not determined": sent as such they would be a
# citable path with nothing behind it (only confidence / method), so they are left out
_SEGMENT_MAIN_FIELD = {"tempo": "bpm", "harmony": "key_estimate"}


def _undetermined(key: str, value: Any) -> bool:
    main = _SEGMENT_MAIN_FIELD.get(key, "value")
    return isinstance(value, Mapping) and main in value and value[main] in (None, "unknown")


def _segments_block(seg: Mapping[str, Any]) -> dict:
    summary = seg.get("summary") if isinstance(seg.get("summary"), Mapping) else seg
    out: dict[str, Any] = {}
    for key in SEGMENT_KEYS:
        value = summary.get(key)
        if _undetermined(key, value):
            continue
        if key == "tempo" and isinstance(value, Mapping):
            value = {k: value[k] for k in TEMPO_KEYS if k in value}     # candidates list not needed
        if key == "harmony" and isinstance(value, Mapping):
            value = {k: value[k] for k in HARMONY_KEYS if k in value}   # per-segment keys not needed
        _put(out, key, value)
    raw = seg.get("sections", summary.get("sections"))
    sections = []
    for s in (list(raw)[:MAX_SECTIONS] if isinstance(raw, (list, tuple)) else []):
        if not isinstance(s, Mapping):
            continue
        item: dict[str, Any] = {}
        _put(item, "label", s.get("label"))
        _put(item, "start", s.get("start", s.get("start_s")))
        _put(item, "end", s.get("end", s.get("end_s")))
        if item:
            sections.append(item)
    if sections:
        out["sections"] = sections
    notes = seg.get("notes")
    if out and isinstance(notes, (list, tuple)):
        _put(out, "notes", [n for n in notes if isinstance(n, str)][:MAX_NOTES])   # analyzer's own caveats
    return out


def _record_status(record: Mapping[str, Any]) -> str | None:
    status = record.get("status")
    if status is None and isinstance(record.get("match"), Mapping):
        status = record["match"].get("status")
    return str(status).strip().lower() if status is not None else None


def _source_names(*raws: Any) -> list[str]:
    """Source NAMES ('musicbrainz'), never URLs, from record- or value-level source fields."""
    names: list[str] = []
    for raw in raws:
        for item in raw if isinstance(raw, (list, tuple)) else [raw]:
            if isinstance(item, Mapping):
                item = item.get("name") or item.get("source") or item.get("provider")
            if isinstance(item, str) and item.strip() and not _PATH_VALUE_RE.match(item) and "://" not in item:
                if item.strip() not in names:
                    names.append(item.strip())
    return names


# Sources whose data expires within hours (Discogs, 6 h): a dispute that rests on them alone
# would change the payload — and so the cache key — when the value expires, and a plan made
# later would no longer find the interpretation. Such a dispute is the planner's business
# (it reports it); the AI gets the same facts before and after the expiry.
_EPHEMERAL_SOURCES = frozenset({"discogs"})


def _durable_dispute(conflict: Mapping[str, Any]) -> bool:
    values = conflict.get("values")
    if not isinstance(values, (list, tuple)) or not any(
            isinstance(v, Mapping) and str(v.get("source", "")).lower() in _EPHEMERAL_SOURCES for v in values):
        return True
    durable = {canonical_json(v.get("value")) for v in values
               if isinstance(v, Mapping) and str(v.get("source", "")).lower() not in _EPHEMERAL_SOURCES}
    return len(durable) > 1


def _conflicted_fields(record: Mapping[str, Any]) -> set[str]:
    raw = record.get("conflicts")
    fields = {c["field"] for c in raw if isinstance(c, Mapping) and isinstance(c.get("field"), str)
              and _durable_dispute(c)} if isinstance(raw, (list, tuple)) else set()
    for f in list(fields):
        fields |= _CONFLICT_GROUPS.get(f, frozenset())
    return fields


# Facts that DESCRIBE a recording without identifying it: an allowlist, so a new research field
# (composer, label, performer ...) is never sent before someone decides it is not identifying.
DESCRIPTIVE_FACTS = frozenset({"year", "first_release_date", "release_group_type", "publication_date_wikidata"})


def _facts_block(record: Mapping[str, Any]) -> dict:
    """Matched facts: value only — descriptive fields only (``DESCRIPTIVE_FACTS``: identifiers,
    URLs, titles / credits and tags are never sent), without fields the sources disagree on,
    plus ``facts_source`` naming the sources."""
    raw = record.get("facts", record.get("fields"))
    conflicted = _conflicted_fields(record)
    out: dict[str, Any] = {}
    sources: list[Any] = [record.get("sources"), record.get("source")]
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            if (not isinstance(k, str) or k not in DESCRIPTIVE_FACTS or _pathlike_key(k) or _identifier_key(k)
                    or _identity_fact_key(k) or k.lower() in _TAG_FIELDS or k in conflicted):
                continue
            value = v.get("value", _MISSING) if isinstance(v, Mapping) else v
            if isinstance(value, (list, tuple)):          # e.g. Wikidata genres without a label
                value = [x for x in value if not (isinstance(x, str) and _QID_RE.match(x.strip()))]
            if value is _MISSING or (isinstance(value, str) and _QID_RE.match(value.strip())):
                continue
            before = len(out)
            _put(out, k, value)
            if len(out) > before and isinstance(v, Mapping):
                sources.append(v.get("source"))
    names = _source_names(*sources)
    if out and names:
        out["facts_source"] = ", ".join(sorted(names))
    return out


def _tag_names(raw: Any) -> list[str]:
    items: list[tuple[str, float]] = []
    for t in raw if isinstance(raw, (list, tuple)) else []:
        if isinstance(t, str):
            items.append((t, 0.0))
        elif isinstance(t, Mapping) and t.get("scope") == "work":
            continue                       # describes the song (a cover's original), not this recording
        elif isinstance(t, Mapping) and isinstance(t.get("name", t.get("tag")), str):
            count = t.get("count", 0)
            ok = isinstance(count, (int, float)) and not isinstance(count, bool) and math.isfinite(count)
            items.append((t.get("name", t.get("tag")), float(count) if ok else 0.0))
    items.sort(key=lambda nc: (-nc[1], nc[0].casefold()))          # most-voted first, deterministic
    out: list[str] = []
    for name, _ in items:
        n = name.strip()[:60]
        if n and n.casefold() not in (o.casefold() for o in out):
            out.append(n)
    return out[:MAX_COMMUNITY_TAGS]


def _community_tags(record: Mapping[str, Any]) -> dict | None:
    raw = None
    for k in _TAG_FIELDS:
        if record.get(k):
            raw = record[k]
            break
    if raw is None and isinstance(record.get("facts"), Mapping):
        for k in _TAG_FIELDS:
            if record["facts"].get(k):
                raw = record["facts"][k]
                break
    tags = _tag_names(raw)
    if not tags:
        return None
    entries = [t for t in raw if isinstance(t, Mapping)] if isinstance(raw, (list, tuple)) else []
    licenses = sorted({t["license"] for t in entries if isinstance(t.get("license"), str) and t["license"]})
    note = COMMUNITY_TAGS_NOTE + (f" ({', '.join(licenses)})" if licenses else "")
    block: dict[str, Any] = {"kind": "user_tags", "verified": False, "license_note": note, "tags": tags}
    names = _source_names(*[t.get("source") for t in entries]) or _source_names(record.get("sources"),
                                                                              record.get("source"))
    if names:
        block["source"] = ", ".join(sorted(names))
    return block


def build_payload(track: TrackFeatures, *, segments: dict | None = None, research: dict | None = None,
                  use_community_tags: bool = False) -> dict:
    """Canonical, JSON-native input for one track (see the module docstring). ``segments`` is the
    track's segment analysis; ``research`` its research record (used only when ``status`` is
    ``matched``). Contains no file paths, folder names, file names, artist or title: only
    measured features, verified facts and curator labels (``PRIVACY_NOTE``)."""
    fs = track.feature_sources or {}

    def manual(name: str) -> bool:
        return fs.get(name) == "manual"

    payload: dict[str, Any] = {}
    measured: dict[str, Any] = {}

    def put_measured(key: str, value: Any, **provenance: Any) -> None:
        """A value and its provenance — the provenance only when the value itself is usable."""
        if value is None or _clean(value) is _MISSING:
            return
        _put(measured, key, value)
        for name, v in provenance.items():
            _put(measured, name, v)

    if not manual("bpm"):
        put_measured("bpm", track.bpm, bpm_source=fs.get("bpm"))
    if not manual("camelot"):
        put_measured("key", track.camelot, key_source=fs.get("camelot"), key_confidence=track.key_confidence)
    if not manual("energy"):
        put_measured("energy", track.energy, energy_source=fs.get("energy"),
                     energy_confidence=track.energy_confidence)
    if track.duration_s and track.duration_s > 0:
        _put(measured, "duration_s", track.duration_s)
    if isinstance(segments, Mapping) and not segments.get("error"):
        block = _segments_block(segments)
        if block:
            measured["segments"] = block
    if measured:
        payload["measured"] = measured

    if research is not None and not isinstance(research, Mapping) and hasattr(research, "to_dict"):
        research = research.to_dict()                       # a research.models.ResearchRecord
    if isinstance(research, Mapping) and _record_status(research) == "matched":
        facts = _facts_block(research)
        if facts:
            payload["facts"] = facts
        if use_community_tags:
            tags = _community_tags(research)
            if tags:
                # a tag that repeats the artist or the title would hand the model the identity
                identity = {normalize_tag(x) for x in (track.artist, track.title, primary_artist(track.artist))
                            if isinstance(x, str) and x.strip()} - {""}
                tags["tags"] = [t for t in tags["tags"] if normalize_tag(t) not in identity]
                if tags["tags"]:
                    payload["community_tags"] = tags

    curator: dict[str, Any] = {}
    for name in CURATOR_FIELDS:
        if manual(name):
            _put(curator, name, getattr(track, name, None))
    if manual("bpm"):
        _put(curator, "bpm", track.bpm)
    if manual("camelot"):
        _put(curator, "key", track.camelot)
    if curator:
        payload["curator"] = curator
    return canonical_payload(payload)


def canonical_payload(payload: dict) -> dict:
    """Payload with canonical number formatting (what the cache key and the provider see)."""
    return json.loads(canonical_json(payload))


def interpretable(payload: Mapping[str, Any]) -> bool:
    """True when the payload holds something beyond tempo / key / duration: energy, segments,
    facts, user tags or curator input. Asking a model about BPM alone only invites guessing."""
    measured = payload.get("measured") or {}
    return bool("energy" in measured or "segments" in measured or payload.get("facts")
                or payload.get("community_tags") or payload.get("curator"))


def cache_key(payload: Mapping[str, Any], *, provider: str, model: str) -> str:
    """Cache key: sha256 of the canonical {prompt_version, prompt_digest, schema_version, provider,
    model, input payload}."""
    return make_key({"prompt_version": PROMPT_VERSION, "prompt_digest": PROMPT_DIGEST,
                     "schema_version": SCHEMA_VERSION, "provider": provider, "model": model, "input": payload})


def input_digest(payload: Mapping[str, Any]) -> str:
    """Provider-independent digest of the same input (cache-only lookups without a provider)."""
    return make_key({"prompt_version": PROMPT_VERSION, "prompt_digest": PROMPT_DIGEST,
                     "schema_version": SCHEMA_VERSION, "input": payload})


def user_message(payload: Mapping[str, Any], *, errors: Sequence[str] = ()) -> str:
    """The user message: the canonical payload JSON, plus the validator's complaints on a re-ask."""
    text = canonical_json(payload)
    if errors:
        shown = [e[:200] for e in list(errors)[:_MAX_REASK_ERRORS]]
        more = len(errors) - len(shown)
        text += ("\n\nYour previous answer for this input was REJECTED by the validator:\n"
                 + "\n".join(f"- {e}" for e in shown)
                 + (f"\n- ... and {more} more" if more > 0 else "")
                 + "\nReturn a corrected JSON object for the input above. Cite as evidence only key paths "
                   "that exist in that JSON input.")
    return text


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

@dataclass
class InterpretResult:
    # track_id -> clean interpretation + {"source", "prompt_version", "schema_version",
    #                                     "created_at", "request_id", "from_cache",
    #                                     "inputs" (payload sections used, e.g. ["curator", "measured"]),
    #                                     "license_note" (only when non-commercial user tags were used)}
    interpretations: dict[str, dict] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=lambda: dict.fromkeys(
        ("cache_hits", "requests", "valid", "invalid", "errors", "skipped", "reasks", "dropped_claims"), 0))
    warnings: list[str] = field(default_factory=list)


class _Warnings:
    """Per-category warnings: the first few in detail, then one summary line."""

    def __init__(self, sink: list[str]) -> None:
        self.sink = sink
        self.counts: dict[str, int] = {}

    def add(self, category: str, message: str) -> None:
        n = self.counts[category] = self.counts.get(category, 0) + 1
        if n <= MAX_DETAILED_WARNINGS:
            log.warning("%s", message)
            self.sink.append(message)

    def finish(self) -> None:
        for category, n in self.counts.items():
            if n > MAX_DETAILED_WARNINGS:
                msg = f"AI: ... and {n - MAX_DETAILED_WARNINGS} more '{category}' warnings"
                log.warning("%s", msg)
                self.sink.append(msg)


def _iso(now: Callable[[], datetime]) -> str:
    t = now()
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.isoformat(timespec="seconds")


def _inputs(payload: Mapping[str, Any]) -> list[str]:
    """Input categories the answer was built from (a ``track`` identity block, which payloads no
    longer carry, would not be an input)."""
    return sorted(k for k in payload if k != "track")


def _record(entry: Mapping[str, Any], *, from_cache: bool) -> dict:
    out = dict(entry["output"])
    out["dropped"] = list(entry.get("dropped") or [])
    inputs = [x for x in entry.get("inputs") or [] if isinstance(x, str)]
    out.update(source=f"ai:{entry['provider']}/{entry['model']}", prompt_version=PROMPT_VERSION,
               schema_version=SCHEMA_VERSION, created_at=entry["created_at"],
               request_id=entry.get("request_id"), from_cache=from_cache, inputs=inputs)
    if "community_tags" in inputs:      # derived from CC BY-NC-SA user tags: say so wherever it goes
        out["license_note"] = "built partly from " + COMMUNITY_TAGS_NOTE
    return out


def _max_tokens_from_env() -> int:
    raw = os.environ.get("VEG_AI_MAX_TOKENS")
    if not raw:
        return DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError:
        raise AIConfigError(f"VEG_AI_MAX_TOKENS must be an integer, got {raw!r}") from None
    if value < 256:
        raise AIConfigError(f"VEG_AI_MAX_TOKENS must be at least 256, got {value}")
    return value


def _call(provider: AIProvider, system: str, user: str, max_tokens: int) -> AIResult:
    """One provider call; a provider that raises (a third-party plugin, a bug) becomes an error
    result instead of aborting the run."""
    try:
        result = provider.complete_json(system=system, user=user, schema=INTERPRETATION_SCHEMA,
                                        max_tokens=max_tokens)
    except Exception as e:   # noqa: BLE001 - provider boundary: degrade, never abort the run
        log.debug("AI provider %s raised", getattr(provider, "name", "?"), exc_info=True)
        return AIResult(None, "", str(getattr(provider, "name", "?")), str(getattr(provider, "model", "?")),
                        error=f"sdk: provider raised {type(e).__name__}: {str(e)[:300]}", error_kind="sdk")
    if not isinstance(result, AIResult):
        return AIResult(None, "", str(getattr(provider, "name", "?")), str(getattr(provider, "model", "?")),
                        error=f"bad_response: provider returned {type(result).__name__}", error_kind="bad_response")
    return result


def interpret_tracks(tracks: Sequence[TrackFeatures], *, provider: AIProvider | None, mode: str = "cache",
                     cache: AICache, segments: Mapping[str, dict] | None = None,
                     research: Mapping[str, dict] | None = None, use_community_tags: bool = False,
                     progress: Callable[[int, int, str], None] | None = None, max_consecutive_failures: int = 5,
                     now: Callable[[], datetime] | None = None, retry_invalid: bool = False,
                     max_tokens: int | None = None) -> InterpretResult:
    """Interpret ``tracks`` (see the module docstring). ``segments`` / ``research`` are keyed by
    ``track_id``. ``retry_invalid`` (online only) asks again for inputs whose cached result is
    negative. New answers are saved at least every ``SAVE_EVERY_S`` seconds / ``SAVE_EVERY_N``
    answers and at the end, also on Ctrl+C (what was paid for is kept; whole-file rewrites stay
    bounded on large libraries). Never raises for provider, network, JSON or cache-write problems."""
    if mode not in AI_MODES:
        raise AIConfigError(f"unknown AI mode {mode!r}; use one of {', '.join(AI_MODES)}")
    result = InterpretResult()
    if mode == "off":
        return result
    if mode == "online" and provider is None:
        raise AIConfigError("AI mode 'online' needs a provider (set VEG_AI_PROVIDER or pass one)")
    if max_consecutive_failures < 1:
        raise AIConfigError(f"max_consecutive_failures must be >= 1, got {max_consecutive_failures}")
    use_provider = provider if mode == "online" else None
    # only a run that calls the provider needs (and validates) the token limit: a cache-only run
    # (e.g. `plan`) must not fail over an unrelated VEG_AI_MAX_TOKENS value
    tokens = DEFAULT_MAX_TOKENS if use_provider is None else (
        int(max_tokens) if max_tokens is not None else _max_tokens_from_env())
    now = now or (lambda: datetime.now(timezone.utc))
    stats, warn = result.stats, _Warnings(result.warnings)
    segments, research = segments or {}, research or {}
    name = str(getattr(provider, "name", "") or "unknown") if provider is not None else None
    model = str(getattr(provider, "model", "") or "unknown") if provider is not None else None
    consecutive, breaker_reason = 0, None
    no_cache: list[str] = []
    total = len(tracks)

    unsaved, last_save = 0, time.monotonic()

    def save() -> None:
        nonlocal unsaved, last_save
        unsaved, last_save = 0, time.monotonic()
        try:
            cache.save()
        except OSError as e:
            warn.add("cache", f"AI cache {cache.path} could not be saved ({type(e).__name__}: {e}); "
                              f"new answers are kept for this run only")

    def stored() -> None:
        nonlocal unsaved
        unsaved += 1
        if unsaved >= SAVE_EVERY_N or time.monotonic() - last_save >= SAVE_EVERY_S:
            save()

    try:
        for i, track in enumerate(tracks, start=1):
            if progress is not None:
                progress(i, total, track.display_name)
            payload = build_payload(track, segments=segments.get(track.track_id),
                                    research=research.get(track.track_id), use_community_tags=use_community_tags)
            if not interpretable(payload):
                stats["skipped"] += 1
                log.debug("AI: %s skipped: only tempo / key / duration known", track.display_name)
                continue
            key = cache_key(payload, provider=name, model=model) if provider is not None else None
            digest = input_digest(payload)
            if key is not None:
                entry = cache.get(key)
            else:
                found = cache.find(digest)
                entry = found[1] if found else None
            if entry is not None:
                stats["cache_hits"] += 1
                if entry["valid"]:
                    result.interpretations[track.track_id] = _record(entry, from_cache=True)
                    stats["valid"] += 1
                    stats["dropped_claims"] += len(entry.get("dropped") or [])
                    continue
                if not (use_provider is not None and retry_invalid):
                    stats["invalid"] += 1
                    warn.add("invalid", f"AI: {track.display_name}: the cached answer is invalid "
                                        f"({'; '.join(entry['errors'][:2]) or 'no details'}); run "
                                        f"`interpret --mode online --retry-invalid` to ask again")
                    continue
            if use_provider is None:
                stats["skipped"] += 1
                no_cache.append(track.display_name)
                continue
            if breaker_reason is not None:
                stats["skipped"] += 1
                continue

            res, errs, clean = _ask(use_provider, payload, tokens, stats)
            if res.error is not None and res.error_kind != INVALID_OUTPUT:
                stats["errors"] += 1
                consecutive += 1
                warn.add("error", f"AI: {track.display_name}: {res.error}")
                if res.error_kind in FATAL_ERROR_KINDS:
                    breaker_reason = f"a non-recoverable provider error ({res.error_kind})"
                elif consecutive >= max_consecutive_failures:
                    breaker_reason = f"{consecutive} consecutive provider errors"
                if breaker_reason is not None:
                    remaining = total - i
                    msg = (f"AI: stopped calling {name}/{model} after {breaker_reason}; "
                           f"{remaining} remaining track(s) not interpreted")
                    log.warning("%s", msg)
                    result.warnings.append(msg)
                continue
            consecutive = 0                     # the provider answered (even if invalidly)
            valid = clean is not None and not errs
            entry = {"output": {k: v for k, v in clean.items() if k != "dropped"} if clean is not None else None,
                     "valid": valid, "errors": list(errs), "dropped": list(clean.get("dropped", []) if clean else []),
                     "provider": name, "model": res.model or model, "request_id": res.request_id,
                     "usage": dict(res.usage), "created_at": _iso(now), "input_digest": digest,
                     "inputs": _inputs(payload)}
            try:
                cache.put(key, entry)
            except (TypeError, ValueError) as e:        # a provider returned non-JSON-native data
                warn.add("cache", f"AI: {track.display_name}: answer not cached ({e})")
            else:
                stored()
            if valid:
                stats["valid"] += 1
                stats["dropped_claims"] += len(entry["dropped"])
                result.interpretations[track.track_id] = _record(entry, from_cache=False)
            else:
                stats["invalid"] += 1
                warn.add("invalid", f"AI: {track.display_name}: invalid answer after one re-ask "
                                    f"({'; '.join(errs[:2])}); cached as invalid")
    finally:
        save()
    if no_cache:
        msg = (f"AI: {len(no_cache)} track(s) have no cached interpretation (e.g. {', '.join(no_cache[:3])}); "
               f"run the interpretation online to create them")
        log.warning("%s", msg)
        result.warnings.append(msg)
    warn.finish()
    return result


def _ask(provider: AIProvider, payload: dict, max_tokens: int,
         stats: dict[str, int]) -> tuple[AIResult, list[str], dict | None]:
    """Ask once; when the answer is invalid (unparsable or structurally wrong) ask ONE more time
    quoting the validator's errors. Returns (last result, errors, clean)."""
    errs: list[str] = []
    clean: dict | None = None
    res = AIResult(None, "", "", "")
    for attempt in range(2):
        stats["requests"] += 1
        if attempt:
            stats["reasks"] += 1
        res = _call(provider, SYSTEM_PROMPT, user_message(payload, errors=errs), max_tokens)
        if res.error is not None and res.error_kind != INVALID_OUTPUT:
            return res, errs, clean                     # provider failure (also on the re-ask): not cached
        if res.error is not None:                       # unparsable answer
            clean, errs = None, [res.error]
        else:
            try:
                clean, errs = validate_interpretation(res.data, payload)
            except Exception as e:   # noqa: BLE001 - untrusted model output must never abort the run
                log.warning("AI: the validator failed on an answer (%s: %s)", type(e).__name__, e, exc_info=True)
                clean, errs = None, [f"the answer could not be validated ({type(e).__name__})"]
        if clean is not None and not errs:
            break
    return res, errs, clean


__all__ = ["PROMPT_VERSION", "PROMPT_DIGEST", "PRIVACY_NOTE", "AI_MODES", "SYSTEM_PROMPT", "ROLE_DEFINITIONS",
           "PHASE_DEFINITIONS",
           "DEFAULT_MAX_TOKENS", "InterpretResult", "build_payload", "canonical_payload", "interpretable",
           "cache_key", "input_digest", "user_message", "interpret_tracks"]
