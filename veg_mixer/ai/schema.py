"""Strict JSON schema of an AI track interpretation (VEmix 3.0, spec §4) and its validator.

An interpretation is a set of CLAIMS about one track: moods, energy trajectory, danceability,
groove, sonic density, dramaturgical roles, suitable phases of a set and risky pairings. Every
claim carries a confidence (0..1) and EVIDENCE: key paths into the input payload the model was
given (``measured.segments.energy_out``, ``curator.mood``, ``facts.first_release_date``).

Two layers:

* ``INTERPRETATION_SCHEMA`` is sent to the provider as a structured-output schema. It uses only
  the subset that both Anthropic structured outputs and OpenAI strict mode accept: every object
  lists ALL its properties in ``required`` and has ``additionalProperties: false``; there are no
  numeric or length constraints (``minimum``, ``maxLength``, ``minItems`` > 1, ...). Ranges are
  stated in the descriptions and checked here, client side.
* ``validate_interpretation`` is a stdlib validator for that subset (type, enum, required,
  additionalProperties, items, anyOf) plus the semantic rules a schema cannot express. It is the
  "no fabrication" guard:

  - a claim that cites a key path which does NOT exist in the input is dropped: the model
    referred to data it was not given;
  - identity fields are not in the payload (artist and title are not sent since interpret-v2,
    so citing them is citing data that does not exist); in a payload that still carries a
    ``track`` block they are never evidence (such entries are removed from the evidence list);
  - provenance fields (``*_source``, ``*_confidence``, ``confidence``, ``method``, ...) say how
    reliable a value is, not what it is: a claim citing ONLY them is dropped, and they never
    satisfy the requirements below (``measured.segments.trajectory.confidence`` exists even when
    the analyzer could not determine a trajectory);
  - moods, danceability and groove must cite at least one path from their evidence whitelist
    (``EVIDENCE_WHITELISTS``, spec §1: the AI must not invent facts about a composition): a mood
    needs energy / dynamics / segment / key / tempo data or the curator's mood labels — a
    catalogue fact, a user tag or the duration alone never supports a mood; danceability and
    groove need rhythm data (tempo, percussive density) or the curator's danceability / tempo —
    key, mode, energy or loudness alone say nothing about rhythm;
  - energy trajectory and sonic density describe how a track evolves: they need
    ``measured.segments.*`` evidence (the tempo block excluded) — one whole-track energy number
    cannot show a trajectory;
  - roles and phases are dramaturgical decisions: tempo, key, duration, a genre label, user
    tags, catalogue facts or popularity ALONE are never sufficient (spec §4) — at least one piece
    of evidence must be energy / segment data or curator input other than the curator's own
    tempo / key / genre.

  Dropped claims are listed in ``clean["dropped"]`` with the reason, so a report can show what
  the model said and why it was not used. Structural problems (wrong types, unknown enum
  values, a confidence outside 0..1, missing or extra keys) are returned as ``errors``; the
  caller decides whether to re-ask.

What the validator cannot do: judge whether the cited data really supports the claim (that
``measured.segments.trajectory = "rising"`` fits ``tension_builder``). Confidence and evidence
make that reviewable; the planner weighs AI claims below measurements.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any

from ..scenario import ROLES
from ..textnorm import normalize_tag

# v2: per-field evidence whitelists for moods / danceability / groove (an answer validated under
# v1 may hold claims v2 drops). Part of every cache key: v1 answers are never served again.
SCHEMA_VERSION = "interpretation-v2"

# Dramaturgical phases of a set (spec §5). The planner's scenario phases are free names; these
# are the standard ones an interpretation may suggest.
PHASES = ("intro", "build", "first_peak", "breather", "second_build", "climax", "descent", "closer")
TRAJECTORIES = ("rising", "falling", "stable", "wave", "unknown")
DENSITIES = ("sparse", "moderate", "dense", "unknown")

# Client-side limits (the provider schema cannot express them).
CONFIDENCE_EPS = 1e-6          # float noise tolerated around 0 and 1 (clipped)
MAX_EVIDENCE = 12              # evidence entries kept per claim
MAX_ITEMS = {"moods": 12, "roles": len(ROLES), "positions": len(PHASES), "risky_pairings": 10}
MAX_LABEL_LEN = 48             # a mood label is a word or two, not a sentence
MAX_TEXT_LEN = 300             # risky_pairings.with / .reason
MAX_NOTES_LEN = 1000

# Unsupported by Anthropic structured outputs / OpenAI strict mode: never put these in the schema.
UNSUPPORTED_KEYWORDS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "minProperties", "maxProperties", "uniqueItems", "maxItems",
})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CONFIDENCE = {"type": "number",
               "description": "0..1: how strongly the cited input data supports this claim"}
_EVIDENCE = {"type": "array", "items": {"type": "string"},
             "description": "key paths into the input JSON that support the claim, dot notation with "
                            "numeric list indices, e.g. 'measured.segments.energy_out', 'curator.mood'; "
                            "only paths that exist in the input"}


def _obj(properties: dict[str, dict], description: str | None = None) -> dict:
    out: dict[str, Any] = {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}
    if description:
        out["description"] = description
    return out


def _claim(fields: dict[str, dict], description: str) -> dict:
    return _obj({**fields, "confidence": dict(_CONFIDENCE), "evidence": copy.deepcopy(_EVIDENCE)}, description)


def _nullable_score(description: str) -> dict:
    return _claim({"value": {"anyOf": [{"type": "number"}, {"type": "null"}],
                             "description": "0..1, or null when the data does not allow an estimate"}},
                  description)


INTERPRETATION_SCHEMA: dict[str, Any] = _obj({
    "moods": {"type": "array", "description": "moods the data supports (may be empty)",
              "items": _claim({"label": {"type": "string",
                                         "description": "one or two words, preferably from the mood vocabulary"}},
                              "one mood")},
    "energy_trajectory": _claim({"value": {"type": "string", "enum": list(TRAJECTORIES)}},
                                "how energy develops over the track; needs measured.segments evidence"),
    "danceability": _nullable_score("estimated danceability"),
    "groove": _nullable_score("estimated strength of the rhythmic groove"),
    "sonic_density": _claim({"value": {"type": "string", "enum": list(DENSITIES)}},
                            "how dense the arrangement is; needs measured.segments evidence"),
    "roles": {"type": "array", "description": "dramaturgical roles the track can play (may be empty)",
              "items": _claim({"role": {"type": "string", "enum": list(ROLES)}}, "one role")},
    "positions": {"type": "array", "description": "phases of a set where the track fits (may be empty)",
                  "items": _claim({"phase": {"type": "string", "enum": list(PHASES)}}, "one phase")},
    "risky_pairings": {"type": "array",
                       "description": "kinds of neighbouring tracks that would clash with this one (may be empty)",
                       "items": _claim({"with": {"type": "string",
                                                 "description": "a kind of track (mood, role, tempo, energy), "
                                                                "not a specific song"},
                                        "reason": {"type": "string"}}, "one risky pairing")},
    "notes": {"type": "string", "description": "at most two short sentences about the data; may be empty"},
}, "interpretation of one track, built only from the input data")

TOP_LEVEL_KEYS: tuple[str, ...] = tuple(INTERPRETATION_SCHEMA["required"])
_ITEM_SCHEMAS = {k: INTERPRETATION_SCHEMA["properties"][k]["items"]
                 for k in ("moods", "roles", "positions", "risky_pairings")}


def schema_subset_problems(schema: Any, path: str = "$") -> list[str]:
    """Problems that would make ``schema`` unacceptable as a strict structured-output schema:
    unsupported keywords, objects without ``additionalProperties: false`` or with properties
    missing from ``required``. Empty list = OK."""
    problems: list[str] = []
    if isinstance(schema, dict):
        for kw in sorted(UNSUPPORTED_KEYWORDS & set(schema)):
            problems.append(f"{path}: unsupported keyword {kw!r}")
        if schema.get("minItems", 0) > 1:
            problems.append(f"{path}: minItems > 1 is not supported")
        if schema.get("type") == "object":
            props = schema.get("properties", {})
            if schema.get("additionalProperties") is not False:
                problems.append(f"{path}: object without additionalProperties: false")
            missing = [k for k in props if k not in schema.get("required", [])]
            if missing:
                problems.append(f"{path}: properties not listed in required: {', '.join(missing)}")
        for key, sub in schema.items():
            if key == "properties" and isinstance(sub, dict):
                for name, s in sub.items():
                    problems += schema_subset_problems(s, f"{path}.{name}")
            elif key == "items":
                problems += schema_subset_problems(sub, f"{path}[]")
            elif key == "anyOf" and isinstance(sub, list):
                for i, s in enumerate(sub):
                    problems += schema_subset_problems(s, f"{path}|{i}")
    return problems


# ---------------------------------------------------------------------------
# Generic validator for the schema subset
# ---------------------------------------------------------------------------

def _type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


_FLOAT_MAX = 1.7976931348623157e308     # an int beyond it cannot become a float (OverflowError)


def _is_type(v: Any, t: str) -> bool:
    if t == "null":
        return v is None
    if t == "boolean":
        return isinstance(v, bool)
    if t == "integer":   # JSON Schema: any number without a fractional part (2.0 is an integer)
        return (isinstance(v, int) and not isinstance(v, bool)) or (
            isinstance(v, float) and math.isfinite(v) and v.is_integer())
    if t == "number":   # JSON has no NaN/inf: a non-finite (or float-overflowing) number is garbage
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
        return -_FLOAT_MAX <= v <= _FLOAT_MAX if isinstance(v, int) else math.isfinite(v)
    if t == "string":
        return isinstance(v, str)
    if t == "array":
        return isinstance(v, list)
    if t == "object":
        return isinstance(v, dict)
    return False


def check_instance(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Errors of ``value`` against a schema of the supported subset (type, enum, anyOf,
    properties/required/additionalProperties, items). Empty list = valid."""
    if "anyOf" in schema:
        if any(not check_instance(value, s, path) for s in schema["anyOf"]):
            return []
        return [f"{path}: {_type_name(value)} {_short(value)} matches none of the allowed forms"]
    errors: list[str] = []
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_is_type(value, x) for x in types):
            what = "non-finite number" if isinstance(value, float) and not math.isfinite(value) else _type_name(value)
            return [f"{path}: expected {' or '.join(types)}, got {what}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {_short(value)} is not one of {', '.join(map(str, schema['enum']))}")
    if isinstance(value, dict) and schema.get("type") == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errors.append(f"{path}: missing key {k!r}")
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errors.append(f"{path}: unexpected key {_short(k)}")
        for k, sub in props.items():
            if k in value:
                errors += check_instance(value[k], sub, f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += check_instance(item, schema["items"], f"{path}[{i}]")
    return errors


def _short(v: Any, n: int = 60) -> str:
    try:
        s = repr(v)
    except RecursionError:          # an absurdly nested answer: name the type only
        s = f"<deeply nested {type(v).__name__}>"
    return s if len(s) <= n else s[:n - 3] + "..."


# ---------------------------------------------------------------------------
# Evidence paths
# ---------------------------------------------------------------------------

# ASCII digits only: \d also matches '²' / other Unicode digits, which int() rejects.
_INDEX_RE = re.compile(r"\[\s*([0-9]+)\s*\]")
_DOTS_RE = re.compile(r"\.{2,}")
# Path components that mark a label / popularity value: never enough for a dramaturgical decision.
_WEAK_HINTS = ("genre", "popular", "rating", "listener", "playcount", "play_count", "chart", "rank")
# measured.* / curator.* entries that say nothing about energy or arrangement on their own: tempo,
# key and duration (spec §4: BPM is never a sufficient basis — also when the curator typed it).
_WEAK_MEASURED = frozenset({"bpm", "bpm_source", "key", "key_confidence", "duration_s"})
# Leaves that describe the reliability / origin of a value, not the value.
_PROVENANCE_LEAVES = frozenset({"confidence", "source", "sources", "method", "verified", "kind", "license_note",
                                "facts_source", "identity_source"})
# Leaves that qualify a value without being one (a BPM's octave ambiguity, the analyzer's notes).
_QUALIFIER_LEAVES = frozenset({"half_double_ambiguous", "notes"})


def normalize_evidence_path(text: str) -> str:
    """'$.measured.segments.sections[0].label' / 'input.curator.mood' -> canonical dot path."""
    s = str(text).strip().strip("`'\" ")
    if s.startswith("$"):
        s = s[1:]
    s = s.lstrip(".")
    if s.startswith("input."):
        s = s[len("input."):]
    s = _INDEX_RE.sub(r".\1", s)
    return _DOTS_RE.sub(".", s).strip(".")


def path_exists(payload: Any, path: str) -> bool:
    """True when the dot path names an existing key / list index of ``payload``."""
    if not path:
        return False
    node = payload
    for part in path.split("."):
        if isinstance(node, dict):
            if part not in node:
                return False
            node = node[part]
        elif isinstance(node, list):
            if not (part.isascii() and part.isdigit()) or int(part) >= len(node):
                return False
            node = node[int(part)]
        else:
            return False
    return True


def _admissible(path: str) -> bool:
    """Identity fields and bare top-level sections are not evidence."""
    parts = path.split(".")
    return len(parts) >= 2 and parts[0] != "track"


def _is_provenance(path: str) -> bool:
    """True for a source / confidence / method field: it qualifies a value, it is not one."""
    leaf = next((p for p in reversed(path.split(".")) if not p.isdigit()), "")
    return leaf in _PROVENANCE_LEAVES or leaf.endswith(("_source", "_confidence"))


def _resolve(payload: Any, path: str) -> Any:
    node = payload
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def _has_content(node: Any, key: str | None = None) -> bool:
    """True when the cited value (or something inside a cited block) is an actual value:
    provenance / qualifier leaves, None and 'unknown' are not."""
    if key is not None and (key in _PROVENANCE_LEAVES or key in _QUALIFIER_LEAVES
                            or key.endswith(("_source", "_confidence"))):
        return False
    if isinstance(node, dict):
        return any(_has_content(v, str(k)) for k, v in node.items())
    if isinstance(node, list):
        return any(_has_content(v) for v in node)
    return node is not None and node != "unknown"


def _is_weak_for_dramaturgy(path: str) -> bool:
    parts = path.split(".")
    head = parts[0]
    if head not in ("measured", "curator"):
        return True                                  # facts, community tags, identity
    if any(h in p for p in parts for h in _WEAK_HINTS):
        return True                                  # genre labels, popularity
    if _is_provenance(path):
        return True                                  # provenance, not content
    if len(parts) >= 2 and parts[1] in _WEAK_MEASURED:
        return True                                  # tempo / key / duration, measured or typed in
    if head == "measured" and len(parts) >= 3 and parts[1] == "segments" and parts[2] in _NON_ENERGY_SEGMENT_KEYS:
        return True
    return False


# measured.segments.* entries that say nothing about energy or arrangement: the tempo block and
# the analyzer's own caveats.
_NON_ENERGY_SEGMENT_KEYS = frozenset({"tempo", "notes"})


def _is_segment(path: str) -> bool:
    parts = path.split(".")
    return (len(parts) >= 3 and parts[0] == "measured" and parts[1] == "segments"
            and parts[2] not in _NON_ENERGY_SEGMENT_KEYS)


# Evidence whitelists per claim field (spec §1: the AI must not invent facts about a composition).
# A claim of such a field must cite at least one CONTENT path (not provenance) that equals one of
# its prefixes or lies under one ('measured.segments.tempo' covers 'measured.segments.tempo.bpm',
# 'curator.vibe_tags' covers 'curator.vibe_tags.0'); further cited paths that exist are kept but
# do not count. The prefixes are paths of ``interpret.build_payload``: extend the table when the
# payload gains a key (the analyzer's timbre vector is not sent — unlabelled numbers — so there is
# no brightness / timbre path to list).
#
# moods: what a mood can be read from — energy / intensity / loudness / dynamics, how the track
#   evolves and sounds (segment trajectory, density, percussive / bass density, vocal presence),
#   key and mode (Camelot 'A' = minor, 'B' = major; the segment harmony block) and tempo; or the
#   curator's mood, vibe tags and genre, and the curator's own energy / tempo / key / vocal
#   values (the payload lists a value the curator overrode under ``curator`` INSTEAD of
#   ``measured``). Catalogue facts, user tags, section labels, the analyzer's notes and the
#   duration never support a mood on their own.
# danceability / groove: rhythm only — tempo (BPM, the segment tempo block incl. its half/double
#   ambiguity) and percussive (onset) density, or the curator's danceability / tempo. Key, mode,
#   energy or loudness alone say nothing about rhythm.
MOOD_EVIDENCE: tuple[str, ...] = (
    "measured.energy", "measured.segments.energy_in", "measured.segments.energy_out",
    "measured.segments.energy_mean", "measured.segments.energy_peak", "measured.segments.dynamic_range_db",
    "measured.segments.trajectory", "measured.segments.density", "measured.segments.percussive_density",
    "measured.segments.bass_density", "measured.segments.vocal_presence_proxy",
    "measured.key", "measured.segments.harmony", "measured.bpm", "measured.segments.tempo",
    "curator.mood", "curator.vibe_tags", "curator.genre",
    "curator.energy", "curator.bpm", "curator.key", "curator.vocal_presence",
)
RHYTHM_EVIDENCE: tuple[str, ...] = (
    "measured.bpm", "measured.segments.tempo", "measured.segments.percussive_density",
    "curator.danceability", "curator.bpm",
)
EVIDENCE_WHITELISTS: dict[str, tuple[str, ...]] = {
    "moods": MOOD_EVIDENCE,
    "danceability": RHYTHM_EVIDENCE,
    "groove": RHYTHM_EVIDENCE,
}
_RHYTHM_REASON = ("needs rhythm evidence: measured tempo / BPM or percussive density, or the curator's "
                  "danceability / BPM (key, mode, energy, loudness, facts or user tags alone are not enough)")
WHITELIST_REASONS: dict[str, str] = {
    "moods": ("a mood needs mood-relevant evidence: measured energy, dynamics, segment trajectory / density / "
              "vocals, key / mode or tempo, or the curator's mood, vibe tags or genre (catalogue facts, user "
              "tags, section labels or duration alone are not enough)"),
    "danceability": _RHYTHM_REASON,
    "groove": _RHYTHM_REASON,
}


def under_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    """True when the dot path equals one of ``prefixes`` or lies under one (component-wise:
    'measured.energy' does not cover 'measured.energy_source')."""
    return any(path == p or path.startswith(p + ".") for p in prefixes)


# Evidence requirement per claim kind: (predicate one path must satisfy, reason when none does).
# The keys of ``EVIDENCE_WHITELISTS`` are requirements too (see ``_requirement``).
_REQUIREMENTS = {
    "segments": (_is_segment, "needs measured.segments evidence (time-resolved analysis)"),
    "dramaturgy": (lambda p: not _is_weak_for_dramaturgy(p),
                   "tempo, key, duration, genre labels, user tags, facts or popularity alone are not "
                   "sufficient for a dramaturgical decision"),
}


def _requirement(name: str) -> tuple[Any, str]:
    """(predicate, reason) of a requirement; whitelists are read at call time (data-driven)."""
    if name in EVIDENCE_WHITELISTS:
        prefixes = tuple(EVIDENCE_WHITELISTS[name])
        return (lambda p: under_prefix(p, prefixes)), WHITELIST_REASONS.get(name, f"{name}: evidence not "
                                                                                  f"in its whitelist")
    return _REQUIREMENTS[name]


# ---------------------------------------------------------------------------
# Interpretation validator
# ---------------------------------------------------------------------------

class _Checker:
    def __init__(self, payload: dict, errors: list[str]) -> None:
        self.payload = payload
        self.errors = errors
        self.dropped: list[dict] = []

    # --- helpers -------------------------------------------------------------

    def confidence(self, value: Any, where: str) -> float | None:
        """Confidence clipped to 0..1 (float noise only); None + error for anything else."""
        if not _is_type(value, "number"):
            self.errors.append(f"{where}.confidence: expected a number, got {_type_name(value)}")
            return None
        v = float(value)
        if v < -CONFIDENCE_EPS or v > 1.0 + CONFIDENCE_EPS:
            self.errors.append(f"{where}.confidence: {v:g} is outside 0..1")
            return None
        return round(min(1.0, max(0.0, v)), 3)

    def unit_score(self, value: Any, where: str) -> tuple[bool, float | None]:
        """(ok, value) for a 0..1 score or null."""
        if value is None:
            return True, None
        v = float(value)
        if v < -CONFIDENCE_EPS or v > 1.0 + CONFIDENCE_EPS:
            self.errors.append(f"{where}.value: {v:g} is outside 0..1")
            return False, None
        return True, round(min(1.0, max(0.0, v)), 3)

    def evidence(self, raw: list, *, requirement: str | None, claims_something: bool) -> tuple[list[str], str | None]:
        """(admissible evidence, reason to drop the claim or None)."""
        paths: list[str] = []
        for entry in raw:
            p = normalize_evidence_path(entry)
            if not path_exists(self.payload, p):
                if claims_something:
                    return [], f"evidence {_short(entry)} is not a key path of the input"
                continue                                   # 'unknown' claims make no assertion
            if _admissible(p) and p not in paths:
                paths.append(p)
        paths = paths[:MAX_EVIDENCE]
        if not claims_something:
            return paths, None
        if not paths:
            return [], ("no evidence" if not raw else "no admissible evidence (identity fields and whole "
                                                      "sections are not evidence)")
        content = [p for p in paths if not _is_provenance(p)
                   and _has_content(_resolve(self.payload, p), None if p.rsplit(".", 1)[-1].isdigit()
                                    else p.rsplit(".", 1)[-1])]
        if not content:
            return [], ("only provenance fields or undetermined values cited (a source / confidence says how "
                        "reliable a value is, not what it is)")
        if requirement is not None:
            pred, reason = _requirement(requirement)
            if not any(pred(p) for p in content):
                return [], reason
        return paths, None

    def drop(self, field: str, claim: dict, reason: str) -> None:
        shown = {k: (v[:MAX_EVIDENCE] if k == "evidence" and isinstance(v, list) else v)
                 for k, v in claim.items()}
        self.dropped.append({"field": field, "reason": reason, "claim": _jsonable(shown)})

    # --- claims --------------------------------------------------------------

    def scalar(self, data: dict, key: str, *, unknown: Any, requirement: str) -> dict:
        """energy_trajectory / sonic_density / danceability / groove."""
        default = {"value": unknown, "confidence": 0.0, "evidence": []}
        if key not in data:
            return default
        item = data[key]
        errs = check_instance(item, INTERPRETATION_SCHEMA["properties"][key], key)
        if errs:
            self.errors.extend(errs)
            return default
        conf = self.confidence(item["confidence"], key)
        if conf is None:
            return default
        value = item["value"]
        if key in ("danceability", "groove"):
            ok, value = self.unit_score(value, key)
            if not ok:
                return default
        claims = value != unknown
        paths, reason = self.evidence(item["evidence"], requirement=requirement, claims_something=claims)
        if reason:
            self.drop(key, item, reason)
            return default
        # an "unknown" / null value asserts nothing: its confidence carries no information
        return {"value": value, "confidence": conf if claims else 0.0, "evidence": paths}

    def items(self, data: dict, key: str, name_field: str | None, requirement: str | None) -> list[dict]:
        """moods / roles / positions / risky_pairings."""
        raw = data.get(key, [])
        if not isinstance(raw, list):
            self.errors.append(f"{key}: expected array, got {_type_name(raw)}")
            return []
        merged: dict[str, dict] = {}
        out: list[dict] = []
        for i, item in enumerate(raw):
            where = f"{key}[{i}]"
            errs = check_instance(item, _ITEM_SCHEMAS[key], where)
            if errs:
                self.errors.extend(errs)
                continue
            conf = self.confidence(item["confidence"], where)
            if conf is None:
                continue
            claim = self._claim_fields(key, item, where)
            if claim is None:
                continue
            paths, reason = self.evidence(item["evidence"], requirement=requirement, claims_something=True)
            if reason:
                self.drop(where, item, reason)
                continue
            claim.update(confidence=conf, evidence=paths)
            if name_field is None:                       # risky pairings: kept as listed
                out.append(claim)
                continue
            name = claim[name_field]
            prev = merged.get(name)
            if prev is None:
                merged[name] = claim
            else:                                        # duplicate label: max confidence, union evidence
                prev["confidence"] = max(prev["confidence"], conf)
                prev["evidence"] = (prev["evidence"] + [p for p in paths if p not in prev["evidence"]])[:MAX_EVIDENCE]
        if name_field is not None:
            out = sorted(merged.values(), key=lambda c: (-c["confidence"], c[name_field]))
        limit = MAX_ITEMS[key]
        for extra in out[limit:]:
            self.drop(key, extra, f"more than {limit} items")
        return out[:limit]

    def _claim_fields(self, key: str, item: dict, where: str) -> dict | None:
        if key == "moods":
            label = normalize_tag(item["label"])
            if not label:
                self.errors.append(f"{where}.label: empty")
                return None
            if len(label) > MAX_LABEL_LEN:
                self.errors.append(f"{where}.label: longer than {MAX_LABEL_LEN} characters")
                return None
            return {"label": label}
        if key == "roles":
            return {"role": item["role"]}
        if key == "positions":
            return {"phase": item["phase"]}
        with_, reason = item["with"].strip(), item["reason"].strip()
        if not with_ or not reason:
            self.errors.append(f"{where}: 'with' and 'reason' must not be empty")
            return None
        return {"with": with_[:MAX_TEXT_LEN], "reason": reason[:MAX_TEXT_LEN]}


def validate_interpretation(data: Any, payload: dict) -> tuple[dict | None, list[str]]:
    """Validate a provider's answer against ``INTERPRETATION_SCHEMA`` and the evidence rules.

    Returns ``(clean, errors)``. ``clean`` always has every top-level key (missing or invalid
    parts become "unknown" / null / empty) plus ``dropped`` — claims removed for evidence
    reasons, each ``{"field", "reason", "claim"}``. ``errors`` lists structural problems
    (types, enums, missing / extra keys, confidences outside 0..1); a caller treats a non-empty
    list as an invalid answer. ``clean`` is None only when the top level is unusable (not an
    object, or none of the expected keys). Never raises for malformed input.
    """
    if not isinstance(data, dict):
        return None, [f"top level is {_type_name(data)}, expected an object"]
    if not any(k in data for k in TOP_LEVEL_KEYS):
        return None, [f"none of the expected keys ({', '.join(TOP_LEVEL_KEYS)}) is present"]
    try:
        return _validate(data, payload)
    except RecursionError:          # nesting near the interpreter limit (json.loads accepted it)
        return None, ["answer is nested too deeply"]


def _validate(data: dict, payload: Any) -> tuple[dict, list[str]]:
    errors: list[str] = [f"missing key {k!r}" for k in TOP_LEVEL_KEYS if k not in data]
    errors += [f"unexpected key {_short(k)}" for k in data if k not in TOP_LEVEL_KEYS]
    ck = _Checker(payload if isinstance(payload, dict) else {}, errors)
    clean: dict[str, Any] = {
        "moods": ck.items(data, "moods", "label", "moods"),
        "energy_trajectory": ck.scalar(data, "energy_trajectory", unknown="unknown", requirement="segments"),
        "danceability": ck.scalar(data, "danceability", unknown=None, requirement="danceability"),
        "groove": ck.scalar(data, "groove", unknown=None, requirement="groove"),
        "sonic_density": ck.scalar(data, "sonic_density", unknown="unknown", requirement="segments"),
        "roles": ck.items(data, "roles", "role", "dramaturgy"),
        "positions": ck.items(data, "positions", "phase", "dramaturgy"),
        "risky_pairings": ck.items(data, "risky_pairings", None, None),
        "notes": "",
    }
    notes = data.get("notes", "")
    if isinstance(notes, str):
        clean["notes"] = notes.strip()[:MAX_NOTES_LEN]
    else:
        errors.append(f"notes: expected string, got {_type_name(notes)}")
    clean["dropped"] = ck.dropped
    return clean, errors


def _jsonable(v: Any) -> Any:
    """JSON-native copy (non-finite floats -> None, unknown objects -> repr)."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if v is None or isinstance(v, (str, int, bool)):
        return v
    return repr(v)


__all__ = ["SCHEMA_VERSION", "PHASES", "ROLES", "TRAJECTORIES", "DENSITIES", "INTERPRETATION_SCHEMA",
           "TOP_LEVEL_KEYS", "UNSUPPORTED_KEYWORDS", "MOOD_EVIDENCE", "RHYTHM_EVIDENCE", "EVIDENCE_WHITELISTS",
           "WHITELIST_REASONS", "validate_interpretation", "check_instance", "schema_subset_problems",
           "normalize_evidence_path", "path_exists", "under_prefix"]
