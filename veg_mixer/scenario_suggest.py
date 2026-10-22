"""AI-proposed scenario drafts (VEmix 3.0, spec §5: "AI may propose the structure").

``suggest_scenario`` asks an AI provider for a DRAFT scenario — phases with relative
durations, an energy curve, roles, moods, tolerances and diversity rules — from a text
brief written by the user and an anonymous summary of the library (``library_summary``: the
track count, the total duration, energy and tempo quantiles with the share of tracks whose
energy is known, and mood label counts). It never sends audio, file paths, file names or
track / artist names, and the draft contains no anchors or track references: those are the
curator's decisions.

The answer is untrusted: an OpenAI-compatible server without structured outputs answers in
plain JSON mode, so not even its shape is guaranteed. It is checked against
``SUGGESTION_SCHEMA`` (``ai.schema.check_instance``), mapped type-safely onto the scenario
format (``draft_to_scenario``: a wrong-typed part is ignored and its default used, a single
string where a list is expected becomes a one-item list, numbers are clipped — every repair is
recorded) and validated with the normal scenario loader (``scenario.parse_scenario``) before it
is written. Phase names become unique ASCII snake_case identifiers (``ascii_identifier``).
The loader accepts only ASCII names (letters, digits, '_' and '-', at most 48 characters,
lowercased on load), so Cyrillic is transliterated and a name without an ASCII form becomes
``phase_<n>``. An answer that is unparsable or does not make a valid scenario — and a
retryable provider error (rate limit, server error, timeout, connection) — gets ONE re-ask
quoting the problems. Then ``SuggestProviderError`` (a RuntimeError: exit 5, like
``enrich`` / ``interpret`` when every request failed) if the provider never answered
(network, key, quota, server, timeout); otherwise ``ScenarioError`` (exit 2) — a refusal, a
truncated or unparsable answer, or a draft still invalid after the re-ask. Never a KeyError /
TypeError / AttributeError. The plan itself is still computed by the deterministic planner
(``dramaturgy``), never by the model.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .ai.providers import INVALID_OUTPUT
from .ai.schema import check_instance
from .fsutil import atomic_write_text
from .models import TrackFeatures
from .scenario import ROLES, ScenarioError, parse_scenario

log = logging.getLogger(__name__)

SUGGEST_PROMPT_VERSION = "suggest-v2"
MAX_PHASES = 10
MAX_NAME_LEN = 40              # + a '_<n>' uniqueness suffix stays within the loader's 48 characters
MAX_SHAPE_ERRORS = 5           # schema deviations quoted per rejected attempt
MAX_LOGGED_REPAIRS = 10

_NUM = {"type": "number"}
_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}

SUGGESTION_SCHEMA: dict = {
    "type": "object", "additionalProperties": False,
    "required": ["scenario", "description", "target_duration_min", "phases", "moods", "transitions", "diversity",
                 "rationale"],
    "properties": {
        "scenario": _STR,
        "description": _STR,
        "target_duration_min": _NUM,
        "phases": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "share", "energy_start", "energy_end", "roles", "prefer_moods", "avoid_moods",
                         "contrast", "purpose"],
            "properties": {"name": {"type": "string",
                                    "description": "ASCII snake_case identifier in English, e.g. first_peak"},
                           "share": _NUM, "energy_start": _NUM, "energy_end": _NUM,
                           "roles": {"type": "array", "items": {"type": "string", "enum": list(ROLES)}},
                           "prefer_moods": _STRS, "avoid_moods": _STRS, "contrast": {"type": "boolean"},
                           "purpose": _STR}}},
        "moods": {"type": "object", "additionalProperties": False, "required": ["prefer", "avoid"],
                  "properties": {"prefer": _STRS, "avoid": _STRS}},
        "transitions": {"type": "object", "additionalProperties": False,
                        "required": ["max_bpm_jump", "max_energy_jump", "contrast_energy_jump"],
                        "properties": {"max_bpm_jump": _NUM, "max_energy_jump": _NUM, "contrast_energy_jump": _NUM}},
        "diversity": {"type": "object", "additionalProperties": False,
                      "required": ["artist_min_gap", "max_tracks_per_artist", "avoid_same_song"],
                      "properties": {"artist_min_gap": {"type": "integer"},
                                     "max_tracks_per_artist": {"type": "integer"},
                                     "avoid_same_song": {"type": "boolean"}}},
        "rationale": _STR,
    },
}

SYSTEM_PROMPT = f"""You design the dramaturgy of a music playlist (prompt {SUGGEST_PROMPT_VERSION}).
Return ONE JSON object that matches the given schema: a scenario DRAFT for a deterministic planner.

Rules:
- Work only from the user's brief and the library summary. Do not name tracks or artists and do not
  invent facts about the library; the summary is all you know about it.
- Keep the schema's shape exactly: 'phases' is a list of objects; 'moods', 'transitions' and 'diversity'
  are objects; 'roles', 'prefer_moods', 'avoid_moods', 'moods.prefer' and 'moods.avoid' are lists of strings.
- Energy is on a 0..1 scale (the library summary shows what the library actually covers: do not plan a
  curve far outside the available range).
- phases: 3 to {MAX_PHASES} phases in playing order. 'name' is a short ASCII snake_case identifier in
  English (lowercase letters a-z, digits and '_', e.g. intro, build, first_peak, breather, climax, closer),
  unique within the scenario, also when the brief is written in another language; 'purpose' says what the
  phase does and may use the brief's language. 'share' is the relative duration (they are normalised);
  energy_start / energy_end are the target energy at the phase start and end; 'contrast' = true allows a
  deliberate energy jump when ENTERING that phase (e.g. a sudden quiet section after a peak).
- roles must come from: {", ".join(ROLES)}.
- Moods are short labels (e.g. ominous, playful, tense, euphoric, melancholic, sensual, mysterious).
- transitions: max_bpm_jump in BPM (folded half/double time, typically 8-20), max_energy_jump 0.15-0.45,
  contrast_energy_jump larger than max_energy_jump (<= 0.8).
- diversity: artist_min_gap 1-5 other tracks between two tracks of one artist; max_tracks_per_artist >= 1.
- rationale: 2-5 sentences explaining the structure in terms of the brief."""


def library_summary(tracks: Sequence[TrackFeatures], *, interpretations: Mapping[str, dict] | None = None) -> dict:
    """Anonymous summary of the library — no names, no paths: the track count, the total
    duration (minutes), energy and BPM quantiles, the share of tracks with a known energy (with a
    note on what those values are) and mood label counts (the tracks' moods, plus the moods of
    ``interpretations`` when given)."""
    def q(values: list[float]) -> dict | None:
        if not values:
            return None
        v = sorted(values)
        pick = lambda p: round(v[min(len(v) - 1, int(round(p * (len(v) - 1))))], 3)   # noqa: E731
        return {"min": pick(0.0), "p25": pick(0.25), "median": pick(0.5), "p75": pick(0.75), "max": pick(1.0)}

    energies = [float(t.energy) for t in tracks if t.energy is not None and math.isfinite(float(t.energy))]
    bpms = [float(t.bpm) for t in tracks if t.bpm and math.isfinite(float(t.bpm))]
    moods: dict[str, int] = {}
    for t in tracks:
        if t.mood:
            for m in str(t.mood).replace(",", ";").split(";"):
                if m.strip():
                    moods[m.strip().lower()] = moods.get(m.strip().lower(), 0) + 1
    for interp in (interpretations or {}).values():
        for m in interp.get("moods") or []:
            if isinstance(m, dict) and m.get("label"):
                key = str(m["label"]).lower()
                moods[key] = moods.get(key, 0) + 1
    return {
        "tracks": len(tracks),
        "total_duration_min": round(sum(float(t.duration_s or 0.0) for t in tracks) / 60.0, 1),
        "energy": q(energies), "energy_known_share": round(len(energies) / max(1, len(tracks)), 3),
        "energy_note": "curated values where given, otherwise a low-confidence automatic estimate",
        "bpm": q(bpms),
        "moods": dict(sorted(moods.items(), key=lambda kv: -kv[1])[:25]),
    }


# ---------------------------------------------------------------------------
# Phase names: ASCII snake_case identifiers
# ---------------------------------------------------------------------------

# Practical Russian romanisation (+ the Ukrainian / Belarusian letters) and a few Latin letters
# that NFKD does not decompose. Deterministic: the same name always gives the same identifier.
_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
    "э": "e", "ю": "yu", "я": "ya", "є": "ye", "і": "i", "ї": "yi", "ґ": "g", "ў": "u",
    "ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "ł": "l", "đ": "d", "þ": "th",
})
_NON_ID_RE = re.compile(r"[^a-z0-9]+")


def ascii_identifier(text: str, *, max_len: int = MAX_NAME_LEN) -> str:
    """Deterministic ASCII snake_case form of ``text``: Cyrillic transliterated, accents
    stripped, every run of other characters one '_' ('Первый пик!' -> 'pervyy_pik',
    'First-Peak' -> 'first_peak'); '' when nothing ASCII remains (e.g. CJK)."""
    s = unicodedata.normalize("NFC", str(text)).lower().translate(_TRANSLIT)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = _NON_ID_RE.sub("_", s).strip("_")
    return s[:max_len].rstrip("_")


# ---------------------------------------------------------------------------
# Type-safe mapping of the model's JSON
# ---------------------------------------------------------------------------

def _kind(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "a boolean"
    if isinstance(v, (int, float)):
        return "a number"
    return {str: "a string", list: "a list", dict: "an object"}.get(type(v), type(v).__name__)


def _brief(v: Any) -> str:
    """A short, safe rendering of a scalar (a container is named by its kind only)."""
    if v is None or isinstance(v, (bool, int, float)):
        return repr(v) if not isinstance(v, int) or abs(v) < 10 ** 15 else "a huge number"
    if isinstance(v, str):
        s = repr(v)
        return s if len(s) <= 40 else s[:36] + "...'"
    return _kind(v)


class _Repairs:
    """Typed readers that never raise: each one records what it had to repair."""

    def __init__(self, sink: list[str]) -> None:
        self.notes = sink

    def note(self, message: str) -> None:
        self.notes.append(message)

    def mapping(self, value: Any, where: str) -> Mapping[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return value
        self.note(f"{where}: expected an object, got {_kind(value)}; defaults used")
        return {}

    def number(self, value: Any, where: str, lo: float, hi: float, default: float) -> float:
        if value is None:
            self.note(f"{where}: missing; {default:g} used")
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            self.note(f"{where}: expected a number, got {_kind(value)}; {default:g} used")
            return default
        try:
            v = float(value)
        except (ValueError, OverflowError):
            self.note(f"{where}: {_brief(value)} is not a number; {default:g} used")
            return default
        if not math.isfinite(v):
            self.note(f"{where}: {_brief(value)} is not a finite number; {default:g} used")
            return default
        if isinstance(value, str):
            self.note(f"{where}: the string {_brief(value)} read as a number")
        if v < lo or v > hi:
            clipped = min(hi, max(lo, v))
            self.note(f"{where}: {v:g} clipped to {clipped:g}")
            return clipped
        return v

    def flag(self, value: Any, where: str, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        self.note(f"{where}: expected true or false, got {_brief(value)}; {str(default).lower()} used")
        return default

    def text(self, value: Any, where: str, limit: int) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            self.note(f"{where}: expected a string, got {_kind(value)}; ignored")
            return None
        return " ".join(value.split())[:limit] or None

    def strings(self, value: Any, where: str, limit: int) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            self.note(f"{where}: a single string instead of a list; used as a one-item list")
            value = [value]
        elif not isinstance(value, list):
            self.note(f"{where}: expected a list of strings, got {_kind(value)}; ignored")
            return []
        out: list[str] = []
        for j, item in enumerate(value):
            if not isinstance(item, str):
                self.note(f"{where}[{j}]: expected a string, got {_kind(item)}; ignored")
                continue
            s = " ".join(item.split())[:60]
            if s and s not in out:
                out.append(s)
        if len(out) > limit:
            self.note(f"{where}: {len(out)} items; the first {limit} kept")
        return out[:limit]

    def phase_name(self, value: Any, where: str, n: int, seen: set[str]) -> str:
        fallback = f"phase_{n}"
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            self.note(f"{where}: {'missing' if value is None else 'expected a string, got ' + _kind(value)}; "
                      f"named {fallback}")
            name = fallback
        else:
            name = ascii_identifier(str(value))
            if not name:
                self.note(f"{where}: {_brief(value)} has no ASCII form; named {fallback}")
                name = fallback
            elif not str(value).isascii():
                self.note(f"{where}: {_brief(value)} written as {name!r} (phase names are ASCII identifiers)")
        base, k = name, 2
        while name in seen:
            name = f"{base}_{k}"
            k += 1
        if name != base:
            self.note(f"{where}: duplicate name {base!r} renamed {name!r}")
        seen.add(name)
        return name


def draft_to_scenario(data: Mapping[str, Any], *, warnings: list[str] | None = None) -> dict:
    """Map the model's JSON onto the scenario format (values clipped to valid ranges).

    Type-safe for any JSON value: a part of the wrong type is ignored (its default is used), a
    single string where a list is expected becomes a one-item list, and phase names become
    unique ASCII snake_case identifiers (``ascii_identifier``). Every such repair is appended to
    ``warnings`` (when given). Raises ScenarioError only when ``data`` is not an object."""
    if not isinstance(data, Mapping):
        raise ScenarioError(f"suggest: the AI answer is {_kind(data)}, expected a JSON object")
    fix = _Repairs(warnings if warnings is not None else [])

    phases = []
    seen: set[str] = set()
    raw_phases = data.get("phases")
    items: list = []
    if isinstance(raw_phases, list):
        items = raw_phases[:MAX_PHASES]
        if len(raw_phases) > MAX_PHASES:
            fix.note(f"phases: {len(raw_phases)} phases; the first {MAX_PHASES} used")
    elif raw_phases is not None:
        fix.note(f"phases: expected a list, got {_kind(raw_phases)}; ignored")
    for i, p in enumerate(items):
        where = f"phases[{i}]"
        if not isinstance(p, Mapping):
            fix.note(f"{where}: expected an object, got {_kind(p)}; ignored")
            continue
        name = fix.phase_name(p.get("name"), f"{where}.name", i + 1, seen)
        roles: list[str] = []
        for r in fix.strings(p.get("roles"), f"{where}.roles", 2 * len(ROLES)):
            norm = r.strip().lower().replace(" ", "_").replace("-", "_")
            if norm not in ROLES:
                fix.note(f"{where}.roles: unknown role {_brief(r)} ignored")
            elif norm not in roles:
                roles.append(norm)
        phase = {"name": name, "share": fix.number(p.get("share"), f"{where}.share", 0.01, 100.0, 1.0),
                 "energy": [round(fix.number(p.get("energy_start"), f"{where}.energy_start", 0.0, 1.0, 0.5), 3),
                            round(fix.number(p.get("energy_end"), f"{where}.energy_end", 0.0, 1.0, 0.5), 3)],
                 "roles": roles,
                 "contrast": fix.flag(p.get("contrast"), f"{where}.contrast", False)}
        description = fix.text(p.get("purpose"), f"{where}.purpose", 200)
        if description:
            phase["description"] = description
        moods = {"prefer": fix.strings(p.get("prefer_moods"), f"{where}.prefer_moods", 8),
                 "avoid": fix.strings(p.get("avoid_moods"), f"{where}.avoid_moods", 8)}
        if moods["prefer"] or moods["avoid"]:
            phase["moods"] = moods
        phases.append(phase)

    raw_name = data.get("scenario")
    if raw_name is not None and (isinstance(raw_name, bool) or not isinstance(raw_name, (str, int, float))):
        fix.note(f"scenario: expected a string, got {_kind(raw_name)}; named 'draft'")
        raw_name = None
    scenario_name = "".join(ch if ch.isalnum() or ch in "_-" else "_"
                            for ch in str(raw_name if raw_name is not None else "").strip())[:60] or "draft"
    tr = fix.mapping(data.get("transitions"), "transitions")
    mj = round(fix.number(tr.get("max_energy_jump"), "transitions.max_energy_jump", 0.05, 0.9, 0.3), 3)
    cj = fix.number(tr.get("contrast_energy_jump"), "transitions.contrast_energy_jump", 0.05, 0.95, max(mj, 0.5))
    if cj < mj:
        fix.note(f"transitions.contrast_energy_jump: {cj:g} raised to max_energy_jump {mj:g}")
        cj = mj
    dv = fix.mapping(data.get("diversity"), "diversity")
    moods_top = fix.mapping(data.get("moods"), "moods")
    out = {
        "scenario": scenario_name,
        "description": fix.text(data.get("description"), "description", 500),
        "target_duration_min": round(fix.number(data.get("target_duration_min"), "target_duration_min", 1.0,
                                                24 * 60.0, 60.0), 1),
        "phases": phases,
        "moods": {"prefer": fix.strings(moods_top.get("prefer"), "moods.prefer", 10),
                  "avoid": fix.strings(moods_top.get("avoid"), "moods.avoid", 10)},
        "transitions": {"max_bpm_jump": round(fix.number(tr.get("max_bpm_jump"), "transitions.max_bpm_jump",
                                                         2.0, 60.0, 16.0), 1),
                        "max_energy_jump": mj,
                        "contrast_energy_jump": round(cj, 3)},
        "diversity": {"artist_min_gap": int(round(fix.number(dv.get("artist_min_gap"), "diversity.artist_min_gap",
                                                             0, 10, 2))),
                      "max_tracks_per_artist": int(round(fix.number(dv.get("max_tracks_per_artist"),
                                                                    "diversity.max_tracks_per_artist", 1, 50, 3))),
                      "avoid_same_song": fix.flag(dv.get("avoid_same_song"), "diversity.avoid_same_song", True)},
    }
    if out["description"] is None:
        out.pop("description")
    return out


# ---------------------------------------------------------------------------
# Provider round trip
# ---------------------------------------------------------------------------

def _shape_errors(data: Any) -> list[str]:
    try:
        return check_instance(data, SUGGESTION_SCHEMA, "answer")
    except RecursionError:          # absurd nesting (parse_json_text already refuses most of it)
        return ["answer: nested too deeply"]
    except (OverflowError, ValueError) as e:      # defensive: a value no number type can hold
        return [f"answer: unusable value ({e})"]


def _rejection(reason: str, shape: list[str]) -> str:
    """One rejected attempt: why the loader refused the draft, plus how the answer deviated from
    the schema (the likely cause, quoted in the re-ask)."""
    if not shape:
        return reason
    shown = "; ".join(e[:160] for e in shape[:MAX_SHAPE_ERRORS])
    more = len(shape) - MAX_SHAPE_ERRORS
    return f"{reason} (the answer does not match the schema: {shown}{f'; ... and {more} more' if more > 0 else ''})"


def _report(repairs: list[str], sink: list[str] | None) -> None:
    for r in repairs[:MAX_LOGGED_REPAIRS]:
        log.warning("suggest: the AI draft was repaired: %s", r)
    if len(repairs) > MAX_LOGGED_REPAIRS:
        log.warning("suggest: ... and %d more repairs of the AI draft", len(repairs) - MAX_LOGGED_REPAIRS)
    if sink is not None:
        sink.extend(repairs)


def suggest_scenario(provider, brief: str, summary: Mapping[str, Any], *, target_duration_min: float | None = None,
                     max_tokens: int = 8000, warnings: list[str] | None = None) -> tuple[dict, str, list[str]]:
    """Ask ``provider`` for a draft. Returns (scenario dict, rationale, problems): the dict is
    validated by ``parse_scenario``; ``problems`` describe REJECTED attempts. An answer that is
    unparsable, not an object or does not make a valid scenario (and a retryable provider error)
    gets one re-ask quoting the problems; then SuggestProviderError when the provider never
    answered, else ScenarioError. Repairs of the accepted answer (see ``draft_to_scenario``) are
    logged as warnings and appended to ``warnings`` when a list is given."""
    if not isinstance(brief, str) or not brief.strip():
        raise ScenarioError("suggest: the brief is empty")
    user = json.dumps({"brief": brief.strip(), "target_duration_min": target_duration_min,
                       "library_summary": summary}, ensure_ascii=False, sort_keys=True)
    problems: list[str] = []
    answered = False                  # did the provider return anything to judge at all?
    for attempt in range(2):
        prompt = user if attempt == 0 else (user + "\n\nYour previous answer was rejected: "
                                            + "; ".join(problems[-5:]) + ". Return a corrected JSON object.")
        result = provider.complete_json(system=SYSTEM_PROMPT, user=prompt, schema=SUGGESTION_SCHEMA,
                                        max_tokens=max_tokens)
        error, data = getattr(result, "error", None), getattr(result, "data", None)
        # the model's own answer (even a refusal, a truncated or unparsable one) is about the brief;
        # network, key, quota and server failures are not
        answered = answered or not error or getattr(result, "error_kind", None) in _ANSWER_KINDS
        if error or not isinstance(data, dict):
            problems.append(f"provider error: {error}" if error else f"the answer is {_kind(data)}, not a JSON object")
            # an unparsable answer is worth one re-ask; a refusal, a truncated answer or a
            # non-retryable API error would only fail again
            if error and not (getattr(result, "retryable", False)
                              or getattr(result, "error_kind", None) == INVALID_OUTPUT):
                break
            continue
        shape = _shape_errors(data)
        if shape:
            log.debug("suggest: the answer deviates from the schema: %s", "; ".join(shape[:20]))
        repairs: list[str] = []
        draft = draft_to_scenario(data, warnings=repairs)
        if target_duration_min:
            draft["target_duration_min"] = round(float(target_duration_min), 1)
            repairs = [r for r in repairs if not r.startswith("target_duration_min")]   # overridden anyway
        try:
            parse_scenario(draft, source="<AI draft>")
        except ScenarioError as e:
            problems.append(_rejection(str(e), shape))
            continue
        _report(repairs, warnings)
        rationale = data.get("rationale")
        return draft, " ".join(rationale.split()) if isinstance(rationale, str) else "", problems
    if not answered:
        # network, key, quota, server, timeout: the same class of failure as interpret / enrich
        # (exit 5); a refusal or a truncated answer is an answer and ends in ScenarioError below
        raise SuggestProviderError("suggest: the AI provider gave no answer — " + "; ".join(problems[-3:]))
    raise ScenarioError("suggest: no valid scenario draft — " + "; ".join(problems[-3:]))


_ANSWER_KINDS = frozenset({INVALID_OUTPUT, "refused", "incomplete", "bad_response"})


class SuggestProviderError(RuntimeError):
    """Every request failed at the provider (no answer to judge): a runtime failure, exit 5."""


def write_draft(path: Path, draft: dict, *, rationale: str, provider_label: str, brief: str) -> None:
    header = ["# VEmix 3.0 — ЧЕРНОВИК сценария, предложенный AI (" + provider_label + ")",
              "# Проверьте и подстройте: фазы, кривую энергии, допуски. Якоря, обязательные и",
              "# исключённые треки AI не предлагает — добавьте их сами (anchors / required / excluded).",
              "# Бриф: " + " ".join(brief.split())[:300]]
    if rationale:
        words, line, lines = rationale.split(), "# Обоснование AI:", []
        for w in words:
            if len(line) + len(w) + 1 > 100:
                lines.append(line)
                line = "#  "
            line += " " + w
        lines.append(line)
        header += lines
    body = yaml.safe_dump(draft, allow_unicode=True, sort_keys=False, width=100)
    atomic_write_text(Path(path), "\n".join(header) + "\n\n" + body)


__all__ = ["SUGGEST_PROMPT_VERSION", "SUGGESTION_SCHEMA", "SYSTEM_PROMPT", "library_summary", "ascii_identifier",
           "draft_to_scenario", "suggest_scenario", "write_draft", "SuggestProviderError"]
