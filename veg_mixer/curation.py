"""Client-specific curation rules (spec §6) and daypart definitions (spec §7).

A client profile (YAML or JSON, see ``profiles/client.yaml``) expresses EDITORIAL
preferences — it never triggers downloading or "discovering" music, and nothing in
here claims that an unknown track is popular, familiar or suitable.

Rule precedence in :func:`curate` (first group wins, see docs/ARCHITECTURE.md):

1. hard rules (an allowlist cannot override them)
   blocklist (tracks / artists / tags) → ``excluded`` in curator metadata →
   ANY of the track's languages listed in ``language.excluded`` ('ru;en' is excluded
   by ``excluded: [ru]``);
2. ``allowlist.only`` — nothing outside the allowlist is eligible;
3. soft rules
   a language not in ``language.allowed`` (a multi-language track passes only when ALL
   its languages are allowed) → unknown language, including unrecognised tag values
   (``unknown_policy``) → Cyrillic script with unknown language
   (``cyrillic_script_policy``) → ``avoid`` preferences → ``require_approved`` →
   ``manual_review_required``.
   An allowlisted track is never EXCLUDED by a soft rule, but the allowlist does not
   hide unverified facts: unknown language, the Cyrillic hint and "not approved" stay
   on the track as review flags.

Review flags never remove a track: it stays eligible with a human-readable reason, and
an eligible track with any review reason (curation's or analysis') has
``requires_review`` set. A Cyrillic artist/title whose language comes from a file tag
(not the curator) and is not a Cyrillic-script language is flagged as a conflict.

``editorial_score`` (−1..+1) is the mean weight of the preferences a track matches
(high +1, medium 0, low −0.6, avoid −1). A familiarity-based preference (e.g.
``recognizable_pop``) always takes part in the mean: a curator tag ('hit') counts the
full weight, curated ``familiarity`` f counts weight × (2f − 1), and an UNKNOWN
familiarity counts 0 (neutral) — so curated-familiar > unknown > curated-obscure and
an unrated track can never outrank a curated hit.

Text matching (tags, artists, languages, paths) is shared with sequencing and QC
through :mod:`veg_mixer.textnorm`.
"""

from __future__ import annotations

import copy
import difflib
import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .models import (
    ClientProfile,
    CurationDecision,
    CurationResult,
    DaypartProfile,
    LanguageRules,
    LevelSetting,
    ListRules,
    PreferenceLevel,
    TrackFeatures,
)
from .textnorm import (  # noqa: F401  (ISO639_2_TO_1 / normalize_language: public re-exports)
    ISO639_2_TO_1,
    UNKNOWN_LANGUAGE_TOKENS,
    normalize_artist,
    normalize_language,
    normalize_tag,
    parse_languages,
    rel_path_key,
    split_artists,
    split_values,
)

log = logging.getLogger(__name__)


class ProfileError(ValueError):
    """Invalid client profile; the message names the file and the offending key."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNKNOWN_POLICIES = ("review", "exclude", "allow")
CYRILLIC_POLICIES = ("exclude", "review", "ignore")

PREFERENCE_WEIGHTS: dict[PreferenceLevel, float] = {
    PreferenceLevel.HIGH: 1.0,
    PreferenceLevel.MEDIUM: 0.0,
    PreferenceLevel.LOW: -0.6,
    PreferenceLevel.AVOID: -1.0,   # only reached for allowlisted tracks (others are excluded)
}

# In ``preference_tags`` this token means "matched through curated familiarity".
FAMILIARITY_TOKEN = "@familiarity"
# Preferences without an explicit ``preference_tags`` entry whose name looks like this
# implicitly use curated familiarity (e.g. the spec's ``recognizable_pop``).
FAMILIARITY_NAME_RE = re.compile(r"recogni[sz]|familiar|popular|well[ _-]?known|(?:^|[ _-])hits?(?:$|[ _-])",
                                 re.IGNORECASE)

FREE_TEXT_KEYS = ("description", "notes")
TOP_LEVEL_KEYS = ("client", "language", "preferences", "preference_tags", "vibe_consistency",
                  "energy_jumps", "manual_review_required", "require_approved", "blocklist",
                  "allowlist", "dayparts")
LANGUAGE_KEYS = ("allowed", "excluded", "unknown_policy", "cyrillic_script_policy")
LIST_RULE_KEYS = ("tracks", "artists", "tags")
DAYPART_KEYS = ("energy_range", "energy_curve", "opening_tracks", "opening_max_energy", "bpm_range",
                "preferred_tags", "avoid_tags", "preferred_moods", "preferred_genres",
                "target_duration_min", "max_tracks")

# Human-readable reasons (stable wording: reports and tests rely on them).
CYRILLIC_EXCLUDE_REASON = ("Cyrillic script in artist/title; language unverified — "
                           "excluded pending curator confirmation")
CYRILLIC_REVIEW_REASON = ("Cyrillic script in artist/title; language unverified — "
                          "needs curator confirmation")
UNKNOWN_LANGUAGE_REVIEW_REASON = "language unknown — needs curator review (not assumed to be English)"
UNKNOWN_LANGUAGE_EXCLUDE_REASON = ("language unknown — excluded by the client profile "
                                   "(language.unknown_policy: exclude)")
NOT_APPROVED_REVIEW_REASON = "not yet approved by curator"
NOT_APPROVED_EXCLUDE_REASON = "not approved by curator (the client profile sets require_approved)"
CYRILLIC_TAG_CONFLICT_REASON = ("Cyrillic script in artist/title but the file tag says language '{lang}' "
                                "(not confirmed by a curator) — needs curator confirmation")
# The hint analysis.py leaves on a track with a Cyrillic artist/title. Curation always
# re-decides the Cyrillic question from ``script_hint`` and replaces this hint with its own
# verdict (a review/exclusion reason above, or nothing once a curator set the language).
ANALYSIS_CYRILLIC_HINT_REASON = "Cyrillic script in artist/title — language not verified"

# Kept for backward compatibility: the shared tables live in textnorm (ISO639_2_TO_1 is
# re-exported from there; tag placeholders such as 'und' / 'zxx' mean "unknown").
UNKNOWN_LANGUAGE_VALUES = UNKNOWN_LANGUAGE_TOKENS
# Languages normally written in Cyrillic: with one of these a Cyrillic artist/title is
# expected; with any other (tag-sourced) language it is a conflict worth a review.
CYRILLIC_SCRIPT_LANGUAGES = frozenset({"ru", "uk", "be", "bg", "sr", "kk", "mk", "mn", "ky", "tg", "tt"})

# Daypart names become output folder names / file-name suffixes (pipeline): no path
# separators, no characters Windows forbids, no '.'/'..', no reserved device names.
_DAYPART_NAME_FORBIDDEN_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_WINDOWS_RESERVED_NAME_RE = re.compile(r"^(?:con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?$", re.IGNORECASE)
DAYPART_NAME_MAX_LEN = 64

# ---------------------------------------------------------------------------
# Dayparts (spec §7) — EDITORIAL STARTING POINTS, not measured truths.
#
# Energy values are on the 0..1 scale of TrackFeatures.energy (curated, or the
# low-confidence heuristic). energy_range bounds selection, energy_curve is the target
# energy at the start/end of the block, opening_max_energy caps the first
# ``opening_tracks`` tracks ("avoid aggressive openings"). Tags/moods/genres are
# matched case-insensitively against vibe_tags / mood / genre. Tune per client in
# the profile; anything not overridden there falls back to these numbers.
# ---------------------------------------------------------------------------

DEFAULT_DAYPARTS: dict[str, DaypartProfile] = {
    # gentle to moderate energy, accessible melodic music, no aggressive openings
    "morning": DaypartProfile(
        name="morning",
        energy_range=(0.15, 0.60),
        energy_curve=(0.30, 0.50),
        opening_tracks=3,
        opening_max_energy=0.40,
        bpm_range=None,
        preferred_tags=["melodic", "accessible", "acoustic", "mellow", "warm"],
        avoid_tags=["aggressive", "harsh", "heavy"],
        preferred_moods=["calm", "warm", "relaxed", "uplifting", "bright"],
        preferred_genres=["pop", "acoustic pop", "soft rock", "singer-songwriter"],
    ),
    # moderate energy, recognizable pop, a steady working atmosphere
    "afternoon": DaypartProfile(
        name="afternoon",
        energy_range=(0.30, 0.70),
        energy_curve=(0.45, 0.55),
        opening_tracks=3,
        opening_max_energy=None,
        bpm_range=None,
        preferred_tags=["recognizable", "pop", "upbeat", "feel-good", "steady"],
        avoid_tags=["aggressive", "harsh", "experimental"],
        preferred_moods=["upbeat", "positive", "cheerful", "steady"],
        preferred_genres=["pop", "dance pop", "indie pop", "soft rock"],
    ),
    # moderate to moderately high energy, more groove/rhythm, no abrupt style changes
    "evening": DaypartProfile(
        name="evening",
        energy_range=(0.40, 0.85),
        energy_curve=(0.50, 0.70),
        opening_tracks=3,
        opening_max_energy=0.60,
        bpm_range=None,
        preferred_tags=["groove", "rhythmic", "funky", "danceable"],
        avoid_tags=["harsh"],
        preferred_moods=["groovy", "energetic", "confident", "warm"],
        preferred_genres=["funk", "disco", "nu-disco", "dance pop", "r&b", "soul"],
    ),
}


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def default_profile() -> ClientProfile:
    """Permissive profile used when no ``--profile`` is given: nothing is excluded by
    language, but an unknown language and a Cyrillic artist/title are still FLAGGED for
    review (spec §6: unknown language is never silently accepted); default dayparts."""
    return ClientProfile(
        client="default",
        language=LanguageRules(allowed=[], excluded=[], unknown_policy="review", cyrillic_script_policy="review"),
        dayparts=copy.deepcopy(DEFAULT_DAYPARTS),
    )


def load_profile(path: Path) -> ClientProfile:
    """Load and strictly validate a client profile (.yaml/.yml or .json)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".yaml", ".yml", ".json"):
        raise ProfileError(f"{path}: unsupported profile format '{suffix}' (use .yaml, .yml or .json)")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise ProfileError(f"{path}: cannot read profile: {e}") from e
    if suffix == ".json":
        try:
            data = json.loads(text, object_pairs_hook=_json_no_duplicates)
        except json.JSONDecodeError as e:
            raise ProfileError(f"{path}: invalid JSON at line {e.lineno}: {e.msg}") from None
        except _DuplicateKeyError as e:
            raise ProfileError(f"{path}: invalid JSON: duplicate key '{e}'") from None
    else:
        try:
            data = yaml.load(text, Loader=_StrictYamlLoader)   # SafeLoader + duplicate-key check
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            where = f" at line {mark.line + 1}" if mark is not None else ""
            problem = getattr(e, "problem", None) or str(e).splitlines()[0]
            raise ProfileError(f"{path}: invalid YAML{where}: {problem}") from None
    profile = parse_profile(data, source=str(path))
    profile.source_path = path
    log.info("profile '%s' loaded from %s", profile.client, path)
    return profile


class _StrictYamlLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys (plain PyYAML silently keeps the
    last one — e.g. a pasted second ``dayparts:`` block would replace the first)."""

    def construct_mapping(self, node, deep=False):
        seen: dict[Any, int] = {}
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=True)
            try:
                first = seen.get(key)
            except TypeError:          # unhashable key: let the base class report it
                continue
            if first is not None:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"found duplicate key {key!r} (first defined at line {first})", key_node.start_mark)
            seen[key] = key_node.start_mark.line + 1
        return super().construct_mapping(node, deep=deep)


class _DuplicateKeyError(ValueError):
    pass


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKeyError(key)
        out[key] = value
    return out


def parse_profile(data: Any, *, source: str = "<profile>") -> ClientProfile:
    """Validate an already-parsed profile mapping (YAML/JSON object)."""
    p = _Parser(source)
    if data is None:
        raise ProfileError(f"{source}: profile is empty")
    p.mapping(data, "", TOP_LEVEL_KEYS)

    client = data.get("client")
    if isinstance(client, bool) or not isinstance(client, (str, int, float)) or not str(client).strip():
        p.fail("client", "required: a non-empty client name")
    profile = ClientProfile(client=str(client).strip())

    lang = data.get("language")
    if lang is not None:
        p.mapping(lang, "language", LANGUAGE_KEYS)
        profile.language = LanguageRules(
            allowed=p.languages(lang.get("allowed"), "language.allowed"),
            excluded=p.languages(lang.get("excluded"), "language.excluded"),
            unknown_policy=p.choice(lang.get("unknown_policy", "review"), "language.unknown_policy",
                                    UNKNOWN_POLICIES),
            cyrillic_script_policy=p.choice(lang.get("cyrillic_script_policy", "review"),
                                            "language.cyrillic_script_policy", CYRILLIC_POLICIES),
        )
        both = sorted(set(profile.language.allowed) & set(profile.language.excluded))
        if both:
            p.fail("language", f"code(s) {', '.join(both)} are both allowed and excluded")

    prefs = data.get("preferences")
    if prefs is not None:
        p.mapping(prefs, "preferences", None)
        levels = [lv.value for lv in PreferenceLevel]
        for name, level in prefs.items():
            if _is_free_text(name, level):
                continue
            key = f"preferences.{name}"
            p.name(name, "preferences")
            profile.preferences[name] = PreferenceLevel(p.choice(level, key, levels))

    ptags = data.get("preference_tags")
    if ptags is not None:
        p.mapping(ptags, "preference_tags", None)
        for name, tags in ptags.items():
            if _is_free_text(name, tags):
                continue
            p.name(name, "preference_tags")
            if name not in profile.preferences:
                p.fail(f"preference_tags.{name}", "no preference with this name in 'preferences'"
                       + _did_you_mean(name, profile.preferences))
            profile.preference_tags[name] = p.strings(tags, f"preference_tags.{name}")

    profile.vibe_consistency = LevelSetting(p.choice(data.get("vibe_consistency", "medium"), "vibe_consistency",
                                                     [lv.value for lv in LevelSetting]))
    profile.energy_jumps = LevelSetting(p.choice(data.get("energy_jumps", "medium"), "energy_jumps",
                                                 [lv.value for lv in LevelSetting]))
    profile.manual_review_required = p.boolean(data.get("manual_review_required", False), "manual_review_required")
    profile.require_approved = p.boolean(data.get("require_approved", False), "require_approved")
    profile.blocklist = p.list_rules(data.get("blocklist"), "blocklist", allow_only=False)
    profile.allowlist = p.list_rules(data.get("allowlist"), "allowlist", allow_only=True)
    if profile.allowlist.only and not (profile.allowlist.tracks or profile.allowlist.artists
                                       or profile.allowlist.tags):
        p.fail("allowlist.only", "is true but the allowlist is empty — every track would be excluded")

    profile.dayparts = copy.deepcopy(DEFAULT_DAYPARTS)
    dayparts = data.get("dayparts")
    if dayparts is not None:
        p.mapping(dayparts, "dayparts", None)
        for raw_name, spec in dayparts.items():
            if _is_free_text(raw_name, spec):
                continue
            p.name(raw_name, "dayparts")
            name = raw_name.strip().lower()
            problem = daypart_name_problem(name)
            if problem:
                p.fail(f"dayparts.{raw_name}", problem)
            base = profile.dayparts.get(name) or DaypartProfile(name=name)
            profile.dayparts[name] = p.daypart(name, spec, base, f"dayparts.{raw_name}")
    return profile


def get_daypart(profile: ClientProfile, name: str) -> DaypartProfile:
    """Daypart by name (case-insensitive); profile values merged over DEFAULT_DAYPARTS.
    Returns a copy. Raises KeyError listing the available names."""
    available = {**DEFAULT_DAYPARTS, **{k.lower(): v for k, v in profile.dayparts.items()}}
    key = (name or "").strip().lower()
    if key not in available:
        raise KeyError(f"unknown daypart '{name}'; available: {', '.join(sorted(available))}")
    problem = daypart_name_problem(key)       # profiles built in code bypass load_profile()
    if problem:
        raise ProfileError(f"profile '{profile.client}': dayparts.{name}: {problem}")
    return copy.deepcopy(available[key])


def daypart_name_problem(name: str) -> str | None:
    """Why ``name`` cannot be used as a daypart name, or None when it is safe.

    The name becomes an output sub-folder (playlist) or a file-name suffix (mix), so it
    must be a single, portable path component: no '/', '\\', ':' or other characters
    Windows forbids, not '.'/'..', no trailing dot/space, no reserved device name."""
    if not name or not name.strip():
        return "the name is empty"
    if len(name) > DAYPART_NAME_MAX_LEN:
        return f"the name is longer than {DAYPART_NAME_MAX_LEN} characters"
    bad = sorted(set(_DAYPART_NAME_FORBIDDEN_RE.findall(name)))
    if bad:
        shown = ", ".join(repr(c) for c in bad)
        return (f"the name is used as an output folder name and must not contain {shown} "
                f"(e.g. write '10-14' or 'midday' instead of '10:00-14:00')")
    if name.strip(".") == "" or name != name.rstrip(". "):
        return "the name is used as an output folder name and must not be '.'/'..' or end with a dot or space"
    if _WINDOWS_RESERVED_NAME_RE.match(name):
        return f"'{name}' is a reserved device name on Windows and cannot be an output folder name"
    return None


def _is_free_text(name: Any, value: Any) -> bool:
    """'description: ...' / 'notes: ...' inside a name-keyed section is a comment, not an entry."""
    return name in FREE_TEXT_KEYS and (value is None or isinstance(value, str))


def _did_you_mean(name: Any, options) -> str:
    close = difflib.get_close_matches(str(name), list(options), n=1)
    return f" (did you mean '{close[0]}'?)" if close else ""


@dataclass
class _Parser:
    """Validation helpers that raise ProfileError('<file>: <key.path>: <problem>')."""
    source: str

    def fail(self, key: str, message: str) -> None:
        raise ProfileError(f"{self.source}: {key or '<top level>'}: {message}")

    def mapping(self, value: Any, key: str, allowed: tuple[str, ...] | None) -> None:
        if not isinstance(value, dict):
            self.fail(key, f"expected a mapping (key: value), got {_type(value)}")
        if allowed is None:
            return
        for k in value:
            if not isinstance(k, str):
                self.fail(key, f"key {k!r} must be text")
            if k in FREE_TEXT_KEYS:
                if value[k] is not None and not isinstance(value[k], str):
                    self.fail(_join(key, k), "must be free text")
                continue
            if k not in allowed:
                self.fail(_join(key, k), "unknown key" + _did_you_mean(k, allowed + FREE_TEXT_KEYS)
                          + f"; allowed keys: {', '.join(allowed)}")

    def name(self, value: Any, key: str) -> None:
        if not isinstance(value, str) or not value.strip():
            self.fail(key, f"names must be non-empty text, got {value!r}")

    def choice(self, value: Any, key: str, options) -> str:
        if isinstance(value, bool):
            self.fail(key, f"got the boolean {value} — YAML reads unquoted yes/no/on/off as booleans; "
                           f"use one of: {', '.join(options)}")
        if not isinstance(value, str) or value.strip().lower() not in options:
            self.fail(key, f"{value!r} is not one of: {', '.join(options)}")
        return value.strip().lower()

    def boolean(self, value: Any, key: str) -> bool:
        if not isinstance(value, bool):
            self.fail(key, f"expected true or false, got {value!r}")
        return value

    def strings(self, value: Any, key: str) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            self.fail(key, f"expected a list of text values, got {_type(value)}")
        out: list[str] = []
        for i, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, (str, int, float)) or not str(item).strip():
                self.fail(f"{key}[{i}]", f"expected non-empty text, got {item!r}")
            text = " ".join(str(item).split())
            if text not in out:
                out.append(text)
        return out

    def languages(self, value: Any, key: str) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            self.fail(key, f"expected a list of ISO 639-1 codes, got {_type(value)}")
        out: list[str] = []
        for i, item in enumerate(value):
            if isinstance(item, bool):
                self.fail(f"{key}[{i}]", f"got the boolean {item} — YAML reads unquoted no/yes/on/off as "
                                         f"booleans; quote the code (e.g. 'no' for Norwegian)")
            if not isinstance(item, str) or not re.fullmatch(r"[a-z]{2}", item.strip().lower()):
                self.fail(f"{key}[{i}]", f"expected an ISO 639-1 two-letter code such as 'en', got {item!r}")
            code = item.strip().lower()
            if code not in out:
                out.append(code)
        return out

    def list_rules(self, value: Any, key: str, *, allow_only: bool) -> ListRules:
        if value is None:
            return ListRules()
        if isinstance(value, dict) and "only" in value and not allow_only:
            self.fail(f"{key}.only", "'only' is valid in the allowlist only")
        self.mapping(value, key, LIST_RULE_KEYS + (("only",) if allow_only else ()))
        return ListRules(
            tracks=self.strings(value.get("tracks"), f"{key}.tracks"),
            artists=self.strings(value.get("artists"), f"{key}.artists"),
            tags=self.strings(value.get("tags"), f"{key}.tags"),
            only=self.boolean(value.get("only", False), f"{key}.only") if allow_only else False,
        )

    def number(self, value: Any, key: str, lo: float, hi: float, *, lo_exclusive: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            self.fail(key, f"expected a number, got {value!r}")
        if (value <= lo if lo_exclusive else value < lo) or value > hi:
            bound = f"> {lo:g} and <= {hi:g}" if lo_exclusive else f"within {lo:g}..{hi:g}"
            self.fail(key, f"{value!r} is out of range (must be {bound})")
        return float(value)

    def integer(self, value: Any, key: str, lo: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self.fail(key, f"expected a whole number, got {value!r}")
        if value < lo:
            self.fail(key, f"{value} is out of range (must be >= {lo})")
        return value

    def pair(self, value: Any, key: str, lo: float, hi: float, *, ordered: bool,
             lo_exclusive: bool = False) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            self.fail(key, f"expected two numbers [a, b], got {value!r}")
        a = self.number(value[0], f"{key}[0]", lo, hi, lo_exclusive=lo_exclusive)
        b = self.number(value[1], f"{key}[1]", lo, hi, lo_exclusive=lo_exclusive)
        if ordered and a > b:
            self.fail(key, f"lower bound {a:g} is greater than upper bound {b:g}")
        return a, b

    def daypart(self, name: str, spec: Any, base: DaypartProfile, key: str) -> DaypartProfile:
        dp = copy.deepcopy(base)
        dp.name = name
        if spec is None:          # "evening:" with everything commented out -> defaults
            return dp
        self.mapping(spec, key, DAYPART_KEYS)
        for k, v in spec.items():
            kp = f"{key}.{k}"
            if k in FREE_TEXT_KEYS:
                continue
            if k == "energy_range":
                dp.energy_range = self.pair(v, kp, 0.0, 1.0, ordered=True)
            elif k == "energy_curve":
                dp.energy_curve = None if v is None else self.pair(v, kp, 0.0, 1.0, ordered=False)
            elif k == "bpm_range":
                dp.bpm_range = None if v is None else self.pair(v, kp, 0.0, 400.0, ordered=True, lo_exclusive=True)
            elif k == "opening_tracks":
                dp.opening_tracks = self.integer(v, kp, 0)
            elif k == "opening_max_energy":
                dp.opening_max_energy = None if v is None else self.number(v, kp, 0.0, 1.0)
            elif k == "target_duration_min":
                dp.target_duration_s = None if v is None else self.number(v, kp, 0.0, 24 * 60.0,
                                                                          lo_exclusive=True) * 60.0
            elif k == "max_tracks":
                dp.max_tracks = None if v is None else self.integer(v, kp, 1)
            else:   # preferred_tags / avoid_tags / preferred_moods / preferred_genres
                setattr(dp, k, self.strings(v, kp))
        _warn_daypart_inconsistency(dp, key, self.source)
        return dp


def _warn_daypart_inconsistency(dp: DaypartProfile, key: str, source: str) -> None:
    """Merged values (profile + defaults) may disagree; warn rather than fail, since the
    curator may have changed only one of them."""
    lo, hi = dp.energy_range
    if dp.energy_curve and not all(lo <= v <= hi for v in dp.energy_curve):
        log.warning("%s: %s: energy_curve %s lies outside energy_range %s", source, key,
                    list(dp.energy_curve), list(dp.energy_range))
    if dp.opening_max_energy is not None and dp.opening_max_energy < lo:
        log.warning("%s: %s: opening_max_energy %.2f is below energy_range %s", source, key,
                    dp.opening_max_energy, list(dp.energy_range))
    clash = {normalize_tag(t) for t in dp.preferred_tags} & {normalize_tag(t) for t in dp.avoid_tags}
    if clash:
        log.warning("%s: %s: tag(s) both preferred and avoided: %s", source, key, ", ".join(sorted(clash)))


def _join(a: str, b: str) -> str:
    return f"{a}.{b}" if a else b


def _type(value: Any) -> str:
    return "nothing" if value is None else type(value).__name__


# ---------------------------------------------------------------------------
# Matching helpers (public: sequencing uses the same tag semantics)
# ---------------------------------------------------------------------------

_DASH_RE = re.compile(r"\s*[-–—]\s*")

# ``normalize_tag``, ``normalize_language`` and ``ISO639_2_TO_1`` are re-exported from
# textnorm (one implementation for curation, sequencing, analysis and QC).


def track_tags(track: TrackFeatures) -> dict[str, str]:
    """Normalized tag -> original label, from vibe_tags, genre and mood (multi-valued
    genre/mood strings such as 'Pop/Rock' also contribute their parts)."""
    tags: dict[str, str] = {}
    for tag in track.vibe_tags or []:
        if tag:
            tags.setdefault(normalize_tag(tag), tag)
    for value in (track.genre, track.mood):
        if not value:
            continue
        tags.setdefault(normalize_tag(value), value)
        for part in split_values(value):
            tags.setdefault(normalize_tag(part), part)
    tags.pop("", None)
    return tags


@dataclass(frozen=True)
class LanguageFacts:
    """A track's language tag parsed into a set (see textnorm.parse_languages).

    ``codes`` — ISO 639-1 codes in order of appearance; ``unresolved`` — tag values no
    table recognises (they count as UNKNOWN language, never as a known one)."""
    codes: tuple[str, ...]
    unresolved: tuple[str, ...]

    @classmethod
    def of(cls, value: Any) -> "LanguageFacts":
        codes, unresolved = parse_languages(value)
        return cls(codes, unresolved)

    @property
    def unknown(self) -> bool:
        """No usable language, or part of the tag could not be recognised."""
        return not self.codes or bool(self.unresolved)

    @property
    def label(self) -> str:
        return ";".join(self.codes + self.unresolved)


def track_languages(track: TrackFeatures) -> LanguageFacts:
    """Parsed language set of a track ('rus;eng' -> codes ('ru', 'en'))."""
    return LanguageFacts.of(track.language)


def _norm_title(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(_DASH_RE.sub(" - ", text.casefold()).split())


def _track_keys(track: TrackFeatures) -> set[str]:
    rel = rel_path_key(track.rel_path or track.path.name)
    name = PurePosixPath(rel).name
    stem = name.rsplit(".", 1)[0] if "." in name else name
    keys = {rel, name, _norm_title(stem)}
    if track.artist and track.title:
        keys.add(_norm_title(f"{track.artist} - {track.title}"))
    keys.discard("")
    return keys


def _artist_keys(artist: str | None) -> set[str]:
    """The full credit and every credited part ('A/B', 'A & B', 'A feat. B', ...)."""
    return set(split_artists(artist))


@dataclass
class _ListMatcher:
    tracks: dict[str, str] = field(default_factory=dict)     # normalized key -> original entry
    artists: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, rules: ListRules) -> "_ListMatcher":
        m = cls()
        for entry in rules.tracks:
            for key in (rel_path_key(entry), _norm_title(entry)):
                if key:
                    m.tracks.setdefault(key, entry)
        for entry in rules.artists:
            key = normalize_artist(entry)
            if key:
                m.artists.setdefault(key, entry)
        for entry in rules.tags:
            m.tags.setdefault(normalize_tag(entry), entry)
        return m

    def hits(self, track: TrackFeatures, tags: dict[str, str]) -> list[tuple[str, str]]:
        """(kind, matched entry) for every rule that matches the track."""
        out: list[tuple[str, str]] = []
        for key in sorted(_track_keys(track)):
            if key in self.tracks and ("track", self.tracks[key]) not in out:
                out.append(("track", self.tracks[key]))
        for key in sorted(_artist_keys(track.artist)):
            if key in self.artists and ("artist", self.artists[key]) not in out:
                out.append(("artist", self.artists[key]))
        for key in tags:
            if key in self.tags and ("tag", self.tags[key]) not in out:
                out.append(("tag", self.tags[key]))
        return out


@dataclass
class _PreferenceMatcher:
    name: str
    level: PreferenceLevel
    tags: dict[str, str]          # normalized -> as written in the profile
    uses_familiarity: bool


def preference_match_tags(profile: ClientProfile) -> dict[str, list[str]]:
    """Effective tags per preference: the explicit ``preference_tags`` entry, or the
    preference name itself (+ ``@familiarity`` for recognizable/popular-style names)."""
    out: dict[str, list[str]] = {}
    for name in profile.preferences:
        explicit = profile.preference_tags.get(name)
        if explicit is not None:
            out[name] = list(explicit)
        else:
            out[name] = [name] + ([FAMILIARITY_TOKEN] if FAMILIARITY_NAME_RE.search(name) else [])
    return out


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------

@dataclass
class _Rules:
    profile: ClientProfile
    block: _ListMatcher
    allow: _ListMatcher
    allowed_langs: list[str]
    excluded_langs: list[str]
    preferences: list[_PreferenceMatcher]

    @classmethod
    def build(cls, profile: ClientProfile) -> "_Rules":
        # profiles built in code bypass load_profile(): a misspelt policy must not act as "allow"
        for key, value, options in (("language.unknown_policy", profile.language.unknown_policy, UNKNOWN_POLICIES),
                                    ("language.cyrillic_script_policy", profile.language.cyrillic_script_policy,
                                     CYRILLIC_POLICIES)):
            if value not in options:
                raise ProfileError(f"profile '{profile.client}': {key}: {value!r} is not one of: {', '.join(options)}")
        prefs = []
        for name, tags in preference_match_tags(profile).items():
            try:
                level = PreferenceLevel(profile.preferences[name])
            except ValueError:
                raise ProfileError(f"profile '{profile.client}': preferences.{name}: "
                                   f"{profile.preferences[name]!r} is not a preference level") from None
            uses_fam = any(t.strip().lower() == FAMILIARITY_TOKEN for t in tags)
            norm = {normalize_tag(t): t for t in tags if t.strip().lower() != FAMILIARITY_TOKEN}
            prefs.append(_PreferenceMatcher(name, level, norm, uses_fam))
        return cls(
            profile=profile,
            block=_ListMatcher.build(profile.blocklist),
            allow=_ListMatcher.build(profile.allowlist),
            allowed_langs=_profile_codes(profile.language.allowed),
            excluded_langs=_profile_codes(profile.language.excluded),
            preferences=prefs,
        )


def _profile_codes(values: list[str]) -> list[str]:
    """ISO 639-1 codes of a profile language list (hand-built profiles may say 'rus')."""
    out: list[str] = []
    for value in values:
        for code in parse_languages(value)[0]:
            if code not in out:
                out.append(code)
    return out


def curate(tracks: list[TrackFeatures], profile: ClientProfile) -> CurationResult:
    """Apply the client profile: one CurationDecision per input track, in input order.

    Inputs are never mutated; each decision carries a copy of the track whose
    ``exclusion_reasons`` / ``review_reasons`` (and ``requires_review``) are filled in.
    """
    rules = _Rules.build(profile)
    decisions = [_decide(t, rules) for t in tracks]
    eligible = [d.track for d in decisions if d.eligible]
    excluded = [d for d in decisions if not d.eligible]
    review = sum(1 for d in decisions if d.eligible and (d.review_reasons or d.track.requires_review))
    log.info("curation (%s): %d eligible (%d flagged for review), %d excluded",
             profile.client, len(eligible), review, len(excluded))
    return CurationResult(eligible=eligible, excluded=excluded, decisions=decisions)


def editorial_scores(result: CurationResult) -> dict[str, float]:
    """``{track_id: editorial_score}`` for the eligible tracks (input for sequencing)."""
    return {d.track.track_id: d.editorial_score for d in result.decisions if d.eligible}


def _decide(track: TrackFeatures, rules: _Rules) -> CurationDecision:
    profile = rules.profile
    t = copy.deepcopy(track)
    tags = track_tags(t)
    allow_hits = rules.allow.hits(t, tags)
    allowlisted = bool(allow_hits)
    excl: list[str] = []
    review: list[str] = []

    # 1. hard rules — the allowlist cannot override them
    for kind, entry in rules.block.hits(t, tags):
        excl.append(f"blocklisted {kind} '{entry}'"
                    + (" (also allowlisted; the blocklist takes precedence)" if allowlisted else ""))
    if t.excluded:
        who = "curator metadata" if t.feature_sources.get("excluded") == "manual" else "track metadata"
        excl.append(f"excluded in {who} (excluded=true)" + (f" — note: {t.notes}" if t.notes else ""))
    lang = track_languages(t)
    banned = [c for c in lang.codes if c in rules.excluded_langs]   # ANY excluded language is enough
    if banned:
        excl.append(f"language '{', '.join(banned)}' is excluded by the client profile"
                    + (f" (track language '{lang.label}')" if len(lang.codes) > 1 or lang.unresolved else "")
                    + (" (an allowlist cannot override an excluded language)" if allowlisted else ""))

    # 2. allowlist.only
    if not excl and profile.allowlist.only and not allowlisted:
        excl.append("not on the client allowlist (allowlist.only is set)")

    # 3. soft rules — an allowlisted track is not excluded by them, but unverified facts
    #    (unknown language, Cyrillic hint, not approved) stay on it as review flags
    if not excl:
        overridden: list[str] = []
        for item in _soft_rules(t, lang, tags, rules):
            if not item.exclude:
                review.append(item.reason)
            elif not allowlisted:
                excl.append(item.reason)
            elif item.review_if_allowlisted:
                review.append(item.review_if_allowlisted)
            else:
                overridden.append(item.reason)
        if overridden:
            log.debug("%s: allowlisted (%s); overrides: %s", t.rel_path,
                      ", ".join(f"{k} '{e}'" for k, e in allow_hits), "; ".join(overridden))

    # the analysis hint is superseded by curation's own Cyrillic verdict (see _soft_rules)
    t.review_reasons = [r for r in t.review_reasons if r != ANALYSIS_CYRILLIC_HINT_REASON]

    # requires_review is never overridden: the track stays eligible, with a reason
    if t.requires_review:
        if t.feature_sources.get("requires_review") == "manual":
            review.append("flagged for review in curator metadata"
                          + (f" — note: {t.notes}" if t.notes else ""))
        elif not t.review_reasons:
            review.append("flagged for review (requires_review is set)")

    eligible = not excl
    if not eligible:
        review = []              # moot for an excluded track; keep reports unambiguous
    review = _unique(review)
    _extend_unique(t.exclusion_reasons, excl)
    _extend_unique(t.review_reasons, review)
    if eligible and (review or (t.review_reasons and not _review_signed_off(t))):
        # Any open reason — curation's own or one left by analysis (Cyrillic hint, truncated
        # file, key/tempo not found ...) — must reach the playlist flag and QC.
        t.requires_review = True
    return CurationDecision(track=t, eligible=eligible, exclusion_reasons=excl, review_reasons=review,
                            editorial_score=_editorial_score(t, tags, rules), allowlisted=allowlisted)


def _review_signed_off(t: TrackFeatures) -> bool:
    """The curator explicitly set ``requires_review: false`` for this track — a sign-off
    for the reasons analysis left on it (curation's own reasons still set the flag)."""
    return t.feature_sources.get("requires_review") == "manual" and not t.requires_review


@dataclass(frozen=True)
class _SoftFinding:
    """One soft-rule result, in rule order."""
    reason: str
    exclude: bool                              # False: a review flag
    review_if_allowlisted: str | None = None   # what an exclusion becomes for an allowlisted
                                               # track (None: a client policy it overrides)


def _soft_rules(t: TrackFeatures, lang: LanguageFacts, tags: dict[str, str],
                rules: _Rules) -> list[_SoftFinding]:
    """Soft rules in precedence order. Exclusions of UNVERIFIED facts (unknown language,
    Cyrillic hint, not approved) carry the review flag they become for an allowlisted
    track; client policies the allowlist exists to override (language not in the allowed
    list, ``avoid`` preferences) carry None."""
    profile = rules.profile
    out: list[_SoftFinding] = []

    if lang.codes and rules.allowed_langs:
        outside = [c for c in lang.codes if c not in rules.allowed_langs]   # ALL must be allowed
        if outside:
            allowed = ", ".join(rules.allowed_langs)
            if len(lang.codes) == 1 and not lang.unresolved:
                reason = f"language '{outside[0]}' is not in the allowed list ({allowed})"
            else:
                reason = (f"language '{', '.join(outside)}' (track language '{lang.label}') is not in the "
                          f"allowed list ({allowed}); every language of a multi-language track must be allowed")
            out.append(_SoftFinding(reason, exclude=True))

    cyrillic = (t.script_hint or "").lower() == "cyrillic"
    cyr_policy = profile.language.cyrillic_script_policy
    if lang.unknown:              # no language, or a tag value nobody recognises
        detail = ""
        if lang.unresolved:
            detail = (f"; unrecognised language tag value(s) {', '.join(repr(u) for u in lang.unresolved)}"
                      + (f" in '{lang.label}'" if lang.codes else ""))
        policy = profile.language.unknown_policy
        if policy == "exclude":
            out.append(_SoftFinding(UNKNOWN_LANGUAGE_EXCLUDE_REASON + detail, exclude=True,
                                    review_if_allowlisted=UNKNOWN_LANGUAGE_REVIEW_REASON + detail))
        elif policy == "review":
            out.append(_SoftFinding(UNKNOWN_LANGUAGE_REVIEW_REASON + detail, exclude=False))
        if cyrillic:
            if cyr_policy == "exclude":
                out.append(_SoftFinding(CYRILLIC_EXCLUDE_REASON, exclude=True,
                                        review_if_allowlisted=CYRILLIC_REVIEW_REASON))
            elif cyr_policy == "review":
                out.append(_SoftFinding(CYRILLIC_REVIEW_REASON, exclude=False))
    elif (cyrillic and cyr_policy != "ignore" and t.feature_sources.get("language") != "manual"
          and not any(c in CYRILLIC_SCRIPT_LANGUAGES for c in lang.codes)):
        # e.g. 'Кино - Группа крови' tagged TLAN 'eng' by a tagger: the tag is not proof
        out.append(_SoftFinding(CYRILLIC_TAG_CONFLICT_REASON.format(lang=lang.label), exclude=False))

    for pref in rules.preferences:
        if pref.level is PreferenceLevel.AVOID:
            matched = [key for key in pref.tags if key in tags]
            if matched:
                out.append(_SoftFinding(f"matches avoided preference '{pref.name}' (tag '{tags[matched[0]]}')",
                                        exclude=True))

    if not t.approved:
        if profile.require_approved:
            out.append(_SoftFinding(NOT_APPROVED_EXCLUDE_REASON, exclude=True,
                                    review_if_allowlisted=NOT_APPROVED_REVIEW_REASON))
        elif profile.manual_review_required:
            out.append(_SoftFinding(NOT_APPROVED_REVIEW_REASON, exclude=False))
    return out


def _editorial_score(t: TrackFeatures, tags: dict[str, str], rules: _Rules) -> float:
    """Mean preference weight (−1..+1) over the preferences that apply to the track.

    * a preference matched by a curator tag counts its full weight;
    * a familiarity-based preference (``@familiarity``) without a tag match ALWAYS takes
      part in the mean: curated familiarity f counts weight × (2f − 1) (f=1 → full
      weight, f=0.5 → neutral, f=0 → opposite) and UNKNOWN familiarity counts 0. The
      score is therefore monotonic in familiarity — curated-familiar > unknown >
      curated-obscure — and an unrated track never outranks a curated hit;
    * other preferences the track does not match do not take part.
    0.0 when nothing applies."""
    familiarity = t.familiarity
    if familiarity is not None and (t.feature_sources.get("familiarity", "").startswith("auto")
                                    or not math.isfinite(familiarity)):
        familiarity = None        # never estimated by design; ignore anything that claims to be
    contributions: list[float] = []
    for pref in rules.preferences:
        weight = PREFERENCE_WEIGHTS[pref.level]
        if any(key in tags for key in pref.tags):
            contributions.append(weight)
        elif pref.uses_familiarity:
            f = None if familiarity is None else min(max(familiarity, 0.0), 1.0)
            contributions.append(0.0 if f is None else weight * (2.0 * f - 1.0))
    if not contributions:
        return 0.0
    return round(min(1.0, max(-1.0, sum(contributions) / len(contributions))), 4)


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    _extend_unique(out, items)
    return out


def _extend_unique(target: list[str], items: list[str]) -> None:
    for item in items:
        if item not in target:
            target.append(item)
