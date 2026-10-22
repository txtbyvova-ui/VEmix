"""Deciding whether a catalogue recording IS the local track (spec §3: "оценивай
вероятность правильного совпадения", "при неоднозначном совпадении не применяй").

Catalogue search scores are not match probabilities: the MusicBrainz ``score`` (0..100)
ranks results WITHIN one query, so for "Michael Jackson — Thriller" the original album
recording, a live recording and a "360 Reality Audio mix" all score 100. This module
re-ranks candidates with its own explicit, explainable score:

* title similarity — on the "core" title: bracketed qualifiers and version tails removed
  (``scenario._relaxed_title``), diacritics folded, '&' == 'and';
* artist similarity — full credit, primary artist (featured guests dropped), any credited
  part (``textnorm.split_artists``), leading "The" ignored;
* duration — within the tolerance scores ~1, beyond it falls to 0 at 3× the tolerance;
* version penalties — markers such as live, remix, instrumental, demo, karaoke, edit,
  mix, 360 / Dolby Atmos found in the candidate's title or disambiguation (or in the
  secondary types of EVERY release group it appears on) but not in the local title;
  half the penalty in the opposite direction (a local "(Live)" vs a studio recording);
  music videos are penalised too;
* small bonuses — an official release, the file's ISRC listed on it;
* release hints from the file's tags (album, year) — tie-breakers among plausible
  candidates (base score ≥ ``plausible_score``): the local album among the candidate's
  release titles (``album_bonus``; a small ``album_penalty`` when the album is known and on
  none of a complete release list), a release dated in the local year (``year_bonus``).
  Only when the file names neither album nor year does the earliest first release among
  the plausible candidates get ``earliest_bonus`` (the original rather than a re-recording
  — a prior, replaced by the file's own evidence when there is some);
* identity reliability — artist/title parsed from the FILE NAME (not tags) scale the
  score down (``filename_identity_factor``).

Every bonus / penalty that contributed is listed in the candidate's ``evidence``
(component -> signed adjustment) and its ``reasons``.

A match is applied automatically only when ALL gates pass: score ≥ ``accept_score`` —
also WITHOUT the album/year bonus (album and year break ties, they never turn a weak
artist/title/duration match into a match) — margin over the runner-up ≥ ``min_margin``,
both durations known and within ``duration_tolerance_s``, title (and, when known,
artist) similar enough. Anything else is 'ambiguous' (plausible candidates kept for the
curator, nothing applied) or 'not_found'. The weights are editorial defaults, not a
trained model: the score is a ranking device with an honest name, not a calibrated
probability.
"""

from __future__ import annotations

import difflib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..scenario import _relaxed_title
from ..textnorm import primary_artist, split_artists
from .models import STATUS_AMBIGUOUS, STATUS_MATCHED, STATUS_NOT_FOUND


@dataclass(frozen=True)
class MatchConfig:
    accept_score: float = 0.85          # minimum (identity-adjusted) score to apply a match
    min_margin: float = 0.08            # minimum lead over the runner-up
    duration_tolerance_s: float = 7.0   # |local − catalogue| allowed for an automatic match
    min_candidate_score: float = 0.40   # below: not even a candidate (not_found)
    min_title_similarity: float = 0.85
    min_artist_similarity: float = 0.80
    filename_identity_factor: float = 0.92   # artist/title parsed from the file name
    max_candidates: int = 5             # candidates kept in the record
    w_title: float = 0.45
    w_artist: float = 0.35
    w_duration: float = 0.20
    official_bonus: float = 0.03
    earliest_bonus: float = 0.04        # only when the file names neither album nor year
    isrc_bonus: float = 0.10
    album_bonus: float = 0.05           # the local album is one of the candidate's releases
    album_penalty: float = 0.03         # local album known, on none of a complete release list
    year_bonus: float = 0.04            # a release of the candidate dates from the local year
    min_album_similarity: float = 0.85
    max_complete_releases: int = 24     # longer lists may be truncated (lookups return ≤ 25)
    plausible_score: float = 0.80       # base score from which the tie-break bonuses apply
    video_penalty: float = 0.20
    max_penalty: float = 0.60
    id_confidence: float = 0.99         # identity taken from an MBID tag in the file


DEFAULT_MATCH_CONFIG = MatchConfig()

# (marker, penalty, pattern). Order matters only for readability; 'mix' does not match
# inside 'remix' (word boundaries).
VERSION_MARKERS: tuple[tuple[str, float, str], ...] = (
    ("live", 0.30, r"\blive\b|\bin concert\b|\bunplugged\b"),
    ("karaoke", 0.40, r"\bkaraoke\b|\bbacking track\b"),
    ("instrumental", 0.30, r"\binstrumental\b"),
    ("a cappella", 0.30, r"\ba ?cappella\b|\bacapella\b"),
    ("remix", 0.25, r"\bremix(?:es|ed)?\b|\brmx\b|\brework\b|\bbootleg\b"),
    ("demo", 0.25, r"\bdemo\b"),
    ("360", 0.20, r"\b360\b|\bdolby atmos\b|\bspatial audio\b|\bbinaural\b"),
    ("mix", 0.15, r"\bmix\b|\bmixed\b|\bdub\b"),
    ("acoustic", 0.15, r"\bacoustic\b"),
    ("edit", 0.10, r"\bedit\b"),
    ("speed", 0.30, r"\bsped up\b|\bslowed\b|\bnightcore\b"),
)
_MARKER_RES = tuple((name, pen, re.compile(pat, re.IGNORECASE)) for name, pen, pat in VERSION_MARKERS)
MARKER_PENALTY = {name: pen for name, pen, _ in VERSION_MARKERS}
# phrases that name the ORIGINAL version and must not count as markers
_NEUTRAL_RE = re.compile(r"\b(?:original|album|single|lp)\s+(?:mix|version|edit)\b|\bstudio version\b"
                         r"|\bremaster(?:ed)?\b|\bmono\b|\bstereo\b", re.IGNORECASE)
# MusicBrainz release-group secondary type -> marker (applied when EVERY release has it)
SECONDARY_TYPE_MARKERS = {"live": "live", "remix": "remix", "dj-mix": "mix", "demo": "demo"}

_DASH_TAIL_RE = re.compile(r"\s+[-–—]\s+(.+)$")
_NON_WORD_RE = re.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# normalisation and similarities
# ---------------------------------------------------------------------------

def fold(text: Any) -> str:
    """Comparison form: Unicode-compatible, diacritics removed, case-folded, '&' = 'and',
    punctuation dropped, whitespace collapsed ('Beyoncé' == 'beyonce')."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = s.replace("&", " and ")
    s = _NON_WORD_RE.sub(" ", s.replace("'", "").replace("’", ""))
    return " ".join(s.split())


def marker_counts(text: Any) -> Counter[str]:
    """Occurrences of version markers in a title / disambiguation."""
    out: Counter[str] = Counter()
    if not text:
        return out
    s = _NEUTRAL_RE.sub(" ", str(text))
    for name, _pen, rx in _MARKER_RES:
        n = len(rx.findall(s))
        if n:
            out[name] += n
    return out


def core_title(text: Any) -> str:
    """Folded title without version qualifiers: brackets, '- 2008 Remaster' tails and
    ' - Live' / ' - Radio Edit' style tails (version markers are scored separately)."""
    if not text:
        return ""
    s = str(text)
    m = _DASH_TAIL_RE.search(s)
    if m and (marker_counts(m.group(1)) or _NEUTRAL_RE.search(m.group(1))):
        s = s[:m.start()]
    return fold(_relaxed_title(s)) or fold(s)


def artist_key(text: Any) -> str:
    s = fold(text)
    return s[4:] if s.startswith("the ") and len(s) > 4 else s


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def title_similarity(local: Any, remote: Any) -> float:
    """1.0 for the same title (after folding), 0.97 for the same core title (version
    qualifiers differ — scored by the marker penalties), else a string ratio ≤ 0.9."""
    if not local or not remote:
        return 0.0
    if fold(local) == fold(remote):
        return 1.0
    a, b = core_title(local), core_title(remote)
    if a and a == b:
        return 0.97
    return 0.9 * _ratio(a, b)


def artist_similarity(local: Any, credit: Any, names: Iterable[str] = ()) -> float | None:
    """Similarity of the local artist to a catalogue artist credit (None when the local
    artist or the credit is unknown).

    ``names`` are the individually credited artists (MusicBrainz ``artist-credit``). The
    catalogue credit is never split by punctuation — 'Earth, Wind & Fire' is one artist —
    so partial matches use only these structured names: 1.0 same credit; 0.95 same main
    artist (guests differ); 0.9 every artist of a local multi-artist credit is credited;
    0.85 the local main artist is one of the credited artists; else a string ratio."""
    la = artist_key(local)
    names = [n for n in names if n]
    if not la or not (credit or names):
        return None
    full = artist_key(credit) if credit else artist_key(" & ".join(names))
    if la == full:
        return 1.0
    remote = {artist_key(n) for n in names} if names else {full}
    remote.discard("")
    lp = artist_key(primary_artist(local))
    rp = artist_key(names[0]) if names else artist_key(primary_artist(credit))
    if lp and lp == rp:
        return 0.95
    parts = {artist_key(p) for p in split_artists(local)[1:]} - {"", la}
    if parts and parts <= remote:
        return 0.9
    if lp in remote:
        return 0.85
    return 0.9 * max(_ratio(la, full), _ratio(lp, rp))


def _duration(value: Any) -> float | None:
    """A positive finite duration, else None (NaN would pass every comparison gate)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        f = float(value)
    except OverflowError:            # an integer too large for a float
        return None
    return f if math.isfinite(f) and f > 0 else None


def duration_similarity(local_s: float | None, remote_s: float | None, tolerance_s: float) -> float | None:
    """1.0 within 1 s, falling linearly to 0 at 1 s + 3× the tolerance; None when unknown
    (missing, non-positive or non-finite)."""
    local_s, remote_s = _duration(local_s), _duration(remote_s)
    if local_s is None or remote_s is None:
        return None
    diff = abs(local_s - remote_s)
    return max(0.0, 1.0 - max(0.0, diff - 1.0) / (3.0 * max(tolerance_s, 0.5)))


def _year(date: Any) -> int | None:
    # isascii(): '²'.isdigit() is True but int('²') raises
    if isinstance(date, str) and len(date) >= 4 and date[:4].isascii() and date[:4].isdigit():
        y = int(date[:4])
        return y if y > 0 else None
    return None


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

@dataclass
class LocalIdentity:
    """What we know about the local track (tags / file name / probe). ``album`` and
    ``year`` come from the file's tags and only re-rank candidates (tie-breakers)."""
    artist: str | None
    title: str | None
    duration_s: float | None
    isrc: str | None = None
    identity_factor: float = 1.0      # < 1 when artist/title come from the file name
    identity_note: str | None = None
    album: str | None = None
    year: int | None = None

    @property
    def has_release_hints(self) -> bool:
        return bool(self.album) or self.year is not None


@dataclass
class ScoredCandidate:
    source: str
    id: str
    url: str
    title: str | None
    artist: str | None
    date: str | None
    length_s: float | None
    disambiguation: str | None
    relevance: float | None            # catalogue's own ranking value (not a probability)
    raw_score: float                   # unclamped, before the identity factor (ranking)
    score: float                       # 0..1 after the identity factor (reported / gated)
    components: dict[str, float | None] = field(default_factory=dict)
    duration_diff_s: float | None = None
    reasons: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict, repr=False)   # the compact recording
    # bonus / penalty components that contributed: name -> signed adjustment
    # ('official', 'earliest_release', 'isrc', 'album', 'year')
    evidence: dict[str, float] = field(default_factory=dict)
    # 0..1 score WITHOUT the album/year bonus (after the identity factor): what artist, title,
    # duration and the other evidence alone support — the accept gate uses it. None = ``score``.
    identity_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "id": self.id, "url": self.url, "title": self.title,
            "artist": self.artist, "date": self.date,
            "length_s": None if self.length_s is None else round(self.length_s, 1),
            "disambiguation": self.disambiguation or None,
            "score": round(self.score, 4), "relevance": self.relevance,
            "evidence": {k: round(v, 4) for k, v in self.evidence.items()},
            "reasons": list(self.reasons),
        }


@dataclass
class MatchDecision:
    status: str                         # 'matched' | 'ambiguous' | 'not_found'
    chosen: ScoredCandidate | None
    confidence: float | None
    candidates: list[ScoredCandidate]
    reasons: list[str] = field(default_factory=list)


def score_recordings(local: LocalIdentity, recordings: Iterable[dict[str, Any]], *,
                     source: str = "musicbrainz", url_for=None,
                     cfg: MatchConfig = DEFAULT_MATCH_CONFIG) -> list[ScoredCandidate]:
    """Score compact recordings (``musicbrainz.compact_recording`` shape) against the
    local identity; best first (ties: earlier first release, then id — deterministic)."""
    recs = [r for r in recordings if isinstance(r, dict) and r.get("id")]
    local_markers = marker_counts(local.title)
    prelim: list[tuple[dict, dict[str, float | None], float, float | None]] = []
    for rec in recs:
        length_ms = _duration(rec.get("length_ms"))
        length_s = length_ms / 1000.0 if length_ms is not None else None
        names = [c.get("name") for c in rec.get("artist_credit") or [] if isinstance(c, dict)]
        comps = {
            "title": title_similarity(local.title, rec.get("title")) if local.title else None,
            "artist": artist_similarity(local.artist, rec.get("artist_phrase"), names),
            "duration": duration_similarity(local.duration_s, length_s, cfg.duration_tolerance_s),
        }
        weights = {"title": cfg.w_title, "artist": cfg.w_artist, "duration": cfg.w_duration}
        known = {k: v for k, v in comps.items() if v is not None}
        wsum = sum(weights[k] for k in known)
        base = sum(weights[k] * v for k, v in known.items()) / wsum if wsum > 0 else 0.0
        prelim.append((rec, comps, base, length_s))

    # the earliest first release among plausible candidates gets a small bonus — a prior used
    # only when the file's tags name neither album nor year (otherwise those decide)
    earliest = None
    if not local.has_release_hints:
        plausible_years = [_year(rec.get("first_release_date")) for rec, _c, base, _l in prelim
                           if base >= cfg.plausible_score]
        plausible_years = [y for y in plausible_years if y is not None]
        earliest = min(plausible_years) if plausible_years else None

    out: list[ScoredCandidate] = []
    for rec, comps, base, length_s in prelim:
        reasons: list[str] = []
        for k in ("title", "artist", "duration"):
            if comps[k] is None:
                reasons.append(f"{k}: unknown")
            else:
                reasons.append(f"{k} {comps[k]:.2f}")
        local_d, remote_d = _duration(local.duration_s), _duration(length_s)
        diff = abs(local_d - remote_d) if (local_d is not None and remote_d is not None) else None
        if diff is not None:
            reasons.append(f"duration difference {diff:.1f} s")
        penalty, pen_reasons = _version_penalty(local_markers, rec, cfg)
        reasons += pen_reasons
        if rec.get("video"):
            penalty += cfg.video_penalty
            reasons.append(f"music video recording -{cfg.video_penalty:.2f}")
        hint_bonus, hint_penalty, hint_comps, hint_evidence, hint_reasons = _release_hints(local, rec, base, cfg)
        comps.update(hint_comps)
        penalty = min(penalty + hint_penalty, cfg.max_penalty)
        bonus = 0.0
        evidence: dict[str, float] = {}
        if any(isinstance(r, dict) and str(r.get("status") or "").casefold() == "official"
               for r in rec.get("releases") or []):
            bonus += cfg.official_bonus
            evidence["official"] = cfg.official_bonus
            reasons.append(f"on an official release +{cfg.official_bonus:.2f}")
        year = _year(rec.get("first_release_date"))
        if earliest is not None and year == earliest and base >= cfg.plausible_score:
            bonus += cfg.earliest_bonus
            evidence["earliest_release"] = cfg.earliest_bonus
            reasons.append(f"earliest first release ({year}) +{cfg.earliest_bonus:.2f}")
        if local.isrc and local.isrc in (rec.get("isrcs") or []):
            bonus += cfg.isrc_bonus
            evidence["isrc"] = cfg.isrc_bonus
            reasons.append(f"file ISRC {local.isrc} listed +{cfg.isrc_bonus:.2f}")
        evidence.update(hint_evidence)
        reasons += hint_reasons
        raw = base + bonus + hint_bonus - penalty
        score = min(1.0, max(0.0, raw)) * local.identity_factor
        identity = min(1.0, max(0.0, raw - hint_bonus)) * local.identity_factor
        if local.identity_factor < 1.0:
            reasons.append(f"{local.identity_note or 'identity less reliable'} x{local.identity_factor:.2f}")
        rel = rec.get("relevance")
        if isinstance(rel, (int, float)) and not isinstance(rel, bool):
            reasons.append(f"{source} relevance {rel:g} (ranking within the query, not a probability)")
        out.append(ScoredCandidate(
            source=source, id=str(rec["id"]), url=url_for(rec["id"]) if url_for else "",
            title=rec.get("title"), artist=rec.get("artist_phrase"), date=rec.get("first_release_date") or None,
            length_s=length_s, disambiguation=rec.get("disambiguation") or None,
            relevance=rel if isinstance(rel, (int, float)) and not isinstance(rel, bool) else None,
            raw_score=raw, score=score, components=comps, duration_diff_s=diff, reasons=reasons, data=rec,
            evidence=evidence, identity_score=identity))
    out.sort(key=lambda c: (-c.raw_score, _year(c.date) or 9999, c.id))
    return out


def _release_hints(local: LocalIdentity, rec: dict[str, Any], base: float, cfg: MatchConfig
                   ) -> tuple[float, float, dict[str, float | None], dict[str, float], list[str]]:
    """Album / year of the file's tags against the candidate's releases: (bonus, penalty,
    components, evidence, reasons). Bonuses only for plausible candidates (tie-breakers);
    the album penalty only when the release list is complete (lookups and searches list at
    most ~25 releases — a missing album on a truncated list is no evidence)."""
    bonus = penalty = 0.0
    comps: dict[str, float | None] = {}
    evidence: dict[str, float] = {}
    reasons: list[str] = []
    plausible = base >= cfg.plausible_score
    releases = [r for r in rec.get("releases") or [] if isinstance(r, dict)]
    if local.album:
        titles: list[str] = []
        for r in releases:
            for t in (r.get("title"), r.get("rg_title")):
                if isinstance(t, str) and t and t not in titles:
                    titles.append(t)
        if titles:
            sim, best_title = max(((title_similarity(local.album, t), t) for t in titles), key=lambda x: x[0])
            comps["album"] = sim
            if sim >= cfg.min_album_similarity:
                if plausible:
                    bonus += cfg.album_bonus
                    evidence["album"] = cfg.album_bonus
                reasons.append(f"album '{local.album}' matches release '{best_title}'"
                               + (f" +{cfg.album_bonus:.2f}" if plausible else ""))
            elif len(releases) <= cfg.max_complete_releases:
                penalty += cfg.album_penalty
                evidence["album"] = -cfg.album_penalty
                reasons.append(f"album '{local.album}' is on none of its {len(releases)} release(s) "
                               f"-{cfg.album_penalty:.2f}")
            else:
                reasons.append(f"album '{local.album}' not among the {len(releases)} listed releases "
                               f"(list may be incomplete): not scored")
        else:
            comps["album"] = None
    if local.year is not None:
        years = {y for y in (_year(r.get("date")) for r in releases) if y is not None}
        first = _year(rec.get("first_release_date"))
        if first is not None:
            years.add(first)
        if years:
            comps["year"] = 1.0 if local.year in years else 0.0
            if local.year in years:
                if plausible:
                    bonus += cfg.year_bonus
                    evidence["year"] = cfg.year_bonus
                reasons.append(f"year {local.year}: a release dates from it"
                               + (f" +{cfg.year_bonus:.2f}" if plausible else ""))
            else:
                reasons.append(f"year {local.year}: no listed release from that year (first {min(years)})")
        else:
            comps["year"] = None
    return bonus, penalty, comps, evidence, reasons


def _version_penalty(local_markers: Counter[str], rec: dict[str, Any], cfg: MatchConfig) -> tuple[float, list[str]]:
    cand = marker_counts(rec.get("title")) + marker_counts(rec.get("disambiguation"))
    rg_markers = _release_group_markers(rec.get("releases") or [])
    reasons: list[str] = []
    penalty = 0.0
    extra = {m for m, n in cand.items() if n > local_markers.get(m, 0)}
    extra |= {m for m in rg_markers if local_markers.get(m, 0) == 0}
    for m in sorted(extra):
        penalty += MARKER_PENALTY[m]
        where = "release groups" if m in rg_markers and m not in cand else "title/disambiguation"
        reasons.append(f"version '{m}' ({where}) not in the local title -{MARKER_PENALTY[m]:.2f}")
    missing = {m for m, n in local_markers.items() if n > cand.get(m, 0) and m not in rg_markers}
    for m in sorted(missing):
        half = MARKER_PENALTY[m] / 2
        penalty += half
        reasons.append(f"local title says '{m}', candidate does not -{half:.2f}")
    return penalty, reasons


def _release_group_markers(releases: list[Any]) -> set[str]:
    """Markers implied by release-group secondary types shared by EVERY release."""
    rels = [r for r in releases if isinstance(r, dict)]
    if not rels:
        return set()
    common: set[str] | None = None
    for r in rels:
        types = {str(t).casefold() for t in r.get("secondary_types") or [] if t}
        markers = {SECONDARY_TYPE_MARKERS[t] for t in types if t in SECONDARY_TYPE_MARKERS}
        common = markers if common is None else common & markers
        if not common:
            return set()
    return common or set()


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------

def decide(local: LocalIdentity, scored: list[ScoredCandidate], *,
           cfg: MatchConfig = DEFAULT_MATCH_CONFIG) -> MatchDecision:
    """Apply the gates (see the module docstring) to ranked candidates."""
    kept = [c for c in scored if c.score >= cfg.min_candidate_score][:cfg.max_candidates]
    if not kept:
        reasons = ["no catalogue result" if not scored else
                   f"best candidate scores {scored[0].score:.2f} < {cfg.min_candidate_score:.2f}"]
        return MatchDecision(STATUS_NOT_FOUND, None, None, scored[:cfg.max_candidates], reasons)
    best = kept[0]
    runner = scored[1] if len(scored) > 1 else None
    # the margin uses the UNCLAMPED scores: two candidates above 1.0 (bonuses) would
    # otherwise look tied; scaled by the identity factor like the scores themselves
    margin = (best.raw_score - (runner.raw_score if runner else 0.0)) * local.identity_factor
    failed: list[str] = []
    identity = best.score if best.identity_score is None else best.identity_score
    if best.score < cfg.accept_score:
        failed.append(f"best score {best.score:.2f} < {cfg.accept_score:.2f}")
    elif identity < cfg.accept_score:
        failed.append(f"best score without the album/year bonus {identity:.2f} < {cfg.accept_score:.2f} "
                      f"(album and year only break ties, they cannot confirm a match)")
    if margin < cfg.min_margin:
        failed.append(f"margin over the runner-up {margin:.2f} < {cfg.min_margin:.2f} "
                      f"(candidates too close to choose)")
    if best.duration_diff_s is None:
        failed.append("duration unknown on one side: identity cannot be verified")
    elif best.duration_diff_s > cfg.duration_tolerance_s:
        failed.append(f"duration differs by {best.duration_diff_s:.1f} s (> {cfg.duration_tolerance_s:g} s)")
    t = best.components.get("title")
    if t is not None and t < cfg.min_title_similarity:
        failed.append(f"title similarity {t:.2f} < {cfg.min_title_similarity:.2f}")
    a = best.components.get("artist")
    if a is not None and a < cfg.min_artist_similarity:
        failed.append(f"artist similarity {a:.2f} < {cfg.min_artist_similarity:.2f}")
    if failed:
        return MatchDecision(STATUS_AMBIGUOUS, None, best.score, kept, failed)
    reasons = [f"best {best.score:.2f}"
               + (f", runner-up {runner.score:.2f} (margin {margin:.2f})" if runner else ", no runner-up")]
    hints = [f"{k} {best.evidence[k]:+.2f}" for k in ("album", "year") if k in best.evidence]
    if hints:
        reasons.append("album/year evidence (tie-breaker): " + ", ".join(hints))
    return MatchDecision(STATUS_MATCHED, best, best.score, kept, reasons)


def decide_identified(local: LocalIdentity, candidate: ScoredCandidate, *, evidence: str,
                      cfg: MatchConfig = DEFAULT_MATCH_CONFIG) -> MatchDecision:
    """A recording named by an identifier in the file's own tags (MBID). It is trusted
    unless the file contradicts it: a duration beyond the tolerance or a clearly
    different title/artist make it 'ambiguous' for the curator."""
    failed: list[str] = []
    diff = candidate.duration_diff_s
    if diff is not None and diff > cfg.duration_tolerance_s:
        failed.append(f"{evidence} points to a recording whose duration differs by {diff:.1f} s")
    t, a = candidate.components.get("title"), candidate.components.get("artist")
    if t is not None and t < 0.6:
        failed.append(f"{evidence} points to a recording with a different title ({t:.2f})")
    if a is not None and a < 0.6:
        failed.append(f"{evidence} points to a recording by a different artist ({a:.2f})")
    if failed:
        return MatchDecision(STATUS_AMBIGUOUS, None, candidate.score, [candidate], failed)
    candidate.score = candidate.identity_score = cfg.id_confidence
    candidate.reasons.append(f"identified by {evidence} in the file's tags")
    return MatchDecision(STATUS_MATCHED, candidate, cfg.id_confidence, [candidate],
                         [f"identified by {evidence} in the file's tags"])


__all__ = ["MatchConfig", "DEFAULT_MATCH_CONFIG", "VERSION_MARKERS", "LocalIdentity", "ScoredCandidate",
           "MatchDecision", "fold", "core_title", "artist_key", "marker_counts", "title_similarity",
           "artist_similarity", "duration_similarity", "score_recordings", "decide", "decide_identified"]
