"""Shared text normalization: tags, artists, languages and library paths.

One implementation for every module (curation, sequencing, QC, analysis, metadata)
so editorial rules cannot diverge between selection and verification.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# ---------------------------------------------------------------------------
# Tags / labels
# ---------------------------------------------------------------------------

_TAG_SEP_RE = re.compile(r"[\s_\-]+")
MULTI_VALUE_SPLIT_RE = re.compile(r"\s*[;,/|]\s*")


def normalize_tag(text: Any) -> str:
    """Case-, separator- and Unicode-insensitive label key: 'Feel-Good' == 'feel good'
    == 'FEEL_GOOD'. Empty string for None/blank."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text)).casefold().strip()
    return _TAG_SEP_RE.sub(" ", s).strip()


def split_values(text: Any) -> list[str]:
    """Split a multi-valued tag such as 'Pop/Rock', 'pop; dance' into its parts
    (original spelling kept, blanks dropped)."""
    if text is None:
        return []
    return [p.strip() for p in MULTI_VALUE_SPLIT_RE.split(str(text)) if p.strip()]


def label_keys(text: Any) -> set[str]:
    """Normalized keys of a (possibly multi-valued) label: the whole value and each part."""
    if text is None or not str(text).strip():
        return set()
    keys = {normalize_tag(text)}
    keys.update(normalize_tag(p) for p in split_values(text))
    keys.discard("")
    return keys


# ---------------------------------------------------------------------------
# Artists
# ---------------------------------------------------------------------------

_ARTIST_SPLIT_RE = re.compile(
    r"\s*(?:,|;|/|\s&\s|\s\+\s|\sx\s|\s(?:feat\.?|ft\.?|featuring|with|vs\.?)\s)\s*", re.IGNORECASE)
_ARTIST_FEAT_TAIL_RE = re.compile(r"\s*[(\[]\s*(?:feat\.?|ft\.?|featuring)\s[^)\]]*[)\]]\s*$", re.IGNORECASE)


def normalize_artist(name: Any) -> str:
    if name is None:
        return ""
    s = unicodedata.normalize("NFC", str(name)).casefold().strip()
    return re.sub(r"\s+", " ", s)


def split_artists(artist: Any) -> list[str]:
    """Normalized credited artists. The full credit is always included (so 'AC/DC' and
    'Simon & Garfunkel' still match by their full name), followed by its parts:
    'A/B' (ID3v2.3 separator), 'A, B', 'A & B', 'A x B', 'A feat. B', 'A (feat. B)'."""
    full = normalize_artist(artist)
    if not full:
        return []
    out = [full]
    base = _ARTIST_FEAT_TAIL_RE.sub("", full)
    pieces = [normalize_artist(p) for p in _ARTIST_SPLIT_RE.split(base)]
    for extra in _ARTIST_FEAT_TAIL_RE.findall(full):
        inner = re.sub(r"^[(\[]\s*(?:feat\.?|ft\.?|featuring)\s", "", extra.strip(), flags=re.IGNORECASE)
        pieces += [normalize_artist(p) for p in _ARTIST_SPLIT_RE.split(inner.rstrip(")]"))]
    for p in [base, *pieces]:
        if p and p not in out:
            out.append(p)
    return out


def primary_artist(artist: Any) -> str | None:
    """Main credited artist (featured guests dropped) — used for repetition rules."""
    parts = split_artists(artist)
    if not parts:
        return None
    base = _ARTIST_FEAT_TAIL_RE.sub("", parts[0])
    base = re.split(r"\s(?:feat\.?|ft\.?|featuring)\s", base, flags=re.IGNORECASE)[0].strip()
    return base or parts[0]


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

ISO639_2_TO_1 = {
    "eng": "en", "rus": "ru", "fra": "fr", "fre": "fr", "spa": "es", "ita": "it", "deu": "de", "ger": "de",
    "por": "pt", "nld": "nl", "dut": "nl", "swe": "sv", "nor": "no", "nob": "nb", "nno": "nn", "dan": "da",
    "fin": "fi", "pol": "pl", "ces": "cs", "cze": "cs", "slk": "sk", "slo": "sk", "ukr": "uk", "bel": "be",
    "bul": "bg", "srp": "sr", "hrv": "hr", "slv": "sl", "ron": "ro", "rum": "ro", "hun": "hu", "ell": "el",
    "gre": "el", "tur": "tr", "ara": "ar", "heb": "he", "fas": "fa", "per": "fa", "hin": "hi", "jpn": "ja",
    "kor": "ko", "zho": "zh", "chi": "zh", "vie": "vi", "tha": "th", "ind": "id", "msa": "ms", "may": "ms",
    "cat": "ca", "eus": "eu", "baq": "eu", "glg": "gl", "gle": "ga", "cym": "cy", "wel": "cy", "est": "et",
    "lav": "lv", "lit": "lt", "kaz": "kk", "aze": "az", "hye": "hy", "arm": "hy", "kat": "ka", "geo": "ka",
    "isl": "is", "ice": "is", "lat": "la", "epo": "eo", "afr": "af", "swa": "sw", "tgl": "tl", "fil": "tl",
}
LANGUAGE_NAMES = {
    "english": "en", "russian": "ru", "русский": "ru", "french": "fr", "français": "fr", "francais": "fr",
    "spanish": "es", "español": "es", "espanol": "es", "italian": "it", "italiano": "it", "german": "de",
    "deutsch": "de", "portuguese": "pt", "português": "pt", "portugues": "pt", "dutch": "nl",
    "ukrainian": "uk", "українська": "uk", "украинский": "uk", "belarusian": "be", "беларуская": "be",
    "polish": "pl", "polski": "pl", "swedish": "sv", "japanese": "ja", "korean": "ko", "chinese": "zh",
    "turkish": "tr", "arabic": "ar", "hebrew": "he", "greek": "el", "czech": "cs", "finnish": "fi",
    "danish": "da", "norwegian": "no", "hungarian": "hu", "romanian": "ro", "bulgarian": "bg",
    "serbian": "sr", "croatian": "hr", "kazakh": "kk", "латышский": "lv", "английский": "en",
    "французский": "fr", "испанский": "es", "итальянский": "it", "немецкий": "de",
}
# All ISO 639-1 codes: a 2-letter token outside this list ('jp', 'xx', 'un') is NOT a
# language claim and stays unresolved (-> unknown-language policy).
ISO639_1_CODES = frozenset("""
aa ab ae af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch co cr cs cu cv cy da de dv dz ee el
en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht hu hy hz ia id ie ig ii ik io is it iu
ja jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg mh mi mk ml mn mr ms mt my na
nb nd ne ng nl nn no nr nv ny oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so
sq sr ss st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo za zh zu
""".split())
# "no linguistic content", "undetermined", "multiple", placeholders -> unknown
UNKNOWN_LANGUAGE_TOKENS = frozenset({
    "und", "zxx", "mul", "mis", "unknown", "xxx", "none", "n/a", "na", "-", "?",
    "no linguistic content", "instrumental", "no vocals", "no lyrics", "без слов", "инструментал",
})
_LANG_SPLIT_RE = re.compile(r"\s*[;,/|&+]\s*|\s+and\s+|\s+и\s+")
_ISO1_RE = re.compile(r"^[a-z]{2}$")
_LETTERS_RE = re.compile(r"^[a-z]+$")


def _resolve_language_token(tok: str) -> list[str] | None:
    """ISO 639-1 code(s) for one token, or None when it is not a recognisable language."""
    base = re.split(r"[-_]", tok, maxsplit=1)[0]          # en-US / pt_BR -> en / pt
    if _ISO1_RE.match(base):
        return [base] if base in ISO639_1_CODES else None
    if base in ISO639_2_TO_1:
        return [ISO639_2_TO_1[base]]
    if tok in LANGUAGE_NAMES:
        return [LANGUAGE_NAMES[tok]]
    if base in LANGUAGE_NAMES:
        return [LANGUAGE_NAMES[base]]
    # ID3v2.3 allows several 3-letter codes concatenated ('ruseng'): split only when every
    # triplet is a known ISO 639-2 code, so ordinary words are never mis-read as languages.
    if _LETTERS_RE.match(tok) and len(tok) % 3 == 0 and len(tok) >= 6:
        parts = [tok[i:i + 3] for i in range(0, len(tok), 3)]
        if all(p in ISO639_2_TO_1 for p in parts):
            return [ISO639_2_TO_1[p] for p in parts]
    return None


def parse_languages(value: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(ISO 639-1 codes in order of appearance, unresolved tokens) of a language tag.

    'rus;eng' / 'rus eng' / 'ruseng' -> (('ru', 'en'), ()); 'en-US' -> (('en',), ());
    'klingon' -> ((), ('klingon',)); 'jp' -> ((), ('jp',)); 'und' / 'zxx' / '' -> ((), ()).
    Language NAMES are matched before whitespace splitting, so 'English and French' works."""
    if value is None:
        return (), ()
    if isinstance(value, (list, tuple, set, frozenset)):
        parts = [str(v) for v in value]
    else:
        text = unicodedata.normalize("NFC", str(value)).strip()
        parts = [text] if text.casefold() in LANGUAGE_NAMES else _LANG_SPLIT_RE.split(text)
    tokens: list[str] = []
    for part in parts:
        p = part.strip().casefold()
        words = p.split()
        if len(words) <= 1 or p in LANGUAGE_NAMES or p in UNKNOWN_LANGUAGE_TOKENS:
            tokens.append(p)
            continue
        # whitespace-separated values ('rus eng', 'ru en'): split only when EVERY word is a
        # language, otherwise free text ('No vocals', 'sung in russian') would turn ordinary
        # 2-letter words ('no', 'is', 'it') into language claims; names / 3-letter codes in
        # free text still count, bare 2-letter words then stay unresolved
        if all(w in UNKNOWN_LANGUAGE_TOKENS or _resolve_language_token(w) is not None for w in words):
            tokens.extend(words)
        else:
            tokens.extend(w if len(w) != 2 else f"?{w}" for w in words)
    codes: list[str] = []
    unresolved: list[str] = []
    for raw in tokens:
        tok = raw.strip().casefold()
        if not tok or tok in UNKNOWN_LANGUAGE_TOKENS:
            continue
        resolved = None if tok.startswith("?") else _resolve_language_token(tok)
        tok = tok.lstrip("?")
        if resolved is None:
            if tok not in unresolved:
                unresolved.append(tok)
            continue
        for code in resolved:
            if code not in codes:
                codes.append(code)
    return tuple(codes), tuple(unresolved)


def normalize_language(value: Any) -> str | None:
    """Canonical stored form: ISO 639-1 codes joined with ';' (e.g. 'ru;en'), followed by
    any unresolved tokens as given (lower-case). None when nothing usable is present."""
    codes, unresolved = parse_languages(value)
    parts = list(codes) + list(unresolved)
    return ";".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Library paths
# ---------------------------------------------------------------------------

def normalize_rel_path(value: Any) -> str:
    """Library-relative POSIX path key: forward slashes, no leading './', NFC."""
    if value is None:
        return ""
    s = unicodedata.normalize("NFC", str(value).strip()).replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s


def rel_path_key(value: Any) -> str:
    """Case-insensitive matching key for library-relative paths."""
    return normalize_rel_path(value).casefold()
