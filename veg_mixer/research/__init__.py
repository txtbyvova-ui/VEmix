"""Music Research Agent (VEmix 3.0, spec §3): sourced catalogue facts about library
tracks — MusicBrainz identity and facts, Wikidata cross-check, optional Discogs
cross-check — cached, rate-limited, licence- and provenance-aware.

Research results live in :class:`ResearchRecord` objects, never in ``TrackFeatures``:
external statements are their own category of information (spec §1). See
``enrich`` for the workflow and the sources deliberately not used.
"""

from __future__ import annotations

from .cache import (
    DEFAULT_TTL_S,
    DISCOGS_MAX_AGE_S,
    NEGATIVE_TTL_S,
    RESEARCH_CACHE_SCHEMA,
    RESEARCH_CACHE_SCHEMA_VERSION,
    ResearchCache,
)
from .enrich import (
    DISCOGS_NOTICE,
    REVIEW_CSV_COLUMNS,
    STAT_KEYS,
    EnrichResult,
    default_client,
    default_sources,
    enrich_tracks,
    export_review_csv,
    format_conflict,
    report_document,
)
from .matching import DEFAULT_MATCH_CONFIG, MatchConfig
from .models import (
    DEFAULT_SOURCES,
    DISCOGS_ATTRIBUTION,
    KNOWN_SOURCES,
    LICENSE_CC0,
    LICENSE_DISCOGS,
    LICENSE_MB_SUPPLEMENTARY,
    RESEARCH_CACHE_NAME,
    RESEARCH_MODES,
    STATUSES,
    ResearchRecord,
)

__all__ = [
    "RESEARCH_CACHE_NAME", "RESEARCH_MODES", "KNOWN_SOURCES", "DEFAULT_SOURCES", "STATUSES",
    "LICENSE_CC0", "LICENSE_MB_SUPPLEMENTARY", "LICENSE_DISCOGS", "DISCOGS_ATTRIBUTION",
    "RESEARCH_CACHE_SCHEMA", "RESEARCH_CACHE_SCHEMA_VERSION", "DEFAULT_TTL_S", "NEGATIVE_TTL_S",
    "DISCOGS_MAX_AGE_S", "ResearchRecord", "ResearchCache", "EnrichResult", "MatchConfig",
    "DEFAULT_MATCH_CONFIG", "STAT_KEYS", "REVIEW_CSV_COLUMNS", "DISCOGS_NOTICE", "default_client",
    "default_sources", "enrich_tracks", "export_review_csv", "format_conflict", "report_document",
]
