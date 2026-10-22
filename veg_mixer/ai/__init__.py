"""Optional AI track understanding (VEmix 3.0, spec §4).

Structured, evidence-backed interpretations of tracks — moods, energy trajectory,
danceability / groove, sonic density, dramaturgical roles, suitable phases and risky pairings,
each with a confidence — built ONLY from measured audio features, verified external facts and
the curator's metadata. Audio is never sent anywhere; only a JSON summary without file paths,
file names, artist or title (``PRIVACY_NOTE`` says so in one line for the console).

* ``schema``    — strict output schema + validator (the "no fabrication" evidence guard);
* ``providers`` — pluggable providers: Anthropic (official SDK, optional), OpenAI-compatible
  servers (Ollama / LM Studio / llama.cpp, stdlib HTTP), a deterministic fake;
* ``cache``     — content-addressed cache of answers (``veg_ai_cache.json``);
* ``interpret`` — payload, prompt, re-ask / negative caching / circuit breaker.

Interpretations are advisory and kept apart from ``TrackFeatures`` (spec §1: measured data,
external facts and AI interpretation are separate categories). Nothing here is imported by the
2.0 pipeline, and the ``anthropic`` package is imported only when that provider is created.
"""

from __future__ import annotations

from .cache import AI_CACHE_NAME, AI_CACHE_SCHEMA, AICache, canonical_json
from .interpret import (
    AI_MODES,
    DEFAULT_MAX_TOKENS,
    PHASE_DEFINITIONS,
    PRIVACY_NOTE,
    PROMPT_DIGEST,
    PROMPT_VERSION,
    ROLE_DEFINITIONS,
    SYSTEM_PROMPT,
    InterpretResult,
    build_payload,
    cache_key,
    input_digest,
    interpret_tracks,
    interpretable,
)
from .providers import (
    DEFAULT_ANTHROPIC_MODEL,
    PROVIDER_NAMES,
    AIConfigError,
    AIProvider,
    AIResult,
    AIUnavailableError,
    AnthropicProvider,
    FakeProvider,
    OpenAICompatibleProvider,
    get_provider,
)
from .schema import (
    EVIDENCE_WHITELISTS,
    INTERPRETATION_SCHEMA,
    PHASES,
    ROLES,
    SCHEMA_VERSION,
    check_instance,
    schema_subset_problems,
    validate_interpretation,
)

__all__ = [
    "PROMPT_VERSION", "PROMPT_DIGEST", "PRIVACY_NOTE", "SCHEMA_VERSION", "EVIDENCE_WHITELISTS", "check_instance",
    "AI_CACHE_NAME", "AI_CACHE_SCHEMA", "AI_MODES",
    "DEFAULT_MAX_TOKENS", "ROLES", "PHASES", "ROLE_DEFINITIONS", "PHASE_DEFINITIONS", "INTERPRETATION_SCHEMA",
    "SYSTEM_PROMPT", "validate_interpretation", "schema_subset_problems", "AIProvider", "AIResult", "AIConfigError",
    "AIUnavailableError", "AnthropicProvider", "OpenAICompatibleProvider", "FakeProvider", "get_provider",
    "PROVIDER_NAMES", "DEFAULT_ANTHROPIC_MODEL", "AICache", "canonical_json", "build_payload", "interpretable",
    "cache_key", "input_digest", "InterpretResult", "interpret_tracks",
]
