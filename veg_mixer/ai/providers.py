"""Pluggable AI providers for track interpretation (VEmix 3.0, spec §4).

The interpreter talks to a provider through one method, ``complete_json(system=, user=,
schema=, max_tokens=)``, that returns an :class:`AIResult` and NEVER raises for API, network or
JSON problems: every failure becomes ``AIResult.error`` (+ ``error_kind``, ``retryable``), so
one bad answer or a dropped connection degrades to "no interpretation" instead of aborting the
run (the CLI would otherwise map an ``OSError`` / ``ValueError`` to the wrong exit code).

Providers (``get_provider(name)``; ``name`` None -> ``VEG_AI_PROVIDER``):

* ``anthropic`` — Claude through the official Anthropic Python SDK (optional dependency,
  ``pip install -r requirements-ai.txt``; imported lazily so the package stays optional).
  Credentials are resolved by the SDK itself (``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` /
  ``ant auth login`` profile; ``ANTHROPIC_BASE_URL`` is honoured): this module never reads,
  accepts or logs a key. Request: ``client.beta.messages.create`` with structured output
  (``output_config.format = json_schema``) and ``effort`` (``VEG_AI_EFFORT``, default
  ``medium``). Server-side refusal fallback is opted in by default (beta
  ``server-side-fallback-2026-07-01``, ``fallbacks="default"``): when a safety classifier
  declines, the API re-runs the request on Anthropic's recommended fallback model inside the
  same call. It works on the Claude API only — set ``VEG_AI_FALLBACKS=off`` behind a proxy or
  gateway that rejects it. No ``temperature`` / ``top_p`` / ``top_k``, no
  ``thinking={"type": "disabled"}`` and no forced ``tool_choice``: Claude Opus 5.5 rejects all of
  them (400); its thinking cannot be switched off and counts towards ``max_tokens``.
  ``stop_reason`` is checked before the content is read: ``refusal``, ``max_tokens`` and
  ``model_context_window_exceeded`` give an error, never a half-parsed answer. After a fallback
  the answer is the text AFTER the last ``fallback`` content block (before it is the declining
  model's partial output) and the reported model is the one that served it (``usage.iterations``
  / the block's ``to.model``). Default model ``claude-opus-5-5`` (``VEG_AI_MODEL`` or
  the ``model`` argument override it). SDK retries (``max_retries``, default 2) cover 408/409/
  429/5xx and connection errors.
* ``openai-compatible`` (aliases ``ollama``, ``lmstudio``, ``llamacpp``) — any server with an
  OpenAI-style ``/chat/completions`` endpoint, through the stdlib client in ``veg_mixer.net``
  (timeouts, retries with ``Retry-After``). Structured output via ``response_format``
  ``json_schema`` (strict); a server that rejects it (400 mentioning response_format) is asked
  once more with ``json_object`` and the schema in the prompt, and is remembered as such.
  ``temperature=0`` and ``seed=0`` keep local models repeatable where the server supports it.
  The model is REQUIRED (argument or ``VEG_AI_MODEL``). An optional key comes only from
  ``VEG_AI_API_KEY`` — or ``OPENAI_API_KEY`` when the base URL is OpenAI's own API (that
  variable is often set globally; it is never sent to a local or third-party server) — and is
  sent as a Bearer header (never logged; header values are redacted in debug logs and the key is
  scrubbed from error texts). Base URL:
  argument > ``VEG_AI_BASE_URL`` > alias default (``ollama``: ``OLLAMA_API_BASE`` + ``/v1``, else
  http://localhost:11434/v1; ``lmstudio``: http://localhost:1234/v1; ``llamacpp``:
  http://localhost:8080/v1). Plain ``openai-compatible`` has no default: data is never sent to
  a cloud service the user did not name.
* ``fake`` — deterministic provider for tests and demos: canned answers, a function, or (by
  default) a trivial echo of curator / segment fields with the evidence they came from.

Only the JSON payload built by ``interpret.build_payload`` (or the anonymous library summary of
``scenario_suggest``) is ever sent — never audio, file paths, folder or file names, artist or
title. Timeout: argument > ``VEG_AI_TIMEOUT_S`` > 120 s.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import logging
import math
import os
import urllib.parse
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from ..net import (
    HttpClient,
    HttpResponse,
    NetDecodeError,
    NetError,
    NetHTTPError,
    NetTimeout,
    Transport,
    redact_headers,
)
from .schema import DENSITIES, TRAJECTORIES

log = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_EFFORT = "medium"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
RESPONSE_FORMAT_NAME = "vemix_track_interpretation"
INSTALL_HINT = "pip install -r requirements-ai.txt"
DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "llamacpp": "http://localhost:8080/v1",
}
PROVIDER_NAMES = ("anthropic", "openai-compatible", "ollama", "lmstudio", "llamacpp", "fake")
_ALIASES = {
    "claude": "anthropic", "openai": "openai-compatible", "openai_compatible": "openai-compatible",
    "openaicompatible": "openai-compatible", "lm-studio": "lmstudio", "lm_studio": "lmstudio",
    "llama.cpp": "llamacpp", "llama-cpp": "llamacpp", "llama_cpp": "llamacpp",
}
_OFF = frozenset({"off", "0", "false", "no", "none", "disabled"})
_ON = frozenset({"", "on", "1", "true", "yes", "default", "enabled"})

# error_kind values. A FATAL kind will fail the same way for every track (bad credentials,
# unknown model, missing package): the interpreter stops calling the provider at once.
FATAL_ERROR_KINDS = frozenset({"auth", "permission", "not_found", "config"})
# The provider answered, but not with a usable JSON object: the interpreter may re-ask.
INVALID_OUTPUT = "invalid_output"


class AIConfigError(ValueError):
    """Invalid AI provider configuration (unknown name, missing model / base URL, bad env value)."""


class AIUnavailableError(AIConfigError):
    """The provider's optional dependency is not installed."""


@dataclass(frozen=True)
class AIResult:
    """Outcome of one provider call. ``data`` is the parsed JSON (not yet validated) or None.

    ``model`` is the model that produced the answer (for Anthropic with a server-side fallback
    this can differ from the requested one). ``error_kind`` classifies a failure: ``auth``,
    ``permission``, ``not_found``, ``config``, ``bad_request``, ``rate_limit``, ``server``,
    ``timeout``, ``connection``, ``refused``, ``incomplete``, ``bad_response``,
    ``invalid_output``, ``http`` or ``sdk``."""
    data: dict | None
    raw_text: str
    provider: str
    model: str
    request_id: str | None = None
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    error: str | None = None
    retryable: bool = False
    error_kind: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.data is not None


@runtime_checkable
class AIProvider(Protocol):
    name: str
    model: str

    def complete_json(self, *, system: str, user: str, schema: dict, max_tokens: int = 4000) -> AIResult: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not valid JSON")


def parse_json_text(text: str) -> tuple[Any, str | None]:
    """Parse a model's JSON answer: plain JSON, a ```json fenced block, or (lenient, for local
    models) the first JSON object in the text. NaN / Infinity are rejected.
    Returns (data, None) or (None, error message)."""
    s = (text or "").strip().lstrip("﻿")
    if not s:
        return None, "empty answer"
    if s.startswith("```"):
        body = s.split("\n", 1)[1] if "\n" in s else s[3:]       # '```json {...}```' on one line
        s = body.rsplit("```", 1)[0].strip()
    try:
        return json.loads(s, parse_constant=_reject_constant), None
    except ValueError as e:
        first_error = str(e)
    except RecursionError:          # absurd nesting: not an answer (and must not escape)
        return None, "answer is not valid JSON (nested too deeply)"
    start = s.find("{")
    if start >= 0:
        try:
            data, _ = json.JSONDecoder(parse_constant=_reject_constant).raw_decode(s, start)
            return data, None
        except (ValueError, RecursionError):
            pass
    return None, f"answer is not valid JSON ({first_error})"


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _timeout(timeout_s: float | None, env: Mapping[str, str]) -> float:
    raw: Any = timeout_s if timeout_s is not None else env.get("VEG_AI_TIMEOUT_S") or DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise AIConfigError(f"AI timeout must be a number of seconds, got {raw!r} (VEG_AI_TIMEOUT_S)") from None
    if not math.isfinite(value) or value <= 0:
        raise AIConfigError(f"AI timeout must be > 0 seconds, got {raw!r}")
    return value


def _retries(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise AIConfigError(f"max_retries must be an integer, got {value!r}") from None
    if n < 0:
        raise AIConfigError(f"max_retries must be >= 0, got {n}")
    return n


def _usage_ints(pairs: Mapping[str, Any]) -> dict[str, int]:
    return {k: int(v) for k, v in pairs.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


# ---------------------------------------------------------------------------
# Fake provider (tests / demos)
# ---------------------------------------------------------------------------

Responder = Callable[[str, str, dict], Any]


class FakeProvider:
    """Deterministic provider.

    ``responses``:

    * None — a trivial echo of the payload (curator mood / danceability, segment trajectory and
      density) with the evidence it came from; everything else unknown. A demo, not an
      interpretation: its source label is ``ai:fake/<model>``;
    * a dict — returned for every call;
    * a list / tuple — consumed in order, one per call (an error result when exhausted);
    * a callable ``(system, user, schema) -> answer``.

    An answer may be a dict, a JSON string (parsed like a real model's text), an
    :class:`AIResult` (returned as is) or an exception instance (raised — to test that the
    interpreter survives a misbehaving provider). Every call is recorded in ``calls``.
    """

    def __init__(self, responses: Any = None, *, model: str = "fake-1", name: str = "fake") -> None:
        self.name = name
        self.model = model
        self._responses = responses
        self._queue = list(responses) if isinstance(responses, (list, tuple)) else None
        self.calls: list[dict] = []

    @staticmethod
    def payload_of(user: str) -> dict:
        """The input payload at the start of a user message (re-asks append text after it)."""
        try:
            data, _ = json.JSONDecoder().raw_decode(user.lstrip())
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def complete_json(self, *, system: str, user: str, schema: dict, max_tokens: int = 4000) -> AIResult:
        self.calls.append({"system": system, "user": user, "schema": schema, "max_tokens": max_tokens})
        answer = self._next(system, user, schema)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, AIResult):
            return answer
        if isinstance(answer, str):
            data, err = parse_json_text(answer)
            if err:
                return AIResult(None, answer, self.name, self.model, stop_reason="end_turn",
                                error=f"{INVALID_OUTPUT}: {err}", error_kind=INVALID_OUTPUT)
            return AIResult(data, answer, self.name, self.model, stop_reason="end_turn")
        if isinstance(answer, dict):
            return AIResult(copy.deepcopy(answer), json.dumps(answer, ensure_ascii=False), self.name, self.model,
                            stop_reason="end_turn")
        return AIResult(None, "", self.name, self.model, error=f"bad_response: fake answer of type "
                        f"{type(answer).__name__}", error_kind="bad_response")

    def _next(self, system: str, user: str, schema: dict) -> Any:
        if self._queue is not None:
            if not self._queue:
                return AIResult(None, "", self.name, self.model, error="bad_response: fake provider has no more "
                                "answers", error_kind="bad_response")
            return self._queue.pop(0)
        if callable(self._responses):
            return self._responses(system, user, schema)
        if isinstance(self._responses, dict):
            return self._responses
        return echo_interpretation(self.payload_of(user))


def echo_interpretation(payload: Mapping[str, Any]) -> dict:
    """The fake provider's default answer: restates curator / segment fields with evidence."""
    from ..moods import mood_labels

    def unknown(v: Any = "unknown") -> dict:
        return {"value": v, "confidence": 0.0, "evidence": []}

    curator = payload.get("curator") or {}
    seg = (payload.get("measured") or {}).get("segments") or {}
    out: dict[str, Any] = {"moods": [], "energy_trajectory": unknown(), "danceability": unknown(None),
                           "groove": unknown(None), "sonic_density": unknown(), "roles": [], "positions": [],
                           "risky_pairings": [],
                           "notes": "fake provider: restates curator and segment fields; not an interpretation"}
    for label in mood_labels(curator.get("mood"))[:6]:
        out["moods"].append({"label": label, "confidence": 0.6, "evidence": ["curator.mood"]})
    traj, traj_path = seg.get("trajectory"), "measured.segments.trajectory"
    if isinstance(traj, Mapping):                       # audio_intel: {"value", "confidence", ...}
        traj, traj_path = traj.get("value"), f"{traj_path}.value"
    if traj in TRAJECTORIES and traj != "unknown":
        out["energy_trajectory"] = {"value": traj, "confidence": 0.5, "evidence": [traj_path]}
    dens, dens_path = seg.get("density"), "measured.segments.density"
    if isinstance(dens, Mapping):                       # audio_intel: {"value", "score", "confidence"}
        dens, dens_path = dens.get("value"), f"{dens_path}.value"
    if isinstance(dens, (int, float)) and not isinstance(dens, bool) and math.isfinite(dens):
        value = "sparse" if dens < 0.33 else "moderate" if dens < 0.66 else "dense"
        out["sonic_density"] = {"value": value, "confidence": 0.4, "evidence": [dens_path]}
    elif dens in DENSITIES and dens != "unknown":
        out["sonic_density"] = {"value": dens, "confidence": 0.4, "evidence": [dens_path]}
    dance = curator.get("danceability")
    if isinstance(dance, (int, float)) and not isinstance(dance, bool) and 0.0 <= dance <= 1.0:
        out["danceability"] = {"value": float(dance), "confidence": 0.9, "evidence": ["curator.danceability"]}
    return out


# ---------------------------------------------------------------------------
# Anthropic (official SDK)
# ---------------------------------------------------------------------------

def _import_anthropic() -> Any:
    try:
        import anthropic
    except ImportError as e:
        raise AIUnavailableError(f"the 'anthropic' provider needs the official Anthropic SDK, which is not "
                                 f"installed ({e}); install it with: {INSTALL_HINT}") from None
    return anthropic


def _fallbacks_enabled(value: bool | None, env: Mapping[str, str]) -> bool:
    if value is not None:
        return bool(value)
    raw = str(env.get("VEG_AI_FALLBACKS", "")).strip().lower()
    if raw in _OFF:
        return False
    if raw in _ON:
        return True
    raise AIConfigError(f"VEG_AI_FALLBACKS must be 'default' or 'off', got {raw!r}")


def _served_model(response: Any, content: list) -> str | None:
    """The model that produced the returned answer. With a server-side fallback the SDK
    reports it in ``usage.iterations`` (an entry of type ``fallback_message``) and in the last
    ``fallback`` content block (``to.model``); otherwise it is ``response.model``."""
    iterations = getattr(getattr(response, "usage", None), "iterations", None)
    if isinstance(iterations, (list, tuple)):
        for it in reversed(iterations):
            model = getattr(it, "model", None)
            if getattr(it, "type", None) == "fallback_message" and isinstance(model, str) and model:
                return model
    for block in reversed(content):
        if getattr(block, "type", None) == "fallback":
            model = getattr(getattr(block, "to", None), "model", None)
            if isinstance(model, str) and model:
                return model
            break
    model = getattr(response, "model", None)
    return model if isinstance(model, str) and model else None


class AnthropicProvider:
    """Claude via ``anthropic`` (``client.beta.messages.create``). See the module docstring.

    ``client`` injects a ready client (tests pass a fake object with ``.beta.messages.create``);
    otherwise ``anthropic.Anthropic(timeout=, max_retries=[, base_url=])`` is created here and
    resolves credentials itself."""

    name = "anthropic"

    def __init__(self, model: str | None = None, *, client: Any = None, base_url: str | None = None,
                 timeout_s: float | None = None, max_retries: int = DEFAULT_MAX_RETRIES,
                 effort: str | None = None, fallbacks: bool | None = None,
                 env: Mapping[str, str] | None = None) -> None:
        env = _env(env)
        self.model = (model or env.get("VEG_AI_MODEL") or DEFAULT_ANTHROPIC_MODEL).strip()
        eff = (effort or env.get("VEG_AI_EFFORT") or DEFAULT_EFFORT).strip().lower()
        if eff not in EFFORTS:
            raise AIConfigError(f"unknown effort {eff!r} (VEG_AI_EFFORT); use one of {', '.join(EFFORTS)}")
        self.effort = eff
        self.fallbacks = _fallbacks_enabled(fallbacks, env)
        self.timeout_s = _timeout(timeout_s, env)
        self.max_retries = _retries(max_retries)
        try:
            self._sdk: Any = _import_anthropic()
        except AIUnavailableError:
            if client is None:
                raise
            self._sdk = None          # injected client: error mapping falls back to generic
        if client is None:
            kwargs: dict[str, Any] = {"timeout": self.timeout_s, "max_retries": self.max_retries}
            if base_url:
                kwargs["base_url"] = base_url
            try:
                client = self._sdk.Anthropic(**kwargs)
            except self._sdk.AnthropicError as e:
                raise AIConfigError(f"could not create the Anthropic client: {type(e).__name__}: {e}") from None
        self._client = client

    def request_kwargs(self, *, system: str, user: str, schema: dict, max_tokens: int = 4000) -> dict:
        """The exact keyword arguments of ``client.beta.messages.create``."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(max_tokens),
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}, "effort": self.effort},
        }
        if self.fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def complete_json(self, *, system: str, user: str, schema: dict, max_tokens: int = 4000) -> AIResult:
        kwargs = self.request_kwargs(system=system, user=user, schema=schema, max_tokens=max_tokens)
        try:
            response = self._client.beta.messages.create(**kwargs)
        except Exception as e:   # noqa: BLE001 - mapped below; never escapes to the caller
            return self._error(e)
        return self._result(response)

    # --- response ------------------------------------------------------------

    def _result(self, response: Any) -> AIResult:
        request_id = getattr(response, "_request_id", None)
        stop = getattr(response, "stop_reason", None)
        content = list(getattr(response, "content", None) or [])
        served = _served_model(response, content) or self.model
        usage_obj = getattr(response, "usage", None)
        usage = _usage_ints({k: getattr(usage_obj, k, None) for k in
                             ("input_tokens", "output_tokens", "cache_read_input_tokens",
                              "cache_creation_input_tokens")})
        if request_id:
            log.debug("anthropic: request %s stop_reason=%s model=%s", request_id, stop, served)
        if served != self.model:
            log.info("anthropic: answer served by %s (server-side fallback from %s)", served, self.model)

        def fail(kind: str, message: str, raw: str = "", retryable: bool = False) -> AIResult:
            return AIResult(None, raw, self.name, str(served), request_id=request_id, stop_reason=stop,
                            usage=usage, error=f"{kind}: {message}", retryable=retryable, error_kind=kind)

        # stop_reason BEFORE content: a refused or truncated answer is never parsed
        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            return fail("refused", f"the model declined to answer (category: {category or 'unspecified'})")
        # With a server-side fallback, `content` holds the declining model's partial output, a
        # `fallback` block, then the serving model's output: the answer is AFTER the last such block.
        last_fallback = max((i for i, b in enumerate(content) if getattr(b, "type", None) == "fallback"),
                            default=-1)
        text = next((getattr(b, "text", None) for b in content[last_fallback + 1:]
                     if getattr(b, "type", None) == "text"), None)
        if stop in ("max_tokens", "model_context_window_exceeded"):
            return fail("incomplete", f"the answer stopped ({stop}) before it was complete (raise "
                        "VEG_AI_MAX_TOKENS or lower VEG_AI_EFFORT)", text or "")
        if not isinstance(text, str):
            return fail("bad_response", f"no text block in the response (stop_reason {stop!r})")
        data, err = parse_json_text(text)
        if err:
            return fail(INVALID_OUTPUT, err, text)
        return AIResult(data, text, self.name, str(served), request_id=request_id, stop_reason=stop, usage=usage)

    def _error(self, e: Exception) -> AIResult:
        sdk = self._sdk
        kind, retryable = "sdk", False
        message = getattr(e, "message", None) or str(e) or type(e).__name__
        request_id = getattr(e, "request_id", None)
        if sdk is not None:
            # most specific first: APITimeoutError is a subclass of APIConnectionError
            if isinstance(e, sdk.AuthenticationError):
                kind = "auth"
            elif isinstance(e, sdk.PermissionDeniedError):
                kind = "permission"
            elif isinstance(e, sdk.NotFoundError):
                kind, message = "not_found", f"{message} (is the model id {self.model!r} right?)"
            elif isinstance(e, sdk.BadRequestError):
                kind = "bad_request"
            elif isinstance(e, sdk.RateLimitError):
                kind, retryable = "rate_limit", True
            elif isinstance(e, sdk.APIStatusError):
                status = int(getattr(e, "status_code", 0) or 0)
                kind, retryable = ("server", True) if status >= 500 else ("http", False)
                message = f"HTTP {status}: {message}"
            elif isinstance(e, sdk.APITimeoutError):
                kind, retryable = "timeout", True
            elif isinstance(e, sdk.APIConnectionError):
                kind, retryable = "connection", True
            elif isinstance(e, getattr(sdk, "CredentialsError", ())):
                kind = "auth"
        if kind == "sdk" and isinstance(e, TypeError):
            if "authentication method" in str(e):
                kind, message = "auth", ("no Anthropic credentials found: set ANTHROPIC_API_KEY (or "
                                         "ANTHROPIC_AUTH_TOKEN), or sign in with `ant auth login`")
            else:   # e.g. an SDK too old for output_config / fallbacks
                kind, message = "config", f"the Anthropic SDK rejected the request ({e}); {INSTALL_HINT}"
        log.warning("anthropic: %s error%s: %s", kind, f" (request {request_id})" if request_id else "", message)
        return AIResult(None, "", self.name, self.model, request_id=request_id, error=f"{kind}: {message}",
                        retryable=retryable, error_kind=kind)


# ---------------------------------------------------------------------------
# OpenAI-compatible servers (stdlib)
# ---------------------------------------------------------------------------

def _resolve_base_url(alias: str, base_url: str | None, env: Mapping[str, str]) -> str:
    url = (base_url or env.get("VEG_AI_BASE_URL") or "").strip()
    if not url and alias == "ollama":
        host = (env.get("OLLAMA_API_BASE") or "").strip().rstrip("/")
        url = (host if host.endswith("/v1") else f"{host}/v1") if host else DEFAULT_BASE_URLS["ollama"]
    if not url:
        url = DEFAULT_BASE_URLS.get(alias, "")
    if not url:
        raise AIConfigError("the 'openai-compatible' provider needs a base URL: pass it or set VEG_AI_BASE_URL "
                            "(e.g. http://localhost:11434/v1)")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise AIConfigError(f"AI base URL must be an http(s) URL, got {url!r}")
    return url.rstrip("/")


def _host_of(netloc: str) -> str:
    host = netloc.rsplit("@", 1)[-1].lower()
    if host.startswith("["):                       # [::1]:8080
        return host[1:].split("]", 1)[0]
    return host.split(":")[0]


def _is_local(netloc: str) -> bool:
    host = _host_of(netloc)
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")


def _is_openai_host(netloc: str) -> bool:
    host = _host_of(netloc)
    return host == "openai.com" or host.endswith(".openai.com")


def _api_key(env: Mapping[str, str], base_url: str) -> str | None:
    """``VEG_AI_API_KEY`` for any server. ``OPENAI_API_KEY`` only when the server IS OpenAI's
    API: that variable is often set globally for other tools, and sending it as a Bearer
    token to a local / LAN / third-party server would hand an OpenAI credential to that host."""
    key = (env.get("VEG_AI_API_KEY") or "").strip()
    if not key and _is_openai_host(urllib.parse.urlsplit(base_url).netloc):
        key = (env.get("OPENAI_API_KEY") or "").strip()
    return key or None


def _number(value: Any, name: str, cast: Callable[[Any], Any]) -> Any:
    try:
        out = cast(value)
    except (TypeError, ValueError, OverflowError):
        raise AIConfigError(f"{name} must be a number, got {value!r}") from None
    if isinstance(out, float) and not math.isfinite(out):
        raise AIConfigError(f"{name} must be finite, got {value!r}")
    return out


class OpenAICompatibleProvider:
    """``POST {base_url}/chat/completions`` through ``veg_mixer.net.HttpClient``. See the module
    docstring. ``http`` injects a client (tests pass one with a fake transport)."""

    def __init__(self, model: str | None = None, *, alias: str = "openai-compatible", base_url: str | None = None,
                 timeout_s: float | None = None, max_retries: int = DEFAULT_MAX_RETRIES,
                 http: HttpClient | None = None, temperature: float = 0.0, seed: int = 0,
                 env: Mapping[str, str] | None = None) -> None:
        env = _env(env)
        self.name = alias
        self.model = (model or env.get("VEG_AI_MODEL") or "").strip()
        if not self.model:
            raise AIConfigError(f"the {alias!r} provider needs a model name: pass it or set VEG_AI_MODEL "
                                f"(e.g. the name shown by `ollama list`)")
        self.base_url = _resolve_base_url(alias, base_url, env)
        self.timeout_s = _timeout(timeout_s, env)
        self.temperature = _number(temperature, "temperature", float)
        self.seed = _number(seed, "seed", int)
        # Never stored anywhere else, never logged; scrubbed from error texts.
        self._api_key = _api_key(env, self.base_url)
        base = http or HttpClient(timeout_s=self.timeout_s, max_retries=_retries(max_retries))
        # HttpClient logs a failed request's error message; a server that echoes the key back in
        # an error body would put it in the log. Scrub it from error bodies before HttpClient
        # sees them (a copy of an injected client: the caller's object is not modified).
        self._http = dataclasses.replace(base, transport=_scrubbing_transport(base.transport, self._api_key)) \
            if self._api_key else base
        self._json_schema_ok = True      # False once the server rejected response_format json_schema
        netloc = urllib.parse.urlsplit(self.base_url).netloc
        if self._api_key and self.base_url.startswith("http://") and not _is_local(netloc):
            log.warning("AI provider %s: the API key is sent over plain HTTP to %s", alias, netloc)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def request_body(self, *, system: str, user: str, schema: dict, max_tokens: int = 4000,
                     json_schema: bool = True) -> dict:
        if json_schema:
            fmt: dict[str, Any] = {"type": "json_schema",
                                   "json_schema": {"name": RESPONSE_FORMAT_NAME, "strict": True, "schema": schema}}
        else:
            fmt = {"type": "json_object"}
            system = (f"{system}\n\nReturn ONLY one JSON object that conforms to this JSON schema:\n"
                      f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}")
        return {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": fmt, "max_tokens": int(max_tokens),
                "temperature": self.temperature, "seed": self.seed}

    def _headers(self) -> dict[str, str]:
        # identity encoding: error bodies stay plain text, so the key can be scrubbed from them
        auth = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        return {"Accept-Encoding": "identity", **auth}

    def _post(self, body: dict) -> tuple[Any, Any]:
        headers = self._headers()
        log.debug("AI request: POST %s headers=%s", self.endpoint, redact_headers(headers))
        return self._http.post_json(self.endpoint, body, headers=headers)

    def complete_json(self, *, system: str, user: str, schema: dict, max_tokens: int = 4000) -> AIResult:
        body = self.request_body(system=system, user=user, schema=schema, max_tokens=max_tokens,
                                 json_schema=self._json_schema_ok)
        try:
            data, resp = self._post(body)
        except NetHTTPError as e:
            if not (self._json_schema_ok and e.status == 400 and _mentions_response_format(str(e))):
                return self._net_error(e)
            log.info("AI provider %s rejected response_format json_schema (%s); retrying with json_object",
                     self.name, self._scrub(str(e)))
            self._json_schema_ok = False
            body = self.request_body(system=system, user=user, schema=schema, max_tokens=max_tokens,
                                     json_schema=False)
            try:
                data, resp = self._post(body)
            except NetError as e2:
                return self._net_error(e2)
        except NetError as e:
            return self._net_error(e)
        return self._parse(data, resp)

    def _parse(self, data: Any, resp: Any) -> AIResult:
        headers = getattr(resp, "headers", {}) or {}
        request_id = headers.get("x-request-id") or headers.get("request-id")

        def fail(kind: str, message: str, raw: str = "", stop: str | None = None, usage: dict | None = None) -> AIResult:
            return AIResult(None, raw, self.name, self.model, request_id=request_id, stop_reason=stop,
                            usage=usage or {}, error=f"{kind}: {self._scrub(message)}", error_kind=kind)

        if not isinstance(data, dict):
            return fail("bad_response", f"response is {type(data).__name__}, expected an object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return fail("bad_response", "response has no choices")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        stop = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
        raw_usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        usage = _usage_ints({"input_tokens": raw_usage.get("prompt_tokens"),
                             "output_tokens": raw_usage.get("completion_tokens")})
        model = data.get("model") if isinstance(data.get("model"), str) and data.get("model") else self.model
        if message.get("refusal"):
            return fail("refused", f"the model declined to answer: {str(message['refusal'])[:200]}", "", stop, usage)
        content = message.get("content")
        if stop == "length":
            return fail("incomplete", "the answer hit max_tokens before it was complete",
                        content if isinstance(content, str) else "", stop, usage)
        if not isinstance(content, str) or not content.strip():
            return fail("bad_response", "empty answer", "", stop, usage)
        parsed, err = parse_json_text(content)
        if err:
            return fail(INVALID_OUTPUT, err, content, stop, usage)
        return AIResult(parsed, content, self.name, model, request_id=request_id, stop_reason=stop, usage=usage)

    def _net_error(self, e: NetError) -> AIResult:
        status = getattr(e, "status", None)
        if isinstance(e, NetTimeout):
            kind = "timeout"
        elif isinstance(e, NetDecodeError):
            kind = "bad_response"
        elif isinstance(e, NetHTTPError):
            kind = {401: "auth", 403: "permission", 404: "not_found", 400: "bad_request",
                    429: "rate_limit"}.get(status or 0, "server" if (status or 0) >= 500 else "http")
        else:
            kind = "connection"
        message = self._scrub(str(e))
        log.warning("AI provider %s: %s error: %s", self.name, kind, message)
        return AIResult(None, "", self.name, self.model, error=f"{kind}: {message}", retryable=bool(e.retryable),
                        error_kind=kind)

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "<redacted>") if self._api_key else text


def _scrubbing_transport(inner: Transport, secret: str) -> Transport:
    """Transport wrapper that replaces ``secret`` in error response bodies with '<redacted>'
    (``HttpClient`` logs an error body's message). A gzip error body — sent although the
    request asked for ``identity`` — is decompressed (bounded) first so it can be scrubbed."""
    needle = secret.encode("utf-8")

    def transport(method: str, url: str, headers: Mapping[str, str], body: bytes | None,
                  timeout: float) -> HttpResponse:
        resp = inner(method, url, headers, body, timeout)
        if resp.status < 400:
            return resp
        raw, hdrs = resp.body, resp.headers
        if hdrs.get("content-encoding", "").lower() == "gzip" or raw[:2] == b"\x1f\x8b":
            plain = _gunzip_small(raw)
            if plain is None:            # undecodable: never let unscrubbed bytes through
                return HttpResponse(resp.status, {k: v for k, v in hdrs.items() if k != "content-encoding"},
                                    b"undecodable error body", resp.url)
            raw, hdrs = plain, {k: v for k, v in hdrs.items() if k != "content-encoding"}
        if needle in raw or raw is not resp.body:
            resp = HttpResponse(resp.status, hdrs, raw.replace(needle, b"<redacted>"), resp.url)
        return resp

    return transport


def _gunzip_small(raw: bytes, limit: int = 1024 * 1024) -> bytes | None:
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = d.decompress(raw, limit)
    except zlib.error:
        return None
    return out if d.eof else None


def _mentions_response_format(message: str) -> bool:
    m = message.lower()
    return "response_format" in m or "json_schema" in m or "response format" in m


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def get_provider(name: str | None = None, *, model: str | None = None, base_url: str | None = None,
                 timeout_s: float | None = None, **kw: Any) -> AIProvider:
    """Create a provider by name (None -> ``VEG_AI_PROVIDER``). Extra keyword arguments go to the
    provider's constructor (``env=``, ``client=``, ``http=``, ``responses=``, ``max_retries=``,
    ``effort=``, ``fallbacks=``). Raises :class:`AIConfigError` (a ValueError) for an unknown or
    missing name and incomplete configuration, :class:`AIUnavailableError` when the SDK is missing."""
    env = _env(kw.get("env"))
    raw = name if name is not None else env.get("VEG_AI_PROVIDER")
    key = str(raw or "").strip().lower()
    key = _ALIASES.get(key, key)
    if not key:
        raise AIConfigError(f"no AI provider selected: set VEG_AI_PROVIDER or pass a provider name "
                            f"({', '.join(PROVIDER_NAMES)})")
    if key not in PROVIDER_NAMES:
        raise AIConfigError(f"unknown AI provider {raw!r}; available: {', '.join(PROVIDER_NAMES)}")
    if key == "fake":            # network / SDK options do not apply to the fake provider
        return FakeProvider(kw.get("responses"), model=model or "fake-1")
    if key == "anthropic":
        return AnthropicProvider(model, base_url=base_url, timeout_s=timeout_s, **_options(AnthropicProvider, key, kw))
    return OpenAICompatibleProvider(model, alias=key, base_url=base_url, timeout_s=timeout_s,
                                    **_options(OpenAICompatibleProvider, key, kw))


def _options(cls: type, key: str, kw: dict[str, Any]) -> dict[str, Any]:
    """Constructor options of ``cls``. None means "not given" (a CLI passes every option): it is
    dropped so the constructor's default / environment applies; a set value of a foreign option
    is a configuration error."""
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self", "model", "alias", "base_url", "timeout_s"}
    out: dict[str, Any] = {}
    for k, v in kw.items():
        if v is None:
            continue
        if k not in accepted:
            raise AIConfigError(f"option {k!r} does not apply to the {key!r} AI provider")
        out[k] = v
    return out


__all__ = ["AIConfigError", "AIUnavailableError", "AIResult", "AIProvider", "FakeProvider", "AnthropicProvider",
           "OpenAICompatibleProvider", "get_provider", "parse_json_text", "echo_interpretation", "PROVIDER_NAMES",
           "DEFAULT_ANTHROPIC_MODEL", "DEFAULT_TIMEOUT_S", "DEFAULT_EFFORT", "EFFORTS", "FALLBACK_BETA",
           "FATAL_ERROR_KINDS", "INVALID_OUTPUT", "RESPONSE_FORMAT_NAME", "INSTALL_HINT"]
