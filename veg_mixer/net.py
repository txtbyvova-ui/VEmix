"""Minimal HTTP client for the optional online features (VEmix 3.0): music-metadata
research and OpenAI-compatible AI servers. Standard library only (``urllib``).

Guarantees:

* every request has a timeout, a descriptive User-Agent and a per-host minimum interval
  (MusicBrainz allows ~1 request/s per client; going faster gets EVERY request refused);
* transient failures (connection errors, timeouts, HTTP 408/425/429/500/502/503/504/529)
  are retried with exponential backoff, honouring ``Retry-After`` (capped);
* responses are size-limited — both the body on the wire and its gzip-decoded form
  (a small compressed body cannot expand without bound) — and decoded before JSON
  parsing; non-standard ``NaN`` / ``Infinity`` literals become ``None`` and unpaired
  ``\\uD800``-style escapes U+FFFD (a lone surrogate cannot be written as UTF-8);
* nothing escapes as a bare ``OSError`` / ``ValueError``: every failure is a
  :class:`NetError` subclass, so callers can degrade to "no data" instead of the CLI
  mapping a network hiccup to a usage or runtime error;
* secrets are never logged: header values whose name looks like a credential are
  redacted, and query strings are not logged at all.

The transport is injectable (``HttpClient(transport=...)``) so tests never touch the
network.
"""

from __future__ import annotations

import email.utils
import http.client
import json
import logging
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

log = logging.getLogger(__name__)

PROJECT_URL = "https://github.com/txtbyvova-ui/VEmix"
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MAX_RETRIES = 3
MAX_RETRY_AFTER_S = 60.0
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
_SECRET_HEADER_HINTS = ("key", "token", "auth", "secret", "cookie", "password")


class NetError(Exception):
    """Any failure of an online request (network, HTTP status, decoding)."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False,
                 url: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.url = url


class NetTimeout(NetError):
    pass


class NetHTTPError(NetError):
    pass


class NetDecodeError(NetError):
    pass


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]          # lower-cased names
    body: bytes
    url: str


# transport(method, url, headers, body, timeout) -> HttpResponse; raises NetError on failure
Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], HttpResponse]


def user_agent(contact: str | None = None, version: str | None = None) -> str:
    """'VEmix/<version> ( <contact> )' — MusicBrainz and Wikidata require a contact;
    the project URL is the default contact (override with VEG_CONTACT)."""
    from . import __version__
    return f"VEmix/{version or __version__} ( {contact or PROJECT_URL} )"


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k: ("<redacted>" if any(h in k.lower() for h in _SECRET_HEADER_HINTS) else v)
            for k, v in headers.items()}


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.lower()


def _log_url(url: str) -> str:
    """URL without its query string (queries may carry user data or keys)."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follows redirects only to http/https, and never carries credentials to another host:
    urllib's default handler copies EVERY request header — ``Authorization`` with a Discogs
    token or an AI key included — to whatever URL the server names (and allows ftp://)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme.lower() not in ("http", "https"):
            raise urllib.error.HTTPError(newurl, code, "redirect to a non-HTTP URL refused", headers, fp)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and target.netloc.lower() != urllib.parse.urlsplit(req.full_url).netloc.lower():
            for name in list(new.headers):
                if any(h in name.lower() for h in _SECRET_HEADER_HINTS):
                    del new.headers[name]
            for name in list(new.unredirected_hdrs):
                if any(h in name.lower() for h in _SECRET_HEADER_HINTS):
                    del new.unredirected_hdrs[name]
        return new


_OPENER = urllib.request.build_opener(_SafeRedirectHandler())


def _urlopen(req: urllib.request.Request, timeout: float):
    """The one place a request leaves the process (tests replace it: no network in tests)."""
    return _OPENER.open(req, timeout=timeout)


def urllib_transport(method: str, url: str, headers: Mapping[str, str], body: bytes | None,
                     timeout: float) -> HttpResponse:
    try:
        req = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
        with _urlopen(req, timeout) as resp:      # noqa: S310 (https/http only, see HttpClient)
            raw = resp.read(MAX_RESPONSE_BYTES + 1)
            return HttpResponse(resp.status, {k.lower(): v for k, v in resp.headers.items()}, raw, resp.geturl())
    except urllib.error.HTTPError as e:
        try:
            raw = e.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException):
            raw = b""
        return HttpResponse(e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, raw, url)
    except TimeoutError as e:
        raise NetTimeout(f"timed out after {timeout:.0f} s", retryable=True, url=_log_url(url)) from e
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        retry = isinstance(reason, (TimeoutError, ConnectionError, OSError))
        cls = NetTimeout if isinstance(reason, TimeoutError) else NetError
        raise cls(f"connection failed: {reason}", retryable=retry, url=_log_url(url)) from e
    except (ConnectionError, OSError) as e:
        raise NetError(f"connection failed: {e}", retryable=True, url=_log_url(url)) from e
    except http.client.HTTPException as e:
        # IncompleteRead / RemoteDisconnected-like protocol errors (not OSError subclasses);
        # InvalidURL is also an HTTPException and is not retryable
        retry = not isinstance(e, http.client.InvalidURL)
        raise NetError(f"HTTP protocol error: {type(e).__name__}", retryable=retry, url=_log_url(url)) from e
    except ValueError:
        # malformed URL, or a header value http.client refuses (a line break or a character
        # outside Latin-1). Its message echoes that value — possibly a credential such as
        # 'Authorization: Discogs token=…' — or the full URL with its query, so neither the
        # message nor the original exception (``from None``) is passed on
        raise NetError("invalid request: malformed URL or header value (not shown)",
                       url=_log_url(url)) from None


@dataclass
class HttpClient:
    """Rate-limited, retrying JSON-over-HTTP client (thread-safe)."""
    user_agent: str = field(default_factory=user_agent)
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES
    # host -> minimum seconds between two requests to that host
    min_interval_s: dict[str, float] = field(default_factory=dict)
    transport: Transport = urllib_transport
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    jitter_seed: int = 0
    stats: dict[str, int] = field(default_factory=lambda: {"requests": 0, "retries": 0, "failures": 0})

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._next_at: dict[str, float] = {}
        self._rng = random.Random(self.jitter_seed)

    # --- public -------------------------------------------------------------

    def get_json(self, url: str, *, params: Mapping[str, Any] | None = None,
                 headers: Mapping[str, str] | None = None) -> tuple[Any, HttpResponse]:
        if params:
            # errors="replace": a lone surrogate (an undecodable tag or file name) becomes '?'
            # instead of a bare UnicodeEncodeError
            query = urllib.parse.urlencode(params, doseq=True, errors="replace")
            url = f"{url}{'&' if '?' in url else '?'}{query}"
        resp = self.request("GET", url, headers={"Accept": "application/json", **(headers or {})})
        return self._decode_json(resp), resp

    def post_json(self, url: str, payload: Any, *, headers: Mapping[str, str] | None = None) -> tuple[Any, HttpResponse]:
        try:
            text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except ValueError as e:      # NaN / Infinity (not JSON), a circular reference
            raise NetError(f"request body is not valid JSON ({e})", url=_log_url(url)) from None
        body = text.encode("utf-8", "replace")      # a lone surrogate becomes '?', not a crash
        resp = self.request("POST", url, body=body,
                            headers={"Accept": "application/json", "Content-Type": "application/json",
                                     **(headers or {})})
        return self._decode_json(resp), resp

    def request(self, method: str, url: str, *, headers: Mapping[str, str] | None = None,
                body: bytes | None = None) -> HttpResponse:
        scheme = urllib.parse.urlsplit(url).scheme.lower()
        if scheme not in ("http", "https"):
            raise NetError(f"unsupported URL scheme '{scheme}'", url=_log_url(url))
        hdrs = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip", **(headers or {})}
        last: NetError | None = None
        for attempt in range(max(0, self.max_retries) + 1):
            self._wait_turn(url)
            self.stats["requests"] += 1
            try:
                resp = self.transport(method, url, hdrs, body, self.timeout_s)
            except NetError as e:
                last = e
                if not e.retryable or attempt >= self.max_retries:
                    break
                self._backoff(attempt, None, url, str(e))
                continue
            if 200 <= resp.status < 300:
                if len(resp.body) > MAX_RESPONSE_BYTES:
                    self.stats["failures"] += 1
                    raise NetDecodeError(f"response larger than {MAX_RESPONSE_BYTES} bytes", url=_log_url(url))
                return resp
            message = _error_message(resp)
            retry = resp.status in RETRY_STATUSES
            last = NetHTTPError(f"HTTP {resp.status}: {message}", status=resp.status, retryable=retry,
                                url=_log_url(url))
            if not retry or attempt >= self.max_retries:
                break
            self._backoff(attempt, resp.headers.get("retry-after"), url, f"HTTP {resp.status}")
        self.stats["failures"] += 1
        assert last is not None
        log.info("request failed: %s %s: %s", method, _log_url(url), last)
        raise last

    # --- internals ----------------------------------------------------------

    def _wait_turn(self, url: str) -> None:
        host = _host(url)
        interval = self.min_interval_s.get(host, 0.0)
        if interval <= 0:
            return
        with self._lock:
            now = self.clock()
            at = max(now, self._next_at.get(host, 0.0))
            self._next_at[host] = at + interval
        if at > now:
            self.sleep(at - now)

    def _backoff(self, attempt: int, retry_after: str | None, url: str, why: str) -> None:
        delay = _parse_retry_after(retry_after)
        if delay is not None:
            delay = min(MAX_RETRY_AFTER_S, delay)
        else:
            delay = min(30.0, 2.0 ** attempt) + self._rng.uniform(0.0, 0.5)
        self.stats["retries"] += 1
        log.info("retrying %s in %.1f s (%s)", _log_url(url), delay, why)
        self.sleep(delay)

    @staticmethod
    def _decode_json(resp: HttpResponse) -> Any:
        raw = _body_bytes(resp)
        try:
            text = raw.decode("utf-8-sig")
            # NaN / Infinity are not JSON: map them to None instead of letting them reach
            # caches and reports (which must stay JSON-native)
            data = json.loads(text, parse_constant=lambda _c: None)
        except (UnicodeDecodeError, ValueError, RecursionError) as e:
            raise NetDecodeError(f"response is not valid JSON: {e}", url=_log_url(resp.url)) from e
        if _SURROGATE_ESCAPE_RE.search(text):
            data = _replace_lone_surrogates(data)
        return data


# a '\uD800'..'\uDFFF' escape: json.loads turns an UNPAIRED one into a lone surrogate, a
# str that cannot be encoded as UTF-8 — every later cache / report write would fail
_SURROGATE_ESCAPE_RE = re.compile(r"\\u[dD][89a-fA-F][0-9a-fA-F]{2}")
_LONE_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def _replace_lone_surrogates(obj: Any) -> Any:
    """``obj`` with lone surrogates in strings (keys included) replaced by U+FFFD."""
    if isinstance(obj, str):
        return _LONE_SURROGATE_RE.sub("�", obj)
    if isinstance(obj, dict):
        return {_replace_lone_surrogates(k): _replace_lone_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_lone_surrogates(v) for v in obj]
    return obj


def _parse_retry_after(value: str | None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header (delta-seconds or an HTTP date);
    None when absent or unparsable. Never negative."""
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None or when.tzinfo is None:
            return None
        delay = when.timestamp() - time.time()
    if delay != delay:          # NaN
        return None
    return max(0.0, delay)


def _gunzip(raw: bytes, limit: int, url: str) -> bytes:
    """Decompress a gzip body, refusing output larger than ``limit`` (a few KB of
    compressed data can otherwise expand to gigabytes)."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = d.decompress(raw, limit + 1)
        if len(out) > limit:
            raise NetDecodeError(f"decompressed response larger than {limit} bytes", url=_log_url(url))
        out += d.flush()
    except zlib.error as e:
        raise NetDecodeError(f"bad gzip body: {e}", url=_log_url(url)) from e
    if len(out) > limit:
        raise NetDecodeError(f"decompressed response larger than {limit} bytes", url=_log_url(url))
    if not d.eof:
        raise NetDecodeError("bad gzip body: truncated stream", url=_log_url(url))
    return out


def _body_bytes(resp: HttpResponse) -> bytes:
    """Response body, gzip-decoded when compressed (bounded, see ``_gunzip``)."""
    raw = resp.body
    if resp.headers.get("content-encoding", "").lower() == "gzip" or raw[:2] == b"\x1f\x8b":
        return _gunzip(raw, MAX_RESPONSE_BYTES, resp.url)
    return raw


def _error_message(resp: HttpResponse) -> str:
    try:
        body = _body_bytes(resp)
    except NetDecodeError:
        return "undecodable error body"
    try:
        data = json.loads(body.decode("utf-8", errors="replace") or "null")
    except (ValueError, RecursionError):
        return body[:200].decode("utf-8", errors="replace").strip() or "no body"
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("type") or err)[:300]
        if err:
            return str(err)[:300]
        return str(data.get("message") or data)[:300]
    return str(data)[:300]


__all__ = ["NetError", "NetTimeout", "NetHTTPError", "NetDecodeError", "HttpResponse", "Transport", "HttpClient",
           "urllib_transport", "user_agent", "redact_headers", "PROJECT_URL"]
