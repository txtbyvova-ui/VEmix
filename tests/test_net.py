"""HTTP client for the optional online features (VEmix 3.0 ``net.py``): retries,
Retry-After, gzip, size limits, scheme checks, error mapping and secret redaction.
No test touches the network: every request goes to an injected transport."""

from __future__ import annotations

import gzip
import http.client
import json
import logging
import urllib.error
import urllib.request
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone

import pytest

from veg_mixer import net
from veg_mixer import net as net_module
from veg_mixer.net import (
    HttpClient,
    HttpResponse,
    NetDecodeError,
    NetError,
    NetHTTPError,
    NetTimeout,
    redact_headers,
    urllib_transport,
    user_agent,
)


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


class Script:
    """Transport returning scripted responses / raising scripted errors, in order."""

    def __init__(self, *items, clock: FakeClock | None = None) -> None:
        self.items = list(items)
        self.calls: list[dict] = []
        self.clock = clock

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": body,
                           "timeout": timeout, "t": self.clock.t if self.clock else None})
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, Exception):
            raise item
        return item


def resp(data=None, status=200, headers=None, body: bytes | None = None, url="https://example.org/x"):
    if body is None:
        body = json.dumps(data).encode("utf-8")
    return HttpResponse(status, {k.lower(): v for k, v in (headers or {}).items()}, body, url)


def client(transport, clock=None, **kw) -> HttpClient:
    clock = clock or FakeClock()
    return HttpClient(transport=transport, sleep=clock.sleep, clock=clock.monotonic, **kw)


# ---------------------------------------------------------------------------
# retries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", sorted(net.RETRY_STATUSES))
def test_retry_statuses_then_success(status):
    clock = FakeClock()
    t = Script(resp({"e": 1}, status=status), resp({"ok": True}))
    c = client(t, clock)
    data, r = c.get_json("https://example.org/api")
    assert data == {"ok": True} and r.status == 200
    assert len(t.calls) == 2
    assert c.stats == {"requests": 2, "retries": 1, "failures": 0}
    assert len(clock.sleeps) == 1 and 1.0 <= clock.sleeps[0] <= 1.5      # 2**0 + jitter


def test_retries_exhausted_raise_http_error_with_status():
    clock = FakeClock()
    t = Script(resp({"error": "busy"}, status=503))
    c = client(t, clock, max_retries=2)
    with pytest.raises(NetHTTPError) as ei:
        c.get_json("https://example.org/api")
    assert ei.value.status == 503 and ei.value.retryable
    assert "busy" in str(ei.value)
    assert len(t.calls) == 3 and c.stats["failures"] == 1
    # exponential backoff: ~1 s, ~2 s
    assert 1.0 <= clock.sleeps[0] <= 1.5 and 2.0 <= clock.sleeps[1] <= 2.5


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
def test_non_retry_4xx_fail_immediately(status):
    clock = FakeClock()
    t = Script(resp({"error": "nope"}, status=status))
    c = client(t, clock)
    with pytest.raises(NetHTTPError) as ei:
        c.get_json("https://example.org/api")
    assert ei.value.status == status and not ei.value.retryable
    assert len(t.calls) == 1 and clock.sleeps == []


def test_retry_after_seconds_is_honoured_and_capped():
    clock = FakeClock()
    t = Script(resp(status=429, body=b"", headers={"Retry-After": "3"}),
               resp(status=429, body=b"", headers={"Retry-After": "3600"}), resp({"ok": 1}))
    c = client(t, clock)
    c.get_json("https://example.org/api")
    assert clock.sleeps == [3.0, net.MAX_RETRY_AFTER_S]


def test_retry_after_http_date_and_garbage():
    clock = FakeClock()
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    t = Script(resp(status=503, body=b"", headers={"Retry-After": when}),
               resp(status=503, body=b"", headers={"Retry-After": "soon-ish"}), resp({"ok": 1}))
    c = client(t, clock)
    c.get_json("https://example.org/api")
    assert 20.0 <= clock.sleeps[0] <= 31.0          # HTTP-date: about 30 s from now
    assert 2.0 <= clock.sleeps[1] <= 2.5            # unparsable: exponential backoff


def test_transport_errors_retryable_vs_not():
    clock = FakeClock()
    t = Script(NetTimeout("timed out", retryable=True), resp({"ok": 1}))
    c = client(t, clock)
    assert c.get_json("https://example.org/a")[0] == {"ok": 1}
    t2 = Script(NetError("bad request line", retryable=False))
    c2 = client(t2, clock)
    with pytest.raises(NetError):
        c2.get_json("https://example.org/a")
    assert len(t2.calls) == 1


def test_negative_max_retries_still_makes_one_attempt():
    t = Script(resp({"ok": 1}))
    c = client(t, max_retries=-3)
    assert c.get_json("https://example.org/a")[0] == {"ok": 1}


# ---------------------------------------------------------------------------
# decoding / limits
# ---------------------------------------------------------------------------

def test_gzip_body_with_and_without_header():
    payload = {"title": "Größe", "n": 3}
    gz = gzip.compress(json.dumps(payload).encode("utf-8"))
    for headers in ({"Content-Encoding": "gzip"}, {}):            # magic bytes detected too
        c = client(Script(resp(body=gz, headers=headers)))
        assert c.get_json("https://example.org/a")[0] == payload


def test_gzip_bomb_and_corrupt_gzip_are_decode_errors(monkeypatch):
    monkeypatch.setattr(net, "MAX_RESPONSE_BYTES", 10_000)
    bomb = gzip.compress(b"[" + b"0," * 200_000 + b"0]")
    assert len(bomb) < 10_000
    c = client(Script(resp(body=bomb, headers={"Content-Encoding": "gzip"})))
    with pytest.raises(NetDecodeError, match="decompressed response larger"):
        c.get_json("https://example.org/a")
    broken = gzip.compress(b'{"a": 1}')[:-6]
    c = client(Script(resp(body=broken, headers={"Content-Encoding": "gzip"})))
    with pytest.raises(NetDecodeError, match="gzip"):
        c.get_json("https://example.org/a")


def test_size_limit(monkeypatch):
    monkeypatch.setattr(net, "MAX_RESPONSE_BYTES", 100)
    c = client(Script(resp(body=b'{"a": "' + b"x" * 200 + b'"}')))
    with pytest.raises(NetDecodeError, match="larger than 100 bytes"):
        c.get_json("https://example.org/a")
    assert c.stats["failures"] == 1


def test_invalid_json_and_non_standard_constants():
    c = client(Script(resp(body=b"<html>Service Unavailable</html>")))
    with pytest.raises(NetDecodeError, match="not valid JSON"):
        c.get_json("https://example.org/a")
    # NaN / Infinity are not JSON: they become None instead of leaking into caches
    c = client(Script(resp(body=b'{"a": NaN, "b": [Infinity, -Infinity], "c": 1.5}')))
    assert c.get_json("https://example.org/a")[0] == {"a": None, "b": [None, None], "c": 1.5}
    # a UTF-8 BOM is tolerated
    c = client(Script(resp(body=b'\xef\xbb\xbf{"a": 1}')))
    assert c.get_json("https://example.org/a")[0] == {"a": 1}


def test_unpaired_surrogate_escapes_become_replacement_chars():
    # json.loads keeps an unpaired '\ud800' as a lone surrogate, which no UTF-8 write accepts
    body = b'{"t": "a\\ud800b", "\\udc00k": ["\\ud83d\\ude00", "ok"]}'
    data = client(Script(resp(body=body))).get_json("https://example.org/a")[0]
    assert data == {"t": "a�b", "�k": ["\U0001F600", "ok"]}       # a valid pair is kept
    json.dumps(data, ensure_ascii=False).encode("utf-8")


def test_unencodable_request_data_never_escapes_as_valueerror():
    t = Script(resp({"ok": 1}))
    c = client(t)
    # a lone surrogate (undecodable tag / file name) in a query parameter or a JSON body
    assert c.get_json("https://example.org/a", params={"q": "a\udc80b"})[0] == {"ok": 1}
    assert "q=a%3Fb" in t.calls[0]["url"]
    c.post_json("https://example.org/p", {"x": "a\udc80b"})
    assert json.loads(t.calls[1]["body"].decode("utf-8")) == {"x": "a?b"}
    # NaN is not JSON: a NetError (callers degrade), never an invalid body on the wire
    with pytest.raises(NetError, match="not valid JSON"):
        c.post_json("https://example.org/p", {"x": float("nan")})
    assert len(t.calls) == 2


def test_error_message_from_gzipped_error_body():
    gz = gzip.compress(json.dumps({"error": {"message": "rate limited"}}).encode())
    c = client(Script(resp(status=403, body=gz, headers={"Content-Encoding": "gzip"})))
    with pytest.raises(NetHTTPError, match="rate limited"):
        c.get_json("https://example.org/a")


# ---------------------------------------------------------------------------
# request building / safety
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.org/x", "javascript:alert(1)", "/relative"])
def test_scheme_check_never_reaches_transport(url):
    t = Script(resp({"ok": 1}))
    with pytest.raises(NetError, match="unsupported URL scheme"):
        client(t).get_json(url)
    assert t.calls == []


def test_user_agent_params_and_post_body():
    t = Script(resp({"ok": 1}))
    c = client(t, user_agent=user_agent("ops@example.invalid", version="9.9"))
    c.get_json("https://example.org/ws?fmt=json", params={"query": 'recording:"a b" AND artist:"c"', "limit": 10})
    call = t.calls[0]
    assert call["headers"]["User-Agent"] == "VEmix/9.9 ( ops@example.invalid )"
    assert call["headers"]["Accept"] == "application/json" and call["headers"]["Accept-Encoding"] == "gzip"
    assert call["url"].startswith("https://example.org/ws?fmt=json&query=recording%3A%22a+b%22+AND")
    assert "limit=10" in call["url"]
    c.post_json("https://example.org/p", {"x": "é"})
    post = t.calls[1]
    assert post["method"] == "POST" and json.loads(post["body"].decode("utf-8")) == {"x": "é"}
    assert post["headers"]["Content-Type"] == "application/json"


def test_default_user_agent_names_project_contact():
    ua = user_agent()
    assert ua.startswith("VEmix/") and net.PROJECT_URL in ua


def test_redact_headers():
    h = {"Authorization": "Discogs token=SECRET", "X-Api-Key": "k", "Cookie": "c", "x-auth-token": "t",
         "Accept": "application/json", "User-Agent": "VEmix/3 ( x )"}
    r = redact_headers(h)
    assert r["Authorization"] == r["X-Api-Key"] == r["Cookie"] == r["x-auth-token"] == "<redacted>"
    assert r["Accept"] == "application/json" and r["User-Agent"] == "VEmix/3 ( x )"


def test_rate_limit_per_host():
    clock = FakeClock()
    t = Script(resp({"ok": 1}), clock=clock)
    c = client(t, clock, min_interval_s={"a.example": 1.1})
    for _ in range(3):
        c.get_json("https://a.example/x")
    c.get_json("https://b.example/x")          # other host: no wait
    times = [call["t"] for call in t.calls]
    assert times[1] - times[0] >= 1.1 - 1e-9 and times[2] - times[1] >= 1.1 - 1e-9
    assert times[3] == times[2]


def test_failed_request_log_has_no_query_string(caplog):
    t = Script(resp({"error": "nope"}, status=404))
    c = client(t)
    with caplog.at_level(logging.DEBUG, logger="veg_mixer.net"):
        with pytest.raises(NetHTTPError):
            c.get_json("https://example.org/a?token=SUPERSECRET&q=x")
    assert "SUPERSECRET" not in caplog.text
    assert "https://example.org/a" in caplog.text


# ---------------------------------------------------------------------------
# urllib transport error mapping (urlopen monkeypatched — no network)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc, cls, retryable", [
    (TimeoutError("slow"), NetTimeout, True),
    (urllib.error.URLError(TimeoutError("slow")), NetTimeout, True),
    (urllib.error.URLError(ConnectionRefusedError("refused")), NetError, True),
    (ConnectionResetError("reset"), NetError, True),
    (http.client.IncompleteRead(b"partial"), NetError, True),
    (http.client.InvalidURL("bad url"), NetError, False),
    (ValueError("unknown url type"), NetError, False),
])
def test_urllib_transport_maps_errors(monkeypatch, exc, cls, retryable):
    def boom(*_a, **_k):
        raise exc
    monkeypatch.setattr(net_module, "_urlopen", boom)
    with pytest.raises(cls) as ei:
        urllib_transport("GET", "https://example.org/x?secret=1", {}, None, 5.0)
    assert ei.value.retryable is retryable
    assert "secret" not in (ei.value.url or "")


def test_urllib_transport_rejects_malformed_url_without_bare_valueerror():
    # Request() itself raises ValueError for these; the transport must map it (no network)
    with pytest.raises(NetError):
        urllib_transport("GET", "not a url", {}, None, 5.0)


def test_invalid_header_value_error_does_not_echo_the_secret(caplog):
    # http.client refuses a header value with a line break BEFORE connecting (no network);
    # its ValueError message contains the value — here a credential — and must not leak
    secret = "SECRET-TOKEN-42"
    with pytest.raises(NetError) as ei:
        urllib_transport("GET", "http://127.0.0.1:9/x", {"Authorization": f"Discogs token={secret}\nX: 1"}, None, 1.0)
    assert secret not in str(ei.value) and ei.value.__cause__ is None and not ei.value.retryable
    # through the client too: neither the error nor the log carries it
    c = client(lambda *a: urllib_transport("GET", "http://127.0.0.1:9/x",
                                           {"Authorization": f"Discogs token={secret}\nX: 1"}, None, 1.0))
    with caplog.at_level(logging.DEBUG, logger="veg_mixer.net"):
        with pytest.raises(NetError) as ei2:
            c.get_json("http://127.0.0.1:9/x")
    assert secret not in str(ei2.value) and secret not in caplog.text


def test_urllib_transport_returns_http_error_responses(monkeypatch):
    import io
    from email.message import Message

    def http_error(*_a, **_k):
        headers = Message()
        headers["Retry-After"] = "2"
        raise urllib.error.HTTPError("https://example.org/x", 503, "busy", headers, io.BytesIO(b'{"error": "busy"}'))
    monkeypatch.setattr(net_module, "_urlopen", http_error)
    r = urllib_transport("GET", "https://example.org/x", {}, None, 5.0)
    assert r.status == 503 and r.headers["retry-after"] == "2" and r.body == b'{"error": "busy"}'


def test_redirects_never_carry_credentials_to_another_host_or_scheme():
    """urllib's default redirect handler copies every header (Authorization included) to the
    redirect target and follows ftp://; the transport must not."""
    import http.server
    import socketserver
    import threading

    from veg_mixer import net as net_mod

    seen: dict = {}

    class Target(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen["auth"] = self.headers.get("Authorization")
            seen["ua"] = self.headers.get("User-Agent")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    target = socketserver.TCPServer(("127.0.0.1", 0), Target)
    port_b = target.server_address[1]

    class Origin(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", "ftp://example.invalid/x" if self.path == "/ftp"
                             else f"http://localhost:{port_b}/b")
            self.end_headers()

        def log_message(self, *a):
            pass

    origin = socketserver.TCPServer(("127.0.0.1", 0), Origin)
    for srv in (origin, target):
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port_a = origin.server_address[1]
        r = net_mod.urllib_transport("GET", f"http://127.0.0.1:{port_a}/a",
                                     {"Authorization": "Bearer SECRET", "User-Agent": "ua"}, None, 5)
        assert r.status == 200 and seen["auth"] is None and seen["ua"] == "ua"
        r2 = net_mod.urllib_transport("GET", f"http://127.0.0.1:{port_a}/ftp", {"Authorization": "x"}, None, 5)
        assert r2.status == 302
    finally:
        origin.shutdown()
        target.shutdown()
