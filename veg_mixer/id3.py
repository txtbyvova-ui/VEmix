"""Minimal ID3v2 text-frame reader.

ffprobe reports only the FIRST value of a multi-valued ID3v2.4 text frame (values are
NUL-separated, e.g. TLAN 'eng\\0rus'), which would hide a second language from the
language rules. This reader returns every value of the requested text frames. It is
deliberately small: ID3v2.3/2.4, no unsynchronisation (rare in practice; such tags are
skipped and the caller keeps ffprobe's value).
"""

from __future__ import annotations

from pathlib import Path

_ENCODINGS = {0: "latin-1", 1: "utf-16", 2: "utf-16-be", 3: "utf-8"}


def _syncsafe(b: bytes) -> int:
    return (b[0] & 0x7F) << 21 | (b[1] & 0x7F) << 14 | (b[2] & 0x7F) << 7 | (b[3] & 0x7F)


def read_text_frames(path: Path, frame_ids: tuple[str, ...] = ("TLAN",)) -> dict[str, list[str]]:
    """{frame id: [values]} for the requested ID3v2 text frames; {} when the file has no
    readable ID3v2.3/2.4 tag. Never raises for malformed tags."""
    try:
        with open(path, "rb") as f:
            header = f.read(10)
            if len(header) < 10 or header[:3] != b"ID3" or header[3] not in (3, 4):
                return {}
            version, flags = header[3], header[5]
            if flags & 0x80:                       # unsynchronisation: not supported
                return {}
            size = _syncsafe(header[6:10])
            data = f.read(size)
    except OSError:
        return {}
    pos = 0
    if flags & 0x40 and len(data) >= 4:            # extended header
        ext = _syncsafe(data[:4]) if version == 4 else int.from_bytes(data[:4], "big") + 4
        pos = ext
    wanted = set(frame_ids)
    out: dict[str, list[str]] = {}
    while pos + 10 <= len(data):
        fid = data[pos:pos + 4]
        if fid == b"\x00\x00\x00\x00" or not fid.isalnum():
            break                                  # padding / garbage
        raw_size = data[pos + 4:pos + 8]
        fsize = _syncsafe(raw_size) if version == 4 else int.from_bytes(raw_size, "big")
        body = data[pos + 10:pos + 10 + fsize]
        pos += 10 + fsize
        name = fid.decode("latin-1")
        if name not in wanted or not body:
            continue
        enc = _ENCODINGS.get(body[0])
        if enc is None:
            continue
        try:
            text = body[1:].decode(enc, errors="replace")
        except LookupError:
            continue
        values = [v.strip("﻿ \t\r\n") for v in text.split("\x00") if v.strip("﻿ \t\r\n")]
        if values:
            out.setdefault(name, []).extend(values)
    return out
