"""Small filesystem helpers shared by the pipeline (atomic writes, path safety)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8", newline: str = "\n") -> None:
    """Write ``text`` to ``path`` via a temp file in the same directory + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline=newline) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def path_key(path: Path | str) -> str:
    """Identity key of a file system path: resolved, case-folded where the OS is
    case-insensitive (Windows), Unicode-NFC. Two paths with equal keys are the same file."""
    resolved = os.path.realpath(os.path.abspath(str(path)))
    return unicodedata.normalize("NFC", os.path.normcase(resolved))


def name_key(path: Path | str) -> str:
    """Collision key for OUTPUT names: like path_key but always case-insensitive (outputs
    may be copied to case-insensitive media / players)."""
    return path_key(path).casefold()


def is_within(path: Path | str, parent: Path | str) -> bool:
    p, root = path_key(path), path_key(parent)
    return p == root or p.startswith(root.rstrip(os.sep) + os.sep)


def assert_outputs_safe(outputs: Iterable[Path], sources: Iterable[Path]) -> None:
    """Refuse output plans that would overwrite a source file or write two outputs
    to the same name. Raises ValueError (exit 2) listing the problems."""
    source_keys = {path_key(s): Path(s) for s in sources}
    problems: list[str] = []
    seen: dict[str, Path] = {}
    for out in outputs:
        key = path_key(out)
        if key in source_keys:
            problems.append(f"output {out} is a library source file ({source_keys[key]})")
        nk = name_key(out)
        if nk in seen:
            problems.append(f"two outputs map to the same file: {seen[nk]} and {out}")
        else:
            seen[nk] = Path(out)
    if problems:
        head = "; ".join(problems[:5]) + (f"; ... {len(problems) - 5} more" if len(problems) > 5 else "")
        raise ValueError(f"refusing to write outputs: {head}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def remove_stale_dirs(parent: Path, prefix: str, *, max_idle_s: float = 6 * 3600) -> list[Path]:
    """Remove our own work directories (``prefix``*) left behind by killed runs that
    have been idle for more than ``max_idle_s``. Returns the removed paths."""
    removed: list[Path] = []
    try:
        entries = list(Path(parent).iterdir())
    except OSError:
        return removed
    now = time.time()
    for d in entries:
        if not d.name.startswith(prefix) or not d.is_dir():
            continue
        try:
            newest = max([d.stat().st_mtime] + [p.stat().st_mtime for p in d.rglob("*")])
        except OSError:
            continue
        if now - newest > max_idle_s:
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                removed.append(d)
    return removed


def mp3_audio_sha256(path: Path, chunk: int = 1 << 20) -> tuple[str, int]:
    """(sha256, size) of an MP3's AUDIO payload: a leading ID3v2 tag and trailing ID3v1 /
    APEv2 tags are skipped, so a curator's tag-only edit (title, artwork) does not make a
    delivered file look like different audio."""
    with open(path, "rb") as f:
        head = f.read(10)
        start = 0
        if len(head) == 10 and head[:3] == b"ID3":
            size = (head[6] & 0x7F) << 21 | (head[7] & 0x7F) << 14 | (head[8] & 0x7F) << 7 | (head[9] & 0x7F)
            start = 10 + size + (10 if head[5] & 0x10 else 0)
        f.seek(0, os.SEEK_END)
        end = f.tell()
        if end - start >= 128:
            f.seek(end - 128)
            if f.read(3) == b"TAG":
                end -= 128
        if end - start >= 32:
            f.seek(end - 32)
            footer = f.read(32)
            if footer[:8] == b"APETAGEX":
                tag_size = int.from_bytes(footer[12:16], "little")
                has_header = int.from_bytes(footer[20:24], "little") & 0x80000000
                end -= tag_size + (32 if has_header else 0)
        start = min(start, end)
        h = hashlib.sha256()
        f.seek(start)
        remaining = end - start
        while remaining > 0:
            block = f.read(min(chunk, remaining))
            if not block:
                break
            h.update(block)
            remaining -= len(block)
    return h.hexdigest(), end - start


def remove_stale_files(parent: Path, pattern: str, *, max_idle_s: float = 6 * 3600,
                       recursive: bool = False) -> list[Path]:
    """Remove our own hidden temporary FILES (e.g. '.*.veg-*' encode temps next to
    deliverables) left behind by killed runs and idle for more than ``max_idle_s``."""
    removed: list[Path] = []
    try:
        candidates = list(Path(parent).rglob(pattern) if recursive else Path(parent).glob(pattern))
    except OSError:
        return removed
    now = time.time()
    for f in candidates:
        try:
            if f.is_file() and now - f.stat().st_mtime > max_idle_s:
                f.unlink()
                removed.append(f)
        except OSError:
            continue
    return removed
