"""Shared primitives for resumable dataset downloads."""
from __future__ import annotations

import csv
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def download_file(url: str, destination: Path, dry_run: bool = False) -> bool:
    """Download once; a non-empty existing file is treated as complete."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"SKIP existing: {destination.name}")
        return False
    if dry_run:
        print(f"DRY RUN download: {url} -> {destination}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
        print(f"RESUME {destination.name} at {offset:,} bytes")
    else:
        print(f"DOWNLOAD {url}")
    with urllib.request.urlopen(request) as response:
        # Servers that ignore Range must not have their full response appended.
        mode = "ab" if offset and response.status == 206 else "wb"
        with temporary.open(mode) as handle:
            shutil.copyfileobj(response, handle)
    temporary.replace(destination)
    return True


def count_audio_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES)


def disk_usage(path: Path) -> str:
    if not path.exists():
        return "0 B"
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if total < 1024 or unit == "TiB":
            return f"{total:.1f} {unit}"
        total /= 1024
    return "0 B"


def write_rows(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


TIMED_OUT = object()


def imap_with_timeout(fn, items, workers: int = 4, timeout: float = 120.0):
    """Ordered ``(item, fn(item))`` over ``items`` on ``workers`` daemon threads.

    An item still running ``timeout`` seconds after the consumer starts waiting for it yields
    ``(item, TIMED_OUT)`` and its slot is handed to the next item, so one hung read (e.g. a
    Google Drive file stuck after a network drop) cannot stall the run; the stuck daemon thread
    is abandoned and does not block interpreter exit. Exceptions are returned as results.
    """
    import threading

    items = list(items)
    n = len(items)
    out: list = [None] * n
    done = [threading.Event() for _ in range(n)]
    released = [False] * n
    lock = threading.Lock()
    slots = threading.Semaphore(workers)

    def release(i: int) -> None:
        with lock:
            if not released[i]:
                released[i] = True
                slots.release()

    def run(i: int) -> None:
        try:
            out[i] = fn(items[i])
        except BaseException as exc:  # noqa: BLE001 - handed back to the caller
            out[i] = exc
        finally:
            done[i].set()
            release(i)

    started = 0

    def start(i: int) -> None:
        threading.Thread(target=run, args=(i,), daemon=True).start()

    for i in range(n):
        while started <= i:  # item i must be running before we wait for it
            slots.acquire()
            start(started)
            started += 1
        while started < min(n, i + 1 + 4 * workers) and slots.acquire(blocking=False):  # prefetch
            start(started)
            started += 1
        if done[i].wait(timeout):
            yield items[i], out[i]
        else:
            release(i)
            yield items[i], TIMED_OUT
