"""
utils.py — Foundational helpers used across the system.

Contains: logging, image path resolution + encoding (with downscale for large
files), image-ID extraction, a retry decorator with exponential backoff, a
CostTracker for the operational analysis, robust CSV read/write, and a
fuzzy enum matcher that snaps free-text model output onto allowed values.
"""
from __future__ import annotations

import csv
import difflib
import functools
import io
import logging
import mimetypes
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"


def get_logger(name: str = "evidence_review") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger()

# ---------------------------------------------------------------------------
# Image IDs and path resolution
# ---------------------------------------------------------------------------

def image_id_from_path(image_path: str) -> str:
    """`images/test/case_001/img_1.jpg` -> `img_1` (filename without extension)."""
    return Path(image_path.strip()).stem


def split_image_paths(image_paths: str) -> list[str]:
    """Split a semicolon-separated `image_paths` cell into clean relative paths."""
    if not image_paths:
        return []
    return [p.strip() for p in image_paths.split(";") if p.strip()]


def resolve_image_path(rel_path: str) -> Path:
    """Resolve a CSV image path to an absolute file on disk.

    The CSVs reference images as `images/test/...`, but the files live under
    `dataset/images/test/...`. We try the dataset-relative location first, then
    a few sensible fallbacks so the resolver is robust to path variations.
    """
    rel = rel_path.strip()
    candidates = [
        config.DATASET_DIR / rel,          # dataset/images/test/...  (correct)
        config.REPO_ROOT / rel,            # repo/images/...          (fallback)
        Path(rel),                         # already absolute/relative to cwd
    ]
    for c in candidates:
        if c.exists():
            return c
    # Return the primary expected location even if missing, so callers can log it.
    return config.DATASET_DIR / rel


# ---------------------------------------------------------------------------
# Image encoding (with downscale for oversized files)
# ---------------------------------------------------------------------------

@dataclass
class EncodedImage:
    image_id: str
    rel_path: str
    data: bytes
    mime_type: str
    exists: bool
    resized: bool = False
    error: Optional[str] = None


def _sniff_mime(raw: bytes) -> Optional[str]:
    """Identify image format from magic bytes (extension can lie)."""
    if len(raw) < 12:
        return None
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:4] in (b"GIF8",):
        return "image/gif"
    if raw[:4] == b"\x00\x00\x01\x00":
        return "image/x-icon"
    # ISO-BMFF container (AVIF / HEIF) — 'ftyp' box at offset 4. These are NOT
    # accepted by most vision APIs, so flag them so load_image transcodes to JPEG.
    if raw[4:8] == b"ftyp":
        brand = raw[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"mif1", b"heim"):
            return "image/heic"
        return "image/avif"
    return None


def load_image(rel_path: str) -> EncodedImage:
    """Load an image as raw bytes, downscaling/recompressing if it exceeds the
    configured byte threshold. Returns an EncodedImage (never raises)."""
    image_id = image_id_from_path(rel_path)
    abs_path = resolve_image_path(rel_path)
    if not abs_path.exists():
        return EncodedImage(image_id, rel_path, b"", "image/jpeg", exists=False,
                            error=f"file not found: {abs_path}")
    try:
        raw = abs_path.read_bytes()
        # Detect the TRUE format from magic bytes — several dataset files are
        # WebP/PNG/AVIF with a .jpg extension. Some providers reject a wrong mime
        # or unsupported format (e.g. AVIF), so normalize when needed.
        mime = _sniff_mime(raw) or mimetypes.guess_type(str(abs_path))[0] or "image/jpeg"
        safe = mime in ("image/jpeg", "image/png", "image/webp", "image/gif")
        if safe and len(raw) <= config.MAX_IMAGE_BYTES:
            return EncodedImage(image_id, rel_path, raw, mime, exists=True)
        # Unsupported format (e.g. AVIF) or oversized: transcode/downscale to JPEG.
        data, out_mime, changed = _to_jpeg(raw)
        return EncodedImage(image_id, rel_path, data, out_mime, exists=True, resized=changed)
    except Exception as exc:  # pragma: no cover - defensive
        return EncodedImage(image_id, rel_path, b"", "image/jpeg", exists=True,
                            error=f"read/encode error: {exc}")


def _to_jpeg(raw: bytes) -> tuple[bytes, str, bool]:
    """Transcode to JPEG (and downscale if the longest edge exceeds
    MAX_IMAGE_DIMENSION). Used for oversized images and for formats that some
    providers don't accept (e.g. AVIF). Falls back to original bytes if Pillow
    cannot handle it."""
    try:
        from PIL import Image
    except Exception:
        return raw, "image/jpeg", False
    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        longest = max(img.size)
        if longest > config.MAX_IMAGE_DIMENSION:
            scale = config.MAX_IMAGE_DIMENSION / longest
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=config.JPEG_QUALITY, optimize=True)
        return buf.getvalue(), "image/jpeg", True
    except Exception:
        return raw, "image/jpeg", False


# ---------------------------------------------------------------------------
# Retry with exponential backoff (+ jitter)
# ---------------------------------------------------------------------------

def retry(max_retries: int | None = None,
          base_delay: float | None = None,
          max_delay: float | None = None,
          exceptions: tuple[type[BaseException], ...] = (Exception,)) -> Callable:
    """Decorator: retry a callable with exponential backoff and jitter.

    Tuned for transient API failures (429 rate limits, 5xx, timeouts). The final
    failure is re-raised so the caller can decide on a fallback row.
    """
    _max = config.MAX_RETRIES if max_retries is None else max_retries
    _base = config.RETRY_BASE_DELAY if base_delay is None else base_delay
    _cap = config.RETRY_MAX_DELAY if max_delay is None else max_delay

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    attempt += 1
                    if attempt > _max:
                        log.error("Giving up after %d attempts: %s", _max, exc)
                        raise
                    delay = min(_cap, _base * (2 ** (attempt - 1)))
                    delay += random.uniform(0, delay * 0.25)  # jitter
                    log.warning("Attempt %d/%d failed (%s). Retrying in %.1fs",
                                attempt, _max, type(exc).__name__, delay)
                    time.sleep(delay)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Cost / usage tracking (for evaluation/evaluation_report.md)
# ---------------------------------------------------------------------------

@dataclass
class CostTracker:
    calls: int = 0
    cached_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    images: int = 0
    errors: int = 0
    started: float = field(default_factory=time.time)

    def record(self, input_tokens: int = 0, output_tokens: int = 0,
               images: int = 0, cached: bool = False, calls: int = 1) -> None:
        if cached:
            self.cached_hits += 1
            return
        self.calls += int(calls or 1)
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.images += int(images or 0)

    def record_error(self) -> None:
        self.errors += 1

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def estimated_cost(self) -> float:
        in_cost = self.input_tokens / 1_000_000 * config.PRICE_INPUT_PER_M
        out_cost = self.output_tokens / 1_000_000 * config.PRICE_OUTPUT_PER_M
        return in_cost + out_cost

    def summary(self) -> dict:
        return {
            "model_calls": self.calls,
            "cache_hits": self.cached_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "images_processed": self.images,
            "errors": self.errors,
            "elapsed_seconds": round(self.elapsed, 1),
            "estimated_cost_usd": round(self.estimated_cost(), 4),
        }

    def pretty(self) -> str:
        s = self.summary()
        return (
            f"calls={s['model_calls']} cache_hits={s['cache_hits']} "
            f"in_tok={s['input_tokens']} out_tok={s['output_tokens']} "
            f"images={s['images_processed']} errors={s['errors']} "
            f"time={s['elapsed_seconds']}s est_cost=${s['estimated_cost_usd']}"
        )


# ---------------------------------------------------------------------------
# CSV read / write (csv module handles quoted fields & embedded commas)
# ---------------------------------------------------------------------------

def read_csv_dicts(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # QUOTE_ALL mirrors the provided CSVs (every field quoted), which is the
    # safest choice for transcripts containing commas, pipes, and quotes.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Fuzzy enum matching — snap model output onto allowed values
# ---------------------------------------------------------------------------

def _normalize(value: str) -> str:
    return "".join(ch for ch in value.strip().lower() if ch.isalnum() or ch == "_").strip("_")


def closest_enum(value: Optional[str], allowed: Iterable[str], default: str) -> str:
    """Return the closest allowed value for `value`.

    Strategy: exact match -> normalized exact -> substring -> fuzzy ratio.
    Falls back to `default` if nothing is close enough. This guards against the
    model emitting e.g. "dented" instead of "dent" or "rear bumper" vs
    "rear_bumper".
    """
    allowed = list(allowed)
    if value is None:
        return default
    raw = str(value).strip()
    if raw in allowed:
        return raw
    norm = _normalize(raw)
    if not norm:
        return default
    norm_map = {_normalize(a): a for a in allowed}
    if norm in norm_map:
        return norm_map[norm]
    # substring containment (e.g. "rearbumper area" -> "rear_bumper")
    for a_norm, a in norm_map.items():
        if a_norm and (a_norm in norm or norm in a_norm):
            return a
    # fuzzy
    match = difflib.get_close_matches(norm, list(norm_map.keys()), n=1, cutoff=0.8)
    if match:
        return norm_map[match[0]]
    return default
