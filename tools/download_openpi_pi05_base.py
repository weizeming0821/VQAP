#!/usr/bin/env python3
"""Pre-download the openpi pi0.5 base checkpoint and tokenizer."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
import urllib.parse


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OPENPI_SRC = REPO_ROOT / "openpi-vqap" / "src"
DEFAULT_CACHE_DIR = REPO_ROOT / "openpi_cache"

PI05_BASE_PARAMS_URL = "gs://openpi-assets/checkpoints/pi05_base/params"
PI05_BASE_ASSETS_URL = "gs://openpi-assets/checkpoints/pi05_base/assets"
PALIGEMMA_TOKENIZER_URL = "gs://big_vision/paligemma_tokenizer.model"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the openpi pi0.5 base checkpoint into a local cache that "
            "can be reused by later training runs."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Local cache root used as OPENPI_DATA_HOME.",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip downloading gs://openpi-assets/checkpoints/pi05_base/params.",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip downloading gs://big_vision/paligemma_tokenizer.model.",
    )
    parser.add_argument(
        "--include-assets",
        action="store_true",
        help="Also download gs://openpi-assets/checkpoints/pi05_base/assets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the target is already cached.",
    )
    args = parser.parse_args()

    if args.skip_weights and args.skip_tokenizer and not args.include_assets:
        raise ValueError("Nothing to download. Remove a --skip flag or add --include-assets.")

    return args


def ensure_openpi_importable() -> None:
    """Add openpi-vqap/src to sys.path for direct script execution."""
    openpi_src_str = str(OPENPI_SRC)
    if openpi_src_str not in sys.path:
        sys.path.insert(0, openpi_src_str)


def build_local_cache_path(cache_dir: Path, url: str) -> Path:
    """Mirror openpi.shared.download cache path resolution."""
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        return Path(url).resolve()
    return (cache_dir / parsed.netloc / parsed.path.strip("/")).resolve()


def format_size(num_bytes: int) -> str:
    """Format a byte count in a human-readable form."""
    suffixes = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(num_bytes)
    for suffix in suffixes:
        if size < 1024.0 or suffix == suffixes[-1]:
            return f"{size:.1f}{suffix}"
        size /= 1024.0
    return f"{num_bytes}B"


def compute_path_size(path: Path) -> int:
    """Return the recursive size of a file or directory."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def format_duration(seconds: float) -> str:
    """Format elapsed seconds in a compact form."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def download_one(
    *,
    download_module,
    index: int,
    total: int,
    name: str,
    url: str,
    cache_dir: Path,
    force: bool,
    extra_kwargs: dict[str, object] | None = None,
) -> Path:
    """Download one artifact and print a readable progress header."""
    extra_kwargs = extra_kwargs or {}
    target_path = build_local_cache_path(cache_dir, url)
    status = "cached" if target_path.exists() and not force else "download"

    print()
    print("=" * 80)
    print(f"[{index}/{total}] {name}")
    print(f"  url:    {url}")
    print(f"  target: {target_path}")
    print(f"  mode:   {status}")
    print("=" * 80)

    start_time = time.time()
    path = download_module.maybe_download(url, force_download=force, **extra_kwargs)
    elapsed = time.time() - start_time

    size_bytes = compute_path_size(path)
    print(f"[done] {name}")
    print(f"  path:    {path}")
    print(f"  size:    {format_size(size_bytes)}")
    print(f"  elapsed: {format_duration(elapsed)}")
    return path


def main() -> int:
    """Run the downloader."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENPI_DATA_HOME"] = str(cache_dir)

    ensure_openpi_importable()
    from openpi.shared import download

    jobs: list[tuple[str, str, dict[str, object]]] = []
    if not args.skip_weights:
        jobs.append(("pi05 base params", PI05_BASE_PARAMS_URL, {}))
    if not args.skip_tokenizer:
        jobs.append(("paligemma tokenizer", PALIGEMMA_TOKENIZER_URL, {"gs": {"token": "anon"}}))
    if args.include_assets:
        jobs.append(("pi05 base assets", PI05_BASE_ASSETS_URL, {}))

    print(f"OPENPI_DATA_HOME={cache_dir}")
    print(f"openpi src: {OPENPI_SRC}")
    print(f"planned downloads: {len(jobs)}")

    for index, (name, url, extra_kwargs) in enumerate(jobs, start=1):
        download_one(
            download_module=download,
            index=index,
            total=len(jobs),
            name=name,
            url=url,
            cache_dir=cache_dir,
            force=args.force,
            extra_kwargs=extra_kwargs,
        )

    print()
    print("All downloads finished.")
    print(f"Reuse this cache in later runs with: export OPENPI_DATA_HOME={cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
