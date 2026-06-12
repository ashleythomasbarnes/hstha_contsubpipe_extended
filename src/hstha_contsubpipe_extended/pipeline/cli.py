"""Command-line interface for the continuum-subtraction pipeline."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .galaxy_config import DEFAULT_CONFIG_DIR
from .models import ContsubResult
from .runner import run_all


def print_results(results: Sequence[ContsubResult]) -> None:
    """Print a compact command-line summary."""

    for result in results:
        path = result.contsub_file or ""
        message = f" - {result.message}" if result.message else ""
        print(f"{result.galaxy:10s} {result.status:8s} {result.narrow_filter:6s} {path}{message}")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description="Run configured HST H-alpha continuum subtraction.")
    parser.add_argument(
        "--config-dir", default=DEFAULT_CONFIG_DIR, help="Directory containing YAML config files."
    )
    parser.add_argument(
        "--galaxy", action="append", help="Galaxy to run. May be passed more than once."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve files and outputs without writing FITS files.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output FITS files."
    )
    parser.add_argument(
        "--stop-on-error", action="store_true", help="Stop at the first failed galaxy."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_all(
        galaxies=args.galaxy,
        config_dir=args.config_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite if args.overwrite else None,
        keep_going=not args.stop_on_error,
    )
    print_results(results)
    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
