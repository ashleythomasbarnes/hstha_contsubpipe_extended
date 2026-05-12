"""Linear HST narrowband continuum subtraction.

This module implements the deliberately plain first-pass workflow:

1. find one narrowband image and two broadband continuum images;
2. convert HST count-rate images to flux density using ``PHOTFLAM``;
3. estimate the continuum at the narrowband pivot wavelength in linear space;
4. write the continuum-subtracted narrowband image.

The file-discovery and per-galaxy choices are kept in YAML so reruns for a
single source or with different filters do not require code edits.
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from astropy.io import fits

from .core.config import clear_config_cache, get_configs
from .core.context import build_generic_context
from .core.files import resolve_file

LOGGER = logging.getLogger(__name__)


DEFAULT_CONFIG_DIR = "config"
DEFAULT_GALAXIES_FILE = "galaxies.yaml"


@dataclass(frozen=True)
class ImageSet:
    """Resolved input images for one galaxy."""

    galaxy: str
    blue_filter: str
    red_filter: str
    narrow_filter: str
    blue_file: Path
    red_file: Path
    narrow_file: Path


@dataclass(frozen=True)
class ContsubResult:
    """Output summary for one galaxy."""

    galaxy: str
    status: str
    narrow_filter: str = ""
    contsub_file: Path | None = None
    continuum_file: Path | None = None
    blue_file: Path | None = None
    red_file: Path | None = None
    narrow_file: Path | None = None
    weight_blue: float | None = None
    weight_red: float | None = None
    message: str = ""


def load_galaxy_config(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    filename: str = DEFAULT_GALAXIES_FILE,
) -> dict[str, Any]:
    """Load the galaxy/run configuration YAML file."""

    config_path = Path(config_dir) / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Galaxy config not found: {config_path}")
    with config_path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def configured_galaxies(galaxy_config: Mapping[str, Any]) -> list[str]:
    """Return the configured galaxy names in their YAML order."""

    galaxies = galaxy_config.get("galaxies", [])
    names: list[str] = []
    for entry in galaxies:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, Mapping) and "name" in entry:
            names.append(str(entry["name"]))
        else:
            raise ValueError(f"Invalid galaxy entry in config/galaxies.yaml: {entry!r}")
    return names


def _galaxy_entry(galaxy_config: Mapping[str, Any], galaxy: str) -> dict[str, Any]:
    """Merge defaults and any per-galaxy overrides."""

    defaults = dict(galaxy_config.get("defaults", {}) or {})
    overrides = dict(galaxy_config.get("overrides", {}).get(galaxy, {}) or {})

    for entry in galaxy_config.get("galaxies", []):
        if isinstance(entry, Mapping) and entry.get("name") == galaxy:
            overrides.update({k: v for k, v in entry.items() if k != "name"})
            break

    merged = defaults
    merged.update(overrides)
    return merged


def _filter_digits(filter_name: str) -> str:
    """Return the numeric part used by the HST image-product directories."""

    return "".join(char for char in filter_name if char.isdigit())


def _resolve_template_path(
    template_key: str,
    context_extra: Mapping[str, Any],
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> str:
    """Resolve a path template from ``files.yaml`` with dynamic context."""

    paths, params, files = get_configs(config_dir=config_dir)
    context = build_generic_context(paths=paths, params=params, extra=context_extra)
    return resolve_file(files_cfg=files, key=template_key, context=context)


def _select_match(
    matches: Sequence[Path],
    preferred_tokens: Sequence[str],
    galaxy: str,
    filter_name: str,
) -> Path:
    """Select a unique file from glob matches, using configured token priority."""

    if not matches:
        raise FileNotFoundError(f"No input file found for {galaxy} {filter_name}")

    ordered_matches = sorted(matches)
    for token in preferred_tokens:
        token_matches = [
            path
            for path in ordered_matches
            if f"_{token.lower()}_" in path.name.lower()
            or f"/{token.lower()}" in str(path).lower()
        ]
        if len(token_matches) == 1:
            if len(matches) > 1:
                LOGGER.warning(
                    "Multiple matches for %s %s; selected %s by preferred token '%s'",
                    galaxy,
                    filter_name,
                    token_matches[0],
                    token,
                )
            return token_matches[0]

    if len(ordered_matches) == 1:
        return ordered_matches[0]

    match_list = "\n  ".join(str(path) for path in ordered_matches)
    raise ValueError(
        f"Multiple input files found for {galaxy} {filter_name}; add an override "
        f"in config/galaxies.yaml.\n  {match_list}"
    )


def find_filter_file(
    galaxy: str,
    filter_name: str,
    galaxy_settings: Mapping[str, Any],
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> Path:
    """Find the configured image file for one galaxy/filter combination.

    Per-filter overrides should use keys like ``f555w_file`` or ``f658n_file``.
    If no override is given, ``files.yaml:file_templates.hst_image_glob`` is
    expanded and globbed.
    """

    filter_name = filter_name.lower()
    override_key = f"{filter_name}_file"
    if override_key in galaxy_settings:
        return Path(galaxy_settings[override_key]).expanduser().resolve()

    search_galaxy = str(galaxy_settings.get("search_galaxy", galaxy))
    pattern = _resolve_template_path(
        "hst_image_glob",
        {
            "galaxy": galaxy,
            "search_galaxy": search_galaxy,
            "filter": filter_name,
            "filter_digits": _filter_digits(filter_name),
        },
        config_dir=config_dir,
    )
    matches = [Path(match).resolve() for match in glob.glob(pattern)]
    return _select_match(
        matches=matches,
        preferred_tokens=galaxy_settings.get("preferred_instruments", []),
        galaxy=galaxy,
        filter_name=filter_name,
    )


def resolve_image_set(
    galaxy: str,
    galaxy_config: Mapping[str, Any],
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> ImageSet:
    """Resolve all input images needed to continuum-subtract one galaxy."""

    settings = _galaxy_entry(galaxy_config, galaxy)
    broad_filters = [str(item).lower() for item in settings["broad_filters"]]
    if len(broad_filters) != 2:
        raise ValueError(f"{galaxy}: broad_filters must contain exactly two filters")

    narrow_filters = [str(item).lower() for item in settings["narrow_filters"]]
    blue_filter, red_filter = broad_filters
    blue_file = find_filter_file(galaxy, blue_filter, settings, config_dir=config_dir)
    red_file = find_filter_file(galaxy, red_filter, settings, config_dir=config_dir)

    narrow_file: Path | None = None
    narrow_filter = ""
    errors: list[str] = []
    for candidate in narrow_filters:
        try:
            narrow_file = find_filter_file(galaxy, candidate, settings, config_dir=config_dir)
            narrow_filter = candidate
            break
        except FileNotFoundError as exc:
            errors.append(str(exc))

    if narrow_file is None:
        raise FileNotFoundError(f"{galaxy}: no narrowband file found. " + " ".join(errors))

    return ImageSet(
        galaxy=galaxy,
        blue_filter=blue_filter,
        red_filter=red_filter,
        narrow_filter=narrow_filter,
        blue_file=blue_file,
        red_file=red_file,
        narrow_file=narrow_file,
    )


def convert_hst_count_rate_to_flux_density(hdu: fits.PrimaryHDU) -> fits.PrimaryHDU:
    """Convert an HST count-rate image to ``1e-20 erg s-1 cm-2 A-1 pixel-1``.

    HST drizzled science images are expected to be in count-rate units. The
    ``PHOTFLAM`` header keyword converts them to flux density.
    """

    if "PHOTFLAM" not in hdu.header:
        raise KeyError("Input HDU is missing PHOTFLAM")

    out = hdu.copy()
    out.data = np.asarray(out.data, dtype=np.float32) * float(out.header["PHOTFLAM"]) * 1e20
    out.header["BUNIT"] = ("1e-20 erg/s/cm2/A/pixel", "PHOTFLAM-scaled flux density")
    return out


def linear_continuum_subtract(
    narrow_hdu: fits.PrimaryHDU,
    blue_hdu: fits.PrimaryHDU,
    red_hdu: fits.PrimaryHDU,
    blue_filter: str,
    red_filter: str,
    narrow_filter: str,
) -> tuple[fits.PrimaryHDU, fits.PrimaryHDU, float, float]:
    """Subtract a linear continuum estimate from the narrowband image."""

    if narrow_hdu.data is None or blue_hdu.data is None or red_hdu.data is None:
        raise ValueError("All input HDUs must contain image data")

    shapes = {narrow_hdu.data.shape, blue_hdu.data.shape, red_hdu.data.shape}
    if len(shapes) != 1:
        raise ValueError(
            "Input image shapes differ. Reprojection is not part of this plain "
            f"linear workflow. Shapes: {sorted(shapes)}"
        )

    lam_narrow = float(narrow_hdu.header["PHOTPLAM"])
    lam_blue = float(blue_hdu.header["PHOTPLAM"])
    lam_red = float(red_hdu.header["PHOTPLAM"])
    denominator = abs(lam_blue - lam_red)
    if denominator == 0:
        raise ValueError("Broadband PHOTPLAM values are identical")

    weight_blue = abs(lam_red - lam_narrow) / denominator
    weight_red = abs(lam_blue - lam_narrow) / denominator

    continuum_data = (
        np.asarray(blue_hdu.data, dtype=np.float32) * weight_blue
        + np.asarray(red_hdu.data, dtype=np.float32) * weight_red
    )
    continuum_data[~np.isfinite(continuum_data)] = 0.0
    contsub_data = np.asarray(narrow_hdu.data, dtype=np.float32) - continuum_data

    continuum_hdu = narrow_hdu.copy()
    contsub_hdu = narrow_hdu.copy()
    continuum_hdu.data = np.asarray(continuum_data, dtype=np.float32)
    contsub_hdu.data = np.asarray(contsub_data, dtype=np.float32)

    for hdu, product in ((continuum_hdu, "CONTINUUM"), (contsub_hdu, "CONTSUB")):
        hdu.header["CONTSUB"] = (True, "Produced by hstha_contsubpipe_extended")
        hdu.header["CSPROD"] = (product, "Continuum-subtraction product")
        hdu.header["CSSPACE"] = ("linear", "Continuum interpolation space")
        hdu.header["CSBLUE"] = (blue_filter, "Blue continuum filter")
        hdu.header["CSRED"] = (red_filter, "Red continuum filter")
        hdu.header["CSNB"] = (narrow_filter, "Narrowband filter")
        hdu.header["CSWBLUE"] = (weight_blue, "Blue continuum weight")
        hdu.header["CSWRED"] = (weight_red, "Red continuum weight")
        hdu.header["BUNIT"] = ("1e-20 erg/s/cm2/A/pixel", "PHOTFLAM-scaled flux density")

    return contsub_hdu, continuum_hdu, weight_blue, weight_red


def _output_path(
    template_key: str,
    image_set: ImageSet,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> Path:
    """Resolve one output path from ``files.yaml``."""

    return Path(
        _resolve_template_path(
            template_key,
            {
                "galaxy": image_set.galaxy,
                "narrow_filter": image_set.narrow_filter,
                "blue_filter": image_set.blue_filter,
                "red_filter": image_set.red_filter,
                "contsub_space": "linear",
            },
            config_dir=config_dir,
        )
    )


def run_galaxy(
    galaxy: str,
    galaxy_config: Mapping[str, Any],
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    dry_run: bool = False,
    overwrite: bool | None = None,
) -> ContsubResult:
    """Run or plan continuum subtraction for one galaxy."""

    settings = _galaxy_entry(galaxy_config, galaxy)
    if overwrite is None:
        overwrite = bool(settings.get("overwrite", False))

    image_set = resolve_image_set(galaxy, galaxy_config, config_dir=config_dir)
    contsub_file = _output_path("outputs.contsub", image_set, config_dir=config_dir)
    continuum_file = _output_path("outputs.continuum", image_set, config_dir=config_dir)

    if dry_run:
        return ContsubResult(
            galaxy=galaxy,
            status="planned",
            narrow_filter=image_set.narrow_filter,
            contsub_file=contsub_file,
            continuum_file=continuum_file,
            blue_file=image_set.blue_file,
            red_file=image_set.red_file,
            narrow_file=image_set.narrow_file,
        )

    if contsub_file.exists() and not overwrite:
        return ContsubResult(
            galaxy=galaxy,
            status="skipped",
            narrow_filter=image_set.narrow_filter,
            contsub_file=contsub_file,
            continuum_file=continuum_file,
            blue_file=image_set.blue_file,
            red_file=image_set.red_file,
            narrow_file=image_set.narrow_file,
            message="output exists; use overwrite=True to replace it",
        )

    hdu_index = int(settings.get("hdu_index", 0))
    with (
        fits.open(image_set.narrow_file, memmap=True) as narrow_hdul,
        fits.open(image_set.blue_file, memmap=True) as blue_hdul,
        fits.open(image_set.red_file, memmap=True) as red_hdul,
    ):
        narrow_hdu = convert_hst_count_rate_to_flux_density(narrow_hdul[hdu_index])
        blue_hdu = convert_hst_count_rate_to_flux_density(blue_hdul[hdu_index])
        red_hdu = convert_hst_count_rate_to_flux_density(red_hdul[hdu_index])

        contsub_hdu, continuum_hdu, weight_blue, weight_red = linear_continuum_subtract(
            narrow_hdu=narrow_hdu,
            blue_hdu=blue_hdu,
            red_hdu=red_hdu,
            blue_filter=image_set.blue_filter,
            red_filter=image_set.red_filter,
            narrow_filter=image_set.narrow_filter,
        )

        contsub_file.parent.mkdir(parents=True, exist_ok=True)
        contsub_hdu.writeto(contsub_file, overwrite=overwrite)

        if bool(settings.get("write_continuum", True)):
            continuum_file.parent.mkdir(parents=True, exist_ok=True)
            continuum_hdu.writeto(continuum_file, overwrite=overwrite)
        else:
            continuum_file = None

    return ContsubResult(
        galaxy=galaxy,
        status="written",
        narrow_filter=image_set.narrow_filter,
        contsub_file=contsub_file,
        continuum_file=continuum_file,
        blue_file=image_set.blue_file,
        red_file=image_set.red_file,
        narrow_file=image_set.narrow_file,
        weight_blue=weight_blue,
        weight_red=weight_red,
    )


def _write_manifest(results: Sequence[ContsubResult], config_dir: str | Path) -> Path:
    """Write a CSV manifest summarizing files, products, and failures."""

    paths, params, files = get_configs(config_dir=config_dir)
    context = build_generic_context(paths=paths, params=params)
    manifest_file = Path(resolve_file(files_cfg=files, key="outputs.manifest", context=context))
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "galaxy",
        "status",
        "narrow_filter",
        "blue_file",
        "red_file",
        "narrow_file",
        "contsub_file",
        "continuum_file",
        "weight_blue",
        "weight_red",
        "message",
    ]
    with manifest_file.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: str(getattr(result, field) or "") for field in fields})
    return manifest_file


def run_all(
    galaxies: Iterable[str] | None = None,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    dry_run: bool = False,
    overwrite: bool | None = None,
    keep_going: bool = True,
) -> list[ContsubResult]:
    """Run continuum subtraction for the configured galaxy sample."""

    # Config files are intentionally edited between reruns, especially in
    # notebooks and interactive sessions. Reload them for each pipeline run.
    clear_config_cache()
    galaxy_config = load_galaxy_config(config_dir=config_dir)
    selected_galaxies = (
        list(galaxies) if galaxies is not None else configured_galaxies(galaxy_config)
    )

    results: list[ContsubResult] = []
    for galaxy in selected_galaxies:
        try:
            result = run_galaxy(
                galaxy=galaxy,
                galaxy_config=galaxy_config,
                config_dir=config_dir,
                dry_run=dry_run,
                overwrite=overwrite,
            )
        except Exception as exc:
            if not keep_going:
                raise
            result = ContsubResult(galaxy=galaxy, status="failed", message=str(exc))
        results.append(result)
        LOGGER.info("%s: %s %s", result.galaxy, result.status, result.message)

    if not dry_run:
        manifest_file = _write_manifest(results, config_dir=config_dir)
        LOGGER.info("Wrote manifest: %s", manifest_file)

    return results


def _print_results(results: Sequence[ContsubResult]) -> None:
    """Print a compact command-line summary."""

    for result in results:
        path = result.contsub_file or ""
        message = f" - {result.message}" if result.message else ""
        print(f"{result.galaxy:10s} {result.status:8s} {result.narrow_filter:6s} {path}{message}")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description="Run linear HST H-alpha continuum subtraction.")
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
    _print_results(results)
    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
