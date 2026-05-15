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
from contextlib import ExitStack
import csv
import glob
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table

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
    blue_error_file: Path | None = None
    red_error_file: Path | None = None
    narrow_error_file: Path | None = None


@dataclass(frozen=True)
class ContsubResult:
    """Output summary for one galaxy."""

    galaxy: str
    status: str
    narrow_filter: str = ""
    contsub_file: Path | None = None
    continuum_file: Path | None = None
    contsub_error_file: Path | None = None
    continuum_error_file: Path | None = None
    halpha_file: Path | None = None
    halpha_error_file: Path | None = None
    blue_file: Path | None = None
    red_file: Path | None = None
    narrow_file: Path | None = None
    blue_error_file: Path | None = None
    red_error_file: Path | None = None
    narrow_error_file: Path | None = None
    weight_blue: float | None = None
    weight_red: float | None = None
    extinction_ebv: float | None = None
    nii_to_halpha: float | None = None
    narrowband_width: float | None = None
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
    product: str = "science",
) -> Path:
    """Find the configured image file for one galaxy/filter combination.

    Per-filter overrides should use keys like ``f555w_file`` or ``f658n_file``.
    Error-image overrides use keys like ``f555w_error_file``.
    If no override is given, ``files.yaml:file_templates.hst_image_glob`` is
    expanded and globbed for science images, or ``hst_error_glob`` for errors.
    """

    filter_name = filter_name.lower()
    if product not in {"science", "error"}:
        raise ValueError(f"Unknown product {product!r}; expected 'science' or 'error'")

    override_key = f"{filter_name}_file" if product == "science" else f"{filter_name}_error_file"
    if override_key in galaxy_settings:
        return Path(galaxy_settings[override_key]).expanduser().resolve()

    if product == "error":
        science_override = f"{filter_name}_file"
        if science_override in galaxy_settings:
            candidate = Path(
                str(galaxy_settings[science_override]).replace(
                    "_exp_drc_sci.fits", "_err_drc_wht.fits"
                )
            ).expanduser()
            if candidate.exists():
                return candidate.resolve()

    search_galaxy = str(galaxy_settings.get("search_galaxy", galaxy))
    template_key = "hst_image_glob" if product == "science" else "hst_error_glob"
    pattern = _resolve_template_path(
        template_key,
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
        filter_name=f"{filter_name} {product}",
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

    require_errors = bool(settings.get("require_errors", True))
    blue_error_file = red_error_file = narrow_error_file = None
    try:
        blue_error_file = find_filter_file(
            galaxy, blue_filter, settings, config_dir=config_dir, product="error"
        )
        red_error_file = find_filter_file(
            galaxy, red_filter, settings, config_dir=config_dir, product="error"
        )
        narrow_error_file = find_filter_file(
            galaxy, narrow_filter, settings, config_dir=config_dir, product="error"
        )
    except FileNotFoundError:
        if require_errors:
            raise
        LOGGER.warning("%s: error images missing; error products will not be written", galaxy)

    return ImageSet(
        galaxy=galaxy,
        blue_filter=blue_filter,
        red_filter=red_filter,
        narrow_filter=narrow_filter,
        blue_file=blue_file,
        red_file=red_file,
        narrow_file=narrow_file,
        blue_error_file=blue_error_file,
        red_error_file=red_error_file,
        narrow_error_file=narrow_error_file,
    )


def convert_hst_count_rate_to_flux_density(
    hdu: fits.PrimaryHDU,
    photflam: float | None = None,
) -> fits.PrimaryHDU:
    """Convert an HST count-rate image to ``1e-20 erg s-1 cm-2 A-1 pixel-1``.

    HST drizzled science images are expected to be in count-rate units. The
    ``PHOTFLAM`` header keyword converts them to flux density.
    """

    if photflam is None and "PHOTFLAM" not in hdu.header:
        raise KeyError("Input HDU is missing PHOTFLAM")
    if photflam is None:
        photflam = float(hdu.header["PHOTFLAM"])

    out = hdu.copy()
    out.data = np.asarray(out.data, dtype=np.float32) * photflam * 1e20
    out.header["BUNIT"] = ("1e-20 erg/s/cm2/A/pixel", "PHOTFLAM-scaled flux density")
    return out


def convert_inverse_variance_to_error(hdu: fits.PrimaryHDU) -> fits.PrimaryHDU:
    """Convert an inverse-variance weight image to a 1-sigma error image."""

    data = np.asarray(hdu.data, dtype=np.float32)
    err = np.full(data.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(data) & (data > 0)
    err[valid] = np.sqrt(1.0 / data[valid])
    out = fits.PrimaryHDU(err, header=hdu.header.copy())
    out.header["BUNIT"] = ("count-rate error", "Converted from inverse variance")
    return out


def replace_zeros_with_nan(hdu: fits.PrimaryHDU) -> fits.PrimaryHDU:
    """Return a copy of ``hdu`` with exact zero pixels changed to NaN."""

    out = hdu.copy()
    data = np.asarray(out.data, dtype=np.float32).copy()
    data[data == 0] = np.nan
    out.data = data
    return out


def _valid_data_bounds(data: np.ndarray) -> tuple[slice, slice]:
    """Return y/x slices trimming all-NaN padding from image edges."""

    finite = np.isfinite(data)
    valid_y = np.where(np.any(finite, axis=1))[0]
    valid_x = np.where(np.any(finite, axis=0))[0]
    if len(valid_y) == 0 or len(valid_x) == 0:
        return slice(0, data.shape[0]), slice(0, data.shape[1])
    return (
        slice(int(valid_y[0]), int(valid_y[-1]) + 1),
        slice(int(valid_x[0]), int(valid_x[-1]) + 1),
    )


def crop_hdu(hdu: fits.PrimaryHDU, y_slice: slice, x_slice: slice) -> fits.PrimaryHDU:
    """Crop an image HDU and update CRPIX to keep the WCS aligned."""

    out = hdu.copy()
    out.data = np.asarray(out.data, dtype=np.float32)[y_slice, x_slice]
    if "CRPIX1" in out.header:
        out.header["CRPIX1"] -= x_slice.start or 0
    if "CRPIX2" in out.header:
        out.header["CRPIX2"] -= y_slice.start or 0
    return out


def preprocess_hst_data(
    narrow_hdu: fits.PrimaryHDU,
    blue_hdu: fits.PrimaryHDU,
    red_hdu: fits.PrimaryHDU,
    error_hdus: Sequence[fits.PrimaryHDU] | None = None,
) -> tuple[fits.PrimaryHDU, fits.PrimaryHDU, fits.PrimaryHDU, list[fits.PrimaryHDU]]:
    """Replace zero science pixels with NaN and remove narrowband NaN padding.

    The old pipeline crops the narrowband image after converting zero-valued
    science pixels to NaN. Here the same crop is applied to all science and
    error HDUs, which keeps the images on the same grid without adding a
    reprojection step to this plain first-pass workflow.
    """

    narrow_hdu = replace_zeros_with_nan(narrow_hdu)
    blue_hdu = replace_zeros_with_nan(blue_hdu)
    red_hdu = replace_zeros_with_nan(red_hdu)

    y_slice, x_slice = _valid_data_bounds(np.asarray(narrow_hdu.data))
    cropped_errors = [crop_hdu(err_hdu, y_slice, x_slice) for err_hdu in (error_hdus or [])]
    return (
        crop_hdu(narrow_hdu, y_slice, x_slice),
        crop_hdu(blue_hdu, y_slice, x_slice),
        crop_hdu(red_hdu, y_slice, x_slice),
        cropped_errors,
    )


def _ccm89_a_lambda_over_av(wavelength_angstrom: float, r_v: float = 3.1) -> float:
    """Return A(lambda)/A(V) for the CCM89 optical/NIR extinction law."""

    wavelength_micron = wavelength_angstrom / 10000.0
    x = 1.0 / wavelength_micron
    if 0.3 <= x < 1.1:
        a = 0.574 * x**1.61
        b = -0.527 * x**1.61
    elif 1.1 <= x <= 3.3:
        y = x - 1.82
        a = (
            1
            + 0.17699 * y
            - 0.50447 * y**2
            - 0.02427 * y**3
            + 0.72085 * y**4
            + 0.01979 * y**5
            - 0.77530 * y**6
            + 0.32999 * y**7
        )
        b = (
            1.41338 * y
            + 2.28305 * y**2
            + 1.07233 * y**3
            - 5.38434 * y**4
            - 0.62251 * y**5
            + 5.30260 * y**6
            - 2.09002 * y**7
        )
    else:
        raise ValueError(f"CCM89 helper only supports 0.3 <= 1/lambda <= 3.3; got {x:.3f}")
    return float(a + b / r_v)


def extinction_correction_factor(
    wavelength_angstrom: float,
    ebv: float,
    r_v: float = 3.1,
) -> float:
    """Return the multiplicative foreground-extinction correction factor."""

    av = ebv * r_v
    a_lambda = av * _ccm89_a_lambda_over_av(wavelength_angstrom, r_v=r_v)
    return float(10 ** (0.4 * a_lambda))


def apply_scalar_factor(
    hdu: fits.PrimaryHDU, factor: float, bunit: str | None = None
) -> fits.PrimaryHDU:
    """Return a copy of an HDU with data multiplied by a scalar factor."""

    out = hdu.copy()
    out.data = np.asarray(out.data, dtype=np.float32) * factor
    if bunit is not None:
        out.header["BUNIT"] = bunit
    return out


def convert_flux_density_to_flux(
    hdu: fits.PrimaryHDU,
    narrowband_width: float,
) -> fits.PrimaryHDU:
    """Convert flux density to integrated flux using the narrowband width."""

    out = apply_scalar_factor(hdu, narrowband_width, "1e-20 erg/s/cm2/pixel")
    out.header["PHYSUNIT"] = ("erg/s/cm2/pixel", "Data are scaled by 1e-20")
    out.header["CSWIDTH"] = (float(narrowband_width), "Narrowband width used for flux conversion")
    return out


def get_narrowband_width(
    narrow_hdu: fits.PrimaryHDU, narrow_filter: str, settings: Mapping[str, Any]
) -> float:
    """Get the width used to convert flux density to integrated flux."""

    widths = settings.get("narrowband_widths", {}) or {}
    if narrow_filter in widths:
        return float(widths[narrow_filter])

    header_key = str(settings.get("narrowband_width_header", "PHOTBW"))
    if header_key in narrow_hdu.header:
        return float(narrow_hdu.header[header_key])

    raise KeyError(
        f"No narrowband width found for {narrow_filter}. Add "
        f"narrowband_widths.{narrow_filter} in config/galaxies.yaml or set "
        f"narrowband_width_header to a FITS header keyword."
    )


def _normalize_galaxy_name(name: Any) -> str:
    """Normalize galaxy names for robust sample-table matching."""

    text = str(name).strip().lower()
    match = re.fullmatch(r"([a-z]+)0*([0-9]+)([a-z]*)", text)
    if match:
        prefix, number, suffix = match.groups()
        return f"{prefix}{int(number)}{suffix}"
    return text


def foreground_ebv(
    galaxy: str,
    galaxy_settings: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float | None:
    """Read foreground E(B-V) for one galaxy from the configured sample table."""

    ext_cfg = params.get("foreground_extinction", {}) or {}
    if not bool(ext_cfg.get("enabled", False)):
        return None

    if "ebv" in galaxy_settings:
        return float(galaxy_settings["ebv"])

    sample_table_path = str(ext_cfg.get("sample_table_path", "") or "")
    if not sample_table_path:
        raise ValueError("foreground_extinction.enabled is true but sample_table_path is empty")

    table = Table.read(Path(sample_table_path).expanduser())
    name_column = str(ext_cfg.get("galaxy_column", "name"))
    ebv_column = str(ext_cfg.get("ebv_column", "mwext_sf11"))
    sample_name = str(galaxy_settings.get("sample_name", galaxy))
    wanted = _normalize_galaxy_name(sample_name)

    names = [_normalize_galaxy_name(value) for value in table[name_column]]
    matches = [idx for idx, name in enumerate(names) if name == wanted]
    if not matches:
        raise ValueError(f"{galaxy}: no sample-table row found for sample_name={sample_name!r}")
    return float(table[ebv_column][matches[0]])


def linear_continuum_subtract(
    narrow_hdu: fits.PrimaryHDU,
    blue_hdu: fits.PrimaryHDU,
    red_hdu: fits.PrimaryHDU,
    blue_filter: str,
    red_filter: str,
    narrow_filter: str,
    narrow_error_hdu: fits.PrimaryHDU | None = None,
    blue_error_hdu: fits.PrimaryHDU | None = None,
    red_error_hdu: fits.PrimaryHDU | None = None,
) -> tuple[
    fits.PrimaryHDU,
    fits.PrimaryHDU,
    fits.PrimaryHDU | None,
    fits.PrimaryHDU | None,
    float,
    float,
]:
    """Subtract a linear continuum estimate and optionally propagate errors."""

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

    contsub_error_hdu = None
    continuum_error_hdu = None
    if narrow_error_hdu is not None and blue_error_hdu is not None and red_error_hdu is not None:
        error_shapes = {
            narrow_error_hdu.data.shape,
            blue_error_hdu.data.shape,
            red_error_hdu.data.shape,
            narrow_hdu.data.shape,
        }
        if len(error_shapes) != 1:
            raise ValueError(f"Input error image shapes differ. Shapes: {sorted(error_shapes)}")

        continuum_error_data = np.sqrt(
            (np.asarray(blue_error_hdu.data, dtype=np.float32) * weight_blue) ** 2
            + (np.asarray(red_error_hdu.data, dtype=np.float32) * weight_red) ** 2
        )
        contsub_error_data = np.sqrt(
            np.asarray(narrow_error_hdu.data, dtype=np.float32) ** 2 + continuum_error_data**2
        )
        continuum_error_hdu = narrow_hdu.copy()
        contsub_error_hdu = narrow_hdu.copy()
        continuum_error_hdu.data = np.asarray(continuum_error_data, dtype=np.float32)
        contsub_error_hdu.data = np.asarray(contsub_error_data, dtype=np.float32)

        for hdu, product in (
            (continuum_error_hdu, "CONTINUUM_ERR"),
            (contsub_error_hdu, "CONTSUB_ERR"),
        ):
            hdu.header["CONTSUB"] = (True, "Produced by hstha_contsubpipe_extended")
            hdu.header["CSPROD"] = (product, "Continuum-subtraction error product")
            hdu.header["CSSPACE"] = ("linear", "Continuum interpolation space")
            hdu.header["CSWBLUE"] = (weight_blue, "Blue continuum weight")
            hdu.header["CSWRED"] = (weight_red, "Red continuum weight")
            hdu.header["BUNIT"] = ("1e-20 erg/s/cm2/A/pixel", "PHOTFLAM-scaled error")

    return (
        contsub_hdu,
        continuum_hdu,
        contsub_error_hdu,
        continuum_error_hdu,
        weight_blue,
        weight_red,
    )


def _output_path(
    template_key: str,
    image_set: ImageSet,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    product: str | None = None,
) -> Path:
    """Resolve one output path from ``files.yaml``."""

    product_name = product or template_key.split(".")[-1]
    return Path(
        _resolve_template_path(
            template_key,
            {
                "galaxy": image_set.galaxy,
                "narrow_filter": image_set.narrow_filter,
                "blue_filter": image_set.blue_filter,
                "red_filter": image_set.red_filter,
                "contsub_space": "linear",
                "product": product_name,
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
    _, params, _ = get_configs(config_dir=config_dir)
    contsub_file = _output_path("outputs.contsub", image_set, config_dir=config_dir)
    continuum_file = _output_path("outputs.continuum", image_set, config_dir=config_dir)
    contsub_error_file = _output_path("outputs.contsub_error", image_set, config_dir=config_dir)
    continuum_error_file = _output_path(
        "outputs.continuum_error", image_set, config_dir=config_dir
    )
    halpha_file = _output_path("outputs.halpha", image_set, config_dir=config_dir)
    halpha_error_file = _output_path("outputs.halpha_error", image_set, config_dir=config_dir)

    if dry_run:
        return ContsubResult(
            galaxy=galaxy,
            status="planned",
            narrow_filter=image_set.narrow_filter,
            contsub_file=contsub_file,
            continuum_file=continuum_file,
            contsub_error_file=contsub_error_file,
            continuum_error_file=continuum_error_file,
            halpha_file=halpha_file,
            halpha_error_file=halpha_error_file,
            blue_file=image_set.blue_file,
            red_file=image_set.red_file,
            narrow_file=image_set.narrow_file,
            blue_error_file=image_set.blue_error_file,
            red_error_file=image_set.red_error_file,
            narrow_error_file=image_set.narrow_error_file,
        )

    output_files = [
        contsub_file,
        continuum_file,
        contsub_error_file,
        continuum_error_file,
        halpha_file,
        halpha_error_file,
    ]
    if any(path.exists() for path in output_files) and not overwrite:
        return ContsubResult(
            galaxy=galaxy,
            status="skipped",
            narrow_filter=image_set.narrow_filter,
            contsub_file=contsub_file,
            continuum_file=continuum_file,
            contsub_error_file=contsub_error_file,
            continuum_error_file=continuum_error_file,
            halpha_file=halpha_file,
            halpha_error_file=halpha_error_file,
            blue_file=image_set.blue_file,
            red_file=image_set.red_file,
            narrow_file=image_set.narrow_file,
            blue_error_file=image_set.blue_error_file,
            red_error_file=image_set.red_error_file,
            narrow_error_file=image_set.narrow_error_file,
            message="output exists; use overwrite=True to replace it",
        )

    hdu_index = int(settings.get("hdu_index", 0))
    with ExitStack() as stack:
        narrow_hdul = stack.enter_context(fits.open(image_set.narrow_file, memmap=True))
        blue_hdul = stack.enter_context(fits.open(image_set.blue_file, memmap=True))
        red_hdul = stack.enter_context(fits.open(image_set.red_file, memmap=True))

        narrow_hdu = narrow_hdul[hdu_index].copy()
        blue_hdu = blue_hdul[hdu_index].copy()
        red_hdu = red_hdul[hdu_index].copy()

        narrow_error_hdu = blue_error_hdu = red_error_hdu = None
        if (
            image_set.narrow_error_file is not None
            and image_set.blue_error_file is not None
            and image_set.red_error_file is not None
        ):
            narrow_error_hdul = stack.enter_context(
                fits.open(image_set.narrow_error_file, memmap=True)
            )
            blue_error_hdul = stack.enter_context(
                fits.open(image_set.blue_error_file, memmap=True)
            )
            red_error_hdul = stack.enter_context(fits.open(image_set.red_error_file, memmap=True))
            narrow_error_hdu = convert_inverse_variance_to_error(narrow_error_hdul[hdu_index])
            blue_error_hdu = convert_inverse_variance_to_error(blue_error_hdul[hdu_index])
            red_error_hdu = convert_inverse_variance_to_error(red_error_hdul[hdu_index])

        error_inputs = [
            hdu for hdu in (narrow_error_hdu, blue_error_hdu, red_error_hdu) if hdu is not None
        ]
        narrow_hdu, blue_hdu, red_hdu, error_inputs = preprocess_hst_data(
            narrow_hdu=narrow_hdu,
            blue_hdu=blue_hdu,
            red_hdu=red_hdu,
            error_hdus=error_inputs,
        )
        if error_inputs:
            narrow_error_hdu, blue_error_hdu, red_error_hdu = error_inputs

        narrow_hdu = convert_hst_count_rate_to_flux_density(narrow_hdu)
        blue_hdu = convert_hst_count_rate_to_flux_density(blue_hdu)
        red_hdu = convert_hst_count_rate_to_flux_density(red_hdu)

        if (
            narrow_error_hdu is not None
            and blue_error_hdu is not None
            and red_error_hdu is not None
        ):
            narrow_error_hdu = convert_hst_count_rate_to_flux_density(
                narrow_error_hdu,
                photflam=float(narrow_hdu.header["PHOTFLAM"]),
            )
            blue_error_hdu = convert_hst_count_rate_to_flux_density(
                blue_error_hdu,
                photflam=float(blue_hdu.header["PHOTFLAM"]),
            )
            red_error_hdu = convert_hst_count_rate_to_flux_density(
                red_error_hdu,
                photflam=float(red_hdu.header["PHOTFLAM"]),
            )

        ebv = foreground_ebv(galaxy, settings, params)
        r_v = float((params.get("foreground_extinction", {}) or {}).get("r_v", 3.1))
        if ebv is not None:
            for band_hdu, err_hdu in (
                (narrow_hdu, narrow_error_hdu),
                (blue_hdu, blue_error_hdu),
                (red_hdu, red_error_hdu),
            ):
                factor = extinction_correction_factor(
                    float(band_hdu.header["PHOTPLAM"]),
                    ebv=ebv,
                    r_v=r_v,
                )
                band_hdu.data = np.asarray(band_hdu.data, dtype=np.float32) * factor
                if err_hdu is not None:
                    err_hdu.data = np.asarray(err_hdu.data, dtype=np.float32) * factor
                band_hdu.header["CSEBV"] = (float(ebv), "Foreground E(B-V)")
                band_hdu.header["CSEXTCOR"] = (factor, "Foreground extinction correction")

        (
            contsub_hdu,
            continuum_hdu,
            contsub_error_hdu,
            continuum_error_hdu,
            weight_blue,
            weight_red,
        ) = linear_continuum_subtract(
            narrow_hdu=narrow_hdu,
            blue_hdu=blue_hdu,
            red_hdu=red_hdu,
            blue_filter=image_set.blue_filter,
            red_filter=image_set.red_filter,
            narrow_filter=image_set.narrow_filter,
            narrow_error_hdu=narrow_error_hdu,
            blue_error_hdu=blue_error_hdu,
            red_error_hdu=red_error_hdu,
        )

        narrowband_width = get_narrowband_width(
            narrow_hdu=narrow_hdu,
            narrow_filter=image_set.narrow_filter,
            settings=settings,
        )
        products: dict[Path, fits.PrimaryHDU | None] = {
            contsub_file: convert_flux_density_to_flux(contsub_hdu, narrowband_width),
            continuum_file: convert_flux_density_to_flux(continuum_hdu, narrowband_width),
            contsub_error_file: (
                convert_flux_density_to_flux(contsub_error_hdu, narrowband_width)
                if contsub_error_hdu is not None
                else None
            ),
            continuum_error_file: (
                convert_flux_density_to_flux(continuum_error_hdu, narrowband_width)
                if continuum_error_hdu is not None
                else None
            ),
        }

        nii_to_halpha = float(settings.get("nii_to_halpha", 0.0))
        halpha_hdu = apply_scalar_factor(
            products[contsub_file],
            1.0 / (1.0 + nii_to_halpha),
            "1e-20 erg/s/cm2/pixel",
        )
        halpha_hdu.header["CSNIIRAT"] = (nii_to_halpha, "Assumed total [NII]/H-alpha")
        halpha_hdu.header["CSPROD"] = ("HALPHA_NII_CORR", "H-alpha after fixed [NII] correction")
        products[halpha_file] = halpha_hdu

        if products[contsub_error_file] is not None:
            halpha_error_hdu = apply_scalar_factor(
                products[contsub_error_file],
                1.0 / (1.0 + nii_to_halpha),
                "1e-20 erg/s/cm2/pixel",
            )
            halpha_error_hdu.header["CSNIIRAT"] = (
                nii_to_halpha,
                "Assumed total [NII]/H-alpha",
            )
            halpha_error_hdu.header["CSPROD"] = (
                "HALPHA_NII_CORR_ERR",
                "H-alpha error after fixed [NII] correction",
            )
            products[halpha_error_file] = halpha_error_hdu
        else:
            products[halpha_error_file] = None

        if not bool(settings.get("write_continuum", True)):
            products[continuum_file] = None
            products[continuum_error_file] = None
            continuum_file = None
            continuum_error_file = None

        for path, hdu in products.items():
            if hdu is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            hdu.writeto(path, overwrite=overwrite)

    return ContsubResult(
        galaxy=galaxy,
        status="written",
        narrow_filter=image_set.narrow_filter,
        contsub_file=contsub_file,
        continuum_file=continuum_file,
        contsub_error_file=contsub_error_file,
        continuum_error_file=continuum_error_file,
        halpha_file=halpha_file,
        halpha_error_file=halpha_error_file,
        blue_file=image_set.blue_file,
        red_file=image_set.red_file,
        narrow_file=image_set.narrow_file,
        blue_error_file=image_set.blue_error_file,
        red_error_file=image_set.red_error_file,
        narrow_error_file=image_set.narrow_error_file,
        weight_blue=weight_blue,
        weight_red=weight_red,
        extinction_ebv=ebv,
        nii_to_halpha=nii_to_halpha,
        narrowband_width=narrowband_width,
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
        "contsub_error_file",
        "continuum_error_file",
        "halpha_file",
        "halpha_error_file",
        "blue_error_file",
        "red_error_file",
        "narrow_error_file",
        "weight_blue",
        "weight_red",
        "extinction_ebv",
        "nii_to_halpha",
        "narrowband_width",
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
