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
import math
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

try:
    from synphot import SpectralElement, units as synphot_units
except ImportError:  # pragma: no cover - exercised only when synphot is installed.
    SpectralElement = None
    synphot_units = None


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
    bandpass_source: str = ""
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


def pixel_area_arcsec2(header: fits.Header) -> float:
    """Return the projected pixel area in square arcseconds from FITS WCS keywords."""

    if all(key in header for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        area_deg2 = abs(
            float(header["CD1_1"]) * float(header["CD2_2"])
            - float(header["CD1_2"]) * float(header["CD2_1"])
        )
    elif all(key in header for key in ("CD1_1", "CD2_2")):
        area_deg2 = abs(float(header["CD1_1"]) * float(header["CD2_2"]))
    elif all(key in header for key in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2")):
        pc_det = float(header["PC1_1"]) * float(header["PC2_2"]) - float(header["PC1_2"]) * float(
            header["PC2_1"]
        )
        area_deg2 = abs(pc_det * float(header["CDELT1"]) * float(header["CDELT2"]))
    elif all(key in header for key in ("CDELT1", "CDELT2")):
        area_deg2 = abs(float(header["CDELT1"]) * float(header["CDELT2"]))
    elif "CD1_1" in header:
        area_deg2 = abs(float(header["CD1_1"])) ** 2
    elif "PC1_1" in header:
        area_deg2 = abs(float(header["PC1_1"])) ** 2
    else:
        raise KeyError("Cannot determine pixel area from CD, PC+CDELT, or CDELT WCS keywords")

    return area_deg2 * 3600.0**2


def convert_perpix_to_perarcsec(hdu: fits.PrimaryHDU) -> fits.PrimaryHDU:
    """Convert integrated flux per pixel to surface brightness per arcsec squared."""

    out = hdu.copy()
    pix_area = pixel_area_arcsec2(out.header)
    out.data = np.asarray(out.data, dtype=np.float32) / pix_area
    out.header["BUNIT"] = ("1e-20 erg/s/cm2/arcsec2", "Surface brightness")
    out.header["PHYSUNIT"] = ("erg/s/cm2/arcsec2", "Data are scaled by 1e-20")
    out.header["CSPIXA2"] = (float(pix_area), "Pixel area in arcsec2")
    return out


def convert_output_units(hdu: fits.PrimaryHDU, settings: Mapping[str, Any]) -> fits.PrimaryHDU:
    """Convert final products to the configured output unit."""

    output_unit = str(settings.get("output_unit", "erg/s/cm2/arcsec2")).lower()
    aliases = {
        "erg/s/cm2/arcsec2",
        "ergcm2s/arcsec^2",
        "erg/s/cm^2/arcsec^2",
        "surface_brightness",
    }
    if output_unit in aliases:
        return convert_perpix_to_perarcsec(hdu)
    if output_unit in {"erg/s/cm2/pixel", "erg/s/cm^2/pixel", "per_pixel"}:
        return hdu
    raise ValueError(
        f"Unsupported output_unit={output_unit!r}; use 'erg/s/cm2/arcsec2' or 'erg/s/cm2/pixel'"
    )


def get_narrowband_width(
    narrow_hdu: fits.PrimaryHDU,
    narrow_filter: str,
    settings: Mapping[str, Any],
    bandpass_width: float | None = None,
) -> float:
    """Get the width used to convert flux density to integrated flux."""

    widths = settings.get("narrowband_widths", {}) or {}
    if narrow_filter in widths:
        return float(widths[narrow_filter])

    if bandpass_width is not None:
        return float(bandpass_width)

    header_key = str(settings.get("narrowband_width_header", "PHOTBW"))
    if header_key in narrow_hdu.header:
        return float(narrow_hdu.header[header_key])

    raise KeyError(
        f"No narrowband width found for {narrow_filter}. Add "
        f"narrowband_widths.{narrow_filter} in config/galaxies.yaml, configure "
        f"external bandpasses, or set "
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


def _normalize_instrument(instrument: Any) -> str:
    """Normalize HST instrument labels to the bandpass-table convention."""

    text = str(instrument).strip().upper()
    if text in {"WFC3", "WFC3_UVIS", "UVIS1"}:
        return "UVIS"
    if text == "UVIS2":
        return "UVIS2"
    if text in {"ACS_WFC", "WFC"}:
        return "ACS"
    return text


def infer_bandpass_instrument(hdu: fits.PrimaryHDU, filename: Path) -> str:
    """Infer the external bandpass instrument label for one HST image."""

    path_text = str(filename).lower()
    if "_acs_" in path_text or "/acs" in path_text:
        return "ACS"
    if "_uvis_" in path_text or "/uvis" in path_text:
        return "UVIS"

    instrume = str(hdu.header.get("INSTRUME", "")).upper()
    detector = str(hdu.header.get("DETECTOR", "")).upper()
    if instrume == "ACS":
        return "ACS"
    if instrume == "WFC3" and detector == "UVIS":
        return "UVIS"
    if detector:
        return _normalize_instrument(detector)
    return _normalize_instrument(instrume)


def _bandpass_key_from_dat_file(path: Path) -> tuple[str, str]:
    """Return ``(instrument, filter)`` from an HST throughput filename."""

    name = path.name.split(".dat")[0]
    name = name.replace("HST_", "").replace(".F", "_F")
    name = name.replace("WFC_", "")
    name = name.replace("WFC3_", "")
    name = name.replace("UVIS1", "UVIS")
    name = name.replace("UVIS2", "UVIS")
    instrument, filter_name = name.split("_", 1)
    return _normalize_instrument(instrument), filter_name.upper()


def _bandpass_from_curve(path: Path) -> dict[str, float | str]:
    """Read a throughput curve and return the same bandpass keys as the old pipeline."""

    if SpectralElement is not None and synphot_units is not None:
        area = 45238.93416 * synphot_units.AREA
        bp = SpectralElement.from_file(path)
        return {
            "equivwidth": float(bp.equivwidth().value),
            "integrate": float(bp.integrate().value),
            "rmswidth": float(bp.rmswidth().value),
            "photbw": float(bp.photbw().value),
            "fwhm": float(bp.fwhm().value),
            "rectwidth": float(bp.rectwidth().value),
            "pivot": float(bp.pivot().value),
            "unit_response": float(bp.unit_response(area).value),
            "source": str(path),
            "source_kind": "filter_curve",
        }

    arr = np.loadtxt(path)
    wave = np.asarray(arr[:, 0], dtype=float)
    throughput = np.asarray(arr[:, 1], dtype=float)
    valid = np.isfinite(wave) & np.isfinite(throughput) & (throughput > 0)
    wave = wave[valid]
    throughput = throughput[valid]
    if wave.size < 2:
        raise ValueError(f"Not enough valid throughput samples in {path}")

    integral_t_lambda = np.trapz(throughput * wave, wave)
    integral_t_over_lambda = np.trapz(throughput / wave, wave)
    pivot = math.sqrt(integral_t_lambda / integral_t_over_lambda)
    integrate = np.trapz(throughput, wave)
    rectwidth = integrate / np.nanmax(throughput)

    mean_wave = np.trapz(wave * throughput, wave) / integrate
    rmswidth = math.sqrt(np.trapz(((wave - mean_wave) ** 2) * throughput, wave) / integrate)
    half_max = np.nanmax(throughput) / 2.0
    above_half = wave[throughput >= half_max]
    fwhm = float(above_half[-1] - above_half[0]) if above_half.size > 1 else float("nan")

    return {
        "equivwidth": float(integrate),
        "integrate": float(integrate),
        "rmswidth": float(rmswidth),
        "photbw": float(rectwidth),
        "fwhm": fwhm,
        "rectwidth": float(rectwidth),
        "pivot": float(pivot),
        "unit_response": float("nan"),
        "source": str(path),
        "source_kind": "filter_curve_numeric",
    }


def load_bandpass_catalog(params: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Load all HST filter curves, plus optional precomputed table columns."""

    bandpass_cfg = params.get("bandpass", {}) or {}
    root_value = str(bandpass_cfg.get("filter_root", "") or "")
    if not root_value:
        return {}
    root = Path(root_value).expanduser()

    catalog: dict[tuple[str, str], dict[str, Any]] = {}
    for curve_path in sorted(root.glob("*.dat")):
        try:
            instrument, filter_name = _bandpass_key_from_dat_file(curve_path)
            catalog[(instrument, filter_name)] = _bandpass_from_curve(curve_path)
        except Exception as exc:
            LOGGER.warning("Could not read bandpass curve %s: %s", curve_path, exc)

    table_name = str(bandpass_cfg.get("table_file", "filter_table.fits"))
    table_path = root / table_name
    if table_path.exists():
        table = Table.read(table_path)
        for row in table:
            first = str(row[0]).strip().upper()
            second = str(row[1]).strip().upper()
            if first in {"ACS", "UVIS", "UVIS1", "UVIS2", "WFC3"}:
                instrument, filter_name = _normalize_instrument(first), second
            else:
                filter_name, instrument = first, _normalize_instrument(second)
            entry = catalog.setdefault((instrument, filter_name), {})
            entry["table_photplam"] = float(row["photplam"])
            entry["table_photbw"] = float(row["photbw"])
            if "photflam" in table.colnames:
                scale = float(bandpass_cfg.get("table_photflam_scale", 1e-19))
                entry["table_photflam"] = float(row["photflam"]) * scale
            entry["table_source"] = str(table_path)

    return catalog


def _bandpass_entry_for(
    instrument: str,
    filter_key: str,
    catalog: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any] | None]:
    """Return the best external entry for an instrument/filter pair."""

    candidates = [instrument]
    if instrument == "UVIS2":
        candidates.append("UVIS")
    if instrument == "UVIS":
        candidates.append("UVIS2")
    for candidate in candidates:
        entry = catalog.get((candidate, filter_key))
        if entry is not None:
            return candidate, entry
    return instrument, None


def _resolve_bandpass_value(
    hdu: fits.PrimaryHDU,
    filename: Path,
    entry: Mapping[str, Any] | None,
    quantity: str,
    source_setting: str,
    header_key: str,
    external_keys: Sequence[str],
    allow_header: bool,
) -> tuple[float, str]:
    """Resolve one bandpass quantity from configured source preferences."""

    source = source_setting.lower()
    if source in {"header", "fits"}:
        if header_key not in hdu.header:
            raise KeyError(f"{filename} is missing required header keyword {header_key}")
        return float(hdu.header[header_key]), f"header:{header_key}"

    source_key_map = {
        "filter": external_keys,
        "filter_curve": external_keys,
        "external": external_keys,
        "table": [f"table_{quantity}", *external_keys],
        "filter_table": [f"table_{quantity}", *external_keys],
        "filter_rectwidth": ["rectwidth", "table_photbw"],
        "rectwidth": ["rectwidth", "table_photbw"],
        "filter_photbw": ["photbw", "table_photbw"],
        "photbw": ["photbw", "table_photbw"],
        "unit_response": ["unit_response"],
        "filter_unit_response": ["unit_response"],
    }
    keys = source_key_map.get(source, external_keys)
    if entry is not None:
        for key in keys:
            if key in entry and np.isfinite(float(entry[key])):
                src = str(entry.get("source") or entry.get("table_source") or "external")
                if key.startswith("table_"):
                    src = str(entry.get("table_source") or src)
                return float(entry[key]), f"{source}:{Path(src).name}"

    if allow_header and header_key in hdu.header:
        return float(hdu.header[header_key]), f"header:{header_key}"

    raise KeyError(
        f"Could not resolve {quantity} for {filename}. Tried source={source_setting!r}; "
        f"set bandpass.{quantity}_source or enable fallback_to_header."
    )


def bandpass_for_image(
    hdu: fits.PrimaryHDU,
    filename: Path,
    filter_name: str,
    catalog: Mapping[tuple[str, str], Mapping[str, Any]],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve pivot, width, and PHOTFLAM from configured sources."""

    bandpass_cfg = params.get("bandpass", {}) or {}
    allow_header = bool(bandpass_cfg.get("fallback_to_header", True))
    filter_key = filter_name.upper()
    instrument = infer_bandpass_instrument(hdu, filename)
    matched_instrument, entry = _bandpass_entry_for(instrument, filter_key, catalog)

    header_pivot_key = str(bandpass_cfg.get("header_pivot_key", "PHOTPLAM"))
    header_width_key = str(bandpass_cfg.get("header_width_key", "PHOTBW"))
    header_photflam_key = str(bandpass_cfg.get("header_photflam_key", "PHOTFLAM"))

    pivot, pivot_source = _resolve_bandpass_value(
        hdu=hdu,
        filename=filename,
        entry=entry,
        quantity="photplam",
        source_setting=str(bandpass_cfg.get("pivot_source", "filter")),
        header_key=header_pivot_key,
        external_keys=["pivot", "table_photplam"],
        allow_header=allow_header,
    )
    width, width_source = _resolve_bandpass_value(
        hdu=hdu,
        filename=filename,
        entry=entry,
        quantity="photbw",
        source_setting=str(bandpass_cfg.get("width_source", "filter_rectwidth")),
        header_key=header_width_key,
        external_keys=["rectwidth", "photbw", "table_photbw"],
        allow_header=allow_header,
    )
    photflam, photflam_source = _resolve_bandpass_value(
        hdu=hdu,
        filename=filename,
        entry=entry,
        quantity="photflam",
        source_setting=str(bandpass_cfg.get("photflam_source", "header")),
        header_key=header_photflam_key,
        external_keys=["table_photflam", "unit_response"],
        allow_header=allow_header,
    )

    return {
        "pivot": pivot,
        "width": width,
        "photflam": photflam,
        "instrument": matched_instrument,
        "source": pivot_source,
        "pivot_source": pivot_source,
        "width_source": width_source,
        "photflam_source": photflam_source,
    }


def linear_continuum_subtract(
    narrow_hdu: fits.PrimaryHDU,
    blue_hdu: fits.PrimaryHDU,
    red_hdu: fits.PrimaryHDU,
    blue_filter: str,
    red_filter: str,
    narrow_filter: str,
    narrow_pivot: float | None = None,
    blue_pivot: float | None = None,
    red_pivot: float | None = None,
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

    lam_narrow = float(narrow_pivot if narrow_pivot is not None else narrow_hdu.header["PHOTPLAM"])
    lam_blue = float(blue_pivot if blue_pivot is not None else blue_hdu.header["PHOTPLAM"])
    lam_red = float(red_pivot if red_pivot is not None else red_hdu.header["PHOTPLAM"])
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
    bandpass_catalog = load_bandpass_catalog(params)
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

        narrow_bandpass = bandpass_for_image(
            hdu=narrow_hdu,
            filename=image_set.narrow_file,
            filter_name=image_set.narrow_filter,
            catalog=bandpass_catalog,
            params=params,
        )
        blue_bandpass = bandpass_for_image(
            hdu=blue_hdu,
            filename=image_set.blue_file,
            filter_name=image_set.blue_filter,
            catalog=bandpass_catalog,
            params=params,
        )
        red_bandpass = bandpass_for_image(
            hdu=red_hdu,
            filename=image_set.red_file,
            filter_name=image_set.red_filter,
            catalog=bandpass_catalog,
            params=params,
        )
        for hdu, bandpass in (
            (narrow_hdu, narrow_bandpass),
            (blue_hdu, blue_bandpass),
            (red_hdu, red_bandpass),
        ):
            source_label = str(bandpass["source"])
            hdu.header["CSBPSRC"] = (source_label[:48], "BP source")
            hdu.header["CSBPINS"] = (str(bandpass["instrument"]), "Bandpass instrument")
            hdu.header["CSBPWAV"] = (float(bandpass["pivot"]), "Bandpass pivot wavelength")
            hdu.header["CSBPWID"] = (float(bandpass["width"]), "Bandpass width")
            hdu.header["CSBPFLAM"] = (float(bandpass["photflam"]), "PHOTFLAM used")
            hdu.header["CSWAVSRC"] = (str(bandpass["pivot_source"])[:48], "Pivot source")
            hdu.header["CSWIDSRC"] = (str(bandpass["width_source"])[:48], "Width source")
            hdu.header["CSFLSRC"] = (str(bandpass["photflam_source"])[:48], "PHOTFLAM source")

        narrow_hdu = convert_hst_count_rate_to_flux_density(
            narrow_hdu, photflam=float(narrow_bandpass["photflam"])
        )
        blue_hdu = convert_hst_count_rate_to_flux_density(
            blue_hdu, photflam=float(blue_bandpass["photflam"])
        )
        red_hdu = convert_hst_count_rate_to_flux_density(
            red_hdu, photflam=float(red_bandpass["photflam"])
        )

        if (
            narrow_error_hdu is not None
            and blue_error_hdu is not None
            and red_error_hdu is not None
        ):
            narrow_error_hdu = convert_hst_count_rate_to_flux_density(
                narrow_error_hdu,
                photflam=float(narrow_bandpass["photflam"]),
            )
            blue_error_hdu = convert_hst_count_rate_to_flux_density(
                blue_error_hdu,
                photflam=float(blue_bandpass["photflam"]),
            )
            red_error_hdu = convert_hst_count_rate_to_flux_density(
                red_error_hdu,
                photflam=float(red_bandpass["photflam"]),
            )

        ebv = foreground_ebv(galaxy, settings, params)
        r_v = float((params.get("foreground_extinction", {}) or {}).get("r_v", 3.1))
        if ebv is not None:
            for band_hdu, err_hdu in (
                (narrow_hdu, narrow_error_hdu),
                (blue_hdu, blue_error_hdu),
                (red_hdu, red_error_hdu),
            ):
                pivot = float(band_hdu.header["CSBPWAV"])
                factor = extinction_correction_factor(
                    pivot,
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
            narrow_pivot=float(narrow_bandpass["pivot"]),
            blue_pivot=float(blue_bandpass["pivot"]),
            red_pivot=float(red_bandpass["pivot"]),
            narrow_error_hdu=narrow_error_hdu,
            blue_error_hdu=blue_error_hdu,
            red_error_hdu=red_error_hdu,
        )

        narrowband_width = get_narrowband_width(
            narrow_hdu=narrow_hdu,
            narrow_filter=image_set.narrow_filter,
            settings=settings,
            bandpass_width=float(narrow_bandpass["width"]),
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

        products = {
            path: convert_output_units(hdu, settings) if hdu is not None else None
            for path, hdu in products.items()
        }

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
        bandpass_source=str(narrow_bandpass["source"]),
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
        "bandpass_source",
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
