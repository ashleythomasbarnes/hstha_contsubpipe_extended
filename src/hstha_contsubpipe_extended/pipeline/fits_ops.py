"""FITS image operations and unit conversions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_closing


def convert_hst_count_rate_to_flux_density(
    hdu: fits.PrimaryHDU,
    photflam: float | None = None,
) -> fits.PrimaryHDU:
    """Convert an HST count-rate image to ``1e-20 erg s-1 cm-2 A-1 pixel-1``."""

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


def valid_data_bounds(data: np.ndarray) -> tuple[slice, slice]:
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
    """Replace zero science pixels with NaN and remove narrowband NaN padding."""

    narrow_hdu = replace_zeros_with_nan(narrow_hdu)
    blue_hdu = replace_zeros_with_nan(blue_hdu)
    red_hdu = replace_zeros_with_nan(red_hdu)

    y_slice, x_slice = valid_data_bounds(np.asarray(narrow_hdu.data))
    cropped_errors = [crop_hdu(err_hdu, y_slice, x_slice) for err_hdu in (error_hdus or [])]
    return (
        crop_hdu(narrow_hdu, y_slice, x_slice),
        crop_hdu(blue_hdu, y_slice, x_slice),
        crop_hdu(red_hdu, y_slice, x_slice),
        cropped_errors,
    )


def _require_matching_shapes(hdus: Sequence[fits.PrimaryHDU], description: str) -> tuple[int, ...]:
    if any(hdu.data is None for hdu in hdus):
        raise ValueError(f"{description} HDUs must contain image data")
    shapes = {np.asarray(hdu.data).shape for hdu in hdus}
    if len(shapes) != 1:
        raise ValueError(f"{description} image shapes differ. Shapes: {sorted(shapes)}")
    return next(iter(shapes))


def _mask_hdu_to_coverage(
    hdu: fits.PrimaryHDU,
    coverage_mask: np.ndarray,
    closing_size: int,
    closing_iterations: int,
    retained_pixels: int,
) -> fits.PrimaryHDU:
    out = hdu.copy()
    data = np.asarray(out.data, dtype=np.float32).copy()
    data[~coverage_mask] = np.nan
    out.data = data
    out.header["CSCOVMSK"] = (True, "Matched to common filter coverage")
    out.header["CSCOVCLS"] = (closing_size, "Coverage mask closing kernel")
    out.header["CSCOVIT"] = (closing_iterations, "Coverage mask closing iterations")
    out.header["CSCOVPIX"] = (retained_pixels, "Pixels retained by coverage mask")
    return out


def match_spatial_coverage(
    blue_hdu: fits.PrimaryHDU,
    narrow_hdu: fits.PrimaryHDU,
    red_hdu: fits.PrimaryHDU,
    error_hdus: Sequence[fits.PrimaryHDU | None] | None = None,
    closing_size: int = 10,
    closing_iterations: int = 5,
) -> tuple[
    fits.PrimaryHDU, fits.PrimaryHDU, fits.PrimaryHDU, list[fits.PrimaryHDU | None], dict[str, Any]
]:
    """Mask science and error HDUs to the common finite footprint of all filters."""

    _require_matching_shapes((blue_hdu, narrow_hdu, red_hdu), "Science")
    if closing_size < 1:
        raise ValueError("coverage_mask.closing_size must be >= 1")
    if closing_iterations < 0:
        raise ValueError("coverage_mask.closing_iterations must be >= 0")

    blue_data = np.asarray(blue_hdu.data, dtype=np.float32)
    narrow_data = np.asarray(narrow_hdu.data, dtype=np.float32)
    red_data = np.asarray(red_hdu.data, dtype=np.float32)
    raw_mask = np.isfinite(blue_data) & np.isfinite(narrow_data) & np.isfinite(red_data)
    coverage_mask = raw_mask
    if closing_size > 1 and closing_iterations > 0:
        structure = np.ones((closing_size, closing_size), dtype=bool)
        coverage_mask = binary_closing(
            raw_mask,
            structure=structure,
            iterations=closing_iterations,
            border_value=1,
        )

    retained_pixels = int(np.count_nonzero(coverage_mask))
    record = {
        "raw_pixels": int(np.count_nonzero(raw_mask)),
        "retained_pixels": retained_pixels,
        "masked_pixels": int(coverage_mask.size - retained_pixels),
        "closing_size": int(closing_size),
        "closing_iterations": int(closing_iterations),
    }

    masked_errors: list[fits.PrimaryHDU | None] = []
    for err_hdu in error_hdus or ():
        if err_hdu is None:
            masked_errors.append(None)
            continue
        _require_matching_shapes((blue_hdu, err_hdu), "Science and error")
        masked_errors.append(
            _mask_hdu_to_coverage(
                err_hdu,
                coverage_mask,
                closing_size,
                closing_iterations,
                retained_pixels,
            )
        )

    return (
        _mask_hdu_to_coverage(
            blue_hdu,
            coverage_mask,
            closing_size,
            closing_iterations,
            retained_pixels,
        ),
        _mask_hdu_to_coverage(
            narrow_hdu,
            coverage_mask,
            closing_size,
            closing_iterations,
            retained_pixels,
        ),
        _mask_hdu_to_coverage(
            red_hdu,
            coverage_mask,
            closing_size,
            closing_iterations,
            retained_pixels,
        ),
        masked_errors,
        record,
    )


def apply_scalar_factor(
    hdu: fits.PrimaryHDU, factor: float, bunit: str | None = None
) -> fits.PrimaryHDU:
    """Return a copy of an HDU with data multiplied by a scalar factor."""

    out = hdu.copy()
    out.data = np.asarray(out.data, dtype=np.float32) * factor
    if bunit is not None:
        out.header["BUNIT"] = bunit
    return out


def apply_background_correction(
    hdu: fits.PrimaryHDU,
    surface_brightness_offset: float,
    filter_name: str,
    instrument: str,
) -> tuple[fits.PrimaryHDU, dict[str, float | str]]:
    """Subtract a flux-density background estimate from one science HDU."""

    offset_arcsec2 = float(surface_brightness_offset)
    pix_area = pixel_area_arcsec2(hdu.header)
    offset_pixel = offset_arcsec2 * pix_area

    out = hdu.copy()
    out.data = np.asarray(out.data, dtype=np.float32) - offset_pixel
    out.header["CSBKG"] = (True, "Background correction applied")
    out.header["CSBKGF"] = (filter_name, "Background correction filter")
    out.header["CSBKGI"] = (instrument, "Background correction instrument")
    out.header["CSBKGSA"] = (
        offset_arcsec2,
        "Bkg offset per arcsec2",
    )
    out.header["CSBKGPA"] = (pix_area, "Pixel area used for background correction")
    out.header["CSBKGPP"] = (
        offset_pixel,
        "Subtracted bkg per pixel",
    )
    return out, {
        "filter": filter_name,
        "instrument": instrument,
        "offset_arcsec2": offset_arcsec2,
        "pixel_area_arcsec2": pix_area,
        "offset_pixel": offset_pixel,
    }


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
