"""FITS image operations and unit conversions."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from astropy.io import fits


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
