"""Continuum subtraction algorithms."""

from __future__ import annotations

import numpy as np
from astropy.io import fits


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
