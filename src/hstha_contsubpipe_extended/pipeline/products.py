"""Output product construction and writing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from astropy.io import fits

from .discovery import resolve_template_path
from .fits_ops import (
    apply_scalar_factor,
    convert_flux_density_to_flux,
    convert_output_units,
)
from .galaxy_config import DEFAULT_CONFIG_DIR
from .models import ImageSet, PipelineContext


def output_path(
    template_key: str,
    image_set: ImageSet,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    product: str | None = None,
) -> Path:
    """Resolve one output path from ``files.yaml``."""

    product_name = product or template_key.split(".")[-1]
    return Path(
        resolve_template_path(
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


def plan_output_paths(context: PipelineContext) -> None:
    """Populate all output paths in the pipeline context."""

    if context.image_set is None:
        raise ValueError("Cannot plan outputs before inputs are resolved")

    image_set = context.image_set
    context.contsub_file = output_path("outputs.contsub", image_set, config_dir=context.config_dir)
    context.continuum_file = output_path(
        "outputs.continuum", image_set, config_dir=context.config_dir
    )
    context.contsub_error_file = output_path(
        "outputs.contsub_error", image_set, config_dir=context.config_dir
    )
    context.continuum_error_file = output_path(
        "outputs.continuum_error", image_set, config_dir=context.config_dir
    )
    context.halpha_file = output_path("outputs.halpha", image_set, config_dir=context.config_dir)
    context.halpha_error_file = output_path(
        "outputs.halpha_error", image_set, config_dir=context.config_dir
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


def build_products(context: PipelineContext) -> None:
    """Build final FITS product HDUs and attach them to the context."""

    required = (
        context.image_set,
        context.narrow_hdu,
        context.narrow_bandpass,
        context.contsub_hdu,
        context.continuum_hdu,
        context.contsub_file,
        context.continuum_file,
        context.contsub_error_file,
        context.continuum_error_file,
        context.halpha_file,
        context.halpha_error_file,
    )
    if any(item is None for item in required):
        raise ValueError("Cannot build products before subtraction and output planning are complete")

    image_set = context.image_set
    narrow_bandpass = context.narrow_bandpass
    narrowband_width = get_narrowband_width(
        narrow_hdu=context.narrow_hdu,
        narrow_filter=image_set.narrow_filter,
        settings=context.settings,
        bandpass_width=float(narrow_bandpass["width"]),
    )
    context.narrowband_width = narrowband_width

    products: dict[Path, fits.PrimaryHDU | None] = {
        context.contsub_file: convert_flux_density_to_flux(context.contsub_hdu, narrowband_width),
        context.continuum_file: convert_flux_density_to_flux(
            context.continuum_hdu, narrowband_width
        ),
        context.contsub_error_file: (
            convert_flux_density_to_flux(context.contsub_error_output_hdu, narrowband_width)
            if context.contsub_error_output_hdu is not None
            else None
        ),
        context.continuum_error_file: (
            convert_flux_density_to_flux(context.continuum_error_output_hdu, narrowband_width)
            if context.continuum_error_output_hdu is not None
            else None
        ),
    }

    nii_to_halpha = float(context.settings.get("nii_to_halpha", 0.0))
    context.nii_to_halpha = nii_to_halpha
    halpha_hdu = apply_scalar_factor(
        products[context.contsub_file],
        1.0 / (1.0 + nii_to_halpha),
        "1e-20 erg/s/cm2/pixel",
    )
    halpha_hdu.header["CSNIIRAT"] = (nii_to_halpha, "Assumed total [NII]/H-alpha")
    halpha_hdu.header["CSPROD"] = ("HALPHA_NII_CORR", "H-alpha after fixed [NII] correction")
    products[context.halpha_file] = halpha_hdu

    if products[context.contsub_error_file] is not None:
        halpha_error_hdu = apply_scalar_factor(
            products[context.contsub_error_file],
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
        products[context.halpha_error_file] = halpha_error_hdu
    else:
        products[context.halpha_error_file] = None

    if not bool(context.settings.get("write_continuum", True)):
        products[context.continuum_file] = None
        products[context.continuum_error_file] = None
        context.continuum_file = None
        context.continuum_error_file = None

    context.products = {
        path: convert_output_units(hdu, context.settings) if hdu is not None else None
        for path, hdu in products.items()
    }


def write_products(context: PipelineContext) -> None:
    """Write all non-empty product HDUs to disk."""

    for path, hdu in context.products.items():
        if hdu is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        hdu.writeto(path, overwrite=bool(context.overwrite))
