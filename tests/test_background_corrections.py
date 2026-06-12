from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from hstha_contsubpipe_extended.pipeline.fits_ops import apply_background_correction
from hstha_contsubpipe_extended.pipeline.models import ImageSet, PipelineContext
from hstha_contsubpipe_extended.pipeline.stages import _stage_apply_background_corrections


def _header_with_pixel_area(area_arcsec2: float = 2.0) -> fits.Header:
    header = fits.Header()
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = area_arcsec2 / 3600.0
    return header


def _hdu(data):
    return fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=_header_with_pixel_area())


def _context(settings):
    return PipelineContext(
        galaxy="ngc1365",
        galaxy_config={},
        config_dir="config",
        settings=settings,
        params={},
        files={},
        image_set=ImageSet(
            galaxy="ngc1365",
            blue_filter="f555w",
            red_filter="f814w",
            narrow_filter="f657n",
            blue_file=Path("blue.fits"),
            red_file=Path("red.fits"),
            narrow_file=Path("narrow.fits"),
        ),
        blue_hdu=_hdu([[10.0]]),
        narrow_hdu=_hdu([[20.0]]),
        red_hdu=_hdu([[30.0]]),
        blue_error_hdu=_hdu([[1.0]]),
        narrow_error_hdu=_hdu([[2.0]]),
        red_error_hdu=_hdu([[3.0]]),
        blue_bandpass={"instrument": "UVIS"},
        narrow_bandpass={"instrument": "UVIS"},
        red_bandpass={"instrument": "UVIS"},
    )


def test_apply_background_correction_converts_arcsec2_offset_to_pixel_offset():
    corrected, record = apply_background_correction(
        hdu=_hdu([[10.0, 20.0]]),
        surface_brightness_offset=-5.0,
        filter_name="f555w",
        instrument="uvis",
    )

    np.testing.assert_allclose(corrected.data, [[0.0, 10.0]])
    assert record["offset_arcsec2"] == -5.0
    assert record["pixel_area_arcsec2"] == 2.0
    assert record["offset_pixel"] == -10.0
    assert corrected.header["CSBKG"] is True
    assert corrected.header["CSBKGSA"] == -5.0
    assert corrected.header["CSBKGPP"] == -10.0


def test_background_correction_stage_applies_matching_science_offsets_only():
    context = _context(
        {
            "background_corrections": {
                "f555w": {"uvis": -1.0},
                "f657n": {"uvis": 2.0},
                "f814w": {"uvis": -3.0},
            }
        }
    )

    _stage_apply_background_corrections(context)

    np.testing.assert_allclose(context.blue_hdu.data, [[8.0]])
    np.testing.assert_allclose(context.narrow_hdu.data, [[24.0]])
    np.testing.assert_allclose(context.red_hdu.data, [[24.0]])
    np.testing.assert_allclose(context.blue_error_hdu.data, [[1.0]])
    np.testing.assert_allclose(context.narrow_error_hdu.data, [[2.0]])
    np.testing.assert_allclose(context.red_error_hdu.data, [[3.0]])
    assert [record["filter"] for record in context.background_corrections] == [
        "f555w",
        "f657n",
        "f814w",
    ]
    assert [record["image"] for record in context.background_corrections] == [
        "blue",
        "narrow",
        "red",
    ]


def test_background_correction_requires_numeric_values():
    context = _context({"background_corrections": {"f555w": {"uvis": "not-a-number"}}})

    with pytest.raises(ValueError, match="must be numeric"):
        _stage_apply_background_corrections(context)


def test_background_correction_raises_for_selected_filter_instrument_mismatch():
    context = _context({"background_corrections": {"f555w": {"acs": -1.0}}})

    with pytest.raises(ValueError, match="not for resolved instrument"):
        _stage_apply_background_corrections(context)
