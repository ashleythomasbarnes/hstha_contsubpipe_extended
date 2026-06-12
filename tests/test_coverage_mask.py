import numpy as np
import pytest
from astropy.io import fits

from hstha_contsubpipe_extended.pipeline.fits_ops import match_spatial_coverage
from hstha_contsubpipe_extended.pipeline.models import PipelineContext
from hstha_contsubpipe_extended.pipeline.stages import _stage_match_spatial_coverage


def _hdu(data):
    return fits.PrimaryHDU(np.asarray(data, dtype=np.float32))


def _context():
    return PipelineContext(
        galaxy="ngc1365",
        galaxy_config={},
        config_dir="config",
        settings={
            "coverage_mask": {
                "enabled": True,
                "closing_size": 1,
                "closing_iterations": 1,
            }
        },
        params={},
        files={},
        blue_hdu=_hdu([[np.nan, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        narrow_hdu=_hdu([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        red_hdu=_hdu([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, np.nan]]),
        blue_error_hdu=_hdu(np.ones((3, 3))),
        narrow_error_hdu=_hdu(np.ones((3, 3)) * 2.0),
        red_error_hdu=_hdu(np.ones((3, 3)) * 3.0),
    )


def test_match_spatial_coverage_stage_applies_common_mask_to_science_and_errors():
    context = _context()

    _stage_match_spatial_coverage(context)

    expected_invalid = np.array([[True, True, False], [False, False, False], [False, False, True]])
    for hdu in (
        context.blue_hdu,
        context.narrow_hdu,
        context.red_hdu,
        context.blue_error_hdu,
        context.narrow_error_hdu,
        context.red_error_hdu,
    ):
        np.testing.assert_array_equal(np.isnan(hdu.data), expected_invalid)
        assert hdu.header["CSCOVMSK"] is True
        assert hdu.header["CSCOVPIX"] == 6

    assert context.coverage_mask == {
        "raw_pixels": 6,
        "retained_pixels": 6,
        "masked_pixels": 3,
        "closing_size": 1,
        "closing_iterations": 1,
    }


def test_match_spatial_coverage_raises_for_shape_mismatch():
    with pytest.raises(ValueError, match="Science image shapes differ"):
        match_spatial_coverage(
            blue_hdu=_hdu([[1.0]]),
            narrow_hdu=_hdu([[1.0, 2.0]]),
            red_hdu=_hdu([[1.0]]),
        )
