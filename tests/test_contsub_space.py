from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from hstha_contsubpipe_extended.core.config import clear_config_cache
from hstha_contsubpipe_extended.pipeline.models import ImageSet, PipelineContext
from hstha_contsubpipe_extended.pipeline.products import plan_output_paths
from hstha_contsubpipe_extended.pipeline.subtraction import linear_continuum_subtract


def _hdu(data):
    return fits.PrimaryHDU(np.asarray(data, dtype=np.float32))


def _subtract(**kwargs):
    return linear_continuum_subtract(
        narrow_hdu=_hdu([[20.0, 30.0], [40.0, 50.0]]),
        blue_hdu=_hdu([[4.0, 9.0], [16.0, 25.0]]),
        red_hdu=_hdu([[16.0, 36.0], [64.0, 100.0]]),
        blue_filter="f555w",
        red_filter="f814w",
        narrow_filter="f657n",
        narrow_pivot=2.0,
        blue_pivot=1.0,
        red_pivot=3.0,
        **kwargs,
    )


def test_linear_continuum_subtraction_matches_weighted_sum():
    contsub_hdu, continuum_hdu, *_ = _subtract(contsub_space="linear")

    expected_continuum = np.array([[10.0, 22.5], [40.0, 62.5]], dtype=np.float32)
    np.testing.assert_allclose(continuum_hdu.data, expected_continuum)
    np.testing.assert_allclose(contsub_hdu.data, _hdu([[10.0, 7.5], [0.0, -12.5]]).data)
    assert continuum_hdu.header["CSSPACE"] == "linear"
    assert contsub_hdu.header["CSSPACE"] == "linear"


def test_log_continuum_subtraction_matches_geometric_interpolation():
    contsub_hdu, continuum_hdu, *_ = _subtract(contsub_space="log")

    expected_continuum = np.array([[8.0, 18.0], [32.0, 50.0]], dtype=np.float32)
    np.testing.assert_allclose(continuum_hdu.data, expected_continuum, rtol=1e-6)
    np.testing.assert_allclose(
        contsub_hdu.data,
        _hdu([[12.0, 12.0], [8.0, 0.0]]).data,
        rtol=1e-6,
        atol=1e-5,
    )
    assert continuum_hdu.header["CSSPACE"] == "log"
    assert contsub_hdu.header["CSSPACE"] == "log"


def test_log_error_propagation_uses_relative_log_errors():
    contsub_hdu, continuum_hdu, contsub_error_hdu, continuum_error_hdu, *_ = (
        linear_continuum_subtract(
            narrow_hdu=_hdu([[20.0]]),
            blue_hdu=_hdu([[4.0]]),
            red_hdu=_hdu([[16.0]]),
            blue_filter="f555w",
            red_filter="f814w",
            narrow_filter="f657n",
            narrow_pivot=2.0,
            blue_pivot=1.0,
            red_pivot=3.0,
            narrow_error_hdu=_hdu([[1.5]]),
            blue_error_hdu=_hdu([[0.4]]),
            red_error_hdu=_hdu([[3.2]]),
            contsub_space="log",
        )
    )

    expected_continuum = 8.0
    expected_continuum_error = expected_continuum * np.sqrt((0.1 * 0.5) ** 2 + (0.2 * 0.5) ** 2)
    expected_contsub_error = np.sqrt(1.5**2 + expected_continuum_error**2)

    np.testing.assert_allclose(continuum_hdu.data, [[expected_continuum]], rtol=1e-6)
    np.testing.assert_allclose(contsub_hdu.data, [[12.0]], rtol=1e-6)
    np.testing.assert_allclose(continuum_error_hdu.data, [[expected_continuum_error]], rtol=1e-6)
    np.testing.assert_allclose(contsub_error_hdu.data, [[expected_contsub_error]], rtol=1e-6)
    assert continuum_error_hdu.header["CSSPACE"] == "log"
    assert contsub_error_hdu.header["CSSPACE"] == "log"


def test_log_space_nonpositive_continuum_inputs_become_zero_continuum():
    contsub_hdu, continuum_hdu, contsub_error_hdu, continuum_error_hdu, *_ = (
        linear_continuum_subtract(
            narrow_hdu=_hdu([[10.0, 20.0]]),
            blue_hdu=_hdu([[0.0, -1.0]]),
            red_hdu=_hdu([[9.0, 16.0]]),
            blue_filter="f555w",
            red_filter="f814w",
            narrow_filter="f657n",
            narrow_pivot=2.0,
            blue_pivot=1.0,
            red_pivot=3.0,
            narrow_error_hdu=_hdu([[1.0, 2.0]]),
            blue_error_hdu=_hdu([[0.1, 0.1]]),
            red_error_hdu=_hdu([[0.3, 0.4]]),
            contsub_space="log",
        )
    )

    np.testing.assert_allclose(continuum_hdu.data, [[0.0, 0.0]])
    np.testing.assert_allclose(contsub_hdu.data, [[10.0, 20.0]])
    np.testing.assert_allclose(continuum_error_hdu.data, [[0.0, 0.0]])
    np.testing.assert_allclose(contsub_error_hdu.data, [[1.0, 2.0]])


def test_invalid_contsub_space_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported contsub_space"):
        _subtract(contsub_space="sqrt")


def test_log_contsub_space_resolves_log_output_paths(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "paths.yaml").write_text('contsub:\n  output_root: "products"\n')
    (config_dir / "params.yaml").write_text("{}\n")
    (config_dir / "files.yaml").write_text(
        "\n".join(
            [
                "outputs:",
                '  contsub: "{contsub.output_root}/{galaxy}_{narrow_filter}_contsub_flux_{contsub_space}.fits"',
                '  continuum: "{contsub.output_root}/{galaxy}_{narrow_filter}_continuum_flux_{contsub_space}.fits"',
                '  contsub_error: "{contsub.output_root}/{galaxy}_{narrow_filter}_contsub_flux_err_{contsub_space}.fits"',
                '  continuum_error: "{contsub.output_root}/{galaxy}_{narrow_filter}_continuum_flux_err_{contsub_space}.fits"',
                '  halpha: "{contsub.output_root}/{galaxy}_{narrow_filter}_halpha_flux_nii_corr_{contsub_space}.fits"',
                '  halpha_error: "{contsub.output_root}/{galaxy}_{narrow_filter}_halpha_flux_nii_corr_err_{contsub_space}.fits"',
                "",
            ]
        )
    )
    clear_config_cache()
    context = PipelineContext(
        galaxy="ngc0000",
        galaxy_config={},
        config_dir=config_dir,
        settings={"contsub_space": "log"},
        params={},
        files={},
        image_set=ImageSet(
            galaxy="ngc0000",
            blue_filter="f555w",
            red_filter="f814w",
            narrow_filter="f657n",
            blue_file=Path("blue.fits"),
            red_file=Path("red.fits"),
            narrow_file=Path("narrow.fits"),
        ),
    )

    try:
        plan_output_paths(context)
    finally:
        clear_config_cache()

    assert context.contsub_file.name == "ngc0000_f657n_contsub_flux_log.fits"
    assert context.halpha_error_file.name == "ngc0000_f657n_halpha_flux_nii_corr_err_log.fits"
