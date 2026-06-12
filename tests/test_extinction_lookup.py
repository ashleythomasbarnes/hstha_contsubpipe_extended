import pytest
from astropy.table import Table

from hstha_contsubpipe_extended.pipeline.extinction import foreground_ebv


def _sample_table(tmp_path, rows):
    path = tmp_path / "sample_table.fits"
    table = Table(rows=rows, names=("name", "mwext_sf11"))
    table.write(path)
    return path


def _params(sample_table_path):
    return {
        "foreground_extinction": {
            "enabled": True,
            "sample_table_path": str(sample_table_path),
            "galaxy_column": "name",
            "ebv_column": "mwext_sf11",
        }
    }


def test_foreground_ebv_exact_sample_table_match(tmp_path):
    sample_table = _sample_table(tmp_path, [("ngc2997", 0.11)])

    ebv = foreground_ebv("ngc2997", {}, _params(sample_table))

    assert ebv == pytest.approx(0.11)


def test_foreground_ebv_falls_back_from_west_subfield_to_base_name(tmp_path):
    sample_table = _sample_table(tmp_path, [("ngc2997", 0.12)])

    ebv = foreground_ebv("ngc2997w", {}, _params(sample_table))

    assert ebv == pytest.approx(0.12)


def test_foreground_ebv_falls_back_from_center_subfield_to_base_name(tmp_path):
    sample_table = _sample_table(tmp_path, [("ngc628", 0.13)])

    ebv = foreground_ebv("ngc628c", {}, _params(sample_table))

    assert ebv == pytest.approx(0.13)


def test_foreground_ebv_falls_back_from_north_subfield_to_base_name(tmp_path):
    sample_table = _sample_table(tmp_path, [("ngc1234", 0.18)])

    ebv = foreground_ebv("ngc1234n", {}, _params(sample_table))

    assert ebv == pytest.approx(0.18)


def test_foreground_ebv_sample_name_override_controls_lookup(tmp_path):
    sample_table = _sample_table(tmp_path, [("ngc2997", 0.14), ("ngc628", 0.15)])

    ebv = foreground_ebv("ngc2997w", {"sample_name": "ngc628e"}, _params(sample_table))

    assert ebv == pytest.approx(0.15)


def test_foreground_ebv_does_not_strip_real_suffix_name(tmp_path):
    sample_table = _sample_table(tmp_path, [("ngc4496", 0.16)])

    with pytest.raises(ValueError, match=r"attempted 'ngc4496a'$"):
        foreground_ebv("ngc4496a", {}, _params(sample_table))


def test_foreground_ebv_missing_row_lists_attempted_names(tmp_path):
    sample_table = _sample_table(tmp_path, [("ngc628", 0.17)])

    with pytest.raises(ValueError, match=r"attempted 'ngc2997w', 'ngc2997'"):
        foreground_ebv("ngc2997w", {}, _params(sample_table))
