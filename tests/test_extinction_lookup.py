from io import BytesIO

import numpy as np
import pytest
import requests
from astropy.io import fits
from astropy.table import Table

from hstha_contsubpipe_extended.pipeline import extinction
from hstha_contsubpipe_extended.pipeline.extinction import (
    ForegroundExtinctionResolution,
    apply_foreground_extinction_to_hdu_pair,
    foreground_ebv,
    get_ned_overview,
    resolve_foreground_extinction,
)
from hstha_contsubpipe_extended.pipeline.models import PipelineContext
from hstha_contsubpipe_extended.pipeline.run_log import context_audit_record


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


def _ned_table(av=0.31, ak=0.03):
    return Table({"a_lambda_V": [av], "a_lambda_K": [ak]})


def test_foreground_ebv_exact_sample_table_match_does_not_call_ned(tmp_path, monkeypatch):
    sample_table = _sample_table(tmp_path, [("ngc2997", 0.11)])
    monkeypatch.setattr(
        extinction,
        "get_ned_overview",
        lambda target: pytest.fail(f"unexpected NED lookup for {target}"),
    )

    resolution = resolve_foreground_extinction("ngc2997", {}, _params(sample_table))

    assert resolution.source == "sample_table"
    assert resolution.resolved_name == "ngc2997"
    assert resolution.ebv == pytest.approx(0.11)
    assert resolution.av == pytest.approx(0.341)


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


def test_override_records_derived_av_and_provenance(tmp_path, monkeypatch):
    sample_table = _sample_table(tmp_path, [("ngc2997", 0.11)])
    params = _params(sample_table)
    params["foreground_extinction"]["r_v"] = 3.2
    monkeypatch.setattr(
        extinction,
        "get_ned_overview",
        lambda target: pytest.fail(f"unexpected NED lookup for {target}"),
    )

    resolution = resolve_foreground_extinction("ngc2997", {"ebv": 0.2}, params)

    assert resolution.status == "applied"
    assert resolution.source == "override"
    assert resolution.ebv == pytest.approx(0.2)
    assert resolution.av == pytest.approx(0.64)
    assert resolution.r_v == pytest.approx(3.2)


def test_foreground_ebv_does_not_strip_real_suffix_name(tmp_path, monkeypatch):
    sample_table = _sample_table(tmp_path, [("ngc4496", 0.16)])
    attempted = []

    def fake_ned(target):
        attempted.append(target)
        return _ned_table()

    monkeypatch.setattr(extinction, "get_ned_overview", fake_ned)

    resolution = resolve_foreground_extinction("ngc4496a", {}, _params(sample_table))

    assert attempted == ["ngc4496a"]
    assert resolution.source == "ned"
    assert resolution.resolved_name == "ngc4496a"


def test_ned_fallback_uses_av_and_retains_ak(tmp_path, monkeypatch):
    sample_table = _sample_table(tmp_path, [("ngc628", 0.17)])
    monkeypatch.setattr(extinction, "get_ned_overview", lambda target: _ned_table(0.31, 0.028))

    resolution = resolve_foreground_extinction("ngc2997", {}, _params(sample_table))

    assert resolution.status == "applied"
    assert resolution.source == "ned"
    assert resolution.resolved_name == "ngc2997"
    assert resolution.av == pytest.approx(0.31)
    assert resolution.ak == pytest.approx(0.028)
    assert resolution.ebv == pytest.approx(0.1)


def test_ned_fallback_tries_subfield_then_base_name(tmp_path, monkeypatch):
    sample_table = _sample_table(tmp_path, [("ngc628", 0.17)])
    attempted = []

    def fake_ned(target):
        attempted.append(target)
        if target == "ngc2997w":
            raise ValueError("unknown object")
        return _ned_table()

    monkeypatch.setattr(extinction, "get_ned_overview", fake_ned)

    resolution = resolve_foreground_extinction("ngc2997w", {}, _params(sample_table))

    assert attempted == ["ngc2997w", "ngc2997"]
    assert resolution.resolved_name == "ngc2997"


@pytest.mark.parametrize(
    "failure",
    [
        requests.Timeout("timed out"),
        requests.HTTPError("server error"),
        ValueError("malformed VOTable"),
    ],
)
def test_ned_failures_warn_and_skip(tmp_path, monkeypatch, caplog, failure):
    sample_table = _sample_table(tmp_path, [("ngc628", 0.17)])

    def fail_ned(target):
        raise failure

    monkeypatch.setattr(extinction, "get_ned_overview", fail_ned)

    resolution = resolve_foreground_extinction("ngc2997w", {}, _params(sample_table))

    assert resolution.status == "skipped"
    assert resolution.source == "ned"
    assert resolution.ebv is None
    assert "ngc2997w" in resolution.failure_reason
    assert "continuing without foreground correction" in caplog.text


@pytest.mark.parametrize(
    "table",
    [
        Table(),
        Table({"a_lambda_K": [0.03]}),
        Table({"a_lambda_V": [np.nan]}),
        Table({"a_lambda_V": np.ma.array([0.31], mask=[True])}),
    ],
)
def test_ned_missing_or_unusable_av_skips(tmp_path, monkeypatch, table):
    sample_table = _sample_table(tmp_path, [("ngc628", 0.17)])
    monkeypatch.setattr(extinction, "get_ned_overview", lambda target: table)

    resolution = resolve_foreground_extinction("ngc2997", {}, _params(sample_table))

    assert resolution.status == "skipped"
    assert resolution.ebv is None


def test_get_ned_overview_requests_and_parses_votable(monkeypatch):
    payload = BytesIO()
    _ned_table().write(payload, format="votable")
    calls = []

    class FakeResponse:
        content = payload.getvalue()

        def raise_for_status(self):
            calls.append("raise_for_status")

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse()

    monkeypatch.setattr(extinction.requests, "get", fake_get)

    table = get_ned_overview("M31")

    assert calls[0] == (extinction.NED_OVERVIEW_URL, {"TARGET": "M31"}, 30)
    assert calls[1] == "raise_for_status"
    assert table["a_lambda_V"][0] == pytest.approx(0.31)


def test_applied_headers_record_extinction_provenance():
    science_hdu = fits.PrimaryHDU(data=np.ones((2, 2)), header=fits.Header({"CSBPWAV": 5500.0}))
    error_hdu = fits.PrimaryHDU(data=np.ones((2, 2)))
    resolution = ForegroundExtinctionResolution(
        status="applied",
        source="ned",
        requested_name="ngc2997w",
        resolved_name="ngc2997",
        ebv=0.1,
        av=0.31,
        ak=0.028,
        r_v=3.1,
    )

    apply_foreground_extinction_to_hdu_pair(
        science_hdu,
        error_hdu,
        ebv=resolution.ebv,
        r_v=resolution.r_v,
        resolution=resolution,
    )

    assert science_hdu.header["CSEBV"] == pytest.approx(0.1)
    assert science_hdu.header["CSEAV"] == pytest.approx(0.31)
    assert science_hdu.header["CSEAK"] == pytest.approx(0.028)
    assert science_hdu.header["CSERV"] == pytest.approx(3.1)
    assert science_hdu.header["CSEXTSRC"] == "ned"
    assert science_hdu.header["CSEXTARG"] == "ngc2997"
    assert science_hdu.header["CSEXTCOR"] > 1.0
    assert np.all(science_hdu.data > 1.0)
    assert np.all(error_hdu.data > 1.0)


def test_context_audit_includes_foreground_extinction_record():
    context = PipelineContext(
        galaxy="ngc2997",
        galaxy_config={},
        config_dir=".",
        settings={},
        params={},
        files={},
    )
    context.foreground_extinction = {"status": "skipped", "failure_reason": "NED timeout"}
    result = context.to_result()

    record = context_audit_record(context, result)

    assert record["derived"]["foreground_extinction"] == context.foreground_extinction
