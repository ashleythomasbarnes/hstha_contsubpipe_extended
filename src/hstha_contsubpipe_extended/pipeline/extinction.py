"""Foreground extinction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import logging
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table


LOGGER = logging.getLogger(__name__)
NED_OVERVIEW_URL = "https://ned.ipac.caltech.edu/NED::API/OverviewOfObject"


@dataclass(frozen=True)
class ForegroundExtinctionResolution:
    """Resolved foreground-extinction input and its provenance."""

    status: str
    source: str
    requested_name: str
    resolved_name: str | None
    ebv: float | None
    av: float | None
    ak: float | None
    r_v: float
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable provenance record."""

        return asdict(self)


def get_ned_overview(target: str) -> Table:
    """Retrieve the NED overview table for a named object."""

    response = requests.get(
        NED_OVERVIEW_URL,
        params={"TARGET": target},
        timeout=30,
    )
    response.raise_for_status()
    return Table.read(BytesIO(response.content), format="votable", table_id=0)


def ccm89_a_lambda_over_av(wavelength_angstrom: float, r_v: float = 3.1) -> float:
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
    a_lambda = av * ccm89_a_lambda_over_av(wavelength_angstrom, r_v=r_v)
    return float(10 ** (0.4 * a_lambda))


def normalize_galaxy_name(name: Any) -> str:
    """Normalize galaxy names for robust sample-table matching."""

    text = str(name).strip().lower()
    match = re.fullmatch(r"([a-z]+)0*([0-9]+)([a-z]*)", text)
    if match:
        prefix, number, suffix = match.groups()
        return f"{prefix}{int(number)}{suffix}"
    return text


def sample_table_candidate_names(name: Any) -> list[str]:
    """Return sample-table lookup candidates, exact first."""

    normalized = normalize_galaxy_name(name)
    candidates = [normalized]

    match = re.fullmatch(r"([a-z]+[0-9]+)([censw])", normalized)
    if match:
        candidates.append(match.group(1))

    return candidates


def _optional_finite_value(table: Table, column: str) -> float | None:
    """Return a finite, unmasked value from the first row when available."""

    if len(table) == 0 or column not in table.colnames:
        return None
    value = table[column][0]
    if np.ma.is_masked(value):
        return None
    numeric_value = float(value)
    return numeric_value if np.isfinite(numeric_value) else None


def resolve_foreground_extinction(
    galaxy: str,
    settings: Mapping[str, Any],
    params: Mapping[str, Any],
) -> ForegroundExtinctionResolution:
    """Resolve foreground extinction from an override, sample table, or NED."""

    ext_cfg = params.get("foreground_extinction", {}) or {}
    r_v = float(ext_cfg.get("r_v", 3.1))
    sample_name = str(settings.get("sample_name", galaxy))
    if not bool(ext_cfg.get("enabled", False)):
        return ForegroundExtinctionResolution(
            status="disabled",
            source="disabled",
            requested_name=sample_name,
            resolved_name=None,
            ebv=None,
            av=None,
            ak=None,
            r_v=r_v,
        )

    if "ebv" in settings:
        ebv = float(settings["ebv"])
        return ForegroundExtinctionResolution(
            status="applied",
            source="override",
            requested_name=sample_name,
            resolved_name=sample_name,
            ebv=ebv,
            av=ebv * r_v,
            ak=None,
            r_v=r_v,
        )

    sample_table_path = str(ext_cfg.get("sample_table_path", "") or "")
    if not sample_table_path:
        raise ValueError("foreground_extinction.enabled is true but sample_table_path is empty")

    table = Table.read(Path(sample_table_path).expanduser())
    name_column = str(ext_cfg.get("galaxy_column", "name"))
    ebv_column = str(ext_cfg.get("ebv_column", "mwext_sf11"))
    candidate_names = sample_table_candidate_names(sample_name)

    names = [normalize_galaxy_name(value) for value in table[name_column]]
    for candidate_name in candidate_names:
        matches = [idx for idx, name in enumerate(names) if name == candidate_name]
        if matches:
            ebv = float(table[ebv_column][matches[0]])
            return ForegroundExtinctionResolution(
                status="applied",
                source="sample_table",
                requested_name=sample_name,
                resolved_name=candidate_name,
                ebv=ebv,
                av=ebv * r_v,
                ak=None,
                r_v=r_v,
            )

    failures: list[str] = []
    for candidate_name in candidate_names:
        try:
            ned_table = get_ned_overview(candidate_name)
            av = _optional_finite_value(ned_table, "a_lambda_V")
            if av is None or av < 0:
                raise ValueError("response has no usable a_lambda_V value")
            ak = _optional_finite_value(ned_table, "a_lambda_K")
        except Exception as exc:
            failures.append(f"{candidate_name!r}: {exc}")
            continue

        return ForegroundExtinctionResolution(
            status="applied",
            source="ned",
            requested_name=sample_name,
            resolved_name=candidate_name,
            ebv=av / r_v,
            av=av,
            ak=ak,
            r_v=r_v,
        )

    attempted = ", ".join(repr(candidate_name) for candidate_name in candidate_names)
    reason = (
        f"no sample-table row found; attempted {attempted}; "
        f"NED fallback failed ({'; '.join(failures)})"
    )
    LOGGER.warning("%s: %s; continuing without foreground correction", galaxy, reason)
    return ForegroundExtinctionResolution(
        status="skipped",
        source="ned",
        requested_name=sample_name,
        resolved_name=None,
        ebv=None,
        av=None,
        ak=None,
        r_v=r_v,
        failure_reason=reason,
    )


def foreground_ebv(
    galaxy: str,
    settings: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float | None:
    """Return resolved foreground E(B-V), or ``None`` when not applied."""

    return resolve_foreground_extinction(galaxy, settings, params).ebv


def apply_foreground_extinction_to_hdu_pair(
    science_hdu: fits.PrimaryHDU,
    error_hdu: fits.PrimaryHDU | None,
    ebv: float,
    r_v: float,
    resolution: ForegroundExtinctionResolution | None = None,
) -> None:
    """Apply foreground extinction in place to a science/error HDU pair."""

    pivot = float(science_hdu.header["CSBPWAV"])
    factor = extinction_correction_factor(pivot, ebv=ebv, r_v=r_v)
    science_hdu.data = np.asarray(science_hdu.data, dtype=np.float32) * factor
    if error_hdu is not None:
        error_hdu.data = np.asarray(error_hdu.data, dtype=np.float32) * factor
    science_hdu.header["CSEBV"] = (float(ebv), "Foreground E(B-V)")
    science_hdu.header["CSEXTCOR"] = (factor, "Foreground extinction correction")
    science_hdu.header["CSERV"] = (float(r_v), "Foreground extinction R(V)")
    if resolution is not None:
        science_hdu.header["CSEXTSRC"] = (resolution.source, "Foreground extinction source")
        if resolution.resolved_name is not None:
            science_hdu.header["CSEXTARG"] = (
                resolution.resolved_name,
                "Foreground extinction lookup target",
            )
        if resolution.av is not None:
            science_hdu.header["CSEAV"] = (float(resolution.av), "Foreground extinction A(V)")
        if resolution.ak is not None:
            science_hdu.header["CSEAK"] = (float(resolution.ak), "Foreground extinction A(K)")
