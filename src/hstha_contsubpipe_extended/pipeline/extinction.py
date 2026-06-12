"""Foreground extinction helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from astropy.io import fits
from astropy.table import Table


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


def foreground_ebv(
    galaxy: str,
    settings: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float | None:
    """Read foreground E(B-V) for one galaxy from the configured sample table."""

    ext_cfg = params.get("foreground_extinction", {}) or {}
    if not bool(ext_cfg.get("enabled", False)):
        return None

    if "ebv" in settings:
        return float(settings["ebv"])

    sample_table_path = str(ext_cfg.get("sample_table_path", "") or "")
    if not sample_table_path:
        raise ValueError("foreground_extinction.enabled is true but sample_table_path is empty")

    table = Table.read(Path(sample_table_path).expanduser())
    name_column = str(ext_cfg.get("galaxy_column", "name"))
    ebv_column = str(ext_cfg.get("ebv_column", "mwext_sf11"))
    sample_name = str(settings.get("sample_name", galaxy))
    candidate_names = sample_table_candidate_names(sample_name)

    names = [normalize_galaxy_name(value) for value in table[name_column]]
    for candidate_name in candidate_names:
        matches = [idx for idx, name in enumerate(names) if name == candidate_name]
        if matches:
            return float(table[ebv_column][matches[0]])

    attempted = ", ".join(repr(candidate_name) for candidate_name in candidate_names)
    raise ValueError(
        f"{galaxy}: no sample-table row found for sample_name={sample_name!r}; "
        f"attempted {attempted}"
    )


def apply_foreground_extinction_to_hdu_pair(
    science_hdu: fits.PrimaryHDU,
    error_hdu: fits.PrimaryHDU | None,
    ebv: float,
    r_v: float,
) -> None:
    """Apply foreground extinction in place to a science/error HDU pair."""

    pivot = float(science_hdu.header["CSBPWAV"])
    factor = extinction_correction_factor(pivot, ebv=ebv, r_v=r_v)
    science_hdu.data = np.asarray(science_hdu.data, dtype=np.float32) * factor
    if error_hdu is not None:
        error_hdu.data = np.asarray(error_hdu.data, dtype=np.float32) * factor
    science_hdu.header["CSEBV"] = (float(ebv), "Foreground E(B-V)")
    science_hdu.header["CSEXTCOR"] = (factor, "Foreground extinction correction")
