"""HST bandpass catalog loading and image metadata resolution."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table

LOGGER = logging.getLogger(__name__)

try:
    from synphot import SpectralElement, units as synphot_units
except ImportError:  # pragma: no cover - exercised only when synphot is installed.
    SpectralElement = None
    synphot_units = None


def normalize_instrument(instrument: Any) -> str:
    """Normalize HST instrument labels to the bandpass-table convention."""

    text = str(instrument).strip().upper()
    if text in {"WFC3", "WFC3_UVIS", "UVIS1"}:
        return "UVIS"
    if text == "UVIS2":
        return "UVIS2"
    if text in {"ACS_WFC", "WFC"}:
        return "ACS"
    return text


def infer_bandpass_instrument(hdu: fits.PrimaryHDU, filename: Path) -> str:
    """Infer the external bandpass instrument label for one HST image."""

    path_text = str(filename).lower()
    if "_acs_" in path_text or "/acs" in path_text:
        return "ACS"
    if "_uvis_" in path_text or "/uvis" in path_text:
        return "UVIS"

    instrume = str(hdu.header.get("INSTRUME", "")).upper()
    detector = str(hdu.header.get("DETECTOR", "")).upper()
    if instrume == "ACS":
        return "ACS"
    if instrume == "WFC3" and detector == "UVIS":
        return "UVIS"
    if detector:
        return normalize_instrument(detector)
    return normalize_instrument(instrume)


def bandpass_key_from_dat_file(path: Path) -> tuple[str, str]:
    """Return ``(instrument, filter)`` from an HST throughput filename."""

    name = path.name.split(".dat")[0]
    name = name.replace("HST_", "").replace(".F", "_F")
    name = name.replace("WFC_", "")
    name = name.replace("WFC3_", "")
    name = name.replace("UVIS1", "UVIS")
    name = name.replace("UVIS2", "UVIS")
    instrument, filter_name = name.split("_", 1)
    return normalize_instrument(instrument), filter_name.upper()


def bandpass_from_curve(path: Path) -> dict[str, float | str]:
    """Read a throughput curve and return the same bandpass keys as the old pipeline."""

    if SpectralElement is not None and synphot_units is not None:
        area = 45238.93416 * synphot_units.AREA
        bp = SpectralElement.from_file(path)
        return {
            "equivwidth": float(bp.equivwidth().value),
            "integrate": float(bp.integrate().value),
            "rmswidth": float(bp.rmswidth().value),
            "photbw": float(bp.photbw().value),
            "fwhm": float(bp.fwhm().value),
            "rectwidth": float(bp.rectwidth().value),
            "pivot": float(bp.pivot().value),
            "unit_response": float(bp.unit_response(area).value),
            "source": str(path),
            "source_kind": "filter_curve",
        }

    arr = np.loadtxt(path)
    wave = np.asarray(arr[:, 0], dtype=float)
    throughput = np.asarray(arr[:, 1], dtype=float)
    valid = np.isfinite(wave) & np.isfinite(throughput) & (throughput > 0)
    wave = wave[valid]
    throughput = throughput[valid]
    if wave.size < 2:
        raise ValueError(f"Not enough valid throughput samples in {path}")

    integral_t_lambda = np.trapz(throughput * wave, wave)
    integral_t_over_lambda = np.trapz(throughput / wave, wave)
    pivot = math.sqrt(integral_t_lambda / integral_t_over_lambda)
    integrate = np.trapz(throughput, wave)
    rectwidth = integrate / np.nanmax(throughput)

    mean_wave = np.trapz(wave * throughput, wave) / integrate
    rmswidth = math.sqrt(np.trapz(((wave - mean_wave) ** 2) * throughput, wave) / integrate)
    half_max = np.nanmax(throughput) / 2.0
    above_half = wave[throughput >= half_max]
    fwhm = float(above_half[-1] - above_half[0]) if above_half.size > 1 else float("nan")

    return {
        "equivwidth": float(integrate),
        "integrate": float(integrate),
        "rmswidth": float(rmswidth),
        "photbw": float(rectwidth),
        "fwhm": fwhm,
        "rectwidth": float(rectwidth),
        "pivot": float(pivot),
        "unit_response": float("nan"),
        "source": str(path),
        "source_kind": "filter_curve_numeric",
    }


def load_bandpass_catalog(params: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Load all HST filter curves, plus optional precomputed table columns."""

    bandpass_cfg = params.get("bandpass", {}) or {}
    root_value = str(bandpass_cfg.get("filter_root", "") or "")
    if not root_value:
        return {}
    root = Path(root_value).expanduser()

    catalog: dict[tuple[str, str], dict[str, Any]] = {}
    for curve_path in sorted(root.glob("*.dat")):
        try:
            instrument, filter_name = bandpass_key_from_dat_file(curve_path)
            catalog[(instrument, filter_name)] = bandpass_from_curve(curve_path)
        except Exception as exc:
            LOGGER.warning("Could not read bandpass curve %s: %s", curve_path, exc)

    table_name = str(bandpass_cfg.get("table_file", "filter_table.fits"))
    table_path = root / table_name
    if table_path.exists():
        table = Table.read(table_path)
        for row in table:
            first = str(row[0]).strip().upper()
            second = str(row[1]).strip().upper()
            if first in {"ACS", "UVIS", "UVIS1", "UVIS2", "WFC3"}:
                instrument, filter_name = normalize_instrument(first), second
            else:
                filter_name, instrument = first, normalize_instrument(second)
            entry = catalog.setdefault((instrument, filter_name), {})
            entry["table_photplam"] = float(row["photplam"])
            entry["table_photbw"] = float(row["photbw"])
            if "photflam" in table.colnames:
                scale = float(bandpass_cfg.get("table_photflam_scale", 1e-19))
                entry["table_photflam"] = float(row["photflam"]) * scale
            entry["table_source"] = str(table_path)

    return catalog


def bandpass_entry_for(
    instrument: str,
    filter_key: str,
    catalog: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any] | None]:
    """Return the best external entry for an instrument/filter pair."""

    candidates = [instrument]
    if instrument == "UVIS2":
        candidates.append("UVIS")
    if instrument == "UVIS":
        candidates.append("UVIS2")
    for candidate in candidates:
        entry = catalog.get((candidate, filter_key))
        if entry is not None:
            return candidate, entry
    return instrument, None


def resolve_bandpass_value(
    hdu: fits.PrimaryHDU,
    filename: Path,
    entry: Mapping[str, Any] | None,
    quantity: str,
    source_setting: str,
    header_key: str,
    external_keys: Sequence[str],
    allow_header: bool,
) -> tuple[float, str]:
    """Resolve one bandpass quantity from configured source preferences."""

    source = source_setting.lower()
    if source in {"header", "fits"}:
        if header_key not in hdu.header:
            raise KeyError(f"{filename} is missing required header keyword {header_key}")
        return float(hdu.header[header_key]), f"header:{header_key}"

    source_key_map = {
        "filter": external_keys,
        "filter_curve": external_keys,
        "external": external_keys,
        "table": [f"table_{quantity}", *external_keys],
        "filter_table": [f"table_{quantity}", *external_keys],
        "filter_rectwidth": ["rectwidth", "table_photbw"],
        "rectwidth": ["rectwidth", "table_photbw"],
        "filter_photbw": ["photbw", "table_photbw"],
        "photbw": ["photbw", "table_photbw"],
        "unit_response": ["unit_response"],
        "filter_unit_response": ["unit_response"],
    }
    keys = source_key_map.get(source, external_keys)
    if entry is not None:
        for key in keys:
            if key in entry and np.isfinite(float(entry[key])):
                src = str(entry.get("source") or entry.get("table_source") or "external")
                if key.startswith("table_"):
                    src = str(entry.get("table_source") or src)
                return float(entry[key]), f"{source}:{Path(src).name}"

    if allow_header and header_key in hdu.header:
        return float(hdu.header[header_key]), f"header:{header_key}"

    raise KeyError(
        f"Could not resolve {quantity} for {filename}. Tried source={source_setting!r}; "
        f"set bandpass.{quantity}_source or enable fallback_to_header."
    )


def bandpass_for_image(
    hdu: fits.PrimaryHDU,
    filename: Path,
    filter_name: str,
    catalog: Mapping[tuple[str, str], Mapping[str, Any]],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve pivot, width, and PHOTFLAM from configured sources."""

    bandpass_cfg = params.get("bandpass", {}) or {}
    allow_header = bool(bandpass_cfg.get("fallback_to_header", True))
    filter_key = filter_name.upper()
    instrument = infer_bandpass_instrument(hdu, filename)
    matched_instrument, entry = bandpass_entry_for(instrument, filter_key, catalog)

    header_pivot_key = str(bandpass_cfg.get("header_pivot_key", "PHOTPLAM"))
    header_width_key = str(bandpass_cfg.get("header_width_key", "PHOTBW"))
    header_photflam_key = str(bandpass_cfg.get("header_photflam_key", "PHOTFLAM"))

    pivot, pivot_source = resolve_bandpass_value(
        hdu=hdu,
        filename=filename,
        entry=entry,
        quantity="photplam",
        source_setting=str(bandpass_cfg.get("pivot_source", "filter")),
        header_key=header_pivot_key,
        external_keys=["pivot", "table_photplam"],
        allow_header=allow_header,
    )
    width, width_source = resolve_bandpass_value(
        hdu=hdu,
        filename=filename,
        entry=entry,
        quantity="photbw",
        source_setting=str(bandpass_cfg.get("width_source", "filter_rectwidth")),
        header_key=header_width_key,
        external_keys=["rectwidth", "photbw", "table_photbw"],
        allow_header=allow_header,
    )
    photflam, photflam_source = resolve_bandpass_value(
        hdu=hdu,
        filename=filename,
        entry=entry,
        quantity="photflam",
        source_setting=str(bandpass_cfg.get("photflam_source", "header")),
        header_key=header_photflam_key,
        external_keys=["table_photflam", "unit_response"],
        allow_header=allow_header,
    )

    return {
        "pivot": pivot,
        "width": width,
        "photflam": photflam,
        "instrument": matched_instrument,
        "source": pivot_source,
        "pivot_source": pivot_source,
        "width_source": width_source,
        "photflam_source": photflam_source,
    }


def annotate_bandpass(hdu: fits.PrimaryHDU, bandpass: Mapping[str, Any]) -> None:
    """Write resolved bandpass metadata to an HDU header."""

    source_label = str(bandpass["source"])
    hdu.header["CSBPSRC"] = (source_label[:48], "BP source")
    hdu.header["CSBPINS"] = (str(bandpass["instrument"]), "Bandpass instrument")
    hdu.header["CSBPWAV"] = (float(bandpass["pivot"]), "Bandpass pivot wavelength")
    hdu.header["CSBPWID"] = (float(bandpass["width"]), "Bandpass width")
    hdu.header["CSBPFLAM"] = (float(bandpass["photflam"]), "PHOTFLAM used")
    hdu.header["CSWAVSRC"] = (str(bandpass["pivot_source"])[:48], "Pivot source")
    hdu.header["CSWIDSRC"] = (str(bandpass["width_source"])[:48], "Width source")
    hdu.header["CSFLSRC"] = (str(bandpass["photflam_source"])[:48], "PHOTFLAM source")
