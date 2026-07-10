#!/usr/bin/env python
"""Compare scratch HST H-alpha products against archived hst_contsub products.

This is a standalone diagnostic script, not a pytest test. It writes a CSV and
summary plots, then exits successfully unless required directories cannot be
read or output files cannot be written.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCRATCH_DIR = Path("/Users/abarnes/Dropbox/Scratch/HSTHA")
DEFAULT_ARCHIVE_ROOT = Path(
    "/Users/abarnes/Library/CloudStorage/Dropbox/Data/Extragalactic"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "hstha_product_comparison_outputs"
DEFAULT_SCRATCH_PATTERN = "*_halpha_flux_nii_corr_log.fits"
DEFAULT_ARCHIVE_PATTERN = "*/hst_contsub/*_hst_ha_as2.fits"
DEFAULT_ALIASES = {"ngc1365": "ngc1365n"}
RATIO_HISTOGRAM_BINS = np.round(np.arange(0.5, 1.75 + 0.05, 0.05), 50)
RATIO_MAP_LIMITS = (0.8, 1.2)
SCRATCH_RE = re.compile(
    r"^(?P<galaxy>[a-z0-9]+)_(?P<filter>f\d+n)_halpha_flux_nii_corr_log\.fits$",
    re.IGNORECASE,
)


CSV_COLUMNS = [
    "scratch_galaxy",
    "archive_galaxy",
    "filter",
    "status",
    "match_mode",
    "alias_used",
    "scratch_path",
    "archive_path",
    "scratch_shape",
    "archive_shape",
    "scratch_dtype",
    "archive_dtype",
    "scratch_bunit",
    "archive_bunit",
    "scratch_finite",
    "archive_finite",
    "common_finite",
    "ratio_pixels",
    "scratch_mean",
    "scratch_median",
    "scratch_std",
    "scratch_min",
    "scratch_max",
    "archive_mean",
    "archive_median",
    "archive_std",
    "archive_min",
    "archive_max",
    "diff_mean",
    "diff_median",
    "diff_std",
    "diff_rms",
    "diff_min",
    "diff_max",
    "max_abs_diff",
    "ratio_mean",
    "ratio_median",
    "ratio_std",
    "ratio_mad",
    "ratio_p01",
    "ratio_p05",
    "ratio_p95",
    "ratio_p99",
    "frac_resid_mean",
    "frac_resid_median",
    "frac_resid_std",
    "frac_resid_mad",
    "p50_abs_frac_resid",
    "p95_abs_frac_resid",
    "p99_abs_frac_resid",
    "max_abs_frac_resid",
    "median_ratio_delta",
    "error",
]


@dataclass(frozen=True)
class ScratchFile:
    galaxy: str
    filter_name: str
    path: Path


@dataclass(frozen=True)
class ArchiveFile:
    galaxy: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare flat scratch HST H-alpha FITS files against archived "
            "hst_contsub products for matching galaxies."
        )
    )
    parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH_DIR)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scratch-pattern", default=DEFAULT_SCRATCH_PATTERN)
    parser.add_argument("--archive-pattern", default=DEFAULT_ARCHIVE_PATTERN)
    parser.add_argument(
        "--galaxy",
        action="append",
        default=[],
        help=(
            "Only compare this scratch galaxy name. May be supplied multiple times. "
            "Exact archive matching is still tried before aliases."
        ),
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="SCRATCH=ARCHIVE",
        help=(
            "Fallback galaxy-name alias. Exact matches are always tried first. "
            "May be supplied multiple times. Default includes ngc1365=ngc1365n."
        ),
    )
    parser.add_argument("--median-ratio-tol", type=float, default=1e-3)
    parser.add_argument("--p95-frac-resid-tol", type=float, default=1e-2)
    parser.add_argument("--max-frac-resid-tol", type=float, default=5e-2)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument(
        "--outlier-panels",
        type=int,
        default=4,
        help="Number of worst ok_not_close rows to show in example_outliers.png.",
    )
    parser.add_argument(
        "--plot-max-side",
        type=int,
        default=900,
        help="Maximum image side length used for the outlier plot panels.",
    )
    return parser.parse_args()


def parse_aliases(alias_args: list[str]) -> dict[str, str]:
    aliases = dict(DEFAULT_ALIASES)
    for item in alias_args:
        if "=" not in item:
            raise ValueError(f"Invalid --alias {item!r}; expected SCRATCH=ARCHIVE")
        scratch, archive = item.split("=", 1)
        scratch = scratch.strip().lower()
        archive = archive.strip().lower()
        if not scratch or not archive:
            raise ValueError(f"Invalid --alias {item!r}; expected SCRATCH=ARCHIVE")
        aliases[scratch] = archive
    return aliases


def discover_scratch_files(scratch_dir: Path, pattern: str) -> tuple[list[ScratchFile], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    scratch_files: list[ScratchFile] = []
    for path in sorted(scratch_dir.glob(pattern)):
        match = SCRATCH_RE.match(path.name)
        if match is None:
            rows.append(
                {
                    "status": "unparsed_scratch_name",
                    "scratch_path": str(path),
                    "error": f"Filename does not match expected scratch pattern: {path.name}",
                }
            )
            continue
        scratch_files.append(
            ScratchFile(
                galaxy=match.group("galaxy").lower(),
                filter_name=match.group("filter").lower(),
                path=path,
            )
        )
    return scratch_files, rows


def discover_archive_files(archive_root: Path, pattern: str) -> dict[str, ArchiveFile]:
    archive_files: dict[str, ArchiveFile] = {}
    for path in sorted(archive_root.glob(pattern)):
        name = path.name.lower()
        if not name.endswith("_hst_ha_as2.fits"):
            continue
        galaxy = name.removesuffix("_hst_ha_as2.fits")
        archive_files[galaxy] = ArchiveFile(galaxy=galaxy, path=path)
    return archive_files


def match_archive(
    scratch: ScratchFile,
    archive_files: dict[str, ArchiveFile],
    aliases: dict[str, str],
) -> tuple[ArchiveFile | None, str, str]:
    if scratch.galaxy in archive_files:
        return archive_files[scratch.galaxy], "exact", ""

    alias = aliases.get(scratch.galaxy, "")
    if alias and alias in archive_files:
        return archive_files[alias], "alias", f"{scratch.galaxy}->{alias}"

    return None, "missing", ""


def blank_row(scratch: ScratchFile | None = None, archive: ArchiveFile | None = None) -> dict[str, Any]:
    return {
        "scratch_galaxy": "" if scratch is None else scratch.galaxy,
        "archive_galaxy": "" if archive is None else archive.galaxy,
        "filter": "" if scratch is None else scratch.filter_name,
        "scratch_path": "" if scratch is None else str(scratch.path),
        "archive_path": "" if archive is None else str(archive.path),
    }


def shape_text(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return ""
    return "x".join(str(value) for value in shape)


def finite_stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def compare_pair(
    scratch: ScratchFile,
    archive: ArchiveFile,
    match_mode: str,
    alias_used: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row = blank_row(scratch, archive)
    row.update({"match_mode": match_mode, "alias_used": alias_used})

    try:
        with fits.open(scratch.path, memmap=True) as scratch_hdul, fits.open(
            archive.path, memmap=True
        ) as archive_hdul:
            scratch_hdu = scratch_hdul[0]
            archive_hdu = archive_hdul[0]
            scratch_data = scratch_hdu.data
            archive_data = archive_hdu.data

            row.update(
                {
                    "scratch_shape": shape_text(None if scratch_data is None else scratch_data.shape),
                    "archive_shape": shape_text(None if archive_data is None else archive_data.shape),
                    "scratch_dtype": "" if scratch_data is None else str(scratch_data.dtype),
                    "archive_dtype": "" if archive_data is None else str(archive_data.dtype),
                    "scratch_bunit": scratch_hdu.header.get("BUNIT", ""),
                    "archive_bunit": archive_hdu.header.get("BUNIT", ""),
                }
            )

            if scratch_data is None or archive_data is None:
                row.update({"status": "read_error", "error": "Primary HDU data is missing"})
                return row

            if scratch_data.shape != archive_data.shape:
                row.update({"status": "shape_mismatch"})
                return row

            scratch_array = np.asarray(scratch_data)
            archive_array = np.asarray(archive_data)
            scratch_finite = np.isfinite(scratch_array)
            archive_finite = np.isfinite(archive_array)
            common = scratch_finite & archive_finite
            ratio_mask = common & (archive_array != 0)

            row.update(
                {
                    "scratch_finite": int(np.count_nonzero(scratch_finite)),
                    "archive_finite": int(np.count_nonzero(archive_finite)),
                    "common_finite": int(np.count_nonzero(common)),
                    "ratio_pixels": int(np.count_nonzero(ratio_mask)),
                }
            )
            row.update(finite_stats(scratch_array, "scratch"))
            row.update(finite_stats(archive_array, "archive"))

            if not np.any(common) or not np.any(ratio_mask):
                row.update({"status": "no_valid_pixels"})
                return row

            scratch_common = scratch_array[common].astype(np.float64, copy=False)
            archive_common = archive_array[common].astype(np.float64, copy=False)
            diff = scratch_common - archive_common

            row.update(
                {
                    "diff_mean": float(np.mean(diff)),
                    "diff_median": float(np.median(diff)),
                    "diff_std": float(np.std(diff)),
                    "diff_rms": float(np.sqrt(np.mean(diff**2))),
                    "diff_min": float(np.min(diff)),
                    "diff_max": float(np.max(diff)),
                    "max_abs_diff": float(np.max(np.abs(diff))),
                }
            )

            ratio = (
                scratch_array[ratio_mask].astype(np.float64, copy=False)
                / archive_array[ratio_mask].astype(np.float64, copy=False)
            )
            frac_resid = ratio - 1.0
            abs_frac_resid = np.abs(frac_resid)
            ratio_median = float(np.median(ratio))
            frac_resid_median = float(np.median(frac_resid))

            row.update(
                {
                    "ratio_mean": float(np.mean(ratio)),
                    "ratio_median": ratio_median,
                    "ratio_std": float(np.std(ratio)),
                    "ratio_mad": float(np.median(np.abs(ratio - ratio_median))),
                    "ratio_p01": percentile(ratio, 1),
                    "ratio_p05": percentile(ratio, 5),
                    "ratio_p95": percentile(ratio, 95),
                    "ratio_p99": percentile(ratio, 99),
                    "frac_resid_mean": float(np.mean(frac_resid)),
                    "frac_resid_median": frac_resid_median,
                    "frac_resid_std": float(np.std(frac_resid)),
                    "frac_resid_mad": float(
                        np.median(np.abs(frac_resid - frac_resid_median))
                    ),
                    "p50_abs_frac_resid": percentile(abs_frac_resid, 50),
                    "p95_abs_frac_resid": percentile(abs_frac_resid, 95),
                    "p99_abs_frac_resid": percentile(abs_frac_resid, 99),
                    "max_abs_frac_resid": float(np.max(abs_frac_resid)),
                    "median_ratio_delta": ratio_median - 1.0,
                }
            )

            close = (
                abs(row["median_ratio_delta"]) <= args.median_ratio_tol
                and row["p95_abs_frac_resid"] <= args.p95_frac_resid_tol
                and row["max_abs_frac_resid"] <= args.max_frac_resid_tol
            )
            row["status"] = "ok_close" if close else "ok_not_close"
            return row

    except Exception as exc:
        row.update({"status": "read_error", "error": f"{type(exc).__name__}: {exc}"})
        return row


def make_missing_row(scratch: ScratchFile, match_mode: str, alias_used: str) -> dict[str, Any]:
    row = blank_row(scratch, None)
    row.update(
        {
            "status": "missing_archive",
            "match_mode": match_mode,
            "alias_used": alias_used,
            "error": f"No archive match for scratch galaxy {scratch.galaxy}",
        }
    )
    return row


def format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.12g}"
    return value


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: format_csv_value(row.get(column, "")) for column in CSV_COLUMNS}
            )


def ok_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") in {"ok_close", "ok_not_close"}]


def float_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def make_ratio_summary_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows_to_plot = sorted(ok_rows(rows), key=lambda row: row["scratch_galaxy"])
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(rows_to_plot)), 5), constrained_layout=True)

    if rows_to_plot:
        x = np.arange(len(rows_to_plot))
        med = np.array([float_value(row, "ratio_median") for row in rows_to_plot])
        p05 = np.array([float_value(row, "ratio_p05") for row in rows_to_plot])
        p95 = np.array([float_value(row, "ratio_p95") for row in rows_to_plot])
        colors = ["#2f6f9f" if row["status"] == "ok_close" else "#c73e1d" for row in rows_to_plot]
        ax.errorbar(x, med, yerr=[med - p05, p95 - med], fmt="none", ecolor="0.35", alpha=0.7)
        ax.scatter(x, med, c=colors, s=36, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([row["scratch_galaxy"] for row in rows_to_plot], rotation=45, ha="right")

    ax.axhline(1.0, color="0.2", lw=1, ls="--")
    ax.set_ylabel("scratch / archive")
    ax.set_title("Median ratio with 5th-95th percentile range")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_fractional_residual_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows_to_plot = sorted(ok_rows(rows), key=lambda row: row["scratch_galaxy"])
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(rows_to_plot)), 5), constrained_layout=True)

    if rows_to_plot:
        x = np.arange(len(rows_to_plot))
        width = 0.25
        p50 = np.array([float_value(row, "p50_abs_frac_resid") for row in rows_to_plot])
        p95 = np.array([float_value(row, "p95_abs_frac_resid") for row in rows_to_plot])
        p99 = np.array([float_value(row, "p99_abs_frac_resid") for row in rows_to_plot])
        ax.bar(x - width, p50, width=width, label="p50", color="#4d7c8a")
        ax.bar(x, p95, width=width, label="p95", color="#d9a441")
        ax.bar(x + width, p99, width=width, label="p99", color="#c73e1d")
        ax.set_xticks(x)
        ax.set_xticklabels([row["scratch_galaxy"] for row in rows_to_plot], rotation=45, ha="right")
        ax.legend()

    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylabel("|(scratch - archive) / archive|")
    ax.set_title("Absolute fractional residual summary")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def downsample_for_plot(values: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return values
    step = max(1, int(np.ceil(max(values.shape) / max_side)))
    return values[::step, ::step]


def make_example_outliers_plot(rows: list[dict[str, Any]], output_path: Path, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    candidates = [
        row
        for row in ok_rows(rows)
        if row.get("status") == "ok_not_close" and row.get("scratch_path") and row.get("archive_path")
    ]
    candidates = sorted(
        candidates,
        key=lambda row: max(
            abs(float_value(row, "median_ratio_delta")),
            float_value(row, "p95_abs_frac_resid"),
        ),
        reverse=True,
    )[: args.outlier_panels]

    if not candidates:
        fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
        ax.text(0.5, 0.5, "No ok_not_close rows to plot", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    fig, axes = plt.subplots(
        len(candidates),
        4,
        figsize=(14, 3.2 * len(candidates)),
        squeeze=False,
        constrained_layout=True,
    )
    column_titles = ["scratch", "archive", "ratio", "difference"]
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title)

    for row_index, row in enumerate(candidates):
        try:
            with fits.open(row["scratch_path"], memmap=True) as scratch_hdul, fits.open(
                row["archive_path"], memmap=True
            ) as archive_hdul:
                scratch = np.asarray(scratch_hdul[0].data)
                archive = np.asarray(archive_hdul[0].data)
                ratio = np.full(scratch.shape, np.nan, dtype=np.float32)
                valid = np.isfinite(scratch) & np.isfinite(archive) & (archive != 0)
                ratio[valid] = scratch[valid] / archive[valid]
                diff = scratch - archive
                arrays = [scratch, archive, ratio, diff]
        except Exception:
            for ax in axes[row_index]:
                ax.text(0.5, 0.5, "plot read error", ha="center", va="center")
                ax.set_axis_off()
            continue

        for col_index, values in enumerate(arrays):
            ax = axes[row_index, col_index]
            image_values = downsample_for_plot(values, args.plot_max_side)
            vmin, vmax = RATIO_MAP_LIMITS if col_index == 2 else robust_limits(image_values)
            cmap = "RdBu_r" if col_index in {2, 3} else "viridis"
            ax.imshow(image_values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_axis_off()
            if col_index == 0:
                label = row["scratch_galaxy"]
                if row.get("alias_used"):
                    label += f" ({row['alias_used']})"
                ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=45)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_ratio_histogram_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    histogram_paths: list[Path] = []

    for row in sorted(ok_rows(rows), key=lambda item: item["scratch_galaxy"]):
        try:
            with fits.open(row["scratch_path"], memmap=True) as scratch_hdul, fits.open(
                row["archive_path"], memmap=True
            ) as archive_hdul:
                scratch = np.asarray(scratch_hdul[0].data)
                archive = np.asarray(archive_hdul[0].data)
                valid = np.isfinite(scratch) & np.isfinite(archive) & (archive != 0)
                ratio = scratch[valid].astype(np.float64, copy=False) / archive[valid].astype(
                    np.float64,
                    copy=False,
                )
        except Exception:
            continue

        if ratio.size == 0:
            continue

        in_range = (ratio >= RATIO_HISTOGRAM_BINS[0]) & (ratio <= RATIO_HISTOGRAM_BINS[-1])
        median_ratio = float_value(row, "ratio_median")
        fraction_in_range = float(np.count_nonzero(in_range) / ratio.size)

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.hist(ratio, bins=RATIO_HISTOGRAM_BINS, color="#4d7c8a", edgecolor="white")
        ax.axvline(1.0, color="0.2", lw=1.2, ls="--", label="ratio = 1")
        if np.isfinite(median_ratio):
            ax.axvline(
                median_ratio,
                color="#c73e1d",
                lw=1.2,
                label=f"median = {median_ratio:.4g}",
            )
        ax.set_xlim(RATIO_HISTOGRAM_BINS[0], RATIO_HISTOGRAM_BINS[-1])
        ax.set_xlabel("scratch / archive")
        ax.set_ylabel("Pixels")
        title = f"{row['scratch_galaxy']} ratio histogram"
        if row.get("alias_used"):
            title += f" ({row['alias_used']})"
        ax.set_title(title)
        ax.text(
            0.02,
            0.96,
            f"bin width = 0.05\nshown range = 0.5-1.75\nin range = {fraction_in_range:.1%}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.25)

        output_path = output_dir / f"{row['scratch_galaxy']}_ratio_histogram.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        histogram_paths.append(output_path)

    return histogram_paths


def make_per_galaxy_map_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    map_paths: list[Path] = []

    for row in sorted(ok_rows(rows), key=lambda item: item["scratch_galaxy"]):
        try:
            with fits.open(row["scratch_path"], memmap=True) as scratch_hdul, fits.open(
                row["archive_path"], memmap=True
            ) as archive_hdul:
                scratch = np.asarray(scratch_hdul[0].data)
                archive = np.asarray(archive_hdul[0].data)
                valid = np.isfinite(scratch) & np.isfinite(archive) & (archive != 0)
                ratio = np.full(scratch.shape, np.nan, dtype=np.float32)
                ratio[valid] = scratch[valid] / archive[valid]
                diff = scratch - archive
        except Exception:
            continue

        panels = [
            ("scratch", scratch, "viridis"),
            ("archive", archive, "viridis"),
            ("scratch / archive", ratio, "RdBu_r"),
            ("scratch - archive", diff, "RdBu_r"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        title = f"{row['scratch_galaxy']} map comparison"
        if row.get("alias_used"):
            title += f" ({row['alias_used']})"
        fig.suptitle(title)

        for ax, (panel_title, values, cmap) in zip(axes.flat, panels):
            image_values = downsample_for_plot(values, args.plot_max_side)
            vmin, vmax = (
                RATIO_MAP_LIMITS
                if panel_title == "scratch / archive"
                else robust_limits(image_values)
            )
            image = ax.imshow(image_values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(panel_title)
            ax.set_axis_off()
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            colorbar.ax.tick_params(labelsize=8)

        output_path = output_dir / f"{row['scratch_galaxy']}_map_comparison.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        map_paths.append(output_path)

    return map_paths


def print_summary(
    rows: list[dict[str, Any]],
    scratch_count: int,
    archive_files: dict[str, ArchiveFile],
    matched_archive_names: set[str],
    aliases: dict[str, str],
    args: argparse.Namespace,
    csv_path: Path,
    ratio_plot_path: Path,
    residual_plot_path: Path,
    outlier_plot_path: Path,
    histogram_paths: list[Path],
    map_paths: list[Path],
) -> None:
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row.get("status", "unknown")] = statuses.get(row.get("status", "unknown"), 0) + 1

    archive_extras = sorted(set(archive_files) - matched_archive_names)
    if args.galaxy:
        archive_extras = []
    alias_rows = [row for row in rows if row.get("match_mode") == "alias"]
    unit_mismatches = [
        row
        for row in ok_rows(rows)
        if str(row.get("scratch_bunit", "")) != str(row.get("archive_bunit", ""))
    ]

    print("\nHST H-alpha product comparison")
    print("==============================")
    print(f"Scratch directory: {args.scratch_dir}")
    print(f"Archive root: {args.archive_root}")
    print(f"Scratch FITS discovered: {scratch_count}")
    print(f"Archive FITS discovered: {len(archive_files)}")
    print(f"CSV: {csv_path}")
    print(f"Ratio plot: {ratio_plot_path}")
    print(f"Residual plot: {residual_plot_path}")
    print(f"Outlier plot: {outlier_plot_path}")
    print(f"Ratio histograms: {len(histogram_paths)} files in {args.output_dir / 'ratio_histograms'}")
    print(f"Map plots: {len(map_paths)} files in {args.output_dir / 'map_plots'}")

    print("\nStatuses")
    for status in sorted(statuses):
        print(f"  {status}: {statuses[status]}")

    if aliases:
        print("\nFallback aliases")
        for scratch_name, archive_name in sorted(aliases.items()):
            print(f"  {scratch_name} -> {archive_name}")

    if alias_rows:
        print("\nAlias matches used")
        for row in alias_rows:
            print(
                f"  {row['scratch_galaxy']} matched to {row['archive_galaxy']} "
                f"via {row['alias_used']}"
            )

    if archive_extras:
        print("\nArchive extras not matched by scratch files")
        for galaxy in archive_extras:
            print(f"  {galaxy}: {archive_files[galaxy].path}")

    for status, title in [
        ("missing_archive", "Missing archive matches"),
        ("shape_mismatch", "Shape mismatches"),
        ("read_error", "Read errors"),
    ]:
        subset = [row for row in rows if row.get("status") == status]
        if subset:
            print(f"\n{title}")
            for row in subset:
                label = row.get("scratch_galaxy") or row.get("scratch_path")
                detail = row.get("error", "")
                if status == "shape_mismatch":
                    detail = f"{row.get('scratch_shape')} vs {row.get('archive_shape')}"
                print(f"  {label}: {detail}")

    if unit_mismatches:
        print("\nBUNIT mismatches among compared rows")
        for row in unit_mismatches:
            print(
                f"  {row['scratch_galaxy']}: scratch={row.get('scratch_bunit')!r}, "
                f"archive={row.get('archive_bunit')!r}"
            )

    comparable = ok_rows(rows)
    if comparable:
        print(f"\nTop {min(args.top_n, len(comparable))} median-ratio outliers")
        for row in sorted(
            comparable, key=lambda item: abs(float_value(item, "median_ratio_delta")), reverse=True
        )[: args.top_n]:
            print(
                f"  {row['scratch_galaxy']:8s} status={row['status']:12s} "
                f"median_ratio={float_value(row, 'ratio_median'):.6g} "
                f"delta={float_value(row, 'median_ratio_delta'):+.6g}"
            )

        print(f"\nTop {min(args.top_n, len(comparable))} p95 fractional-residual outliers")
        for row in sorted(
            comparable, key=lambda item: float_value(item, "p95_abs_frac_resid"), reverse=True
        )[: args.top_n]:
            print(
                f"  {row['scratch_galaxy']:8s} status={row['status']:12s} "
                f"p95_abs_frac_resid={float_value(row, 'p95_abs_frac_resid'):.6g} "
                f"max_abs_frac_resid={float_value(row, 'max_abs_frac_resid'):.6g}"
            )


def main() -> int:
    args = parse_args()
    aliases = parse_aliases(args.alias)

    if not args.scratch_dir.exists():
        raise FileNotFoundError(f"Scratch directory not found: {args.scratch_dir}")
    if not args.archive_root.exists():
        raise FileNotFoundError(f"Archive root not found: {args.archive_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output_dir / ".cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    scratch_files, rows = discover_scratch_files(args.scratch_dir, args.scratch_pattern)
    if args.galaxy:
        requested_galaxies = {galaxy.lower() for galaxy in args.galaxy}
        scratch_files = [scratch for scratch in scratch_files if scratch.galaxy in requested_galaxies]
        missing_requested = requested_galaxies - {scratch.galaxy for scratch in scratch_files}
        for galaxy in sorted(missing_requested):
            rows.append(
                {
                    "scratch_galaxy": galaxy,
                    "status": "missing_scratch",
                    "match_mode": "requested",
                    "error": f"No scratch file found for requested galaxy {galaxy}",
                }
            )
    archive_files = discover_archive_files(args.archive_root, args.archive_pattern)
    matched_archive_names: set[str] = set()

    for scratch in scratch_files:
        archive, match_mode, alias_used = match_archive(scratch, archive_files, aliases)
        if archive is None:
            rows.append(make_missing_row(scratch, match_mode, alias_used))
            continue
        matched_archive_names.add(archive.galaxy)
        rows.append(compare_pair(scratch, archive, match_mode, alias_used, args))

    csv_path = args.output_dir / "hstha_product_comparison.csv"
    ratio_plot_path = args.output_dir / "ratio_summary.png"
    residual_plot_path = args.output_dir / "fractional_residual_summary.png"
    outlier_plot_path = args.output_dir / "example_outliers.png"
    histogram_dir = args.output_dir / "ratio_histograms"
    map_dir = args.output_dir / "map_plots"

    write_csv(rows, csv_path)
    make_ratio_summary_plot(rows, ratio_plot_path)
    make_fractional_residual_plot(rows, residual_plot_path)
    make_example_outliers_plot(rows, outlier_plot_path, args)
    histogram_paths = make_ratio_histogram_plots(rows, histogram_dir)
    map_paths = make_per_galaxy_map_plots(rows, map_dir, args)
    print_summary(
        rows,
        len(scratch_files),
        archive_files,
        matched_archive_names,
        aliases,
        args,
        csv_path,
        ratio_plot_path,
        residual_plot_path,
        outlier_plot_path,
        histogram_paths,
        map_paths,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
