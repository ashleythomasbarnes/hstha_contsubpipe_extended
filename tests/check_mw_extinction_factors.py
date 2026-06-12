#!/usr/bin/env python
"""Compare current MW extinction factors against the previous hard-coded values.

This is a standalone diagnostic script, not a pytest test. It writes a CSV and
summary plot, then exits successfully unless required inputs cannot be read.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hstha_contsubpipe_extended.pipeline.bandpass import load_bandpass_catalog  # noqa: E402
from hstha_contsubpipe_extended.pipeline.extinction import (  # noqa: E402
    extinction_correction_factor,
    normalize_galaxy_name,
)


DEFAULT_SAMPLE_TABLE = Path(
    "/Users/abarnes/Library/CloudStorage/Dropbox/Data/Extragalactic/catalogues/"
    "sample_table/phangs_sample_table_v1p6.fits"
)
DEFAULT_FILTER_ROOT = Path(
    "/Users/abarnes/Library/CloudStorage/Dropbox/Data/Extragalactic/misc/hst_filters"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "mw_extinction_check_outputs"


OLD_AV = {
    "ic5332": 0.046,
    "ngc0628": 0.192,
    "ngc1087": 0.095,
    "ngc1300": 0.083,
    "ngc1365": 0.056,
    "ngc1385": 0.055,
    "ngc1433": 0.025,
    "ngc1512": 0.029,
    "ngc1566": 0.025,
    "ngc1672": 0.064,
    "ngc2835": 0.275,
    "ngc3351": 0.076,
    "ngc3627": 0.091,
    "ngc4254": 0.106,
    "ngc4303": 0.061,
    "ngc4321": 0.072,
    "ngc5068": 0.281,
    "ngc7496": 0.188,
}


OLD_R_COEFFICIENTS = {
    "UVIS": {
        "F547M": 2.650,
        "F555W": 2.855,
        "F658N": 2.2,
        "F657N": 2.2,
        "F814W": 1.536,
        "V": 2.742,
    },
    "ACS": {
        "F550M": 2.620,
        "F555W": 2.792,
        "F658N": 2.2,
        "F657N": 2.2,
        "F814W": 1.526,
        "V": 2.742,
    },
}


FILTERS_TO_CHECK = {
    "UVIS": ["F547M", "F555W", "F657N", "F658N", "F814W"],
    "ACS": ["F550M", "F555W", "F657N", "F658N", "F814W"],
}


CSV_COLUMNS = [
    "galaxy",
    "sample_table_name",
    "instrument",
    "filter",
    "status",
    "old_av",
    "sample_ebv",
    "sample_av",
    "pivot_angstrom",
    "old_factor",
    "current_factor",
    "delta_factor",
    "frac_delta",
    "frac_delta_percent",
    "bandpass_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current sample-table MW extinction correction factors "
            "against the previous hard-coded calculation."
        )
    )
    parser.add_argument("--sample-table", type=Path, default=DEFAULT_SAMPLE_TABLE)
    parser.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rv", type=float, default=3.1)
    parser.add_argument("--ebv-column", default="mwext_sf11")
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def old_extinction_factor(galaxy: str, instrument: str, filter_name: str) -> float:
    coeffs = OLD_R_COEFFICIENTS[instrument]
    old_a_lambda = OLD_AV[galaxy] * coeffs[filter_name] / coeffs["V"]
    return float(10 ** (0.4 * old_a_lambda))


def build_sample_lookup(
    table: Table,
    name_column: str,
    ebv_column: str,
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in table:
        normalized = normalize_galaxy_name(row[name_column])
        lookup[normalized] = {
            "sample_table_name": str(row[name_column]),
            "sample_ebv": float(row[ebv_column]),
        }
    return lookup


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[Any, Any]]:
    if not args.sample_table.exists():
        raise FileNotFoundError(f"Sample table not found: {args.sample_table}")
    if not args.filter_root.exists():
        raise FileNotFoundError(f"Filter root not found: {args.filter_root}")

    table = Table.read(args.sample_table)
    for column in (args.name_column, args.ebv_column):
        if column not in table.colnames:
            raise KeyError(f"Column {column!r} not found in {args.sample_table}")

    sample_lookup = build_sample_lookup(table, args.name_column, args.ebv_column)
    bandpass_catalog = load_bandpass_catalog(
        {
            "bandpass": {
                "filter_root": str(args.filter_root),
                "table_file": "filter_table.fits",
            }
        }
    )
    return sample_lookup, bandpass_catalog


def make_rows(
    sample_lookup: dict[str, dict[str, Any]],
    bandpass_catalog: dict[Any, Any],
    rv: float,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    skipped_filters: list[str] = []
    missing_galaxies: list[str] = []

    for galaxy in OLD_AV:
        normalized_galaxy = normalize_galaxy_name(galaxy)
        sample_entry = sample_lookup.get(normalized_galaxy)
        if sample_entry is None:
            missing_galaxies.append(galaxy)
            continue

        sample_ebv = sample_entry["sample_ebv"]
        sample_av = rv * sample_ebv

        for instrument, filters in FILTERS_TO_CHECK.items():
            for filter_name in filters:
                bandpass_entry = bandpass_catalog.get((instrument, filter_name))
                base_row = {
                    "galaxy": galaxy,
                    "sample_table_name": sample_entry["sample_table_name"],
                    "instrument": instrument,
                    "filter": filter_name,
                    "old_av": OLD_AV[galaxy],
                    "sample_ebv": sample_ebv,
                    "sample_av": sample_av,
                }

                if bandpass_entry is None or "pivot" not in bandpass_entry:
                    label = f"{instrument} {filter_name}"
                    if label not in skipped_filters:
                        skipped_filters.append(label)
                    rows.append(
                        {
                            **base_row,
                            "status": "missing_current_bandpass",
                            "pivot_angstrom": np.nan,
                            "old_factor": np.nan,
                            "current_factor": np.nan,
                            "delta_factor": np.nan,
                            "frac_delta": np.nan,
                            "frac_delta_percent": np.nan,
                            "bandpass_source": "",
                        }
                    )
                    continue

                pivot = float(bandpass_entry["pivot"])
                old_factor = old_extinction_factor(galaxy, instrument, filter_name)
                current_factor = extinction_correction_factor(pivot, sample_ebv, r_v=rv)
                delta = current_factor - old_factor
                frac_delta = delta / old_factor
                rows.append(
                    {
                        **base_row,
                        "status": "ok",
                        "pivot_angstrom": pivot,
                        "old_factor": old_factor,
                        "current_factor": current_factor,
                        "delta_factor": delta,
                        "frac_delta": frac_delta,
                        "frac_delta_percent": 100.0 * frac_delta,
                        "bandpass_source": str(bandpass_entry.get("source", "")),
                    }
                )

    return rows, missing_galaxies, skipped_filters


def finite_ok_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["status"] == "ok" and np.isfinite(float(row["frac_delta"]))
    ]


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def print_summary(
    rows: list[dict[str, Any]],
    missing_galaxies: list[str],
    skipped_filters: list[str],
    rv: float,
    top_n: int,
    csv_path: Path,
    plot_path: Path,
) -> None:
    ok_rows = finite_ok_rows(rows)
    av_rows = {}
    for row in rows:
        av_rows.setdefault(
            row["galaxy"],
            (float(row["old_av"]), float(row["sample_ebv"]), float(row["sample_av"])),
        )

    print("\nMW extinction factor check")
    print("==========================")
    print(f"R_V: {rv:g}")
    print(f"Compared rows: {len(ok_rows)}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")

    print("\nA_V comparison from old constants and sample table")
    print("galaxy     old_Av   sample_E(B-V)   sample_Av   delta_Av")
    for galaxy, (old_av, sample_ebv, sample_av) in av_rows.items():
        print(
            f"{galaxy:8s}  {old_av:7.4f}  {sample_ebv:13.5f}  "
            f"{sample_av:9.4f}  {sample_av - old_av:+9.4f}"
        )

    if ok_rows:
        max_row = max(ok_rows, key=lambda row: abs(float(row["frac_delta"])))
        print("\nLargest correction-factor difference")
        print(
            f"{max_row['galaxy']} {max_row['instrument']} {max_row['filter']}: "
            f"current={max_row['current_factor']:.8f}, old={max_row['old_factor']:.8f}, "
            f"frac_delta={100.0 * float(max_row['frac_delta']):+.3f}%"
        )

        print(f"\nTop {min(top_n, len(ok_rows))} outliers by absolute fractional difference")
        print("galaxy     inst  filter  old_factor  current_factor  frac_delta_%")
        for row in sorted(ok_rows, key=lambda item: abs(float(item["frac_delta"])), reverse=True)[
            :top_n
        ]:
            print(
                f"{row['galaxy']:8s}  {row['instrument']:4s}  {row['filter']:6s}  "
                f"{row['old_factor']:10.6f}  {row['current_factor']:14.6f}  "
                f"{row['frac_delta_percent']:+12.4f}"
            )

        non_ngc7496 = [
            row for row in ok_rows if normalize_galaxy_name(row["galaxy"]) != "ngc7496"
        ]
        if non_ngc7496:
            max_non = max(non_ngc7496, key=lambda row: abs(float(row["frac_delta"])))
            print(
                "\nMax absolute fractional difference excluding ngc7496: "
                f"{100.0 * abs(float(max_non['frac_delta'])):.3f}% "
                f"({max_non['galaxy']} {max_non['instrument']} {max_non['filter']})"
            )

    if skipped_filters:
        print("\nSkipped filters with no current local bandpass pivot:")
        for label in skipped_filters:
            print(f"  - {label}")

    if missing_galaxies:
        print("\nMissing old-reference galaxies in sample table:")
        for galaxy in missing_galaxies:
            print(f"  - {galaxy}")


def make_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ok_rows = finite_ok_rows(rows)
    galaxies = list(OLD_AV)
    filter_labels = [
        f"{instrument} {filter_name}"
        for instrument, filters in FILTERS_TO_CHECK.items()
        for filter_name in filters
    ]

    old_av = np.array([OLD_AV[galaxy] for galaxy in galaxies], dtype=float)
    sample_av_by_galaxy = {}
    for row in rows:
        sample_av_by_galaxy.setdefault(row["galaxy"], float(row["sample_av"]))
    sample_av = np.array([sample_av_by_galaxy[galaxy] for galaxy in galaxies], dtype=float)

    heatmap = np.full((len(galaxies), len(filter_labels)), np.nan, dtype=float)
    max_abs_by_galaxy = np.zeros(len(galaxies), dtype=float)
    galaxy_index = {galaxy: idx for idx, galaxy in enumerate(galaxies)}
    filter_index = {label: idx for idx, label in enumerate(filter_labels)}

    for row in ok_rows:
        label = f"{row['instrument']} {row['filter']}"
        i = galaxy_index[row["galaxy"]]
        j = filter_index[label]
        frac_percent = float(row["frac_delta_percent"])
        heatmap[i, j] = frac_percent
        max_abs_by_galaxy[i] = max(max_abs_by_galaxy[i], abs(frac_percent))

    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.55, 1.0])

    ax_av = fig.add_subplot(grid[0])
    ax_heat = fig.add_subplot(grid[1])
    ax_bar = fig.add_subplot(grid[2])

    ax_av.scatter(old_av, sample_av, color="#2f6f9f", s=38)
    min_av = float(np.nanmin([np.nanmin(old_av), np.nanmin(sample_av)]))
    max_av = float(np.nanmax([np.nanmax(old_av), np.nanmax(sample_av)]))
    pad = 0.02
    ax_av.plot([min_av - pad, max_av + pad], [min_av - pad, max_av + pad], color="0.35", lw=1)
    ax_av.set_xlabel("Old hard-coded A_V")
    ax_av.set_ylabel("Sample-table A_V = R_V * mwext_sf11")
    ax_av.set_title("Foreground extinction input comparison")
    ax_av.grid(alpha=0.25)
    for galaxy, old_value, sample_value in zip(galaxies, old_av, sample_av):
        if galaxy == "ngc7496" or abs(sample_value - old_value) > 0.01:
            ax_av.annotate(
                galaxy,
                (old_value, sample_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    finite_heat = heatmap[np.isfinite(heatmap)]
    vmax = max(1.0, float(np.nanmax(np.abs(finite_heat)))) if finite_heat.size else 1.0
    image = ax_heat.imshow(heatmap, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_heat.set_xticks(np.arange(len(filter_labels)))
    ax_heat.set_xticklabels(filter_labels, rotation=45, ha="right")
    ax_heat.set_yticks(np.arange(len(galaxies)))
    ax_heat.set_yticklabels(galaxies)
    ax_heat.set_title("Current minus old correction factor, fractional difference (%)")
    colorbar = fig.colorbar(image, ax=ax_heat, shrink=0.9)
    colorbar.set_label("Fractional difference (%)")

    bar_colors = ["#c73e1d" if galaxy == "ngc7496" else "#4d7c8a" for galaxy in galaxies]
    ax_bar.bar(galaxies, max_abs_by_galaxy, color=bar_colors)
    ax_bar.set_ylabel("Max |fractional difference| (%)")
    ax_bar.set_title("Largest filter-level correction-factor difference per galaxy")
    ax_bar.tick_params(axis="x", rotation=45)
    ax_bar.grid(axis="y", alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output_dir / ".cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    sample_lookup, bandpass_catalog = load_inputs(args)
    rows, missing_galaxies, skipped_filters = make_rows(sample_lookup, bandpass_catalog, args.rv)

    if missing_galaxies:
        raise RuntimeError(
            "Required old-reference galaxies are missing from the sample table: "
            + ", ".join(missing_galaxies)
        )

    csv_path = args.output_dir / "mw_extinction_factor_comparison.csv"
    plot_path = args.output_dir / "mw_extinction_factor_comparison.png"

    write_csv(rows, csv_path)
    make_plot(rows, plot_path)
    print_summary(
        rows,
        missing_galaxies,
        skipped_filters,
        args.rv,
        args.top_n,
        csv_path,
        plot_path,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
