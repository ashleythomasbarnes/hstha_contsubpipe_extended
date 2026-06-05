"""Continuum-subtraction run manifest writing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from hstha_contsubpipe_extended.core.config import get_configs
from hstha_contsubpipe_extended.core.context import build_generic_context
from hstha_contsubpipe_extended.core.files import resolve_file

from .models import ContsubResult


def write_manifest(results: Sequence[ContsubResult], config_dir: str | Path) -> Path:
    """Write a CSV manifest summarizing files, products, and failures."""

    paths, params, files = get_configs(config_dir=config_dir)
    context = build_generic_context(paths=paths, params=params)
    manifest_file = Path(resolve_file(files_cfg=files, key="outputs.manifest", context=context))
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "galaxy",
        "status",
        "narrow_filter",
        "blue_file",
        "red_file",
        "narrow_file",
        "contsub_file",
        "continuum_file",
        "contsub_error_file",
        "continuum_error_file",
        "halpha_file",
        "halpha_error_file",
        "blue_error_file",
        "red_error_file",
        "narrow_error_file",
        "weight_blue",
        "weight_red",
        "extinction_ebv",
        "nii_to_halpha",
        "narrowband_width",
        "bandpass_source",
        "message",
    ]
    with manifest_file.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: str(getattr(result, field) or "") for field in fields})
    return manifest_file
