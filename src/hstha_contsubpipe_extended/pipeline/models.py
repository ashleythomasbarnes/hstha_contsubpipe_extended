"""Shared models for the continuum-subtraction pipeline."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from astropy.io import fits


@dataclass(frozen=True)
class ImageSet:
    """Resolved input images for one galaxy."""

    galaxy: str
    blue_filter: str
    red_filter: str
    narrow_filter: str
    blue_file: Path
    red_file: Path
    narrow_file: Path
    blue_error_file: Path | None = None
    red_error_file: Path | None = None
    narrow_error_file: Path | None = None


@dataclass(frozen=True)
class ContsubResult:
    """Output summary for one galaxy."""

    galaxy: str
    status: str
    narrow_filter: str = ""
    contsub_file: Path | None = None
    continuum_file: Path | None = None
    contsub_error_file: Path | None = None
    continuum_error_file: Path | None = None
    halpha_file: Path | None = None
    halpha_error_file: Path | None = None
    blue_file: Path | None = None
    red_file: Path | None = None
    narrow_file: Path | None = None
    blue_error_file: Path | None = None
    red_error_file: Path | None = None
    narrow_error_file: Path | None = None
    weight_blue: float | None = None
    weight_red: float | None = None
    extinction_ebv: float | None = None
    nii_to_halpha: float | None = None
    narrowband_width: float | None = None
    bandpass_source: str = ""
    message: str = ""


class PipelineStop(Exception):
    """Raised by a stage after setting the context status/result fields."""


@dataclass
class PipelineContext:
    """Mutable per-galaxy state shared by pipeline stages."""

    galaxy: str
    galaxy_config: Mapping[str, Any]
    config_dir: str | Path
    settings: Mapping[str, Any]
    params: Mapping[str, Any]
    files: Mapping[str, Any]
    dry_run: bool = False
    overwrite: bool | None = None
    stack: ExitStack | None = None

    status: str = "running"
    message: str = ""
    image_set: ImageSet | None = None

    contsub_file: Path | None = None
    continuum_file: Path | None = None
    contsub_error_file: Path | None = None
    continuum_error_file: Path | None = None
    halpha_file: Path | None = None
    halpha_error_file: Path | None = None

    narrow_hdu: fits.PrimaryHDU | None = None
    blue_hdu: fits.PrimaryHDU | None = None
    red_hdu: fits.PrimaryHDU | None = None
    narrow_error_hdu: fits.PrimaryHDU | None = None
    blue_error_hdu: fits.PrimaryHDU | None = None
    red_error_hdu: fits.PrimaryHDU | None = None

    bandpass_catalog: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    narrow_bandpass: dict[str, Any] | None = None
    blue_bandpass: dict[str, Any] | None = None
    red_bandpass: dict[str, Any] | None = None

    contsub_hdu: fits.PrimaryHDU | None = None
    continuum_hdu: fits.PrimaryHDU | None = None
    contsub_error_output_hdu: fits.PrimaryHDU | None = None
    continuum_error_output_hdu: fits.PrimaryHDU | None = None
    products: dict[Path, fits.PrimaryHDU | None] = field(default_factory=dict)

    weight_blue: float | None = None
    weight_red: float | None = None
    extinction_ebv: float | None = None
    nii_to_halpha: float | None = None
    narrowband_width: float | None = None
    bandpass_source: str = ""
    foreground_extinction: dict[str, Any] = field(default_factory=dict)
    background_corrections: list[dict[str, Any]] = field(default_factory=list)
    coverage_mask: dict[str, Any] = field(default_factory=dict)
    stage_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def output_files(self) -> list[Path]:
        """Return all planned output paths that are currently known."""

        return [
            path
            for path in (
                self.contsub_file,
                self.continuum_file,
                self.contsub_error_file,
                self.continuum_error_file,
                self.halpha_file,
                self.halpha_error_file,
            )
            if path is not None
        ]

    def to_result(self) -> ContsubResult:
        """Convert the current context state into a public result object."""

        image_set = self.image_set
        return ContsubResult(
            galaxy=self.galaxy,
            status=self.status,
            narrow_filter=image_set.narrow_filter if image_set is not None else "",
            contsub_file=self.contsub_file,
            continuum_file=self.continuum_file,
            contsub_error_file=self.contsub_error_file,
            continuum_error_file=self.continuum_error_file,
            halpha_file=self.halpha_file,
            halpha_error_file=self.halpha_error_file,
            blue_file=image_set.blue_file if image_set is not None else None,
            red_file=image_set.red_file if image_set is not None else None,
            narrow_file=image_set.narrow_file if image_set is not None else None,
            blue_error_file=image_set.blue_error_file if image_set is not None else None,
            red_error_file=image_set.red_error_file if image_set is not None else None,
            narrow_error_file=image_set.narrow_error_file if image_set is not None else None,
            weight_blue=self.weight_blue,
            weight_red=self.weight_red,
            extinction_ebv=self.extinction_ebv,
            nii_to_halpha=self.nii_to_halpha,
            narrowband_width=self.narrowband_width,
            bandpass_source=self.bandpass_source,
            message=self.message,
        )
