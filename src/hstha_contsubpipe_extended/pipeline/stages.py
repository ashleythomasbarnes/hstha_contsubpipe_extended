"""Stage registry and built-in continuum-subtraction stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from astropy.io import fits

from .bandpass import annotate_bandpass, bandpass_for_image, load_bandpass_catalog
from .discovery import resolve_image_set
from .extinction import apply_foreground_extinction_to_hdu_pair, foreground_ebv
from .fits_ops import (
    apply_background_correction,
    convert_hst_count_rate_to_flux_density,
    convert_inverse_variance_to_error,
    match_spatial_coverage,
    preprocess_hst_data,
)
from .models import PipelineContext, PipelineStop
from .products import build_products, plan_output_paths, write_products
from .subtraction import linear_continuum_subtract


class Stage(Protocol):
    """A pipeline stage that mutates a per-galaxy context."""

    name: str

    def run(self, context: PipelineContext) -> None:
        """Run the stage."""


@dataclass(frozen=True)
class FunctionStage:
    """Small adapter for registering ordinary functions as stages."""

    name: str
    function: Callable[[PipelineContext], None]

    def run(self, context: PipelineContext) -> None:
        """Run the wrapped stage function."""

        self.function(context)


_STAGES: dict[str, Stage] = {}
_DEFAULT_STAGE_NAMES = [
    "resolve_inputs",
    "plan_outputs",
    "skip_existing_outputs",
    "load_images",
    "load_errors",
    "preprocess",
    "resolve_bandpasses",
    "calibrate_flux_density",
    "apply_background_corrections",
    "apply_foreground_extinction",
    "match_spatial_coverage",
    "subtract_continuum",
    "build_products",
    "write_outputs",
]


def register_stage(stage: Stage) -> Stage:
    """Register a stage by name and return it."""

    if not stage.name:
        raise ValueError("Stage name cannot be empty")
    _STAGES[stage.name] = stage
    return stage


def get_stage(name: str) -> Stage:
    """Return a registered stage by name."""

    try:
        return _STAGES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown pipeline stage {name!r}. Available stages: {', '.join(sorted(_STAGES))}"
        ) from exc


def default_stage_names() -> list[str]:
    """Return the default stage sequence."""

    return list(_DEFAULT_STAGE_NAMES)


def build_stage_sequence(stage_names: Sequence[str] | None = None) -> list[Stage]:
    """Build a concrete stage sequence from names."""

    names = list(stage_names) if stage_names is not None else default_stage_names()
    return [get_stage(name) for name in names]


def _require_stack(context: PipelineContext):
    if context.stack is None:
        raise ValueError("PipelineContext.stack is required for FITS-loading stages")
    return context.stack


def _require_image_set(context: PipelineContext):
    if context.image_set is None:
        raise ValueError("Pipeline image set has not been resolved")
    return context.image_set


def _stage_resolve_inputs(context: PipelineContext) -> None:
    context.image_set = resolve_image_set(
        context.galaxy,
        context.galaxy_config,
        config_dir=context.config_dir,
    )


def _stage_plan_outputs(context: PipelineContext) -> None:
    plan_output_paths(context)
    if context.dry_run:
        context.status = "planned"
        raise PipelineStop()


def _stage_skip_existing_outputs(context: PipelineContext) -> None:
    if context.overwrite is None:
        context.overwrite = bool(context.settings.get("overwrite", False))

    if any(path.exists() for path in context.output_files) and not context.overwrite:
        context.status = "skipped"
        context.message = "output exists; use overwrite=True to replace it"
        raise PipelineStop()


def _stage_load_images(context: PipelineContext) -> None:
    image_set = _require_image_set(context)
    stack = _require_stack(context)
    hdu_index = int(context.settings.get("hdu_index", 0))

    narrow_hdul = stack.enter_context(fits.open(image_set.narrow_file, memmap=True))
    blue_hdul = stack.enter_context(fits.open(image_set.blue_file, memmap=True))
    red_hdul = stack.enter_context(fits.open(image_set.red_file, memmap=True))

    context.narrow_hdu = narrow_hdul[hdu_index].copy()
    context.blue_hdu = blue_hdul[hdu_index].copy()
    context.red_hdu = red_hdul[hdu_index].copy()


def _stage_load_errors(context: PipelineContext) -> None:
    image_set = _require_image_set(context)
    if (
        image_set.narrow_error_file is None
        or image_set.blue_error_file is None
        or image_set.red_error_file is None
    ):
        return

    stack = _require_stack(context)
    hdu_index = int(context.settings.get("hdu_index", 0))
    narrow_error_hdul = stack.enter_context(fits.open(image_set.narrow_error_file, memmap=True))
    blue_error_hdul = stack.enter_context(fits.open(image_set.blue_error_file, memmap=True))
    red_error_hdul = stack.enter_context(fits.open(image_set.red_error_file, memmap=True))

    context.narrow_error_hdu = convert_inverse_variance_to_error(narrow_error_hdul[hdu_index])
    context.blue_error_hdu = convert_inverse_variance_to_error(blue_error_hdul[hdu_index])
    context.red_error_hdu = convert_inverse_variance_to_error(red_error_hdul[hdu_index])


def _stage_preprocess(context: PipelineContext) -> None:
    if context.narrow_hdu is None or context.blue_hdu is None or context.red_hdu is None:
        raise ValueError("Science HDUs must be loaded before preprocessing")

    error_inputs = [
        hdu
        for hdu in (context.narrow_error_hdu, context.blue_error_hdu, context.red_error_hdu)
        if hdu is not None
    ]
    context.narrow_hdu, context.blue_hdu, context.red_hdu, error_inputs = preprocess_hst_data(
        narrow_hdu=context.narrow_hdu,
        blue_hdu=context.blue_hdu,
        red_hdu=context.red_hdu,
        error_hdus=error_inputs,
    )
    if error_inputs:
        context.narrow_error_hdu, context.blue_error_hdu, context.red_error_hdu = error_inputs


def _stage_resolve_bandpasses(context: PipelineContext) -> None:
    image_set = _require_image_set(context)
    if context.narrow_hdu is None or context.blue_hdu is None or context.red_hdu is None:
        raise ValueError("Science HDUs must be loaded before bandpass resolution")

    context.bandpass_catalog = load_bandpass_catalog(context.params)
    context.narrow_bandpass = bandpass_for_image(
        hdu=context.narrow_hdu,
        filename=image_set.narrow_file,
        filter_name=image_set.narrow_filter,
        catalog=context.bandpass_catalog,
        params=context.params,
    )
    context.blue_bandpass = bandpass_for_image(
        hdu=context.blue_hdu,
        filename=image_set.blue_file,
        filter_name=image_set.blue_filter,
        catalog=context.bandpass_catalog,
        params=context.params,
    )
    context.red_bandpass = bandpass_for_image(
        hdu=context.red_hdu,
        filename=image_set.red_file,
        filter_name=image_set.red_filter,
        catalog=context.bandpass_catalog,
        params=context.params,
    )

    for hdu, bandpass in (
        (context.narrow_hdu, context.narrow_bandpass),
        (context.blue_hdu, context.blue_bandpass),
        (context.red_hdu, context.red_bandpass),
    ):
        annotate_bandpass(hdu, bandpass)


def _stage_calibrate_flux_density(context: PipelineContext) -> None:
    if (
        context.narrow_hdu is None
        or context.blue_hdu is None
        or context.red_hdu is None
        or context.narrow_bandpass is None
        or context.blue_bandpass is None
        or context.red_bandpass is None
    ):
        raise ValueError("Science HDUs and bandpass metadata are required for calibration")

    context.narrow_hdu = convert_hst_count_rate_to_flux_density(
        context.narrow_hdu, photflam=float(context.narrow_bandpass["photflam"])
    )
    context.blue_hdu = convert_hst_count_rate_to_flux_density(
        context.blue_hdu, photflam=float(context.blue_bandpass["photflam"])
    )
    context.red_hdu = convert_hst_count_rate_to_flux_density(
        context.red_hdu, photflam=float(context.red_bandpass["photflam"])
    )

    if (
        context.narrow_error_hdu is not None
        and context.blue_error_hdu is not None
        and context.red_error_hdu is not None
    ):
        context.narrow_error_hdu = convert_hst_count_rate_to_flux_density(
            context.narrow_error_hdu,
            photflam=float(context.narrow_bandpass["photflam"]),
        )
        context.blue_error_hdu = convert_hst_count_rate_to_flux_density(
            context.blue_error_hdu,
            photflam=float(context.blue_bandpass["photflam"]),
        )
        context.red_error_hdu = convert_hst_count_rate_to_flux_density(
            context.red_error_hdu,
            photflam=float(context.red_bandpass["photflam"]),
        )


def _background_correction_for(
    corrections: Mapping[str, Any],
    galaxy: str,
    filter_name: str,
    instrument: str,
) -> float | None:
    """Return a configured background correction for one selected filter/instrument."""

    filter_key = filter_name.lower()
    if filter_key not in corrections:
        return None

    instrument_corrections = corrections[filter_key]
    if not isinstance(instrument_corrections, Mapping):
        raise ValueError(
            f"{galaxy}: background_corrections.{filter_key} must map instruments to values"
        )

    instrument_key = instrument.lower()
    normalized = {str(key).lower(): value for key, value in instrument_corrections.items()}
    if instrument_key not in normalized:
        raise ValueError(
            f"{galaxy}: background correction configured for {filter_key}, but not for "
            f"resolved instrument {instrument_key!r}"
        )

    try:
        return float(normalized[instrument_key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{galaxy}: background correction for {filter_key}/{instrument_key} must be numeric"
        ) from exc


def _stage_apply_background_corrections(context: PipelineContext) -> None:
    corrections = context.settings.get("background_corrections", {}) or {}
    if not corrections:
        return
    if not isinstance(corrections, Mapping):
        raise ValueError(f"{context.galaxy}: background_corrections must be a mapping")

    image_set = _require_image_set(context)
    if (
        context.narrow_hdu is None
        or context.blue_hdu is None
        or context.red_hdu is None
        or context.narrow_bandpass is None
        or context.blue_bandpass is None
        or context.red_bandpass is None
    ):
        raise ValueError("Calibrated science HDUs and bandpasses are required for correction")

    selected = (
        ("blue", image_set.blue_filter, context.blue_hdu, context.blue_bandpass),
        ("narrow", image_set.narrow_filter, context.narrow_hdu, context.narrow_bandpass),
        ("red", image_set.red_filter, context.red_hdu, context.red_bandpass),
    )
    for label, filter_name, hdu, bandpass in selected:
        instrument = str(bandpass.get("instrument", "")).lower()
        offset = _background_correction_for(
            corrections=corrections,
            galaxy=context.galaxy,
            filter_name=filter_name,
            instrument=instrument,
        )
        if offset is None:
            continue

        corrected_hdu, record = apply_background_correction(
            hdu=hdu,
            surface_brightness_offset=offset,
            filter_name=filter_name,
            instrument=instrument,
        )
        record["image"] = label
        context.background_corrections.append(record)
        if label == "blue":
            context.blue_hdu = corrected_hdu
        elif label == "narrow":
            context.narrow_hdu = corrected_hdu
        else:
            context.red_hdu = corrected_hdu


def _stage_apply_foreground_extinction(context: PipelineContext) -> None:
    if context.narrow_hdu is None or context.blue_hdu is None or context.red_hdu is None:
        raise ValueError("Science HDUs must be calibrated before foreground extinction")

    ebv = foreground_ebv(context.galaxy, context.settings, context.params)
    context.extinction_ebv = ebv
    r_v = float((context.params.get("foreground_extinction", {}) or {}).get("r_v", 3.1))
    if ebv is None:
        return

    for band_hdu, err_hdu in (
        (context.narrow_hdu, context.narrow_error_hdu),
        (context.blue_hdu, context.blue_error_hdu),
        (context.red_hdu, context.red_error_hdu),
    ):
        apply_foreground_extinction_to_hdu_pair(band_hdu, err_hdu, ebv=ebv, r_v=r_v)


def _stage_match_spatial_coverage(context: PipelineContext) -> None:
    cfg = context.settings.get("coverage_mask", {}) or {}
    if isinstance(cfg, bool):
        enabled = cfg
        closing_size = 10
        closing_iterations = 5
    elif isinstance(cfg, Mapping):
        enabled = bool(cfg.get("enabled", True))
        closing_size = int(cfg.get("closing_size", 10))
        closing_iterations = int(cfg.get("closing_iterations", 5))
    else:
        raise ValueError(f"{context.galaxy}: coverage_mask must be a mapping or boolean")

    if not enabled:
        return
    if context.narrow_hdu is None or context.blue_hdu is None or context.red_hdu is None:
        raise ValueError("Science HDUs must be calibrated before coverage matching")

    (
        context.blue_hdu,
        context.narrow_hdu,
        context.red_hdu,
        error_hdus,
        context.coverage_mask,
    ) = match_spatial_coverage(
        blue_hdu=context.blue_hdu,
        narrow_hdu=context.narrow_hdu,
        red_hdu=context.red_hdu,
        error_hdus=(
            context.blue_error_hdu,
            context.narrow_error_hdu,
            context.red_error_hdu,
        ),
        closing_size=closing_size,
        closing_iterations=closing_iterations,
    )
    if error_hdus:
        context.blue_error_hdu, context.narrow_error_hdu, context.red_error_hdu = error_hdus


def _stage_subtract_continuum(context: PipelineContext) -> None:
    image_set = _require_image_set(context)
    contsub_space = str(context.settings.get("contsub_space", "linear")).lower()
    if contsub_space not in {"linear", "log"}:
        raise ValueError(
            f"Unsupported contsub_space {contsub_space!r}; expected 'linear' or 'log'"
        )
    if (
        context.narrow_hdu is None
        or context.blue_hdu is None
        or context.red_hdu is None
        or context.narrow_bandpass is None
        or context.blue_bandpass is None
        or context.red_bandpass is None
    ):
        raise ValueError("Calibrated HDUs and bandpasses are required for subtraction")

    (
        context.contsub_hdu,
        context.continuum_hdu,
        context.contsub_error_output_hdu,
        context.continuum_error_output_hdu,
        context.weight_blue,
        context.weight_red,
    ) = linear_continuum_subtract(
        narrow_hdu=context.narrow_hdu,
        blue_hdu=context.blue_hdu,
        red_hdu=context.red_hdu,
        blue_filter=image_set.blue_filter,
        red_filter=image_set.red_filter,
        narrow_filter=image_set.narrow_filter,
        narrow_pivot=float(context.narrow_bandpass["pivot"]),
        blue_pivot=float(context.blue_bandpass["pivot"]),
        red_pivot=float(context.red_bandpass["pivot"]),
        narrow_error_hdu=context.narrow_error_hdu,
        blue_error_hdu=context.blue_error_hdu,
        red_error_hdu=context.red_error_hdu,
        contsub_space=contsub_space,
    )
    context.bandpass_source = str(context.narrow_bandpass["source"])


def _stage_build_products(context: PipelineContext) -> None:
    build_products(context)


def _stage_write_outputs(context: PipelineContext) -> None:
    write_products(context)
    context.status = "written"


def _register_builtin_stages() -> None:
    for name, function in (
        ("resolve_inputs", _stage_resolve_inputs),
        ("plan_outputs", _stage_plan_outputs),
        ("skip_existing_outputs", _stage_skip_existing_outputs),
        ("load_images", _stage_load_images),
        ("load_errors", _stage_load_errors),
        ("preprocess", _stage_preprocess),
        ("resolve_bandpasses", _stage_resolve_bandpasses),
        ("calibrate_flux_density", _stage_calibrate_flux_density),
        ("apply_background_corrections", _stage_apply_background_corrections),
        ("apply_foreground_extinction", _stage_apply_foreground_extinction),
        ("match_spatial_coverage", _stage_match_spatial_coverage),
        ("subtract_continuum", _stage_subtract_continuum),
        ("build_products", _stage_build_products),
        ("write_outputs", _stage_write_outputs),
    ):
        register_stage(FunctionStage(name, function))


_register_builtin_stages()
