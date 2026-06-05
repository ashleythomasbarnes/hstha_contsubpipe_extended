"""Input image discovery for continuum subtraction."""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from hstha_contsubpipe_extended.core.config import get_configs
from hstha_contsubpipe_extended.core.context import build_generic_context
from hstha_contsubpipe_extended.core.files import resolve_file

from .galaxy_config import DEFAULT_CONFIG_DIR, galaxy_settings
from .models import ImageSet

LOGGER = logging.getLogger(__name__)


def filter_digits(filter_name: str) -> str:
    """Return the numeric part used by the HST image-product directories."""

    return "".join(char for char in filter_name if char.isdigit())


def resolve_template_path(
    template_key: str,
    context_extra: Mapping[str, Any],
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> str:
    """Resolve a path template from ``files.yaml`` with dynamic context."""

    paths, params, files = get_configs(config_dir=config_dir)
    context = build_generic_context(paths=paths, params=params, extra=context_extra)
    return resolve_file(files_cfg=files, key=template_key, context=context)


def select_match(
    matches: Sequence[Path],
    preferred_tokens: Sequence[str],
    galaxy: str,
    filter_name: str,
) -> Path:
    """Select a unique file from glob matches, using configured token priority."""

    if not matches:
        raise FileNotFoundError(f"No input file found for {galaxy} {filter_name}")

    ordered_matches = sorted(matches)
    for token in preferred_tokens:
        token_matches = [
            path
            for path in ordered_matches
            if f"_{token.lower()}_" in path.name.lower()
            or f"/{token.lower()}" in str(path).lower()
        ]
        if len(token_matches) == 1:
            if len(matches) > 1:
                LOGGER.warning(
                    "Multiple matches for %s %s; selected %s by preferred token '%s'",
                    galaxy,
                    filter_name,
                    token_matches[0],
                    token,
                )
            return token_matches[0]

    if len(ordered_matches) == 1:
        return ordered_matches[0]

    match_list = "\n  ".join(str(path) for path in ordered_matches)
    raise ValueError(
        f"Multiple input files found for {galaxy} {filter_name}; add an override "
        f"in config/galaxies.yaml.\n  {match_list}"
    )


def find_filter_file(
    galaxy: str,
    filter_name: str,
    settings: Mapping[str, Any],
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    product: str = "science",
) -> Path:
    """Find the configured image file for one galaxy/filter combination."""

    filter_name = filter_name.lower()
    if product not in {"science", "error"}:
        raise ValueError(f"Unknown product {product!r}; expected 'science' or 'error'")

    override_key = f"{filter_name}_file" if product == "science" else f"{filter_name}_error_file"
    if override_key in settings:
        return Path(settings[override_key]).expanduser().resolve()

    if product == "error":
        science_override = f"{filter_name}_file"
        if science_override in settings:
            candidate = Path(
                str(settings[science_override]).replace(
                    "_exp_drc_sci.fits", "_err_drc_wht.fits"
                )
            ).expanduser()
            if candidate.exists():
                return candidate.resolve()

    search_galaxy = str(settings.get("search_galaxy", galaxy))
    template_key = "hst_image_glob" if product == "science" else "hst_error_glob"
    pattern = resolve_template_path(
        template_key,
        {
            "galaxy": galaxy,
            "search_galaxy": search_galaxy,
            "filter": filter_name,
            "filter_digits": filter_digits(filter_name),
        },
        config_dir=config_dir,
    )
    matches = [Path(match).resolve() for match in glob.glob(pattern)]
    return select_match(
        matches=matches,
        preferred_tokens=settings.get("preferred_instruments", []),
        galaxy=galaxy,
        filter_name=f"{filter_name} {product}",
    )


def resolve_image_set(
    galaxy: str,
    galaxy_config: Mapping[str, Any],
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> ImageSet:
    """Resolve all input images needed to continuum-subtract one galaxy."""

    settings = galaxy_settings(galaxy_config, galaxy)
    broad_filters = [str(item).lower() for item in settings["broad_filters"]]
    if len(broad_filters) != 2:
        raise ValueError(f"{galaxy}: broad_filters must contain exactly two filters")

    narrow_filters = [str(item).lower() for item in settings["narrow_filters"]]
    blue_filter, red_filter = broad_filters
    blue_file = find_filter_file(galaxy, blue_filter, settings, config_dir=config_dir)
    red_file = find_filter_file(galaxy, red_filter, settings, config_dir=config_dir)

    narrow_file: Path | None = None
    narrow_filter = ""
    errors: list[str] = []
    for candidate in narrow_filters:
        try:
            narrow_file = find_filter_file(galaxy, candidate, settings, config_dir=config_dir)
            narrow_filter = candidate
            break
        except FileNotFoundError as exc:
            errors.append(str(exc))

    if narrow_file is None:
        raise FileNotFoundError(f"{galaxy}: no narrowband file found. " + " ".join(errors))

    require_errors = bool(settings.get("require_errors", True))
    blue_error_file = red_error_file = narrow_error_file = None
    try:
        blue_error_file = find_filter_file(
            galaxy, blue_filter, settings, config_dir=config_dir, product="error"
        )
        red_error_file = find_filter_file(
            galaxy, red_filter, settings, config_dir=config_dir, product="error"
        )
        narrow_error_file = find_filter_file(
            galaxy, narrow_filter, settings, config_dir=config_dir, product="error"
        )
    except FileNotFoundError:
        if require_errors:
            raise
        LOGGER.warning("%s: error images missing; error products will not be written", galaxy)

    return ImageSet(
        galaxy=galaxy,
        blue_filter=blue_filter,
        red_filter=red_filter,
        narrow_filter=narrow_filter,
        blue_file=blue_file,
        red_file=red_file,
        narrow_file=narrow_file,
        blue_error_file=blue_error_file,
        red_error_file=red_error_file,
        narrow_error_file=narrow_error_file,
    )
