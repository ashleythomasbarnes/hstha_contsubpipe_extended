"""Galaxy configuration loading and per-target settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_DIR = "config"
DEFAULT_GALAXIES_FILE = "galaxies.yaml"


def load_galaxy_config(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    filename: str = DEFAULT_GALAXIES_FILE,
) -> dict[str, Any]:
    """Load the galaxy/run configuration YAML file."""

    config_path = Path(config_dir) / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Galaxy config not found: {config_path}")
    with config_path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def configured_galaxies(galaxy_config: Mapping[str, Any]) -> list[str]:
    """Return the configured galaxy names in their YAML order."""

    galaxies = galaxy_config.get("galaxies", [])
    names: list[str] = []
    for entry in galaxies:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, Mapping) and "name" in entry:
            names.append(str(entry["name"]))
        else:
            raise ValueError(f"Invalid galaxy entry in config/galaxies.yaml: {entry!r}")
    return names


def galaxy_settings(galaxy_config: Mapping[str, Any], galaxy: str) -> dict[str, Any]:
    """Merge defaults and any per-galaxy overrides."""

    defaults = dict(galaxy_config.get("defaults", {}) or {})
    overrides = dict(galaxy_config.get("overrides", {}).get(galaxy, {}) or {})

    for entry in galaxy_config.get("galaxies", []):
        if isinstance(entry, Mapping) and entry.get("name") == galaxy:
            overrides.update({k: v for k, v in entry.items() if k != "name"})
            break

    merged = defaults
    merged.update(overrides)
    return merged
