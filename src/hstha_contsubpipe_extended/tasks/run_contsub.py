from __future__ import annotations

from hstha_contsubpipe_extended.contsub import ContsubResult, run_all


TASK_NAME = "run_contsub"


def run(
    galaxies: list[str] | None = None,
    dry_run: bool = False,
    overwrite: bool | None = None,
    keep_going: bool = True,
) -> list[ContsubResult]:
    """Run the configured linear HST continuum-subtraction pipeline."""

    return run_all(
        galaxies=galaxies,
        dry_run=dry_run,
        overwrite=overwrite,
        keep_going=keep_going,
    )
