"""Pipeline run orchestration."""

from __future__ import annotations

from contextlib import ExitStack
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hstha_contsubpipe_extended.core.config import clear_config_cache, get_configs

from .galaxy_config import DEFAULT_CONFIG_DIR, configured_galaxies, galaxy_settings, load_galaxy_config
from .manifest import write_manifest
from .models import ContsubResult, PipelineContext, PipelineStop
from .run_log import (
    RunAudit,
    context_audit_record,
    failed_galaxy_audit_record,
    make_run_id,
    utc_now,
    write_run_audit,
)
from .stages import build_stage_sequence, default_stage_names

LOGGER = logging.getLogger(__name__)


class _GalaxyRunError(Exception):
    """Internal wrapper carrying audit details for a failed galaxy."""

    def __init__(
        self,
        original: Exception,
        result: ContsubResult,
        galaxy_record: dict[str, Any],
    ) -> None:
        super().__init__(str(original))
        self.original = original
        self.result = result
        self.galaxy_record = galaxy_record


def _configured_stage_names(params: Mapping[str, Any]) -> list[str]:
    """Return the configured stage sequence from params.yaml."""

    pipeline_cfg = params.get("contsub_pipeline", {}) or {}
    configured = pipeline_cfg.get("stages")
    if configured is None:
        names = default_stage_names()
    elif isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        names = [str(name) for name in configured]
    else:
        raise TypeError("contsub_pipeline.stages must be null or a list of stage names")

    disabled = pipeline_cfg.get("disabled_stages", []) or []
    if isinstance(disabled, (str, bytes)) or not isinstance(disabled, Sequence):
        raise TypeError("contsub_pipeline.disabled_stages must be a list of stage names")
    disabled_names = {str(name) for name in disabled}
    return [name for name in names if name not in disabled_names]


def run_galaxy(
    galaxy: str,
    galaxy_config: Mapping[str, Any],
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    dry_run: bool = False,
    overwrite: bool | None = None,
    stage_names: Sequence[str] | None = None,
) -> ContsubResult:
    """Run or plan continuum subtraction for one galaxy."""

    result, _ = _run_galaxy_with_audit(
        galaxy=galaxy,
        galaxy_config=galaxy_config,
        config_dir=config_dir,
        dry_run=dry_run,
        overwrite=overwrite,
        stage_names=stage_names,
        catch_errors=False,
    )
    return result


def _run_stage_with_event(context: PipelineContext, stage) -> None:
    """Run one stage and append an audit event to the context."""

    started_at = utc_now()
    event: dict[str, Any] = {
        "stage": stage.name,
        "started_at": started_at.isoformat(),
        "ended_at": "",
        "duration_seconds": "",
        "status": "started",
        "error_type": "",
        "error_message": "",
    }
    context.stage_events.append(event)
    try:
        stage.run(context)
    except PipelineStop:
        ended_at = utc_now()
        event["ended_at"] = ended_at.isoformat()
        event["duration_seconds"] = (ended_at - started_at).total_seconds()
        event["status"] = "stopped"
        raise
    except Exception as exc:
        ended_at = utc_now()
        event["ended_at"] = ended_at.isoformat()
        event["duration_seconds"] = (ended_at - started_at).total_seconds()
        event["status"] = "failed"
        event["error_type"] = type(exc).__name__
        event["error_message"] = str(exc)
        raise
    else:
        ended_at = utc_now()
        event["ended_at"] = ended_at.isoformat()
        event["duration_seconds"] = (ended_at - started_at).total_seconds()
        event["status"] = "completed"


def _run_galaxy_with_audit(
    galaxy: str,
    galaxy_config: Mapping[str, Any],
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    dry_run: bool = False,
    overwrite: bool | None = None,
    stage_names: Sequence[str] | None = None,
    catch_errors: bool = True,
) -> tuple[ContsubResult, dict[str, Any]]:
    """Run or plan one galaxy and return result plus audit record."""

    settings = galaxy_settings(galaxy_config, galaxy)
    if overwrite is None:
        overwrite = bool(settings.get("overwrite", False))

    _, params, files = get_configs(config_dir=config_dir)
    names = list(stage_names) if stage_names is not None else _configured_stage_names(params)
    stages = build_stage_sequence(names)

    with ExitStack() as stack:
        context = PipelineContext(
            galaxy=galaxy,
            galaxy_config=galaxy_config,
            config_dir=config_dir,
            settings=settings,
            params=params,
            files=files,
            dry_run=dry_run,
            overwrite=overwrite,
            stack=stack,
        )
        try:
            for stage in stages:
                _run_stage_with_event(context, stage)
        except PipelineStop:
            pass
        except Exception as exc:
            context.status = "failed"
            context.message = str(exc)
            result = context.to_result()
            galaxy_record = context_audit_record(context, result, exception=exc)
            if not catch_errors:
                raise _GalaxyRunError(exc, result, galaxy_record) from exc
            return result, galaxy_record

        if context.status == "running":
            context.status = "written" if "write_outputs" in names else "completed"
        result = context.to_result()
        return result, context_audit_record(context, result)


def run_all(
    galaxies: Iterable[str] | None = None,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    dry_run: bool = False,
    overwrite: bool | None = None,
    keep_going: bool = True,
    stage_names: Sequence[str] | None = None,
) -> list[ContsubResult]:
    """Run continuum subtraction for the configured galaxy sample."""

    # Config files are intentionally edited between reruns, especially in
    # notebooks and interactive sessions. Reload them for each pipeline run.
    clear_config_cache()
    started_at = utc_now()
    run_id = make_run_id(started_at)
    galaxy_config = load_galaxy_config(config_dir=config_dir)
    selected_galaxies = (
        list(galaxies) if galaxies is not None else configured_galaxies(galaxy_config)
    )
    _, params, _ = get_configs(config_dir=config_dir)
    selected_stages = list(stage_names) if stage_names is not None else _configured_stage_names(params)
    audit = RunAudit(
        run_id=run_id,
        started_at=started_at,
        config_dir=str(config_dir),
        selected_galaxies=selected_galaxies,
        selected_stages=selected_stages,
        dry_run=dry_run,
        overwrite=overwrite,
        keep_going=keep_going,
    )

    results: list[ContsubResult] = []
    for galaxy in selected_galaxies:
        try:
            result, galaxy_record = _run_galaxy_with_audit(
                galaxy=galaxy,
                galaxy_config=galaxy_config,
                config_dir=config_dir,
                dry_run=dry_run,
                overwrite=overwrite,
                stage_names=selected_stages,
                catch_errors=keep_going,
            )
        except _GalaxyRunError as exc:
            result = exc.result
            galaxy_record = exc.galaxy_record
            results.append(result)
            audit.results.append(result)
            audit.galaxy_records.append(galaxy_record)
            LOGGER.info("%s: %s %s", result.galaxy, result.status, result.message)
            audit.ended_at = utc_now()
            write_run_audit(audit, config_dir=config_dir)
            raise exc.original from exc.original
        except Exception as exc:
            result = ContsubResult(galaxy=galaxy, status="failed", message=str(exc))
            galaxy_record = failed_galaxy_audit_record(galaxy, result, exc)
            if not keep_going:
                results.append(result)
                audit.results.append(result)
                audit.galaxy_records.append(galaxy_record)
                audit.ended_at = utc_now()
                write_run_audit(audit, config_dir=config_dir)
                raise
        results.append(result)
        audit.results.append(result)
        audit.galaxy_records.append(galaxy_record)
        LOGGER.info("%s: %s %s", result.galaxy, result.status, result.message)

    if not dry_run:
        manifest_file = write_manifest(results, config_dir=config_dir)
        LOGGER.info("Wrote manifest: %s", manifest_file)

    audit.ended_at = utc_now()
    log_file, jsonl_file = write_run_audit(audit, config_dir=config_dir)
    LOGGER.info("Wrote run audit log: %s", log_file)
    LOGGER.info("Wrote run audit JSONL: %s", jsonl_file)

    return results
