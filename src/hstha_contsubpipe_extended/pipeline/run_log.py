"""Timestamped run audit log writing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from hstha_contsubpipe_extended.core.config import get_configs
from hstha_contsubpipe_extended.core.context import build_generic_context
from hstha_contsubpipe_extended.core.files import resolve_file

from .models import ContsubResult, PipelineContext


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def make_run_id(started_at: datetime | None = None) -> str:
    """Return a compact timestamp ID suitable for filenames."""

    timestamp = started_at or utc_now()
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


@dataclass
class RunAudit:
    """Audit data accumulated for one pipeline run."""

    run_id: str
    started_at: datetime
    config_dir: str
    selected_galaxies: list[str]
    selected_stages: list[str]
    dry_run: bool
    overwrite: bool | None
    keep_going: bool
    ended_at: datetime | None = None
    results: list[ContsubResult] = field(default_factory=list)
    galaxy_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        """Return run duration in seconds if the run has ended."""

        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


def isoformat(value: datetime | None) -> str:
    """Format an optional timestamp."""

    return value.isoformat() if value is not None else ""


def json_ready(value: Any) -> Any:
    """Convert values commonly stored in pipeline records into JSON-safe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def _result_paths(result: ContsubResult) -> dict[str, str]:
    return {
        "blue_file": str(result.blue_file or ""),
        "red_file": str(result.red_file or ""),
        "narrow_file": str(result.narrow_file or ""),
        "blue_error_file": str(result.blue_error_file or ""),
        "red_error_file": str(result.red_error_file or ""),
        "narrow_error_file": str(result.narrow_error_file or ""),
        "contsub_file": str(result.contsub_file or ""),
        "continuum_file": str(result.continuum_file or ""),
        "contsub_error_file": str(result.contsub_error_file or ""),
        "continuum_error_file": str(result.continuum_error_file or ""),
        "halpha_file": str(result.halpha_file or ""),
        "halpha_error_file": str(result.halpha_error_file or ""),
    }


def _bandpass_record(context: PipelineContext) -> dict[str, dict[str, Any]]:
    image_set = context.image_set
    if image_set is None:
        return {}

    rows: dict[str, dict[str, Any]] = {}
    for label, filter_name, bandpass in (
        ("narrow", image_set.narrow_filter, context.narrow_bandpass),
        ("blue", image_set.blue_filter, context.blue_bandpass),
        ("red", image_set.red_filter, context.red_bandpass),
    ):
        if bandpass is None:
            rows[label] = {"filter": filter_name}
            continue
        rows[label] = {
            "filter": filter_name,
            "instrument": bandpass.get("instrument", ""),
            "pivot": bandpass.get("pivot", ""),
            "pivot_source": bandpass.get("pivot_source", ""),
            "width": bandpass.get("width", ""),
            "width_source": bandpass.get("width_source", ""),
            "photflam": bandpass.get("photflam", ""),
            "photflam_source": bandpass.get("photflam_source", ""),
        }
    return rows


def context_audit_record(
    context: PipelineContext,
    result: ContsubResult,
    exception: BaseException | None = None,
) -> dict[str, Any]:
    """Build a serializable per-galaxy audit record from a context."""

    return {
        "galaxy": result.galaxy,
        "status": result.status,
        "message": result.message,
        "exception_type": type(exception).__name__ if exception is not None else "",
        "exception_message": str(exception) if exception is not None else "",
        "settings": dict(context.settings),
        "paths": _result_paths(result),
        "stage_events": list(context.stage_events),
        "bandpasses": _bandpass_record(context),
        "derived": {
            "weight_blue": result.weight_blue,
            "weight_red": result.weight_red,
            "extinction_ebv": result.extinction_ebv,
            "nii_to_halpha": result.nii_to_halpha,
            "narrowband_width": result.narrowband_width,
            "bandpass_source": result.bandpass_source,
            "background_corrections": list(context.background_corrections),
            "coverage_mask": dict(context.coverage_mask),
        },
    }


def failed_galaxy_audit_record(
    galaxy: str, result: ContsubResult, exception: BaseException
) -> dict[str, Any]:
    """Build an audit record when a context was not available."""

    return {
        "galaxy": galaxy,
        "status": result.status,
        "message": result.message,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "settings": {},
        "paths": _result_paths(result),
        "stage_events": [],
        "bandpasses": {},
        "derived": {
            "weight_blue": result.weight_blue,
            "weight_red": result.weight_red,
            "extinction_ebv": result.extinction_ebv,
            "nii_to_halpha": result.nii_to_halpha,
            "narrowband_width": result.narrowband_width,
            "bandpass_source": result.bandpass_source,
            "background_corrections": [],
            "coverage_mask": {},
        },
    }


def resolve_run_log_paths(config_dir: str | Path, run_id: str) -> tuple[Path, Path]:
    """Resolve timestamped text and JSONL audit log paths."""

    paths, params, files = get_configs(config_dir=config_dir)
    context = build_generic_context(paths=paths, params=params, extra={"run_id": run_id})
    text_path = Path(resolve_file(files_cfg=files, key="outputs.run_log", context=context))
    jsonl_path = Path(resolve_file(files_cfg=files, key="outputs.run_log_jsonl", context=context))
    return text_path, jsonl_path


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _write_key_values(lines: list[str], values: Mapping[str, Any], indent: str = "  ") -> None:
    for key, value in values.items():
        if isinstance(value, Mapping):
            lines.append(f"{indent}{key}:")
            _write_key_values(lines, value, indent=f"{indent}  ")
        else:
            lines.append(f"{indent}{key}: {_format_value(value)}")


def render_text_log(audit: RunAudit) -> str:
    """Render a human-readable audit log."""

    lines: list[str] = [
        "Continuum Subtraction Run Audit",
        "================================",
        f"run_id: {audit.run_id}",
        f"started_at: {isoformat(audit.started_at)}",
        f"ended_at: {isoformat(audit.ended_at)}",
        f"duration_seconds: {_format_value(audit.duration_seconds)}",
        f"config_dir: {audit.config_dir}",
        f"dry_run: {audit.dry_run}",
        f"overwrite: {audit.overwrite}",
        f"keep_going: {audit.keep_going}",
        f"selected_galaxies: {', '.join(audit.selected_galaxies)}",
        f"selected_stages: {', '.join(audit.selected_stages)}",
        "",
    ]

    for record in audit.galaxy_records:
        lines.extend(
            [
                f"Galaxy: {record['galaxy']}",
                "-" * (8 + len(str(record["galaxy"]))),
                f"status: {record['status']}",
                f"message: {record.get('message', '')}",
                f"exception_type: {record.get('exception_type', '')}",
                f"exception_message: {record.get('exception_message', '')}",
                "",
                "settings:",
            ]
        )
        settings = record.get("settings", {})
        interesting_settings = {
            key: settings.get(key, "")
            for key in (
                "broad_filters",
                "narrow_filters",
                "preferred_instruments",
                "hdu_index",
                "require_errors",
                "write_continuum",
                "contsub_space",
                "background_corrections",
                "coverage_mask",
                "nii_to_halpha",
                "output_unit",
                "overwrite",
                "narrowband_width_header",
                "narrowband_widths",
                "search_galaxy",
                "sample_name",
            )
            if key in settings
        }
        _write_key_values(lines, interesting_settings)

        lines.extend(["", "input/output paths:"])
        _write_key_values(lines, record.get("paths", {}))

        lines.extend(["", "stages:"])
        for event in record.get("stage_events", []):
            lines.append(
                "  "
                + f"{event.get('stage', '')}: {event.get('status', '')} "
                + f"start={event.get('started_at', '')} "
                + f"end={event.get('ended_at', '')} "
                + f"duration={event.get('duration_seconds', '')} "
                + f"error={event.get('error_message', '')}"
            )

        lines.extend(["", "bandpasses:"])
        for label, bandpass in record.get("bandpasses", {}).items():
            lines.append(f"  {label}:")
            _write_key_values(lines, bandpass, indent="    ")

        lines.extend(["", "derived:"])
        _write_key_values(lines, record.get("derived", {}))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_run_log(audit: RunAudit, config_dir: str | Path) -> Path:
    """Write a human-readable run audit log."""

    text_path, _ = resolve_run_log_paths(config_dir=config_dir, run_id=audit.run_id)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(render_text_log(audit))
    return text_path


def iter_jsonl_records(audit: RunAudit):
    """Yield JSONL records for one run audit."""

    yield {
        "record_type": "run_start",
        "run_id": audit.run_id,
        "started_at": isoformat(audit.started_at),
        "config_dir": audit.config_dir,
        "selected_galaxies": audit.selected_galaxies,
        "selected_stages": audit.selected_stages,
        "dry_run": audit.dry_run,
        "overwrite": audit.overwrite,
        "keep_going": audit.keep_going,
    }
    for record in audit.galaxy_records:
        for event in record.get("stage_events", []):
            yield {
                "record_type": "stage_event",
                "run_id": audit.run_id,
                "galaxy": record["galaxy"],
                **event,
            }
        yield {
            "record_type": "galaxy_summary",
            "run_id": audit.run_id,
            **record,
        }
    yield {
        "record_type": "run_end",
        "run_id": audit.run_id,
        "started_at": isoformat(audit.started_at),
        "ended_at": isoformat(audit.ended_at),
        "duration_seconds": audit.duration_seconds,
        "status_counts": {
            status: sum(1 for result in audit.results if result.status == status)
            for status in sorted({result.status for result in audit.results})
        },
    }


def write_run_log_jsonl(audit: RunAudit, config_dir: str | Path) -> Path:
    """Write a structured JSONL run audit log."""

    _, jsonl_path = resolve_run_log_paths(config_dir=config_dir, run_id=audit.run_id)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w") as fh:
        for record in iter_jsonl_records(audit):
            fh.write(json.dumps(json_ready(record), sort_keys=True) + "\n")
    return jsonl_path


def write_run_audit(audit: RunAudit, config_dir: str | Path) -> tuple[Path, Path]:
    """Write both text and JSONL audit logs."""

    return write_run_log(audit, config_dir=config_dir), write_run_log_jsonl(
        audit, config_dir=config_dir
    )
