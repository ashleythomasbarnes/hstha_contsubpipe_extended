"""Modular staged continuum-subtraction pipeline."""

from .models import ContsubResult, ImageSet, PipelineContext, PipelineStop
from .runner import run_all, run_galaxy
from .stages import build_stage_sequence, default_stage_names, get_stage, register_stage

__all__ = [
    "ContsubResult",
    "ImageSet",
    "PipelineContext",
    "PipelineStop",
    "build_stage_sequence",
    "default_stage_names",
    "get_stage",
    "register_stage",
    "run_all",
    "run_galaxy",
]
