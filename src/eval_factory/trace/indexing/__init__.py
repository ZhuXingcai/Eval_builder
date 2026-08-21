"""Deterministic TraceIR indexing projections."""

from eval_factory.trace.indexing.builder import TraceIndexBuilder
from eval_factory.trace.indexing.models import (
    FILE_OBSERVATION_POLICY_VERSION,
    SEGMENTATION_POLICY_VERSION,
    IndexDiagnostic,
    TraceIndexResult,
)
from eval_factory.trace.indexing.paths import ProjectedPath, project_logical_path

__all__ = [
    "FILE_OBSERVATION_POLICY_VERSION",
    "SEGMENTATION_POLICY_VERSION",
    "IndexDiagnostic",
    "ProjectedPath",
    "TraceIndexBuilder",
    "TraceIndexResult",
    "project_logical_path",
]
