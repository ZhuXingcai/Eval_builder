from __future__ import annotations

from eval_factory.contracts.core import ContractAudit
from eval_factory.trace.indexing.files import build_file_observations
from eval_factory.trace.indexing.models import (
    FILE_OBSERVATION_POLICY_VERSION,
    SEGMENTATION_POLICY_VERSION,
    TraceIndexResult,
)
from eval_factory.trace.indexing.segments import build_interaction_segments
from eval_factory.trace.normalization import TraceNormalizationResult


class TraceIndexBuilder:
    file_observation_policy_version = FILE_OBSERVATION_POLICY_VERSION
    segmentation_policy_version = SEGMENTATION_POLICY_VERSION

    def build(
        self,
        *,
        normalization_result: TraceNormalizationResult,
        audit: ContractAudit,
    ) -> TraceIndexResult:
        file_build = build_file_observations(normalization_result)
        content_blobs = tuple(
            sorted(
                {
                    item.content_ref.object_id: item
                    for item in (*normalization_result.content_blobs, *file_build.raw_path_blobs)
                }.values(),
                key=lambda item: item.content_ref.object_id,
            )
        )
        indexed_normalization = normalization_result.__class__(
            recovery_result=normalization_result.recovery_result,
            trace_ir_version_id=normalization_result.trace_ir_version_id,
            trace_ir_schema_version=normalization_result.trace_ir_schema_version,
            normalization_policy_version=normalization_result.normalization_policy_version,
            tool_family_policy_version=normalization_result.tool_family_policy_version,
            declared_segmentation_policy_version=normalization_result.declared_segmentation_policy_version,
            events=normalization_result.events,
            tool_call_records=normalization_result.tool_call_records,
            source_spans=normalization_result.source_spans,
            content_blobs=content_blobs,
            diagnostics=normalization_result.diagnostics,
            unrecoverable_spans=normalization_result.unrecoverable_spans,
            audit=normalization_result.audit,
        )
        segments = build_interaction_segments(
            events=indexed_normalization.events,
            tool_call_records=indexed_normalization.tool_call_records,
            file_observations=file_build.observations,
        )
        return TraceIndexResult(
            normalization_result=indexed_normalization,
            file_observation_policy_version=self.file_observation_policy_version,
            segmentation_policy_version=self.segmentation_policy_version,
            file_observations=file_build.observations,
            interaction_segments=segments,
            diagnostics=file_build.diagnostics,
            audit=audit,
        )
