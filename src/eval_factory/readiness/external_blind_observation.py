from __future__ import annotations

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.labeling_v2 import LabelSpecV2
from eval_factory.labeling.decision import (
    LabelDecisionMerger,
    LabelDecisionMergeRequest,
    LabelDecisionRoutingPolicy,
)
from eval_factory.labeling.structured import (
    StructuredFactSet,
    StructuredPredicateCompiler,
)
from eval_factory.readiness.external_observation_models import (
    BlindLabelObservationRecordV1,
)
from eval_factory.readiness.external_reference_authoring import (
    APPROVED_EXTERNAL_LABEL_NAMES,
    external_trace_envelope_ref,
)
from eval_factory.trace.indexing.models import TraceIndexResult


class ExternalBlindObservationError(RuntimeError):
    pass


class ExternalBlindStructuredObserver:
    def __init__(
        self,
        *,
        observation_policy_ref: ObjectRef,
    ) -> None:
        if (
            observation_policy_ref.object_type != "external-observation-policy"
            or observation_policy_ref.object_version != "v2"
        ):
            raise ExternalBlindObservationError("blind observation policy ref is invalid")
        self.observation_policy_ref = observation_policy_ref
        self.compiler = StructuredPredicateCompiler()
        self.merger = LabelDecisionMerger()

    def observe(
        self,
        *,
        trace_index: TraceIndexResult,
        label_specs: tuple[LabelSpecV2, ...],
        audit: ContractAudit,
    ) -> tuple[BlindLabelObservationRecordV1, ...]:
        specs = {spec.name: spec for spec in label_specs}
        if set(specs) != set(APPROVED_EXTERNAL_LABEL_NAMES):
            raise ExternalBlindObservationError("blind observation label portfolio is not exact")
        trace_envelope_ref = external_trace_envelope_ref(
            trace_index,
            audit=audit,
        )
        normalization = trace_index.normalization_result
        recovery = normalization.recovery_result
        registered = recovery.base_parse_result.registered_source
        fact_set = StructuredFactSet(
            source_trace_id=registered.source.source_trace_id,
            trace_ir_version_id=normalization.trace_ir_version_id,
            trace_envelope_ref=trace_envelope_ref,
            capabilities=recovery.capabilities,
            source_spans=normalization.source_spans,
            events=tuple(event.model_copy(update={"content_ref": None}) for event in normalization.events),
            tool_call_records=normalization.tool_call_records,
            file_observations=trace_index.file_observations,
            interaction_segments=trace_index.interaction_segments,
            audit=audit,
        )
        records: list[BlindLabelObservationRecordV1] = []
        for name in (
            "powershell_error_signature",
            "search_tool_usage",
        ):
            spec = specs[name]
            structured = self.compiler.compile(
                label_spec=spec,
                fact_set=fact_set,
                audit=audit,
            )
            merged = self.merger.merge(
                LabelDecisionMergeRequest(
                    label_spec=spec,
                    structured_result=structured,
                    semantic_result=None,
                    routing_policy=LabelDecisionRoutingPolicy(),
                    audit=audit,
                )
            )
            records.append(
                BlindLabelObservationRecordV1.create(
                    source_trace_id=registered.source.source_trace_id,
                    raw_sha256=registered.source.raw_sha256,
                    trace_envelope_ref=trace_envelope_ref,
                    label_spec_ref=_label_spec_ref(spec),
                    observation_policy_ref=self.observation_policy_ref,
                    label_decision=merged.label_decision,
                    route=merged.route,
                    audit=audit,
                )
            )
        return tuple(
            sorted(
                records,
                key=lambda record: record.label_spec_ref.object_id,
            )
        )


def _label_spec_ref(spec: LabelSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-spec",
        object_id=spec.label_spec_id,
        object_version=spec.label_version,
        object_sha256=spec.label_spec_sha256,
    )


__all__ = [
    "ExternalBlindObservationError",
    "ExternalBlindStructuredObserver",
]
