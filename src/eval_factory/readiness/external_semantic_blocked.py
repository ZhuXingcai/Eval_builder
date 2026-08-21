from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelSpecV2,
    LabelUnresolvedReason,
)
from eval_factory.labeling.decision import (
    LABEL_DECISION_MERGE_POLICY_VERSION,
    LabelDecisionRoute,
)
from eval_factory.readiness.external_observation_models import (
    BlindLabelObservationRecordV1,
    BlindLabelObservationSetV1,
)
from eval_factory.readiness.external_reference_models import (
    ExternalReferenceAuthorKindV1,
    ExternalReferenceRecordV1,
    ExternalReferenceSetV1,
    ExternalReferenceStateV1,
)

EXTERNAL_BLOCKED_SEMANTIC_POLICY_VERSION = "external-blocked-semantic-evidence/r8-10-v1"


class ExternalBlockedSemanticEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExternalBlockedSemanticEvidenceCompilation:
    reference_set: ExternalReferenceSetV1
    observation_set: BlindLabelObservationSetV1
    semantic_source_count: int


@dataclass(frozen=True, slots=True)
class _SourceBinding:
    source_trace_id: str
    raw_sha256: str
    trace_envelope_ref: ObjectRef
    annotation_contract_ref: ObjectRef
    structured_capability_complete: bool


class ExternalBlockedSemanticEvidenceBuilder:
    def extend(
        self,
        *,
        reference_set: ExternalReferenceSetV1,
        observation_set: BlindLabelObservationSetV1,
        semantic_label_spec: LabelSpecV2,
        reference_model_profile: str,
        reference_prompt_version: str,
        audit: ContractAudit,
    ) -> ExternalBlockedSemanticEvidenceCompilation:
        residual = semantic_label_spec.semantic_residual
        if residual is None:
            raise ExternalBlockedSemanticEvidenceError("blocked semantic evidence requires a semantic label")
        semantic_label_ref = _label_spec_ref(semantic_label_spec)
        if reference_set.source_population_ref != observation_set.source_population_ref:
            raise ExternalBlockedSemanticEvidenceError("reference and observation populations differ")
        if set(reference_set.label_spec_refs) != set(observation_set.label_spec_refs):
            raise ExternalBlockedSemanticEvidenceError("reference and observation label portfolios differ")
        if semantic_label_ref in reference_set.label_spec_refs:
            raise ExternalBlockedSemanticEvidenceError(
                "semantic label already exists in the private authorities"
            )
        if (
            reference_model_profile == residual.model_profile
            or reference_prompt_version == residual.prompt_version
        ):
            raise ExternalBlockedSemanticEvidenceError(
                "reference and observation semantic authorities must be distinct"
            )

        bindings = _source_bindings(reference_set, observation_set)
        semantic_references: list[ExternalReferenceRecordV1] = []
        semantic_observations: list[BlindLabelObservationRecordV1] = []
        for binding in bindings:
            semantic_references.append(
                ExternalReferenceRecordV1.create(
                    source_trace_id=binding.source_trace_id,
                    raw_sha256=binding.raw_sha256,
                    trace_envelope_ref=binding.trace_envelope_ref,
                    label_spec_ref=semantic_label_ref,
                    label_name=semantic_label_spec.name,
                    annotation_contract_ref=(binding.annotation_contract_ref),
                    reference_policy_ref=(reference_set.reference_policy_ref),
                    expected_decision=LabelDecisionValueV2.ABSTAIN,
                    evidence_refs=(),
                    structured_capability_complete=(binding.structured_capability_complete),
                    author_kind=(ExternalReferenceAuthorKindV1.SEMANTIC_AGENT),
                    reference_state=ExternalReferenceStateV1.BLOCKED,
                    rule_version=None,
                    model_profile=reference_model_profile,
                    prompt_version=reference_prompt_version,
                    audit=audit,
                )
            )
            decision = _blocked_observation_decision(
                binding=binding,
                label_spec_ref=semantic_label_ref,
                observation_policy_ref=(observation_set.observation_policy_ref),
                model_profile=residual.model_profile,
                prompt_version=residual.prompt_version,
                audit=audit,
            )
            semantic_observations.append(
                BlindLabelObservationRecordV1.create(
                    source_trace_id=binding.source_trace_id,
                    raw_sha256=binding.raw_sha256,
                    trace_envelope_ref=binding.trace_envelope_ref,
                    label_spec_ref=semantic_label_ref,
                    observation_policy_ref=(observation_set.observation_policy_ref),
                    label_decision=decision,
                    route=LabelDecisionRoute.TYPED_UNRESOLVED_QUEUE,
                    audit=audit,
                )
            )

        labels = (*reference_set.label_spec_refs, semantic_label_ref)
        combined_references = ExternalReferenceSetV1.create(
            source_population_ref=reference_set.source_population_ref,
            label_spec_refs=labels,
            reference_policy_ref=reference_set.reference_policy_ref,
            records=(
                *reference_set.records,
                *semantic_references,
            ),
            audit=audit,
        )
        combined_observations = BlindLabelObservationSetV1.create(
            source_population_ref=observation_set.source_population_ref,
            label_spec_refs=labels,
            observation_policy_ref=(observation_set.observation_policy_ref),
            records=(
                *observation_set.records,
                *semantic_observations,
            ),
            audit=audit,
        )
        return ExternalBlockedSemanticEvidenceCompilation(
            reference_set=combined_references,
            observation_set=combined_observations,
            semantic_source_count=len(bindings),
        )


def _source_bindings(
    reference_set: ExternalReferenceSetV1,
    observation_set: BlindLabelObservationSetV1,
) -> tuple[_SourceBinding, ...]:
    references: dict[str, list[ExternalReferenceRecordV1]] = {}
    for reference_record in reference_set.records:
        references.setdefault(
            reference_record.source_trace_id,
            [],
        ).append(reference_record)
    observations: dict[str, list[BlindLabelObservationRecordV1]] = {}
    for observation_record in observation_set.records:
        observations.setdefault(
            observation_record.source_trace_id,
            [],
        ).append(observation_record)
    if not references or set(references) != set(observations):
        raise ExternalBlockedSemanticEvidenceError("reference and observation source inventories differ")

    bindings: list[_SourceBinding] = []
    for source_trace_id in sorted(references):
        source_references = references[source_trace_id]
        source_observations = observations[source_trace_id]
        raw_hashes = {reference.raw_sha256 for reference in source_references}
        raw_hashes.update(observation.raw_sha256 for observation in source_observations)
        trace_refs = {reference.trace_envelope_ref for reference in source_references}
        trace_refs.update(observation.trace_envelope_ref for observation in source_observations)
        annotation_refs = {record.annotation_contract_ref for record in source_references}
        if len(raw_hashes) != 1 or len(trace_refs) != 1 or len(annotation_refs) != 1:
            raise ExternalBlockedSemanticEvidenceError("semantic source binding is ambiguous")
        bindings.append(
            _SourceBinding(
                source_trace_id=source_trace_id,
                raw_sha256=next(iter(raw_hashes)),
                trace_envelope_ref=next(iter(trace_refs)),
                annotation_contract_ref=next(iter(annotation_refs)),
                structured_capability_complete=all(
                    record.structured_capability_complete for record in source_references
                ),
            )
        )
    return tuple(bindings)


def _blocked_observation_decision(
    *,
    binding: _SourceBinding,
    label_spec_ref: ObjectRef,
    observation_policy_ref: ObjectRef,
    model_profile: str,
    prompt_version: str,
    audit: ContractAudit,
) -> LabelDecisionV2:
    seed = {
        "source_trace_id": binding.source_trace_id,
        "raw_sha256": binding.raw_sha256,
        "trace_envelope_ref": binding.trace_envelope_ref.model_dump(
            mode="json",
        ),
        "label_spec_ref": label_spec_ref.model_dump(mode="json"),
        "observation_policy_ref": observation_policy_ref.model_dump(
            mode="json",
        ),
        "decision": LabelDecisionValueV2.ABSTAIN.value,
        "execution_status": LabelExecutionStatus.UNRESOLVED.value,
        "structured_capability_complete": (binding.structured_capability_complete),
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "unresolved_reasons": [LabelUnresolvedReason.MODEL_UNAVAILABLE.value],
        "policy_version": LABEL_DECISION_MERGE_POLICY_VERSION,
        "blocked_policy_version": (EXTERNAL_BLOCKED_SEMANTIC_POLICY_VERSION),
    }
    digest = hashlib.sha256(
        json.dumps(
            seed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    decision_audit = audit.model_copy(
        update={
            "input_refs": tuple(
                sorted(
                    (
                        binding.trace_envelope_ref,
                        label_spec_ref,
                        observation_policy_ref,
                    ),
                    key=_ref_key,
                )
            )
        }
    )
    return LabelDecisionV2(
        label_decision_id=f"label-decision://sha256/{digest}",
        label_spec_ref=label_spec_ref,
        trace_envelope_ref=binding.trace_envelope_ref,
        decision=LabelDecisionValueV2.ABSTAIN,
        execution_status=LabelExecutionStatus.UNRESOLVED,
        structured_capability_complete=(binding.structured_capability_complete),
        confidence=0.0,
        rule_version="structured-labeling/r3-02-v1",
        model_profile=model_profile,
        prompt_version=prompt_version,
        unresolved_reasons=frozenset({LabelUnresolvedReason.MODEL_UNAVAILABLE}),
        policy_version=LABEL_DECISION_MERGE_POLICY_VERSION,
        decision_sha256=digest,
        audit=decision_audit,
    )


def _label_spec_ref(value: LabelSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-spec",
        object_id=value.label_spec_id,
        object_version=value.label_version,
        object_sha256=value.label_spec_sha256,
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "EXTERNAL_BLOCKED_SEMANTIC_POLICY_VERSION",
    "ExternalBlockedSemanticEvidenceBuilder",
    "ExternalBlockedSemanticEvidenceCompilation",
    "ExternalBlockedSemanticEvidenceError",
]
