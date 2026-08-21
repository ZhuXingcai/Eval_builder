from __future__ import annotations

import hashlib

from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.agent_system.trace_candidate import (
    TraceCandidatePreparation,
)
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
    FactoryItemRunBindingV2,
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.orchestration_v2 import dataset_item_id_v2


class FactoryItemRunMaterializerError(RuntimeError):
    pass


class FactoryItemRunMaterializer:
    """Creates one deterministic child run and CORE_SELECTION authority."""

    def __init__(self, store: FactoryControlStore) -> None:
        self.store = store

    def materialize(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        source_trace_ref: ObjectRef,
        candidate: TraceCandidatePreparation,
        audit: ContractAudit,
    ) -> FactoryItemRunBindingV2:
        decision = candidate.decision
        prompt = candidate.extracted_prompt
        intent = candidate.inferred_intent
        rewrite = candidate.rewrite_candidate
        if (
            decision.disposition is not TraceCandidateDispositionV2.CANDIDATE
            or decision.cleaned_trace_ref is None
            or prompt is None
            or intent is None
            or rewrite is None
            or prompt.source_spans[0].source_trace_id != decision.source_trace_id
            or intent.extracted_prompt_ref != prompt.to_ref()
            or rewrite.extracted_prompt_ref != prompt.to_ref()
            or rewrite.inferred_intent_ref != intent.to_ref()
            or source_trace_ref.object_type != "trace-source"
            or source_trace_ref.object_id != decision.source_ref.object_id
            or source_trace_ref.object_sha256 != decision.source_ref.object_sha256
        ):
            raise FactoryItemRunMaterializerError(
                "candidate material is incomplete or stale",
            )
        parent = self.store.get_run(request.dataset_run_id)
        if (
            request.requirement_spec_ref != requirement.to_ref()
            or request.factory_policy_ref != policy.to_ref()
            or parent.requirement_spec_ref != requirement.to_ref()
            or parent.policy_ref != policy.to_ref()
        ):
            raise FactoryItemRunMaterializerError(
                "candidate parent authority is stale",
            )
        item_id = dataset_item_id_v2(
            request.dataset_run_id,
            source_trace_ref,
        )
        digest = hashlib.sha256(item_id.encode()).hexdigest()
        item_run_id = f"factory-run://dataset-item/sha256/{digest}"
        child_requirement = EvaluationRequirementSpecV2.create(
            requirement_spec_id=(f"evaluation-requirement-spec://dataset-item/{digest}"),
            run_id=item_run_id,
            source_ref=requirement.source_ref,
            goals=requirement.goals,
            constraints=requirement.constraints,
            assumptions=requirement.assumptions,
            open_questions=requirement.open_questions,
            requirement_version=1,
            audit=audit,
        )
        child = self.store.create_run(
            policy=policy,
            requirement=child_requirement,
            idempotency_key=(f"{request.idempotency_key}:child:{digest}"),
        )
        try:
            binding = self.store.get_item_binding(
                request.dataset_run_id,
                item_id,
            )
        except FactoryControlNotFoundError:
            binding = self.store.commit_item_binding(
                FactoryItemRunBindingV2.create(
                    dataset_run_ref=parent.to_ref(),
                    item_run_ref=child.to_ref(),
                    item_id=item_id,
                    binding_version=1,
                    predecessor_binding_ref=None,
                    trace_candidate_decision_ref=decision.to_ref(),
                    extracted_prompt_ref=prompt.to_ref(),
                    inferred_intent_ref=intent.to_ref(),
                    rewrite_candidate_ref=rewrite.to_ref(),
                    audit=audit,
                ),
                idempotency_key=(f"{request.idempotency_key}:binding:{digest}"),
            )
        bound_parent = self.store.get_run_by_ref(
            binding.dataset_run_ref,
        )
        if (
            bound_parent.run_id != parent.run_id
            or bound_parent.policy_ref != policy.to_ref()
            or bound_parent.requirement_spec_ref != requirement.to_ref()
        ):
            raise FactoryItemRunMaterializerError(
                "item binding parent authority is stale",
            )
        expected = FactoryItemRunBindingV2.create(
            dataset_run_ref=binding.dataset_run_ref,
            item_run_ref=child.to_ref(),
            item_id=item_id,
            binding_version=1,
            predecessor_binding_ref=None,
            trace_candidate_decision_ref=decision.to_ref(),
            extracted_prompt_ref=prompt.to_ref(),
            inferred_intent_ref=intent.to_ref(),
            rewrite_candidate_ref=rewrite.to_ref(),
            audit=audit,
        )
        if binding != expected:
            raise FactoryItemRunMaterializerError(
                "existing item binding differs from candidate authority",
            )
        head = FactoryItemStageHeadV2.create(
            item_binding_ref=binding.to_ref(),
            item_run_ref=binding.item_run_ref,
            stage=FactoryItemStageV2.CORE_SELECTION,
            stage_version=1,
            predecessor_head_ref=None,
            dependency_result_refs=(
                decision.to_ref(),
                prompt.to_ref(),
                intent.to_ref(),
            ),
            result_ref=rewrite.to_ref(),
            outcome=FactoryItemStageOutcomeV2.SUCCEEDED,
            reason_codes=(),
            audit=audit,
        )
        committed = self.store.commit_item_stage_head(
            head,
            idempotency_key=(f"{request.idempotency_key}:core-head:{digest}"),
        )
        if committed != head:
            raise FactoryItemRunMaterializerError(
                "CORE_SELECTION head differs from candidate authority",
            )
        return binding


__all__ = [
    "FactoryItemRunMaterializer",
    "FactoryItemRunMaterializerError",
]
