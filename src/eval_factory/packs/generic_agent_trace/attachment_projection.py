from __future__ import annotations

from eval_factory.agent_system.attachment_execution_material import (
    FactoryAttachmentExecutionMaterialStore,
)
from eval_factory.agent_system.attachment_subgraph import (
    AttachmentSupervisedExecution,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    AttachmentSubgraphOutcomeV2,
    PlanKindV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    AttachmentReconstructionCapabilityRequestV1,
)


class GenericAgentAttachmentProjectionError(RuntimeError):
    pass


class GenericAgentAttachmentProjection:
    """Projects successful Harness attachment work into Factory authority."""

    def __init__(
        self,
        *,
        dataset_run_id: str,
        store: FactoryControlStore,
        materials: FactoryAttachmentExecutionMaterialStore,
    ) -> None:
        self.dataset_run_id = dataset_run_id
        self.store = store
        self.materials = materials

    def commit(
        self,
        *,
        request: AttachmentReconstructionCapabilityRequestV1,
        execution: AttachmentSupervisedExecution,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> tuple[ObjectRef, ...]:
        result = execution.subgraph_result
        if result.outcome is not AttachmentSubgraphOutcomeV2.SUCCEEDED:
            raise GenericAgentAttachmentProjectionError(
                "only successful attachment execution may be projected",
            )
        matches = tuple(
            binding
            for binding in self.store.list_item_bindings(
                self.dataset_run_id,
            )
            if self.store.get_item_run(binding).run_id == request.run_id
        )
        if len(matches) != 1:
            raise GenericAgentAttachmentProjectionError(
                "attachment item binding is not unique",
            )
        binding = matches[0]
        current_domain = self.store.get_domain_result(
            request.run_id,
            PlanKindV2.ATTACHMENT_GENERATION,
        )
        if current_domain.result_ref != result.to_ref() or result.plan_ref != request.plan_ref:
            raise GenericAgentAttachmentProjectionError(
                "attachment execution differs from owner authority",
            )
        task_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.TASK_AUTHORING,
        )
        material_ref = self.materials.put(execution)
        stage_head = FactoryItemStageHeadV2.create(
            item_binding_ref=binding.to_ref(),
            item_run_ref=binding.item_run_ref,
            stage=FactoryItemStageV2.ATTACHMENT,
            stage_version=1,
            predecessor_head_ref=None,
            dependency_result_refs=(task_head.result_ref,),
            result_ref=result.to_ref(),
            outcome=FactoryItemStageOutcomeV2.SUCCEEDED,
            reason_codes=(),
            audit=audit,
        )
        committed = self.store.commit_item_stage_head(
            stage_head,
            idempotency_key=f"{idempotency_key}.head",
            material_ref=material_ref,
        )
        if self.materials.get(material_ref) != execution:
            raise GenericAgentAttachmentProjectionError(
                "attachment execution material drifted",
            )
        return sorted_refs(
            (
                committed.to_ref(),
                material_ref,
            )
        )


__all__ = [
    "GenericAgentAttachmentProjection",
    "GenericAgentAttachmentProjectionError",
]
