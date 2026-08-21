from __future__ import annotations

from typing import Protocol

from eval_factory.agent_system.attachment_execution_material import (
    FactoryAttachmentExecutionMaterialStore,
)
from eval_factory.agent_system.attachment_quality_runtime import (
    FactoryAttachmentQualityContext,
    FactoryAttachmentQualityRuntime,
    FactoryAttachmentQualityView,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.agent_system.task_authoring_material import (
    FactoryTaskAuthoringMaterialStore,
)
from eval_factory.contracts.core import ContractAudit
from eval_factory.contracts.dataset_runtime_v2 import FactoryItemStageV2
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    AttachmentQualityCapabilityRequestV2,
)


class GenericAgentAttachmentQualityExecutionError(RuntimeError):
    pass


class GenericAgentQualityContextSource(Protocol):
    def get(self, item_id: str) -> FactoryAttachmentQualityContext: ...


class GenericAgentAttachmentQualityExecution:
    """Executes the full item-quality owner workflow inside one capability."""

    def __init__(
        self,
        *,
        dataset_run_id: str,
        store: FactoryControlStore,
        task_authoring_materials: FactoryTaskAuthoringMaterialStore,
        attachment_materials: FactoryAttachmentExecutionMaterialStore,
        runtime: FactoryAttachmentQualityRuntime,
        contexts: GenericAgentQualityContextSource,
    ) -> None:
        self.dataset_run_id = dataset_run_id
        self.store = store
        self.task_authoring_materials = task_authoring_materials
        self.attachment_materials = attachment_materials
        self.runtime = runtime
        self.contexts = contexts

    async def execute(
        self,
        request: AttachmentQualityCapabilityRequestV2,
        *,
        audit: ContractAudit,
    ) -> FactoryAttachmentQualityView:
        matches = tuple(
            binding
            for binding in self.store.list_item_bindings(
                self.dataset_run_id,
            )
            if binding.to_ref() == request.item_binding_ref
        )
        if len(matches) != 1:
            raise GenericAgentAttachmentQualityExecutionError(
                "quality item binding is not unique",
            )
        binding = matches[0]
        task_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.TASK_AUTHORING,
        )
        attachment_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.ATTACHMENT,
        )
        if (
            self.store.get_item_stage_material_ref(task_head.to_ref()) != request.task_authoring_material_ref
            or self.store.get_item_stage_material_ref(
                attachment_head.to_ref(),
            )
            != request.attachment_execution_material_ref
        ):
            raise GenericAgentAttachmentQualityExecutionError(
                "quality request differs from current item authority",
            )
        task_authoring = self.task_authoring_materials.get(
            request.task_authoring_material_ref,
        )
        attachment = self.attachment_materials.get(
            request.attachment_execution_material_ref,
        )
        producer_task_view = task_authoring.producer_task_view_result.producer_task_view
        if producer_task_view is None:
            raise GenericAgentAttachmentQualityExecutionError(
                "quality request has no producer task view",
            )
        return await self.runtime.advance(
            binding=binding,
            producer_task_view=producer_task_view,
            leakage_reference_set=(task_authoring.leakage_reference_set),
            task_contract_set=task_authoring.task_contract_set,
            supervised_execution=attachment,
            context=self.contexts.get(binding.item_id),
            audit=audit,
        )


__all__ = [
    "GenericAgentAttachmentQualityExecution",
    "GenericAgentAttachmentQualityExecutionError",
    "GenericAgentQualityContextSource",
]
