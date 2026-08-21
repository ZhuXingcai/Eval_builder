from __future__ import annotations

from dataclasses import dataclass

from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.agent_system.task_authoring_bridge import (
    FactoryTaskAuthoringBridgeResult,
)
from eval_factory.agent_system.task_authoring_material import (
    FactoryTaskAuthoringMaterialStore,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryItemRunBindingV2,
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.task_v2 import r4_task_contract_set_ref


class FactoryTaskAuthoringCommitError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryTaskAuthoringCommit:
    binding: FactoryItemRunBindingV2
    stage_head: FactoryItemStageHeadV2
    material_ref: ObjectRef
    result: FactoryTaskAuthoringBridgeResult


class FactoryTaskAuthoringCommitter:
    """Commits one R4 result under its current item binding."""

    def __init__(
        self,
        *,
        store: FactoryControlStore,
        materials: FactoryTaskAuthoringMaterialStore,
    ) -> None:
        self.store = store
        self.materials = materials

    def commit(
        self,
        *,
        dataset_run_id: str,
        binding: FactoryItemRunBindingV2,
        result: FactoryTaskAuthoringBridgeResult,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> FactoryTaskAuthoringCommit:
        current = self.store.get_item_binding(
            dataset_run_id,
            binding.item_id,
        )
        if current != binding:
            raise FactoryTaskAuthoringCommitError(
                "task authoring item binding is stale",
            )
        core_head = self.store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.CORE_SELECTION,
        )
        try:
            task_head = self.store.get_item_stage_head(
                binding.item_id,
                FactoryItemStageV2.TASK_AUTHORING,
            )
        except FactoryControlNotFoundError:
            material_ref = self.materials.put(result)
            task_head = self.store.commit_item_stage_head(
                FactoryItemStageHeadV2.create(
                    item_binding_ref=binding.to_ref(),
                    item_run_ref=binding.item_run_ref,
                    stage=FactoryItemStageV2.TASK_AUTHORING,
                    stage_version=1,
                    predecessor_head_ref=None,
                    dependency_result_refs=(core_head.result_ref,),
                    result_ref=r4_task_contract_set_ref(
                        result.task_contract_set,
                    ),
                    outcome=FactoryItemStageOutcomeV2.SUCCEEDED,
                    reason_codes=(),
                    audit=audit,
                ),
                idempotency_key=idempotency_key,
                material_ref=material_ref,
            )
        material_ref = self.store.get_item_stage_material_ref(
            task_head.to_ref(),
        )
        restored = self.materials.get(material_ref)
        if (
            task_head.item_binding_ref != binding.to_ref()
            or task_head.item_run_ref != binding.item_run_ref
            or task_head.dependency_result_refs != (core_head.result_ref,)
            or task_head.result_ref != r4_task_contract_set_ref(restored.task_contract_set)
            or restored != result
        ):
            raise FactoryTaskAuthoringCommitError(
                "task authoring stage authority drifted",
            )
        return FactoryTaskAuthoringCommit(
            binding=binding,
            stage_head=task_head,
            material_ref=material_ref,
            result=restored,
        )


__all__ = [
    "FactoryTaskAuthoringCommit",
    "FactoryTaskAuthoringCommitError",
    "FactoryTaskAuthoringCommitter",
]
