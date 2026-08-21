from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from eval_factory.agent_system.batch_quality_material import (
    FactoryBatchQualityMaterialStore,
    FactoryBatchQualityResult,
)
from eval_factory.agent_system.candidate_output import (
    AuthorizedCandidateExport,
)
from eval_factory.agent_system.candidate_projection_material import (
    FactoryCandidateProjectionResult,
)
from eval_factory.agent_system.delivery_runtime import (
    FactoryDeliveryWaitingView,
)
from eval_factory.agent_system.release_runtime import (
    FactoryDatasetReleaseContext,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import FactoryRunPolicyV2
from eval_factory.contracts.core import ContractAudit
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetAggregateResultV2,
    FactoryItemStageV2,
)
from eval_factory.packs.generic_agent_trace.item_materials import (
    GenericAgentFactoryItemMaterialSource,
)


class GenericAgentDeliveryPreparationError(RuntimeError):
    pass


class GenericAgentReleaseContextSource(Protocol):
    def release_context(self) -> FactoryDatasetReleaseContext: ...


@dataclass(frozen=True, slots=True)
class GenericAgentDeliveryMaterial:
    aggregate: FactoryDatasetAggregateResultV2
    candidates: tuple[FactoryCandidateProjectionResult, ...]
    exports: tuple[AuthorizedCandidateExport, ...]
    policy: FactoryRunPolicyV2


class GenericAgentDeliveryPreparation:
    """Creates release candidates and the final review before export."""

    def __init__(
        self,
        *,
        dataset_run_id: str,
        store: FactoryControlStore,
        batch_materials: FactoryBatchQualityMaterialStore,
        item_materials: GenericAgentFactoryItemMaterialSource,
        policy: FactoryRunPolicyV2,
        release_context: FactoryDatasetReleaseContext | None = None,
        release_context_source: (GenericAgentReleaseContextSource | None) = None,
    ) -> None:
        self.dataset_run_id = dataset_run_id
        self.store = store
        self.batch_materials = batch_materials
        self.item_materials = item_materials
        self.release_context = release_context
        self.release_context_source = release_context_source
        self.policy = policy
        if (release_context is None) == (release_context_source is None):
            raise ValueError(
                "delivery preparation requires one release context source",
            )

    def prepare_review(
        self,
        *,
        audit: ContractAudit,
    ) -> FactoryDeliveryWaitingView:
        aggregate = self.store.get_dataset_aggregate(
            self.dataset_run_id,
        )
        batch = self.batch_materials.get(
            self.store.get_dataset_aggregate_material_ref(
                self.dataset_run_id,
            )
        )
        if not isinstance(batch, FactoryBatchQualityResult):
            raise GenericAgentDeliveryPreparationError(
                "delivery requires complete Batch Quality material",
            )
        bindings = {
            binding.to_ref(): binding
            for binding in self.store.list_item_bindings(
                self.dataset_run_id,
            )
        }
        try:
            items = tuple(
                self.item_materials.release_source(
                    bindings[reference].item_id,
                )
                for reference in aggregate.candidate_binding_refs
            )
        except KeyError as exc:
            raise GenericAgentDeliveryPreparationError(
                "delivery candidate binding is unavailable",
            ) from exc
        view = self._context().prepare_review(
            dataset_run_id=self.dataset_run_id,
            aggregate=aggregate,
            items=items,
            batch_quality=batch.report,
            factory_policy=self.policy,
            audit=audit,
        )
        if not isinstance(view.delivery, FactoryDeliveryWaitingView):
            raise GenericAgentDeliveryPreparationError(
                "delivery preparation executed output unexpectedly",
            )
        return view.delivery

    def material(self) -> GenericAgentDeliveryMaterial:
        aggregate = self.store.get_dataset_aggregate(
            self.dataset_run_id,
        )
        context = self._context()
        candidates = []
        for reference in aggregate.candidate_binding_refs:
            binding = next(
                (
                    value
                    for value in self.store.list_item_bindings(
                        self.dataset_run_id,
                    )
                    if value.to_ref() == reference
                ),
                None,
            )
            if binding is None:
                raise GenericAgentDeliveryPreparationError(
                    "delivery candidate binding is unavailable",
                )
            head = self.store.get_item_stage_head(
                binding.item_id,
                FactoryItemStageV2.RELEASE_CANDIDATE,
            )
            material_ref = self.store.get_item_stage_material_ref(
                head.to_ref(),
            )
            candidates.append(
                context.runtime.candidates.materials.get(
                    material_ref,
                )
            )
        return GenericAgentDeliveryMaterial(
            aggregate=aggregate,
            candidates=tuple(candidates),
            exports=context.exports,
            policy=self.policy,
        )

    def _context(self) -> FactoryDatasetReleaseContext:
        if self.release_context is not None:
            return self.release_context
        assert self.release_context_source is not None
        return self.release_context_source.release_context()


__all__ = [
    "GenericAgentDeliveryMaterial",
    "GenericAgentDeliveryPreparation",
    "GenericAgentDeliveryPreparationError",
    "GenericAgentReleaseContextSource",
]
