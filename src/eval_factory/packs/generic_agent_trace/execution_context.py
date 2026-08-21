from __future__ import annotations

from collections.abc import Mapping

from eval_factory.agent_system.batch_quality_runtime import (
    FactoryBatchQualityContext,
)
from eval_factory.agent_system.dataset_runtime import (
    FactoryBatchQualityContextFactory,
    FactoryDatasetReleaseContextFactory,
    FactoryItemSpecialistContextFactory,
)
from eval_factory.agent_system.delivery_runtime import FactoryDeliveryRuntime
from eval_factory.agent_system.item_run_materializer import (
    FactoryItemRunMaterializer,
)
from eval_factory.agent_system.item_specialist_runtime import (
    FactoryItemSpecialistContext,
)
from eval_factory.agent_system.job_store_bridge import (
    FactoryJobStoreAuthority,
    FactoryJobStoreBridge,
)
from eval_factory.agent_system.job_store_witness_bridge import (
    FactoryJobStoreWitnessBridge,
)
from eval_factory.agent_system.release_runtime import (
    FactoryDatasetReleaseContext,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.agent_system.trace_candidate_store import (
    FactoryTraceCandidateMaterialStore,
)
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
)
from eval_factory.contracts.orchestration import TraceSourceRef
from eval_factory.harness.contracts import sorted_refs, static_object_ref
from eval_factory.packs.generic_agent_trace.item_materials import (
    GenericAgentCriteriaContext,
    GenericAgentFactoryItemMaterialSource,
    GenericAgentGradingContext,
)


class GenericAgentExecutionContextError(RuntimeError):
    pass


class GenericAgentFactoryExecutionContext:
    """Materializes shared owner authority after Trace dispositions."""

    def __init__(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        requirement: EvaluationRequirementSpecV2,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
        source_refs: tuple[ObjectRef, ...],
        trace_sources: Mapping[ObjectRef, TraceSourceRef],
        store: FactoryControlStore,
        candidate_store: FactoryTraceCandidateMaterialStore,
        item_materializer: FactoryItemRunMaterializer,
        job_store_bridge: FactoryJobStoreBridge,
        item_context_factory: FactoryItemSpecialistContextFactory,
        batch_context_factory: FactoryBatchQualityContextFactory,
        release_context_factory: FactoryDatasetReleaseContextFactory,
    ) -> None:
        self.request = request
        self.requirement = requirement
        self.policy = policy
        self.audit = audit
        self.source_refs = sorted_refs(source_refs)
        self.trace_sources = dict(trace_sources)
        if set(self.trace_sources) != set(self.source_refs):
            raise ValueError(
                "execution context Trace sources differ from scope refs",
            )
        self.store = store
        self.candidate_store = candidate_store
        self.item_materializer = item_materializer
        self.job_store_bridge = job_store_bridge
        self.item_context_factory = item_context_factory
        self.batch_context_factory = batch_context_factory
        self.release_context_factory = release_context_factory
        self._authority: FactoryJobStoreAuthority | None = None

    def ensure(self) -> FactoryJobStoreAuthority:
        candidate_materials = tuple(
            self.candidate_store.get(
                run_id=self.request.dataset_run_id,
                trace_source_ref=source_ref,
            )
            for source_ref in self.source_refs
        )
        candidates = tuple(
            material
            for material in candidate_materials
            if material.preparation.decision.disposition is TraceCandidateDispositionV2.CANDIDATE
        )
        candidate_refs = sorted_refs(
            self._item_source_ref(
                material.authority.trace_source_ref,
            )
            for material in candidates
        )
        for material in candidates:
            source_ref = self._item_source_ref(
                material.authority.trace_source_ref,
            )
            self.item_materializer.materialize(
                request=self.request,
                policy=self.policy,
                requirement=self.requirement,
                source_trace_ref=source_ref,
                candidate=material.preparation,
                audit=self.audit,
            )
        authority = self.job_store_bridge.prepare(
            dataset_run_id=self.request.dataset_run_id,
            source_trace_refs=(candidate_refs if candidate_refs else self.source_refs),
            audit=self.audit,
        )
        if self._authority is not None and self._authority != authority:
            raise GenericAgentExecutionContextError(
                "owner execution authority drifted",
            )
        self._authority = authority
        return authority

    def criteria_context(
        self,
        item_id: str,
    ) -> GenericAgentCriteriaContext:
        context = self._item_context(item_id)
        bindings = context.evaluator_bindings
        return GenericAgentCriteriaContext(
            quality_context=context.quality,
            binding_definitions=bindings,
            binding_definitions_ref=static_object_ref(
                object_type="evaluator-binding-definitions",
                object_id=(f"evaluator-binding-definitions://{self.request.dataset_run_id}/{item_id}"),
                object_version="v2",
                payload={
                    "bindings": [value.model_dump(mode="json") for value in bindings],
                },
            ),
            tool_catalog=context.tool_catalog,
        )

    def grading_context(
        self,
        item_id: str,
    ) -> GenericAgentGradingContext:
        context = self._item_context(item_id)
        authorizations = context.model_authorizations
        return GenericAgentGradingContext(
            model_authorizations=authorizations,
            model_authorizations_ref=static_object_ref(
                object_type="model-domain-authorizations",
                object_id=(f"model-domain-authorizations://{self.request.dataset_run_id}/{item_id}"),
                object_version="v2",
                payload={
                    "authorizations": [value.model_dump(mode="json") for value in authorizations],
                },
            ),
            evaluated_at=context.evaluated_at,
        )

    def batch_context(self) -> FactoryBatchQualityContext:
        return self.batch_context_factory.build(
            job_authority=self.ensure(),
        )

    def source_trace_ref(self, item_id: str) -> ObjectRef:
        authority = self.ensure()
        try:
            index = authority.graph.item_ids.index(item_id)
        except ValueError as exc:
            raise GenericAgentExecutionContextError(
                "Item is absent from owner execution graph",
            ) from exc
        return authority.graph.source_trace_refs[index]

    def release_context(self) -> FactoryDatasetReleaseContext:
        authority = self.ensure()
        aggregate = self.store.get_dataset_aggregate(
            self.request.dataset_run_id,
        )
        bindings = {
            binding.to_ref(): binding
            for binding in self.store.list_item_bindings(
                self.request.dataset_run_id,
            )
        }
        try:
            item_ids = tuple(bindings[reference].item_id for reference in aggregate.candidate_binding_refs)
        except KeyError as exc:
            raise GenericAgentExecutionContextError(
                "release candidate binding is unavailable",
            ) from exc
        return self.release_context_factory.build(
            job_authority=authority,
            item_ids=item_ids,
        )

    def delivery_runtime(self) -> FactoryDeliveryRuntime:
        return self.release_context().runtime.delivery

    def _item_context(
        self,
        item_id: str,
    ) -> FactoryItemSpecialistContext:
        authority = self.ensure()
        binding = self.store.get_item_binding(
            self.request.dataset_run_id,
            item_id,
        )
        return self.item_context_factory.build(
            binding=binding,
            job_authority=authority,
        )

    def _item_source_ref(
        self,
        source_ref: ObjectRef,
    ) -> ObjectRef:
        source = self.trace_sources[source_ref]
        return source_ref.model_copy(
            update={"object_version": source.adapter_version},
        )


class GenericAgentBatchExecutionContext:
    def __init__(
        self,
        *,
        execution: GenericAgentFactoryExecutionContext,
        item_materials: GenericAgentFactoryItemMaterialSource,
        witness: FactoryJobStoreWitnessBridge,
    ) -> None:
        self.execution = execution
        self.item_materials = item_materials
        self.witness = witness

    def batch_context(self) -> FactoryBatchQualityContext:
        return self.execution.batch_context()

    def commit_item_witnesses(
        self,
        *,
        audit: ContractAudit,
    ) -> None:
        authority = self.execution.ensure()
        bindings = self.execution.store.list_item_bindings(
            self.execution.request.dataset_run_id,
        )
        if not bindings:
            return
        self.witness.commit_items(
            graph=authority.graph,
            sources=tuple(self.item_materials.witness_source(binding.item_id) for binding in bindings),
            audit=audit,
        )

    def commit_batch_witness(
        self,
        *,
        batch_quality_ref: ObjectRef,
        audit: ContractAudit,
    ) -> None:
        self.witness.commit_batch(
            graph=self.execution.ensure().graph,
            batch_quality_ref=batch_quality_ref,
            audit=audit,
        )


__all__ = [
    "GenericAgentBatchExecutionContext",
    "GenericAgentExecutionContextError",
    "GenericAgentFactoryExecutionContext",
]
