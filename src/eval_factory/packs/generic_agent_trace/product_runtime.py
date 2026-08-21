from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.agent_system.dataset_runtime import FactoryDatasetRuntime
from eval_factory.agent_system.graph_bootstrap import (
    FactoryDatasetGraphBootstrapResult,
)
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalNotFoundError,
)
from eval_factory.agent_system.graph_runtime import FactoryDatasetGraphRun
from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunStatusV2,
    PlanReviewStateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetOutcomeV2,
    FactoryDatasetNextActionV2,
    FactoryDatasetRunRequestV2,
    FactoryDatasetRunViewV2,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs.generic_agent_trace.workflow_bootstrap import (
    GenericAgentWorkflowBootstrapResult,
)


class GenericAgentGraphProductError(RuntimeError):
    pass


class GenericAgentGraphRuntimeConfigV1(ContractModelV2):
    schema_version: Literal["generic-agent-trace/graph-runtime-config/v1"] = (
        "generic-agent-trace/graph-runtime-config/v1"
    )
    claim_scope: Literal["DEVELOPMENT_FIXTURE_ONLY"] = "DEVELOPMENT_FIXTURE_ONLY"
    main_session_id: str = Field(min_length=1, max_length=512)
    control_session_id: str = Field(min_length=1, max_length=512)
    coordinator_session_id: str = Field(min_length=1, max_length=512)
    quality_session_id: str = Field(min_length=1, max_length=512)
    requirement_session_id: str = Field(min_length=1, max_length=512)
    task_session_id: str = Field(min_length=1, max_length=512)
    trace_session_id: str = Field(min_length=1, max_length=512)
    team_id: str = Field(min_length=1, max_length=512)
    team_incarnation_id: str = Field(min_length=1, max_length=512)
    graph_binding_id: str = Field(min_length=1, max_length=512)
    thread_id: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_sessions(self) -> Self:
        members = tuple(self.member_session_ids().values())
        if len(set(members)) != len(members):
            raise ValueError(
                "Graph Team member sessions must be distinct",
            )
        return self

    def member_session_ids(self) -> dict[str, str]:
        return {
            "control": self.control_session_id,
            "coordinator": self.coordinator_session_id,
            "quality": self.quality_session_id,
            "requirement": self.requirement_session_id,
            "task": self.task_session_id,
            "trace": self.trace_session_id,
        }


@dataclass(frozen=True, slots=True)
class GenericAgentGraphProductPaths:
    team_store: Path
    request_store: Path
    graph_journal: Path
    graph_checkpoint: Path

    def resolved(self) -> GenericAgentGraphProductPaths:
        return GenericAgentGraphProductPaths(
            team_store=self.team_store.expanduser().resolve(),
            request_store=self.request_store.expanduser().resolve(),
            graph_journal=self.graph_journal.expanduser().resolve(),
            graph_checkpoint=self.graph_checkpoint.expanduser().resolve(),
        )


@dataclass(frozen=True, slots=True)
class GenericAgentGraphProduct:
    graph: FactoryDatasetGraphBootstrapResult
    workflow: GenericAgentWorkflowBootstrapResult
    dataset_runtime: FactoryDatasetRuntime
    dataset_store: FactoryControlStore
    request: FactoryDatasetRunRequestV2
    audit: ContractAudit

    async def advance(self) -> FactoryDatasetRunViewV2:
        terminal = self._terminal_aggregate_view()
        if terminal is not None:
            return terminal
        runtime = self.graph.runtime
        journal = self.graph.journal
        try:
            binding = journal.get_binding(
                self.graph.binding.binding_id,
            )
        except FactoryGraphJournalNotFoundError:
            run = await runtime.start(
                self.graph.binding,
                idempotency_key=(f"start-product-graph:{self.graph.binding.object_sha256}"),
            )
        else:
            checkpoint = journal.current_checkpoint(
                binding.binding_id,
            )
            if checkpoint is None:
                run = await runtime.start(
                    binding,
                    idempotency_key=(f"start-product-graph:{binding.object_sha256}"),
                )
            elif checkpoint.outcome.value == "WAITING_REVIEW":
                run = await self._continue_review(
                    binding_ref=binding.to_ref(),
                    checkpoint_ref=checkpoint.to_ref(),
                )
            else:
                run = await runtime.continue_current(
                    binding.to_ref(),
                )
        runtime.reconcile(run.binding.to_ref())
        current = self.dataset_runtime.current_view(
            request=self.request,
            audit=self.audit,
        )
        terminal = self._terminal_aggregate_view(current)
        if terminal is not None:
            return terminal
        if current.pending_review_refs:
            return current
        pending = self._pending_domain_reviews()
        if not pending:
            return current
        return FactoryDatasetRunViewV2.create(
            request_ref=current.request_ref,
            dataset_run_ref=current.dataset_run_ref,
            status=FactoryRunStatusV2.WAITING_REVIEW,
            pending_review_refs=pending,
            item_binding_refs=current.item_binding_refs,
            candidate_count=current.candidate_count,
            rejected_count=current.rejected_count,
            blocked_count=current.blocked_count,
            incomplete_count=current.incomplete_count,
            aggregate_result_ref=current.aggregate_result_ref,
            delivery_manifest_ref=current.delivery_manifest_ref,
            next_action=FactoryDatasetNextActionV2.REVIEW_PLAN,
            audit=self.audit,
        )

    def _terminal_aggregate_view(
        self,
        current: FactoryDatasetRunViewV2 | None = None,
    ) -> FactoryDatasetRunViewV2 | None:
        try:
            aggregate = self.dataset_store.get_dataset_aggregate(
                self.request.dataset_run_id,
            )
        except FactoryControlNotFoundError:
            return None
        if aggregate.outcome in {
            CandidateDatasetOutcomeV2.COMPLETE,
            CandidateDatasetOutcomeV2.PARTIAL,
        }:
            return None
        view = current or self.dataset_runtime.current_view(
            request=self.request,
            audit=self.audit,
        )
        if view.aggregate_result_ref != aggregate.to_ref():
            raise GenericAgentGraphProductError(
                "terminal aggregate differs from Factory run view",
            )
        return FactoryDatasetRunViewV2.create(
            request_ref=view.request_ref,
            dataset_run_ref=view.dataset_run_ref,
            status=FactoryRunStatusV2.BLOCKED,
            pending_review_refs=(),
            item_binding_refs=view.item_binding_refs,
            candidate_count=view.candidate_count,
            rejected_count=view.rejected_count,
            blocked_count=view.blocked_count,
            incomplete_count=view.incomplete_count,
            aggregate_result_ref=aggregate.to_ref(),
            delivery_manifest_ref=None,
            next_action=FactoryDatasetNextActionV2.NONE,
            audit=self.audit,
        )

    def _pending_domain_reviews(self) -> tuple[ObjectRef, ...]:
        run_ids = {self.request.dataset_run_id}
        run_ids.update(
            self.dataset_store.get_item_run(binding).run_id
            for binding in self.dataset_store.list_item_bindings(
                self.request.dataset_run_id,
            )
        )
        page = self.dataset_runtime.plan_reviews.list_reviews(
            state=PlanReviewStateV2.PENDING_REVIEW,
            limit=500,
        )
        return sorted_refs(value.request.to_ref() for value in page.items if value.run_id in run_ids)

    async def _continue_review(
        self,
        *,
        binding_ref: ObjectRef,
        checkpoint_ref: ObjectRef,
    ) -> FactoryDatasetGraphRun:
        runtime = self.graph.runtime
        binding = self.graph.journal.get_binding_by_ref(
            binding_ref,
        )
        prior_run = self.dataset_store.get_run_by_ref(
            binding.factory_run_ref,
        )
        if prior_run.status is not FactoryRunStatusV2.WAITING_REVIEW or prior_run.pending_review_ref is None:
            raise GenericAgentGraphProductError(
                "waiting Graph checkpoint has no review authority",
            )
        review = self.dataset_runtime.plan_reviews.show(
            prior_run.pending_review_ref.object_id,
        )
        current_run = self.dataset_store.get_run(
            prior_run.run_id,
        )
        if review.result.state is not PlanReviewStateV2.RESUMED:
            if current_run.to_ref() != prior_run.to_ref():
                raise GenericAgentGraphProductError(
                    "Factory authority advanced without a resumed review",
                )
            return runtime.show(binding_ref)
        return await runtime.resume(
            graph_binding_ref=binding_ref,
            expected_checkpoint_ref=checkpoint_ref,
            review_result_ref=review.result.to_ref(),
            idempotency_key=(f"resume-product-graph:{review.result.object_sha256}"),
        )


__all__ = [
    "GenericAgentGraphProduct",
    "GenericAgentGraphProductError",
    "GenericAgentGraphProductPaths",
    "GenericAgentGraphRuntimeConfigV1",
]
