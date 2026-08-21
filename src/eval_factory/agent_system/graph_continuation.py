from __future__ import annotations

from typing import Protocol

from eval_factory.agent_system.graph import FactoryGraphDriftError
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalConflictError,
    FactoryGraphJournalStore,
)
from eval_factory.agent_system.graph_transition import (
    FactoryGraphBindingAuthority,
)
from eval_factory.agent_system.plan_review import PlanReviewViewV1
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunStatusV2,
    FactoryRunV2,
    PlanReviewStateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.graph_models import (
    GraphCheckpointOutcomeV1,
    GraphCheckpointPhaseV1,
    HarnessGraphCheckpointV1,
    HarnessGraphExecutionBindingV1,
)


class FactoryGraphContinuationReviewSource(Protocol):
    def show_by_result_ref(
        self,
        reference: ObjectRef,
    ) -> PlanReviewViewV1: ...


class FactoryGraphContinuationFactorySource(Protocol):
    def get_run_by_ref(self, reference: ObjectRef) -> FactoryRunV2: ...


class FactoryGraphContinuationService:
    """Reconciles the exact Factory successor produced by PlanReview."""

    def __init__(
        self,
        *,
        journal: FactoryGraphJournalStore,
        authority: FactoryGraphBindingAuthority,
        factory_source: FactoryGraphContinuationFactorySource,
        review_source: FactoryGraphContinuationReviewSource,
        audit: ContractAudit,
    ) -> None:
        self._journal = journal
        self._authority = authority
        self._factory_source = factory_source
        self._review_source = review_source
        self._audit = audit

    def resume(
        self,
        *,
        graph_binding_ref: ObjectRef,
        expected_checkpoint_ref: ObjectRef,
        review_result_ref: ObjectRef,
        idempotency_key: str,
    ) -> HarnessGraphCheckpointV1:
        prior_binding = self._journal.get_binding_by_ref(
            graph_binding_ref,
        )
        expected = self._journal.get_checkpoint_by_ref(
            expected_checkpoint_ref,
        )
        current_checkpoint = self._journal.current_checkpoint(
            prior_binding.binding_id,
        )
        if (
            current_checkpoint is not None
            and current_checkpoint.phase is GraphCheckpointPhaseV1.RECONCILED
            and current_checkpoint.predecessor_checkpoint_ref == expected_checkpoint_ref
        ):
            return current_checkpoint
        if (
            current_checkpoint is None
            or current_checkpoint.to_ref() != expected_checkpoint_ref
            or expected.binding_ref != graph_binding_ref
        ):
            raise FactoryGraphDriftError(
                "PlanReview continuation checkpoint is stale",
            )
        review = self._review_source.show_by_result_ref(
            review_result_ref,
        )
        if (
            review.result.to_ref() != review_result_ref
            or review.result.state is not PlanReviewStateV2.RESUMED
        ):
            raise FactoryGraphDriftError(
                "PlanReview continuation requires current RESUMED authority",
            )
        prior_run = self._factory_source.get_run_by_ref(
            prior_binding.factory_run_ref,
        )
        request_run = self._factory_source.get_run_by_ref(
            review.request.run_ref,
        )
        if (
            prior_run.status is not FactoryRunStatusV2.WAITING_REVIEW
            or prior_run.pending_review_ref != review.request.to_ref()
            or review.run_id != prior_run.run_id
            or request_run.run_id != prior_run.run_id
            or request_run.run_version >= prior_run.run_version
        ):
            raise FactoryGraphDriftError(
                "PlanReview result does not succeed the checkpoint run",
            )
        successor = self._authority.capture_current(
            graph_binding_ref,
            audit=self._audit,
        )
        self._validate_review_successor(
            prior=prior_binding,
            successor=successor,
            review=review,
        )
        successor = self._journal.commit_binding(
            successor,
            idempotency_key=(f"{idempotency_key}.binding.{successor.object_sha256}"),
        )
        checkpoint = HarnessGraphCheckpointV1.create(
            checkpoint_id=(f"{successor.binding_id}.{expected.transition_number}.{expected.node}.reconciled"),
            binding_ref=successor.to_ref(),
            phase=GraphCheckpointPhaseV1.RECONCILED,
            node=expected.node,
            transition_number=expected.transition_number,
            predecessor_checkpoint_ref=expected.to_ref(),
            factory_run_ref=successor.factory_run_ref,
            team_ref=successor.team_ref,
            team_checkpoint_ref=successor.team_checkpoint_ref,
            blackboard_head_refs=successor.blackboard_head_refs,
            planner_assessment_ref=None,
            outcome=GraphCheckpointOutcomeV1.COMMITTED,
            reason_codes=(),
            audit=self._audit,
        )
        try:
            return self._journal.commit_checkpoint(
                checkpoint,
                idempotency_key=f"{idempotency_key}.checkpoint",
            )
        except FactoryGraphJournalConflictError:
            replay = self._journal.current_checkpoint(
                successor.binding_id,
            )
            if replay == checkpoint:
                return replay
            raise

    @staticmethod
    def _validate_review_successor(
        *,
        prior: HarnessGraphExecutionBindingV1,
        successor: HarnessGraphExecutionBindingV1,
        review: PlanReviewViewV1,
    ) -> None:
        invariant_fields = (
            "binding_id",
            "session_ref",
            "requirement_ref",
            "requirement_policy_ref",
            "factory_request_ref",
            "factory_policy_ref",
            "pack_manifest_ref",
            "composition_ref",
            "blueprint_ref",
            "team_ref",
            "roster_ref",
            "task_graph_ref",
            "execution_authority_ref",
            "team_checkpoint_ref",
            "expected_team_version",
            "expected_task_graph_revision",
            "expected_authority_version",
            "blackboard_head_refs",
            "thread_id",
            "max_transitions",
        )
        if any(getattr(prior, field) != getattr(successor, field) for field in invariant_fields):
            raise FactoryGraphDriftError(
                "PlanReview continuation changed unrelated authority",
            )
        if (
            successor.factory_run_ref != review.current_run_ref
            or successor.expected_factory_run_version <= prior.expected_factory_run_version
        ):
            raise FactoryGraphDriftError(
                "PlanReview continuation did not advance Factory authority",
            )


__all__ = [
    "FactoryGraphContinuationFactorySource",
    "FactoryGraphContinuationReviewSource",
    "FactoryGraphContinuationService",
]
