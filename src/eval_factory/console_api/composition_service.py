from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from eval_factory.agent_system.candidate_output import (
    CandidateDatasetOutputAssembler,
    CandidateOutputIntegrityError,
    CandidateOutputWrite,
)
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalIntegrityError,
    FactoryGraphJournalNotFoundError,
    FactoryGraphJournalStore,
)
from eval_factory.agent_system.graph_projection import (
    FactoryGraphSessionReconciler,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewService,
    PlanReviewViewV1,
)
from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.console_api.agent_contracts import (
    AgentShellDeliverySummaryV1,
    AgentShellExecutionPhaseV1,
    AgentShellFactorySummaryV1,
    AgentShellGraphSummaryV1,
    AgentShellInteractionCardV1,
    AgentShellInteractionKindV1,
    AgentShellInteractionStateV1,
    AgentShellMemberProjectionV1,
    AgentShellPlanReviewSummaryV1,
    AgentShellTeamMemberSummaryV1,
    AgentShellTeamMessageSummaryV1,
    AgentShellTeamSummaryV1,
    AgentShellTeamTaskSummaryV1,
    AgentShellWorkspaceDescriptorV1,
    AgentShellWorkspaceKindV1,
    AgentShellWorkspaceStatusV1,
)
from eval_factory.console_api.composition_types import (
    AgentShellCompositionConflictError,
    AgentShellCompositionIntegrityError,
    AgentShellCompositionNotFoundError,
    AgentShellCompositionPreflight,
    AgentShellGraphStarter,
)
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunStatusV2,
    PlanReviewStateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunViewV2,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.harness.graph_models import (
    GraphCheckpointOutcomeV1,
    HarnessGraphCheckpointV1,
    HarnessGraphExecutionBindingV1,
)
from eval_factory.harness.runtime_models import (
    HarnessSessionProjectionV1,
    HarnessTurnOutcomeV1,
    HarnessTurnResultV1,
    ProviderEvidenceClassV1,
    RequirementInterpretationOutcomeV1,
    RequirementInterpretationV1,
)
from eval_factory.harness.session_service import HarnessSessionService
from eval_factory.harness.source_admission import (
    HarnessSourceAdmissionNotFoundError,
    HarnessSourceAdmissionStore,
    HarnessSourceAdmissionWrite,
)
from eval_factory.team.models import (
    TeamTaskStatusV1,
)
from eval_factory.team.service import TeamSessionReconciler
from eval_factory.team.store import TeamNotFoundError, TeamStore


@dataclass(frozen=True, slots=True)
class AgentShellCompositionProjection:
    execution_phase: AgentShellExecutionPhaseV1
    requirement_outcome: RequirementInterpretationOutcomeV1 | None
    evidence_class: ProviderEvidenceClassV1 | None
    source_admission_ref: ObjectRef | None
    factory: AgentShellFactorySummaryV1 | None
    graph: AgentShellGraphSummaryV1 | None
    team: AgentShellTeamSummaryV1 | None
    plan_reviews: tuple[AgentShellPlanReviewSummaryV1, ...]
    pending_interactions: tuple[AgentShellInteractionCardV1, ...]
    workspaces: tuple[AgentShellWorkspaceDescriptorV1, ...]
    delivery: AgentShellDeliverySummaryV1 | None
    source_fingerprint: str


class AgentShellCompositionService:
    """Projects independent owner authorities without copying their state."""

    def __init__(
        self,
        *,
        sessions: HarnessSessionService,
        sources: HarnessSourceAdmissionStore,
        team_store: TeamStore,
        factory_store: FactoryControlStore,
        graph_journal: FactoryGraphJournalStore,
        plan_reviews: PlanReviewService,
        candidate_output: CandidateDatasetOutputAssembler,
        graph_starter: AgentShellGraphStarter,
    ) -> None:
        if plan_reviews.store is not factory_store or candidate_output.store is not factory_store:
            raise ValueError(
                "Agent Shell composition must share Factory authority",
            )
        roots = (
            sessions.store.path.expanduser().resolve(),
            sources.root.expanduser().resolve(),
            team_store.path.expanduser().resolve(),
            factory_store.path.expanduser().resolve(),
            graph_journal.path.expanduser().resolve(),
            candidate_output.root.expanduser().resolve(),
        )
        if any(
            left == right or left in right.parents or right in left.parents
            for index, left in enumerate(roots)
            for right in roots[index + 1 :]
        ):
            raise ValueError(
                "Agent Shell owner stores require non-overlapping paths",
            )
        self.sessions = sessions
        self.sources = sources
        self.team_store = team_store
        self.factory_store = factory_store
        self.graph_journal = graph_journal
        self.plan_reviews = plan_reviews
        self.candidate_output = candidate_output
        self.graph_starter = graph_starter
        self._team_reconciler = TeamSessionReconciler(
            team_store=team_store,
            session_sink=sessions.store,
        )
        self._graph_reconciler = FactoryGraphSessionReconciler(
            journal=graph_journal,
            session_sink=sessions.store,
        )

    def before_turn(
        self,
        session_id: str,
        *,
        idempotency_key: str,
    ) -> None:
        projection = self.sessions.get_session(session_id)
        if self._binding_or_none(
            projection.session.session_ref,
        ) is not None and not self.sessions.has_message_command(
            session_id=session_id,
            idempotency_key=idempotency_key,
        ):
            raise AgentShellCompositionConflictError(
                "Graph-bound session does not accept a new requirement turn",
            )

    async def after_turn(
        self,
        session_id: str,
        *,
        turn: HarnessTurnResultV1,
        principal: str,
        audit: ContractAudit,
    ) -> None:
        if turn.outcome is not HarnessTurnOutcomeV1.READY:
            return
        projection = self.sessions.get_session(session_id)
        binding = self._binding_or_none(
            projection.session.session_ref,
        )
        if binding is None:
            preflight = self.graph_starter.preflight(session_id)
            if not preflight.ready:
                return
            await self.graph_starter.start(
                session_id,
                turn=turn,
                principal=principal,
                audit=audit,
            )
        binding = self._binding_or_none(
            projection.session.session_ref,
        )
        if binding is None:
            raise AgentShellCompositionIntegrityError(
                "Graph starter returned without a committed binding",
            )
        self._reconcile_outboxes(
            binding,
            audit=audit,
        )

    async def reconcile(
        self,
        session_id: str,
        *,
        expected_binding_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> int:
        projection = self.sessions.get_session(session_id)
        binding = self._binding_or_none(
            projection.session.session_ref,
        )
        if binding is None:
            if expected_binding_ref is not None:
                raise AgentShellCompositionIntegrityError(
                    "reconcile Graph binding disappeared",
                )
            return 0
        reconciled = self._reconcile_outboxes(
            binding,
            audit=audit,
        )
        if binding.to_ref() != expected_binding_ref:
            return reconciled
        await self.graph_starter.advance(
            session_id,
            binding=binding,
            audit=audit,
        )
        successor = self._require_binding(
            projection.session.session_ref,
        )
        return reconciled + self._reconcile_outboxes(
            successor,
            audit=audit,
        )

    def reconcile_binding_ref(
        self,
        session_id: str,
    ) -> ObjectRef | None:
        projection = self.sessions.get_session(session_id)
        binding = self._binding_or_none(
            projection.session.session_ref,
        )
        return binding.to_ref() if binding is not None else None

    def _reconcile_outboxes(
        self,
        binding: HarnessGraphExecutionBindingV1,
        *,
        audit: ContractAudit,
    ) -> int:
        team_id = self._team_id(binding)
        team_events = self._team_reconciler.reconcile(
            team_id,
            audit=audit,
        )
        graph_events = self._graph_reconciler.reconcile(
            binding.binding_id,
            audit=audit,
        )
        return len(team_events) + len(graph_events)

    def project(
        self,
        session: HarnessSessionProjectionV1,
    ) -> AgentShellCompositionProjection:
        interpretation = self._interpretation(session)
        source = self._source_or_none(session.session.session_id)
        binding = self._binding_or_none(session.session.session_ref)
        if binding is None:
            preflight = self._preflight(
                session=session,
                interpretation=interpretation,
                source=source,
            )
            phase = self._phase_without_graph(
                interpretation=interpretation,
                source=source,
                preflight=preflight,
            )
            interactions = self._interactions(
                session=session,
                interpretation=interpretation,
                reviews=(),
                checkpoint=None,
                preflight=preflight,
            )
            workspaces = self._workspaces(
                source=source,
                team=None,
                reviews=(),
                delivery=None,
                execution_phase=phase,
            )
            return self._projection(
                session=session,
                phase=phase,
                interpretation=interpretation,
                source=source,
                factory=None,
                graph=None,
                team=None,
                reviews=(),
                interactions=interactions,
                workspaces=workspaces,
                delivery=None,
            )
        (
            run_view,
            checkpoint,
            team,
            reviews,
            delivery,
        ) = self._owner_projection(
            session=session,
            binding=binding,
        )
        factory = self._factory_summary(
            binding=binding,
            view=run_view,
        )
        graph = self._graph_summary(
            binding=binding,
            checkpoint=checkpoint,
        )
        phase = self._phase_with_graph(
            view=run_view,
            checkpoint=checkpoint,
            reviews=reviews,
            delivery=delivery,
        )
        interactions = self._interactions(
            session=session,
            interpretation=interpretation,
            reviews=reviews,
            checkpoint=checkpoint,
            preflight=None,
        )
        workspaces = self._workspaces(
            source=source,
            team=team,
            reviews=reviews,
            delivery=delivery,
            execution_phase=phase,
        )
        return self._projection(
            session=session,
            phase=phase,
            interpretation=interpretation,
            source=source,
            factory=factory,
            graph=graph,
            team=team,
            reviews=reviews,
            interactions=interactions,
            workspaces=workspaces,
            delivery=delivery,
        )

    def member_projection(
        self,
        session_id: str,
        member_id: str,
    ) -> AgentShellMemberProjectionV1:
        session = self.sessions.get_session(session_id)
        binding = self._require_binding(session.session.session_ref)
        team_id = self._team_id(binding)
        source = self.team_store.projection_source(
            team_id,
            member_id,
        )
        return AgentShellMemberProjectionV1(
            team_ref=source.snapshot.team.to_ref(),
            member=self._member_summary(source.member),
            task_refs=sorted_refs(value.to_ref() for value in source.tasks),
            artifact_envelope_refs=sorted_refs(value.to_ref() for value in source.artifact_envelopes),
            messages=tuple(
                AgentShellTeamMessageSummaryV1(
                    message_ref=value.to_ref(),
                    sender_member_ref=value.sender_member_ref,
                    audience=value.audience,
                    message_kind=value.message_kind,
                    recipient_member_refs=value.recipient_member_refs,
                    task_ref=value.task_ref,
                    artifact_envelope_refs=value.artifact_envelope_refs,
                    reply_to_message_ref=value.reply_to_message_ref,
                )
                for value in source.messages
            ),
            subscription_refs=sorted_refs(value.to_ref() for value in source.subscriptions),
            acceptance_check_refs=source.acceptance_check_refs,
            source_fingerprint=source.source_fingerprint,
        )

    def _owner_projection(
        self,
        *,
        session: HarnessSessionProjectionV1,
        binding: HarnessGraphExecutionBindingV1,
    ) -> tuple[
        FactoryDatasetRunViewV2,
        HarnessGraphCheckpointV1 | None,
        AgentShellTeamSummaryV1,
        tuple[AgentShellPlanReviewSummaryV1, ...],
        AgentShellDeliverySummaryV1 | None,
    ]:
        if binding.session_ref != session.session.session_ref:
            raise AgentShellCompositionConflictError(
                "Graph binding belongs to another Harness session",
            )
        if (
            binding.composition_ref != session.session.composition_ref
            or binding.requirement_policy_ref != session.session.current_requirement_policy_ref
        ):
            raise AgentShellCompositionConflictError(
                "Graph binding uses stale Harness authority",
            )
        try:
            run = self.factory_store.get_run_by_ref(
                binding.factory_run_ref,
            )
            current_run = self.factory_store.get_run(run.run_id)
        except FactoryControlNotFoundError as exc:
            raise AgentShellCompositionIntegrityError(
                "Graph binding references missing Factory authority",
            ) from exc
        if (
            current_run.to_ref() != binding.factory_run_ref
            or current_run.run_version != binding.expected_factory_run_version
        ):
            raise AgentShellCompositionIntegrityError(
                "Graph binding does not reference current Factory authority",
            )
        run_view = self.graph_starter.current_view(
            session.session.session_id,
            binding=binding,
        )
        if run_view.dataset_run_ref != current_run.to_ref():
            raise AgentShellCompositionIntegrityError(
                "Factory projection differs from current Graph binding",
            )
        checkpoint = self.graph_journal.current_checkpoint(
            binding.binding_id,
        )
        if checkpoint is not None and (
            checkpoint.binding_ref != binding.to_ref()
            or checkpoint.factory_run_ref != binding.factory_run_ref
            or checkpoint.team_ref != binding.team_ref
            or checkpoint.team_checkpoint_ref != binding.team_checkpoint_ref
        ):
            raise AgentShellCompositionIntegrityError(
                "Graph checkpoint differs from current binding",
            )
        team_id = self._team_id(binding)
        team = self._team_summary(team_id, binding)
        reviews = self._review_summaries(run.run_id)
        delivery = self._delivery_summary(
            run_id=run.run_id,
            view=run_view,
        )
        return run_view, checkpoint, team, reviews, delivery

    def _team_summary(
        self,
        team_id: str,
        binding: HarnessGraphExecutionBindingV1,
    ) -> AgentShellTeamSummaryV1:
        try:
            historical = self.team_store.get_snapshot_at(
                binding.team_ref,
            )
            snapshot = self.team_store.get_snapshot(team_id)
            checkpoint = self.team_store.current_checkpoint(team_id)
        except TeamNotFoundError as exc:
            raise AgentShellCompositionIntegrityError(
                "Graph binding references missing Team authority",
            ) from exc
        if (
            historical.team.team_id != team_id
            or snapshot.team.to_ref() != binding.team_ref
            or snapshot.roster.to_ref() != binding.roster_ref
            or snapshot.graph.to_ref() != binding.task_graph_ref
            or snapshot.authority.to_ref() != binding.execution_authority_ref
            or snapshot.team.team_version != binding.expected_team_version
            or snapshot.graph.revision != binding.expected_task_graph_revision
            or snapshot.authority.authority_version != binding.expected_authority_version
            or checkpoint is None
            or checkpoint.to_ref() != binding.team_checkpoint_ref
        ):
            raise AgentShellCompositionIntegrityError(
                "Team projection differs from current Graph binding",
            )
        work = self.team_store.list_task_work(team_id)
        return AgentShellTeamSummaryV1(
            team_ref=snapshot.team.to_ref(),
            team_version=snapshot.team.team_version,
            lifecycle=snapshot.team.lifecycle,
            roster_ref=snapshot.roster.to_ref(),
            task_graph_ref=snapshot.graph.to_ref(),
            authority_ref=snapshot.authority.to_ref(),
            checkpoint_ref=checkpoint.to_ref(),
            members=tuple(self._member_summary(value) for value in snapshot.roster.members),
            tasks=tuple(
                AgentShellTeamTaskSummaryV1(
                    task_ref=value.task.to_ref(),
                    task_id=value.task.task_id,
                    task_kind=value.task.task_kind,
                    status=value.status,
                    assigned_member_id=value.task.assigned_member_id,
                    dependency_task_ids=value.task.dependency_task_ids,
                    capability_definition_ref=(value.task.capability_definition_ref),
                    attempt=value.attempt,
                    event_version=value.event_version,
                    claim_ref=(value.claim.to_ref() if value.claim is not None else None),
                    lease_ref=(value.lease.to_ref() if value.lease is not None else None),
                    latest_event_ref=(
                        value.latest_event.to_ref() if value.latest_event is not None else None
                    ),
                )
                for value in work
            ),
            artifact_head_refs=sorted_refs(
                value.to_ref()
                for value in self.team_store.current_artifact_heads(
                    team_id,
                )
            ),
            open_conflict_refs=sorted_refs(
                value.to_ref() for value in self.team_store.open_conflicts(team_id)
            ),
        )

    @staticmethod
    def _member_summary(value: object) -> AgentShellTeamMemberSummaryV1:
        from eval_factory.team.models import TeamMemberV1

        if not isinstance(value, TeamMemberV1):
            raise AgentShellCompositionIntegrityError(
                "Team roster contains an invalid member",
            )
        return AgentShellTeamMemberSummaryV1(
            member_ref=value.to_ref(),
            member_id=value.member_id,
            role=value.role,
            status=value.status,
            independent_session_ref=value.independent_session_ref,
            capability_definition_refs=value.capability_definition_refs,
            is_coordinator=value.is_coordinator,
        )

    def _review_summaries(
        self,
        run_id: str,
    ) -> tuple[AgentShellPlanReviewSummaryV1, ...]:
        run_ids = {run_id}
        run_ids.update(
            self.factory_store.get_item_run(binding).run_id
            for binding in self.factory_store.list_item_bindings(run_id)
        )
        values: dict[ObjectRef, PlanReviewViewV1] = {}
        for current_run_id in sorted(run_ids):
            page = self.plan_reviews.list_reviews(
                run_id=current_run_id,
                state=None,
                limit=500,
            )
            for value in page.items:
                values[value.request.to_ref()] = value
        if len(values) > 500:
            raise AgentShellCompositionIntegrityError(
                "Agent Shell PlanReview projection exceeds its limit",
            )
        return tuple(
            AgentShellPlanReviewSummaryV1(
                run_id=value.run_id,
                request_ref=value.request.to_ref(),
                result_ref=value.result.to_ref(),
                plan_ref=value.plan.to_ref(),
                plan_kind=value.request.plan_kind,
                plan_version=value.request.plan_version,
                state=value.result.state,
                decision_ref=(value.decision.to_ref() if value.decision is not None else None),
                revision_ref=(value.revision.to_ref() if value.revision is not None else None),
            )
            for _reference, value in sorted(
                values.items(),
                key=lambda item: _ref_key(item[0]),
            )
        )

    def _delivery_summary(
        self,
        *,
        run_id: str,
        view: FactoryDatasetRunViewV2,
    ) -> AgentShellDeliverySummaryV1 | None:
        if view.status is not FactoryRunStatusV2.COMPLETED:
            return None
        try:
            write = self.candidate_output.get(run_id)
        except CandidateOutputIntegrityError as exc:
            raise AgentShellCompositionIntegrityError(
                "completed Factory run has no valid candidate delivery",
            ) from exc
        if view.delivery_manifest_ref != write.manifest.to_ref():
            raise AgentShellCompositionIntegrityError(
                "candidate delivery differs from Factory projection",
            )
        return self._public_delivery(write)

    @staticmethod
    def _public_delivery(
        write: CandidateOutputWrite,
    ) -> AgentShellDeliverySummaryV1:
        return AgentShellDeliverySummaryV1(
            completion_ref=write.completion.to_ref(),
            manifest_ref=write.manifest.to_ref(),
            inventory_ref=write.inventory.to_ref(),
            item_count=write.manifest.item_count,
            file_count=len(write.inventory.files),
            total_bytes=sum(value.size_bytes for value in write.inventory.files),
            bundle_sha256=write.inventory.bundle_sha256,
        )

    @staticmethod
    def _factory_summary(
        *,
        binding: HarnessGraphExecutionBindingV1,
        view: FactoryDatasetRunViewV2,
    ) -> AgentShellFactorySummaryV1:
        return AgentShellFactorySummaryV1(
            run_ref=view.dataset_run_ref,
            run_version=binding.expected_factory_run_version,
            status=view.status,
            next_action=view.next_action,
            pending_review_refs=view.pending_review_refs,
            candidate_count=view.candidate_count,
            rejected_count=view.rejected_count,
            blocked_count=view.blocked_count,
            incomplete_count=view.incomplete_count,
            aggregate_result_ref=view.aggregate_result_ref,
            delivery_manifest_ref=view.delivery_manifest_ref,
        )

    @staticmethod
    def _graph_summary(
        *,
        binding: HarnessGraphExecutionBindingV1,
        checkpoint: HarnessGraphCheckpointV1 | None,
    ) -> AgentShellGraphSummaryV1:
        return AgentShellGraphSummaryV1(
            binding_ref=binding.to_ref(),
            checkpoint_ref=(checkpoint.to_ref() if checkpoint is not None else None),
            checkpoint_phase=(checkpoint.phase if checkpoint is not None else None),
            checkpoint_outcome=(checkpoint.outcome if checkpoint is not None else None),
            node=(checkpoint.node if checkpoint is not None else None),
            transition_number=(checkpoint.transition_number if checkpoint is not None else 0),
            reason_codes=(checkpoint.reason_codes if checkpoint is not None else ()),
        )

    def _interactions(
        self,
        *,
        session: HarnessSessionProjectionV1,
        interpretation: RequirementInterpretationV1 | None,
        reviews: tuple[AgentShellPlanReviewSummaryV1, ...],
        checkpoint: HarnessGraphCheckpointV1 | None,
        preflight: AgentShellCompositionPreflight | None,
    ) -> tuple[AgentShellInteractionCardV1, ...]:
        values: list[AgentShellInteractionCardV1] = []
        if (
            interpretation is not None
            and interpretation.outcome is RequirementInterpretationOutcomeV1.CLARIFICATION_REQUIRED
        ):
            values.append(
                AgentShellInteractionCardV1(
                    interaction_id=interpretation.object_id,
                    kind=AgentShellInteractionKindV1.CLARIFICATION,
                    state=AgentShellInteractionStateV1.PENDING,
                    subject_ref=interpretation.to_ref(),
                    questions=interpretation.clarification_questions,
                )
            )
        for review in reviews:
            if review.state in {
                PlanReviewStateV2.RESUMED,
                PlanReviewStateV2.SUPERSEDED,
            }:
                continue
            values.append(
                AgentShellInteractionCardV1(
                    interaction_id=review.request_ref.object_id,
                    kind=AgentShellInteractionKindV1.PLAN_REVIEW,
                    state=(
                        AgentShellInteractionStateV1.BLOCKED
                        if review.state
                        in {
                            PlanReviewStateV2.REJECTED,
                            PlanReviewStateV2.DEFERRED,
                        }
                        else AgentShellInteractionStateV1.PENDING
                    ),
                    subject_ref=review.request_ref,
                    reason_codes=(review.state.value,),
                )
            )
        if checkpoint is not None and checkpoint.outcome is GraphCheckpointOutcomeV1.VERIFICATION_REQUIRED:
            values.append(
                AgentShellInteractionCardV1(
                    interaction_id=checkpoint.object_id,
                    kind=AgentShellInteractionKindV1.VERIFICATION,
                    state=AgentShellInteractionStateV1.BLOCKED,
                    subject_ref=checkpoint.to_ref(),
                    reason_codes=checkpoint.reason_codes,
                )
            )
        if preflight is not None and not preflight.ready:
            source_ref = session.session.current_interpretation_ref
            if source_ref is not None and preflight.reason_codes:
                values.append(
                    AgentShellInteractionCardV1(
                        interaction_id=(
                            f"shell-composition.{session.session.session_ref.object_sha256[:32]}"
                        ),
                        kind=AgentShellInteractionKindV1.PERMISSION,
                        state=AgentShellInteractionStateV1.BLOCKED,
                        subject_ref=source_ref,
                        reason_codes=preflight.reason_codes,
                    )
                )
        return tuple(
            sorted(
                values,
                key=lambda value: value.interaction_id,
            )
        )

    def _workspaces(
        self,
        *,
        source: HarnessSourceAdmissionWrite | None,
        team: AgentShellTeamSummaryV1 | None,
        reviews: tuple[AgentShellPlanReviewSummaryV1, ...],
        delivery: AgentShellDeliverySummaryV1 | None,
        execution_phase: AgentShellExecutionPhaseV1,
    ) -> tuple[AgentShellWorkspaceDescriptorV1, ...]:
        by_kind: dict[
            AgentShellWorkspaceKindV1,
            tuple[ObjectRef, ...],
        ] = {value: () for value in AgentShellWorkspaceKindV1}
        by_kind[AgentShellWorkspaceKindV1.CONVERSATION] = ()
        by_kind[AgentShellWorkspaceKindV1.ACTIVITY] = ()
        if source is not None:
            by_kind[AgentShellWorkspaceKindV1.TRACE] = (source.admission.to_ref(),)
        if team is not None:
            by_kind[AgentShellWorkspaceKindV1.TEAM] = (team.team_ref,)
            task_groups = {
                AgentShellWorkspaceKindV1.TASK: {
                    "task-authoring",
                },
                AgentShellWorkspaceKindV1.ATTACHMENT: {
                    "attachment-reconstruction",
                },
                AgentShellWorkspaceKindV1.RUBRIC: {
                    "criteria-rubric",
                },
                AgentShellWorkspaceKindV1.GRADING: {
                    "grading-design",
                },
                AgentShellWorkspaceKindV1.QUALITY: {
                    "batch-quality",
                    "quality-review",
                },
            }
            for kind, task_kinds in task_groups.items():
                by_kind[kind] = sorted_refs(
                    value.task_ref for value in team.tasks if value.task_kind in task_kinds
                )
        by_kind[AgentShellWorkspaceKindV1.PLAN_REVIEW] = sorted_refs(value.request_ref for value in reviews)
        if delivery is not None:
            by_kind[AgentShellWorkspaceKindV1.DELIVERY] = (delivery.manifest_ref,)
        blocked = execution_phase in {
            AgentShellExecutionPhaseV1.BLOCKED,
            AgentShellExecutionPhaseV1.VERIFICATION_REQUIRED,
        }
        return tuple(
            AgentShellWorkspaceDescriptorV1(
                kind=kind,
                status=self._workspace_status(
                    kind=kind,
                    refs=by_kind[kind],
                    team=team,
                    delivery=delivery,
                    blocked=blocked,
                ),
                owner_refs=by_kind[kind],
                item_count=len(by_kind[kind]),
                reason_codes=(
                    ("EXECUTION_BLOCKED",)
                    if blocked
                    and kind
                    not in {
                        AgentShellWorkspaceKindV1.CONVERSATION,
                        AgentShellWorkspaceKindV1.ACTIVITY,
                    }
                    and not by_kind[kind]
                    else ()
                ),
            )
            for kind in AgentShellWorkspaceKindV1
        )

    @staticmethod
    def _workspace_status(
        *,
        kind: AgentShellWorkspaceKindV1,
        refs: tuple[ObjectRef, ...],
        team: AgentShellTeamSummaryV1 | None,
        delivery: AgentShellDeliverySummaryV1 | None,
        blocked: bool,
    ) -> AgentShellWorkspaceStatusV1:
        if kind in {
            AgentShellWorkspaceKindV1.CONVERSATION,
            AgentShellWorkspaceKindV1.ACTIVITY,
        }:
            return AgentShellWorkspaceStatusV1.AVAILABLE
        if kind is AgentShellWorkspaceKindV1.DELIVERY and delivery is not None:
            return AgentShellWorkspaceStatusV1.COMPLETE
        if refs:
            if team is not None:
                task_refs = set(refs)
                matching = tuple(value for value in team.tasks if value.task_ref in task_refs)
                if matching and all(value.status is TeamTaskStatusV1.COMPLETED for value in matching):
                    return AgentShellWorkspaceStatusV1.COMPLETE
                if matching and any(
                    value.status
                    in {
                        TeamTaskStatusV1.BLOCKED,
                        TeamTaskStatusV1.FAILED,
                        TeamTaskStatusV1.CANCELLED,
                    }
                    for value in matching
                ):
                    return AgentShellWorkspaceStatusV1.BLOCKED
            return AgentShellWorkspaceStatusV1.AVAILABLE
        if blocked:
            return AgentShellWorkspaceStatusV1.BLOCKED
        return AgentShellWorkspaceStatusV1.EMPTY

    def _projection(
        self,
        *,
        session: HarnessSessionProjectionV1,
        phase: AgentShellExecutionPhaseV1,
        interpretation: RequirementInterpretationV1 | None,
        source: HarnessSourceAdmissionWrite | None,
        factory: AgentShellFactorySummaryV1 | None,
        graph: AgentShellGraphSummaryV1 | None,
        team: AgentShellTeamSummaryV1 | None,
        reviews: tuple[AgentShellPlanReviewSummaryV1, ...],
        interactions: tuple[AgentShellInteractionCardV1, ...],
        workspaces: tuple[AgentShellWorkspaceDescriptorV1, ...],
        delivery: AgentShellDeliverySummaryV1 | None,
    ) -> AgentShellCompositionProjection:
        values = {
            "session_fingerprint": session.source_fingerprint,
            "source_ref": (_ref_value(source.admission.to_ref()) if source is not None else None),
            "factory": (factory.model_dump(mode="json") if factory is not None else None),
            "graph": (graph.model_dump(mode="json") if graph is not None else None),
            "team": (team.model_dump(mode="json") if team is not None else None),
            "reviews": [value.model_dump(mode="json") for value in reviews],
            "interactions": [value.model_dump(mode="json") for value in interactions],
            "workspaces": [value.model_dump(mode="json") for value in workspaces],
            "delivery": (delivery.model_dump(mode="json") if delivery is not None else None),
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                values,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return AgentShellCompositionProjection(
            execution_phase=phase,
            requirement_outcome=(interpretation.outcome if interpretation is not None else None),
            evidence_class=(interpretation.evidence_class if interpretation is not None else None),
            source_admission_ref=(source.admission.to_ref() if source is not None else None),
            factory=factory,
            graph=graph,
            team=team,
            plan_reviews=reviews,
            pending_interactions=interactions,
            workspaces=workspaces,
            delivery=delivery,
            source_fingerprint=fingerprint,
        )

    def _interpretation(
        self,
        session: HarnessSessionProjectionV1,
    ) -> RequirementInterpretationV1 | None:
        reference = session.session.current_interpretation_ref
        return self.sessions.get_interpretation(reference) if reference is not None else None

    def _source_or_none(
        self,
        session_id: str,
    ) -> HarnessSourceAdmissionWrite | None:
        try:
            return self.sources.get_for_session(session_id)
        except HarnessSourceAdmissionNotFoundError:
            return None

    def _preflight(
        self,
        *,
        session: HarnessSessionProjectionV1,
        interpretation: RequirementInterpretationV1 | None,
        source: HarnessSourceAdmissionWrite | None,
    ) -> AgentShellCompositionPreflight | None:
        if (
            interpretation is None
            or interpretation.outcome is not RequirementInterpretationOutcomeV1.READY
            or source is None
        ):
            return None
        return self.graph_starter.preflight(
            session.session.session_id,
        )

    @staticmethod
    def _phase_without_graph(
        *,
        interpretation: RequirementInterpretationV1 | None,
        source: HarnessSourceAdmissionWrite | None,
        preflight: AgentShellCompositionPreflight | None,
    ) -> AgentShellExecutionPhaseV1:
        if interpretation is None:
            return AgentShellExecutionPhaseV1.WAITING_REQUIREMENT
        if interpretation.outcome is not RequirementInterpretationOutcomeV1.READY:
            return (
                AgentShellExecutionPhaseV1.WAITING_REQUIREMENT
                if interpretation.outcome is RequirementInterpretationOutcomeV1.CLARIFICATION_REQUIRED
                else AgentShellExecutionPhaseV1.BLOCKED
            )
        if source is None:
            return AgentShellExecutionPhaseV1.WAITING_SOURCE
        if preflight is not None and not preflight.ready:
            return AgentShellExecutionPhaseV1.BLOCKED
        return AgentShellExecutionPhaseV1.READY

    @staticmethod
    def _phase_with_graph(
        *,
        view: FactoryDatasetRunViewV2,
        checkpoint: HarnessGraphCheckpointV1 | None,
        reviews: tuple[AgentShellPlanReviewSummaryV1, ...],
        delivery: AgentShellDeliverySummaryV1 | None,
    ) -> AgentShellExecutionPhaseV1:
        if checkpoint is not None and checkpoint.outcome is GraphCheckpointOutcomeV1.VERIFICATION_REQUIRED:
            return AgentShellExecutionPhaseV1.VERIFICATION_REQUIRED
        if delivery is not None:
            return AgentShellExecutionPhaseV1.COMPLETED
        if view.status in {
            FactoryRunStatusV2.BLOCKED,
            FactoryRunStatusV2.FAILED,
            FactoryRunStatusV2.CANCELLED,
        }:
            return AgentShellExecutionPhaseV1.BLOCKED
        if any(
            value.state
            not in {
                PlanReviewStateV2.RESUMED,
                PlanReviewStateV2.SUPERSEDED,
            }
            for value in reviews
        ):
            return AgentShellExecutionPhaseV1.WAITING_REVIEW
        if view.status is FactoryRunStatusV2.COMPLETED:
            raise AgentShellCompositionIntegrityError(
                "completed Factory run has no candidate delivery projection",
            )
        return AgentShellExecutionPhaseV1.RUNNING

    def _binding_or_none(
        self,
        session_ref: ObjectRef,
    ) -> HarnessGraphExecutionBindingV1 | None:
        try:
            return self.graph_journal.get_binding_for_session(
                session_ref,
            )
        except FactoryGraphJournalNotFoundError:
            return None
        except FactoryGraphJournalIntegrityError as exc:
            raise AgentShellCompositionIntegrityError(
                "Graph binding lookup failed integrity validation",
            ) from exc

    def _require_binding(
        self,
        session_ref: ObjectRef,
    ) -> HarnessGraphExecutionBindingV1:
        binding = self._binding_or_none(session_ref)
        if binding is None:
            raise AgentShellCompositionNotFoundError(
                "Agent Shell session has no Graph binding",
            )
        return binding

    def _team_id(
        self,
        binding: HarnessGraphExecutionBindingV1,
    ) -> str:
        return self.team_store.get_snapshot_at(
            binding.team_ref,
        ).team.team_id


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _ref_value(value: ObjectRef) -> dict[str, object]:
    return value.model_dump(mode="json")


__all__ = [
    "AgentShellCompositionProjection",
    "AgentShellCompositionService",
]
