from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from eval_factory.agent_system.graph import (
    FactoryGraphDriftError,
    FactoryGraphSignalV2,
    FactoryGraphState,
)
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalConflictError,
    FactoryGraphJournalFaultPoint,
    FactoryGraphJournalInjectedCrash,
    FactoryGraphJournalIntegrityError,
    FactoryGraphJournalNotFoundError,
    FactoryGraphJournalStore,
    StaticFactoryGraphJournalFaultInjector,
)
from eval_factory.agent_system.graph_recovery import (
    FactoryGraphRecoveryService,
)
from eval_factory.agent_system.graph_runtime import (
    FactoryGraphContinuationService,
    FactoryGraphSessionReconciler,
    FactoryGraphTransitionService,
)
from eval_factory.agent_system.plan_review import PlanReviewViewV1
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunStatusV2,
    FactoryRunV2,
    PlanKindV2,
    PlanReviewRequestV2,
    PlanReviewResultV2,
    PlanReviewStateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.harness.graph_models import (
    GraphCheckpointOutcomeV1,
    GraphCheckpointPhaseV1,
    HarnessGraphCheckpointV1,
    HarnessGraphExecutionBindingV1,
)
from eval_factory.harness.session_models import (
    HarnessSessionRefV1,
    SessionCheckpointPayloadV1,
    SessionEventV1,
)

HASH = "a" * 64
BINDING_FAULTS = (
    FactoryGraphJournalFaultPoint.AFTER_BINDING_RECORD,
    FactoryGraphJournalFaultPoint.AFTER_BINDING_HEAD,
    FactoryGraphJournalFaultPoint.AFTER_IDEMPOTENCY,
)
CHECKPOINT_FAULTS = (
    FactoryGraphJournalFaultPoint.AFTER_CHECKPOINT_RECORD,
    FactoryGraphJournalFaultPoint.AFTER_CHECKPOINT_HEAD,
    FactoryGraphJournalFaultPoint.AFTER_CHECKPOINT_OUTBOX,
    FactoryGraphJournalFaultPoint.AFTER_IDEMPOTENCY,
)


@dataclass(frozen=True, slots=True)
class _BindingAuthority:
    binding: HarnessGraphExecutionBindingV1

    def resolve_current(
        self,
        reference: ObjectRef,
    ) -> HarnessGraphExecutionBindingV1:
        assert reference == self.binding.to_ref()
        return self.binding

    def capture_current(
        self,
        reference: ObjectRef,
        *,
        audit: ContractAudit,
    ) -> HarnessGraphExecutionBindingV1:
        del audit
        assert reference == self.binding.to_ref()
        return self.binding


@dataclass(frozen=True, slots=True)
class _ContinuationAuthority:
    prior: HarnessGraphExecutionBindingV1
    successor: HarnessGraphExecutionBindingV1

    def resolve_current(
        self,
        reference: ObjectRef,
    ) -> HarnessGraphExecutionBindingV1:
        assert reference == self.prior.to_ref()
        return self.prior

    def capture_current(
        self,
        reference: ObjectRef,
        *,
        audit: ContractAudit,
    ) -> HarnessGraphExecutionBindingV1:
        del audit
        assert reference == self.prior.to_ref()
        return self.successor


@dataclass(frozen=True, slots=True)
class _DriftedAuthority:
    successor: HarnessGraphExecutionBindingV1

    def resolve_current(
        self,
        reference: ObjectRef,
    ) -> HarnessGraphExecutionBindingV1:
        del reference
        raise FactoryGraphDriftError("source advanced after PRE")

    def capture_current(
        self,
        reference: ObjectRef,
        *,
        audit: ContractAudit,
    ) -> HarnessGraphExecutionBindingV1:
        del reference, audit
        return self.successor


@dataclass(frozen=True, slots=True)
class _FactoryRuns:
    values: dict[ObjectRef, FactoryRunV2]

    def get_run_by_ref(self, reference: ObjectRef) -> FactoryRunV2:
        return self.values[reference]


@dataclass(frozen=True, slots=True)
class _ReviewSource:
    view: PlanReviewViewV1

    def show_by_result_ref(
        self,
        reference: ObjectRef,
    ) -> PlanReviewViewV1:
        assert reference == self.view.result.to_ref()
        return self.view


@dataclass
class _SessionSink:
    calls: int = 0

    def append_team_event(
        self,
        *,
        session_ref: ObjectRef,
        source_outbox_ref: ObjectRef,
        team_authority_version: int,
        payload: SessionCheckpointPayloadV1,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> SessionEventV1:
        del source_outbox_ref, idempotency_key
        self.calls += 1
        return SessionEventV1.create(
            event_id=f"session-event.graph-{self.calls}",
            session=HarnessSessionRefV1(
                session_ref=session_ref,
                incarnation_id="session-incarnation-stage4",
                session_version=self.calls,
                composition_ref=_ref("harness-composition"),
            ),
            sequence=self.calls,
            authority_version=team_authority_version,
            turn_id=None,
            step_id=None,
            command_ref=None,
            payload=payload,
            occurred_at=audit.created_at,
            audit=audit,
        )


@dataclass
class _FailingSessionSink:
    def append_team_event(
        self,
        *,
        session_ref: ObjectRef,
        source_outbox_ref: ObjectRef,
        team_authority_version: int,
        payload: SessionCheckpointPayloadV1,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> SessionEventV1:
        del (
            session_ref,
            source_outbox_ref,
            team_authority_version,
            payload,
            audit,
            idempotency_key,
        )
        raise RuntimeError("injected session projection crash")


@dataclass
class _IdempotentSessionSink:
    calls: int = 0
    events: dict[str, SessionEventV1] = field(default_factory=dict)

    def append_team_event(
        self,
        *,
        session_ref: ObjectRef,
        source_outbox_ref: ObjectRef,
        team_authority_version: int,
        payload: SessionCheckpointPayloadV1,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> SessionEventV1:
        del source_outbox_ref
        self.calls += 1
        existing = self.events.get(idempotency_key)
        if existing is not None:
            return existing
        sequence = len(self.events) + 1
        event = SessionEventV1.create(
            event_id=f"session-event.graph-idempotent-{sequence}",
            session=HarnessSessionRefV1(
                session_ref=session_ref,
                incarnation_id="session-incarnation-stage4",
                session_version=sequence,
                composition_ref=_ref("harness-composition"),
            ),
            sequence=sequence,
            authority_version=team_authority_version,
            turn_id=None,
            step_id=None,
            command_ref=None,
            payload=payload,
            occurred_at=audit.created_at,
            audit=audit,
        )
        self.events[idempotency_key] = event
        return event


def _ref(
    object_type: str,
    suffix: str = "example",
    *,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 19, tzinfo=UTC),
        created_by="graph-journal-test",
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage4",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _binding(
    *,
    factory_run_suffix: str = "run-1",
    team_suffix: str = "team-1",
    factory_run_ref: ObjectRef | None = None,
    expected_factory_run_version: int = 2,
) -> HarnessGraphExecutionBindingV1:
    return HarnessGraphExecutionBindingV1.create(
        binding_id="graph-binding.stage4",
        session_ref=_ref("harness-session"),
        requirement_ref=_ref(
            "evaluation-requirement-spec",
            version="v2",
        ),
        requirement_policy_ref=_ref("harness-requirement-policy"),
        factory_run_ref=factory_run_ref
        or _ref(
            "factory-run",
            factory_run_suffix,
            version="v2",
        ),
        factory_request_ref=_ref(
            "factory-dataset-run-request",
            version="v2",
        ),
        factory_policy_ref=_ref("factory-run-policy", version="v2"),
        expected_factory_run_version=(expected_factory_run_version),
        pack_manifest_ref=_ref("eval-pack-manifest"),
        composition_ref=_ref("harness-composition"),
        blueprint_ref=_ref("evaluation-blueprint"),
        team_ref=_ref("agent-team", team_suffix),
        roster_ref=_ref("team-roster"),
        task_graph_ref=_ref("team-task-graph"),
        execution_authority_ref=_ref("execution-authority"),
        team_checkpoint_ref=_ref("team-checkpoint"),
        expected_team_version=1,
        expected_task_graph_revision=1,
        expected_authority_version=1,
        blackboard_head_refs=(_ref("artifact-head"),),
        thread_id="stage4-thread",
        max_transitions=64,
        audit=_audit(),
    )


def _checkpoint(
    binding: HarnessGraphExecutionBindingV1,
    *,
    phase: GraphCheckpointPhaseV1,
    transition: int,
    predecessor: HarnessGraphCheckpointV1 | None,
    outcome: GraphCheckpointOutcomeV1,
    reason_codes: tuple[str, ...] = (),
) -> HarnessGraphCheckpointV1:
    return HarnessGraphCheckpointV1.create(
        checkpoint_id=(f"graph-checkpoint.stage4.{transition}.{phase.value.casefold()}"),
        binding_ref=binding.to_ref(),
        phase=phase,
        node="dispatch_ready_work",
        transition_number=transition,
        predecessor_checkpoint_ref=(predecessor.to_ref() if predecessor else None),
        factory_run_ref=binding.factory_run_ref,
        team_ref=binding.team_ref,
        team_checkpoint_ref=binding.team_checkpoint_ref,
        blackboard_head_refs=binding.blackboard_head_refs,
        planner_assessment_ref=None,
        outcome=outcome,
        reason_codes=reason_codes,
        audit=_audit(),
    )


def test_journal_requires_separate_authority_path(tmp_path: Path) -> None:
    path = tmp_path / "journal.sqlite3"
    with pytest.raises(ValueError, match="separate"):
        FactoryGraphJournalStore(
            path,
            authority_paths=(path,),
        )


@pytest.mark.parametrize("fault_point", BINDING_FAULTS)
def test_binding_fault_rolls_back_every_write_family(
    tmp_path: Path,
    fault_point: FactoryGraphJournalFaultPoint,
) -> None:
    path = tmp_path / f"binding-{fault_point.value}.sqlite3"
    faulted = FactoryGraphJournalStore(
        path,
        fault_injector=StaticFactoryGraphJournalFaultInjector(
            frozenset({fault_point}),
        ),
    )
    binding = _binding()
    with pytest.raises(
        FactoryGraphJournalInjectedCrash,
        match=fault_point.value,
    ):
        faulted.commit_binding(
            binding,
            idempotency_key="binding-fault",
        )

    restarted = FactoryGraphJournalStore(path)
    with pytest.raises(FactoryGraphJournalNotFoundError):
        restarted.get_binding(binding.binding_id)
    assert restarted.rebuild_current_heads().binding_count == 0


@pytest.mark.parametrize("fault_point", CHECKPOINT_FAULTS)
def test_checkpoint_fault_rolls_back_record_head_outbox_and_key(
    tmp_path: Path,
    fault_point: FactoryGraphJournalFaultPoint,
) -> None:
    path = tmp_path / f"checkpoint-{fault_point.value}.sqlite3"
    store = FactoryGraphJournalStore(path)
    binding = _binding()
    store.commit_binding(binding, idempotency_key="binding")
    checkpoint = _checkpoint(
        binding,
        phase=GraphCheckpointPhaseV1.PRE_TRANSITION,
        transition=1,
        predecessor=None,
        outcome=GraphCheckpointOutcomeV1.READY,
    )
    faulted = FactoryGraphJournalStore(
        path,
        fault_injector=StaticFactoryGraphJournalFaultInjector(
            frozenset({fault_point}),
        ),
    )
    with pytest.raises(
        FactoryGraphJournalInjectedCrash,
        match=fault_point.value,
    ):
        faulted.commit_checkpoint(
            checkpoint,
            idempotency_key="checkpoint-fault",
        )

    restarted = FactoryGraphJournalStore(path)
    assert restarted.current_checkpoint(binding.binding_id) is None
    assert restarted.pending_deliveries(binding.binding_id) == ()
    assert restarted.list_checkpoints(binding.binding_id) == ()


def test_binding_and_checkpoint_replay_survive_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "journal.sqlite3"
    store = FactoryGraphJournalStore(path)
    binding = _binding()
    assert (
        store.commit_binding(
            binding,
            idempotency_key="bind-1",
        )
        == binding
    )
    assert (
        store.commit_binding(
            binding,
            idempotency_key="bind-1",
        )
        == binding
    )
    pre = _checkpoint(
        binding,
        phase=GraphCheckpointPhaseV1.PRE_TRANSITION,
        transition=1,
        predecessor=None,
        outcome=GraphCheckpointOutcomeV1.READY,
    )
    post = _checkpoint(
        binding,
        phase=GraphCheckpointPhaseV1.POST_TRANSITION,
        transition=1,
        predecessor=pre,
        outcome=GraphCheckpointOutcomeV1.COMMITTED,
    )
    store.commit_checkpoint(pre, idempotency_key="checkpoint-1-pre")
    store.commit_checkpoint(post, idempotency_key="checkpoint-1-post")

    restarted = FactoryGraphJournalStore(path)
    assert restarted.get_binding(binding.binding_id) == binding
    assert restarted.current_checkpoint(binding.binding_id) == post
    assert restarted.list_checkpoints(binding.binding_id) == (pre, post)
    assert restarted.rebuild_current_heads().binding_count == 1
    assert restarted.rebuild_current_heads().checkpoint_count == 1


def test_transition_service_commits_exact_pre_and_post(
    tmp_path: Path,
) -> None:
    store = FactoryGraphJournalStore(tmp_path / "journal.sqlite3")
    binding = _binding()
    store.commit_binding(binding, idempotency_key="bind")
    service = FactoryGraphTransitionService(
        journal=store,
        authority=_BindingAuthority(binding),
        audit=_audit(),
    )
    state: FactoryGraphState = {
        "factory_run_id": "run-1",
        "expected_run_version": 2,
        "transition_count": 1,
        "signal": FactoryGraphSignalV2.CONTINUE,
        "graph_binding_ref": binding.to_ref(),
        "run_ref": binding.factory_run_ref,
        "run_status": FactoryRunStatusV2.RUNNING,
    }

    pre_ref = service.begin_transition(
        node="dispatch_ready_work",
        state=state,
    )
    update = service.complete_transition(
        node="dispatch_ready_work",
        state=state,
        pre_checkpoint_ref=pre_ref,
    )

    checkpoints = store.list_checkpoints(binding.binding_id)
    assert tuple(value.phase for value in checkpoints) == (
        GraphCheckpointPhaseV1.PRE_TRANSITION,
        GraphCheckpointPhaseV1.POST_TRANSITION,
    )
    assert checkpoints[1].predecessor_checkpoint_ref == pre_ref
    assert checkpoints[1].outcome is GraphCheckpointOutcomeV1.COMMITTED
    assert update["graph_binding_ref"] == binding.to_ref()
    assert update["graph_checkpoint_ref"] == checkpoints[1].to_ref()
    assert len(store.pending_deliveries(binding.binding_id)) == 2

    sink = _SessionSink()
    reconciler = FactoryGraphSessionReconciler(
        journal=store,
        session_sink=sink,
    )
    projected = reconciler.reconcile(
        binding.binding_id,
        audit=_audit(),
    )
    assert len(projected) == 2
    assert sink.calls == 2
    assert not store.pending_deliveries(binding.binding_id)
    assert reconciler.reconcile(binding.binding_id, audit=_audit()) == ()


def test_before_owner_crash_reuses_unfinished_matching_pre(
    tmp_path: Path,
) -> None:
    path = tmp_path / "journal.sqlite3"
    store = FactoryGraphJournalStore(path)
    binding = _binding()
    store.commit_binding(binding, idempotency_key="bind")
    service = FactoryGraphTransitionService(
        journal=store,
        authority=_BindingAuthority(binding),
        audit=_audit(),
    )
    state: FactoryGraphState = {
        "factory_run_id": "run-1",
        "expected_run_version": 2,
        "transition_count": 3,
        "signal": FactoryGraphSignalV2.CONTINUE,
        "graph_binding_ref": binding.to_ref(),
    }
    first = service.begin_transition(
        node="dispatch_ready_work",
        state=state,
    )
    restarted = FactoryGraphTransitionService(
        journal=FactoryGraphJournalStore(path),
        authority=_BindingAuthority(binding),
        audit=_audit(),
    )

    assert (
        restarted.begin_transition(
            node="dispatch_ready_work",
            state=state,
        )
        == first
    )
    with pytest.raises(
        FactoryGraphDriftError,
        match="another command",
    ):
        restarted.begin_transition(
            node="planner_assessment",
            state=state,
        )
    update = restarted.complete_transition(
        node="dispatch_ready_work",
        state=state,
        pre_checkpoint_ref=first,
    )
    checkpoints = store.list_checkpoints(binding.binding_id)
    assert len(checkpoints) == 2
    assert checkpoints[0].phase is GraphCheckpointPhaseV1.PRE_TRANSITION
    assert checkpoints[1].phase is GraphCheckpointPhaseV1.POST_TRANSITION
    assert update["graph_checkpoint_ref"] == checkpoints[1].to_ref()


@pytest.mark.parametrize(
    "fault_point",
    (
        None,
        FactoryGraphJournalFaultPoint.AFTER_BINDING_RECORD,
        FactoryGraphJournalFaultPoint.AFTER_BINDING_HEAD,
        FactoryGraphJournalFaultPoint.AFTER_IDEMPOTENCY,
        FactoryGraphJournalFaultPoint.AFTER_CHECKPOINT_RECORD,
        FactoryGraphJournalFaultPoint.AFTER_CHECKPOINT_HEAD,
        FactoryGraphJournalFaultPoint.AFTER_CHECKPOINT_OUTBOX,
    ),
)
def test_recovery_marks_changed_source_as_verification_required(
    tmp_path: Path,
    fault_point: FactoryGraphJournalFaultPoint | None,
) -> None:
    path = tmp_path / "recovery.sqlite3"
    store = FactoryGraphJournalStore(path)
    prior = _binding()
    successor = _binding(factory_run_suffix="run-after-owner")
    store.commit_binding(prior, idempotency_key="binding-prior")
    pre = _checkpoint(
        prior,
        phase=GraphCheckpointPhaseV1.PRE_TRANSITION,
        transition=1,
        predecessor=None,
        outcome=GraphCheckpointOutcomeV1.READY,
    )
    store.commit_checkpoint(pre, idempotency_key="pre")
    authority = _DriftedAuthority(successor)
    recovery_journal = (
        store
        if fault_point is None
        else FactoryGraphJournalStore(
            path,
            fault_injector=StaticFactoryGraphJournalFaultInjector(
                frozenset({fault_point}),
            ),
        )
    )
    recovery = FactoryGraphRecoveryService(
        journal=recovery_journal,
        authority=authority,
        audit=_audit(),
    )

    if fault_point is not None:
        with pytest.raises(
            FactoryGraphJournalInjectedCrash,
            match=fault_point.value,
        ):
            recovery.reconcile_pre(
                graph_binding_ref=prior.to_ref(),
                expected_pre_checkpoint_ref=pre.to_ref(),
                idempotency_key="recover",
            )
        assert store.current_checkpoint(prior.binding_id) == pre
        recovery = FactoryGraphRecoveryService(
            journal=store,
            authority=authority,
            audit=_audit(),
        )

    reconciled = recovery.reconcile_pre(
        graph_binding_ref=prior.to_ref(),
        expected_pre_checkpoint_ref=pre.to_ref(),
        idempotency_key="recover",
    )

    assert reconciled.outcome is GraphCheckpointOutcomeV1.VERIFICATION_REQUIRED
    assert reconciled.binding_ref == successor.to_ref()
    assert reconciled.reason_codes == ("OWNER_COMMIT_AFTER_PRE_REQUIRES_VERIFICATION",)
    assert store.get_binding(prior.binding_id) == successor
    assert (
        recovery.reconcile_pre(
            graph_binding_ref=prior.to_ref(),
            expected_pre_checkpoint_ref=pre.to_ref(),
            idempotency_key="recover",
        )
        == reconciled
    )


@pytest.mark.parametrize(
    "fault_point",
    (
        None,
        FactoryGraphJournalFaultPoint.AFTER_BINDING_RECORD,
        FactoryGraphJournalFaultPoint.AFTER_BINDING_HEAD,
        FactoryGraphJournalFaultPoint.AFTER_IDEMPOTENCY,
        FactoryGraphJournalFaultPoint.AFTER_CHECKPOINT_RECORD,
        FactoryGraphJournalFaultPoint.AFTER_CHECKPOINT_HEAD,
        FactoryGraphJournalFaultPoint.AFTER_CHECKPOINT_OUTBOX,
    ),
)
def test_continuation_reconciles_only_exact_resumed_review_successor(
    tmp_path: Path,
    fault_point: FactoryGraphJournalFaultPoint | None,
) -> None:
    policy_ref = _ref("factory-run-policy", version="v2")
    requirement_ref = _ref(
        "evaluation-requirement-spec",
        version="v2",
    )
    plan_ref = _ref("dataset-build-plan", version="v2")
    compiled_plan_ref = _ref(
        "compiled-dataset-build-plan",
        version="v2",
    )
    source_run = FactoryRunV2.create(
        run_id="factory-run.stage4-review",
        run_version=1,
        status=FactoryRunStatusV2.PLANNING,
        policy_ref=policy_ref,
        requirement_spec_ref=requirement_ref,
        current_plan_ref=plan_ref,
        compiled_plan_ref=compiled_plan_ref,
        active_task_refs=(),
        result_refs=(),
        pending_review_ref=None,
        planner_assessment_ref=None,
        completion_ref=None,
        delivery_manifest_ref=None,
        transition_count=1,
        model_requests_used=0,
        model_tokens_used=0,
        cost_micro_usd_used=0,
        audit=_audit(),
    )
    request = PlanReviewRequestV2.create(
        review_request_id="review.stage4",
        run_ref=source_run.to_ref(),
        plan_ref=plan_ref,
        plan_kind=PlanKindV2.GLOBAL_BUILD,
        plan_version=1,
        presentation_ref=_ref(
            "plan-review-presentation",
            version="v2",
        ),
        requested_by="stage4-reviewer",
        audit=_audit(),
    )
    waiting_run = FactoryRunV2.create(
        run_id=source_run.run_id,
        run_version=2,
        status=FactoryRunStatusV2.WAITING_REVIEW,
        policy_ref=policy_ref,
        requirement_spec_ref=requirement_ref,
        current_plan_ref=plan_ref,
        compiled_plan_ref=compiled_plan_ref,
        active_task_refs=(),
        result_refs=(),
        pending_review_ref=request.to_ref(),
        planner_assessment_ref=None,
        completion_ref=None,
        delivery_manifest_ref=None,
        transition_count=2,
        model_requests_used=0,
        model_tokens_used=0,
        cost_micro_usd_used=0,
        audit=_audit(),
    )
    resumed_run = FactoryRunV2.create(
        run_id=source_run.run_id,
        run_version=3,
        status=FactoryRunStatusV2.RUNNING,
        policy_ref=policy_ref,
        requirement_spec_ref=requirement_ref,
        current_plan_ref=plan_ref,
        compiled_plan_ref=compiled_plan_ref,
        active_task_refs=(),
        result_refs=(),
        pending_review_ref=None,
        planner_assessment_ref=None,
        completion_ref=None,
        delivery_manifest_ref=None,
        transition_count=3,
        model_requests_used=0,
        model_tokens_used=0,
        cost_micro_usd_used=0,
        audit=_audit(),
    )
    result = PlanReviewResultV2.create(
        result_id="review-result.stage4.resumed",
        review_request_ref=request.to_ref(),
        plan_ref=plan_ref,
        plan_version=1,
        state=PlanReviewStateV2.RESUMED,
        decision_ref=_ref("plan-decision", version="v2"),
        successor_plan_ref=None,
        resume_token_ref=_ref("plan-resume-token", version="v2"),
        audit=_audit(),
    )
    view = cast(
        PlanReviewViewV1,
        SimpleNamespace(
            run_id=source_run.run_id,
            current_run_ref=resumed_run.to_ref(),
            request=request,
            result=result,
        ),
    )
    prior = _binding(
        factory_run_ref=waiting_run.to_ref(),
        expected_factory_run_version=2,
    )
    successor = HarnessGraphExecutionBindingV1.create(
        binding_id=prior.binding_id,
        session_ref=prior.session_ref,
        requirement_ref=prior.requirement_ref,
        requirement_policy_ref=prior.requirement_policy_ref,
        factory_run_ref=resumed_run.to_ref(),
        factory_request_ref=prior.factory_request_ref,
        factory_policy_ref=prior.factory_policy_ref,
        expected_factory_run_version=3,
        pack_manifest_ref=prior.pack_manifest_ref,
        composition_ref=prior.composition_ref,
        blueprint_ref=prior.blueprint_ref,
        team_ref=prior.team_ref,
        roster_ref=prior.roster_ref,
        task_graph_ref=prior.task_graph_ref,
        execution_authority_ref=prior.execution_authority_ref,
        team_checkpoint_ref=prior.team_checkpoint_ref,
        expected_team_version=prior.expected_team_version,
        expected_task_graph_revision=(prior.expected_task_graph_revision),
        expected_authority_version=(prior.expected_authority_version),
        blackboard_head_refs=prior.blackboard_head_refs,
        thread_id=prior.thread_id,
        max_transitions=prior.max_transitions,
        audit=_audit(),
    )
    journal_path = tmp_path / "continuation.sqlite3"
    journal = FactoryGraphJournalStore(journal_path)
    journal.commit_binding(prior, idempotency_key="binding-prior")
    pre = _checkpoint(
        prior,
        phase=GraphCheckpointPhaseV1.PRE_TRANSITION,
        transition=1,
        predecessor=None,
        outcome=GraphCheckpointOutcomeV1.READY,
    )
    post = HarnessGraphCheckpointV1.create(
        checkpoint_id="checkpoint.waiting-review",
        binding_ref=prior.to_ref(),
        phase=GraphCheckpointPhaseV1.POST_TRANSITION,
        node=pre.node,
        transition_number=pre.transition_number,
        predecessor_checkpoint_ref=pre.to_ref(),
        factory_run_ref=prior.factory_run_ref,
        team_ref=prior.team_ref,
        team_checkpoint_ref=prior.team_checkpoint_ref,
        blackboard_head_refs=prior.blackboard_head_refs,
        planner_assessment_ref=None,
        outcome=GraphCheckpointOutcomeV1.WAITING_REVIEW,
        reason_codes=(),
        audit=_audit(),
    )
    journal.commit_checkpoint(pre, idempotency_key="pre")
    journal.commit_checkpoint(post, idempotency_key="post")
    authority = _ContinuationAuthority(prior, successor)
    factory_source = _FactoryRuns(
        {
            source_run.to_ref(): source_run,
            waiting_run.to_ref(): waiting_run,
            resumed_run.to_ref(): resumed_run,
        }
    )
    review_source = _ReviewSource(view)
    service_journal = (
        journal
        if fault_point is None
        else FactoryGraphJournalStore(
            journal_path,
            fault_injector=StaticFactoryGraphJournalFaultInjector(
                frozenset({fault_point}),
            ),
        )
    )
    service = FactoryGraphContinuationService(
        journal=service_journal,
        authority=authority,
        factory_source=factory_source,
        review_source=review_source,
        audit=_audit(),
    )

    if fault_point is not None:
        with pytest.raises(
            FactoryGraphJournalInjectedCrash,
            match=fault_point.value,
        ):
            service.resume(
                graph_binding_ref=prior.to_ref(),
                expected_checkpoint_ref=post.to_ref(),
                review_result_ref=result.to_ref(),
                idempotency_key="resume-stage4",
            )
        assert journal.current_checkpoint(prior.binding_id) == post
        service = FactoryGraphContinuationService(
            journal=journal,
            authority=authority,
            factory_source=factory_source,
            review_source=review_source,
            audit=_audit(),
        )

    reconciled = service.resume(
        graph_binding_ref=prior.to_ref(),
        expected_checkpoint_ref=post.to_ref(),
        review_result_ref=result.to_ref(),
        idempotency_key="resume-stage4",
    )

    assert reconciled.phase is GraphCheckpointPhaseV1.RECONCILED
    assert reconciled.binding_ref == successor.to_ref()
    assert reconciled.factory_run_ref == resumed_run.to_ref()
    assert journal.get_binding(prior.binding_id) == successor
    assert (
        service.resume(
            graph_binding_ref=prior.to_ref(),
            expected_checkpoint_ref=post.to_ref(),
            review_result_ref=result.to_ref(),
            idempotency_key="resume-stage4",
        )
        == reconciled
    )


def test_journal_rejects_changed_idempotency_and_checkpoint_gap(
    tmp_path: Path,
) -> None:
    store = FactoryGraphJournalStore(tmp_path / "journal.sqlite3")
    binding = _binding()
    store.commit_binding(binding, idempotency_key="bind")
    changed = _binding(factory_run_suffix="run-2")
    with pytest.raises(
        FactoryGraphJournalConflictError,
        match="idempotency",
    ):
        store.commit_binding(changed, idempotency_key="bind")

    invalid_post = _checkpoint(
        binding,
        phase=GraphCheckpointPhaseV1.POST_TRANSITION,
        transition=1,
        predecessor=_checkpoint(
            binding,
            phase=GraphCheckpointPhaseV1.PRE_TRANSITION,
            transition=1,
            predecessor=None,
            outcome=GraphCheckpointOutcomeV1.READY,
        ),
        outcome=GraphCheckpointOutcomeV1.COMMITTED,
    )
    with pytest.raises(
        FactoryGraphJournalConflictError,
        match="first",
    ):
        store.commit_checkpoint(
            invalid_post,
            idempotency_key="invalid-post",
        )


def test_binding_successor_preserves_pinned_authority(
    tmp_path: Path,
) -> None:
    store = FactoryGraphJournalStore(tmp_path / "journal.sqlite3")
    binding = _binding()
    store.commit_binding(binding, idempotency_key="bind-1")
    successor = _binding(
        factory_run_suffix="run-2",
        team_suffix="team-2",
    )
    assert (
        store.commit_binding(
            successor,
            idempotency_key="bind-2",
        )
        == successor
    )

    changed = HarnessGraphExecutionBindingV1.create(
        binding_id=binding.binding_id,
        session_ref=binding.session_ref,
        requirement_ref=binding.requirement_ref,
        requirement_policy_ref=binding.requirement_policy_ref,
        factory_run_ref=successor.factory_run_ref,
        factory_request_ref=binding.factory_request_ref,
        factory_policy_ref=binding.factory_policy_ref,
        expected_factory_run_version=(successor.expected_factory_run_version),
        pack_manifest_ref=binding.pack_manifest_ref,
        composition_ref=_ref("harness-composition", "changed"),
        blueprint_ref=binding.blueprint_ref,
        team_ref=successor.team_ref,
        roster_ref=binding.roster_ref,
        task_graph_ref=binding.task_graph_ref,
        execution_authority_ref=binding.execution_authority_ref,
        team_checkpoint_ref=binding.team_checkpoint_ref,
        expected_team_version=successor.expected_team_version,
        expected_task_graph_revision=(successor.expected_task_graph_revision),
        expected_authority_version=(successor.expected_authority_version),
        blackboard_head_refs=binding.blackboard_head_refs,
        thread_id=binding.thread_id,
        max_transitions=binding.max_transitions,
        audit=_audit(),
    )
    with pytest.raises(
        FactoryGraphJournalConflictError,
        match="pinned",
    ):
        store.commit_binding(changed, idempotency_key="bind-3")


def test_outbox_delivery_is_idempotent_and_session_bound(
    tmp_path: Path,
) -> None:
    store = FactoryGraphJournalStore(tmp_path / "journal.sqlite3")
    binding = _binding()
    store.commit_binding(binding, idempotency_key="bind")
    pre = _checkpoint(
        binding,
        phase=GraphCheckpointPhaseV1.PRE_TRANSITION,
        transition=1,
        predecessor=None,
        outcome=GraphCheckpointOutcomeV1.READY,
    )
    store.commit_checkpoint(pre, idempotency_key="pre")
    pending = store.pending_deliveries(binding.binding_id)
    assert len(pending) == 1
    assert pending[0].checkpoint == pre
    assert pending[0].session_ref == binding.session_ref

    store.mark_delivered(
        checkpoint_ref=pre.to_ref(),
        session_ref=binding.session_ref,
    )
    store.mark_delivered(
        checkpoint_ref=pre.to_ref(),
        session_ref=binding.session_ref,
    )
    assert store.pending_deliveries(binding.binding_id) == ()

    with pytest.raises(
        FactoryGraphJournalConflictError,
        match="another session",
    ):
        store.mark_delivered(
            checkpoint_ref=pre.to_ref(),
            session_ref=_ref("harness-session", "other"),
        )


def test_post_checkpoint_survives_session_projection_crash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "journal.sqlite3"
    store = FactoryGraphJournalStore(path)
    binding = _binding()
    store.commit_binding(binding, idempotency_key="bind")
    pre = _checkpoint(
        binding,
        phase=GraphCheckpointPhaseV1.PRE_TRANSITION,
        transition=1,
        predecessor=None,
        outcome=GraphCheckpointOutcomeV1.READY,
    )
    post = _checkpoint(
        binding,
        phase=GraphCheckpointPhaseV1.POST_TRANSITION,
        transition=1,
        predecessor=pre,
        outcome=GraphCheckpointOutcomeV1.COMMITTED,
    )
    store.commit_checkpoint(pre, idempotency_key="pre")
    store.commit_checkpoint(post, idempotency_key="post")
    store.mark_delivered(
        checkpoint_ref=pre.to_ref(),
        session_ref=binding.session_ref,
    )

    crashing = FactoryGraphSessionReconciler(
        journal=store,
        session_sink=_FailingSessionSink(),
    )
    with pytest.raises(
        RuntimeError,
        match="session projection crash",
    ):
        crashing.reconcile(binding.binding_id, audit=_audit())

    restarted = FactoryGraphJournalStore(path)
    pending = restarted.pending_deliveries(binding.binding_id)
    assert tuple(value.checkpoint for value in pending) == (post,)
    sink = _SessionSink()
    projected = FactoryGraphSessionReconciler(
        journal=restarted,
        session_sink=sink,
    ).reconcile(binding.binding_id, audit=_audit())
    assert len(projected) == 1
    assert sink.calls == 1
    assert restarted.pending_deliveries(binding.binding_id) == ()


def test_session_event_replays_after_delivery_marker_crash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "journal.sqlite3"
    store = FactoryGraphJournalStore(path)
    binding = _binding()
    store.commit_binding(binding, idempotency_key="bind")
    pre = _checkpoint(
        binding,
        phase=GraphCheckpointPhaseV1.PRE_TRANSITION,
        transition=1,
        predecessor=None,
        outcome=GraphCheckpointOutcomeV1.READY,
    )
    post = _checkpoint(
        binding,
        phase=GraphCheckpointPhaseV1.POST_TRANSITION,
        transition=1,
        predecessor=pre,
        outcome=GraphCheckpointOutcomeV1.COMMITTED,
    )
    store.commit_checkpoint(pre, idempotency_key="pre")
    store.commit_checkpoint(post, idempotency_key="post")
    store.mark_delivered(
        checkpoint_ref=pre.to_ref(),
        session_ref=binding.session_ref,
    )
    sink = _IdempotentSessionSink()
    faulted = FactoryGraphJournalStore(
        path,
        fault_injector=StaticFactoryGraphJournalFaultInjector(
            frozenset({FactoryGraphJournalFaultPoint.AFTER_DELIVERY}),
        ),
    )
    crashing = FactoryGraphSessionReconciler(
        journal=faulted,
        session_sink=sink,
    )

    with pytest.raises(
        FactoryGraphJournalInjectedCrash,
        match=FactoryGraphJournalFaultPoint.AFTER_DELIVERY.value,
    ):
        crashing.reconcile(binding.binding_id, audit=_audit())

    restarted = FactoryGraphJournalStore(path)
    assert len(restarted.pending_deliveries(binding.binding_id)) == 1
    projected = FactoryGraphSessionReconciler(
        journal=restarted,
        session_sink=sink,
    ).reconcile(binding.binding_id, audit=_audit())
    assert len(projected) == 1
    assert sink.calls == 2
    assert len(sink.events) == 1
    assert restarted.pending_deliveries(binding.binding_id) == ()


def test_head_drift_requires_explicit_repair(tmp_path: Path) -> None:
    path = tmp_path / "journal.sqlite3"
    store = FactoryGraphJournalStore(path)
    binding = _binding()
    store.commit_binding(binding, idempotency_key="bind")
    pre = _checkpoint(
        binding,
        phase=GraphCheckpointPhaseV1.PRE_TRANSITION,
        transition=1,
        predecessor=None,
        outcome=GraphCheckpointOutcomeV1.READY,
    )
    store.commit_checkpoint(pre, idempotency_key="pre")

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "DELETE FROM graph_checkpoint_current_heads",
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        FactoryGraphJournalIntegrityError,
        match="heads",
    ):
        store.rebuild_current_heads()
    repaired = store.rebuild_current_heads(repair=True)
    assert repaired.checkpoint_count == 1
    assert store.current_checkpoint(binding.binding_id) == pre


def test_continuation_rejects_unrelated_or_nonadvancing_successor() -> None:
    prior = _binding()
    changed_team = prior.model_copy(
        update={
            "team_ref": _ref("agent-team", "changed"),
        },
    )
    changed_view = cast(
        PlanReviewViewV1,
        SimpleNamespace(
            current_run_ref=changed_team.factory_run_ref,
        ),
    )
    with pytest.raises(
        FactoryGraphDriftError,
        match="unrelated",
    ):
        FactoryGraphContinuationService._validate_review_successor(
            prior=prior,
            successor=changed_team,
            review=changed_view,
        )

    same_view = cast(
        PlanReviewViewV1,
        SimpleNamespace(current_run_ref=prior.factory_run_ref),
    )
    with pytest.raises(
        FactoryGraphDriftError,
        match="did not advance",
    ):
        FactoryGraphContinuationService._validate_review_successor(
            prior=prior,
            successor=prior,
            review=same_view,
        )
