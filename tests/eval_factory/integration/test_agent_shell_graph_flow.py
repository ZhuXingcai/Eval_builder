from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.eval_factory.integration.test_harness_agent_loop import (
    _audit,
)
from tests.eval_factory.integration.test_harness_agent_loop import (
    _components as _harness_components,
)
from tests.eval_factory.unit.test_harness_source_admission import (
    _admit,
    _ref,
)
from tests.integration.test_evalfactory_agent_run_cli import (
    RAW_ROOT,
    _attachment_config,
    _config,
    _core_config,
    _fixture,
    _policy,
    _r4_config,
    _real_trace_root,
    _release_config,
    _request,
    _specialist_config,
)

from eval_factory.agent_system.candidate_output import (
    CandidateDatasetOutputAssembler,
)
from eval_factory.agent_system.dataset_runtime_fixture import (
    build_fixture_dataset_components,
)
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalStore,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.console_api import (
    AgentShellCompositionConflictError,
    AgentShellCreateSessionCommandV1,
    AgentShellExecutionPhaseV1,
    AgentShellPostMessageCommandV1,
    AgentShellReconcileCommandV1,
    AgentShellService,
    AgentShellWorkspaceKindV1,
    HarnessSourceAdmissionService,
)
from eval_factory.console_api.composition_service import (
    AgentShellCompositionService,
)
from eval_factory.contracts.agent_system_v2 import PlanDecisionKindV2
from eval_factory.contracts.core import ObjectRef
from eval_factory.harness import (
    HarnessSessionStore,
    HarnessSourceAdmissionStore,
    RequirementInterpretationProposalV1,
)
from eval_factory.packs import (
    build_generic_agent_trace_execution_pack,
)
from eval_factory.packs.generic_agent_trace.product_runtime import (
    GenericAgentGraphProductPaths,
)
from eval_factory.packs.generic_agent_trace.shell_composition import (
    GenericAgentFixtureShellGraphStarter,
    GenericAgentShellCompositionConfigV1,
)
from eval_factory.team import TeamStore

pytestmark = pytest.mark.skipif(
    not (RAW_ROOT / "manifest.csv").is_file(),
    reason="private 91-trace corpus is not installed",
)


@dataclass(frozen=True, slots=True)
class _Host:
    shell: AgentShellService
    sources: HarnessSourceAdmissionService
    sessions: HarnessSessionStore
    reviews: PlanReviewService


def _principal_ref(principal: str) -> ObjectRef:
    digest = hashlib.sha256(principal.encode()).hexdigest()
    return ObjectRef(
        object_type="principal",
        object_id=f"principal://sha256/{digest}",
        object_version="v1",
        object_sha256=digest,
    )


def _ready_proposal() -> RequirementInterpretationProposalV1:
    return RequirementInterpretationProposalV1(
        outcome="READY",
        assistant_message="Requirement is ready for Pack 1.2.0.",
        goals=("Build generic Agent evaluation data.",),
        source_expectations=("Use the admitted trace manifest.",),
        target_capabilities=("generic-agent-trace",),
        quality_intent="Auditable candidate data.",
        delivery_intent="Candidate JSON and JSONL.",
        max_model_requests=1_000,
        max_model_tokens=2_000_000,
        max_cost_micro_usd=25_000_000,
    )


def _host(root: Path) -> _Host:
    principal = "user://agent-shell"
    proposal = _ready_proposal()
    sessions, session_store, _agent, _provider = _harness_components(
        root / "harness-runtime",
        proposal,
    )
    source_store = HarnessSourceAdmissionStore(
        root / "source-admission",
    )
    source_service = HarnessSourceAdmissionService(
        sessions=sessions,
        store=source_store,
        trace_schema_ref=_ref("json-schema", "raw-traj-v1"),
        producer_capability_ref=_ref(
            "harness-capability-definition",
            "trace-ingestion",
        ),
        governing_versions=_audit().governing_versions,
    )
    specialist = _specialist_config()
    release = _release_config(
        approval_policy=specialist.job_store.approval_policy,
    )
    runtime_config = _config(
        include_core=True,
        include_r4=True,
        include_attachment=True,
        specialist_config=specialist,
    )
    components = build_fixture_dataset_components(
        factory_store_path=root / "factory.sqlite3",
        private_store_path=root / "factory-private",
        gateway_store_path=root / "factory-gateway.sqlite3",
        config=runtime_config,
        fixture=_fixture(),
        requested_by=principal,
        core_config=_core_config(),
        r4_config=_r4_config(),
        core_workspace_path=root / "core-workspace",
        attachment_config=_attachment_config(),
        attachment_workspace_path=root / "attachment-workspace",
        specialist_config=specialist,
        job_store_path=root / "job-store.sqlite3",
        specialist_workspace_path=root / "specialist-workspace",
        release_config=release,
        candidate_output_path=root / "candidate-output",
    )
    paths = GenericAgentGraphProductPaths(
        team_store=root / "team.sqlite3",
        request_store=root / "capability-requests.sqlite3",
        graph_journal=root / "graph-journal.sqlite3",
        graph_checkpoint=root / "graph-checkpoint.sqlite3",
    )
    policy = _policy(
        include_attachment=True,
        include_specialists=True,
    )
    request = _request(
        include_attachment=True,
        include_specialists=True,
    )
    starter = GenericAgentFixtureShellGraphStarter(
        components=components,
        session_store=session_store,
        source_store=source_store,
        config=GenericAgentShellCompositionConfigV1(
            allowed_task_kinds=policy.allowed_task_kinds,
            pipeline_policy_refs=request.pipeline_policy_refs,
            gateway_registry_refs=tuple(
                sorted(
                    set(request.gateway_registry_refs),
                    key=lambda value: (
                        value.object_type,
                        value.object_id,
                        value.object_version,
                        value.object_sha256,
                    ),
                )
            ),
            output_target_ref=request.output_target_ref,
            max_transitions=policy.max_transitions,
            max_plan_revisions=policy.max_plan_revisions,
            max_agent_attempts=policy.max_agent_attempts,
        ),
        paths=paths,
        candidate_output_root=root / "candidate-output",
    )
    team_store = TeamStore(paths.team_store)
    graph_journal = FactoryGraphJournalStore(paths.graph_journal)
    composition = AgentShellCompositionService(
        sessions=sessions,
        sources=source_store,
        team_store=team_store,
        factory_store=components.runtime.store,
        graph_journal=graph_journal,
        plan_reviews=components.runtime.plan_reviews,
        candidate_output=CandidateDatasetOutputAssembler(
            store=components.runtime.store,
            root=root / "candidate-output",
        ),
        graph_starter=starter,
    )
    return _Host(
        shell=AgentShellService(
            sessions=sessions,
            composition_ref=build_generic_agent_trace_execution_pack(
                audit=_audit(),
            ).composition.to_ref(),
            principal_resolver=_principal_ref,
            governing_versions=_audit().governing_versions,
            composition=composition,
        ),
        sources=source_service,
        sessions=session_store,
        reviews=components.runtime.plan_reviews,
    )


@pytest.mark.asyncio
async def test_ready_admitted_source_starts_pack_graph_and_replays_after_restart(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    created = host.shell.create_session(
        AgentShellCreateSessionCommandV1(
            session_id="session-shell-graph",
            incarnation_id="session-shell-graph-incarnation",
            idempotency_key="create-session-shell-graph",
        ),
        principal="user://agent-shell",
    )
    (tmp_path / "source-fixture").mkdir()
    manifest_path, _manifest_sha256 = _real_trace_root(
        tmp_path / "source-fixture",
        instance_ids=("LH_077",),
    )
    trace_path = next(manifest_path.parent.glob("*.jsonl"))
    admitted = _admit(
        host.sources,
        session_id="session-shell-graph",
        manifest=manifest_path.read_bytes(),
        traces=((trace_path.name, trace_path.read_bytes()),),
        expected_session_version=created.session.session_version,
    )

    projection = await host.shell.post_message(
        "session-shell-graph",
        AgentShellPostMessageCommandV1(
            expected_session_version=created.session.session_version,
            content="Build the admitted generic Agent evaluation dataset.",
            artifact_envelope_refs=admitted.artifact_envelope_refs,
            idempotency_key="start-session-shell-graph",
        ),
        principal="user://agent-shell",
    )

    assert projection.execution_phase is AgentShellExecutionPhaseV1.WAITING_REVIEW
    assert projection.factory is not None
    assert projection.graph is not None
    assert projection.team is not None
    assert projection.plan_reviews
    assert projection.pending_interactions
    assert projection.pending_interactions[0].kind.value == "PLAN_REVIEW"
    assert projection.source_admission_ref == admitted.admission_ref
    assert next(
        value for value in projection.workspaces if value.kind is AgentShellWorkspaceKindV1.TEAM
    ).owner_refs == (projection.team.team_ref,)
    member = host.shell.member(
        "session-shell-graph",
        projection.team.members[0].member_id,
    )
    assert member.team_ref == projection.team.team_ref
    assert "team-message-body" not in member.model_dump_json()
    assert str(tmp_path) not in projection.model_dump_json()

    with pytest.raises(
        AgentShellCompositionConflictError,
        match="does not accept",
    ):
        await host.shell.post_message(
            "session-shell-graph",
            AgentShellPostMessageCommandV1(
                expected_session_version=(projection.session.session_version),
                content="Replace the active requirement.",
                artifact_envelope_refs=(admitted.artifact_envelope_refs),
                idempotency_key="replace-session-shell-graph",
            ),
            principal="user://agent-shell",
        )
    assert (
        host.sessions.get_projection(
            "session-shell-graph",
        ).session.session_version
        == projection.session.session_version
    )

    review_id = projection.plan_reviews[0].request_ref.object_id
    review = host.reviews.show(review_id)
    host.reviews.decide(
        review_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=review.request.plan_version,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by="user://agent-shell",
            reason_code="SHELL_GLOBAL_PLAN_APPROVED",
            idempotency_key="approve-shell-global",
        ),
        audit=_audit(),
    )
    host.reviews.resume(
        review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=review.request.plan_version,
            resumed_by="user://agent-shell",
            idempotency_key="resume-shell-global",
        ),
        audit=_audit(),
    )
    cursor = projection.reconnect_cursor
    reconcile = AgentShellReconcileCommandV1(
        expected_session_version=projection.session.session_version,
        idempotency_key="reconcile-session-shell-graph",
    )
    first = await host.shell.reconcile(
        "session-shell-graph",
        reconcile,
        principal="user://agent-shell",
    )
    replay = await host.shell.reconcile(
        "session-shell-graph",
        reconcile,
        principal="user://agent-shell",
    )
    assert replay == first
    assert replay.reconnect_cursor > cursor
    assert replay.graph is not None
    assert projection.graph is not None
    assert replay.graph.binding_ref != projection.graph.binding_ref

    restarted = _host(tmp_path)
    recovered = restarted.shell.get_session(
        "session-shell-graph",
    )
    assert recovered == replay


@pytest.mark.asyncio
async def test_ready_message_must_explicitly_confirm_admitted_source(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    created = host.shell.create_session(
        AgentShellCreateSessionCommandV1(
            session_id="session-shell-graph",
            incarnation_id="session-shell-graph-incarnation",
            idempotency_key="create-session-shell-graph",
        ),
        principal="user://agent-shell",
    )
    (tmp_path / "source-fixture").mkdir()
    manifest_path, _manifest_sha256 = _real_trace_root(
        tmp_path / "source-fixture",
        instance_ids=("LH_077",),
    )
    trace_path = next(manifest_path.parent.glob("*.jsonl"))
    _admit(
        host.sources,
        session_id="session-shell-graph",
        manifest=manifest_path.read_bytes(),
        traces=((trace_path.name, trace_path.read_bytes()),),
        expected_session_version=created.session.session_version,
    )

    projection = await host.shell.post_message(
        "session-shell-graph",
        AgentShellPostMessageCommandV1(
            expected_session_version=created.session.session_version,
            content="Build the dataset without confirming its source.",
            artifact_envelope_refs=(),
            idempotency_key="unconfirmed-session-shell-graph",
        ),
        principal="user://agent-shell",
    )

    assert projection.execution_phase is AgentShellExecutionPhaseV1.BLOCKED
    assert projection.pending_interactions[0].reason_codes == ("SOURCE_CONFIRMATION_REQUIRED",)
    assert projection.graph is None
    assert projection.factory is None
    assert projection.team is None
