from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from tests.eval_factory.integration.test_harness_agent_loop import (
    _audit,
    _components,
)
from tests.eval_factory.unit.test_harness_source_admission import (
    _admit,
    _ref,
)

from eval_factory.agent_system.candidate_output import (
    CandidateDatasetOutputAssembler,
)
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalIntegrityError,
    FactoryGraphJournalStore,
)
from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.console_api import (
    AgentShellCloseSessionCommandV1,
    AgentShellCompositionIntegrityError,
    AgentShellCompositionPreflight,
    AgentShellCreateSessionCommandV1,
    AgentShellExecutionPhaseV1,
    AgentShellPostMessageCommandV1,
    AgentShellProjectionV1,
    AgentShellReconcileCommandV1,
    AgentShellService,
    AgentShellWorkspaceKindV1,
    AgentShellWorkspaceStatusV1,
    HarnessSourceAdmissionService,
)
from eval_factory.console_api.composition_service import (
    AgentShellCompositionService,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunViewV2,
)
from eval_factory.harness import (
    HarnessSessionConflictError,
    HarnessSessionStatusV1,
    HarnessSessionStore,
    HarnessSourceAdmissionStore,
    HarnessTurnResultV1,
    RequirementInterpretationProposalV1,
)
from eval_factory.harness.graph_models import (
    HarnessGraphExecutionBindingV1,
)
from eval_factory.team import TeamStore


@dataclass
class _Starter:
    result: AgentShellCompositionPreflight
    preflight_calls: int = 0
    start_calls: int = 0
    advance_calls: int = 0

    def preflight(
        self,
        session_id: str,
    ) -> AgentShellCompositionPreflight:
        assert session_id == "session-shell-composition"
        self.preflight_calls += 1
        return self.result

    async def start(
        self,
        session_id: str,
        *,
        turn: HarnessTurnResultV1,
        principal: str,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2:
        del turn, audit
        assert session_id == "session-shell-composition"
        assert principal == "user://agent-shell"
        self.start_calls += 1
        return cast(FactoryDatasetRunViewV2, object())

    async def advance(
        self,
        session_id: str,
        *,
        binding: HarnessGraphExecutionBindingV1,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2:
        del session_id, binding, audit
        self.advance_calls += 1
        return cast(FactoryDatasetRunViewV2, object())

    def current_view(
        self,
        session_id: str,
        *,
        binding: HarnessGraphExecutionBindingV1,
    ) -> FactoryDatasetRunViewV2:
        del session_id, binding
        raise AssertionError("no Graph binding is expected in this unit fixture")


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
        assistant_message="Requirement is ready.",
        goals=("Build generic Agent evaluation data.",),
        source_expectations=("Use the admitted trace manifest.",),
        target_capabilities=("generic-agent-trace",),
        quality_intent="Auditable candidate data.",
        delivery_intent="Candidate JSON and JSONL.",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=10_000_000,
    )


def _shell(
    tmp_path: Path,
    *,
    proposal: RequirementInterpretationProposalV1,
    starter: _Starter,
) -> tuple[
    AgentShellService,
    HarnessSourceAdmissionService,
    HarnessSessionStore,
]:
    sessions, session_store, _agent, _provider = _components(
        tmp_path / "harness",
        proposal,
    )
    source_store = HarnessSourceAdmissionStore(
        tmp_path / "source",
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
    factory_store = FactoryControlStore(
        tmp_path / "factory.sqlite3",
    )
    reviews = PlanReviewService(factory_store)
    composition = AgentShellCompositionService(
        sessions=sessions,
        sources=source_store,
        team_store=TeamStore(tmp_path / "team.sqlite3"),
        factory_store=factory_store,
        graph_journal=FactoryGraphJournalStore(
            tmp_path / "graph-journal.sqlite3",
        ),
        plan_reviews=reviews,
        candidate_output=CandidateDatasetOutputAssembler(
            store=factory_store,
            root=tmp_path / "candidate-output",
        ),
        graph_starter=starter,
    )
    shell = AgentShellService(
        sessions=sessions,
        composition_ref=_ref(
            "harness-composition",
            "shell-composition",
        ),
        principal_resolver=_principal_ref,
        governing_versions=_audit().governing_versions,
        composition=composition,
    )
    return shell, source_service, session_store


def _create(shell: AgentShellService) -> AgentShellProjectionV1:
    return shell.create_session(
        AgentShellCreateSessionCommandV1(
            session_id="session-shell-composition",
            incarnation_id="session-shell-composition-incarnation",
            idempotency_key="create-session-shell-composition",
        ),
        principal="user://agent-shell",
    )


def test_shell_close_hides_session_without_erasing_projection(
    tmp_path: Path,
) -> None:
    starter = _Starter(
        AgentShellCompositionPreflight(
            ready=False,
            reason_codes=("SOURCE_CONFIRMATION_REQUIRED",),
        )
    )
    shell, _source_service, store = _shell(
        tmp_path,
        proposal=_ready_proposal(),
        starter=starter,
    )
    created = _create(shell)
    command = AgentShellCloseSessionCommandV1(
        expected_session_version=created.session.session_version,
        idempotency_key="close-session-shell-composition",
    )

    closed = shell.close_session(
        created.session.session_id,
        command,
        principal="user://agent-shell",
    )

    assert closed.session.status is HarnessSessionStatusV1.CLOSED
    assert shell.list_sessions(offset=0, limit=100).total == 0
    assert (
        store.get_projection(created.session.session_id).session.status
        is HarnessSessionStatusV1.CLOSED
    )
    assert (
        shell.close_session(
            created.session.session_id,
            command,
            principal="user://agent-shell",
        )
        == closed
    )


@pytest.mark.asyncio
async def test_shell_projects_waiting_source_then_starts_ready_source(
    tmp_path: Path,
) -> None:
    starter = _Starter(
        AgentShellCompositionPreflight(
            ready=False,
            reason_codes=("SOURCE_CONFIRMATION_REQUIRED",),
        )
    )
    shell, source_service, _store = _shell(
        tmp_path,
        proposal=_ready_proposal(),
        starter=starter,
    )
    created = _create(shell)
    assert created.execution_phase is AgentShellExecutionPhaseV1.WAITING_REQUIREMENT

    admitted = _admit(
        source_service,
        session_id="session-shell-composition",
    )
    blocked = await shell.post_message(
        "session-shell-composition",
        AgentShellPostMessageCommandV1(
            expected_session_version=created.session.session_version,
            content="Build the admitted evaluation dataset.",
            artifact_envelope_refs=admitted.artifact_envelope_refs,
            idempotency_key="ready-session-shell-composition",
        ),
        principal="user://agent-shell",
    )

    assert blocked.execution_phase is AgentShellExecutionPhaseV1.BLOCKED
    assert blocked.source_admission_ref == admitted.admission_ref
    assert blocked.pending_interactions[0].reason_codes == ("SOURCE_CONFIRMATION_REQUIRED",)
    assert starter.start_calls == 0
    starter.result = AgentShellCompositionPreflight(
        ready=True,
        reason_codes=(),
    )

    with pytest.raises(
        AgentShellCompositionIntegrityError,
        match="committed binding",
    ):
        await shell.post_message(
            "session-shell-composition",
            AgentShellPostMessageCommandV1(
                expected_session_version=created.session.session_version,
                content="Build the admitted evaluation dataset.",
                artifact_envelope_refs=admitted.artifact_envelope_refs,
                idempotency_key="ready-session-shell-composition",
            ),
            principal="user://agent-shell",
        )
    replay = shell.get_session(
        "session-shell-composition",
    )
    assert replay.execution_phase is AgentShellExecutionPhaseV1.READY
    assert starter.start_calls == 1
    assert tuple(value.kind for value in replay.workspaces) == tuple(AgentShellWorkspaceKindV1)
    trace_workspace = next(
        value for value in replay.workspaces if value.kind is AgentShellWorkspaceKindV1.TRACE
    )
    assert trace_workspace.status is AgentShellWorkspaceStatusV1.AVAILABLE
    assert trace_workspace.owner_refs == (admitted.admission_ref,)
    serialized = replay.model_dump_json()
    assert str(tmp_path) not in serialized
    assert "harness-source-content" not in serialized
    assert '"request"' not in serialized


@pytest.mark.asyncio
async def test_shell_reconcile_command_replays_and_rejects_changed_input(
    tmp_path: Path,
) -> None:
    starter = _Starter(
        AgentShellCompositionPreflight(
            ready=False,
            reason_codes=("SOURCE_NOT_ADMITTED",),
        )
    )
    shell, _source_service, _store = _shell(
        tmp_path,
        proposal=_ready_proposal(),
        starter=starter,
    )
    created = _create(shell)
    command = AgentShellReconcileCommandV1(
        expected_session_version=created.session.session_version,
        idempotency_key="reconcile-session-shell-composition",
    )

    first = await shell.reconcile(
        "session-shell-composition",
        command,
        principal="user://agent-shell",
    )
    replay = await shell.reconcile(
        "session-shell-composition",
        command,
        principal="user://agent-shell",
    )

    assert replay == first
    with pytest.raises(HarnessSessionConflictError, match="idempotency"):
        await shell.reconcile(
            "session-shell-composition",
            command,
            principal="user://another-agent-shell",
        )


def test_shell_maps_graph_lookup_drift_to_composition_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starter = _Starter(
        AgentShellCompositionPreflight(
            ready=False,
            reason_codes=("SOURCE_NOT_ADMITTED",),
        )
    )
    shell, _source_service, _store = _shell(
        tmp_path,
        proposal=_ready_proposal(),
        starter=starter,
    )
    _create(shell)
    composition = shell._require_composition()

    def fail_lookup(_: ObjectRef) -> HarnessGraphExecutionBindingV1:
        raise FactoryGraphJournalIntegrityError(
            "private graph detail",
        )

    monkeypatch.setattr(
        composition.graph_journal,
        "get_binding_for_session",
        fail_lookup,
    )
    with pytest.raises(
        AgentShellCompositionIntegrityError,
        match="integrity",
    ):
        shell.get_session(
            "session-shell-composition",
        )
