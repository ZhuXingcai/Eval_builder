from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from eval_factory.agent_system.plan_review import (
    PlanReviewConflictError,
    PlanReviewDecisionSubmissionV1,
    PlanReviewEditSubmissionV1,
    PlanReviewNotResumableError,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.cli import app
from eval_factory.console_api import (
    GlobalPlanEditableFieldsV1,
    PlanReviewEditPayloadV1,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    FactoryRunStatusV2,
    PlanDecisionKindV2,
    PlanKindV2,
    PlanReviewStateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

HASH = "a" * 64
NOW = datetime(2026, 8, 6, tzinfo=UTC)
USER = "user://plan-owner"


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit(actor: str = "plan-review-test") -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by=actor,
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="plan-review-v1",
                sha256=HASH,
            ),
        ),
    )


def _registry() -> AgentRegistry:
    capability = AgentCapabilityV2.create(
        capability_id="agent-capability://trace-extraction",
        task_kinds=("trace-extraction",),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("extracted-user-prompt",),
        model_capabilities=("structured-output",),
        tool_ids=("trace-query",),
        data_purposes=("evaluation-dataset-construction",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=_audit(),
    )
    definition = AgentDefinitionV2.create(
        agent_definition_id="agent-definition://trace-extraction",
        agent_role="trace-extraction-agent",
        agent_version="v1",
        capability_refs=(capability.to_ref(),),
        prompt_template_ref=_ref("prompt-template"),
        model_policy_ref=_ref("model-routing-policy"),
        tool_ids=("trace-query",),
        data_purpose="evaluation-dataset-construction",
        allowed_data_classifications=("RESTRICTED_TRACE_DERIVED",),
        validator_refs=(_ref("validator"),),
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
        workspace_isolated=True,
        network_allowed=False,
        audit=_audit(),
    )
    return AgentRegistry(capabilities=(capability,), definitions=(definition,))


def _policy() -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://plan-review",
        allowed_task_kinds=("trace-extraction",),
        max_transitions=32,
        max_plan_revisions=4,
        max_agent_attempts=2,
        max_model_requests=10,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
        audit=_audit(),
    )


def _plan(
    run_ref: ObjectRef,
    *,
    version: int = 1,
    predecessor: ObjectRef | None = None,
    extra_goal: str | None = None,
) -> DatasetBuildPlanV2:
    task = DatasetBuildPlanTaskV2(
        task_key="extract",
        stage="core",
        task_kind="trace-extraction",
        agent_role="trace-extraction-agent",
        dependency_task_keys=(),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("extracted-user-prompt",),
        required_capability_ids=("agent-capability://trace-extraction",),
        acceptance_check_refs=(_ref("acceptance-check"),),
        plan_review_kind=PlanKindV2.GLOBAL_BUILD,
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
    )
    goals = ["Build source-grounded rewrite candidates."]
    if extra_goal is not None:
        goals.append(extra_goal)
    return DatasetBuildPlanV2.create(
        plan_id="dataset-build-plan://plan-review",
        run_ref=run_ref,
        plan_version=version,
        predecessor_plan_ref=predecessor,
        goals=tuple(goals),
        user_constraints=("Intent cannot replace the original prompt.",),
        assumptions=(),
        unresolved_questions=(),
        stage_order=("core",),
        tasks=(task,),
        required_review_kinds=(PlanKindV2.GLOBAL_BUILD,),
        total_model_requests=2,
        total_model_tokens=4096,
        total_cost_micro_usd=100_000,
        audit=_audit(),
    )


def _service(tmp_path: Path) -> tuple[PlanReviewService, FactoryControlStore]:
    store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    requirement = EvaluationRequirementSpecV2.create(
        requirement_spec_id="evaluation-requirement-spec://plan-review",
        run_id="factory-run://plan-review",
        source_ref=_ref("evaluation-requirement-source"),
        goals=("Build source-grounded rewrite candidates.",),
        constraints=("Intent cannot replace the original prompt.",),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )
    run = store.create_run(
        policy=_policy(),
        requirement=requirement,
        idempotency_key="create-run",
    )
    compiler = DatasetBuildPlanCompiler(_registry())
    plan = _plan(run.to_ref())
    compiled = compiler.compile(plan=plan, policy=_policy(), audit=_audit())
    store.commit_plan(
        run_id=run.run_id,
        expected_run_version=0,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-plan",
    )
    return PlanReviewService(store, compiler=compiler, clock=lambda: NOW), store


def _open(service: PlanReviewService):
    return service.open_global_plan(
        run_id="factory-run://plan-review",
        title="Review the core dataset build plan",
        summary_lines=(
            "One deterministic extraction task.",
            "Original prompts remain source-bound.",
        ),
        editable_paths=("goals", "user_constraints"),
        warning_codes=("ATTACHMENT_PHASE_DEFERRED",),
        requested_by=USER,
        idempotency_key="open-review",
        audit=_audit(),
    )


def _decision(kind: PlanDecisionKindV2, key: str) -> PlanReviewDecisionSubmissionV1:
    return PlanReviewDecisionSubmissionV1(
        expected_plan_version=1,
        decision=kind,
        decided_by=USER,
        reason_code=f"{kind.value}_BY_USER",
        idempotency_key=key,
    )


def test_plan_review_approve_resume_and_exact_replay(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    opened = _open(service)
    replay = _open(service)

    assert replay == opened
    assert opened.result.state is PlanReviewStateV2.PENDING_REVIEW
    assert store.get_run(opened.run_id).status is FactoryRunStatusV2.WAITING_REVIEW
    assert service.export(opened.request.review_request_id) == opened
    assert (
        service.list_reviews(
            run_id=opened.run_id,
            state=PlanReviewStateV2.PENDING_REVIEW,
        ).total
        == 1
    )

    approved = service.decide(
        opened.request.review_request_id,
        _decision(PlanDecisionKindV2.APPROVE, "approve-review"),
        audit=_audit(),
    )
    assert approved.result.state is PlanReviewStateV2.APPROVED
    assert (
        service.decide(
            opened.request.review_request_id,
            _decision(PlanDecisionKindV2.APPROVE, "approve-review"),
            audit=_audit(),
        )
        == approved
    )

    resumed = service.resume(
        opened.request.review_request_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-review",
        ),
        audit=_audit(),
    )
    assert resumed.result.state is PlanReviewStateV2.RESUMED
    assert store.get_run(opened.run_id).status is FactoryRunStatusV2.PLANNING
    assert (
        service.require_resumed_review(
            run_id=opened.run_id,
            plan_kind=PlanKindV2.GLOBAL_BUILD,
            plan_ref=opened.plan.to_ref(),
        )
        == resumed
    )
    assert (
        service.resume(
            opened.request.review_request_id,
            PlanReviewResumeSubmissionV1(
                expected_plan_version=1,
                resumed_by=USER,
                idempotency_key="resume-review",
            ),
            audit=_audit(),
        )
        == resumed
    )


def test_plan_review_edit_compiles_and_atomically_advances_plan_head(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    opened = _open(service)
    waiting_run = store.get_run(opened.run_id)
    edited_plan = _plan(
        waiting_run.to_ref(),
        version=2,
        predecessor=opened.plan.to_ref(),
        extra_goal="Expose every model route in the audit output.",
    )
    edited = service.edit_global_plan(
        opened.request.review_request_id,
        PlanReviewEditSubmissionV1(
            expected_plan_version=1,
            edited_plan=edited_plan,
            changed_paths=("goals",),
            invalidated_object_refs=(),
            decided_by=USER,
            reason_code="ADD_ROUTE_AUDIT_GOAL",
            idempotency_key="edit-review",
        ),
        audit=_audit(),
    )

    assert edited.result.state is PlanReviewStateV2.REVISION_REQUESTED
    assert edited.revision is not None
    assert edited.plan == edited_plan
    base, _ = store.get_plan(opened.run_id)
    assert base.plan_version == 1

    with pytest.raises(PlanReviewConflictError, match=r"stale|contiguous"):
        service.edit_global_plan(
            opened.request.review_request_id,
            PlanReviewEditSubmissionV1(
                expected_plan_version=2,
                edited_plan=edited_plan,
                changed_paths=("goals",),
                invalidated_object_refs=(),
                decided_by=USER,
                reason_code="STALE_EDIT",
                idempotency_key="stale-edit",
            ),
            audit=_audit(),
        )

    resumed = service.resume(
        opened.request.review_request_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-edited-review",
        ),
        audit=_audit(),
    )
    current_plan, current_compiled = store.get_plan(opened.run_id)
    assert resumed.result.plan_version == 2
    assert current_plan == edited_plan
    assert current_compiled.source_plan_ref == edited_plan.to_ref()


@pytest.mark.parametrize(
    ("decision", "state", "run_status"),
    (
        (
            PlanDecisionKindV2.REJECT,
            PlanReviewStateV2.REJECTED,
            FactoryRunStatusV2.FAILED,
        ),
        (
            PlanDecisionKindV2.DEFER,
            PlanReviewStateV2.DEFERRED,
            FactoryRunStatusV2.WAITING_REVIEW,
        ),
        (
            PlanDecisionKindV2.REQUEST_MORE,
            PlanReviewStateV2.REVISION_REQUESTED,
            FactoryRunStatusV2.WAITING_REVIEW,
        ),
    ),
)
def test_plan_review_terminal_and_paused_decisions(
    tmp_path: Path,
    decision: PlanDecisionKindV2,
    state: PlanReviewStateV2,
    run_status: FactoryRunStatusV2,
) -> None:
    service, store = _service(tmp_path)
    opened = _open(service)
    decided = service.decide(
        opened.request.review_request_id,
        _decision(decision, f"decision-{decision.value.casefold()}"),
        audit=_audit(),
    )

    assert decided.result.state is state
    assert store.get_run(opened.run_id).status is run_status
    if decision is not PlanDecisionKindV2.REJECT:
        with pytest.raises(PlanReviewNotResumableError):
            service.resume(
                opened.request.review_request_id,
                PlanReviewResumeSubmissionV1(
                    expected_plan_version=1,
                    resumed_by=USER,
                    idempotency_key=f"resume-{decision.value.casefold()}",
                ),
                audit=_audit(),
            )


def test_two_clients_commit_one_plan_decision_authority(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    opened = _open(service)

    def decide(kind: PlanDecisionKindV2) -> str:
        try:
            return service.decide(
                opened.request.review_request_id,
                _decision(kind, f"race-{kind.value.casefold()}"),
                audit=_audit(kind.value),
            ).result.state.value
        except PlanReviewConflictError:
            return "CONFLICT"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(
            executor.map(
                decide,
                (PlanDecisionKindV2.APPROVE, PlanDecisionKindV2.REJECT),
            )
        )

    assert outcomes.count("CONFLICT") == 1
    assert service.show(opened.request.review_request_id).decision is not None


def test_agent_plan_cli_uses_shared_show_decide_and_resume_service(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    opened = _open(service)
    base = ["agent", "plan"]
    runner = CliRunner()

    pending = runner.invoke(
        app,
        [*base, "pending", "--factory-store", str(store.path)],
    )
    show = runner.invoke(
        app,
        [
            *base,
            "show",
            "--factory-store",
            str(store.path),
            "--review",
            opened.request.review_request_id,
        ],
    )
    exported = runner.invoke(
        app,
        [
            *base,
            "export",
            "--factory-store",
            str(store.path),
            "--review",
            opened.request.review_request_id,
        ],
    )
    assert pending.exit_code == show.exit_code == exported.exit_code == 0
    assert json.loads(show.stdout) == json.loads(exported.stdout)
    assert json.loads(pending.stdout)["total"] == 1

    approved = runner.invoke(
        app,
        [
            *base,
            "approve",
            "--factory-store",
            str(store.path),
            "--review",
            opened.request.review_request_id,
            "--expected-plan-version",
            "1",
            "--user",
            USER,
            "--reason-code",
            "APPROVE_BY_CLI",
            "--idempotency-key",
            "approve-cli",
        ],
    )
    assert approved.exit_code == 0, approved.stderr
    assert json.loads(approved.stdout)["result"]["state"] == "APPROVED"

    resumed = runner.invoke(
        app,
        [
            *base,
            "resume",
            "--factory-store",
            str(store.path),
            "--review",
            opened.request.review_request_id,
            "--expected-plan-version",
            "1",
            "--user",
            USER,
            "--idempotency-key",
            "resume-cli",
        ],
    )
    assert resumed.exit_code == 0, resumed.stderr
    assert json.loads(resumed.stdout)["result"]["state"] == "RESUMED"


def test_agent_plan_cli_edit_accepts_strict_successor_and_registry(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    opened = _open(service)
    registry = _registry()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "capabilities": [value.model_dump(mode="json") for value in registry.capabilities],
                "definitions": [value.model_dump(mode="json") for value in registry.definitions],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    edited_plan = _plan(
        store.get_run(opened.run_id).to_ref(),
        version=2,
        predecessor=opened.plan.to_ref(),
        extra_goal="Expose model route refs in delivery.",
    )
    submission = PlanReviewEditPayloadV1(
        expected_plan_version=1,
        edited_plan=GlobalPlanEditableFieldsV1(
            goals=edited_plan.goals,
            user_constraints=edited_plan.user_constraints,
            assumptions=edited_plan.assumptions,
            unresolved_questions=edited_plan.unresolved_questions,
            stage_order=edited_plan.stage_order,
            tasks=edited_plan.tasks,
            required_review_kinds=edited_plan.required_review_kinds,
            total_model_requests=edited_plan.total_model_requests,
            total_model_tokens=edited_plan.total_model_tokens,
            total_cost_micro_usd=edited_plan.total_cost_micro_usd,
        ),
        changed_paths=("goals",),
        invalidated_object_refs=(),
        decided_by=USER,
        reason_code="EDIT_BY_CLI",
        idempotency_key="edit-cli",
    )
    submission_path = tmp_path / "edit.json"
    submission_path.write_text(submission.model_dump_json(), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "agent",
            "plan",
            "edit",
            "--factory-store",
            str(store.path),
            "--review",
            opened.request.review_request_id,
            "--submission",
            str(submission_path),
            "--registry",
            str(registry_path),
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["result"]["state"] == "REVISION_REQUESTED"
