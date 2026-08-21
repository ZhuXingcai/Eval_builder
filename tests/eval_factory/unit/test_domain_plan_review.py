from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from eval_factory.agent_system.attachment_planning import (
    AttachmentGenerationPlanCompiler,
)
from eval_factory.agent_system.delivery_planning import (
    DatasetDeliveryPlanCompiler,
)
from eval_factory.agent_system.plan_adapters import (
    AttachmentGenerationPlanAdapter,
    DatasetDeliveryPlanAdapter,
    GlobalBuildPlanAdapter,
    ReviewablePlanAdapterError,
    ReviewablePlanAdapterRegistry,
)
from eval_factory.agent_system.plan_review import (
    AttachmentPlanReviewEditSubmissionV1,
    DatasetDeliveryPlanReviewEditSubmissionV1,
    PlanReviewConflictError,
    PlanReviewDecisionSubmissionV1,
    PlanReviewIntegrityError,
    PlanReviewNotResumableError,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
    PlanReviewViewV1,
)
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.agent_system.store import (
    FactoryControlConflictError,
    FactoryControlHeadRebuild,
    FactoryControlInjectedCrash,
    FactoryControlStore,
    FactoryControlStoreFaultPoint,
    StaticFactoryControlStoreFaultInjector,
)
from eval_factory.cli import app
from eval_factory.console_api import (
    AttachmentPlanEditableFieldsV1,
    DatasetDeliveryPlanEditableFieldsV1,
    PlanReviewEditPayloadV1,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
    AttachmentGenerationPlanV2,
    AttachmentMockWorkV2,
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
from eval_factory.contracts.dataset_runtime_v2 import (
    DatasetDeliveryPlanV2,
)

HASH = "a" * 64
NOW = datetime(2026, 8, 6, tzinfo=UTC)
RUN_ID = "factory-run://domain-plan-review"
USER = "user://domain-plan-owner"


def _ref(
    object_type: str,
    suffix: str = "example",
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit(actor: str = "domain-plan-review-test") -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by=actor,
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="domain-plan-review-v1",
                sha256=HASH,
            ),
        ),
    )


def _registry() -> AgentRegistry:
    trace_capability = AgentCapabilityV2.create(
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
    attachment_capability = AgentCapabilityV2.create(
        capability_id="agent-capability://attachment-mock",
        task_kinds=("attachment-mock",),
        input_object_types=("attachment-planning-context",),
        output_object_types=("attachment-group-result",),
        model_capabilities=("structured-output",),
        tool_ids=("attachment-execution",),
        data_purposes=("attachment-production",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=_audit(),
    )
    trace_definition = AgentDefinitionV2.create(
        agent_definition_id="agent-definition://trace-extraction",
        agent_role="trace-extraction-agent",
        agent_version="v1",
        capability_refs=(trace_capability.to_ref(),),
        prompt_template_ref=_ref("prompt-template", "trace"),
        model_policy_ref=_ref("model-routing-policy", "trace"),
        tool_ids=("trace-query",),
        data_purpose="evaluation-dataset-construction",
        allowed_data_classifications=("RESTRICTED_TRACE_DERIVED",),
        validator_refs=(_ref("validator", "trace"),),
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
        workspace_isolated=True,
        network_allowed=False,
        audit=_audit(),
    )
    attachment_definition = AgentDefinitionV2.create(
        agent_definition_id="agent-definition://attachment-mock",
        agent_role="attachment-mock-agent",
        agent_version="v1",
        capability_refs=(attachment_capability.to_ref(),),
        prompt_template_ref=_ref("prompt-template", "attachment"),
        model_policy_ref=_ref("model-routing-policy", "attachment"),
        tool_ids=("attachment-execution",),
        data_purpose="attachment-production",
        allowed_data_classifications=("RESTRICTED_TRACE_DERIVED",),
        validator_refs=(_ref("validator", "attachment"),),
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
        workspace_isolated=True,
        network_allowed=False,
        audit=_audit(),
    )
    return AgentRegistry(
        capabilities=(trace_capability, attachment_capability),
        definitions=(trace_definition, attachment_definition),
    )


def _policy() -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://domain-plan-review",
        allowed_task_kinds=(
            "attachment-mock",
            "criteria-rubric",
            "grading-design",
            "trace-extraction",
        ),
        max_transitions=64,
        max_plan_revisions=4,
        max_agent_attempts=2,
        max_model_requests=20,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
        audit=_audit(),
    )


def _global_plan(run_ref: ObjectRef) -> DatasetBuildPlanV2:
    task = DatasetBuildPlanTaskV2(
        task_key="extract",
        stage="core",
        task_kind="trace-extraction",
        agent_role="trace-extraction-agent",
        dependency_task_keys=(),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("extracted-user-prompt",),
        required_capability_ids=("agent-capability://trace-extraction",),
        acceptance_check_refs=(_ref("acceptance-check", "trace"),),
        plan_review_kind=PlanKindV2.GLOBAL_BUILD,
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
    )
    return DatasetBuildPlanV2.create(
        plan_id="dataset-build-plan://domain-plan-review",
        run_ref=run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        goals=("Build source-grounded rewrite candidates.",),
        user_constraints=("Do not expose raw traces.",),
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


def _work(
    key: str,
    *,
    dependencies: tuple[str, ...] = (),
    tool_ids: tuple[str, ...] = ("attachment-execution",),
) -> AttachmentMockWorkV2:
    return AttachmentMockWorkV2(
        work_key=key,
        artifact_group_ref=_ref("artifact-execution-group", key),
        artifact_ids=(f"artifact-{key}",),
        agent_role="attachment-mock-agent",
        dependency_work_keys=dependencies,
        input_object_types=("attachment-planning-context",),
        output_object_types=("attachment-group-result",),
        required_capability_ids=("agent-capability://attachment-mock",),
        allowed_tool_ids=tool_ids,
        data_purposes=("attachment-production",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        workspace_policy_ref=_ref("agent-workspace-policy"),
        acceptance_check_refs=(_ref("acceptance-check", key),),
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
    )


def _attachment_plan(
    run_ref: ObjectRef,
    *,
    version: int = 1,
    predecessor: ObjectRef | None = None,
    works: tuple[AttachmentMockWorkV2, ...] | None = None,
    max_parallel_groups: int = 2,
) -> AttachmentGenerationPlanV2:
    values = works or (
        _work("work-a"),
        _work("work-b", dependencies=("work-a",)),
    )
    return AttachmentGenerationPlanV2.create(
        plan_id="attachment-generation-plan://domain-plan-review",
        run_ref=run_ref,
        plan_version=version,
        predecessor_plan_ref=predecessor,
        producer_task_view_ref=_ref("producer-task-view"),
        evidence_bundle_ref=_ref("evidence-bundle", version="v1"),
        attachment_planning_context_ref=_ref("attachment-planning-context"),
        works=values,
        max_parallel_groups=max_parallel_groups,
        quality_policy_ref=_ref("attachment-quality-policy"),
        solvability_policy_ref=_ref("solvability-policy"),
        total_model_requests=sum(value.max_model_requests for value in values),
        total_model_tokens=sum(value.max_model_tokens for value in values),
        total_cost_micro_usd=sum(value.max_cost_micro_usd for value in values),
        audit=_audit(),
    )


def _setup(
    tmp_path: Path,
) -> tuple[
    PlanReviewService,
    FactoryControlStore,
    AttachmentGenerationPlanCompiler,
]:
    store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    requirement = EvaluationRequirementSpecV2.create(
        requirement_spec_id="evaluation-requirement-spec://domain-plan-review",
        run_id=RUN_ID,
        source_ref=_ref("evaluation-requirement-source"),
        goals=("Build source-grounded rewrite candidates.",),
        constraints=("Do not expose raw traces.",),
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
    registry = _registry()
    global_compiler = DatasetBuildPlanCompiler(registry)
    attachment_compiler = AttachmentGenerationPlanCompiler(registry)
    global_plan = _global_plan(run.to_ref())
    compiled_global = global_compiler.compile(
        plan=global_plan,
        policy=_policy(),
        audit=_audit(),
    )
    store.commit_plan(
        run_id=run.run_id,
        expected_run_version=run.run_version,
        plan=global_plan,
        compiled_plan=compiled_global,
        audit=_audit(),
        idempotency_key="commit-global-plan",
    )
    adapters = ReviewablePlanAdapterRegistry(
        (
            GlobalBuildPlanAdapter(global_compiler),
            AttachmentGenerationPlanAdapter(attachment_compiler),
        )
    )
    service = PlanReviewService(
        store,
        compiler=global_compiler,
        adapter_registry=adapters,
        clock=lambda: NOW,
    )
    return service, store, attachment_compiler


def _commit_attachment_plan(
    store: FactoryControlStore,
    compiler: AttachmentGenerationPlanCompiler,
) -> AttachmentGenerationPlanV2:
    run = store.get_run(RUN_ID)
    plan = _attachment_plan(run.to_ref())
    compiled = compiler.compile(plan=plan, policy=_policy(), audit=_audit())
    committed = store.commit_domain_plan(
        run_id=RUN_ID,
        expected_run_version=run.run_version,
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-attachment-plan",
    )
    assert committed.status is FactoryRunStatusV2.PLANNING
    return plan


def _open_attachment(service: PlanReviewService):
    return service.open_plan(
        run_id=RUN_ID,
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        requested_by=USER,
        idempotency_key="open-attachment-review",
        audit=_audit(),
    )


def _commit_delivery_plan(
    store: FactoryControlStore,
) -> DatasetDeliveryPlanV2:
    run = store.get_run(RUN_ID)
    plan = DatasetDeliveryPlanV2.create(
        plan_id="dataset-delivery-plan://review",
        run_ref=run.to_ref(),
        plan_version=1,
        predecessor_plan_ref=None,
        aggregate_result_ref=_ref(
            "factory-dataset-aggregate-result",
        ),
        candidate_item_refs=(_ref("evaluation-item"),),
        candidate_projection_refs=(_ref("release-projection-result"),),
        candidate_stage_head_refs=(_ref("factory-item-stage-head"),),
        rejected_binding_refs=(),
        blocked_binding_refs=(),
        output_target_ref=_ref("candidate-output-target"),
        max_files=100,
        max_total_bytes=1_000_000,
        audit=_audit(),
    )
    compiler = DatasetDeliveryPlanCompiler()
    compiled = compiler.compile(
        plan=plan,
        policy=_policy(),
        audit=_audit(),
    )
    store.commit_domain_plan(
        run_id=RUN_ID,
        expected_run_version=run.run_version,
        plan_kind=PlanKindV2.FINAL_DELIVERY,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-final-delivery-plan",
    )
    return plan


def _open_delivery(
    store: FactoryControlStore,
) -> tuple[PlanReviewService, PlanReviewViewV1]:
    service = PlanReviewService(
        store,
        clock=lambda: NOW,
    )
    opened = service.open_plan(
        run_id=RUN_ID,
        plan_kind=PlanKindV2.FINAL_DELIVERY,
        requested_by=USER,
        idempotency_key="open-final-delivery-review",
        audit=_audit(),
    )
    return service, opened


def test_adapter_registry_fails_closed_and_attachment_scope_cannot_widen() -> None:
    compiler = AttachmentGenerationPlanCompiler(_registry())
    adapter = AttachmentGenerationPlanAdapter(compiler)
    registry = ReviewablePlanAdapterRegistry((adapter,))
    base = _attachment_plan(_ref("factory-run"))
    successor = _attachment_plan(
        _ref("factory-run", "successor"),
        version=2,
        predecessor=base.to_ref(),
        works=(
            _work(
                "work-a",
                tool_ids=("attachment-execution", "shell"),
            ),
            _work("work-b", dependencies=("work-a",)),
        ),
    )

    assert (
        registry.resolve(
            PlanKindV2.ATTACHMENT_GENERATION,
            "attachment-generation-plan",
        )
        is adapter
    )
    assert registry.adapters == (adapter,)
    with pytest.raises(ReviewablePlanAdapterError, match="duplicate"):
        ReviewablePlanAdapterRegistry((adapter, adapter))
    with pytest.raises(ReviewablePlanAdapterError, match="unknown"):
        registry.resolve(
            PlanKindV2.ATTACHMENT_GENERATION,
            "dataset-build-plan",
        )
    with pytest.raises(ReviewablePlanAdapterError, match="tool"):
        adapter.validate_successor(base, successor)


def test_parse_only_adapters_fail_closed_for_invalid_material_and_edit() -> None:
    global_adapter = GlobalBuildPlanAdapter(None)
    attachment_adapter = AttachmentGenerationPlanAdapter(None)
    global_plan = _global_plan(_ref("factory-run"))
    attachment_plan = _attachment_plan(_ref("factory-run"))

    with pytest.raises(
        ReviewablePlanAdapterError,
        match="global build plan material",
    ):
        global_adapter.parse(b"{}")
    with pytest.raises(
        ReviewablePlanAdapterError,
        match="compiled global build plan",
    ):
        global_adapter.parse_compiled(b"{}")
    with pytest.raises(
        ReviewablePlanAdapterError,
        match="deterministic compiler",
    ):
        global_adapter.compile(
            global_plan,
            policy=_policy(),
            audit=_audit(),
        )
    with pytest.raises(
        ReviewablePlanAdapterError,
        match="attachment generation plan material",
    ):
        attachment_adapter.parse(b"{}")
    with pytest.raises(
        ReviewablePlanAdapterError,
        match="compiled attachment plan",
    ):
        attachment_adapter.parse_compiled(b"{}")
    with pytest.raises(
        ReviewablePlanAdapterError,
        match="deterministic compiler",
    ):
        attachment_adapter.compile(
            attachment_plan,
            policy=_policy(),
            audit=_audit(),
        )


def test_attachment_plan_approve_and_resume_reuses_shared_state_machine(
    tmp_path: Path,
) -> None:
    service, store, compiler = _setup(tmp_path)
    plan = _commit_attachment_plan(store, compiler)
    opened = _open_attachment(service)

    assert opened.request.plan_kind is PlanKindV2.ATTACHMENT_GENERATION
    assert opened.plan == plan
    assert opened.presentation.editable_paths == (
        "max_parallel_groups",
        "works",
    )
    with pytest.raises(
        PlanReviewNotResumableError,
        match="one resumed plan review",
    ):
        service.require_resumed_plan(
            run_id=RUN_ID,
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
            plan_ref=plan.to_ref(),
        )
    approved = service.decide(
        opened.request.review_request_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by=USER,
            reason_code="ATTACHMENT_PLAN_APPROVED",
            idempotency_key="approve-attachment-plan",
        ),
        audit=_audit(),
    )
    resumed = service.resume(
        opened.request.review_request_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-attachment-plan",
        ),
        audit=_audit(),
    )

    assert approved.result.state is PlanReviewStateV2.APPROVED
    assert resumed.result.state is PlanReviewStateV2.RESUMED
    assert (
        service.require_resumed_plan(
            run_id=RUN_ID,
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
            plan_ref=plan.to_ref(),
        )
        == resumed
    )
    assert store.get_run(RUN_ID).status is FactoryRunStatusV2.PLANNING
    material = store.get_domain_plan(
        RUN_ID,
        PlanKindV2.ATTACHMENT_GENERATION,
    )
    assert material.plan_ref == plan.to_ref()


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
def test_attachment_plan_uses_shared_terminal_and_pause_actions(
    tmp_path: Path,
    decision: PlanDecisionKindV2,
    state: PlanReviewStateV2,
    run_status: FactoryRunStatusV2,
) -> None:
    service, store, compiler = _setup(tmp_path)
    _commit_attachment_plan(store, compiler)
    opened = _open_attachment(service)

    decided = service.decide(
        opened.request.review_request_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=decision,
            decided_by=USER,
            reason_code=f"{decision.value}_ATTACHMENT_PLAN",
            idempotency_key=(f"attachment-{decision.value.casefold()}"),
        ),
        audit=_audit(),
    )

    assert decided.result.state is state
    assert store.get_run(RUN_ID).status is run_status
    if decision is not PlanDecisionKindV2.REJECT:
        with pytest.raises(PlanReviewNotResumableError):
            service.resume(
                opened.request.review_request_id,
                PlanReviewResumeSubmissionV1(
                    expected_plan_version=1,
                    resumed_by=USER,
                    idempotency_key=(f"resume-{decision.value.casefold()}"),
                ),
                audit=_audit(),
            )


def test_attachment_plan_two_clients_commit_one_authority(
    tmp_path: Path,
) -> None:
    service, store, compiler = _setup(tmp_path)
    _commit_attachment_plan(store, compiler)
    opened = _open_attachment(service)

    def decide(decision: PlanDecisionKindV2) -> str:
        try:
            return service.decide(
                opened.request.review_request_id,
                PlanReviewDecisionSubmissionV1(
                    expected_plan_version=1,
                    decision=decision,
                    decided_by=USER,
                    reason_code=f"{decision.value}_RACE",
                    idempotency_key=(f"attachment-race-{decision.value.casefold()}"),
                ),
                audit=_audit(decision.value),
            ).result.state.value
        except PlanReviewConflictError:
            return "CONFLICT"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(
            executor.map(
                decide,
                (
                    PlanDecisionKindV2.APPROVE,
                    PlanDecisionKindV2.REJECT,
                ),
            )
        )

    assert outcomes.count("CONFLICT") == 1
    assert service.show(opened.request.review_request_id).decision is not None


def test_attachment_work_edit_has_compiler_derived_invalidation(
    tmp_path: Path,
) -> None:
    service, store, compiler = _setup(tmp_path)
    base = _commit_attachment_plan(store, compiler)
    opened = _open_attachment(service)
    reduced_work = base.works[0].model_copy(update={"max_model_requests": 1})
    successor = _attachment_plan(
        store.get_run(RUN_ID).to_ref(),
        version=2,
        predecessor=base.to_ref(),
        works=(reduced_work, base.works[1]),
    )

    edited = service.edit_plan(
        opened.request.review_request_id,
        AttachmentPlanReviewEditSubmissionV1(
            expected_plan_version=1,
            edited_plan=successor,
            changed_paths=("works",),
            decided_by=USER,
            reason_code="REDUCE_WORK_MODEL_REQUESTS",
            idempotency_key="edit-attachment-work",
        ),
        audit=_audit(),
    )

    assert edited.revision is not None
    assert edited.revision.invalidated_object_refs == (base.works[0].artifact_group_ref,)


def test_attachment_edit_promotes_domain_head_and_rebuilds_without_global_drift(
    tmp_path: Path,
) -> None:
    service, store, compiler = _setup(tmp_path)
    base = _commit_attachment_plan(store, compiler)
    global_before = store.get_plan(RUN_ID)
    opened = _open_attachment(service)
    waiting_run = store.get_run(RUN_ID)
    successor = _attachment_plan(
        waiting_run.to_ref(),
        version=2,
        predecessor=base.to_ref(),
        max_parallel_groups=1,
    )
    edited = service.edit_plan(
        opened.request.review_request_id,
        AttachmentPlanReviewEditSubmissionV1(
            expected_plan_version=1,
            edited_plan=successor,
            changed_paths=("max_parallel_groups",),
            decided_by=USER,
            reason_code="REDUCE_ATTACHMENT_PARALLELISM",
            idempotency_key="edit-attachment-plan",
        ),
        audit=_audit(),
    )

    assert edited.result.state is PlanReviewStateV2.REVISION_REQUESTED
    assert edited.plan == successor
    assert (
        store.get_domain_plan(
            RUN_ID,
            PlanKindV2.ATTACHMENT_GENERATION,
        ).plan_ref
        == base.to_ref()
    )
    with pytest.raises(PlanReviewConflictError, match=r"stale|contiguous"):
        service.edit_plan(
            opened.request.review_request_id,
            AttachmentPlanReviewEditSubmissionV1(
                expected_plan_version=2,
                edited_plan=successor,
                changed_paths=("max_parallel_groups",),
                decided_by=USER,
                reason_code="STALE_ATTACHMENT_EDIT",
                idempotency_key="stale-attachment-edit",
            ),
            audit=_audit(),
        )

    service.resume(
        opened.request.review_request_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-edited-attachment-plan",
        ),
        audit=_audit(),
    )
    current = store.get_domain_plan(
        RUN_ID,
        PlanKindV2.ATTACHMENT_GENERATION,
    )
    assert current.plan_ref == successor.to_ref()
    assert store.get_plan(RUN_ID) == global_before

    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM domain_plan_current_heads")
    rebuild = store.rebuild_current_heads()
    assert rebuild == FactoryControlHeadRebuild(
        run_head_count=1,
        plan_head_count=1,
        domain_plan_head_count=1,
    )
    assert (
        store.get_domain_plan(
            RUN_ID,
            PlanKindV2.ATTACHMENT_GENERATION,
        ).plan_ref
        == successor.to_ref()
    )


def test_final_delivery_edit_narrows_limits_and_invalidates_base(
    tmp_path: Path,
) -> None:
    _service, store, _attachment_compiler = _setup(tmp_path)
    compiler = DatasetDeliveryPlanCompiler()
    run = store.get_run(RUN_ID)
    base = DatasetDeliveryPlanV2.create(
        plan_id="dataset-delivery-plan://review",
        run_ref=run.to_ref(),
        plan_version=1,
        predecessor_plan_ref=None,
        aggregate_result_ref=_ref(
            "factory-dataset-aggregate-result",
        ),
        candidate_item_refs=(_ref("evaluation-item"),),
        candidate_projection_refs=(_ref("release-projection-result"),),
        candidate_stage_head_refs=(_ref("factory-item-stage-head"),),
        rejected_binding_refs=(),
        blocked_binding_refs=(),
        output_target_ref=_ref("candidate-output-target"),
        max_files=100,
        max_total_bytes=1_000_000,
        audit=_audit(),
    )
    compiled = compiler.compile(
        plan=base,
        policy=_policy(),
        audit=_audit(),
    )
    store.commit_domain_plan(
        run_id=RUN_ID,
        expected_run_version=run.run_version,
        plan_kind=PlanKindV2.FINAL_DELIVERY,
        plan=base,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-final-delivery-plan",
    )
    service = PlanReviewService(
        store,
        clock=lambda: NOW,
    )
    opened = service.open_plan(
        run_id=RUN_ID,
        plan_kind=PlanKindV2.FINAL_DELIVERY,
        requested_by=USER,
        idempotency_key="open-final-delivery-review",
        audit=_audit(),
    )
    waiting = store.get_run(RUN_ID)
    successor = DatasetDeliveryPlanV2.create(
        plan_id=base.plan_id,
        run_ref=waiting.to_ref(),
        plan_version=2,
        predecessor_plan_ref=base.to_ref(),
        aggregate_result_ref=base.aggregate_result_ref,
        candidate_item_refs=base.candidate_item_refs,
        candidate_projection_refs=(base.candidate_projection_refs),
        candidate_stage_head_refs=base.candidate_stage_head_refs,
        rejected_binding_refs=base.rejected_binding_refs,
        blocked_binding_refs=base.blocked_binding_refs,
        output_target_ref=base.output_target_ref,
        max_files=50,
        max_total_bytes=500_000,
        audit=_audit(),
    )

    edited = service.edit_plan(
        opened.request.review_request_id,
        DatasetDeliveryPlanReviewEditSubmissionV1(
            expected_plan_version=1,
            edited_plan=successor,
            changed_paths=(
                "max_files",
                "max_total_bytes",
            ),
            decided_by=USER,
            reason_code="NARROW_DELIVERY_LIMITS",
            idempotency_key="edit-final-delivery-plan",
        ),
        audit=_audit(),
    )

    assert edited.result.state is (PlanReviewStateV2.REVISION_REQUESTED)
    assert edited.revision is not None
    assert edited.revision.invalidated_object_refs == (base.to_ref(),)
    assert (
        store.get_domain_plan(
            RUN_ID,
            PlanKindV2.FINAL_DELIVERY,
        ).plan_ref
        == base.to_ref()
    )
    service.resume(
        opened.request.review_request_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-final-delivery-edit",
        ),
        audit=_audit(),
    )
    assert (
        store.get_domain_plan(
            RUN_ID,
            PlanKindV2.FINAL_DELIVERY,
        ).plan_ref
        == successor.to_ref()
    )

    widened = successor.model_copy(
        update={
            "max_files": 101,
        }
    )
    with pytest.raises(
        ReviewablePlanAdapterError,
        match="widens",
    ):
        DatasetDeliveryPlanAdapter(compiler).validate_successor(
            base,
            widened,
        )


@pytest.mark.parametrize(
    "fault_point",
    (
        FactoryControlStoreFaultPoint.AFTER_DOMAIN_PLAN,
        FactoryControlStoreFaultPoint.AFTER_DOMAIN_PLAN_PROMOTION,
        FactoryControlStoreFaultPoint.AFTER_DOMAIN_PLAN_HEAD,
        FactoryControlStoreFaultPoint.AFTER_RUN,
        FactoryControlStoreFaultPoint.AFTER_RUN_HEAD,
        FactoryControlStoreFaultPoint.AFTER_OUTBOX,
        FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY,
    ),
)
def test_domain_plan_commit_faults_roll_back_all_authority(
    tmp_path: Path,
    fault_point: FactoryControlStoreFaultPoint,
) -> None:
    _, store, compiler = _setup(tmp_path)
    run = store.get_run(RUN_ID)
    history = store.get_run_history(RUN_ID)
    plan = _attachment_plan(run.to_ref())
    compiled = compiler.compile(
        plan=plan,
        policy=_policy(),
        audit=_audit(),
    )
    faulted = FactoryControlStore(
        store.path,
        fault_injector=StaticFactoryControlStoreFaultInjector(crash_points=frozenset({fault_point})),
    )

    with pytest.raises(
        FactoryControlInjectedCrash,
        match=fault_point.value,
    ):
        faulted.commit_domain_plan(
            run_id=RUN_ID,
            expected_run_version=run.run_version,
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
            plan=plan,
            compiled_plan=compiled,
            audit=_audit(),
            idempotency_key="fault-domain-plan",
            material_ref=_ref(
                "attachment-preparation-material",
                fault_point.value,
            ),
        )

    assert faulted.get_run_history(RUN_ID) == history
    with sqlite3.connect(store.path) as connection:
        for table in (
            "domain_plan_materials",
            "domain_plan_material_refs",
            "domain_plan_promotions",
            "domain_plan_current_heads",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)


def test_domain_plan_commit_replays_and_changed_same_key_conflicts(
    tmp_path: Path,
) -> None:
    _, store, compiler = _setup(tmp_path)
    run = store.get_run(RUN_ID)
    plan = _attachment_plan(run.to_ref())
    compiled = compiler.compile(
        plan=plan,
        policy=_policy(),
        audit=_audit(),
    )
    material_ref = _ref(
        "attachment-preparation-material",
        "domain-plan-replay",
    )
    committed = store.commit_domain_plan(
        run_id=RUN_ID,
        expected_run_version=run.run_version,
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="domain-plan-replay",
        material_ref=material_ref,
    )
    replay = store.commit_domain_plan(
        run_id=RUN_ID,
        expected_run_version=run.run_version,
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="domain-plan-replay",
        material_ref=material_ref,
    )

    assert replay == committed
    assert (
        store.get_domain_plan_material_ref(
            run_id=RUN_ID,
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        )
        == material_ref
    )
    with pytest.raises(
        FactoryControlConflictError,
        match="idempotency",
    ):
        store.commit_domain_plan(
            run_id=RUN_ID,
            expected_run_version=run.run_version,
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
            plan=plan,
            compiled_plan=compiled,
            audit=_audit(),
            idempotency_key="domain-plan-replay",
            material_ref=_ref(
                "attachment-preparation-material",
                "changed",
            ),
        )
    changed = _attachment_plan(
        run.to_ref(),
        max_parallel_groups=1,
    )
    changed_compiled = compiler.compile(
        plan=changed,
        policy=_policy(),
        audit=_audit(),
    )
    with pytest.raises(
        FactoryControlConflictError,
        match="idempotency",
    ):
        store.commit_domain_plan(
            run_id=RUN_ID,
            expected_run_version=run.run_version,
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
            plan=changed,
            compiled_plan=changed_compiled,
            audit=_audit(),
            idempotency_key="domain-plan-replay",
        )


def test_domain_revision_corruption_never_falls_back_to_review_material(
    tmp_path: Path,
) -> None:
    service, store, compiler = _setup(tmp_path)
    base = _commit_attachment_plan(store, compiler)
    opened = _open_attachment(service)
    successor = _attachment_plan(
        store.get_run(RUN_ID).to_ref(),
        version=2,
        predecessor=base.to_ref(),
        max_parallel_groups=1,
    )
    service.edit_plan(
        opened.request.review_request_id,
        AttachmentPlanReviewEditSubmissionV1(
            expected_plan_version=1,
            edited_plan=successor,
            changed_paths=("max_parallel_groups",),
            decided_by=USER,
            reason_code="REDUCE_ATTACHMENT_PARALLELISM",
            idempotency_key="edit-before-corruption",
        ),
        audit=_audit(),
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE domain_plan_materials
            SET record_json = '{}'
            WHERE plan_object_id = ?
            """,
            (successor.object_id,),
        )

    with pytest.raises(
        PlanReviewIntegrityError,
        match="domain plan material",
    ):
        service.show(opened.request.review_request_id)


def test_attachment_plan_cli_edit_and_restart_resume(
    tmp_path: Path,
) -> None:
    service, store, compiler = _setup(tmp_path)
    base = _commit_attachment_plan(store, compiler)
    opened = _open_attachment(service)
    registry = _registry()
    registry_path = tmp_path / "agent-registry.json"
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
    payload = PlanReviewEditPayloadV1(
        expected_plan_version=1,
        edited_plan=AttachmentPlanEditableFieldsV1(
            works=base.works,
            max_parallel_groups=1,
        ),
        changed_paths=("max_parallel_groups",),
        invalidated_object_refs=(),
        decided_by=USER,
        reason_code="REDUCE_ATTACHMENT_PARALLELISM",
        idempotency_key="edit-attachment-cli",
    )
    payload_path = tmp_path / "attachment-edit.json"
    payload_path.write_text(
        payload.model_dump_json(),
        encoding="utf-8",
    )
    runner = CliRunner()

    edited = runner.invoke(
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
            str(payload_path),
            "--registry",
            str(registry_path),
        ],
    )
    resumed = runner.invoke(
        app,
        [
            "agent",
            "plan",
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
            "resume-attachment-cli",
        ],
    )

    assert edited.exit_code == 0, edited.stderr
    assert json.loads(edited.stdout)["plan"]["max_parallel_groups"] == 1
    assert resumed.exit_code == 0, resumed.stderr
    assert json.loads(resumed.stdout)["result"]["state"] == "RESUMED"


def test_final_delivery_cli_edit_preserves_fixed_authority_and_resumes(
    tmp_path: Path,
) -> None:
    _service, store, _compiler = _setup(tmp_path)
    base = _commit_delivery_plan(store)
    _delivery_service, opened = _open_delivery(store)
    registry = _registry()
    registry_path = tmp_path / "agent-registry.json"
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
    payload = PlanReviewEditPayloadV1(
        expected_plan_version=1,
        edited_plan=DatasetDeliveryPlanEditableFieldsV1(
            max_files=50,
            max_total_bytes=500_000,
        ),
        changed_paths=(
            "max_files",
            "max_total_bytes",
        ),
        decided_by=USER,
        reason_code="NARROW_DELIVERY_LIMITS",
        idempotency_key="edit-final-delivery-cli",
    )
    payload_path = tmp_path / "delivery-edit.json"
    payload_path.write_text(
        payload.model_dump_json(),
        encoding="utf-8",
    )
    runner = CliRunner()

    edited = runner.invoke(
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
            str(payload_path),
            "--registry",
            str(registry_path),
        ],
    )
    resumed = runner.invoke(
        app,
        [
            "agent",
            "plan",
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
            "resume-final-delivery-cli",
        ],
    )

    assert edited.exit_code == 0, edited.stderr
    edited_plan = json.loads(edited.stdout)["plan"]
    assert edited_plan["max_files"] == 50
    assert edited_plan["max_total_bytes"] == 500_000
    for field_name in (
        "aggregate_result_ref",
        "candidate_item_refs",
        "candidate_projection_refs",
        "candidate_stage_head_refs",
        "rejected_binding_refs",
        "blocked_binding_refs",
        "output_target_ref",
        "production_release_allowed",
    ):
        assert edited_plan[field_name] == base.model_dump(mode="json")[field_name]
    assert resumed.exit_code == 0, resumed.stderr
    assert json.loads(resumed.stdout)["result"]["state"] == "RESUMED"
