from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_criteria_plan_review import (
    _commit as _commit_criteria_plan,
)
from test_criteria_plan_review import (
    _open as _open_criteria_plan,
)
from test_criteria_plan_review import (
    _service as _criteria_service,
)
from test_domain_plan_review import (
    USER as DOMAIN_USER,
)
from test_domain_plan_review import (
    _commit_attachment_plan,
    _commit_delivery_plan,
    _open_attachment,
    _open_delivery,
)
from test_domain_plan_review import (
    _setup as _domain_setup,
)
from test_grading_plan_review import (
    _commit as _commit_grading_plan,
)
from test_grading_plan_review import (
    _open as _open_grading_plan,
)
from test_grading_plan_review import (
    _service as _grading_service,
)
from test_plan_review import USER, _open, _plan, _service

from eval_factory.console_api import (
    AttachmentPlanEditableFieldsV1,
    CriteriaRubricPlanEditableFieldsV1,
    DatasetDeliveryPlanEditableFieldsV1,
    GlobalPlanEditableFieldsV1,
    GradingDesignPlanEditableFieldsV1,
    PlanReviewEditPayloadV1,
    create_app,
)
from eval_factory.console_api.openapi import (
    check_contract,
    contract_bytes,
    main,
    write_contract,
)
from eval_factory.contracts.agent_system_v2 import PlanReviewStateV2

ROOT = Path(__file__).resolve().parents[3]


def test_frontend_openapi_contract_is_generated_from_console_api() -> None:
    path = ROOT / "web/eval_factory_console/src/generated/openapi.json"
    assert path.read_bytes() == contract_bytes()


def test_openapi_generator_write_check_drift_and_cli(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "openapi.json"
    write_contract(path)
    check_contract(path)
    path.write_text("drift", encoding="utf-8")
    with pytest.raises(SystemExit, match="contract drift"):
        check_contract(path)

    monkeypatch.setattr(sys, "argv", ["openapi", str(path), "--write"])
    main()
    monkeypatch.setattr(sys, "argv", ["openapi", str(path), "--check"])
    main()


def test_console_api_reads_the_same_plan_review_projection(tmp_path) -> None:
    service, _ = _service(tmp_path)
    opened = _open(service)
    client = TestClient(create_app(service))

    contract = client.get("/api/plan-reviews/contract")
    page = client.get("/api/plan-reviews")
    shown = client.get(
        "/api/plan-reviews/show",
        params={"review_id": opened.request.review_request_id},
    )
    exported = client.get(
        "/api/plan-reviews/export",
        params={"review_id": opened.request.review_request_id},
    )

    assert contract.status_code == page.status_code == shown.status_code == 200
    assert "APPROVE" in contract.json()["decision_actions"]
    assert page.json()["total"] == 1
    assert shown.json() == exported.json()
    assert shown.json() == opened.model_dump(mode="json")


def test_console_api_approve_and_resume_share_service_authority(tmp_path) -> None:
    service, _ = _service(tmp_path)
    opened = _open(service)
    client = TestClient(create_app(service))
    review_id = opened.request.review_request_id
    headers = {"X-Eval-Factory-Principal": USER}

    approved = client.post(
        "/api/plan-reviews/decision",
        headers=headers,
        json={
            "review_id": review_id,
            "submission": {
                "expected_plan_version": 1,
                "decision": "APPROVE",
                "decided_by": USER,
                "reason_code": "APPROVE_BY_WEB",
                "idempotency_key": "approve-web",
            },
        },
    )
    resumed = client.post(
        "/api/plan-reviews/resume",
        headers=headers,
        json={
            "review_id": review_id,
            "submission": {
                "expected_plan_version": 1,
                "resumed_by": USER,
                "idempotency_key": "resume-web",
            },
        },
    )

    assert approved.status_code == resumed.status_code == 200
    assert approved.json()["result"]["state"] == PlanReviewStateV2.APPROVED
    assert resumed.json()["result"]["state"] == PlanReviewStateV2.RESUMED


def test_console_api_edit_and_stale_conflict_are_typed(tmp_path) -> None:
    service, store = _service(tmp_path)
    opened = _open(service)
    client = TestClient(create_app(service))
    edited_plan = _plan(
        store.get_run(opened.run_id).to_ref(),
        version=2,
        predecessor=opened.plan.to_ref(),
        extra_goal="Expose route refs in the delivery audit.",
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
        reason_code="EDIT_BY_WEB",
        idempotency_key="edit-web",
    )
    command = {
        "review_id": opened.request.review_request_id,
        "submission": submission.model_dump(mode="json"),
    }

    edited = client.post(
        "/api/plan-reviews/edit",
        headers={"X-Eval-Factory-Principal": USER},
        json=command,
    )
    stale = client.post(
        "/api/plan-reviews/decision",
        headers={"X-Eval-Factory-Principal": USER},
        json={
            "review_id": opened.request.review_request_id,
            "submission": {
                "expected_plan_version": 1,
                "decision": "APPROVE",
                "decided_by": USER,
                "reason_code": "STALE_APPROVAL",
                "idempotency_key": "stale-web",
            },
        },
    )
    wrong_principal = client.post(
        "/api/plan-reviews/edit",
        headers={"X-Eval-Factory-Principal": "user://other"},
        json=command,
    )
    invalid = client.post(
        "/api/plan-reviews/edit",
        headers={"X-Eval-Factory-Principal": USER},
        content=b"{}",
    )

    assert edited.status_code == 200
    assert edited.json()["result"]["state"] == "REVISION_REQUESTED"
    assert stale.status_code == 409
    assert stale.json()["error_code"] == "PLAN_REVIEW_CONFLICT"
    assert wrong_principal.status_code == 403
    assert wrong_principal.json()["error_code"] == "PRINCIPAL_MISMATCH"
    assert invalid.status_code == 422
    assert invalid.json()["error_code"] == "INVALID_COMMAND"


def test_console_api_edits_and_resumes_attachment_plan(
    tmp_path: Path,
) -> None:
    service, store, compiler = _domain_setup(tmp_path)
    base = _commit_attachment_plan(store, compiler)
    opened = _open_attachment(service)
    client = TestClient(create_app(service))
    payload = PlanReviewEditPayloadV1(
        expected_plan_version=1,
        edited_plan=AttachmentPlanEditableFieldsV1(
            works=base.works,
            max_parallel_groups=1,
        ),
        changed_paths=("max_parallel_groups",),
        invalidated_object_refs=(),
        decided_by=DOMAIN_USER,
        reason_code="REDUCE_ATTACHMENT_PARALLELISM",
        idempotency_key="edit-attachment-web",
    )

    edited = client.post(
        "/api/plan-reviews/edit",
        headers={
            "X-Eval-Factory-Principal": DOMAIN_USER,
        },
        json={
            "review_id": opened.request.review_request_id,
            "submission": payload.model_dump(mode="json"),
        },
    )
    resumed = client.post(
        "/api/plan-reviews/resume",
        headers={
            "X-Eval-Factory-Principal": DOMAIN_USER,
        },
        json={
            "review_id": opened.request.review_request_id,
            "submission": {
                "expected_plan_version": 1,
                "resumed_by": DOMAIN_USER,
                "idempotency_key": "resume-attachment-web",
            },
        },
    )

    assert edited.status_code == 200, edited.json()
    assert edited.json()["plan"]["schema_version"] == "eval-factory/attachment-generation-plan/v2"
    assert edited.json()["plan"]["max_parallel_groups"] == 1
    assert resumed.status_code == 200, resumed.json()
    assert resumed.json()["result"]["state"] == "RESUMED"


def test_console_api_edits_and_resumes_criteria_rubric_plan(
    tmp_path: Path,
) -> None:
    service, compiler = _criteria_service(tmp_path)
    base = _commit_criteria_plan(service, compiler)
    opened = _open_criteria_plan(service)
    client = TestClient(create_app(service))
    changed_goal = base.criterion_goals[0].model_copy(
        update={"goal_summary": "Judge the revised response goal."}
    )
    payload = PlanReviewEditPayloadV1(
        expected_plan_version=1,
        edited_plan=CriteriaRubricPlanEditableFieldsV1(
            criterion_goals=(changed_goal,),
            allowed_evaluator_binding_ids=(base.allowed_evaluator_binding_ids),
            allowed_reference_modes=base.allowed_reference_modes,
            selected_reference_mode=base.selected_reference_mode,
            acceptance_check_refs=base.acceptance_check_refs,
            max_attempts=base.max_attempts,
            max_model_requests=base.max_model_requests,
            max_model_tokens=8_000,
            max_cost_micro_usd=base.max_cost_micro_usd,
        ),
        changed_paths=("criterion_goals", "max_model_tokens"),
        decided_by=DOMAIN_USER,
        reason_code="REFINE_CRITERIA_RUBRIC",
        idempotency_key="edit-criteria-rubric-web",
    )

    edited = client.post(
        "/api/plan-reviews/edit",
        headers={"X-Eval-Factory-Principal": DOMAIN_USER},
        json={
            "review_id": opened.request.review_request_id,
            "submission": payload.model_dump(mode="json"),
        },
    )
    resumed = client.post(
        "/api/plan-reviews/resume",
        headers={"X-Eval-Factory-Principal": DOMAIN_USER},
        json={
            "review_id": opened.request.review_request_id,
            "submission": {
                "expected_plan_version": 1,
                "resumed_by": DOMAIN_USER,
                "idempotency_key": "resume-criteria-rubric-web",
            },
        },
    )

    assert edited.status_code == 200, edited.json()
    assert edited.json()["plan"]["schema_version"] == "eval-factory/criteria-rubric-plan/v2"
    assert edited.json()["plan"]["max_model_tokens"] == 8_000
    assert resumed.status_code == 200, resumed.json()
    assert resumed.json()["result"]["state"] == "RESUMED"


def test_console_api_edits_and_resumes_grading_design_plan(
    tmp_path: Path,
) -> None:
    service, compiler = _grading_service(tmp_path)
    base = _commit_grading_plan(service, compiler)
    opened = _open_grading_plan(service)
    client = TestClient(create_app(service))
    payload = PlanReviewEditPayloadV1(
        expected_plan_version=1,
        edited_plan=GradingDesignPlanEditableFieldsV1(
            judge_tasks=base.judge_tasks,
            aggregation_mode=base.aggregation_mode,
            passing_score_basis_points=7500,
            minimum_confidence_basis_points=8500,
            escalate_on_reference_unavailable=(base.escalate_on_reference_unavailable),
            allowed_judge_model_profile_refs=(base.allowed_judge_model_profile_refs[1:]),
            acceptance_check_refs=(base.acceptance_check_refs),
            max_attempts=base.max_attempts,
            max_model_requests=base.max_model_requests,
            max_model_tokens=8_000,
            max_cost_micro_usd=base.max_cost_micro_usd,
        ),
        changed_paths=(
            "allowed_judge_model_profile_refs",
            "max_model_tokens",
            "minimum_confidence_basis_points",
            "passing_score_basis_points",
        ),
        decided_by=DOMAIN_USER,
        reason_code="REFINE_GRADING_DESIGN",
        idempotency_key="edit-grading-design-web",
    )

    edited = client.post(
        "/api/plan-reviews/edit",
        headers={"X-Eval-Factory-Principal": DOMAIN_USER},
        json={
            "review_id": opened.request.review_request_id,
            "submission": payload.model_dump(mode="json"),
        },
    )
    resumed = client.post(
        "/api/plan-reviews/resume",
        headers={"X-Eval-Factory-Principal": DOMAIN_USER},
        json={
            "review_id": opened.request.review_request_id,
            "submission": {
                "expected_plan_version": 1,
                "resumed_by": DOMAIN_USER,
                "idempotency_key": ("resume-grading-design-web"),
            },
        },
    )

    assert edited.status_code == 200, edited.json()
    edited_plan = edited.json()["plan"]
    assert edited_plan["schema_version"] == "eval-factory/grading-design-plan/v2"
    assert edited_plan["max_model_tokens"] == 8_000
    for field_name in (
        "criteria_rubric_result_ref",
        "rubric_set_ref",
        "evaluator_spec_ref",
        "reference_policy_ref",
        "tool_policy_ref",
        "generator_model_profile_ref",
        "judge_input_schema_ref",
        "judge_output_schema_ref",
        "required_output_fields",
        "judge_prompt_template_ref",
        "model_policy_ref",
    ):
        assert edited_plan[field_name] == base.model_dump(mode="json")[field_name]
    assert resumed.status_code == 200, resumed.json()
    assert resumed.json()["result"]["state"] == "RESUMED"


def test_console_api_edits_and_resumes_final_delivery_plan(
    tmp_path: Path,
) -> None:
    _service, store, _compiler = _domain_setup(tmp_path)
    base = _commit_delivery_plan(store)
    service, opened = _open_delivery(store)
    client = TestClient(create_app(service))
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
        decided_by=DOMAIN_USER,
        reason_code="NARROW_DELIVERY_LIMITS",
        idempotency_key="edit-final-delivery-web",
    )

    edited = client.post(
        "/api/plan-reviews/edit",
        headers={"X-Eval-Factory-Principal": DOMAIN_USER},
        json={
            "review_id": opened.request.review_request_id,
            "submission": payload.model_dump(mode="json"),
        },
    )
    resumed = client.post(
        "/api/plan-reviews/resume",
        headers={"X-Eval-Factory-Principal": DOMAIN_USER},
        json={
            "review_id": opened.request.review_request_id,
            "submission": {
                "expected_plan_version": 1,
                "resumed_by": DOMAIN_USER,
                "idempotency_key": "resume-final-delivery-web",
            },
        },
    )

    assert edited.status_code == 200, edited.json()
    edited_plan = edited.json()["plan"]
    assert edited_plan["max_files"] == 50
    assert edited_plan["max_total_bytes"] == 500_000
    for field_name in (
        "aggregate_result_ref",
        "candidate_item_refs",
        "candidate_projection_refs",
        "rejected_binding_refs",
        "blocked_binding_refs",
        "output_target_ref",
        "production_release_allowed",
    ):
        assert edited_plan[field_name] == base.model_dump(mode="json")[field_name]
    assert resumed.status_code == 200, resumed.json()
    assert resumed.json()["result"]["state"] == "RESUMED"
