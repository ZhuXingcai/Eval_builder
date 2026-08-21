from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from eval_factory.approval.interaction_models import (
    TrustedAuthenticatedUserContextV2,
    UserCheckpointSourceContextV2,
)
from eval_factory.cli import app
from eval_factory.contracts.approval import UserDecision
from eval_factory.contracts.checkpoint_interaction_v2 import (
    UserCheckpointDecisionSubmissionV2,
    UserCheckpointInteractionPolicyV2,
    user_checkpoint_interaction_v2_ref,
)
from eval_factory.contracts.orchestration import JobStatus
from eval_factory.orchestration import JobStore

ROOT = Path(__file__).resolve().parents[2]


def _helpers() -> tuple[object, object]:
    unit_tests = str(ROOT / "tests/eval_factory/unit")
    if unit_tests not in sys.path:
        sys.path.insert(0, unit_tests)
    return (
        importlib.import_module("test_user_approval_requests"),
        importlib.import_module("test_user_decisions"),
    )


def _invoke(arguments: list[str]) -> tuple[object, dict[str, object]]:
    result = CliRunner().invoke(app, arguments)
    stream = result.stdout if result.exit_code == 0 else result.stderr
    return result, json.loads(stream)


def test_checkpoint_cli_open_show_decide_and_resume(
    tmp_path: Path,
) -> None:
    requests, decisions = _helpers()
    approval_policy, source, compilation, _ = decisions._label_request()
    job_spec = requests._job_spec(approval_policy)
    generation_policy = requests._generation_policy()
    handling_policy = decisions._handling_policy()
    audit = decisions._audit()
    authentication = decisions._authentication()
    context = UserCheckpointSourceContextV2.create(
        job_spec=job_spec,
        approval_policy=approval_policy,
        generation_policy=generation_policy,
        handling_policy=handling_policy,
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )
    interaction_policy = UserCheckpointInteractionPolicyV2.create(
        max_interactions_per_job=100,
        max_requests_per_interaction=10,
        max_page_size=100,
        max_presentation_bytes=1_000_000,
        max_source_context_bytes=2_000_000,
        max_ready_work_refs=100,
        audit=audit,
    )
    store_path = tmp_path / "factory.sqlite3"
    checkpoint_store = tmp_path / "checkpoint-store"
    store = JobStore(store_path)
    store.create_job(job_spec)
    running = store.transition_job(
        job_spec.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-cli",
    )
    context_path = tmp_path / "context.json"
    policy_path = tmp_path / "policy.json"
    context_path.write_bytes(context.canonical_json())
    policy_path.write_bytes(interaction_policy.canonical_json())

    opened, opened_payload = _invoke(
        [
            "pipeline",
            "checkpoint",
            "open",
            "--job-store",
            str(store_path),
            "--checkpoint-store",
            str(checkpoint_store),
            "--context",
            str(context_path),
            "--policy",
            str(policy_path),
            "--expected-job-version",
            str(running.row_version),
            "--idempotency-key",
            "open-r7-07-cli",
        ]
    )
    assert opened.exit_code == 0, opened.output
    assert opened_payload["outcome"] == "PAUSED"
    interaction_payload = opened_payload["interaction"]
    assert isinstance(interaction_payload, dict)
    interaction_id = str(interaction_payload["interaction_id"])

    shown, shown_payload = _invoke(
        [
            "pipeline",
            "checkpoint",
            "show",
            "--job-store",
            str(store_path),
            "--checkpoint-store",
            str(checkpoint_store),
            "--interaction",
            interaction_id,
        ]
    )
    assert shown.exit_code == 0, shown.output
    assert shown_payload["request"]["checkpoint"] == "LABEL_PLAN"
    assert shown_payload["presentation"]["label_plan"] is not None
    assert "label_specs" not in shown.output
    assert "task_rewrite_plan_versions" not in shown.output

    interaction = JobStore(store_path)
    from eval_factory.approval import (
        UserCheckpointInteractionService,
        UserCheckpointMaterialStore,
    )

    interaction_service = UserCheckpointInteractionService(
        interaction,
        UserCheckpointMaterialStore(
            checkpoint_store,
            max_source_context_bytes=256 * 1024 * 1024,
            max_presentation_bytes=64 * 1024 * 1024,
        ),
    )
    current = interaction_service.get_current_interaction(job_spec.job_id)
    submission = UserCheckpointDecisionSubmissionV2(
        interaction_ref=user_checkpoint_interaction_v2_ref(current),
        decision=UserDecision.ACCEPT,
        reason="Accept from CLI.",
        idempotency_key="decide-r7-07-cli",
    )
    submission_path = tmp_path / "submission.json"
    authentication_path = tmp_path / "authentication.json"
    submission_path.write_bytes(submission.canonical_json())
    authentication_path.write_bytes(
        TrustedAuthenticatedUserContextV2(
            authenticated_user=authentication.authenticated_user,
            authentication_context_ref=(authentication.authentication_context_ref),
        ).canonical_json()
    )

    decided, decided_payload = _invoke(
        [
            "pipeline",
            "checkpoint",
            "decide",
            "--job-store",
            str(store_path),
            "--checkpoint-store",
            str(checkpoint_store),
            "--interaction",
            interaction_id,
            "--submission",
            str(submission_path),
            "--authentication",
            str(authentication_path),
            "--expected-job-version",
            str(opened_payload["job_version"]),
        ]
    )
    assert decided.exit_code == 0, decided.output
    assert decided_payload["resume_disposition"] == "DIRECT"

    resumed, resumed_payload = _invoke(
        [
            "pipeline",
            "checkpoint",
            "resume",
            "--job-store",
            str(store_path),
            "--checkpoint-store",
            str(checkpoint_store),
            "--job",
            job_spec.job_id,
            "--expected-job-version",
            str(decided_payload["job_version"]),
            "--idempotency-key",
            "resume-r7-07-cli",
        ]
    )
    assert resumed.exit_code == 0, resumed.output
    assert resumed_payload["job_status"] == "RUNNING"
    assert resumed_payload["interaction"]["state"] == "RESUMED"


def test_checkpoint_cli_returns_closed_errors_for_pending_and_bad_inputs(
    tmp_path: Path,
) -> None:
    requests, decisions = _helpers()
    approval_policy, source, compilation, _ = decisions._label_request()
    job_spec = requests._job_spec(approval_policy)
    context = UserCheckpointSourceContextV2.create(
        job_spec=job_spec,
        approval_policy=approval_policy,
        generation_policy=requests._generation_policy(),
        handling_policy=decisions._handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )
    store_path = tmp_path / "errors.sqlite3"
    checkpoint_store = tmp_path / "checkpoint-store"
    store = JobStore(store_path)
    store.create_job(job_spec)
    running = store.transition_job(
        job_spec.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-cli-errors",
    )
    from eval_factory.approval import (
        UserCheckpointInteractionService,
        UserCheckpointMaterialStore,
    )

    service = UserCheckpointInteractionService(
        store,
        UserCheckpointMaterialStore(
            checkpoint_store,
            max_source_context_bytes=256 * 1024 * 1024,
            max_presentation_bytes=64 * 1024 * 1024,
        ),
    )
    opened = service.open_checkpoint(
        source_context=context,
        policy=UserCheckpointInteractionPolicyV2.create(
            max_interactions_per_job=100,
            max_requests_per_interaction=10,
            max_page_size=100,
            max_presentation_bytes=1_000_000,
            max_source_context_bytes=2_000_000,
            max_ready_work_refs=100,
            audit=decisions._audit(),
        ),
        expected_job_version=running.row_version,
        idempotency_key="open-r7-07-cli-errors",
        audit=decisions._audit(),
    )
    assert opened.interaction is not None

    generic_resume, generic_payload = _invoke(
        [
            "pipeline",
            "resume",
            "--job-store",
            str(store_path),
            "--job",
            job_spec.job_id,
            "--expected-job-version",
            str(opened.job_version),
            "--idempotency-key",
            "generic-resume-r7-07-cli-errors",
        ]
    )
    assert generic_resume.exit_code == 2
    assert generic_payload["error_code"] == "CHECKPOINT_REQUIRED"

    checkpoint_resume, checkpoint_payload = _invoke(
        [
            "pipeline",
            "checkpoint",
            "resume",
            "--job-store",
            str(store_path),
            "--checkpoint-store",
            str(checkpoint_store),
            "--job",
            job_spec.job_id,
            "--expected-job-version",
            str(opened.job_version),
            "--idempotency-key",
            "checkpoint-resume-r7-07-cli-errors",
        ]
    )
    assert checkpoint_resume.exit_code == 2
    assert checkpoint_payload["error_code"] == "CHECKPOINT_NOT_RESUMABLE"

    submission = UserCheckpointDecisionSubmissionV2(
        interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
        decision=UserDecision.ACCEPT,
        reason="Wrong user.",
        idempotency_key="decide-r7-07-cli-wrong-user",
    )
    authentication = decisions._authentication("another-user")
    submission_path = tmp_path / "wrong-submission.json"
    authentication_path = tmp_path / "wrong-authentication.json"
    submission_path.write_bytes(submission.canonical_json())
    authentication_path.write_bytes(
        TrustedAuthenticatedUserContextV2(
            authenticated_user=authentication.authenticated_user,
            authentication_context_ref=(authentication.authentication_context_ref),
        ).canonical_json()
    )
    decided, decided_payload = _invoke(
        [
            "pipeline",
            "checkpoint",
            "decide",
            "--job-store",
            str(store_path),
            "--checkpoint-store",
            str(checkpoint_store),
            "--interaction",
            opened.interaction.interaction_id,
            "--submission",
            str(submission_path),
            "--authentication",
            str(authentication_path),
            "--expected-job-version",
            str(opened.job_version),
        ]
    )
    assert decided.exit_code == 2
    assert decided_payload["error_code"] == "AUTHENTICATION_MISMATCH"

    missing, missing_payload = _invoke(
        [
            "pipeline",
            "checkpoint",
            "show",
            "--job-store",
            str(store_path),
            "--checkpoint-store",
            str(checkpoint_store),
            "--interaction",
            "user-checkpoint-interaction://missing",
        ]
    )
    assert missing.exit_code == 2
    assert missing_payload["error_code"] == "NOT_FOUND"

    invalid_page, page_payload = _invoke(
        [
            "pipeline",
            "checkpoint",
            "list",
            "--job-store",
            str(store_path),
            "--checkpoint-store",
            str(checkpoint_store),
            "--job",
            job_spec.job_id,
            "--offset",
            "-1",
        ]
    )
    assert invalid_page.exit_code == 2
    assert page_payload["error_code"] == "POLICY_ERROR"
