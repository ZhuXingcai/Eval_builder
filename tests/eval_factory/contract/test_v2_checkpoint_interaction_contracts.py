from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.approval import UserDecision
from eval_factory.contracts.checkpoint_interaction_v2 import (
    UserCheckpointDecisionSubmissionV2,
    UserCheckpointInteractionPolicyV2,
    user_checkpoint_interaction_policy_v2_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)

NOW = datetime(2026, 8, 2, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[3]


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r7-07-contract-test",
        governing_versions=(
            VersionBinding(
                component="checkpoint-interaction",
                version="checkpoint-interaction/r7-07-v1",
            ),
        ),
        input_refs=(),
    )


def test_checkpoint_policy_is_strict_and_frozen() -> None:
    policy = UserCheckpointInteractionPolicyV2.create(
        max_interactions_per_job=100,
        max_requests_per_interaction=10,
        max_page_size=100,
        max_presentation_bytes=1_000_000,
        max_source_context_bytes=2_000_000,
        max_ready_work_refs=100,
        audit=_audit(),
    )

    assert user_checkpoint_interaction_policy_v2_ref(policy).object_type == (
        "user-checkpoint-interaction-policy"
    )
    with pytest.raises(ValidationError):
        policy.max_page_size = 1
    with pytest.raises(ValidationError):
        UserCheckpointInteractionPolicyV2.model_validate(
            {
                **policy.model_dump(mode="python"),
                "unknown": True,
            }
        )


def test_decision_submission_rejects_identity_and_authority_fields() -> None:
    interaction_ref = ObjectRef(
        object_type="user-checkpoint-interaction",
        object_id="user-checkpoint-interaction://example",
        object_version="v2",
        object_sha256="a" * 64,
    )
    payload = UserCheckpointDecisionSubmissionV2(
        interaction_ref=interaction_ref,
        decision=UserDecision.ACCEPT,
        reason="Accept.",
        idempotency_key="decision-submission-example",
    ).model_dump(mode="python")

    for field, value in {
        "authenticated_user": "forged-user",
        "authentication_context_ref": interaction_ref,
        "checkpoint": "LABEL_PLAN",
        "job_status": "RUNNING",
        "release_state": "APPROVED",
    }.items():
        with pytest.raises(ValidationError):
            UserCheckpointDecisionSubmissionV2.model_validate(
                {
                    **payload,
                    field: value,
                }
            )


def test_generated_checkpoint_schemas_are_closed_and_gold_is_safe() -> None:
    manifest = json.loads(
        (ROOT / "specs/002-eval-dataset-factory/contracts/v2/manifest.json").read_text(encoding="utf-8")
    )
    checkpoint_contracts = [
        value
        for value in manifest["contracts"]
        if value["python_type"].startswith("eval_factory.contracts.checkpoint_interaction_v2.")
    ]
    assert len(checkpoint_contracts) == 9
    assert {value["owner"] for value in checkpoint_contracts} == {"factory"}
    for contract in checkpoint_contracts:
        schema = json.loads(
            (ROOT / "specs/002-eval-dataset-factory/contracts/v2" / contract["schema_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert schema["additionalProperties"] is False

    gold = (ROOT / "evals/golden/eval_factory/approval" / "r7-07-checkpoint-interaction-v1.json").read_text(
        encoding="utf-8"
    )
    forbidden = {
        "raw_trace",
        "private_reference",
        "grader_prompt",
        "credential",
        "authentication_token",
        "production_registry",
    }
    assert all(value not in gold.casefold() for value in forbidden)
