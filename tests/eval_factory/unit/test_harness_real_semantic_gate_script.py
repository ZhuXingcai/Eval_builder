from __future__ import annotations

from scripts.run_harness_stage1_real_semantic import _semantic_gate_reasons


def _turn(
    *,
    outcome: str,
    policy: bool,
    failure_code: str | None = None,
) -> dict[str, object]:
    return {
        "evidence_class": "REAL_SEMANTIC",
        "failure_code": failure_code,
        "outcome": outcome,
        "requirement_policy_ref": {"object_type": "harness-requirement-policy"} if policy else None,
    }


def test_semantic_gate_accepts_clarification_and_ready_authority() -> None:
    assert (
        _semantic_gate_reasons(
            incomplete=_turn(
                outcome="CLARIFICATION_REQUIRED",
                policy=False,
            ),
            complete=_turn(
                outcome="READY",
                policy=True,
            ),
        )
        == ()
    )


def test_semantic_gate_preserves_provider_failure_as_blocked() -> None:
    assert _semantic_gate_reasons(
        incomplete=_turn(
            outcome="BLOCKED_CAPABILITY",
            policy=False,
            failure_code="MODEL_PROVIDER_FAILED",
        ),
        complete=_turn(
            outcome="BLOCKED_CAPABILITY",
            policy=False,
            failure_code="MODEL_PROVIDER_FAILED",
        ),
    ) == (
        "COMPLETE_MODEL_PROVIDER_FAILED",
        "COMPLETE_REQUIREMENT_NOT_READY",
        "INCOMPLETE_MODEL_PROVIDER_FAILED",
        "INCOMPLETE_REQUIREMENT_NOT_CLARIFIED",
        "READY_REQUIREMENT_POLICY_MISSING",
    )
