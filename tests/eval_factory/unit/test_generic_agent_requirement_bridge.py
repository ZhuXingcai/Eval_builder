from __future__ import annotations

from datetime import UTC, datetime

from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.harness import (
    HarnessRequirementPolicyV1,
    ProviderEvidenceClassV1,
    RequirementInterpretationOutcomeV1,
    RequirementInterpretationV1,
)
from eval_factory.packs.generic_agent_trace.requirement_bridge import (
    GenericAgentRequirementBridge,
    GenericAgentRequirementBridgeInputV1,
    RequirementBridgeFailureCodeV1,
    RequirementBridgeOutcomeV1,
)

HASH = "a" * 64


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 18, tzinfo=UTC),
        created_by="requirement-bridge-test",
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage2",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


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


def _interpretation(
    outcome: RequirementInterpretationOutcomeV1 = (RequirementInterpretationOutcomeV1.READY),
) -> RequirementInterpretationV1:
    ready = outcome is RequirementInterpretationOutcomeV1.READY
    clarification = outcome is RequirementInterpretationOutcomeV1.CLARIFICATION_REQUIRED
    return RequirementInterpretationV1.create(
        interpretation_id="requirement-interpretation.test",
        session_ref=_ref("harness-session"),
        command_ref=_ref("session-command"),
        user_message_ref=_ref("harness-message", "user"),
        assistant_message_ref=_ref("harness-message", "assistant"),
        proposal_ref=_ref("requirement-interpretation-proposal"),
        gateway_result_ref=_ref(
            "gateway-invocation-result",
            version="v2",
        ),
        outcome=outcome,
        evidence_class=ProviderEvidenceClassV1.MECHANISM_FIXTURE,
        goals=(("Build a safe generic Agent candidate dataset.",) if ready else ()),
        constraints=(("Do not expose restricted trace content.",) if ready else ()),
        assumptions=(("Admitted sources are immutable.",) if ready else ()),
        source_expectations=(("Use the admitted trace manifest.",) if ready else ()),
        target_capabilities=(("generic-agent-trace",) if ready else ()),
        quality_intent=("Require auditable lineage." if ready else None),
        delivery_intent=("Deliver candidate JSONL." if ready else None),
        missing_field_codes=(("SOURCE",) if clarification else ()),
        clarification_questions=(("Which admitted source should be used?",) if clarification else ()),
        reason_codes=(() if ready or clarification else ("INTERPRETATION_BLOCKED",)),
        audit=_audit(),
    )


def _policy(
    interpretation_ref: ObjectRef,
) -> HarnessRequirementPolicyV1:
    return HarnessRequirementPolicyV1.create(
        policy_id="harness-requirement-policy.test",
        interpretation_ref=interpretation_ref,
        data_classification="RESTRICTED",
        residency="LOCAL",
        max_model_requests=12,
        max_model_tokens=120_000,
        max_cost_micro_usd=2_000_000,
        audit=_audit(),
    )


def _bridge_input(
    *,
    interpretation: RequirementInterpretationV1 | None = None,
    policy: HarnessRequirementPolicyV1 | None = None,
    requirement_source_ref: ObjectRef | None = None,
) -> GenericAgentRequirementBridgeInputV1:
    value = interpretation or _interpretation()
    return GenericAgentRequirementBridgeInputV1.create(
        interpretation=value,
        requirement_policy=policy or _policy(value.to_ref()),
        dataset_run_id="factory-run://harness-stage2/test",
        requirement_source_ref=requirement_source_ref
        or _ref(
            "evaluation-requirement-source",
            version="v2",
        ),
        manifest_ref=_ref("trace-manifest", version="v2"),
        source_authorization_ref=_ref(
            "trace-source-authorization",
            version="v2",
        ),
        allowed_task_kinds=(
            "attachment-mock",
            "criteria-rubric",
            "grading-design",
            "task-rewrite",
            "trace-extraction",
        ),
        pipeline_policy_refs=(
            _ref("batch-quality-policy", version="v2"),
            _ref("release-projection-policy", version="v2"),
        ),
        gateway_registry_refs=(
            _ref("agent-registry", version="v2"),
            _ref("model-catalog", version="v2"),
            _ref("prompt-registry", version="v2"),
        ),
        output_target_ref=_ref(
            "candidate-output-target",
            version="v2",
        ),
        max_transitions=512,
        max_plan_revisions=8,
        max_agent_attempts=2,
        idempotency_key="requirement-bridge-test",
        audit=_audit(),
    )


def test_ready_requirement_compiles_exact_factory_authority() -> None:
    bridge_input = _bridge_input()

    result = GenericAgentRequirementBridge().compile(
        bridge_input,
        audit=_audit(),
    )

    assert result.outcome is RequirementBridgeOutcomeV1.SUCCEEDED
    assert result.failure_codes == ()
    assert result.evidence_class is ProviderEvidenceClassV1.MECHANISM_FIXTURE
    assert result.requirement is not None
    assert result.factory_policy is not None
    assert result.run_request is not None
    assert result.requirement.goals == bridge_input.interpretation.goals
    assert result.requirement.source_ref == bridge_input.requirement_source_ref
    assert result.factory_policy.max_model_requests == bridge_input.requirement_policy.max_model_requests
    assert result.factory_policy.max_model_tokens == bridge_input.requirement_policy.max_model_tokens
    assert result.factory_policy.max_cost_micro_usd == bridge_input.requirement_policy.max_cost_micro_usd
    assert result.factory_policy.production_release_allowed is False
    assert result.run_request.requirement_spec_ref == result.requirement.to_ref()
    assert result.run_request.factory_policy_ref == result.factory_policy.to_ref()
    assert result.run_request.max_transitions == bridge_input.max_transitions


def test_non_ready_interpretation_creates_no_factory_authority() -> None:
    interpretation = _interpretation(
        RequirementInterpretationOutcomeV1.CLARIFICATION_REQUIRED,
    )
    bridge_input = _bridge_input(interpretation=interpretation)

    result = GenericAgentRequirementBridge().compile(
        bridge_input,
        audit=_audit(),
    )

    assert result.outcome is RequirementBridgeOutcomeV1.BLOCKED_INPUT
    assert result.failure_codes == (RequirementBridgeFailureCodeV1.INTERPRETATION_NOT_READY,)
    assert result.requirement is None
    assert result.factory_policy is None
    assert result.run_request is None


def test_stale_policy_and_source_authority_fail_closed_together() -> None:
    interpretation = _interpretation()
    policy = _policy(
        _ref("requirement-interpretation", "stale"),
    )
    bridge_input = _bridge_input(
        interpretation=interpretation,
        policy=policy,
        requirement_source_ref=_ref(
            "unadmitted-requirement-source",
            version="v2",
        ),
    )

    result = GenericAgentRequirementBridge().compile(
        bridge_input,
        audit=_audit(),
    )

    assert result.outcome is RequirementBridgeOutcomeV1.BLOCKED_INPUT
    assert result.failure_codes == (
        RequirementBridgeFailureCodeV1.INTERPRETATION_POLICY_STALE,
        RequirementBridgeFailureCodeV1.SOURCE_AUTHORITY_INCOMPLETE,
    )
    assert result.requirement is None
    assert result.factory_policy is None
    assert result.run_request is None


def test_unsupported_target_capability_is_not_inferred() -> None:
    source = _interpretation()
    interpretation = RequirementInterpretationV1.create(
        interpretation_id=source.interpretation_id,
        session_ref=source.session_ref,
        command_ref=source.command_ref,
        user_message_ref=source.user_message_ref,
        assistant_message_ref=source.assistant_message_ref,
        proposal_ref=source.proposal_ref,
        gateway_result_ref=source.gateway_result_ref,
        outcome=source.outcome,
        evidence_class=source.evidence_class,
        goals=source.goals,
        constraints=source.constraints,
        assumptions=source.assumptions,
        source_expectations=source.source_expectations,
        target_capabilities=("medical-agent",),
        quality_intent=source.quality_intent,
        delivery_intent=source.delivery_intent,
        missing_field_codes=(),
        clarification_questions=(),
        reason_codes=(),
        audit=_audit(),
    )

    result = GenericAgentRequirementBridge().compile(
        _bridge_input(interpretation=interpretation),
        audit=_audit(),
    )

    assert result.failure_codes == (RequirementBridgeFailureCodeV1.TARGET_CAPABILITY_UNSUPPORTED,)
    assert result.run_request is None
