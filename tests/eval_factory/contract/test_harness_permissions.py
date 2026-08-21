from __future__ import annotations

from datetime import UTC, datetime

import pytest

from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.harness import (
    BalancedAutonomyEvaluator,
    BalancedAutonomyPolicyV1,
    CapabilitySideEffectV1,
    ExecutionAuthorityV1,
    GraphMutationKindV1,
    MemberExecutionGrantV1,
    PermissionActionV1,
    PermissionDecisionV1,
    PermissionOutcomeV1,
    PermissionReasonV1,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs import build_generic_agent_trace_pack

HASH = "a" * 64


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v1",
        object_version="v1",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 17, tzinfo=UTC),
        created_by="permission-contract-test",
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage0",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _authority() -> tuple[
    BalancedAutonomyPolicyV1,
    ExecutionAuthorityV1,
    ObjectRef,
    ObjectRef,
    ObjectRef,
    ObjectRef,
]:
    registration = build_generic_agent_trace_pack(audit=_audit())
    policy = registration.permission_policies[0]
    capability = registration.capability_definitions[0]
    provider = next(
        value
        for value in registration.provider_bindings
        if value.capability_definition_ref == capability.to_ref()
    )
    principal_ref = _ref("principal", "requirement-member")
    data_scope_ref = _ref("artifact-envelope", "requirement-input")
    grant = MemberExecutionGrantV1(
        member_id="member-requirement",
        principal_ref=principal_ref,
        capability_definition_refs=(capability.to_ref(),),
        provider_binding_refs=(provider.to_ref(),),
        task_ids=("task-requirement",),
        data_scope_refs=(data_scope_ref,),
        data_purposes=("evaluation-data-production",),
        data_classifications=("INTERNAL",),
        allowed_side_effects=tuple(sorted(CapabilitySideEffectV1, key=lambda value: value.value)),
    )
    authority = ExecutionAuthorityV1.create(
        authority_id="execution-authority.generic-agent-eval",
        authority_version=1,
        predecessor_authority_ref=None,
        team_id="team-generic-agent-eval",
        team_incarnation_id="team-incarnation-001",
        roster_ref=_ref("team-roster"),
        task_graph_ref=_ref("team-task-graph"),
        permission_policy_ref=policy.to_ref(),
        grants=(grant,),
        max_model_requests=10,
        max_model_tokens=100_000,
        max_cost_micro_usd=5_000_000,
        used_model_requests=2,
        used_model_tokens=10_000,
        used_cost_micro_usd=500_000,
        audit=_audit(),
    )
    return (
        policy,
        authority,
        capability.to_ref(),
        provider.to_ref(),
        principal_ref,
        data_scope_ref,
    )


def _action(**overrides: object) -> PermissionActionV1:
    (
        policy,
        authority,
        capability_ref,
        provider_ref,
        principal_ref,
        data_scope_ref,
    ) = _authority()
    values: dict[str, object] = {
        "action_id": "permission-action-001",
        "authority_ref": authority.to_ref(),
        "permission_policy_ref": policy.to_ref(),
        "member_id": "member-requirement",
        "principal_ref": principal_ref,
        "capability_definition_ref": capability_ref,
        "provider_binding_ref": provider_ref,
        "task_id": "task-requirement",
        "data_scope_refs": (data_scope_ref,),
        "data_purpose": "evaluation-data-production",
        "data_classification": "INTERNAL",
        "side_effect": CapabilitySideEffectV1.READ_ONLY,
        "graph_mutation": GraphMutationKindV1.NONE,
        "local_execution": True,
        "deterministic": True,
        "source_admission": False,
        "external_execution": False,
        "destructive": False,
        "release": False,
        "model_requests_delta": 0,
        "model_tokens_delta": 0,
        "cost_micro_usd_delta": 0,
        "audit": _audit(),
    }
    values.update(overrides)
    return PermissionActionV1.create(**values)  # type: ignore[arg-type]


def _evaluate(action: PermissionActionV1) -> PermissionDecisionV1:
    policy, authority, *_ = _authority()
    return BalancedAutonomyEvaluator().evaluate(
        policy=policy,
        authority=authority,
        action=action,
        audit=_audit(),
    )


def test_balanced_autonomy_allows_work_inside_exact_authority() -> None:
    read_decision = _evaluate(_action())
    assert read_decision.outcome is PermissionOutcomeV1.ALLOW
    assert read_decision.reason_codes == (PermissionReasonV1.INSIDE_APPROVED_AUTHORITY,)

    write_decision = _evaluate(
        _action(
            action_id="permission-action-002",
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            graph_mutation=GraphMutationKindV1.IN_SCOPE,
        )
    )
    assert write_decision.outcome is PermissionOutcomeV1.ALLOW

    budgeted_model_decision = _evaluate(
        _action(
            action_id="permission-action-003",
            deterministic=False,
            model_requests_delta=1,
            model_tokens_delta=5_000,
            cost_micro_usd_delta=100_000,
        )
    )
    assert budgeted_model_decision.outcome is PermissionOutcomeV1.ALLOW


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        (
            {"source_admission": True},
            PermissionReasonV1.SOURCE_ADMISSION,
        ),
        (
            {
                "external_execution": True,
                "local_execution": False,
                "side_effect": CapabilitySideEffectV1.EXTERNAL_EFFECT,
            },
            PermissionReasonV1.EXTERNAL_EXECUTION,
        ),
        (
            {"graph_mutation": GraphMutationKindV1.AUTHORITY_WIDENING},
            PermissionReasonV1.GRAPH_AUTHORITY_WIDENING,
        ),
        (
            {
                "destructive": True,
                "side_effect": CapabilitySideEffectV1.IRREVERSIBLE_WRITE,
            },
            PermissionReasonV1.DESTRUCTIVE_ACTION,
        ),
        (
            {"release": True},
            PermissionReasonV1.RELEASE,
        ),
        (
            {"model_requests_delta": 9},
            PermissionReasonV1.MODEL_BUDGET_WIDENING,
        ),
        (
            {
                "side_effect": CapabilitySideEffectV1.REVERSIBLE_WRITE,
                "deterministic": False,
            },
            PermissionReasonV1.NONDETERMINISTIC_MUTATION,
        ),
    ),
)
def test_balanced_autonomy_requests_approval_for_widening(
    overrides: dict[str, object],
    reason: PermissionReasonV1,
) -> None:
    decision = _evaluate(_action(**overrides))
    assert decision.outcome is PermissionOutcomeV1.REQUIRE_APPROVAL
    assert reason in decision.reason_codes


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        (
            {"member_id": "member-unknown"},
            PermissionReasonV1.UNKNOWN_PRINCIPAL,
        ),
        (
            {"capability_definition_ref": _ref("harness-capability-definition", "other")},
            PermissionReasonV1.CAPABILITY_NOT_GRANTED,
        ),
        (
            {"provider_binding_ref": _ref("harness-capability-provider", "other")},
            PermissionReasonV1.PROVIDER_NOT_GRANTED,
        ),
        (
            {"task_id": "task-other"},
            PermissionReasonV1.TASK_NOT_GRANTED,
        ),
        (
            {"data_scope_refs": (_ref("artifact-envelope", "other"),)},
            PermissionReasonV1.DATA_SCOPE_NOT_GRANTED,
        ),
        (
            {"data_purpose": "other-purpose"},
            PermissionReasonV1.PURPOSE_NOT_GRANTED,
        ),
        (
            {"data_classification": "RESTRICTED"},
            PermissionReasonV1.CLASSIFICATION_NOT_GRANTED,
        ),
    ),
)
def test_balanced_autonomy_denies_ungranted_authority(
    overrides: dict[str, object],
    reason: PermissionReasonV1,
) -> None:
    decision = _evaluate(_action(**overrides))
    assert decision.outcome is PermissionOutcomeV1.DENY
    assert reason in decision.reason_codes


def test_stale_authority_and_policy_are_denied() -> None:
    stale_action = _action(
        authority_ref=_ref("execution-authority", "stale"),
    )
    stale_decision = _evaluate(stale_action)
    assert stale_decision.outcome is PermissionOutcomeV1.DENY
    assert stale_decision.reason_codes == (PermissionReasonV1.STALE_AUTHORITY,)

    wrong_policy_action = _action(
        permission_policy_ref=_ref("balanced-autonomy-policy", "other"),
    )
    wrong_policy_decision = _evaluate(wrong_policy_action)
    assert wrong_policy_decision.outcome is PermissionOutcomeV1.DENY
    assert wrong_policy_decision.reason_codes == (PermissionReasonV1.WRONG_PERMISSION_POLICY,)


def test_grant_ref_inventories_are_canonical() -> None:
    policy, authority, *_ = _authority()
    grant = authority.grants[0]
    assert grant.capability_definition_refs == sorted_refs(grant.capability_definition_refs)
    assert authority.permission_policy_ref == policy.to_ref()


def test_taskless_coordinator_grant_has_no_execution_scope() -> None:
    grant = MemberExecutionGrantV1(
        member_id="member-coordinator",
        principal_ref=_ref("principal", "coordinator"),
        capability_definition_refs=(_ref("harness-capability-definition"),),
        provider_binding_refs=(_ref("harness-capability-provider"),),
        task_ids=(),
        data_scope_refs=(),
        data_purposes=("evaluation-data-production",),
        data_classifications=("INTERNAL",),
        allowed_side_effects=(CapabilitySideEffectV1.READ_ONLY,),
    )

    assert grant.task_ids == ()
