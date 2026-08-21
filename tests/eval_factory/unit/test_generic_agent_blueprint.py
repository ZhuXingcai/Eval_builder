from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from eval_factory.blueprints import BlueprintBudgetV1
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.harness import (
    CapabilityInvocationOutcomeV1,
    CapabilityProviderExecution,
    CapabilityRuntimeRegistry,
    CapabilitySideEffectV1,
    ExecutionAuthorityV1,
    MemberExecutionGrantV1,
)
from eval_factory.packs.generic_agent_trace.blueprint import (
    GenericAgentBlueprintCompiler,
    GenericAgentBlueprintCompilerError,
    GenericAgentBlueprintInputV1,
)
from eval_factory.packs.generic_agent_trace.capabilities import (
    AttachmentQualityCapabilityProvider,
    AttachmentReconstructionCapabilityProvider,
    BatchQualityCapabilityProvider,
    CriteriaRubricCapabilityProvider,
    DeliveryCapabilityProvider,
    GradingDesignCapabilityProvider,
    PlanReviewCapabilityProvider,
    RequirementPlanningCapabilityProvider,
    TaskAuthoringCapabilityProvider,
    TraceIngestionCapabilityProvider,
)
from eval_factory.packs.generic_agent_trace.manifest import (
    build_generic_agent_trace_pack,
    build_generic_agent_trace_pack_v1_0,
)

HASH = "a" * 64

PROVIDER_CLASSES = (
    RequirementPlanningCapabilityProvider,
    TraceIngestionCapabilityProvider,
    TaskAuthoringCapabilityProvider,
    AttachmentReconstructionCapabilityProvider,
    CriteriaRubricCapabilityProvider,
    GradingDesignCapabilityProvider,
    AttachmentQualityCapabilityProvider,
    BatchQualityCapabilityProvider,
    PlanReviewCapabilityProvider,
    DeliveryCapabilityProvider,
)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 18, tzinfo=UTC),
        created_by="generic-blueprint-test",
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


@dataclass(frozen=True)
class _Provider:
    provider_binding_ref: ObjectRef
    implementation_id: str
    request_model: type[BaseModel]
    request_schema_ref: ObjectRef
    result_model: type[BaseModel]
    result_schema_ref: ObjectRef

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: object,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        del request, call, audit
        return CapabilityProviderExecution(
            outcome=CapabilityInvocationOutcomeV1.FAILED,
            failure_code="NOT_EXECUTED",
        )


def _runtime_registry() -> CapabilityRuntimeRegistry:
    registration = build_generic_agent_trace_pack(audit=_audit())
    definitions = {value.capability_id: value for value in registration.capability_definitions}
    bindings = {value.capability_definition_ref: value for value in registration.provider_bindings}
    providers = tuple(
        _Provider(
            provider_binding_ref=bindings[definitions[value.CAPABILITY_ID].to_ref()].to_ref(),
            implementation_id=bindings[definitions[value.CAPABILITY_ID].to_ref()].implementation_id,
            request_model=value.REQUEST_MODEL,
            request_schema_ref=definitions[value.CAPABILITY_ID].request_schema_ref,
            result_model=value.RESULT_MODEL,
            result_schema_ref=definitions[value.CAPABILITY_ID].result_schema_ref,
        )
        for value in PROVIDER_CLASSES
    )
    return CapabilityRuntimeRegistry(
        registration=registration,
        providers=providers,
    )


def _authority(
    registry: CapabilityRuntimeRegistry,
    *,
    capability_refs: tuple[ObjectRef, ...] | None = None,
) -> ExecutionAuthorityV1:
    registration = registry.registration
    grant = MemberExecutionGrantV1(
        member_id="member-coordinator",
        principal_ref=_ref("principal", "coordinator"),
        capability_definition_refs=capability_refs or registration.manifest.capability_definition_refs,
        provider_binding_refs=registration.manifest.provider_binding_refs,
        task_ids=("task-blueprint",),
        data_scope_refs=(),
        data_purposes=("evaluation-data-production",),
        data_classifications=("INTERNAL",),
        allowed_side_effects=tuple(
            sorted(
                CapabilitySideEffectV1,
                key=lambda value: value.value,
            ),
        ),
    )
    return ExecutionAuthorityV1.create(
        authority_id="execution-authority.blueprint-test",
        authority_version=1,
        predecessor_authority_ref=None,
        team_id="team-blueprint-test",
        team_incarnation_id="team-blueprint-test-001",
        roster_ref=_ref("team-roster"),
        task_graph_ref=_ref("team-task-graph"),
        permission_policy_ref=(registration.permission_policies[0].to_ref()),
        grants=(grant,),
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        used_model_requests=10,
        used_model_tokens=100_000,
        used_cost_micro_usd=10_000_000,
        audit=_audit(),
    )


def _input(
    registry: CapabilityRuntimeRegistry,
    *,
    authority: ExecutionAuthorityV1 | None = None,
    budget: BlueprintBudgetV1 | None = None,
) -> GenericAgentBlueprintInputV1:
    registration = registry.registration
    value = authority or _authority(registry)
    return GenericAgentBlueprintInputV1.create(
        requirement_ref=_ref(
            "evaluation-requirement-spec",
            version="v2",
        ),
        team_ref=_ref("agent-team"),
        roster_ref=value.roster_ref,
        task_graph_ref=value.task_graph_ref,
        authority=value,
        evaluator_refs=(_ref("evaluator-profile"),),
        review_gate_refs=registration.manifest.permission_policy_refs,
        budget=budget
        or BlueprintBudgetV1(
            max_model_requests=90,
            max_model_tokens=900_000,
            max_cost_micro_usd=90_000_000,
            max_parallel_tasks=8,
        ),
        delivery_profile_ref=_ref("delivery-profile"),
        audit=_audit(),
    )


def test_fixed_blueprint_compiles_ten_capabilities_and_connected_dag() -> None:
    registry = _runtime_registry()

    blueprint = GenericAgentBlueprintCompiler(
        registration=registry.registration,
        runtime_registry=registry,
    ).compile(
        _input(registry),
        audit=_audit(),
    )

    assert len(blueprint.capability_bindings) == 10
    assert len(blueprint.data_flow_edges) == 19
    assert {value.capability_id for value in blueprint.capability_bindings} == {
        value.capability_id for value in registry.registration.capability_definitions
    }
    assert blueprint.execution_provider_refs == registry.registration.manifest.provider_binding_refs
    assert blueprint.review_gate_refs == registry.registration.manifest.permission_policy_refs


def test_blueprint_rejects_budget_widening() -> None:
    registry = _runtime_registry()
    widened = BlueprintBudgetV1(
        max_model_requests=91,
        max_model_tokens=900_001,
        max_cost_micro_usd=90_000_001,
        max_parallel_tasks=8,
    )

    with pytest.raises(
        GenericAgentBlueprintCompilerError,
        match="budget widens",
    ):
        GenericAgentBlueprintCompiler(
            registration=registry.registration,
            runtime_registry=registry,
        ).compile(
            _input(registry, budget=widened),
            audit=_audit(),
        )


def test_blueprint_rejects_incomplete_execution_grants() -> None:
    registry = _runtime_registry()
    authority = _authority(
        registry,
        capability_refs=(registry.registration.manifest.capability_definition_refs[:-1]),
    )

    with pytest.raises(
        GenericAgentBlueprintCompilerError,
        match="grants differ",
    ):
        GenericAgentBlueprintCompiler(
            registration=registry.registration,
            runtime_registry=registry,
        ).compile(
            _input(registry, authority=authority),
            audit=_audit(),
        )


def test_blueprint_rejects_legacy_or_runtime_composition_drift() -> None:
    registry = _runtime_registry()
    legacy = build_generic_agent_trace_pack_v1_0(audit=_audit())

    with pytest.raises(
        GenericAgentBlueprintCompilerError,
        match="composition is not exact",
    ):
        GenericAgentBlueprintCompiler(
            registration=legacy,
            runtime_registry=registry,
        ).compile(
            _input(registry),
            audit=_audit(),
        )
