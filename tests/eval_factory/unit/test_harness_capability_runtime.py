from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from eval_factory.contracts.agent_system_v2 import PlanKindV2
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.harness import (
    ArtifactEnvelopeV1,
    ArtifactModalityV1,
    CapabilityConsumerFacade,
    CapabilityConsumerKindV1,
    CapabilityInvocationContextV1,
    CapabilityInvocationOutcomeV1,
    CapabilityPermissionScopeV1,
    CapabilityProviderExecution,
    CapabilityRuntimeInputError,
    CapabilityRuntimeRegistrationError,
    CapabilityRuntimeRegistry,
    CapabilitySideEffectV1,
    ExecutionAuthorityV1,
    HarnessCapabilityRuntime,
    MemberExecutionGrantV1,
    PermissionOutcomeV1,
)
from eval_factory.harness.contracts import sorted_refs
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
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    PlanReviewCapabilityRequestV1,
)
from eval_factory.packs.generic_agent_trace.manifest import (
    build_generic_agent_trace_pack,
)

HASH = "a" * 64
RESULT_HASH = "b" * 64

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
        created_by="capability-runtime-test",
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
    sha256: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=sha256,
    )


@dataclass
class _StaticProvider:
    provider_binding_ref: ObjectRef
    implementation_id: str
    request_model: type[BaseModel]
    request_schema_ref: ObjectRef
    result_model: type[BaseModel]
    result_schema_ref: ObjectRef
    execution: CapabilityProviderExecution
    calls: int = 0

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: object,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution:
        del request, call, audit
        self.calls += 1
        return self.execution


def _provider_set(
    *,
    target_execution: CapabilityProviderExecution | None = None,
) -> tuple[
    object,
    tuple[_StaticProvider, ...],
    _StaticProvider,
]:
    registration = build_generic_agent_trace_pack(audit=_audit())
    by_id = {value.capability_id: value for value in registration.capability_definitions}
    bindings = {value.capability_definition_ref: value for value in registration.provider_bindings}
    providers: list[_StaticProvider] = []
    target: _StaticProvider | None = None
    for provider_type in PROVIDER_CLASSES:
        definition = by_id[provider_type.CAPABILITY_ID]
        binding = bindings[definition.to_ref()]
        execution = CapabilityProviderExecution(
            outcome=CapabilityInvocationOutcomeV1.SUCCEEDED,
            canonical_result_ref=_ref(
                "capability-owner-result",
                provider_type.CAPABILITY_ID,
                version="v2",
                sha256=RESULT_HASH,
            ),
            content_ref=_ref(
                "capability-owner-result",
                provider_type.CAPABILITY_ID,
                version="v2",
                sha256=RESULT_HASH,
            ),
        )
        if provider_type.CAPABILITY_ID == "capability.plan-review" and target_execution is not None:
            execution = target_execution
        provider = _StaticProvider(
            provider_binding_ref=binding.to_ref(),
            implementation_id=binding.implementation_id,
            request_model=provider_type.REQUEST_MODEL,
            request_schema_ref=definition.request_schema_ref,
            result_model=provider_type.RESULT_MODEL,
            result_schema_ref=definition.result_schema_ref,
            execution=execution,
        )
        providers.append(provider)
        if provider_type.CAPABILITY_ID == "capability.plan-review":
            target = provider
    assert target is not None
    return registration, tuple(providers), target


def _runtime_inputs() -> tuple[
    CapabilityRuntimeRegistry,
    HarnessCapabilityRuntime,
    PlanReviewCapabilityRequestV1,
    ArtifactEnvelopeV1,
    ExecutionAuthorityV1,
    CapabilityInvocationContextV1,
    CapabilityPermissionScopeV1,
    _StaticProvider,
]:
    registration, providers, target = _provider_set()
    registry = CapabilityRuntimeRegistry(
        registration=registration,
        providers=providers,
    )
    runtime = HarnessCapabilityRuntime(registry)
    definition = next(
        value
        for value in registration.capability_definitions
        if value.capability_id == "capability.plan-review"
    )
    provider_binding = next(
        value
        for value in registration.provider_bindings
        if value.capability_definition_ref == definition.to_ref()
    )
    producer = next(
        value
        for value in registration.capability_definitions
        if value.capability_id == "capability.requirement-planning"
    )
    compiled_plan_ref = _ref(
        "compiled-dataset-build-plan",
        version="v2",
    )
    input_artifact = ArtifactEnvelopeV1.create(
        artifact_id="artifact.compiled-plan",
        subject_ref=compiled_plan_ref,
        schema_ref=producer.result_schema_ref,
        content_ref=compiled_plan_ref,
        media_type="application/json",
        modality=ArtifactModalityV1.DOCUMENT,
        domain_tags=("generic-agent-trace",),
        semantic_role="compiled-build-plan",
        purpose="evaluation-data-production",
        classification="INTERNAL",
        lineage_refs=(),
        producer_capability_ref=producer.to_ref(),
        producer_task_ref=None,
        validation_refs=(),
        revision=1,
        predecessor_envelope_ref=None,
        audit=_audit(),
    )
    principal_ref = _ref("principal", "caller")
    grant = MemberExecutionGrantV1(
        member_id="member-review",
        principal_ref=principal_ref,
        capability_definition_refs=(definition.to_ref(),),
        provider_binding_refs=(provider_binding.to_ref(),),
        task_ids=("task-review",),
        data_scope_refs=(input_artifact.to_ref(),),
        data_purposes=("evaluation-data-production",),
        data_classifications=("INTERNAL",),
        allowed_side_effects=(CapabilitySideEffectV1.REVERSIBLE_WRITE,),
    )
    policy = registration.permission_policies[0]
    authority = ExecutionAuthorityV1.create(
        authority_id="execution-authority.runtime-test",
        authority_version=1,
        predecessor_authority_ref=None,
        team_id="team-runtime-test",
        team_incarnation_id="team-runtime-test-001",
        roster_ref=_ref("team-roster"),
        task_graph_ref=_ref("team-task-graph"),
        permission_policy_ref=policy.to_ref(),
        grants=(grant,),
        max_model_requests=10,
        max_model_tokens=10_000,
        max_cost_micro_usd=1_000_000,
        used_model_requests=0,
        used_model_tokens=0,
        used_cost_micro_usd=0,
        audit=_audit(),
    )
    context = CapabilityInvocationContextV1(
        session_ref=_ref("harness-session"),
        authority_ref=authority.to_ref(),
        principal_ref=principal_ref,
        data_purpose="evaluation-data-production",
        data_classification="INTERNAL",
        idempotency_key="runtime-test-call",
    )
    permission_scope = CapabilityPermissionScopeV1(
        member_id="member-review",
        task_id="task-review",
        data_scope_refs=(input_artifact.to_ref(),),
        local_execution=True,
        deterministic=True,
    )
    request = PlanReviewCapabilityRequestV1.create(
        run_id="factory-run://runtime-test",
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        plan_ref=_ref(
            "attachment-generation-plan",
            version="v2",
        ),
        audit=_audit(),
    )
    return (
        registry,
        runtime,
        request,
        input_artifact,
        authority,
        context,
        permission_scope,
        target,
    )


@pytest.mark.asyncio
async def test_all_consumer_kinds_use_one_runtime_path() -> None:
    (
        registry,
        runtime,
        request,
        input_artifact,
        authority,
        context,
        permission_scope,
        target,
    ) = _runtime_inputs()
    results = []
    for kind in CapabilityConsumerKindV1:
        binding = registry.bind_consumer(
            capability_id="capability.plan-review",
            consumer_id=f"consumer.{kind.value.casefold()}",
            consumer_version="v1",
            consumer_kind=kind,
            projection_ref=None,
            audit=_audit(),
        )
        invocation = await CapabilityConsumerFacade(
            runtime=runtime,
            binding=binding,
        ).invoke(
            context=context,
            permission_scope=permission_scope,
            authority=authority,
            request=request,
            input_artifacts=(input_artifact,),
            audit=_audit(),
        )
        results.append(invocation)

    assert target.calls == len(CapabilityConsumerKindV1)
    assert {value.result.canonical_result_ref for value in results} == {target.execution.canonical_result_ref}
    assert len({value.call.to_ref() for value in results}) == 1
    assert all(value.permission_decision.outcome is PermissionOutcomeV1.ALLOW for value in results)
    assert all(len(value.output_artifacts) == 1 for value in results)
    assert all(value.output_artifacts[0].semantic_role == "review-decision" for value in results)


def test_runtime_accepts_multiple_inputs_for_one_declared_fan_in_role() -> None:
    (
        registry,
        _,
        request,
        input_artifact,
        authority,
        context,
        permission_scope,
        _,
    ) = _runtime_inputs()
    second = ArtifactEnvelopeV1.create(
        artifact_id="artifact.compiled-plan.second",
        subject_ref=_ref(
            "compiled-dataset-build-plan",
            "second",
            version="v2",
        ),
        schema_ref=input_artifact.schema_ref,
        content_ref=_ref(
            "compiled-dataset-build-plan",
            "second",
            version="v2",
        ),
        media_type=input_artifact.media_type,
        modality=input_artifact.modality,
        domain_tags=input_artifact.domain_tags,
        semantic_role=input_artifact.semantic_role,
        purpose=input_artifact.purpose,
        classification=input_artifact.classification,
        lineage_refs=(),
        producer_capability_ref=(input_artifact.producer_capability_ref),
        producer_task_ref=None,
        validation_refs=(),
        revision=1,
        predecessor_envelope_ref=None,
        audit=_audit(),
    )
    refs = sorted_refs((input_artifact.to_ref(), second.to_ref()))
    original_grant = authority.grants[0]
    grant = original_grant.model_copy(
        update={"data_scope_refs": refs},
    )
    expanded = ExecutionAuthorityV1.create(
        authority_id="execution-authority.runtime-fan-in-test",
        authority_version=1,
        predecessor_authority_ref=None,
        team_id=authority.team_id,
        team_incarnation_id=authority.team_incarnation_id,
        roster_ref=authority.roster_ref,
        task_graph_ref=authority.task_graph_ref,
        permission_policy_ref=authority.permission_policy_ref,
        grants=(grant,),
        max_model_requests=authority.max_model_requests,
        max_model_tokens=authority.max_model_tokens,
        max_cost_micro_usd=authority.max_cost_micro_usd,
        used_model_requests=0,
        used_model_tokens=0,
        used_cost_micro_usd=0,
        audit=_audit(),
    )
    consumer = registry.bind_consumer(
        capability_id="capability.plan-review",
        consumer_id="consumer.fan-in",
        consumer_version="v1",
        consumer_kind=CapabilityConsumerKindV1.INTERNAL_SERVICE,
        projection_ref=None,
        audit=_audit(),
    )

    HarnessCapabilityRuntime._validate_invocation(
        resolved=registry.resolve(consumer),
        context=context.model_copy(
            update={"authority_ref": expanded.to_ref()},
        ),
        permission_scope=permission_scope.model_copy(
            update={"data_scope_refs": refs},
        ),
        authority=expanded,
        request=request,
        input_artifacts=(input_artifact, second),
    )


@pytest.mark.asyncio
async def test_permission_block_publishes_no_output_and_skips_provider() -> None:
    (
        registry,
        runtime,
        request,
        input_artifact,
        authority,
        context,
        permission_scope,
        target,
    ) = _runtime_inputs()
    binding = registry.bind_consumer(
        capability_id="capability.plan-review",
        consumer_id="consumer.internal",
        consumer_version="v1",
        consumer_kind=CapabilityConsumerKindV1.INTERNAL_SERVICE,
        projection_ref=None,
        audit=_audit(),
    )
    blocked_scope = permission_scope.model_copy(
        update={"task_id": "task-not-granted"},
    )

    invocation = await runtime.invoke(
        consumer=binding,
        context=context,
        permission_scope=blocked_scope,
        authority=authority,
        request=request,
        input_artifacts=(input_artifact,),
        audit=_audit(),
    )

    assert target.calls == 0
    assert invocation.permission_decision.outcome is PermissionOutcomeV1.DENY
    assert invocation.result.outcome is CapabilityInvocationOutcomeV1.BLOCKED_POLICY
    assert invocation.result.canonical_result_ref is None
    assert invocation.output_artifacts == ()


@pytest.mark.asyncio
async def test_provider_non_success_publishes_no_canonical_output() -> None:
    execution = CapabilityProviderExecution(
        outcome=CapabilityInvocationOutcomeV1.ABSTAINED,
        failure_code="PLAN_REVIEW_NOT_READY",
    )
    registration, providers, target = _provider_set(
        target_execution=execution,
    )
    registry = CapabilityRuntimeRegistry(
        registration=registration,
        providers=providers,
    )
    (
        _,
        _,
        request,
        input_artifact,
        authority,
        context,
        permission_scope,
        _,
    ) = _runtime_inputs()
    binding = registry.bind_consumer(
        capability_id="capability.plan-review",
        consumer_id="consumer.internal",
        consumer_version="v1",
        consumer_kind=CapabilityConsumerKindV1.INTERNAL_SERVICE,
        projection_ref=None,
        audit=_audit(),
    )

    invocation = await HarnessCapabilityRuntime(registry).invoke(
        consumer=binding,
        context=context,
        permission_scope=permission_scope,
        authority=authority,
        request=request,
        input_artifacts=(input_artifact,),
        audit=_audit(),
    )

    assert target.calls == 1
    assert invocation.result.outcome is CapabilityInvocationOutcomeV1.ABSTAINED
    assert invocation.result.failure_code == "PLAN_REVIEW_NOT_READY"
    assert invocation.result.canonical_result_ref is None
    assert invocation.output_artifacts == ()


def test_registry_rejects_missing_or_drifted_implementation() -> None:
    registration, providers, _ = _provider_set()
    with pytest.raises(
        CapabilityRuntimeRegistrationError,
        match="one installed",
    ):
        CapabilityRuntimeRegistry(
            registration=registration,
            providers=providers[:-1],
        )

    changed = providers[0]
    changed.request_schema_ref = _ref(
        "json-schema",
        "drifted",
        version="v1",
    )
    with pytest.raises(
        CapabilityRuntimeRegistrationError,
        match="authority drifted",
    ):
        CapabilityRuntimeRegistry(
            registration=registration,
            providers=providers,
        )


def test_provider_execution_and_scope_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="requires result"):
        CapabilityProviderExecution(
            outcome=CapabilityInvocationOutcomeV1.SUCCEEDED,
        )
    with pytest.raises(ValueError, match="closed failure"):
        CapabilityProviderExecution(
            outcome=CapabilityInvocationOutcomeV1.FAILED,
        )
    with pytest.raises(ValueError, match="validation refs"):
        CapabilityProviderExecution(
            outcome=CapabilityInvocationOutcomeV1.FAILED,
            validation_refs=(
                _ref("validation", "z"),
                _ref("validation", "a"),
            ),
            failure_code="FAILED",
        )
    with pytest.raises(ValueError, match="domain tags"):
        CapabilityProviderExecution(
            outcome=CapabilityInvocationOutcomeV1.FAILED,
            failure_code="FAILED",
            domain_tags=("z", "a"),
        )
    with pytest.raises(ValidationError, match="exactly local or external"):
        CapabilityPermissionScopeV1(
            member_id="member-review",
            task_id="task-review",
            local_execution=False,
            external_execution=False,
            deterministic=True,
        )


@pytest.mark.asyncio
async def test_runtime_rejects_stale_scope_before_provider() -> None:
    (
        registry,
        runtime,
        request,
        input_artifact,
        authority,
        context,
        permission_scope,
        target,
    ) = _runtime_inputs()
    binding = registry.bind_consumer(
        capability_id="capability.plan-review",
        consumer_id="consumer.internal",
        consumer_version="v1",
        consumer_kind=CapabilityConsumerKindV1.INTERNAL_SERVICE,
        projection_ref=None,
        audit=_audit(),
    )
    wrong_scope = permission_scope.model_copy(
        update={
            "data_scope_refs": sorted_refs(
                (_ref("artifact-envelope", "other"),),
            ),
        },
    )

    with pytest.raises(
        CapabilityRuntimeInputError,
        match="permission scope",
    ):
        await runtime.invoke(
            consumer=binding,
            context=context,
            permission_scope=wrong_scope,
            authority=authority,
            request=request,
            input_artifacts=(input_artifact,),
            audit=_audit(),
        )
    assert target.calls == 0
