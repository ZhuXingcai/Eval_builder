from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from eval_factory.contracts.agent_system_v2 import (
    AttachmentQualityOutcomeV2,
    AttachmentSubgraphOutcomeV2,
    CriteriaRubricOutcomeV2,
    GradingDesignOutcomeV2,
    PlanKindV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetOutcomeV2,
)
from eval_factory.harness import (
    ArtifactEnvelopeV1,
    ArtifactModalityV1,
    CapabilityConsumerKindV1,
    CapabilityInvocationContextV1,
    CapabilityInvocationOutcomeV1,
    CapabilityPermissionScopeV1,
    CapabilityProviderExecution,
    CapabilityRuntimeRegistry,
    ExecutionAuthorityV1,
    HarnessCapabilityRuntime,
    MemberExecutionGrantV1,
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
    AttachmentQualityCapabilityRequestV1,
    AttachmentReconstructionCapabilityRequestV1,
    BatchQualityCapabilityRequestV1,
    CriteriaRubricCapabilityRequestV1,
    DeliveryCapabilityRequestV1,
    GradingDesignCapabilityRequestV1,
    PlanReviewCapabilityRequestV1,
    RequirementPlanningCapabilityRequestV1,
    TaskAuthoringCapabilityRequestV1,
    TraceIngestionCapabilityRequestV1,
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
        created_by="capability-adapter-test",
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
    version: str = "v2",
    sha256: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=sha256,
    )


@dataclass
class _Materials:
    values: dict[ObjectRef, object]
    many: dict[ObjectRef, tuple[object, ...]]

    def get[ValueT](
        self,
        reference: ObjectRef,
        expected_type: type[ValueT],
    ) -> ValueT:
        del expected_type
        return self.values[reference]  # type: ignore[return-value]

    def get_many[ValueT](
        self,
        reference: ObjectRef,
        expected_type: type[ValueT],
    ) -> tuple[ValueT, ...]:
        del expected_type
        return self.many[reference]  # type: ignore[return-value]


@dataclass
class _PlaceholderProvider:
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
        raise AssertionError("non-target provider was invoked")


class _Compiler:
    def __init__(self, result_ref: ObjectRef) -> None:
        self.result = _RefValue(result_ref)

    def compile(self, **kwargs: object) -> _RefValue:
        del kwargs
        return self.result


class _TraceService:
    def __init__(self, result_ref: ObjectRef) -> None:
        self.result_ref = result_ref

    def execute(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(
            trace_envelope_ref=self.result_ref,
            stored_index=object(),
        )


class _TraceCandidate:
    def __init__(self, references: tuple[ObjectRef, ...]) -> None:
        self.references = references

    def validation_refs(self) -> tuple[ObjectRef, ...]:
        return self.references


class _TraceCandidatePreparer:
    def __init__(self, candidate: _TraceCandidate) -> None:
        self.candidate = candidate
        self.calls = 0

    async def prepare(self, **kwargs: object) -> _TraceCandidate:
        del kwargs
        self.calls += 1
        return self.candidate


class _TraceCandidateStore:
    def __init__(self) -> None:
        self.calls = 0

    def commit(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(preparation=kwargs["preparation"])


class _TaskBridge:
    def __init__(self, result_ref: ObjectRef) -> None:
        self.contract_set = SimpleNamespace(
            contract_set_id=result_ref.object_id,
            contract_set_sha256=result_ref.object_sha256,
            task_prompt_safety_gate_ref=_ref("task-prompt-safety-gate"),
        )

    async def run(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(task_contract_set=self.contract_set)


class _TaskProjection:
    def __init__(self, references: tuple[ObjectRef, ...]) -> None:
        self.references = references
        self.calls = 0

    def commit(self, **kwargs: object) -> tuple[ObjectRef, ...]:
        del kwargs
        self.calls += 1
        return self.references


class _AttachmentRunner:
    def __init__(self, result_ref: ObjectRef) -> None:
        self.result = _OutcomeValue(
            result_ref,
            outcome=AttachmentSubgraphOutcomeV2.SUCCEEDED,
            reason_codes=(),
            work_result_refs=(_ref("agent-result-envelope"),),
        )

    async def run(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(subgraph_result=self.result)


class _CriteriaRunner:
    def __init__(self, result_ref: ObjectRef) -> None:
        self.result = _OutcomeValue(
            result_ref,
            outcome=CriteriaRubricOutcomeV2.SUCCEEDED,
            reason_codes=(),
        )
        self.envelope = _RefValue(_ref("agent-result-envelope"))

    async def run(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(
            result=self.result,
            envelope=self.envelope,
        )


class _GradingRunner:
    def __init__(self, result_ref: ObjectRef) -> None:
        self.result = _OutcomeValue(
            result_ref,
            outcome=GradingDesignOutcomeV2.SUCCEEDED,
            reason_codes=(),
            validation_ref=_ref("judge-design-validation"),
        )
        self.envelope = _RefValue(_ref("agent-result-envelope"))

    async def run(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(
            result=self.result,
            envelope=self.envelope,
        )


class _QualityAgent:
    def __init__(self, result_ref: ObjectRef) -> None:
        self.result = _OutcomeValue(
            result_ref,
            outcome=AttachmentQualityOutcomeV2.PASSED,
            reason_codes=(),
            validator_result_refs=(_ref("quality-validation"),),
        )

    def assess(self, **kwargs: object) -> _OutcomeValue:
        del kwargs
        return self.result


class _BatchRuntime:
    def __init__(self, result_ref: ObjectRef) -> None:
        self.aggregate = _OutcomeValue(
            result_ref,
            outcome=CandidateDatasetOutcomeV2.COMPLETE,
            reason_codes=(),
            batch_quality_ref=_ref("batch-quality-report"),
        )

    async def advance(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(aggregate=self.aggregate)


class _PlanReviewService:
    def __init__(self, result_ref: ObjectRef) -> None:
        self.result = _RefValue(
            result_ref,
            decision_ref=_ref("plan-decision"),
        )

    def require_resumed_review(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(result=self.result)


class _PlanReviewPreparation:
    def __init__(self, plan_ref: ObjectRef) -> None:
        self.plan_ref = plan_ref
        self.calls = 0

    async def prepare(
        self,
        request: object,
        *,
        audit: object,
    ) -> SimpleNamespace:
        del request, audit
        self.calls += 1
        return SimpleNamespace(
            run_id="factory-run://prepared-review",
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
            plan_ref=self.plan_ref,
        )


class _DeliveryRuntime:
    def __init__(self, result_ref: ObjectRef) -> None:
        self.result = SimpleNamespace(
            manifest=_RefValue(result_ref),
            completion=_RefValue(_ref("factory-run-completion")),
            inventory=_RefValue(_ref("candidate-dataset-inventory")),
        )

    def advance(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        return self.result


class _RefValue:
    def __init__(self, reference: ObjectRef, **values: object) -> None:
        self.reference = reference
        for name, value in values.items():
            setattr(self, name, value)

    def to_ref(self) -> ObjectRef:
        return self.reference


class _OutcomeValue(_RefValue):
    pass


@dataclass(frozen=True)
class _AdapterCase:
    capability_id: str
    request: BaseModel
    provider: object
    expected_ref: ObjectRef


def _request(
    capability_id: str,
    materials: _Materials,
) -> BaseModel:
    values: dict[str, ObjectRef] = {}

    def material(
        name: str,
        object_type: str,
        value: object | None = None,
    ) -> ObjectRef:
        reference = _ref(object_type, name)
        values[name] = reference
        materials.values[reference] = value or object()
        return reference

    if capability_id == "capability.requirement-planning":
        return RequirementPlanningCapabilityRequestV1.create(
            plan_ref=material("plan", "dataset-build-plan"),
            policy_ref=material("policy", "factory-run-policy"),
            audit=_audit(),
        )
    if capability_id == "capability.trace-ingestion":
        return TraceIngestionCapabilityRequestV1.create(
            trace_source_ref=material("trace", "trace-source"),
            job_id="job-parity",
            attempt=1,
            audit=_audit(),
        )
    if capability_id == "capability.task-authoring":
        return TaskAuthoringCapabilityRequestV1.create(
            decision_ref=material(
                "decision",
                "trace-candidate-decision",
            ),
            extracted_prompt_ref=material(
                "prompt",
                "extracted-user-prompt",
            ),
            intent_ref=material("intent", "inferred-user-intent"),
            rewrite_ref=material(
                "rewrite",
                "task-rewrite-candidate",
            ),
            requirement_ref=material(
                "requirement",
                "evaluation-requirement-spec",
            ),
            audit=_audit(),
        )
    if capability_id == "capability.attachment-reconstruction":
        return AttachmentReconstructionCapabilityRequestV1.create(
            run_id="factory-run://parity",
            plan_ref=material(
                "attachment-plan",
                "attachment-generation-plan",
            ),
            compiled_plan_ref=material(
                "compiled-attachment",
                "compiled-attachment-generation-plan",
            ),
            preparation_ref=material(
                "preparation",
                "attachment-r5-preparation",
            ),
            prior_batch_ref=None,
            lease_duration_seconds=600,
            audit=_audit(),
        )
    if capability_id == "capability.criteria-rubric":
        bindings_ref = _ref(
            "evaluator-binding-definitions",
            "bindings",
        )
        materials.many[bindings_ref] = (object(),)
        return CriteriaRubricCapabilityRequestV1.create(
            run_id="factory-run://parity",
            plan_ref=material(
                "criteria-plan",
                "criteria-rubric-plan",
            ),
            compiled_plan_ref=material(
                "compiled-criteria",
                "compiled-criteria-rubric-plan",
            ),
            task_draft_ref=material("task-draft", "task-draft"),
            attachment_quality_ref=material(
                "attachment-quality",
                "attachment-quality-assessment",
            ),
            solvability_ref=material(
                "solvability",
                "solvability-assessment",
            ),
            binding_definitions_ref=bindings_ref,
            tool_catalog_ref=material(
                "tool-catalog",
                "tool-capability-catalog",
            ),
            audit=_audit(),
        )
    if capability_id == "capability.grading-design":
        authorizations_ref = _ref(
            "model-domain-authorizations",
            "authorizations",
        )
        materials.many[authorizations_ref] = ()
        return GradingDesignCapabilityRequestV1.create(
            run_id="factory-run://parity",
            plan_ref=material("grading-plan", "grading-design-plan"),
            compiled_plan_ref=material(
                "compiled-grading",
                "compiled-grading-design-plan",
            ),
            criteria_result_ref=material(
                "criteria-result",
                "criteria-rubric-result",
            ),
            criteria_route_ref=material(
                "criteria-route",
                "model-route-decision",
            ),
            rubric_set_ref=material("rubric", "rubric-set"),
            evaluator_spec_ref=material(
                "evaluator",
                "evaluator-spec",
            ),
            reference_policy_ref=material(
                "reference-policy",
                "reference-policy",
            ),
            tool_policy_ref=material("tool-policy", "tool-policy"),
            model_authorizations_ref=authorizations_ref,
            evaluated_at=datetime(2026, 8, 18, tzinfo=UTC),
            audit=_audit(),
        )
    if capability_id == "capability.quality-review":
        group_ref = material("group", "attachment-group-result")
        envelope_ref = material("envelope", "agent-result-envelope")
        return AttachmentQualityCapabilityRequestV1.create(
            subgraph_result_ref=material(
                "subgraph",
                "attachment-subgraph-result",
            ),
            group_result_refs=(group_ref,),
            work_envelope_refs=(envelope_ref,),
            source_validation_ref=material(
                "source-validation",
                "deterministic-item-validation-result",
            ),
            item_quality_ref=material(
                "item-quality",
                "item-quality-compilation-result",
            ),
            audit=_audit(),
        )
    if capability_id == "capability.batch-quality":
        sources_ref = _ref("batch-quality-sources", "sources")
        materials.many[sources_ref] = ()
        return BatchQualityCapabilityRequestV1.create(
            dataset_run_id="factory-run://parity",
            core_vertical_result_ref=_ref("core-vertical-result"),
            sources_ref=sources_ref,
            rejected_binding_refs=(),
            blocked_binding_refs=(),
            resolved_job_work_graph_ref=material(
                "graph",
                "resolved-job-work-graph",
            ),
            duplicate_policy_ref=material(
                "duplicate-policy",
                "duplicate-detection-policy",
            ),
            cross_item_policy_ref=material(
                "cross-policy",
                "cross-item-safety-policy",
            ),
            batch_policy_ref=material(
                "batch-policy",
                "batch-quality-policy",
            ),
            audit=_audit(),
        )
    if capability_id == "capability.plan-review":
        return PlanReviewCapabilityRequestV1.create(
            run_id="factory-run://parity",
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
            plan_ref=_ref("attachment-generation-plan"),
            audit=_audit(),
        )
    if capability_id == "capability.delivery":
        projection_ref = material(
            "projection",
            "release-projection-result",
            SimpleNamespace(
                projection=_RefValue(
                    _ref("release-projection-result", "projection"),
                ),
            ),
        )
        export_ref = material(
            "export",
            "authorized-candidate-export",
        )
        return DeliveryCapabilityRequestV1.create(
            dataset_run_id="factory-run://parity",
            aggregate_ref=material(
                "aggregate",
                "factory-dataset-aggregate-result",
            ),
            candidate_projection_refs=(projection_ref,),
            export_refs=(export_ref,),
            policy_ref=material("delivery-policy", "factory-run-policy"),
            audit=_audit(),
        )
    raise AssertionError(capability_id)


def _case(capability_id: str) -> _AdapterCase:
    registration = build_generic_agent_trace_pack(audit=_audit())
    expected_ref = _ref(
        ("r4-task-contract-set" if capability_id == "capability.task-authoring" else "owner-result"),
        capability_id.removeprefix("capability."),
        sha256=RESULT_HASH,
    )
    materials = _Materials(values={}, many={})
    request = _request(capability_id, materials)
    if capability_id == "capability.requirement-planning":
        provider = RequirementPlanningCapabilityProvider(
            registration,
            compiler=_Compiler(expected_ref),  # type: ignore[arg-type]
            materials=materials,
        )
    elif capability_id == "capability.trace-ingestion":
        provider = TraceIngestionCapabilityProvider(
            registration,
            service=_TraceService(expected_ref),  # type: ignore[arg-type]
            materials=materials,
        )
    elif capability_id == "capability.task-authoring":
        provider = TaskAuthoringCapabilityProvider(
            registration,
            bridge=_TaskBridge(expected_ref),  # type: ignore[arg-type]
            materials=materials,
        )
    elif capability_id == "capability.attachment-reconstruction":
        provider = AttachmentReconstructionCapabilityProvider(
            registration,
            runner=_AttachmentRunner(expected_ref),  # type: ignore[arg-type]
            materials=materials,
            facade=object(),  # type: ignore[arg-type]
        )
    elif capability_id == "capability.criteria-rubric":
        provider = CriteriaRubricCapabilityProvider(
            registration,
            runner=_CriteriaRunner(expected_ref),  # type: ignore[arg-type]
            materials=materials,
        )
    elif capability_id == "capability.grading-design":
        provider = GradingDesignCapabilityProvider(
            registration,
            runner=_GradingRunner(expected_ref),  # type: ignore[arg-type]
            materials=materials,
        )
    elif capability_id == "capability.quality-review":
        provider = AttachmentQualityCapabilityProvider(
            registration,
            agent=_QualityAgent(expected_ref),  # type: ignore[arg-type]
            materials=materials,
        )
    elif capability_id == "capability.batch-quality":
        provider = BatchQualityCapabilityProvider(
            registration,
            runtime=_BatchRuntime(expected_ref),  # type: ignore[arg-type]
            duplicate_facade=object(),  # type: ignore[arg-type]
            cross_item_facade=object(),  # type: ignore[arg-type]
            materials=materials,
        )
    elif capability_id == "capability.plan-review":
        provider = PlanReviewCapabilityProvider(
            registration,
            service=_PlanReviewService(expected_ref),  # type: ignore[arg-type]
        )
    else:
        provider = DeliveryCapabilityProvider(
            registration,
            runtime=_DeliveryRuntime(expected_ref),  # type: ignore[arg-type]
            materials=materials,
        )
    return _AdapterCase(
        capability_id=capability_id,
        request=request,
        provider=provider,
        expected_ref=expected_ref,
    )


def _runtime(
    case: _AdapterCase,
) -> tuple[
    HarnessCapabilityRuntime,
    object,
    ExecutionAuthorityV1,
    CapabilityInvocationContextV1,
    CapabilityPermissionScopeV1,
    tuple[ArtifactEnvelopeV1, ...],
]:
    provider = case.provider
    registration = provider.registration if hasattr(provider, "registration") else None
    if registration is None:
        registration = build_generic_agent_trace_pack(audit=_audit())
    definitions = {value.capability_id: value for value in registration.capability_definitions}
    bindings = {value.capability_definition_ref: value for value in registration.provider_bindings}
    providers: list[object] = []
    for provider_type in PROVIDER_CLASSES:
        definition = definitions[provider_type.CAPABILITY_ID]
        binding = bindings[definition.to_ref()]
        if case.capability_id == provider_type.CAPABILITY_ID:
            providers.append(provider)
        else:
            providers.append(
                _PlaceholderProvider(
                    provider_binding_ref=binding.to_ref(),
                    implementation_id=binding.implementation_id,
                    request_model=provider_type.REQUEST_MODEL,
                    request_schema_ref=definition.request_schema_ref,
                    result_model=provider_type.RESULT_MODEL,
                    result_schema_ref=definition.result_schema_ref,
                ),
            )
    registry = CapabilityRuntimeRegistry(
        registration=registration,
        providers=tuple(providers),  # type: ignore[arg-type]
    )
    binding = registry.bind_consumer(
        capability_id=case.capability_id,
        consumer_id=f"consumer.{case.capability_id}",
        consumer_version="v1",
        consumer_kind=CapabilityConsumerKindV1.INTERNAL_SERVICE,
        projection_ref=None,
        audit=_audit(),
    )
    definition = definitions[case.capability_id]
    artifacts = tuple(
        ArtifactEnvelopeV1.create(
            artifact_id=f"artifact.{role}",
            subject_ref=_ref("owner-input", role),
            schema_ref=definition.request_schema_ref,
            content_ref=_ref("owner-input", role),
            media_type="application/json",
            modality=ArtifactModalityV1.DOCUMENT,
            domain_tags=("generic-agent-trace",),
            semantic_role=role,
            purpose="evaluation-data-production",
            classification="INTERNAL",
            lineage_refs=(),
            producer_capability_ref=definition.to_ref(),
            producer_task_ref=None,
            validation_refs=(),
            revision=1,
            predecessor_envelope_ref=None,
            audit=_audit(),
        )
        for role in definition.input_artifact_roles
    )
    principal_ref = _ref("principal", "adapter")
    policy = registration.permission_policies[0]
    artifact_refs = sorted_refs(value.to_ref() for value in artifacts)
    grant = MemberExecutionGrantV1(
        member_id="member-adapter",
        principal_ref=principal_ref,
        capability_definition_refs=(definition.to_ref(),),
        provider_binding_refs=(bindings[definition.to_ref()].to_ref(),),
        task_ids=("task-adapter",),
        data_scope_refs=artifact_refs,
        data_purposes=("evaluation-data-production",),
        data_classifications=("INTERNAL",),
        allowed_side_effects=(definition.side_effect,),
    )
    authority = ExecutionAuthorityV1.create(
        authority_id=f"execution-authority.{case.capability_id}",
        authority_version=1,
        predecessor_authority_ref=None,
        team_id="team-adapter",
        team_incarnation_id="team-adapter-001",
        roster_ref=_ref("team-roster", version="v1"),
        task_graph_ref=_ref("team-task-graph", version="v1"),
        permission_policy_ref=policy.to_ref(),
        grants=(grant,),
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        used_model_requests=0,
        used_model_tokens=0,
        used_cost_micro_usd=0,
        audit=_audit(),
    )
    context = CapabilityInvocationContextV1(
        session_ref=_ref("harness-session", version="v1"),
        authority_ref=authority.to_ref(),
        principal_ref=principal_ref,
        data_purpose="evaluation-data-production",
        data_classification="INTERNAL",
        idempotency_key=f"parity-{case.capability_id}",
    )
    permission_scope = CapabilityPermissionScopeV1(
        member_id="member-adapter",
        task_id="task-adapter",
        data_scope_refs=artifact_refs,
        local_execution=True,
        deterministic=True,
    )
    return (
        HarnessCapabilityRuntime(registry),
        binding,
        authority,
        context,
        permission_scope,
        artifacts,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capability_id",
    tuple(value.CAPABILITY_ID for value in PROVIDER_CLASSES),
)
async def test_owner_adapter_runtime_canonical_ref_parity(
    capability_id: str,
) -> None:
    case = _case(capability_id)
    runtime, binding, authority, context, scope, artifacts = _runtime(case)

    invocation = await runtime.invoke(
        consumer=binding,  # type: ignore[arg-type]
        context=context,
        permission_scope=scope,
        authority=authority,
        request=case.request,
        input_artifacts=artifacts,
        audit=_audit(),
    )

    assert invocation.result.outcome is CapabilityInvocationOutcomeV1.SUCCEEDED
    assert invocation.result.canonical_result_ref == case.expected_ref
    assert invocation.output_artifacts[0].subject_ref == case.expected_ref


@pytest.mark.asyncio
async def test_trace_adapter_commits_candidate_authority_inside_runtime() -> None:
    case = _case("capability.trace-ingestion")
    registration = case.provider.registration  # type: ignore[attr-defined]
    materials = _Materials(values={}, many={})
    request = _request("capability.trace-ingestion", materials)
    candidate_refs = (
        _ref("trace-candidate-decision", "candidate"),
        _ref("extracted-user-prompt", "candidate"),
        _ref("inferred-user-intent", "candidate"),
        _ref("task-rewrite-candidate", "candidate"),
    )
    candidate = _TraceCandidate(candidate_refs)
    preparer = _TraceCandidatePreparer(candidate)
    store = _TraceCandidateStore()
    provider = TraceIngestionCapabilityProvider(
        registration,
        service=_TraceService(case.expected_ref),  # type: ignore[arg-type]
        materials=materials,
        candidate_preparer=preparer,  # type: ignore[arg-type]
        candidate_materials=store,  # type: ignore[arg-type]
    )
    candidate_case = _AdapterCase(
        capability_id=case.capability_id,
        request=request,
        provider=provider,
        expected_ref=case.expected_ref,
    )
    runtime, binding, authority, context, scope, artifacts = _runtime(
        candidate_case,
    )

    invocation = await runtime.invoke(
        consumer=binding,  # type: ignore[arg-type]
        context=context,
        permission_scope=scope,
        authority=authority,
        request=request,
        input_artifacts=artifacts,
        audit=_audit(),
    )

    assert preparer.calls == store.calls == 1
    assert set(candidate_refs).issubset(
        invocation.result.validation_refs,
    )
    assert any(value.object_type == "permission-decision" for value in invocation.result.validation_refs)
    with pytest.raises(ValueError, match="must appear together"):
        TraceIngestionCapabilityProvider(
            registration,
            service=_TraceService(case.expected_ref),  # type: ignore[arg-type]
            materials=materials,
            candidate_preparer=preparer,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_task_adapter_projects_result_into_factory_authority() -> None:
    case = _case("capability.task-authoring")
    registration = case.provider.registration  # type: ignore[attr-defined]
    materials = _Materials(values={}, many={})
    request = _request("capability.task-authoring", materials)
    projection_refs = (
        _ref("factory-item-stage-head", "task-authoring"),
        _ref("task-authoring-material", "task-authoring"),
    )
    projection = _TaskProjection(projection_refs)
    provider = TaskAuthoringCapabilityProvider(
        registration,
        bridge=_TaskBridge(case.expected_ref),  # type: ignore[arg-type]
        materials=materials,
        result_projection=projection,  # type: ignore[arg-type]
    )
    projected_case = _AdapterCase(
        capability_id=case.capability_id,
        request=request,
        provider=provider,
        expected_ref=case.expected_ref,
    )
    runtime, binding, authority, context, scope, artifacts = _runtime(
        projected_case,
    )

    invocation = await runtime.invoke(
        consumer=binding,  # type: ignore[arg-type]
        context=context,
        permission_scope=scope,
        authority=authority,
        request=request,
        input_artifacts=artifacts,
        audit=_audit(),
    )

    assert projection.calls == 1
    assert set(projection_refs).issubset(
        invocation.result.validation_refs,
    )


@pytest.mark.asyncio
async def test_plan_review_adapter_prepares_plan_inside_runtime() -> None:
    case = _case("capability.plan-review")
    registration = case.provider.registration  # type: ignore[attr-defined]
    prepared_plan = _ref(
        "attachment-generation-plan",
        "prepared-review",
    )
    preparation = _PlanReviewPreparation(prepared_plan)
    provider = PlanReviewCapabilityProvider(
        registration,
        service=_PlanReviewService(case.expected_ref),  # type: ignore[arg-type]
        review_preparation=preparation,  # type: ignore[arg-type]
    )
    prepared_case = _AdapterCase(
        capability_id=case.capability_id,
        request=case.request,
        provider=provider,
        expected_ref=case.expected_ref,
    )
    runtime, binding, authority, context, scope, artifacts = _runtime(
        prepared_case,
    )

    invocation = await runtime.invoke(
        consumer=binding,  # type: ignore[arg-type]
        context=context,
        permission_scope=scope,
        authority=authority,
        request=case.request,
        input_artifacts=artifacts,
        audit=_audit(),
    )

    assert preparation.calls == 1
    assert invocation.result.outcome is (CapabilityInvocationOutcomeV1.SUCCEEDED)
