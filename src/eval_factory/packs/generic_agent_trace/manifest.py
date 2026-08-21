from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from eval_factory.blueprints.models import (
    ArtifactSchemaBindingV1,
    BlueprintBudgetV1,
    BlueprintCapabilityBindingV1,
    BlueprintDataFlowEdgeV1,
    EvaluationBlueprintV1,
)
from eval_factory.contracts.agent_system_v2 import (
    AttachmentGenerationPlanV2,
    AttachmentQualityAssessmentV2,
    AttachmentSubgraphResultV2,
    CompiledDatasetBuildPlanV2,
    CriteriaRubricPlanV2,
    CriteriaRubricResultV2,
    DatasetBuildPlanV2,
    GradingDesignPlanV2,
    GradingDesignResultV2,
    PlanReviewRequestV2,
    PlanReviewResultV2,
    TaskRewriteCandidateV2,
    TaskRewritePlanV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetDeliveryManifestV2,
    DatasetDeliveryPlanV2,
    FactoryDatasetAggregateResultV2,
)
from eval_factory.contracts.orchestration import TraceSourceRef
from eval_factory.contracts.task_v2 import R4TaskContractSetV2
from eval_factory.contracts.trace import TraceEnvelope
from eval_factory.harness.artifacts import ArtifactModalityV1
from eval_factory.harness.capability import (
    CapabilityConsumerKindV1,
    CapabilityDefinitionV1,
    CapabilityIdempotencyV1,
    CapabilityProviderBindingV1,
    CapabilityProviderKindV1,
    CapabilitySideEffectV1,
)
from eval_factory.harness.composition import (
    EvalPackManifestV1,
    HarnessCompositionV1,
    PackAgentProfileV1,
    StaticPackRegistrationV1,
    StaticPackRegistry,
)
from eval_factory.harness.contracts import (
    schema_ref_for,
    sorted_refs,
    static_object_ref,
)
from eval_factory.harness.interaction import BalancedAutonomyPolicyV1
from eval_factory.harness.projections import (
    ProjectionDefinitionV1,
    ProjectionVisibilityV1,
)
from eval_factory.harness.session_models import SessionEventV1
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    AttachmentQualityCapabilityRequestV1,
    AttachmentQualityCapabilityRequestV2,
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
from eval_factory.team.models import MemberContextProjectionV1, TeamCheckpointV1

GENERIC_AGENT_TRACE_PACK_ID = "generic-agent-trace-pack"
GENERIC_AGENT_TRACE_PACK_LEGACY_VERSION = "1.0.0"
GENERIC_AGENT_TRACE_PACK_VERSION = "1.1.0"
GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION = "1.2.0"
GENERIC_AGENT_EVAL_PROFILE_ID = "generic-agent-eval"

_ALL_CONSUMERS = tuple(CapabilityConsumerKindV1)
_SERVICE_CONSUMERS = (
    CapabilityConsumerKindV1.AGENT_TOOL,
    CapabilityConsumerKindV1.CLI,
    CapabilityConsumerKindV1.GRAPH_NODE,
    CapabilityConsumerKindV1.HTTP_API,
    CapabilityConsumerKindV1.INTERNAL_SERVICE,
    CapabilityConsumerKindV1.UI_WORKSPACE,
)


class GenericAgentTracePackManifestV1(EvalPackManifestV1):
    """Named built-in manifest type using the shared Eval Pack schema."""


@dataclass(frozen=True, slots=True)
class _CapabilitySpec:
    capability_id: str
    request_model: type[BaseModel]
    result_model: type[BaseModel]
    input_roles: tuple[str, ...]
    output_role: str
    provider_kind: CapabilityProviderKindV1
    side_effect: CapabilitySideEffectV1
    idempotency: CapabilityIdempotencyV1
    implementation_id: str


def build_generic_agent_trace_pack(
    *,
    audit: ContractAudit,
) -> StaticPackRegistrationV1:
    return _build_generic_agent_trace_pack(
        capability_specs=_capability_specs(),
        pack_version=GENERIC_AGENT_TRACE_PACK_VERSION,
        audit=audit,
    )


def build_generic_agent_trace_pack_v1_0(
    *,
    audit: ContractAudit,
) -> StaticPackRegistrationV1:
    return _build_generic_agent_trace_pack(
        capability_specs=_legacy_capability_specs(),
        pack_version=GENERIC_AGENT_TRACE_PACK_LEGACY_VERSION,
        audit=audit,
    )


def build_generic_agent_trace_execution_pack(
    *,
    audit: ContractAudit,
) -> StaticPackRegistrationV1:
    return _build_generic_agent_trace_pack(
        capability_specs=_execution_capability_specs(),
        pack_version=GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION,
        audit=audit,
    )


def _build_generic_agent_trace_pack(
    *,
    capability_specs: tuple[_CapabilitySpec, ...],
    pack_version: str,
    audit: ContractAudit,
) -> StaticPackRegistrationV1:
    policy = BalancedAutonomyPolicyV1.create(
        policy_id="permission-policy.generic-agent-eval",
        audit=audit,
    )
    schema_refs = _schema_refs(capability_specs)
    definitions = tuple(
        _build_capability(
            spec,
            schema_refs=schema_refs,
            capability_version=pack_version,
            audit=audit,
        )
        for spec in capability_specs
    )
    providers = tuple(
        CapabilityProviderBindingV1.create(
            provider_id=f"provider.{spec.capability_id}",
            provider_version=pack_version,
            capability_definition_ref=definition.to_ref(),
            implementation_id=spec.implementation_id,
            provider_kind=spec.provider_kind,
            supported_consumers=_SERVICE_CONSUMERS,
            policy_refs=(policy.to_ref(),),
            audit=audit,
        )
        for spec, definition in zip(capability_specs, definitions, strict=True)
    )
    projections = _build_projections(
        schema_refs=schema_refs,
        capability_ids=frozenset(value.capability_id for value in definitions),
        audit=audit,
    )
    profiles = _build_profiles(
        definitions=definitions,
        profile_version=pack_version,
        context_projection_ref=next(
            projection.to_ref()
            for projection in projections
            if projection.projection_id == "projection.team-workbench"
        ),
        audit=audit,
    )
    graph_template_ref = static_object_ref(
        object_type="graph-template",
        object_id="graph-template://generic-agent-eval/v1",
        object_version="v1",
        payload={
            "pack_id": GENERIC_AGENT_TRACE_PACK_ID,
            "capability_ids": [definition.capability_id for definition in definitions],
            "topology": (
                "requirement-trace-task-quality-delivery"
                if pack_version == GENERIC_AGENT_TRACE_PACK_LEGACY_VERSION
                else "requirement-trace-task-specialists-batch-delivery"
            ),
        },
    )
    blueprint_template_ref = static_object_ref(
        object_type="blueprint-template",
        object_id="blueprint-template://generic-agent-eval/v1",
        object_version="v1",
        payload=EvaluationBlueprintV1.model_json_schema(),
    )
    delivery_adapter_ref = static_object_ref(
        object_type="delivery-adapter",
        object_id="delivery-adapter://generic-agent-candidate-dataset/v1",
        object_version="v1",
        payload={
            "implementation_id": "eval-factory.agent-system.candidate-output",
            "formats": ["json", "jsonl"],
        },
    )
    certification_ref = static_object_ref(
        object_type="pack-certification-suite",
        object_id="pack-certification-suite://generic-agent-trace/v1",
        object_version="v1",
        payload={
            "contract": "standalone-agent-parity",
            "claim_scope": "STAGE_0_CONTRACT_ONLY",
        },
    )
    manifest = GenericAgentTracePackManifestV1.create(
        pack_id=GENERIC_AGENT_TRACE_PACK_ID,
        pack_version=pack_version,
        artifact_schema_refs=tuple(schema_refs.values()),
        capability_definition_refs=tuple(value.to_ref() for value in definitions),
        provider_binding_refs=tuple(value.to_ref() for value in providers),
        agent_profile_refs=tuple(value.to_ref() for value in profiles),
        graph_template_ref=graph_template_ref,
        blueprint_template_ref=blueprint_template_ref,
        permission_policy_refs=(policy.to_ref(),),
        projection_refs=tuple(value.to_ref() for value in projections),
        delivery_adapter_refs=(delivery_adapter_ref,),
        certification_refs=(certification_ref,),
        audit=audit,
    )
    composition = HarnessCompositionV1.create(
        profile_id=GENERIC_AGENT_EVAL_PROFILE_ID,
        profile_version=pack_version,
        pack_manifest_refs=(manifest.to_ref(),),
        capability_definition_refs=manifest.capability_definition_refs,
        provider_binding_refs=manifest.provider_binding_refs,
        agent_profile_refs=manifest.agent_profile_refs,
        permission_policy_refs=manifest.permission_policy_refs,
        projection_refs=manifest.projection_refs,
        blueprint_template_ref=manifest.blueprint_template_ref,
        audit=audit,
    )
    return StaticPackRegistrationV1(
        manifest=manifest,
        capability_definitions=definitions,
        provider_bindings=providers,
        agent_profiles=profiles,
        permission_policies=(policy,),
        projections=projections,
        composition=composition,
    )


def generic_agent_trace_pack_registry(
    *,
    audit: ContractAudit,
) -> StaticPackRegistry:
    return StaticPackRegistry(
        (
            build_generic_agent_trace_pack_v1_0(audit=audit),
            build_generic_agent_trace_pack(audit=audit),
            build_generic_agent_trace_execution_pack(audit=audit),
        ),
    )


def build_generic_agent_evaluation_blueprint(
    *,
    registration: StaticPackRegistrationV1,
    requirement_ref: ObjectRef,
    team_ref: ObjectRef,
    roster_ref: ObjectRef,
    task_graph_ref: ObjectRef,
    authority_ref: ObjectRef,
    evaluator_refs: tuple[ObjectRef, ...] | None = None,
    review_gate_refs: tuple[ObjectRef, ...] | None = None,
    budget: BlueprintBudgetV1 | None = None,
    delivery_profile_ref: ObjectRef | None = None,
    audit: ContractAudit,
) -> EvaluationBlueprintV1:
    if (
        registration.manifest.pack_id != GENERIC_AGENT_TRACE_PACK_ID
        or registration.manifest.pack_version
        not in {
            GENERIC_AGENT_TRACE_PACK_VERSION,
            GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION,
        }
    ):
        raise ValueError("generic Agent Blueprint requires the built-in Pack version")
    definitions = {definition.capability_id: definition for definition in registration.capability_definitions}
    providers = {provider.capability_definition_ref: provider for provider in registration.provider_bindings}
    capability_bindings = _blueprint_capability_bindings(
        definitions=definitions,
        providers=providers,
        execution_mode=(registration.manifest.pack_version == GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION),
    )
    data_flow_edges = _blueprint_data_flow_edges(
        definitions,
        execution_mode=(registration.manifest.pack_version == GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION),
    )
    schema_bindings = tuple(
        ArtifactSchemaBindingV1(
            semantic_role=definition.output_artifact_roles[0],
            schema_ref=definition.result_schema_ref,
            modality=ArtifactModalityV1.DOCUMENT,
            media_types=("application/json",),
        )
        for definition in sorted(
            definitions.values(),
            key=lambda value: value.output_artifact_roles[0],
        )
    )
    return EvaluationBlueprintV1.create(
        blueprint_id="evaluation-blueprint.generic-agent-eval",
        blueprint_version=1,
        requirement_ref=requirement_ref,
        composition_ref=registration.composition.to_ref(),
        pack_manifest_refs=(registration.manifest.to_ref(),),
        artifact_schema_bindings=schema_bindings,
        capability_bindings=capability_bindings,
        data_flow_edges=data_flow_edges,
        team_ref=team_ref,
        roster_ref=roster_ref,
        task_graph_ref=task_graph_ref,
        authority_ref=authority_ref,
        graph_template_ref=registration.manifest.graph_template_ref,
        execution_provider_refs=registration.manifest.provider_binding_refs,
        evaluator_refs=evaluator_refs or (_default_evaluator_ref(),),
        review_gate_refs=(
            registration.manifest.permission_policy_refs if review_gate_refs is None else review_gate_refs
        ),
        budget=budget
        or BlueprintBudgetV1(
            max_model_requests=100,
            max_model_tokens=1_000_000,
            max_cost_micro_usd=100_000_000,
            max_parallel_tasks=8,
        ),
        delivery_profile_ref=delivery_profile_ref or _default_delivery_profile_ref(),
        audit=audit,
    )


def _default_evaluator_ref() -> ObjectRef:
    return static_object_ref(
        object_type="evaluator-profile",
        object_id="evaluator-profile://generic-agent-quality/v1",
        object_version="v1",
        payload={"pack_id": GENERIC_AGENT_TRACE_PACK_ID},
    )


def _default_delivery_profile_ref() -> ObjectRef:
    return static_object_ref(
        object_type="delivery-profile",
        object_id="delivery-profile://generic-agent-candidate/v1",
        object_version="v1",
        payload={
            "pack_id": GENERIC_AGENT_TRACE_PACK_ID,
            "authority": "candidate-only",
        },
    )


def _capability_specs() -> tuple[_CapabilitySpec, ...]:
    return (
        _CapabilitySpec(
            capability_id="capability.requirement-planning",
            request_model=RequirementPlanningCapabilityRequestV1,
            result_model=CompiledDatasetBuildPlanV2,
            input_roles=("evaluation-requirement",),
            output_role="compiled-build-plan",
            provider_kind=CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
            side_effect=CapabilitySideEffectV1.READ_ONLY,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.planner",
        ),
        _CapabilitySpec(
            capability_id="capability.trace-ingestion",
            request_model=TraceIngestionCapabilityRequestV1,
            result_model=TraceEnvelope,
            input_roles=("compiled-build-plan",),
            output_role="trace-evidence",
            provider_kind=CapabilityProviderKindV1.LOCAL_DETERMINISTIC,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.trace.ingestion",
        ),
        _CapabilitySpec(
            capability_id="capability.task-authoring",
            request_model=TaskAuthoringCapabilityRequestV1,
            result_model=R4TaskContractSetV2,
            input_roles=("trace-evidence",),
            output_role="task-candidate",
            provider_kind=CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.task-authoring",
        ),
        _CapabilitySpec(
            capability_id="capability.attachment-reconstruction",
            request_model=AttachmentReconstructionCapabilityRequestV1,
            result_model=AttachmentSubgraphResultV2,
            input_roles=("task-candidate",),
            output_role="attachment-package",
            provider_kind=CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.attachment-subgraph",
        ),
        _CapabilitySpec(
            capability_id="capability.criteria-rubric",
            request_model=CriteriaRubricCapabilityRequestV1,
            result_model=CriteriaRubricResultV2,
            input_roles=("attachment-package", "task-candidate"),
            output_role="evaluation-contract",
            provider_kind=CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.criteria-subgraph",
        ),
        _CapabilitySpec(
            capability_id="capability.grading-design",
            request_model=GradingDesignCapabilityRequestV1,
            result_model=GradingDesignResultV2,
            input_roles=("evaluation-contract",),
            output_role="grading-contract",
            provider_kind=CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.grading-subgraph",
        ),
        _CapabilitySpec(
            capability_id="capability.quality-review",
            request_model=AttachmentQualityCapabilityRequestV1,
            result_model=AttachmentQualityAssessmentV2,
            input_roles=("attachment-package", "task-candidate"),
            output_role="quality-assessment",
            provider_kind=CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
            side_effect=CapabilitySideEffectV1.READ_ONLY,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.attachment-quality",
        ),
        _CapabilitySpec(
            capability_id="capability.batch-quality",
            request_model=BatchQualityCapabilityRequestV1,
            result_model=FactoryDatasetAggregateResultV2,
            input_roles=(
                "evaluation-contract",
                "grading-contract",
                "quality-assessment",
                "task-candidate",
            ),
            output_role="batch-quality",
            provider_kind=CapabilityProviderKindV1.LOCAL_DETERMINISTIC,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.batch-quality",
        ),
        _CapabilitySpec(
            capability_id="capability.plan-review",
            request_model=PlanReviewCapabilityRequestV1,
            result_model=PlanReviewResultV2,
            input_roles=("compiled-build-plan",),
            output_role="review-decision",
            provider_kind=CapabilityProviderKindV1.LOCAL_DETERMINISTIC,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.plan-review",
        ),
        _CapabilitySpec(
            capability_id="capability.delivery",
            request_model=DeliveryCapabilityRequestV1,
            result_model=CandidateDatasetDeliveryManifestV2,
            input_roles=(
                "batch-quality",
                "evaluation-contract",
                "grading-contract",
                "quality-assessment",
                "review-decision",
                "task-candidate",
            ),
            output_role="candidate-dataset-delivery",
            provider_kind=CapabilityProviderKindV1.LOCAL_DETERMINISTIC,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.candidate-output",
        ),
    )


def _execution_capability_specs() -> tuple[_CapabilitySpec, ...]:
    return tuple(
        (
            _CapabilitySpec(
                capability_id=spec.capability_id,
                request_model=(
                    AttachmentQualityCapabilityRequestV2
                    if spec.capability_id == "capability.quality-review"
                    else spec.request_model
                ),
                result_model=spec.result_model,
                input_roles=(
                    ("trace-evidence",)
                    if spec.capability_id == "capability.batch-quality"
                    else (
                        ("batch-quality", "review-decision")
                        if spec.capability_id == "capability.delivery"
                        else spec.input_roles
                    )
                ),
                output_role=spec.output_role,
                provider_kind=spec.provider_kind,
                side_effect=(
                    CapabilitySideEffectV1.REVERSIBLE_WRITE
                    if spec.capability_id == "capability.quality-review"
                    else spec.side_effect
                ),
                idempotency=spec.idempotency,
                implementation_id=spec.implementation_id,
            )
            if spec.capability_id
            in {
                "capability.quality-review",
                "capability.batch-quality",
                "capability.delivery",
            }
            else spec
        )
        for spec in _capability_specs()
    )


def _legacy_capability_specs() -> tuple[_CapabilitySpec, ...]:
    return (
        _CapabilitySpec(
            capability_id="capability.requirement-planning",
            request_model=DatasetBuildPlanV2,
            result_model=CompiledDatasetBuildPlanV2,
            input_roles=("evaluation-requirement",),
            output_role="compiled-build-plan",
            provider_kind=CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
            side_effect=CapabilitySideEffectV1.READ_ONLY,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.planner",
        ),
        _CapabilitySpec(
            capability_id="capability.trace-ingestion",
            request_model=TraceSourceRef,
            result_model=TraceEnvelope,
            input_roles=("admitted-source",),
            output_role="trace-evidence",
            provider_kind=CapabilityProviderKindV1.LOCAL_DETERMINISTIC,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.trace.ingestion",
        ),
        _CapabilitySpec(
            capability_id="capability.task-authoring",
            request_model=TaskRewritePlanV2,
            result_model=TaskRewriteCandidateV2,
            input_roles=("compiled-build-plan", "trace-evidence"),
            output_role="task-candidate",
            provider_kind=CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.task-authoring",
        ),
        _CapabilitySpec(
            capability_id="capability.attachment-reconstruction",
            request_model=AttachmentGenerationPlanV2,
            result_model=AttachmentSubgraphResultV2,
            input_roles=("task-candidate",),
            output_role="attachment-package",
            provider_kind=CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.attachment-subgraph",
        ),
        _CapabilitySpec(
            capability_id="capability.criteria-rubric",
            request_model=CriteriaRubricPlanV2,
            result_model=CriteriaRubricResultV2,
            input_roles=("attachment-package", "task-candidate"),
            output_role="evaluation-contract",
            provider_kind=CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.criteria-subgraph",
        ),
        _CapabilitySpec(
            capability_id="capability.grading-design",
            request_model=GradingDesignPlanV2,
            result_model=GradingDesignResultV2,
            input_roles=("evaluation-contract",),
            output_role="grading-contract",
            provider_kind=CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.grading-subgraph",
        ),
        _CapabilitySpec(
            capability_id="capability.quality-review",
            request_model=AttachmentSubgraphResultV2,
            result_model=AttachmentQualityAssessmentV2,
            input_roles=("attachment-package", "task-candidate"),
            output_role="quality-assessment",
            provider_kind=CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
            side_effect=CapabilitySideEffectV1.READ_ONLY,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.attachment-quality",
        ),
        _CapabilitySpec(
            capability_id="capability.plan-review",
            request_model=PlanReviewRequestV2,
            result_model=PlanReviewResultV2,
            input_roles=("compiled-build-plan",),
            output_role="review-decision",
            provider_kind=CapabilityProviderKindV1.LOCAL_DETERMINISTIC,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.plan-review",
        ),
        _CapabilitySpec(
            capability_id="capability.delivery",
            request_model=DatasetDeliveryPlanV2,
            result_model=CandidateDatasetDeliveryManifestV2,
            input_roles=(
                "evaluation-contract",
                "grading-contract",
                "quality-assessment",
                "review-decision",
                "task-candidate",
            ),
            output_role="candidate-dataset-delivery",
            provider_kind=CapabilityProviderKindV1.LOCAL_DETERMINISTIC,
            side_effect=CapabilitySideEffectV1.REVERSIBLE_WRITE,
            idempotency=CapabilityIdempotencyV1.IDEMPOTENCY_KEY_REQUIRED,
            implementation_id="eval-factory.agent-system.candidate-output",
        ),
    )


def _schema_refs(
    specs: tuple[_CapabilitySpec, ...],
) -> dict[type[BaseModel], ObjectRef]:
    result: dict[type[BaseModel], ObjectRef] = {}
    for spec in specs:
        for model_type in (spec.request_model, spec.result_model):
            result.setdefault(
                model_type,
                schema_ref_for(
                    model_type,
                    schema_id=_schema_id(model_type),
                    object_version=_model_contract_version(model_type),
                ),
            )
    result[SessionEventV1] = schema_ref_for(
        SessionEventV1,
        schema_id="harness-session-event-v1",
        object_version="v1",
    )
    result[TeamCheckpointV1] = schema_ref_for(
        TeamCheckpointV1,
        schema_id="team-checkpoint-v1",
        object_version="v1",
    )
    result[MemberContextProjectionV1] = schema_ref_for(
        MemberContextProjectionV1,
        schema_id="member-context-projection-v1",
        object_version="v1",
    )
    return result


def _build_capability(
    spec: _CapabilitySpec,
    *,
    schema_refs: dict[type[BaseModel], ObjectRef],
    capability_version: str,
    audit: ContractAudit,
) -> CapabilityDefinitionV1:
    return CapabilityDefinitionV1.create(
        capability_id=spec.capability_id,
        capability_version=capability_version,
        request_schema_ref=schema_refs[spec.request_model],
        result_schema_ref=schema_refs[spec.result_model],
        input_artifact_roles=spec.input_roles,
        output_artifact_roles=(spec.output_role,),
        data_purposes=("evaluation-data-production",),
        data_classifications=("INTERNAL", "RESTRICTED"),
        side_effect=spec.side_effect,
        idempotency=spec.idempotency,
        supported_consumers=_ALL_CONSUMERS,
        legacy_agent_capability_ref=None,
        audit=audit,
    )


def _build_projections(
    *,
    schema_refs: dict[type[BaseModel], ObjectRef],
    capability_ids: frozenset[str],
    audit: ContractAudit,
) -> tuple[ProjectionDefinitionV1, ...]:
    return (
        ProjectionDefinitionV1.create(
            projection_id="projection.member-context",
            projection_version="1.0.0",
            reducer_id="reducer.member-context",
            source_event_families=("CAPABILITY", "MESSAGE", "TEAM"),
            source_artifact_roles=("evaluation-contract", "task-candidate"),
            output_schema_ref=schema_refs[MemberContextProjectionV1],
            visibility=ProjectionVisibilityV1.MEMBER_SCOPED,
            content_safe=True,
            audit=audit,
        ),
        ProjectionDefinitionV1.create(
            projection_id="projection.session-transcript",
            projection_version="1.0.0",
            reducer_id="reducer.session-transcript",
            source_event_families=(
                "CAPABILITY",
                "CHECKPOINT",
                "INTERACTION",
                "LIFECYCLE",
                "MESSAGE",
                "TEAM",
            ),
            source_artifact_roles=(),
            output_schema_ref=schema_refs[SessionEventV1],
            visibility=ProjectionVisibilityV1.USER_SAFE,
            content_safe=True,
            audit=audit,
        ),
        ProjectionDefinitionV1.create(
            projection_id="projection.team-workbench",
            projection_version="1.0.0",
            reducer_id="reducer.team-workbench",
            source_event_families=("CHECKPOINT", "TEAM"),
            source_artifact_roles=(
                *(("batch-quality",) if "capability.batch-quality" in capability_ids else ()),
                "evaluation-contract",
                "quality-assessment",
                "task-candidate",
                "trace-evidence",
            ),
            output_schema_ref=schema_refs[TeamCheckpointV1],
            visibility=ProjectionVisibilityV1.USER_SAFE,
            content_safe=True,
            audit=audit,
        ),
    )


def _build_profiles(
    *,
    definitions: tuple[CapabilityDefinitionV1, ...],
    profile_version: str,
    context_projection_ref: ObjectRef,
    audit: ContractAudit,
) -> tuple[PackAgentProfileV1, ...]:
    by_id = {definition.capability_id: definition for definition in definitions}
    model_policy_ref = static_object_ref(
        object_type="model-routing-policy",
        object_id="model-routing-policy://generic-agent-eval/v1",
        object_version="v2",
        payload={
            "pack_id": GENERIC_AGENT_TRACE_PACK_ID,
            "policy": "governed-gateway",
        },
    )
    profiles = (
        (
            "profile.coordinator",
            "coordinator",
            (
                "capability.delivery",
                "capability.plan-review",
                "capability.requirement-planning",
            ),
            True,
        ),
        (
            "profile.requirement",
            "requirement",
            ("capability.requirement-planning",),
            False,
        ),
        (
            "profile.trace",
            "trace",
            ("capability.trace-ingestion",),
            False,
        ),
        (
            "profile.task",
            "task",
            (
                "capability.attachment-reconstruction",
                "capability.criteria-rubric",
                "capability.grading-design",
                "capability.task-authoring",
            ),
            False,
        ),
        (
            "profile.quality",
            "quality",
            (
                *(("capability.batch-quality",) if "capability.batch-quality" in by_id else ()),
                "capability.quality-review",
            ),
            False,
        ),
    )
    return tuple(
        PackAgentProfileV1.create(
            profile_id=profile_id,
            profile_version=profile_version,
            role=role,
            capability_definition_refs=sorted_refs(
                by_id[capability_id].to_ref() for capability_id in capability_ids
            ),
            context_projection_ref=context_projection_ref,
            model_policy_ref=model_policy_ref,
            coordinator=coordinator,
            audit=audit,
        )
        for profile_id, role, capability_ids, coordinator in profiles
    )


def _blueprint_capability_bindings(
    *,
    definitions: dict[str, CapabilityDefinitionV1],
    providers: dict[ObjectRef, CapabilityProviderBindingV1],
    execution_mode: bool,
) -> tuple[BlueprintCapabilityBindingV1, ...]:
    upstream = {
        "capability.requirement-planning": (),
        "capability.trace-ingestion": ("capability.requirement-planning",),
        "capability.task-authoring": ("capability.trace-ingestion",),
        "capability.attachment-reconstruction": ("capability.task-authoring",),
        "capability.criteria-rubric": (
            "capability.attachment-reconstruction",
            "capability.task-authoring",
        ),
        "capability.grading-design": ("capability.criteria-rubric",),
        "capability.quality-review": (
            "capability.attachment-reconstruction",
            "capability.task-authoring",
        ),
        "capability.batch-quality": (
            *(("capability.trace-ingestion",) if execution_mode else ()),
            "capability.criteria-rubric",
            "capability.grading-design",
            "capability.quality-review",
            "capability.task-authoring",
        ),
        "capability.plan-review": ("capability.requirement-planning",),
        "capability.delivery": (
            "capability.batch-quality",
            "capability.criteria-rubric",
            "capability.grading-design",
            "capability.plan-review",
            "capability.quality-review",
            "capability.task-authoring",
        ),
    }
    return tuple(
        BlueprintCapabilityBindingV1(
            capability_id=capability_id,
            capability_definition_ref=definition.to_ref(),
            provider_binding_ref=providers[definition.to_ref()].to_ref(),
            input_schema_refs=sorted_refs(
                definitions[parent_id].result_schema_ref for parent_id in upstream[capability_id]
            ),
            output_schema_refs=(definition.result_schema_ref,),
        )
        for capability_id, definition in sorted(definitions.items())
    )


def _blueprint_data_flow_edges(
    definitions: dict[str, CapabilityDefinitionV1],
    *,
    execution_mode: bool,
) -> tuple[BlueprintDataFlowEdgeV1, ...]:
    pairs = (
        ("capability.requirement-planning", "capability.plan-review"),
        ("capability.requirement-planning", "capability.trace-ingestion"),
        ("capability.trace-ingestion", "capability.task-authoring"),
        *((("capability.trace-ingestion", "capability.batch-quality"),) if execution_mode else ()),
        ("capability.task-authoring", "capability.attachment-reconstruction"),
        ("capability.task-authoring", "capability.criteria-rubric"),
        ("capability.task-authoring", "capability.quality-review"),
        ("capability.task-authoring", "capability.delivery"),
        ("capability.attachment-reconstruction", "capability.criteria-rubric"),
        ("capability.attachment-reconstruction", "capability.quality-review"),
        ("capability.criteria-rubric", "capability.grading-design"),
        ("capability.task-authoring", "capability.batch-quality"),
        ("capability.quality-review", "capability.batch-quality"),
        ("capability.criteria-rubric", "capability.batch-quality"),
        ("capability.grading-design", "capability.batch-quality"),
        ("capability.batch-quality", "capability.delivery"),
        ("capability.criteria-rubric", "capability.delivery"),
        ("capability.grading-design", "capability.delivery"),
        ("capability.quality-review", "capability.delivery"),
        ("capability.plan-review", "capability.delivery"),
    )
    return tuple(
        BlueprintDataFlowEdgeV1(
            edge_id=f"edge.{producer_id.removeprefix('capability.')}."
            f"{consumer_id.removeprefix('capability.')}",
            producer_capability_id=producer_id,
            consumer_capability_id=consumer_id,
            artifact_schema_ref=definitions[producer_id].result_schema_ref,
        )
        for producer_id, consumer_id in sorted(pairs)
    )


def _schema_id(model_type: type[BaseModel]) -> str:
    return model_type.__name__.replace("V2", "-v2").replace("V1", "-v1").casefold()


def _model_contract_version(model_type: type[BaseModel]) -> str:
    schema_version = model_type.model_fields["schema_version"].default
    if not isinstance(schema_version, str):
        raise ValueError("contract model schema version must be a string literal")
    return schema_version.rsplit("/", maxsplit=1)[-1]


__all__ = [
    "GENERIC_AGENT_EVAL_PROFILE_ID",
    "GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION",
    "GENERIC_AGENT_TRACE_PACK_ID",
    "GENERIC_AGENT_TRACE_PACK_LEGACY_VERSION",
    "GENERIC_AGENT_TRACE_PACK_VERSION",
    "GenericAgentTracePackManifestV1",
    "build_generic_agent_evaluation_blueprint",
    "build_generic_agent_trace_execution_pack",
    "build_generic_agent_trace_pack",
    "build_generic_agent_trace_pack_v1_0",
    "generic_agent_trace_pack_registry",
]
