from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from env_mock_agent.facade import RegistryAttachmentRoutingFacade
from env_mock_agent.providers import ProviderRegistry, TextProvider
from env_mock_agent.runtimes import FakeRuntime, RuntimeRegistry
from env_mock_agent.schemas import RuntimeName
from eval_factory.attachment_planning import (
    ArtifactBuildContractDefinition,
    ArtifactBuildSpecCompiler,
    ArtifactRoutingPolicyError,
    ArtifactRoutingRequestBuilder,
    ArtifactRoutingRunner,
    AttachmentPlanningBridge,
    artifact_routing_request_carried_sha256,
)
from eval_factory.contracts import (
    ArtifactEvidenceMatrixV2,
    ArtifactEvidenceRowV2,
    ArtifactEvidenceTargetV2,
    ArtifactRouteEntryOutcomeV2,
    ArtifactRoutingAggregateOutcomeV2,
    ArtifactRoutingPolicyV2,
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
    artifact_evidence_matrix_carried_sha256,
    artifact_evidence_row_carried_sha256,
    artifact_evidence_target_carried_sha256,
    artifact_evidence_target_ref,
    artifact_routing_policy_carried_sha256,
    attachment_planning_context_ref,
    producer_storage_authorization_carried_sha256,
    producer_storage_authorization_ref,
    producer_task_view_carried_sha256,
)
from eval_factory.contracts.attachment import (
    ArtifactEvidenceRow,
    EvidenceCoverage,
    EvidenceStrength,
    ReconstructionMode,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.safety import (
    Disposition,
    EvidenceBundle,
    OriginClass,
    ProvenanceDecision,
    Visibility,
)
from eval_factory.contracts.task import AttachmentCriticality
from eval_factory.contracts.task_v2 import ProducerAttachmentRequirementV2
from eval_factory.provenance import (
    EvidenceBundleCompiler,
    EvidenceBundleCompileRequest,
    EvidenceViewEngine,
    EvidenceViewPrincipal,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewRequest,
    EvidenceViewSubject,
    ProjectionEvidenceBinding,
    evidence_bundle_ref,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
FOURTH_HASH = "d" * 64
SOURCE_TRACE_ID = "source-trace://artifact-routing/r5-05"
TRACE_IR_VERSION_ID = "trace-ir://artifact-routing/r5-05"
PRODUCER_PRINCIPAL_ID = "principal://attachment-producer/r5-05"
ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = ROOT / "evals/golden/eval_factory/attachment_routing" / "r5-05-provider-first-v1.json"


def _audit(
    *,
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 7, 28, tzinfo=UTC),
        created_by="artifact-routing-test",
        governing_versions=(
            VersionBinding(
                component="artifact-routing",
                version="r5-05",
            ),
        ),
        input_refs=input_refs,
    )


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v1",
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _routing_policy() -> ArtifactRoutingPolicyV2:
    model_refs = (
        _ref("model-profile", "sdk"),
        _ref("model-profile", "cli"),
        _ref("model-profile", "pi"),
    )
    provider_policy_ref = _ref(
        "provider-capability-policy",
        "current",
    )
    runtime_policy_ref = _ref(
        "runtime-capability-policy",
        "current",
    )
    validator_policy_ref = _ref("validator-policy", "current")
    policy = ArtifactRoutingPolicyV2(
        artifact_routing_policy_id="artifact-routing-policy://pending",
        deterministic_provider_first=True,
        approved_provider_ids=("text",),
        runtime_order=(
            "claude_agent_sdk",
            "claude_code_cli",
            "pi_rpc",
        ),
        runtime_model_profile_refs=model_refs,
        provider_capability_policy_ref=provider_policy_ref,
        runtime_capability_policy_ref=runtime_policy_ref,
        validator_policy_ref=validator_policy_ref,
        silent_degradation_allowed=False,
        policy_version="artifact-routing/r5-05-v1",
        artifact_routing_policy_sha256=HASH,
        audit=_audit(
            input_refs=tuple(
                sorted(
                    (
                        *model_refs,
                        provider_policy_ref,
                        runtime_policy_ref,
                        validator_policy_ref,
                    ),
                    key=lambda ref: (
                        ref.object_type,
                        ref.object_id,
                        ref.object_version,
                        ref.object_sha256,
                    ),
                )
            )
        ),
    )
    digest = artifact_routing_policy_carried_sha256(policy)
    return policy.model_copy(
        update={
            "artifact_routing_policy_id": f"artifact-routing-policy://sha256/{digest}",
            "artifact_routing_policy_sha256": digest,
        }
    )


def _basis(
    requirements: tuple[ProducerAttachmentRequirementV2, ...] = (),
    *,
    include_bundle_evidence: bool = False,
) -> tuple[
    object,
    ProducerTaskViewV2,
    ProducerStorageAuthorizationV2,
    EvidenceBundle,
    object,
]:
    source_ref = _ref("file-version", "unused")
    decision = ProvenanceDecision(
        provenance_decision_id="provenance-decision://artifact-routing/unused",
        subject_ref=source_ref,
        origin_class=OriginClass.PREEXISTING_WORKSPACE_INPUT,
        visibility=Visibility.STAGE_PROJECTION,
        disposition=Disposition.ALLOW_INPUT_EVIDENCE,
        rule_ids=("artifact-routing-parent/v1",),
        source_event_refs=(_ref("trace-event", "unused"),),
        confidence=1.0,
        review_required=False,
        policy_version="artifact-routing-parent/test-v1",
        subject_sha256=source_ref.object_sha256,
        audit=_audit(),
    )
    principal = EvidenceViewPrincipal(
        principal_id=PRODUCER_PRINCIPAL_ID,
        principal_type=EvidenceViewPrincipalType.ATTACHMENT_PRODUCER,
        allowed_purposes=frozenset({EvidenceViewPurpose.ATTACHMENT_PRODUCTION}),
        max_subjects=5,
        max_characters=1000,
    )
    view_result = EvidenceViewEngine().project(
        EvidenceViewRequest(
            principal=principal,
            purpose=EvidenceViewPurpose.ATTACHMENT_PRODUCTION,
            subjects=(
                EvidenceViewSubject(
                    subject_ref=source_ref,
                    decision=decision,
                    projection_text="unused safe input",
                ),
            ),
            max_characters=1000,
            audit=_audit(),
        )
    )
    bindings = (
        (
            ProjectionEvidenceBinding(
                projection_item_id=(view_result.included_items[0].projection_item_id),
                source_spans=(
                    SourceSpanRef(
                        span_id="source-span://routing/bundle",
                        source_trace_id=SOURCE_TRACE_ID,
                        raw_sha256=HASH,
                    ),
                ),
                polarity=EvidencePolarity.POSITIVE,
                capability="trace-evidence",
                capability_complete=True,
            ),
        )
        if include_bundle_evidence
        else ()
    )
    bundle = (
        EvidenceBundleCompiler()
        .compile(
            EvidenceBundleCompileRequest(
                source_trace_id=SOURCE_TRACE_ID,
                trace_ir_version_id=TRACE_IR_VERSION_ID,
                consumer_stage="attachment-producer",
                purpose="attachment-production",
                view_result=view_result,
                bindings=bindings,
                max_characters=1000,
                audit=_audit(),
            )
        )
        .evidence_bundle
    )
    authorization = ProducerStorageAuthorizationV2(
        authorization_id="producer-storage-authorization://pending",
        producer_principal_id=PRODUCER_PRINCIPAL_ID,
        purpose="ATTACHMENT_PRODUCTION",
        source_task_draft_sha256=OTHER_HASH,
        projection_policy_ref=bundle.projection_policy_ref,
        evidence_bundle_refs=(evidence_bundle_ref(bundle),),
        authorized_subject_refs=tuple(
            sorted(
                (item.subject_ref for item in bundle.evidence),
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        ),
        raw_store_access=False,
        canonical_store_access=False,
        quarantine_store_access=False,
        private_reference_store_access=False,
        credentials_issued=False,
        policy_version="producer-task-view/r4-08-v1",
        authorization_sha256=HASH,
        audit=_audit(),
    )
    authorization_digest = producer_storage_authorization_carried_sha256(authorization)
    authorization = authorization.model_copy(
        update={
            "authorization_id": (f"producer-storage-authorization://sha256/{authorization_digest}"),
            "authorization_sha256": authorization_digest,
        }
    )
    producer_view = ProducerTaskViewV2(
        producer_task_view_id="producer-task-view://pending",
        producer_task_view_version=1,
        supersedes_producer_task_view_ref=None,
        query_instruction="Inspect the supplied input-state workspace.",
        attachment_requirements=requirements,
        allowed_tools=(),
        safe_evidence_bundle_refs=(evidence_bundle_ref(bundle),),
        forbidden_outputs=("original final answer",),
        projection_policy_ref=bundle.projection_policy_ref,
        storage_authorization_ref=producer_storage_authorization_ref(authorization),
        prompt_boundary_enforcement_ref=_ref(
            "prompt-boundary-enforcement",
            "artifact-routing",
            version="prompt-injection-as-data/r2-07-v1",
        ),
        source_task_draft_sha256=OTHER_HASH,
        source_contestant_tool_policy_sha256=THIRD_HASH,
        source_contract_chain_sha256=FOURTH_HASH,
        policy_version="producer-task-view/r4-08-v1",
        producer_task_view_sha256=HASH,
        audit=_audit(),
    )
    view_digest = producer_task_view_carried_sha256(producer_view)
    producer_view = producer_view.model_copy(
        update={
            "producer_task_view_id": (f"producer-task-view://sha256/{view_digest}"),
            "producer_task_view_sha256": view_digest,
        }
    )
    context = AttachmentPlanningBridge().compile(
        producer_task_view=producer_view,
        storage_authorization=authorization,
        evidence_bundle=bundle,
        audit=_audit(),
    )
    return context, producer_view, authorization, bundle, view_result


class _UnexpectedFacade:
    async def route(self, request):
        raise AssertionError(f"facade must not be called: {request}")


async def _not_required_case():
    context, producer_view, authorization, bundle, view_result = _basis()
    policy = _routing_policy()
    request = ArtifactRoutingRequestBuilder().build(
        attachment_planning_context=context,
        producer_task_view=producer_view,
        storage_authorization=authorization,
        producer_view_result=view_result,
        producer_evidence_bundle=bundle,
        artifact_evidence_matrix=None,
        artifact_targets=(),
        routing_policy=policy,
        contract_definitions=(),
        audit=_audit(),
    )

    execution = await ArtifactRoutingRunner().run(
        request,
        facade=_UnexpectedFacade(),
        audit=_audit(),
    )
    result = ArtifactBuildSpecCompiler().compile(
        request=request,
        execution_result=execution,
        attachment_planning_context=context,
        producer_task_view=producer_view,
        storage_authorization=authorization,
        producer_view_result=view_result,
        producer_evidence_bundle=bundle,
        artifact_evidence_matrix=None,
        artifact_targets=(),
        routing_policy=policy,
        audit=_audit(),
    )
    return result, request, execution


@pytest.mark.asyncio
async def test_no_requirements_returns_not_required_without_probe_or_plan() -> None:
    result, request, execution = await _not_required_case()
    assert result.outcome is ArtifactRoutingAggregateOutcomeV2.NOT_REQUIRED
    assert result.routing_plan is None
    assert request.build_contracts == ()
    assert request.facade_requests == ()
    assert execution.executions == ()


def _requirement(
    criticality: AttachmentCriticality,
) -> ProducerAttachmentRequirementV2:
    return ProducerAttachmentRequirementV2(
        dependency_id="attachment-dependency://routing/input",
        description="Provide the input-state text file.",
        criticality=criticality,
    )


def _requirement_evidence() -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id="evidence-ref://routing/requirement",
        subject_ref=_ref("requirement-projection", "routing/input"),
        source_spans=(
            SourceSpanRef(
                span_id="source-span://routing/requirement",
                source_trace_id=SOURCE_TRACE_ID,
                raw_sha256=HASH,
            ),
        ),
        polarity=EvidencePolarity.POSITIVE,
        capability="artifact-requirement",
        capability_complete=True,
    )


def _matrix(
    context,
    producer_view: ProducerTaskViewV2,
    bundle: EvidenceBundle,
    *,
    criticality: AttachmentCriticality,
    mode: ReconstructionMode,
) -> tuple[ArtifactEvidenceMatrixV2, ArtifactEvidenceTargetV2]:
    target = ArtifactEvidenceTargetV2(
        artifact_evidence_target_id="artifact-evidence-target://pending",
        attachment_planning_context_ref=attachment_planning_context_ref(context),
        attachment_dependency_id="attachment-dependency://routing/input",
        artifact_id="artifact://routing/input",
        logical_path="inputs/source.txt",
        media_type="text/plain",
        criticality=criticality,
        requirement_evidence=(_requirement_evidence(),),
        candidate_source_refs=(),
        policy_version="artifact-evidence-mode/r5-02-v1",
        artifact_evidence_target_sha256=HASH,
        audit=_audit(),
    )
    target_digest = artifact_evidence_target_carried_sha256(target)
    target = target.model_copy(
        update={
            "artifact_evidence_target_id": (f"artifact-evidence-target://sha256/{target_digest}"),
            "artifact_evidence_target_sha256": target_digest,
        }
    )
    frozen_row = ArtifactEvidenceRow(
        artifact_id=target.artifact_id,
        logical_path=target.logical_path,
        media_type=target.media_type,
        path_evidence=EvidenceStrength.INFERRED,
        type_evidence=EvidenceStrength.INFERRED,
        structure_coverage=EvidenceCoverage.NONE,
        untainted_content_coverage=EvidenceCoverage.NONE,
        pre_mutation_coverage=EvidenceCoverage.NONE,
        provenance_confidence=0.0,
        truncation="NONE",
        criticality=criticality,
        blocking_uncertainties=(
            ("critical-capability-gap",)
            if mode is ReconstructionMode.BLOCKED and criticality is AttachmentCriticality.CRITICAL
            else ()
        ),
        evidence=tuple(
            sorted(
                (*bundle.evidence, _requirement_evidence()),
                key=lambda item: item.evidence_ref_id,
            )
        ),
        selected_mode=mode,
    )
    row = ArtifactEvidenceRowV2(
        artifact_evidence_row_id="artifact-evidence-row://pending",
        attachment_planning_context_ref=attachment_planning_context_ref(context),
        artifact_evidence_target_ref=artifact_evidence_target_ref(target),
        attachment_dependency_id=target.attachment_dependency_id,
        row=frozen_row,
        r2_row_sha256=frozen_row.canonical_sha256(),
        r2_policy_version="evidence-compilation/r2-06-v1",
        policy_version="artifact-evidence-mode/r5-02-v1",
        artifact_evidence_row_sha256=HASH,
        audit=_audit(),
    )
    row_digest = artifact_evidence_row_carried_sha256(row)
    row = row.model_copy(
        update={
            "artifact_evidence_row_id": (f"artifact-evidence-row://sha256/{row_digest}"),
            "artifact_evidence_row_sha256": row_digest,
        }
    )
    matrix = ArtifactEvidenceMatrixV2(
        artifact_evidence_matrix_id="artifact-evidence-matrix://pending",
        attachment_planning_context_ref=attachment_planning_context_ref(context),
        producer_task_view_ref=_ref(
            "producer-task-view",
            "current",
            version="v2",
            digest=producer_view.producer_task_view_sha256,
        ).model_copy(update={"object_id": producer_view.producer_task_view_id}),
        safe_evidence_bundle_ref=evidence_bundle_ref(bundle),
        source_r2_matrix_ref=_ref(
            "artifact-evidence-matrix",
            "source-r2",
            version="v1",
        ),
        rows=(row,),
        aggregate_mode=mode.value,
        r2_policy_version="evidence-compilation/r2-06-v1",
        policy_version="artifact-evidence-mode/r5-02-v1",
        artifact_evidence_matrix_sha256=HASH,
        audit=_audit(),
    )
    matrix_digest = artifact_evidence_matrix_carried_sha256(matrix)
    matrix = matrix.model_copy(
        update={
            "artifact_evidence_matrix_id": (f"artifact-evidence-matrix://sha256/{matrix_digest}"),
            "artifact_evidence_matrix_sha256": matrix_digest,
        }
    )
    return matrix, target


def _definition(
    *,
    provider_payload: bool,
) -> ArtifactBuildContractDefinition:
    return ArtifactBuildContractDefinition(
        definition_id="artifact-build-contract-definition://routing/input",
        attachment_dependency_id="attachment-dependency://routing/input",
        asset_type="txt",
        content_contract_ref=_ref(
            "artifact-content-contract",
            "routing/input",
            version="v2",
        ),
        render_contract_ref=_ref(
            "artifact-render-contract",
            "routing/input",
            version="v2",
        ),
        provider_payload_ref=(
            _ref(
                "attachment-provider-payload",
                "routing/input",
                version="v2",
            )
            if provider_payload
            else None
        ),
        source_evidence_set_ref=None,
        required_provider_capability_ids=("attachment-provider/generate/txt/v1",),
        required_runtime_tools=("write",),
        runtime_role="attachment-writer",
        runtime_resume_required=True,
        validator_ids=("secret-validator", "text-validator"),
    )


async def _route_case(
    *,
    criticality: AttachmentCriticality = AttachmentCriticality.REQUIRED,
    mode: ReconstructionMode = ReconstructionMode.PROMPT_ONLY,
    include_definition: bool = True,
    provider_payload: bool = True,
    sdk_available: bool = False,
    include_bundle_evidence: bool = False,
):
    requirement = _requirement(criticality)
    context, producer_view, authorization, bundle, view_result = _basis(
        (requirement,),
        include_bundle_evidence=include_bundle_evidence,
    )
    matrix, target = _matrix(
        context,
        producer_view,
        bundle,
        criticality=criticality,
        mode=mode,
    )
    policy = _routing_policy()
    definitions = (
        (_definition(provider_payload=provider_payload),)
        if include_definition and mode is not ReconstructionMode.BLOCKED
        else ()
    )
    request = ArtifactRoutingRequestBuilder().build(
        attachment_planning_context=context,
        producer_task_view=producer_view,
        storage_authorization=authorization,
        producer_view_result=view_result,
        producer_evidence_bundle=bundle,
        artifact_evidence_matrix=matrix,
        artifact_targets=(target,),
        routing_policy=policy,
        contract_definitions=definitions,
        audit=_audit(),
    )
    providers = ProviderRegistry()
    providers.register(TextProvider())
    runtimes = RuntimeRegistry()
    if sdk_available:
        runtimes.register(RuntimeName.CLAUDE_AGENT_SDK, FakeRuntime())
    facade = RegistryAttachmentRoutingFacade(
        providers,
        runtimes,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )
    execution = await ArtifactRoutingRunner().run(
        request,
        facade=facade,
        audit=_audit(),
    )
    compiler = ArtifactBuildSpecCompiler()
    result = compiler.compile(
        request=request,
        execution_result=execution,
        attachment_planning_context=context,
        producer_task_view=producer_view,
        storage_authorization=authorization,
        producer_view_result=view_result,
        producer_evidence_bundle=bundle,
        artifact_evidence_matrix=matrix,
        artifact_targets=(target,),
        routing_policy=policy,
        audit=_audit(),
    )
    return (
        result,
        compiler,
        request,
        execution,
        context,
        producer_view,
        authorization,
        bundle,
        view_result,
        matrix,
        target,
        policy,
    )


@pytest.mark.asyncio
async def test_provider_route_compiles_resolved_least_privilege_build_spec() -> None:
    case = await _route_case(include_bundle_evidence=True)
    result = case[0]

    assert result.outcome is ArtifactRoutingAggregateOutcomeV2.ROUTED
    assert result.routing_plan is not None
    entry = result.routing_plan.entries[0]
    assert entry.outcome is ArtifactRouteEntryOutcomeV2.ROUTED_PROVIDER
    assert entry.build_spec is not None
    assert entry.build_spec.artifact_build_spec_v2_id == (
        "artifact-build-spec://sha256/" + entry.build_spec.artifact_build_spec_v2_sha256
    )
    assert entry.build_spec.build_spec.provider_preference == ("text",)
    assert entry.build_spec.build_spec.runtime_preference == ()
    assert entry.build_spec.selected_model_profile_ref is None
    assert entry.build_spec.build_spec.authorized_evidence_refs == case[7].evidence
    assert "evidence-ref://routing/requirement" not in {
        item.evidence_ref_id for item in entry.build_spec.build_spec.authorized_evidence_refs
    }

    case[1].validate_current(
        request=case[2],
        execution_result=case[3],
        compilation_result=result,
        attachment_planning_context=case[4],
        producer_task_view=case[5],
        storage_authorization=case[6],
        producer_view_result=case[8],
        producer_evidence_bundle=case[7],
        artifact_evidence_matrix=case[9],
        artifact_targets=(case[10],),
        routing_policy=case[11],
    )

    request = case[2]
    contract = request.build_contracts[0]
    shifted_contract = contract.model_copy(
        update={
            "audit": ContractAudit(
                created_at=datetime(2026, 7, 29, tzinfo=UTC),
                created_by="shifted-routing-audit",
                governing_versions=contract.audit.governing_versions,
                input_refs=contract.audit.input_refs,
            )
        }
    )
    shifted_request = request.model_copy(update={"build_contracts": (shifted_contract,)})
    assert artifact_routing_request_carried_sha256(
        shifted_request
    ) == artifact_routing_request_carried_sha256(request)

    tampered_contract = contract.model_copy(
        update={"audit": _audit(input_refs=(_ref("private-reference", "caller-injected"),))}
    )
    tampered_request = request.model_copy(update={"build_contracts": (tampered_contract,)})
    with pytest.raises(
        ArtifactRoutingPolicyError,
        match="audit lineage",
    ):
        case[1].compile(
            request=tampered_request,
            execution_result=case[3],
            attachment_planning_context=case[4],
            producer_task_view=case[5],
            storage_authorization=case[6],
            producer_view_result=case[8],
            producer_evidence_bundle=case[7],
            artifact_evidence_matrix=case[9],
            artifact_targets=(case[10],),
            routing_policy=case[11],
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_runtime_fallback_binds_exact_sdk_model_profile() -> None:
    result = (
        await _route_case(
            provider_payload=False,
            sdk_available=True,
        )
    )[0]

    assert result.outcome is ArtifactRoutingAggregateOutcomeV2.ROUTED
    assert result.routing_plan is not None
    entry = result.routing_plan.entries[0]
    assert entry.outcome is ArtifactRouteEntryOutcomeV2.ROUTED_RUNTIME
    assert entry.build_spec is not None
    assert entry.build_spec.build_spec.provider_preference == ()
    assert entry.build_spec.build_spec.runtime_preference == ("claude_agent_sdk",)
    assert entry.build_spec.selected_model_profile_ref == _ref(
        "model-profile",
        "sdk",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("criticality", "expected"),
    [
        (
            AttachmentCriticality.REQUIRED,
            ArtifactRoutingAggregateOutcomeV2.BLOCKED_CAPABILITY,
        ),
        (
            AttachmentCriticality.OPTIONAL,
            ArtifactRoutingAggregateOutcomeV2.PARTIALLY_ROUTED,
        ),
    ],
)
async def test_capability_blocks_preserve_required_optional_semantics(
    criticality: AttachmentCriticality,
    expected: ArtifactRoutingAggregateOutcomeV2,
) -> None:
    result = (
        await _route_case(
            criticality=criticality,
            provider_payload=False,
        )
    )[0]

    assert result.outcome is expected
    assert result.routing_plan is not None
    entry = result.routing_plan.entries[0]
    assert entry.outcome is ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY
    assert entry.build_spec is None


@pytest.mark.asyncio
async def test_missing_contract_is_typed_capability_block_without_probe() -> None:
    result = (await _route_case(include_definition=False))[0]

    assert result.outcome is (ArtifactRoutingAggregateOutcomeV2.BLOCKED_CAPABILITY)
    assert result.routing_plan is not None
    entry = result.routing_plan.entries[0]
    assert entry.outcome is ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY
    assert entry.facade_route_decision is None
    assert entry.build_spec is None


@pytest.mark.asyncio
async def test_blocked_evidence_row_is_policy_block_without_build_spec() -> None:
    result = (
        await _route_case(
            criticality=AttachmentCriticality.CRITICAL,
            mode=ReconstructionMode.BLOCKED,
            include_definition=False,
        )
    )[0]

    assert result.outcome is ArtifactRoutingAggregateOutcomeV2.BLOCKED_POLICY
    assert result.routing_plan is not None
    entry = result.routing_plan.entries[0]
    assert entry.outcome is ArtifactRouteEntryOutcomeV2.BLOCKED_POLICY
    assert entry.facade_route_request_ref is None
    assert entry.build_spec is None


@pytest.mark.asyncio
async def test_routing_gold_cases_are_executable() -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    assert gold["schema_version"] == ("eval-factory/r5-05-provider-first-gold/v1")
    assert gold["claim_scope"] == ("CAPABILITY_PREFLIGHT_AND_BUILD_SPEC_ONLY")

    for case in gold["cases"]:
        scenario = case["scenario"]
        if scenario == "NOT_REQUIRED":
            result = (await _not_required_case())[0]
        else:
            kwargs: dict[str, object] = {
                "criticality": AttachmentCriticality(case["criticality"]),
            }
            if scenario == "SDK_FALLBACK":
                kwargs.update(
                    provider_payload=False,
                    sdk_available=True,
                )
            elif scenario == "ALL_UNAVAILABLE":
                kwargs.update(provider_payload=False)
            elif scenario == "MISSING_CONTRACT":
                kwargs.update(include_definition=False)
            elif scenario == "BLOCKED_EVIDENCE":
                kwargs.update(
                    mode=ReconstructionMode.BLOCKED,
                    include_definition=False,
                )
            result = (await _route_case(**kwargs))[0]

        assert result.outcome.value == case["expected_outcome"]
        if result.routing_plan is None:
            assert case["expected_entry_outcome"] is None
            assert case["expected_build_spec_count"] == 0
            continue
        assert result.routing_plan.entries[0].outcome.value == (case["expected_entry_outcome"])
        build_spec_count = sum(entry.build_spec is not None for entry in result.routing_plan.entries)
        assert build_spec_count == case["expected_build_spec_count"]
