from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_producer_task_view import (
    PRODUCER_PRINCIPAL,
    SOURCE_TRACE_ID,
    TRACE_ID,
    _draft,
    _evaluation_contract,
    _rubric_set,
    _span,
    _tool_contract,
)
from test_producer_task_view import (
    _audit as producer_audit,
)
from test_prompt_only_dependencies import (
    _case as dependency_case,
)
from test_prompt_only_dependencies import (
    _compile as compile_dependency,
)
from test_prompt_only_dependencies import _rehash_draft
from test_prompt_only_dependencies import (
    _request as dependency_request,
)

from env_mock_agent.facade import (
    FacadeObjectRef,
    PublicSourceFetchRequestV2,
    PublicSourceFetchResultV2,
    PublicSourceRetrievalFacade,
    PublicSourceRetrievalFacadeError,
    PublicSourceRetrievalFacadeFailureKind,
    PublicSourceSearchHitV2,
    PublicSourceSearchRequestV2,
    PublicSourceSearchResultV2,
    public_source_fetch_request_ref,
    public_source_fetch_result_carried_sha256,
    public_source_search_hit_carried_sha256,
    public_source_search_request_ref,
    public_source_search_result_carried_sha256,
)
from eval_factory.attachment_planning import (
    ApprovedSourceEvidenceCompiler,
    ApprovedSourceEvidenceOutcome,
    ApprovedSourceEvidencePolicyError,
    ApprovedSourceEvidenceReason,
    ApprovedSourceRetrievalExecutionOutcome,
    ApprovedSourceRetrievalExecutionResult,
    ApprovedSourceRetrievalIntentDefinition,
    ApprovedSourceRetrievalMode,
    ApprovedSourceRetrievalRequestBuilder,
    ApprovedSourceRetrievalRunner,
    AttachmentPlanningBridge,
    PromptOnlyDependencyPlanningProjector,
    PublicSourceCheck,
    PublicSourceUsageBasis,
    VerifiedPublicSourceChecks,
    approved_source_execution_result_carried_sha256,
    approved_source_retrieval_request_carried_sha256,
)
from eval_factory.contracts import (
    PublicSourceRetrievalPolicyV2,
    artifact_evidence_target_ref,
    public_source_retrieval_policy_carried_sha256,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.safety import (
    Disposition,
    OriginClass,
    ProvenanceDecision,
    Visibility,
)
from eval_factory.provenance.bundles import (
    EvidenceBundleCompiler,
    EvidenceBundleCompileRequest,
    ProjectionEvidenceBinding,
)
from eval_factory.provenance.views import (
    EvidenceViewEngine,
    EvidenceViewPrincipal,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewRequest,
    EvidenceViewSubject,
)
from eval_factory.task_authoring import ProducerTaskViewCompiler

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
POLICY_VERSION = "public-source-retrieval/r5-04-v1"
ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = ROOT / "evals/golden/eval_factory/source_evidence" / "r5-04-retrieval-provenance-v1.json"


def _audit(
    created_at: datetime = datetime(2026, 7, 28, tzinfo=UTC),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="approved-source-evidence-test",
        governing_versions=(
            VersionBinding(
                component="approved-source-evidence",
                version="r5-04",
            ),
        ),
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


def _policy(**overrides: object) -> PublicSourceRetrievalPolicyV2:
    values: dict[str, object] = {
        "public_source_retrieval_policy_id": "public-source-retrieval-policy://pending",
        "approved_search_provider_ids": ("search-provider://fake",),
        "approved_fetch_provider_ids": ("fetch-provider://fake",),
        "allowed_schemes": ("https",),
        "allowed_host_suffixes": ("example.gov",),
        "max_search_results": 3,
        "max_fetch_bytes": 4096,
        "query_egress_policy_ref": _ref(
            "query-egress-policy",
            "public-source",
        ),
        "network_policy_ref": _ref(
            "network-policy",
            "public-source",
        ),
        "source_usage_policy_ref": _ref(
            "source-usage-policy",
            "public-source",
        ),
        "safety_scan_policy_ref": _ref(
            "safety-scan-policy",
            "public-source",
        ),
        "license_policy_ref": _ref(
            "source-license-policy",
            "public-source",
        ),
        "policy_version": POLICY_VERSION,
        "public_source_retrieval_policy_sha256": HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    policy = PublicSourceRetrievalPolicyV2(**values)
    digest = public_source_retrieval_policy_carried_sha256(policy)
    return policy.model_copy(
        update={
            "public_source_retrieval_policy_id": (f"public-source-retrieval-policy://sha256/{digest}"),
            "public_source_retrieval_policy_sha256": digest,
        }
    )


def _discovery(case):
    request = dependency_request(case)
    result = compile_dependency(case, request=request)
    assert result.discovery is not None
    return result.discovery


def _search_intent(case) -> ApprovedSourceRetrievalIntentDefinition:
    return ApprovedSourceRetrievalIntentDefinition(
        intent_id="approved-source-intent://search",
        attachment_dependency_id=(case[1].attachment_requirements[0].dependency_id),
        mode=ApprovedSourceRetrievalMode.SEARCH_THEN_FETCH,
        search_provider_id="search-provider://fake",
        fetch_provider_id="fetch-provider://fake",
        external_lead_projection_item_id=None,
        query_approval_ref=_ref(
            "public-search-query-approval",
            "requirement",
        ),
        source_approval_ref=_ref(
            "public-source-approval",
            "example-gov",
        ),
        usage_basis=PublicSourceUsageBasis.FACTS_ONLY,
    )


def _lead_case():
    source_draft = _draft()
    dependency = source_draft.attachment_dependencies[0].model_copy(
        update={"description": "Use inputs/source.csv."}
    )
    draft = _rehash_draft(source_draft.model_copy(update={"attachment_dependencies": (dependency,)}))
    rubric_set = _rubric_set(draft)
    evaluator_spec, reference_policy = _evaluation_contract(rubric_set)
    tool_policy, contestant_policy = _tool_contract(
        draft,
        rubric_set,
        evaluator_spec,
    )
    source_ref = _ref(
        "external-source",
        "trace-lead",
        digest=OTHER_HASH,
    )
    decision = ProvenanceDecision(
        provenance_decision_id="provenance-decision://external/trace-lead",
        subject_ref=source_ref,
        origin_class=OriginClass.AGENT_RETRIEVED_EXTERNAL,
        visibility=Visibility.STAGE_PROJECTION,
        disposition=Disposition.ALLOW_EXTERNAL_LEAD_ONLY,
        rule_ids=("external-lead/test-v1",),
        source_event_refs=(_ref("trace-event", "external/lead"),),
        confidence=1.0,
        review_required=False,
        policy_version="external-lead/test-v1",
        subject_sha256=source_ref.object_sha256,
        audit=producer_audit(),
    )
    view_result = EvidenceViewEngine().project(
        EvidenceViewRequest(
            principal=EvidenceViewPrincipal(
                principal_id=PRODUCER_PRINCIPAL,
                principal_type=(EvidenceViewPrincipalType.ATTACHMENT_PRODUCER),
                allowed_purposes=frozenset({EvidenceViewPurpose.ATTACHMENT_PRODUCTION}),
                max_subjects=10,
                max_characters=1000,
            ),
            purpose=EvidenceViewPurpose.ATTACHMENT_PRODUCTION,
            subjects=(
                EvidenceViewSubject(
                    subject_ref=source_ref,
                    decision=decision,
                    external_uri="https://example.gov/lead.txt",
                ),
            ),
            max_characters=1000,
            audit=producer_audit(),
        )
    )
    bundle = (
        EvidenceBundleCompiler()
        .compile(
            EvidenceBundleCompileRequest(
                source_trace_id=SOURCE_TRACE_ID,
                trace_ir_version_id=TRACE_ID,
                consumer_stage="attachment-producer",
                purpose="attachment-production",
                view_result=view_result,
                bindings=(
                    ProjectionEvidenceBinding(
                        projection_item_id=(view_result.included_items[0].projection_item_id),
                        source_spans=(_span("external-lead"),),
                        polarity=EvidencePolarity.POSITIVE,
                        capability="attachment-production",
                        capability_complete=True,
                    ),
                ),
                max_characters=1000,
                audit=producer_audit(),
            )
        )
        .evidence_bundle
    )
    projected = ProducerTaskViewCompiler().compile(
        task_draft=draft,
        rubric_set=rubric_set,
        evaluator_spec=evaluator_spec,
        reference_policy=reference_policy,
        tool_policy=tool_policy,
        contestant_tool_policy=contestant_policy,
        producer_view_result=view_result,
        producer_evidence_bundle=bundle,
        audit=producer_audit(),
    )
    assert projected.producer_task_view is not None
    assert projected.storage_authorization is not None
    context = AttachmentPlanningBridge().compile(
        producer_task_view=projected.producer_task_view,
        storage_authorization=projected.storage_authorization,
        evidence_bundle=bundle,
        audit=_audit(),
    )
    planning = PromptOnlyDependencyPlanningProjector().compile(
        task_draft=draft,
        producer_task_view=projected.producer_task_view,
        storage_authorization=projected.storage_authorization,
        evidence_bundle=bundle,
        attachment_planning_context=context,
        audit=_audit(),
    )
    return (
        draft,
        projected.producer_task_view,
        projected.storage_authorization,
        view_result,
        bundle,
        context,
        planning,
    )


def _lead_intent(case) -> ApprovedSourceRetrievalIntentDefinition:
    return ApprovedSourceRetrievalIntentDefinition(
        intent_id="approved-source-intent://lead",
        attachment_dependency_id=(case[1].attachment_requirements[0].dependency_id),
        mode=ApprovedSourceRetrievalMode.REFETCH_EXTERNAL_LEAD,
        search_provider_id=None,
        fetch_provider_id="fetch-provider://fake",
        external_lead_projection_item_id=(case[3].included_items[0].projection_item_id),
        query_approval_ref=None,
        source_approval_ref=_ref(
            "public-source-approval",
            "example-gov-lead",
        ),
        usage_basis=PublicSourceUsageBasis.FACTS_ONLY,
    )


def _build_request(case, *, intents=None):
    discovery = None
    if case[1].attachment_requirements:
        discovery = _discovery(case)
    definitions = (
        ()
        if intents is None and discovery is None
        else ((_search_intent(case),) if intents is None else intents)
    )
    request = ApprovedSourceRetrievalRequestBuilder().build(
        attachment_planning_context=case[5],
        producer_task_view=case[1],
        storage_authorization=case[2],
        producer_view_result=case[3],
        producer_evidence_bundle=case[4],
        dependency_discovery=discovery,
        retrieval_policy=_policy(),
        intent_definitions=definitions,
        audit=_audit(),
    )
    return request, discovery


class _FakeFacade(PublicSourceRetrievalFacade):
    def __init__(self) -> None:
        self.search_requests: list[PublicSourceSearchRequestV2] = []
        self.fetch_requests: list[PublicSourceFetchRequestV2] = []

    async def search(
        self,
        request: PublicSourceSearchRequestV2,
    ) -> PublicSourceSearchResultV2:
        self.search_requests.append(request)
        hits = []
        for rank, uri in (
            (1, "https://example.gov/first.txt"),
            (2, "https://example.gov/second.txt"),
        ):
            hit = PublicSourceSearchHitV2(
                search_hit_id="public-source-search-hit://pending",
                source_uri=uri,
                rank=rank,
                provider_id=request.provider_id,
                search_hit_sha256=HASH,
            )
            digest = public_source_search_hit_carried_sha256(hit)
            hits.append(
                hit.model_copy(
                    update={
                        "search_hit_id": (f"public-source-search-hit://sha256/{digest}"),
                        "search_hit_sha256": digest,
                    }
                )
            )
        result = PublicSourceSearchResultV2(
            search_result_id="public-source-search-result://pending",
            search_request_ref=public_source_search_request_ref(request),
            provider_id=request.provider_id,
            hits=tuple(hits),
            search_result_sha256=HASH,
        )
        digest = public_source_search_result_carried_sha256(result)
        return result.model_copy(
            update={
                "search_result_id": (f"public-source-search-result://sha256/{digest}"),
                "search_result_sha256": digest,
            }
        )

    async def fetch(
        self,
        request: PublicSourceFetchRequestV2,
    ) -> PublicSourceFetchResultV2:
        self.fetch_requests.append(request)
        content_ref = FacadeObjectRef(
            object_type="public-source-content",
            object_id=f"public-source-content://sha256/{OTHER_HASH}",
            object_version="v1",
            object_sha256=OTHER_HASH,
        )
        result = PublicSourceFetchResultV2(
            fetch_result_id="public-source-fetch-result://pending",
            fetch_request_ref=public_source_fetch_request_ref(request),
            requested_source_uri=request.source_uri,
            canonical_source_uri=request.source_uri,
            retrieved_at=datetime(2026, 7, 28, tzinfo=UTC),
            status_code=200,
            media_type="text/plain",
            bytes_count=23,
            content_ref=content_ref,
            content_sha256=OTHER_HASH,
            provider_id=request.provider_id,
            response_metadata_sha256=THIRD_HASH,
            fetch_result_sha256=HASH,
        )
        digest = public_source_fetch_result_carried_sha256(result)
        return result.model_copy(
            update={
                "fetch_result_id": (f"public-source-fetch-result://sha256/{digest}"),
                "fetch_result_sha256": digest,
            }
        )


class _EmptySearchFacade(_FakeFacade):
    async def search(
        self,
        request: PublicSourceSearchRequestV2,
    ) -> PublicSourceSearchResultV2:
        result = await super().search(request)
        empty = result.model_copy(
            update={
                "hits": (),
                "search_result_id": "public-source-search-result://pending",
                "search_result_sha256": HASH,
            }
        )
        digest = public_source_search_result_carried_sha256(empty)
        return empty.model_copy(
            update={
                "search_result_id": (f"public-source-search-result://sha256/{digest}"),
                "search_result_sha256": digest,
            }
        )


class _SearchPolicyBlockedFacade(_FakeFacade):
    async def search(
        self,
        request: PublicSourceSearchRequestV2,
    ) -> PublicSourceSearchResultV2:
        raise PublicSourceRetrievalFacadeError(
            PublicSourceRetrievalFacadeFailureKind.POLICY,
            "URI_HOST_NOT_APPROVED",
            "source host is not approved",
        )


class _FetchUnavailableFacade(_FakeFacade):
    async def fetch(
        self,
        request: PublicSourceFetchRequestV2,
    ) -> PublicSourceFetchResultV2:
        raise PublicSourceRetrievalFacadeError(
            PublicSourceRetrievalFacadeFailureKind.CAPABILITY,
            "FETCH_PROVIDER_UNAVAILABLE",
            "fetch provider is unavailable",
        )


def _verified_checks(execution, *, failed=frozenset(), unavailable=frozenset()):
    fetch = execution.fetch_result
    scan_refs: dict[PublicSourceCheck, ObjectRef] = {
        PublicSourceCheck.SECRET: _ref(
            "secret-scan-result",
            "content",
            digest=fetch.content_sha256,
        ),
        PublicSourceCheck.CONFIGURED_PII: _ref(
            "configured-pii-scan-result",
            "content",
            digest=fetch.content_sha256,
        ),
        PublicSourceCheck.PROMPT_INJECTION: _ref(
            "prompt-injection-scan-result",
            "content",
            digest=fetch.content_sha256,
        ),
        PublicSourceCheck.ANSWER_LEAKAGE: _ref(
            "answer-leakage-scan-result",
            "content",
            digest=fetch.content_sha256,
        ),
        PublicSourceCheck.LICENSE: _ref(
            "source-license-assessment",
            "content",
            digest=fetch.content_sha256,
        ),
    }
    for check in failed | unavailable:
        scan_refs.pop(check, None)
    return VerifiedPublicSourceChecks(
        intent_id=execution.intent_id,
        fetch_result_ref=execution.fetch_result_ref,
        content_ref=ObjectRef.model_validate(fetch.content_ref.model_dump(mode="python")),
        content_sha256=fetch.content_sha256,
        secret_scan_ref=scan_refs.get(PublicSourceCheck.SECRET),
        pii_scan_ref=scan_refs.get(PublicSourceCheck.CONFIGURED_PII),
        prompt_injection_scan_ref=scan_refs.get(PublicSourceCheck.PROMPT_INJECTION),
        answer_leakage_scan_ref=scan_refs.get(PublicSourceCheck.ANSWER_LEAKAGE),
        license_assessment_ref=scan_refs.get(PublicSourceCheck.LICENSE),
        unavailable_checks=unavailable,
        failed_checks=failed,
    )


def _compile(
    case,
    request,
    discovery,
    execution_result,
    checks,
):
    return ApprovedSourceEvidenceCompiler().compile(
        request=request,
        execution_result=execution_result,
        verified_checks=checks,
        attachment_planning_context=case[5],
        producer_task_view=case[1],
        storage_authorization=case[2],
        producer_view_result=case[3],
        producer_evidence_bundle=case[4],
        dependency_discovery=discovery,
        retrieval_policy=_policy(),
        audit=_audit(),
    )


def test_no_attachment_requirements_return_not_required_without_discovery() -> None:
    case = dependency_case(no_requirements=True)
    request, discovery = _build_request(case)

    result = _compile(
        case,
        request,
        discovery,
        execution_result=None,
        checks=(),
    )

    assert discovery is None
    assert request.intents == ()
    assert request.not_required_target_refs == ()
    assert result.outcome is ApprovedSourceEvidenceOutcome.NOT_REQUIRED
    assert result.source_evidence_set is None


def test_search_request_is_derived_from_current_requirement_and_partitions_targets() -> None:
    case = dependency_case(description="Use inputs/source.csv.")
    request, discovery = _build_request(case)
    assert discovery is not None

    assert len(request.intents) == 1
    intent = request.intents[0]
    target = discovery.targets[0]
    assert intent.query == case[1].attachment_requirements[0].description
    assert intent.artifact_evidence_target_ref == (artifact_evidence_target_ref(target))
    assert request.covered_target_refs == (artifact_evidence_target_ref(target),)
    assert request.not_required_target_refs == ()


def test_zero_approved_intents_returns_not_required_with_complete_partition() -> None:
    case = dependency_case(description="Use inputs/source.csv.")
    request, discovery = _build_request(case, intents=())
    assert discovery is not None

    result = _compile(
        case,
        request,
        discovery,
        execution_result=None,
        checks=(),
    )

    assert request.covered_target_refs == ()
    assert request.not_required_target_refs == tuple(
        artifact_evidence_target_ref(target) for target in discovery.targets
    )
    assert result.outcome is ApprovedSourceEvidenceOutcome.NOT_REQUIRED
    assert result.source_evidence_set is None


@pytest.mark.asyncio
async def test_runner_selects_first_policy_allowed_search_hit() -> None:
    case = dependency_case(description="Use inputs/source.csv.")
    request, _ = _build_request(case)
    facade = _FakeFacade()

    execution_result = await ApprovedSourceRetrievalRunner().run(
        request,
        facade=facade,
        audit=_audit(),
    )

    assert execution_result.executions[0].search_result is not None
    assert execution_result.executions[0].fetch_request.source_uri == ("https://example.gov/first.txt")
    assert facade.fetch_requests[0].source_uri == ("https://example.gov/first.txt")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("facade", "expected_outcome", "expected_reason"),
    [
        pytest.param(
            _EmptySearchFacade(),
            ApprovedSourceRetrievalExecutionOutcome.BLOCKED_CAPABILITY,
            ApprovedSourceEvidenceReason.SEARCH_RESULT_EMPTY,
            id="empty-search",
        ),
        pytest.param(
            _SearchPolicyBlockedFacade(),
            ApprovedSourceRetrievalExecutionOutcome.BLOCKED_POLICY,
            ApprovedSourceEvidenceReason.URI_POLICY_DENIED,
            id="search-policy",
        ),
        pytest.param(
            _FetchUnavailableFacade(),
            ApprovedSourceRetrievalExecutionOutcome.BLOCKED_CAPABILITY,
            ApprovedSourceEvidenceReason.FETCH_FAILED,
            id="fetch-unavailable",
        ),
    ],
)
async def test_runner_blocks_all_or_none_on_search_fetch_failure(
    facade: PublicSourceRetrievalFacade,
    expected_outcome: ApprovedSourceRetrievalExecutionOutcome,
    expected_reason: ApprovedSourceEvidenceReason,
) -> None:
    case = dependency_case(description="Use inputs/source.csv.")
    request, _ = _build_request(case)

    result = await ApprovedSourceRetrievalRunner().run(
        request,
        facade=facade,
        audit=_audit(),
    )

    assert result.outcome is expected_outcome
    assert result.executions == ()
    assert result.unresolved_intent_ids == (request.intents[0].intent_id,)
    assert result.reasons == frozenset({expected_reason})


@pytest.mark.asyncio
async def test_external_lead_is_refetched_and_preserved_only_as_restricted_lineage() -> None:
    case = _lead_case()
    request, discovery = _build_request(
        case,
        intents=(_lead_intent(case),),
    )
    facade = _FakeFacade()
    execution_result = await ApprovedSourceRetrievalRunner().run(
        request,
        facade=facade,
        audit=_audit(),
    )
    checks = tuple(_verified_checks(execution) for execution in execution_result.executions)

    result = _compile(
        case,
        request,
        discovery,
        execution_result,
        checks,
    )

    assert facade.search_requests == []
    assert facade.fetch_requests[0].source_uri == ("https://example.gov/lead.txt")
    assert result.source_evidence_set is not None
    evidence = result.source_evidence_set.source_evidence[0]
    lead = case[3].included_items[0]
    assert evidence.external_lead_ref == lead.projected_ref
    assert evidence.external_lead_decision_ref is not None
    assert evidence.search_result_ref is None
    serialized = str(evidence.model_dump(mode="json"))
    assert lead.source_ref.object_id not in serialized


@pytest.mark.asyncio
async def test_complete_search_fetch_checks_compile_fresh_source_evidence() -> None:
    case = dependency_case(description="Use inputs/source.csv.")
    request, discovery = _build_request(case)
    facade = _FakeFacade()
    execution_result = await ApprovedSourceRetrievalRunner().run(
        request,
        facade=facade,
        audit=_audit(),
    )
    checks = tuple(_verified_checks(execution) for execution in execution_result.executions)

    result = _compile(
        case,
        request,
        discovery,
        execution_result,
        checks,
    )

    assert result.outcome is ApprovedSourceEvidenceOutcome.COMPILED
    assert result.source_evidence_set is not None
    evidence = result.source_evidence_set.source_evidence[0]
    assert evidence.source_evidence_v1.source_uri == ("https://example.gov/first.txt")
    assert evidence.source_evidence_v1.content_sha256 == OTHER_HASH
    assert evidence.content_provenance_decision.origin_class is (OriginClass.SYSTEM_OR_HARNESS_CONTEXT)
    assert evidence.content_provenance_decision.disposition is (Disposition.ALLOW_INPUT_EVIDENCE)
    assert evidence.content_provenance_decision.rule_ids == ("public-source-refetch/r5-04-v1",)
    assert result.source_evidence_set.covered_target_refs == (request.covered_target_refs)


@pytest.mark.asyncio
async def test_safety_failure_blocks_entire_set_without_disclosing_source_facts() -> None:
    case = dependency_case(description="Use inputs/source.csv.")
    request, discovery = _build_request(case)
    execution_result = await ApprovedSourceRetrievalRunner().run(
        request,
        facade=_FakeFacade(),
        audit=_audit(),
    )
    checks = (
        _verified_checks(
            execution_result.executions[0],
            failed=frozenset({PublicSourceCheck.SECRET}),
        ),
    )

    result = _compile(
        case,
        request,
        discovery,
        execution_result,
        checks,
    )

    assert result.outcome is ApprovedSourceEvidenceOutcome.BLOCKED_SAFETY
    assert result.source_evidence_set is None
    assert result.reasons == frozenset({ApprovedSourceEvidenceReason.SECRET_SCAN_FAILED})
    serialized = str(result.model_dump(mode="json"))
    for forbidden in (
        "https://example.gov",
        OTHER_HASH,
        "secret-scan-result",
        "public-source-content",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_license_denial_blocks_policy_without_partial_evidence() -> None:
    case = dependency_case(description="Use inputs/source.csv.")
    request, discovery = _build_request(case)
    execution_result = await ApprovedSourceRetrievalRunner().run(
        request,
        facade=_FakeFacade(),
        audit=_audit(),
    )
    checks = (
        _verified_checks(
            execution_result.executions[0],
            failed=frozenset({PublicSourceCheck.LICENSE}),
        ),
    )

    result = _compile(
        case,
        request,
        discovery,
        execution_result,
        checks,
    )

    assert result.outcome is ApprovedSourceEvidenceOutcome.BLOCKED_POLICY
    assert result.source_evidence_set is None
    assert result.reasons == frozenset({ApprovedSourceEvidenceReason.LICENSE_DENIED})


@pytest.mark.asyncio
async def test_missing_scan_capability_blocks_without_partial_evidence() -> None:
    case = dependency_case(description="Use inputs/source.csv.")
    request, discovery = _build_request(case)
    execution_result = await ApprovedSourceRetrievalRunner().run(
        request,
        facade=_FakeFacade(),
        audit=_audit(),
    )
    checks = (
        _verified_checks(
            execution_result.executions[0],
            unavailable=frozenset({PublicSourceCheck.PROMPT_INJECTION}),
        ),
    )

    result = _compile(
        case,
        request,
        discovery,
        execution_result,
        checks,
    )

    assert result.outcome is ApprovedSourceEvidenceOutcome.BLOCKED_CAPABILITY
    assert result.source_evidence_set is None
    assert result.reasons == frozenset({ApprovedSourceEvidenceReason.SCAN_CAPABILITY_UNAVAILABLE})


def test_execution_policy_block_maps_to_policy_without_results() -> None:
    case = dependency_case(description="Use inputs/source.csv.")
    request, discovery = _build_request(case)
    value = ApprovedSourceRetrievalExecutionResult(
        execution_result_id=("approved-source-retrieval-execution-result://pending"),
        request_ref=ObjectRef(
            object_type="approved-source-retrieval-request",
            object_id=request.request_id,
            object_version="r5-04",
            object_sha256=request.request_sha256,
        ),
        outcome=ApprovedSourceRetrievalExecutionOutcome.BLOCKED_POLICY,
        executions=(),
        unresolved_intent_ids=(request.intents[0].intent_id,),
        reasons=frozenset({ApprovedSourceEvidenceReason.URI_POLICY_DENIED}),
        execution_result_sha256=HASH,
        audit=_audit(),
    )
    digest = approved_source_execution_result_carried_sha256(value)
    execution_result = value.model_copy(
        update={
            "execution_result_id": (f"approved-source-retrieval-execution-result://sha256/{digest}"),
            "execution_result_sha256": digest,
        }
    )

    result = _compile(
        case,
        request,
        discovery,
        execution_result,
        checks=(),
    )

    assert result.outcome is ApprovedSourceEvidenceOutcome.BLOCKED_POLICY
    assert result.source_evidence_set is None
    assert result.reasons == frozenset({ApprovedSourceEvidenceReason.URI_POLICY_DENIED})


def test_request_identity_ignores_audit_actor_time_and_binds_policy() -> None:
    case = dependency_case(description="Use inputs/source.csv.")
    request, _ = _build_request(case)
    changed_audit = request.model_copy(
        update={
            "audit": request.audit.model_copy(
                update={
                    "created_at": datetime(2026, 7, 29, tzinfo=UTC),
                    "created_by": "approved-source-evidence-other-actor",
                }
            )
        }
    )

    assert approved_source_retrieval_request_carried_sha256(request) == (
        approved_source_retrieval_request_carried_sha256(changed_audit)
    )
    changed_limit = request.model_copy(update={"max_fetch_bytes": request.max_fetch_bytes + 1})
    assert approved_source_retrieval_request_carried_sha256(request) != (
        approved_source_retrieval_request_carried_sha256(changed_limit)
    )


@pytest.mark.asyncio
async def test_validate_current_rejects_fetch_result_time_drift() -> None:
    case = dependency_case(description="Use inputs/source.csv.")
    request, discovery = _build_request(case)
    execution_result = await ApprovedSourceRetrievalRunner().run(
        request,
        facade=_FakeFacade(),
        audit=_audit(),
    )
    checks = tuple(_verified_checks(execution) for execution in execution_result.executions)
    result = _compile(
        case,
        request,
        discovery,
        execution_result,
        checks,
    )
    assert result.source_evidence_set is not None

    ApprovedSourceEvidenceCompiler().validate_current(
        request=request,
        execution_result=execution_result,
        verified_checks=checks,
        attachment_planning_context=case[5],
        producer_task_view=case[1],
        storage_authorization=case[2],
        producer_view_result=case[3],
        producer_evidence_bundle=case[4],
        dependency_discovery=discovery,
        retrieval_policy=_policy(),
        source_evidence_set=result.source_evidence_set,
    )

    execution = execution_result.executions[0]
    stale_fetch = execution.fetch_result.model_copy(
        update={
            "retrieved_at": datetime(2026, 7, 29, tzinfo=UTC),
        }
    )
    stale_execution = execution.model_copy(update={"fetch_result": stale_fetch})
    stale_result = execution_result.model_copy(update={"executions": (stale_execution,)})
    with pytest.raises(ApprovedSourceEvidencePolicyError, match="current"):
        ApprovedSourceEvidenceCompiler().validate_current(
            request=request,
            execution_result=stale_result,
            verified_checks=checks,
            attachment_planning_context=case[5],
            producer_task_view=case[1],
            storage_authorization=case[2],
            producer_view_result=case[3],
            producer_evidence_bundle=case[4],
            dependency_discovery=discovery,
            retrieval_policy=_policy(),
            source_evidence_set=result.source_evidence_set,
        )


async def _compile_gold_setup(setup: str):
    if setup == "NO_REQUIREMENTS":
        case = dependency_case(no_requirements=True)
        request, discovery = _build_request(case)
        return (
            _compile(
                case,
                request,
                discovery,
                execution_result=None,
                checks=(),
            ),
            request,
        )
    if setup == "NO_APPROVED_INTENTS":
        case = dependency_case(description="Use inputs/source.csv.")
        request, discovery = _build_request(case, intents=())
        return (
            _compile(
                case,
                request,
                discovery,
                execution_result=None,
                checks=(),
            ),
            request,
        )

    case = (
        _lead_case()
        if setup == "LEAD_REFETCH_COMPILED"
        else dependency_case(description="Use inputs/source.csv.")
    )
    intents = (_lead_intent(case),) if setup == "LEAD_REFETCH_COMPILED" else None
    request, discovery = _build_request(case, intents=intents)
    execution_result = await ApprovedSourceRetrievalRunner().run(
        request,
        facade=_FakeFacade(),
        audit=_audit(),
    )
    failed_by_setup = {
        "SECRET_FAILED": PublicSourceCheck.SECRET,
        "CONFIGURED_PII_FAILED": PublicSourceCheck.CONFIGURED_PII,
        "PROMPT_INJECTION_FAILED": PublicSourceCheck.PROMPT_INJECTION,
        "ANSWER_LEAKAGE_FAILED": PublicSourceCheck.ANSWER_LEAKAGE,
        "LICENSE_FAILED": PublicSourceCheck.LICENSE,
    }
    failed_check = failed_by_setup.get(setup)
    unavailable = (
        frozenset({PublicSourceCheck.PROMPT_INJECTION}) if setup == "SCAN_UNAVAILABLE" else frozenset()
    )
    checks = tuple(
        _verified_checks(
            execution,
            failed=(frozenset({failed_check}) if failed_check is not None else frozenset()),
            unavailable=unavailable,
        )
        for execution in execution_result.executions
    )
    return (
        _compile(
            case,
            request,
            discovery,
            execution_result,
            checks,
        ),
        request,
    )


@pytest.mark.asyncio
async def test_retrieval_provenance_gold_is_content_free_and_executable() -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    assert gold["schema_version"] == ("eval-factory/r5-04-retrieval-provenance-gold/v1")
    assert gold["claim_scope"] == ("CONTENT_FREE_PUBLIC_SOURCE_RETRIEVAL_POLICY_ONLY")

    serialized = json.dumps(gold, sort_keys=True)
    for forbidden in (
        "query_text",
        "source_uri",
        "snippet",
        "header",
        "content_path",
        "credential",
        "trace_text",
        "final_answer",
    ):
        assert forbidden not in serialized

    for case in gold["cases"]:
        assert set(case) == {
            "case_id",
            "setup",
            "expected_outcome",
            "expected_reason",
            "covered_targets",
            "not_required_targets",
        }
        result, request = await _compile_gold_setup(case["setup"])
        assert result.outcome.value == case["expected_outcome"]
        expected_reason = case["expected_reason"]
        if expected_reason is None:
            assert result.reasons == frozenset()
        else:
            assert {item.value for item in result.reasons} == {expected_reason}
        assert len(request.covered_target_refs) == case["covered_targets"]
        assert len(request.not_required_target_refs) == (case["not_required_targets"])
