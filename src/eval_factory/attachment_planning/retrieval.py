from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit

from env_mock_agent.facade import (
    FacadeObjectRef,
    PublicSourceFetchRequestV2,
    PublicSourceFetchResultV2,
    PublicSourceRetrievalFacade,
    PublicSourceRetrievalFacadeError,
    PublicSourceRetrievalFacadeFailureKind,
    PublicSourceSearchRequestV2,
    PublicSourceSearchResultV2,
    public_source_fetch_request_carried_sha256,
    public_source_fetch_request_ref,
    public_source_fetch_result_carried_sha256,
    public_source_fetch_result_ref,
    public_source_search_request_carried_sha256,
    public_source_search_request_ref,
    public_source_search_result_carried_sha256,
    public_source_search_result_ref,
)
from eval_factory.attachment_planning.bridge import AttachmentPlanningBridge
from eval_factory.attachment_planning.models import AttachmentPlanningPolicyError
from eval_factory.attachment_planning.retrieval_models import (
    APPROVED_SOURCE_RETRIEVAL_POLICY_VERSION,
    ApprovedSourceEvidenceOutcome,
    ApprovedSourceEvidencePolicyError,
    ApprovedSourceEvidenceReason,
    ApprovedSourceEvidenceResult,
    ApprovedSourceRetrievalExecutionOutcome,
    ApprovedSourceRetrievalExecutionResult,
    ApprovedSourceRetrievalIntent,
    ApprovedSourceRetrievalIntentDefinition,
    ApprovedSourceRetrievalMode,
    ApprovedSourceRetrievalRequest,
    PublicSourceCheck,
    PublicSourceRetrievalExecution,
    VerifiedPublicSourceChecks,
    approved_source_evidence_result_carried_sha256,
    approved_source_execution_result_carried_sha256,
    approved_source_retrieval_request_carried_sha256,
    approved_source_retrieval_request_ref,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactEvidenceTargetV2,
    AttachmentPlanningContextV2,
    PromptOnlyDependencyDiscoveryV2,
    PublicSourceRetrievalPolicyV2,
    PublicSourceSafetyAssessmentV2,
    SourceEvidenceClaimBindingV2,
    SourceEvidenceSetV2,
    SourceEvidenceV2,
    artifact_evidence_target_ref,
    attachment_planning_context_ref,
    prompt_only_dependency_discovery_carried_sha256,
    prompt_only_dependency_discovery_ref,
    public_source_retrieval_policy_carried_sha256,
    public_source_retrieval_policy_ref,
    public_source_safety_assessment_carried_sha256,
    public_source_safety_assessment_ref,
    source_evidence_set_carried_sha256,
    source_evidence_v2_carried_sha256,
    source_evidence_v2_ref,
    validate_artifact_evidence_target_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.safety import (
    Disposition,
    EvidenceBundle,
    OriginClass,
    ProvenanceDecision,
    SourceEvidence,
    Visibility,
)
from eval_factory.contracts.task_v2 import (
    ProducerAttachmentRequirementV2,
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
    producer_task_view_ref,
)
from eval_factory.provenance.bundles import (
    EvidenceCompilationPolicyError,
    evidence_bundle_ref,
    projection_policy_ref,
    validate_evidence_bundle_identity,
    validate_projection_policy,
)
from eval_factory.provenance.views import (
    EvidenceProjectionItem,
    EvidenceProjectionMode,
    EvidenceViewPolicyError,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewResult,
    validate_evidence_view_result_identity,
)

_SAFETY_CHECK_REASONS = {
    PublicSourceCheck.SECRET: ApprovedSourceEvidenceReason.SECRET_SCAN_FAILED,
    PublicSourceCheck.CONFIGURED_PII: ApprovedSourceEvidenceReason.PII_SCAN_FAILED,
    PublicSourceCheck.PROMPT_INJECTION: (ApprovedSourceEvidenceReason.PROMPT_INJECTION_SCAN_FAILED),
    PublicSourceCheck.ANSWER_LEAKAGE: (ApprovedSourceEvidenceReason.ANSWER_LEAKAGE_SCAN_FAILED),
}


class ApprovedSourceRetrievalRequestBuilder:
    policy_version = APPROVED_SOURCE_RETRIEVAL_POLICY_VERSION

    def build(
        self,
        *,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
        dependency_discovery: PromptOnlyDependencyDiscoveryV2 | None,
        retrieval_policy: PublicSourceRetrievalPolicyV2,
        intent_definitions: tuple[
            ApprovedSourceRetrievalIntentDefinition,
            ...,
        ],
        audit: ContractAudit,
    ) -> ApprovedSourceRetrievalRequest:
        self._validate_basis(
            attachment_planning_context=attachment_planning_context,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            producer_view_result=producer_view_result,
            producer_evidence_bundle=producer_evidence_bundle,
            dependency_discovery=dependency_discovery,
            retrieval_policy=retrieval_policy,
        )
        requirements = {item.dependency_id: item for item in producer_task_view.attachment_requirements}
        context_ref = attachment_planning_context_ref(attachment_planning_context)
        view_ref = producer_task_view_ref(producer_task_view)
        evidence_view_ref = _evidence_view_result_ref(producer_view_result)
        bundle_ref = evidence_bundle_ref(producer_evidence_bundle)
        policy_ref = public_source_retrieval_policy_ref(retrieval_policy)

        if not requirements:
            if dependency_discovery is not None or intent_definitions:
                raise ApprovedSourceEvidencePolicyError(
                    "attachment-free task cannot carry discovery or retrieval intents"
                )
            return _request(
                context_ref=context_ref,
                view_ref=view_ref,
                evidence_view_ref=evidence_view_ref,
                bundle_ref=bundle_ref,
                discovery_ref=None,
                policy=retrieval_policy,
                policy_ref=policy_ref,
                intents=(),
                covered_target_refs=(),
                not_required_target_refs=(),
                audit=audit,
            )

        if dependency_discovery is None:
            raise ApprovedSourceEvidencePolicyError(
                "attachment requirements require current dependency discovery"
            )
        targets = {item.attachment_dependency_id: item for item in dependency_discovery.targets}
        if set(targets) != set(requirements):
            raise ApprovedSourceEvidencePolicyError(
                "dependency discovery targets must exactly cover requirements"
            )
        definitions = tuple(sorted(intent_definitions, key=lambda item: item.intent_id))
        _require_unique(
            "retrieval intent definitions",
            tuple(item.intent_id for item in definitions),
        )
        _require_unique(
            "retrieval intent dependency IDs",
            tuple(item.attachment_dependency_id for item in definitions),
        )
        leads = {
            item.projection_item_id: item
            for item in producer_view_result.included_items
            if item.projection_mode is EvidenceProjectionMode.EXTERNAL_LEAD
        }
        intents = tuple(
            self._compile_intent(
                definition=definition,
                requirement=requirements.get(definition.attachment_dependency_id),
                target=targets.get(definition.attachment_dependency_id),
                leads=leads,
                retrieval_policy=retrieval_policy,
            )
            for definition in definitions
        )
        covered_refs = tuple(
            sorted(
                (item.artifact_evidence_target_ref for item in intents),
                key=_ref_key,
            )
        )
        covered_keys = {_ref_key(ref) for ref in covered_refs}
        not_required_refs = tuple(
            sorted(
                (
                    artifact_evidence_target_ref(target)
                    for target in dependency_discovery.targets
                    if _ref_key(artifact_evidence_target_ref(target)) not in covered_keys
                ),
                key=_ref_key,
            )
        )
        return _request(
            context_ref=context_ref,
            view_ref=view_ref,
            evidence_view_ref=evidence_view_ref,
            bundle_ref=bundle_ref,
            discovery_ref=prompt_only_dependency_discovery_ref(dependency_discovery),
            policy=retrieval_policy,
            policy_ref=policy_ref,
            intents=intents,
            covered_target_refs=covered_refs,
            not_required_target_refs=not_required_refs,
            audit=audit,
        )

    def _compile_intent(
        self,
        *,
        definition: ApprovedSourceRetrievalIntentDefinition,
        requirement: ProducerAttachmentRequirementV2 | None,
        target: ArtifactEvidenceTargetV2 | None,
        leads: dict[str, EvidenceProjectionItem],
        retrieval_policy: PublicSourceRetrievalPolicyV2,
    ) -> ApprovedSourceRetrievalIntent:
        if requirement is None or target is None:
            raise ApprovedSourceEvidencePolicyError("retrieval intent selects unknown dependency")
        dependency_id = definition.attachment_dependency_id
        if target.attachment_dependency_id != dependency_id:
            raise ApprovedSourceEvidencePolicyError("retrieval intent target dependency is mismatched")
        if definition.fetch_provider_id not in retrieval_policy.approved_fetch_provider_ids:
            raise ApprovedSourceEvidencePolicyError("retrieval intent fetch provider is not approved")
        target_ref = artifact_evidence_target_ref(target)
        if definition.mode is ApprovedSourceRetrievalMode.SEARCH_THEN_FETCH:
            if definition.search_provider_id not in retrieval_policy.approved_search_provider_ids:
                raise ApprovedSourceEvidencePolicyError("retrieval intent search provider is not approved")
            return ApprovedSourceRetrievalIntent(
                intent_id=definition.intent_id,
                attachment_dependency_id=dependency_id,
                artifact_evidence_target_ref=target_ref,
                mode=definition.mode,
                search_provider_id=definition.search_provider_id,
                fetch_provider_id=definition.fetch_provider_id,
                query=requirement.description,
                query_approval_ref=definition.query_approval_ref,
                source_approval_ref=definition.source_approval_ref,
                usage_basis=definition.usage_basis,
            )
        lead = leads.get(definition.external_lead_projection_item_id or "")
        if lead is None:
            raise ApprovedSourceEvidencePolicyError("retrieval intent selects unknown external lead")
        _validate_external_lead(lead)
        return ApprovedSourceRetrievalIntent(
            intent_id=definition.intent_id,
            attachment_dependency_id=dependency_id,
            artifact_evidence_target_ref=target_ref,
            mode=definition.mode,
            fetch_provider_id=definition.fetch_provider_id,
            external_lead_projection_item_id=lead.projection_item_id,
            external_lead_ref=lead.projected_ref,
            external_lead_decision_ref=_provenance_decision_ref(lead.child_decision),
            source_uri=lead.external_uri,
            source_approval_ref=definition.source_approval_ref,
            usage_basis=definition.usage_basis,
        )

    def _validate_basis(
        self,
        *,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
        dependency_discovery: PromptOnlyDependencyDiscoveryV2 | None,
        retrieval_policy: PublicSourceRetrievalPolicyV2,
    ) -> None:
        try:
            AttachmentPlanningBridge().validate_current(
                producer_task_view=producer_task_view,
                storage_authorization=storage_authorization,
                evidence_bundle=producer_evidence_bundle,
                attachment_planning_context=attachment_planning_context,
            )
            validate_evidence_view_result_identity(producer_view_result)
            validate_projection_policy(producer_view_result.projection_policy)
            validate_evidence_bundle_identity(producer_evidence_bundle)
        except (
            AttachmentPlanningPolicyError,
            EvidenceViewPolicyError,
            EvidenceCompilationPolicyError,
        ) as exc:
            raise ApprovedSourceEvidencePolicyError(
                "approved source retrieval basis is stale or invalid"
            ) from exc
        _validate_retrieval_policy_identity(retrieval_policy)
        if retrieval_policy.policy_version != self.policy_version:
            raise ApprovedSourceEvidencePolicyError("public source retrieval policy is stale or unsupported")
        if (
            producer_view_result.principal_id != attachment_planning_context.producer_principal_id
            or producer_view_result.principal_type is not EvidenceViewPrincipalType.ATTACHMENT_PRODUCER
            or producer_view_result.purpose is not EvidenceViewPurpose.ATTACHMENT_PRODUCTION
        ):
            raise ApprovedSourceEvidencePolicyError(
                "producer evidence view principal or purpose is mismatched"
            )
        expected_policy_ref = projection_policy_ref(producer_view_result.projection_policy)
        if (
            expected_policy_ref != attachment_planning_context.projection_policy_ref
            or expected_policy_ref != producer_evidence_bundle.projection_policy_ref
        ):
            raise ApprovedSourceEvidencePolicyError("producer evidence view policy is mismatched")
        requirements = producer_task_view.attachment_requirements
        if not requirements:
            if dependency_discovery is not None:
                raise ApprovedSourceEvidencePolicyError(
                    "attachment-free task cannot carry dependency discovery"
                )
            return
        if dependency_discovery is None:
            raise ApprovedSourceEvidencePolicyError("attachment requirements require dependency discovery")
        digest = prompt_only_dependency_discovery_carried_sha256(dependency_discovery)
        if (
            dependency_discovery.prompt_only_dependency_discovery_sha256 != digest
            or dependency_discovery.prompt_only_dependency_discovery_id
            != f"prompt-only-dependency-discovery://sha256/{digest}"
        ):
            raise ApprovedSourceEvidencePolicyError("dependency discovery identity is stale")
        if dependency_discovery.attachment_planning_context_ref != attachment_planning_context_ref(
            attachment_planning_context
        ) or dependency_discovery.producer_task_view_ref != producer_task_view_ref(producer_task_view):
            raise ApprovedSourceEvidencePolicyError("dependency discovery source graph is mismatched")
        for target in dependency_discovery.targets:
            try:
                validate_artifact_evidence_target_identity(target)
            except ValueError as exc:
                raise ApprovedSourceEvidencePolicyError(
                    "dependency discovery target identity is stale"
                ) from exc


class ApprovedSourceRetrievalRunner:
    policy_version = APPROVED_SOURCE_RETRIEVAL_POLICY_VERSION

    async def run(
        self,
        request: ApprovedSourceRetrievalRequest,
        *,
        facade: PublicSourceRetrievalFacade,
        audit: ContractAudit,
    ) -> ApprovedSourceRetrievalExecutionResult:
        _validate_request_identity(request)
        if not request.intents:
            raise ApprovedSourceEvidencePolicyError("retrieval runner requires at least one approved intent")
        executions: list[PublicSourceRetrievalExecution] = []
        for intent in request.intents:
            try:
                execution = await self._execute_intent(
                    request=request,
                    intent=intent,
                    facade=facade,
                )
            except _RunnerBlocked as blocked:
                return _execution_result(
                    request=request,
                    outcome=blocked.outcome,
                    executions=(),
                    unresolved_intent_ids=(intent.intent_id,),
                    reasons=frozenset({blocked.reason}),
                    audit=audit,
                )
            executions.append(execution)
        return _execution_result(
            request=request,
            outcome=ApprovedSourceRetrievalExecutionOutcome.EXECUTED,
            executions=tuple(executions),
            unresolved_intent_ids=(),
            reasons=frozenset(),
            audit=audit,
        )

    async def _execute_intent(
        self,
        *,
        request: ApprovedSourceRetrievalRequest,
        intent: ApprovedSourceRetrievalIntent,
        facade: PublicSourceRetrievalFacade,
    ) -> PublicSourceRetrievalExecution:
        search_request = None
        search_result = None
        source_uri = intent.source_uri
        if intent.mode is ApprovedSourceRetrievalMode.SEARCH_THEN_FETCH:
            assert intent.query is not None
            assert intent.search_provider_id is not None
            assert intent.query_approval_ref is not None
            search_request = _search_request(
                request=request,
                intent=intent,
            )
            try:
                search_result = await facade.search(search_request)
            except PublicSourceRetrievalFacadeError as exc:
                raise _blocked_from_facade(exc, search=True) from exc
            _validate_search_result(
                request=search_request,
                result=search_result,
            )
            allowed_hits = tuple(
                hit
                for hit in search_result.hits
                if _uri_allowed(
                    hit.source_uri,
                    request.allowed_schemes,
                    request.allowed_host_suffixes,
                )
            )
            if not allowed_hits:
                raise _RunnerBlocked(
                    ApprovedSourceRetrievalExecutionOutcome.BLOCKED_CAPABILITY,
                    ApprovedSourceEvidenceReason.SEARCH_RESULT_EMPTY,
                )
            source_uri = allowed_hits[0].source_uri
        assert source_uri is not None
        fetch_request = _fetch_request(
            request=request,
            intent=intent,
            source_uri=source_uri,
        )
        try:
            fetch_result = await facade.fetch(fetch_request)
        except PublicSourceRetrievalFacadeError as exc:
            raise _blocked_from_facade(exc, search=False) from exc
        _validate_fetch_result(
            request=fetch_request,
            result=fetch_result,
        )
        return PublicSourceRetrievalExecution(
            intent_id=intent.intent_id,
            search_request=search_request,
            search_result=search_result,
            fetch_request=fetch_request,
            fetch_result=fetch_result,
        )


class ApprovedSourceEvidenceCompiler:
    policy_version = APPROVED_SOURCE_RETRIEVAL_POLICY_VERSION

    def compile(
        self,
        *,
        request: ApprovedSourceRetrievalRequest,
        execution_result: ApprovedSourceRetrievalExecutionResult | None,
        verified_checks: tuple[VerifiedPublicSourceChecks, ...],
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
        dependency_discovery: PromptOnlyDependencyDiscoveryV2 | None,
        retrieval_policy: PublicSourceRetrievalPolicyV2,
        audit: ContractAudit,
    ) -> ApprovedSourceEvidenceResult:
        self._validate_current_request(
            request=request,
            attachment_planning_context=attachment_planning_context,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            producer_view_result=producer_view_result,
            producer_evidence_bundle=producer_evidence_bundle,
            dependency_discovery=dependency_discovery,
            retrieval_policy=retrieval_policy,
        )
        if not request.intents:
            if execution_result is not None or verified_checks:
                raise ApprovedSourceEvidencePolicyError(
                    "NOT_REQUIRED request cannot carry execution or checks"
                )
            return _evidence_result(
                request=request,
                outcome=ApprovedSourceEvidenceOutcome.NOT_REQUIRED,
                source_evidence_set=None,
                unresolved_intent_ids=(),
                reasons=frozenset(),
                audit=audit,
            )
        if execution_result is None:
            return _evidence_result(
                request=request,
                outcome=ApprovedSourceEvidenceOutcome.BLOCKED_CAPABILITY,
                source_evidence_set=None,
                unresolved_intent_ids=tuple(item.intent_id for item in request.intents),
                reasons=frozenset({ApprovedSourceEvidenceReason.FETCH_RESULT_MISSING}),
                audit=audit,
            )
        _validate_execution_result_identity(execution_result)
        if execution_result.request_ref != approved_source_retrieval_request_ref(request):
            raise ApprovedSourceEvidencePolicyError("retrieval execution result request is mismatched")
        if execution_result.outcome is not ApprovedSourceRetrievalExecutionOutcome.EXECUTED:
            mapped_outcome = (
                ApprovedSourceEvidenceOutcome.BLOCKED_POLICY
                if execution_result.outcome is ApprovedSourceRetrievalExecutionOutcome.BLOCKED_POLICY
                else ApprovedSourceEvidenceOutcome.BLOCKED_CAPABILITY
            )
            return _evidence_result(
                request=request,
                outcome=mapped_outcome,
                source_evidence_set=None,
                unresolved_intent_ids=(execution_result.unresolved_intent_ids),
                reasons=execution_result.reasons,
                audit=audit,
            )
        execution_by_intent = {item.intent_id: item for item in execution_result.executions}
        intent_ids = {item.intent_id for item in request.intents}
        if set(execution_by_intent) != intent_ids:
            raise ApprovedSourceEvidencePolicyError(
                "retrieval executions must exactly cover approved intents"
            )
        checks_by_intent = {item.intent_id: item for item in verified_checks}
        if len(checks_by_intent) != len(verified_checks):
            raise ApprovedSourceEvidencePolicyError("duplicate verified source checks")
        if set(checks_by_intent) != intent_ids:
            raise ApprovedSourceEvidencePolicyError("verified checks must exactly cover approved intents")
        self._validate_execution_checks(
            execution_by_intent,
            checks_by_intent,
        )
        blocked = _reduce_checks(checks_by_intent)
        if blocked is not None:
            outcome, unresolved_ids, reasons = blocked
            return _evidence_result(
                request=request,
                outcome=outcome,
                source_evidence_set=None,
                unresolved_intent_ids=unresolved_ids,
                reasons=reasons,
                audit=audit,
            )
        if dependency_discovery is None:
            raise ApprovedSourceEvidencePolicyError("compiled source evidence requires dependency discovery")
        discovery_ref = request.dependency_discovery_ref
        if discovery_ref is None:
            raise ApprovedSourceEvidencePolicyError("compiled source evidence request requires discovery ref")
        evidence_values = tuple(
            sorted(
                (
                    self._compile_source_evidence(
                        request=request,
                        intent=intent,
                        execution=execution_by_intent[intent.intent_id],
                        checks=checks_by_intent[intent.intent_id],
                        retrieval_policy=retrieval_policy,
                        audit=audit,
                    )
                    for intent in request.intents
                ),
                key=lambda item: item.source_evidence_id,
            )
        )
        request_refs = _retrieval_request_refs(execution_result.executions)
        evidence_set = SourceEvidenceSetV2(
            source_evidence_set_id="source-evidence-set://pending",
            attachment_planning_context_ref=(request.attachment_planning_context_ref),
            producer_task_view_ref=request.producer_task_view_ref,
            prompt_only_dependency_discovery_ref=(discovery_ref),
            retrieval_policy_ref=request.retrieval_policy_ref,
            retrieval_request_refs=request_refs,
            source_evidence=evidence_values,
            covered_target_refs=request.covered_target_refs,
            not_required_target_refs=request.not_required_target_refs,
            policy_version=self.policy_version,
            source_evidence_set_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    request.attachment_planning_context_ref,
                    request.producer_task_view_ref,
                    discovery_ref,
                    request.retrieval_policy_ref,
                    *request_refs,
                    *tuple(source_evidence_v2_ref(item) for item in evidence_values),
                    *request.covered_target_refs,
                    *request.not_required_target_refs,
                ),
            ),
        )
        digest = source_evidence_set_carried_sha256(evidence_set)
        evidence_set = evidence_set.model_copy(
            update={
                "source_evidence_set_id": (f"source-evidence-set://sha256/{digest}"),
                "source_evidence_set_sha256": digest,
            }
        )
        return _evidence_result(
            request=request,
            outcome=ApprovedSourceEvidenceOutcome.COMPILED,
            source_evidence_set=evidence_set,
            unresolved_intent_ids=(),
            reasons=frozenset(),
            audit=audit,
        )

    def validate_current(
        self,
        *,
        request: ApprovedSourceRetrievalRequest,
        execution_result: ApprovedSourceRetrievalExecutionResult,
        verified_checks: tuple[VerifiedPublicSourceChecks, ...],
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
        dependency_discovery: PromptOnlyDependencyDiscoveryV2,
        retrieval_policy: PublicSourceRetrievalPolicyV2,
        source_evidence_set: SourceEvidenceSetV2,
    ) -> None:
        try:
            _validate_source_evidence_set_identity(source_evidence_set)
            rebuilt = self.compile(
                request=request,
                execution_result=execution_result,
                verified_checks=verified_checks,
                attachment_planning_context=attachment_planning_context,
                producer_task_view=producer_task_view,
                storage_authorization=storage_authorization,
                producer_view_result=producer_view_result,
                producer_evidence_bundle=producer_evidence_bundle,
                dependency_discovery=dependency_discovery,
                retrieval_policy=retrieval_policy,
                audit=source_evidence_set.audit,
            )
            if (
                rebuilt.outcome is not ApprovedSourceEvidenceOutcome.COMPILED
                or rebuilt.source_evidence_set is None
                or not _same_evidence_set_except_audit_actor_time(
                    rebuilt.source_evidence_set,
                    source_evidence_set,
                )
            ):
                raise ApprovedSourceEvidencePolicyError(
                    "source evidence set does not match authoritative inputs"
                )
        except (ApprovedSourceEvidencePolicyError, ValueError) as exc:
            raise ApprovedSourceEvidencePolicyError(
                f"current approved source evidence validation failed: {exc}"
            ) from exc

    def _validate_current_request(
        self,
        *,
        request: ApprovedSourceRetrievalRequest,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
        dependency_discovery: PromptOnlyDependencyDiscoveryV2 | None,
        retrieval_policy: PublicSourceRetrievalPolicyV2,
    ) -> None:
        _validate_request_identity(request)
        definitions = tuple(
            ApprovedSourceRetrievalIntentDefinition(
                intent_id=item.intent_id,
                attachment_dependency_id=item.attachment_dependency_id,
                mode=item.mode,
                search_provider_id=item.search_provider_id,
                fetch_provider_id=item.fetch_provider_id,
                external_lead_projection_item_id=(item.external_lead_projection_item_id),
                query_approval_ref=item.query_approval_ref,
                source_approval_ref=item.source_approval_ref,
                usage_basis=item.usage_basis,
            )
            for item in request.intents
        )
        rebuilt = ApprovedSourceRetrievalRequestBuilder().build(
            attachment_planning_context=attachment_planning_context,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            producer_view_result=producer_view_result,
            producer_evidence_bundle=producer_evidence_bundle,
            dependency_discovery=dependency_discovery,
            retrieval_policy=retrieval_policy,
            intent_definitions=definitions,
            audit=request.audit,
        )
        if not _same_request_except_audit_actor_time(rebuilt, request):
            raise ApprovedSourceEvidencePolicyError(
                "approved source retrieval request is stale or mismatched"
            )

    def _validate_execution_checks(
        self,
        executions: dict[str, PublicSourceRetrievalExecution],
        checks: dict[str, VerifiedPublicSourceChecks],
    ) -> None:
        for intent_id, execution in executions.items():
            _validate_execution(execution)
            check = checks[intent_id]
            fetch = execution.fetch_result
            content_ref = _factory_ref(fetch.content_ref)
            if (
                check.fetch_result_ref != _factory_ref(public_source_fetch_result_ref(fetch))
                or check.content_ref != content_ref
                or check.content_sha256 != fetch.content_sha256
            ):
                raise ApprovedSourceEvidencePolicyError(
                    "verified checks are not bound to the exact fetch result"
                )

    def _compile_source_evidence(
        self,
        *,
        request: ApprovedSourceRetrievalRequest,
        intent: ApprovedSourceRetrievalIntent,
        execution: PublicSourceRetrievalExecution,
        checks: VerifiedPublicSourceChecks,
        retrieval_policy: PublicSourceRetrievalPolicyV2,
        audit: ContractAudit,
    ) -> SourceEvidenceV2:
        assert checks.secret_scan_ref is not None
        assert checks.pii_scan_ref is not None
        assert checks.prompt_injection_scan_ref is not None
        assert checks.answer_leakage_scan_ref is not None
        assert checks.license_assessment_ref is not None
        assessment = PublicSourceSafetyAssessmentV2(
            public_source_safety_assessment_id=("public-source-safety-assessment://pending"),
            fetch_result_ref=checks.fetch_result_ref,
            content_ref=checks.content_ref,
            content_sha256=checks.content_sha256,
            secret_scan_ref=checks.secret_scan_ref,
            pii_scan_ref=checks.pii_scan_ref,
            prompt_injection_scan_ref=(checks.prompt_injection_scan_ref),
            answer_leakage_scan_ref=checks.answer_leakage_scan_ref,
            license_assessment_ref=checks.license_assessment_ref,
            passed=True,
            policy_version=self.policy_version,
            public_source_safety_assessment_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    checks.fetch_result_ref,
                    checks.content_ref,
                    checks.secret_scan_ref,
                    checks.pii_scan_ref,
                    checks.prompt_injection_scan_ref,
                    checks.answer_leakage_scan_ref,
                    checks.license_assessment_ref,
                ),
            ),
        )
        assessment_digest = public_source_safety_assessment_carried_sha256(assessment)
        assessment = assessment.model_copy(
            update={
                "public_source_safety_assessment_id": (
                    f"public-source-safety-assessment://sha256/{assessment_digest}"
                ),
                "public_source_safety_assessment_sha256": (assessment_digest),
            }
        )
        claim = SourceEvidenceClaimBindingV2(
            artifact_evidence_target_ref=(intent.artifact_evidence_target_ref),
            attachment_dependency_id=intent.attachment_dependency_id,
            supported_claim_ids=(intent.attachment_dependency_id,),
            usage_basis=intent.usage_basis.value,
        )
        route_ref = (
            intent.external_lead_ref
            if intent.external_lead_ref is not None
            else _search_result_factory_ref(execution)
        )
        assert route_ref is not None
        assessment_ref = public_source_safety_assessment_ref(assessment)
        fetch_ref = _factory_ref(public_source_fetch_result_ref(execution.fetch_result))
        provenance = _fresh_content_provenance(
            content_ref=checks.content_ref,
            content_sha256=checks.content_sha256,
            route_refs=tuple(
                ref
                for ref in (
                    route_ref,
                    intent.external_lead_decision_ref,
                    fetch_ref,
                    assessment_ref,
                )
                if ref is not None
            ),
            audit=audit,
        )
        provenance_ref = ObjectRef(
            object_type="provenance-decision",
            object_id=provenance.provenance_decision_id,
            object_version="v1",
            object_sha256=provenance.canonical_sha256(),
        )
        fetch = execution.fetch_result
        frozen = SourceEvidence(
            source_evidence_id=(
                f"source-evidence://sha256/{
                    _payload_sha256(
                        {
                            'intent_id': intent.intent_id,
                            'fetch_result_ref': _ref_payload(fetch_ref),
                            'target_ref': _ref_payload(intent.artifact_evidence_target_ref),
                        }
                    )
                }"
            ),
            source_uri=fetch.canonical_source_uri,
            retrieved_at=fetch.retrieved_at.isoformat(),
            retrieval_policy_version=retrieval_policy.policy_version,
            usage_basis=intent.usage_basis.value,
            content_ref=checks.content_ref,
            content_sha256=checks.content_sha256,
            supported_claim_ids=claim.supported_claim_ids,
            provenance_decision_ref=provenance_ref,
            audit=_safe_audit(
                audit,
                (
                    fetch_ref,
                    assessment_ref,
                    provenance_ref,
                    intent.artifact_evidence_target_ref,
                ),
            ),
        )
        search_result_ref = (
            None
            if execution.search_result is None
            else _factory_ref(public_source_search_result_ref(execution.search_result))
        )
        value = SourceEvidenceV2(
            source_evidence_id="source-evidence-v2://pending",
            attachment_planning_context_ref=(request.attachment_planning_context_ref),
            producer_task_view_ref=request.producer_task_view_ref,
            prompt_only_dependency_discovery_ref=(_required_discovery_ref(request)),
            retrieval_policy_ref=request.retrieval_policy_ref,
            external_lead_ref=intent.external_lead_ref,
            external_lead_decision_ref=(intent.external_lead_decision_ref),
            search_result_ref=search_result_ref,
            fetch_result_ref=fetch_ref,
            safety_assessment_ref=assessment_ref,
            claim_bindings=(claim,),
            source_evidence_v1=frozen,
            content_provenance_decision=provenance,
            policy_version=self.policy_version,
            source_evidence_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    request.attachment_planning_context_ref,
                    request.producer_task_view_ref,
                    _required_discovery_ref(request),
                    request.retrieval_policy_ref,
                    route_ref,
                    fetch_ref,
                    assessment_ref,
                    provenance_ref,
                    intent.artifact_evidence_target_ref,
                ),
            ),
        )
        digest = source_evidence_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "source_evidence_id": (f"source-evidence-v2://sha256/{digest}"),
                "source_evidence_sha256": digest,
            }
        )


class _RunnerBlocked(RuntimeError):
    def __init__(
        self,
        outcome: ApprovedSourceRetrievalExecutionOutcome,
        reason: ApprovedSourceEvidenceReason,
    ) -> None:
        super().__init__(reason.value)
        self.outcome = outcome
        self.reason = reason


def _request(
    *,
    context_ref: ObjectRef,
    view_ref: ObjectRef,
    evidence_view_ref: ObjectRef,
    bundle_ref: ObjectRef,
    discovery_ref: ObjectRef | None,
    policy: PublicSourceRetrievalPolicyV2,
    policy_ref: ObjectRef,
    intents: tuple[ApprovedSourceRetrievalIntent, ...],
    covered_target_refs: tuple[ObjectRef, ...],
    not_required_target_refs: tuple[ObjectRef, ...],
    audit: ContractAudit,
) -> ApprovedSourceRetrievalRequest:
    value = ApprovedSourceRetrievalRequest(
        request_id="approved-source-retrieval-request://pending",
        attachment_planning_context_ref=context_ref,
        producer_task_view_ref=view_ref,
        producer_evidence_view_ref=evidence_view_ref,
        producer_evidence_bundle_ref=bundle_ref,
        dependency_discovery_ref=discovery_ref,
        retrieval_policy_ref=policy_ref,
        allowed_schemes=policy.allowed_schemes,
        allowed_host_suffixes=policy.allowed_host_suffixes,
        max_search_results=policy.max_search_results,
        max_fetch_bytes=policy.max_fetch_bytes,
        intents=intents,
        covered_target_refs=covered_target_refs,
        not_required_target_refs=not_required_target_refs,
        policy_version=APPROVED_SOURCE_RETRIEVAL_POLICY_VERSION,
        request_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            tuple(
                ref
                for ref in (
                    context_ref,
                    view_ref,
                    evidence_view_ref,
                    bundle_ref,
                    discovery_ref,
                    policy_ref,
                    *covered_target_refs,
                    *not_required_target_refs,
                )
                if ref is not None
            ),
        ),
    )
    digest = approved_source_retrieval_request_carried_sha256(value)
    return value.model_copy(
        update={
            "request_id": (f"approved-source-retrieval-request://sha256/{digest}"),
            "request_sha256": digest,
        }
    )


def _search_request(
    *,
    request: ApprovedSourceRetrievalRequest,
    intent: ApprovedSourceRetrievalIntent,
) -> PublicSourceSearchRequestV2:
    assert intent.query is not None
    assert intent.search_provider_id is not None
    assert intent.query_approval_ref is not None
    value = PublicSourceSearchRequestV2(
        search_request_id="public-source-search-request://pending",
        query=intent.query,
        provider_id=intent.search_provider_id,
        result_limit=request.max_search_results,
        allowed_schemes=request.allowed_schemes,
        allowed_host_suffixes=request.allowed_host_suffixes,
        query_approval_ref=_facade_ref(intent.query_approval_ref),
        retrieval_policy_ref=_facade_ref(request.retrieval_policy_ref),
        idempotency_key=(
            f"public-source-search://sha256/{
                _payload_sha256(
                    {
                        'request_id': request.request_id,
                        'intent_id': intent.intent_id,
                    }
                )
            }"
        ),
        search_request_sha256="0" * 64,
    )
    digest = public_source_search_request_carried_sha256(value)
    return value.model_copy(
        update={
            "search_request_id": (f"public-source-search-request://sha256/{digest}"),
            "search_request_sha256": digest,
        }
    )


def _fetch_request(
    *,
    request: ApprovedSourceRetrievalRequest,
    intent: ApprovedSourceRetrievalIntent,
    source_uri: str,
) -> PublicSourceFetchRequestV2:
    value = PublicSourceFetchRequestV2(
        fetch_request_id="public-source-fetch-request://pending",
        source_uri=source_uri,
        provider_id=intent.fetch_provider_id,
        max_bytes=request.max_fetch_bytes,
        allowed_schemes=request.allowed_schemes,
        allowed_host_suffixes=request.allowed_host_suffixes,
        source_approval_ref=_facade_ref(intent.source_approval_ref),
        retrieval_policy_ref=_facade_ref(request.retrieval_policy_ref),
        idempotency_key=(
            f"public-source-fetch://sha256/{
                _payload_sha256(
                    {
                        'request_id': request.request_id,
                        'intent_id': intent.intent_id,
                        'source_uri': source_uri,
                    }
                )
            }"
        ),
        fetch_request_sha256="0" * 64,
    )
    digest = public_source_fetch_request_carried_sha256(value)
    return value.model_copy(
        update={
            "fetch_request_id": (f"public-source-fetch-request://sha256/{digest}"),
            "fetch_request_sha256": digest,
        }
    )


def _execution_result(
    *,
    request: ApprovedSourceRetrievalRequest,
    outcome: ApprovedSourceRetrievalExecutionOutcome,
    executions: tuple[PublicSourceRetrievalExecution, ...],
    unresolved_intent_ids: tuple[str, ...],
    reasons: frozenset[ApprovedSourceEvidenceReason],
    audit: ContractAudit,
) -> ApprovedSourceRetrievalExecutionResult:
    value = ApprovedSourceRetrievalExecutionResult(
        execution_result_id=("approved-source-retrieval-execution-result://pending"),
        request_ref=approved_source_retrieval_request_ref(request),
        outcome=outcome,
        executions=executions,
        unresolved_intent_ids=tuple(sorted(unresolved_intent_ids)),
        reasons=reasons,
        execution_result_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (approved_source_retrieval_request_ref(request),),
        ),
    )
    digest = approved_source_execution_result_carried_sha256(value)
    return value.model_copy(
        update={
            "execution_result_id": (f"approved-source-retrieval-execution-result://sha256/{digest}"),
            "execution_result_sha256": digest,
        }
    )


def _evidence_result(
    *,
    request: ApprovedSourceRetrievalRequest,
    outcome: ApprovedSourceEvidenceOutcome,
    source_evidence_set: SourceEvidenceSetV2 | None,
    unresolved_intent_ids: tuple[str, ...],
    reasons: frozenset[ApprovedSourceEvidenceReason],
    audit: ContractAudit,
) -> ApprovedSourceEvidenceResult:
    value = ApprovedSourceEvidenceResult(
        result_id="approved-source-evidence-result://pending",
        request_ref=approved_source_retrieval_request_ref(request),
        outcome=outcome,
        source_evidence_set=source_evidence_set,
        unresolved_intent_ids=tuple(sorted(unresolved_intent_ids)),
        reasons=reasons,
        result_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            tuple(
                ref
                for ref in (
                    approved_source_retrieval_request_ref(request),
                    (
                        None
                        if source_evidence_set is None
                        else ObjectRef(
                            object_type="source-evidence-set",
                            object_id=(source_evidence_set.source_evidence_set_id),
                            object_version="v2",
                            object_sha256=(source_evidence_set.source_evidence_set_sha256),
                        )
                    ),
                )
                if ref is not None
            ),
        ),
    )
    digest = approved_source_evidence_result_carried_sha256(value)
    return value.model_copy(
        update={
            "result_id": (f"approved-source-evidence-result://sha256/{digest}"),
            "result_sha256": digest,
        }
    )


def _reduce_checks(
    checks: dict[str, VerifiedPublicSourceChecks],
) -> (
    tuple[
        ApprovedSourceEvidenceOutcome,
        tuple[str, ...],
        frozenset[ApprovedSourceEvidenceReason],
    ]
    | None
):
    safety_ids: set[str] = set()
    safety_reasons: set[ApprovedSourceEvidenceReason] = set()
    policy_ids: set[str] = set()
    capability_ids: set[str] = set()
    for intent_id, value in checks.items():
        safety_failures = value.failed_checks & set(_SAFETY_CHECK_REASONS)
        if safety_failures:
            safety_ids.add(intent_id)
            safety_reasons.update(_SAFETY_CHECK_REASONS[item] for item in safety_failures)
        if PublicSourceCheck.LICENSE in value.failed_checks:
            policy_ids.add(intent_id)
        if value.unavailable_checks:
            capability_ids.add(intent_id)
    if safety_ids:
        return (
            ApprovedSourceEvidenceOutcome.BLOCKED_SAFETY,
            tuple(sorted(safety_ids)),
            frozenset(safety_reasons),
        )
    if policy_ids:
        return (
            ApprovedSourceEvidenceOutcome.BLOCKED_POLICY,
            tuple(sorted(policy_ids)),
            frozenset({ApprovedSourceEvidenceReason.LICENSE_DENIED}),
        )
    if capability_ids:
        return (
            ApprovedSourceEvidenceOutcome.BLOCKED_CAPABILITY,
            tuple(sorted(capability_ids)),
            frozenset({ApprovedSourceEvidenceReason.SCAN_CAPABILITY_UNAVAILABLE}),
        )
    return None


def _fresh_content_provenance(
    *,
    content_ref: ObjectRef,
    content_sha256: str,
    route_refs: tuple[ObjectRef, ...],
    audit: ContractAudit,
) -> ProvenanceDecision:
    derived_from = tuple(sorted(set(route_refs), key=_ref_key))
    payload = {
        "subject_ref": _ref_payload(content_ref),
        "origin_class": OriginClass.SYSTEM_OR_HARNESS_CONTEXT.value,
        "visibility": Visibility.STAGE_PROJECTION.value,
        "disposition": Disposition.ALLOW_INPUT_EVIDENCE.value,
        "derived_from": [_ref_payload(ref) for ref in derived_from],
        "rule_ids": ["public-source-refetch/r5-04-v1"],
        "policy_version": "public-source-refetch/r5-04-v1",
        "subject_sha256": content_sha256,
    }
    digest = _payload_sha256(payload)
    return ProvenanceDecision(
        provenance_decision_id=(f"provenance-decision://public-source/sha256/{digest}"),
        subject_ref=content_ref,
        origin_class=OriginClass.SYSTEM_OR_HARNESS_CONTEXT,
        taint_labels=frozenset(),
        content_risk_labels=frozenset(),
        visibility=Visibility.STAGE_PROJECTION,
        disposition=Disposition.ALLOW_INPUT_EVIDENCE,
        derived_from=derived_from,
        rule_ids=("public-source-refetch/r5-04-v1",),
        source_event_refs=(),
        confidence=1.0,
        review_required=False,
        policy_version="public-source-refetch/r5-04-v1",
        subject_sha256=content_sha256,
        audit=_safe_audit(
            audit,
            (content_ref, *derived_from),
        ),
    )


def _validate_external_lead(item: EvidenceProjectionItem) -> None:
    decision = item.child_decision
    if (
        item.projection_mode is not EvidenceProjectionMode.EXTERNAL_LEAD
        or item.content is not None
        or item.structure_fields
        or item.external_uri is None
        or decision.subject_ref != item.projected_ref
        or decision.origin_class is not OriginClass.AGENT_RETRIEVED_EXTERNAL
        or decision.disposition is not Disposition.ALLOW_EXTERNAL_LEAD_ONLY
        or decision.taint_labels
        or decision.content_risk_labels
    ):
        raise ApprovedSourceEvidencePolicyError("external lead projection is unsafe or invalid")


def _validate_retrieval_policy_identity(
    policy: PublicSourceRetrievalPolicyV2,
) -> None:
    digest = public_source_retrieval_policy_carried_sha256(policy)
    if (
        policy.public_source_retrieval_policy_sha256 != digest
        or policy.public_source_retrieval_policy_id != f"public-source-retrieval-policy://sha256/{digest}"
    ):
        raise ApprovedSourceEvidencePolicyError("public source retrieval policy identity is stale")


def _validate_request_identity(
    request: ApprovedSourceRetrievalRequest,
) -> None:
    digest = approved_source_retrieval_request_carried_sha256(request)
    if (
        request.request_sha256 != digest
        or request.request_id != f"approved-source-retrieval-request://sha256/{digest}"
    ):
        raise ApprovedSourceEvidencePolicyError("approved source retrieval request identity is stale")


def _validate_execution_result_identity(
    result: ApprovedSourceRetrievalExecutionResult,
) -> None:
    digest = approved_source_execution_result_carried_sha256(result)
    if (
        result.execution_result_sha256 != digest
        or result.execution_result_id != f"approved-source-retrieval-execution-result://sha256/{digest}"
    ):
        raise ApprovedSourceEvidencePolicyError("retrieval execution result identity is stale")
    for execution in result.executions:
        _validate_execution(execution)


def _validate_execution(
    execution: PublicSourceRetrievalExecution,
) -> None:
    if execution.search_request is not None:
        search_digest = public_source_search_request_carried_sha256(execution.search_request)
        if (
            execution.search_request.search_request_sha256 != search_digest
            or execution.search_request.search_request_id
            != f"public-source-search-request://sha256/{search_digest}"
        ):
            raise ApprovedSourceEvidencePolicyError("search request identity is stale")
    if execution.search_result is not None:
        digest = public_source_search_result_carried_sha256(execution.search_result)
        if (
            execution.search_result.search_result_sha256 != digest
            or execution.search_result.search_result_id != f"public-source-search-result://sha256/{digest}"
        ):
            raise ApprovedSourceEvidencePolicyError("search result identity is stale")
    fetch_digest = public_source_fetch_request_carried_sha256(execution.fetch_request)
    if (
        execution.fetch_request.fetch_request_sha256 != fetch_digest
        or execution.fetch_request.fetch_request_id != f"public-source-fetch-request://sha256/{fetch_digest}"
    ):
        raise ApprovedSourceEvidencePolicyError("fetch request identity is stale")
    result_digest = public_source_fetch_result_carried_sha256(execution.fetch_result)
    if (
        execution.fetch_result.fetch_result_sha256 != result_digest
        or execution.fetch_result.fetch_result_id != f"public-source-fetch-result://sha256/{result_digest}"
    ):
        raise ApprovedSourceEvidencePolicyError("fetch result identity is stale")


def _validate_search_result(
    *,
    request: PublicSourceSearchRequestV2,
    result: PublicSourceSearchResultV2,
) -> None:
    digest = public_source_search_result_carried_sha256(result)
    if (
        result.search_result_sha256 != digest
        or result.search_result_id != f"public-source-search-result://sha256/{digest}"
        or result.search_request_ref != public_source_search_request_ref(request)
        or result.provider_id != request.provider_id
    ):
        raise ApprovedSourceEvidencePolicyError("search result is stale or mismatched")


def _validate_fetch_result(
    *,
    request: PublicSourceFetchRequestV2,
    result: PublicSourceFetchResultV2,
) -> None:
    digest = public_source_fetch_result_carried_sha256(result)
    if (
        result.fetch_result_sha256 != digest
        or result.fetch_result_id != f"public-source-fetch-result://sha256/{digest}"
        or result.fetch_request_ref != public_source_fetch_request_ref(request)
        or result.provider_id != request.provider_id
        or result.requested_source_uri != request.source_uri
        or not 200 <= result.status_code < 300
        or result.bytes_count > request.max_bytes
    ):
        raise ApprovedSourceEvidencePolicyError("fetch result is stale or mismatched")


def _blocked_from_facade(
    error: PublicSourceRetrievalFacadeError,
    *,
    search: bool,
) -> _RunnerBlocked:
    if error.kind is PublicSourceRetrievalFacadeFailureKind.POLICY:
        return _RunnerBlocked(
            ApprovedSourceRetrievalExecutionOutcome.BLOCKED_POLICY,
            ApprovedSourceEvidenceReason.URI_POLICY_DENIED,
        )
    return _RunnerBlocked(
        ApprovedSourceRetrievalExecutionOutcome.BLOCKED_CAPABILITY,
        (ApprovedSourceEvidenceReason.SEARCH_FAILED if search else ApprovedSourceEvidenceReason.FETCH_FAILED),
    )


def _retrieval_request_refs(
    executions: tuple[PublicSourceRetrievalExecution, ...],
) -> tuple[ObjectRef, ...]:
    refs: list[ObjectRef] = []
    for execution in executions:
        if execution.search_request is not None:
            refs.append(_factory_ref(public_source_search_request_ref(execution.search_request)))
        refs.append(_factory_ref(public_source_fetch_request_ref(execution.fetch_request)))
    return tuple(sorted(set(refs), key=_ref_key))


def _validate_source_evidence_set_identity(
    evidence_set: SourceEvidenceSetV2,
) -> None:
    for evidence in evidence_set.source_evidence:
        digest = source_evidence_v2_carried_sha256(evidence)
        if (
            evidence.source_evidence_sha256 != digest
            or evidence.source_evidence_id != f"source-evidence-v2://sha256/{digest}"
        ):
            raise ApprovedSourceEvidencePolicyError("source evidence identity is stale")
    digest = source_evidence_set_carried_sha256(evidence_set)
    if (
        evidence_set.source_evidence_set_sha256 != digest
        or evidence_set.source_evidence_set_id != f"source-evidence-set://sha256/{digest}"
    ):
        raise ApprovedSourceEvidencePolicyError("source evidence set identity is stale")


def _same_request_except_audit_actor_time(
    expected: ApprovedSourceRetrievalRequest,
    observed: ApprovedSourceRetrievalRequest,
) -> bool:
    return expected.model_dump(mode="json", exclude={"audit"}) == observed.model_dump(
        mode="json", exclude={"audit"}
    ) and _same_audit_lineage(expected.audit, observed.audit)


def _same_evidence_set_except_audit_actor_time(
    expected: SourceEvidenceSetV2,
    observed: SourceEvidenceSetV2,
) -> bool:
    if expected.model_dump(
        mode="json",
        exclude={"audit", "source_evidence"},
    ) != observed.model_dump(
        mode="json",
        exclude={"audit", "source_evidence"},
    ) or len(expected.source_evidence) != len(observed.source_evidence):
        return False
    for expected_item, observed_item in zip(
        expected.source_evidence,
        observed.source_evidence,
        strict=True,
    ):
        if (
            expected_item.model_dump(
                mode="json",
                exclude={
                    "audit": True,
                    "source_evidence_v1": {"audit"},
                    "content_provenance_decision": {"audit"},
                },
            )
            != observed_item.model_dump(
                mode="json",
                exclude={
                    "audit": True,
                    "source_evidence_v1": {"audit"},
                    "content_provenance_decision": {"audit"},
                },
            )
            or not _same_audit_lineage(
                expected_item.audit,
                observed_item.audit,
            )
            or not _same_audit_lineage(
                expected_item.source_evidence_v1.audit,
                observed_item.source_evidence_v1.audit,
            )
            or not _same_audit_lineage(
                expected_item.content_provenance_decision.audit,
                observed_item.content_provenance_decision.audit,
            )
        ):
            return False
    return _same_audit_lineage(expected.audit, observed.audit)


def _same_audit_lineage(
    expected: ContractAudit,
    observed: ContractAudit,
) -> bool:
    return (
        expected.governing_versions == observed.governing_versions
        and expected.input_refs == observed.input_refs
    )


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=tuple(sorted(set(refs), key=_ref_key)),
    )


def _evidence_view_result_ref(result: EvidenceViewResult) -> ObjectRef:
    digest = _identity_digest(result.view_result_id)
    return ObjectRef(
        object_type="evidence-view-result",
        object_id=result.view_result_id,
        object_version="r2-05",
        object_sha256=digest,
    )


def _provenance_decision_ref(
    decision: ProvenanceDecision,
) -> ObjectRef:
    return ObjectRef(
        object_type="provenance-decision",
        object_id=decision.provenance_decision_id,
        object_version=decision.policy_version,
        object_sha256=decision.canonical_sha256(),
    )


def _factory_ref(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef.model_validate(ref.model_dump(mode="python"))


def _search_result_factory_ref(
    execution: PublicSourceRetrievalExecution,
) -> ObjectRef:
    if execution.search_result is None:
        raise ApprovedSourceEvidencePolicyError("search route requires a search result")
    return _factory_ref(public_source_search_result_ref(execution.search_result))


def _required_discovery_ref(
    request: ApprovedSourceRetrievalRequest,
) -> ObjectRef:
    if request.dependency_discovery_ref is None:
        raise ApprovedSourceEvidencePolicyError("compiled source evidence requires discovery ref")
    return request.dependency_discovery_ref


def _facade_ref(ref: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _uri_allowed(
    uri: str,
    allowed_schemes: tuple[str, ...],
    allowed_host_suffixes: tuple[str, ...],
) -> bool:
    parsed = urlsplit(uri)
    host = (parsed.hostname or "").casefold().rstrip(".")
    return (
        parsed.scheme in allowed_schemes
        and bool(host)
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
        and any(host == suffix or host.endswith(f".{suffix}") for suffix in allowed_host_suffixes)
    )


def _identity_digest(identifier: str) -> str:
    marker = "://sha256/"
    if marker not in identifier:
        raise ApprovedSourceEvidencePolicyError("source identity is not content-addressed")
    digest = identifier.rsplit(marker, maxsplit=1)[1]
    if len(digest) != 64:
        raise ApprovedSourceEvidencePolicyError("source identity digest is invalid")
    return digest


def _require_unique(label: str, values: tuple[object, ...]) -> None:
    if len(values) != len(set(values)):
        raise ApprovedSourceEvidencePolicyError(f"{label} must be unique")


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
