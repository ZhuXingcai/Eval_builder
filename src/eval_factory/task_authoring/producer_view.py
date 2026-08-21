from __future__ import annotations

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.task_v2 import (
    ContestantToolPolicyV2,
    EvaluatorSpecV2,
    ProducerAttachmentRequirementV2,
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
    ReferencePolicyV2,
    RubricSetV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    ToolPolicyV2,
    ToolRuleActionV2,
    contestant_tool_policy_carried_sha256,
    contestant_tool_policy_ref,
    contestant_tool_rule_carried_sha256,
    evaluator_binding_carried_sha256,
    evaluator_failure_rule_carried_sha256,
    evaluator_spec_carried_sha256,
    evaluator_spec_ref,
    producer_storage_authorization_carried_sha256,
    producer_storage_authorization_ref,
    producer_task_view_carried_sha256,
    producer_task_view_ref,
    reference_policy_carried_sha256,
    reference_policy_ref,
    rubric_criterion_carried_sha256,
    rubric_reachability_carried_sha256,
    rubric_set_carried_sha256,
    rubric_set_ref,
    task_draft_carried_sha256,
    task_draft_ref,
    tool_policy_carried_sha256,
    tool_policy_ref,
    tool_rule_carried_sha256,
)
from eval_factory.provenance.bundles import (
    EvidenceCompilationPolicyError,
    evidence_bundle_ref,
    projection_policy_ref,
    validate_evidence_bundle_identity,
    validate_projection_policy,
)
from eval_factory.provenance.injection import (
    ProducerTaskDataBoundaryRequest,
    PromptBoundaryEnforcementRequest,
    PromptBoundaryEnforcementResult,
    PromptBoundarySegment,
    PromptBoundarySourceRole,
    PromptBoundarySurface,
    PromptInjectionBoundaryEnforcer,
    PromptInjectionBoundaryPolicyError,
)
from eval_factory.provenance.views import (
    EvidenceViewPolicyError,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewResult,
    validate_evidence_view_result_identity,
)
from eval_factory.task_authoring.producer_models import (
    PRODUCER_TASK_VIEW_POLICY_VERSION,
    ProducerStorageAccessOutcome,
    ProducerStorageAccessReason,
    ProducerStorageAccessRequest,
    ProducerStorageAccessResult,
    ProducerTaskViewPolicyError,
    ProducerTaskViewProjectionOutcome,
    ProducerTaskViewProjectionReason,
    ProducerTaskViewProjectionResult,
    producer_payload_sha256,
    producer_storage_access_request_carried_sha256,
    producer_storage_access_request_ref,
)

_DENIED_STORAGE_REF_MARKERS = frozenset(
    {
        "answer-bearing",
        "configured-pii",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "private-reference",
        "quarantine",
        "raw-trace",
        "raw-traj",
        "restricted-pii",
        "secret",
        "sensitive-pii",
        "trace-raw",
    }
)


class ProducerTaskViewCompiler:
    policy_version = PRODUCER_TASK_VIEW_POLICY_VERSION

    def compile(
        self,
        *,
        task_draft: TaskDraftV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
        contestant_tool_policy: ContestantToolPolicyV2,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
        audit: ContractAudit,
        previous_producer_task_view: ProducerTaskViewV2 | None = None,
    ) -> ProducerTaskViewProjectionResult:
        validate_task_draft_identity(task_draft)
        if previous_producer_task_view is not None:
            validate_producer_task_view_identity(previous_producer_task_view)
        if task_draft.prompt_safety_status is not TaskDraftPromptSafetyStatusV2.PASSED:
            reason = (
                ProducerTaskViewProjectionReason.PROMPT_SAFETY_PENDING
                if task_draft.prompt_safety_status is TaskDraftPromptSafetyStatusV2.PENDING
                else ProducerTaskViewProjectionReason.PROMPT_SAFETY_BLOCKED
            )
            return _projection_result(
                task_draft=task_draft,
                outcome=ProducerTaskViewProjectionOutcome.BLOCKED_SAFETY,
                producer_task_view=None,
                storage_authorization=None,
                unresolved_reasons=frozenset({reason}),
                audit=audit,
                internal_refs=(task_draft_ref(task_draft),),
            )

        validate_rubric_set_identity(rubric_set)
        validate_evaluator_spec_identity(evaluator_spec)
        validate_reference_policy_identity(reference_policy, evaluator_spec)
        validate_tool_policy_identity(tool_policy)
        validate_contestant_tool_policy_identity(contestant_tool_policy)
        validate_r4_contract_chain(
            task_draft=task_draft,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
            contestant_tool_policy=contestant_tool_policy,
        )
        _validate_producer_evidence(
            producer_view_result,
            producer_evidence_bundle,
        )

        draft_ref = task_draft_ref(task_draft)
        rubric_ref = rubric_set_ref(rubric_set)
        evaluator_ref = evaluator_spec_ref(evaluator_spec)
        reference_ref = reference_policy_ref(reference_policy)
        tool_ref = tool_policy_ref(tool_policy)
        contestant_ref = contestant_tool_policy_ref(contestant_tool_policy)
        bundle_ref = evidence_bundle_ref(producer_evidence_bundle)
        policy_ref = projection_policy_ref(producer_view_result.projection_policy)
        contract_chain_sha256 = r4_contract_chain_sha256(
            draft_ref=draft_ref,
            rubric_ref=rubric_ref,
            evaluator_ref=evaluator_ref,
            reference_ref=reference_ref,
            tool_ref=tool_ref,
            contestant_ref=contestant_ref,
        )
        requirements = tuple(
            ProducerAttachmentRequirementV2(
                dependency_id=item.dependency_id,
                description=item.description,
                criticality=item.criticality,
            )
            for item in sorted(
                task_draft.attachment_dependencies,
                key=lambda dependency: dependency.dependency_id,
            )
        )
        allowed_tools = tuple(sorted(rule.tool_id for rule in contestant_tool_policy.rules))
        storage_authorization = _compile_storage_authorization(
            task_draft=task_draft,
            producer_view_result=producer_view_result,
            producer_evidence_bundle=producer_evidence_bundle,
            audit=audit,
        )
        authorization_ref = producer_storage_authorization_ref(storage_authorization)
        projection_payload = _producer_projection_payload(
            task_draft=task_draft,
            attachment_requirements=requirements,
            allowed_tools=allowed_tools,
            bundle_ref=bundle_ref,
            projection_policy_ref=policy_ref,
            storage_authorization_ref=authorization_ref,
            contestant_tool_policy=contestant_tool_policy,
            contract_chain_sha256=contract_chain_sha256,
        )
        projection_ref = _producer_projection_ref(projection_payload)
        boundary = _enforce_producer_boundary(
            task_draft=task_draft,
            contestant_tool_policy=contestant_tool_policy,
            producer_projection_ref=projection_ref,
            bundle_refs=(bundle_ref,),
            audit=audit,
        )
        boundary_ref = _prompt_boundary_ref(boundary)
        view = ProducerTaskViewV2(
            producer_task_view_id="producer-task-view://pending",
            producer_task_view_version=(
                1
                if previous_producer_task_view is None
                else previous_producer_task_view.producer_task_view_version + 1
            ),
            supersedes_producer_task_view_ref=(
                None
                if previous_producer_task_view is None
                else producer_task_view_ref(previous_producer_task_view)
            ),
            query_instruction=task_draft.visible_prompt,
            attachment_requirements=requirements,
            allowed_tools=allowed_tools,
            safe_evidence_bundle_refs=(bundle_ref,),
            forbidden_outputs=task_draft.forbidden_outputs,
            projection_policy_ref=policy_ref,
            storage_authorization_ref=authorization_ref,
            prompt_boundary_enforcement_ref=boundary_ref,
            source_task_draft_sha256=task_draft.task_draft_sha256,
            source_contestant_tool_policy_sha256=(contestant_tool_policy.contestant_tool_policy_sha256),
            source_contract_chain_sha256=contract_chain_sha256,
            policy_version=self.policy_version,
            producer_task_view_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    bundle_ref,
                    policy_ref,
                    authorization_ref,
                    boundary_ref,
                ),
            ),
        )
        view_digest = producer_task_view_carried_sha256(view)
        view = view.model_copy(
            update={
                "producer_task_view_id": (f"producer-task-view://sha256/{view_digest}"),
                "producer_task_view_sha256": view_digest,
            }
        )
        return _projection_result(
            task_draft=task_draft,
            outcome=ProducerTaskViewProjectionOutcome.PROJECTED,
            producer_task_view=view,
            storage_authorization=storage_authorization,
            unresolved_reasons=frozenset(),
            audit=audit,
            internal_refs=(
                draft_ref,
                rubric_ref,
                evaluator_ref,
                reference_ref,
                tool_ref,
                contestant_ref,
                bundle_ref,
                producer_task_view_ref(view),
                authorization_ref,
            ),
        )

    def validate_current(
        self,
        *,
        task_draft: TaskDraftV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
        contestant_tool_policy: ContestantToolPolicyV2,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
    ) -> None:
        try:
            validate_producer_task_view_identity(producer_task_view)
            validate_storage_authorization_identity(storage_authorization)
            if producer_task_view.storage_authorization_ref != producer_storage_authorization_ref(
                storage_authorization
            ):
                raise ProducerTaskViewPolicyError(
                    "producer view storage authorization ref is stale or mismatched"
                )
            rebuilt = self.compile(
                task_draft=task_draft,
                rubric_set=rubric_set,
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                tool_policy=tool_policy,
                contestant_tool_policy=contestant_tool_policy,
                producer_view_result=producer_view_result,
                producer_evidence_bundle=producer_evidence_bundle,
                audit=producer_task_view.audit,
            )
            if (
                rebuilt.outcome is not ProducerTaskViewProjectionOutcome.PROJECTED
                or rebuilt.producer_task_view is None
                or rebuilt.storage_authorization is None
            ):
                raise ProducerTaskViewPolicyError("authoritative inputs no longer project producer contracts")
            if not _same_contract_except_audit_actor_time(
                rebuilt.producer_task_view,
                producer_task_view,
            ):
                raise ProducerTaskViewPolicyError(
                    "current ProducerTaskView does not match authoritative inputs"
                )
            if not _same_contract_except_audit_actor_time(
                rebuilt.storage_authorization,
                storage_authorization,
            ):
                raise ProducerTaskViewPolicyError(
                    "current storage authorization does not match authoritative inputs"
                )
        except ProducerTaskViewPolicyError as exc:
            raise ProducerTaskViewPolicyError(f"current producer contract validation failed: {exc}") from exc


class ProducerStorageAccessGate:
    policy_version = PRODUCER_TASK_VIEW_POLICY_VERSION

    def authorize(
        self,
        *,
        authorization: ProducerStorageAuthorizationV2,
        request: ProducerStorageAccessRequest,
        audit: ContractAudit,
    ) -> ProducerStorageAccessResult:
        validate_storage_authorization_identity(authorization)
        _validate_access_request(request)
        authorization_ref = producer_storage_authorization_ref(authorization)
        if request.authorization_ref != authorization_ref:
            raise ProducerTaskViewPolicyError(
                "storage access request authorization ref is stale or mismatched"
            )
        reasons: set[ProducerStorageAccessReason] = set()
        if request.producer_principal_id != authorization.producer_principal_id:
            reasons.add(ProducerStorageAccessReason.PRINCIPAL_MISMATCH)
        if request.purpose != authorization.purpose:
            reasons.add(ProducerStorageAccessReason.PURPOSE_MISMATCH)
        if reasons:
            return _access_result(
                authorization=authorization,
                request=request,
                outcome=ProducerStorageAccessOutcome.DENIED_IDENTITY,
                authorized_subject_ref=None,
                reasons=frozenset(reasons),
                audit=audit,
            )
        if (
            _unsafe_storage_ref(request.requested_subject_ref)
            or request.requested_subject_ref not in authorization.authorized_subject_refs
        ):
            return _access_result(
                authorization=authorization,
                request=request,
                outcome=ProducerStorageAccessOutcome.DENIED_SCOPE,
                authorized_subject_ref=None,
                reasons=frozenset({ProducerStorageAccessReason.SUBJECT_NOT_AUTHORIZED}),
                audit=audit,
            )
        return _access_result(
            authorization=authorization,
            request=request,
            outcome=ProducerStorageAccessOutcome.GRANTED,
            authorized_subject_ref=request.requested_subject_ref,
            reasons=frozenset(),
            audit=audit,
        )


def validate_task_draft_identity(task_draft: TaskDraftV2) -> None:
    digest = task_draft_carried_sha256(task_draft)
    if task_draft.task_draft_sha256 != digest or task_draft.task_draft_id != f"task-draft://sha256/{digest}":
        raise ProducerTaskViewPolicyError("TaskDraft identity is stale or mismatched")


def validate_rubric_set_identity(rubric_set: RubricSetV2) -> None:
    for criterion in rubric_set.criteria:
        reachability_digest = rubric_reachability_carried_sha256(criterion.reachability)
        if (
            criterion.reachability.reachability_sha256 != reachability_digest
            or criterion.reachability.reachability_id != f"rubric-reachability://sha256/{reachability_digest}"
        ):
            raise ProducerTaskViewPolicyError("RubricSet reachability identity is stale or mismatched")
        criterion_digest = rubric_criterion_carried_sha256(criterion)
        if (
            criterion.criterion_sha256 != criterion_digest
            or criterion.criterion_id != f"rubric-criterion://sha256/{criterion_digest}"
        ):
            raise ProducerTaskViewPolicyError("RubricSet criterion identity is stale or mismatched")
    digest = rubric_set_carried_sha256(rubric_set)
    if rubric_set.rubric_set_sha256 != digest or rubric_set.rubric_set_id != f"rubric-set://sha256/{digest}":
        raise ProducerTaskViewPolicyError("RubricSet identity is stale or mismatched")


def validate_evaluator_spec_identity(evaluator_spec: EvaluatorSpecV2) -> None:
    for binding in evaluator_spec.bindings:
        if binding.binding_sha256 != evaluator_binding_carried_sha256(binding):
            raise ProducerTaskViewPolicyError("EvaluatorSpec binding identity is stale or mismatched")
    for rule in evaluator_spec.failure_rules:
        if rule.rule_sha256 != evaluator_failure_rule_carried_sha256(rule):
            raise ProducerTaskViewPolicyError("EvaluatorSpec failure rule identity is stale or mismatched")
    digest = evaluator_spec_carried_sha256(evaluator_spec)
    if (
        evaluator_spec.evaluator_spec_sha256 != digest
        or evaluator_spec.evaluator_spec_id != f"evaluator-spec://sha256/{digest}"
    ):
        raise ProducerTaskViewPolicyError("EvaluatorSpec identity is stale or mismatched")


def validate_reference_policy_identity(
    reference_policy: ReferencePolicyV2,
    evaluator_spec: EvaluatorSpecV2,
) -> None:
    digest = reference_policy_carried_sha256(reference_policy)
    if (
        reference_policy.reference_policy_sha256 != digest
        or reference_policy.reference_policy_id != f"reference-policy://sha256/{digest}"
    ):
        raise ProducerTaskViewPolicyError("ReferencePolicy identity is stale or mismatched")
    if reference_policy.evaluator_spec_ref != evaluator_spec_ref(evaluator_spec):
        raise ProducerTaskViewPolicyError("ReferencePolicy EvaluatorSpec ref is stale or mismatched")
    binding_refs = {_ref_key(ref) for binding in evaluator_spec.bindings for ref in binding.reference_refs}
    policy_refs = {_ref_key(ref) for ref in reference_policy.reference_refs}
    if binding_refs != policy_refs:
        raise ProducerTaskViewPolicyError("ReferencePolicy refs do not match EvaluatorSpec bindings")
    if reference_policy.contestant_access or reference_policy.attachment_producer_access:
        raise ProducerTaskViewPolicyError(
            "ReferencePolicy cannot grant contestant or attachment producer access"
        )


def validate_tool_policy_identity(tool_policy: ToolPolicyV2) -> None:
    for rule in tool_policy.rules:
        if rule.rule_id != rule.tool_id or rule.rule_sha256 != tool_rule_carried_sha256(rule):
            raise ProducerTaskViewPolicyError("ToolPolicy rule identity is stale or mismatched")
    digest = tool_policy_carried_sha256(tool_policy)
    if (
        tool_policy.tool_policy_sha256 != digest
        or tool_policy.tool_policy_id != f"tool-policy://sha256/{digest}"
    ):
        raise ProducerTaskViewPolicyError("ToolPolicy identity is stale or mismatched")


def validate_contestant_tool_policy_identity(
    policy: ContestantToolPolicyV2,
) -> None:
    for rule in policy.rules:
        if rule.rule_id != rule.tool_id or rule.rule_sha256 != contestant_tool_rule_carried_sha256(rule):
            raise ProducerTaskViewPolicyError("contestant tool rule identity is stale or mismatched")
    digest = contestant_tool_policy_carried_sha256(policy)
    if (
        policy.contestant_tool_policy_sha256 != digest
        or policy.contestant_tool_policy_id != f"contestant-tool-policy://sha256/{digest}"
    ):
        raise ProducerTaskViewPolicyError("contestant tool policy identity is stale or mismatched")


def validate_r4_contract_chain(
    *,
    task_draft: TaskDraftV2,
    rubric_set: RubricSetV2,
    evaluator_spec: EvaluatorSpecV2,
    reference_policy: ReferencePolicyV2,
    tool_policy: ToolPolicyV2,
    contestant_tool_policy: ContestantToolPolicyV2,
) -> None:
    draft_ref = task_draft_ref(task_draft)
    rubric_ref = rubric_set_ref(rubric_set)
    evaluator_ref = evaluator_spec_ref(evaluator_spec)
    if rubric_set.task_draft_ref != draft_ref:
        raise ProducerTaskViewPolicyError("RubricSet TaskDraft ref is stale or mismatched")
    if evaluator_spec.rubric_set_ref != rubric_ref:
        raise ProducerTaskViewPolicyError("EvaluatorSpec RubricSet ref is stale or mismatched")
    if reference_policy.evaluator_spec_ref != evaluator_ref:
        raise ProducerTaskViewPolicyError("ReferencePolicy EvaluatorSpec ref is stale or mismatched")
    if (
        tool_policy.task_draft_ref != draft_ref
        or tool_policy.rubric_set_ref != rubric_ref
        or tool_policy.evaluator_spec_ref != evaluator_ref
    ):
        raise ProducerTaskViewPolicyError("ToolPolicy R4 contract chain is stale or mismatched")
    if tool_policy.contestant_projection_ref != contestant_tool_policy_ref(contestant_tool_policy):
        raise ProducerTaskViewPolicyError("ToolPolicy contestant projection ref is stale or mismatched")
    if contestant_tool_policy.source_task_draft_sha256 != task_draft.task_draft_sha256:
        raise ProducerTaskViewPolicyError("contestant tool policy TaskDraft hash is stale or mismatched")
    contestant_tools = tuple(sorted(rule.tool_id for rule in contestant_tool_policy.rules))
    internal_allowed_tools = tuple(
        sorted(rule.tool_id for rule in tool_policy.rules if rule.action is ToolRuleActionV2.ALLOW)
    )
    if contestant_tools != internal_allowed_tools or set(contestant_tools) != set(task_draft.allowed_tools):
        raise ProducerTaskViewPolicyError(
            "producer-visible tools do not match exact TaskDraft and ToolPolicy"
        )


def validate_current_r4_contracts(
    *,
    task_draft: TaskDraftV2,
    rubric_set: RubricSetV2,
    evaluator_spec: EvaluatorSpecV2,
    reference_policy: ReferencePolicyV2,
    tool_policy: ToolPolicyV2,
    contestant_tool_policy: ContestantToolPolicyV2,
    storage_authorization: ProducerStorageAuthorizationV2,
    producer_task_view: ProducerTaskViewV2,
) -> str:
    validate_task_draft_identity(task_draft)
    if task_draft.prompt_safety_status is not TaskDraftPromptSafetyStatusV2.PASSED:
        raise ProducerTaskViewPolicyError("TaskDraft must have PASSED prompt safety")
    validate_rubric_set_identity(rubric_set)
    validate_evaluator_spec_identity(evaluator_spec)
    validate_reference_policy_identity(reference_policy, evaluator_spec)
    validate_tool_policy_identity(tool_policy)
    validate_contestant_tool_policy_identity(contestant_tool_policy)
    validate_r4_contract_chain(
        task_draft=task_draft,
        rubric_set=rubric_set,
        evaluator_spec=evaluator_spec,
        reference_policy=reference_policy,
        tool_policy=tool_policy,
        contestant_tool_policy=contestant_tool_policy,
    )
    validate_storage_authorization_identity(storage_authorization)
    validate_producer_task_view_identity(producer_task_view)
    chain_digest = r4_contract_chain_sha256(
        draft_ref=task_draft_ref(task_draft),
        rubric_ref=rubric_set_ref(rubric_set),
        evaluator_ref=evaluator_spec_ref(evaluator_spec),
        reference_ref=reference_policy_ref(reference_policy),
        tool_ref=tool_policy_ref(tool_policy),
        contestant_ref=contestant_tool_policy_ref(contestant_tool_policy),
    )
    if storage_authorization.source_task_draft_sha256 != task_draft.task_draft_sha256:
        raise ProducerTaskViewPolicyError("storage authorization TaskDraft hash is stale or mismatched")
    if producer_task_view.storage_authorization_ref != (
        producer_storage_authorization_ref(storage_authorization)
    ):
        raise ProducerTaskViewPolicyError("ProducerTaskView storage authorization ref is stale or mismatched")
    if producer_task_view.source_task_draft_sha256 != (task_draft.task_draft_sha256):
        raise ProducerTaskViewPolicyError("ProducerTaskView TaskDraft hash is stale or mismatched")
    if producer_task_view.source_contestant_tool_policy_sha256 != (
        contestant_tool_policy.contestant_tool_policy_sha256
    ):
        raise ProducerTaskViewPolicyError("ProducerTaskView contestant policy hash is stale or mismatched")
    if producer_task_view.source_contract_chain_sha256 != chain_digest:
        raise ProducerTaskViewPolicyError("ProducerTaskView contract chain hash is stale or mismatched")
    if producer_task_view.allowed_tools != tuple(rule.tool_id for rule in contestant_tool_policy.rules):
        raise ProducerTaskViewPolicyError("ProducerTaskView tools are stale or mismatched")
    return chain_digest


def _validate_producer_evidence(
    view_result: EvidenceViewResult,
    bundle: EvidenceBundle,
) -> None:
    try:
        validate_evidence_view_result_identity(view_result)
        validate_projection_policy(view_result.projection_policy)
        validate_evidence_bundle_identity(bundle)
    except (EvidenceViewPolicyError, EvidenceCompilationPolicyError) as exc:
        raise ProducerTaskViewPolicyError("producer evidence identity is stale or invalid") from exc
    if (
        view_result.principal_type is not EvidenceViewPrincipalType.ATTACHMENT_PRODUCER
        or view_result.purpose is not EvidenceViewPurpose.ATTACHMENT_PRODUCTION
    ):
        raise ProducerTaskViewPolicyError("producer evidence view must use attachment producer purpose")
    if (
        view_result.projection_policy.principal_type != EvidenceViewPrincipalType.ATTACHMENT_PRODUCER.value
        or view_result.projection_policy.purpose != EvidenceViewPurpose.ATTACHMENT_PRODUCTION.value
    ):
        raise ProducerTaskViewPolicyError("producer projection policy must use attachment producer purpose")
    if bundle.consumer_stage != "attachment-producer" or bundle.purpose != "attachment-production":
        raise ProducerTaskViewPolicyError("producer evidence bundle stage or purpose is invalid")
    expected_policy_ref = projection_policy_ref(view_result.projection_policy)
    if bundle.projection_policy_ref != expected_policy_ref:
        raise ProducerTaskViewPolicyError("producer evidence bundle projection policy is stale or mismatched")
    included_projected_refs = {_ref_key(item.projected_ref) for item in view_result.included_items}
    if any(_ref_key(evidence.subject_ref) not in included_projected_refs for evidence in bundle.evidence):
        raise ProducerTaskViewPolicyError("producer evidence bundle contains an unauthorized subject")


def _compile_storage_authorization(
    *,
    task_draft: TaskDraftV2,
    producer_view_result: EvidenceViewResult,
    producer_evidence_bundle: EvidenceBundle,
    audit: ContractAudit,
) -> ProducerStorageAuthorizationV2:
    policy_ref = projection_policy_ref(producer_view_result.projection_policy)
    bundle_ref = evidence_bundle_ref(producer_evidence_bundle)
    subject_refs = tuple(
        sorted(
            (item.subject_ref for item in producer_evidence_bundle.evidence),
            key=_ref_key,
        )
    )
    authorization = ProducerStorageAuthorizationV2(
        authorization_id="producer-storage-authorization://pending",
        producer_principal_id=producer_view_result.principal_id,
        purpose="ATTACHMENT_PRODUCTION",
        source_task_draft_sha256=task_draft.task_draft_sha256,
        projection_policy_ref=policy_ref,
        evidence_bundle_refs=(bundle_ref,),
        authorized_subject_refs=subject_refs,
        raw_store_access=False,
        canonical_store_access=False,
        quarantine_store_access=False,
        private_reference_store_access=False,
        credentials_issued=False,
        policy_version=PRODUCER_TASK_VIEW_POLICY_VERSION,
        authorization_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (policy_ref, bundle_ref, *subject_refs),
        ),
    )
    digest = producer_storage_authorization_carried_sha256(authorization)
    return authorization.model_copy(
        update={
            "authorization_id": (f"producer-storage-authorization://sha256/{digest}"),
            "authorization_sha256": digest,
        }
    )


def _producer_projection_payload(
    *,
    task_draft: TaskDraftV2,
    attachment_requirements: tuple[ProducerAttachmentRequirementV2, ...],
    allowed_tools: tuple[str, ...],
    bundle_ref: ObjectRef,
    projection_policy_ref: ObjectRef,
    storage_authorization_ref: ObjectRef,
    contestant_tool_policy: ContestantToolPolicyV2,
    contract_chain_sha256: str,
) -> dict[str, object]:
    return {
        "query_instruction": task_draft.visible_prompt,
        "attachment_requirements": [
            item.model_dump(mode="json", exclude_none=False) for item in attachment_requirements
        ],
        "allowed_tools": list(allowed_tools),
        "safe_evidence_bundle_refs": [_ref_payload(bundle_ref)],
        "forbidden_outputs": list(task_draft.forbidden_outputs),
        "projection_policy_ref": _ref_payload(projection_policy_ref),
        "storage_authorization_ref": _ref_payload(storage_authorization_ref),
        "source_task_draft_sha256": task_draft.task_draft_sha256,
        "source_contestant_tool_policy_sha256": (contestant_tool_policy.contestant_tool_policy_sha256),
        "source_contract_chain_sha256": contract_chain_sha256,
        "policy_version": PRODUCER_TASK_VIEW_POLICY_VERSION,
    }


def _producer_projection_ref(payload: dict[str, object]) -> ObjectRef:
    digest = producer_payload_sha256(payload)
    return ObjectRef(
        object_type="producer-task-view-input",
        object_id=f"producer-task-view-input://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _enforce_producer_boundary(
    *,
    task_draft: TaskDraftV2,
    contestant_tool_policy: ContestantToolPolicyV2,
    producer_projection_ref: ObjectRef,
    bundle_refs: tuple[ObjectRef, ...],
    audit: ContractAudit,
) -> PromptBoundaryEnforcementResult:
    assert task_draft.prompt_safety_gate_ref is not None
    projection_segment = PromptBoundarySegment(
        segment_id=(f"prompt-boundary-segment://producer/projection/{producer_projection_ref.object_sha256}"),
        surface=PromptBoundarySurface.PRODUCER_TASK_DATA,
        source_role=(PromptBoundarySourceRole.PROMPT_SAFETY_PASSED_TASK_CONTRACT),
        source_ref=producer_projection_ref,
        content_sha256=producer_projection_ref.object_sha256,
        untrusted_data_marker=True,
        text_preview=None,
    )
    contestant_ref = contestant_tool_policy_ref(contestant_tool_policy)
    tool_segment = PromptBoundarySegment(
        segment_id=(f"prompt-boundary-segment://producer/tool-policy/{contestant_ref.object_sha256}"),
        surface=PromptBoundarySurface.TOOL_POLICY_CONFIG,
        source_role=PromptBoundarySourceRole.STATIC_POLICY_CONFIG,
        source_ref=contestant_ref,
        content_sha256=contestant_ref.object_sha256,
        untrusted_data_marker=False,
        text_preview=None,
    )
    bundle_segments = tuple(
        PromptBoundarySegment(
            segment_id=(f"prompt-boundary-segment://producer/bundle/{bundle_ref.object_sha256}"),
            surface=PromptBoundarySurface.EVIDENCE_DATA,
            source_role=PromptBoundarySourceRole.EVIDENCE_BUNDLE_REF,
            source_ref=bundle_ref,
            evidence_bundle_ref=bundle_ref,
            untrusted_data_marker=True,
            text_preview=None,
        )
        for bundle_ref in bundle_refs
    )
    boundary_request = PromptBoundaryEnforcementRequest(
        boundary_id=(f"prompt-boundary://producer/{producer_projection_ref.object_sha256}"),
        segments=(projection_segment, tool_segment, *bundle_segments),
        approved_evidence_bundle_refs=bundle_refs,
        audit=_safe_audit(
            audit,
            (
                task_draft_ref(task_draft),
                task_draft.prompt_safety_gate_ref,
                producer_projection_ref,
                contestant_ref,
                *bundle_refs,
            ),
        ),
    )
    try:
        return PromptInjectionBoundaryEnforcer().validate_producer_task_data_boundary(
            ProducerTaskDataBoundaryRequest(
                task_draft_ref=task_draft_ref(task_draft),
                task_prompt_safety_gate_ref=(task_draft.prompt_safety_gate_ref),
                producer_projection_ref=producer_projection_ref,
                contestant_tool_policy_ref=contestant_ref,
                safe_evidence_bundle_refs=bundle_refs,
                boundary_request=boundary_request,
                projection_segment_id=projection_segment.segment_id,
                tool_control_segment_id=tool_segment.segment_id,
                evidence_bundle_segment_ids=tuple(segment.segment_id for segment in bundle_segments),
            )
        )
    except PromptInjectionBoundaryPolicyError as exc:
        raise ProducerTaskViewPolicyError("producer task prompt boundary is invalid") from exc


def r4_contract_chain_sha256(
    *,
    draft_ref: ObjectRef,
    rubric_ref: ObjectRef,
    evaluator_ref: ObjectRef,
    reference_ref: ObjectRef,
    tool_ref: ObjectRef,
    contestant_ref: ObjectRef,
) -> str:
    return producer_payload_sha256(
        {
            "task_draft_ref": _ref_payload(draft_ref),
            "rubric_set_ref": _ref_payload(rubric_ref),
            "evaluator_spec_ref": _ref_payload(evaluator_ref),
            "reference_policy_ref": _ref_payload(reference_ref),
            "tool_policy_ref": _ref_payload(tool_ref),
            "contestant_tool_policy_ref": _ref_payload(contestant_ref),
        }
    )


def _projection_result(
    *,
    task_draft: TaskDraftV2,
    outcome: ProducerTaskViewProjectionOutcome,
    producer_task_view: ProducerTaskViewV2 | None,
    storage_authorization: ProducerStorageAuthorizationV2 | None,
    unresolved_reasons: frozenset[ProducerTaskViewProjectionReason],
    audit: ContractAudit,
    internal_refs: tuple[ObjectRef, ...],
) -> ProducerTaskViewProjectionResult:
    view_ref = producer_task_view_ref(producer_task_view) if producer_task_view is not None else None
    authorization_ref = (
        producer_storage_authorization_ref(storage_authorization)
        if storage_authorization is not None
        else None
    )
    payload = {
        "source_task_draft_sha256": task_draft.task_draft_sha256,
        "outcome": outcome.value,
        "producer_task_view_ref": _maybe_ref_payload(view_ref),
        "storage_authorization_ref": _maybe_ref_payload(authorization_ref),
        "unresolved_reasons": sorted(reason.value for reason in unresolved_reasons),
        "policy_version": PRODUCER_TASK_VIEW_POLICY_VERSION,
    }
    digest = producer_payload_sha256(payload)
    return ProducerTaskViewProjectionResult(
        result_id=f"producer-task-view-result://sha256/{digest}",
        source_task_draft_sha256=task_draft.task_draft_sha256,
        outcome=outcome,
        producer_task_view=producer_task_view,
        storage_authorization=storage_authorization,
        unresolved_reasons=unresolved_reasons,
        policy_version=PRODUCER_TASK_VIEW_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(audit, internal_refs),
    )


def validate_storage_authorization_identity(
    authorization: ProducerStorageAuthorizationV2,
) -> None:
    digest = producer_storage_authorization_carried_sha256(authorization)
    if (
        authorization.authorization_sha256 != digest
        or authorization.authorization_id != f"producer-storage-authorization://sha256/{digest}"
    ):
        raise ProducerTaskViewPolicyError("storage authorization identity is stale or mismatched")


def validate_producer_task_view_identity(view: ProducerTaskViewV2) -> None:
    digest = producer_task_view_carried_sha256(view)
    if (
        view.producer_task_view_sha256 != digest
        or view.producer_task_view_id != f"producer-task-view://sha256/{digest}"
    ):
        raise ProducerTaskViewPolicyError("ProducerTaskView identity is stale or mismatched")


def _validate_access_request(request: ProducerStorageAccessRequest) -> None:
    digest = producer_storage_access_request_carried_sha256(request)
    if (
        request.request_sha256 != digest
        or request.request_id != f"producer-storage-access-request://sha256/{digest}"
    ):
        raise ProducerTaskViewPolicyError("storage access request identity is stale or mismatched")


def _access_result(
    *,
    authorization: ProducerStorageAuthorizationV2,
    request: ProducerStorageAccessRequest,
    outcome: ProducerStorageAccessOutcome,
    authorized_subject_ref: ObjectRef | None,
    reasons: frozenset[ProducerStorageAccessReason],
    audit: ContractAudit,
) -> ProducerStorageAccessResult:
    request_ref = producer_storage_access_request_ref(request)
    authorization_ref = producer_storage_authorization_ref(authorization)
    payload = {
        "request_ref": _ref_payload(request_ref),
        "authorization_ref": _ref_payload(authorization_ref),
        "outcome": outcome.value,
        "authorized_subject_ref": (
            _ref_payload(authorized_subject_ref) if authorized_subject_ref is not None else None
        ),
        "reasons": sorted(reason.value for reason in reasons),
        "policy_version": PRODUCER_TASK_VIEW_POLICY_VERSION,
    }
    digest = producer_payload_sha256(payload)
    audit_refs = (
        (request_ref, authorization_ref, authorized_subject_ref)
        if authorized_subject_ref is not None
        else (request_ref, authorization_ref)
    )
    return ProducerStorageAccessResult(
        result_id=f"producer-storage-access-result://sha256/{digest}",
        request_ref=request_ref,
        authorization_ref=authorization_ref,
        outcome=outcome,
        authorized_subject_ref=authorized_subject_ref,
        reasons=reasons,
        policy_version=PRODUCER_TASK_VIEW_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(audit, audit_refs),
    )


def _unsafe_storage_ref(ref: ObjectRef) -> bool:
    normalized = f"{ref.object_type}:{ref.object_id}".casefold().replace(
        "_",
        "-",
    )
    return any(marker in normalized for marker in _DENIED_STORAGE_REF_MARKERS)


def _prompt_boundary_ref(
    result: PromptBoundaryEnforcementResult,
) -> ObjectRef:
    return ObjectRef(
        object_type="prompt-boundary-enforcement",
        object_id=result.enforcement_id,
        object_version=result.policy_version,
        object_sha256=result.enforcement_sha256,
    )


def _same_contract_except_audit_actor_time(
    expected: ProducerTaskViewV2 | ProducerStorageAuthorizationV2,
    observed: ProducerTaskViewV2 | ProducerStorageAuthorizationV2,
) -> bool:
    if expected.model_dump(mode="json", exclude={"audit"}) != observed.model_dump(
        mode="json",
        exclude={"audit"},
    ):
        return False
    return (
        expected.audit.governing_versions == observed.audit.governing_versions
        and expected.audit.input_refs == observed.audit.input_refs
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


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _maybe_ref_payload(ref: ObjectRef | None) -> dict[str, object] | None:
    if ref is None:
        return None
    return _ref_payload(ref)
