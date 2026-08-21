from __future__ import annotations

from datetime import datetime

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.contracts.task import EvaluationFailureClass, ReferenceMode
from eval_factory.contracts.task_v2 import (
    EvaluatorBindingV2,
    EvaluatorExecutionModeV2,
    EvaluatorFailureRuleV2,
    EvaluatorFailureSignalV2,
    EvaluatorModelDomainV2,
    EvaluatorReferenceDataClassV2,
    EvaluatorReferenceGrantV2,
    EvaluatorSpecV2,
    ReferencePolicyV2,
    RubricSetV2,
    evaluator_binding_carried_sha256,
    evaluator_failure_rule_carried_sha256,
    evaluator_reference_grant_carried_sha256,
    evaluator_reference_grant_ref,
    evaluator_spec_carried_sha256,
    evaluator_spec_ref,
    reference_policy_carried_sha256,
    reference_policy_ref,
    rubric_criterion_carried_sha256,
    rubric_reachability_carried_sha256,
    rubric_set_carried_sha256,
    rubric_set_ref,
)
from eval_factory.task_authoring.evaluation_models import (
    EVALUATION_CONTRACT_POLICY_VERSION,
    REFERENCE_ACCESS_POLICY_VERSION,
    EvaluationContractCompileResult,
    EvaluationContractOutcome,
    EvaluationContractPolicyError,
    EvaluationContractReason,
    EvaluatorAccessPrincipalType,
    EvaluatorBindingDefinition,
    EvaluatorFailureClassification,
    EvaluatorFailureObservation,
    EvaluatorReferenceAccessOutcome,
    EvaluatorReferenceAccessReason,
    EvaluatorReferenceAccessRequest,
    EvaluatorReferenceAccessResult,
    VerifiedModelDomainAuthorization,
    evaluation_payload_sha256,
    evaluator_failure_observation_sha256,
    evaluator_reference_access_request_sha256,
    verified_model_domain_authorization_sha256,
)

_FAILURE_CLASS_BY_SIGNAL = {
    EvaluatorFailureSignalV2.CONTESTANT_OUTPUT_MISSING: (EvaluationFailureClass.CONTESTANT_FAILURE),
    EvaluatorFailureSignalV2.CONTESTANT_OUTPUT_CONTRACT_VIOLATION: (
        EvaluationFailureClass.CONTESTANT_FAILURE
    ),
    EvaluatorFailureSignalV2.EVALUATOR_TIMEOUT: (EvaluationFailureClass.EVALUATOR_FAILURE),
    EvaluatorFailureSignalV2.EVALUATOR_RUNTIME_FAILURE: (EvaluationFailureClass.EVALUATOR_FAILURE),
    EvaluatorFailureSignalV2.EVALUATOR_OUTPUT_CONTRACT_VIOLATION: (EvaluationFailureClass.EVALUATOR_FAILURE),
    EvaluatorFailureSignalV2.ENVIRONMENT_STARTUP_FAILURE: (EvaluationFailureClass.ENVIRONMENT_FAILURE),
    EvaluatorFailureSignalV2.ENVIRONMENT_RUNTIME_FAILURE: (EvaluationFailureClass.ENVIRONMENT_FAILURE),
    EvaluatorFailureSignalV2.REQUIRED_ENVIRONMENT_CAPABILITY_MISSING: (
        EvaluationFailureClass.ENVIRONMENT_FAILURE
    ),
    EvaluatorFailureSignalV2.INSUFFICIENT_OBSERVATION: (EvaluationFailureClass.INDETERMINATE),
    EvaluatorFailureSignalV2.CONFLICTING_FAILURE_SIGNALS: (EvaluationFailureClass.INDETERMINATE),
}

_REFERENCE_SHAPES = {
    ReferenceMode.NONE: (
        EvaluatorReferenceDataClassV2.NONE,
        None,
        False,
        False,
    ),
    ReferenceMode.STRUCTURED_EXPECTATIONS: (
        EvaluatorReferenceDataClassV2.RESTRICTED_EVAL_CONTROL,
        "structured-expectation",
        True,
        False,
    ),
    ReferenceMode.PRIVATE_ANSWER: (
        EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
        "private-reference",
        True,
        False,
    ),
    ReferenceMode.TRACE_BEHAVIOR: (
        EvaluatorReferenceDataClassV2.RESTRICTED_TRACE_SPAN,
        "trace-behavior-reference",
        True,
        False,
    ),
    ReferenceMode.HUMAN_ONLY: (
        EvaluatorReferenceDataClassV2.HUMAN_ONLY_REFERENCE,
        "human-only-reference",
        False,
        True,
    ),
}


class EvaluationContractCompiler:
    policy_version = EVALUATION_CONTRACT_POLICY_VERSION

    def compile(
        self,
        *,
        rubric_set: RubricSetV2,
        binding_definitions: tuple[EvaluatorBindingDefinition, ...],
        reference_mode: ReferenceMode,
        audit: ContractAudit,
    ) -> EvaluationContractCompileResult:
        _validate_rubric_set(rubric_set)
        definitions = _definition_map(binding_definitions)
        criterion_ids_by_binding = _criterion_ids_by_binding(rubric_set)
        required_binding_ids = set(criterion_ids_by_binding)
        definition_ids = set(definitions)
        extra = definition_ids - required_binding_ids
        if extra:
            raise EvaluationContractPolicyError(f"extra evaluator binding definitions: {sorted(extra)}")
        missing = required_binding_ids - definition_ids
        if missing:
            return _compile_result(
                rubric_set=rubric_set,
                outcome=EvaluationContractOutcome.BLOCKED_BINDING,
                evaluator_spec=None,
                reference_policy=None,
                unresolved_reasons=frozenset({EvaluationContractReason.MISSING_EVALUATOR_BINDING}),
                audit=audit,
            )

        mode_error = _validate_definition_mode(
            reference_mode=reference_mode,
            definitions=tuple(definitions.values()),
        )
        if mode_error is not None:
            return _compile_result(
                rubric_set=rubric_set,
                outcome=EvaluationContractOutcome.BLOCKED_POLICY,
                evaluator_spec=None,
                reference_policy=None,
                unresolved_reasons=frozenset({mode_error}),
                audit=audit,
            )

        bindings = tuple(
            _compile_binding(
                definition=definitions[binding_id],
                criterion_ids=criterion_ids_by_binding[binding_id],
            )
            for binding_id in sorted(required_binding_ids)
        )
        rules = _failure_rules()
        evaluator_spec = _compile_evaluator_spec(
            rubric_set=rubric_set,
            bindings=bindings,
            rules=rules,
            audit=audit,
        )
        reference_policy = _compile_reference_policy(
            evaluator_spec=evaluator_spec,
            bindings=bindings,
            mode=reference_mode,
            audit=audit,
        )
        return _compile_result(
            rubric_set=rubric_set,
            outcome=EvaluationContractOutcome.COMPILED,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            unresolved_reasons=frozenset(),
            audit=audit,
        )


class EvaluatorFailureClassifier:
    policy_version = EVALUATION_CONTRACT_POLICY_VERSION

    def classify(
        self,
        *,
        evaluator_spec: EvaluatorSpecV2,
        observation: EvaluatorFailureObservation,
        audit: ContractAudit,
    ) -> EvaluatorFailureClassification:
        _validate_evaluator_spec(evaluator_spec)
        _validate_observation(observation)
        rule_by_signal = {rule.signal: rule for rule in evaluator_spec.failure_rules}
        matched_rules = tuple(
            sorted(
                (rule_by_signal[signal] for signal in observation.signals),
                key=lambda item: item.rule_id,
            )
        )
        mapped_classes = frozenset(item.failure_class for item in matched_rules)
        if (
            not matched_rules
            or EvaluationFailureClass.INDETERMINATE in mapped_classes
            or len(mapped_classes) != 1
        ):
            failure_class = EvaluationFailureClass.INDETERMINATE
        else:
            failure_class = next(iter(mapped_classes))
        spec_ref = evaluator_spec_ref(evaluator_spec)
        observation_ref = _observation_ref(observation)
        seed = {
            "evaluator_spec_ref": _ref_payload(spec_ref),
            "observation_ref": _ref_payload(observation_ref),
            "failure_class": failure_class.value,
            "matched_rule_ids": [item.rule_id for item in matched_rules],
            "policy_version": self.policy_version,
        }
        digest = evaluation_payload_sha256(seed)
        return EvaluatorFailureClassification(
            classification_id=(f"evaluator-failure-classification://sha256/{digest}"),
            evaluator_spec_ref=spec_ref,
            observation_ref=observation_ref,
            failure_class=failure_class,
            matched_rule_ids=tuple(item.rule_id for item in matched_rules),
            policy_version=self.policy_version,
            classification_sha256=digest,
            audit=_safe_audit(audit, (spec_ref, observation_ref)),
        )


class EvaluatorReferenceAccessGate:
    policy_version = REFERENCE_ACCESS_POLICY_VERSION

    def authorize(
        self,
        *,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        request: EvaluatorReferenceAccessRequest,
        model_authorization: VerifiedModelDomainAuthorization | None,
        evaluated_at: datetime,
        audit: ContractAudit,
    ) -> EvaluatorReferenceAccessResult:
        _validate_evaluator_spec(evaluator_spec)
        _validate_reference_policy(reference_policy, evaluator_spec)
        _validate_access_request(request)
        if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
            raise EvaluationContractPolicyError("reference authorization time must be timezone-aware")

        if request.principal_type is not EvaluatorAccessPrincipalType.EVALUATOR:
            return _access_result(
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                request=request,
                outcome=EvaluatorReferenceAccessOutcome.DENIED_IDENTITY,
                grant=None,
                reasons=frozenset({EvaluatorReferenceAccessReason.PRINCIPAL_TYPE_DENIED}),
                audit=audit,
            )

        binding = next(
            (
                item
                for item in evaluator_spec.bindings
                if item.evaluator_binding_id == request.evaluator_binding_id
            ),
            None,
        )
        if binding is None:
            return _access_result(
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                request=request,
                outcome=EvaluatorReferenceAccessOutcome.DENIED_IDENTITY,
                grant=None,
                reasons=frozenset({EvaluatorReferenceAccessReason.EVALUATOR_BINDING_NOT_FOUND}),
                audit=audit,
            )
        if (
            reference_policy.mode is ReferenceMode.HUMAN_ONLY
            or binding.execution_mode is EvaluatorExecutionModeV2.HUMAN_ONLY
        ):
            return _access_result(
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                request=request,
                outcome=EvaluatorReferenceAccessOutcome.HUMAN_ONLY,
                grant=None,
                reasons=frozenset(),
                audit=audit,
            )
        if reference_policy.mode is ReferenceMode.NONE:
            return _access_result(
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                request=request,
                outcome=EvaluatorReferenceAccessOutcome.NO_REFERENCE_REQUIRED,
                grant=None,
                reasons=frozenset(),
                audit=audit,
            )
        if binding.evaluator_principal_id != request.principal_id:
            return _access_result(
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                request=request,
                outcome=EvaluatorReferenceAccessOutcome.DENIED_IDENTITY,
                grant=None,
                reasons=frozenset({EvaluatorReferenceAccessReason.EVALUATOR_PRINCIPAL_MISMATCH}),
                audit=audit,
            )

        approval_ref: ObjectRef | None = None
        if binding.execution_mode is EvaluatorExecutionModeV2.MODEL:
            reasons = _model_authorization_reasons(
                binding=binding,
                reference_policy=reference_policy,
                request=request,
                authorization=model_authorization,
                evaluated_at=evaluated_at,
            )
            if reasons:
                outcome = (
                    EvaluatorReferenceAccessOutcome.DENIED_SCOPE
                    if reasons.intersection(
                        {
                            EvaluatorReferenceAccessReason.MODEL_PROFILE_MISMATCH,
                            EvaluatorReferenceAccessReason.REFERENCE_SCOPE_DENIED,
                        }
                    )
                    else EvaluatorReferenceAccessOutcome.BLOCKED_POLICY
                )
                return _access_result(
                    evaluator_spec=evaluator_spec,
                    reference_policy=reference_policy,
                    request=request,
                    outcome=outcome,
                    grant=None,
                    reasons=frozenset(reasons),
                    audit=audit,
                )
            assert model_authorization is not None
            approval_ref = model_authorization.model_domain_approval_ref
        elif request.model_profile_ref is not None:
            return _access_result(
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                request=request,
                outcome=EvaluatorReferenceAccessOutcome.DENIED_SCOPE,
                grant=None,
                reasons=frozenset({EvaluatorReferenceAccessReason.MODEL_PROFILE_MISMATCH}),
                audit=audit,
            )

        grant = _compile_grant(
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            binding=binding,
            model_domain_approval_ref=approval_ref,
            audit=audit,
        )
        return _access_result(
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            request=request,
            outcome=EvaluatorReferenceAccessOutcome.GRANTED,
            grant=grant,
            reasons=frozenset(),
            audit=audit,
        )


def _validate_rubric_set(rubric_set: RubricSetV2) -> None:
    digest = rubric_set_carried_sha256(rubric_set)
    if rubric_set.rubric_set_sha256 != digest or rubric_set.rubric_set_id != (
        f"rubric-set://sha256/{digest}"
    ):
        raise EvaluationContractPolicyError("RubricSet identity is stale or mismatched")
    for criterion in rubric_set.criteria:
        reachability_digest = rubric_reachability_carried_sha256(criterion.reachability)
        if criterion.reachability.reachability_sha256 != reachability_digest:
            raise EvaluationContractPolicyError("RubricSet reachability identity is stale or mismatched")
        criterion_digest = rubric_criterion_carried_sha256(criterion)
        if (
            criterion.criterion_sha256 != criterion_digest
            or criterion.criterion_id != f"rubric-criterion://sha256/{criterion_digest}"
        ):
            raise EvaluationContractPolicyError("RubricSet criterion identity is stale or mismatched")


def _definition_map(
    definitions: tuple[EvaluatorBindingDefinition, ...],
) -> dict[Identifier, EvaluatorBindingDefinition]:
    result: dict[Identifier, EvaluatorBindingDefinition] = {}
    for definition in definitions:
        if definition.evaluator_binding_id in result:
            raise EvaluationContractPolicyError("duplicate evaluator binding definition")
        result[definition.evaluator_binding_id] = definition
    return result


def _criterion_ids_by_binding(
    rubric_set: RubricSetV2,
) -> dict[Identifier, tuple[Identifier, ...]]:
    collected: dict[Identifier, list[Identifier]] = {}
    for criterion in rubric_set.criteria:
        collected.setdefault(criterion.evaluator_binding, []).append(criterion.criterion_id)
    return {binding_id: tuple(sorted(criterion_ids)) for binding_id, criterion_ids in collected.items()}


def _validate_definition_mode(
    *,
    reference_mode: ReferenceMode,
    definitions: tuple[EvaluatorBindingDefinition, ...],
) -> EvaluationContractReason | None:
    reference_refs = tuple(ref for definition in definitions for ref in definition.reference_refs)
    if reference_mode is ReferenceMode.NONE:
        if reference_refs:
            raise EvaluationContractPolicyError("NONE mode cannot carry reference refs")
    elif not reference_refs:
        return EvaluationContractReason.REFERENCE_MODE_MISMATCH
    expected_ref_type = _REFERENCE_SHAPES[reference_mode][1]
    if expected_ref_type is not None and any(ref.object_type != expected_ref_type for ref in reference_refs):
        raise EvaluationContractPolicyError(f"{reference_mode.value} requires {expected_ref_type} refs")
    if reference_mode is ReferenceMode.HUMAN_ONLY:
        if any(
            definition.execution_mode is not EvaluatorExecutionModeV2.HUMAN_ONLY for definition in definitions
        ):
            return EvaluationContractReason.REFERENCE_MODE_MISMATCH
    elif any(definition.execution_mode is EvaluatorExecutionModeV2.HUMAN_ONLY for definition in definitions):
        return EvaluationContractReason.REFERENCE_MODE_MISMATCH
    return None


def _compile_binding(
    *,
    definition: EvaluatorBindingDefinition,
    criterion_ids: tuple[Identifier, ...],
) -> EvaluatorBindingV2:
    binding = EvaluatorBindingV2(
        evaluator_binding_id=definition.evaluator_binding_id,
        criterion_ids=criterion_ids,
        evaluator_type=definition.evaluator_type,
        execution_mode=definition.execution_mode,
        evaluator_version=definition.evaluator_version,
        input_contract_ref=definition.input_contract_ref,
        output_contract_ref=definition.output_contract_ref,
        evaluator_principal_id=definition.evaluator_principal_id,
        model_profile_ref=definition.model_profile_ref,
        reference_refs=tuple(
            sorted(
                set(definition.reference_refs),
                key=_ref_key,
            )
        ),
        timeout_seconds=definition.timeout_seconds,
        binding_sha256="0" * 64,
    )
    digest = evaluator_binding_carried_sha256(binding)
    return binding.model_copy(update={"binding_sha256": digest})


def _failure_rules() -> tuple[EvaluatorFailureRuleV2, ...]:
    rules = []
    for signal in sorted(EvaluatorFailureSignalV2, key=lambda item: item.value):
        rule = EvaluatorFailureRuleV2(
            rule_id=f"evaluator-failure-rule://{signal.value.casefold().replace('_', '-')}",
            signal=signal,
            failure_class=_FAILURE_CLASS_BY_SIGNAL[signal],
            rule_sha256="0" * 64,
        )
        digest = evaluator_failure_rule_carried_sha256(rule)
        rules.append(
            rule.model_copy(
                update={
                    "rule_id": f"evaluator-failure-rule://sha256/{digest}",
                    "rule_sha256": digest,
                }
            )
        )
    return tuple(rules)


def _compile_evaluator_spec(
    *,
    rubric_set: RubricSetV2,
    bindings: tuple[EvaluatorBindingV2, ...],
    rules: tuple[EvaluatorFailureRuleV2, ...],
    audit: ContractAudit,
) -> EvaluatorSpecV2:
    rubric_ref = rubric_set_ref(rubric_set)
    evaluator_spec = EvaluatorSpecV2(
        evaluator_spec_id="evaluator-spec://pending",
        evaluator_spec_version=1,
        supersedes_evaluator_spec_ref=None,
        rubric_set_ref=rubric_ref,
        bindings=bindings,
        failure_rules=rules,
        policy_version=EVALUATION_CONTRACT_POLICY_VERSION,
        evaluator_spec_sha256="0" * 64,
        audit=_safe_audit(audit, (rubric_ref,)),
    )
    digest = evaluator_spec_carried_sha256(evaluator_spec)
    return evaluator_spec.model_copy(
        update={
            "evaluator_spec_id": f"evaluator-spec://sha256/{digest}",
            "evaluator_spec_sha256": digest,
        }
    )


def _compile_reference_policy(
    *,
    evaluator_spec: EvaluatorSpecV2,
    bindings: tuple[EvaluatorBindingV2, ...],
    mode: ReferenceMode,
    audit: ContractAudit,
) -> ReferencePolicyV2:
    spec_ref = evaluator_spec_ref(evaluator_spec)
    reference_refs = tuple(
        sorted(
            {ref for binding in bindings for ref in binding.reference_refs},
            key=_ref_key,
        )
    )
    data_class, _, evaluator_access, human_only_access = _REFERENCE_SHAPES[mode]
    policy = ReferencePolicyV2(
        reference_policy_id="reference-policy://pending",
        reference_policy_version=1,
        supersedes_reference_policy_ref=None,
        evaluator_spec_ref=spec_ref,
        mode=mode,
        reference_data_class=data_class,
        reference_refs=reference_refs,
        contestant_access=False,
        attachment_producer_access=False,
        evaluator_access=evaluator_access,
        human_only_access=human_only_access,
        policy_version=REFERENCE_ACCESS_POLICY_VERSION,
        reference_policy_sha256="0" * 64,
        audit=_safe_audit(audit, (spec_ref,)),
    )
    digest = reference_policy_carried_sha256(policy)
    return policy.model_copy(
        update={
            "reference_policy_id": f"reference-policy://sha256/{digest}",
            "reference_policy_sha256": digest,
        }
    )


def _compile_result(
    *,
    rubric_set: RubricSetV2,
    outcome: EvaluationContractOutcome,
    evaluator_spec: EvaluatorSpecV2 | None,
    reference_policy: ReferencePolicyV2 | None,
    unresolved_reasons: frozenset[EvaluationContractReason],
    audit: ContractAudit,
) -> EvaluationContractCompileResult:
    rubric_ref = rubric_set_ref(rubric_set)
    spec_ref = evaluator_spec_ref(evaluator_spec) if evaluator_spec is not None else None
    policy_ref = reference_policy_ref(reference_policy) if reference_policy is not None else None
    seed = {
        "rubric_set_ref": _ref_payload(rubric_ref),
        "outcome": outcome.value,
        "evaluator_spec_ref": (_ref_payload(spec_ref) if spec_ref is not None else None),
        "reference_policy_ref": (_ref_payload(policy_ref) if policy_ref is not None else None),
        "unresolved_reasons": sorted(item.value for item in unresolved_reasons),
        "policy_version": EVALUATION_CONTRACT_POLICY_VERSION,
    }
    digest = evaluation_payload_sha256(seed)
    refs = tuple(ref for ref in (rubric_ref, spec_ref, policy_ref) if ref is not None)
    return EvaluationContractCompileResult(
        result_id=f"evaluation-contract-result://sha256/{digest}",
        outcome=outcome,
        evaluator_spec=evaluator_spec,
        reference_policy=reference_policy,
        unresolved_reasons=unresolved_reasons,
        policy_version=EVALUATION_CONTRACT_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(audit, refs),
    )


def _validate_evaluator_spec(evaluator_spec: EvaluatorSpecV2) -> None:
    for binding in evaluator_spec.bindings:
        if binding.binding_sha256 != evaluator_binding_carried_sha256(binding):
            raise EvaluationContractPolicyError("EvaluatorSpec binding identity is stale or mismatched")
    for rule in evaluator_spec.failure_rules:
        if rule.rule_sha256 != evaluator_failure_rule_carried_sha256(rule):
            raise EvaluationContractPolicyError("EvaluatorSpec failure rule identity is stale or mismatched")
    digest = evaluator_spec_carried_sha256(evaluator_spec)
    if (
        evaluator_spec.evaluator_spec_sha256 != digest
        or evaluator_spec.evaluator_spec_id != f"evaluator-spec://sha256/{digest}"
    ):
        raise EvaluationContractPolicyError("EvaluatorSpec identity is stale or mismatched")


def _validate_observation(observation: EvaluatorFailureObservation) -> None:
    digest = evaluator_failure_observation_sha256(observation)
    if (
        observation.observation_sha256 != digest
        or observation.observation_id != f"evaluator-failure-observation://sha256/{digest}"
    ):
        raise EvaluationContractPolicyError("failure observation identity is stale or mismatched")


def _observation_ref(observation: EvaluatorFailureObservation) -> ObjectRef:
    return ObjectRef(
        object_type="evaluator-failure-observation",
        object_id=observation.observation_id,
        object_version="r4-06",
        object_sha256=observation.observation_sha256,
    )


def _validate_reference_policy(
    reference_policy: ReferencePolicyV2,
    evaluator_spec: EvaluatorSpecV2,
) -> None:
    if reference_policy.evaluator_spec_ref != evaluator_spec_ref(evaluator_spec):
        raise EvaluationContractPolicyError("ReferencePolicy EvaluatorSpec ref is stale or mismatched")
    digest = reference_policy_carried_sha256(reference_policy)
    if (
        reference_policy.reference_policy_sha256 != digest
        or reference_policy.reference_policy_id != f"reference-policy://sha256/{digest}"
    ):
        raise EvaluationContractPolicyError("ReferencePolicy identity is stale or mismatched")
    binding_refs = {_ref_key(ref) for binding in evaluator_spec.bindings for ref in binding.reference_refs}
    policy_refs = {_ref_key(ref) for ref in reference_policy.reference_refs}
    if binding_refs != policy_refs:
        raise EvaluationContractPolicyError("ReferencePolicy refs do not match EvaluatorSpec bindings")


def _validate_access_request(
    request: EvaluatorReferenceAccessRequest,
) -> None:
    digest = evaluator_reference_access_request_sha256(request)
    if (
        request.request_sha256 != digest
        or request.request_id != f"evaluator-reference-access-request://sha256/{digest}"
    ):
        raise EvaluationContractPolicyError("reference access request identity is stale or mismatched")


def _model_authorization_reasons(
    *,
    binding: EvaluatorBindingV2,
    reference_policy: ReferencePolicyV2,
    request: EvaluatorReferenceAccessRequest,
    authorization: VerifiedModelDomainAuthorization | None,
    evaluated_at: datetime,
) -> set[EvaluatorReferenceAccessReason]:
    if request.model_profile_ref != binding.model_profile_ref:
        return {EvaluatorReferenceAccessReason.MODEL_PROFILE_MISMATCH}
    if authorization is None:
        return {EvaluatorReferenceAccessReason.MODEL_AUTHORIZATION_MISSING}
    _validate_model_authorization(authorization)
    reasons: set[EvaluatorReferenceAccessReason] = set()
    if authorization.model_profile_ref != binding.model_profile_ref:
        reasons.add(EvaluatorReferenceAccessReason.MODEL_PROFILE_MISMATCH)
    if (
        reference_policy.mode is ReferenceMode.PRIVATE_ANSWER
        and authorization.domain is not EvaluatorModelDomainV2.INTERNAL_EVALUATOR_APPROVED
    ):
        reasons.add(EvaluatorReferenceAccessReason.MODEL_DOMAIN_DENIED)
    if "EVALUATION" not in authorization.allowed_purposes:
        reasons.add(EvaluatorReferenceAccessReason.MODEL_PURPOSE_DENIED)
    if reference_policy.reference_data_class not in authorization.allowed_data_classes:
        reasons.add(EvaluatorReferenceAccessReason.MODEL_DATA_CLASS_DENIED)
    if {_ref_key(ref) for ref in authorization.approved_reference_refs} != {
        _ref_key(ref) for ref in binding.reference_refs
    }:
        reasons.add(EvaluatorReferenceAccessReason.REFERENCE_SCOPE_DENIED)
    if evaluated_at < authorization.valid_from:
        reasons.add(EvaluatorReferenceAccessReason.MODEL_AUTHORIZATION_NOT_YET_VALID)
    if evaluated_at >= authorization.expires_at:
        reasons.add(EvaluatorReferenceAccessReason.MODEL_AUTHORIZATION_EXPIRED)
    if not authorization.internal_endpoint:
        reasons.add(EvaluatorReferenceAccessReason.EXTERNAL_ENDPOINT_DENIED)
    if not authorization.retention_policy_satisfied:
        reasons.add(EvaluatorReferenceAccessReason.RETENTION_POLICY_DENIED)
    if not authorization.training_use_approved:
        reasons.add(EvaluatorReferenceAccessReason.TRAINING_USE_DENIED)
    if not authorization.independent_authority_verified:
        reasons.add(EvaluatorReferenceAccessReason.AUTHORITY_UNVERIFIED)
    if not authorization.non_self_signed:
        reasons.add(EvaluatorReferenceAccessReason.SELF_SIGNED_APPROVAL_DENIED)
    return reasons


def _validate_model_authorization(
    authorization: VerifiedModelDomainAuthorization,
) -> None:
    digest = verified_model_domain_authorization_sha256(authorization)
    if (
        authorization.authorization_sha256 != digest
        or authorization.authorization_id != f"verified-model-domain-authorization://sha256/{digest}"
    ):
        raise EvaluationContractPolicyError("model-domain authorization identity is stale or mismatched")


def _compile_grant(
    *,
    evaluator_spec: EvaluatorSpecV2,
    reference_policy: ReferencePolicyV2,
    binding: EvaluatorBindingV2,
    model_domain_approval_ref: ObjectRef | None,
    audit: ContractAudit,
) -> EvaluatorReferenceGrantV2:
    spec_ref = evaluator_spec_ref(evaluator_spec)
    policy_ref = reference_policy_ref(reference_policy)
    assert binding.evaluator_principal_id is not None
    grant = EvaluatorReferenceGrantV2(
        grant_id="evaluator-reference-grant://pending",
        evaluator_spec_ref=spec_ref,
        reference_policy_ref=policy_ref,
        evaluator_binding_id=binding.evaluator_binding_id,
        evaluator_principal_id=binding.evaluator_principal_id,
        model_profile_ref=binding.model_profile_ref,
        model_domain_approval_ref=model_domain_approval_ref,
        reference_refs=binding.reference_refs,
        purpose="EVALUATION",
        policy_version=REFERENCE_ACCESS_POLICY_VERSION,
        grant_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                spec_ref,
                policy_ref,
                *binding.reference_refs,
                *((model_domain_approval_ref,) if model_domain_approval_ref else ()),
            ),
        ),
    )
    digest = evaluator_reference_grant_carried_sha256(grant)
    return grant.model_copy(
        update={
            "grant_id": f"evaluator-reference-grant://sha256/{digest}",
            "grant_sha256": digest,
        }
    )


def _access_result(
    *,
    evaluator_spec: EvaluatorSpecV2,
    reference_policy: ReferencePolicyV2,
    request: EvaluatorReferenceAccessRequest,
    outcome: EvaluatorReferenceAccessOutcome,
    grant: EvaluatorReferenceGrantV2 | None,
    reasons: frozenset[EvaluatorReferenceAccessReason],
    audit: ContractAudit,
) -> EvaluatorReferenceAccessResult:
    request_ref = _access_request_ref(request)
    spec_ref = evaluator_spec_ref(evaluator_spec)
    policy_ref = reference_policy_ref(reference_policy)
    grant_ref = evaluator_reference_grant_ref(grant) if grant is not None else None
    seed = {
        "request_ref": _ref_payload(request_ref),
        "evaluator_spec_ref": _ref_payload(spec_ref),
        "reference_policy_ref": _ref_payload(policy_ref),
        "outcome": outcome.value,
        "grant_ref": (_ref_payload(grant_ref) if grant_ref is not None else None),
        "reasons": sorted(item.value for item in reasons),
        "policy_version": REFERENCE_ACCESS_POLICY_VERSION,
    }
    digest = evaluation_payload_sha256(seed)
    refs = tuple(ref for ref in (request_ref, spec_ref, policy_ref, grant_ref) if ref is not None)
    return EvaluatorReferenceAccessResult(
        access_result_id=f"evaluator-reference-access-result://sha256/{digest}",
        request_ref=request_ref,
        evaluator_spec_ref=spec_ref,
        reference_policy_ref=policy_ref,
        outcome=outcome,
        grant=grant,
        reasons=reasons,
        policy_version=REFERENCE_ACCESS_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(audit, refs),
    )


def _access_request_ref(
    request: EvaluatorReferenceAccessRequest,
) -> ObjectRef:
    return ObjectRef(
        object_type="evaluator-reference-access-request",
        object_id=request.request_id,
        object_version="r4-06",
        object_sha256=request.request_sha256,
    )


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=tuple(
            sorted(
                set(refs),
                key=_ref_key,
            )
        ),
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
