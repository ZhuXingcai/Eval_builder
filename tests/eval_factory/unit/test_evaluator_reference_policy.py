from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts import (
    EvaluatorExecutionModeV2,
    EvaluatorFailureSignalV2,
    EvaluatorModelDomainV2,
    EvaluatorReferenceDataClassV2,
    RubricCriterionV2,
    RubricJudgedObjectKindV2,
    RubricJudgedObjectV2,
    RubricReachabilityV2,
    RubricSetV2,
    RubricSourceModeV2,
    evaluator_binding_carried_sha256,
    evaluator_failure_rule_carried_sha256,
    evaluator_reference_grant_carried_sha256,
    evaluator_spec_carried_sha256,
    reference_policy_carried_sha256,
    rubric_criterion_carried_sha256,
    rubric_reachability_carried_sha256,
    rubric_set_carried_sha256,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.task import EvaluationFailureClass, ReferenceMode, RubricVisibility
from eval_factory.task_authoring import (
    EVALUATION_CONTRACT_POLICY_VERSION,
    REFERENCE_ACCESS_POLICY_VERSION,
    EvaluationContractCompiler,
    EvaluationContractOutcome,
    EvaluationContractPolicyError,
    EvaluationContractReason,
    EvaluatorAccessPrincipalType,
    EvaluatorBindingDefinition,
    EvaluatorFailureClassifier,
    EvaluatorFailureObservation,
    EvaluatorReferenceAccessGate,
    EvaluatorReferenceAccessOutcome,
    EvaluatorReferenceAccessReason,
    EvaluatorReferenceAccessRequest,
    VerifiedModelDomainAuthorization,
    evaluator_failure_observation_sha256,
    evaluator_reference_access_request_sha256,
    verified_model_domain_authorization_sha256,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
FOURTH_HASH = "d" * 64
PRIMARY_BINDING = "evaluator-binding://r4-06/primary"
SECONDARY_BINDING = "evaluator-binding://r4-06/secondary"
PRIMARY_PRINCIPAL = "principal://evaluator/r4-06/primary"
SECONDARY_PRINCIPAL = "principal://evaluator/r4-06/secondary"
ROOT = Path(__file__).resolve().parents[3]


def _audit(
    created_at: datetime = datetime(2026, 7, 26, tzinfo=UTC),
    *,
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="evaluator-reference-policy-test",
        governing_versions=(VersionBinding(component="evaluation-contract", version="r4-06"),),
        input_refs=input_refs,
    )


def _ref(
    object_type: str,
    suffix: str,
    *,
    digest: str = HASH,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _evidence(suffix: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://evaluation/{suffix}",
        subject_ref=_ref("file-version-projection", suffix),
        source_spans=(
            SourceSpanRef(
                span_id=f"source-span://evaluation/{suffix}",
                source_trace_id="source-trace://r4-06",
                raw_sha256=HASH,
            ),
        ),
        polarity=EvidencePolarity.POSITIVE,
        capability="evaluation-contract",
        capability_complete=True,
    )


def _criterion(
    *,
    suffix: str = "primary",
    evaluator_binding: str = PRIMARY_BINDING,
) -> RubricCriterionV2:
    reachability = RubricReachabilityV2(
        reachability_id="rubric-reachability://pending",
        prompt_requirement_ids=(f"requirement://r4-06/{suffix}",),
        attachment_dependency_ids=(),
        allowed_tool_ids=(),
        evidence=(_evidence(suffix),),
        capability_complete=True,
        reachability_sha256=HASH,
    )
    reachability_digest = rubric_reachability_carried_sha256(reachability)
    reachability = reachability.model_copy(
        update={
            "reachability_id": f"rubric-reachability://sha256/{reachability_digest}",
            "reachability_sha256": reachability_digest,
        }
    )
    criterion = RubricCriterionV2(
        criterion_id="rubric-criterion://pending",
        judged_object=RubricJudgedObjectV2(
            judged_object_id=f"judged-object://r4-06/{suffix}",
            kind=RubricJudgedObjectKindV2.CONTESTANT_RESPONSE,
            description=f"Contestant response for {suffix}.",
        ),
        description=f"Evaluate the {suffix} visible requirement.",
        weight=1.0,
        reachability=reachability,
        visibility=RubricVisibility.EVALUATOR_ONLY,
        evaluator_binding=evaluator_binding,
        approval_status="CANDIDATE",
        criterion_sha256=HASH,
    )
    criterion_digest = rubric_criterion_carried_sha256(criterion)
    return criterion.model_copy(
        update={
            "criterion_id": f"rubric-criterion://sha256/{criterion_digest}",
            "criterion_sha256": criterion_digest,
        }
    )


def _rubric_set(
    *,
    criteria: tuple[RubricCriterionV2, ...] | None = None,
    audit: ContractAudit | None = None,
) -> RubricSetV2:
    resolved = criteria or (_criterion(),)
    rubric = RubricSetV2(
        rubric_set_id="rubric-set://pending",
        rubric_version=1,
        supersedes_rubric_set_ref=None,
        task_draft_ref=_ref("task-draft", "r4-06", version="v2"),
        source_mode=RubricSourceModeV2.GENERATED,
        criteria=resolved,
        total_weight=sum(item.weight for item in resolved),
        imported_from_ref=None,
        model_profile="internal-rubric-author-v1",
        prompt_version="rubric-authoring-prompt/v1",
        policy_version="rubric-authoring/r4-05-v1",
        rubric_set_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = rubric_set_carried_sha256(rubric)
    return rubric.model_copy(
        update={
            "rubric_set_id": f"rubric-set://sha256/{digest}",
            "rubric_set_sha256": digest,
        }
    )


def _reference_ref(mode: ReferenceMode, suffix: str = "primary") -> ObjectRef | None:
    object_type = {
        ReferenceMode.NONE: None,
        ReferenceMode.STRUCTURED_EXPECTATIONS: "structured-expectation",
        ReferenceMode.PRIVATE_ANSWER: "private-reference",
        ReferenceMode.TRACE_BEHAVIOR: "trace-behavior-reference",
        ReferenceMode.HUMAN_ONLY: "human-only-reference",
    }[mode]
    return None if object_type is None else _ref(object_type, f"r4-06/{suffix}")


def _binding_definition(
    *,
    binding_id: str = PRIMARY_BINDING,
    execution_mode: EvaluatorExecutionModeV2 = EvaluatorExecutionModeV2.DETERMINISTIC,
    principal_id: str | None = PRIMARY_PRINCIPAL,
    model_profile_ref: ObjectRef | None = None,
    reference_refs: tuple[ObjectRef, ...] = (),
) -> EvaluatorBindingDefinition:
    if execution_mode is EvaluatorExecutionModeV2.MODEL and model_profile_ref is None:
        model_profile_ref = _ref("model-profile", "r4-06/internal-evaluator")
    if execution_mode is EvaluatorExecutionModeV2.HUMAN_ONLY:
        principal_id = None
        model_profile_ref = None
    return EvaluatorBindingDefinition(
        evaluator_binding_id=binding_id,
        evaluator_type="contract-evaluator",
        execution_mode=execution_mode,
        evaluator_version="r4-06-v1",
        input_contract_ref=_ref("evaluator-input-contract", "r4-06"),
        output_contract_ref=_ref("evaluator-output-contract", "r4-06"),
        evaluator_principal_id=principal_id,
        model_profile_ref=model_profile_ref,
        reference_refs=reference_refs,
        timeout_seconds=300,
    )


def _compile(
    *,
    rubric_set: RubricSetV2 | None = None,
    binding_definitions: tuple[EvaluatorBindingDefinition, ...] | None = None,
    reference_mode: ReferenceMode = ReferenceMode.NONE,
    audit: ContractAudit | None = None,
):
    return EvaluationContractCompiler().compile(
        rubric_set=rubric_set or _rubric_set(),
        binding_definitions=(
            (_binding_definition(),) if binding_definitions is None else binding_definitions
        ),
        reference_mode=reference_mode,
        audit=audit or _audit(),
    )


def _compiled_for_mode(
    mode: ReferenceMode,
    *,
    execution_mode: EvaluatorExecutionModeV2 | None = None,
):
    reference_ref = _reference_ref(mode)
    if execution_mode is None:
        execution_mode = (
            EvaluatorExecutionModeV2.HUMAN_ONLY
            if mode is ReferenceMode.HUMAN_ONLY
            else EvaluatorExecutionModeV2.DETERMINISTIC
        )
    return _compile(
        reference_mode=mode,
        binding_definitions=(
            _binding_definition(
                execution_mode=execution_mode,
                reference_refs=(() if reference_ref is None else (reference_ref,)),
            ),
        ),
    )


def _observation(
    signals: tuple[EvaluatorFailureSignalV2, ...],
    *,
    audit: ContractAudit | None = None,
) -> EvaluatorFailureObservation:
    observation = EvaluatorFailureObservation(
        observation_id="evaluator-failure-observation://pending",
        source_observation_ref=_ref("evaluation-observation", "r4-06"),
        signals=signals,
        evidence_refs=(),
        policy_version=EVALUATION_CONTRACT_POLICY_VERSION,
        observation_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = evaluator_failure_observation_sha256(observation)
    return observation.model_copy(
        update={
            "observation_id": f"evaluator-failure-observation://sha256/{digest}",
            "observation_sha256": digest,
        }
    )


def _access_request(
    *,
    principal_type: EvaluatorAccessPrincipalType = EvaluatorAccessPrincipalType.EVALUATOR,
    principal_id: str = PRIMARY_PRINCIPAL,
    binding_id: str = PRIMARY_BINDING,
    model_profile_ref: ObjectRef | None = None,
    audit: ContractAudit | None = None,
) -> EvaluatorReferenceAccessRequest:
    request = EvaluatorReferenceAccessRequest(
        request_id="evaluator-reference-access-request://pending",
        principal_type=principal_type,
        principal_id=principal_id,
        purpose="EVALUATION",
        evaluator_binding_id=binding_id,
        model_profile_ref=model_profile_ref,
        policy_version=REFERENCE_ACCESS_POLICY_VERSION,
        request_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = evaluator_reference_access_request_sha256(request)
    return request.model_copy(
        update={
            "request_id": f"evaluator-reference-access-request://sha256/{digest}",
            "request_sha256": digest,
        }
    )


def _authorization(
    *,
    model_profile_ref: ObjectRef,
    domain: EvaluatorModelDomainV2,
    data_class: EvaluatorReferenceDataClassV2,
    reference_refs: tuple[ObjectRef, ...],
    valid_from: datetime = datetime(2026, 7, 25, tzinfo=UTC),
    expires_at: datetime = datetime(2026, 7, 28, tzinfo=UTC),
    internal_endpoint: bool = True,
    retention_policy_satisfied: bool = True,
    training_use_approved: bool = True,
    independent_authority_verified: bool = True,
    non_self_signed: bool = True,
) -> VerifiedModelDomainAuthorization:
    authorization = VerifiedModelDomainAuthorization(
        authorization_id="verified-model-domain-authorization://pending",
        model_profile_ref=model_profile_ref,
        model_domain_approval_ref=_ref("model-domain-approval", "r4-06"),
        verification_ref=_ref("model-domain-verification", "r4-06"),
        domain=domain,
        allowed_purposes=frozenset({"EVALUATION"}),
        allowed_data_classes=frozenset({data_class}),
        approved_reference_refs=reference_refs,
        valid_from=valid_from,
        expires_at=expires_at,
        internal_endpoint=internal_endpoint,
        retention_policy_satisfied=retention_policy_satisfied,
        training_use_approved=training_use_approved,
        independent_authority_verified=independent_authority_verified,
        non_self_signed=non_self_signed,
        policy_version=REFERENCE_ACCESS_POLICY_VERSION,
        authorization_sha256=HASH,
        audit=_audit(),
    )
    digest = verified_model_domain_authorization_sha256(authorization)
    return authorization.model_copy(
        update={
            "authorization_id": f"verified-model-domain-authorization://sha256/{digest}",
            "authorization_sha256": digest,
        }
    )


def _authorize(
    compiled,
    request: EvaluatorReferenceAccessRequest,
    *,
    model_authorization: VerifiedModelDomainAuthorization | None = None,
    evaluated_at: datetime = datetime(2026, 7, 26, tzinfo=UTC),
):
    assert compiled.evaluator_spec is not None
    assert compiled.reference_policy is not None
    return EvaluatorReferenceAccessGate().authorize(
        evaluator_spec=compiled.evaluator_spec,
        reference_policy=compiled.reference_policy,
        request=request,
        model_authorization=model_authorization,
        evaluated_at=evaluated_at,
        audit=_audit(),
    )


def test_compiler_resolves_exact_rubric_binding_and_canonical_rules() -> None:
    result = _compile()

    assert result.outcome is EvaluationContractOutcome.COMPILED
    assert result.evaluator_spec is not None
    assert result.reference_policy is not None
    assert not result.unresolved_reasons
    binding = result.evaluator_spec.bindings[0]
    assert binding.evaluator_binding_id == PRIMARY_BINDING
    assert binding.criterion_ids == (_rubric_set().criteria[0].criterion_id,)
    assert binding.binding_sha256 == evaluator_binding_carried_sha256(binding)
    assert result.evaluator_spec.evaluator_spec_sha256 == evaluator_spec_carried_sha256(result.evaluator_spec)
    assert result.reference_policy.reference_policy_sha256 == reference_policy_carried_sha256(
        result.reference_policy
    )
    assert set(item.signal for item in result.evaluator_spec.failure_rules) == set(EvaluatorFailureSignalV2)
    assert all(
        item.rule_sha256 == evaluator_failure_rule_carried_sha256(item)
        for item in result.evaluator_spec.failure_rules
    )


def test_compiler_derives_shared_and_distinct_criterion_coverage() -> None:
    shared = _rubric_set(
        criteria=(
            _criterion(suffix="one"),
            _criterion(suffix="two"),
        )
    )
    shared_result = _compile(rubric_set=shared)
    assert shared_result.evaluator_spec is not None
    assert shared_result.evaluator_spec.bindings[0].criterion_ids == tuple(
        sorted(item.criterion_id for item in shared.criteria)
    )

    distinct = _rubric_set(
        criteria=(
            _criterion(suffix="one"),
            _criterion(
                suffix="two",
                evaluator_binding=SECONDARY_BINDING,
            ),
        )
    )
    first_definition = _binding_definition()
    second_definition = _binding_definition(
        binding_id=SECONDARY_BINDING,
        principal_id=SECONDARY_PRINCIPAL,
    )
    distinct_result = _compile(
        rubric_set=distinct,
        binding_definitions=(second_definition, first_definition),
    )
    assert distinct_result.evaluator_spec is not None
    assert tuple(item.evaluator_binding_id for item in distinct_result.evaluator_spec.bindings) == (
        PRIMARY_BINDING,
        SECONDARY_BINDING,
    )


def test_missing_binding_is_typed_blocked_without_partial_contracts() -> None:
    result = _compile(binding_definitions=())

    assert result.outcome is EvaluationContractOutcome.BLOCKED_BINDING
    assert result.evaluator_spec is None
    assert result.reference_policy is None
    assert result.unresolved_reasons == frozenset({EvaluationContractReason.MISSING_EVALUATOR_BINDING})


def test_duplicate_extra_and_stale_binding_inputs_fail_closed() -> None:
    definition = _binding_definition()
    with pytest.raises(EvaluationContractPolicyError, match="duplicate"):
        _compile(binding_definitions=(definition, definition))
    with pytest.raises(EvaluationContractPolicyError, match="extra"):
        _compile(
            binding_definitions=(
                definition,
                _binding_definition(
                    binding_id=SECONDARY_BINDING,
                    principal_id=SECONDARY_PRINCIPAL,
                ),
            )
        )
    stale = _rubric_set().model_copy(update={"rubric_set_sha256": OTHER_HASH})
    with pytest.raises(EvaluationContractPolicyError, match="RubricSet"):
        _compile(rubric_set=stale)


@pytest.mark.parametrize("mode", tuple(ReferenceMode))
def test_compiler_supports_all_reference_modes(mode: ReferenceMode) -> None:
    result = _compiled_for_mode(mode)

    assert result.outcome is EvaluationContractOutcome.COMPILED
    assert result.evaluator_spec is not None
    assert result.reference_policy is not None
    assert result.reference_policy.mode is mode
    expected_ref = _reference_ref(mode)
    expected_refs = () if expected_ref is None else (expected_ref,)
    assert result.evaluator_spec.bindings[0].reference_refs == expected_refs
    assert result.reference_policy.reference_refs == expected_refs
    assert result.reference_policy.contestant_access is False
    assert result.reference_policy.attachment_producer_access is False


def test_reference_mode_mismatch_blocks_without_silent_none_fallback() -> None:
    private_without_reference = _compile(
        reference_mode=ReferenceMode.PRIVATE_ANSWER,
        binding_definitions=(_binding_definition(),),
    )
    assert private_without_reference.outcome is EvaluationContractOutcome.BLOCKED_POLICY
    assert private_without_reference.evaluator_spec is None
    assert private_without_reference.reference_policy is None
    assert EvaluationContractReason.REFERENCE_MODE_MISMATCH in (private_without_reference.unresolved_reasons)

    with pytest.raises(EvaluationContractPolicyError, match="private-reference"):
        _compile(
            reference_mode=ReferenceMode.PRIVATE_ANSWER,
            binding_definitions=(
                _binding_definition(
                    reference_refs=(_ref("structured-expectation", "wrong-mode"),),
                ),
            ),
        )
    human_ref = _reference_ref(ReferenceMode.HUMAN_ONLY)
    assert human_ref is not None
    result = _compile(
        reference_mode=ReferenceMode.HUMAN_ONLY,
        binding_definitions=(
            _binding_definition(
                reference_refs=(human_ref,),
            ),
        ),
    )
    assert result.outcome is EvaluationContractOutcome.BLOCKED_POLICY


@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        pytest.param(
            EvaluatorFailureSignalV2.CONTESTANT_OUTPUT_MISSING,
            EvaluationFailureClass.CONTESTANT_FAILURE,
            id="contestant",
        ),
        pytest.param(
            EvaluatorFailureSignalV2.EVALUATOR_TIMEOUT,
            EvaluationFailureClass.EVALUATOR_FAILURE,
            id="evaluator",
        ),
        pytest.param(
            EvaluatorFailureSignalV2.ENVIRONMENT_STARTUP_FAILURE,
            EvaluationFailureClass.ENVIRONMENT_FAILURE,
            id="environment",
        ),
        pytest.param(
            EvaluatorFailureSignalV2.INSUFFICIENT_OBSERVATION,
            EvaluationFailureClass.INDETERMINATE,
            id="indeterminate",
        ),
    ],
)
def test_failure_classifier_maps_each_causal_domain(
    signal: EvaluatorFailureSignalV2,
    expected: EvaluationFailureClass,
) -> None:
    compiled = _compile()
    assert compiled.evaluator_spec is not None
    classification = EvaluatorFailureClassifier().classify(
        evaluator_spec=compiled.evaluator_spec,
        observation=_observation((signal,)),
        audit=_audit(),
    )

    assert classification.failure_class is expected
    assert len(classification.matched_rule_ids) == 1


def test_failure_classifier_is_conservative_for_empty_indeterminate_and_conflicting_signals() -> None:
    compiled = _compile()
    assert compiled.evaluator_spec is not None
    classifier = EvaluatorFailureClassifier()
    observations = (
        _observation(()),
        _observation(
            (
                EvaluatorFailureSignalV2.CONTESTANT_OUTPUT_MISSING,
                EvaluatorFailureSignalV2.INSUFFICIENT_OBSERVATION,
            )
        ),
        _observation(
            (
                EvaluatorFailureSignalV2.CONTESTANT_OUTPUT_MISSING,
                EvaluatorFailureSignalV2.EVALUATOR_RUNTIME_FAILURE,
            )
        ),
    )

    for observation in observations:
        classification = classifier.classify(
            evaluator_spec=compiled.evaluator_spec,
            observation=observation,
            audit=_audit(),
        )
        assert classification.failure_class is EvaluationFailureClass.INDETERMINATE


def test_failure_classifier_rejects_stale_spec_observation_and_rule() -> None:
    compiled = _compile()
    assert compiled.evaluator_spec is not None
    observation = _observation((EvaluatorFailureSignalV2.EVALUATOR_RUNTIME_FAILURE,))
    with pytest.raises(EvaluationContractPolicyError, match="EvaluatorSpec"):
        EvaluatorFailureClassifier().classify(
            evaluator_spec=compiled.evaluator_spec.model_copy(update={"evaluator_spec_sha256": OTHER_HASH}),
            observation=observation,
            audit=_audit(),
        )
    with pytest.raises(EvaluationContractPolicyError, match="observation"):
        EvaluatorFailureClassifier().classify(
            evaluator_spec=compiled.evaluator_spec,
            observation=observation.model_copy(update={"observation_sha256": OTHER_HASH}),
            audit=_audit(),
        )
    tampered_rule = compiled.evaluator_spec.failure_rules[0].model_copy(update={"rule_sha256": OTHER_HASH})
    with pytest.raises(EvaluationContractPolicyError, match="failure rule"):
        EvaluatorFailureClassifier().classify(
            evaluator_spec=compiled.evaluator_spec.model_copy(
                update={
                    "failure_rules": (
                        tampered_rule,
                        *compiled.evaluator_spec.failure_rules[1:],
                    )
                }
            ),
            observation=observation,
            audit=_audit(),
        )


def test_exact_deterministic_evaluator_receives_only_binding_reference_refs() -> None:
    compiled = _compiled_for_mode(ReferenceMode.STRUCTURED_EXPECTATIONS)
    result = _authorize(compiled, _access_request())

    assert result.outcome is EvaluatorReferenceAccessOutcome.GRANTED
    assert result.grant is not None
    assert result.grant.reference_refs == (_reference_ref(ReferenceMode.STRUCTURED_EXPECTATIONS),)
    assert result.grant.model_profile_ref is None
    assert result.grant.model_domain_approval_ref is None
    assert result.grant.grant_sha256 == evaluator_reference_grant_carried_sha256(result.grant)


@pytest.mark.parametrize(
    "principal_type",
    (
        EvaluatorAccessPrincipalType.CONTESTANT,
        EvaluatorAccessPrincipalType.ATTACHMENT_PRODUCER,
    ),
)
def test_contestant_and_attachment_producer_are_denied_without_reference_disclosure(
    principal_type: EvaluatorAccessPrincipalType,
) -> None:
    compiled = _compiled_for_mode(ReferenceMode.PRIVATE_ANSWER)
    result = _authorize(
        compiled,
        _access_request(
            principal_type=principal_type,
            principal_id=f"principal://{principal_type.value.casefold()}/r4-06",
        ),
    )

    assert result.outcome is EvaluatorReferenceAccessOutcome.DENIED_IDENTITY
    assert result.grant is None
    assert EvaluatorReferenceAccessReason.PRINCIPAL_TYPE_DENIED in result.reasons
    serialized = result.model_dump_json()
    assert "private-reference" not in serialized
    assert HASH not in serialized


def test_wrong_evaluator_principal_or_binding_is_denied_without_grant() -> None:
    compiled = _compiled_for_mode(ReferenceMode.STRUCTURED_EXPECTATIONS)

    wrong_principal = _authorize(
        compiled,
        _access_request(principal_id=SECONDARY_PRINCIPAL),
    )
    wrong_binding = _authorize(
        compiled,
        _access_request(binding_id=SECONDARY_BINDING),
    )

    assert wrong_principal.outcome is EvaluatorReferenceAccessOutcome.DENIED_IDENTITY
    assert wrong_principal.grant is None
    assert wrong_binding.outcome is EvaluatorReferenceAccessOutcome.DENIED_IDENTITY
    assert wrong_binding.grant is None


def test_none_and_human_only_modes_emit_no_automated_grant() -> None:
    none_result = _authorize(
        _compiled_for_mode(ReferenceMode.NONE),
        _access_request(),
    )
    human_result = _authorize(
        _compiled_for_mode(ReferenceMode.HUMAN_ONLY),
        _access_request(),
    )

    assert none_result.outcome is EvaluatorReferenceAccessOutcome.NO_REFERENCE_REQUIRED
    assert none_result.grant is None
    assert human_result.outcome is EvaluatorReferenceAccessOutcome.HUMAN_ONLY
    assert human_result.grant is None


@pytest.mark.parametrize(
    ("mode", "data_class"),
    [
        pytest.param(
            ReferenceMode.STRUCTURED_EXPECTATIONS,
            EvaluatorReferenceDataClassV2.RESTRICTED_EVAL_CONTROL,
            id="structured",
        ),
        pytest.param(
            ReferenceMode.TRACE_BEHAVIOR,
            EvaluatorReferenceDataClassV2.RESTRICTED_TRACE_SPAN,
            id="trace-behavior",
        ),
    ],
)
def test_model_evaluator_requires_current_internal_authorization(
    mode: ReferenceMode,
    data_class: EvaluatorReferenceDataClassV2,
) -> None:
    compiled = _compiled_for_mode(
        mode,
        execution_mode=EvaluatorExecutionModeV2.MODEL,
    )
    assert compiled.evaluator_spec is not None
    binding = compiled.evaluator_spec.bindings[0]
    assert binding.model_profile_ref is not None
    request = _access_request(model_profile_ref=binding.model_profile_ref)
    authorization = _authorization(
        model_profile_ref=binding.model_profile_ref,
        domain=EvaluatorModelDomainV2.INTERNAL_APPROVED,
        data_class=data_class,
        reference_refs=binding.reference_refs,
    )

    result = _authorize(
        compiled,
        request,
        model_authorization=authorization,
    )

    assert result.outcome is EvaluatorReferenceAccessOutcome.GRANTED
    assert result.grant is not None
    assert result.grant.model_profile_ref == binding.model_profile_ref
    assert result.grant.model_domain_approval_ref == (authorization.model_domain_approval_ref)


def test_private_answer_model_requires_exact_evaluator_approved_scope() -> None:
    compiled = _compiled_for_mode(
        ReferenceMode.PRIVATE_ANSWER,
        execution_mode=EvaluatorExecutionModeV2.MODEL,
    )
    assert compiled.evaluator_spec is not None
    binding = compiled.evaluator_spec.bindings[0]
    assert binding.model_profile_ref is not None
    request = _access_request(model_profile_ref=binding.model_profile_ref)
    authorization = _authorization(
        model_profile_ref=binding.model_profile_ref,
        domain=EvaluatorModelDomainV2.INTERNAL_EVALUATOR_APPROVED,
        data_class=EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
        reference_refs=binding.reference_refs,
    )

    result = _authorize(
        compiled,
        request,
        model_authorization=authorization,
    )

    assert result.outcome is EvaluatorReferenceAccessOutcome.GRANTED
    assert result.grant is not None
    assert result.grant.reference_refs == binding.reference_refs


@pytest.mark.parametrize(
    ("authorization_builder", "reason"),
    [
        pytest.param(
            lambda binding: None,
            EvaluatorReferenceAccessReason.MODEL_AUTHORIZATION_MISSING,
            id="missing",
        ),
        pytest.param(
            lambda binding: _authorization(
                model_profile_ref=binding.model_profile_ref,
                domain=EvaluatorModelDomainV2.INTERNAL_APPROVED,
                data_class=EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
                reference_refs=binding.reference_refs,
            ),
            EvaluatorReferenceAccessReason.MODEL_DOMAIN_DENIED,
            id="wrong-domain",
        ),
        pytest.param(
            lambda binding: _authorization(
                model_profile_ref=_ref("model-profile", "wrong"),
                domain=EvaluatorModelDomainV2.INTERNAL_EVALUATOR_APPROVED,
                data_class=EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
                reference_refs=binding.reference_refs,
            ),
            EvaluatorReferenceAccessReason.MODEL_PROFILE_MISMATCH,
            id="wrong-profile",
        ),
        pytest.param(
            lambda binding: _authorization(
                model_profile_ref=binding.model_profile_ref,
                domain=EvaluatorModelDomainV2.INTERNAL_EVALUATOR_APPROVED,
                data_class=EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
                reference_refs=(),
            ),
            EvaluatorReferenceAccessReason.REFERENCE_SCOPE_DENIED,
            id="missing-reference-scope",
        ),
        pytest.param(
            lambda binding: _authorization(
                model_profile_ref=binding.model_profile_ref,
                domain=EvaluatorModelDomainV2.INTERNAL_EVALUATOR_APPROVED,
                data_class=EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
                reference_refs=binding.reference_refs,
                expires_at=datetime(2026, 7, 25, 12, tzinfo=UTC),
            ),
            EvaluatorReferenceAccessReason.MODEL_AUTHORIZATION_EXPIRED,
            id="expired",
        ),
        pytest.param(
            lambda binding: _authorization(
                model_profile_ref=binding.model_profile_ref,
                domain=EvaluatorModelDomainV2.INTERNAL_EVALUATOR_APPROVED,
                data_class=EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
                reference_refs=binding.reference_refs,
                internal_endpoint=False,
            ),
            EvaluatorReferenceAccessReason.EXTERNAL_ENDPOINT_DENIED,
            id="external-endpoint",
        ),
        pytest.param(
            lambda binding: _authorization(
                model_profile_ref=binding.model_profile_ref,
                domain=EvaluatorModelDomainV2.INTERNAL_EVALUATOR_APPROVED,
                data_class=EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
                reference_refs=binding.reference_refs,
                retention_policy_satisfied=False,
            ),
            EvaluatorReferenceAccessReason.RETENTION_POLICY_DENIED,
            id="retention",
        ),
        pytest.param(
            lambda binding: _authorization(
                model_profile_ref=binding.model_profile_ref,
                domain=EvaluatorModelDomainV2.INTERNAL_EVALUATOR_APPROVED,
                data_class=EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
                reference_refs=binding.reference_refs,
                training_use_approved=False,
            ),
            EvaluatorReferenceAccessReason.TRAINING_USE_DENIED,
            id="training",
        ),
        pytest.param(
            lambda binding: _authorization(
                model_profile_ref=binding.model_profile_ref,
                domain=EvaluatorModelDomainV2.INTERNAL_EVALUATOR_APPROVED,
                data_class=EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
                reference_refs=binding.reference_refs,
                independent_authority_verified=False,
            ),
            EvaluatorReferenceAccessReason.AUTHORITY_UNVERIFIED,
            id="authority",
        ),
        pytest.param(
            lambda binding: _authorization(
                model_profile_ref=binding.model_profile_ref,
                domain=EvaluatorModelDomainV2.INTERNAL_EVALUATOR_APPROVED,
                data_class=EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
                reference_refs=binding.reference_refs,
                non_self_signed=False,
            ),
            EvaluatorReferenceAccessReason.SELF_SIGNED_APPROVAL_DENIED,
            id="self-signed",
        ),
    ],
)
def test_private_model_authorization_failures_block_without_partial_grant(
    authorization_builder,
    reason: EvaluatorReferenceAccessReason,
) -> None:
    compiled = _compiled_for_mode(
        ReferenceMode.PRIVATE_ANSWER,
        execution_mode=EvaluatorExecutionModeV2.MODEL,
    )
    assert compiled.evaluator_spec is not None
    binding = compiled.evaluator_spec.bindings[0]
    assert binding.model_profile_ref is not None
    request = _access_request(model_profile_ref=binding.model_profile_ref)
    authorization = authorization_builder(binding)

    result = _authorize(
        compiled,
        request,
        model_authorization=authorization,
    )

    assert result.outcome in {
        EvaluatorReferenceAccessOutcome.BLOCKED_POLICY,
        EvaluatorReferenceAccessOutcome.DENIED_SCOPE,
    }
    assert result.grant is None
    assert reason in result.reasons


def test_access_request_rejects_caller_supplied_reference_scope() -> None:
    with pytest.raises(ValidationError):
        EvaluatorReferenceAccessRequest.model_validate(
            {
                **_access_request().model_dump(mode="python"),
                "requested_reference_refs": (_ref("private-reference", "caller-requested"),),
            }
        )


def test_reference_authorization_requires_timezone_aware_bounds_and_evaluation_time() -> None:
    profile_ref = _ref("model-profile", "r4-06/timezone")
    with pytest.raises(ValidationError, match="timezone-aware"):
        _authorization(
            model_profile_ref=profile_ref,
            domain=EvaluatorModelDomainV2.INTERNAL_EVALUATOR_APPROVED,
            data_class=EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
            reference_refs=(_ref("private-reference", "r4-06/timezone"),),
            valid_from=datetime(2026, 7, 25),
        )

    compiled = _compiled_for_mode(ReferenceMode.STRUCTURED_EXPECTATIONS)
    with pytest.raises(EvaluationContractPolicyError, match="timezone-aware"):
        _authorize(
            compiled,
            _access_request(),
            evaluated_at=datetime(2026, 7, 26),
        )


def test_compile_classify_and_access_are_stable_across_audit_and_order() -> None:
    rubric_first = _rubric_set(
        criteria=(
            _criterion(suffix="one"),
            _criterion(
                suffix="two",
                evaluator_binding=SECONDARY_BINDING,
            ),
        ),
        audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)),
    )
    rubric_second = _rubric_set(
        criteria=rubric_first.criteria,
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    first_definition = _binding_definition()
    second_definition = _binding_definition(
        binding_id=SECONDARY_BINDING,
        principal_id=SECONDARY_PRINCIPAL,
    )
    first = _compile(
        rubric_set=rubric_first,
        binding_definitions=(first_definition, second_definition),
        audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)),
    )
    second = _compile(
        rubric_set=rubric_second,
        binding_definitions=(second_definition, first_definition),
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )

    assert first.evaluator_spec is not None
    assert second.evaluator_spec is not None
    assert first.reference_policy is not None
    assert second.reference_policy is not None
    assert first.evaluator_spec.evaluator_spec_id == second.evaluator_spec.evaluator_spec_id
    assert first.reference_policy.reference_policy_id == (second.reference_policy.reference_policy_id)

    first_observation = _observation(
        (
            EvaluatorFailureSignalV2.EVALUATOR_RUNTIME_FAILURE,
            EvaluatorFailureSignalV2.EVALUATOR_TIMEOUT,
        ),
        audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)),
    )
    second_observation = _observation(
        tuple(reversed(first_observation.signals)),
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    classifier = EvaluatorFailureClassifier()
    first_classification = classifier.classify(
        evaluator_spec=first.evaluator_spec,
        observation=first_observation,
        audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)),
    )
    second_classification = classifier.classify(
        evaluator_spec=second.evaluator_spec,
        observation=second_observation,
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    assert first_classification.classification_id == (second_classification.classification_id)


def test_r4_06_identities_are_stable_across_python_hash_seed() -> None:
    code = """
import runpy

ns = runpy.run_path("tests/eval_factory/unit/test_evaluator_reference_policy.py")
compiled = ns["_compiled_for_mode"](ns["ReferenceMode"].STRUCTURED_EXPECTATIONS)
spec = compiled.evaluator_spec
policy = compiled.reference_policy
assert spec is not None and policy is not None
classification = ns["EvaluatorFailureClassifier"]().classify(
    evaluator_spec=spec,
    observation=ns["_observation"](
        (
            ns["EvaluatorFailureSignalV2"].EVALUATOR_TIMEOUT,
            ns["EvaluatorFailureSignalV2"].EVALUATOR_RUNTIME_FAILURE,
        )
    ),
    audit=ns["_audit"](),
)
access = ns["_authorize"](compiled, ns["_access_request"]())
assert access.grant is not None
print(spec.evaluator_spec_id)
print(policy.reference_policy_id)
print(classification.classification_id)
print(access.grant.grant_id)
"""
    outputs = []
    for seed in ("1", "99"):
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONPATH": "src",
            },
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout.strip())

    assert len(set(outputs)) == 1
