from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    ApprovalMode,
    EnvironmentAlternative,
    EnvironmentRequirement,
    EnvironmentStrategy,
    EnvironmentStrategyChoice,
    ExampleKind,
    FinalReviewScope,
    LabelPlan,
    PlanExample,
    QueryPackagingChoice,
    UserApprovalPolicy,
    UserApprovalRequest,
    UserDecision,
)
from eval_factory.contracts.approval_v2 import (
    USER_APPROVAL_REQUEST_POLICY_VERSION,
    ApprovalRequestGenerationPolicyV2,
    FinalDatasetReviewPreviewV2,
    UserApprovalRequestCompilationOutcomeV2,
    UserApprovalRequestCompilationResultV2,
    approval_request_generation_policy_v2_ref,
    environment_strategy_carried_sha256,
    environment_strategy_ref,
    final_dataset_review_preview_v2_ref,
    label_plan_carried_sha256,
    label_plan_ref,
    user_approval_policy_carried_sha256,
    user_approval_policy_ref,
    user_approval_request_carried_sha256,
    user_approval_request_compilation_result_v2_ref,
    user_approval_request_ref,
    validate_approval_request_generation_policy_v2_identity,
    validate_final_dataset_review_preview_v2_identity,
    validate_user_approval_policy_identity,
    validate_user_approval_request_compilation_result_v2_identity,
    validate_user_approval_request_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-04/{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _audit(
    *,
    created_at: datetime = NOW,
    created_by: str = "r7-04-contract-test",
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    by_key = {
        (
            ref.object_type,
            ref.object_id,
            ref.object_version,
            ref.object_sha256,
        ): ref
        for ref in input_refs
    }
    ordered_refs = tuple(by_key[key] for key in sorted(by_key))
    return ContractAudit(
        created_at=created_at,
        created_by=created_by,
        governing_versions=(
            VersionBinding(
                component="user-approval-request",
                version=USER_APPROVAL_REQUEST_POLICY_VERSION,
            ),
        ),
        input_refs=ordered_refs,
    )


def _generation_policy() -> ApprovalRequestGenerationPolicyV2:
    return ApprovalRequestGenerationPolicyV2.create(
        max_requests_per_checkpoint=10,
        max_subject_refs_per_request=20,
        max_preview_refs_per_request=10,
        max_examples_per_plan=20,
        max_preview_characters_per_request=20_000,
        max_preview_characters_per_checkpoint=100_000,
        max_final_sample_refs=100,
        audit=_audit(),
    )


def _approval_policy(
    *,
    mode: ApprovalMode = ApprovalMode.PLAN_GATES,
    checkpoints: frozenset[ApprovalCheckpoint] = frozenset(
        {
            ApprovalCheckpoint.LABEL_PLAN,
            ApprovalCheckpoint.TASK_REWRITE_PLAN,
            ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        }
    ),
    scope: FinalReviewScope = FinalReviewScope.NONE,
    audit: ContractAudit | None = None,
) -> UserApprovalPolicy:
    value = UserApprovalPolicy(
        policy_id="user-approval-policy://pending",
        policy_version=USER_APPROVAL_REQUEST_POLICY_VERSION,
        mode=mode,
        enabled_checkpoints=checkpoints,
        final_review_scope=scope,
        explicit_unattended_choice=False,
        audit=audit or _audit(),
    )
    digest = user_approval_policy_carried_sha256(value)
    return value.model_copy(update={"policy_id": f"user-approval-policy://sha256/{digest}"})


def _evidence(suffix: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://r7-04/{suffix}",
        subject_ref=_ref("safe-evidence-projection", suffix),
        source_spans=(
            SourceSpanRef(
                span_id=f"source-span://r7-04/{suffix}",
                source_trace_id=f"source-trace://r7-04/{suffix}",
                raw_sha256=HASH,
            ),
        ),
        polarity=EvidencePolarity.POSITIVE,
        capability="approval-preview",
        capability_complete=True,
    )


def _label_plan(*, audit: ContractAudit | None = None) -> LabelPlan:
    label_spec_ref = _ref("label-spec", "label", version="v2")
    value = LabelPlan(
        label_plan_id="label-plan://pending",
        label_spec_ref=label_spec_ref,
        intent="Detect a capability-complete recovery behavior.",
        boundary="Use executed structured facts and abstain on incomplete evidence.",
        deterministic_clause_refs=(_ref("structured-predicate", "positive", version="v2"),),
        semantic_clause_refs=(),
        examples=tuple(
            PlanExample(
                example_id=f"plan-example://r7-04/{kind.value.casefold()}",
                kind=kind,
                input_summary=f"Bounded {kind.value.casefold()} summary.",
                expected_treatment=f"Apply {kind.value.casefold()} treatment.",
            )
            for kind in (
                ExampleKind.POSITIVE,
                ExampleKind.NEGATIVE,
                ExampleKind.AMBIGUOUS,
                ExampleKind.ABSTAIN,
            )
        ),
        abstain_rules=("Abstain when required capability is incomplete.",),
        expected_model_path=(),
        estimated_model_requests_per_trace=0,
        blind_spots=("Incomplete source ranges cannot prove absence.",),
        audit=audit or _audit(input_refs=(label_spec_ref,)),
    )
    digest = label_plan_carried_sha256(value)
    return value.model_copy(update={"label_plan_id": f"label-plan://sha256/{digest}"})


def _environment_strategy(
    *,
    audit: ContractAudit | None = None,
) -> EnvironmentStrategy:
    task_ref = _ref("task-draft", "environment")
    value = EnvironmentStrategy(
        environment_strategy_id="environment-strategy://pending",
        requirements=(
            EnvironmentRequirement(
                requirement_id="environment-requirement://r7-04/workspace",
                affected_subject_refs=(task_ref,),
                description="A non-standard workspace is required.",
                requires_special_account=False,
                requires_nonstandard_environment=True,
                evidence_refs=(_evidence("environment"),),
                alternatives=tuple(
                    EnvironmentAlternative(
                        strategy=strategy,
                        feasible=True,
                        capability_impact=f"Bounded impact for {strategy.value}.",
                    )
                    for strategy in EnvironmentStrategyChoice
                ),
            ),
        ),
        query_packaging_options=frozenset(QueryPackagingChoice),
        recommended_strategy=EnvironmentStrategyChoice.REWRITE_STANDARD_ENV,
        recommendation_reason="Avoid private credentials.",
        credential_fabrication_forbidden=True,
        audit=audit or _audit(input_refs=(task_ref,)),
    )
    digest = environment_strategy_carried_sha256(value)
    return value.model_copy(update={"environment_strategy_id": (f"environment-strategy://sha256/{digest}")})


def _request(
    *,
    policy_ref: ObjectRef,
    checkpoint: ApprovalCheckpoint = ApprovalCheckpoint.LABEL_PLAN,
    suffix: str = "label",
    audit: ContractAudit | None = None,
) -> UserApprovalRequest:
    subject_ref = _ref("label-spec", suffix)
    plan_ref = _ref("label-plan", suffix)
    projection_ref = _ref("user-approval-projection", suffix)
    preview_ref = _ref("label-plan", suffix)
    value = UserApprovalRequest(
        request_id="user-approval-request://pending",
        checkpoint=checkpoint,
        requested_by="requesting-user",
        approval_policy_ref=policy_ref,
        subject_refs=(subject_ref,),
        plan_ref=plan_ref,
        projection_ref=projection_ref,
        preview_refs=(preview_ref,),
        available_decisions=frozenset(
            {
                UserDecision.ACCEPT,
                UserDecision.ADJUST,
                UserDecision.REJECT,
                UserDecision.REQUEST_MORE_EXAMPLES,
            }
        ),
        affected_stages=("label",),
        idempotency_key=f"request-r7-04-{suffix}",
        audit=audit
        or _audit(
            input_refs=(
                policy_ref,
                subject_ref,
                plan_ref,
                projection_ref,
                preview_ref,
            )
        ),
    )
    digest = user_approval_request_carried_sha256(value)
    return value.model_copy(update={"request_id": f"user-approval-request://sha256/{digest}"})


def _reidentify_request(
    value: UserApprovalRequest,
    **updates: object,
) -> UserApprovalRequest:
    candidate = value.model_copy(update=updates)
    audit_refs = (
        candidate.approval_policy_ref,
        *candidate.subject_refs,
        *((candidate.plan_ref,) if candidate.plan_ref is not None else ()),
        candidate.projection_ref,
        *candidate.preview_refs,
    )
    candidate = candidate.model_copy(
        update={
            "request_id": "user-approval-request://pending",
            "audit": _audit(input_refs=audit_refs),
        }
    )
    digest = user_approval_request_carried_sha256(candidate)
    return candidate.model_copy(update={"request_id": f"user-approval-request://sha256/{digest}"})


def _final_preview(
    scope: FinalReviewScope,
    *,
    audit: ContractAudit | None = None,
) -> FinalDatasetReviewPreviewV2:
    subjects = (_ref("evaluation-item", "a"),)
    prompt_refs = (
        (_ref("user-prompt-projection", "a"),)
        if scope in {FinalReviewScope.PROMPTS, FinalReviewScope.FULL_DATASET}
        else ()
    )
    item_refs = (
        (_ref("user-item-projection", "a"),)
        if scope
        in {
            FinalReviewScope.SELECTED_ITEMS,
            FinalReviewScope.FULL_DATASET,
        }
        else ()
    )
    dataset_ref = _ref("user-dataset-projection", "batch") if scope is FinalReviewScope.FULL_DATASET else None
    quality_refs = (_ref("batch-quality-report", "batch"),)
    finding_refs = (_ref("low-severity-finding-projection", "p2"),)
    sample_refs = (_ref("approval-navigation-sample", "a"),)
    refs = (
        *subjects,
        *prompt_refs,
        *item_refs,
        *((dataset_ref,) if dataset_ref is not None else ()),
        *quality_refs,
        *finding_refs,
        *sample_refs,
    )
    return FinalDatasetReviewPreviewV2.create(
        scope=scope,
        subject_refs=subjects,
        prompt_projection_refs=prompt_refs,
        selected_item_projection_refs=item_refs,
        dataset_projection_ref=dataset_ref,
        quality_summary_refs=quality_refs,
        open_low_severity_finding_refs=finding_refs,
        sample_navigation_refs=sample_refs,
        projection_policy_version="user-approval-projection/r7-04-v1",
        audit=audit or _audit(input_refs=refs),
    )


def test_generation_policy_is_strict_current_and_audit_independent() -> None:
    first = _generation_policy()
    second = ApprovalRequestGenerationPolicyV2.create(
        max_requests_per_checkpoint=first.max_requests_per_checkpoint,
        max_subject_refs_per_request=first.max_subject_refs_per_request,
        max_preview_refs_per_request=first.max_preview_refs_per_request,
        max_examples_per_plan=first.max_examples_per_plan,
        max_preview_characters_per_request=(first.max_preview_characters_per_request),
        max_preview_characters_per_checkpoint=(first.max_preview_characters_per_checkpoint),
        max_final_sample_refs=first.max_final_sample_refs,
        audit=_audit(
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
            created_by="different-actor",
        ),
    )

    assert approval_request_generation_policy_v2_ref(first) == (
        approval_request_generation_policy_v2_ref(second)
    )
    validate_approval_request_generation_policy_v2_identity(first)
    with pytest.raises(ValidationError):
        ApprovalRequestGenerationPolicyV2.model_validate({**first.model_dump(mode="python"), "unknown": True})
    with pytest.raises(ValidationError):
        ApprovalRequestGenerationPolicyV2.model_validate(
            {
                **first.model_dump(mode="python"),
                "schema_version": "eval-factory/wrong/v2",
            }
        )


@pytest.mark.parametrize(
    "scope",
    (
        FinalReviewScope.PROMPTS,
        FinalReviewScope.SELECTED_ITEMS,
        FinalReviewScope.FULL_DATASET,
    ),
)
def test_final_preview_enforces_scope_and_exact_counts(
    scope: FinalReviewScope,
) -> None:
    preview = _final_preview(scope)

    validate_final_dataset_review_preview_v2_identity(preview)
    assert preview.subject_count == len(preview.subject_refs)
    assert preview.prompt_count == len(preview.prompt_projection_refs)
    assert preview.selected_item_count == len(preview.selected_item_projection_refs)
    assert preview.open_low_severity_finding_count == len(preview.open_low_severity_finding_refs)
    assert preview.automated_hard_gates_passed is True
    assert final_dataset_review_preview_v2_ref(preview).object_sha256 == (preview.preview_sha256)

    with pytest.raises(ValidationError):
        FinalDatasetReviewPreviewV2.model_validate(
            {
                **preview.model_dump(mode="python"),
                "subject_count": preview.subject_count + 1,
            }
        )


def test_final_preview_rejects_scope_widening_and_hard_gate_override() -> None:
    prompt = _final_preview(FinalReviewScope.PROMPTS)

    with pytest.raises(ValidationError):
        FinalDatasetReviewPreviewV2.model_validate(
            {
                **prompt.model_dump(mode="python"),
                "selected_item_projection_refs": (_ref("user-item-projection", "forbidden"),),
            }
        )
    with pytest.raises(ValidationError):
        FinalDatasetReviewPreviewV2.model_validate(
            {
                **prompt.model_dump(mode="python"),
                "automated_hard_gates_passed": False,
            }
        )


@pytest.mark.parametrize(
    "unsafe_subject_type",
    ("rubric-set", "evaluator-spec", "reference-policy"),
)
def test_final_preview_rejects_control_subject_refs(
    unsafe_subject_type: str,
) -> None:
    with pytest.raises(ValidationError, match="subject"):
        FinalDatasetReviewPreviewV2.create(
            scope=FinalReviewScope.PROMPTS,
            subject_refs=(_ref(unsafe_subject_type, "control"),),
            prompt_projection_refs=(_ref("user-prompt-projection", "prompt"),),
            selected_item_projection_refs=(),
            dataset_projection_ref=None,
            quality_summary_refs=(_ref("batch-quality-report", "batch"),),
            open_low_severity_finding_refs=(),
            sample_navigation_refs=(),
            projection_policy_version="user-approval-projection/r7-04-v1",
            audit=_audit(),
        )


def test_final_preview_identity_rejects_audit_ref_tampering() -> None:
    preview = _final_preview(FinalReviewScope.PROMPTS)
    tampered = preview.model_copy(
        update={"audit": _audit(input_refs=(_ref("rubric-set", "forbidden-control"),))}
    )

    with pytest.raises(ValueError, match="audit"):
        validate_final_dataset_review_preview_v2_identity(tampered)


def test_request_identity_enforces_checkpoint_specific_ref_shapes() -> None:
    policy = _approval_policy()
    request = _request(policy_ref=user_approval_policy_ref(policy))
    mismatched = _reidentify_request(
        request,
        subject_refs=(_ref("selection-context", "rewrite"),),
        plan_ref=_ref("task-rewrite-plan-version", "rewrite"),
        preview_refs=(_ref("task-rewrite-plan-preview", "rewrite"),),
    )

    with pytest.raises(ValueError, match="LABEL_PLAN"):
        validate_user_approval_request_identity(mismatched)


@pytest.mark.parametrize(
    ("outcome", "checkpoint"),
    (
        (
            UserApprovalRequestCompilationOutcomeV2.DISABLED,
            ApprovalCheckpoint.LABEL_PLAN,
        ),
        (
            UserApprovalRequestCompilationOutcomeV2.NOT_REQUIRED,
            ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        ),
    ),
)
def test_empty_compilation_outcomes_carry_no_synthetic_request(
    outcome: UserApprovalRequestCompilationOutcomeV2,
    checkpoint: ApprovalCheckpoint,
) -> None:
    policy = _approval_policy()
    generation_policy = _generation_policy()
    job_ref = _ref("dataset-job-spec", "job")
    result = UserApprovalRequestCompilationResultV2.create(
        job_id="job://r7-04/contract",
        dataset_job_spec_ref=job_ref,
        approval_policy_ref=user_approval_policy_ref(policy),
        generation_policy_ref=approval_request_generation_policy_v2_ref(generation_policy),
        checkpoint=checkpoint,
        outcome=outcome,
        requests=(),
        request_preview_character_counts=(),
        audit=_audit(),
    )

    assert result.requests == ()
    assert result.request_refs == ()
    assert result.request_count == 0
    assert result.subject_ref_count == 0
    assert result.preview_ref_count == 0
    assert result.preview_character_count == 0
    validate_user_approval_request_compilation_result_v2_identity(result)


def test_requested_result_binds_nested_request_and_derived_counts() -> None:
    policy = _approval_policy()
    generation_policy = _generation_policy()
    request = _request(policy_ref=user_approval_policy_ref(policy))
    result = UserApprovalRequestCompilationResultV2.create(
        job_id="job://r7-04/contract",
        dataset_job_spec_ref=_ref("dataset-job-spec", "job"),
        approval_policy_ref=user_approval_policy_ref(policy),
        generation_policy_ref=approval_request_generation_policy_v2_ref(generation_policy),
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        outcome=UserApprovalRequestCompilationOutcomeV2.REQUESTED,
        requests=(request,),
        request_preview_character_counts=(123,),
        audit=_audit(),
    )

    assert result.request_refs == (user_approval_request_ref(request),)
    assert result.request_count == 1
    assert result.subject_ref_count == 1
    assert result.preview_ref_count == 1
    assert result.preview_character_count == 123
    validate_user_approval_request_compilation_result_v2_identity(result)
    assert user_approval_request_compilation_result_v2_ref(result).object_sha256 == result.result_sha256

    with pytest.raises(ValidationError):
        UserApprovalRequestCompilationResultV2.model_validate(
            {
                **result.model_dump(mode="python"),
                "request_preview_character_counts": (122,),
            }
        )


def test_scaffold_helpers_are_audit_independent_and_reject_stale_ids() -> None:
    policy_first = _approval_policy()
    policy_second = _approval_policy(
        audit=_audit(
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
            created_by="different-actor",
        )
    )
    assert user_approval_policy_ref(policy_first) == user_approval_policy_ref(policy_second)
    validate_user_approval_policy_identity(policy_first)
    policy_payload = policy_first.model_dump(mode="python")
    reparsed_policy = UserApprovalPolicy.model_validate(policy_payload)
    assert user_approval_policy_ref(reparsed_policy) == (user_approval_policy_ref(policy_first))
    unsafe_audit = policy_first.model_copy(
        update={"audit": _audit(input_refs=(_ref("private-reference", "forbidden"),))}
    )
    with pytest.raises(ValueError, match="audit input refs"):
        validate_user_approval_policy_identity(unsafe_audit)

    label_first = _label_plan()
    label_second = _label_plan(
        audit=_audit(
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
            created_by="different-actor",
            input_refs=(label_first.label_spec_ref,),
        )
    )
    assert label_plan_ref(label_first) == label_plan_ref(label_second)
    label_audit_tampered = label_first.model_copy(
        update={"audit": _audit(input_refs=(_ref("rubric-set", "forbidden-control"),))}
    )
    with pytest.raises(ValueError, match="audit"):
        label_plan_ref(label_audit_tampered)

    environment_first = _environment_strategy()
    environment_second = _environment_strategy(
        audit=_audit(
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
            created_by="different-actor",
            input_refs=tuple(
                ref
                for requirement in environment_first.requirements
                for ref in requirement.affected_subject_refs
            ),
        )
    )
    assert environment_strategy_ref(environment_first) == (environment_strategy_ref(environment_second))
    environment_audit_tampered = environment_first.model_copy(
        update={"audit": _audit(input_refs=(_ref("evaluator-spec", "forbidden-control"),))}
    )
    with pytest.raises(ValueError, match="audit"):
        environment_strategy_ref(environment_audit_tampered)

    request_first = _request(policy_ref=user_approval_policy_ref(policy_first))
    request_second = _request(
        policy_ref=user_approval_policy_ref(policy_first),
        audit=_audit(
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
            created_by="different-actor",
            input_refs=request_first.audit.input_refs,
        ),
    )
    assert user_approval_request_ref(request_first) == user_approval_request_ref(request_second)
    validate_user_approval_request_identity(request_first)

    stale = request_first.model_copy(update={"request_id": "user-approval-request://sha256/" + "f" * 64})
    with pytest.raises(ValueError, match="identity is stale"):
        validate_user_approval_request_identity(stale)


def test_label_plan_identity_preserves_model_path_order_and_normalizes_evidence() -> None:
    base = _label_plan()
    semantic_ref = _ref("semantic-residual-spec", "semantic")
    first_evidence = _evidence("a")
    second_evidence = _evidence("b")

    def identified_plan(
        *,
        model_path: tuple[str, ...],
        evidence_refs: tuple[EvidenceRef, ...],
    ) -> LabelPlan:
        first_example = base.examples[0].model_copy(update={"evidence_refs": evidence_refs})
        value = LabelPlan.model_validate(
            {
                **base.model_dump(mode="python"),
                "label_plan_id": "label-plan://pending",
                "semantic_clause_refs": (semantic_ref,),
                "examples": (first_example, *base.examples[1:]),
                "expected_model_path": model_path,
            }
        )
        digest = label_plan_carried_sha256(value)
        return value.model_copy(update={"label_plan_id": f"label-plan://sha256/{digest}"})

    ordered = identified_plan(
        model_path=("model-a", "model-b"),
        evidence_refs=(first_evidence, second_evidence),
    )
    evidence_reordered = identified_plan(
        model_path=("model-a", "model-b"),
        evidence_refs=(second_evidence, first_evidence),
    )
    path_reordered = identified_plan(
        model_path=("model-b", "model-a"),
        evidence_refs=(first_evidence, second_evidence),
    )

    assert label_plan_ref(ordered) == label_plan_ref(evidence_reordered)
    assert label_plan_ref(ordered) != label_plan_ref(path_reordered)


def test_new_contract_schemas_are_closed_and_content_free() -> None:
    def property_names(value: object) -> set[str]:
        if isinstance(value, dict):
            names = set(value.get("properties", {}))
            for child in value.values():
                names.update(property_names(child))
            return names
        if isinstance(value, list):
            names: set[str] = set()
            for child in value:
                names.update(property_names(child))
            return names
        return set()

    forbidden = {
        "raw_trace",
        "trace_text",
        "private_reference",
        "grader_rule",
        "hidden_condition",
        "evaluator_material",
        "attachment_bytes",
        "physical_path",
        "credential",
        "provider_payload",
        "reviewer_id",
        "quorum",
        "claim_lease",
        "user_decision_record",
    }
    for model in (
        ApprovalRequestGenerationPolicyV2,
        FinalDatasetReviewPreviewV2,
        UserApprovalRequestCompilationResultV2,
    ):
        schema = model.model_json_schema()
        assert schema["additionalProperties"] is False
        assert forbidden.isdisjoint(property_names(schema))
