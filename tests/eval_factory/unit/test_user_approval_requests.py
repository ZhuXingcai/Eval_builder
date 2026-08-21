from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_task_rewrite import _compile_plan, _source_chain

from eval_factory.approval.requests import (
    EnvironmentStrategyApprovalSource,
    FinalDatasetReviewApprovalSource,
    LabelPlanApprovalSource,
    TaskRewriteApprovalSource,
    UserApprovalPolicyCompiler,
    UserApprovalPolicyError,
    UserApprovalRequestCompiler,
)
from eval_factory.contracts.approval import (
    ALL_CHECKPOINTS,
    PLAN_CHECKPOINTS,
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
    UserDecision,
)
from eval_factory.contracts.approval_v2 import (
    ApprovalRequestGenerationPolicyV2,
    FinalDatasetReviewPreviewV2,
    UserApprovalRequestCompilationOutcomeV2,
    environment_strategy_carried_sha256,
    final_dataset_review_preview_v2_ref,
    label_plan_carried_sha256,
    user_approval_policy_ref,
    user_approval_request_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.labeling_v2 import (
    LabelSpecV2,
    PredicateOperatorV2,
    StructuredPredicateV2,
)
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    StageNameV2,
)
from eval_factory.contracts.task_v2 import (
    TaskPromptSafetyGateStatusV2,
    task_rewrite_plan_preview_ref,
    task_rewrite_plan_version_ref,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
HASH = "a" * 64
ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = ROOT / "evals/golden/eval_factory/approval" / "r7-04-checkpoint-requests-v1.json"


def _audit(
    *,
    created_at: datetime = NOW,
    created_by: str = "r7-04-request-test",
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    refs_by_key = {
        (
            ref.object_type,
            ref.object_id,
            ref.object_version,
            ref.object_sha256,
        ): ref
        for ref in input_refs
    }
    return ContractAudit(
        created_at=created_at,
        created_by=created_by,
        governing_versions=(
            VersionBinding(
                component="user-approval-request",
                version="r7-04-v1",
            ),
        ),
        input_refs=tuple(refs_by_key[key] for key in sorted(refs_by_key)),
    )


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


def _generation_policy(
    *,
    max_requests_per_checkpoint: int = 20,
    max_subject_refs_per_request: int = 20,
    max_preview_refs_per_request: int = 5,
    max_examples_per_plan: int = 20,
    max_preview_characters_per_request: int = 20_000,
    max_preview_characters_per_checkpoint: int = 100_000,
    max_final_sample_refs: int = 100,
) -> ApprovalRequestGenerationPolicyV2:
    return ApprovalRequestGenerationPolicyV2.create(
        max_requests_per_checkpoint=max_requests_per_checkpoint,
        max_subject_refs_per_request=max_subject_refs_per_request,
        max_preview_refs_per_request=max_preview_refs_per_request,
        max_examples_per_plan=max_examples_per_plan,
        max_preview_characters_per_request=(max_preview_characters_per_request),
        max_preview_characters_per_checkpoint=(max_preview_characters_per_checkpoint),
        max_final_sample_refs=max_final_sample_refs,
        audit=_audit(),
    )


def _policy(
    mode: ApprovalMode | None,
    *,
    custom: frozenset[ApprovalCheckpoint] = frozenset(),
    scope: FinalReviewScope = FinalReviewScope.NONE,
    explicit: bool = True,
):
    return UserApprovalPolicyCompiler().compile(
        mode=mode,
        custom_checkpoints=custom,
        final_review_scope=scope,
        explicit_choice=explicit,
        audit=_audit(),
    )


def _job_spec(policy) -> DatasetJobSpecV2:
    requested_stages = tuple(
        stage
        for stage in StageNameV2
        if stage
        not in {
            StageNameV2.LABEL_PLAN,
            StageNameV2.TASK_REWRITE_PLAN,
            StageNameV2.ENVIRONMENT_STRATEGY,
            StageNameV2.FINAL_DATASET_REVIEW,
        }
        or {
            StageNameV2.LABEL_PLAN: ApprovalCheckpoint.LABEL_PLAN,
            StageNameV2.TASK_REWRITE_PLAN: (ApprovalCheckpoint.TASK_REWRITE_PLAN),
            StageNameV2.ENVIRONMENT_STRATEGY: (ApprovalCheckpoint.ENVIRONMENT_STRATEGY),
            StageNameV2.FINAL_DATASET_REVIEW: (ApprovalCheckpoint.FINAL_DATASET_REVIEW),
        }[stage]
        in policy.enabled_checkpoints
    )
    return DatasetJobSpecV2(
        job_id="job://r7-04/requests",
        traces=(
            TraceSourceRef(
                source_trace_id="source-trace://r7-04/requests",
                source_uri="raw-traj://r7-04/requests",
                raw_sha256=HASH,
                adapter_name="raw-traj-v1",
                adapter_version="v1",
                processing_class="RESTRICTED_TRACE_RAW",
            ),
        ),
        requested_stages=requested_stages,
        privacy_profile="trusted-monitored-local",
        model_profiles=(),
        budget=ResourceBudget(
            max_model_requests=10,
            max_model_tokens=1000,
            max_processes=1,
            max_renderers=1,
            max_network_requests=1,
            max_storage_bytes=1024,
        ),
        concurrency=ConcurrencyLimit(
            model_requests=1,
            processes=1,
            renderers=1,
            network_requests=1,
            artifacts_per_item=1,
            items=1,
        ),
        selection_spec_ref=None,
        approval_policy_ref=user_approval_policy_ref(policy),
        approval_mode=policy.mode,
        enabled_checkpoints=policy.enabled_checkpoints,
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://canary",
        ),
        idempotency_key="job-r7-04-requests",
        audit=_audit(),
    )


def _label_source(suffix: str = "label") -> LabelPlanApprovalSource:
    predicate = StructuredPredicateV2(
        predicate_id=f"structured-predicate://r7-04/{suffix}",
        fact_type="tool-call-record",
        field_path="tool_family",
        operator=PredicateOperatorV2.EQUALS,
        expected_value="search",
        window_events=None,
        required_capability="tool_events",
        rule_version="structured-label/r3-02",
    )
    predicate_ref = ObjectRef(
        object_type="structured-predicate",
        object_id=predicate.predicate_id,
        object_version="v2",
        object_sha256=predicate.canonical_sha256(),
    )
    label_spec = LabelSpecV2(
        label_spec_id=f"label-spec://r7-04/{suffix}",
        label_version="v2",
        name=f"label-{suffix}",
        requirement="Detect a complete search-tool execution.",
        prerequisite_predicates=(),
        positive_predicates=(predicate,),
        negative_predicates=(),
        semantic_residual=None,
        decision_threshold=1.0,
        review_threshold=1.0,
        label_plan_ref=None,
        policy_version="labeling/r3-02-v1",
        label_spec_sha256=("b" * 64),
        audit=_audit(),
    )
    label_spec_ref = ObjectRef(
        object_type="label-spec",
        object_id=label_spec.label_spec_id,
        object_version=label_spec.label_version,
        object_sha256=label_spec.label_spec_sha256,
    )
    plan = LabelPlan(
        label_plan_id="label-plan://pending",
        label_spec_ref=label_spec_ref,
        intent="Detect a complete search-tool execution.",
        boundary="Use executed tool facts, not text mentions.",
        deterministic_clause_refs=(predicate_ref,),
        semantic_clause_refs=(),
        examples=tuple(
            PlanExample(
                example_id=(f"plan-example://r7-04/{suffix}/{kind.value.casefold()}"),
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
        abstain_rules=("Abstain on incomplete tool capability.",),
        expected_model_path=(),
        estimated_model_requests_per_trace=0,
        blind_spots=("Incomplete ranges cannot prove absence.",),
        audit=_audit(input_refs=(label_spec_ref,)),
    )
    digest = label_plan_carried_sha256(plan)
    plan = plan.model_copy(update={"label_plan_id": f"label-plan://sha256/{digest}"})
    return LabelPlanApprovalSource(
        label_spec=label_spec,
        label_plan=plan,
    )


def _environment_source() -> EnvironmentStrategyApprovalSource:
    task_refs = (
        _ref("task-draft", "environment-a"),
        _ref("task-draft", "environment-b"),
    )
    strategy = EnvironmentStrategy(
        environment_strategy_id="environment-strategy://pending",
        requirements=(
            EnvironmentRequirement(
                requirement_id="environment-requirement://r7-04/workspace",
                affected_subject_refs=task_refs,
                description="A non-standard workspace is required.",
                requires_special_account=False,
                requires_nonstandard_environment=True,
                evidence_refs=(),
                alternatives=tuple(
                    EnvironmentAlternative(
                        strategy=choice,
                        feasible=True,
                        capability_impact=f"Impact for {choice.value}.",
                    )
                    for choice in EnvironmentStrategyChoice
                ),
            ),
        ),
        query_packaging_options=frozenset(QueryPackagingChoice),
        recommended_strategy=EnvironmentStrategyChoice.REWRITE_STANDARD_ENV,
        recommendation_reason="Avoid private credentials.",
        credential_fabrication_forbidden=True,
        audit=_audit(input_refs=task_refs),
    )
    digest = environment_strategy_carried_sha256(strategy)
    strategy = strategy.model_copy(
        update={"environment_strategy_id": (f"environment-strategy://sha256/{digest}")}
    )
    return EnvironmentStrategyApprovalSource(strategy=strategy)


def _final_source(
    scope: FinalReviewScope,
    *,
    sample_count: int = 1,
) -> FinalDatasetReviewApprovalSource:
    prompt_refs = (
        (_ref("user-prompt-projection", "prompt"),)
        if scope in {FinalReviewScope.PROMPTS, FinalReviewScope.FULL_DATASET}
        else ()
    )
    item_refs = (
        (_ref("user-item-projection", "item"),)
        if scope
        in {
            FinalReviewScope.SELECTED_ITEMS,
            FinalReviewScope.FULL_DATASET,
        }
        else ()
    )
    dataset_ref = (
        _ref("user-dataset-projection", "dataset") if scope is FinalReviewScope.FULL_DATASET else None
    )
    preview = FinalDatasetReviewPreviewV2.create(
        scope=scope,
        subject_refs=(_ref("evaluation-item", "item"),),
        prompt_projection_refs=prompt_refs,
        selected_item_projection_refs=item_refs,
        dataset_projection_ref=dataset_ref,
        quality_summary_refs=(_ref("batch-quality-report", "batch"),),
        open_low_severity_finding_refs=(_ref("low-severity-finding-projection", "p2"),),
        sample_navigation_refs=tuple(
            _ref("approval-navigation-sample", str(index)) for index in range(sample_count)
        ),
        projection_policy_version="user-approval-projection/r7-04-v1",
        audit=_audit(),
    )
    return FinalDatasetReviewApprovalSource(preview=preview)


def _compile(
    *,
    policy,
    checkpoint: ApprovalCheckpoint,
    sources,
    generation_policy: ApprovalRequestGenerationPolicyV2 | None = None,
    requested_by: str = "requesting-user",
):
    return UserApprovalRequestCompiler().compile(
        job_spec=_job_spec(policy),
        approval_policy=policy,
        generation_policy=generation_policy or _generation_policy(),
        checkpoint=checkpoint,
        sources=sources,
        requested_by=requested_by,
        audit=_audit(),
    )


def test_policy_compiler_resolves_default_named_custom_and_none() -> None:
    default = _policy(None, explicit=False)
    assert default.mode is ApprovalMode.PLAN_GATES
    assert default.enabled_checkpoints == PLAN_CHECKPOINTS

    none = _policy(ApprovalMode.NONE, explicit=True)
    assert none.enabled_checkpoints == frozenset()

    all_checkpoints = _policy(
        ApprovalMode.PLAN_AND_FINAL,
        scope=FinalReviewScope.FULL_DATASET,
    )
    assert all_checkpoints.enabled_checkpoints == ALL_CHECKPOINTS

    custom = _policy(
        ApprovalMode.CUSTOM,
        custom=frozenset(
            {
                ApprovalCheckpoint.LABEL_PLAN,
                ApprovalCheckpoint.FINAL_DATASET_REVIEW,
            }
        ),
        scope=FinalReviewScope.PROMPTS,
    )
    assert custom.mode is ApprovalMode.CUSTOM
    assert custom.final_review_scope is FinalReviewScope.PROMPTS
    UserApprovalPolicyCompiler().validate_current(custom)

    with pytest.raises(UserApprovalPolicyError, match="NONE"):
        _policy(ApprovalMode.NONE, explicit=False)
    with pytest.raises(UserApprovalPolicyError, match="CUSTOM"):
        _policy(ApprovalMode.CUSTOM)


def test_disabled_and_environment_not_required_emit_no_request() -> None:
    none = _policy(ApprovalMode.NONE)
    disabled = _compile(
        policy=none,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(),
    )
    assert disabled.outcome is UserApprovalRequestCompilationOutcomeV2.DISABLED
    assert disabled.requests == ()

    plan_gates = _policy(ApprovalMode.PLAN_GATES)
    not_required = _compile(
        policy=plan_gates,
        checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        sources=(),
    )
    assert not_required.outcome is UserApprovalRequestCompilationOutcomeV2.NOT_REQUIRED
    assert not_required.requests == ()

    with pytest.raises(UserApprovalPolicyError, match="disabled"):
        _compile(
            policy=none,
            checkpoint=ApprovalCheckpoint.LABEL_PLAN,
            sources=(_label_source(),),
        )
    with pytest.raises(UserApprovalPolicyError, match="requires"):
        _compile(
            policy=plan_gates,
            checkpoint=ApprovalCheckpoint.LABEL_PLAN,
            sources=(),
        )


def test_label_plan_requests_are_current_bounded_and_canonical() -> None:
    policy = _policy(ApprovalMode.PLAN_GATES)
    first = _label_source("b")
    second = _label_source("a")

    result = _compile(
        policy=policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(first, second),
    )

    assert result.outcome is UserApprovalRequestCompilationOutcomeV2.REQUESTED
    assert result.request_count == 2
    assert tuple(request.request_id for request in result.requests) == tuple(
        sorted(request.request_id for request in result.requests)
    )
    assert all(
        request.available_decisions
        == frozenset(
            {
                "ACCEPT",
                "ADJUST",
                "REJECT",
                "REQUEST_MORE_EXAMPLES",
            }
        )
        for request in result.requests
    )
    assert all(request.affected_stages == ("label",) for request in result.requests)
    assert all(request.plan_ref == request.preview_refs[0] for request in result.requests)
    for request in result.requests:
        user_approval_request_ref(request)

    UserApprovalRequestCompiler().validate_current(
        result,
        job_spec=_job_spec(policy),
        approval_policy=policy,
        generation_policy=_generation_policy(),
        sources=(second, first),
        requested_by="requesting-user",
    )


def test_label_plan_rejects_stale_clause_and_example_budget() -> None:
    policy = _policy(ApprovalMode.PLAN_GATES)
    source = _label_source()
    stale = replace(
        source,
        label_plan=source.label_plan.model_copy(
            update={"deterministic_clause_refs": (_ref("structured-predicate", "stale"),)}
        ),
    )
    with pytest.raises(UserApprovalPolicyError, match="clause"):
        _compile(
            policy=policy,
            checkpoint=ApprovalCheckpoint.LABEL_PLAN,
            sources=(stale,),
        )
    with pytest.raises(UserApprovalPolicyError, match="example"):
        _compile(
            policy=policy,
            checkpoint=ApprovalCheckpoint.LABEL_PLAN,
            sources=(source,),
            generation_policy=_generation_policy(max_examples_per_plan=3),
        )


def test_label_and_environment_sources_reject_audit_ref_tampering() -> None:
    policy = _policy(ApprovalMode.PLAN_GATES)
    label = _label_source()
    label_tampered = replace(
        label,
        label_plan=label.label_plan.model_copy(
            update={"audit": _audit(input_refs=(_ref("rubric-set", "forbidden-control"),))}
        ),
    )
    with pytest.raises(UserApprovalPolicyError, match="audit"):
        _compile(
            policy=policy,
            checkpoint=ApprovalCheckpoint.LABEL_PLAN,
            sources=(label_tampered,),
        )

    environment = _environment_source()
    environment_tampered = replace(
        environment,
        strategy=environment.strategy.model_copy(
            update={"audit": _audit(input_refs=(_ref("evaluator-spec", "forbidden-control"),))}
        ),
    )
    with pytest.raises(UserApprovalPolicyError, match="audit"):
        _compile(
            policy=policy,
            checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
            sources=(environment_tampered,),
        )


def test_task_rewrite_request_uses_only_passed_safe_preview() -> None:
    policy = _policy(ApprovalMode.PLAN_GATES)
    chain = _source_chain()
    _, compiled = _compile_plan(chain)
    assert compiled.plan_version is not None
    assert compiled.preview_safety_gate is not None
    assert compiled.preview is not None
    source = TaskRewriteApprovalSource(
        plan_version=compiled.plan_version,
        preview_safety_gate=compiled.preview_safety_gate,
        preview=compiled.preview,
    )

    result = _compile(
        policy=policy,
        checkpoint=ApprovalCheckpoint.TASK_REWRITE_PLAN,
        sources=(source,),
    )

    request = result.requests[0]
    assert request.plan_ref == task_rewrite_plan_version_ref(compiled.plan_version)
    assert request.preview_refs == (task_rewrite_plan_preview_ref(compiled.preview),)
    assert request.affected_stages == ("task-authoring",)
    serialized = request.model_dump_json()
    assert "evidence-ref://" not in serialized
    assert "private-reference://" not in serialized
    assert "rubric-set://" not in serialized

    blocked_gate = compiled.preview_safety_gate.model_copy(
        update={"status": TaskPromptSafetyGateStatusV2.BLOCKED}
    )
    with pytest.raises(
        (ValidationError, UserApprovalPolicyError),
    ):
        _compile(
            policy=policy,
            checkpoint=ApprovalCheckpoint.TASK_REWRITE_PLAN,
            sources=(replace(source, preview_safety_gate=blocked_gate),),
        )


@pytest.mark.parametrize(
    "source_field",
    ("plan_version", "preview_safety_gate", "preview"),
)
def test_task_rewrite_request_rejects_audit_ref_tampering(
    source_field: str,
) -> None:
    policy = _policy(ApprovalMode.PLAN_GATES)
    _, compiled = _compile_plan(_source_chain())
    assert compiled.plan_version is not None
    assert compiled.preview_safety_gate is not None
    assert compiled.preview is not None
    source = TaskRewriteApprovalSource(
        plan_version=compiled.plan_version,
        preview_safety_gate=compiled.preview_safety_gate,
        preview=compiled.preview,
    )
    current = getattr(source, source_field)
    tampered_value = current.model_copy(
        update={
            "audit": current.audit.model_copy(
                update={"input_refs": (_ref("reference-policy", "forbidden-control"),)}
            )
        }
    )
    tampered_source = replace(source, **{source_field: tampered_value})

    with pytest.raises(UserApprovalPolicyError, match="audit"):
        _compile(
            policy=policy,
            checkpoint=ApprovalCheckpoint.TASK_REWRITE_PLAN,
            sources=(tampered_source,),
        )


def test_environment_request_binds_all_subjects_and_closed_choices() -> None:
    policy = _policy(ApprovalMode.PLAN_GATES)
    source = _environment_source()

    result = _compile(
        policy=policy,
        checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        sources=(source,),
    )

    request = result.requests[0]
    assert len(request.subject_refs) == 2
    assert request.affected_stages == ("attachment", "task-authoring")
    assert {
        alternative.strategy
        for requirement in source.strategy.requirements
        for alternative in requirement.alternatives
    } == set(EnvironmentStrategyChoice)
    assert source.strategy.query_packaging_options == frozenset(QueryPackagingChoice)
    assert source.strategy.credential_fabrication_forbidden is True


@pytest.mark.parametrize(
    "scope",
    (
        FinalReviewScope.PROMPTS,
        FinalReviewScope.SELECTED_ITEMS,
        FinalReviewScope.FULL_DATASET,
    ),
)
def test_final_review_request_matches_policy_scope(
    scope: FinalReviewScope,
) -> None:
    policy = _policy(ApprovalMode.FINAL_ONLY, scope=scope)
    source = _final_source(scope)

    result = _compile(
        policy=policy,
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        sources=(source,),
    )

    request = result.requests[0]
    assert request.plan_ref is None
    assert request.preview_refs == (final_dataset_review_preview_v2_ref(source.preview),)
    assert request.affected_stages == ("release",)
    assert UserDecision.DEFER in request.available_decisions

    wrong_scope = (
        FinalReviewScope.SELECTED_ITEMS
        if scope is not FinalReviewScope.SELECTED_ITEMS
        else FinalReviewScope.PROMPTS
    )
    with pytest.raises(UserApprovalPolicyError, match="scope"):
        _compile(
            policy=policy,
            checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
            sources=(_final_source(wrong_scope),),
        )


def test_request_budgets_reject_before_partial_result() -> None:
    plan_policy = _policy(ApprovalMode.PLAN_GATES)
    with pytest.raises(UserApprovalPolicyError, match="request limit"):
        _compile(
            policy=plan_policy,
            checkpoint=ApprovalCheckpoint.LABEL_PLAN,
            sources=(_label_source("a"), _label_source("b")),
            generation_policy=_generation_policy(max_requests_per_checkpoint=1),
        )
    with pytest.raises(UserApprovalPolicyError, match="subject"):
        _compile(
            policy=plan_policy,
            checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
            sources=(_environment_source(),),
            generation_policy=_generation_policy(max_subject_refs_per_request=1),
        )
    with pytest.raises(UserApprovalPolicyError, match="character"):
        _compile(
            policy=plan_policy,
            checkpoint=ApprovalCheckpoint.LABEL_PLAN,
            sources=(_label_source(),),
            generation_policy=_generation_policy(
                max_preview_characters_per_request=1,
                max_preview_characters_per_checkpoint=1,
            ),
        )
    final_policy = _policy(
        ApprovalMode.FINAL_ONLY,
        scope=FinalReviewScope.PROMPTS,
    )
    with pytest.raises(UserApprovalPolicyError, match="sample"):
        _compile(
            policy=final_policy,
            checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
            sources=(_final_source(FinalReviewScope.PROMPTS, sample_count=2),),
            generation_policy=_generation_policy(max_final_sample_refs=1),
        )


def test_request_identity_changes_with_requester_and_detects_stale_result() -> None:
    policy = _policy(ApprovalMode.PLAN_GATES)
    source = _label_source()
    first = _compile(
        policy=policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(source,),
        requested_by="requesting-user-a",
    )
    second = _compile(
        policy=policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(source,),
        requested_by="requesting-user-b",
    )

    assert first.result_id != second.result_id
    assert first.requests[0].projection_ref != second.requests[0].projection_ref

    stale = first.model_copy(update={"result_sha256": "f" * 64})
    with pytest.raises(UserApprovalPolicyError, match="current"):
        UserApprovalRequestCompiler().validate_current(
            stale,
            job_spec=_job_spec(policy),
            approval_policy=policy,
            generation_policy=_generation_policy(),
            sources=(source,),
            requested_by="requesting-user-a",
        )


def test_checkpoint_request_gold_is_complete_and_content_free() -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == ("eval-factory/checkpoint-request-gold/r7-04-v1")
    assert payload["generation_policy_version"] == ("user-approval-request/r7-04-v1")
    assert payload["stable_hash_seeds"] == [1, 321]
    assert payload["synthetic_decision_count"] == 0
    scenarios = {scenario["scenario_id"]: scenario for scenario in payload["scenarios"]}
    assert set(scenarios) == {
        "interactive-default",
        "explicit-none",
        "named-all",
        "custom-subset",
        "label-plan",
        "task-rewrite-plan",
        "environment-relevant",
        "environment-not-required",
        "final-prompts",
        "final-selected-items",
        "final-full-dataset",
        "disabled",
        "stale-source",
        "preflight-budget",
    }
    assert scenarios["disabled"]["request_count"] == 0
    assert scenarios["environment-not-required"]["request_count"] == 0
    serialized = json.dumps(payload, sort_keys=True).casefold()
    for forbidden in (
        "raw_trace",
        "trace_text",
        "private_reference",
        "grader_rule",
        "hidden_condition",
        "final_output",
        "attachment_bytes",
        "physical_path",
        "credential",
        "provider_payload",
        "reviewer_id",
        "quorum",
        "claim_lease",
    ):
        assert forbidden not in serialized
