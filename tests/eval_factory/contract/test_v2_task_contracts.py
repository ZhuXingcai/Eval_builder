from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts import (
    TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2,
    ContestantToolPolicyV2,
    ContestantToolRuleV2,
    EvaluatorBindingV2,
    EvaluatorExecutionModeV2,
    EvaluatorFailureRuleV2,
    EvaluatorFailureSignalV2,
    EvaluatorReferenceDataClassV2,
    EvaluatorReferenceGrantV2,
    EvaluatorSpecV2,
    ProducerAttachmentRequirementV2,
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
    PromptLeakageCategoryV2,
    PromptLeakageDetectorV2,
    PromptLeakageFingerprintV2,
    PromptLeakageMatchKindV2,
    PromptLeakageReferenceSetV2,
    R4TaskContractSetV2,
    ReferencePolicyV2,
    RubricCriterionV2,
    RubricJudgedObjectKindV2,
    RubricJudgedObjectV2,
    RubricReachabilityV2,
    RubricSetV2,
    RubricSourceModeV2,
    TaskContractInvalidationV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    TaskEpisodeSegmentEvidenceBindingV2,
    TaskEpisodeV2,
    TaskPromptSafetyCheckOutcomeV2,
    TaskPromptSafetyCheckV2,
    TaskPromptSafetyFindingV2,
    TaskPromptSafetyGateStatusV2,
    TaskPromptSafetyGateV2,
    TaskRequirementLineageV2,
    TaskRewriteApplicationV2,
    TaskRewriteExamplePreviewV2,
    TaskRewritePlanPreviewV2,
    TaskRewritePlanVersionV2,
    TaskRewritePreviewSafetyGateV2,
    TaskRewriteRebuildStageV2,
    ToolDenyReasonV2,
    ToolPolicyV2,
    ToolRuleActionV2,
    ToolRuleV2,
    contestant_tool_policy_carried_sha256,
    contestant_tool_policy_ref,
    contestant_tool_rule_carried_sha256,
    evaluator_binding_carried_sha256,
    evaluator_failure_rule_carried_sha256,
    evaluator_reference_grant_carried_sha256,
    evaluator_reference_grant_ref,
    evaluator_spec_carried_sha256,
    evaluator_spec_ref,
    producer_storage_authorization_carried_sha256,
    producer_storage_authorization_ref,
    producer_task_view_carried_sha256,
    producer_task_view_ref,
    prompt_leakage_reference_set_ref,
    r4_task_contract_set_carried_sha256,
    r4_task_contract_set_ref,
    reference_policy_carried_sha256,
    reference_policy_ref,
    rubric_criterion_carried_sha256,
    rubric_criterion_ref,
    rubric_reachability_carried_sha256,
    rubric_set_carried_sha256,
    rubric_set_ref,
    task_contract_invalidation_carried_sha256,
    task_contract_invalidation_ref,
    task_draft_carried_sha256,
    task_draft_ref,
    task_episode_ref,
    task_prompt_safety_gate_ref,
    task_rewrite_application_carried_sha256,
    task_rewrite_application_ref,
    task_rewrite_plan_carried_sha256,
    task_rewrite_plan_preview_carried_sha256,
    task_rewrite_plan_preview_ref,
    task_rewrite_plan_ref,
    task_rewrite_plan_version_carried_sha256,
    task_rewrite_plan_version_ref,
    task_rewrite_preview_safety_gate_carried_sha256,
    task_rewrite_preview_safety_gate_ref,
    tool_policy_carried_sha256,
    tool_policy_ref,
    tool_rule_carried_sha256,
)
from eval_factory.contracts.approval import (
    ExampleKind,
    PlanExample,
    RewriteFidelity,
    TaskRewritePlan,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.task import (
    AttachmentCriticality,
    AttachmentDependency,
    EvaluationFailureClass,
    EvidencePriority,
    ReferenceMode,
    RequirementConflict,
    RubricVisibility,
)
from eval_factory.contracts.trace import ToolFamily

HASH = "a" * 64
OTHER_HASH = "b" * 64
ROOT = Path(__file__).resolve().parents[3]


def _audit(created_at: datetime = datetime(2026, 7, 25, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="task-v2-contract-test",
        governing_versions=(VersionBinding(component="task-episode-v2", version="r4-01"),),
    )


def _ref(object_type: str, suffix: str, *, digest: str = HASH) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v2" if object_type == "selection-context" else "v1",
        object_sha256=digest,
    )


def _segment_ref(suffix: str) -> ObjectRef:
    return ObjectRef(
        object_type="interaction-segment",
        object_id=f"interaction-segment://{suffix}",
        object_version="deterministic-interaction-segments/v1",
        object_sha256=HASH,
    )


def _episode(**overrides: object) -> TaskEpisodeV2:
    first = _segment_ref("one")
    second = _segment_ref("two")
    values: dict[str, object] = {
        "task_episode_id": "task-episode://sha256/" + HASH,
        "selection_context_ref": _ref("selection-context", "candidate"),
        "trace_envelope_ref": _ref("trace-envelope", "trace"),
        "segment_refs": (first, second),
        "segment_evidence_bindings": (
            TaskEpisodeSegmentEvidenceBindingV2(
                segment_ref=first,
                evidence_ref_ids=("evidence-ref://one",),
            ),
            TaskEpisodeSegmentEvidenceBindingV2(
                segment_ref=second,
                evidence_ref_ids=("evidence-ref://two", "evidence-ref://shared"),
            ),
        ),
        "rationale_ref": _ref("task-episode-rationale", "candidate"),
        "evidence_bundle_ref": _ref("evidence-bundle", "candidate"),
        "model_profile": "internal-task-episode-grouper-v1",
        "prompt_version": "task-episode-grouping-prompt/v1",
        "policy_version": "task-episode-grouping/r4-01-v1",
        "task_episode_sha256": OTHER_HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    return TaskEpisodeV2(**values)


def test_task_episode_v2_accepts_exact_segment_evidence_lineage() -> None:
    episode = _episode()

    assert episode.schema_version == "eval-factory/task-episode/v2"
    assert tuple(binding.segment_ref for binding in episode.segment_evidence_bindings) == (
        episode.segment_refs
    )
    assert episode.selection_context_ref.object_type == "selection-context"
    assert episode.evidence_bundle_ref.object_type == "evidence-bundle"
    assert episode.task_episode_sha256 == OTHER_HASH
    assert task_episode_ref(episode) == ObjectRef(
        object_type="task-episode",
        object_id=episode.task_episode_id,
        object_version="v2",
        object_sha256=episode.task_episode_sha256,
    )


def test_task_episode_v2_rejects_wrong_ref_types_and_lineage_mismatch() -> None:
    with pytest.raises(ValidationError, match="selection_context_ref"):
        _episode(selection_context_ref=_ref("label-decision", "candidate"))

    with pytest.raises(ValidationError, match="trace_envelope_ref"):
        _episode(trace_envelope_ref=_ref("trace-event", "trace"))

    with pytest.raises(ValidationError, match="rationale_ref"):
        _episode(rationale_ref=_ref("final-output", "unsafe"))

    with pytest.raises(ValidationError, match="evidence_bundle_ref"):
        _episode(evidence_bundle_ref=_ref("projection-policy", "candidate"))

    first = _segment_ref("one")
    with pytest.raises(ValidationError, match="exactly match segment_refs"):
        _episode(
            segment_evidence_bindings=(
                TaskEpisodeSegmentEvidenceBindingV2(
                    segment_ref=first,
                    evidence_ref_ids=("evidence-ref://one",),
                ),
            )
        )


def test_task_episode_v2_rejects_duplicate_segments_and_evidence_ids() -> None:
    first = _segment_ref("one")

    with pytest.raises(ValidationError, match="segment_refs must be unique"):
        _episode(segment_refs=(first, first))
    with pytest.raises(ValidationError, match="segment_refs must be unique"):
        _episode(
            segment_refs=(
                first,
                first.model_copy(update={"object_sha256": OTHER_HASH}),
            )
        )

    with pytest.raises(ValidationError, match="evidence_ref_ids must be unique"):
        TaskEpisodeSegmentEvidenceBindingV2(
            segment_ref=first,
            evidence_ref_ids=("evidence-ref://one", "evidence-ref://one"),
        )


def test_task_episode_v2_is_closed_versioned_and_audit_is_not_the_carried_hash() -> None:
    first = _episode(audit=_audit(datetime(2026, 7, 25, tzinfo=UTC)))
    second = _episode(audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)))

    assert first.task_episode_id == second.task_episode_id
    assert first.task_episode_sha256 == second.task_episode_sha256
    assert first.canonical_sha256() != second.canonical_sha256()

    with pytest.raises(ValidationError):
        TaskEpisodeV2.model_validate(
            {
                **first.model_dump(mode="json"),
                "schema_version": "eval-factory/task-episode/v3",
            }
        )
    with pytest.raises(ValidationError):
        TaskEpisodeV2.model_validate(
            {
                **first.model_dump(mode="json"),
                "raw_trace_text": "deny",
            }
        )


def _evidence(suffix: str = "primary") -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://task-draft/{suffix}",
        subject_ref=_ref("file-version-projection", suffix),
        source_spans=(
            SourceSpanRef(
                span_id=f"source-span://task-draft/{suffix}",
                source_trace_id="source-trace://task-draft",
                raw_sha256=HASH,
            ),
        ),
        polarity=EvidencePolarity.POSITIVE,
        capability="task-draft-authoring",
        capability_complete=True,
    )


def _lineage(
    requirement_id: str = "requirement://task-draft/critical",
    *,
    criticality: AttachmentCriticality = AttachmentCriticality.CRITICAL,
    conflict_status: RequirementConflict = RequirementConflict.NONE,
) -> TaskRequirementLineageV2:
    return TaskRequirementLineageV2(
        requirement_id=requirement_id,
        statement="Inspect the safe workspace structure and explain the observed design state.",
        criticality=criticality,
        evidence_priority=EvidencePriority.DIRECT_OBSERVATION,
        evidence=(_evidence(requirement_id.rsplit("/", 1)[-1]),),
        task_episode_refs=(_ref("task-episode", "draft-primary"),),
        conflict_status=conflict_status,
    )


def _dependency() -> AttachmentDependency:
    return AttachmentDependency(
        dependency_id="attachment-dependency://task-draft/workspace",
        description="Input-state workspace required for the task.",
        criticality=AttachmentCriticality.REQUIRED,
        evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
        evidence=(_evidence("attachment"),),
    )


def _draft(**overrides: object) -> TaskDraftV2:
    values: dict[str, object] = {
        "task_draft_id": "task-draft://sha256/" + HASH,
        "task_version": 1,
        "supersedes_task_draft_ref": None,
        "selection_context_ref": _ref("selection-context", "draft"),
        "task_episode_refs": (_ref("task-episode", "draft-primary"),),
        "visible_prompt": "Inspect the safe workspace and explain the observed design state.",
        "task_intent": "Evaluate workspace design-state analysis.",
        "evaluation_claim": "The task is solvable from safe input-state evidence.",
        "required_capabilities": ("workspace-analysis",),
        "allowed_tools": ("file-read",),
        "forbidden_outputs": ("original final answer",),
        "attachment_dependencies": (_dependency(),),
        "requirement_lineage": (_lineage(),),
        "prompt_requirement_ids": ("requirement://task-draft/critical",),
        "uncertainties": (),
        "prompt_safety_status": TaskDraftPromptSafetyStatusV2.PENDING,
        "prompt_safety_gate_ref": None,
        "model_profile": "internal-task-author-v1",
        "prompt_version": "task-draft-authoring-prompt/v1",
        "policy_version": "task-draft-authoring/r4-03-v1",
        "task_draft_sha256": OTHER_HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    return TaskDraftV2(**values)


def test_task_draft_v2_accepts_exact_pending_lineage_and_carried_ref() -> None:
    draft = _draft()

    assert draft.schema_version == "eval-factory/task-draft/v2"
    assert draft.task_version == 1
    assert draft.prompt_safety_status is TaskDraftPromptSafetyStatusV2.PENDING
    assert draft.prompt_safety_gate_ref is None
    assert draft.requirement_lineage[0].task_episode_refs == draft.task_episode_refs
    assert task_draft_ref(draft) == ObjectRef(
        object_type="task-draft",
        object_id=draft.task_draft_id,
        object_version="v2",
        object_sha256=draft.task_draft_sha256,
    )


def test_task_draft_v2_enforces_version_ref_and_prompt_gate_shapes() -> None:
    with pytest.raises(ValidationError, match="version 1"):
        _draft(supersedes_task_draft_ref=_ref("task-draft", "prior"))
    with pytest.raises(ValidationError, match="later task versions"):
        _draft(task_version=2)
    accepted = _draft(
        task_version=2,
        supersedes_task_draft_ref=_ref("task-draft", "prior"),
    )
    assert accepted.task_version == 2

    with pytest.raises(ValidationError, match="selection_context_ref"):
        _draft(selection_context_ref=_ref("label-decision", "wrong"))
    with pytest.raises(ValidationError, match="task_episode_refs"):
        _draft(task_episode_refs=(_ref("interaction-segment", "wrong"),))
    with pytest.raises(ValidationError, match="PENDING"):
        _draft(prompt_safety_gate_ref=_ref("task-prompt-safety-gate", "gate"))
    with pytest.raises(ValidationError, match="requires prompt_safety_gate_ref"):
        _draft(prompt_safety_status=TaskDraftPromptSafetyStatusV2.PASSED)
    with pytest.raises(ValidationError, match="prompt_safety_gate_ref"):
        _draft(
            prompt_safety_status=TaskDraftPromptSafetyStatusV2.BLOCKED,
            prompt_safety_gate_ref=_ref("quality-report", "wrong"),
        )


def test_task_draft_v2_requires_complete_unique_critical_lineage() -> None:
    with pytest.raises(ValidationError, match="critical requirements"):
        _draft(
            requirement_lineage=(
                _lineage(),
                _lineage(
                    "requirement://task-draft/optional",
                    criticality=AttachmentCriticality.OPTIONAL,
                ),
            ),
            prompt_requirement_ids=("requirement://task-draft/optional",),
        )
    with pytest.raises(ValidationError, match="prompt requirement"):
        _draft(prompt_requirement_ids=("requirement://task-draft/critical",) * 2)
    with pytest.raises(ValidationError, match="requirement IDs"):
        _draft(requirement_lineage=(_lineage(), _lineage()))
    with pytest.raises(ValidationError, match="attachment dependency IDs"):
        _draft(attachment_dependencies=(_dependency(), _dependency()))
    with pytest.raises(ValidationError, match="unresolved requirement"):
        _draft(requirement_lineage=(_lineage(conflict_status=RequirementConflict.UNRESOLVED),))
    with pytest.raises(ValidationError, match="TaskDraft task episodes"):
        _draft(
            requirement_lineage=(
                TaskRequirementLineageV2(
                    **{
                        **_lineage().model_dump(mode="python"),
                        "task_episode_refs": (_ref("task-episode", "unknown"),),
                    }
                ),
            )
        )
    with pytest.raises(ValidationError, match="required_capabilities"):
        _draft(required_capabilities=("workspace-analysis", "workspace-analysis"))
    with pytest.raises(ValidationError, match="allowed_tools"):
        _draft(allowed_tools=("file-read", "file-read"))
    with pytest.raises(ValidationError, match="forbidden_outputs"):
        _draft(forbidden_outputs=("original final answer", "original final answer"))
    with pytest.raises(ValidationError, match="attachment evidence IDs"):
        _draft(
            attachment_dependencies=(
                _dependency().model_copy(
                    update={
                        "evidence": (
                            _evidence("attachment"),
                            _evidence("attachment"),
                        )
                    }
                ),
            )
        )
    with pytest.raises(ValidationError, match="unsafe requirement evidence"):
        unsafe = _evidence("unsafe").model_copy(
            update={
                "subject_ref": _ref("raw-trace", "unsafe"),
            }
        )
        _draft(
            requirement_lineage=(
                TaskRequirementLineageV2(
                    **{
                        **_lineage().model_dump(mode="python"),
                        "evidence": (unsafe,),
                    }
                ),
            )
        )


def test_task_draft_v2_is_closed_and_carried_hash_ignores_audit_time() -> None:
    first = _draft(audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)))
    second = _draft(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))

    assert first.task_draft_id == second.task_draft_id
    assert first.task_draft_sha256 == second.task_draft_sha256
    assert first.canonical_sha256() != second.canonical_sha256()

    with pytest.raises(ValidationError):
        TaskDraftV2.model_validate(
            {
                **first.model_dump(mode="json"),
                "schema_version": "eval-factory/task-draft/v3",
            }
        )
    with pytest.raises(ValidationError):
        TaskDraftV2.model_validate(
            {
                **first.model_dump(mode="json"),
                "raw_trace_text": "forbidden",
            }
        )


def _fingerprint(
    category: PromptLeakageCategoryV2 = PromptLeakageCategoryV2.FINAL_ANSWER,
    *,
    suffix: str = "primary",
    match_kind: PromptLeakageMatchKindV2 = PromptLeakageMatchKindV2.TOKEN_WINDOW,
    digest: str = HASH,
) -> PromptLeakageFingerprintV2:
    return PromptLeakageFingerprintV2(
        fingerprint_id=f"prompt-leakage-fingerprint://{suffix}",
        category=category,
        source_subject_ref=_ref("final-output", suffix, digest=digest),
        source_provenance_decision_ref=_ref("provenance-decision", suffix),
        classification_evidence_ref=_ref("safety-classification", suffix),
        match_kind=match_kind,
        digest_sha256=digest,
        token_count=4,
        normalization_version="task-prompt-leakage-normalization/r4-04-v1",
    )


def _reference_set(**overrides: object) -> PromptLeakageReferenceSetV2:
    values: dict[str, object] = {
        "reference_set_id": "prompt-leakage-reference-set://sha256/" + HASH,
        "trace_envelope_ref": _ref("trace-envelope", "prompt-safety"),
        "fingerprints": (_fingerprint(),),
        "complete_categories": TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2,
        "fingerprint_policy_version": "task-prompt-leakage-fingerprint/r4-04-v1",
        "reference_set_sha256": HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    return PromptLeakageReferenceSetV2(**values)


def _safety_finding(
    category: PromptLeakageCategoryV2 = PromptLeakageCategoryV2.FINAL_ANSWER,
    *,
    suffix: str = "primary",
    detector: PromptLeakageDetectorV2 = PromptLeakageDetectorV2.DETERMINISTIC_FINGERPRINT,
) -> TaskPromptSafetyFindingV2:
    return TaskPromptSafetyFindingV2(
        finding_id=f"task-prompt-safety-finding://{suffix}",
        category=category,
        detector=detector,
        prompt_span_start=8,
        prompt_span_end=24,
        matched_fingerprint_ids=(
            ("prompt-leakage-fingerprint://primary",)
            if detector is PromptLeakageDetectorV2.DETERMINISTIC_FINGERPRINT
            else ()
        ),
        rule_id=f"task-prompt-safety-rule://{suffix}",
        non_waivable=True,
    )


def _checks(
    *,
    failed_category: PromptLeakageCategoryV2 | None = None,
    not_evaluated: frozenset[PromptLeakageCategoryV2] = frozenset(),
) -> tuple[TaskPromptSafetyCheckV2, ...]:
    values: list[TaskPromptSafetyCheckV2] = []
    for category in sorted(TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2, key=lambda item: item.value):
        if category is failed_category:
            values.append(
                TaskPromptSafetyCheckV2(
                    category=category,
                    outcome=TaskPromptSafetyCheckOutcomeV2.FAILED,
                    finding_ids=("task-prompt-safety-finding://primary",),
                    detector_ids=("task-prompt-safety-detector://fingerprint",),
                )
            )
        elif category in not_evaluated:
            values.append(
                TaskPromptSafetyCheckV2(
                    category=category,
                    outcome=TaskPromptSafetyCheckOutcomeV2.NOT_EVALUATED,
                    finding_ids=(),
                    detector_ids=(),
                )
            )
        else:
            values.append(
                TaskPromptSafetyCheckV2(
                    category=category,
                    outcome=TaskPromptSafetyCheckOutcomeV2.PASSED,
                    finding_ids=(),
                    detector_ids=("task-prompt-safety-detector://semantic",),
                )
            )
    return tuple(values)


def _gate(**overrides: object) -> TaskPromptSafetyGateV2:
    values: dict[str, object] = {
        "task_prompt_safety_gate_id": "task-prompt-safety-gate://sha256/" + HASH,
        "source_task_draft_ref": _ref("task-draft", "pending"),
        "leakage_reference_set_ref": prompt_leakage_reference_set_ref(_reference_set()),
        "deterministic_scan_ref": _ref("task-prompt-deterministic-scan", "clean"),
        "semantic_assessment_ref": _ref("task-prompt-safety-assessment", "clear"),
        "visible_prompt_sha256": HASH,
        "status": TaskPromptSafetyGateStatusV2.PASSED,
        "checks": _checks(),
        "findings": (),
        "policy_version": "task-prompt-safety/r4-04-v1",
        "model_profile": "internal-task-prompt-safety-v1",
        "prompt_version": "task-prompt-safety-prompt/v1",
        "gate_sha256": OTHER_HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    return TaskPromptSafetyGateV2(**values)


def test_prompt_leakage_reference_set_is_content_free_strict_and_carried() -> None:
    reference_set = _reference_set()

    assert reference_set.complete_categories == TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2
    assert prompt_leakage_reference_set_ref(reference_set) == ObjectRef(
        object_type="prompt-leakage-reference-set",
        object_id=reference_set.reference_set_id,
        object_version="v2",
        object_sha256=reference_set.reference_set_sha256,
    )
    rendered = str(reference_set.model_dump(mode="json"))
    assert "source_text" not in rendered
    assert "logical_path" not in rendered

    with pytest.raises(ValidationError):
        PromptLeakageReferenceSetV2.model_validate(
            {
                **reference_set.model_dump(mode="json"),
                "raw_final_answer": "restricted",
            }
        )
    with pytest.raises(ValidationError, match="fingerprint IDs"):
        _reference_set(fingerprints=(_fingerprint(), _fingerprint()))
    with pytest.raises(ValidationError, match="trace_envelope_ref"):
        _reference_set(trace_envelope_ref=_ref("trace-event", "wrong"))


def test_prompt_safety_finding_and_check_shapes_fail_closed() -> None:
    with pytest.raises(ValidationError, match="ordered"):
        TaskPromptSafetyFindingV2(
            **{
                **_safety_finding().model_dump(mode="python"),
                "prompt_span_end": 8,
            }
        )
    with pytest.raises(ValidationError, match="non_waivable"):
        TaskPromptSafetyFindingV2.model_validate(
            {
                **_safety_finding().model_dump(mode="python"),
                "non_waivable": False,
            }
        )
    with pytest.raises(ValidationError, match="deterministic"):
        TaskPromptSafetyFindingV2(
            **{
                **_safety_finding().model_dump(mode="python"),
                "matched_fingerprint_ids": (),
            }
        )
    with pytest.raises(ValidationError, match="FAILED"):
        TaskPromptSafetyCheckV2(
            category=PromptLeakageCategoryV2.FINAL_ANSWER,
            outcome=TaskPromptSafetyCheckOutcomeV2.FAILED,
            finding_ids=(),
            detector_ids=("task-prompt-safety-detector://fingerprint",),
        )
    with pytest.raises(ValidationError, match="PASSED"):
        TaskPromptSafetyCheckV2(
            category=PromptLeakageCategoryV2.FINAL_ANSWER,
            outcome=TaskPromptSafetyCheckOutcomeV2.PASSED,
            finding_ids=("task-prompt-safety-finding://unexpected",),
            detector_ids=("task-prompt-safety-detector://semantic",),
        )


def test_prompt_safety_gate_enforces_complete_passed_and_blocked_inventories() -> None:
    passed = _gate()
    assert passed.status is TaskPromptSafetyGateStatusV2.PASSED
    assert task_prompt_safety_gate_ref(passed) == ObjectRef(
        object_type="task-prompt-safety-gate",
        object_id=passed.task_prompt_safety_gate_id,
        object_version="v2",
        object_sha256=passed.gate_sha256,
    )

    finding = _safety_finding()
    blocked = _gate(
        status=TaskPromptSafetyGateStatusV2.BLOCKED,
        semantic_assessment_ref=None,
        checks=_checks(
            failed_category=PromptLeakageCategoryV2.FINAL_ANSWER,
            not_evaluated=frozenset({PromptLeakageCategoryV2.TRAJECTORY_SPECIFIC_STEP}),
        ),
        findings=(finding,),
        model_profile=None,
        prompt_version=None,
    )
    assert blocked.status is TaskPromptSafetyGateStatusV2.BLOCKED

    with pytest.raises(ValidationError, match="exactly one check"):
        _gate(checks=_checks()[:-1])
    with pytest.raises(ValidationError, match="semantic assessment"):
        _gate(semantic_assessment_ref=None)
    with pytest.raises(ValidationError, match="cannot contain findings"):
        _gate(findings=(finding,))
    with pytest.raises(ValidationError, match="requires a failed check"):
        _gate(
            status=TaskPromptSafetyGateStatusV2.BLOCKED,
            findings=(finding,),
        )


def test_task_draft_carried_hash_recomputes_content_and_ignores_audit() -> None:
    first = _draft(audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)))
    second = _draft(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))
    tampered = first.model_copy(update={"visible_prompt": first.visible_prompt + " changed"})

    assert task_draft_carried_sha256(first) == task_draft_carried_sha256(second)
    assert task_draft_carried_sha256(first) != task_draft_carried_sha256(tampered)


def _rubric_reachability(**overrides: object) -> RubricReachabilityV2:
    values: dict[str, object] = {
        "reachability_id": "rubric-reachability://sha256/" + HASH,
        "prompt_requirement_ids": ("requirement://task-draft/critical",),
        "attachment_dependency_ids": ("attachment-dependency://task-draft/workspace",),
        "allowed_tool_ids": ("file-read",),
        "evidence": (_evidence("rubric-reachability"),),
        "capability_complete": True,
        "reachability_sha256": HASH,
    }
    values.update(overrides)
    return RubricReachabilityV2(**values)


def _rubric_criterion(
    *,
    suffix: str = "primary",
    weight: float = 1.0,
    **overrides: object,
) -> RubricCriterionV2:
    values: dict[str, object] = {
        "criterion_id": f"rubric-criterion://sha256/{suffix}",
        "judged_object": RubricJudgedObjectV2(
            judged_object_id=f"judged-object://{suffix}",
            kind=RubricJudgedObjectKindV2.WORKSPACE_STATE,
            description="The observable input workspace state.",
        ),
        "description": "Correctly determine the workspace design state from visible inputs.",
        "weight": weight,
        "reachability": _rubric_reachability(),
        "visibility": RubricVisibility.EVALUATOR_ONLY,
        "evaluator_binding": "evaluator-binding://workspace-state",
        "approval_status": "CANDIDATE",
        "criterion_sha256": OTHER_HASH,
    }
    values.update(overrides)
    return RubricCriterionV2(**values)


def _rubric_set(
    *,
    source_mode: RubricSourceModeV2 = RubricSourceModeV2.GENERATED,
    criteria: tuple[RubricCriterionV2, ...] | None = None,
    **overrides: object,
) -> RubricSetV2:
    resolved_criteria = criteria or (_rubric_criterion(),)
    values: dict[str, object] = {
        "rubric_set_id": "rubric-set://sha256/" + HASH,
        "rubric_version": 1,
        "supersedes_rubric_set_ref": None,
        "task_draft_ref": task_draft_ref(_draft()),
        "source_mode": source_mode,
        "criteria": resolved_criteria,
        "total_weight": sum(item.weight for item in resolved_criteria),
        "imported_from_ref": None,
        "model_profile": "internal-rubric-author-v1",
        "prompt_version": "rubric-authoring-prompt/v1",
        "policy_version": "rubric-authoring/r4-05-v1",
        "rubric_set_sha256": HASH,
        "audit": _audit(),
    }
    if source_mode is RubricSourceModeV2.IMPORTED:
        values.update(
            {
                "imported_from_ref": _ref("imported-rubric", "rubric-source"),
                "model_profile": None,
                "prompt_version": None,
            }
        )
    values.update(overrides)
    return RubricSetV2(**values)


def test_rubric_v2_contracts_bind_typed_judged_objects_and_reachability() -> None:
    criterion = _rubric_criterion()
    rubric_set = _rubric_set(criteria=(criterion,))

    assert criterion.judged_object.kind is RubricJudgedObjectKindV2.WORKSPACE_STATE
    assert criterion.approval_status == "CANDIDATE"
    assert criterion.reachability.capability_complete is True
    assert rubric_set.source_mode is RubricSourceModeV2.GENERATED
    assert rubric_criterion_ref(criterion) == ObjectRef(
        object_type="rubric-criterion",
        object_id=criterion.criterion_id,
        object_version="v2",
        object_sha256=criterion.criterion_sha256,
    )
    assert rubric_set_ref(rubric_set) == ObjectRef(
        object_type="rubric-set",
        object_id=rubric_set.rubric_set_id,
        object_version="v2",
        object_sha256=rubric_set.rubric_set_sha256,
    )


def test_rubric_reachability_requires_unique_prompt_lineage_and_complete_evidence() -> None:
    with pytest.raises(ValidationError):
        _rubric_reachability(prompt_requirement_ids=())
    with pytest.raises(ValidationError):
        _rubric_reachability(evidence=())
    with pytest.raises(ValidationError, match="prompt requirement IDs"):
        _rubric_reachability(
            prompt_requirement_ids=(
                "requirement://task-draft/critical",
                "requirement://task-draft/critical",
            )
        )
    with pytest.raises(ValidationError, match="evidence IDs"):
        evidence = _evidence("duplicate")
        _rubric_reachability(evidence=(evidence, evidence))
    with pytest.raises(ValidationError):
        RubricReachabilityV2.model_validate(
            {
                **_rubric_reachability().model_dump(mode="python"),
                "capability_complete": False,
            }
        )


def test_rubric_criterion_requires_positive_weight_and_candidate_only_approval() -> None:
    with pytest.raises(ValidationError):
        _rubric_criterion(weight=0)
    with pytest.raises(ValidationError):
        RubricCriterionV2.model_validate(
            {
                **_rubric_criterion().model_dump(mode="python"),
                "approval_status": "APPROVED",
            }
        )
    with pytest.raises(ValidationError, match="evaluator_binding"):
        _rubric_criterion(evaluator_binding="")


def test_rubric_set_enforces_source_mode_version_and_total_weight() -> None:
    generated = _rubric_set()
    imported = _rubric_set(source_mode=RubricSourceModeV2.IMPORTED)

    assert generated.imported_from_ref is None
    assert generated.model_profile is not None
    assert imported.imported_from_ref is not None
    assert imported.model_profile is None

    with pytest.raises(ValidationError, match="GENERATED"):
        _rubric_set(imported_from_ref=_ref("imported-rubric", "unexpected"))
    with pytest.raises(ValidationError, match="IMPORTED"):
        _rubric_set(
            source_mode=RubricSourceModeV2.IMPORTED,
            imported_from_ref=None,
        )
    with pytest.raises(ValidationError, match="total_weight"):
        _rubric_set(total_weight=2.0)
    with pytest.raises(ValidationError, match="TaskDraft v2"):
        _rubric_set(task_draft_ref=task_draft_ref(_draft()).model_copy(update={"object_version": "v1"}))
    with pytest.raises(ValidationError, match="version 1"):
        _rubric_set(
            supersedes_rubric_set_ref=_ref("rubric-set", "previous"),
        )
    with pytest.raises(ValidationError, match="predecessor"):
        _rubric_set(rubric_version=2)
    with pytest.raises(ValidationError, match="supersedes_rubric_set_ref"):
        _rubric_set(
            rubric_version=2,
            supersedes_rubric_set_ref=_ref("task-draft", "wrong"),
        )
    with pytest.raises(ValidationError, match="RubricSet v2"):
        _rubric_set(
            rubric_version=2,
            supersedes_rubric_set_ref=_ref("rubric-set", "previous"),
        )


def test_rubric_set_rejects_duplicate_criterion_and_judged_object_identities() -> None:
    first = _rubric_criterion(suffix="first")
    duplicate_criterion = _rubric_criterion(
        suffix="second",
        criterion_id=first.criterion_id,
    )
    with pytest.raises(ValidationError, match="criterion IDs"):
        _rubric_set(criteria=(first, duplicate_criterion))

    duplicate_object = _rubric_criterion(
        suffix="second",
        judged_object=first.judged_object,
    )
    with pytest.raises(ValidationError, match="judged object IDs"):
        _rubric_set(criteria=(first, duplicate_object))


def test_rubric_v2_contracts_are_closed_and_audit_independent() -> None:
    first = _rubric_set(audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)))
    second = _rubric_set(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))

    assert first.rubric_set_id == second.rubric_set_id
    assert first.rubric_set_sha256 == second.rubric_set_sha256
    assert first.canonical_sha256() != second.canonical_sha256()
    assert rubric_set_carried_sha256(first) == rubric_set_carried_sha256(second)
    criterion = first.criteria[0]
    assert rubric_criterion_carried_sha256(criterion) != rubric_criterion_carried_sha256(
        criterion.model_copy(update={"description": "Changed criterion."})
    )
    assert rubric_reachability_carried_sha256(criterion.reachability) != rubric_reachability_carried_sha256(
        criterion.reachability.model_copy(update={"allowed_tool_ids": ("different-tool",)})
    )

    with pytest.raises(ValidationError):
        RubricSetV2.model_validate(
            {
                **first.model_dump(mode="json"),
                "schema_version": "eval-factory/rubric-set/v3",
            }
        )
    with pytest.raises(ValidationError):
        RubricSetV2.model_validate(
            {
                **first.model_dump(mode="json"),
                "raw_trace_text": "forbidden",
            }
        )


_FAILURE_CLASS_BY_SIGNAL = {
    EvaluatorFailureSignalV2.CONTESTANT_OUTPUT_MISSING: EvaluationFailureClass.CONTESTANT_FAILURE,
    EvaluatorFailureSignalV2.CONTESTANT_OUTPUT_CONTRACT_VIOLATION: (
        EvaluationFailureClass.CONTESTANT_FAILURE
    ),
    EvaluatorFailureSignalV2.EVALUATOR_TIMEOUT: EvaluationFailureClass.EVALUATOR_FAILURE,
    EvaluatorFailureSignalV2.EVALUATOR_RUNTIME_FAILURE: EvaluationFailureClass.EVALUATOR_FAILURE,
    EvaluatorFailureSignalV2.EVALUATOR_OUTPUT_CONTRACT_VIOLATION: (EvaluationFailureClass.EVALUATOR_FAILURE),
    EvaluatorFailureSignalV2.ENVIRONMENT_STARTUP_FAILURE: EvaluationFailureClass.ENVIRONMENT_FAILURE,
    EvaluatorFailureSignalV2.ENVIRONMENT_RUNTIME_FAILURE: EvaluationFailureClass.ENVIRONMENT_FAILURE,
    EvaluatorFailureSignalV2.REQUIRED_ENVIRONMENT_CAPABILITY_MISSING: (
        EvaluationFailureClass.ENVIRONMENT_FAILURE
    ),
    EvaluatorFailureSignalV2.INSUFFICIENT_OBSERVATION: EvaluationFailureClass.INDETERMINATE,
    EvaluatorFailureSignalV2.CONFLICTING_FAILURE_SIGNALS: EvaluationFailureClass.INDETERMINATE,
}


def _v2_ref(object_type: str, suffix: str, digest: str = HASH) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v2",
        object_sha256=digest,
    )


def _failure_rules() -> tuple[EvaluatorFailureRuleV2, ...]:
    rules = []
    for signal, failure_class in _FAILURE_CLASS_BY_SIGNAL.items():
        rule = EvaluatorFailureRuleV2(
            rule_id=f"evaluator-failure-rule://{signal.value.casefold().replace('_', '-')}",
            signal=signal,
            failure_class=failure_class,
            rule_sha256=HASH,
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


def _evaluator_binding(
    *,
    binding_id: str = "evaluator-binding://workspace-state",
    execution_mode: EvaluatorExecutionModeV2 = EvaluatorExecutionModeV2.DETERMINISTIC,
    reference_refs: tuple[ObjectRef, ...] = (),
    principal_id: str | None = "principal://evaluator/workspace-state",
    model_profile_ref: ObjectRef | None = None,
    **overrides: object,
) -> EvaluatorBindingV2:
    if execution_mode is EvaluatorExecutionModeV2.MODEL and model_profile_ref is None:
        model_profile_ref = _ref("model-profile", "internal-evaluator")
    if execution_mode is EvaluatorExecutionModeV2.HUMAN_ONLY:
        principal_id = None
        model_profile_ref = None
    values: dict[str, object] = {
        "evaluator_binding_id": binding_id,
        "criterion_ids": ("rubric-criterion://sha256/primary",),
        "evaluator_type": "contract-evaluator",
        "execution_mode": execution_mode,
        "evaluator_version": "r4-06-v1",
        "input_contract_ref": _ref("evaluator-input-contract", "r4-06"),
        "output_contract_ref": _ref("evaluator-output-contract", "r4-06"),
        "evaluator_principal_id": principal_id,
        "model_profile_ref": model_profile_ref,
        "reference_refs": reference_refs,
        "timeout_seconds": 300,
        "binding_sha256": HASH,
    }
    values.update(overrides)
    binding = EvaluatorBindingV2(**values)
    return binding.model_copy(update={"binding_sha256": evaluator_binding_carried_sha256(binding)})


def _evaluator_spec(
    *,
    bindings: tuple[EvaluatorBindingV2, ...] | None = None,
    audit: ContractAudit | None = None,
    **overrides: object,
) -> EvaluatorSpecV2:
    values: dict[str, object] = {
        "evaluator_spec_id": "evaluator-spec://sha256/" + HASH,
        "evaluator_spec_version": 1,
        "supersedes_evaluator_spec_ref": None,
        "rubric_set_ref": _v2_ref("rubric-set", "r4-06"),
        "bindings": bindings or (_evaluator_binding(),),
        "failure_rules": _failure_rules(),
        "policy_version": "evaluation-contract/r4-06-v1",
        "evaluator_spec_sha256": HASH,
        "audit": audit or _audit(),
    }
    values.update(overrides)
    spec = EvaluatorSpecV2(**values)
    digest = evaluator_spec_carried_sha256(spec)
    return spec.model_copy(
        update={
            "evaluator_spec_id": f"evaluator-spec://sha256/{digest}",
            "evaluator_spec_sha256": digest,
        }
    )


_REFERENCE_MODE_SHAPES = {
    ReferenceMode.NONE: (
        EvaluatorReferenceDataClassV2.NONE,
        (),
        False,
        False,
    ),
    ReferenceMode.STRUCTURED_EXPECTATIONS: (
        EvaluatorReferenceDataClassV2.RESTRICTED_EVAL_CONTROL,
        (_ref("structured-expectation", "r4-06"),),
        True,
        False,
    ),
    ReferenceMode.PRIVATE_ANSWER: (
        EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
        (_ref("private-reference", "r4-06"),),
        True,
        False,
    ),
    ReferenceMode.TRACE_BEHAVIOR: (
        EvaluatorReferenceDataClassV2.RESTRICTED_TRACE_SPAN,
        (_ref("trace-behavior-reference", "r4-06"),),
        True,
        False,
    ),
    ReferenceMode.HUMAN_ONLY: (
        EvaluatorReferenceDataClassV2.HUMAN_ONLY_REFERENCE,
        (_ref("human-only-reference", "r4-06"),),
        False,
        True,
    ),
}


def _reference_policy(
    *,
    evaluator_spec: EvaluatorSpecV2 | None = None,
    mode: ReferenceMode = ReferenceMode.NONE,
    audit: ContractAudit | None = None,
    **overrides: object,
) -> ReferencePolicyV2:
    data_class, refs, evaluator_access, human_only_access = _REFERENCE_MODE_SHAPES[mode]
    values: dict[str, object] = {
        "reference_policy_id": "reference-policy://sha256/" + HASH,
        "reference_policy_version": 1,
        "supersedes_reference_policy_ref": None,
        "evaluator_spec_ref": evaluator_spec_ref(evaluator_spec or _evaluator_spec()),
        "mode": mode,
        "reference_data_class": data_class,
        "reference_refs": refs,
        "contestant_access": False,
        "attachment_producer_access": False,
        "evaluator_access": evaluator_access,
        "human_only_access": human_only_access,
        "policy_version": "reference-policy/r4-06-v1",
        "reference_policy_sha256": HASH,
        "audit": audit or _audit(),
    }
    values.update(overrides)
    policy = ReferencePolicyV2(**values)
    digest = reference_policy_carried_sha256(policy)
    return policy.model_copy(
        update={
            "reference_policy_id": f"reference-policy://sha256/{digest}",
            "reference_policy_sha256": digest,
        }
    )


def _reference_grant(
    *,
    evaluator_spec: EvaluatorSpecV2,
    reference_policy: ReferencePolicyV2,
    model_profile_ref: ObjectRef | None = None,
    model_domain_approval_ref: ObjectRef | None = None,
    **overrides: object,
) -> EvaluatorReferenceGrantV2:
    values: dict[str, object] = {
        "grant_id": "evaluator-reference-grant://sha256/" + HASH,
        "evaluator_spec_ref": evaluator_spec_ref(evaluator_spec),
        "reference_policy_ref": reference_policy_ref(reference_policy),
        "evaluator_binding_id": evaluator_spec.bindings[0].evaluator_binding_id,
        "evaluator_principal_id": "principal://evaluator/workspace-state",
        "model_profile_ref": model_profile_ref,
        "model_domain_approval_ref": model_domain_approval_ref,
        "reference_refs": reference_policy.reference_refs,
        "purpose": "EVALUATION",
        "policy_version": "reference-access/r4-06-v1",
        "grant_sha256": HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    grant = EvaluatorReferenceGrantV2(**values)
    digest = evaluator_reference_grant_carried_sha256(grant)
    return grant.model_copy(
        update={
            "grant_id": f"evaluator-reference-grant://sha256/{digest}",
            "grant_sha256": digest,
        }
    )


def test_evaluator_v2_contracts_bind_rules_bindings_policy_and_grant() -> None:
    reference_ref = _ref("private-reference", "r4-06")
    binding = _evaluator_binding(
        execution_mode=EvaluatorExecutionModeV2.MODEL,
        reference_refs=(reference_ref,),
    )
    spec = _evaluator_spec(bindings=(binding,))
    policy = _reference_policy(
        evaluator_spec=spec,
        mode=ReferenceMode.PRIVATE_ANSWER,
        reference_refs=(reference_ref,),
    )
    grant = _reference_grant(
        evaluator_spec=spec,
        reference_policy=policy,
        model_profile_ref=binding.model_profile_ref,
        model_domain_approval_ref=_ref("model-domain-approval", "r4-06"),
    )

    assert set(item.failure_class for item in spec.failure_rules) == set(EvaluationFailureClass)
    assert binding.binding_sha256 == evaluator_binding_carried_sha256(binding)
    assert evaluator_spec_ref(spec).object_sha256 == spec.evaluator_spec_sha256
    assert reference_policy_ref(policy).object_sha256 == policy.reference_policy_sha256
    assert evaluator_reference_grant_ref(grant).object_sha256 == grant.grant_sha256
    assert grant.reference_refs == (reference_ref,)


def test_evaluator_binding_enforces_execution_mode_principal_and_model_profile() -> None:
    with pytest.raises(ValidationError, match="principal"):
        _evaluator_binding(principal_id=None)
    with pytest.raises(ValidationError, match="DETERMINISTIC"):
        _evaluator_binding(model_profile_ref=_ref("model-profile", "unexpected"))
    with pytest.raises(ValidationError, match="MODEL"):
        EvaluatorBindingV2.model_validate(
            {
                **_evaluator_binding(
                    execution_mode=EvaluatorExecutionModeV2.MODEL,
                ).model_dump(mode="python"),
                "model_profile_ref": None,
            }
        )
    with pytest.raises(ValidationError, match="HUMAN_ONLY"):
        _evaluator_binding(
            execution_mode=EvaluatorExecutionModeV2.HUMAN_ONLY,
            evaluator_principal_id="principal://unexpected",
        )
    with pytest.raises(ValidationError, match="criterion IDs"):
        _evaluator_binding(
            criterion_ids=(
                "rubric-criterion://sha256/primary",
                "rubric-criterion://sha256/primary",
            )
        )


def test_evaluator_spec_enforces_exact_rubric_rules_and_version_chain() -> None:
    with pytest.raises(ValidationError, match="RubricSet v2"):
        _evaluator_spec(rubric_set_ref=_ref("rubric-set", "r4-06"))
    with pytest.raises(ValidationError, match="binding IDs"):
        binding = _evaluator_binding()
        _evaluator_spec(bindings=(binding, binding))
    with pytest.raises(ValidationError, match="failure signals"):
        rules = _failure_rules()
        duplicate_signal = rules[0].model_copy(
            update={
                "rule_id": "evaluator-failure-rule://duplicate-signal",
                "rule_sha256": OTHER_HASH,
            }
        )
        _evaluator_spec(failure_rules=(*rules[:-1], duplicate_signal))
    with pytest.raises(ValidationError, match="version 1"):
        _evaluator_spec(
            supersedes_evaluator_spec_ref=_v2_ref("evaluator-spec", "previous"),
        )
    with pytest.raises(ValidationError, match="predecessor"):
        _evaluator_spec(evaluator_spec_version=2)
    with pytest.raises(ValidationError, match="EvaluatorSpec v2"):
        _evaluator_spec(
            evaluator_spec_version=2,
            supersedes_evaluator_spec_ref=_ref("evaluator-spec", "previous"),
        )


@pytest.mark.parametrize("mode", tuple(ReferenceMode))
def test_reference_policy_supports_every_mode_with_exact_shape(mode: ReferenceMode) -> None:
    policy = _reference_policy(mode=mode)
    data_class, refs, evaluator_access, human_only_access = _REFERENCE_MODE_SHAPES[mode]

    assert policy.reference_data_class is data_class
    assert policy.reference_refs == refs
    assert policy.evaluator_access is evaluator_access
    assert policy.human_only_access is human_only_access
    assert policy.contestant_access is False
    assert policy.attachment_producer_access is False


def test_reference_policy_rejects_mode_ref_access_and_version_mismatches() -> None:
    with pytest.raises(ValidationError, match="NONE"):
        _reference_policy(
            mode=ReferenceMode.NONE,
            reference_refs=(_ref("private-reference", "unexpected"),),
        )
    with pytest.raises(ValidationError, match="PRIVATE_ANSWER"):
        _reference_policy(
            mode=ReferenceMode.PRIVATE_ANSWER,
            reference_refs=(_ref("structured-expectation", "wrong"),),
        )
    with pytest.raises(ValidationError, match="evaluator_access"):
        _reference_policy(
            mode=ReferenceMode.PRIVATE_ANSWER,
            evaluator_access=False,
        )
    with pytest.raises(ValidationError, match="HUMAN_ONLY"):
        _reference_policy(
            mode=ReferenceMode.HUMAN_ONLY,
            human_only_access=False,
        )
    with pytest.raises(ValidationError, match="version 1"):
        _reference_policy(
            supersedes_reference_policy_ref=_v2_ref("reference-policy", "previous"),
        )
    with pytest.raises(ValidationError, match="predecessor"):
        _reference_policy(reference_policy_version=2)


def test_evaluator_reference_grant_requires_exact_refs_and_model_approval_pair() -> None:
    spec = _evaluator_spec(
        bindings=(
            _evaluator_binding(
                reference_refs=(_ref("structured-expectation", "r4-06"),),
            ),
        )
    )
    policy = _reference_policy(
        evaluator_spec=spec,
        mode=ReferenceMode.STRUCTURED_EXPECTATIONS,
    )
    with pytest.raises(ValidationError):
        _reference_grant(
            evaluator_spec=spec,
            reference_policy=policy,
            reference_refs=(),
        )
    with pytest.raises(ValidationError, match="paired"):
        _reference_grant(
            evaluator_spec=spec,
            reference_policy=policy,
            model_profile_ref=_ref("model-profile", "r4-06"),
        )
    with pytest.raises(ValidationError, match="EvaluatorSpec v2"):
        _reference_grant(
            evaluator_spec=spec,
            reference_policy=policy,
            evaluator_spec_ref=_ref("evaluator-spec", "wrong-version"),
        )


def test_evaluator_v2_carried_hashes_are_closed_and_audit_independent() -> None:
    first_spec = _evaluator_spec(audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)))
    second_spec = _evaluator_spec(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))
    first_policy = _reference_policy(
        evaluator_spec=first_spec,
        audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)),
    )
    second_policy = _reference_policy(
        evaluator_spec=second_spec,
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )

    assert evaluator_spec_carried_sha256(first_spec) == evaluator_spec_carried_sha256(second_spec)
    assert first_spec.canonical_sha256() != second_spec.canonical_sha256()
    assert reference_policy_carried_sha256(first_policy) == reference_policy_carried_sha256(second_policy)
    with pytest.raises(ValidationError):
        EvaluatorSpecV2.model_validate(
            {
                **first_spec.model_dump(mode="json"),
                "raw_private_reference_text": "forbidden",
            }
        )
    with pytest.raises(ValidationError):
        ReferencePolicyV2.model_validate(
            {
                **first_policy.model_dump(mode="json"),
                "schema_version": "eval-factory/reference-policy/v3",
            }
        )


def _tool_rule(
    *,
    tool_id: str = "file-read",
    action: ToolRuleActionV2 = ToolRuleActionV2.ALLOW,
    deny_reason: ToolDenyReasonV2 | None = None,
    **overrides: object,
) -> ToolRuleV2:
    values: dict[str, object] = {
        "rule_id": tool_id,
        "tool_id": tool_id,
        "tool_family": ToolFamily.FILE_READ,
        "action": action,
        "capability_ref": _ref("tool-capability", tool_id),
        "enforcement_profile_ref": _ref("tool-enforcement-profile", tool_id),
        "constraint_profile_ref": _ref("tool-constraint-profile", tool_id),
        "deny_reason": deny_reason,
        "rule_sha256": HASH,
    }
    values.update(overrides)
    rule = ToolRuleV2(**values)
    return rule.model_copy(update={"rule_sha256": tool_rule_carried_sha256(rule)})


def _contestant_tool_rule(
    *,
    tool_id: str = "file-read",
    **overrides: object,
) -> ContestantToolRuleV2:
    values: dict[str, object] = {
        "rule_id": tool_id,
        "tool_id": tool_id,
        "tool_family": ToolFamily.FILE_READ,
        "action": "ALLOW",
        "contestant_descriptor_ref": _ref("contestant-tool-descriptor", tool_id),
        "contestant_constraint_profile_ref": _ref(
            "contestant-tool-constraint-profile",
            tool_id,
        ),
        "rule_sha256": HASH,
    }
    values.update(overrides)
    rule = ContestantToolRuleV2(**values)
    return rule.model_copy(update={"rule_sha256": contestant_tool_rule_carried_sha256(rule)})


def _contestant_tool_policy(
    *,
    rules: tuple[ContestantToolRuleV2, ...] = (),
    audit: ContractAudit | None = None,
    **overrides: object,
) -> ContestantToolPolicyV2:
    values: dict[str, object] = {
        "contestant_tool_policy_id": "contestant-tool-policy://pending",
        "source_task_draft_sha256": HASH,
        "rules": rules,
        "default_action": "DENY",
        "policy_version": "tool-policy/r4-07-v1",
        "contestant_tool_policy_sha256": HASH,
        "audit": audit or _audit(),
    }
    values.update(overrides)
    policy = ContestantToolPolicyV2(**values)
    digest = contestant_tool_policy_carried_sha256(policy)
    return policy.model_copy(
        update={
            "contestant_tool_policy_id": (f"contestant-tool-policy://sha256/{digest}"),
            "contestant_tool_policy_sha256": digest,
        }
    )


def _tool_policy(
    *,
    rules: tuple[ToolRuleV2, ...] | None = None,
    projection: ContestantToolPolicyV2 | None = None,
    audit: ContractAudit | None = None,
    **overrides: object,
) -> ToolPolicyV2:
    projection = projection or _contestant_tool_policy(rules=(_contestant_tool_rule(),))
    values: dict[str, object] = {
        "tool_policy_id": "tool-policy://pending",
        "tool_policy_version": 1,
        "supersedes_tool_policy_ref": None,
        "task_draft_ref": _v2_ref("task-draft", "r4-07"),
        "rubric_set_ref": _v2_ref("rubric-set", "r4-07"),
        "evaluator_spec_ref": _v2_ref("evaluator-spec", "r4-07"),
        "tool_catalog_ref": _ref("tool-capability-catalog", "r4-07"),
        "control_boundary_enforcement_ref": _ref(
            "prompt-boundary-enforcement",
            "r4-07",
        ),
        "rules": rules or (_tool_rule(),),
        "default_action": "DENY",
        "contestant_projection_ref": contestant_tool_policy_ref(projection),
        "policy_version": "tool-policy/r4-07-v1",
        "tool_policy_sha256": HASH,
        "audit": audit or _audit(),
    }
    values.update(overrides)
    policy = ToolPolicyV2(**values)
    digest = tool_policy_carried_sha256(policy)
    return policy.model_copy(
        update={
            "tool_policy_id": f"tool-policy://sha256/{digest}",
            "tool_policy_sha256": digest,
        }
    )


def test_tool_rule_v2_enforces_action_reason_and_internal_ref_shapes() -> None:
    allowed = _tool_rule()
    denied = _tool_rule(
        action=ToolRuleActionV2.DENY,
        deny_reason=ToolDenyReasonV2.NOT_REQUIRED,
    )

    assert allowed.deny_reason is None
    assert denied.deny_reason is ToolDenyReasonV2.NOT_REQUIRED
    assert allowed.rule_sha256 == tool_rule_carried_sha256(allowed)

    with pytest.raises(ValidationError, match="deny_reason"):
        _tool_rule(deny_reason=ToolDenyReasonV2.NOT_REQUIRED)
    with pytest.raises(ValidationError, match="deny_reason"):
        _tool_rule(action=ToolRuleActionV2.DENY)
    with pytest.raises(ValidationError, match="capability_ref"):
        _tool_rule(capability_ref=_ref("contestant-tool-descriptor", "wrong"))


def test_contestant_tool_contracts_are_allow_only_and_public_only() -> None:
    rule = _contestant_tool_rule()
    projection = _contestant_tool_policy(rules=(rule,))

    assert projection.rules == (rule,)
    assert projection.default_action == "DENY"
    assert contestant_tool_policy_ref(projection).object_version == "v2"
    serialized = str(projection.model_dump(mode="json"))
    assert "tool-enforcement-profile" not in serialized
    assert "deny_reason" not in serialized
    assert rule.contestant_descriptor_ref.object_type == "contestant-tool-descriptor"
    assert rule.contestant_constraint_profile_ref.object_type == "contestant-tool-constraint-profile"

    with pytest.raises(ValidationError):
        ContestantToolRuleV2.model_validate(
            {
                **rule.model_dump(mode="json"),
                "action": "DENY",
            }
        )
    with pytest.raises(ValidationError):
        ContestantToolRuleV2.model_validate(
            {
                **rule.model_dump(mode="json"),
                "credential": "forbidden",
            }
        )
    with pytest.raises(ValidationError, match="contestant_descriptor_ref"):
        _contestant_tool_rule(contestant_descriptor_ref=_ref("tool-enforcement-profile", "wrong"))


def test_tool_policy_v2_binds_exact_chain_rules_projection_and_version() -> None:
    projection = _contestant_tool_policy(rules=(_contestant_tool_rule(),))
    policy = _tool_policy(projection=projection)

    assert policy.default_action == "DENY"
    assert policy.contestant_projection_ref == contestant_tool_policy_ref(projection)
    assert tool_policy_ref(policy).object_sha256 == policy.tool_policy_sha256
    assert policy.tool_policy_sha256 == tool_policy_carried_sha256(policy)

    with pytest.raises(ValidationError, match="TaskDraft v2"):
        _tool_policy(task_draft_ref=_ref("task-draft", "wrong-version"))
    with pytest.raises(ValidationError, match="version 1"):
        _tool_policy(supersedes_tool_policy_ref=_v2_ref("tool-policy", "previous"))
    with pytest.raises(ValidationError, match="predecessor"):
        _tool_policy(tool_policy_version=2)
    with pytest.raises(ValidationError, match="ToolPolicy v2"):
        _tool_policy(
            tool_policy_version=2,
            supersedes_tool_policy_ref=_ref("tool-policy", "previous"),
        )


def test_tool_policy_v2_rejects_duplicate_rules_and_is_closed() -> None:
    rule = _tool_rule()
    with pytest.raises(ValidationError, match="rule IDs"):
        _tool_policy(rules=(rule, rule))
    with pytest.raises(ValidationError, match="tool IDs"):
        _tool_policy(
            rules=(
                rule,
                _tool_rule(
                    tool_id="file-read",
                    rule_id="second-file-read-rule",
                    capability_ref=_ref(
                        "tool-capability",
                        "file-read-second",
                        digest=OTHER_HASH,
                    ),
                ),
            )
        )

    policy = _tool_policy()
    with pytest.raises(ValidationError):
        ToolPolicyV2.model_validate(
            {
                **policy.model_dump(mode="json"),
                "runtime_credentials": "forbidden",
            }
        )
    with pytest.raises(ValidationError):
        ToolPolicyV2.model_validate(
            {
                **policy.model_dump(mode="json"),
                "schema_version": "eval-factory/tool-policy/v3",
            }
        )


def test_tool_policy_v2_carried_hashes_ignore_audit_and_bind_behavior() -> None:
    first_projection = _contestant_tool_policy(
        rules=(_contestant_tool_rule(),),
        audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)),
    )
    second_projection = _contestant_tool_policy(
        rules=(_contestant_tool_rule(),),
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    first = _tool_policy(
        projection=first_projection,
        audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)),
    )
    second = _tool_policy(
        projection=second_projection,
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )

    assert first_projection.contestant_tool_policy_sha256 == (second_projection.contestant_tool_policy_sha256)
    assert first.tool_policy_sha256 == second.tool_policy_sha256
    assert first.canonical_sha256() != second.canonical_sha256()
    assert tool_policy_carried_sha256(first) != tool_policy_carried_sha256(
        first.model_copy(update={"default_action": "ALLOW"})
    )


def _producer_requirement(
    *,
    dependency_id: str = "attachment-dependency://producer/workspace",
    **overrides: object,
) -> ProducerAttachmentRequirementV2:
    values: dict[str, object] = {
        "dependency_id": dependency_id,
        "description": "The input-state workspace required by the visible task.",
        "criticality": AttachmentCriticality.REQUIRED,
    }
    values.update(overrides)
    return ProducerAttachmentRequirementV2(**values)


def _producer_storage_authorization(
    *,
    audit: ContractAudit | None = None,
    **overrides: object,
) -> ProducerStorageAuthorizationV2:
    values: dict[str, object] = {
        "authorization_id": "producer-storage-authorization://pending",
        "producer_principal_id": "principal://attachment-producer/r4-08",
        "purpose": "ATTACHMENT_PRODUCTION",
        "source_task_draft_sha256": HASH,
        "projection_policy_ref": _ref("projection-policy", "producer"),
        "evidence_bundle_refs": (_ref("evidence-bundle", "producer"),),
        "authorized_subject_refs": (_ref("file-version-projection", "producer-input"),),
        "raw_store_access": False,
        "canonical_store_access": False,
        "quarantine_store_access": False,
        "private_reference_store_access": False,
        "credentials_issued": False,
        "policy_version": "producer-task-view/r4-08-v1",
        "authorization_sha256": HASH,
        "audit": audit or _audit(),
    }
    values.update(overrides)
    authorization = ProducerStorageAuthorizationV2(**values)
    digest = producer_storage_authorization_carried_sha256(authorization)
    return authorization.model_copy(
        update={
            "authorization_id": (f"producer-storage-authorization://sha256/{digest}"),
            "authorization_sha256": digest,
        }
    )


def _producer_task_view(
    *,
    authorization: ProducerStorageAuthorizationV2 | None = None,
    audit: ContractAudit | None = None,
    **overrides: object,
) -> ProducerTaskViewV2:
    authorization = authorization or _producer_storage_authorization()
    values: dict[str, object] = {
        "producer_task_view_id": "producer-task-view://pending",
        "producer_task_view_version": 1,
        "supersedes_producer_task_view_ref": None,
        "query_instruction": ("Inspect the input-state workspace using only explicitly allowed tools."),
        "attachment_requirements": (_producer_requirement(),),
        "allowed_tools": ("file-read",),
        "safe_evidence_bundle_refs": (_ref("evidence-bundle", "producer"),),
        "forbidden_outputs": (
            "original final answer",
            "private grader controls",
        ),
        "projection_policy_ref": _ref("projection-policy", "producer"),
        "storage_authorization_ref": producer_storage_authorization_ref(authorization),
        "prompt_boundary_enforcement_ref": _ref(
            "prompt-boundary-enforcement",
            "producer",
        ),
        "source_task_draft_sha256": HASH,
        "source_contestant_tool_policy_sha256": OTHER_HASH,
        "source_contract_chain_sha256": "c" * 64,
        "policy_version": "producer-task-view/r4-08-v1",
        "producer_task_view_sha256": HASH,
        "audit": audit or _audit(),
    }
    values.update(overrides)
    view = ProducerTaskViewV2(**values)
    digest = producer_task_view_carried_sha256(view)
    return view.model_copy(
        update={
            "producer_task_view_id": f"producer-task-view://sha256/{digest}",
            "producer_task_view_sha256": digest,
        }
    )


def test_producer_attachment_requirement_v2_is_stripped_and_closed() -> None:
    requirement = _producer_requirement()

    assert set(ProducerAttachmentRequirementV2.model_fields) == {
        "schema_version",
        "dependency_id",
        "description",
        "criticality",
    }
    for forbidden_field in (
        "evidence",
        "evidence_priority",
        "source_spans",
        "task_episode_refs",
        "logical_path",
        "command",
        "credential",
    ):
        with pytest.raises(ValidationError):
            ProducerAttachmentRequirementV2.model_validate(
                {
                    **requirement.model_dump(mode="json"),
                    forbidden_field: "forbidden",
                }
            )


def test_producer_storage_authorization_is_ref_only_and_deny_by_default() -> None:
    authorization = _producer_storage_authorization()

    assert authorization.raw_store_access is False
    assert authorization.canonical_store_access is False
    assert authorization.quarantine_store_access is False
    assert authorization.private_reference_store_access is False
    assert authorization.credentials_issued is False
    assert authorization.authorization_sha256 == (
        producer_storage_authorization_carried_sha256(authorization)
    )
    assert producer_storage_authorization_ref(authorization).object_version == "v2"

    for field_name in (
        "raw_store_access",
        "canonical_store_access",
        "quarantine_store_access",
        "private_reference_store_access",
        "credentials_issued",
    ):
        with pytest.raises(ValidationError):
            ProducerStorageAuthorizationV2.model_validate(
                {
                    **authorization.model_dump(mode="json"),
                    field_name: True,
                }
            )
    with pytest.raises(ValidationError, match="projection_policy_ref"):
        _producer_storage_authorization(projection_policy_ref=_ref("private-reference", "wrong"))
    with pytest.raises(ValidationError):
        ProducerStorageAuthorizationV2.model_validate(
            {
                **authorization.model_dump(mode="json"),
                "store_endpoint": "restricted://store",
            }
        )


def test_producer_storage_authorization_enforces_unique_sorted_scope() -> None:
    first = _ref("evidence-bundle", "a")
    second = _ref("evidence-bundle", "b", digest=OTHER_HASH)
    subject_one = _ref("file-version-projection", "a")
    subject_two = _ref(
        "file-version-projection",
        "b",
        digest=OTHER_HASH,
    )

    with pytest.raises(ValidationError, match="bundle refs"):
        _producer_storage_authorization(
            evidence_bundle_refs=(first, first),
        )
    with pytest.raises(ValidationError, match="subject refs"):
        _producer_storage_authorization(
            authorized_subject_refs=(subject_one, subject_one),
        )
    with pytest.raises(ValidationError, match="sorted"):
        _producer_storage_authorization(
            evidence_bundle_refs=(second, first),
        )
    with pytest.raises(ValidationError, match="sorted"):
        _producer_storage_authorization(
            authorized_subject_refs=(subject_two, subject_one),
        )


def test_producer_task_view_v2_binds_visible_fields_and_version_chain() -> None:
    authorization = _producer_storage_authorization()
    view = _producer_task_view(authorization=authorization)

    assert view.storage_authorization_ref == (producer_storage_authorization_ref(authorization))
    assert view.producer_task_view_sha256 == producer_task_view_carried_sha256(view)
    assert producer_task_view_ref(view).object_version == "v2"

    with pytest.raises(ValidationError, match="version 1"):
        _producer_task_view(
            supersedes_producer_task_view_ref=_v2_ref(
                "producer-task-view",
                "previous",
            )
        )
    with pytest.raises(ValidationError, match="predecessor"):
        _producer_task_view(producer_task_view_version=2)
    with pytest.raises(ValidationError, match="ProducerTaskView v2"):
        _producer_task_view(
            producer_task_view_version=2,
            supersedes_producer_task_view_ref=_ref(
                "producer-task-view",
                "previous",
            ),
        )


def test_producer_task_view_v2_rejects_duplicate_or_internal_fields() -> None:
    requirement = _producer_requirement()
    view = _producer_task_view()

    with pytest.raises(ValidationError, match="attachment requirement"):
        _producer_task_view(
            attachment_requirements=(requirement, requirement),
        )
    with pytest.raises(ValidationError, match="allowed tool"):
        _producer_task_view(allowed_tools=("file-read", "file-read"))
    with pytest.raises(ValidationError, match="evidence bundle"):
        bundle = _ref("evidence-bundle", "producer")
        _producer_task_view(safe_evidence_bundle_refs=(bundle, bundle))
    with pytest.raises(ValidationError, match="forbidden output"):
        _producer_task_view(forbidden_outputs=("original final answer", "original final answer"))

    for forbidden_field in (
        "task_intent",
        "evaluation_claim",
        "requirement_lineage",
        "rubric_set_ref",
        "evaluator_spec_ref",
        "reference_policy_ref",
        "tool_policy_ref",
        "storage_credential",
        "runtime_config",
    ):
        with pytest.raises(ValidationError):
            ProducerTaskViewV2.model_validate(
                {
                    **view.model_dump(mode="json"),
                    forbidden_field: "forbidden",
                }
            )


def test_producer_contract_carried_hashes_ignore_audit_and_bind_behavior() -> None:
    first_authorization = _producer_storage_authorization(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))
    second_authorization = _producer_storage_authorization(audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)))
    first = _producer_task_view(
        authorization=first_authorization,
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    second = _producer_task_view(
        authorization=second_authorization,
        audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)),
    )

    assert first_authorization.authorization_sha256 == (second_authorization.authorization_sha256)
    assert first.producer_task_view_sha256 == second.producer_task_view_sha256
    assert first.canonical_sha256() != second.canonical_sha256()
    assert producer_task_view_carried_sha256(first) != (
        producer_task_view_carried_sha256(first.model_copy(update={"allowed_tools": ()}))
    )


def _rewrite_plan(**overrides: object) -> TaskRewritePlan:
    values: dict[str, object] = {
        "task_rewrite_plan_id": "task-rewrite-plan://sha256/" + HASH,
        "selection_context_ref": _v2_ref("selection-context", "rewrite"),
        "target_capability": "Evaluate workspace design-state analysis.",
        "rewrite_style": "Concise, self-contained engineering task.",
        "fidelity": RewriteFidelity.CAPABILITY_PRESERVING,
        "operational_noise_policy": "Remove retries and runtime chatter.",
        "examples": (
            PlanExample(
                example_id="plan-example://rewrite/primary",
                kind=ExampleKind.REWRITE,
                input_summary="A multi-turn workspace analysis request.",
                expected_treatment="Inspect the workspace and explain its design state.",
                evidence_refs=(_evidence("rewrite-plan"),),
            ),
        ),
        "forbidden_content_rules": (
            "original final answer",
            "private reference",
        ),
        "expected_capability_impact": "Preserve analysis capability.",
        "audit": _audit(),
    }
    values.update(overrides)
    return TaskRewritePlan(**values)


def _r4_contract_set(**overrides: object) -> R4TaskContractSetV2:
    values: dict[str, object] = {
        "contract_set_id": "r4-task-contract-set://sha256/" + HASH,
        "task_draft_ref": _v2_ref("task-draft", "rewrite-source"),
        "task_prompt_safety_gate_ref": _v2_ref(
            "task-prompt-safety-gate",
            "rewrite-source",
        ),
        "rubric_set_ref": _v2_ref("rubric-set", "rewrite-source"),
        "evaluator_spec_ref": _v2_ref("evaluator-spec", "rewrite-source"),
        "reference_policy_ref": _v2_ref("reference-policy", "rewrite-source"),
        "tool_policy_ref": _v2_ref("tool-policy", "rewrite-source"),
        "contestant_tool_policy_ref": _v2_ref(
            "contestant-tool-policy",
            "rewrite-source",
        ),
        "producer_storage_authorization_ref": _v2_ref(
            "producer-storage-authorization",
            "rewrite-source",
        ),
        "producer_task_view_ref": _v2_ref(
            "producer-task-view",
            "rewrite-source",
        ),
        "contract_set_sha256": HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    return R4TaskContractSetV2(**values)


def _rewrite_plan_version(**overrides: object) -> TaskRewritePlanVersionV2:
    values: dict[str, object] = {
        "task_rewrite_plan_version_id": ("task-rewrite-plan-version://sha256/" + HASH),
        "plan_version": 1,
        "supersedes_task_rewrite_plan_version_ref": None,
        "plan": _rewrite_plan(),
        "basis_contract_set_ref": r4_task_contract_set_ref(_r4_contract_set()),
        "policy_version": "task-rewrite/r4-09-v1",
        "task_rewrite_plan_version_sha256": HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    return TaskRewritePlanVersionV2(**values)


def _rewrite_preview_gate(**overrides: object) -> TaskRewritePreviewSafetyGateV2:
    values: dict[str, object] = {
        "gate_id": "task-rewrite-preview-safety-gate://sha256/" + HASH,
        "preview_candidate_ref": _v2_ref(
            "task-rewrite-preview-candidate",
            "safe",
        ),
        "leakage_reference_set_ref": _v2_ref(
            "prompt-leakage-reference-set",
            "rewrite",
        ),
        "preview_content_sha256": HASH,
        "status": TaskPromptSafetyGateStatusV2.PASSED,
        "checks": _checks(),
        "findings": (),
        "semantic_assessment_ref": _v2_ref(
            "task-prompt-safety-assessment",
            "rewrite",
        ),
        "model_profile": "internal-task-rewrite-safety-v1",
        "prompt_version": "task-rewrite-safety-prompt/v1",
        "policy_version": "task-rewrite/r4-09-v1",
        "gate_sha256": HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    return TaskRewritePreviewSafetyGateV2(**values)


def _rewrite_preview(**overrides: object) -> TaskRewritePlanPreviewV2:
    gate = _rewrite_preview_gate()
    values: dict[str, object] = {
        "preview_id": "task-rewrite-plan-preview://sha256/" + HASH,
        "source_plan_version_ref": task_rewrite_plan_version_ref(_rewrite_plan_version()),
        "target_capability": "Evaluate workspace design-state analysis.",
        "rewrite_style": "Concise, self-contained engineering task.",
        "fidelity": RewriteFidelity.CAPABILITY_PRESERVING,
        "operational_noise_policy": "Remove retries and runtime chatter.",
        "examples": (
            TaskRewriteExamplePreviewV2(
                example_id="plan-example://rewrite/primary",
                kind="REWRITE",
                input_summary="A multi-turn workspace analysis request.",
                expected_treatment=("Inspect the workspace and explain its design state."),
            ),
        ),
        "forbidden_content_rules": (
            "original final answer",
            "private reference",
        ),
        "expected_capability_impact": "Preserve analysis capability.",
        "safety_gate_ref": task_rewrite_preview_safety_gate_ref(gate),
        "projection_policy_version": "task-rewrite-preview/r4-09-v1",
        "preview_sha256": HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    return TaskRewritePlanPreviewV2(**values)


def _rewrite_invalidation(**overrides: object) -> TaskContractInvalidationV2:
    source_set = _r4_contract_set()
    source_version = _rewrite_plan_version()
    replacement_version = _rewrite_plan_version(
        plan_version=2,
        supersedes_task_rewrite_plan_version_ref=(task_rewrite_plan_version_ref(source_version)),
        task_rewrite_plan_version_id=("task-rewrite-plan-version://sha256/" + OTHER_HASH),
        task_rewrite_plan_version_sha256=OTHER_HASH,
    )
    invalidated = (
        source_set.task_draft_ref,
        source_set.task_prompt_safety_gate_ref,
        source_set.rubric_set_ref,
        source_set.evaluator_spec_ref,
        source_set.reference_policy_ref,
        source_set.tool_policy_ref,
        source_set.contestant_tool_policy_ref,
        source_set.producer_storage_authorization_ref,
        source_set.producer_task_view_ref,
    )
    values: dict[str, object] = {
        "invalidation_id": "task-contract-invalidation://sha256/" + HASH,
        "source_plan_version_ref": task_rewrite_plan_version_ref(source_version),
        "replacement_plan_version_ref": task_rewrite_plan_version_ref(replacement_version),
        "source_contract_set_ref": r4_task_contract_set_ref(source_set),
        "changed_paths": ("rewrite_style",),
        "invalidated_object_refs": invalidated,
        "preserved_object_refs": (
            _v2_ref("selection-context", "rewrite"),
            _v2_ref("task-episode", "rewrite"),
            _v2_ref("evidence-bundle", "rewrite"),
            _v2_ref("tool-capability-catalog", "rewrite"),
        ),
        "required_rebuild_stages": tuple(TaskRewriteRebuildStageV2),
        "hard_gate_revalidation_required": True,
        "policy_version": "task-rewrite/r4-09-v1",
        "invalidation_sha256": HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    return TaskContractInvalidationV2(**values)


def test_r4_task_contract_set_is_closed_typed_and_audit_independent() -> None:
    first = _r4_contract_set(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))
    second = _r4_contract_set(audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)))

    assert r4_task_contract_set_carried_sha256(first) == (r4_task_contract_set_carried_sha256(second))
    assert r4_task_contract_set_ref(first).object_type == "r4-task-contract-set"
    assert first.canonical_sha256() != second.canonical_sha256()

    with pytest.raises(ValidationError, match="task_draft_ref"):
        _r4_contract_set(task_draft_ref=_v2_ref("rubric-set", "wrong"))
    with pytest.raises(ValidationError):
        R4TaskContractSetV2.model_validate(
            {
                **first.model_dump(mode="json"),
                "private_reference_ref": _v2_ref(
                    "private-reference",
                    "forbidden",
                ).model_dump(mode="json"),
            }
        )


def test_task_rewrite_plan_version_enforces_predecessor_and_carried_refs() -> None:
    version = _rewrite_plan_version()
    assert task_rewrite_plan_ref(version.plan).object_type == "task-rewrite-plan"
    assert task_rewrite_plan_version_ref(version).object_type == ("task-rewrite-plan-version")
    assert task_rewrite_plan_carried_sha256(version.plan) != (
        task_rewrite_plan_carried_sha256(
            version.plan.model_copy(update={"rewrite_style": "Different style."})
        )
    )

    with pytest.raises(ValidationError, match="version 1"):
        _rewrite_plan_version(
            supersedes_task_rewrite_plan_version_ref=_v2_ref(
                "task-rewrite-plan-version",
                "previous",
            )
        )
    with pytest.raises(ValidationError, match="predecessor"):
        _rewrite_plan_version(plan_version=2)
    with pytest.raises(ValidationError, match="basis_contract_set_ref"):
        _rewrite_plan_version(
            basis_contract_set_ref=_v2_ref("task-draft", "wrong"),
        )


def test_task_rewrite_preview_is_recursively_content_minimized() -> None:
    preview = _rewrite_preview()
    assert task_rewrite_plan_preview_ref(preview).object_type == ("task-rewrite-plan-preview")
    rendered = preview.model_dump(mode="json")
    serialized = str(rendered)
    for forbidden in (
        "evidence_refs",
        "source_spans",
        "selection_context_ref",
        "rubric_set_ref",
        "evaluator_spec_ref",
        "reference_policy_ref",
        "tool_policy_ref",
        "storage_authorization_ref",
        "credential",
    ):
        assert forbidden not in serialized

    with pytest.raises(ValidationError):
        TaskRewriteExamplePreviewV2.model_validate(
            {
                **preview.examples[0].model_dump(mode="json"),
                "evidence_refs": [_evidence("leak").model_dump(mode="json")],
            }
        )
    with pytest.raises(ValidationError):
        TaskRewritePlanPreviewV2.model_validate(
            {
                **rendered,
                "private_reference_ref": _v2_ref(
                    "private-reference",
                    "leak",
                ).model_dump(mode="json"),
            }
        )


def test_rewrite_preview_gate_requires_complete_nonwaivable_safety() -> None:
    gate = _rewrite_preview_gate()
    assert task_rewrite_preview_safety_gate_ref(gate).object_type == ("task-rewrite-preview-safety-gate")

    with pytest.raises(ValidationError, match="semantic assessment"):
        _rewrite_preview_gate(semantic_assessment_ref=None)
    with pytest.raises(ValidationError, match="exactly one check"):
        _rewrite_preview_gate(checks=_checks()[:-1])
    with pytest.raises(ValidationError, match="cannot contain findings"):
        _rewrite_preview_gate(findings=(_safety_finding(),))


def test_task_contract_invalidation_requires_exact_ordered_closure() -> None:
    invalidation = _rewrite_invalidation()
    assert invalidation.required_rebuild_stages == tuple(TaskRewriteRebuildStageV2)
    assert task_contract_invalidation_ref(invalidation).object_type == ("task-contract-invalidation")

    with pytest.raises(ValidationError, match="canonical invalidated"):
        _rewrite_invalidation(invalidated_object_refs=tuple(reversed(invalidation.invalidated_object_refs)))
    with pytest.raises(ValidationError, match="canonical rebuild"):
        _rewrite_invalidation(required_rebuild_stages=tuple(reversed(invalidation.required_rebuild_stages)))
    with pytest.raises(ValidationError, match="changed_paths"):
        _rewrite_invalidation(changed_paths=("rewrite_style", "rewrite_style"))


def test_task_rewrite_application_is_candidate_only_and_hash_bound() -> None:
    source_set = _r4_contract_set()
    replacement_set = _r4_contract_set(
        contract_set_id="r4-task-contract-set://sha256/" + OTHER_HASH,
        contract_set_sha256=OTHER_HASH,
        task_draft_ref=_v2_ref(
            "task-draft",
            "rewrite-replacement",
            digest=OTHER_HASH,
        ),
    )
    invalidation = _rewrite_invalidation()
    source_version = _rewrite_plan_version()
    replacement_version = _rewrite_plan_version(
        plan_version=2,
        supersedes_task_rewrite_plan_version_ref=(task_rewrite_plan_version_ref(source_version)),
        task_rewrite_plan_version_id=("task-rewrite-plan-version://sha256/" + OTHER_HASH),
        task_rewrite_plan_version_sha256=OTHER_HASH,
    )
    application = TaskRewriteApplicationV2(
        application_id="task-rewrite-application://sha256/" + HASH,
        source_plan_version_ref=task_rewrite_plan_version_ref(source_version),
        replacement_plan_version_ref=task_rewrite_plan_version_ref(replacement_version),
        invalidation_ref=task_contract_invalidation_ref(invalidation),
        source_contract_set_ref=r4_task_contract_set_ref(source_set),
        replacement_contract_set_ref=r4_task_contract_set_ref(replacement_set),
        candidate_state="CANDIDATE_TASK",
        policy_version="task-rewrite/r4-09-v1",
        application_sha256=HASH,
        audit=_audit(),
    )

    assert task_rewrite_application_ref(application).object_type == ("task-rewrite-application")
    assert task_rewrite_application_carried_sha256(application) != (
        task_rewrite_application_carried_sha256(
            application.model_copy(
                update={"replacement_contract_set_ref": r4_task_contract_set_ref(source_set)}
            )
        )
    )
    with pytest.raises(ValidationError):
        TaskRewriteApplicationV2.model_validate(
            {
                **application.model_dump(mode="json"),
                "candidate_state": "RELEASED",
            }
        )


def test_rewrite_contract_hashes_ignore_audit_and_bind_behavior() -> None:
    first_version = _rewrite_plan_version(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))
    second_version = _rewrite_plan_version(audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)))
    first_gate = _rewrite_preview_gate(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))
    second_gate = _rewrite_preview_gate(audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)))
    first_preview = _rewrite_preview(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))
    second_preview = _rewrite_preview(audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)))
    first_invalidation = _rewrite_invalidation(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))
    second_invalidation = _rewrite_invalidation(audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)))

    assert task_rewrite_plan_version_carried_sha256(first_version) == (
        task_rewrite_plan_version_carried_sha256(second_version)
    )
    assert task_rewrite_preview_safety_gate_carried_sha256(first_gate) == (
        task_rewrite_preview_safety_gate_carried_sha256(second_gate)
    )
    assert task_rewrite_plan_preview_carried_sha256(first_preview) == (
        task_rewrite_plan_preview_carried_sha256(second_preview)
    )
    assert task_contract_invalidation_carried_sha256(first_invalidation) == (
        task_contract_invalidation_carried_sha256(second_invalidation)
    )


def test_task_rewrite_plan_hash_is_stable_across_python_hash_seeds() -> None:
    script = """
from datetime import UTC, datetime
from eval_factory.contracts import task_rewrite_plan_carried_sha256
from eval_factory.contracts.approval import (
    ExampleKind, PlanExample, RewriteFidelity, TaskRewritePlan,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

plan = TaskRewritePlan(
    task_rewrite_plan_id="task-rewrite-plan://seed-test",
    selection_context_ref=ObjectRef(
        object_type="selection-context",
        object_id="selection-context://seed-test",
        object_version="v2",
        object_sha256="a" * 64,
    ),
    target_capability="Evaluate workspace design.",
    rewrite_style="Concise.",
    fidelity=RewriteFidelity.CAPABILITY_PRESERVING,
    operational_noise_policy="Remove retries.",
    examples=(
        PlanExample(
            example_id="plan-example://seed-test",
            kind=ExampleKind.REWRITE,
            input_summary="Input summary.",
            expected_treatment="Expected treatment.",
        ),
    ),
    forbidden_content_rules=("secret", "original final answer"),
    expected_capability_impact="Preserve capability.",
    audit=ContractAudit(
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
        created_by="seed-test",
        governing_versions=(
            VersionBinding(component="task-rewrite", version="r4-09-v1"),
        ),
    ),
)
print(task_rewrite_plan_carried_sha256(plan))
"""
    observed = []
    for seed in ("1", "27", "777"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        process = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )
        observed.append(process.stdout.strip())
    assert len(set(observed)) == 1
