from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_producer_task_view import (
    _audit as producer_audit,
)
from test_producer_task_view import (
    _draft,
    _evaluation_contract,
    _producer_evidence,
    _rubric_set,
    _tool_contract,
)

from eval_factory.attachment_planning import (
    ArtifactEvidenceModeCompiler,
    FakePromptOnlyDependencyFixture,
    FakePromptOnlyDependencyRunner,
    PromptOnlyCandidateRole,
    PromptOnlyDependencyAssessment,
    PromptOnlyDependencyCompiler,
    PromptOnlyDependencyOutcome,
    PromptOnlyDependencyPlanningProjector,
    PromptOnlyDependencyPolicyError,
    PromptOnlyDependencyReason,
    PromptOnlyDependencyRequestBuilder,
)
from eval_factory.contracts import (
    ArtifactEvidenceTargetV2,
    PromptOnlyDependencyDiscoveryV2,
    PromptOnlyDependencyPlanningContextV2,
    artifact_evidence_target_carried_sha256,
    artifact_evidence_target_ref,
    attachment_planning_context_ref,
    producer_task_view_ref,
    prompt_only_dependency_discovery_carried_sha256,
    prompt_only_dependency_planning_context_carried_sha256,
    task_draft_carried_sha256,
)
from eval_factory.contracts.attachment import ReconstructionMode
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.task import (
    AttachmentDependency,
    EvidencePriority,
)
from eval_factory.task_authoring import ProducerTaskViewCompiler

HASH = "a" * 64
OTHER_HASH = "b" * 64
MODEL_PROFILE = "internal-prompt-dependency-v1"
PROMPT_VERSION = "prompt-only-dependency/v1"
ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = (
    ROOT / "evals/golden/eval_factory/attachment_dependencies" / "r5-03-prompt-only-dependencies-v1.json"
)


def _audit(
    created_at: datetime = datetime(2026, 7, 27, tzinfo=UTC),
    *,
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="prompt-only-dependency-test",
        governing_versions=(
            VersionBinding(
                component="prompt-only-dependency",
                version="r5-03",
            ),
        ),
        input_refs=input_refs,
    )


def _rehash_draft(draft):
    digest = task_draft_carried_sha256(draft)
    return draft.model_copy(
        update={
            "task_draft_id": f"task-draft://sha256/{digest}",
            "task_draft_sha256": digest,
        }
    )


def _case(
    *,
    description: str = "Use the explicit input file inputs/source.csv.",
    descriptions: tuple[str, ...] | None = None,
    priority: EvidencePriority = EvidencePriority.EXPLICIT_REQUIREMENT,
    prompt: str = "Use the supplied input and create a concise summary.",
    forbidden_outputs: tuple[str, ...] = ("final/report.docx",),
    no_requirements: bool = False,
):
    source = _draft()
    dependencies = ()
    if not no_requirements:
        parent = source.attachment_dependencies[0]
        dependency_descriptions = descriptions or (description,)
        dependencies = tuple(
            AttachmentDependency(
                dependency_id=(parent.dependency_id if index == 0 else f"{parent.dependency_id}-{index + 1}"),
                description=item_description,
                criticality=parent.criticality,
                evidence_priority=priority,
                evidence=parent.evidence,
            )
            for index, item_description in enumerate(dependency_descriptions)
        )
    draft = _rehash_draft(
        source.model_copy(
            update={
                "visible_prompt": prompt,
                "forbidden_outputs": forbidden_outputs,
                "attachment_dependencies": dependencies,
                "task_draft_sha256": HASH,
            }
        )
    )
    rubric_set = _rubric_set(draft)
    evaluator_spec, reference_policy = _evaluation_contract(rubric_set)
    tool_policy, contestant_policy = _tool_contract(
        draft,
        rubric_set,
        evaluator_spec,
    )
    view_result, bundle = _producer_evidence(empty=True)
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
    from eval_factory.attachment_planning import AttachmentPlanningBridge

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


def _request(case, *, existing_targets: tuple[ArtifactEvidenceTargetV2, ...] = ()):
    return PromptOnlyDependencyRequestBuilder().build(
        attachment_planning_context=case[5],
        producer_task_view=case[1],
        dependency_planning_context=case[6],
        existing_targets=existing_targets,
        model_profile=MODEL_PROFILE,
        prompt_version=PROMPT_VERSION,
        abstain_conditions=(
            "ambiguous input/output role",
            "media type unresolved",
        ),
        audit=_audit(),
    )


def _proposal(
    request,
    *,
    outcome: PromptOnlyDependencyOutcome = PromptOnlyDependencyOutcome.RESOLVED,
    role: PromptOnlyCandidateRole = PromptOnlyCandidateRole.INPUT_DEPENDENCY,
    media_type: str | None = "text/csv",
    reasons: frozenset[PromptOnlyDependencyReason] = frozenset(),
    model_available: bool = True,
):
    assessments = ()
    if request.dependencies and outcome is PromptOnlyDependencyOutcome.RESOLVED:
        assessments = tuple(
            PromptOnlyDependencyAssessment(
                dependency_id=dependency.dependency_id,
                candidate_role=role,
                proposed_media_type=media_type,
                explanation_code="prompt-only-dependency/test-assessment",
            )
            for dependency in request.dependencies
            if dependency.existing_target_ref is None
            and not (
                len(dependency.explicit_path_facts) == 1
                and dependency.explicit_path_facts[0].media_type is not None
            )
        )
    fixture = FakePromptOnlyDependencyFixture(
        fixture_id="prompt-only-dependency-fixture://default",
        outcome=outcome,
        assessments=assessments,
        unresolved_reasons=reasons,
        model_available=model_available,
    )
    return FakePromptOnlyDependencyRunner().run(
        request,
        fixture=fixture,
        audit=_audit(),
    )


def _compile(
    case,
    *,
    request=None,
    proposal=None,
    existing_targets: tuple[ArtifactEvidenceTargetV2, ...] = (),
):
    request = request or _request(case, existing_targets=existing_targets)
    return PromptOnlyDependencyCompiler().compile(
        request=request,
        proposal=proposal,
        attachment_planning_context=case[5],
        producer_task_view=case[1],
        dependency_planning_context=case[6],
        existing_targets=existing_targets,
        audit=_audit(),
    )


def _existing_target(case, *, path: str = "inputs/direct.csv") -> ArtifactEvidenceTargetV2:
    binding = case[6].dependency_evidence_bindings[0]
    requirement = case[1].attachment_requirements[0]
    target = ArtifactEvidenceTargetV2(
        artifact_evidence_target_id="artifact-evidence-target://pending",
        attachment_planning_context_ref=attachment_planning_context_ref(case[5]),
        attachment_dependency_id=requirement.dependency_id,
        artifact_id="artifact://prompt-only/existing",
        logical_path=path,
        media_type="text/csv",
        criticality=requirement.criticality,
        requirement_evidence=binding.evidence,
        candidate_source_refs=(),
        policy_version="artifact-evidence-mode/r5-02-v1",
        artifact_evidence_target_sha256=HASH,
        audit=_audit(),
    )
    digest = artifact_evidence_target_carried_sha256(target)
    return target.model_copy(
        update={
            "artifact_evidence_target_id": (f"artifact-evidence-target://sha256/{digest}"),
            "artifact_evidence_target_sha256": digest,
        }
    )


def test_planning_projector_emits_exact_evidence_sidecar_without_task_text() -> None:
    case = _case()
    planning = case[6]

    assert isinstance(planning, PromptOnlyDependencyPlanningContextV2)
    assert planning.attachment_planning_context_ref == (attachment_planning_context_ref(case[5]))
    assert planning.producer_task_view_ref == producer_task_view_ref(case[1])
    assert planning.source_task_draft_sha256 == case[0].task_draft_sha256
    assert planning.source_trace_id == case[5].source_trace_id
    assert planning.dependency_evidence_bindings[0].evidence == (case[0].attachment_dependencies[0].evidence)
    assert planning.prompt_only_dependency_planning_context_sha256 == (
        prompt_only_dependency_planning_context_carried_sha256(planning)
    )

    serialized = str(planning.model_dump(mode="json"))
    for forbidden in (
        case[0].visible_prompt,
        case[0].task_intent,
        case[0].evaluation_claim,
        "forbidden_outputs",
        "rubric",
        "evaluator",
        "reference_policy",
        "credential",
    ):
        assert forbidden not in serialized


def test_planning_projector_rejects_task_view_dependency_mismatch() -> None:
    case = _case()
    wrong_requirement = (
        case[1].attachment_requirements[0].model_copy(update={"description": "Different description."})
    )
    changed_view = case[1].model_copy(update={"attachment_requirements": (wrong_requirement,)})

    with pytest.raises(PromptOnlyDependencyPolicyError, match="dependency"):
        PromptOnlyDependencyPlanningProjector().compile(
            task_draft=case[0],
            producer_task_view=changed_view,
            storage_authorization=case[2],
            evidence_bundle=case[4],
            attachment_planning_context=case[5],
            audit=_audit(),
        )


def test_request_contains_no_evidence_or_taskdraft_and_has_zero_execution_boundary() -> None:
    case = _case()
    request = _request(case)

    assert request.query_instruction == case[1].query_instruction
    assert request.dependencies[0].evidence_priority is (EvidencePriority.EXPLICIT_REQUIREMENT)
    assert request.prompt_boundary_enforcement_ref.object_type == ("prompt-boundary-enforcement")
    serialized = str(request.model_dump(mode="json"))
    for forbidden in (
        "EvidenceRef",
        "source_spans",
        "task_draft",
        "evaluation_claim",
        "authorized_subject_refs",
        "raw_trace",
        "provider_preference",
        "runtime_preference",
        "credential",
    ):
        assert forbidden not in serialized


def test_explicit_requirement_path_resolves_without_semantic_proposal() -> None:
    case = _case(description="Read the input from data/source.csv.")
    result = _compile(case)

    assert result.outcome is PromptOnlyDependencyOutcome.RESOLVED
    assert result.discovery is not None
    assert isinstance(result.discovery, PromptOnlyDependencyDiscoveryV2)
    target = result.discovery.targets[0]
    assert target.logical_path == "data/source.csv"
    assert target.media_type == "text/csv"
    assert target.requirement_evidence == (case[6].dependency_evidence_bindings[0].evidence)
    assert target.candidate_source_refs == ()
    assert result.discovery.semantic_evaluated is False
    assert result.discovery.model_profile is None
    assert result.discovery.prompt_version is None
    assert result.discovery.prompt_only_dependency_discovery_sha256 == (
        prompt_only_dependency_discovery_carried_sha256(result.discovery)
    )


def test_windows_requirement_path_is_normalized_but_query_path_is_not_authoritative() -> None:
    explicit_case = _case(description=r"Read the input from inputs\source.csv.")
    explicit_result = _compile(explicit_case)
    assert explicit_result.discovery is not None
    assert explicit_result.discovery.targets[0].logical_path == "inputs/source.csv"

    inferred_case = _case(
        description="Use the supplied tabular input.",
        prompt="Read final/report.csv and produce the requested summary.",
    )
    request = _request(inferred_case)
    proposal = _proposal(request, media_type="text/csv")
    inferred_result = _compile(
        inferred_case,
        request=request,
        proposal=proposal,
    )
    assert inferred_result.discovery is not None
    assert inferred_result.discovery.targets[0].logical_path.startswith("inputs/prompt-only/")
    assert inferred_result.discovery.targets[0].logical_path != "final/report.csv"


def test_inferred_requirement_uses_compiler_owned_path_and_semantic_media() -> None:
    case = _case(description="Use the supplied tabular input.")
    request = _request(case)
    proposal = _proposal(request, media_type="text/csv")

    result = _compile(case, request=request, proposal=proposal)

    assert result.outcome is PromptOnlyDependencyOutcome.RESOLVED
    assert result.discovery is not None
    target = result.discovery.targets[0]
    assert target.logical_path.startswith("inputs/prompt-only/")
    assert target.logical_path.endswith(".csv")
    assert "report" not in target.logical_path
    assert target.media_type == "text/csv"
    assert result.discovery.semantic_evaluated is True
    assert result.discovery.model_profile == MODEL_PROFILE
    assert result.discovery.prompt_version == PROMPT_VERSION


def test_multiple_dependencies_resolve_as_one_complete_target_set() -> None:
    case = _case(
        descriptions=(
            "Use inputs/source.csv.",
            "Use config/settings.json.",
        )
    )

    result = _compile(case)

    assert result.outcome is PromptOnlyDependencyOutcome.RESOLVED
    assert result.discovery is not None
    assert {target.logical_path for target in result.discovery.targets} == {
        "config/settings.json",
        "inputs/source.csv",
    }
    assert len(result.discovery.targets) == len(case[1].attachment_requirements)


def test_existing_direct_target_is_preserved_without_semantic_rewrite() -> None:
    case = _case(priority=EvidencePriority.DIRECT_OBSERVATION)
    existing = _existing_target(case)
    request = _request(case, existing_targets=(existing,))

    result = _compile(
        case,
        request=request,
        proposal=None,
        existing_targets=(existing,),
    )

    assert result.outcome is PromptOnlyDependencyOutcome.RESOLVED
    assert result.discovery is not None
    assert result.discovery.targets == (existing,)
    assert result.discovery.existing_target_refs == (artifact_evidence_target_ref(existing),)
    assert result.discovery.discovered_target_refs == ()


def test_existing_target_with_stale_r5_policy_is_rejected() -> None:
    case = _case(priority=EvidencePriority.DIRECT_OBSERVATION)
    existing = _existing_target(case).model_copy(
        update={"policy_version": "artifact-evidence-mode/r5-01-stale"}
    )
    digest = artifact_evidence_target_carried_sha256(existing)
    existing = existing.model_copy(
        update={
            "artifact_evidence_target_id": f"artifact-evidence-target://sha256/{digest}",
            "artifact_evidence_target_sha256": digest,
        }
    )

    with pytest.raises(PromptOnlyDependencyPolicyError, match="policy"):
        _request(case, existing_targets=(existing,))


def test_direct_evidence_without_existing_target_blocks_capability() -> None:
    case = _case(priority=EvidencePriority.DIRECT_OBSERVATION)

    result = _compile(case)

    assert result.outcome is PromptOnlyDependencyOutcome.BLOCKED_CAPABILITY
    assert result.discovery is None
    assert result.reasons == frozenset({PromptOnlyDependencyReason.DIRECT_TARGET_REQUIRED})


def test_forbidden_path_dominates_direct_target_capability_gap() -> None:
    case = _case(
        description="Use final/report.docx as the input.",
        priority=EvidencePriority.DIRECT_OBSERVATION,
        forbidden_outputs=("final/report.docx",),
    )

    result = _compile(case)

    assert result.outcome is PromptOnlyDependencyOutcome.BLOCKED_SAFETY
    assert result.discovery is None
    assert result.reasons == frozenset({PromptOnlyDependencyReason.FORBIDDEN_OUTPUT_MATCH})


def test_no_requirements_returns_not_required() -> None:
    case = _case(no_requirements=True)

    result = _compile(case)

    assert result.outcome is PromptOnlyDependencyOutcome.NOT_REQUIRED
    assert result.discovery is None
    assert result.unresolved_dependency_ids == ()
    assert result.reasons == frozenset()


def test_conflicting_explicit_paths_abstain_without_partial_target() -> None:
    case = _case(description="Use inputs/a.csv or inputs/b.csv.")

    result = _compile(case)

    assert result.outcome is PromptOnlyDependencyOutcome.ABSTAIN
    assert result.discovery is None
    assert result.reasons == frozenset({PromptOnlyDependencyReason.AMBIGUOUS_EXPLICIT_FACTS})


def test_forbidden_dependency_blocks_the_entire_multi_dependency_set() -> None:
    case = _case(
        descriptions=(
            "Use inputs/source.csv.",
            "Use final/report.docx.",
        ),
        forbidden_outputs=("final/report.docx",),
    )

    result = _compile(case)

    assert result.outcome is PromptOnlyDependencyOutcome.BLOCKED_SAFETY
    assert result.discovery is None
    assert result.unresolved_dependency_ids == (case[1].attachment_requirements[1].dependency_id,)


def test_model_unavailable_blocks_without_partial_target() -> None:
    case = _case(description="Use the supplied tabular input.")
    request = _request(case)
    proposal = _proposal(
        request,
        outcome=PromptOnlyDependencyOutcome.BLOCKED_CAPABILITY,
        media_type=None,
        reasons=frozenset({PromptOnlyDependencyReason.MODEL_UNAVAILABLE}),
        model_available=False,
    )

    result = _compile(case, request=request, proposal=proposal)

    assert result.outcome is PromptOnlyDependencyOutcome.BLOCKED_CAPABILITY
    assert result.discovery is None
    assert result.reasons == frozenset({PromptOnlyDependencyReason.MODEL_UNAVAILABLE})


def test_forbidden_path_dominates_model_unavailable_proposal() -> None:
    case = _case(
        description="Use final/report.docx as the input.",
        forbidden_outputs=("final/report.docx",),
    )
    request = _request(case)
    proposal = _proposal(
        request,
        outcome=PromptOnlyDependencyOutcome.BLOCKED_CAPABILITY,
        media_type=None,
        reasons=frozenset({PromptOnlyDependencyReason.MODEL_UNAVAILABLE}),
        model_available=False,
    )

    result = _compile(case, request=request, proposal=proposal)

    assert result.outcome is PromptOnlyDependencyOutcome.BLOCKED_SAFETY
    assert result.discovery is None
    assert result.reasons == frozenset({PromptOnlyDependencyReason.FORBIDDEN_OUTPUT_MATCH})


def test_forbidden_explicit_path_dominates_and_does_not_disclose_path() -> None:
    case = _case(
        description="Use final/report.docx as the input.",
        forbidden_outputs=("final/report.docx",),
    )

    result = _compile(case)

    assert result.outcome is PromptOnlyDependencyOutcome.BLOCKED_SAFETY
    assert result.discovery is None
    assert result.reasons == frozenset({PromptOnlyDependencyReason.FORBIDDEN_OUTPUT_MATCH})
    assert "final/report.docx" not in str(result.model_dump(mode="json"))


def test_requested_output_semantic_role_blocks_safety() -> None:
    case = _case(description="Use the supplied document.")
    request = _request(case)
    proposal = _proposal(
        request,
        role=PromptOnlyCandidateRole.REQUESTED_OUTPUT,
        media_type="application/pdf",
    )

    result = _compile(case, request=request, proposal=proposal)

    assert result.outcome is PromptOnlyDependencyOutcome.BLOCKED_SAFETY
    assert result.discovery is None
    assert result.reasons == frozenset({PromptOnlyDependencyReason.REQUESTED_OUTPUT_PROPOSED})


def test_requested_output_assessment_dominates_another_dependency_ambiguity() -> None:
    case = _case(
        descriptions=(
            "Use the first supplied document.",
            "Use the second supplied document.",
        )
    )
    request = _request(case)
    fixture = FakePromptOnlyDependencyFixture(
        fixture_id="prompt-only-dependency-fixture://mixed-semantic-roles",
        outcome=PromptOnlyDependencyOutcome.RESOLVED,
        assessments=(
            PromptOnlyDependencyAssessment(
                dependency_id=request.dependencies[0].dependency_id,
                candidate_role=PromptOnlyCandidateRole.AMBIGUOUS,
                proposed_media_type=None,
                explanation_code="prompt-only-dependency/ambiguous",
            ),
            PromptOnlyDependencyAssessment(
                dependency_id=request.dependencies[1].dependency_id,
                candidate_role=PromptOnlyCandidateRole.REQUESTED_OUTPUT,
                proposed_media_type="application/pdf",
                explanation_code="prompt-only-dependency/requested-output",
            ),
        ),
    )
    proposal = FakePromptOnlyDependencyRunner().run(
        request,
        fixture=fixture,
        audit=_audit(),
    )

    result = _compile(case, request=request, proposal=proposal)

    assert result.outcome is PromptOnlyDependencyOutcome.BLOCKED_SAFETY
    assert result.discovery is None
    assert result.unresolved_dependency_ids == (request.dependencies[1].dependency_id,)
    assert result.reasons == frozenset({PromptOnlyDependencyReason.REQUESTED_OUTPUT_PROPOSED})


def test_ambiguous_semantic_role_abstains() -> None:
    case = _case(description="Use the supplied document.")
    request = _request(case)
    proposal = _proposal(
        request,
        outcome=PromptOnlyDependencyOutcome.ABSTAIN,
        role=PromptOnlyCandidateRole.AMBIGUOUS,
        media_type=None,
        reasons=frozenset({PromptOnlyDependencyReason.AMBIGUOUS_INPUT_OUTPUT_ROLE}),
    )

    result = _compile(case, request=request, proposal=proposal)

    assert result.outcome is PromptOnlyDependencyOutcome.ABSTAIN
    assert result.discovery is None


def test_fake_runner_rejects_assessments_for_deterministically_resolved_dependencies() -> None:
    case = _case(description="Use inputs/source.csv.")
    request = _request(case)
    fixture = FakePromptOnlyDependencyFixture(
        fixture_id="prompt-only-dependency-fixture://covered",
        outcome=PromptOnlyDependencyOutcome.RESOLVED,
        assessments=(
            PromptOnlyDependencyAssessment(
                dependency_id=request.dependencies[0].dependency_id,
                candidate_role=PromptOnlyCandidateRole.INPUT_DEPENDENCY,
                proposed_media_type="text/csv",
                explanation_code="prompt-only-dependency/covered",
            ),
        ),
    )

    with pytest.raises(PromptOnlyDependencyPolicyError, match="residual"):
        FakePromptOnlyDependencyRunner().run(
            request,
            fixture=fixture,
            audit=_audit(),
        )


def _compile_gold_setup(setup: str):
    if setup == "EXISTING_DIRECT_TARGET":
        case = _case(priority=EvidencePriority.DIRECT_OBSERVATION)
        existing = _existing_target(case)
        request = _request(case, existing_targets=(existing,))
        return _compile(
            case,
            request=request,
            existing_targets=(existing,),
        )
    if setup == "EXPLICIT_REQUIREMENT_PATH":
        return _compile(_case(description="Use inputs/source.csv."))
    if setup == "INFERRED_MEDIA":
        case = _case(description="Use the supplied tabular input.")
        request = _request(case)
        return _compile(
            case,
            request=request,
            proposal=_proposal(request, media_type="text/csv"),
        )
    if setup == "MULTIPLE_EXPLICIT_PATHS":
        return _compile(
            _case(
                descriptions=(
                    "Use inputs/source.csv.",
                    "Use config/settings.json.",
                )
            )
        )
    if setup == "NO_REQUIREMENTS":
        return _compile(_case(no_requirements=True))
    if setup == "DIRECT_TARGET_MISSING":
        return _compile(_case(priority=EvidencePriority.DIRECT_OBSERVATION))
    if setup == "MODEL_UNAVAILABLE":
        case = _case(description="Use the supplied tabular input.")
        request = _request(case)
        return _compile(
            case,
            request=request,
            proposal=_proposal(
                request,
                outcome=PromptOnlyDependencyOutcome.BLOCKED_CAPABILITY,
                media_type=None,
                reasons=frozenset({PromptOnlyDependencyReason.MODEL_UNAVAILABLE}),
                model_available=False,
            ),
        )
    if setup == "AMBIGUOUS_EXPLICIT_PATHS":
        return _compile(_case(description="Use inputs/a.csv or inputs/b.csv."))
    if setup == "AMBIGUOUS_ROLE":
        case = _case(description="Use the supplied document.")
        request = _request(case)
        return _compile(
            case,
            request=request,
            proposal=_proposal(
                request,
                outcome=PromptOnlyDependencyOutcome.ABSTAIN,
                media_type=None,
                reasons=frozenset({PromptOnlyDependencyReason.AMBIGUOUS_INPUT_OUTPUT_ROLE}),
            ),
        )
    if setup == "FORBIDDEN_PATH":
        return _compile(
            _case(
                description="Use final/report.docx as the input.",
                forbidden_outputs=("final/report.docx",),
            )
        )
    if setup == "REQUESTED_OUTPUT":
        case = _case(description="Use the supplied document.")
        request = _request(case)
        return _compile(
            case,
            request=request,
            proposal=_proposal(
                request,
                role=PromptOnlyCandidateRole.REQUESTED_OUTPUT,
                media_type="application/pdf",
            ),
        )
    raise AssertionError(f"unknown gold setup: {setup}")


def test_prompt_only_dependency_gold_cases_are_content_free_and_executable() -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    assert gold["schema_version"] == ("eval-factory/r5-03-prompt-only-dependency-gold/v1")
    assert gold["claim_scope"] == "CONTENT_FREE_DEPENDENCY_DISCOVERY_POLICY_ONLY"

    for case in gold["cases"]:
        assert set(case) == {"case_id", "expected_outcome", "setup"}
        result = _compile_gold_setup(case["setup"])
        assert result.outcome.value == case["expected_outcome"]


def test_resolved_targets_feed_r5_02_as_prompt_only() -> None:
    case = _case(description="Use inputs/source.csv.")
    result = _compile(case)
    assert result.discovery is not None

    matrix_result = ArtifactEvidenceModeCompiler().compile(
        attachment_planning_context=case[5],
        producer_task_view=case[1],
        storage_authorization=case[2],
        evidence_bundle=case[4],
        evidence_view_result=case[3],
        timelines=(),
        targets=result.discovery.targets,
        audit=_audit(),
    )

    assert matrix_result.artifact_evidence_matrix is not None
    assert matrix_result.artifact_evidence_matrix.aggregate_mode == "PROMPT_ONLY"
    assert {row.row.selected_mode for row in matrix_result.artifact_evidence_matrix.rows} == {
        ReconstructionMode.PROMPT_ONLY
    }


def test_discovery_validate_current_rejects_target_mutation() -> None:
    case = _case(description="Use inputs/source.csv.")
    request = _request(case)
    result = _compile(case, request=request)
    assert result.discovery is not None

    PromptOnlyDependencyCompiler().validate_current(
        request=request,
        proposal=None,
        attachment_planning_context=case[5],
        producer_task_view=case[1],
        dependency_planning_context=case[6],
        existing_targets=(),
        discovery=result.discovery,
    )

    changed_target = result.discovery.targets[0].model_copy(update={"logical_path": "inputs/changed.csv"})
    stale = result.discovery.model_copy(update={"targets": (changed_target,)})
    with pytest.raises(PromptOnlyDependencyPolicyError, match="current"):
        PromptOnlyDependencyCompiler().validate_current(
            request=request,
            proposal=None,
            attachment_planning_context=case[5],
            producer_task_view=case[1],
            dependency_planning_context=case[6],
            existing_targets=(),
            discovery=stale,
        )


def test_compiler_rejects_request_audit_lineage_tampering() -> None:
    case = _case()
    request = _request(case)
    stale_audit = request.audit.model_copy(
        update={
            "input_refs": (
                ObjectRef(
                    object_type="unrelated",
                    object_id="unrelated://request-audit",
                    object_version="v1",
                    object_sha256=HASH,
                ),
            )
        }
    )
    stale_request = request.model_copy(update={"audit": stale_audit})

    with pytest.raises(PromptOnlyDependencyPolicyError, match="request"):
        _compile(case, request=stale_request)


def test_result_model_rejects_partial_or_unknown_fields() -> None:
    case = _case()
    result = _compile(case)

    with pytest.raises(ValidationError):
        type(result).model_validate(
            {
                **result.model_dump(mode="python"),
                "artifact_content": "forbidden",
            }
        )
