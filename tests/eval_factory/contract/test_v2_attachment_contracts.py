from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts import (
    ArtifactBuildContractV2,
    ArtifactBuildSpecV2,
    ArtifactEvidenceMatrixV2,
    ArtifactEvidenceRowV2,
    ArtifactEvidenceTargetV2,
    ArtifactRoutePlanEntryV2,
    ArtifactRoutingPlanV2,
    ArtifactRoutingPolicyV2,
    AttachmentPlanningContextV2,
    PromptOnlyDependencyDiscoveryV2,
    PromptOnlyDependencyEvidenceBindingV2,
    PromptOnlyDependencyPlanningContextV2,
    PublicSourceRetrievalPolicyV2,
    PublicSourceSafetyAssessmentV2,
    SourceEvidenceClaimBindingV2,
    SourceEvidenceSetV2,
    SourceEvidenceV2,
    artifact_build_contract_carried_sha256,
    artifact_build_contract_ref,
    artifact_evidence_matrix_carried_sha256,
    artifact_evidence_matrix_ref,
    artifact_evidence_row_carried_sha256,
    artifact_evidence_row_ref,
    artifact_evidence_target_carried_sha256,
    artifact_evidence_target_ref,
    artifact_routing_policy_carried_sha256,
    artifact_routing_policy_ref,
    attachment_planning_context_carried_sha256,
    attachment_planning_context_ref,
    prompt_only_dependency_discovery_carried_sha256,
    prompt_only_dependency_discovery_ref,
    prompt_only_dependency_planning_context_carried_sha256,
    prompt_only_dependency_planning_context_ref,
    public_source_retrieval_policy_carried_sha256,
    public_source_retrieval_policy_ref,
    public_source_safety_assessment_carried_sha256,
    public_source_safety_assessment_ref,
    source_evidence_set_carried_sha256,
    source_evidence_v2_carried_sha256,
    source_evidence_v2_ref,
)
from eval_factory.contracts.attachment import (
    ArtifactBuildSpec,
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
    OriginClass,
    ProvenanceDecision,
    SourceEvidence,
    Visibility,
)
from eval_factory.contracts.task import AttachmentCriticality, EvidencePriority

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
FOURTH_HASH = "d" * 64
MODE_POLICY = "artifact-evidence-mode/r5-02-v1"
R2_POLICY = "evidence-compilation/r2-06-v1"
ROOT = Path(__file__).resolve().parents[3]


def _audit(
    created_at: datetime = datetime(2026, 7, 27, tzinfo=UTC),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="attachment-planning-contract-test",
        governing_versions=(
            VersionBinding(
                component="attachment-planning",
                version="r5-01",
            ),
        ),
    )


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str,
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _context(
    *,
    audit: ContractAudit | None = None,
    **overrides: object,
) -> AttachmentPlanningContextV2:
    values: dict[str, object] = {
        "attachment_planning_context_id": "attachment-planning-context://pending",
        "producer_task_view_ref": _ref(
            "producer-task-view",
            "r5-01",
            version="v2",
        ),
        "producer_storage_authorization_ref": _ref(
            "producer-storage-authorization",
            "r5-01",
            version="v2",
            digest=OTHER_HASH,
        ),
        "safe_evidence_bundle_ref": _ref(
            "evidence-bundle",
            "r5-01",
            version="v1",
            digest=THIRD_HASH,
        ),
        "projection_policy_ref": _ref(
            "projection-policy",
            "r5-01",
            version="evidence-views/r2-05-v1",
        ),
        "source_trace_id": "source-trace://attachment-planning/r5-01",
        "trace_ir_version_id": "trace-ir://attachment-planning/r5-01",
        "producer_principal_id": "principal://attachment-producer/r5-01",
        "source_task_draft_sha256": OTHER_HASH,
        "source_contract_chain_sha256": THIRD_HASH,
        "policy_version": "attachment-planning/r5-01-v1",
        "attachment_planning_context_sha256": HASH,
        "audit": audit or _audit(),
    }
    values.update(overrides)
    context = AttachmentPlanningContextV2(**values)
    digest = attachment_planning_context_carried_sha256(context)
    return context.model_copy(
        update={
            "attachment_planning_context_id": (f"attachment-planning-context://sha256/{digest}"),
            "attachment_planning_context_sha256": digest,
        }
    )


def test_attachment_planning_context_is_minimal_closed_and_carried() -> None:
    context = _context()

    assert set(AttachmentPlanningContextV2.model_fields) == {
        "schema_version",
        "attachment_planning_context_id",
        "producer_task_view_ref",
        "producer_storage_authorization_ref",
        "safe_evidence_bundle_ref",
        "projection_policy_ref",
        "source_trace_id",
        "trace_ir_version_id",
        "producer_principal_id",
        "source_task_draft_sha256",
        "source_contract_chain_sha256",
        "policy_version",
        "attachment_planning_context_sha256",
        "audit",
    }
    assert context.attachment_planning_context_sha256 == (attachment_planning_context_carried_sha256(context))
    context_ref = attachment_planning_context_ref(context)
    assert context_ref.object_type == "attachment-planning-context"
    assert context_ref.object_version == "v2"
    assert context_ref.object_sha256 == context.attachment_planning_context_sha256

    for forbidden_field in (
        "query_instruction",
        "attachment_requirements",
        "allowed_tools",
        "forbidden_outputs",
        "evidence",
        "source_spans",
        "excluded_subject_refs",
        "authorized_subject_refs",
        "content",
        "relative_path",
        "media_type",
        "provider_preference",
        "runtime_preference",
        "model_profile",
        "command",
        "credential",
        "endpoint",
        "rubric_set_ref",
        "evaluator_spec_ref",
        "reference_policy_ref",
        "selection_context_ref",
        "lineage_refs",
    ):
        with pytest.raises(ValidationError):
            AttachmentPlanningContextV2.model_validate(
                {
                    **context.model_dump(mode="json"),
                    forbidden_field: "forbidden",
                }
            )


@pytest.mark.parametrize(
    ("field_name", "wrong_ref", "message"),
    [
        pytest.param(
            "producer_task_view_ref",
            _ref("task-draft", "wrong", version="v2"),
            "producer_task_view_ref",
            id="wrong-producer-view-type",
        ),
        pytest.param(
            "producer_task_view_ref",
            _ref("producer-task-view", "wrong", version="v1"),
            "ProducerTaskView v2",
            id="wrong-producer-view-version",
        ),
        pytest.param(
            "producer_storage_authorization_ref",
            _ref("producer-storage-authorization", "wrong", version="v1"),
            "ProducerStorageAuthorization v2",
            id="wrong-authorization-version",
        ),
        pytest.param(
            "safe_evidence_bundle_ref",
            _ref("evidence-bundle", "wrong", version="v2"),
            "EvidenceBundle v1",
            id="wrong-bundle-version",
        ),
        pytest.param(
            "projection_policy_ref",
            _ref("private-reference", "wrong", version="v1"),
            "projection_policy_ref",
            id="wrong-projection-policy-type",
        ),
    ],
)
def test_attachment_planning_context_requires_exact_ref_wrappers(
    field_name: str,
    wrong_ref: ObjectRef,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _context(**{field_name: wrong_ref})


def test_attachment_planning_context_hash_ignores_audit_and_binds_behavior() -> None:
    first = _context(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))
    second = _context(audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)))

    assert first.attachment_planning_context_sha256 == (second.attachment_planning_context_sha256)
    assert first.attachment_planning_context_id == second.attachment_planning_context_id
    assert first.canonical_sha256() != second.canonical_sha256()
    assert attachment_planning_context_carried_sha256(first) != (
        attachment_planning_context_carried_sha256(
            first.model_copy(
                update={
                    "source_contract_chain_sha256": OTHER_HASH,
                }
            )
        )
    )


def _evidence(
    suffix: str,
    *,
    source_trace_id: str = "source-trace://attachment-planning/r5-01",
) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://attachment-mode/{suffix}",
        subject_ref=_ref(
            "requirement-projection",
            suffix,
            version="v1",
        ),
        source_spans=(
            SourceSpanRef(
                span_id=f"source-span://attachment-mode/{suffix}",
                source_trace_id=source_trace_id,
                raw_sha256=HASH,
            ),
        ),
        polarity=EvidencePolarity.POSITIVE,
        capability="artifact-requirement",
        capability_complete=True,
    )


def _target(
    *,
    context: AttachmentPlanningContextV2 | None = None,
    dependency_id: str = "attachment-dependency://mode/input",
    artifact_id: str = "artifact://mode/input",
    path: str = "input.txt",
    media_type: str = "text/plain",
    criticality: AttachmentCriticality = AttachmentCriticality.REQUIRED,
    evidence: tuple[EvidenceRef, ...] | None = None,
    candidate_source_refs: tuple[ObjectRef, ...] = (),
    audit: ContractAudit | None = None,
) -> ArtifactEvidenceTargetV2:
    context = context or _context()
    target = ArtifactEvidenceTargetV2(
        artifact_evidence_target_id="artifact-evidence-target://pending",
        attachment_planning_context_ref=attachment_planning_context_ref(context),
        attachment_dependency_id=dependency_id,
        artifact_id=artifact_id,
        logical_path=path,
        media_type=media_type,
        criticality=criticality,
        requirement_evidence=evidence or (_evidence("requirement"),),
        candidate_source_refs=candidate_source_refs,
        policy_version=MODE_POLICY,
        artifact_evidence_target_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = artifact_evidence_target_carried_sha256(target)
    return target.model_copy(
        update={
            "artifact_evidence_target_id": (f"artifact-evidence-target://sha256/{digest}"),
            "artifact_evidence_target_sha256": digest,
        }
    )


def _r2_row(
    *,
    artifact_id: str = "artifact://mode/input",
    path: str = "input.txt",
    media_type: str = "text/plain",
    criticality: AttachmentCriticality = AttachmentCriticality.REQUIRED,
    mode: ReconstructionMode = ReconstructionMode.PROMPT_ONLY,
    evidence: tuple[EvidenceRef, ...] | None = None,
) -> ArtifactEvidenceRow:
    pre_mutation = (
        EvidenceCoverage.COMPLETE if mode is ReconstructionMode.TRACE_RICH else EvidenceCoverage.NONE
    )
    content = EvidenceCoverage.COMPLETE if mode is ReconstructionMode.TRACE_RICH else EvidenceCoverage.NONE
    structure = (
        EvidenceCoverage.COMPLETE if mode is ReconstructionMode.SKELETON_GUIDED else EvidenceCoverage.NONE
    )
    return ArtifactEvidenceRow(
        artifact_id=artifact_id,
        logical_path=path,
        media_type=media_type,
        path_evidence=EvidenceStrength.INFERRED,
        type_evidence=EvidenceStrength.INFERRED,
        structure_coverage=structure,
        untainted_content_coverage=content,
        pre_mutation_coverage=pre_mutation,
        provenance_confidence=1.0 if mode is ReconstructionMode.TRACE_RICH else 0.0,
        truncation="NONE",
        criticality=criticality,
        blocking_uncertainties=(
            ("artifact-evidence/test-blocker",) if mode is ReconstructionMode.BLOCKED else ()
        ),
        evidence=evidence or (_evidence("requirement"),),
        selected_mode=mode,
    )


def _row(
    *,
    context: AttachmentPlanningContextV2 | None = None,
    target: ArtifactEvidenceTargetV2 | None = None,
    mode: ReconstructionMode = ReconstructionMode.PROMPT_ONLY,
    criticality: AttachmentCriticality | None = None,
    audit: ContractAudit | None = None,
) -> ArtifactEvidenceRowV2:
    context = context or _context()
    target = target or _target(
        context=context,
        criticality=criticality or AttachmentCriticality.REQUIRED,
    )
    nested = _r2_row(
        artifact_id=target.artifact_id,
        path=target.logical_path,
        media_type=target.media_type,
        criticality=criticality or target.criticality,
        mode=mode,
        evidence=target.requirement_evidence,
    )
    row = ArtifactEvidenceRowV2(
        artifact_evidence_row_id="artifact-evidence-row://pending",
        attachment_planning_context_ref=attachment_planning_context_ref(context),
        artifact_evidence_target_ref=artifact_evidence_target_ref(target),
        attachment_dependency_id=target.attachment_dependency_id,
        row=nested,
        r2_row_sha256=nested.canonical_sha256(),
        r2_policy_version=R2_POLICY,
        policy_version=MODE_POLICY,
        artifact_evidence_row_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = artifact_evidence_row_carried_sha256(row)
    return row.model_copy(
        update={
            "artifact_evidence_row_id": (f"artifact-evidence-row://sha256/{digest}"),
            "artifact_evidence_row_sha256": digest,
        }
    )


def _matrix(
    *,
    context: AttachmentPlanningContextV2 | None = None,
    rows: tuple[ArtifactEvidenceRowV2, ...] | None = None,
    aggregate_mode: str = "PROMPT_ONLY",
    audit: ContractAudit | None = None,
) -> ArtifactEvidenceMatrixV2:
    context = context or _context()
    values = rows or (_row(context=context),)
    matrix = ArtifactEvidenceMatrixV2(
        artifact_evidence_matrix_id="artifact-evidence-matrix://pending",
        attachment_planning_context_ref=attachment_planning_context_ref(context),
        producer_task_view_ref=context.producer_task_view_ref,
        safe_evidence_bundle_ref=context.safe_evidence_bundle_ref,
        source_r2_matrix_ref=_ref(
            "artifact-evidence-matrix",
            "source-r2",
            version="v1",
            digest=FOURTH_HASH,
        ),
        rows=values,
        aggregate_mode=aggregate_mode,
        r2_policy_version=R2_POLICY,
        policy_version=MODE_POLICY,
        artifact_evidence_matrix_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = artifact_evidence_matrix_carried_sha256(matrix)
    return matrix.model_copy(
        update={
            "artifact_evidence_matrix_id": (f"artifact-evidence-matrix://sha256/{digest}"),
            "artifact_evidence_matrix_sha256": digest,
        }
    )


def test_artifact_evidence_target_v2_is_strict_sorted_and_carried() -> None:
    context = _context()
    target = _target(context=context)

    assert target.attachment_planning_context_ref == (attachment_planning_context_ref(context))
    assert target.artifact_evidence_target_sha256 == (artifact_evidence_target_carried_sha256(target))
    assert artifact_evidence_target_ref(target).object_version == "v2"
    assert set(ArtifactEvidenceTargetV2.model_fields) == {
        "schema_version",
        "artifact_evidence_target_id",
        "attachment_planning_context_ref",
        "attachment_dependency_id",
        "artifact_id",
        "logical_path",
        "media_type",
        "criticality",
        "requirement_evidence",
        "candidate_source_refs",
        "policy_version",
        "artifact_evidence_target_sha256",
        "audit",
    }

    with pytest.raises(ValidationError, match="requirement evidence"):
        _target(context=context, evidence=(_evidence("z"), _evidence("a")))
    with pytest.raises(ValidationError, match="candidate source"):
        _target(
            context=context,
            candidate_source_refs=(
                _ref("file-version", "z", version="v1"),
                _ref("file-version", "a", version="v1"),
            ),
        )
    with pytest.raises(ValidationError):
        ArtifactEvidenceTargetV2.model_validate(
            {
                **target.model_dump(mode="json"),
                "provider_preference": ["forbidden"],
            }
        )


def test_artifact_evidence_row_v2_wraps_exact_r2_row_and_ref() -> None:
    context = _context()
    target = _target(context=context)
    row = _row(context=context, target=target)

    assert row.r2_row_sha256 == row.row.canonical_sha256()
    assert row.artifact_evidence_row_sha256 == (artifact_evidence_row_carried_sha256(row))
    assert artifact_evidence_row_ref(row).object_version == "v2"
    with pytest.raises(ValidationError, match="R2 row hash"):
        ArtifactEvidenceRowV2.model_validate(
            {
                **row.model_dump(mode="python"),
                "r2_row_sha256": OTHER_HASH,
            }
        )
    with pytest.raises(ValidationError):
        ArtifactEvidenceRowV2.model_validate(
            {
                **row.model_dump(mode="json"),
                "runtime_preference": ["forbidden"],
            }
        )


@pytest.mark.parametrize(
    ("modes", "criticalities", "aggregate"),
    [
        pytest.param(
            (ReconstructionMode.TRACE_RICH,),
            (AttachmentCriticality.REQUIRED,),
            "TRACE_RICH",
            id="common-trace-rich",
        ),
        pytest.param(
            (
                ReconstructionMode.TRACE_RICH,
                ReconstructionMode.SKELETON_GUIDED,
                ReconstructionMode.PROMPT_ONLY,
            ),
            (
                AttachmentCriticality.REQUIRED,
                AttachmentCriticality.REQUIRED,
                AttachmentCriticality.OPTIONAL,
            ),
            "MIXED",
            id="heterogeneous-nonblocked",
        ),
        pytest.param(
            (
                ReconstructionMode.TRACE_RICH,
                ReconstructionMode.BLOCKED,
            ),
            (
                AttachmentCriticality.REQUIRED,
                AttachmentCriticality.OPTIONAL,
            ),
            "BLOCKED",
            id="optional-blocked-dominates",
        ),
    ],
)
def test_artifact_evidence_matrix_v2_enforces_r5_aggregate(
    modes: tuple[ReconstructionMode, ...],
    criticalities: tuple[AttachmentCriticality, ...],
    aggregate: str,
) -> None:
    context = _context()
    rows = []
    for index, (mode, criticality) in enumerate(zip(modes, criticalities, strict=True)):
        target = _target(
            context=context,
            dependency_id=f"attachment-dependency://mode/{index}",
            artifact_id=f"artifact://mode/{index}",
            path=f"input-{index}.txt",
            criticality=criticality,
            evidence=(_evidence(f"requirement-{index}"),),
        )
        rows.append(
            _row(
                context=context,
                target=target,
                mode=mode,
                criticality=criticality,
            )
        )
    matrix = _matrix(
        context=context,
        rows=tuple(rows),
        aggregate_mode=aggregate,
    )

    assert matrix.aggregate_mode == aggregate
    assert matrix.artifact_evidence_matrix_sha256 == (artifact_evidence_matrix_carried_sha256(matrix))
    assert artifact_evidence_matrix_ref(matrix).object_version == "v2"
    wrong_aggregate = "MIXED" if aggregate != "MIXED" else "BLOCKED"
    with pytest.raises(ValidationError, match="aggregate"):
        ArtifactEvidenceMatrixV2.model_validate(
            {
                **matrix.model_dump(mode="python"),
                "aggregate_mode": wrong_aggregate,
            }
        )


def test_artifact_evidence_matrix_v2_requires_row_policy_alignment() -> None:
    context = _context()
    row = _row(context=context).model_copy(update={"policy_version": "artifact-evidence-mode/stale"})

    with pytest.raises(ValidationError, match="row policy"):
        ArtifactEvidenceMatrixV2.model_validate(
            {
                **_matrix(context=context).model_dump(mode="python"),
                "rows": (row,),
            }
        )

    row = _row(context=context).model_copy(update={"r2_policy_version": "evidence-compilation/stale"})
    with pytest.raises(ValidationError, match="row R2 policy"):
        ArtifactEvidenceMatrixV2.model_validate(
            {
                **_matrix(context=context).model_dump(mode="python"),
                "rows": (row,),
            }
        )


def test_attachment_mode_contract_hashes_ignore_audit_time() -> None:
    first_audit = _audit(datetime(2026, 7, 27, tzinfo=UTC))
    second_audit = _audit(datetime(2026, 7, 28, tzinfo=UTC))
    first_context = _context(audit=first_audit)
    second_context = _context(audit=second_audit)
    first_target = _target(context=first_context, audit=first_audit)
    second_target = _target(context=second_context, audit=second_audit)
    first_row = _row(
        context=first_context,
        target=first_target,
        audit=first_audit,
    )
    second_row = _row(
        context=second_context,
        target=second_target,
        audit=second_audit,
    )
    first = _matrix(
        context=first_context,
        rows=(first_row,),
        audit=first_audit,
    )
    second = _matrix(
        context=second_context,
        rows=(second_row,),
        audit=second_audit,
    )

    assert first_target.artifact_evidence_target_sha256 == (second_target.artifact_evidence_target_sha256)
    assert first_row.artifact_evidence_row_sha256 == (second_row.artifact_evidence_row_sha256)
    assert first.artifact_evidence_matrix_sha256 == (second.artifact_evidence_matrix_sha256)
    assert first.canonical_sha256() != second.canonical_sha256()


def test_attachment_mode_contract_hashes_are_stable_across_python_hash_seed() -> None:
    script = """
from tests.eval_factory.contract.test_v2_attachment_contracts import (
    _context,
    _matrix,
    _row,
    _target,
)

context = _context()
target = _target(context=context)
row = _row(context=context, target=target)
matrix = _matrix(context=context, rows=(row,))
print(
    target.artifact_evidence_target_sha256,
    row.artifact_evidence_row_sha256,
    matrix.artifact_evidence_matrix_sha256,
)
"""
    outputs = []
    for seed in ("1", "99"):
        process = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONPATH": "src",
            },
            timeout=30,
        )
        assert process.returncode == 0, process.stderr
        outputs.append(process.stdout.strip())

    assert len(set(outputs)) == 1


def _prompt_dependency_binding() -> PromptOnlyDependencyEvidenceBindingV2:
    return PromptOnlyDependencyEvidenceBindingV2(
        attachment_dependency_id="attachment-dependency://mode/input",
        evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
        evidence=(_evidence("prompt-only-requirement"),),
    )


def _prompt_dependency_context(
    *,
    context: AttachmentPlanningContextV2 | None = None,
    audit: ContractAudit | None = None,
) -> PromptOnlyDependencyPlanningContextV2:
    context = context or _context(audit=audit)
    value = PromptOnlyDependencyPlanningContextV2(
        prompt_only_dependency_planning_context_id=("prompt-only-dependency-planning-context://pending"),
        attachment_planning_context_ref=attachment_planning_context_ref(context),
        producer_task_view_ref=context.producer_task_view_ref,
        task_prompt_safety_gate_ref=_ref(
            "task-prompt-safety-gate",
            "prompt-only",
            version="v2",
        ),
        source_task_draft_sha256=context.source_task_draft_sha256,
        source_trace_id=context.source_trace_id,
        dependency_evidence_bindings=(_prompt_dependency_binding(),),
        policy_version="prompt-only-dependency-planning/r5-03-v1",
        prompt_only_dependency_planning_context_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = prompt_only_dependency_planning_context_carried_sha256(value)
    return value.model_copy(
        update={
            "prompt_only_dependency_planning_context_id": (
                f"prompt-only-dependency-planning-context://sha256/{digest}"
            ),
            "prompt_only_dependency_planning_context_sha256": digest,
        }
    )


def _prompt_dependency_discovery(
    *,
    context: AttachmentPlanningContextV2 | None = None,
    planning: PromptOnlyDependencyPlanningContextV2 | None = None,
    audit: ContractAudit | None = None,
) -> PromptOnlyDependencyDiscoveryV2:
    context = context or _context(audit=audit)
    planning = planning or _prompt_dependency_context(
        context=context,
        audit=audit,
    )
    target = _target(
        context=context,
        evidence=planning.dependency_evidence_bindings[0].evidence,
        audit=audit,
    )
    target_ref = artifact_evidence_target_ref(target)
    value = PromptOnlyDependencyDiscoveryV2(
        prompt_only_dependency_discovery_id=("prompt-only-dependency-discovery://pending"),
        attachment_planning_context_ref=attachment_planning_context_ref(context),
        producer_task_view_ref=context.producer_task_view_ref,
        dependency_planning_context_ref=(prompt_only_dependency_planning_context_ref(planning)),
        prompt_boundary_enforcement_ref=_ref(
            "prompt-boundary-enforcement",
            "prompt-only",
            version="prompt-injection-as-data/r2-07-v1",
        ),
        targets=(target,),
        existing_target_refs=(),
        discovered_target_refs=(target_ref,),
        deterministic_dependency_ids=(target.attachment_dependency_id,),
        semantic_dependency_ids=(),
        semantic_evaluated=False,
        model_profile=None,
        prompt_version=None,
        policy_version="prompt-only-dependency/r5-03-v1",
        prompt_only_dependency_discovery_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = prompt_only_dependency_discovery_carried_sha256(value)
    return value.model_copy(
        update={
            "prompt_only_dependency_discovery_id": (f"prompt-only-dependency-discovery://sha256/{digest}"),
            "prompt_only_dependency_discovery_sha256": digest,
        }
    )


def test_prompt_only_dependency_sidecar_is_closed_sorted_and_carried() -> None:
    planning = _prompt_dependency_context()

    assert set(PromptOnlyDependencyEvidenceBindingV2.model_fields) == {
        "schema_version",
        "attachment_dependency_id",
        "evidence_priority",
        "evidence",
    }
    assert set(PromptOnlyDependencyPlanningContextV2.model_fields) == {
        "schema_version",
        "prompt_only_dependency_planning_context_id",
        "attachment_planning_context_ref",
        "producer_task_view_ref",
        "task_prompt_safety_gate_ref",
        "source_task_draft_sha256",
        "source_trace_id",
        "dependency_evidence_bindings",
        "policy_version",
        "prompt_only_dependency_planning_context_sha256",
        "audit",
    }
    assert planning.prompt_only_dependency_planning_context_sha256 == (
        prompt_only_dependency_planning_context_carried_sha256(planning)
    )
    assert prompt_only_dependency_planning_context_ref(planning).object_version == "v2"

    with pytest.raises(ValidationError, match="evidence"):
        PromptOnlyDependencyEvidenceBindingV2(
            attachment_dependency_id="attachment-dependency://mode/input",
            evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
            evidence=(),
        )
    with pytest.raises(ValidationError, match="unsafe or incomplete"):
        PromptOnlyDependencyEvidenceBindingV2(
            attachment_dependency_id="attachment-dependency://mode/input",
            evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
            evidence=(_evidence("negative").model_copy(update={"polarity": EvidencePolarity.NEGATIVE}),),
        )
    with pytest.raises(ValidationError, match="unsafe or incomplete"):
        PromptOnlyDependencyEvidenceBindingV2(
            attachment_dependency_id="attachment-dependency://mode/input",
            evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
            evidence=(_evidence("incomplete").model_copy(update={"capability_complete": False}),),
        )
    with pytest.raises(ValidationError):
        PromptOnlyDependencyPlanningContextV2.model_validate(
            {
                **planning.model_dump(mode="json"),
                "query_instruction": "forbidden",
            }
        )


def test_prompt_only_dependency_discovery_is_complete_closed_and_carried() -> None:
    discovery = _prompt_dependency_discovery()

    assert set(PromptOnlyDependencyDiscoveryV2.model_fields) == {
        "schema_version",
        "prompt_only_dependency_discovery_id",
        "attachment_planning_context_ref",
        "producer_task_view_ref",
        "dependency_planning_context_ref",
        "prompt_boundary_enforcement_ref",
        "targets",
        "existing_target_refs",
        "discovered_target_refs",
        "deterministic_dependency_ids",
        "semantic_dependency_ids",
        "semantic_evaluated",
        "model_profile",
        "prompt_version",
        "policy_version",
        "prompt_only_dependency_discovery_sha256",
        "audit",
    }
    assert discovery.prompt_only_dependency_discovery_sha256 == (
        prompt_only_dependency_discovery_carried_sha256(discovery)
    )
    assert prompt_only_dependency_discovery_ref(discovery).object_version == "v2"
    assert discovery.existing_target_refs == ()
    assert discovery.discovered_target_refs == (artifact_evidence_target_ref(discovery.targets[0]),)

    with pytest.raises(ValidationError, match="semantic"):
        PromptOnlyDependencyDiscoveryV2.model_validate(
            {
                **discovery.model_dump(mode="python"),
                "semantic_evaluated": True,
            }
        )
    with pytest.raises(ValidationError):
        PromptOnlyDependencyDiscoveryV2.model_validate(
            {
                **discovery.model_dump(mode="json"),
                "artifact_content": "forbidden",
            }
        )


def test_prompt_only_dependency_hashes_ignore_audit_time() -> None:
    first_audit = _audit(datetime(2026, 7, 27, tzinfo=UTC))
    second_audit = _audit(datetime(2026, 7, 28, tzinfo=UTC))
    first_context = _context(audit=first_audit)
    second_context = _context(audit=second_audit)
    first_planning = _prompt_dependency_context(
        context=first_context,
        audit=first_audit,
    )
    second_planning = _prompt_dependency_context(
        context=second_context,
        audit=second_audit,
    )
    first = _prompt_dependency_discovery(
        context=first_context,
        planning=first_planning,
        audit=first_audit,
    )
    second = _prompt_dependency_discovery(
        context=second_context,
        planning=second_planning,
        audit=second_audit,
    )

    assert first_planning.prompt_only_dependency_planning_context_sha256 == (
        second_planning.prompt_only_dependency_planning_context_sha256
    )
    assert first.prompt_only_dependency_discovery_sha256 == (second.prompt_only_dependency_discovery_sha256)
    assert first.canonical_sha256() != second.canonical_sha256()


def _public_source_policy(
    *,
    audit: ContractAudit | None = None,
    **overrides: object,
) -> PublicSourceRetrievalPolicyV2:
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
            version="v1",
        ),
        "network_policy_ref": _ref(
            "network-policy",
            "public-source",
            version="v1",
        ),
        "source_usage_policy_ref": _ref(
            "source-usage-policy",
            "public-source",
            version="v1",
        ),
        "safety_scan_policy_ref": _ref(
            "safety-scan-policy",
            "public-source",
            version="v1",
        ),
        "license_policy_ref": _ref(
            "source-license-policy",
            "public-source",
            version="v1",
        ),
        "policy_version": "public-source-retrieval/r5-04-v1",
        "public_source_retrieval_policy_sha256": HASH,
        "audit": audit or _audit(),
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


def _public_source_safety_assessment(
    *,
    audit: ContractAudit | None = None,
    **overrides: object,
) -> PublicSourceSafetyAssessmentV2:
    content_ref = _ref(
        "public-source-content",
        "content",
        version="v1",
        digest=OTHER_HASH,
    )
    values: dict[str, object] = {
        "public_source_safety_assessment_id": ("public-source-safety-assessment://pending"),
        "fetch_result_ref": _ref(
            "public-source-fetch-result",
            "fetch",
            version="v2",
            digest=THIRD_HASH,
        ),
        "content_ref": content_ref,
        "content_sha256": OTHER_HASH,
        "secret_scan_ref": _ref(
            "secret-scan-result",
            "content",
            version="v1",
            digest=OTHER_HASH,
        ),
        "pii_scan_ref": _ref(
            "configured-pii-scan-result",
            "content",
            version="v1",
            digest=OTHER_HASH,
        ),
        "prompt_injection_scan_ref": _ref(
            "prompt-injection-scan-result",
            "content",
            version="v1",
            digest=OTHER_HASH,
        ),
        "answer_leakage_scan_ref": _ref(
            "answer-leakage-scan-result",
            "content",
            version="v1",
            digest=OTHER_HASH,
        ),
        "license_assessment_ref": _ref(
            "source-license-assessment",
            "content",
            version="v1",
            digest=OTHER_HASH,
        ),
        "passed": True,
        "policy_version": "public-source-retrieval/r5-04-v1",
        "public_source_safety_assessment_sha256": HASH,
        "audit": audit or _audit(),
    }
    values.update(overrides)
    assessment = PublicSourceSafetyAssessmentV2(**values)
    digest = public_source_safety_assessment_carried_sha256(assessment)
    return assessment.model_copy(
        update={
            "public_source_safety_assessment_id": (f"public-source-safety-assessment://sha256/{digest}"),
            "public_source_safety_assessment_sha256": digest,
        }
    )


def _source_evidence_contract(
    *,
    audit: ContractAudit | None = None,
    **overrides: object,
) -> SourceEvidenceV2:
    audit = audit or _audit()
    context = _context(audit=audit)
    discovery = _prompt_dependency_discovery(context=context, audit=audit)
    target = discovery.targets[0]
    policy = _public_source_policy(audit=audit)
    safety = _public_source_safety_assessment(audit=audit)
    content_ref = safety.content_ref
    provenance = ProvenanceDecision(
        provenance_decision_id="provenance-decision://public-source/content",
        subject_ref=content_ref,
        origin_class=OriginClass.SYSTEM_OR_HARNESS_CONTEXT,
        taint_labels=frozenset(),
        content_risk_labels=frozenset(),
        visibility=Visibility.STAGE_PROJECTION,
        disposition=Disposition.ALLOW_INPUT_EVIDENCE,
        derived_from=(
            safety.fetch_result_ref,
            public_source_safety_assessment_ref(safety),
        ),
        rule_ids=("public-source-refetch/r5-04-v1",),
        source_event_refs=(),
        confidence=1.0,
        review_required=False,
        policy_version="public-source-refetch/r5-04-v1",
        subject_sha256=safety.content_sha256,
        audit=audit,
    )
    provenance_ref = ObjectRef(
        object_type="provenance-decision",
        object_id=provenance.provenance_decision_id,
        object_version="v1",
        object_sha256=provenance.canonical_sha256(),
    )
    claim = SourceEvidenceClaimBindingV2(
        artifact_evidence_target_ref=artifact_evidence_target_ref(target),
        attachment_dependency_id=target.attachment_dependency_id,
        supported_claim_ids=(target.attachment_dependency_id,),
        usage_basis="FACTS_ONLY",
    )
    frozen = SourceEvidence(
        source_evidence_id="source-evidence://public-source/content",
        source_uri="https://example.gov/source.txt",
        retrieved_at="2026-07-28T00:00:00+00:00",
        retrieval_policy_version=policy.policy_version,
        usage_basis="FACTS_ONLY",
        content_ref=content_ref,
        content_sha256=safety.content_sha256,
        supported_claim_ids=claim.supported_claim_ids,
        provenance_decision_ref=provenance_ref,
        audit=audit,
    )
    values: dict[str, object] = {
        "source_evidence_id": "source-evidence-v2://pending",
        "attachment_planning_context_ref": attachment_planning_context_ref(context),
        "producer_task_view_ref": discovery.producer_task_view_ref,
        "prompt_only_dependency_discovery_ref": (prompt_only_dependency_discovery_ref(discovery)),
        "retrieval_policy_ref": public_source_retrieval_policy_ref(policy),
        "external_lead_ref": None,
        "external_lead_decision_ref": None,
        "search_result_ref": _ref(
            "public-source-search-result",
            "search",
            version="v2",
        ),
        "fetch_result_ref": safety.fetch_result_ref,
        "safety_assessment_ref": public_source_safety_assessment_ref(safety),
        "claim_bindings": (claim,),
        "source_evidence_v1": frozen,
        "content_provenance_decision": provenance,
        "policy_version": policy.policy_version,
        "source_evidence_sha256": HASH,
        "audit": audit,
    }
    values.update(overrides)
    value = SourceEvidenceV2(**values)
    digest = source_evidence_v2_carried_sha256(value)
    return value.model_copy(
        update={
            "source_evidence_id": f"source-evidence-v2://sha256/{digest}",
            "source_evidence_sha256": digest,
        }
    )


def test_public_source_retrieval_policy_is_strict_sorted_and_carried() -> None:
    first = _public_source_policy(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))
    second = _public_source_policy(audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)))

    assert first.public_source_retrieval_policy_sha256 == (
        public_source_retrieval_policy_carried_sha256(first)
    )
    assert first.public_source_retrieval_policy_sha256 == (second.public_source_retrieval_policy_sha256)
    assert public_source_retrieval_policy_ref(first).object_version == "v2"
    assert first.canonical_sha256() != second.canonical_sha256()

    with pytest.raises(ValidationError, match=r"sorted|unique"):
        _public_source_policy(
            approved_fetch_provider_ids=(
                "fetch-provider://z",
                "fetch-provider://a",
            )
        )
    with pytest.raises(ValidationError):
        PublicSourceRetrievalPolicyV2.model_validate(
            {
                **first.model_dump(mode="json"),
                "credential": "forbidden",
            }
        )


def test_public_source_safety_assessment_requires_complete_hash_bound_checks() -> None:
    assessment = _public_source_safety_assessment()

    assert assessment.public_source_safety_assessment_sha256 == (
        public_source_safety_assessment_carried_sha256(assessment)
    )
    assert public_source_safety_assessment_ref(assessment).object_version == "v2"
    assert assessment.passed is True

    with pytest.raises(ValidationError, match="content"):
        _public_source_safety_assessment(content_sha256=THIRD_HASH)
    with pytest.raises(ValidationError):
        PublicSourceSafetyAssessmentV2.model_validate(
            {
                **assessment.model_dump(mode="json"),
                "finding_text": "forbidden",
            }
        )


def test_source_evidence_v2_requires_exact_one_retrieval_route_and_fresh_provenance() -> None:
    evidence = _source_evidence_contract()

    assert evidence.source_evidence_sha256 == (source_evidence_v2_carried_sha256(evidence))
    assert source_evidence_v2_ref(evidence).object_version == "v2"
    assert evidence.content_provenance_decision.origin_class is (OriginClass.SYSTEM_OR_HARNESS_CONTEXT)
    assert evidence.content_provenance_decision.disposition is (Disposition.ALLOW_INPUT_EVIDENCE)
    assert evidence.content_provenance_decision.rule_ids == ("public-source-refetch/r5-04-v1",)

    with pytest.raises(ValidationError, match="route"):
        _source_evidence_contract(
            external_lead_ref=_ref(
                "external-source-projection",
                "lead",
                version="evidence-views/r2-05-v1",
            ),
            external_lead_decision_ref=_ref(
                "provenance-decision",
                "lead",
                version="v1",
            ),
        )
    with pytest.raises(ValidationError):
        SourceEvidenceV2.model_validate(
            {
                **evidence.model_dump(mode="json"),
                "retrieved_content": "forbidden",
            }
        )


def test_source_evidence_v2_hash_ignores_nested_audit_actor_time() -> None:
    first = _source_evidence_contract(audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)))
    second = _source_evidence_contract(audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)))

    assert first.source_evidence_sha256 == second.source_evidence_sha256
    assert first.source_evidence_id == second.source_evidence_id
    assert (
        first.source_evidence_v1.provenance_decision_ref.object_sha256
        != second.source_evidence_v1.provenance_decision_ref.object_sha256
    )
    assert first.canonical_sha256() != second.canonical_sha256()


def test_source_evidence_v2_hash_ignores_lead_decision_audit_hash() -> None:
    lead_ref = _ref(
        "external-source-projection",
        "lead",
        version="evidence-views/r2-05-v1",
    )
    first = _source_evidence_contract(
        external_lead_ref=lead_ref,
        external_lead_decision_ref=_ref(
            "provenance-decision",
            "lead",
            version="evidence-views/r2-05-v1",
            digest=HASH,
        ),
        search_result_ref=None,
    )
    second = _source_evidence_contract(
        external_lead_ref=lead_ref,
        external_lead_decision_ref=_ref(
            "provenance-decision",
            "lead",
            version="evidence-views/r2-05-v1",
            digest=OTHER_HASH,
        ),
        search_result_ref=None,
    )

    assert first.source_evidence_sha256 == second.source_evidence_sha256
    assert first.canonical_sha256() != second.canonical_sha256()


def test_source_evidence_set_rejects_partial_or_overlapping_target_partition() -> None:
    evidence = _source_evidence_contract()
    covered = tuple(binding.artifact_evidence_target_ref for binding in evidence.claim_bindings)
    request_refs = tuple(
        sorted(
            (
                _ref(
                    "public-source-fetch-request",
                    "fetch",
                    version="v2",
                ),
                _ref(
                    "public-source-search-request",
                    "search",
                    version="v2",
                ),
            ),
            key=lambda ref: (
                ref.object_type,
                ref.object_id,
                ref.object_version,
                ref.object_sha256,
            ),
        )
    )
    value = SourceEvidenceSetV2(
        source_evidence_set_id="source-evidence-set://pending",
        attachment_planning_context_ref=(evidence.attachment_planning_context_ref),
        producer_task_view_ref=evidence.producer_task_view_ref,
        prompt_only_dependency_discovery_ref=(evidence.prompt_only_dependency_discovery_ref),
        retrieval_policy_ref=evidence.retrieval_policy_ref,
        retrieval_request_refs=request_refs,
        source_evidence=(evidence,),
        covered_target_refs=covered,
        not_required_target_refs=(),
        policy_version=evidence.policy_version,
        source_evidence_set_sha256=HASH,
        audit=_audit(),
    )
    digest = source_evidence_set_carried_sha256(value)
    value = value.model_copy(
        update={
            "source_evidence_set_id": (f"source-evidence-set://sha256/{digest}"),
            "source_evidence_set_sha256": digest,
        }
    )

    assert value.source_evidence_set_sha256 == (source_evidence_set_carried_sha256(value))
    with pytest.raises(ValidationError, match="disjoint"):
        SourceEvidenceSetV2.model_validate(
            {
                **value.model_dump(mode="python"),
                "not_required_target_refs": covered,
            }
        )


def _routing_policy(**overrides: object) -> ArtifactRoutingPolicyV2:
    values: dict[str, object] = {
        "artifact_routing_policy_id": "artifact-routing-policy://pending",
        "deterministic_provider_first": True,
        "approved_provider_ids": ("text-provider",),
        "runtime_order": (
            "claude_agent_sdk",
            "claude_code_cli",
            "pi_rpc",
        ),
        "runtime_model_profile_refs": (
            _ref("model-profile", "claude-agent-sdk", version="v1"),
            _ref("model-profile", "claude-code-cli", version="v1"),
            _ref("model-profile", "pi-rpc", version="v1"),
        ),
        "provider_capability_policy_ref": _ref(
            "provider-capability-policy",
            "current",
            version="v1",
        ),
        "runtime_capability_policy_ref": _ref(
            "runtime-capability-policy",
            "current",
            version="v1",
        ),
        "validator_policy_ref": _ref(
            "validator-policy",
            "current",
            version="v1",
        ),
        "silent_degradation_allowed": False,
        "policy_version": "artifact-routing/r5-05-v1",
        "artifact_routing_policy_sha256": HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    policy = ArtifactRoutingPolicyV2(**values)
    digest = artifact_routing_policy_carried_sha256(policy)
    return policy.model_copy(
        update={
            "artifact_routing_policy_id": f"artifact-routing-policy://sha256/{digest}",
            "artifact_routing_policy_sha256": digest,
        }
    )


def _build_contract(**overrides: object) -> ArtifactBuildContractV2:
    values: dict[str, object] = {
        "artifact_build_contract_id": "artifact-build-contract://pending",
        "attachment_planning_context_ref": _ref(
            "attachment-planning-context",
            "current",
            version="v2",
        ),
        "producer_task_view_ref": _ref(
            "producer-task-view",
            "current",
            version="v2",
        ),
        "artifact_evidence_matrix_ref": _ref(
            "artifact-evidence-matrix",
            "current",
            version="v2",
        ),
        "artifact_evidence_row_ref": _ref(
            "artifact-evidence-row",
            "current",
            version="v2",
        ),
        "artifact_evidence_target_ref": _ref(
            "artifact-evidence-target",
            "current",
            version="v2",
        ),
        "attachment_dependency_id": "attachment-dependency://routing/input",
        "artifact_id": "artifact://routing/input",
        "logical_path": "inputs/source.txt",
        "media_type": "text/plain",
        "asset_type": "txt",
        "mode": ReconstructionMode.PROMPT_ONLY,
        "criticality": AttachmentCriticality.REQUIRED,
        "content_contract_ref": _ref(
            "artifact-content-contract",
            "input",
            version="v2",
        ),
        "render_contract_ref": _ref(
            "artifact-render-contract",
            "input",
            version="v2",
        ),
        "provider_payload_ref": _ref(
            "attachment-provider-payload",
            "input",
            version="v2",
        ),
        "source_evidence_set_ref": None,
        "required_provider_capability_ids": ("attachment-provider/generate/txt/v1",),
        "required_runtime_tools": ("ls", "write"),
        "runtime_role": "attachment-writer",
        "runtime_resume_required": True,
        "validator_ids": ("secret-validator", "text-validator"),
        "policy_version": "artifact-routing/r5-05-v1",
        "artifact_build_contract_sha256": HASH,
        "audit": _audit(),
    }
    values.update(overrides)
    contract = ArtifactBuildContractV2(**values)
    digest = artifact_build_contract_carried_sha256(contract)
    return contract.model_copy(
        update={
            "artifact_build_contract_id": f"artifact-build-contract://sha256/{digest}",
            "artifact_build_contract_sha256": digest,
        }
    )


def test_artifact_routing_policy_is_provider_first_closed_and_carried() -> None:
    policy = _routing_policy()
    shifted = policy.model_copy(
        update={
            "audit": _audit(datetime(2026, 7, 29, tzinfo=UTC)),
        }
    )

    assert policy.runtime_order == (
        "claude_agent_sdk",
        "claude_code_cli",
        "pi_rpc",
    )
    assert policy.artifact_routing_policy_sha256 == artifact_routing_policy_carried_sha256(policy)
    assert artifact_routing_policy_carried_sha256(shifted) == (policy.artifact_routing_policy_sha256)
    assert shifted.canonical_sha256() != policy.canonical_sha256()
    assert artifact_routing_policy_ref(policy).object_version == "v2"
    for field in (
        "deterministic_provider_first",
        "runtime_order",
        "runtime_model_profile_refs",
        "silent_degradation_allowed",
    ):
        assert field in type(policy).model_fields

    with pytest.raises(ValidationError):
        ArtifactRoutingPolicyV2.model_validate(
            {
                **policy.model_dump(mode="python"),
                "runtime_command": "must-not-cross",
            }
        )
    with pytest.raises(ValidationError, match="runtime order"):
        _routing_policy(
            runtime_order=("pi_rpc", "claude_agent_sdk", "claude_code_cli"),
        )


def test_artifact_build_contract_is_complete_closed_and_carried() -> None:
    contract = _build_contract()
    shifted = contract.model_copy(
        update={
            "audit": _audit(datetime(2026, 7, 29, tzinfo=UTC)),
        }
    )

    assert contract.asset_type == "txt"
    assert contract.required_runtime_tools == ("ls", "write")
    assert contract.artifact_build_contract_sha256 == artifact_build_contract_carried_sha256(contract)
    assert artifact_build_contract_carried_sha256(shifted) == (contract.artifact_build_contract_sha256)
    assert shifted.canonical_sha256() != contract.canonical_sha256()
    assert artifact_build_contract_ref(contract).object_version == "v2"
    assert ArtifactBuildSpecV2 is not None
    assert ArtifactRoutePlanEntryV2 is not None
    assert ArtifactRoutingPlanV2 is not None
    assert ArtifactBuildSpec is not None

    with pytest.raises(ValidationError):
        ArtifactBuildContractV2.model_validate(
            {
                **contract.model_dump(mode="python"),
                "provider_payload": {"secret": "must-not-cross"},
            }
        )
    with pytest.raises(ValidationError, match="sorted"):
        _build_contract(required_runtime_tools=("write", "ls"))
