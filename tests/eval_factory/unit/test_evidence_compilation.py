from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.attachment import EvidenceCoverage, ReconstructionMode
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    TypedAttribute,
    VersionBinding,
)
from eval_factory.contracts.safety import Disposition, OriginClass, ProvenanceDecision, Visibility
from eval_factory.contracts.task import AttachmentCriticality
from eval_factory.contracts.trace import Completeness, FileObservation, FileOperation
from eval_factory.provenance import (
    ArtifactEvidenceMatrixCompiler,
    ArtifactEvidenceMatrixCompileRequest,
    ArtifactEvidenceTarget,
    EvidenceBundleCompiler,
    EvidenceBundleCompileRequest,
    EvidenceCompilationPolicyError,
    EvidenceCompilationUncertainty,
    EvidenceViewEngine,
    EvidenceViewPrincipal,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewRequest,
    EvidenceViewSubject,
    FileVersionTimelineBuilder,
    artifact_evidence_matrix_carried_sha256,
    artifact_evidence_matrix_ref,
    validate_artifact_evidence_matrix_identity,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
SOURCE_TRACE_ID = "source-trace://evidence-compilation"
TRACE_ID = "trace-ir://evidence-compilation"


def _audit(created_at: datetime = datetime(2026, 7, 24, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="evidence-compilation-test",
        governing_versions=(VersionBinding(component="evidence-compilation", version="r2-06"),),
    )


def _ref(object_type: str, object_id: str, digest: str = HASH) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v1",
        object_sha256=digest,
    )


def _span(span_id: str = "source-span://safe") -> SourceSpanRef:
    return SourceSpanRef(span_id=span_id, source_trace_id=SOURCE_TRACE_ID, raw_sha256=HASH)


def _decision(
    *,
    subject: ObjectRef,
    disposition: Disposition = Disposition.ALLOW_INPUT_EVIDENCE,
    visibility: Visibility = Visibility.STAGE_PROJECTION,
    confidence: float = 1.0,
) -> ProvenanceDecision:
    return ProvenanceDecision(
        provenance_decision_id=f"provenance-decision://{subject.object_id.rsplit('/', 1)[-1]}",
        subject_ref=subject,
        origin_class=OriginClass.PREEXISTING_WORKSPACE_INPUT,
        visibility=visibility,
        disposition=disposition,
        rule_ids=("evidence-compilation-parent/v1",),
        source_event_refs=(_ref("trace-event", "trace-event://evidence-compilation"),),
        confidence=confidence,
        review_required=disposition in {Disposition.NEEDS_REVIEW, Disposition.QUARANTINE, Disposition.REJECT},
        policy_version="evidence-compilation-parent/test-v1",
        subject_sha256=subject.object_sha256,
        audit=_audit(),
    )


def _principal() -> EvidenceViewPrincipal:
    return EvidenceViewPrincipal(
        principal_id="principal://attachment-producer",
        principal_type=EvidenceViewPrincipalType.ATTACHMENT_PRODUCER,
        allowed_purposes=frozenset({EvidenceViewPurpose.ATTACHMENT_PRODUCTION}),
        max_subjects=10,
        max_characters=1000,
    )


def _view_subject(
    *,
    source_ref: ObjectRef,
    disposition: Disposition = Disposition.ALLOW_INPUT_EVIDENCE,
    text: str | None = "safe input content",
    structure: tuple[TypedAttribute, ...] = (),
    confidence: float = 1.0,
) -> EvidenceViewSubject:
    return EvidenceViewSubject(
        subject_ref=source_ref,
        decision=_decision(subject=source_ref, disposition=disposition, confidence=confidence),
        projection_text=text,
        structure_fields=structure,
    )


def _view(*subjects: EvidenceViewSubject):
    return EvidenceViewEngine().project(
        EvidenceViewRequest(
            principal=_principal(),
            purpose=EvidenceViewPurpose.ATTACHMENT_PRODUCTION,
            subjects=subjects,
            max_characters=1000,
            audit=_audit(),
        )
    )


def _binding(item_id: str, *, complete: bool = True):
    from eval_factory.provenance import ProjectionEvidenceBinding

    return ProjectionEvidenceBinding(
        projection_item_id=item_id,
        source_spans=(_span(),),
        polarity=EvidencePolarity.POSITIVE,
        capability="trace-evidence",
        capability_complete=complete,
    )


def _compile_bundle(view_result, *bindings):
    return EvidenceBundleCompiler().compile(
        EvidenceBundleCompileRequest(
            source_trace_id=SOURCE_TRACE_ID,
            trace_ir_version_id=TRACE_ID,
            consumer_stage="attachment-producer",
            purpose="attachment-production",
            view_result=view_result,
            bindings=bindings,
            max_characters=1000,
            audit=_audit(),
        )
    )


def _requirement_evidence(artifact_id: str = "artifact://input") -> EvidenceRef:
    subject = _ref("requirement", f"requirement://{artifact_id}", THIRD_HASH)
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://requirement/{artifact_id.rsplit('/', 1)[-1]}",
        subject_ref=subject,
        source_spans=(_span(f"source-span://requirement/{artifact_id.rsplit('/', 1)[-1]}"),),
        polarity=EvidencePolarity.POSITIVE,
        capability="task-requirement",
        capability_complete=True,
    )


def _target(
    *,
    artifact_id: str = "artifact://input",
    path: str = "input.txt",
    media_type: str = "text/plain",
    criticality: AttachmentCriticality = AttachmentCriticality.REQUIRED,
    candidates: tuple[ObjectRef, ...],
) -> ArtifactEvidenceTarget:
    return ArtifactEvidenceTarget(
        artifact_id=artifact_id,
        logical_path=path,
        media_type=media_type,
        criticality=criticality,
        requirement_evidence=(_requirement_evidence(artifact_id),),
        candidate_source_refs=candidates,
    )


def _observation(
    *,
    path: str = "input.txt",
    operation: FileOperation = FileOperation.READ,
    sequence: int = 1,
    completeness: Completeness = Completeness.COMPLETE,
    truncated: bool | None = False,
    content: bool = True,
) -> FileObservation:
    content_ref = _ref("content-blob", f"content://{path}/{sequence}") if content else None
    return FileObservation(
        observation_id=f"file-observation://{path.replace('/', '-')}/{sequence}",
        trace_ir_version_id=TRACE_ID,
        logical_path=path,
        raw_path_ref=_ref("content-blob", f"raw-path://{path}"),
        operation=operation,
        sequence=sequence,
        observed_start=None,
        observed_end=None,
        completeness=completeness,
        truncated=truncated,
        content_ref=content_ref,
        content_sha256=content_ref.object_sha256 if content_ref is not None else None,
        file_version_id=f"file-version://r1/{path.replace('/', '-')}/{sequence}",
        source_event_refs=(_ref("trace-event", f"trace-event://{path}/{sequence}"),),
    )


def _timeline(*observations: FileObservation):
    return FileVersionTimelineBuilder().build(
        trace_ir_version_id=TRACE_ID,
        file_observations=observations,
        audit=_audit(),
    )[0]


def _matrix(bundle_result, view_result, *, targets: tuple[ArtifactEvidenceTarget, ...], timelines=()):
    return ArtifactEvidenceMatrixCompiler().compile(
        ArtifactEvidenceMatrixCompileRequest(
            producer_task_view_ref=_ref("producer-task-view", "producer-task-view://r2-06"),
            evidence_bundle=bundle_result.evidence_bundle,
            view_result=view_result,
            timelines=timelines,
            targets=targets,
            audit=_audit(),
        )
    )


def test_bundle_includes_only_projected_refs_with_exact_source_spans() -> None:
    source_ref = _ref("file-version", "file-version://input")
    view_result = _view(_view_subject(source_ref=source_ref))

    result = _compile_bundle(view_result, _binding(view_result.included_items[0].projection_item_id))

    assert result.evidence_bundle.projection_policy_ref.object_id == (
        view_result.projection_policy.projection_policy_id
    )
    assert [item.subject_ref for item in result.evidence_bundle.evidence] == [
        view_result.included_items[0].projected_ref
    ]
    assert result.evidence_bundle.evidence[0].source_spans == (_span(),)
    assert result.evidence_bundle.excluded_subject_refs == ()
    assert result.evidence_bundle.tainted_content_included is False


def test_missing_span_binding_records_gap_without_fallback_span() -> None:
    source_ref = _ref("file-version", "file-version://missing-span")
    view_result = _view(_view_subject(source_ref=source_ref))

    result = _compile_bundle(view_result)

    assert result.evidence_bundle.evidence == ()
    assert result.evidence_bundle.excluded_subject_refs == (source_ref,)
    assert EvidenceCompilationUncertainty.MISSING_SOURCE_SPAN_BINDING in result.uncertainties
    assert SOURCE_TRACE_ID not in str(result.evidence_bundle.evidence)


def test_view_exclusions_stay_excluded_and_never_become_bundle_evidence() -> None:
    safe_ref = _ref("file-version", "file-version://safe")
    unsafe_ref = _ref("file-version", "file-version://unsafe", OTHER_HASH)
    view_result = _view(
        _view_subject(source_ref=unsafe_ref, disposition=Disposition.QUARANTINE),
        _view_subject(source_ref=safe_ref),
    )

    result = _compile_bundle(view_result, _binding(view_result.included_items[0].projection_item_id))

    assert unsafe_ref in result.evidence_bundle.excluded_subject_refs
    assert all(item.subject_ref != unsafe_ref for item in result.evidence_bundle.evidence)
    assert EvidenceCompilationUncertainty.VIEW_SUBJECT_EXCLUDED in result.uncertainties


def test_duplicate_unknown_and_stale_bundle_bindings_fail_closed() -> None:
    source_ref = _ref("file-version", "file-version://safe")
    view_result = _view(_view_subject(source_ref=source_ref))
    binding = _binding(view_result.included_items[0].projection_item_id)

    with pytest.raises(EvidenceCompilationPolicyError, match="duplicate"):
        _compile_bundle(view_result, binding, binding)
    with pytest.raises(EvidenceCompilationPolicyError, match="unknown projection item"):
        _compile_bundle(view_result, _binding("projection-item://missing"))

    stale_policy = view_result.model_copy(
        update={
            "projection_policy": view_result.projection_policy.model_copy(
                update={"projection_policy_id": "projection-policy://tampered"}
            )
        }
    )
    with pytest.raises(EvidenceCompilationPolicyError, match="projection policy"):
        _compile_bundle(stale_policy, _binding(stale_policy.included_items[0].projection_item_id))


def test_complete_pre_mutation_safe_content_selects_trace_rich() -> None:
    source_ref = _ref("file-version", "file-version://input")
    view_result = _view(_view_subject(source_ref=source_ref, text="safe input content"))
    bundle = _compile_bundle(view_result, _binding(view_result.included_items[0].projection_item_id))

    result = _matrix(
        bundle,
        view_result,
        targets=(_target(candidates=(source_ref,)),),
        timelines=(_timeline(_observation(path="input.txt", operation=FileOperation.READ)),),
    )
    row = result.artifact_evidence_matrix.rows[0]

    assert row.selected_mode is ReconstructionMode.TRACE_RICH
    assert row.pre_mutation_coverage is EvidenceCoverage.COMPLETE
    assert row.untainted_content_coverage is EvidenceCoverage.COMPLETE
    assert result.artifact_evidence_matrix.aggregate_mode == "TRACE_RICH"


def test_structure_content_and_requirement_only_paths_select_expected_modes() -> None:
    content_ref = _ref("file-version", "file-version://content")
    structure_ref = _ref("file-version", "file-version://structure", OTHER_HASH)
    view_result = _view(
        _view_subject(source_ref=content_ref, text="safe content"),
        _view_subject(
            source_ref=structure_ref,
            disposition=Disposition.ALLOW_STRUCTURE_ONLY,
            text=None,
            structure=(
                TypedAttribute(key="normalized-path", value="structure.txt"),
                TypedAttribute(key="media-type", value="text/plain"),
            ),
        ),
    )
    bundle = _compile_bundle(
        view_result,
        _binding(view_result.included_items[0].projection_item_id, complete=False),
        _binding(view_result.included_items[1].projection_item_id),
    )

    result = _matrix(
        bundle,
        view_result,
        targets=(
            _target(artifact_id="artifact://content", path="content.txt", candidates=(content_ref,)),
            _target(artifact_id="artifact://structure", path="structure.txt", candidates=(structure_ref,)),
            _target(
                artifact_id="artifact://requirement",
                path="requirement.txt",
                criticality=AttachmentCriticality.OPTIONAL,
                candidates=(),
            ),
        ),
        timelines=(
            _timeline(
                _observation(
                    path="content.txt",
                    operation=FileOperation.READ,
                    completeness=Completeness.PARTIAL,
                )
            ),
            _timeline(_observation(path="structure.txt", operation=FileOperation.READ, content=False)),
        ),
    )

    modes = {row.artifact_id: row.selected_mode for row in result.artifact_evidence_matrix.rows}
    assert modes == {
        "artifact://content": ReconstructionMode.TRACE_RICH,
        "artifact://structure": ReconstructionMode.SKELETON_GUIDED,
        "artifact://requirement": ReconstructionMode.PROMPT_ONLY,
    }
    assert result.artifact_evidence_matrix.aggregate_mode == "MIXED"


def test_critical_missing_or_unknown_evidence_selects_blocked() -> None:
    source_ref = _ref("file-version", "file-version://first-write")
    view_result = _view(_view_subject(source_ref=source_ref))
    bundle = _compile_bundle(view_result, _binding(view_result.included_items[0].projection_item_id))

    result = _matrix(
        bundle,
        view_result,
        targets=(
            _target(
                criticality=AttachmentCriticality.CRITICAL,
                candidates=(source_ref,),
            ),
        ),
        timelines=(_timeline(_observation(path="input.txt", operation=FileOperation.WRITE)),),
    )
    row = result.artifact_evidence_matrix.rows[0]

    assert row.selected_mode is ReconstructionMode.BLOCKED
    assert row.pre_mutation_coverage is EvidenceCoverage.NONE
    assert row.blocking_uncertainties
    assert result.artifact_evidence_matrix.aggregate_mode == "BLOCKED"


def test_stale_mismatched_duplicate_and_unknown_matrix_inputs_fail_closed() -> None:
    source_ref = _ref("file-version", "file-version://input")
    view_result = _view(_view_subject(source_ref=source_ref))
    bundle = _compile_bundle(view_result, _binding(view_result.included_items[0].projection_item_id))
    target = _target(candidates=(source_ref,))

    stale_bundle = bundle.evidence_bundle.model_copy(
        update={"projection_policy_ref": _ref("projection-policy", "projection-policy://stale")}
    )
    with pytest.raises(EvidenceCompilationPolicyError, match="projection policy"):
        ArtifactEvidenceMatrixCompiler().compile(
            ArtifactEvidenceMatrixCompileRequest(
                producer_task_view_ref=_ref("producer-task-view", "producer-task-view://r2-06"),
                evidence_bundle=stale_bundle,
                view_result=view_result,
                timelines=(_timeline(_observation()),),
                targets=(target,),
                audit=_audit(),
            )
        )

    with pytest.raises(EvidenceCompilationPolicyError, match="duplicate artifact"):
        _matrix(bundle, view_result, targets=(target, target), timelines=(_timeline(_observation()),))
    with pytest.raises(EvidenceCompilationPolicyError, match="unknown target source"):
        _matrix(
            bundle,
            view_result,
            targets=(_target(candidates=(_ref("file-version", "file-version://unknown"),)),),
            timelines=(_timeline(_observation()),),
        )
    with pytest.raises(EvidenceCompilationPolicyError, match="trace"):
        _matrix(
            bundle,
            view_result,
            targets=(target,),
            timelines=(
                _timeline(_observation()).model_copy(update={"trace_ir_version_id": "trace-ir://other"}),
            ),
        )


def test_ids_and_hashes_ignore_audit_timestamp() -> None:
    source_ref = _ref("file-version", "file-version://input")
    view_result = _view(_view_subject(source_ref=source_ref))
    first = EvidenceBundleCompiler().compile(
        EvidenceBundleCompileRequest(
            source_trace_id=SOURCE_TRACE_ID,
            trace_ir_version_id=TRACE_ID,
            consumer_stage="attachment-producer",
            purpose="attachment-production",
            view_result=view_result,
            bindings=(_binding(view_result.included_items[0].projection_item_id),),
            max_characters=1000,
            audit=_audit(datetime(2026, 7, 24, tzinfo=UTC)),
        )
    )
    second = EvidenceBundleCompiler().compile(
        EvidenceBundleCompileRequest(
            source_trace_id=SOURCE_TRACE_ID,
            trace_ir_version_id=TRACE_ID,
            consumer_stage="attachment-producer",
            purpose="attachment-production",
            view_result=view_result,
            bindings=(_binding(view_result.included_items[0].projection_item_id),),
            max_characters=1000,
            audit=_audit(datetime(2026, 7, 25, tzinfo=UTC)),
        )
    )

    assert first.evidence_bundle.evidence_bundle_id == second.evidence_bundle.evidence_bundle_id
    assert first.evidence_bundle.bundle_sha256 == second.evidence_bundle.bundle_sha256
    assert first.canonical_sha256() != second.canonical_sha256()


def test_ids_are_stable_across_python_hash_seed(tmp_path: Path) -> None:
    script = tmp_path / "check_evidence_compilation_seed.py"
    script.write_text(
        """
from datetime import UTC, datetime
from eval_factory.contracts.core import ContractAudit, ObjectRef, SourceSpanRef, VersionBinding
from eval_factory.contracts.core import EvidencePolarity
from eval_factory.contracts.safety import Disposition, OriginClass, ProvenanceDecision, Visibility
from eval_factory.provenance import (
    EvidenceBundleCompileRequest,
    EvidenceBundleCompiler,
    EvidenceViewEngine,
    EvidenceViewPrincipal,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewRequest,
    EvidenceViewSubject,
    ProjectionEvidenceBinding,
)

def ref(kind, object_id):
    return ObjectRef(object_type=kind, object_id=object_id, object_version='v1', object_sha256='a' * 64)

audit = ContractAudit(
    created_at=datetime(2026, 7, 24, tzinfo=UTC),
    created_by='seed-test',
    governing_versions=(VersionBinding(component='evidence-compilation', version='r2-06'),),
)
subject = ref('file-version', 'file-version://input')
decision = ProvenanceDecision(
    provenance_decision_id='provenance-decision://input',
    subject_ref=subject,
    origin_class=OriginClass.PREEXISTING_WORKSPACE_INPUT,
    visibility=Visibility.STAGE_PROJECTION,
    disposition=Disposition.ALLOW_INPUT_EVIDENCE,
    rule_ids=('view-parent/v1',),
    source_event_refs=(ref('trace-event', 'trace-event://input'),),
    confidence=1.0,
    review_required=False,
    policy_version='view-parent/test-v1',
    subject_sha256='a' * 64,
    audit=audit,
)
principal = EvidenceViewPrincipal(
    principal_id='principal://attachment-producer',
    principal_type=EvidenceViewPrincipalType.ATTACHMENT_PRODUCER,
    allowed_purposes=frozenset({EvidenceViewPurpose.ATTACHMENT_PRODUCTION}),
    max_subjects=10,
    max_characters=1000,
)
view = EvidenceViewEngine().project(
    EvidenceViewRequest(
        principal=principal,
        purpose=EvidenceViewPurpose.ATTACHMENT_PRODUCTION,
        subjects=(EvidenceViewSubject(subject_ref=subject, decision=decision, projection_text='safe'),),
        max_characters=1000,
        audit=audit,
    )
)
span = SourceSpanRef(span_id='source-span://safe', source_trace_id='source-trace://evidence-compilation', raw_sha256='a' * 64)
bundle = EvidenceBundleCompiler().compile(
    EvidenceBundleCompileRequest(
        source_trace_id='source-trace://evidence-compilation',
        trace_ir_version_id='trace-ir://evidence-compilation',
        consumer_stage='attachment-producer',
        purpose='attachment-production',
        view_result=view,
        bindings=(ProjectionEvidenceBinding(
            projection_item_id=view.included_items[0].projection_item_id,
            source_spans=(span,),
            polarity=EvidencePolarity.POSITIVE,
            capability='trace-evidence',
            capability_complete=True,
        ),),
        max_characters=1000,
        audit=audit,
    )
).evidence_bundle
print(bundle.evidence_bundle_id)
print(bundle.bundle_sha256)
print(bundle.evidence[0].evidence_ref_id)
""",
        encoding="utf-8",
    )
    outputs = []
    for seed in ("1", "99"):
        result = subprocess.run(
            [sys.executable, str(script)],
            check=False,
            capture_output=True,
            text=True,
            env={**dict(PYTHONHASHSEED=seed), "PYTHONPATH": "src"},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())

    assert len(set(outputs)) == 1


def test_request_model_rejects_unexpected_raw_trace_field() -> None:
    source_ref = _ref("file-version", "file-version://input")
    view_result = _view(_view_subject(source_ref=source_ref))
    request = EvidenceBundleCompileRequest(
        source_trace_id=SOURCE_TRACE_ID,
        trace_ir_version_id=TRACE_ID,
        consumer_stage="attachment-producer",
        purpose="attachment-production",
        view_result=view_result,
        bindings=(_binding(view_result.included_items[0].projection_item_id),),
        max_characters=1000,
        audit=_audit(),
    )

    with pytest.raises(ValidationError):
        EvidenceBundleCompileRequest.model_validate(
            {**request.model_dump(mode="json"), "raw_trace_text": "forbidden"}
        )


def test_matrix_public_identity_authority_preserves_existing_id() -> None:
    source_ref = _ref("file-version", "file-version://identity")
    view_result = _view(_view_subject(source_ref=source_ref))
    bundle = _compile_bundle(
        view_result,
        _binding(view_result.included_items[0].projection_item_id),
    )
    matrix = _matrix(
        bundle,
        view_result,
        targets=(_target(candidates=(source_ref,)),),
        timelines=(
            _timeline(
                _observation(
                    path="input.txt",
                    operation=FileOperation.READ,
                )
            ),
        ),
    ).artifact_evidence_matrix

    digest = artifact_evidence_matrix_carried_sha256(matrix)
    assert digest == "eb0631d02215f771cbb89c490f7e9fa734f0c2f6871dd87e76027c3c2352d589"
    assert matrix.artifact_evidence_matrix_id == (f"artifact-evidence-matrix://sha256/{digest}")
    assert artifact_evidence_matrix_ref(matrix).object_version == "v1"
    assert artifact_evidence_matrix_ref(matrix).object_sha256 == digest
    validate_artifact_evidence_matrix_identity(matrix)

    with pytest.raises(
        EvidenceCompilationPolicyError,
        match="matrix identity",
    ):
        validate_artifact_evidence_matrix_identity(
            matrix.model_copy(update={"artifact_evidence_matrix_id": ("artifact-evidence-matrix://tampered")})
        )
