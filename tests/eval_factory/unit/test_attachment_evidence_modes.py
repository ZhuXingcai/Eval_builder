from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.attachment_planning import (
    ARTIFACT_EVIDENCE_MODE_POLICY_VERSION,
    ArtifactEvidenceModeCompiler,
    ArtifactEvidenceModeOutcome,
    ArtifactEvidenceModePolicyError,
    ArtifactEvidenceModeReason,
    AttachmentPlanningBridge,
)
from eval_factory.contracts import (
    ArtifactEvidenceTargetV2,
    ProducerAttachmentRequirementV2,
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
    artifact_evidence_matrix_carried_sha256,
    artifact_evidence_target_carried_sha256,
    attachment_planning_context_ref,
    producer_storage_authorization_carried_sha256,
    producer_storage_authorization_ref,
    producer_task_view_carried_sha256,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    TypedAttribute,
    VersionBinding,
)
from eval_factory.contracts.safety import (
    Disposition,
    OriginClass,
    ProvenanceDecision,
    Visibility,
)
from eval_factory.contracts.task import AttachmentCriticality
from eval_factory.contracts.trace import (
    Completeness,
    FileObservation,
    FileOperation,
)
from eval_factory.provenance import (
    ArtifactEvidenceMatrixCompiler,
    ArtifactEvidenceMatrixCompileRequest,
    ArtifactEvidenceTarget,
    EvidenceBundleCompiler,
    EvidenceBundleCompileRequest,
    EvidenceViewEngine,
    EvidenceViewPrincipal,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewRequest,
    EvidenceViewSubject,
    FileVersionTimeline,
    FileVersionTimelineBuilder,
    ProjectionEvidenceBinding,
    artifact_evidence_matrix_ref,
    evidence_bundle_ref,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
FOURTH_HASH = "d" * 64
SOURCE_TRACE_ID = "source-trace://attachment-mode/r5-02"
TRACE_IR_VERSION_ID = "trace-ir://attachment-mode/r5-02"
PRODUCER_PRINCIPAL_ID = "principal://attachment-producer/r5-02"
ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = ROOT / "evals/golden/eval_factory/attachment_modes" / "r5-02-mode-matrix-v1.json"


def _audit(
    created_at: datetime = datetime(2026, 7, 27, tzinfo=UTC),
    *,
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="attachment-evidence-mode-test",
        governing_versions=(
            VersionBinding(
                component="attachment-evidence-mode",
                version="r5-02",
            ),
        ),
        input_refs=input_refs,
    )


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v1",
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _span(suffix: str) -> SourceSpanRef:
    return SourceSpanRef(
        span_id=f"source-span://attachment-mode/{suffix}",
        source_trace_id=SOURCE_TRACE_ID,
        raw_sha256=HASH,
    )


def _decision(
    subject: ObjectRef,
    *,
    disposition: Disposition = Disposition.ALLOW_INPUT_EVIDENCE,
) -> ProvenanceDecision:
    return ProvenanceDecision(
        provenance_decision_id=f"provenance-decision://attachment-mode/{subject.object_sha256}",
        subject_ref=subject,
        origin_class=OriginClass.PREEXISTING_WORKSPACE_INPUT,
        visibility=Visibility.STAGE_PROJECTION,
        disposition=disposition,
        rule_ids=("attachment-mode-parent/v1",),
        source_event_refs=(_ref("trace-event", f"attachment-mode/{subject.object_sha256}"),),
        confidence=1.0,
        review_required=False,
        policy_version="attachment-mode-parent/test-v1",
        subject_sha256=subject.object_sha256,
        audit=_audit(),
    )


def _view_subject(
    source_ref: ObjectRef,
    *,
    structure_only: bool = False,
    path: str,
) -> EvidenceViewSubject:
    return EvidenceViewSubject(
        subject_ref=source_ref,
        decision=_decision(
            source_ref,
            disposition=(
                Disposition.ALLOW_STRUCTURE_ONLY if structure_only else Disposition.ALLOW_INPUT_EVIDENCE
            ),
        ),
        projection_text=None if structure_only else "safe input-state content",
        structure_fields=(
            (
                TypedAttribute(key="normalized-path", value=path),
                TypedAttribute(key="media-type", value="text/plain"),
            )
            if structure_only
            else ()
        ),
    )


def _requirement_evidence(suffix: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://attachment-mode/requirement/{suffix}",
        subject_ref=_ref("requirement-projection", suffix),
        source_spans=(_span(f"requirement/{suffix}"),),
        polarity=EvidencePolarity.POSITIVE,
        capability="artifact-requirement",
        capability_complete=True,
    )


def _observation(
    path: str,
    *,
    operation: FileOperation,
    content: bool,
) -> FileObservation:
    content_ref = _ref("content-blob", f"attachment-mode/{path}") if content else None
    return FileObservation(
        observation_id=f"file-observation://attachment-mode/{path}",
        trace_ir_version_id=TRACE_IR_VERSION_ID,
        logical_path=path,
        raw_path_ref=_ref("content-blob", f"raw-path/{path}"),
        operation=operation,
        sequence=1,
        observed_start=None,
        observed_end=None,
        completeness=Completeness.COMPLETE,
        truncated=False,
        content_ref=content_ref,
        content_sha256=(content_ref.object_sha256 if content_ref is not None else None),
        file_version_id=f"file-version://attachment-mode/{path}",
        source_event_refs=(_ref("trace-event", f"attachment-mode/{path}"),),
    )


def _timeline(
    path: str,
    *,
    operation: FileOperation,
    content: bool,
) -> FileVersionTimeline:
    return FileVersionTimelineBuilder().build(
        trace_ir_version_id=TRACE_IR_VERSION_ID,
        file_observations=(
            _observation(
                path,
                operation=operation,
                content=content,
            ),
        ),
        audit=_audit(),
    )[0]


def _storage_authorization(
    bundle,
) -> ProducerStorageAuthorizationV2:
    authorization = ProducerStorageAuthorizationV2(
        authorization_id="producer-storage-authorization://pending",
        producer_principal_id=PRODUCER_PRINCIPAL_ID,
        purpose="ATTACHMENT_PRODUCTION",
        source_task_draft_sha256=OTHER_HASH,
        projection_policy_ref=bundle.projection_policy_ref,
        evidence_bundle_refs=(evidence_bundle_ref(bundle),),
        authorized_subject_refs=tuple(
            sorted(
                (item.subject_ref for item in bundle.evidence),
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        ),
        raw_store_access=False,
        canonical_store_access=False,
        quarantine_store_access=False,
        private_reference_store_access=False,
        credentials_issued=False,
        policy_version="producer-task-view/r4-08-v1",
        authorization_sha256=HASH,
        audit=_audit(),
    )
    digest = producer_storage_authorization_carried_sha256(authorization)
    return authorization.model_copy(
        update={
            "authorization_id": (f"producer-storage-authorization://sha256/{digest}"),
            "authorization_sha256": digest,
        }
    )


def _producer_view(
    *,
    bundle,
    authorization: ProducerStorageAuthorizationV2,
    requirements: tuple[ProducerAttachmentRequirementV2, ...],
) -> ProducerTaskViewV2:
    view = ProducerTaskViewV2(
        producer_task_view_id="producer-task-view://pending",
        producer_task_view_version=1,
        supersedes_producer_task_view_ref=None,
        query_instruction="Inspect the supplied input-state artifacts.",
        attachment_requirements=tuple(
            sorted(
                requirements,
                key=lambda item: item.dependency_id,
            )
        ),
        allowed_tools=(),
        safe_evidence_bundle_refs=(evidence_bundle_ref(bundle),),
        forbidden_outputs=("original final answer",),
        projection_policy_ref=bundle.projection_policy_ref,
        storage_authorization_ref=producer_storage_authorization_ref(authorization),
        prompt_boundary_enforcement_ref=_ref(
            "prompt-boundary-enforcement",
            "attachment-mode",
            version="prompt-injection-as-data/r2-07-v1",
        ),
        source_task_draft_sha256=OTHER_HASH,
        source_contestant_tool_policy_sha256=THIRD_HASH,
        source_contract_chain_sha256=FOURTH_HASH,
        policy_version="producer-task-view/r4-08-v1",
        producer_task_view_sha256=HASH,
        audit=_audit(),
    )
    digest = producer_task_view_carried_sha256(view)
    return view.model_copy(
        update={
            "producer_task_view_id": (f"producer-task-view://sha256/{digest}"),
            "producer_task_view_sha256": digest,
        }
    )


def _target(
    *,
    context,
    setup: str,
    source_ref: ObjectRef | None,
    path: str,
    criticality: AttachmentCriticality,
) -> ArtifactEvidenceTargetV2:
    suffix = setup.casefold().replace("_", "-")
    target = ArtifactEvidenceTargetV2(
        artifact_evidence_target_id="artifact-evidence-target://pending",
        attachment_planning_context_ref=attachment_planning_context_ref(context),
        attachment_dependency_id=(f"attachment-dependency://attachment-mode/{suffix}"),
        artifact_id=f"artifact://attachment-mode/{suffix}",
        logical_path=path,
        media_type="text/plain",
        criticality=criticality,
        requirement_evidence=(_requirement_evidence(suffix),),
        candidate_source_refs=((source_ref,) if source_ref is not None else ()),
        policy_version=ARTIFACT_EVIDENCE_MODE_POLICY_VERSION,
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


def _case_inputs(
    setups: tuple[str, ...],
    *,
    omit_targets: bool = False,
):
    source_refs: dict[str, ObjectRef] = {}
    paths: dict[str, str] = {}
    subjects = []
    for setup in setups:
        if setup in {"PROMPT_ONLY", "MISSING_TARGET"}:
            continue
        suffix = setup.casefold().replace("_", "-")
        path = f"{suffix}.txt"
        source_ref = _ref(
            "file-version",
            suffix,
            digest={
                "TRACE_RICH": HASH,
                "SKELETON_GUIDED": OTHER_HASH,
                "BLOCKED_FIRST_WRITE": THIRD_HASH,
            }[setup],
        )
        source_refs[setup] = source_ref
        paths[setup] = path
        subjects.append(
            _view_subject(
                source_ref,
                structure_only=setup == "SKELETON_GUIDED",
                path=path,
            )
        )
    if not subjects:
        baseline_ref = _ref("file-version", "unused-baseline")
        subjects.append(
            _view_subject(
                baseline_ref,
                path="unused.txt",
            )
        )

    principal = EvidenceViewPrincipal(
        principal_id=PRODUCER_PRINCIPAL_ID,
        principal_type=EvidenceViewPrincipalType.ATTACHMENT_PRODUCER,
        allowed_purposes=frozenset({EvidenceViewPurpose.ATTACHMENT_PRODUCTION}),
        max_subjects=20,
        max_characters=4000,
    )
    view_result = EvidenceViewEngine().project(
        EvidenceViewRequest(
            principal=principal,
            purpose=EvidenceViewPurpose.ATTACHMENT_PRODUCTION,
            subjects=tuple(subjects),
            max_characters=4000,
            audit=_audit(),
        )
    )
    included_by_source = {item.source_ref: item for item in view_result.included_items}
    bindings = tuple(
        ProjectionEvidenceBinding(
            projection_item_id=(included_by_source[source_ref].projection_item_id),
            source_spans=(_span(setup.casefold()),),
            polarity=EvidencePolarity.POSITIVE,
            capability="trace-evidence",
            capability_complete=True,
        )
        for setup, source_ref in sorted(source_refs.items())
    )
    bundle = (
        EvidenceBundleCompiler()
        .compile(
            EvidenceBundleCompileRequest(
                source_trace_id=SOURCE_TRACE_ID,
                trace_ir_version_id=TRACE_IR_VERSION_ID,
                consumer_stage="attachment-producer",
                purpose="attachment-production",
                view_result=view_result,
                bindings=bindings,
                max_characters=4000,
                audit=_audit(),
            )
        )
        .evidence_bundle
    )
    authorization = _storage_authorization(bundle)

    requirements = []
    for setup in setups:
        suffix = setup.casefold().replace("_", "-")
        requirements.append(
            ProducerAttachmentRequirementV2(
                dependency_id=(f"attachment-dependency://attachment-mode/{suffix}"),
                description=f"Input-state artifact for {suffix}.",
                criticality=(
                    AttachmentCriticality.CRITICAL
                    if setup == "BLOCKED_FIRST_WRITE"
                    else AttachmentCriticality.REQUIRED
                ),
            )
        )
    producer_view = _producer_view(
        bundle=bundle,
        authorization=authorization,
        requirements=tuple(requirements),
    )
    context = AttachmentPlanningBridge().compile(
        producer_task_view=producer_view,
        storage_authorization=authorization,
        evidence_bundle=bundle,
        audit=_audit(),
    )

    targets = []
    timelines = []
    for setup in setups:
        if setup == "MISSING_TARGET" or omit_targets:
            continue
        source_ref = source_refs.get(setup)
        suffix = setup.casefold().replace("_", "-")
        path = paths.get(setup, f"{suffix}.txt")
        criticality = (
            AttachmentCriticality.CRITICAL
            if setup == "BLOCKED_FIRST_WRITE"
            else AttachmentCriticality.REQUIRED
        )
        targets.append(
            _target(
                context=context,
                setup=setup,
                source_ref=source_ref,
                path=path,
                criticality=criticality,
            )
        )
        if setup == "TRACE_RICH":
            timelines.append(
                _timeline(
                    path,
                    operation=FileOperation.READ,
                    content=True,
                )
            )
        elif setup == "SKELETON_GUIDED":
            timelines.append(
                _timeline(
                    path,
                    operation=FileOperation.READ,
                    content=False,
                )
            )
        elif setup == "BLOCKED_FIRST_WRITE":
            timelines.append(
                _timeline(
                    path,
                    operation=FileOperation.WRITE,
                    content=True,
                )
            )
    return (
        context,
        producer_view,
        authorization,
        bundle,
        view_result,
        tuple(timelines),
        tuple(targets),
    )


def _compile(inputs):
    return ArtifactEvidenceModeCompiler().compile(
        attachment_planning_context=inputs[0],
        producer_task_view=inputs[1],
        storage_authorization=inputs[2],
        evidence_bundle=inputs[3],
        evidence_view_result=inputs[4],
        timelines=inputs[5],
        targets=inputs[6],
        audit=_audit(),
    )


def test_mode_gold_matrix_matches_expected_rows_and_task_modes() -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    assert gold["schema_version"] == ("eval-factory/r5-02-mode-matrix-gold/v1")
    assert gold["claim_scope"] == "DETERMINISTIC_MODE_POLICY_ONLY"

    for case in gold["cases"]:
        inputs = _case_inputs(tuple(case["setups"]))
        result = _compile(inputs)

        assert result.outcome.value == case["expected_outcome"]
        if result.artifact_evidence_matrix is None:
            assert case["expected_task_mode"] is None
            assert case["expected_row_modes"] == []
            continue
        observed_modes = sorted(row.row.selected_mode.value for row in result.artifact_evidence_matrix.rows)
        assert observed_modes == sorted(case["expected_row_modes"])
        assert result.artifact_evidence_matrix.aggregate_mode == (case["expected_task_mode"])


def test_compiler_wraps_exact_r2_rows_and_source_matrix() -> None:
    inputs = _case_inputs(("TRACE_RICH", "SKELETON_GUIDED", "PROMPT_ONLY"))
    result = _compile(inputs)
    assert result.outcome is ArtifactEvidenceModeOutcome.COMPILED
    assert result.artifact_evidence_matrix is not None
    matrix = result.artifact_evidence_matrix

    direct_targets = tuple(
        ArtifactEvidenceTarget(
            artifact_id=target.artifact_id,
            logical_path=target.logical_path,
            media_type=target.media_type,
            criticality=target.criticality,
            requirement_evidence=target.requirement_evidence,
            candidate_source_refs=target.candidate_source_refs,
        )
        for target in inputs[6]
    )
    direct = (
        ArtifactEvidenceMatrixCompiler()
        .compile(
            ArtifactEvidenceMatrixCompileRequest(
                producer_task_view_ref=inputs[0].producer_task_view_ref,
                evidence_bundle=inputs[3],
                view_result=inputs[4],
                timelines=inputs[5],
                targets=direct_targets,
                audit=_audit(),
            )
        )
        .artifact_evidence_matrix
    )

    assert tuple(row.row for row in matrix.rows) == direct.rows
    assert matrix.source_r2_matrix_ref == artifact_evidence_matrix_ref(direct)
    assert matrix.aggregate_mode == "MIXED"
    assert matrix.artifact_evidence_matrix_sha256 == (artifact_evidence_matrix_carried_sha256(matrix))


def test_missing_target_blocks_without_partial_matrix() -> None:
    inputs = _case_inputs(("TRACE_RICH",), omit_targets=True)

    result = _compile(inputs)

    assert result.outcome is ArtifactEvidenceModeOutcome.BLOCKED_CAPABILITY
    assert result.artifact_evidence_matrix is None
    assert result.missing_dependency_ids == ("attachment-dependency://attachment-mode/trace-rich",)
    assert result.reasons == frozenset({ArtifactEvidenceModeReason.MISSING_ARTIFACT_TARGET})


def test_no_requirements_returns_not_required() -> None:
    result = _compile(_case_inputs(()))

    assert result.outcome is ArtifactEvidenceModeOutcome.NOT_REQUIRED
    assert result.artifact_evidence_matrix is None
    assert result.missing_dependency_ids == ()
    assert result.reasons == frozenset()


def test_unknown_target_and_criticality_mismatch_fail_closed() -> None:
    inputs = _case_inputs(("PROMPT_ONLY",))
    target = inputs[6][0]
    unknown = target.model_copy(
        update={"attachment_dependency_id": ("attachment-dependency://attachment-mode/unknown")}
    )
    digest = artifact_evidence_target_carried_sha256(unknown)
    unknown = unknown.model_copy(
        update={
            "artifact_evidence_target_id": (f"artifact-evidence-target://sha256/{digest}"),
            "artifact_evidence_target_sha256": digest,
        }
    )
    with pytest.raises(
        ArtifactEvidenceModePolicyError,
        match="unknown target",
    ):
        _compile((*inputs[:6], (unknown,)))

    mismatched = _target(
        context=inputs[0],
        setup="PROMPT_ONLY",
        source_ref=None,
        path="prompt-only.txt",
        criticality=AttachmentCriticality.OPTIONAL,
    )
    with pytest.raises(
        ArtifactEvidenceModePolicyError,
        match="criticality",
    ):
        _compile((*inputs[:6], (mismatched,)))

    stale_policy = target.model_copy(update={"policy_version": "artifact-evidence-mode/stale"})
    digest = artifact_evidence_target_carried_sha256(stale_policy)
    stale_policy = stale_policy.model_copy(
        update={
            "artifact_evidence_target_id": (f"artifact-evidence-target://sha256/{digest}"),
            "artifact_evidence_target_sha256": digest,
        }
    )
    with pytest.raises(
        ArtifactEvidenceModePolicyError,
        match="target policy",
    ):
        _compile((*inputs[:6], (stale_policy,)))


def test_requirement_evidence_must_be_current_safe_and_trace_bound() -> None:
    inputs = _case_inputs(("PROMPT_ONLY",))
    target = inputs[6][0]
    cross_trace = target.requirement_evidence[0].model_copy(
        update={
            "source_spans": (
                SourceSpanRef(
                    span_id="source-span://attachment-mode/wrong",
                    source_trace_id="source-trace://other",
                    raw_sha256=HASH,
                ),
            )
        }
    )
    changed = target.model_copy(update={"requirement_evidence": (cross_trace,)})
    digest = artifact_evidence_target_carried_sha256(changed)
    changed = changed.model_copy(
        update={
            "artifact_evidence_target_id": (f"artifact-evidence-target://sha256/{digest}"),
            "artifact_evidence_target_sha256": digest,
        }
    )

    with pytest.raises(
        ArtifactEvidenceModePolicyError,
        match="source trace",
    ):
        _compile((*inputs[:6], (changed,)))


def test_validate_current_accepts_exact_and_rejects_matrix_mutation() -> None:
    inputs = _case_inputs(("TRACE_RICH",))
    result = _compile(inputs)
    assert result.artifact_evidence_matrix is not None
    compiler = ArtifactEvidenceModeCompiler()

    compiler.validate_current(
        attachment_planning_context=inputs[0],
        producer_task_view=inputs[1],
        storage_authorization=inputs[2],
        evidence_bundle=inputs[3],
        evidence_view_result=inputs[4],
        timelines=inputs[5],
        targets=inputs[6],
        artifact_evidence_matrix=result.artifact_evidence_matrix,
    )

    matrix = result.artifact_evidence_matrix
    shifted_rows = tuple(
        row.model_copy(
            update={
                "audit": ContractAudit(
                    created_at=datetime(2026, 7, 29, tzinfo=UTC),
                    created_by="shifted-row-audit",
                    governing_versions=row.audit.governing_versions,
                    input_refs=row.audit.input_refs,
                )
            }
        )
        for row in matrix.rows
    )
    audit_only_change = matrix.model_copy(
        update={
            "rows": shifted_rows,
            "audit": ContractAudit(
                created_at=datetime(2026, 7, 28, tzinfo=UTC),
                created_by="shifted-matrix-audit",
                governing_versions=matrix.audit.governing_versions,
                input_refs=matrix.audit.input_refs,
            ),
        }
    )
    assert audit_only_change.artifact_evidence_matrix_sha256 == (
        artifact_evidence_matrix_carried_sha256(audit_only_change)
    )
    compiler.validate_current(
        attachment_planning_context=inputs[0],
        producer_task_view=inputs[1],
        storage_authorization=inputs[2],
        evidence_bundle=inputs[3],
        evidence_view_result=inputs[4],
        timelines=inputs[5],
        targets=inputs[6],
        artifact_evidence_matrix=audit_only_change,
    )

    with pytest.raises(
        ArtifactEvidenceModePolicyError,
        match="current",
    ):
        compiler.validate_current(
            attachment_planning_context=inputs[0],
            producer_task_view=inputs[1],
            storage_authorization=inputs[2],
            evidence_bundle=inputs[3],
            evidence_view_result=inputs[4],
            timelines=inputs[5],
            targets=inputs[6],
            artifact_evidence_matrix=(
                result.artifact_evidence_matrix.model_copy(update={"aggregate_mode": "PROMPT_ONLY"})
            ),
        )


def test_result_and_matrix_do_not_disclose_content_or_caller_audit() -> None:
    inputs = _case_inputs(("TRACE_RICH",))
    private_ref = _ref(
        "private-reference",
        "caller-audit",
        digest=FOURTH_HASH,
    )
    result = ArtifactEvidenceModeCompiler().compile(
        attachment_planning_context=inputs[0],
        producer_task_view=inputs[1],
        storage_authorization=inputs[2],
        evidence_bundle=inputs[3],
        evidence_view_result=inputs[4],
        timelines=inputs[5],
        targets=inputs[6],
        audit=_audit(input_refs=(private_ref,)),
    )

    serialized = str(result.model_dump(mode="json"))
    for forbidden in (
        "safe input-state content",
        "private-reference://caller-audit",
        "query_instruction",
        "authorized_subject_refs",
        "provider_preference",
        "runtime_preference",
        "build_spec",
        "credential",
    ):
        assert forbidden not in serialized


def test_result_model_is_closed() -> None:
    result = _compile(_case_inputs(("PROMPT_ONLY",)))

    with pytest.raises(ValidationError):
        type(result).model_validate(
            {
                **result.model_dump(mode="json"),
                "artifact_content": "forbidden",
            }
        )

    with pytest.raises(ValidationError, match="result planning context"):
        type(result).model_validate(
            {
                **result.model_dump(mode="python"),
                "attachment_planning_context_ref": _ref(
                    "attachment-planning-context",
                    "other",
                    version="v2",
                    digest=OTHER_HASH,
                ),
            }
        )
