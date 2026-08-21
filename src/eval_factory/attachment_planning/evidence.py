from __future__ import annotations

import hashlib
import json

from eval_factory.attachment_planning.bridge import (
    AttachmentPlanningBridge,
)
from eval_factory.attachment_planning.models import (
    ARTIFACT_EVIDENCE_MODE_POLICY_VERSION,
    ArtifactEvidenceModeCompilationResult,
    ArtifactEvidenceModeOutcome,
    ArtifactEvidenceModePolicyError,
    ArtifactEvidenceModeReason,
    AttachmentPlanningPolicyError,
)
from eval_factory.contracts.attachment import ArtifactEvidenceRow
from eval_factory.contracts.attachment_v2 import (
    ArtifactEvidenceMatrixV2,
    ArtifactEvidenceRowV2,
    ArtifactEvidenceTargetV2,
    AttachmentPlanningContextV2,
    artifact_evidence_aggregate_mode_v2,
    artifact_evidence_matrix_carried_sha256,
    artifact_evidence_matrix_ref,
    artifact_evidence_row_carried_sha256,
    artifact_evidence_row_ref,
    artifact_evidence_target_ref,
    attachment_planning_context_ref,
    validate_artifact_evidence_target_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    ObjectRef,
)
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.task_v2 import (
    ProducerAttachmentRequirementV2,
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
)
from eval_factory.provenance.bundles import (
    EVIDENCE_COMPILATION_POLICY_VERSION,
    ArtifactEvidenceMatrixCompiler,
    ArtifactEvidenceMatrixCompileRequest,
    ArtifactEvidenceTarget,
    EvidenceCompilationPolicyError,
    EvidenceCompilationUncertainty,
    projection_policy_ref,
    validate_artifact_evidence_matrix_identity,
    validate_evidence_bundle_identity,
    validate_projection_policy,
)
from eval_factory.provenance.bundles import (
    artifact_evidence_matrix_ref as r2_artifact_evidence_matrix_ref,
)
from eval_factory.provenance.timelines import FileVersionTimeline
from eval_factory.provenance.views import (
    EvidenceViewPolicyError,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewResult,
    validate_evidence_view_result_identity,
)

_DENIED_REQUIREMENT_REF_MARKERS = frozenset(
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


class ArtifactEvidenceModeCompiler:
    policy_version = ARTIFACT_EVIDENCE_MODE_POLICY_VERSION

    def compile(
        self,
        *,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        evidence_bundle: EvidenceBundle,
        evidence_view_result: EvidenceViewResult,
        timelines: tuple[FileVersionTimeline, ...],
        targets: tuple[ArtifactEvidenceTargetV2, ...],
        audit: ContractAudit,
    ) -> ArtifactEvidenceModeCompilationResult:
        context_ref = attachment_planning_context_ref(attachment_planning_context)
        self._validate_current_basis(
            attachment_planning_context=attachment_planning_context,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            evidence_bundle=evidence_bundle,
            evidence_view_result=evidence_view_result,
        )
        requirements = {item.dependency_id: item for item in producer_task_view.attachment_requirements}
        targets_by_dependency = self._validate_targets(
            attachment_planning_context=attachment_planning_context,
            requirements=requirements,
            targets=targets,
        )

        if not requirements:
            if targets:
                raise ArtifactEvidenceModePolicyError(
                    "artifact targets are not allowed without producer attachment requirements"
                )
            return _compilation_result(
                context_ref=context_ref,
                outcome=ArtifactEvidenceModeOutcome.NOT_REQUIRED,
                matrix=None,
                missing_dependency_ids=(),
                reasons=frozenset(),
                uncertainties=(),
                audit=audit,
            )

        missing_dependency_ids = tuple(sorted(set(requirements) - set(targets_by_dependency)))
        if missing_dependency_ids:
            return _compilation_result(
                context_ref=context_ref,
                outcome=ArtifactEvidenceModeOutcome.BLOCKED_CAPABILITY,
                matrix=None,
                missing_dependency_ids=missing_dependency_ids,
                reasons=frozenset({ArtifactEvidenceModeReason.MISSING_ARTIFACT_TARGET}),
                uncertainties=(),
                audit=audit,
            )

        self._validate_timelines(
            timelines,
            attachment_planning_context.trace_ir_version_id,
        )
        r2_targets = tuple(
            ArtifactEvidenceTarget(
                artifact_id=target.artifact_id,
                logical_path=target.logical_path,
                media_type=target.media_type,
                criticality=target.criticality,
                requirement_evidence=target.requirement_evidence,
                candidate_source_refs=target.candidate_source_refs,
            )
            for target in targets
        )
        try:
            r2_result = ArtifactEvidenceMatrixCompiler().compile(
                ArtifactEvidenceMatrixCompileRequest(
                    producer_task_view_ref=(attachment_planning_context.producer_task_view_ref),
                    evidence_bundle=evidence_bundle,
                    view_result=evidence_view_result,
                    timelines=timelines,
                    targets=r2_targets,
                    audit=_safe_audit(
                        audit,
                        (
                            context_ref,
                            attachment_planning_context.safe_evidence_bundle_ref,
                            *tuple(artifact_evidence_target_ref(target) for target in targets),
                        ),
                    ),
                )
            )
            validate_artifact_evidence_matrix_identity(r2_result.artifact_evidence_matrix)
        except EvidenceCompilationPolicyError as exc:
            raise ArtifactEvidenceModePolicyError("R2 artifact evidence classification failed") from exc

        targets_by_artifact = {target.artifact_id: target for target in targets}
        wrapped_rows = tuple(
            _wrap_r2_row(
                context_ref=context_ref,
                target=targets_by_artifact[row.artifact_id],
                row=row,
                audit=audit,
            )
            for row in r2_result.artifact_evidence_matrix.rows
        )
        aggregate_mode = artifact_evidence_aggregate_mode_v2(wrapped_rows)
        source_matrix_ref = r2_artifact_evidence_matrix_ref(r2_result.artifact_evidence_matrix)
        matrix = ArtifactEvidenceMatrixV2(
            artifact_evidence_matrix_id=("artifact-evidence-matrix://pending"),
            attachment_planning_context_ref=context_ref,
            producer_task_view_ref=(attachment_planning_context.producer_task_view_ref),
            safe_evidence_bundle_ref=(attachment_planning_context.safe_evidence_bundle_ref),
            source_r2_matrix_ref=source_matrix_ref,
            rows=wrapped_rows,
            aggregate_mode=aggregate_mode,
            r2_policy_version=EVIDENCE_COMPILATION_POLICY_VERSION,
            policy_version=self.policy_version,
            artifact_evidence_matrix_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    context_ref,
                    attachment_planning_context.producer_task_view_ref,
                    attachment_planning_context.safe_evidence_bundle_ref,
                    source_matrix_ref,
                    *tuple(artifact_evidence_row_ref(row) for row in wrapped_rows),
                ),
            ),
        )
        matrix_digest = artifact_evidence_matrix_carried_sha256(matrix)
        matrix = matrix.model_copy(
            update={
                "artifact_evidence_matrix_id": (f"artifact-evidence-matrix://sha256/{matrix_digest}"),
                "artifact_evidence_matrix_sha256": matrix_digest,
            }
        )
        return _compilation_result(
            context_ref=context_ref,
            outcome=ArtifactEvidenceModeOutcome.COMPILED,
            matrix=matrix,
            missing_dependency_ids=(),
            reasons=frozenset(),
            uncertainties=r2_result.uncertainties,
            audit=audit,
        )

    def validate_current(
        self,
        *,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        evidence_bundle: EvidenceBundle,
        evidence_view_result: EvidenceViewResult,
        timelines: tuple[FileVersionTimeline, ...],
        targets: tuple[ArtifactEvidenceTargetV2, ...],
        artifact_evidence_matrix: ArtifactEvidenceMatrixV2,
    ) -> None:
        try:
            _validate_v2_matrix_identity(artifact_evidence_matrix)
            rebuilt = self.compile(
                attachment_planning_context=(attachment_planning_context),
                producer_task_view=producer_task_view,
                storage_authorization=storage_authorization,
                evidence_bundle=evidence_bundle,
                evidence_view_result=evidence_view_result,
                timelines=timelines,
                targets=targets,
                audit=artifact_evidence_matrix.audit,
            )
            if (
                rebuilt.outcome is not ArtifactEvidenceModeOutcome.COMPILED
                or rebuilt.artifact_evidence_matrix is None
                or not _same_matrix_except_audit_actor_time(
                    rebuilt.artifact_evidence_matrix,
                    artifact_evidence_matrix,
                )
            ):
                raise ArtifactEvidenceModePolicyError(
                    "artifact evidence matrix does not match authoritative inputs"
                )
        except ArtifactEvidenceModePolicyError as exc:
            raise ArtifactEvidenceModePolicyError(
                f"current artifact evidence matrix validation failed: {exc}"
            ) from exc

    def _validate_current_basis(
        self,
        *,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        evidence_bundle: EvidenceBundle,
        evidence_view_result: EvidenceViewResult,
    ) -> None:
        try:
            AttachmentPlanningBridge().validate_current(
                producer_task_view=producer_task_view,
                storage_authorization=storage_authorization,
                evidence_bundle=evidence_bundle,
                attachment_planning_context=(attachment_planning_context),
            )
            validate_evidence_view_result_identity(evidence_view_result)
            validate_projection_policy(evidence_view_result.projection_policy)
            validate_evidence_bundle_identity(evidence_bundle)
        except (
            AttachmentPlanningPolicyError,
            EvidenceViewPolicyError,
            EvidenceCompilationPolicyError,
        ) as exc:
            raise ArtifactEvidenceModePolicyError(
                "attachment mode source identity is stale or invalid"
            ) from exc

        if (
            evidence_view_result.principal_id != attachment_planning_context.producer_principal_id
            or evidence_view_result.principal_type is not EvidenceViewPrincipalType.ATTACHMENT_PRODUCER
            or evidence_view_result.purpose is not EvidenceViewPurpose.ATTACHMENT_PRODUCTION
        ):
            raise ArtifactEvidenceModePolicyError("evidence view must use the exact attachment producer")
        expected_policy_ref = projection_policy_ref(evidence_view_result.projection_policy)
        if (
            expected_policy_ref != attachment_planning_context.projection_policy_ref
            or expected_policy_ref != evidence_bundle.projection_policy_ref
        ):
            raise ArtifactEvidenceModePolicyError("evidence view projection policy is stale or mismatched")
        included_projected_refs = {item.projected_ref for item in evidence_view_result.included_items}
        if any(item.subject_ref not in included_projected_refs for item in evidence_bundle.evidence):
            raise ArtifactEvidenceModePolicyError("evidence bundle contains an unknown projected subject")

    def _validate_targets(
        self,
        *,
        attachment_planning_context: AttachmentPlanningContextV2,
        requirements: dict[str, ProducerAttachmentRequirementV2],
        targets: tuple[ArtifactEvidenceTargetV2, ...],
    ) -> dict[str, ArtifactEvidenceTargetV2]:
        context_ref = attachment_planning_context_ref(attachment_planning_context)
        by_dependency: dict[str, ArtifactEvidenceTargetV2] = {}
        artifact_ids: set[str] = set()
        logical_paths: set[str] = set()
        for target in targets:
            try:
                validate_artifact_evidence_target_identity(target)
            except ValueError as exc:
                raise ArtifactEvidenceModePolicyError(
                    "artifact evidence target identity is stale or invalid"
                ) from exc
            if target.policy_version != self.policy_version:
                raise ArtifactEvidenceModePolicyError(
                    "artifact evidence target policy is stale or unsupported"
                )
            if target.attachment_planning_context_ref != context_ref:
                raise ArtifactEvidenceModePolicyError(
                    "artifact target planning context is stale or mismatched"
                )
            requirement = requirements.get(target.attachment_dependency_id)
            if requirement is None:
                raise ArtifactEvidenceModePolicyError("unknown target attachment dependency")
            expected_criticality = requirement.criticality
            if target.criticality is not expected_criticality:
                raise ArtifactEvidenceModePolicyError(
                    "artifact target criticality does not match ProducerTaskView"
                )
            if target.attachment_dependency_id in by_dependency:
                raise ArtifactEvidenceModePolicyError("duplicate artifact target dependency")
            if target.artifact_id in artifact_ids:
                raise ArtifactEvidenceModePolicyError("duplicate artifact target ID")
            if target.logical_path in logical_paths:
                raise ArtifactEvidenceModePolicyError("duplicate artifact target logical path")
            _validate_requirement_evidence(
                target,
                attachment_planning_context.source_trace_id,
            )
            by_dependency[target.attachment_dependency_id] = target
            artifact_ids.add(target.artifact_id)
            logical_paths.add(target.logical_path)
        return by_dependency

    def _validate_timelines(
        self,
        timelines: tuple[FileVersionTimeline, ...],
        trace_ir_version_id: str,
    ) -> None:
        observed_paths: set[str] = set()
        for timeline in timelines:
            if timeline.trace_ir_version_id != trace_ir_version_id:
                raise ArtifactEvidenceModePolicyError("file timeline trace is stale or mismatched")
            if timeline.logical_path in observed_paths:
                raise ArtifactEvidenceModePolicyError("duplicate file timeline logical path")
            observed_paths.add(timeline.logical_path)


def _wrap_r2_row(
    *,
    context_ref: ObjectRef,
    target: ArtifactEvidenceTargetV2,
    row: ArtifactEvidenceRow,
    audit: ContractAudit,
) -> ArtifactEvidenceRowV2:
    if (
        row.logical_path != target.logical_path
        or row.media_type != target.media_type
        or row.criticality is not target.criticality
    ):
        raise ArtifactEvidenceModePolicyError("R2 row does not match the exact artifact target")
    wrapped = ArtifactEvidenceRowV2(
        artifact_evidence_row_id="artifact-evidence-row://pending",
        attachment_planning_context_ref=context_ref,
        artifact_evidence_target_ref=artifact_evidence_target_ref(target),
        attachment_dependency_id=target.attachment_dependency_id,
        row=row,
        r2_row_sha256=row.canonical_sha256(),
        r2_policy_version=EVIDENCE_COMPILATION_POLICY_VERSION,
        policy_version=ARTIFACT_EVIDENCE_MODE_POLICY_VERSION,
        artifact_evidence_row_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                context_ref,
                artifact_evidence_target_ref(target),
            ),
        ),
    )
    digest = artifact_evidence_row_carried_sha256(wrapped)
    return wrapped.model_copy(
        update={
            "artifact_evidence_row_id": (f"artifact-evidence-row://sha256/{digest}"),
            "artifact_evidence_row_sha256": digest,
        }
    )


def _validate_v2_matrix_identity(
    matrix: ArtifactEvidenceMatrixV2,
) -> None:
    for row in matrix.rows:
        digest = artifact_evidence_row_carried_sha256(row)
        if (
            row.artifact_evidence_row_sha256 != digest
            or row.artifact_evidence_row_id != f"artifact-evidence-row://sha256/{digest}"
        ):
            raise ArtifactEvidenceModePolicyError("artifact evidence row identity is stale or invalid")
    digest = artifact_evidence_matrix_carried_sha256(matrix)
    if (
        matrix.artifact_evidence_matrix_sha256 != digest
        or matrix.artifact_evidence_matrix_id != f"artifact-evidence-matrix://sha256/{digest}"
    ):
        raise ArtifactEvidenceModePolicyError("artifact evidence matrix identity is stale or invalid")


def _validate_requirement_evidence(
    target: ArtifactEvidenceTargetV2,
    source_trace_id: str,
) -> None:
    evidence_ids = tuple(item.evidence_ref_id for item in target.requirement_evidence)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ArtifactEvidenceModePolicyError("duplicate artifact requirement evidence")
    if evidence_ids != tuple(sorted(evidence_ids)):
        raise ArtifactEvidenceModePolicyError("artifact requirement evidence must be sorted")
    for evidence in target.requirement_evidence:
        if evidence.polarity is not EvidencePolarity.POSITIVE or not evidence.capability_complete:
            raise ArtifactEvidenceModePolicyError("artifact requirement evidence capability is incomplete")
        if _unsafe_requirement_ref(evidence.subject_ref):
            raise ArtifactEvidenceModePolicyError("artifact requirement evidence subject is unsafe")
        if any(span.source_trace_id != source_trace_id for span in evidence.source_spans):
            raise ArtifactEvidenceModePolicyError("artifact requirement evidence source trace is mismatched")


def _unsafe_requirement_ref(ref: ObjectRef) -> bool:
    normalized = f"{ref.object_type}:{ref.object_id}".casefold().replace(
        "_",
        "-",
    )
    return any(marker in normalized for marker in _DENIED_REQUIREMENT_REF_MARKERS)


def _compilation_result(
    *,
    context_ref: ObjectRef,
    outcome: ArtifactEvidenceModeOutcome,
    matrix: ArtifactEvidenceMatrixV2 | None,
    missing_dependency_ids: tuple[str, ...],
    reasons: frozenset[ArtifactEvidenceModeReason],
    uncertainties: tuple[EvidenceCompilationUncertainty, ...],
    audit: ContractAudit,
) -> ArtifactEvidenceModeCompilationResult:
    matrix_ref = artifact_evidence_matrix_ref(matrix) if matrix is not None else None
    payload = {
        "attachment_planning_context_ref": context_ref.model_dump(
            mode="json",
            exclude_none=False,
        ),
        "outcome": outcome.value,
        "artifact_evidence_matrix_ref": (
            None
            if matrix_ref is None
            else matrix_ref.model_dump(
                mode="json",
                exclude_none=False,
            )
        ),
        "missing_dependency_ids": list(missing_dependency_ids),
        "reasons": sorted(item.value for item in reasons),
        "uncertainties": [item.value for item in uncertainties],
        "policy_version": ARTIFACT_EVIDENCE_MODE_POLICY_VERSION,
    }
    digest = _payload_sha256(payload)
    refs = (context_ref,) if matrix_ref is None else (context_ref, matrix_ref)
    return ArtifactEvidenceModeCompilationResult(
        result_id=(f"artifact-evidence-mode-result://sha256/{digest}"),
        attachment_planning_context_ref=context_ref,
        outcome=outcome,
        artifact_evidence_matrix=matrix,
        missing_dependency_ids=missing_dependency_ids,
        reasons=reasons,
        uncertainties=uncertainties,
        policy_version=ARTIFACT_EVIDENCE_MODE_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(audit, refs),
    )


def _same_matrix_except_audit_actor_time(
    expected: ArtifactEvidenceMatrixV2,
    observed: ArtifactEvidenceMatrixV2,
) -> bool:
    if expected.model_dump(
        mode="json",
        exclude={"audit", "rows"},
    ) != observed.model_dump(
        mode="json",
        exclude={"audit", "rows"},
    ):
        return False
    if len(expected.rows) != len(observed.rows):
        return False
    for expected_row, observed_row in zip(expected.rows, observed.rows, strict=True):
        if expected_row.model_dump(
            mode="json",
            exclude={"audit"},
        ) != observed_row.model_dump(
            mode="json",
            exclude={"audit"},
        ):
            return False
        if not _same_audit_lineage(expected_row.audit, observed_row.audit):
            return False
    return _same_audit_lineage(expected.audit, observed.audit)


def _same_audit_lineage(
    expected: ContractAudit,
    observed: ContractAudit,
) -> bool:
    return (
        expected.governing_versions == observed.governing_versions
        and expected.input_refs == observed.input_refs
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


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
