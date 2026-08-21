from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from eval_factory.contracts.attachment import (
    ArtifactEvidenceMatrix,
    ArtifactEvidenceRow,
    EvidenceCoverage,
    EvidenceStrength,
    ReconstructionMode,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    EvidencePolarity,
    EvidenceRef,
    Identifier,
    ObjectRef,
    RelativePath,
    SourceSpanRef,
)
from eval_factory.contracts.safety import EvidenceBundle, ProjectionPolicy
from eval_factory.contracts.task import AttachmentCriticality
from eval_factory.contracts.trace import Completeness
from eval_factory.provenance.timelines import (
    FileVersionOrigin,
    FileVersionTimeline,
    FileVersionUncertainty,
)
from eval_factory.provenance.views import (
    EvidenceProjectionItem,
    EvidenceProjectionMode,
    EvidenceViewResult,
)

EVIDENCE_COMPILATION_POLICY_VERSION: Literal["evidence-compilation/r2-06-v1"] = (
    "evidence-compilation/r2-06-v1"
)
_CONFIDENCE_THRESHOLD = 0.80


class EvidenceCompilationPolicyError(RuntimeError):
    pass


class EvidenceCompilationUncertainty(StrEnum):
    VIEW_SUBJECT_EXCLUDED = "VIEW_SUBJECT_EXCLUDED"
    MISSING_SOURCE_SPAN_BINDING = "MISSING_SOURCE_SPAN_BINDING"
    INCOMPLETE_CAPABILITY = "INCOMPLETE_CAPABILITY"
    MISSING_TIMELINE = "MISSING_TIMELINE"
    TIMELINE_UNCERTAINTY = "TIMELINE_UNCERTAINTY"
    MISSING_PRE_MUTATION_EVIDENCE = "MISSING_PRE_MUTATION_EVIDENCE"


class ProjectionEvidenceBinding(ContractModel):
    schema_version: Literal["eval-factory/projection-evidence-binding/r2-06"] = (
        "eval-factory/projection-evidence-binding/r2-06"
    )
    projection_item_id: Identifier
    source_spans: tuple[SourceSpanRef, ...] = Field(min_length=1)
    polarity: EvidencePolarity
    capability: Identifier
    capability_complete: bool

    @field_validator("polarity", mode="before")
    @classmethod
    def parse_polarity(cls, value: object) -> EvidencePolarity:
        if isinstance(value, EvidencePolarity):
            return value
        if isinstance(value, str):
            return EvidencePolarity(value)
        raise TypeError("polarity must be an EvidencePolarity")


class EvidenceBundleCompileRequest(ContractModel):
    schema_version: Literal["eval-factory/evidence-bundle-compile-request/r2-06"] = (
        "eval-factory/evidence-bundle-compile-request/r2-06"
    )
    source_trace_id: Identifier
    trace_ir_version_id: Identifier
    consumer_stage: Identifier
    purpose: Identifier
    view_result: EvidenceViewResult
    bindings: tuple[ProjectionEvidenceBinding, ...] = ()
    max_characters: int = Field(ge=0)
    audit: ContractAudit


class EvidenceBundleCompilationResult(ContractModel):
    schema_version: Literal["eval-factory/evidence-bundle-compilation-result/r2-06"] = (
        "eval-factory/evidence-bundle-compilation-result/r2-06"
    )
    evidence_bundle: EvidenceBundle
    uncertainties: tuple[EvidenceCompilationUncertainty, ...] = ()
    policy_version: Literal["evidence-compilation/r2-06-v1"] = EVIDENCE_COMPILATION_POLICY_VERSION
    audit: ContractAudit

    @field_validator("uncertainties", mode="before")
    @classmethod
    def parse_uncertainties(cls, value: object) -> tuple[EvidenceCompilationUncertainty, ...]:
        if isinstance(value, tuple):
            return tuple(
                item
                if isinstance(item, EvidenceCompilationUncertainty)
                else EvidenceCompilationUncertainty(item)
                for item in value
            )
        if isinstance(value, (list, set, frozenset)):
            return tuple(EvidenceCompilationUncertainty(item) for item in value)
        raise TypeError("uncertainties must be a collection")


class ArtifactEvidenceTarget(ContractModel):
    schema_version: Literal["eval-factory/artifact-evidence-target/r2-06"] = (
        "eval-factory/artifact-evidence-target/r2-06"
    )
    artifact_id: Identifier
    logical_path: RelativePath
    media_type: str = Field(min_length=3, max_length=255)
    criticality: AttachmentCriticality
    requirement_evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    candidate_source_refs: tuple[ObjectRef, ...] = ()

    @field_validator("criticality", mode="before")
    @classmethod
    def parse_criticality(cls, value: object) -> AttachmentCriticality:
        if isinstance(value, AttachmentCriticality):
            return value
        if isinstance(value, str):
            return AttachmentCriticality(value)
        raise TypeError("criticality must be an AttachmentCriticality")


class ArtifactEvidenceMatrixCompileRequest(ContractModel):
    schema_version: Literal["eval-factory/artifact-evidence-matrix-compile-request/r2-06"] = (
        "eval-factory/artifact-evidence-matrix-compile-request/r2-06"
    )
    producer_task_view_ref: ObjectRef
    evidence_bundle: EvidenceBundle
    view_result: EvidenceViewResult
    timelines: tuple[FileVersionTimeline, ...] = ()
    targets: tuple[ArtifactEvidenceTarget, ...] = Field(min_length=1)
    audit: ContractAudit


class ArtifactEvidenceMatrixCompilationResult(ContractModel):
    schema_version: Literal["eval-factory/artifact-evidence-matrix-compilation-result/r2-06"] = (
        "eval-factory/artifact-evidence-matrix-compilation-result/r2-06"
    )
    artifact_evidence_matrix: ArtifactEvidenceMatrix
    uncertainties: tuple[EvidenceCompilationUncertainty, ...] = ()
    policy_version: Literal["evidence-compilation/r2-06-v1"] = EVIDENCE_COMPILATION_POLICY_VERSION
    audit: ContractAudit

    @field_validator("uncertainties", mode="before")
    @classmethod
    def parse_uncertainties(cls, value: object) -> tuple[EvidenceCompilationUncertainty, ...]:
        if isinstance(value, tuple):
            return tuple(
                item
                if isinstance(item, EvidenceCompilationUncertainty)
                else EvidenceCompilationUncertainty(item)
                for item in value
            )
        if isinstance(value, (list, set, frozenset)):
            return tuple(EvidenceCompilationUncertainty(item) for item in value)
        raise TypeError("uncertainties must be a collection")


class EvidenceBundleCompiler:
    policy_version = EVIDENCE_COMPILATION_POLICY_VERSION

    def compile(self, request: EvidenceBundleCompileRequest) -> EvidenceBundleCompilationResult:
        validate_projection_policy(request.view_result.projection_policy)
        included_by_id = {item.projection_item_id: item for item in request.view_result.included_items}
        bindings = _bindings_by_projection_id(request.bindings, included_by_id)
        evidence: list[EvidenceRef] = []
        excluded = list(_view_excluded_refs(request.view_result))
        uncertainties: set[EvidenceCompilationUncertainty] = set()
        if excluded:
            uncertainties.add(EvidenceCompilationUncertainty.VIEW_SUBJECT_EXCLUDED)
        returned_characters = 0

        for item in request.view_result.included_items:
            binding = bindings.get(item.projection_item_id)
            if binding is None:
                excluded.append(item.source_ref)
                uncertainties.add(EvidenceCompilationUncertainty.MISSING_SOURCE_SPAN_BINDING)
                continue
            _validate_binding_spans(binding, request.source_trace_id)
            if not binding.capability_complete:
                uncertainties.add(EvidenceCompilationUncertainty.INCOMPLETE_CAPABILITY)
            evidence_ref = _evidence_ref(
                item=item,
                binding=binding,
                policy_version=self.policy_version,
            )
            evidence.append(evidence_ref)
            returned_characters += _item_returned_characters(item)

        evidence_tuple = tuple(sorted(evidence, key=lambda item: item.evidence_ref_id))
        excluded_tuple = _unique_object_refs(tuple(excluded))
        policy_ref = projection_policy_ref(request.view_result.projection_policy)
        bundle_seed = {
            "source_trace_id": request.source_trace_id,
            "trace_ir_version_id": request.trace_ir_version_id,
            "consumer_stage": request.consumer_stage,
            "purpose": request.purpose,
            "projection_policy_ref": policy_ref.model_dump(mode="json", exclude_none=False),
            "evidence": [item.model_dump(mode="json", exclude_none=False) for item in evidence_tuple],
            "excluded_subject_refs": [
                item.model_dump(mode="json", exclude_none=False) for item in excluded_tuple
            ],
            "returned_characters": returned_characters,
            "max_characters": request.max_characters,
            "policy_version": self.policy_version,
        }
        evidence_bundle = EvidenceBundle(
            evidence_bundle_id=_stable_id("evidence-bundle", bundle_seed),
            source_trace_id=request.source_trace_id,
            trace_ir_version_id=request.trace_ir_version_id,
            consumer_stage=request.consumer_stage,
            purpose=request.purpose,
            projection_policy_ref=policy_ref,
            evidence=evidence_tuple,
            excluded_subject_refs=excluded_tuple,
            returned_characters=returned_characters,
            max_characters=request.max_characters,
            tainted_content_included=False,
            bundle_sha256=_stable_hash(bundle_seed),
            audit=request.audit,
        )
        return EvidenceBundleCompilationResult(
            evidence_bundle=evidence_bundle,
            uncertainties=tuple(sorted(uncertainties, key=lambda item: item.value)),
            policy_version=self.policy_version,
            audit=request.audit,
        )


class ArtifactEvidenceMatrixCompiler:
    policy_version = EVIDENCE_COMPILATION_POLICY_VERSION

    def compile(
        self,
        request: ArtifactEvidenceMatrixCompileRequest,
    ) -> ArtifactEvidenceMatrixCompilationResult:
        validate_projection_policy(request.view_result.projection_policy)
        _validate_matrix_binding(request)
        timelines = _timelines_by_path(request.timelines, request.evidence_bundle.trace_ir_version_id)
        view_index = _ViewIndex(request.view_result, request.evidence_bundle)
        _validate_targets(request.targets, view_index)
        rows = tuple(
            _row_for_target(
                target=target,
                timeline=timelines.get(target.logical_path),
                view_index=view_index,
                policy_version=self.policy_version,
            )
            for target in sorted(request.targets, key=lambda item: (item.logical_path, item.artifact_id))
        )
        aggregate_mode = _aggregate_mode(rows)
        matrix = ArtifactEvidenceMatrix(
            artifact_evidence_matrix_id=("artifact-evidence-matrix://pending"),
            producer_task_view_ref=request.producer_task_view_ref,
            rows=rows,
            aggregate_mode=aggregate_mode,
            audit=request.audit,
        )
        matrix_digest = artifact_evidence_matrix_carried_sha256(matrix)
        matrix = matrix.model_copy(
            update={"artifact_evidence_matrix_id": (f"artifact-evidence-matrix://sha256/{matrix_digest}")}
        )
        return ArtifactEvidenceMatrixCompilationResult(
            artifact_evidence_matrix=matrix,
            uncertainties=_result_uncertainties(rows),
            policy_version=self.policy_version,
            audit=request.audit,
        )


class _ViewIndex:
    def __init__(self, view_result: EvidenceViewResult, bundle: EvidenceBundle) -> None:
        self.included_by_source = {item.source_ref: item for item in view_result.included_items}
        self.excluded_by_source = {item.subject_ref: item for item in view_result.excluded_subjects}
        self.bundle_evidence_by_projected = {item.subject_ref: item for item in bundle.evidence}
        self.bundle_excluded_refs = frozenset(bundle.excluded_subject_refs)

    @property
    def known_source_refs(self) -> frozenset[ObjectRef]:
        return (
            frozenset(self.included_by_source)
            | frozenset(self.excluded_by_source)
            | self.bundle_excluded_refs
        )

    def included_item(self, source_ref: ObjectRef) -> EvidenceProjectionItem | None:
        return self.included_by_source.get(source_ref)

    def bundle_evidence(self, item: EvidenceProjectionItem) -> EvidenceRef | None:
        return self.bundle_evidence_by_projected.get(item.projected_ref)

    def excluded_reason(self, source_ref: ObjectRef) -> str | None:
        exclusion = self.excluded_by_source.get(source_ref)
        if exclusion is not None:
            return exclusion.reason.value
        if source_ref in self.bundle_excluded_refs:
            return EvidenceCompilationUncertainty.MISSING_SOURCE_SPAN_BINDING.value
        return None


def _bindings_by_projection_id(
    bindings: tuple[ProjectionEvidenceBinding, ...],
    included_by_id: dict[str, EvidenceProjectionItem],
) -> dict[str, ProjectionEvidenceBinding]:
    by_id: dict[str, ProjectionEvidenceBinding] = {}
    for binding in bindings:
        if binding.projection_item_id in by_id:
            raise EvidenceCompilationPolicyError("duplicate projection evidence binding")
        if binding.projection_item_id not in included_by_id:
            raise EvidenceCompilationPolicyError("unknown projection item binding")
        by_id[binding.projection_item_id] = binding
    return by_id


def _validate_binding_spans(binding: ProjectionEvidenceBinding, source_trace_id: str) -> None:
    for span in binding.source_spans:
        if span.source_trace_id != source_trace_id:
            raise EvidenceCompilationPolicyError("source span trace mismatch")


def _evidence_ref(
    *,
    item: EvidenceProjectionItem,
    binding: ProjectionEvidenceBinding,
    policy_version: str,
) -> EvidenceRef:
    payload = {
        "projection_item_id": item.projection_item_id,
        "projected_ref": item.projected_ref.model_dump(mode="json", exclude_none=False),
        "source_spans": [span.model_dump(mode="json", exclude_none=False) for span in binding.source_spans],
        "polarity": binding.polarity.value,
        "capability": binding.capability,
        "capability_complete": binding.capability_complete,
        "policy_version": policy_version,
    }
    return EvidenceRef(
        evidence_ref_id=_stable_id("evidence-ref", payload),
        subject_ref=item.projected_ref,
        source_spans=binding.source_spans,
        polarity=binding.polarity,
        capability=binding.capability,
        capability_complete=binding.capability_complete,
    )


def _validate_matrix_binding(request: ArtifactEvidenceMatrixCompileRequest) -> None:
    expected_policy_ref = projection_policy_ref(request.view_result.projection_policy)
    if request.evidence_bundle.projection_policy_ref != expected_policy_ref:
        raise EvidenceCompilationPolicyError("bundle projection policy does not match view result")
    projected_refs = {item.projected_ref for item in request.view_result.included_items}
    for evidence in request.evidence_bundle.evidence:
        if evidence.subject_ref not in projected_refs:
            raise EvidenceCompilationPolicyError("bundle evidence references unknown projected subject")


def _timelines_by_path(
    timelines: tuple[FileVersionTimeline, ...],
    trace_ir_version_id: str,
) -> dict[str, FileVersionTimeline]:
    by_path: dict[str, FileVersionTimeline] = {}
    for timeline in timelines:
        if timeline.trace_ir_version_id != trace_ir_version_id:
            raise EvidenceCompilationPolicyError("timeline trace does not match evidence bundle")
        if timeline.logical_path in by_path:
            raise EvidenceCompilationPolicyError("duplicate timeline logical path")
        by_path[timeline.logical_path] = timeline
    return by_path


def _validate_targets(targets: tuple[ArtifactEvidenceTarget, ...], view_index: _ViewIndex) -> None:
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for target in targets:
        if target.artifact_id in seen_ids or target.logical_path in seen_paths:
            raise EvidenceCompilationPolicyError("duplicate artifact target identity")
        seen_ids.add(target.artifact_id)
        seen_paths.add(target.logical_path)
        for source_ref in target.candidate_source_refs:
            if source_ref not in view_index.known_source_refs:
                raise EvidenceCompilationPolicyError("unknown target source reference")
            item = view_index.included_item(source_ref)
            if (
                item is not None
                and view_index.bundle_evidence(item) is None
                and source_ref not in view_index.bundle_excluded_refs
            ):
                raise EvidenceCompilationPolicyError("missing bundle projected evidence")


def _row_for_target(
    *,
    target: ArtifactEvidenceTarget,
    timeline: FileVersionTimeline | None,
    view_index: _ViewIndex,
    policy_version: str,
) -> ArtifactEvidenceRow:
    items = tuple(
        item for ref in target.candidate_source_refs if (item := view_index.included_item(ref)) is not None
    )
    bundle_evidence = tuple(
        evidence for item in items if (evidence := view_index.bundle_evidence(item)) is not None
    )
    relevant_evidence = _unique_evidence_refs(target.requirement_evidence + bundle_evidence)
    blocking = _blocking_uncertainties(
        target=target,
        timeline=timeline,
        view_index=view_index,
    )
    structure_coverage = _structure_coverage(target, items)
    pre_mutation_coverage = _pre_mutation_coverage(timeline)
    untainted_content_coverage = _untainted_content_coverage(
        pre_mutation_coverage=pre_mutation_coverage,
        items=items,
        view_index=view_index,
    )
    provenance_confidence = _provenance_confidence(items, view_index)
    truncation = _truncation(timeline, target.candidate_source_refs)
    path_evidence = _path_evidence(target, timeline, items)
    type_evidence = _type_evidence(target, items)
    mode = _selected_mode(
        target=target,
        structure_coverage=structure_coverage,
        pre_mutation_coverage=pre_mutation_coverage,
        untainted_content_coverage=untainted_content_coverage,
        provenance_confidence=provenance_confidence,
        truncation=truncation,
        blocking_uncertainties=blocking,
    )
    return ArtifactEvidenceRow(
        artifact_id=target.artifact_id,
        logical_path=target.logical_path,
        media_type=target.media_type,
        path_evidence=path_evidence,
        type_evidence=type_evidence,
        structure_coverage=structure_coverage,
        untainted_content_coverage=untainted_content_coverage,
        pre_mutation_coverage=pre_mutation_coverage,
        provenance_confidence=provenance_confidence,
        truncation=truncation,
        criticality=target.criticality,
        blocking_uncertainties=blocking,
        evidence=relevant_evidence,
        selected_mode=mode,
    )


def _path_evidence(
    target: ArtifactEvidenceTarget,
    timeline: FileVersionTimeline | None,
    items: tuple[EvidenceProjectionItem, ...],
) -> EvidenceStrength:
    if timeline is not None or _structure_value(items, "normalized-path") == target.logical_path:
        return EvidenceStrength.DIRECT
    if target.requirement_evidence:
        return EvidenceStrength.INFERRED
    return EvidenceStrength.MISSING


def _type_evidence(
    target: ArtifactEvidenceTarget,
    items: tuple[EvidenceProjectionItem, ...],
) -> EvidenceStrength:
    if _structure_value(items, "media-type") == target.media_type:
        return EvidenceStrength.DIRECT
    if target.requirement_evidence:
        return EvidenceStrength.INFERRED
    return EvidenceStrength.MISSING


def _structure_coverage(
    target: ArtifactEvidenceTarget,
    items: tuple[EvidenceProjectionItem, ...],
) -> EvidenceCoverage:
    structure_fields = {
        field.key: field.value
        for item in items
        if item.projection_mode is EvidenceProjectionMode.STRUCTURE
        for field in item.structure_fields
    }
    if not structure_fields:
        return EvidenceCoverage.NONE
    if (
        structure_fields.get("normalized-path") == target.logical_path
        and structure_fields.get("media-type") == target.media_type
    ):
        return EvidenceCoverage.COMPLETE
    return EvidenceCoverage.PARTIAL


def _pre_mutation_coverage(timeline: FileVersionTimeline | None) -> EvidenceCoverage:
    if timeline is None:
        return EvidenceCoverage.NONE
    version = timeline.versions[0]
    if FileVersionUncertainty.NO_PRE_MUTATION_READ in version.uncertainties:
        return EvidenceCoverage.NONE
    if version.origin is not FileVersionOrigin.PREEXISTING_WORKSPACE_INPUT:
        return EvidenceCoverage.UNKNOWN
    if (
        FileVersionUncertainty.UNKNOWN_MUTATION_BOUNDARY in version.uncertainties
        or FileVersionUncertainty.UNKNOWN_COMPLETENESS in version.uncertainties
        or FileVersionUncertainty.TRUNCATION_UNKNOWN in version.uncertainties
    ):
        return EvidenceCoverage.UNKNOWN
    if not version.content_refs or not version.content_sha256s:
        return EvidenceCoverage.UNKNOWN
    if version.completeness is Completeness.COMPLETE:
        return EvidenceCoverage.COMPLETE
    if version.completeness is Completeness.PARTIAL:
        return EvidenceCoverage.PARTIAL
    return EvidenceCoverage.UNKNOWN


def _untainted_content_coverage(
    *,
    pre_mutation_coverage: EvidenceCoverage,
    items: tuple[EvidenceProjectionItem, ...],
    view_index: _ViewIndex,
) -> EvidenceCoverage:
    content_items = tuple(item for item in items if item.projection_mode is EvidenceProjectionMode.CONTENT)
    if not content_items:
        return EvidenceCoverage.NONE
    if pre_mutation_coverage not in {EvidenceCoverage.COMPLETE, EvidenceCoverage.PARTIAL}:
        return EvidenceCoverage.NONE
    if any(
        (evidence := view_index.bundle_evidence(item)) is not None and not evidence.capability_complete
        for item in content_items
    ):
        return EvidenceCoverage.PARTIAL
    return pre_mutation_coverage


def _provenance_confidence(
    items: tuple[EvidenceProjectionItem, ...],
    view_index: _ViewIndex,
) -> float:
    confidences = [
        item.child_decision.confidence for item in items if view_index.bundle_evidence(item) is not None
    ]
    if not confidences:
        return 0.0
    return min(confidences)


def _truncation(
    timeline: FileVersionTimeline | None,
    candidate_refs: tuple[ObjectRef, ...],
) -> Literal["NONE", "PRESENT", "UNKNOWN"]:
    if timeline is None:
        return "UNKNOWN" if candidate_refs else "NONE"
    version = timeline.versions[0]
    if version.truncated is True:
        return "PRESENT"
    if version.truncated is None:
        return "UNKNOWN"
    return "NONE"


def _blocking_uncertainties(
    *,
    target: ArtifactEvidenceTarget,
    timeline: FileVersionTimeline | None,
    view_index: _ViewIndex,
) -> tuple[str, ...]:
    values: set[str] = set()
    if timeline is None and target.candidate_source_refs:
        values.add(_uncertainty_id(EvidenceCompilationUncertainty.MISSING_TIMELINE.value))
    if timeline is not None:
        for uncertainty in timeline.versions[0].uncertainties:
            if uncertainty is FileVersionUncertainty.NO_PRE_MUTATION_READ:
                continue
            values.add(_uncertainty_id(f"timeline/{uncertainty.value}"))
    for source_ref in target.candidate_source_refs:
        reason = view_index.excluded_reason(source_ref)
        if reason is not None:
            values.add(_uncertainty_id(f"view/{reason}"))
        item = view_index.included_item(source_ref)
        if item is not None and view_index.bundle_evidence(item) is None:
            values.add(_uncertainty_id(EvidenceCompilationUncertainty.MISSING_SOURCE_SPAN_BINDING.value))
    if (
        target.criticality is AttachmentCriticality.CRITICAL
        and _pre_mutation_coverage(timeline) is EvidenceCoverage.NONE
    ):
        values.add(_uncertainty_id(EvidenceCompilationUncertainty.MISSING_PRE_MUTATION_EVIDENCE.value))
    return tuple(sorted(values))


def _selected_mode(
    *,
    target: ArtifactEvidenceTarget,
    structure_coverage: EvidenceCoverage,
    pre_mutation_coverage: EvidenceCoverage,
    untainted_content_coverage: EvidenceCoverage,
    provenance_confidence: float,
    truncation: str,
    blocking_uncertainties: tuple[str, ...],
) -> ReconstructionMode:
    critical_blocked = bool(blocking_uncertainties) and target.criticality is AttachmentCriticality.CRITICAL
    if critical_blocked:
        return ReconstructionMode.BLOCKED
    if (
        pre_mutation_coverage in {EvidenceCoverage.COMPLETE, EvidenceCoverage.PARTIAL}
        and untainted_content_coverage in {EvidenceCoverage.COMPLETE, EvidenceCoverage.PARTIAL}
        and truncation == "NONE"
        and provenance_confidence >= _CONFIDENCE_THRESHOLD
    ):
        return ReconstructionMode.TRACE_RICH
    if structure_coverage in {EvidenceCoverage.COMPLETE, EvidenceCoverage.PARTIAL}:
        return ReconstructionMode.SKELETON_GUIDED
    if target.requirement_evidence:
        return ReconstructionMode.PROMPT_ONLY
    return ReconstructionMode.BLOCKED


def _aggregate_mode(
    rows: tuple[ArtifactEvidenceRow, ...],
) -> Literal["TRACE_RICH", "SKELETON_GUIDED", "PROMPT_ONLY", "MIXED", "BLOCKED"]:
    modes = {row.selected_mode for row in rows}
    if len(modes) == 1:
        return next(iter(modes)).value
    if any(
        row.criticality is AttachmentCriticality.CRITICAL and row.selected_mode is ReconstructionMode.BLOCKED
        for row in rows
    ):
        return "BLOCKED"
    return "MIXED"


def _result_uncertainties(
    rows: tuple[ArtifactEvidenceRow, ...],
) -> tuple[EvidenceCompilationUncertainty, ...]:
    values: set[EvidenceCompilationUncertainty] = set()
    for row in rows:
        for uncertainty in row.blocking_uncertainties:
            if EvidenceCompilationUncertainty.MISSING_SOURCE_SPAN_BINDING.value in uncertainty:
                values.add(EvidenceCompilationUncertainty.MISSING_SOURCE_SPAN_BINDING)
            elif EvidenceCompilationUncertainty.MISSING_TIMELINE.value in uncertainty:
                values.add(EvidenceCompilationUncertainty.MISSING_TIMELINE)
            elif "timeline/" in uncertainty:
                values.add(EvidenceCompilationUncertainty.TIMELINE_UNCERTAINTY)
            elif "view/" in uncertainty:
                values.add(EvidenceCompilationUncertainty.VIEW_SUBJECT_EXCLUDED)
            elif EvidenceCompilationUncertainty.MISSING_PRE_MUTATION_EVIDENCE.value in uncertainty:
                values.add(EvidenceCompilationUncertainty.MISSING_PRE_MUTATION_EVIDENCE)
    return tuple(sorted(values, key=lambda item: item.value))


def _structure_value(items: tuple[EvidenceProjectionItem, ...], key: str) -> object | None:
    for item in items:
        if item.projection_mode is not EvidenceProjectionMode.STRUCTURE:
            continue
        for field in item.structure_fields:
            if field.key == key:
                return field.value
    return None


def _view_excluded_refs(view_result: EvidenceViewResult) -> tuple[ObjectRef, ...]:
    return tuple(item.subject_ref for item in view_result.excluded_subjects)


def _item_returned_characters(item: EvidenceProjectionItem) -> int:
    total = len(item.content or "")
    total += len(item.external_uri or "")
    for field in item.structure_fields:
        if isinstance(field.value, str):
            total += len(field.value)
    return total


def projection_policy_ref(policy: ProjectionPolicy) -> ObjectRef:
    return ObjectRef(
        object_type="projection-policy",
        object_id=policy.projection_policy_id,
        object_version=policy.policy_version,
        object_sha256=policy.canonical_sha256(),
    )


def validate_projection_policy(policy: ProjectionPolicy) -> None:
    expected_payload = {
        "principal_type": policy.principal_type,
        "purpose": policy.purpose,
        "recursive_allow_fields": list(policy.recursive_allow_fields),
        "denied_object_types": list(policy.denied_object_types),
        "source_schema_versions": list(policy.source_schema_versions),
        "policy_version": policy.policy_version,
    }
    expected_id = _stable_id("projection-policy", expected_payload)
    if policy.projection_policy_id != expected_id:
        raise EvidenceCompilationPolicyError("projection policy identity is stale or invalid")


def evidence_bundle_carried_sha256(bundle: EvidenceBundle) -> str:
    return _stable_hash(
        {
            "source_trace_id": bundle.source_trace_id,
            "trace_ir_version_id": bundle.trace_ir_version_id,
            "consumer_stage": bundle.consumer_stage,
            "purpose": bundle.purpose,
            "projection_policy_ref": bundle.projection_policy_ref.model_dump(
                mode="json",
                exclude_none=False,
            ),
            "evidence": [item.model_dump(mode="json", exclude_none=False) for item in bundle.evidence],
            "excluded_subject_refs": [
                item.model_dump(mode="json", exclude_none=False) for item in bundle.excluded_subject_refs
            ],
            "returned_characters": bundle.returned_characters,
            "max_characters": bundle.max_characters,
            "policy_version": EVIDENCE_COMPILATION_POLICY_VERSION,
        }
    )


def evidence_bundle_ref(bundle: EvidenceBundle) -> ObjectRef:
    return ObjectRef(
        object_type="evidence-bundle",
        object_id=bundle.evidence_bundle_id,
        object_version="v1",
        object_sha256=bundle.bundle_sha256,
    )


def validate_evidence_bundle_identity(bundle: EvidenceBundle) -> None:
    digest = evidence_bundle_carried_sha256(bundle)
    if bundle.bundle_sha256 != digest or bundle.evidence_bundle_id != f"evidence-bundle://sha256/{digest}":
        raise EvidenceCompilationPolicyError("evidence bundle identity is stale or invalid")


def artifact_evidence_matrix_carried_sha256(
    matrix: ArtifactEvidenceMatrix,
) -> str:
    return _stable_hash(
        {
            "producer_task_view_ref": matrix.producer_task_view_ref.model_dump(
                mode="json",
                exclude_none=False,
            ),
            "rows": [item.model_dump(mode="json", exclude_none=False) for item in matrix.rows],
            "aggregate_mode": matrix.aggregate_mode,
            "policy_version": EVIDENCE_COMPILATION_POLICY_VERSION,
        }
    )


def artifact_evidence_matrix_ref(
    matrix: ArtifactEvidenceMatrix,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-evidence-matrix",
        object_id=matrix.artifact_evidence_matrix_id,
        object_version="v1",
        object_sha256=artifact_evidence_matrix_carried_sha256(matrix),
    )


def validate_artifact_evidence_matrix_identity(
    matrix: ArtifactEvidenceMatrix,
) -> None:
    digest = artifact_evidence_matrix_carried_sha256(matrix)
    if matrix.artifact_evidence_matrix_id != (f"artifact-evidence-matrix://sha256/{digest}"):
        raise EvidenceCompilationPolicyError("artifact evidence matrix identity is stale or invalid")


def _unique_object_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    refs: dict[tuple[str, str, str, str], ObjectRef] = {}
    for value in values:
        refs.setdefault(_object_ref_key(value), value)
    return tuple(refs[key] for key in sorted(refs))


def _unique_evidence_refs(values: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
    refs: dict[str, EvidenceRef] = {}
    for value in values:
        refs.setdefault(value.evidence_ref_id, value)
    return tuple(refs[key] for key in sorted(refs))


def _object_ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (value.object_type, value.object_id, value.object_version, value.object_sha256)


def _uncertainty_id(value: str) -> str:
    return f"artifact-evidence/{value}"


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(kind: str, payload: object) -> str:
    return f"{kind}://sha256/{_stable_hash(payload)}"
