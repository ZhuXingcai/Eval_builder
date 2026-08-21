from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.semantic_review_v2 import (
    AttachmentRepairRequestV2,
    AttachmentRepairResultV2,
    AttachmentRepairSourceV2,
    AttachmentSemanticFindingResolutionV2,
    AttachmentSemanticFindingScopeV2,
    AttachmentSemanticReviewFindingCodeV2,
    AttachmentSemanticReviewFindingV2,
    attachment_repair_request_ref,
    attachment_repair_result_ref,
    attachment_semantic_finding_policy,
    attachment_semantic_finding_resolution_ref,
    attachment_semantic_review_finding_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    RelativePath,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.quality import Severity

SEMANTIC_REVIEW_POLICY_VERSION: Literal["semantic-review/r5-09-v1"] = "semantic-review/r5-09-v1"
CANDIDATE_REVISION_POLICY_VERSION: Literal["candidate-revision/r5-09-v1"] = "candidate-revision/r5-09-v1"
TARGETED_REPAIR_POLICY_VERSION: Literal["targeted-repair/r5-09-v1"] = "targeted-repair/r5-09-v1"


class SemanticReviewRoundV2(StrEnum):
    COVERAGE_SOLVABILITY = "COVERAGE_SOLVABILITY"
    REALISM_CONSISTENCY = "REALISM_CONSISTENCY"
    LEAKAGE_EXECUTABILITY = "LEAKAGE_EXECUTABILITY"


class SemanticReviewerRoleV2(StrEnum):
    COVERAGE_SOLVABILITY_REVIEWER = "COVERAGE_SOLVABILITY_REVIEWER"
    REALISM_CONSISTENCY_REVIEWER = "REALISM_CONSISTENCY_REVIEWER"
    LEAKAGE_EXECUTABILITY_REVIEWER = "LEAKAGE_EXECUTABILITY_REVIEWER"


ROUND_ROLES: dict[SemanticReviewRoundV2, SemanticReviewerRoleV2] = {
    SemanticReviewRoundV2.COVERAGE_SOLVABILITY: (SemanticReviewerRoleV2.COVERAGE_SOLVABILITY_REVIEWER),
    SemanticReviewRoundV2.REALISM_CONSISTENCY: (SemanticReviewerRoleV2.REALISM_CONSISTENCY_REVIEWER),
    SemanticReviewRoundV2.LEAKAGE_EXECUTABILITY: (SemanticReviewerRoleV2.LEAKAGE_EXECUTABILITY_REVIEWER),
}


class RevisionDeterministicValidationOutcomeV2(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PASSED = "PASSED"
    UPSTREAM_INCOMPLETE = "UPSTREAM_INCOMPLETE"
    REQUIRES_REPAIR = "REQUIRES_REPAIR"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


class SemanticReviewRoundOutcomeV2(StrEnum):
    ACCEPTED = "ACCEPTED"
    REQUIRES_REPAIR = "REQUIRES_REPAIR"
    REJECTED = "REJECTED"
    ABSTAINED = "ABSTAINED"
    BLOCKED = "BLOCKED"


class SemanticReviewWorkflowOutcomeV2(StrEnum):
    PASSED = "PASSED"
    REQUIRES_REPAIR = "REQUIRES_REPAIR"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    UPSTREAM_INCOMPLETE = "UPSTREAM_INCOMPLETE"


class SemanticFindingScopeV2(StrEnum):
    ITEM = "ITEM"
    ARTIFACT = "ARTIFACT"


class TargetedRepairPlanV2(ContractModelV2):
    schema_version: Literal["eval-factory/targeted-repair-plan/v2"] = "eval-factory/targeted-repair-plan/v2"
    targeted_repair_plan_id: Identifier
    source_round: AttachmentRepairSourceV2
    candidate_revision_ref: ObjectRef
    deterministic_validation_result_ref: ObjectRef
    triggering_round_result_ref: ObjectRef | None = None
    targeted_finding_refs: tuple[ObjectRef, ...]
    targeted_artifact_ids: tuple[Identifier, ...]
    untouched_artifact_version_refs: tuple[ObjectRef, ...]
    max_repair_waves: Literal[1] = 1
    policy_version: Literal["targeted-repair/r5-09-v1"] = TARGETED_REPAIR_POLICY_VERSION
    plan_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_plan(self) -> TargetedRepairPlanV2:
        if not self.targeted_finding_refs or not self.targeted_artifact_ids:
            raise ValueError("targeted repair plan requires findings and artifacts")
        _require_sorted_unique_refs(
            "targeted repair finding refs",
            self.targeted_finding_refs,
        )
        _require_sorted_unique(
            "targeted repair artifact IDs",
            self.targeted_artifact_ids,
        )
        _require_sorted_unique_refs(
            "untouched artifact version refs",
            self.untouched_artifact_version_refs,
        )
        _validate_final_identity(
            object_id=self.targeted_repair_plan_id,
            object_sha256=self.plan_sha256,
            expected_prefix="targeted-repair-plan",
            observed=targeted_repair_plan_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        source_round: AttachmentRepairSourceV2,
        candidate_revision_ref: ObjectRef,
        deterministic_validation_result_ref: ObjectRef,
        triggering_round_result_ref: ObjectRef | None,
        targeted_finding_refs: tuple[ObjectRef, ...],
        targeted_artifact_ids: tuple[Identifier, ...],
        untouched_artifact_version_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> TargetedRepairPlanV2:
        audit_refs = (
            candidate_revision_ref,
            deterministic_validation_result_ref,
            *targeted_finding_refs,
            *untouched_artifact_version_refs,
            *((triggering_round_result_ref,) if triggering_round_result_ref is not None else ()),
        )
        value = cls(
            targeted_repair_plan_id="targeted-repair-plan://pending",
            source_round=source_round,
            candidate_revision_ref=candidate_revision_ref,
            deterministic_validation_result_ref=(deterministic_validation_result_ref),
            triggering_round_result_ref=triggering_round_result_ref,
            targeted_finding_refs=_sorted_refs(targeted_finding_refs),
            targeted_artifact_ids=tuple(sorted(targeted_artifact_ids)),
            untouched_artifact_version_refs=_sorted_refs(untouched_artifact_version_refs),
            plan_sha256="0" * 64,
            audit=_safe_audit(audit, audit_refs),
        )
        return _finalize(
            value,
            id_field="targeted_repair_plan_id",
            hash_field="plan_sha256",
            prefix="targeted-repair-plan",
            digest=targeted_repair_plan_carried_sha256(value),
        )


class RepairedArtifactBuildResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/repaired-artifact-build-result/v2"] = (
        "eval-factory/repaired-artifact-build-result/v2"
    )
    repaired_artifact_build_result_id: Identifier
    artifact_id: Identifier
    source_artifact_version_ref: ObjectRef
    source_artifact_build_result_ref: ObjectRef
    source_output_ref: ObjectRef
    build_spec_ref: ObjectRef
    repair_plan_ref: ObjectRef
    facade_repair_request_ref: ObjectRef
    facade_repair_result: AttachmentRepairResultV2
    targeted_finding_refs: tuple[ObjectRef, ...]
    output_ref: ObjectRef
    output_sha256: Sha256
    worker_version: str
    artifact_version: int = Field(ge=2)
    policy_version: Literal["targeted-repair/r5-09-v1"] = TARGETED_REPAIR_POLICY_VERSION
    repaired_result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> RepairedArtifactBuildResultV2:
        facade_ref = _object_ref_from_facade(attachment_repair_result_ref(self.facade_repair_result))
        if (
            self.facade_repair_request_ref
            != _object_ref_from_facade(self.facade_repair_result.repair_request_ref)
            or self.facade_repair_result.artifact_id != self.artifact_id
            or self.facade_repair_result.source_output_ref.object_sha256
            != self.source_output_ref.object_sha256
            or self.facade_repair_result.output_ref is None
            or self.facade_repair_result.output_sha256 is None
            or _object_ref_from_facade(self.facade_repair_result.output_ref) != self.output_ref
            or self.output_sha256 != self.facade_repair_result.output_sha256
            or self.worker_version != self.facade_repair_result.worker_version
        ):
            raise ValueError("repaired artifact result does not match facade result")
        if self.output_sha256 == self.source_output_ref.object_sha256:
            raise ValueError("repaired artifact output must change")
        _require_sorted_unique_refs(
            "repaired artifact finding refs",
            self.targeted_finding_refs,
        )
        _validate_audit(
            self.audit,
            (
                self.source_artifact_version_ref,
                self.source_artifact_build_result_ref,
                self.source_output_ref,
                self.build_spec_ref,
                self.repair_plan_ref,
                self.facade_repair_request_ref,
                facade_ref,
                *self.targeted_finding_refs,
            ),
            "repaired artifact result",
        )
        _validate_final_identity(
            object_id=self.repaired_artifact_build_result_id,
            object_sha256=self.repaired_result_sha256,
            expected_prefix="repaired-artifact-build-result",
            observed=repaired_artifact_build_result_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        source: CandidateArtifactVersionV2,
        repair_plan_ref: ObjectRef,
        facade_request: AttachmentRepairRequestV2,
        facade_result: AttachmentRepairResultV2,
        targeted_finding_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> RepairedArtifactBuildResultV2:
        if (
            facade_result.output_ref is None
            or facade_result.output_sha256 is None
            or facade_result.worker_version is None
        ):
            raise ValueError("successful facade repair result is required")
        request_ref = _object_ref_from_facade(attachment_repair_request_ref(facade_request))
        result_ref = _object_ref_from_facade(attachment_repair_result_ref(facade_result))
        value = cls(
            repaired_artifact_build_result_id=("repaired-artifact-build-result://pending"),
            artifact_id=source.artifact_id,
            source_artifact_version_ref=candidate_artifact_version_ref(source),
            source_artifact_build_result_ref=(source.base_artifact_build_result_ref),
            source_output_ref=source.output_ref,
            build_spec_ref=source.build_spec_ref,
            repair_plan_ref=repair_plan_ref,
            facade_repair_request_ref=request_ref,
            facade_repair_result=facade_result,
            targeted_finding_refs=_sorted_refs(targeted_finding_refs),
            output_ref=_object_ref_from_facade(facade_result.output_ref),
            output_sha256=facade_result.output_sha256,
            worker_version=facade_result.worker_version,
            artifact_version=source.artifact_version + 1,
            repaired_result_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    candidate_artifact_version_ref(source),
                    source.base_artifact_build_result_ref,
                    source.output_ref,
                    source.build_spec_ref,
                    repair_plan_ref,
                    request_ref,
                    result_ref,
                    *targeted_finding_refs,
                ),
            ),
        )
        return _finalize(
            value,
            id_field="repaired_artifact_build_result_id",
            hash_field="repaired_result_sha256",
            prefix="repaired-artifact-build-result",
            digest=repaired_artifact_build_result_carried_sha256(value),
        )


class CandidateArtifactVersionV2(ContractModelV2):
    schema_version: Literal["eval-factory/candidate-artifact-version/v2"] = (
        "eval-factory/candidate-artifact-version/v2"
    )
    candidate_artifact_version_id: Identifier
    artifact_id: Identifier
    attachment_dependency_id: Identifier
    artifact_version: int = Field(ge=1)
    base_artifact_build_result_ref: ObjectRef
    predecessor_artifact_version_ref: ObjectRef | None = None
    repaired_artifact_result_ref: ObjectRef | None = None
    build_spec_ref: ObjectRef
    execution_request_ref: ObjectRef
    execution_result_ref: ObjectRef
    output_ref: ObjectRef
    output_sha256: Sha256
    logical_path: RelativePath
    media_type: str = Field(
        min_length=3,
        max_length=255,
        pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$",
    )
    declared_validator_ids: tuple[Identifier, ...]
    derivation_root_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    policy_version: Literal["candidate-revision/r5-09-v1"] = CANDIDATE_REVISION_POLICY_VERSION
    artifact_version_sha256: Sha256

    @model_validator(mode="after")
    def validate_artifact_version(self) -> CandidateArtifactVersionV2:
        _require_ref(
            self.base_artifact_build_result_ref,
            "artifact-build-result",
            "base_artifact_build_result_ref",
        )
        _require_ref(self.build_spec_ref, "artifact-build-spec", "build_spec_ref")
        _require_ref(
            self.execution_request_ref,
            "attachment-execution-request",
            "execution_request_ref",
        )
        _require_ref(
            self.execution_result_ref,
            "attachment-execution-result",
            "execution_result_ref",
        )
        _require_ref(self.output_ref, "attachment-output", "output_ref")
        if self.output_ref.object_sha256 != self.output_sha256:
            raise ValueError("candidate artifact output ref/hash mismatch")
        _require_sorted_unique(
            "declared validator IDs",
            self.declared_validator_ids,
        )
        _require_sorted_unique_refs(
            "artifact derivation roots",
            self.derivation_root_refs,
        )
        if self.artifact_version == 1:
            if (
                self.predecessor_artifact_version_ref is not None
                or self.repaired_artifact_result_ref is not None
            ):
                raise ValueError("base candidate artifact cannot carry repair predecessor")
        elif self.predecessor_artifact_version_ref is None or self.repaired_artifact_result_ref is None:
            raise ValueError("repaired candidate artifact requires predecessor and repair result")
        _validate_final_identity(
            object_id=self.candidate_artifact_version_id,
            object_sha256=self.artifact_version_sha256,
            expected_prefix="candidate-artifact-version",
            observed=candidate_artifact_version_carried_sha256(self),
        )
        return self

    @classmethod
    def create_base(
        cls,
        *,
        artifact_id: Identifier,
        attachment_dependency_id: Identifier,
        artifact_build_result_ref: ObjectRef,
        build_spec_ref: ObjectRef,
        execution_request_ref: ObjectRef,
        execution_result_ref: ObjectRef,
        output_ref: ObjectRef,
        logical_path: RelativePath,
        media_type: str,
        declared_validator_ids: tuple[Identifier, ...],
        derivation_root_refs: tuple[ObjectRef, ...],
    ) -> CandidateArtifactVersionV2:
        value = cls(
            candidate_artifact_version_id="candidate-artifact-version://pending",
            artifact_id=artifact_id,
            attachment_dependency_id=attachment_dependency_id,
            artifact_version=1,
            base_artifact_build_result_ref=artifact_build_result_ref,
            predecessor_artifact_version_ref=None,
            repaired_artifact_result_ref=None,
            build_spec_ref=build_spec_ref,
            execution_request_ref=execution_request_ref,
            execution_result_ref=execution_result_ref,
            output_ref=output_ref,
            output_sha256=output_ref.object_sha256,
            logical_path=logical_path,
            media_type=media_type,
            declared_validator_ids=tuple(sorted(declared_validator_ids)),
            derivation_root_refs=_sorted_refs(derivation_root_refs),
            artifact_version_sha256="0" * 64,
        )
        return _finalize(
            value,
            id_field="candidate_artifact_version_id",
            hash_field="artifact_version_sha256",
            prefix="candidate-artifact-version",
            digest=candidate_artifact_version_carried_sha256(value),
        )

    @classmethod
    def create_repaired(
        cls,
        *,
        source: CandidateArtifactVersionV2,
        repaired_result: RepairedArtifactBuildResultV2,
    ) -> CandidateArtifactVersionV2:
        if (
            repaired_result.source_artifact_version_ref != candidate_artifact_version_ref(source)
            or repaired_result.artifact_id != source.artifact_id
        ):
            raise ValueError("repaired result belongs to another artifact version")
        value = cls(
            candidate_artifact_version_id="candidate-artifact-version://pending",
            artifact_id=source.artifact_id,
            attachment_dependency_id=source.attachment_dependency_id,
            artifact_version=source.artifact_version + 1,
            base_artifact_build_result_ref=source.base_artifact_build_result_ref,
            predecessor_artifact_version_ref=candidate_artifact_version_ref(source),
            repaired_artifact_result_ref=repaired_artifact_build_result_ref(repaired_result),
            build_spec_ref=source.build_spec_ref,
            execution_request_ref=source.execution_request_ref,
            execution_result_ref=source.execution_result_ref,
            output_ref=repaired_result.output_ref,
            output_sha256=repaired_result.output_sha256,
            logical_path=source.logical_path,
            media_type=source.media_type,
            declared_validator_ids=source.declared_validator_ids,
            derivation_root_refs=_sorted_refs(
                (
                    *source.derivation_root_refs,
                    repaired_artifact_build_result_ref(repaired_result),
                )
            ),
            artifact_version_sha256="0" * 64,
        )
        return _finalize(
            value,
            id_field="candidate_artifact_version_id",
            hash_field="artifact_version_sha256",
            prefix="candidate-artifact-version",
            digest=candidate_artifact_version_carried_sha256(value),
        )


class AttachmentCandidateRevisionV2(ContractModelV2):
    schema_version: Literal["eval-factory/attachment-candidate-revision/v2"] = (
        "eval-factory/attachment-candidate-revision/v2"
    )
    attachment_candidate_revision_id: Identifier
    revision: int = Field(ge=1)
    base_reconstruction_result_ref: ObjectRef
    predecessor_revision_ref: ObjectRef | None = None
    artifact_versions: tuple[CandidateArtifactVersionV2, ...]
    artifact_version_refs: tuple[ObjectRef, ...]
    candidate_output_refs: tuple[ObjectRef, ...]
    changed_artifact_ids: tuple[Identifier, ...]
    repair_plan_ref: ObjectRef | None = None
    repair_result_refs: tuple[ObjectRef, ...] = ()
    accepted_artifact_refs: tuple[ObjectRef, ...] = ()
    environment_spec_ref: None = None
    provenance_manifest_ref: None = None
    quality_report_ref: None = None
    package_sha256: None = None
    input_state_only: None = None
    policy_version: Literal["candidate-revision/r5-09-v1"] = CANDIDATE_REVISION_POLICY_VERSION
    candidate_revision_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_revision(self) -> AttachmentCandidateRevisionV2:
        _require_ref(
            self.base_reconstruction_result_ref,
            "attachment-reconstruction-result",
            "base_reconstruction_result_ref",
        )
        artifact_ids = tuple(item.artifact_id for item in self.artifact_versions)
        _require_sorted_unique("candidate artifact IDs", artifact_ids)
        expected_version_refs = tuple(candidate_artifact_version_ref(item) for item in self.artifact_versions)
        if self.artifact_version_refs != expected_version_refs:
            raise ValueError("candidate artifact version refs must match nested versions")
        expected_outputs = _sorted_refs(tuple(item.output_ref for item in self.artifact_versions))
        if self.candidate_output_refs != expected_outputs:
            raise ValueError("candidate output refs must match artifact versions")
        _require_sorted_unique("changed artifact IDs", self.changed_artifact_ids)
        _require_sorted_unique_refs("repair result refs", self.repair_result_refs)
        if self.revision == 1:
            if (
                self.predecessor_revision_ref is not None
                or self.changed_artifact_ids
                or self.repair_plan_ref is not None
                or self.repair_result_refs
            ):
                raise ValueError("initial candidate revision cannot carry repair lineage")
        elif (
            self.predecessor_revision_ref is None
            or not self.changed_artifact_ids
            or self.repair_plan_ref is None
            or not self.repair_result_refs
        ):
            raise ValueError("later candidate revision requires complete repair lineage")
        if self.accepted_artifact_refs:
            raise ValueError("accepted artifact refs remain empty before R5-10")
        _validate_audit(
            self.audit,
            (
                self.base_reconstruction_result_ref,
                *self.artifact_version_refs,
                *self.repair_result_refs,
                *((self.predecessor_revision_ref,) if self.predecessor_revision_ref is not None else ()),
                *((self.repair_plan_ref,) if self.repair_plan_ref is not None else ()),
            ),
            "candidate revision",
        )
        _validate_final_identity(
            object_id=self.attachment_candidate_revision_id,
            object_sha256=self.candidate_revision_sha256,
            expected_prefix="attachment-candidate-revision",
            observed=attachment_candidate_revision_carried_sha256(self),
        )
        return self

    @classmethod
    def create_initial(
        cls,
        *,
        base_reconstruction_result_ref: ObjectRef,
        artifact_versions: tuple[CandidateArtifactVersionV2, ...],
        audit: ContractAudit,
    ) -> AttachmentCandidateRevisionV2:
        ordered = tuple(sorted(artifact_versions, key=lambda item: item.artifact_id))
        version_refs = tuple(candidate_artifact_version_ref(item) for item in ordered)
        value = cls(
            attachment_candidate_revision_id="attachment-candidate-revision://pending",
            revision=1,
            base_reconstruction_result_ref=base_reconstruction_result_ref,
            predecessor_revision_ref=None,
            artifact_versions=ordered,
            artifact_version_refs=version_refs,
            candidate_output_refs=_sorted_refs(tuple(item.output_ref for item in ordered)),
            changed_artifact_ids=(),
            repair_plan_ref=None,
            repair_result_refs=(),
            accepted_artifact_refs=(),
            environment_spec_ref=None,
            provenance_manifest_ref=None,
            quality_report_ref=None,
            package_sha256=None,
            input_state_only=None,
            candidate_revision_sha256="0" * 64,
            audit=_safe_audit(audit, (base_reconstruction_result_ref, *version_refs)),
        )
        return _finalize(
            value,
            id_field="attachment_candidate_revision_id",
            hash_field="candidate_revision_sha256",
            prefix="attachment-candidate-revision",
            digest=attachment_candidate_revision_carried_sha256(value),
        )

    @classmethod
    def create_successor(
        cls,
        *,
        predecessor: AttachmentCandidateRevisionV2,
        repair_plan: TargetedRepairPlanV2,
        repaired_results: tuple[RepairedArtifactBuildResultV2, ...],
        audit: ContractAudit,
    ) -> AttachmentCandidateRevisionV2:
        repaired_by_id = {item.artifact_id: item for item in repaired_results}
        if tuple(sorted(repaired_by_id)) != repair_plan.targeted_artifact_ids:
            raise ValueError("repair results must exactly cover targeted artifacts")
        artifact_versions = tuple(
            CandidateArtifactVersionV2.create_repaired(
                source=item,
                repaired_result=repaired_by_id[item.artifact_id],
            )
            if item.artifact_id in repaired_by_id
            else item
            for item in predecessor.artifact_versions
        )
        version_refs = tuple(candidate_artifact_version_ref(item) for item in artifact_versions)
        repair_refs = tuple(
            repaired_artifact_build_result_ref(item)
            for item in sorted(repaired_results, key=lambda value: value.artifact_id)
        )
        value = cls(
            attachment_candidate_revision_id="attachment-candidate-revision://pending",
            revision=predecessor.revision + 1,
            base_reconstruction_result_ref=(predecessor.base_reconstruction_result_ref),
            predecessor_revision_ref=attachment_candidate_revision_ref(predecessor),
            artifact_versions=artifact_versions,
            artifact_version_refs=version_refs,
            candidate_output_refs=_sorted_refs(tuple(item.output_ref for item in artifact_versions)),
            changed_artifact_ids=repair_plan.targeted_artifact_ids,
            repair_plan_ref=targeted_repair_plan_ref(repair_plan),
            repair_result_refs=repair_refs,
            accepted_artifact_refs=(),
            environment_spec_ref=None,
            provenance_manifest_ref=None,
            quality_report_ref=None,
            package_sha256=None,
            input_state_only=None,
            candidate_revision_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    predecessor.base_reconstruction_result_ref,
                    attachment_candidate_revision_ref(predecessor),
                    targeted_repair_plan_ref(repair_plan),
                    *version_refs,
                    *repair_refs,
                ),
            ),
        )
        return _finalize(
            value,
            id_field="attachment_candidate_revision_id",
            hash_field="candidate_revision_sha256",
            prefix="attachment-candidate-revision",
            digest=attachment_candidate_revision_carried_sha256(value),
        )


class DeterministicRepairTargetV2(ContractModelV2):
    schema_version: Literal["eval-factory/deterministic-repair-target/v2"] = (
        "eval-factory/deterministic-repair-target/v2"
    )
    artifact_id: Identifier
    finding_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    finding_codes: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target(self) -> DeterministicRepairTargetV2:
        _require_sorted_unique_refs(
            "deterministic repair finding refs",
            self.finding_refs,
        )
        if len(self.finding_codes) != len(set(self.finding_codes)):
            raise ValueError("deterministic repair finding codes must be unique")
        if len(self.finding_refs) != len(self.finding_codes):
            raise ValueError("deterministic repair finding refs/codes must align")
        return self


class RevisionDeterministicValidationV2(ContractModelV2):
    schema_version: Literal["eval-factory/revision-deterministic-validation/v2"] = (
        "eval-factory/revision-deterministic-validation/v2"
    )
    revision_deterministic_validation_id: Identifier
    candidate_revision_ref: ObjectRef
    source_deterministic_validation_result_ref: ObjectRef
    artifact_validation_result_refs: tuple[ObjectRef, ...]
    finding_refs: tuple[ObjectRef, ...]
    output_refs: tuple[ObjectRef, ...]
    repair_targets: tuple[DeterministicRepairTargetV2, ...] = ()
    outcome: RevisionDeterministicValidationOutcomeV2
    policy_version: Literal["semantic-review/r5-09-v1"] = SEMANTIC_REVIEW_POLICY_VERSION
    validation_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_revision_result(self) -> RevisionDeterministicValidationV2:
        _require_ref(
            self.candidate_revision_ref,
            "attachment-candidate-revision",
            "candidate_revision_ref",
        )
        _require_ref(
            self.source_deterministic_validation_result_ref,
            "deterministic-item-validation-result",
            "source_deterministic_validation_result_ref",
        )
        _require_sorted_unique_refs(
            "artifact validation result refs",
            self.artifact_validation_result_refs,
        )
        _require_sorted_unique_refs("deterministic finding refs", self.finding_refs)
        _require_sorted_unique_refs("revision output refs", self.output_refs)
        target_ids = tuple(item.artifact_id for item in self.repair_targets)
        _require_sorted_unique("deterministic repair target IDs", target_ids)
        if (self.outcome is RevisionDeterministicValidationOutcomeV2.REQUIRES_REPAIR) != bool(
            self.repair_targets
        ):
            raise ValueError("REQUIRES_REPAIR outcome must exactly match repair targets")
        _validate_audit(
            self.audit,
            (
                self.candidate_revision_ref,
                self.source_deterministic_validation_result_ref,
            ),
            "revision deterministic validation",
        )
        _validate_final_identity(
            object_id=self.revision_deterministic_validation_id,
            object_sha256=self.validation_sha256,
            expected_prefix="revision-deterministic-validation",
            observed=revision_deterministic_validation_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        candidate_revision_ref: ObjectRef,
        source_deterministic_validation_result_ref: ObjectRef,
        artifact_validation_result_refs: tuple[ObjectRef, ...],
        finding_refs: tuple[ObjectRef, ...],
        output_refs: tuple[ObjectRef, ...],
        outcome: RevisionDeterministicValidationOutcomeV2,
        audit: ContractAudit,
        repair_targets: tuple[DeterministicRepairTargetV2, ...] = (),
    ) -> RevisionDeterministicValidationV2:
        value = cls(
            revision_deterministic_validation_id=("revision-deterministic-validation://pending"),
            candidate_revision_ref=candidate_revision_ref,
            source_deterministic_validation_result_ref=(source_deterministic_validation_result_ref),
            artifact_validation_result_refs=_sorted_refs(artifact_validation_result_refs),
            finding_refs=_sorted_refs(finding_refs),
            output_refs=_sorted_refs(output_refs),
            repair_targets=tuple(sorted(repair_targets, key=lambda item: item.artifact_id)),
            outcome=outcome,
            validation_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    candidate_revision_ref,
                    source_deterministic_validation_result_ref,
                ),
            ),
        )
        return _finalize(
            value,
            id_field="revision_deterministic_validation_id",
            hash_field="validation_sha256",
            prefix="revision-deterministic-validation",
            digest=revision_deterministic_validation_carried_sha256(value),
        )


class SemanticReviewPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/semantic-review-policy/v2"] = (
        "eval-factory/semantic-review-policy/v2"
    )
    semantic_review_policy_id: Identifier
    rounds_in_order: tuple[
        SemanticReviewRoundV2,
        SemanticReviewRoundV2,
        SemanticReviewRoundV2,
    ]
    roles_in_order: tuple[
        SemanticReviewerRoleV2,
        SemanticReviewerRoleV2,
        SemanticReviewerRoleV2,
    ]
    model_profile_refs: tuple[ObjectRef, ObjectRef, ObjectRef]
    prompt_versions: tuple[str, str, str]
    projection_policy_refs: tuple[ObjectRef, ObjectRef, ObjectRef]
    max_repair_waves_per_transition: Literal[1] = 1
    policy_version: Literal["semantic-review/r5-09-v1"] = SEMANTIC_REVIEW_POLICY_VERSION
    policy_sha256: Sha256

    @model_validator(mode="after")
    def validate_policy(self) -> SemanticReviewPolicyV2:
        if self.rounds_in_order != tuple(SemanticReviewRoundV2):
            raise ValueError("semantic review rounds must use canonical order")
        if self.roles_in_order != tuple(ROUND_ROLES[item] for item in self.rounds_in_order):
            raise ValueError("semantic review roles must match rounds")
        for ref in self.model_profile_refs:
            _require_ref(ref, "model-profile", "model_profile_refs")
        for ref in self.projection_policy_refs:
            _require_ref(ref, "projection-policy", "projection_policy_refs")
        _validate_final_identity(
            object_id=self.semantic_review_policy_id,
            object_sha256=self.policy_sha256,
            expected_prefix="semantic-review-policy",
            observed=semantic_review_policy_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        model_profile_refs: Mapping[SemanticReviewRoundV2, ObjectRef],
        prompt_versions: Mapping[SemanticReviewRoundV2, str],
        projection_policy_refs: Mapping[SemanticReviewRoundV2, ObjectRef],
    ) -> SemanticReviewPolicyV2:
        rounds = (
            SemanticReviewRoundV2.COVERAGE_SOLVABILITY,
            SemanticReviewRoundV2.REALISM_CONSISTENCY,
            SemanticReviewRoundV2.LEAKAGE_EXECUTABILITY,
        )
        value = cls(
            semantic_review_policy_id="semantic-review-policy://pending",
            rounds_in_order=rounds,
            roles_in_order=(
                ROUND_ROLES[rounds[0]],
                ROUND_ROLES[rounds[1]],
                ROUND_ROLES[rounds[2]],
            ),
            model_profile_refs=(
                model_profile_refs[rounds[0]],
                model_profile_refs[rounds[1]],
                model_profile_refs[rounds[2]],
            ),
            prompt_versions=(
                prompt_versions[rounds[0]],
                prompt_versions[rounds[1]],
                prompt_versions[rounds[2]],
            ),
            projection_policy_refs=(
                projection_policy_refs[rounds[0]],
                projection_policy_refs[rounds[1]],
                projection_policy_refs[rounds[2]],
            ),
            policy_sha256="0" * 64,
        )
        return _finalize(
            value,
            id_field="semantic_review_policy_id",
            hash_field="policy_sha256",
            prefix="semantic-review-policy",
            digest=semantic_review_policy_carried_sha256(value),
        )

    def model_profile_for(self, round_: SemanticReviewRoundV2) -> ObjectRef:
        return self.model_profile_refs[self.rounds_in_order.index(round_)]

    def prompt_for(self, round_: SemanticReviewRoundV2) -> str:
        return self.prompt_versions[self.rounds_in_order.index(round_)]

    def projection_for(self, round_: SemanticReviewRoundV2) -> ObjectRef:
        return self.projection_policy_refs[self.rounds_in_order.index(round_)]


class CoverageSolvabilityReviewViewV2(ContractModelV2):
    schema_version: Literal["eval-factory/coverage-solvability-review-view/v2"] = (
        "eval-factory/coverage-solvability-review-view/v2"
    )
    review_view_id: Identifier
    round: Literal[SemanticReviewRoundV2.COVERAGE_SOLVABILITY] = SemanticReviewRoundV2.COVERAGE_SOLVABILITY
    reviewer_role: Literal[SemanticReviewerRoleV2.COVERAGE_SOLVABILITY_REVIEWER] = (
        SemanticReviewerRoleV2.COVERAGE_SOLVABILITY_REVIEWER
    )
    candidate_revision_ref: ObjectRef
    deterministic_validation_result_ref: ObjectRef
    projection_policy_ref: ObjectRef
    query_public_view_ref: ObjectRef
    candidate_manifest_ref: ObjectRef
    public_rubric_requirement_refs: tuple[ObjectRef, ...]
    safe_evidence_bundle_refs: tuple[ObjectRef, ...]
    deterministic_finding_refs: tuple[ObjectRef, ...]
    allowed_ref_inventory: tuple[ObjectRef, ...]
    denied_data_families: tuple[Identifier, ...]
    policy_version: Literal["semantic-review/r5-09-v1"] = SEMANTIC_REVIEW_POLICY_VERSION
    view_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_view(self) -> CoverageSolvabilityReviewViewV2:
        expected = _sorted_refs(
            (
                self.candidate_revision_ref,
                self.deterministic_validation_result_ref,
                self.projection_policy_ref,
                self.query_public_view_ref,
                self.candidate_manifest_ref,
                *self.public_rubric_requirement_refs,
                *self.safe_evidence_bundle_refs,
                *self.deterministic_finding_refs,
            )
        )
        if self.allowed_ref_inventory != expected:
            raise ValueError("coverage view allowed inventory is not exact")
        if self.denied_data_families != _denied_families():
            raise ValueError("coverage view denied data families are not exact")
        _validate_audit(self.audit, expected, "coverage review view")
        _validate_final_identity(
            object_id=self.review_view_id,
            object_sha256=self.view_sha256,
            expected_prefix="coverage-solvability-review-view",
            observed=_view_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        candidate_revision_ref: ObjectRef,
        deterministic_validation_result_ref: ObjectRef,
        projection_policy_ref: ObjectRef,
        query_public_view_ref: ObjectRef,
        candidate_manifest_ref: ObjectRef,
        public_rubric_requirement_refs: tuple[ObjectRef, ...],
        safe_evidence_bundle_refs: tuple[ObjectRef, ...],
        deterministic_finding_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> CoverageSolvabilityReviewViewV2:
        inventory = _sorted_refs(
            (
                candidate_revision_ref,
                deterministic_validation_result_ref,
                projection_policy_ref,
                query_public_view_ref,
                candidate_manifest_ref,
                *public_rubric_requirement_refs,
                *safe_evidence_bundle_refs,
                *deterministic_finding_refs,
            )
        )
        value = cls(
            review_view_id="coverage-solvability-review-view://pending",
            candidate_revision_ref=candidate_revision_ref,
            deterministic_validation_result_ref=(deterministic_validation_result_ref),
            projection_policy_ref=projection_policy_ref,
            query_public_view_ref=query_public_view_ref,
            candidate_manifest_ref=candidate_manifest_ref,
            public_rubric_requirement_refs=_sorted_refs(public_rubric_requirement_refs),
            safe_evidence_bundle_refs=_sorted_refs(safe_evidence_bundle_refs),
            deterministic_finding_refs=_sorted_refs(deterministic_finding_refs),
            allowed_ref_inventory=inventory,
            denied_data_families=_denied_families(),
            view_sha256="0" * 64,
            audit=_safe_audit(audit, inventory),
        )
        return _finalize(
            value,
            id_field="review_view_id",
            hash_field="view_sha256",
            prefix="coverage-solvability-review-view",
            digest=_view_carried_sha256(value),
        )


class RealismConsistencyReviewViewV2(ContractModelV2):
    schema_version: Literal["eval-factory/realism-consistency-review-view/v2"] = (
        "eval-factory/realism-consistency-review-view/v2"
    )
    review_view_id: Identifier
    round: Literal[SemanticReviewRoundV2.REALISM_CONSISTENCY] = SemanticReviewRoundV2.REALISM_CONSISTENCY
    reviewer_role: Literal[SemanticReviewerRoleV2.REALISM_CONSISTENCY_REVIEWER] = (
        SemanticReviewerRoleV2.REALISM_CONSISTENCY_REVIEWER
    )
    candidate_revision_ref: ObjectRef
    deterministic_validation_result_ref: ObjectRef
    projection_policy_ref: ObjectRef
    current_artifact_version_refs: tuple[ObjectRef, ...]
    current_output_refs: tuple[ObjectRef, ...]
    world_ledger_snapshot_ref: ObjectRef
    approved_source_evidence_refs: tuple[ObjectRef, ...]
    prior_round_result_ref: ObjectRef
    prior_finding_refs: tuple[ObjectRef, ...]
    prior_resolution_refs: tuple[ObjectRef, ...]
    prior_repair_plan_refs: tuple[ObjectRef, ...]
    prior_repair_result_refs: tuple[ObjectRef, ...]
    allowed_ref_inventory: tuple[ObjectRef, ...]
    denied_data_families: tuple[Identifier, ...]
    policy_version: Literal["semantic-review/r5-09-v1"] = SEMANTIC_REVIEW_POLICY_VERSION
    view_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_view(self) -> RealismConsistencyReviewViewV2:
        expected = _view_expected_refs(self)
        if self.allowed_ref_inventory != expected:
            raise ValueError("realism view allowed inventory is not exact")
        if self.denied_data_families != _denied_families():
            raise ValueError("realism view denied data families are not exact")
        _validate_audit(self.audit, expected, "realism review view")
        _validate_final_identity(
            object_id=self.review_view_id,
            object_sha256=self.view_sha256,
            expected_prefix="realism-consistency-review-view",
            observed=_view_carried_sha256(self),
        )
        return self


class LeakageExecutabilityReviewViewV2(ContractModelV2):
    schema_version: Literal["eval-factory/leakage-executability-review-view/v2"] = (
        "eval-factory/leakage-executability-review-view/v2"
    )
    review_view_id: Identifier
    round: Literal[SemanticReviewRoundV2.LEAKAGE_EXECUTABILITY] = SemanticReviewRoundV2.LEAKAGE_EXECUTABILITY
    reviewer_role: Literal[SemanticReviewerRoleV2.LEAKAGE_EXECUTABILITY_REVIEWER] = (
        SemanticReviewerRoleV2.LEAKAGE_EXECUTABILITY_REVIEWER
    )
    candidate_revision_ref: ObjectRef
    deterministic_validation_result_ref: ObjectRef
    projection_policy_ref: ObjectRef
    current_artifact_version_refs: tuple[ObjectRef, ...]
    current_output_refs: tuple[ObjectRef, ...]
    taint_lineage_projection_refs: tuple[ObjectRef, ...]
    evaluator_contract_ref: ObjectRef
    contestant_tool_policy_ref: ObjectRef
    leakage_reference_set_ref: ObjectRef
    executability_result_refs: tuple[ObjectRef, ...]
    prior_round_result_ref: ObjectRef
    prior_finding_refs: tuple[ObjectRef, ...]
    prior_resolution_refs: tuple[ObjectRef, ...]
    prior_repair_plan_refs: tuple[ObjectRef, ...]
    prior_repair_result_refs: tuple[ObjectRef, ...]
    allowed_ref_inventory: tuple[ObjectRef, ...]
    denied_data_families: tuple[Identifier, ...]
    policy_version: Literal["semantic-review/r5-09-v1"] = SEMANTIC_REVIEW_POLICY_VERSION
    view_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_view(self) -> LeakageExecutabilityReviewViewV2:
        expected = _view_expected_refs(self)
        if self.allowed_ref_inventory != expected:
            raise ValueError("leakage view allowed inventory is not exact")
        if self.denied_data_families != _denied_families():
            raise ValueError("leakage view denied data families are not exact")
        _validate_audit(self.audit, expected, "leakage review view")
        _validate_final_identity(
            object_id=self.review_view_id,
            object_sha256=self.view_sha256,
            expected_prefix="leakage-executability-review-view",
            observed=_view_carried_sha256(self),
        )
        return self


class SemanticReviewFindingV2(ContractModelV2):
    schema_version: Literal["eval-factory/semantic-review-finding/v2"] = (
        "eval-factory/semantic-review-finding/v2"
    )
    semantic_review_finding_id: Identifier
    facade_finding_ref: ObjectRef
    facade_review_result_ref: ObjectRef
    round: SemanticReviewRoundV2
    scope: SemanticFindingScopeV2
    code: AttachmentSemanticReviewFindingCodeV2
    candidate_revision_ref: ObjectRef
    subject_refs: tuple[ObjectRef, ...]
    artifact_ids: tuple[Identifier, ...]
    evidence_ref_ids: tuple[Identifier, ...]
    predecessor_finding_ref: ObjectRef | None = None
    severity: Severity
    status: Literal["OPEN"] = "OPEN"
    non_waivable: bool
    artifact_repair_allowed: bool
    policy_version: Literal["semantic-review/r5-09-v1"] = SEMANTIC_REVIEW_POLICY_VERSION
    finding_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_finding(self) -> SemanticReviewFindingV2:
        policy = attachment_semantic_finding_policy(self.code)
        expected_scope = (
            SemanticFindingScopeV2.ARTIFACT
            if policy.scope is AttachmentSemanticFindingScopeV2.ARTIFACT
            else SemanticFindingScopeV2.ITEM
        )
        if (
            self.round.value != policy.round.value
            or self.scope is not expected_scope
            or self.severity.value != policy.severity.value
            or self.non_waivable is not policy.non_waivable
            or self.artifact_repair_allowed is not policy.artifact_repair_allowed
        ):
            raise ValueError("factory semantic finding policy does not match code")
        _require_sorted_unique_refs(
            "semantic finding subject refs",
            self.subject_refs,
        )
        _require_sorted_unique(
            "semantic finding artifact IDs",
            self.artifact_ids,
        )
        _require_sorted_unique(
            "semantic finding evidence IDs",
            self.evidence_ref_ids,
        )
        _validate_final_identity(
            object_id=self.semantic_review_finding_id,
            object_sha256=self.finding_sha256,
            expected_prefix="semantic-review-finding",
            observed=semantic_review_finding_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        facade_finding: AttachmentSemanticReviewFindingV2,
        facade_review_result_ref: ObjectRef,
        audit: ContractAudit,
    ) -> SemanticReviewFindingV2:
        policy = attachment_semantic_finding_policy(facade_finding.code)
        facade_ref = _object_ref_from_facade(attachment_semantic_review_finding_ref(facade_finding))
        value = cls(
            semantic_review_finding_id="semantic-review-finding://pending",
            facade_finding_ref=facade_ref,
            facade_review_result_ref=facade_review_result_ref,
            round=SemanticReviewRoundV2(facade_finding.round.value),
            scope=(
                SemanticFindingScopeV2.ARTIFACT
                if facade_finding.scope is AttachmentSemanticFindingScopeV2.ARTIFACT
                else SemanticFindingScopeV2.ITEM
            ),
            code=facade_finding.code,
            candidate_revision_ref=_object_ref_from_facade(facade_finding.candidate_revision_ref),
            subject_refs=_sorted_refs(
                tuple(_object_ref_from_facade(ref) for ref in facade_finding.subject_refs)
            ),
            artifact_ids=facade_finding.artifact_ids,
            evidence_ref_ids=facade_finding.evidence_ref_ids,
            predecessor_finding_ref=(
                _object_ref_from_facade(facade_finding.predecessor_finding_ref)
                if facade_finding.predecessor_finding_ref is not None
                else None
            ),
            severity=Severity(policy.severity.value),
            status="OPEN",
            non_waivable=policy.non_waivable,
            artifact_repair_allowed=policy.artifact_repair_allowed,
            finding_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (facade_ref, facade_review_result_ref),
            ),
        )
        return _finalize(
            value,
            id_field="semantic_review_finding_id",
            hash_field="finding_sha256",
            prefix="semantic-review-finding",
            digest=semantic_review_finding_carried_sha256(value),
        )


class SemanticFindingResolutionV2(ContractModelV2):
    schema_version: Literal["eval-factory/semantic-finding-resolution/v2"] = (
        "eval-factory/semantic-finding-resolution/v2"
    )
    semantic_finding_resolution_id: Identifier
    facade_resolution_ref: ObjectRef
    resolving_stage_run_ref: ObjectRef
    prior_finding_ref: ObjectRef
    old_subject_refs: tuple[ObjectRef, ...]
    current_subject_refs: tuple[ObjectRef, ...]
    repair_plan_ref: ObjectRef
    repair_result_refs: tuple[ObjectRef, ...]
    deterministic_revalidation_ref: ObjectRef
    disposition: Literal["CONFIRMED_RESOLVED", "PERSISTS"]
    successor_finding_ref: ObjectRef | None = None
    evidence_ref_ids: tuple[Identifier, ...]
    policy_version: Literal["semantic-review/r5-09-v1"] = SEMANTIC_REVIEW_POLICY_VERSION
    resolution_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_resolution(self) -> SemanticFindingResolutionV2:
        _require_sorted_unique_refs(
            "factory resolution old subjects",
            self.old_subject_refs,
        )
        _require_sorted_unique_refs(
            "factory resolution current subjects",
            self.current_subject_refs,
        )
        _require_sorted_unique_refs(
            "factory resolution repair results",
            self.repair_result_refs,
        )
        if (
            not self.old_subject_refs
            or not self.current_subject_refs
            or self.old_subject_refs == self.current_subject_refs
        ):
            raise ValueError("factory semantic resolution requires changed subjects")
        if self.disposition == "CONFIRMED_RESOLVED":
            if self.successor_finding_ref is not None:
                raise ValueError("confirmed resolution cannot carry successor")
        elif self.successor_finding_ref is None:
            raise ValueError("persistent resolution requires successor")
        _validate_final_identity(
            object_id=self.semantic_finding_resolution_id,
            object_sha256=self.resolution_sha256,
            expected_prefix="semantic-finding-resolution",
            observed=semantic_finding_resolution_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        facade_resolution: AttachmentSemanticFindingResolutionV2,
        resolving_stage_run_ref: ObjectRef,
        prior_finding_ref: ObjectRef,
        successor_finding_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> SemanticFindingResolutionV2:
        facade_ref = _object_ref_from_facade(attachment_semantic_finding_resolution_ref(facade_resolution))
        value = cls(
            semantic_finding_resolution_id=("semantic-finding-resolution://pending"),
            facade_resolution_ref=facade_ref,
            resolving_stage_run_ref=resolving_stage_run_ref,
            prior_finding_ref=prior_finding_ref,
            old_subject_refs=_sorted_refs(
                tuple(_object_ref_from_facade(ref) for ref in facade_resolution.old_subject_refs)
            ),
            current_subject_refs=_sorted_refs(
                tuple(_object_ref_from_facade(ref) for ref in facade_resolution.current_subject_refs)
            ),
            repair_plan_ref=_object_ref_from_facade(facade_resolution.repair_plan_ref),
            repair_result_refs=_sorted_refs(
                tuple(_object_ref_from_facade(ref) for ref in facade_resolution.repair_result_refs)
            ),
            deterministic_revalidation_ref=_object_ref_from_facade(
                facade_resolution.deterministic_revalidation_ref
            ),
            disposition=facade_resolution.disposition.value,
            successor_finding_ref=successor_finding_ref,
            evidence_ref_ids=facade_resolution.evidence_ref_ids,
            resolution_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    facade_ref,
                    resolving_stage_run_ref,
                    prior_finding_ref,
                    _object_ref_from_facade(facade_resolution.repair_plan_ref),
                    _object_ref_from_facade(facade_resolution.deterministic_revalidation_ref),
                    *tuple(_object_ref_from_facade(ref) for ref in facade_resolution.repair_result_refs),
                ),
            ),
        )
        return _finalize(
            value,
            id_field="semantic_finding_resolution_id",
            hash_field="resolution_sha256",
            prefix="semantic-finding-resolution",
            digest=semantic_finding_resolution_carried_sha256(value),
        )


class SemanticReviewRoundResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/semantic-review-round-result/v2"] = (
        "eval-factory/semantic-review-round-result/v2"
    )
    semantic_review_round_result_id: Identifier
    round: SemanticReviewRoundV2
    reviewer_role: SemanticReviewerRoleV2
    stage_run_ref: ObjectRef
    context_view_ref: ObjectRef
    clean_context_attestation_ref: ObjectRef
    facade_review_request_ref: ObjectRef
    facade_review_result_ref: ObjectRef
    candidate_revision_ref: ObjectRef
    deterministic_validation_result_ref: ObjectRef
    predecessor_round_result_ref: ObjectRef | None = None
    prior_finding_refs: tuple[ObjectRef, ...] = ()
    finding_refs: tuple[ObjectRef, ...] = ()
    resolution_refs: tuple[ObjectRef, ...] = ()
    prior_repair_plan_refs: tuple[ObjectRef, ...] = ()
    prior_repair_result_refs: tuple[ObjectRef, ...] = ()
    outcome: SemanticReviewRoundOutcomeV2
    accepted: bool
    policy_version: Literal["semantic-review/r5-09-v1"] = SEMANTIC_REVIEW_POLICY_VERSION
    round_result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_round_result(self) -> SemanticReviewRoundResultV2:
        if self.reviewer_role is not ROUND_ROLES[self.round]:
            raise ValueError("semantic round result role does not match round")
        if self.accepted is not (self.outcome is SemanticReviewRoundOutcomeV2.ACCEPTED):
            raise ValueError("semantic round accepted flag does not match outcome")
        if (self.round is SemanticReviewRoundV2.COVERAGE_SOLVABILITY) != (
            self.predecessor_round_result_ref is None
        ):
            raise ValueError("semantic round predecessor does not match ordering")
        for label, refs in (
            ("semantic round prior finding refs", self.prior_finding_refs),
            ("semantic round finding refs", self.finding_refs),
            ("semantic round resolution refs", self.resolution_refs),
            ("semantic round prior repair plan refs", self.prior_repair_plan_refs),
            ("semantic round prior repair result refs", self.prior_repair_result_refs),
        ):
            _require_sorted_unique_refs(label, refs)
        if bool(self.prior_repair_plan_refs) != bool(self.prior_repair_result_refs):
            raise ValueError("semantic round prior repair refs must be present together")
        expected_audit_refs = _sorted_refs(
            (
                self.stage_run_ref,
                self.context_view_ref,
                self.clean_context_attestation_ref,
                self.facade_review_request_ref,
                self.facade_review_result_ref,
                self.candidate_revision_ref,
                self.deterministic_validation_result_ref,
                *self.prior_finding_refs,
                *self.finding_refs,
                *self.resolution_refs,
                *self.prior_repair_plan_refs,
                *self.prior_repair_result_refs,
                *(
                    (self.predecessor_round_result_ref,)
                    if self.predecessor_round_result_ref is not None
                    else ()
                ),
            )
        )
        _validate_audit(
            self.audit,
            expected_audit_refs,
            "semantic round result",
        )
        _validate_final_identity(
            object_id=self.semantic_review_round_result_id,
            object_sha256=self.round_result_sha256,
            expected_prefix="semantic-review-round-result",
            observed=semantic_review_round_result_carried_sha256(self),
        )
        return self


class SemanticReviewWorkflowResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/semantic-review-workflow-result/v2"] = (
        "eval-factory/semantic-review-workflow-result/v2"
    )
    semantic_review_workflow_result_id: Identifier
    initial_candidate_revision_ref: ObjectRef
    initial_deterministic_validation_result_ref: ObjectRef
    review_policy_ref: ObjectRef
    current_candidate_revision: AttachmentCandidateRevisionV2
    current_candidate_revision_ref: ObjectRef
    deterministic_validation_result_refs: tuple[ObjectRef, ...]
    repair_plan_refs: tuple[ObjectRef, ...] = ()
    repair_result_refs: tuple[ObjectRef, ...] = ()
    round_results: tuple[SemanticReviewRoundResultV2, ...]
    round_result_refs: tuple[ObjectRef, ...]
    stage_result_refs: tuple[ObjectRef, ...]
    semantic_findings: tuple[SemanticReviewFindingV2, ...] = ()
    semantic_resolutions: tuple[SemanticFindingResolutionV2, ...] = ()
    current_finding_refs: tuple[ObjectRef, ...] = ()
    stale_finding_refs: tuple[ObjectRef, ...] = ()
    resolution_refs: tuple[ObjectRef, ...] = ()
    outcome: SemanticReviewWorkflowOutcomeV2
    accepted_artifact_refs: tuple[ObjectRef, ...] = ()
    environment_spec_ref: None = None
    provenance_manifest_ref: None = None
    quality_report_ref: None = None
    package_sha256: None = None
    input_state_only: None = None
    policy_version: Literal["semantic-review/r5-09-v1"] = SEMANTIC_REVIEW_POLICY_VERSION
    workflow_result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_workflow(self) -> SemanticReviewWorkflowResultV2:
        _require_ref(
            self.review_policy_ref,
            "semantic-review-policy",
            "review_policy_ref",
        )
        if self.current_candidate_revision_ref != attachment_candidate_revision_ref(
            self.current_candidate_revision
        ):
            raise ValueError("workflow current candidate revision ref is stale")
        expected_round_refs = tuple(semantic_review_round_result_ref(item) for item in self.round_results)
        if self.round_result_refs != expected_round_refs:
            raise ValueError("workflow round refs must match nested results")
        observed_rounds = tuple(item.round for item in self.round_results)
        if observed_rounds != tuple(SemanticReviewRoundV2)[: len(observed_rounds)]:
            raise ValueError("workflow semantic rounds must be a canonical prefix")
        if len(self.stage_result_refs) != len(self.round_results):
            raise ValueError("workflow stage result refs must cover completed rounds")
        nested_finding_refs = tuple(semantic_review_finding_ref(item) for item in self.semantic_findings)
        _require_sorted_unique_refs(
            "workflow nested semantic finding refs",
            nested_finding_refs,
        )
        nested_resolution_refs = tuple(
            semantic_finding_resolution_ref(item) for item in self.semantic_resolutions
        )
        if self.resolution_refs != nested_resolution_refs:
            raise ValueError("workflow resolution refs must match nested resolutions")
        inventoried_semantic_refs = {
            ref
            for ref in (
                *self.current_finding_refs,
                *self.stale_finding_refs,
            )
            if ref.object_type == "semantic-review-finding"
        }
        if inventoried_semantic_refs != set(nested_finding_refs):
            raise ValueError("workflow semantic findings must match current/stale inventory")
        if len(self.deterministic_validation_result_refs) != len(
            set(self.deterministic_validation_result_refs)
        ):
            raise ValueError("workflow deterministic validation refs must be unique")
        if len(self.stage_result_refs) != len(set(self.stage_result_refs)):
            raise ValueError("workflow stage result refs must be unique")
        for label, refs in (
            ("workflow repair plan refs", self.repair_plan_refs),
            ("workflow repair result refs", self.repair_result_refs),
            ("workflow current finding refs", self.current_finding_refs),
            ("workflow stale finding refs", self.stale_finding_refs),
            ("workflow resolution refs", self.resolution_refs),
        ):
            _require_sorted_unique_refs(label, refs)
        if set(self.current_finding_refs) & set(self.stale_finding_refs):
            raise ValueError("workflow current and stale findings must be disjoint")
        if self.outcome is SemanticReviewWorkflowOutcomeV2.PASSED and (
            observed_rounds != tuple(SemanticReviewRoundV2)
            or not self.round_results[-1].accepted
            or self.current_finding_refs
        ):
            raise ValueError("PASSED semantic workflow is incomplete or blocked")
        if self.accepted_artifact_refs:
            raise ValueError("accepted artifact refs remain empty before R5-10")
        _validate_audit(
            self.audit,
            _sorted_refs(
                (
                    self.initial_candidate_revision_ref,
                    self.initial_deterministic_validation_result_ref,
                    self.review_policy_ref,
                    self.current_candidate_revision_ref,
                    *self.deterministic_validation_result_refs,
                    *self.repair_plan_refs,
                    *self.repair_result_refs,
                    *self.round_result_refs,
                    *self.stage_result_refs,
                    *self.current_finding_refs,
                    *self.stale_finding_refs,
                    *self.resolution_refs,
                )
            ),
            "semantic workflow result",
        )
        _validate_final_identity(
            object_id=self.semantic_review_workflow_result_id,
            object_sha256=self.workflow_result_sha256,
            expected_prefix="semantic-review-workflow-result",
            observed=semantic_review_workflow_result_carried_sha256(self),
        )
        return self


def candidate_artifact_version_carried_sha256(
    value: CandidateArtifactVersionV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "candidate_artifact_version_id",
                "artifact_version_sha256",
            },
            exclude_none=False,
        )
    )


def semantic_review_finding_carried_sha256(
    value: SemanticReviewFindingV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "semantic_review_finding_id",
                "finding_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def semantic_review_finding_ref(
    value: SemanticReviewFindingV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="semantic-review-finding",
        object_id=value.semantic_review_finding_id,
        object_version="v2",
        object_sha256=value.finding_sha256,
    )


def semantic_finding_resolution_carried_sha256(
    value: SemanticFindingResolutionV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "semantic_finding_resolution_id",
                "resolution_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def semantic_finding_resolution_ref(
    value: SemanticFindingResolutionV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="semantic-finding-resolution",
        object_id=value.semantic_finding_resolution_id,
        object_version="v2",
        object_sha256=value.resolution_sha256,
    )


def semantic_review_finding_is_current(
    finding: SemanticReviewFindingV2,
    revision: AttachmentCandidateRevisionV2,
) -> bool:
    revision_ref = attachment_candidate_revision_ref(revision)
    expected: tuple[ObjectRef, ...]
    if finding.scope is SemanticFindingScopeV2.ITEM:
        expected = (revision_ref,)
    else:
        artifact_ids = set(finding.artifact_ids)
        expected = _sorted_refs(
            (
                revision_ref,
                *(
                    ref
                    for artifact, ref in zip(
                        revision.artifact_versions,
                        revision.artifact_version_refs,
                        strict=True,
                    )
                    if artifact.artifact_id in artifact_ids
                ),
                *(
                    artifact.output_ref
                    for artifact in revision.artifact_versions
                    if artifact.artifact_id in artifact_ids
                ),
            )
        )
    return finding.subject_refs == expected


def semantic_finding_resolution_is_current(
    resolution: SemanticFindingResolutionV2,
    revision: AttachmentCandidateRevisionV2,
) -> bool:
    current_universe = {
        attachment_candidate_revision_ref(revision),
        *revision.artifact_version_refs,
        *revision.candidate_output_refs,
    }
    return (
        bool(resolution.current_subject_refs)
        and set(resolution.current_subject_refs) <= current_universe
        and attachment_candidate_revision_ref(revision) in resolution.current_subject_refs
    )


def targeted_repair_plan_carried_sha256(
    value: TargetedRepairPlanV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "targeted_repair_plan_id",
                "plan_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def targeted_repair_plan_ref(value: TargetedRepairPlanV2) -> ObjectRef:
    return ObjectRef(
        object_type="targeted-repair-plan",
        object_id=value.targeted_repair_plan_id,
        object_version="v2",
        object_sha256=value.plan_sha256,
    )


def repaired_artifact_build_result_carried_sha256(
    value: RepairedArtifactBuildResultV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "repaired_artifact_build_result_id",
                "repaired_result_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def repaired_artifact_build_result_ref(
    value: RepairedArtifactBuildResultV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="repaired-artifact-build-result",
        object_id=value.repaired_artifact_build_result_id,
        object_version="v2",
        object_sha256=value.repaired_result_sha256,
    )


def candidate_artifact_version_ref(
    value: CandidateArtifactVersionV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="candidate-artifact-version",
        object_id=value.candidate_artifact_version_id,
        object_version="v2",
        object_sha256=value.artifact_version_sha256,
    )


def attachment_candidate_revision_carried_sha256(
    value: AttachmentCandidateRevisionV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "attachment_candidate_revision_id",
                "candidate_revision_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def attachment_candidate_revision_ref(
    value: AttachmentCandidateRevisionV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="attachment-candidate-revision",
        object_id=value.attachment_candidate_revision_id,
        object_version="v2",
        object_sha256=value.candidate_revision_sha256,
    )


def revision_deterministic_validation_carried_sha256(
    value: RevisionDeterministicValidationV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "revision_deterministic_validation_id",
                "validation_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def revision_deterministic_validation_ref(
    value: RevisionDeterministicValidationV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="revision-deterministic-validation",
        object_id=value.revision_deterministic_validation_id,
        object_version="v2",
        object_sha256=value.validation_sha256,
    )


def semantic_review_policy_carried_sha256(value: SemanticReviewPolicyV2) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "semantic_review_policy_id",
                "policy_sha256",
            },
            exclude_none=False,
        )
    )


def semantic_review_policy_ref(value: SemanticReviewPolicyV2) -> ObjectRef:
    return ObjectRef(
        object_type="semantic-review-policy",
        object_id=value.semantic_review_policy_id,
        object_version="v2",
        object_sha256=value.policy_sha256,
    )


def semantic_review_context_view_ref(
    value: (
        CoverageSolvabilityReviewViewV2 | RealismConsistencyReviewViewV2 | LeakageExecutabilityReviewViewV2
    ),
) -> ObjectRef:
    return ObjectRef(
        object_type="semantic-review-context-view",
        object_id=value.review_view_id,
        object_version="v2",
        object_sha256=value.view_sha256,
    )


def create_realism_consistency_review_view(
    *,
    candidate_revision_ref: ObjectRef,
    deterministic_validation_result_ref: ObjectRef,
    projection_policy_ref: ObjectRef,
    current_artifact_version_refs: tuple[ObjectRef, ...],
    current_output_refs: tuple[ObjectRef, ...],
    world_ledger_snapshot_ref: ObjectRef,
    approved_source_evidence_refs: tuple[ObjectRef, ...],
    prior_round_result_ref: ObjectRef,
    prior_finding_refs: tuple[ObjectRef, ...],
    prior_resolution_refs: tuple[ObjectRef, ...],
    prior_repair_plan_refs: tuple[ObjectRef, ...],
    prior_repair_result_refs: tuple[ObjectRef, ...],
    audit: ContractAudit,
) -> RealismConsistencyReviewViewV2:
    inventory = _sorted_refs(
        (
            candidate_revision_ref,
            deterministic_validation_result_ref,
            projection_policy_ref,
            *current_artifact_version_refs,
            *current_output_refs,
            world_ledger_snapshot_ref,
            *approved_source_evidence_refs,
            prior_round_result_ref,
            *prior_finding_refs,
            *prior_resolution_refs,
            *prior_repair_plan_refs,
            *prior_repair_result_refs,
        )
    )
    pending = RealismConsistencyReviewViewV2(
        review_view_id="realism-consistency-review-view://pending",
        candidate_revision_ref=candidate_revision_ref,
        deterministic_validation_result_ref=deterministic_validation_result_ref,
        projection_policy_ref=projection_policy_ref,
        current_artifact_version_refs=_sorted_refs(current_artifact_version_refs),
        current_output_refs=_sorted_refs(current_output_refs),
        world_ledger_snapshot_ref=world_ledger_snapshot_ref,
        approved_source_evidence_refs=_sorted_refs(approved_source_evidence_refs),
        prior_round_result_ref=prior_round_result_ref,
        prior_finding_refs=_sorted_refs(prior_finding_refs),
        prior_resolution_refs=_sorted_refs(prior_resolution_refs),
        prior_repair_plan_refs=_sorted_refs(prior_repair_plan_refs),
        prior_repair_result_refs=_sorted_refs(prior_repair_result_refs),
        allowed_ref_inventory=inventory,
        denied_data_families=_denied_families(),
        view_sha256="0" * 64,
        audit=_safe_audit(audit, inventory),
    )
    return _finalize(
        pending,
        id_field="review_view_id",
        hash_field="view_sha256",
        prefix="realism-consistency-review-view",
        digest=_view_carried_sha256(pending),
    )


def create_leakage_executability_review_view(
    *,
    candidate_revision_ref: ObjectRef,
    deterministic_validation_result_ref: ObjectRef,
    projection_policy_ref: ObjectRef,
    current_artifact_version_refs: tuple[ObjectRef, ...],
    current_output_refs: tuple[ObjectRef, ...],
    taint_lineage_projection_refs: tuple[ObjectRef, ...],
    evaluator_contract_ref: ObjectRef,
    contestant_tool_policy_ref: ObjectRef,
    leakage_reference_set_ref: ObjectRef,
    executability_result_refs: tuple[ObjectRef, ...],
    prior_round_result_ref: ObjectRef,
    prior_finding_refs: tuple[ObjectRef, ...],
    prior_resolution_refs: tuple[ObjectRef, ...],
    prior_repair_plan_refs: tuple[ObjectRef, ...],
    prior_repair_result_refs: tuple[ObjectRef, ...],
    audit: ContractAudit,
) -> LeakageExecutabilityReviewViewV2:
    inventory = _sorted_refs(
        (
            candidate_revision_ref,
            deterministic_validation_result_ref,
            projection_policy_ref,
            *current_artifact_version_refs,
            *current_output_refs,
            *taint_lineage_projection_refs,
            evaluator_contract_ref,
            contestant_tool_policy_ref,
            leakage_reference_set_ref,
            *executability_result_refs,
            prior_round_result_ref,
            *prior_finding_refs,
            *prior_resolution_refs,
            *prior_repair_plan_refs,
            *prior_repair_result_refs,
        )
    )
    pending = LeakageExecutabilityReviewViewV2(
        review_view_id="leakage-executability-review-view://pending",
        candidate_revision_ref=candidate_revision_ref,
        deterministic_validation_result_ref=deterministic_validation_result_ref,
        projection_policy_ref=projection_policy_ref,
        current_artifact_version_refs=_sorted_refs(current_artifact_version_refs),
        current_output_refs=_sorted_refs(current_output_refs),
        taint_lineage_projection_refs=_sorted_refs(taint_lineage_projection_refs),
        evaluator_contract_ref=evaluator_contract_ref,
        contestant_tool_policy_ref=contestant_tool_policy_ref,
        leakage_reference_set_ref=leakage_reference_set_ref,
        executability_result_refs=_sorted_refs(executability_result_refs),
        prior_round_result_ref=prior_round_result_ref,
        prior_finding_refs=_sorted_refs(prior_finding_refs),
        prior_resolution_refs=_sorted_refs(prior_resolution_refs),
        prior_repair_plan_refs=_sorted_refs(prior_repair_plan_refs),
        prior_repair_result_refs=_sorted_refs(prior_repair_result_refs),
        allowed_ref_inventory=inventory,
        denied_data_families=_denied_families(),
        view_sha256="0" * 64,
        audit=_safe_audit(audit, inventory),
    )
    return _finalize(
        pending,
        id_field="review_view_id",
        hash_field="view_sha256",
        prefix="leakage-executability-review-view",
        digest=_view_carried_sha256(pending),
    )


def semantic_review_round_result_carried_sha256(
    value: SemanticReviewRoundResultV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "semantic_review_round_result_id",
                "round_result_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def semantic_review_round_result_ref(
    value: SemanticReviewRoundResultV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="semantic-review-round-result",
        object_id=value.semantic_review_round_result_id,
        object_version="v2",
        object_sha256=value.round_result_sha256,
    )


def semantic_review_workflow_result_carried_sha256(
    value: SemanticReviewWorkflowResultV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "semantic_review_workflow_result_id",
                "workflow_result_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def semantic_review_workflow_result_ref(
    value: SemanticReviewWorkflowResultV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="semantic-review-workflow-result",
        object_id=value.semantic_review_workflow_result_id,
        object_version="v2",
        object_sha256=value.workflow_result_sha256,
    )


def _view_carried_sha256(value: ContractModelV2) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={"review_view_id", "view_sha256", "audit"},
            exclude_none=False,
        )
    )


def _view_expected_refs(
    value: RealismConsistencyReviewViewV2 | LeakageExecutabilityReviewViewV2,
) -> tuple[ObjectRef, ...]:
    excluded = {
        "review_view_id",
        "round",
        "reviewer_role",
        "allowed_ref_inventory",
        "denied_data_families",
        "policy_version",
        "view_sha256",
        "audit",
    }
    refs: list[ObjectRef] = []
    for name in value.__class__.model_fields:
        if name in excluded:
            continue
        field_value = getattr(value, name)
        if isinstance(field_value, ObjectRef):
            refs.append(field_value)
        elif isinstance(field_value, tuple):
            refs.extend(item for item in field_value if isinstance(item, ObjectRef))
    return _sorted_refs(tuple(refs))


def _denied_families() -> tuple[Identifier, ...]:
    return (
        "BUILD_TRANSCRIPT",
        "HIDDEN_REASONING",
        "RAW_PRIVATE_REFERENCE",
        "RAW_TRACE",
    )


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": _sorted_refs(refs)})


def _validate_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(refs):
        raise ValueError(f"{label} audit refs are stale")


def _require_ref(ref: ObjectRef, object_type: str, field_name: str) -> None:
    if ref.object_type != object_type:
        raise ValueError(f"{field_name} must reference {object_type}")


def _sorted_refs(refs: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(ref): ref for ref in refs}
    return tuple(unique[key] for key in sorted(unique))


def _require_sorted_unique_refs(
    label: str,
    refs: tuple[ObjectRef, ...],
) -> None:
    if refs != _sorted_refs(refs) or len(refs) != len(set(refs)):
        raise ValueError(f"{label} must be sorted and unique")


def _require_sorted_unique(label: str, values: tuple[object, ...]) -> None:
    if len(values) != len(set(values)) or values != tuple(sorted(values, key=repr)):
        raise ValueError(f"{label} must be sorted and unique")


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _object_ref_from_facade(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
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


def _validate_final_identity(
    *,
    object_id: str,
    object_sha256: str,
    expected_prefix: str,
    observed: str,
) -> None:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        return
    if object_sha256 != observed or object_id != f"{expected_prefix}://sha256/{observed}":
        raise ValueError(f"{expected_prefix} identity is stale")


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
    *,
    id_field: str,
    hash_field: str,
    prefix: str,
    digest: str,
) -> ModelT:
    return value.model_copy(
        update={
            id_field: f"{prefix}://sha256/{digest}",
            hash_field: digest,
        }
    )
