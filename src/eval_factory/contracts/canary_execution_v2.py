from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.cli_v2 import PipelineControlConfigV2
from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import (
    ContractModelV2,
    canonical_value_v2,
)
from eval_factory.contracts.observability_v2 import (
    BatchAuditOutcomeV2,
)
from eval_factory.contracts.orchestration import JobStatus
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedWorkUnitV2,
    StageNameV2,
    WorkDependencyJoinModeV2,
    WorkUnitScopeV2,
    dataset_job_spec_v2_ref,
    resolved_work_unit_v2_ref,
    trace_source_v2_ref,
)
from eval_factory.contracts.resource_v2 import (
    NonModelResourceVectorV2,
)
from eval_factory.contracts.review_v2 import (
    SemanticReviewRoundV2,
)

R6_CANARY_POLICY_VERSION: Literal["r6-canary-execution/r6-parent-v1"] = "r6-canary-execution/r6-parent-v1"
R6_SEMANTIC_REVIEW_FANOUT_POLICY_VERSION: Literal["semantic-review-fanout/r6-parent-v1"] = (
    "semantic-review-fanout/r6-parent-v1"
)
R6_CANARY_EXECUTION_PROFILE: Literal["r6-canary/r1-r5-v1"] = "r6-canary/r1-r5-v1"
R6_CANARY_CLAIM_SCOPE: Literal["DEVELOPMENT_CANARY_ONLY"] = "DEVELOPMENT_CANARY_ONLY"

R6_CANARY_STAGES = (
    StageNameV2.TRACE_INDEX,
    StageNameV2.SAFETY,
    StageNameV2.LABEL,
    StageNameV2.TASK_AUTHORING,
    StageNameV2.ATTACHMENT,
    StageNameV2.ITEM_QUALITY,
)
_R6_CANARY_REQUIRED_GOVERNING_VERSIONS = frozenset(
    {
        ("artifact-execution", "r5-06"),
        ("artifact-results", "r5-07"),
        ("deterministic-validation", "r5-08"),
        ("semantic-review", "r5-09"),
        ("item-quality", "r5-10"),
    }
)
_R6_CANARY_PROHIBITED_GOVERNING_TOKENS = frozenset(
    {
        "attestation",
        "r7",
        "r8",
        "registry",
        "registries",
        "release",
    }
)
_R6_CANARY_MAX_SEED_PAYLOAD_BYTES = 1_048_576


class R6CanaryObjectCodecV2(StrEnum):
    STRUCTURED_FACT_SET = "STRUCTURED_FACT_SET"
    LABEL_SPEC = "LABEL_SPEC"
    SEMANTIC_RESIDUAL_FIXTURE = "SEMANTIC_RESIDUAL_FIXTURE"
    TASK_EPISODE_FIXTURE = "TASK_EPISODE_FIXTURE"
    TASK_DRAFT_FIXTURE = "TASK_DRAFT_FIXTURE"
    TASK_PROMPT_SAFETY_FIXTURE = "TASK_PROMPT_SAFETY_FIXTURE"
    RUBRIC_GENERATION_FIXTURE = "RUBRIC_GENERATION_FIXTURE"
    PROMPT_ONLY_DEPENDENCY_FIXTURE = "PROMPT_ONLY_DEPENDENCY_FIXTURE"
    SEMANTIC_REVIEW_POLICY = "SEMANTIC_REVIEW_POLICY"
    CANARY_ATTACHMENT_FIXTURE = "CANARY_ATTACHMENT_FIXTURE"
    CANARY_REVIEW_FIXTURE = "CANARY_REVIEW_FIXTURE"
    TRACE_INDEX_OUTPUT = "TRACE_INDEX_OUTPUT"
    SAFETY_OUTPUT = "SAFETY_OUTPUT"
    LABEL_OUTPUT = "LABEL_OUTPUT"
    TASK_AUTHORING_OUTPUT = "TASK_AUTHORING_OUTPUT"
    ATTACHMENT_PREPARATION_OUTPUT = "ATTACHMENT_PREPARATION_OUTPUT"
    PROVIDER_OPERATION_OUTPUT = "PROVIDER_OPERATION_OUTPUT"
    ARTIFACT_GROUP_OUTPUT = "ARTIFACT_GROUP_OUTPUT"
    ATTACHMENT_OUTPUT = "ATTACHMENT_OUTPUT"
    SEMANTIC_REVIEW_ROUND_OUTPUT = "SEMANTIC_REVIEW_ROUND_OUTPUT"
    ITEM_QUALITY_OUTPUT = "ITEM_QUALITY_OUTPUT"


class R6CanaryExpectedItemOutcomeV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


_CODEC_OBJECT_TYPES: dict[R6CanaryObjectCodecV2, str] = {
    R6CanaryObjectCodecV2.STRUCTURED_FACT_SET: ("structured-fact-set"),
    R6CanaryObjectCodecV2.LABEL_SPEC: "label-spec",
    R6CanaryObjectCodecV2.SEMANTIC_RESIDUAL_FIXTURE: ("fake-semantic-residual-fixture"),
    R6CanaryObjectCodecV2.TASK_EPISODE_FIXTURE: ("fake-task-episode-grouping-fixture"),
    R6CanaryObjectCodecV2.TASK_DRAFT_FIXTURE: ("fake-task-draft-authoring-fixture"),
    R6CanaryObjectCodecV2.TASK_PROMPT_SAFETY_FIXTURE: ("fake-task-prompt-safety-fixture"),
    R6CanaryObjectCodecV2.RUBRIC_GENERATION_FIXTURE: ("fake-rubric-generation-fixture"),
    R6CanaryObjectCodecV2.PROMPT_ONLY_DEPENDENCY_FIXTURE: ("fake-prompt-only-dependency-fixture"),
    R6CanaryObjectCodecV2.SEMANTIC_REVIEW_POLICY: ("semantic-review-policy"),
    R6CanaryObjectCodecV2.CANARY_ATTACHMENT_FIXTURE: ("r6-canary-attachment-fixture"),
    R6CanaryObjectCodecV2.CANARY_REVIEW_FIXTURE: ("r6-canary-review-fixture"),
    R6CanaryObjectCodecV2.TRACE_INDEX_OUTPUT: ("r6-canary-trace-index-output"),
    R6CanaryObjectCodecV2.SAFETY_OUTPUT: ("r6-canary-safety-output"),
    R6CanaryObjectCodecV2.LABEL_OUTPUT: ("r6-canary-label-output"),
    R6CanaryObjectCodecV2.TASK_AUTHORING_OUTPUT: ("r6-canary-task-authoring-output"),
    R6CanaryObjectCodecV2.ATTACHMENT_PREPARATION_OUTPUT: ("r6-canary-attachment-preparation-output"),
    R6CanaryObjectCodecV2.PROVIDER_OPERATION_OUTPUT: ("r6-canary-provider-operation-output"),
    R6CanaryObjectCodecV2.ARTIFACT_GROUP_OUTPUT: ("r6-canary-artifact-group-output"),
    R6CanaryObjectCodecV2.ATTACHMENT_OUTPUT: ("r6-canary-attachment-output"),
    R6CanaryObjectCodecV2.SEMANTIC_REVIEW_ROUND_OUTPUT: ("r6-canary-semantic-review-round-output"),
    R6CanaryObjectCodecV2.ITEM_QUALITY_OUTPUT: ("r6-canary-item-quality-output"),
}
_SEED_CODECS = frozenset(
    {
        R6CanaryObjectCodecV2.STRUCTURED_FACT_SET,
        R6CanaryObjectCodecV2.LABEL_SPEC,
        R6CanaryObjectCodecV2.SEMANTIC_RESIDUAL_FIXTURE,
        R6CanaryObjectCodecV2.TASK_EPISODE_FIXTURE,
        R6CanaryObjectCodecV2.TASK_DRAFT_FIXTURE,
        R6CanaryObjectCodecV2.TASK_PROMPT_SAFETY_FIXTURE,
        R6CanaryObjectCodecV2.RUBRIC_GENERATION_FIXTURE,
        R6CanaryObjectCodecV2.PROMPT_ONLY_DEPENDENCY_FIXTURE,
        R6CanaryObjectCodecV2.SEMANTIC_REVIEW_POLICY,
        R6CanaryObjectCodecV2.CANARY_ATTACHMENT_FIXTURE,
        R6CanaryObjectCodecV2.CANARY_REVIEW_FIXTURE,
    }
)


class R6CanaryControlPlaneV2(ContractModelV2):
    schema_version: Literal["eval-factory/r6-canary-control-plane/v2"] = (
        "eval-factory/r6-canary-control-plane/v2"
    )
    resource_domain_ref: ObjectRef
    resource_capacity: NonModelResourceVectorV2
    worker_principal_ref: ObjectRef
    worker_version: str = Field(min_length=1, max_length=128)
    policy_version: Literal["r6-canary-execution/r6-parent-v1"] = R6_CANARY_POLICY_VERSION

    @model_validator(mode="after")
    def validate_control_plane(self) -> Self:
        _require_ref(
            self.resource_domain_ref,
            "resource-domain",
            "v1",
            "resource_domain_ref",
        )
        _require_ref(
            self.worker_principal_ref,
            "worker-principal",
            "v1",
            "worker_principal_ref",
        )
        if self.resource_capacity.processes < 1 or self.resource_capacity.storage_bytes < 1:
            raise ValueError("canary resource capacity requires process and storage allowance")
        return self


class R6CanarySeedObjectV2(ContractModelV2):
    schema_version: Literal["eval-factory/r6-canary-seed-object/v2"] = "eval-factory/r6-canary-seed-object/v2"
    object_ref: ObjectRef
    codec: R6CanaryObjectCodecV2
    payload_json: str = Field(
        min_length=2,
        max_length=_R6_CANARY_MAX_SEED_PAYLOAD_BYTES,
    )

    @model_validator(mode="after")
    def validate_seed(self) -> Self:
        if len(self.payload_json.encode("utf-8")) > _R6_CANARY_MAX_SEED_PAYLOAD_BYTES:
            raise ValueError("seed payload_json exceeds the UTF-8 byte limit")
        if self.codec not in _SEED_CODECS:
            raise ValueError("runtime output codec cannot be supplied as a seed")
        expected_type = _CODEC_OBJECT_TYPES[self.codec]
        if self.object_ref.object_type != expected_type:
            raise ValueError(f"{self.codec.value} must reference {expected_type}")
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("seed payload_json is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("seed payload_json must contain one object")
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if canonical != self.payload_json:
            raise ValueError("seed payload_json must use canonical encoding")
        return self


class R6CanaryTraceBindingV2(ContractModelV2):
    schema_version: Literal["eval-factory/r6-canary-trace-binding/v2"] = (
        "eval-factory/r6-canary-trace-binding/v2"
    )
    source_trace_ref: ObjectRef
    structured_fact_set_ref: ObjectRef
    label_spec_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    semantic_fixture_refs: tuple[ObjectRef, ...] = ()
    task_fixture_refs: tuple[ObjectRef, ...] = ()
    attachment_fixture_refs: tuple[ObjectRef, ...] = ()
    expected_outcome: R6CanaryExpectedItemOutcomeV2

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if self.source_trace_ref.object_type != "trace-source":
            raise ValueError("source_trace_ref must reference trace-source")
        if self.structured_fact_set_ref.object_type != "structured-fact-set":
            raise ValueError("structured_fact_set_ref must reference structured-fact-set")
        _require_refs(
            self.label_spec_refs,
            "label-spec",
            "label_spec_refs",
        )
        for label, refs in (
            (
                "semantic_fixture_refs",
                self.semantic_fixture_refs,
            ),
            ("task_fixture_refs", self.task_fixture_refs),
            (
                "attachment_fixture_refs",
                self.attachment_fixture_refs,
            ),
        ):
            _require_sorted_unique_refs(label, refs)
        all_fixture_refs = (
            *self.label_spec_refs,
            *self.semantic_fixture_refs,
            *self.task_fixture_refs,
            *self.attachment_fixture_refs,
        )
        if len(all_fixture_refs) != len(set(all_fixture_refs)):
            raise ValueError("trace fixture ref partitions must be disjoint")
        return self

    @property
    def seed_refs(self) -> tuple[ObjectRef, ...]:
        return (
            *self.label_spec_refs,
            *self.semantic_fixture_refs,
            *self.task_fixture_refs,
            *self.attachment_fixture_refs,
        )


class R6CanaryExecutionManifestV2(ContractModelV2):
    schema_version: Literal["eval-factory/r6-canary-execution-manifest/v2"] = (
        "eval-factory/r6-canary-execution-manifest/v2"
    )
    manifest_id: Identifier
    job_spec: DatasetJobSpecV2
    control: PipelineControlConfigV2
    execution_profile: Literal["r6-canary/r1-r5-v1"] = R6_CANARY_EXECUTION_PROFILE
    control_plane: R6CanaryControlPlaneV2
    seed_objects: tuple[R6CanarySeedObjectV2, ...] = Field(
        min_length=1,
        max_length=256,
    )
    trace_bindings: tuple[R6CanaryTraceBindingV2, ...] = Field(
        min_length=1,
        max_length=128,
    )
    claim_scope: Literal["DEVELOPMENT_CANARY_ONLY"] = R6_CANARY_CLAIM_SCOPE
    policy_version: Literal["r6-canary-execution/r6-parent-v1"] = R6_CANARY_POLICY_VERSION
    manifest_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.job_spec.requested_stages != R6_CANARY_STAGES:
            raise ValueError("canary Job stages must be the exact R1-R5 candidate chain")
        if self.job_spec.approval_mode is not ApprovalMode.NONE or self.job_spec.enabled_checkpoints:
            raise ValueError("canary Job cannot enable approval checkpoints")
        if self.job_spec.export_target.channel != "CANARY":
            raise ValueError("canary Job export target must use CANARY channel")
        if self.job_spec.model_profiles:
            raise ValueError("fixture-backed canary cannot claim model profiles")
        governing_versions = {
            (binding.component, binding.version) for binding in self.audit.governing_versions
        }
        if not _R6_CANARY_REQUIRED_GOVERNING_VERSIONS.issubset(governing_versions):
            raise ValueError("canary manifest is missing required R5 governing versions")
        if any(
            _contains_post_r6_governing_marker(
                binding.component,
                binding.version,
            )
            for binding in self.audit.governing_versions
        ):
            raise ValueError("canary manifest contains a post-R6 governing version")
        seed_refs = tuple(value.object_ref for value in self.seed_objects)
        _require_sorted_unique_refs("seed_objects", seed_refs)
        binding_sources = tuple(value.source_trace_ref for value in self.trace_bindings)
        _require_sorted_unique_refs(
            "trace_bindings",
            binding_sources,
        )
        expected_sources = tuple(
            sorted(
                (trace_source_v2_ref(trace) for trace in self.job_spec.traces),
                key=_ref_key,
            )
        )
        if binding_sources != expected_sources:
            raise ValueError("trace bindings must exactly cover Job traces")
        seed_ref_set = set(seed_refs)
        for binding in self.trace_bindings:
            if not set(binding.seed_refs).issubset(seed_ref_set):
                raise ValueError("trace binding references a missing seed object")
        expected_audit_refs = _sorted_refs(
            (
                dataset_job_spec_v2_ref(self.job_spec),
                self.control_plane.resource_domain_ref,
                self.control_plane.worker_principal_ref,
                *seed_refs,
                *binding_sources,
            )
        )
        if self.audit.input_refs != expected_audit_refs:
            raise ValueError("canary manifest audit refs are incomplete")
        validate_r6_canary_execution_manifest_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        job_spec: DatasetJobSpecV2,
        control: PipelineControlConfigV2,
        control_plane: R6CanaryControlPlaneV2,
        seed_objects: tuple[R6CanarySeedObjectV2, ...],
        trace_bindings: tuple[R6CanaryTraceBindingV2, ...],
        audit: ContractAudit,
    ) -> R6CanaryExecutionManifestV2:
        ordered_seeds = tuple(
            sorted(
                seed_objects,
                key=lambda value: _ref_key(value.object_ref),
            )
        )
        ordered_bindings = tuple(
            sorted(
                trace_bindings,
                key=lambda value: _ref_key(value.source_trace_ref),
            )
        )
        audit_refs = _sorted_refs(
            (
                dataset_job_spec_v2_ref(job_spec),
                control_plane.resource_domain_ref,
                control_plane.worker_principal_ref,
                *(value.object_ref for value in ordered_seeds),
                *(value.source_trace_ref for value in ordered_bindings),
            )
        )
        manifest = cls(
            manifest_id=("r6-canary-execution-manifest://pending"),
            job_spec=job_spec,
            control=control,
            control_plane=control_plane,
            seed_objects=ordered_seeds,
            trace_bindings=ordered_bindings,
            manifest_sha256="0" * 64,
            audit=_safe_audit(audit, audit_refs),
        )
        digest = r6_canary_execution_manifest_v2_carried_sha256(manifest)
        return manifest.model_copy(
            update={
                "manifest_id": (f"r6-canary-execution-manifest://sha256/{digest}"),
                "manifest_sha256": digest,
            }
        )


class SemanticReviewRoundWorkV2(ContractModelV2):
    schema_version: Literal["eval-factory/semantic-review-round-work/v2"] = (
        "eval-factory/semantic-review-round-work/v2"
    )
    round: SemanticReviewRoundV2
    work_unit: ResolvedWorkUnitV2

    @model_validator(mode="after")
    def validate_round_work(self) -> Self:
        if (
            self.work_unit.scope is not WorkUnitScopeV2.ITEM
            or self.work_unit.stage is not StageNameV2.ITEM_QUALITY
            or self.work_unit.semantic_review_round is not self.round
        ):
            raise ValueError(
                "semantic review round requires ITEM ITEM_QUALITY work with matching round identity"
            )
        return self


class SemanticReviewFanoutV2(ContractModelV2):
    schema_version: Literal["eval-factory/semantic-review-fanout/v2"] = (
        "eval-factory/semantic-review-fanout/v2"
    )
    semantic_review_fanout_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    parent_item_quality_work_unit_ref: ObjectRef
    round_work: tuple[SemanticReviewRoundWorkV2, ...] = Field(
        min_length=3,
        max_length=3,
    )
    policy_version: Literal["semantic-review-fanout/r6-parent-v1"] = R6_SEMANTIC_REVIEW_FANOUT_POLICY_VERSION
    fanout_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_fanout(self) -> Self:
        _require_ref(
            self.resolved_job_work_graph_ref,
            "resolved-job-work-graph",
            "v2",
            "resolved_job_work_graph_ref",
        )
        _require_ref(
            self.parent_item_quality_work_unit_ref,
            "resolved-work-unit",
            "v2",
            "parent_item_quality_work_unit_ref",
        )
        rounds = tuple(value.round for value in self.round_work)
        if rounds != tuple(SemanticReviewRoundV2):
            raise ValueError("semantic review round work must use exact round order")
        units = tuple(value.work_unit for value in self.round_work)
        first = units[0]
        if resolved_work_unit_v2_ref(first) == self.parent_item_quality_work_unit_ref:
            raise ValueError("semantic review work must be distinct from its finalizer")
        if first.join_mode is not WorkDependencyJoinModeV2.ALL_SUCCEEDED:
            raise ValueError("coverage review round requires ALL_SUCCEEDED join")
        for previous, current in pairwise(units):
            if current.depends_on_work_unit_refs != (resolved_work_unit_v2_ref(previous),):
                raise ValueError("semantic review round work dependencies do not follow round order")
            if current.join_mode is not WorkDependencyJoinModeV2.ALL_SUCCEEDED:
                raise ValueError("later semantic review rounds require ALL_SUCCEEDED joins")
        identity = {
            (
                unit.job_id,
                unit.item_id,
                unit.source_trace_ref,
            )
            for unit in units
        }
        if len(identity) != 1:
            raise ValueError("semantic review round work crosses Job, Item, or trace identity")
        unit_refs = tuple(resolved_work_unit_v2_ref(unit) for unit in units)
        _require_sorted_unique_refs(
            "semantic review work units",
            tuple(sorted(unit_refs, key=_ref_key)),
        )
        expected_audit = _sorted_refs(
            (
                self.resolved_job_work_graph_ref,
                self.parent_item_quality_work_unit_ref,
                *unit_refs,
            )
        )
        if self.audit.input_refs != expected_audit:
            raise ValueError("semantic review fanout audit refs are incomplete")
        validate_semantic_review_fanout_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        resolved_job_work_graph_ref: ObjectRef,
        parent_item_quality_work_unit_ref: ObjectRef,
        round_work: tuple[SemanticReviewRoundWorkV2, ...],
        audit: ContractAudit,
    ) -> SemanticReviewFanoutV2:
        unit_refs = tuple(resolved_work_unit_v2_ref(value.work_unit) for value in round_work)
        fanout = cls(
            semantic_review_fanout_id=("semantic-review-fanout://pending"),
            resolved_job_work_graph_ref=(resolved_job_work_graph_ref),
            parent_item_quality_work_unit_ref=(parent_item_quality_work_unit_ref),
            round_work=round_work,
            fanout_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                _sorted_refs(
                    (
                        resolved_job_work_graph_ref,
                        parent_item_quality_work_unit_ref,
                        *unit_refs,
                    )
                ),
            ),
        )
        digest = semantic_review_fanout_v2_carried_sha256(fanout)
        return fanout.model_copy(
            update={
                "semantic_review_fanout_id": (f"semantic-review-fanout://sha256/{digest}"),
                "fanout_sha256": digest,
            }
        )


class R6CanaryDatasetResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/r6-canary-dataset-result/v2"] = (
        "eval-factory/r6-canary-dataset-result/v2"
    )
    manifest_ref: ObjectRef
    job_id: Identifier
    job_status: JobStatus
    item_ids: tuple[Identifier, ...]
    succeeded_item_ids: tuple[Identifier, ...] = ()
    blocked_item_ids: tuple[Identifier, ...] = ()
    failed_item_ids: tuple[Identifier, ...] = ()
    quality_result_refs: tuple[ObjectRef, ...] = ()
    package_manifest_refs: tuple[ObjectRef, ...] = ()
    audit_report_ref: ObjectRef
    audit_outcome: BatchAuditOutcomeV2
    claim_scope: Literal["DEVELOPMENT_CANARY_ONLY"] = R6_CANARY_CLAIM_SCOPE
    policy_version: Literal["r6-canary-execution/r6-parent-v1"] = R6_CANARY_POLICY_VERSION

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.manifest_ref,
            "r6-canary-execution-manifest",
            "v2",
            "manifest_ref",
        )
        _require_ref(
            self.audit_report_ref,
            "batch-audit-report",
            "v2",
            "audit_report_ref",
        )
        _require_sorted_unique_values("item_ids", self.item_ids)
        partitions = (
            self.succeeded_item_ids,
            self.blocked_item_ids,
            self.failed_item_ids,
        )
        for label, values in zip(
            (
                "succeeded_item_ids",
                "blocked_item_ids",
                "failed_item_ids",
            ),
            partitions,
            strict=True,
        ):
            _require_sorted_unique_values(label, values)
        partition_values = tuple(value for values in partitions for value in values)
        if len(partition_values) != len(set(partition_values)) or set(partition_values) != set(self.item_ids):
            raise ValueError("candidate result item partitions must be disjoint and complete")
        _require_refs(
            self.quality_result_refs,
            "item-quality-compilation-result",
            "quality_result_refs",
            object_version="v2",
        )
        _require_refs(
            self.package_manifest_refs,
            "final-package-manifest",
            "package_manifest_refs",
            object_version="v2",
        )
        succeeded = len(self.succeeded_item_ids)
        if len(self.quality_result_refs) != succeeded or len(self.package_manifest_refs) != succeeded:
            raise ValueError("successful items require exact quality and package refs")
        if succeeded:
            if self.job_status is not JobStatus.SUCCEEDED:
                raise ValueError("candidate Job with successful items must be SUCCEEDED")
        elif self.job_status not in {
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            raise ValueError("candidate Job without successful items must be FAILED or CANCELLED")
        return self


def r6_canary_execution_manifest_v2_carried_sha256(
    value: R6CanaryExecutionManifestV2,
) -> str:
    return _carried_sha256(
        value,
        "manifest_id",
        "manifest_sha256",
    )


def r6_canary_execution_manifest_v2_ref(
    value: R6CanaryExecutionManifestV2,
) -> ObjectRef:
    validate_r6_canary_execution_manifest_v2_identity(value)
    return ObjectRef(
        object_type="r6-canary-execution-manifest",
        object_id=value.manifest_id,
        object_version="v2",
        object_sha256=value.manifest_sha256,
    )


def validate_r6_canary_execution_manifest_v2_identity(
    value: R6CanaryExecutionManifestV2,
) -> None:
    _validate_identity(
        object_id=value.manifest_id,
        object_sha256=value.manifest_sha256,
        prefix="r6-canary-execution-manifest",
        observed=r6_canary_execution_manifest_v2_carried_sha256(value),
    )


def semantic_review_fanout_v2_carried_sha256(
    value: SemanticReviewFanoutV2,
) -> str:
    return _carried_sha256(
        value,
        "semantic_review_fanout_id",
        "fanout_sha256",
    )


def semantic_review_fanout_v2_ref(
    value: SemanticReviewFanoutV2,
) -> ObjectRef:
    validate_semantic_review_fanout_v2_identity(value)
    return ObjectRef(
        object_type="semantic-review-fanout",
        object_id=value.semantic_review_fanout_id,
        object_version="v2",
        object_sha256=value.fanout_sha256,
    )


def validate_semantic_review_fanout_v2_identity(
    value: SemanticReviewFanoutV2,
) -> None:
    _validate_identity(
        object_id=value.semantic_review_fanout_id,
        object_sha256=value.fanout_sha256,
        prefix="semantic-review-fanout",
        observed=semantic_review_fanout_v2_carried_sha256(value),
    )


def _carried_sha256(
    value: ContractModelV2,
    id_field: str,
    sha_field: str,
) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={id_field, sha_field, "audit"},
    )
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_identity(
    *,
    object_id: str,
    object_sha256: str,
    prefix: str,
    observed: str,
) -> None:
    if object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    if object_sha256 != observed or object_id != f"{prefix}://sha256/{observed}":
        raise ValueError(f"{prefix} carried identity is stale or invalid")


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=refs,
    )


def _require_ref(
    ref: ObjectRef,
    object_type: str,
    object_version: str,
    label: str,
) -> None:
    if ref.object_type != object_type or ref.object_version != object_version:
        raise ValueError(f"{label} must reference {object_type} {object_version}")


def _require_refs(
    refs: tuple[ObjectRef, ...],
    object_type: str,
    label: str,
    *,
    object_version: str | None = None,
) -> None:
    _require_sorted_unique_refs(label, refs)
    for ref in refs:
        if ref.object_type != object_type:
            raise ValueError(f"{label} must reference {object_type}")
        if object_version is not None and ref.object_version != object_version:
            raise ValueError(f"{label} must reference {object_type} {object_version}")


def _require_sorted_unique_refs(
    label: str,
    refs: tuple[ObjectRef, ...],
) -> None:
    if refs != _sorted_refs(refs):
        raise ValueError(f"{label} must be sorted and unique")


def _sorted_refs(
    refs: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(refs), key=_ref_key))


def _ref_key(
    ref: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _require_sorted_unique_values(
    label: str,
    values: tuple[str, ...],
) -> None:
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{label} must be sorted and unique")


def _contains_post_r6_governing_marker(
    component: str,
    version: str,
) -> bool:
    tokens = {
        token
        for value in (component, version)
        for token in "".join(
            character if character.isalnum() else " " for character in value.casefold()
        ).split()
    }
    return bool(tokens.intersection(_R6_CANARY_PROHIBITED_GOVERNING_TOKENS))
