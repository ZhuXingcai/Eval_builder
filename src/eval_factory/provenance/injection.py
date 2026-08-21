from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import ContractAudit, ContractModel, Identifier, ObjectRef, Sha256
from eval_factory.contracts.safety import (
    NON_WAIVABLE_RISKS,
    NON_WAIVABLE_TAINTS,
    ContentRiskLabel,
    Disposition,
    ProvenanceDecision,
    TaintLabel,
)

PROMPT_INJECTION_AS_DATA_POLICY_VERSION: Literal["prompt-injection-as-data/r2-07-v1"] = (
    "prompt-injection-as-data/r2-07-v1"
)

_RAW_TRACE_OBJECT_TYPES = frozenset({"raw-trace", "raw-traj", "trace-raw"})


class PromptInjectionBoundaryPolicyError(RuntimeError):
    pass


class PromptBoundarySurface(StrEnum):
    SYSTEM_INSTRUCTION = "SYSTEM_INSTRUCTION"
    DEVELOPER_INSTRUCTION = "DEVELOPER_INSTRUCTION"
    TOOL_DEFINITION = "TOOL_DEFINITION"
    TOOL_ARGUMENT_SCHEMA = "TOOL_ARGUMENT_SCHEMA"
    MCP_CONFIG = "MCP_CONFIG"
    PLUGIN_CONFIG = "PLUGIN_CONFIG"
    MODEL_PROFILE = "MODEL_PROFILE"
    RUNTIME_CONFIG = "RUNTIME_CONFIG"
    TOOL_POLICY_CONFIG = "TOOL_POLICY_CONFIG"
    DEPENDENCY_DISCOVERY_DATA = "DEPENDENCY_DISCOVERY_DATA"
    PRODUCER_TASK_DATA = "PRODUCER_TASK_DATA"
    RUBRIC_AUTHORING_DATA = "RUBRIC_AUTHORING_DATA"
    SAFETY_REVIEW_DATA = "SAFETY_REVIEW_DATA"
    TASK_REWRITE_DATA = "TASK_REWRITE_DATA"
    USER_PROMPT_DATA = "USER_PROMPT_DATA"
    QUERY_INSTRUCTION_DATA = "QUERY_INSTRUCTION_DATA"
    EVIDENCE_DATA = "EVIDENCE_DATA"
    ATTACHMENT_REQUIREMENT_DATA = "ATTACHMENT_REQUIREMENT_DATA"
    CONTESTANT_VISIBLE_DATA = "CONTESTANT_VISIBLE_DATA"
    BLOCKED_METADATA = "BLOCKED_METADATA"


class PromptBoundarySourceRole(StrEnum):
    STATIC_HARNESS_CONTROL = "STATIC_HARNESS_CONTROL"
    STATIC_POLICY_CONFIG = "STATIC_POLICY_CONFIG"
    APPROVED_TASK_TEXT = "APPROVED_TASK_TEXT"
    CONTESTANT_VISIBLE_TASK_TEXT = "CONTESTANT_VISIBLE_TASK_TEXT"
    CANDIDATE_TASK_CONTRACT = "CANDIDATE_TASK_CONTRACT"
    CANDIDATE_TASK_TEXT = "CANDIDATE_TASK_TEXT"
    PROMPT_ONLY_DEPENDENCY_CONTRACT = "PROMPT_ONLY_DEPENDENCY_CONTRACT"
    TASK_REWRITE_CONTRACT = "TASK_REWRITE_CONTRACT"
    PROMPT_SAFETY_PASSED_TASK_CONTRACT = "PROMPT_SAFETY_PASSED_TASK_CONTRACT"
    SAFE_EVIDENCE_PROJECTION = "SAFE_EVIDENCE_PROJECTION"
    EVIDENCE_BUNDLE_REF = "EVIDENCE_BUNDLE_REF"
    TRACE_EVENT_TEXT = "TRACE_EVENT_TEXT"
    TOOL_RESULT_TEXT = "TOOL_RESULT_TEXT"
    FILE_TEXT = "FILE_TEXT"
    RETRIEVED_EXTERNAL_TEXT = "RETRIEVED_EXTERNAL_TEXT"
    USER_SUPPLIED_TEXT = "USER_SUPPLIED_TEXT"
    QUARANTINED_OR_PRIVATE_REF = "QUARANTINED_OR_PRIVATE_REF"
    UNKNOWN = "UNKNOWN"


class PromptBoundaryViolationReason(StrEnum):
    BLOCKED_METADATA_CONTAINS_TEXT = "BLOCKED_METADATA_CONTAINS_TEXT"
    CONTROL_SURFACE_SOURCE = "CONTROL_SURFACE_SOURCE"
    DISPOSITION_DENIED = "DISPOSITION_DENIED"
    DUPLICATE_SEGMENT_ID = "DUPLICATE_SEGMENT_ID"
    INVALID_CANDIDATE_TASK_BOUNDARY = "INVALID_CANDIDATE_TASK_BOUNDARY"
    INVALID_CANDIDATE_TASK_CONTRACT_BOUNDARY = "INVALID_CANDIDATE_TASK_CONTRACT_BOUNDARY"
    INVALID_PROMPT_ONLY_DEPENDENCY_BOUNDARY = "INVALID_PROMPT_ONLY_DEPENDENCY_BOUNDARY"
    INVALID_PRODUCER_TASK_BOUNDARY = "INVALID_PRODUCER_TASK_BOUNDARY"
    INVALID_TASK_REWRITE_BOUNDARY = "INVALID_TASK_REWRITE_BOUNDARY"
    MISSING_EVIDENCE_BUNDLE_REF = "MISSING_EVIDENCE_BUNDLE_REF"
    MISSING_PRODUCER_TASK_SEGMENT = "MISSING_PRODUCER_TASK_SEGMENT"
    MISSING_PROJECTION_POLICY_REF = "MISSING_PROJECTION_POLICY_REF"
    MISSING_PROVENANCE_DECISION = "MISSING_PROVENANCE_DECISION"
    MISSING_UNTRUSTED_DATA_MARKER = "MISSING_UNTRUSTED_DATA_MARKER"
    NON_WAIVABLE_RESTRICTION = "NON_WAIVABLE_RESTRICTION"
    PROMPT_INJECTION_RISK = "PROMPT_INJECTION_RISK"
    QUARANTINE_OR_PRIVATE_SOURCE = "QUARANTINE_OR_PRIVATE_SOURCE"
    RAW_TRACE_SOURCE = "RAW_TRACE_SOURCE"
    STALE_OR_MISMATCHED_HASH = "STALE_OR_MISMATCHED_HASH"
    STATIC_CONTROL_MARKED_UNTRUSTED = "STATIC_CONTROL_MARKED_UNTRUSTED"
    UNTRUSTED_INSTRUCTION_TAINT = "UNTRUSTED_INSTRUCTION_TAINT"
    UNKNOWN_SOURCE_ROLE = "UNKNOWN_SOURCE_ROLE"


_CONTROL_SURFACES = frozenset(
    {
        PromptBoundarySurface.SYSTEM_INSTRUCTION,
        PromptBoundarySurface.DEVELOPER_INSTRUCTION,
        PromptBoundarySurface.TOOL_DEFINITION,
        PromptBoundarySurface.TOOL_ARGUMENT_SCHEMA,
        PromptBoundarySurface.MCP_CONFIG,
        PromptBoundarySurface.PLUGIN_CONFIG,
        PromptBoundarySurface.MODEL_PROFILE,
        PromptBoundarySurface.RUNTIME_CONFIG,
        PromptBoundarySurface.TOOL_POLICY_CONFIG,
    }
)
_STATIC_CONTROL_ROLES = frozenset(
    {
        PromptBoundarySourceRole.STATIC_HARNESS_CONTROL,
        PromptBoundarySourceRole.STATIC_POLICY_CONFIG,
    }
)
_ROLES_REQUIRING_DECISION = frozenset(
    {
        PromptBoundarySourceRole.APPROVED_TASK_TEXT,
        PromptBoundarySourceRole.CONTESTANT_VISIBLE_TASK_TEXT,
        PromptBoundarySourceRole.SAFE_EVIDENCE_PROJECTION,
        PromptBoundarySourceRole.TRACE_EVENT_TEXT,
        PromptBoundarySourceRole.TOOL_RESULT_TEXT,
        PromptBoundarySourceRole.FILE_TEXT,
        PromptBoundarySourceRole.RETRIEVED_EXTERNAL_TEXT,
        PromptBoundarySourceRole.USER_SUPPLIED_TEXT,
    }
)
_MARKED_DATA_ROLES = _ROLES_REQUIRING_DECISION | frozenset(
    {
        PromptBoundarySourceRole.CANDIDATE_TASK_CONTRACT,
        PromptBoundarySourceRole.CANDIDATE_TASK_TEXT,
        PromptBoundarySourceRole.EVIDENCE_BUNDLE_REF,
        PromptBoundarySourceRole.PROMPT_ONLY_DEPENDENCY_CONTRACT,
        PromptBoundarySourceRole.PROMPT_SAFETY_PASSED_TASK_CONTRACT,
        PromptBoundarySourceRole.TASK_REWRITE_CONTRACT,
    }
)
_COLLECTABLE_REASONS = frozenset(
    {
        PromptBoundaryViolationReason.DISPOSITION_DENIED,
        PromptBoundaryViolationReason.NON_WAIVABLE_RESTRICTION,
        PromptBoundaryViolationReason.PROMPT_INJECTION_RISK,
        PromptBoundaryViolationReason.UNTRUSTED_INSTRUCTION_TAINT,
    }
)


class PromptBoundarySegment(ContractModel):
    schema_version: Literal["eval-factory/prompt-boundary-segment/r2-07"] = (
        "eval-factory/prompt-boundary-segment/r2-07"
    )
    segment_id: Identifier
    surface: PromptBoundarySurface
    source_role: PromptBoundarySourceRole
    source_ref: ObjectRef
    decision: ProvenanceDecision | None = None
    projection_policy_ref: ObjectRef | None = None
    evidence_bundle_ref: ObjectRef | None = None
    content_sha256: Sha256 | None = None
    untrusted_data_marker: bool
    text_preview: str | None = Field(default=None, min_length=1, max_length=2000)

    @field_validator("surface", mode="before")
    @classmethod
    def parse_surface(cls, value: object) -> PromptBoundarySurface:
        if isinstance(value, PromptBoundarySurface):
            return value
        if isinstance(value, str):
            return PromptBoundarySurface(value)
        raise TypeError("surface must be a PromptBoundarySurface")

    @field_validator("source_role", mode="before")
    @classmethod
    def parse_source_role(cls, value: object) -> PromptBoundarySourceRole:
        if isinstance(value, PromptBoundarySourceRole):
            return value
        if isinstance(value, str):
            return PromptBoundarySourceRole(value)
        raise TypeError("source_role must be a PromptBoundarySourceRole")


class PromptBoundaryViolation(ContractModel):
    schema_version: Literal["eval-factory/prompt-boundary-violation/r2-07"] = (
        "eval-factory/prompt-boundary-violation/r2-07"
    )
    segment_id: Identifier
    source_ref: ObjectRef
    surface: PromptBoundarySurface
    source_role: PromptBoundarySourceRole
    reason: PromptBoundaryViolationReason
    detail: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("surface", mode="before")
    @classmethod
    def parse_surface(cls, value: object) -> PromptBoundarySurface:
        if isinstance(value, PromptBoundarySurface):
            return value
        if isinstance(value, str):
            return PromptBoundarySurface(value)
        raise TypeError("surface must be a PromptBoundarySurface")

    @field_validator("source_role", mode="before")
    @classmethod
    def parse_source_role(cls, value: object) -> PromptBoundarySourceRole:
        if isinstance(value, PromptBoundarySourceRole):
            return value
        if isinstance(value, str):
            return PromptBoundarySourceRole(value)
        raise TypeError("source_role must be a PromptBoundarySourceRole")

    @field_validator("reason", mode="before")
    @classmethod
    def parse_reason(cls, value: object) -> PromptBoundaryViolationReason:
        if isinstance(value, PromptBoundaryViolationReason):
            return value
        if isinstance(value, str):
            return PromptBoundaryViolationReason(value)
        raise TypeError("reason must be a PromptBoundaryViolationReason")


class PromptBoundaryEnforcementRequest(ContractModel):
    schema_version: Literal["eval-factory/prompt-boundary-enforcement-request/r2-07"] = (
        "eval-factory/prompt-boundary-enforcement-request/r2-07"
    )
    boundary_id: Identifier
    segments: tuple[PromptBoundarySegment, ...] = Field(min_length=1)
    approved_projection_policy_refs: tuple[ObjectRef, ...] = ()
    approved_evidence_bundle_refs: tuple[ObjectRef, ...] = ()
    collect_blocked_metadata: bool = False
    audit: ContractAudit


class PromptBoundaryEnforcementResult(ContractModel):
    schema_version: Literal["eval-factory/prompt-boundary-enforcement-result/r2-07"] = (
        "eval-factory/prompt-boundary-enforcement-result/r2-07"
    )
    enforcement_id: Identifier
    accepted_segments: tuple[PromptBoundarySegment, ...] = ()
    rejected_segments: tuple[PromptBoundaryViolation, ...] = ()
    trace_injection_execution_count: Literal[0] = 0
    policy_version: Literal["prompt-injection-as-data/r2-07-v1"] = PROMPT_INJECTION_AS_DATA_POLICY_VERSION
    enforcement_sha256: Sha256
    audit: ContractAudit


class ProducerTaskViewBoundaryRequest(ContractModel):
    schema_version: Literal["eval-factory/producer-task-view-boundary-request/r2-07"] = (
        "eval-factory/producer-task-view-boundary-request/r2-07"
    )
    producer_task_view_ref: ObjectRef
    boundary_request: PromptBoundaryEnforcementRequest
    query_instruction_segment_id: Identifier
    control_segment_ids: tuple[Identifier, ...] = ()
    safe_evidence_bundle_refs: tuple[ObjectRef, ...] = ()


class CandidateTaskPromptBoundaryRequest(ContractModel):
    schema_version: Literal["eval-factory/candidate-task-prompt-boundary-request/r4-04"] = (
        "eval-factory/candidate-task-prompt-boundary-request/r4-04"
    )
    task_draft_ref: ObjectRef
    visible_prompt_ref: ObjectRef
    boundary_request: PromptBoundaryEnforcementRequest
    prompt_segment_id: Identifier

    @model_validator(mode="after")
    def validate_refs(self) -> CandidateTaskPromptBoundaryRequest:
        if self.task_draft_ref.object_type != "task-draft":
            raise ValueError("task_draft_ref must reference task-draft")
        if self.visible_prompt_ref.object_type != "task-draft-visible-prompt":
            raise ValueError("visible_prompt_ref must reference task-draft-visible-prompt")
        return self


class CandidateTaskContractBoundaryRequest(ContractModel):
    schema_version: Literal["eval-factory/candidate-task-contract-boundary-request/r4-05"] = (
        "eval-factory/candidate-task-contract-boundary-request/r4-05"
    )
    task_draft_ref: ObjectRef
    candidate_projection_ref: ObjectRef
    boundary_request: PromptBoundaryEnforcementRequest
    projection_segment_id: Identifier

    @model_validator(mode="after")
    def validate_refs(self) -> CandidateTaskContractBoundaryRequest:
        if self.task_draft_ref.object_type != "task-draft":
            raise ValueError("task_draft_ref must reference task-draft")
        if self.task_draft_ref.object_version != "v2":
            raise ValueError("task_draft_ref must reference TaskDraft v2")
        if self.candidate_projection_ref.object_type != "task-draft-rubric-input":
            raise ValueError("candidate_projection_ref must reference task-draft-rubric-input")
        if self.candidate_projection_ref.object_version != "v2":
            raise ValueError("candidate_projection_ref must reference rubric input v2")
        return self


class ProducerTaskDataBoundaryRequest(ContractModel):
    schema_version: Literal["eval-factory/producer-task-data-boundary-request/r4-08"] = (
        "eval-factory/producer-task-data-boundary-request/r4-08"
    )
    task_draft_ref: ObjectRef
    task_prompt_safety_gate_ref: ObjectRef
    producer_projection_ref: ObjectRef
    contestant_tool_policy_ref: ObjectRef
    safe_evidence_bundle_refs: tuple[ObjectRef, ...] = ()
    boundary_request: PromptBoundaryEnforcementRequest
    projection_segment_id: Identifier
    tool_control_segment_id: Identifier
    evidence_bundle_segment_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_refs(self) -> ProducerTaskDataBoundaryRequest:
        if self.task_draft_ref.object_type != "task-draft" or self.task_draft_ref.object_version != "v2":
            raise ValueError("task_draft_ref must reference TaskDraft v2")
        if (
            self.task_prompt_safety_gate_ref.object_type != "task-prompt-safety-gate"
            or self.task_prompt_safety_gate_ref.object_version != "v2"
        ):
            raise ValueError("task_prompt_safety_gate_ref must reference prompt safety gate v2")
        if (
            self.producer_projection_ref.object_type != "producer-task-view-input"
            or self.producer_projection_ref.object_version != "v2"
        ):
            raise ValueError("producer_projection_ref must reference producer projection v2")
        if (
            self.contestant_tool_policy_ref.object_type != "contestant-tool-policy"
            or self.contestant_tool_policy_ref.object_version != "v2"
        ):
            raise ValueError("contestant_tool_policy_ref must reference contestant tool policy v2")
        for ref in self.safe_evidence_bundle_refs:
            if ref.object_type != "evidence-bundle":
                raise ValueError("safe_evidence_bundle_refs must reference evidence-bundle")
        if len(self.safe_evidence_bundle_refs) != len(set(self.safe_evidence_bundle_refs)):
            raise ValueError("safe evidence bundle refs must be unique")
        if len(self.evidence_bundle_segment_ids) != len(set(self.evidence_bundle_segment_ids)):
            raise ValueError("evidence bundle segment IDs must be unique")
        return self


class TaskRewriteDataBoundaryRequest(ContractModel):
    schema_version: Literal["eval-factory/task-rewrite-data-boundary-request/r4-09"] = (
        "eval-factory/task-rewrite-data-boundary-request/r4-09"
    )
    source_task_draft_ref: ObjectRef
    replacement_plan_version_ref: ObjectRef
    replacement_preview_ref: ObjectRef
    rewrite_projection_ref: ObjectRef
    boundary_request: PromptBoundaryEnforcementRequest
    projection_segment_id: Identifier

    @model_validator(mode="after")
    def validate_refs(self) -> TaskRewriteDataBoundaryRequest:
        _require_v2_ref(
            self.source_task_draft_ref,
            "task-draft",
            "source_task_draft_ref",
        )
        _require_v2_ref(
            self.replacement_plan_version_ref,
            "task-rewrite-plan-version",
            "replacement_plan_version_ref",
        )
        _require_v2_ref(
            self.replacement_preview_ref,
            "task-rewrite-plan-preview",
            "replacement_preview_ref",
        )
        _require_v2_ref(
            self.rewrite_projection_ref,
            "task-rewrite-input",
            "rewrite_projection_ref",
        )
        return self


class PromptOnlyDependencyDataBoundaryRequest(ContractModel):
    schema_version: Literal["eval-factory/prompt-only-dependency-data-boundary-request/r5-03"] = (
        "eval-factory/prompt-only-dependency-data-boundary-request/r5-03"
    )
    attachment_planning_context_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    dependency_planning_context_ref: ObjectRef
    discovery_projection_ref: ObjectRef
    boundary_request: PromptBoundaryEnforcementRequest
    projection_segment_id: Identifier

    @model_validator(mode="after")
    def validate_refs(self) -> PromptOnlyDependencyDataBoundaryRequest:
        _require_v2_ref(
            self.attachment_planning_context_ref,
            "attachment-planning-context",
            "attachment_planning_context_ref",
        )
        _require_v2_ref(
            self.producer_task_view_ref,
            "producer-task-view",
            "producer_task_view_ref",
        )
        _require_v2_ref(
            self.dependency_planning_context_ref,
            "prompt-only-dependency-planning-context",
            "dependency_planning_context_ref",
        )
        _require_v2_ref(
            self.discovery_projection_ref,
            "prompt-only-dependency-input",
            "discovery_projection_ref",
        )
        return self


class PromptInjectionBoundaryEnforcer:
    policy_version = PROMPT_INJECTION_AS_DATA_POLICY_VERSION

    def enforce(self, request: PromptBoundaryEnforcementRequest) -> PromptBoundaryEnforcementResult:
        self._validate_unique_segments(request.segments)
        approved_policy_refs = {_object_ref_key(item) for item in request.approved_projection_policy_refs}
        approved_bundle_refs = {_object_ref_key(item) for item in request.approved_evidence_bundle_refs}
        accepted: list[PromptBoundarySegment] = []
        rejected: list[PromptBoundaryViolation] = []

        for segment in request.segments:
            violation = _segment_violation(segment, approved_policy_refs, approved_bundle_refs)
            if violation is None:
                accepted.append(segment)
                continue
            if request.collect_blocked_metadata and violation.reason in _COLLECTABLE_REASONS:
                rejected.append(violation)
                continue
            raise PromptInjectionBoundaryPolicyError(_violation_message(violation))

        result_payload = {
            "boundary_id": request.boundary_id,
            "accepted_segments": [_segment_facts(item) for item in accepted],
            "rejected_segments": [_violation_facts(item) for item in rejected],
            "trace_injection_execution_count": 0,
            "policy_version": self.policy_version,
        }
        return PromptBoundaryEnforcementResult(
            enforcement_id=_stable_id("prompt-boundary-enforcement", result_payload),
            accepted_segments=tuple(accepted),
            rejected_segments=tuple(rejected),
            trace_injection_execution_count=0,
            policy_version=self.policy_version,
            enforcement_sha256=_stable_hash(result_payload),
            audit=request.audit,
        )

    def validate_producer_task_view_boundary(
        self,
        request: ProducerTaskViewBoundaryRequest,
    ) -> PromptBoundaryEnforcementResult:
        result = self.enforce(request.boundary_request)
        accepted_by_id = {segment.segment_id: segment for segment in result.accepted_segments}
        query = accepted_by_id.get(request.query_instruction_segment_id)
        if query is None:
            raise PromptInjectionBoundaryPolicyError(
                "producer task view query instruction segment is missing"
            )
        if query.surface is not PromptBoundarySurface.QUERY_INSTRUCTION_DATA:
            raise PromptInjectionBoundaryPolicyError("producer task view query instruction must be data")
        if query.source_role not in {
            PromptBoundarySourceRole.APPROVED_TASK_TEXT,
            PromptBoundarySourceRole.CONTESTANT_VISIBLE_TASK_TEXT,
        }:
            raise PromptInjectionBoundaryPolicyError("producer task view query instruction source is invalid")

        for segment_id in request.control_segment_ids:
            segment = accepted_by_id.get(segment_id)
            if segment is None:
                raise PromptInjectionBoundaryPolicyError("producer task view control segment is missing")
            if segment.surface not in _CONTROL_SURFACES:
                raise PromptInjectionBoundaryPolicyError(
                    "producer task view control segment must use a control surface"
                )
            if segment.source_role not in _STATIC_CONTROL_ROLES:
                raise PromptInjectionBoundaryPolicyError("producer task view control segment must be static")

        approved_bundle_refs = {
            _object_ref_key(item) for item in request.boundary_request.approved_evidence_bundle_refs
        }
        for bundle_ref in request.safe_evidence_bundle_refs:
            if _object_ref_key(bundle_ref) not in approved_bundle_refs:
                raise PromptInjectionBoundaryPolicyError(
                    "producer task view evidence bundle ref is not approved"
                )
        return result

    def validate_candidate_task_prompt_boundary(
        self,
        request: CandidateTaskPromptBoundaryRequest,
    ) -> PromptBoundaryEnforcementResult:
        result = self.enforce(request.boundary_request)
        accepted_by_id = {segment.segment_id: segment for segment in result.accepted_segments}
        prompt = accepted_by_id.get(request.prompt_segment_id)
        if prompt is None:
            raise PromptInjectionBoundaryPolicyError("candidate task prompt segment is missing")
        if prompt.surface is not PromptBoundarySurface.SAFETY_REVIEW_DATA:
            raise PromptInjectionBoundaryPolicyError("candidate task prompt must use safety review data")
        if prompt.source_role is not PromptBoundarySourceRole.CANDIDATE_TASK_TEXT:
            raise PromptInjectionBoundaryPolicyError("candidate task prompt must use candidate task text")
        if prompt.source_ref != request.visible_prompt_ref:
            raise PromptInjectionBoundaryPolicyError(
                "candidate task visible prompt ref is stale or mismatched"
            )
        if prompt.content_sha256 != request.visible_prompt_ref.object_sha256:
            raise PromptInjectionBoundaryPolicyError("candidate task prompt hash is stale or mismatched")
        if len(result.accepted_segments) != 1:
            raise PromptInjectionBoundaryPolicyError(
                "candidate task safety review boundary accepts exactly one prompt segment"
            )
        return result

    def validate_candidate_task_contract_boundary(
        self,
        request: CandidateTaskContractBoundaryRequest,
    ) -> PromptBoundaryEnforcementResult:
        segment_by_id = {segment.segment_id: segment for segment in request.boundary_request.segments}
        projection = segment_by_id.get(request.projection_segment_id)
        if projection is None:
            raise PromptInjectionBoundaryPolicyError("candidate task contract projection segment is missing")
        if projection.surface is not PromptBoundarySurface.RUBRIC_AUTHORING_DATA:
            raise PromptInjectionBoundaryPolicyError("candidate task contract must use rubric authoring data")
        if projection.source_role is not PromptBoundarySourceRole.CANDIDATE_TASK_CONTRACT:
            raise PromptInjectionBoundaryPolicyError(
                "candidate task contract must use candidate task contract role"
            )
        result = self.enforce(request.boundary_request)
        if projection.source_ref != request.candidate_projection_ref:
            raise PromptInjectionBoundaryPolicyError(
                "candidate task contract projection ref is stale or mismatched"
            )
        if projection.content_sha256 != request.candidate_projection_ref.object_sha256:
            raise PromptInjectionBoundaryPolicyError(
                "candidate task contract projection hash is stale or mismatched"
            )
        if len(result.accepted_segments) != 1:
            raise PromptInjectionBoundaryPolicyError(
                "candidate task rubric authoring boundary accepts exactly one projection segment"
            )
        return result

    def validate_producer_task_data_boundary(
        self,
        request: ProducerTaskDataBoundaryRequest,
    ) -> PromptBoundaryEnforcementResult:
        segment_by_id = {segment.segment_id: segment for segment in request.boundary_request.segments}
        projection = segment_by_id.get(request.projection_segment_id)
        if projection is None:
            raise PromptInjectionBoundaryPolicyError("producer task projection segment is missing")
        if projection.surface is not PromptBoundarySurface.PRODUCER_TASK_DATA:
            raise PromptInjectionBoundaryPolicyError("producer task contract must use producer task data")
        if projection.source_role is not PromptBoundarySourceRole.PROMPT_SAFETY_PASSED_TASK_CONTRACT:
            raise PromptInjectionBoundaryPolicyError(
                "producer task contract must use prompt-safety-passed task role"
            )
        tool_control = segment_by_id.get(request.tool_control_segment_id)
        if tool_control is None:
            raise PromptInjectionBoundaryPolicyError("producer task tool control segment is missing")
        if (
            tool_control.surface is not PromptBoundarySurface.TOOL_POLICY_CONFIG
            or tool_control.source_role is not PromptBoundarySourceRole.STATIC_POLICY_CONFIG
            or tool_control.source_ref != request.contestant_tool_policy_ref
        ):
            raise PromptInjectionBoundaryPolicyError(
                "producer task tool control must bind exact static contestant policy"
            )
        if len(request.evidence_bundle_segment_ids) != len(request.safe_evidence_bundle_refs):
            raise PromptInjectionBoundaryPolicyError(
                "producer task evidence bundle segment inventory is incomplete"
            )
        for segment_id, bundle_ref in zip(
            request.evidence_bundle_segment_ids,
            request.safe_evidence_bundle_refs,
            strict=True,
        ):
            segment = segment_by_id.get(segment_id)
            if (
                segment is None
                or segment.surface is not PromptBoundarySurface.EVIDENCE_DATA
                or segment.source_role is not PromptBoundarySourceRole.EVIDENCE_BUNDLE_REF
                or segment.source_ref != bundle_ref
                or segment.evidence_bundle_ref != bundle_ref
            ):
                raise PromptInjectionBoundaryPolicyError(
                    "producer task evidence bundle segment is stale or mismatched"
                )
        approved_bundles = tuple(
            sorted(
                request.boundary_request.approved_evidence_bundle_refs,
                key=_object_ref_key,
            )
        )
        expected_bundles = tuple(
            sorted(
                request.safe_evidence_bundle_refs,
                key=_object_ref_key,
            )
        )
        if approved_bundles != expected_bundles:
            raise PromptInjectionBoundaryPolicyError(
                "producer task approved evidence bundle refs are stale or mismatched"
            )
        result = self.enforce(request.boundary_request)
        if projection.source_ref != request.producer_projection_ref:
            raise PromptInjectionBoundaryPolicyError("producer task projection ref is stale or mismatched")
        if projection.content_sha256 != request.producer_projection_ref.object_sha256:
            raise PromptInjectionBoundaryPolicyError("producer task projection hash is stale or mismatched")
        expected_segment_ids = {
            request.projection_segment_id,
            request.tool_control_segment_id,
            *request.evidence_bundle_segment_ids,
        }
        if {segment.segment_id for segment in result.accepted_segments} != expected_segment_ids or len(
            result.accepted_segments
        ) != len(expected_segment_ids):
            raise PromptInjectionBoundaryPolicyError(
                "producer task boundary must accept the exact segment inventory"
            )
        return result

    def validate_task_rewrite_data_boundary(
        self,
        request: TaskRewriteDataBoundaryRequest,
    ) -> PromptBoundaryEnforcementResult:
        segment_by_id = {segment.segment_id: segment for segment in request.boundary_request.segments}
        projection = segment_by_id.get(request.projection_segment_id)
        if projection is None:
            raise PromptInjectionBoundaryPolicyError("task rewrite projection segment is missing")
        if (
            projection.surface is not PromptBoundarySurface.TASK_REWRITE_DATA
            or projection.source_role is not PromptBoundarySourceRole.TASK_REWRITE_CONTRACT
            or projection.source_ref != request.rewrite_projection_ref
            or projection.content_sha256 != request.rewrite_projection_ref.object_sha256
        ):
            raise PromptInjectionBoundaryPolicyError("task rewrite projection is stale or mismatched")
        result = self.enforce(request.boundary_request)
        if result.accepted_segments != (projection,):
            raise PromptInjectionBoundaryPolicyError("task rewrite boundary accepts exactly one projection")
        return result

    def validate_prompt_only_dependency_data_boundary(
        self,
        request: PromptOnlyDependencyDataBoundaryRequest,
    ) -> PromptBoundaryEnforcementResult:
        segment_by_id = {segment.segment_id: segment for segment in request.boundary_request.segments}
        projection = segment_by_id.get(request.projection_segment_id)
        if projection is None:
            raise PromptInjectionBoundaryPolicyError("prompt-only dependency projection segment is missing")
        if (
            projection.surface is not PromptBoundarySurface.DEPENDENCY_DISCOVERY_DATA
            or projection.source_role is not PromptBoundarySourceRole.PROMPT_ONLY_DEPENDENCY_CONTRACT
            or projection.source_ref != request.discovery_projection_ref
            or projection.content_sha256 != request.discovery_projection_ref.object_sha256
        ):
            raise PromptInjectionBoundaryPolicyError(
                "prompt-only dependency projection is stale or mismatched"
            )
        result = self.enforce(request.boundary_request)
        if result.accepted_segments != (projection,):
            raise PromptInjectionBoundaryPolicyError(
                "prompt-only dependency boundary accepts exactly one projection"
            )
        return result

    @staticmethod
    def _validate_unique_segments(segments: tuple[PromptBoundarySegment, ...]) -> None:
        seen: set[str] = set()
        for segment in segments:
            if segment.segment_id in seen:
                raise PromptInjectionBoundaryPolicyError("duplicate prompt boundary segment id")
            seen.add(segment.segment_id)


def _segment_violation(
    segment: PromptBoundarySegment,
    approved_policy_refs: set[tuple[str, str, str, str]],
    approved_bundle_refs: set[tuple[str, str, str, str]],
) -> PromptBoundaryViolation | None:
    if segment.surface is PromptBoundarySurface.BLOCKED_METADATA:
        if segment.text_preview is not None:
            return _violation(segment, PromptBoundaryViolationReason.BLOCKED_METADATA_CONTAINS_TEXT)
        return None
    if segment.source_ref.object_type in _RAW_TRACE_OBJECT_TYPES:
        return _violation(segment, PromptBoundaryViolationReason.RAW_TRACE_SOURCE)
    if segment.source_role is PromptBoundarySourceRole.QUARANTINED_OR_PRIVATE_REF:
        return _violation(segment, PromptBoundaryViolationReason.QUARANTINE_OR_PRIVATE_SOURCE)
    if segment.source_role is PromptBoundarySourceRole.UNKNOWN:
        return _violation(segment, PromptBoundaryViolationReason.UNKNOWN_SOURCE_ROLE)
    if segment.surface is PromptBoundarySurface.SAFETY_REVIEW_DATA:
        if (
            segment.source_role is not PromptBoundarySourceRole.CANDIDATE_TASK_TEXT
            or segment.source_ref.object_type != "task-draft-visible-prompt"
            or segment.content_sha256 is None
            or segment.content_sha256 != segment.source_ref.object_sha256
        ):
            return _violation(
                segment,
                PromptBoundaryViolationReason.INVALID_CANDIDATE_TASK_BOUNDARY,
            )
    elif segment.source_role is PromptBoundarySourceRole.CANDIDATE_TASK_TEXT:
        return _violation(
            segment,
            PromptBoundaryViolationReason.INVALID_CANDIDATE_TASK_BOUNDARY,
        )
    if segment.surface is PromptBoundarySurface.RUBRIC_AUTHORING_DATA:
        if (
            segment.source_role is not PromptBoundarySourceRole.CANDIDATE_TASK_CONTRACT
            or segment.source_ref.object_type != "task-draft-rubric-input"
            or segment.content_sha256 is None
            or segment.content_sha256 != segment.source_ref.object_sha256
        ):
            return _violation(
                segment,
                PromptBoundaryViolationReason.INVALID_CANDIDATE_TASK_CONTRACT_BOUNDARY,
            )
    elif segment.source_role is PromptBoundarySourceRole.CANDIDATE_TASK_CONTRACT:
        return _violation(
            segment,
            PromptBoundaryViolationReason.INVALID_CANDIDATE_TASK_CONTRACT_BOUNDARY,
        )
    if segment.surface is PromptBoundarySurface.PRODUCER_TASK_DATA:
        if (
            segment.source_role is not PromptBoundarySourceRole.PROMPT_SAFETY_PASSED_TASK_CONTRACT
            or segment.source_ref.object_type != "producer-task-view-input"
            or segment.content_sha256 is None
            or segment.content_sha256 != segment.source_ref.object_sha256
        ):
            return _violation(
                segment,
                PromptBoundaryViolationReason.INVALID_PRODUCER_TASK_BOUNDARY,
            )
    elif segment.source_role is PromptBoundarySourceRole.PROMPT_SAFETY_PASSED_TASK_CONTRACT:
        return _violation(
            segment,
            PromptBoundaryViolationReason.INVALID_PRODUCER_TASK_BOUNDARY,
        )
    if segment.surface is PromptBoundarySurface.TASK_REWRITE_DATA:
        if (
            segment.source_role is not PromptBoundarySourceRole.TASK_REWRITE_CONTRACT
            or segment.source_ref.object_type != "task-rewrite-input"
            or segment.content_sha256 is None
            or segment.content_sha256 != segment.source_ref.object_sha256
        ):
            return _violation(
                segment,
                PromptBoundaryViolationReason.INVALID_TASK_REWRITE_BOUNDARY,
            )
    elif segment.source_role is PromptBoundarySourceRole.TASK_REWRITE_CONTRACT:
        return _violation(
            segment,
            PromptBoundaryViolationReason.INVALID_TASK_REWRITE_BOUNDARY,
        )
    if segment.surface is PromptBoundarySurface.DEPENDENCY_DISCOVERY_DATA:
        if (
            segment.source_role is not PromptBoundarySourceRole.PROMPT_ONLY_DEPENDENCY_CONTRACT
            or segment.source_ref.object_type != "prompt-only-dependency-input"
            or segment.content_sha256 is None
            or segment.content_sha256 != segment.source_ref.object_sha256
            or segment.text_preview is not None
        ):
            return _violation(
                segment,
                PromptBoundaryViolationReason.INVALID_PROMPT_ONLY_DEPENDENCY_BOUNDARY,
            )
    elif segment.source_role is PromptBoundarySourceRole.PROMPT_ONLY_DEPENDENCY_CONTRACT:
        return _violation(
            segment,
            PromptBoundaryViolationReason.INVALID_PROMPT_ONLY_DEPENDENCY_BOUNDARY,
        )
    if segment.surface in _CONTROL_SURFACES:
        if segment.source_role not in _STATIC_CONTROL_ROLES:
            return _violation(segment, PromptBoundaryViolationReason.CONTROL_SURFACE_SOURCE)
        if segment.untrusted_data_marker:
            return _violation(segment, PromptBoundaryViolationReason.STATIC_CONTROL_MARKED_UNTRUSTED)
        return None
    if segment.source_role in _MARKED_DATA_ROLES and not segment.untrusted_data_marker:
        return _violation(segment, PromptBoundaryViolationReason.MISSING_UNTRUSTED_DATA_MARKER)
    if segment.source_role in _ROLES_REQUIRING_DECISION and segment.decision is None:
        return _violation(segment, PromptBoundaryViolationReason.MISSING_PROVENANCE_DECISION)
    if segment.decision is not None:
        stale = _decision_stale(segment)
        if stale:
            return _violation(segment, PromptBoundaryViolationReason.STALE_OR_MISMATCHED_HASH)
        unsafe = _unsafe_decision_violation(segment)
        if unsafe is not None:
            return unsafe
    if segment.source_role is PromptBoundarySourceRole.SAFE_EVIDENCE_PROJECTION:
        if (
            segment.projection_policy_ref is None
            or _object_ref_key(segment.projection_policy_ref) not in approved_policy_refs
        ):
            return _violation(segment, PromptBoundaryViolationReason.MISSING_PROJECTION_POLICY_REF)
        if (
            segment.evidence_bundle_ref is None
            or _object_ref_key(segment.evidence_bundle_ref) not in approved_bundle_refs
        ):
            return _violation(segment, PromptBoundaryViolationReason.MISSING_EVIDENCE_BUNDLE_REF)
    if segment.source_role is PromptBoundarySourceRole.EVIDENCE_BUNDLE_REF:
        bundle_ref = segment.evidence_bundle_ref or segment.source_ref
        if _object_ref_key(bundle_ref) not in approved_bundle_refs:
            return _violation(segment, PromptBoundaryViolationReason.MISSING_EVIDENCE_BUNDLE_REF)
    return None


def _decision_stale(segment: PromptBoundarySegment) -> bool:
    decision = segment.decision
    if decision is None:
        return False
    if decision.subject_ref != segment.source_ref:
        return True
    if decision.subject_sha256 != segment.source_ref.object_sha256:
        return True
    return segment.content_sha256 is not None and segment.content_sha256 != decision.subject_sha256


def _unsafe_decision_violation(segment: PromptBoundarySegment) -> PromptBoundaryViolation | None:
    decision = segment.decision
    if decision is None:
        return None
    if ContentRiskLabel.PROMPT_INJECTION in decision.content_risk_labels:
        return _violation(segment, PromptBoundaryViolationReason.PROMPT_INJECTION_RISK)
    if TaintLabel.UNTRUSTED_INSTRUCTION_DERIVED in decision.taint_labels:
        return _violation(segment, PromptBoundaryViolationReason.UNTRUSTED_INSTRUCTION_TAINT)
    if decision.taint_labels & NON_WAIVABLE_TAINTS or decision.content_risk_labels & NON_WAIVABLE_RISKS:
        return _violation(segment, PromptBoundaryViolationReason.NON_WAIVABLE_RESTRICTION)
    if decision.disposition in {Disposition.NEEDS_REVIEW, Disposition.QUARANTINE, Disposition.REJECT}:
        return _violation(segment, PromptBoundaryViolationReason.DISPOSITION_DENIED)
    return None


def _violation(
    segment: PromptBoundarySegment,
    reason: PromptBoundaryViolationReason,
) -> PromptBoundaryViolation:
    return PromptBoundaryViolation(
        segment_id=segment.segment_id,
        source_ref=segment.source_ref,
        surface=segment.surface,
        source_role=segment.source_role,
        reason=reason,
    )


def _violation_message(violation: PromptBoundaryViolation) -> str:
    messages = {
        PromptBoundaryViolationReason.BLOCKED_METADATA_CONTAINS_TEXT: "blocked metadata cannot contain text",
        PromptBoundaryViolationReason.CONTROL_SURFACE_SOURCE: "trace-derived data cannot enter a control surface",
        PromptBoundaryViolationReason.DISPOSITION_DENIED: "provenance disposition is denied for prompt data",
        PromptBoundaryViolationReason.DUPLICATE_SEGMENT_ID: "duplicate prompt boundary segment id",
        PromptBoundaryViolationReason.INVALID_CANDIDATE_TASK_BOUNDARY: (
            "candidate task text requires an exact safety review data boundary "
            "and non-stale or mismatched hash"
        ),
        PromptBoundaryViolationReason.INVALID_CANDIDATE_TASK_CONTRACT_BOUNDARY: (
            "candidate task contract requires an exact rubric authoring data boundary "
            "and non-stale or mismatched hash"
        ),
        PromptBoundaryViolationReason.INVALID_PROMPT_ONLY_DEPENDENCY_BOUNDARY: (
            "prompt-only dependency contract requires an exact dependency discovery data boundary"
        ),
        PromptBoundaryViolationReason.INVALID_PRODUCER_TASK_BOUNDARY: (
            "producer task contract requires exact producer task data and stale or mismatched "
            "bindings are denied"
        ),
        PromptBoundaryViolationReason.INVALID_TASK_REWRITE_BOUNDARY: (
            "task rewrite contract requires an exact untrusted data boundary"
        ),
        PromptBoundaryViolationReason.MISSING_EVIDENCE_BUNDLE_REF: "safe evidence bundle ref is missing or unapproved",
        PromptBoundaryViolationReason.MISSING_PRODUCER_TASK_SEGMENT: "producer task view segment is missing",
        PromptBoundaryViolationReason.MISSING_PROJECTION_POLICY_REF: "projection policy ref is missing or unapproved",
        PromptBoundaryViolationReason.MISSING_PROVENANCE_DECISION: "provenance decision is required for this segment",
        PromptBoundaryViolationReason.MISSING_UNTRUSTED_DATA_MARKER: "untrusted data marker is required",
        PromptBoundaryViolationReason.NON_WAIVABLE_RESTRICTION: "non-waivable restriction cannot enter prompt data",
        PromptBoundaryViolationReason.PROMPT_INJECTION_RISK: "prompt injection risk cannot enter prompt data",
        PromptBoundaryViolationReason.QUARANTINE_OR_PRIVATE_SOURCE: "quarantine or private source cannot enter prompt data",
        PromptBoundaryViolationReason.RAW_TRACE_SOURCE: "raw trace source cannot enter prompt data",
        PromptBoundaryViolationReason.STALE_OR_MISMATCHED_HASH: "stale or mismatched hash binding",
        PromptBoundaryViolationReason.STATIC_CONTROL_MARKED_UNTRUSTED: "static control segment cannot be marked as untrusted data",
        PromptBoundaryViolationReason.UNTRUSTED_INSTRUCTION_TAINT: "untrusted instruction taint cannot enter prompt data",
        PromptBoundaryViolationReason.UNKNOWN_SOURCE_ROLE: "unknown source role cannot enter prompt data",
    }
    return messages[violation.reason]


def _segment_facts(segment: PromptBoundarySegment) -> dict[str, object]:
    return {
        "segment_id": segment.segment_id,
        "surface": segment.surface.value,
        "source_role": segment.source_role.value,
        "source_ref": segment.source_ref.model_dump(mode="json", exclude_none=False),
        "decision": _decision_facts(segment.decision),
        "projection_policy_ref": _maybe_ref(segment.projection_policy_ref),
        "evidence_bundle_ref": _maybe_ref(segment.evidence_bundle_ref),
        "content_sha256": segment.content_sha256,
        "untrusted_data_marker": segment.untrusted_data_marker,
        "text_preview_sha256": _text_preview_sha(segment.text_preview),
    }


def _violation_facts(violation: PromptBoundaryViolation) -> dict[str, object]:
    return {
        "segment_id": violation.segment_id,
        "source_ref": violation.source_ref.model_dump(mode="json", exclude_none=False),
        "surface": violation.surface.value,
        "source_role": violation.source_role.value,
        "reason": violation.reason.value,
        "detail": violation.detail,
    }


def _decision_facts(decision: ProvenanceDecision | None) -> dict[str, object] | None:
    if decision is None:
        return None
    return {
        "provenance_decision_id": decision.provenance_decision_id,
        "subject_ref": decision.subject_ref.model_dump(mode="json", exclude_none=False),
        "origin_class": decision.origin_class.value,
        "taint_labels": sorted(item.value for item in decision.taint_labels),
        "content_risk_labels": sorted(item.value for item in decision.content_risk_labels),
        "visibility": decision.visibility.value,
        "disposition": decision.disposition.value,
        "derived_from": [item.model_dump(mode="json", exclude_none=False) for item in decision.derived_from],
        "rule_ids": list(decision.rule_ids),
        "source_event_refs": [
            item.model_dump(mode="json", exclude_none=False) for item in decision.source_event_refs
        ],
        "confidence": decision.confidence,
        "review_required": decision.review_required,
        "policy_version": decision.policy_version,
        "subject_sha256": decision.subject_sha256,
    }


def _maybe_ref(value: ObjectRef | None) -> dict[str, object] | None:
    if value is None:
        return None
    return value.model_dump(mode="json", exclude_none=False)


def _text_preview_sha(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode()).hexdigest()


def _object_ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (value.object_type, value.object_id, value.object_version, value.object_sha256)


def _require_v2_ref(
    ref: ObjectRef,
    expected: str,
    field_name: str,
) -> None:
    if ref.object_type != expected or ref.object_version != "v2":
        raise ValueError(f"{field_name} must reference {expected} v2")


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(namespace: str, payload: object) -> str:
    return f"{namespace}://{_stable_hash(payload)}"
