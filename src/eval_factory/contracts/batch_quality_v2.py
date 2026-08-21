from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.quality import Severity
from eval_factory.contracts.task_v2 import PromptLeakageCategoryV2

BATCH_QUALITY_POLICY_VERSION: Literal["batch-quality/r7-03-v1"] = "batch-quality/r7-03-v1"


class LineageAuditCheckCodeV2(StrEnum):
    SOURCE_TRACE_LABELS = "SOURCE_TRACE_LABELS"
    LABEL_SELECTION = "LABEL_SELECTION"
    SELECTION_TASK = "SELECTION_TASK"
    TASK_R4 = "TASK_R4"
    R4_ITEM_QUALITY = "R4_ITEM_QUALITY"
    PACKAGE_PROVENANCE_SET = "PACKAGE_PROVENANCE_SET"
    PACKAGE_ENVIRONMENT_SET = "PACKAGE_ENVIRONMENT_SET"
    PACKAGE_MEMBER_DERIVATION = "PACKAGE_MEMBER_DERIVATION"
    CROSS_ITEM_LINEAGE_ALIAS = "CROSS_ITEM_LINEAGE_ALIAS"


BATCH_QUALITY_ITEM_LINEAGE_CHECKS: tuple[
    LineageAuditCheckCodeV2,
    ...,
] = (
    LineageAuditCheckCodeV2.SOURCE_TRACE_LABELS,
    LineageAuditCheckCodeV2.LABEL_SELECTION,
    LineageAuditCheckCodeV2.SELECTION_TASK,
    LineageAuditCheckCodeV2.TASK_R4,
    LineageAuditCheckCodeV2.R4_ITEM_QUALITY,
    LineageAuditCheckCodeV2.PACKAGE_PROVENANCE_SET,
    LineageAuditCheckCodeV2.PACKAGE_ENVIRONMENT_SET,
    LineageAuditCheckCodeV2.PACKAGE_MEMBER_DERIVATION,
)
BATCH_QUALITY_BATCH_LINEAGE_CHECKS: tuple[
    LineageAuditCheckCodeV2,
    ...,
] = (LineageAuditCheckCodeV2.CROSS_ITEM_LINEAGE_ALIAS,)


class LineageAuditCheckOutcomeV2(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    INDETERMINATE = "INDETERMINATE"


class ItemLineageAuditOutcomeV2(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    INDETERMINATE = "INDETERMINATE"


class BatchFindingScopeV2(StrEnum):
    ITEM = "ITEM"
    BATCH = "BATCH"


class BatchFindingCategoryV2(StrEnum):
    EXACT_TASK_DUPLICATE = "EXACT_TASK_DUPLICATE"
    NEAR_TASK_DUPLICATE = "NEAR_TASK_DUPLICATE"
    EXACT_ATTACHMENT_DUPLICATE = "EXACT_ATTACHMENT_DUPLICATE"
    NEAR_ATTACHMENT_DUPLICATE = "NEAR_ATTACHMENT_DUPLICATE"
    CROSS_ITEM_CONTAMINATION = "CROSS_ITEM_CONTAMINATION"
    CROSS_ITEM_LEAKAGE = "CROSS_ITEM_LEAKAGE"
    ANSWER_REUSE = "ANSWER_REUSE"
    LINEAGE_INCOMPLETE = "LINEAGE_INCOMPLETE"
    LINEAGE_INDETERMINATE = "LINEAGE_INDETERMINATE"
    CROSS_ITEM_LINEAGE_ALIAS = "CROSS_ITEM_LINEAGE_ALIAS"


_ANSWER_BEARING_CATEGORIES = frozenset(
    {
        PromptLeakageCategoryV2.FINAL_ANSWER,
        PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
        PromptLeakageCategoryV2.PRIVATE_REFERENCE,
    }
)
_CONTROL_LEAKAGE_CATEGORIES = frozenset(PromptLeakageCategoryV2) - (_ANSWER_BEARING_CATEGORIES)


class BatchQualityPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/batch-quality-policy/v2"] = "eval-factory/batch-quality-policy/v2"
    batch_quality_policy_id: Identifier
    max_items: int = Field(ge=2, le=100_000)
    max_label_decisions: int = Field(ge=2, le=10_000_000)
    max_package_members: int = Field(ge=0, le=10_000_000)
    max_lineage_checks: int = Field(ge=1, le=10_000_000)
    max_direct_evidence_refs: int = Field(ge=0, le=10_000_000)
    max_findings: int = Field(ge=0, le=10_000_000)
    max_finding_subject_refs: int = Field(ge=0, le=100_000_000)
    max_report_revisions: int = Field(ge=0, le=100_000)
    max_blocked_items: int = Field(ge=0, le=100_000)
    required_item_lineage_checks: tuple[LineageAuditCheckCodeV2, ...] = BATCH_QUALITY_ITEM_LINEAGE_CHECKS
    required_batch_lineage_checks: tuple[LineageAuditCheckCodeV2, ...] = BATCH_QUALITY_BATCH_LINEAGE_CHECKS
    ownership_mode: Literal["TARGET_ITEM_RELATION_BATCH"] = "TARGET_ITEM_RELATION_BATCH"
    finding_mode: Literal["DIRECT_EVIDENCE_ONLY"] = "DIRECT_EVIDENCE_ONLY"
    exact_task_duplicate_severity: Literal[Severity.P1] = Severity.P1
    other_duplicate_severity: Literal[Severity.P2] = Severity.P2
    answer_bearing_safety_severity: Literal[Severity.P0] = Severity.P0
    other_safety_severity: Literal[Severity.P1] = Severity.P1
    lineage_severity: Literal[Severity.P1] = Severity.P1
    policy_version: Literal["batch-quality/r7-03-v1"] = BATCH_QUALITY_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @field_validator(
        "required_item_lineage_checks",
        "required_batch_lineage_checks",
        mode="before",
    )
    @classmethod
    def parse_check_codes(
        cls,
        value: object,
    ) -> tuple[LineageAuditCheckCodeV2, ...]:
        if not isinstance(value, (tuple, list)):
            raise TypeError("required lineage checks must be a collection")
        return tuple(
            item if isinstance(item, LineageAuditCheckCodeV2) else LineageAuditCheckCodeV2(item)
            for item in value
        )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.required_item_lineage_checks != (BATCH_QUALITY_ITEM_LINEAGE_CHECKS):
            raise ValueError("required Item lineage checks are not canonical")
        if self.required_batch_lineage_checks != (BATCH_QUALITY_BATCH_LINEAGE_CHECKS):
            raise ValueError("required batch lineage checks are not canonical")
        if self.max_report_revisions > self.max_items:
            raise ValueError("report revision limit cannot exceed Item limit")
        if self.max_blocked_items > self.max_items:
            raise ValueError("blocked Item limit cannot exceed Item limit")
        _validate_audit(self.audit, (), "batch quality policy")
        _validate_identity(
            object_id=self.batch_quality_policy_id,
            object_sha256=self.policy_sha256,
            expected_prefix="batch-quality-policy",
            observed=batch_quality_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        max_items: int,
        max_label_decisions: int,
        max_package_members: int,
        max_lineage_checks: int,
        max_direct_evidence_refs: int,
        max_findings: int,
        max_finding_subject_refs: int,
        max_report_revisions: int,
        max_blocked_items: int,
        audit: ContractAudit,
    ) -> BatchQualityPolicyV2:
        value = cls(
            batch_quality_policy_id="batch-quality-policy://pending",
            max_items=max_items,
            max_label_decisions=max_label_decisions,
            max_package_members=max_package_members,
            max_lineage_checks=max_lineage_checks,
            max_direct_evidence_refs=max_direct_evidence_refs,
            max_findings=max_findings,
            max_finding_subject_refs=max_finding_subject_refs,
            max_report_revisions=max_report_revisions,
            max_blocked_items=max_blocked_items,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        digest = batch_quality_policy_v2_carried_sha256(value)
        return _finalize(
            value,
            id_field="batch_quality_policy_id",
            hash_field="policy_sha256",
            prefix="batch-quality-policy",
            digest=digest,
        )


class LineageAuditCheckV2(ContractModelV2):
    schema_version: Literal["eval-factory/lineage-audit-check/v2"] = "eval-factory/lineage-audit-check/v2"
    lineage_audit_check_id: Identifier
    code: LineageAuditCheckCodeV2
    outcome: LineageAuditCheckOutcomeV2
    item_ids: tuple[Identifier, ...] = Field(min_length=1)
    subject_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    evidence_refs: tuple[ObjectRef, ...] = ()
    evaluated_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    indeterminate_count: int = Field(ge=0)
    policy_ref: ObjectRef
    check_sha256: Sha256
    audit: ContractAudit

    @field_validator("code", mode="before")
    @classmethod
    def parse_code(cls, value: object) -> LineageAuditCheckCodeV2:
        return _parse_enum(value, LineageAuditCheckCodeV2, "code")

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> LineageAuditCheckOutcomeV2:
        return _parse_enum(value, LineageAuditCheckOutcomeV2, "outcome")

    @model_validator(mode="after")
    def validate_check(self) -> Self:
        _require_sorted_unique("lineage check Item IDs", self.item_ids)
        _require_sorted_unique_refs(
            "lineage check subject refs",
            self.subject_refs,
        )
        _require_sorted_unique_refs(
            "lineage check evidence refs",
            self.evidence_refs,
        )
        _require_ref(
            self.policy_ref,
            "batch-quality-policy",
            "v2",
            "policy_ref",
        )
        if self.code is LineageAuditCheckCodeV2.CROSS_ITEM_LINEAGE_ALIAS:
            if len(self.item_ids) < 2:
                raise ValueError("cross-Item lineage alias check requires multiple Items")
        elif len(self.item_ids) != 1:
            raise ValueError("Item lineage check requires exactly one Item")
        if self.outcome is LineageAuditCheckOutcomeV2.PASSED:
            if (
                self.evaluated_count < 1
                or self.failed_count
                or self.indeterminate_count
                or not self.evidence_refs
            ):
                raise ValueError("PASSED lineage check requires passing evidence")
        elif self.outcome is LineageAuditCheckOutcomeV2.FAILED:
            if (
                self.failed_count < 1
                or self.evaluated_count < self.failed_count
                or self.indeterminate_count
                or not self.evidence_refs
            ):
                raise ValueError("FAILED lineage check requires failed evidence")
        elif self.outcome is LineageAuditCheckOutcomeV2.INDETERMINATE:
            if (
                self.indeterminate_count < 1
                or self.evaluated_count < self.indeterminate_count
                or self.failed_count
                or not self.evidence_refs
            ):
                raise ValueError("INDETERMINATE lineage check requires indeterminate evidence")
        elif self.evaluated_count or self.failed_count or self.indeterminate_count or self.evidence_refs:
            raise ValueError("SKIPPED lineage check cannot claim evidence")
        _validate_audit(
            self.audit,
            (*self.subject_refs, *self.evidence_refs, self.policy_ref),
            "lineage audit check",
        )
        _validate_identity(
            object_id=self.lineage_audit_check_id,
            object_sha256=self.check_sha256,
            expected_prefix="lineage-audit-check",
            observed=lineage_audit_check_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        code: LineageAuditCheckCodeV2,
        outcome: LineageAuditCheckOutcomeV2,
        item_ids: tuple[str, ...],
        subject_refs: tuple[ObjectRef, ...],
        evidence_refs: tuple[ObjectRef, ...],
        evaluated_count: int,
        failed_count: int,
        indeterminate_count: int,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> LineageAuditCheckV2:
        value = cls(
            lineage_audit_check_id="lineage-audit-check://pending",
            code=code,
            outcome=outcome,
            item_ids=tuple(sorted(set(item_ids))),
            subject_refs=_sorted_refs(subject_refs),
            evidence_refs=_sorted_refs(evidence_refs),
            evaluated_count=evaluated_count,
            failed_count=failed_count,
            indeterminate_count=indeterminate_count,
            policy_ref=policy_ref,
            check_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (*subject_refs, *evidence_refs, policy_ref),
            ),
        )
        digest = lineage_audit_check_v2_carried_sha256(value)
        return _finalize(
            value,
            id_field="lineage_audit_check_id",
            hash_field="check_sha256",
            prefix="lineage-audit-check",
            digest=digest,
        )


class ItemLineageAuditResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/item-lineage-audit-result/v2"] = (
        "eval-factory/item-lineage-audit-result/v2"
    )
    item_lineage_audit_result_id: Identifier
    item_id: Identifier
    source_trace_ref: ObjectRef
    trace_envelope_ref: ObjectRef
    label_decision_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    selection_context_ref: ObjectRef
    task_draft_ref: ObjectRef
    r4_task_contract_set_ref: ObjectRef
    item_quality_result_ref: ObjectRef
    base_quality_report_ref: ObjectRef
    final_package_manifest_ref: ObjectRef
    provenance_manifest_ref: ObjectRef
    environment_spec_ref: ObjectRef
    checks: tuple[LineageAuditCheckV2, ...]
    check_refs: tuple[ObjectRef, ...]
    outcome: ItemLineageAuditOutcomeV2
    passed_check_count: int = Field(ge=0)
    failed_check_count: int = Field(ge=0)
    skipped_check_count: int = Field(ge=0)
    indeterminate_check_count: int = Field(ge=0)
    policy_ref: ObjectRef
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> ItemLineageAuditOutcomeV2:
        return _parse_enum(value, ItemLineageAuditOutcomeV2, "outcome")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref_type(
            self.source_trace_ref,
            "trace-source",
            "source_trace_ref",
        )
        _require_ref_type(
            self.trace_envelope_ref,
            "trace-envelope",
            "trace_envelope_ref",
        )
        _require_sorted_unique_refs(
            "lineage label decision refs",
            self.label_decision_refs,
        )
        for ref in self.label_decision_refs:
            _require_ref_type(ref, "label-decision", "label_decision_refs")
        for ref, object_type, field_name in (
            (
                self.selection_context_ref,
                "selection-context",
                "selection_context_ref",
            ),
            (self.task_draft_ref, "task-draft", "task_draft_ref"),
            (
                self.r4_task_contract_set_ref,
                "r4-task-contract-set",
                "r4_task_contract_set_ref",
            ),
            (
                self.item_quality_result_ref,
                "item-quality-compilation-result",
                "item_quality_result_ref",
            ),
            (
                self.base_quality_report_ref,
                "quality-report",
                "base_quality_report_ref",
            ),
            (
                self.final_package_manifest_ref,
                "final-package-manifest",
                "final_package_manifest_ref",
            ),
            (
                self.provenance_manifest_ref,
                "provenance-manifest",
                "provenance_manifest_ref",
            ),
            (
                self.environment_spec_ref,
                "environment-spec",
                "environment_spec_ref",
            ),
            (self.policy_ref, "batch-quality-policy", "policy_ref"),
        ):
            _require_ref(ref, object_type, "v2", field_name)
        if tuple(value.code for value in self.checks) != (BATCH_QUALITY_ITEM_LINEAGE_CHECKS):
            raise ValueError("Item lineage check inventory is not exact")
        expected_refs = tuple(lineage_audit_check_v2_ref(value) for value in self.checks)
        if self.check_refs != expected_refs:
            raise ValueError("Item lineage check refs do not match nested checks")
        if len(self.check_refs) != len(set(self.check_refs)):
            raise ValueError("Item lineage check refs must be unique")
        if any(
            value.item_ids != (self.item_id,) or value.policy_ref != self.policy_ref for value in self.checks
        ):
            raise ValueError("Item lineage check ownership is invalid")
        counts = _lineage_outcome_counts(self.checks)
        if counts != (
            self.passed_check_count,
            self.failed_check_count,
            self.skipped_check_count,
            self.indeterminate_check_count,
        ):
            raise ValueError("Item lineage check counts are not exact")
        if self.outcome is not _item_lineage_outcome(self.checks):
            raise ValueError("Item lineage outcome is not derived")
        refs = (
            self.source_trace_ref,
            self.trace_envelope_ref,
            *self.label_decision_refs,
            self.selection_context_ref,
            self.task_draft_ref,
            self.r4_task_contract_set_ref,
            self.item_quality_result_ref,
            self.base_quality_report_ref,
            self.final_package_manifest_ref,
            self.provenance_manifest_ref,
            self.environment_spec_ref,
            *self.check_refs,
            self.policy_ref,
        )
        _validate_audit(self.audit, refs, "Item lineage audit result")
        _validate_identity(
            object_id=self.item_lineage_audit_result_id,
            object_sha256=self.result_sha256,
            expected_prefix="item-lineage-audit-result",
            observed=item_lineage_audit_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        item_id: str,
        source_trace_ref: ObjectRef,
        trace_envelope_ref: ObjectRef,
        label_decision_refs: tuple[ObjectRef, ...],
        selection_context_ref: ObjectRef,
        task_draft_ref: ObjectRef,
        r4_task_contract_set_ref: ObjectRef,
        item_quality_result_ref: ObjectRef,
        base_quality_report_ref: ObjectRef,
        final_package_manifest_ref: ObjectRef,
        provenance_manifest_ref: ObjectRef,
        environment_spec_ref: ObjectRef,
        checks: tuple[LineageAuditCheckV2, ...],
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ItemLineageAuditResultV2:
        by_code = {value.code: value for value in checks}
        ordered = tuple(by_code[code] for code in BATCH_QUALITY_ITEM_LINEAGE_CHECKS if code in by_code)
        check_refs = tuple(lineage_audit_check_v2_ref(value) for value in ordered)
        counts = _lineage_outcome_counts(ordered)
        refs = (
            source_trace_ref,
            trace_envelope_ref,
            *label_decision_refs,
            selection_context_ref,
            task_draft_ref,
            r4_task_contract_set_ref,
            item_quality_result_ref,
            base_quality_report_ref,
            final_package_manifest_ref,
            provenance_manifest_ref,
            environment_spec_ref,
            *check_refs,
            policy_ref,
        )
        value = cls(
            item_lineage_audit_result_id=("item-lineage-audit-result://pending"),
            item_id=item_id,
            source_trace_ref=source_trace_ref,
            trace_envelope_ref=trace_envelope_ref,
            label_decision_refs=_sorted_refs(label_decision_refs),
            selection_context_ref=selection_context_ref,
            task_draft_ref=task_draft_ref,
            r4_task_contract_set_ref=r4_task_contract_set_ref,
            item_quality_result_ref=item_quality_result_ref,
            base_quality_report_ref=base_quality_report_ref,
            final_package_manifest_ref=final_package_manifest_ref,
            provenance_manifest_ref=provenance_manifest_ref,
            environment_spec_ref=environment_spec_ref,
            checks=ordered,
            check_refs=check_refs,
            outcome=_item_lineage_outcome(ordered),
            passed_check_count=counts[0],
            failed_check_count=counts[1],
            skipped_check_count=counts[2],
            indeterminate_check_count=counts[3],
            policy_ref=policy_ref,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        digest = item_lineage_audit_result_v2_carried_sha256(value)
        return _finalize(
            value,
            id_field="item_lineage_audit_result_id",
            hash_field="result_sha256",
            prefix="item-lineage-audit-result",
            digest=digest,
        )


class BatchReviewerFindingV2(ContractModelV2):
    schema_version: Literal["eval-factory/batch-reviewer-finding/v2"] = (
        "eval-factory/batch-reviewer-finding/v2"
    )
    batch_reviewer_finding_id: Identifier
    scope: BatchFindingScopeV2
    owner_item_id: Identifier | None = None
    category: BatchFindingCategoryV2
    restricted_category: PromptLeakageCategoryV2 | None = None
    severity: Severity
    status: Literal["OPEN"] = "OPEN"
    affected_item_ids: tuple[Identifier, ...] = Field(min_length=1)
    blocked_item_ids: tuple[Identifier, ...] = ()
    subject_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    direct_evidence_ref: ObjectRef
    non_waivable: bool
    release_blocking: bool
    policy_ref: ObjectRef
    finding_sha256: Sha256
    audit: ContractAudit

    @field_validator("scope", mode="before")
    @classmethod
    def parse_scope(cls, value: object) -> BatchFindingScopeV2:
        return _parse_enum(value, BatchFindingScopeV2, "scope")

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(cls, value: object) -> BatchFindingCategoryV2:
        return _parse_enum(value, BatchFindingCategoryV2, "category")

    @field_validator("restricted_category", mode="before")
    @classmethod
    def parse_restricted_category(
        cls,
        value: object,
    ) -> PromptLeakageCategoryV2 | None:
        if value is None:
            return None
        return _parse_enum(
            value,
            PromptLeakageCategoryV2,
            "restricted_category",
        )

    @field_validator("severity", mode="before")
    @classmethod
    def parse_severity(cls, value: object) -> Severity:
        return _parse_enum(value, Severity, "severity")

    @model_validator(mode="after")
    def validate_finding(self) -> Self:
        _require_sorted_unique(
            "finding affected Item IDs",
            self.affected_item_ids,
        )
        _require_sorted_unique(
            "finding blocked Item IDs",
            self.blocked_item_ids,
        )
        if not set(self.blocked_item_ids).issubset(self.affected_item_ids):
            raise ValueError("blocked Items must be affected Items")
        _require_sorted_unique_refs("finding subject refs", self.subject_refs)
        _require_ref(
            self.policy_ref,
            "batch-quality-policy",
            "v2",
            "policy_ref",
        )
        expected = _finding_policy(
            self.category,
            restricted_category=self.restricted_category,
        )
        if (
            self.scope is not expected[0]
            or self.severity is not expected[1]
            or self.non_waivable is not expected[2]
            or self.release_blocking is not expected[3]
        ):
            raise ValueError("batch finding policy is not exact")
        if self.scope is BatchFindingScopeV2.ITEM:
            if self.owner_item_id is None or self.affected_item_ids != (self.owner_item_id,):
                raise ValueError("ITEM finding requires one exact owner and affected Item")
        elif self.owner_item_id is not None or len(self.affected_item_ids) < 2:
            raise ValueError("BATCH finding requires multiple affected Items and no owner")
        expected_blocked = self.affected_item_ids if self.release_blocking else ()
        if self.blocked_item_ids != expected_blocked:
            raise ValueError("finding blocked Items are not policy-derived")
        _require_finding_evidence_type(
            self.category,
            self.direct_evidence_ref,
        )
        _validate_audit(
            self.audit,
            (
                *self.subject_refs,
                self.direct_evidence_ref,
                self.policy_ref,
            ),
            "batch reviewer finding",
        )
        _validate_identity(
            object_id=self.batch_reviewer_finding_id,
            object_sha256=self.finding_sha256,
            expected_prefix="batch-reviewer-finding",
            observed=batch_reviewer_finding_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        category: BatchFindingCategoryV2,
        owner_item_id: str | None,
        affected_item_ids: tuple[str, ...],
        subject_refs: tuple[ObjectRef, ...],
        direct_evidence_ref: ObjectRef,
        restricted_category: PromptLeakageCategoryV2 | None,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> BatchReviewerFindingV2:
        scope, severity, non_waivable, release_blocking = _finding_policy(
            category,
            restricted_category=restricted_category,
        )
        affected = tuple(sorted(set(affected_item_ids)))
        value = cls(
            batch_reviewer_finding_id="batch-reviewer-finding://pending",
            scope=scope,
            owner_item_id=owner_item_id,
            category=category,
            restricted_category=restricted_category,
            severity=severity,
            affected_item_ids=affected,
            blocked_item_ids=affected if release_blocking else (),
            subject_refs=_sorted_refs(subject_refs),
            direct_evidence_ref=direct_evidence_ref,
            non_waivable=non_waivable,
            release_blocking=release_blocking,
            policy_ref=policy_ref,
            finding_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (*subject_refs, direct_evidence_ref, policy_ref),
            ),
        )
        digest = batch_reviewer_finding_v2_carried_sha256(value)
        return _finalize(
            value,
            id_field="batch_reviewer_finding_id",
            hash_field="finding_sha256",
            prefix="batch-reviewer-finding",
            digest=digest,
        )


class ItemQualityReportRevisionV2(ContractModelV2):
    schema_version: Literal["eval-factory/item-quality-report-revision/v2"] = (
        "eval-factory/item-quality-report-revision/v2"
    )
    item_quality_report_revision_id: Identifier
    item_id: Identifier
    revision: Literal[1] = 1
    base_item_quality_result_ref: ObjectRef
    base_quality_report_ref: ObjectRef
    final_package_manifest_ref: ObjectRef
    provenance_manifest_ref: ObjectRef
    environment_spec_ref: ObjectRef
    accepted_artifact_refs: tuple[ObjectRef, ...]
    package_sha256: Sha256
    findings: tuple[BatchReviewerFindingV2, ...] = Field(min_length=1)
    finding_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    open_p0_count: int = Field(ge=0)
    open_p1_count: int = Field(ge=0)
    unresolved_non_waivable_count: int = Field(ge=0)
    approvable: Literal[False] = False
    release_revalidation_required: Literal[True] = True
    policy_ref: ObjectRef
    revision_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_revision(self) -> Self:
        for ref, object_type, field_name in (
            (
                self.base_item_quality_result_ref,
                "item-quality-compilation-result",
                "base_item_quality_result_ref",
            ),
            (
                self.base_quality_report_ref,
                "quality-report",
                "base_quality_report_ref",
            ),
            (
                self.final_package_manifest_ref,
                "final-package-manifest",
                "final_package_manifest_ref",
            ),
            (
                self.provenance_manifest_ref,
                "provenance-manifest",
                "provenance_manifest_ref",
            ),
            (
                self.environment_spec_ref,
                "environment-spec",
                "environment_spec_ref",
            ),
            (self.policy_ref, "batch-quality-policy", "policy_ref"),
        ):
            _require_ref(ref, object_type, "v2", field_name)
        _require_sorted_unique_refs(
            "revision accepted artifact refs",
            self.accepted_artifact_refs,
        )
        for ref in self.accepted_artifact_refs:
            _require_ref(
                ref,
                "candidate-artifact-version",
                "v2",
                "accepted_artifact_refs",
            )
        expected_refs = tuple(batch_reviewer_finding_v2_ref(value) for value in self.findings)
        if self.finding_refs != expected_refs:
            raise ValueError("revision finding refs do not match nested findings")
        if len(self.finding_refs) != len(set(self.finding_refs)):
            raise ValueError("revision finding refs must be unique")
        if any(
            value.scope is not BatchFindingScopeV2.ITEM
            or value.owner_item_id != self.item_id
            or value.affected_item_ids != (self.item_id,)
            or value.policy_ref != self.policy_ref
            for value in self.findings
        ):
            raise ValueError("revision contains a finding owned elsewhere")
        counts = _finding_blocker_counts(self.findings)
        if counts != (
            self.open_p0_count,
            self.open_p1_count,
            self.unresolved_non_waivable_count,
        ):
            raise ValueError("revision blocker counts are not exact")
        if not any(value.release_blocking for value in self.findings):
            raise ValueError("revision requires one release-blocking finding")
        refs = (
            self.base_item_quality_result_ref,
            self.base_quality_report_ref,
            self.final_package_manifest_ref,
            self.provenance_manifest_ref,
            self.environment_spec_ref,
            *self.accepted_artifact_refs,
            *self.finding_refs,
            self.policy_ref,
        )
        _validate_audit(self.audit, refs, "Item quality report revision")
        _validate_identity(
            object_id=self.item_quality_report_revision_id,
            object_sha256=self.revision_sha256,
            expected_prefix="item-quality-report-revision",
            observed=item_quality_report_revision_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        item_id: str,
        base_item_quality_result_ref: ObjectRef,
        base_quality_report_ref: ObjectRef,
        final_package_manifest_ref: ObjectRef,
        provenance_manifest_ref: ObjectRef,
        environment_spec_ref: ObjectRef,
        accepted_artifact_refs: tuple[ObjectRef, ...],
        package_sha256: str,
        findings: tuple[BatchReviewerFindingV2, ...],
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ItemQualityReportRevisionV2:
        ordered = _sorted_nested(
            findings,
            batch_reviewer_finding_v2_ref,
        )
        finding_refs = tuple(batch_reviewer_finding_v2_ref(value) for value in ordered)
        counts = _finding_blocker_counts(ordered)
        refs = (
            base_item_quality_result_ref,
            base_quality_report_ref,
            final_package_manifest_ref,
            provenance_manifest_ref,
            environment_spec_ref,
            *accepted_artifact_refs,
            *finding_refs,
            policy_ref,
        )
        value = cls(
            item_quality_report_revision_id=("item-quality-report-revision://pending"),
            item_id=item_id,
            base_item_quality_result_ref=base_item_quality_result_ref,
            base_quality_report_ref=base_quality_report_ref,
            final_package_manifest_ref=final_package_manifest_ref,
            provenance_manifest_ref=provenance_manifest_ref,
            environment_spec_ref=environment_spec_ref,
            accepted_artifact_refs=_sorted_refs(accepted_artifact_refs),
            package_sha256=package_sha256,
            findings=ordered,
            finding_refs=finding_refs,
            open_p0_count=counts[0],
            open_p1_count=counts[1],
            unresolved_non_waivable_count=counts[2],
            policy_ref=policy_ref,
            revision_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        digest = item_quality_report_revision_v2_carried_sha256(value)
        return _finalize(
            value,
            id_field="item_quality_report_revision_id",
            hash_field="revision_sha256",
            prefix="item-quality-report-revision",
            digest=digest,
        )


class BatchQualityReportV2(ContractModelV2):
    schema_version: Literal["eval-factory/batch-quality-report/v2"] = "eval-factory/batch-quality-report/v2"
    batch_quality_report_id: Identifier
    batch_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    policy_ref: ObjectRef
    duplicate_policy_ref: ObjectRef
    duplicate_result_ref: ObjectRef
    cross_item_policy_ref: ObjectRef
    cross_item_result_ref: ObjectRef
    item_ids: tuple[Identifier, ...] = Field(min_length=2)
    item_quality_result_refs: tuple[ObjectRef, ...] = Field(min_length=2)
    base_item_quality_report_refs: tuple[ObjectRef, ...] = Field(min_length=2)
    current_item_quality_report_refs: tuple[ObjectRef, ...] = Field(min_length=2)
    lineage_audits: tuple[ItemLineageAuditResultV2, ...]
    lineage_audit_refs: tuple[ObjectRef, ...]
    batch_lineage_checks: tuple[LineageAuditCheckV2, ...] = ()
    batch_lineage_check_refs: tuple[ObjectRef, ...] = ()
    findings: tuple[BatchReviewerFindingV2, ...] = ()
    finding_refs: tuple[ObjectRef, ...] = ()
    item_quality_report_revisions: tuple[
        ItemQualityReportRevisionV2,
        ...,
    ] = ()
    item_quality_report_revision_refs: tuple[ObjectRef, ...] = ()
    duplicate_pair_refs: tuple[ObjectRef, ...] = ()
    duplicate_cluster_refs: tuple[ObjectRef, ...] = ()
    visible_match_refs: tuple[ObjectRef, ...] = ()
    answer_reuse_pair_refs: tuple[ObjectRef, ...] = ()
    safety_cluster_refs: tuple[ObjectRef, ...] = ()
    item_scoped_finding_refs: tuple[ObjectRef, ...] = ()
    batch_scoped_finding_refs: tuple[ObjectRef, ...] = ()
    open_finding_refs: tuple[ObjectRef, ...] = ()
    affected_item_ids: tuple[Identifier, ...] = ()
    blocked_item_ids: tuple[Identifier, ...] = ()
    invalidated_result_refs: tuple[ObjectRef, ...] = ()
    lineage_audit_count: int = Field(ge=0)
    lineage_check_count: int = Field(ge=0)
    finding_count: int = Field(ge=0)
    item_scoped_finding_count: int = Field(ge=0)
    batch_scoped_finding_count: int = Field(ge=0)
    open_p0_count: int = Field(ge=0)
    open_p1_count: int = Field(ge=0)
    open_p2_count: int = Field(ge=0)
    unresolved_non_waivable_count: int = Field(ge=0)
    report_revision_count: int = Field(ge=0)
    affected_item_count: int = Field(ge=0)
    blocked_item_count: int = Field(ge=0)
    approvable: bool
    policy_version: Literal["batch-quality/r7-03-v1"] = BATCH_QUALITY_POLICY_VERSION
    report_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        for ref, object_type, field_name in (
            (
                self.resolved_job_work_graph_ref,
                "resolved-job-work-graph",
                "resolved_job_work_graph_ref",
            ),
            (self.policy_ref, "batch-quality-policy", "policy_ref"),
            (
                self.duplicate_policy_ref,
                "duplicate-detection-policy",
                "duplicate_policy_ref",
            ),
            (
                self.duplicate_result_ref,
                "duplicate-detection-result",
                "duplicate_result_ref",
            ),
            (
                self.cross_item_policy_ref,
                "cross-item-safety-policy",
                "cross_item_policy_ref",
            ),
            (
                self.cross_item_result_ref,
                "cross-item-safety-result",
                "cross_item_result_ref",
            ),
        ):
            _require_ref(ref, object_type, "v2", field_name)
        _validate_item_report_bindings(self)
        _validate_nested_report_values(self)
        _validate_report_evidence(self)
        _validate_report_ownership(self)
        _validate_report_counts(self)
        _validate_report_approvable(self)
        _validate_audit(
            self.audit,
            _batch_quality_report_refs(self),
            "BatchQualityReport",
        )
        _validate_identity(
            object_id=self.batch_quality_report_id,
            object_sha256=self.report_sha256,
            expected_prefix="batch-quality-report",
            observed=batch_quality_report_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        batch_id: str,
        resolved_job_work_graph_ref: ObjectRef,
        policy_ref: ObjectRef,
        duplicate_policy_ref: ObjectRef,
        duplicate_result_ref: ObjectRef,
        cross_item_policy_ref: ObjectRef,
        cross_item_result_ref: ObjectRef,
        item_ids: tuple[str, ...],
        item_quality_result_refs: tuple[ObjectRef, ...],
        base_item_quality_report_refs: tuple[ObjectRef, ...],
        current_item_quality_report_refs: tuple[ObjectRef, ...],
        lineage_audits: tuple[ItemLineageAuditResultV2, ...],
        batch_lineage_checks: tuple[LineageAuditCheckV2, ...],
        findings: tuple[BatchReviewerFindingV2, ...],
        item_quality_report_revisions: tuple[
            ItemQualityReportRevisionV2,
            ...,
        ],
        duplicate_pair_refs: tuple[ObjectRef, ...],
        duplicate_cluster_refs: tuple[ObjectRef, ...],
        visible_match_refs: tuple[ObjectRef, ...],
        answer_reuse_pair_refs: tuple[ObjectRef, ...],
        safety_cluster_refs: tuple[ObjectRef, ...],
        invalidated_result_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> BatchQualityReportV2:
        ordered_lineages = tuple(sorted(lineage_audits, key=lambda value: value.item_id))
        lineage_refs = tuple(item_lineage_audit_result_v2_ref(value) for value in ordered_lineages)
        ordered_batch_checks = _sorted_nested(
            batch_lineage_checks,
            lineage_audit_check_v2_ref,
        )
        batch_check_refs = tuple(lineage_audit_check_v2_ref(value) for value in ordered_batch_checks)
        ordered_findings = _sorted_nested(
            findings,
            batch_reviewer_finding_v2_ref,
        )
        finding_refs = tuple(batch_reviewer_finding_v2_ref(value) for value in ordered_findings)
        ordered_revisions = tuple(
            sorted(
                item_quality_report_revisions,
                key=lambda value: value.item_id,
            )
        )
        revision_refs = tuple(item_quality_report_revision_v2_ref(value) for value in ordered_revisions)
        item_finding_refs = _sorted_refs(
            tuple(
                batch_reviewer_finding_v2_ref(value)
                for value in ordered_findings
                if value.scope is BatchFindingScopeV2.ITEM
            )
        )
        batch_finding_refs = _sorted_refs(
            tuple(
                batch_reviewer_finding_v2_ref(value)
                for value in ordered_findings
                if value.scope is BatchFindingScopeV2.BATCH
            )
        )
        affected = tuple(
            sorted({item_id for finding in ordered_findings for item_id in finding.affected_item_ids})
        )
        blocked = tuple(
            sorted({item_id for finding in ordered_findings for item_id in finding.blocked_item_ids})
        )
        lineage_check_count = sum(len(value.checks) for value in ordered_lineages) + len(ordered_batch_checks)
        counts = _finding_severity_counts(ordered_findings)
        approvable = (
            not blocked
            and not ordered_revisions
            and not invalidated_result_refs
            and all(value.outcome is ItemLineageAuditOutcomeV2.PASSED for value in ordered_lineages)
            and all(
                value.outcome
                in {
                    LineageAuditCheckOutcomeV2.PASSED,
                    LineageAuditCheckOutcomeV2.SKIPPED,
                }
                for value in ordered_batch_checks
            )
        )
        refs = (
            resolved_job_work_graph_ref,
            policy_ref,
            duplicate_policy_ref,
            duplicate_result_ref,
            cross_item_policy_ref,
            cross_item_result_ref,
            *item_quality_result_refs,
            *base_item_quality_report_refs,
            *current_item_quality_report_refs,
            *lineage_refs,
            *batch_check_refs,
            *finding_refs,
            *revision_refs,
            *duplicate_pair_refs,
            *duplicate_cluster_refs,
            *visible_match_refs,
            *answer_reuse_pair_refs,
            *safety_cluster_refs,
            *invalidated_result_refs,
        )
        value = cls(
            batch_quality_report_id="batch-quality-report://pending",
            batch_id=batch_id,
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            policy_ref=policy_ref,
            duplicate_policy_ref=duplicate_policy_ref,
            duplicate_result_ref=duplicate_result_ref,
            cross_item_policy_ref=cross_item_policy_ref,
            cross_item_result_ref=cross_item_result_ref,
            item_ids=item_ids,
            item_quality_result_refs=item_quality_result_refs,
            base_item_quality_report_refs=base_item_quality_report_refs,
            current_item_quality_report_refs=current_item_quality_report_refs,
            lineage_audits=ordered_lineages,
            lineage_audit_refs=lineage_refs,
            batch_lineage_checks=ordered_batch_checks,
            batch_lineage_check_refs=batch_check_refs,
            findings=ordered_findings,
            finding_refs=finding_refs,
            item_quality_report_revisions=ordered_revisions,
            item_quality_report_revision_refs=revision_refs,
            duplicate_pair_refs=_sorted_refs(duplicate_pair_refs),
            duplicate_cluster_refs=_sorted_refs(duplicate_cluster_refs),
            visible_match_refs=_sorted_refs(visible_match_refs),
            answer_reuse_pair_refs=_sorted_refs(answer_reuse_pair_refs),
            safety_cluster_refs=_sorted_refs(safety_cluster_refs),
            item_scoped_finding_refs=item_finding_refs,
            batch_scoped_finding_refs=batch_finding_refs,
            open_finding_refs=finding_refs,
            affected_item_ids=affected,
            blocked_item_ids=blocked,
            invalidated_result_refs=_sorted_refs(invalidated_result_refs),
            lineage_audit_count=len(ordered_lineages),
            lineage_check_count=lineage_check_count,
            finding_count=len(ordered_findings),
            item_scoped_finding_count=len(item_finding_refs),
            batch_scoped_finding_count=len(batch_finding_refs),
            open_p0_count=counts[0],
            open_p1_count=counts[1],
            open_p2_count=counts[2],
            unresolved_non_waivable_count=counts[3],
            report_revision_count=len(ordered_revisions),
            affected_item_count=len(affected),
            blocked_item_count=len(blocked),
            approvable=approvable,
            report_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        digest = batch_quality_report_v2_carried_sha256(value)
        return _finalize(
            value,
            id_field="batch_quality_report_id",
            hash_field="report_sha256",
            prefix="batch-quality-report",
            digest=digest,
        )


def batch_quality_policy_v2_carried_sha256(
    value: BatchQualityPolicyV2,
) -> str:
    return _carried_sha256(
        value,
        exclude={
            "batch_quality_policy_id",
            "policy_sha256",
            "audit",
        },
    )


def batch_quality_policy_v2_ref(value: BatchQualityPolicyV2) -> ObjectRef:
    return ObjectRef(
        object_type="batch-quality-policy",
        object_id=value.batch_quality_policy_id,
        object_version="v2",
        object_sha256=value.policy_sha256,
    )


def lineage_audit_check_v2_carried_sha256(
    value: LineageAuditCheckV2,
) -> str:
    return _carried_sha256(
        value,
        exclude={
            "lineage_audit_check_id",
            "check_sha256",
            "audit",
        },
    )


def lineage_audit_check_v2_ref(value: LineageAuditCheckV2) -> ObjectRef:
    return ObjectRef(
        object_type="lineage-audit-check",
        object_id=value.lineage_audit_check_id,
        object_version="v2",
        object_sha256=value.check_sha256,
    )


def item_lineage_audit_result_v2_carried_sha256(
    value: ItemLineageAuditResultV2,
) -> str:
    return _payload_sha256(
        {
            "item_id": value.item_id,
            "source_trace_ref": _ref_payload(value.source_trace_ref),
            "trace_envelope_ref": _ref_payload(value.trace_envelope_ref),
            "label_decision_refs": [_ref_payload(ref) for ref in value.label_decision_refs],
            "selection_context_ref": _ref_payload(value.selection_context_ref),
            "task_draft_ref": _ref_payload(value.task_draft_ref),
            "r4_task_contract_set_ref": _ref_payload(value.r4_task_contract_set_ref),
            "item_quality_result_ref": _ref_payload(value.item_quality_result_ref),
            "base_quality_report_ref": _ref_payload(value.base_quality_report_ref),
            "final_package_manifest_ref": _ref_payload(value.final_package_manifest_ref),
            "provenance_manifest_ref": _ref_payload(value.provenance_manifest_ref),
            "environment_spec_ref": _ref_payload(value.environment_spec_ref),
            "check_refs": [_ref_payload(ref) for ref in value.check_refs],
            "outcome": value.outcome.value,
            "passed_check_count": value.passed_check_count,
            "failed_check_count": value.failed_check_count,
            "skipped_check_count": value.skipped_check_count,
            "indeterminate_check_count": value.indeterminate_check_count,
            "policy_ref": _ref_payload(value.policy_ref),
        }
    )


def item_lineage_audit_result_v2_ref(
    value: ItemLineageAuditResultV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="item-lineage-audit-result",
        object_id=value.item_lineage_audit_result_id,
        object_version="v2",
        object_sha256=value.result_sha256,
    )


def batch_reviewer_finding_v2_carried_sha256(
    value: BatchReviewerFindingV2,
) -> str:
    return _carried_sha256(
        value,
        exclude={
            "batch_reviewer_finding_id",
            "finding_sha256",
            "audit",
        },
    )


def batch_reviewer_finding_v2_ref(
    value: BatchReviewerFindingV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="batch-reviewer-finding",
        object_id=value.batch_reviewer_finding_id,
        object_version="v2",
        object_sha256=value.finding_sha256,
    )


def item_quality_report_revision_v2_carried_sha256(
    value: ItemQualityReportRevisionV2,
) -> str:
    return _payload_sha256(
        {
            "item_id": value.item_id,
            "revision": value.revision,
            "base_item_quality_result_ref": _ref_payload(value.base_item_quality_result_ref),
            "base_quality_report_ref": _ref_payload(value.base_quality_report_ref),
            "final_package_manifest_ref": _ref_payload(value.final_package_manifest_ref),
            "provenance_manifest_ref": _ref_payload(value.provenance_manifest_ref),
            "environment_spec_ref": _ref_payload(value.environment_spec_ref),
            "accepted_artifact_refs": [_ref_payload(ref) for ref in value.accepted_artifact_refs],
            "package_sha256": value.package_sha256,
            "finding_refs": [_ref_payload(ref) for ref in value.finding_refs],
            "open_p0_count": value.open_p0_count,
            "open_p1_count": value.open_p1_count,
            "unresolved_non_waivable_count": (value.unresolved_non_waivable_count),
            "approvable": value.approvable,
            "release_revalidation_required": (value.release_revalidation_required),
            "policy_ref": _ref_payload(value.policy_ref),
        }
    )


def item_quality_report_revision_v2_ref(
    value: ItemQualityReportRevisionV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="item-quality-report-revision",
        object_id=value.item_quality_report_revision_id,
        object_version="v2",
        object_sha256=value.revision_sha256,
    )


def batch_quality_report_v2_carried_sha256(
    value: BatchQualityReportV2,
) -> str:
    return _payload_sha256(
        {
            "batch_id": value.batch_id,
            "resolved_job_work_graph_ref": _ref_payload(value.resolved_job_work_graph_ref),
            "policy_ref": _ref_payload(value.policy_ref),
            "duplicate_policy_ref": _ref_payload(value.duplicate_policy_ref),
            "duplicate_result_ref": _ref_payload(value.duplicate_result_ref),
            "cross_item_policy_ref": _ref_payload(value.cross_item_policy_ref),
            "cross_item_result_ref": _ref_payload(value.cross_item_result_ref),
            "item_ids": list(value.item_ids),
            "item_quality_result_refs": [_ref_payload(ref) for ref in value.item_quality_result_refs],
            "base_item_quality_report_refs": [
                _ref_payload(ref) for ref in value.base_item_quality_report_refs
            ],
            "current_item_quality_report_refs": [
                _ref_payload(ref) for ref in value.current_item_quality_report_refs
            ],
            "lineage_audit_refs": [_ref_payload(ref) for ref in value.lineage_audit_refs],
            "batch_lineage_check_refs": [_ref_payload(ref) for ref in value.batch_lineage_check_refs],
            "finding_refs": [_ref_payload(ref) for ref in value.finding_refs],
            "item_quality_report_revision_refs": [
                _ref_payload(ref) for ref in value.item_quality_report_revision_refs
            ],
            "duplicate_pair_refs": [_ref_payload(ref) for ref in value.duplicate_pair_refs],
            "duplicate_cluster_refs": [_ref_payload(ref) for ref in value.duplicate_cluster_refs],
            "visible_match_refs": [_ref_payload(ref) for ref in value.visible_match_refs],
            "answer_reuse_pair_refs": [_ref_payload(ref) for ref in value.answer_reuse_pair_refs],
            "safety_cluster_refs": [_ref_payload(ref) for ref in value.safety_cluster_refs],
            "item_scoped_finding_refs": [_ref_payload(ref) for ref in value.item_scoped_finding_refs],
            "batch_scoped_finding_refs": [_ref_payload(ref) for ref in value.batch_scoped_finding_refs],
            "open_finding_refs": [_ref_payload(ref) for ref in value.open_finding_refs],
            "affected_item_ids": list(value.affected_item_ids),
            "blocked_item_ids": list(value.blocked_item_ids),
            "invalidated_result_refs": [_ref_payload(ref) for ref in value.invalidated_result_refs],
            "lineage_audit_count": value.lineage_audit_count,
            "lineage_check_count": value.lineage_check_count,
            "finding_count": value.finding_count,
            "item_scoped_finding_count": value.item_scoped_finding_count,
            "batch_scoped_finding_count": value.batch_scoped_finding_count,
            "open_p0_count": value.open_p0_count,
            "open_p1_count": value.open_p1_count,
            "open_p2_count": value.open_p2_count,
            "unresolved_non_waivable_count": (value.unresolved_non_waivable_count),
            "report_revision_count": value.report_revision_count,
            "affected_item_count": value.affected_item_count,
            "blocked_item_count": value.blocked_item_count,
            "approvable": value.approvable,
            "policy_version": value.policy_version,
        }
    )


def batch_quality_report_v2_ref(value: BatchQualityReportV2) -> ObjectRef:
    return ObjectRef(
        object_type="batch-quality-report",
        object_id=value.batch_quality_report_id,
        object_version="v2",
        object_sha256=value.report_sha256,
    )


def validate_batch_quality_policy_v2_identity(
    value: BatchQualityPolicyV2,
) -> None:
    _require_current_identity(
        object_id=value.batch_quality_policy_id,
        object_sha256=value.policy_sha256,
        expected_prefix="batch-quality-policy",
        observed=batch_quality_policy_v2_carried_sha256(value),
    )


def validate_lineage_audit_check_v2_identity(
    value: LineageAuditCheckV2,
) -> None:
    _require_current_identity(
        object_id=value.lineage_audit_check_id,
        object_sha256=value.check_sha256,
        expected_prefix="lineage-audit-check",
        observed=lineage_audit_check_v2_carried_sha256(value),
    )


def validate_item_lineage_audit_result_v2_identity(
    value: ItemLineageAuditResultV2,
) -> None:
    _require_current_identity(
        object_id=value.item_lineage_audit_result_id,
        object_sha256=value.result_sha256,
        expected_prefix="item-lineage-audit-result",
        observed=item_lineage_audit_result_v2_carried_sha256(value),
    )


def validate_batch_reviewer_finding_v2_identity(
    value: BatchReviewerFindingV2,
) -> None:
    _require_current_identity(
        object_id=value.batch_reviewer_finding_id,
        object_sha256=value.finding_sha256,
        expected_prefix="batch-reviewer-finding",
        observed=batch_reviewer_finding_v2_carried_sha256(value),
    )


def validate_item_quality_report_revision_v2_identity(
    value: ItemQualityReportRevisionV2,
) -> None:
    _require_current_identity(
        object_id=value.item_quality_report_revision_id,
        object_sha256=value.revision_sha256,
        expected_prefix="item-quality-report-revision",
        observed=item_quality_report_revision_v2_carried_sha256(value),
    )


def validate_batch_quality_report_v2_identity(
    value: BatchQualityReportV2,
) -> None:
    _require_current_identity(
        object_id=value.batch_quality_report_id,
        object_sha256=value.report_sha256,
        expected_prefix="batch-quality-report",
        observed=batch_quality_report_v2_carried_sha256(value),
    )


def _validate_item_report_bindings(report: BatchQualityReportV2) -> None:
    item_count = len(report.item_ids)
    if (
        len(
            {
                item_count,
                len(report.item_quality_result_refs),
                len(report.base_item_quality_report_refs),
                len(report.current_item_quality_report_refs),
                len(report.lineage_audits),
            }
        )
        != 1
    ):
        raise ValueError("batch report Item binding lengths differ")
    _require_sorted_unique("batch report Item IDs", report.item_ids)
    for label, refs, object_type in (
        (
            "batch item quality result refs",
            report.item_quality_result_refs,
            "item-quality-compilation-result",
        ),
        (
            "batch base quality report refs",
            report.base_item_quality_report_refs,
            "quality-report",
        ),
    ):
        _require_unique_refs(label, refs)
        for ref in refs:
            _require_ref(ref, object_type, "v2", label)
    if tuple(value.item_id for value in report.lineage_audits) != (report.item_ids):
        raise ValueError("lineage audit Item inventory is not aligned")
    revision_by_item = {value.item_id: value for value in report.item_quality_report_revisions}
    if len(revision_by_item) != len(report.item_quality_report_revisions):
        raise ValueError("duplicate Item quality report revision")
    for item_id, base_ref, current_ref in zip(
        report.item_ids,
        report.base_item_quality_report_refs,
        report.current_item_quality_report_refs,
        strict=True,
    ):
        revision = revision_by_item.get(item_id)
        expected = item_quality_report_revision_v2_ref(revision) if revision is not None else base_ref
        if current_ref != expected:
            raise ValueError("current Item quality report ref is not exact")
    if any(item_id not in set(report.item_ids) for item_id in revision_by_item):
        raise ValueError("Item quality revision belongs to an unknown Item")


def _validate_nested_report_values(report: BatchQualityReportV2) -> None:
    expected_lineage_refs = tuple(
        item_lineage_audit_result_v2_ref(lineage_value) for lineage_value in report.lineage_audits
    )
    if report.lineage_audit_refs != expected_lineage_refs:
        raise ValueError("lineage audit refs do not match nested values")
    for lineage_value in report.lineage_audits:
        validate_item_lineage_audit_result_v2_identity(lineage_value)
        if lineage_value.policy_ref != report.policy_ref:
            raise ValueError("lineage audit policy is mismatched")
    expected_batch_check_refs = tuple(
        lineage_audit_check_v2_ref(check_value) for check_value in report.batch_lineage_checks
    )
    if report.batch_lineage_check_refs != expected_batch_check_refs:
        raise ValueError("batch lineage check refs do not match nested values")
    if tuple(value.code for value in report.batch_lineage_checks) != (BATCH_QUALITY_BATCH_LINEAGE_CHECKS):
        raise ValueError("batch lineage check inventory is not exact")
    for check_value in report.batch_lineage_checks:
        validate_lineage_audit_check_v2_identity(check_value)
        if (
            check_value.code is not LineageAuditCheckCodeV2.CROSS_ITEM_LINEAGE_ALIAS
            or check_value.policy_ref != report.policy_ref
        ):
            raise ValueError("batch lineage check is invalid")
    expected_finding_refs = tuple(
        batch_reviewer_finding_v2_ref(finding_value) for finding_value in report.findings
    )
    if report.finding_refs != expected_finding_refs:
        raise ValueError("finding refs do not match nested values")
    for finding_value in report.findings:
        validate_batch_reviewer_finding_v2_identity(finding_value)
        if finding_value.policy_ref != report.policy_ref:
            raise ValueError("finding policy is mismatched")
    expected_revision_refs = tuple(
        item_quality_report_revision_v2_ref(revision_value)
        for revision_value in report.item_quality_report_revisions
    )
    if report.item_quality_report_revision_refs != expected_revision_refs:
        raise ValueError("revision refs do not match nested values")
    for revision_value in report.item_quality_report_revisions:
        validate_item_quality_report_revision_v2_identity(revision_value)
        if revision_value.policy_ref != report.policy_ref:
            raise ValueError("revision policy is mismatched")
    for label, refs in (
        ("batch lineage check refs", report.batch_lineage_check_refs),
        ("finding refs", report.finding_refs),
    ):
        _require_sorted_unique_refs(label, refs)
    _require_unique_refs("lineage audit refs", report.lineage_audit_refs)
    _require_unique_refs(
        "Item quality report revision refs",
        report.item_quality_report_revision_refs,
    )


def _validate_report_evidence(report: BatchQualityReportV2) -> None:
    for label, refs, object_type in (
        (
            "duplicate pair refs",
            report.duplicate_pair_refs,
            "duplicate-pair-evidence",
        ),
        (
            "duplicate cluster refs",
            report.duplicate_cluster_refs,
            "duplicate-cluster",
        ),
        (
            "visible match refs",
            report.visible_match_refs,
            "cross-item-visible-match-evidence",
        ),
        (
            "answer reuse pair refs",
            report.answer_reuse_pair_refs,
            "answer-reuse-pair-evidence",
        ),
        (
            "safety cluster refs",
            report.safety_cluster_refs,
            "cross-item-safety-cluster",
        ),
    ):
        _require_sorted_unique_refs(label, refs)
        for ref in refs:
            _require_ref(ref, object_type, "v2", label)
    failed_lineage_refs = tuple(
        lineage_audit_check_v2_ref(check)
        for result in report.lineage_audits
        for check in result.checks
        if check.outcome
        in {
            LineageAuditCheckOutcomeV2.FAILED,
            LineageAuditCheckOutcomeV2.INDETERMINATE,
        }
    ) + tuple(
        lineage_audit_check_v2_ref(check)
        for check in report.batch_lineage_checks
        if check.outcome
        in {
            LineageAuditCheckOutcomeV2.FAILED,
            LineageAuditCheckOutcomeV2.INDETERMINATE,
        }
    )
    expected_direct = _sorted_refs(
        (
            *report.duplicate_pair_refs,
            *report.visible_match_refs,
            *report.answer_reuse_pair_refs,
            *failed_lineage_refs,
        )
    )
    finding_direct_refs = tuple(value.direct_evidence_ref for value in report.findings)
    _require_unique_refs("finding direct evidence refs", finding_direct_refs)
    observed_direct = _sorted_refs(finding_direct_refs)
    if observed_direct != expected_direct:
        raise ValueError("findings do not exactly cover authoritative direct evidence")


def _validate_report_ownership(report: BatchQualityReportV2) -> None:
    item_refs = _sorted_refs(
        tuple(
            batch_reviewer_finding_v2_ref(value)
            for value in report.findings
            if value.scope is BatchFindingScopeV2.ITEM
        )
    )
    batch_refs = _sorted_refs(
        tuple(
            batch_reviewer_finding_v2_ref(value)
            for value in report.findings
            if value.scope is BatchFindingScopeV2.BATCH
        )
    )
    if (
        report.item_scoped_finding_refs != item_refs
        or report.batch_scoped_finding_refs != batch_refs
        or report.open_finding_refs != report.finding_refs
        or set(item_refs) & set(batch_refs)
    ):
        raise ValueError("finding ownership inventories are not exact")
    affected = tuple(sorted({item_id for value in report.findings for item_id in value.affected_item_ids}))
    blocked = tuple(sorted({item_id for value in report.findings for item_id in value.blocked_item_ids}))
    if (
        report.affected_item_ids != affected
        or report.blocked_item_ids != blocked
        or not set(blocked).issubset(report.item_ids)
        or not set(affected).issubset(report.item_ids)
    ):
        raise ValueError("finding Item partitions are not exact")
    revision_items = {value.item_id for value in report.item_quality_report_revisions}
    item_finding_owners = {
        value.owner_item_id for value in report.findings if value.scope is BatchFindingScopeV2.ITEM
    }
    if revision_items != item_finding_owners:
        raise ValueError("Item findings and quality report revisions do not align")
    finding_refs_by_item: dict[str, set[ObjectRef]] = {}
    for value in report.findings:
        if value.owner_item_id is not None:
            finding_refs_by_item.setdefault(value.owner_item_id, set()).add(
                batch_reviewer_finding_v2_ref(value)
            )
    for revision in report.item_quality_report_revisions:
        if set(revision.finding_refs) != finding_refs_by_item.get(
            revision.item_id,
            set(),
        ):
            raise ValueError("revision finding inventory is not exact")


def _validate_report_counts(report: BatchQualityReportV2) -> None:
    lineage_check_count = sum(len(value.checks) for value in report.lineage_audits) + len(
        report.batch_lineage_checks
    )
    severity_counts = _finding_severity_counts(report.findings)
    expected = (
        len(report.lineage_audits),
        lineage_check_count,
        len(report.findings),
        len(report.item_scoped_finding_refs),
        len(report.batch_scoped_finding_refs),
        severity_counts[0],
        severity_counts[1],
        severity_counts[2],
        severity_counts[3],
        len(report.item_quality_report_revisions),
        len(report.affected_item_ids),
        len(report.blocked_item_ids),
    )
    observed = (
        report.lineage_audit_count,
        report.lineage_check_count,
        report.finding_count,
        report.item_scoped_finding_count,
        report.batch_scoped_finding_count,
        report.open_p0_count,
        report.open_p1_count,
        report.open_p2_count,
        report.unresolved_non_waivable_count,
        report.report_revision_count,
        report.affected_item_count,
        report.blocked_item_count,
    )
    if observed != expected:
        raise ValueError("BatchQualityReport counts are not exact")


def _validate_report_approvable(report: BatchQualityReportV2) -> None:
    expected = (
        not report.blocked_item_ids
        and not report.item_quality_report_revisions
        and not report.invalidated_result_refs
        and not report.open_p0_count
        and not report.open_p1_count
        and not report.unresolved_non_waivable_count
        and all(value.outcome is ItemLineageAuditOutcomeV2.PASSED for value in report.lineage_audits)
        and all(
            value.outcome
            in {
                LineageAuditCheckOutcomeV2.PASSED,
                LineageAuditCheckOutcomeV2.SKIPPED,
            }
            for value in report.batch_lineage_checks
        )
    )
    if report.approvable is not expected:
        raise ValueError("BatchQualityReport approvable is not derived")


def _batch_quality_report_refs(
    report: BatchQualityReportV2,
) -> tuple[ObjectRef, ...]:
    return (
        report.resolved_job_work_graph_ref,
        report.policy_ref,
        report.duplicate_policy_ref,
        report.duplicate_result_ref,
        report.cross_item_policy_ref,
        report.cross_item_result_ref,
        *report.item_quality_result_refs,
        *report.base_item_quality_report_refs,
        *report.current_item_quality_report_refs,
        *report.lineage_audit_refs,
        *report.batch_lineage_check_refs,
        *report.finding_refs,
        *report.item_quality_report_revision_refs,
        *report.duplicate_pair_refs,
        *report.duplicate_cluster_refs,
        *report.visible_match_refs,
        *report.answer_reuse_pair_refs,
        *report.safety_cluster_refs,
        *report.invalidated_result_refs,
    )


def _finding_policy(
    category: BatchFindingCategoryV2,
    *,
    restricted_category: PromptLeakageCategoryV2 | None,
) -> tuple[BatchFindingScopeV2, Severity, bool, bool]:
    duplicate = {
        BatchFindingCategoryV2.EXACT_TASK_DUPLICATE: (
            Severity.P1,
            True,
        ),
        BatchFindingCategoryV2.NEAR_TASK_DUPLICATE: (
            Severity.P2,
            False,
        ),
        BatchFindingCategoryV2.EXACT_ATTACHMENT_DUPLICATE: (
            Severity.P2,
            False,
        ),
        BatchFindingCategoryV2.NEAR_ATTACHMENT_DUPLICATE: (
            Severity.P2,
            False,
        ),
    }
    if category in duplicate:
        if restricted_category is not None:
            raise ValueError("duplicate finding cannot carry restricted category")
        severity, blocking = duplicate[category]
        return BatchFindingScopeV2.BATCH, severity, False, blocking
    if category is BatchFindingCategoryV2.CROSS_ITEM_CONTAMINATION:
        if restricted_category not in {
            PromptLeakageCategoryV2.FINAL_ANSWER,
            PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
        }:
            raise ValueError("contamination finding requires final/completed category")
        return BatchFindingScopeV2.ITEM, Severity.P0, True, True
    if category is BatchFindingCategoryV2.CROSS_ITEM_LEAKAGE:
        if restricted_category is None or restricted_category in {
            PromptLeakageCategoryV2.FINAL_ANSWER,
            PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
        }:
            raise ValueError("leakage finding requires one leakage restricted category")
        severity = (
            Severity.P0 if restricted_category is PromptLeakageCategoryV2.PRIVATE_REFERENCE else Severity.P1
        )
        return BatchFindingScopeV2.ITEM, severity, True, True
    if restricted_category is not None:
        raise ValueError("finding category does not accept restricted category")
    if category is BatchFindingCategoryV2.ANSWER_REUSE:
        return BatchFindingScopeV2.BATCH, Severity.P0, True, True
    if category in {
        BatchFindingCategoryV2.LINEAGE_INCOMPLETE,
        BatchFindingCategoryV2.LINEAGE_INDETERMINATE,
    }:
        return BatchFindingScopeV2.ITEM, Severity.P1, True, True
    if category is BatchFindingCategoryV2.CROSS_ITEM_LINEAGE_ALIAS:
        return BatchFindingScopeV2.BATCH, Severity.P1, True, True
    raise ValueError("unsupported batch finding category")


def _require_finding_evidence_type(
    category: BatchFindingCategoryV2,
    ref: ObjectRef,
) -> None:
    expected = {
        BatchFindingCategoryV2.EXACT_TASK_DUPLICATE: ("duplicate-pair-evidence"),
        BatchFindingCategoryV2.NEAR_TASK_DUPLICATE: ("duplicate-pair-evidence"),
        BatchFindingCategoryV2.EXACT_ATTACHMENT_DUPLICATE: ("duplicate-pair-evidence"),
        BatchFindingCategoryV2.NEAR_ATTACHMENT_DUPLICATE: ("duplicate-pair-evidence"),
        BatchFindingCategoryV2.CROSS_ITEM_CONTAMINATION: ("cross-item-visible-match-evidence"),
        BatchFindingCategoryV2.CROSS_ITEM_LEAKAGE: ("cross-item-visible-match-evidence"),
        BatchFindingCategoryV2.ANSWER_REUSE: ("answer-reuse-pair-evidence"),
        BatchFindingCategoryV2.LINEAGE_INCOMPLETE: ("lineage-audit-check"),
        BatchFindingCategoryV2.LINEAGE_INDETERMINATE: ("lineage-audit-check"),
        BatchFindingCategoryV2.CROSS_ITEM_LINEAGE_ALIAS: ("lineage-audit-check"),
    }[category]
    _require_ref(ref, expected, "v2", "direct_evidence_ref")


def _lineage_outcome_counts(
    checks: tuple[LineageAuditCheckV2, ...],
) -> tuple[int, int, int, int]:
    return (
        sum(value.outcome is LineageAuditCheckOutcomeV2.PASSED for value in checks),
        sum(value.outcome is LineageAuditCheckOutcomeV2.FAILED for value in checks),
        sum(value.outcome is LineageAuditCheckOutcomeV2.SKIPPED for value in checks),
        sum(value.outcome is LineageAuditCheckOutcomeV2.INDETERMINATE for value in checks),
    )


def _item_lineage_outcome(
    checks: tuple[LineageAuditCheckV2, ...],
) -> ItemLineageAuditOutcomeV2:
    if any(value.outcome is LineageAuditCheckOutcomeV2.FAILED for value in checks):
        return ItemLineageAuditOutcomeV2.FAILED
    if any(value.outcome is LineageAuditCheckOutcomeV2.INDETERMINATE for value in checks):
        return ItemLineageAuditOutcomeV2.INDETERMINATE
    return ItemLineageAuditOutcomeV2.PASSED


def _finding_blocker_counts(
    findings: tuple[BatchReviewerFindingV2, ...],
) -> tuple[int, int, int]:
    return (
        sum(value.severity is Severity.P0 for value in findings),
        sum(value.severity is Severity.P1 for value in findings),
        sum(value.non_waivable for value in findings),
    )


def _finding_severity_counts(
    findings: tuple[BatchReviewerFindingV2, ...],
) -> tuple[int, int, int, int]:
    return (
        sum(value.severity is Severity.P0 for value in findings),
        sum(value.severity is Severity.P1 for value in findings),
        sum(value.severity is Severity.P2 for value in findings),
        sum(value.non_waivable for value in findings),
    )


def _carried_sha256(
    value: ContractModelV2,
    *,
    exclude: set[str],
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude=exclude,
            exclude_none=False,
        )
    )


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _ref_payload(value: ObjectRef) -> dict[str, object]:
    return value.model_dump(mode="json", exclude_none=False)


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _require_sorted_unique_refs(
    label: str,
    values: tuple[ObjectRef, ...],
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    _require_sorted_unique(label, keys)


def _require_unique_refs(
    label: str,
    values: tuple[ObjectRef, ...],
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} must be unique")


def _require_sorted_unique(
    label: str,
    values: tuple[Any, ...],
) -> None:
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


def _require_ref(
    ref: ObjectRef,
    expected_type: str,
    expected_version: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type or ref.object_version != expected_version:
        raise ValueError(f"{field_name} must reference {expected_type} {expected_version}")


def _require_ref_type(
    ref: ObjectRef,
    expected_type: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type:
        raise ValueError(f"{field_name} must reference {expected_type}")


def _validate_audit(
    audit: ContractAudit,
    expected_refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(expected_refs):
        raise ValueError(f"{label} audit input refs are not exact")


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    governing = (
        *(value for value in audit.governing_versions if value.component != "batch-quality"),
        VersionBinding(
            component="batch-quality",
            version=BATCH_QUALITY_POLICY_VERSION,
        ),
    )
    return audit.model_copy(
        update={
            "governing_versions": governing,
            "input_refs": _sorted_refs(refs),
        }
    )


def _validate_identity(
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


def _require_current_identity(
    *,
    object_id: str,
    object_sha256: str,
    expected_prefix: str,
    observed: str,
) -> None:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        raise ValueError(f"{expected_prefix} identity is pending")
    _validate_identity(
        object_id=object_id,
        object_sha256=object_sha256,
        expected_prefix=expected_prefix,
        observed=observed,
    )


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


def _sorted_nested[ValueT](
    values: tuple[ValueT, ...],
    ref_builder: Callable[[ValueT], ObjectRef],
) -> tuple[ValueT, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: _ref_key(ref_builder(value)),
        )
    )


def _parse_enum[EnumT: StrEnum](
    value: object,
    enum_type: type[EnumT],
    field_name: str,
) -> EnumT:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        return enum_type(value)
    raise TypeError(f"{field_name} must be a {enum_type.__name__}")
