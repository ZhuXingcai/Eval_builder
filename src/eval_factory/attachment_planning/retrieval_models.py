from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from env_mock_agent.facade import (
    FacadeObjectRef,
    PublicSourceFetchRequestV2,
    PublicSourceFetchResultV2,
    PublicSourceSearchRequestV2,
    PublicSourceSearchResultV2,
    public_source_fetch_request_ref,
    public_source_fetch_result_ref,
    public_source_search_request_ref,
)
from eval_factory.contracts.attachment_v2 import SourceEvidenceSetV2
from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import canonical_value_v2

APPROVED_SOURCE_RETRIEVAL_POLICY_VERSION: Literal["public-source-retrieval/r5-04-v1"] = (
    "public-source-retrieval/r5-04-v1"
)


class ApprovedSourceEvidencePolicyError(RuntimeError):
    pass


class ApprovedSourceRetrievalMode(StrEnum):
    REFETCH_EXTERNAL_LEAD = "REFETCH_EXTERNAL_LEAD"
    SEARCH_THEN_FETCH = "SEARCH_THEN_FETCH"


class PublicSourceUsageBasis(StrEnum):
    FACTS_ONLY = "FACTS_ONLY"
    STRUCTURE_AND_STYLE_ONLY = "STRUCTURE_AND_STYLE_ONLY"


class PublicSourceCheck(StrEnum):
    SECRET = "SECRET"
    CONFIGURED_PII = "CONFIGURED_PII"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    ANSWER_LEAKAGE = "ANSWER_LEAKAGE"
    LICENSE = "LICENSE"


class ApprovedSourceRetrievalExecutionOutcome(StrEnum):
    EXECUTED = "EXECUTED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"


class ApprovedSourceEvidenceOutcome(StrEnum):
    COMPILED = "COMPILED"
    NOT_REQUIRED = "NOT_REQUIRED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    BLOCKED_SAFETY = "BLOCKED_SAFETY"


class ApprovedSourceEvidenceReason(StrEnum):
    ANSWER_LEAKAGE_SCAN_FAILED = "ANSWER_LEAKAGE_SCAN_FAILED"
    FETCH_FAILED = "FETCH_FAILED"
    FETCH_PROVIDER_UNAVAILABLE = "FETCH_PROVIDER_UNAVAILABLE"
    FETCH_RESULT_MISSING = "FETCH_RESULT_MISSING"
    LICENSE_DENIED = "LICENSE_DENIED"
    NETWORK_POLICY_DENIED = "NETWORK_POLICY_DENIED"
    PII_SCAN_FAILED = "PII_SCAN_FAILED"
    PROMPT_INJECTION_SCAN_FAILED = "PROMPT_INJECTION_SCAN_FAILED"
    QUERY_EGRESS_DENIED = "QUERY_EGRESS_DENIED"
    SCAN_CAPABILITY_UNAVAILABLE = "SCAN_CAPABILITY_UNAVAILABLE"
    SEARCH_FAILED = "SEARCH_FAILED"
    SEARCH_PROVIDER_UNAVAILABLE = "SEARCH_PROVIDER_UNAVAILABLE"
    SEARCH_RESULT_EMPTY = "SEARCH_RESULT_EMPTY"
    SECRET_SCAN_FAILED = "SECRET_SCAN_FAILED"
    SOURCE_POLICY_DENIED = "SOURCE_POLICY_DENIED"
    URI_POLICY_DENIED = "URI_POLICY_DENIED"


class ApprovedSourceRetrievalIntentDefinition(ContractModel):
    schema_version: Literal["eval-factory/approved-source-retrieval-intent-definition/r5-04"] = (
        "eval-factory/approved-source-retrieval-intent-definition/r5-04"
    )
    intent_id: Identifier
    attachment_dependency_id: Identifier
    mode: ApprovedSourceRetrievalMode
    search_provider_id: Identifier | None = None
    fetch_provider_id: Identifier
    external_lead_projection_item_id: Identifier | None = None
    query_approval_ref: ObjectRef | None = None
    source_approval_ref: ObjectRef
    usage_basis: PublicSourceUsageBasis

    @field_validator("mode", mode="before")
    @classmethod
    def parse_mode(cls, value: object) -> ApprovedSourceRetrievalMode:
        if isinstance(value, ApprovedSourceRetrievalMode):
            return value
        if isinstance(value, str):
            return ApprovedSourceRetrievalMode(value)
        raise TypeError("mode must be an ApprovedSourceRetrievalMode")

    @field_validator("usage_basis", mode="before")
    @classmethod
    def parse_usage_basis(cls, value: object) -> PublicSourceUsageBasis:
        if isinstance(value, PublicSourceUsageBasis):
            return value
        if isinstance(value, str):
            return PublicSourceUsageBasis(value)
        raise TypeError("usage_basis must be a PublicSourceUsageBasis")

    @model_validator(mode="after")
    def validate_definition(self) -> ApprovedSourceRetrievalIntentDefinition:
        _require_ref_type(
            self.source_approval_ref,
            "public-source-approval",
            "source_approval_ref",
        )
        if self.mode is ApprovedSourceRetrievalMode.REFETCH_EXTERNAL_LEAD:
            if (
                self.external_lead_projection_item_id is None
                or self.search_provider_id is not None
                or self.query_approval_ref is not None
            ):
                raise ValueError("lead re-fetch requires one lead and no search authority")
        elif (
            self.external_lead_projection_item_id is not None
            or self.search_provider_id is None
            or self.query_approval_ref is None
        ):
            raise ValueError("search-then-fetch requires search provider and query approval only")
        if self.query_approval_ref is not None:
            _require_ref_type(
                self.query_approval_ref,
                "public-search-query-approval",
                "query_approval_ref",
            )
        return self


class ApprovedSourceRetrievalIntent(ContractModel):
    schema_version: Literal["eval-factory/approved-source-retrieval-intent/r5-04"] = (
        "eval-factory/approved-source-retrieval-intent/r5-04"
    )
    intent_id: Identifier
    attachment_dependency_id: Identifier
    artifact_evidence_target_ref: ObjectRef
    mode: ApprovedSourceRetrievalMode
    search_provider_id: Identifier | None = None
    fetch_provider_id: Identifier
    external_lead_projection_item_id: Identifier | None = None
    external_lead_ref: ObjectRef | None = None
    external_lead_decision_ref: ObjectRef | None = None
    query: str | None = Field(default=None, min_length=1, max_length=2000)
    source_uri: str | None = Field(default=None, min_length=3, max_length=2048)
    query_approval_ref: ObjectRef | None = None
    source_approval_ref: ObjectRef
    usage_basis: PublicSourceUsageBasis

    @field_validator("mode", mode="before")
    @classmethod
    def parse_mode(cls, value: object) -> ApprovedSourceRetrievalMode:
        if isinstance(value, ApprovedSourceRetrievalMode):
            return value
        if isinstance(value, str):
            return ApprovedSourceRetrievalMode(value)
        raise TypeError("mode must be an ApprovedSourceRetrievalMode")

    @field_validator("usage_basis", mode="before")
    @classmethod
    def parse_usage_basis(cls, value: object) -> PublicSourceUsageBasis:
        if isinstance(value, PublicSourceUsageBasis):
            return value
        if isinstance(value, str):
            return PublicSourceUsageBasis(value)
        raise TypeError("usage_basis must be a PublicSourceUsageBasis")

    @model_validator(mode="after")
    def validate_intent(self) -> ApprovedSourceRetrievalIntent:
        _require_ref_version(
            self.artifact_evidence_target_ref,
            "artifact-evidence-target",
            "v2",
            "artifact_evidence_target_ref",
        )
        _require_ref_type(
            self.source_approval_ref,
            "public-source-approval",
            "source_approval_ref",
        )
        if self.mode is ApprovedSourceRetrievalMode.REFETCH_EXTERNAL_LEAD:
            if (
                self.external_lead_projection_item_id is None
                or self.external_lead_ref is None
                or self.external_lead_decision_ref is None
                or self.source_uri is None
                or self.search_provider_id is not None
                or self.query is not None
                or self.query_approval_ref is not None
            ):
                raise ValueError("lead re-fetch requires exact lead lineage and URI only")
            _require_ref_type(
                self.external_lead_decision_ref,
                "provenance-decision",
                "external_lead_decision_ref",
            )
        elif (
            self.external_lead_projection_item_id is not None
            or self.external_lead_ref is not None
            or self.external_lead_decision_ref is not None
            or self.source_uri is not None
            or self.search_provider_id is None
            or self.query is None
            or self.query_approval_ref is None
        ):
            raise ValueError("search-then-fetch requires exact query approval and no lead")
        if self.query_approval_ref is not None:
            _require_ref_type(
                self.query_approval_ref,
                "public-search-query-approval",
                "query_approval_ref",
            )
        return self


class ApprovedSourceRetrievalRequest(ContractModel):
    schema_version: Literal["eval-factory/approved-source-retrieval-request/r5-04"] = (
        "eval-factory/approved-source-retrieval-request/r5-04"
    )
    request_id: Identifier
    attachment_planning_context_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    producer_evidence_view_ref: ObjectRef
    producer_evidence_bundle_ref: ObjectRef
    dependency_discovery_ref: ObjectRef | None = None
    retrieval_policy_ref: ObjectRef
    allowed_schemes: tuple[str, ...] = Field(min_length=1)
    allowed_host_suffixes: tuple[str, ...] = Field(min_length=1)
    max_search_results: int = Field(ge=1, le=100)
    max_fetch_bytes: int = Field(ge=1)
    intents: tuple[ApprovedSourceRetrievalIntent, ...] = ()
    covered_target_refs: tuple[ObjectRef, ...] = ()
    not_required_target_refs: tuple[ObjectRef, ...] = ()
    policy_version: Literal["public-source-retrieval/r5-04-v1"] = APPROVED_SOURCE_RETRIEVAL_POLICY_VERSION
    request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> ApprovedSourceRetrievalRequest:
        for ref, expected_type, version, field_name in (
            (
                self.attachment_planning_context_ref,
                "attachment-planning-context",
                "v2",
                "attachment_planning_context_ref",
            ),
            (
                self.producer_task_view_ref,
                "producer-task-view",
                "v2",
                "producer_task_view_ref",
            ),
            (
                self.producer_evidence_view_ref,
                "evidence-view-result",
                "r2-05",
                "producer_evidence_view_ref",
            ),
            (
                self.producer_evidence_bundle_ref,
                "evidence-bundle",
                "v1",
                "producer_evidence_bundle_ref",
            ),
            (
                self.retrieval_policy_ref,
                "public-source-retrieval-policy",
                "v2",
                "retrieval_policy_ref",
            ),
        ):
            _require_ref_version(ref, expected_type, version, field_name)
        if self.dependency_discovery_ref is None:
            if self.intents or self.covered_target_refs or self.not_required_target_refs:
                raise ValueError("request without discovery cannot carry intents or targets")
        else:
            _require_ref_version(
                self.dependency_discovery_ref,
                "prompt-only-dependency-discovery",
                "v2",
                "dependency_discovery_ref",
            )
        for label, values in (
            ("allowed schemes", self.allowed_schemes),
            ("allowed host suffixes", self.allowed_host_suffixes),
        ):
            _require_unique(label, values)
            if values != tuple(sorted(values)):
                raise ValueError(f"{label} must be sorted")
        intent_ids = tuple(item.intent_id for item in self.intents)
        dependency_ids = tuple(item.attachment_dependency_id for item in self.intents)
        target_refs = tuple(item.artifact_evidence_target_ref for item in self.intents)
        _require_unique("retrieval intent IDs", intent_ids)
        _require_unique("retrieval intent dependency IDs", dependency_ids)
        _require_unique(
            "retrieval intent target refs",
            tuple(_ref_key(ref) for ref in target_refs),
        )
        if intent_ids != tuple(sorted(intent_ids)):
            raise ValueError("retrieval intents must be sorted by intent ID")
        for label, refs in (
            ("covered target refs", self.covered_target_refs),
            ("not-required target refs", self.not_required_target_refs),
        ):
            keys = tuple(_ref_key(ref) for ref in refs)
            _require_unique(label, keys)
            if keys != tuple(sorted(keys)):
                raise ValueError(f"{label} must be sorted")
            for ref in refs:
                _require_ref_version(
                    ref,
                    "artifact-evidence-target",
                    "v2",
                    label,
                )
        covered_keys = {_ref_key(ref) for ref in self.covered_target_refs}
        not_required_keys = {_ref_key(ref) for ref in self.not_required_target_refs}
        if covered_keys & not_required_keys:
            raise ValueError("covered and not-required target refs must be disjoint")
        if covered_keys != {_ref_key(ref) for ref in target_refs}:
            raise ValueError("covered target refs must exactly equal retrieval intent targets")
        return self


class PublicSourceRetrievalExecution(ContractModel):
    schema_version: Literal["eval-factory/public-source-retrieval-execution/r5-04"] = (
        "eval-factory/public-source-retrieval-execution/r5-04"
    )
    intent_id: Identifier
    search_request: PublicSourceSearchRequestV2 | None = None
    search_result: PublicSourceSearchResultV2 | None = None
    fetch_request: PublicSourceFetchRequestV2
    fetch_result: PublicSourceFetchResultV2

    @model_validator(mode="after")
    def validate_execution(self) -> PublicSourceRetrievalExecution:
        if (self.search_request is None) != (self.search_result is None):
            raise ValueError("search request and result must be both present or both absent")
        if (
            self.search_request is not None
            and self.search_result is not None
            and self.search_result.search_request_ref != public_source_search_request_ref(self.search_request)
        ):
            raise ValueError("search result must bind the exact search request")
        if self.fetch_result.fetch_request_ref != public_source_fetch_request_ref(self.fetch_request):
            raise ValueError("fetch result must bind the exact fetch request")
        return self

    @property
    def fetch_result_ref(self) -> ObjectRef:
        return _factory_ref(public_source_fetch_result_ref(self.fetch_result))


class ApprovedSourceRetrievalExecutionResult(ContractModel):
    schema_version: Literal["eval-factory/approved-source-retrieval-execution-result/r5-04"] = (
        "eval-factory/approved-source-retrieval-execution-result/r5-04"
    )
    execution_result_id: Identifier
    request_ref: ObjectRef
    outcome: ApprovedSourceRetrievalExecutionOutcome
    executions: tuple[PublicSourceRetrievalExecution, ...] = ()
    unresolved_intent_ids: tuple[Identifier, ...] = ()
    reasons: frozenset[ApprovedSourceEvidenceReason] = frozenset()
    policy_version: Literal["public-source-retrieval/r5-04-v1"] = APPROVED_SOURCE_RETRIEVAL_POLICY_VERSION
    execution_result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> ApprovedSourceRetrievalExecutionOutcome:
        if isinstance(value, ApprovedSourceRetrievalExecutionOutcome):
            return value
        if isinstance(value, str):
            return ApprovedSourceRetrievalExecutionOutcome(value)
        raise TypeError("outcome must be an ApprovedSourceRetrievalExecutionOutcome")

    @field_validator("reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[ApprovedSourceEvidenceReason]:
        return _parse_reason_set(value)

    @model_validator(mode="after")
    def validate_result(self) -> ApprovedSourceRetrievalExecutionResult:
        _require_ref_type(
            self.request_ref,
            "approved-source-retrieval-request",
            "request_ref",
        )
        if self.unresolved_intent_ids != tuple(sorted(set(self.unresolved_intent_ids))):
            raise ValueError("unresolved intent IDs must be unique and sorted")
        execution_ids = tuple(item.intent_id for item in self.executions)
        _require_unique("retrieval executions", execution_ids)
        if execution_ids != tuple(sorted(execution_ids)):
            raise ValueError("retrieval executions must be sorted by intent ID")
        if self.outcome is ApprovedSourceRetrievalExecutionOutcome.EXECUTED:
            if not self.executions or self.unresolved_intent_ids or self.reasons:
                raise ValueError("EXECUTED requires executions and no unresolved state")
        elif self.executions or not self.unresolved_intent_ids or not self.reasons:
            raise ValueError(f"{self.outcome.value} must carry only unresolved safe diagnostics")
        return self


class VerifiedPublicSourceChecks(ContractModel):
    schema_version: Literal["eval-factory/verified-public-source-checks/r5-04"] = (
        "eval-factory/verified-public-source-checks/r5-04"
    )
    intent_id: Identifier
    fetch_result_ref: ObjectRef
    content_ref: ObjectRef
    content_sha256: Sha256
    secret_scan_ref: ObjectRef | None = None
    pii_scan_ref: ObjectRef | None = None
    prompt_injection_scan_ref: ObjectRef | None = None
    answer_leakage_scan_ref: ObjectRef | None = None
    license_assessment_ref: ObjectRef | None = None
    unavailable_checks: frozenset[PublicSourceCheck] = frozenset()
    failed_checks: frozenset[PublicSourceCheck] = frozenset()

    @field_validator("unavailable_checks", "failed_checks", mode="before")
    @classmethod
    def parse_checks(cls, value: object) -> frozenset[PublicSourceCheck]:
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise TypeError("checks must be a collection")
        return frozenset(PublicSourceCheck(item) for item in value)

    @model_validator(mode="after")
    def validate_checks(self) -> VerifiedPublicSourceChecks:
        _require_ref_version(
            self.fetch_result_ref,
            "public-source-fetch-result",
            "v2",
            "fetch_result_ref",
        )
        _require_ref_version(
            self.content_ref,
            "public-source-content",
            "v1",
            "content_ref",
        )
        if self.content_ref.object_sha256 != self.content_sha256:
            raise ValueError("content_ref must bind content_sha256")
        if self.unavailable_checks & self.failed_checks:
            raise ValueError("unavailable and failed check sets must be disjoint")
        refs = {
            PublicSourceCheck.SECRET: self.secret_scan_ref,
            PublicSourceCheck.CONFIGURED_PII: self.pii_scan_ref,
            PublicSourceCheck.PROMPT_INJECTION: (self.prompt_injection_scan_ref),
            PublicSourceCheck.ANSWER_LEAKAGE: self.answer_leakage_scan_ref,
            PublicSourceCheck.LICENSE: self.license_assessment_ref,
        }
        expected_types = {
            PublicSourceCheck.SECRET: "secret-scan-result",
            PublicSourceCheck.CONFIGURED_PII: "configured-pii-scan-result",
            PublicSourceCheck.PROMPT_INJECTION: ("prompt-injection-scan-result"),
            PublicSourceCheck.ANSWER_LEAKAGE: "answer-leakage-scan-result",
            PublicSourceCheck.LICENSE: "source-license-assessment",
        }
        for check, ref in refs.items():
            unavailable_or_failed = check in self.unavailable_checks or check in self.failed_checks
            if unavailable_or_failed:
                if ref is not None:
                    raise ValueError("failed or unavailable checks cannot carry evidence refs")
                continue
            if ref is None:
                raise ValueError("passed checks require exact evidence refs")
            _require_ref_type(ref, expected_types[check], f"{check.value}_ref")
            if ref.object_sha256 != self.content_sha256:
                raise ValueError("check evidence ref must bind content_sha256")
        return self


class ApprovedSourceEvidenceResult(ContractModel):
    schema_version: Literal["eval-factory/approved-source-evidence-result/r5-04"] = (
        "eval-factory/approved-source-evidence-result/r5-04"
    )
    result_id: Identifier
    request_ref: ObjectRef
    outcome: ApprovedSourceEvidenceOutcome
    source_evidence_set: SourceEvidenceSetV2 | None = None
    unresolved_intent_ids: tuple[Identifier, ...] = ()
    reasons: frozenset[ApprovedSourceEvidenceReason] = frozenset()
    policy_version: Literal["public-source-retrieval/r5-04-v1"] = APPROVED_SOURCE_RETRIEVAL_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> ApprovedSourceEvidenceOutcome:
        if isinstance(value, ApprovedSourceEvidenceOutcome):
            return value
        if isinstance(value, str):
            return ApprovedSourceEvidenceOutcome(value)
        raise TypeError("outcome must be an ApprovedSourceEvidenceOutcome")

    @field_validator("reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[ApprovedSourceEvidenceReason]:
        return _parse_reason_set(value)

    @model_validator(mode="after")
    def validate_result(self) -> ApprovedSourceEvidenceResult:
        _require_ref_type(
            self.request_ref,
            "approved-source-retrieval-request",
            "request_ref",
        )
        if self.unresolved_intent_ids != tuple(sorted(set(self.unresolved_intent_ids))):
            raise ValueError("unresolved intent IDs must be unique and sorted")
        if self.outcome is ApprovedSourceEvidenceOutcome.COMPILED:
            if self.source_evidence_set is None or self.unresolved_intent_ids or self.reasons:
                raise ValueError("COMPILED requires a set and no unresolved state")
        elif self.outcome is ApprovedSourceEvidenceOutcome.NOT_REQUIRED:
            if self.source_evidence_set is not None or self.unresolved_intent_ids or self.reasons:
                raise ValueError("NOT_REQUIRED cannot carry evidence or unresolved state")
        elif self.source_evidence_set is not None or not self.unresolved_intent_ids or not self.reasons:
            raise ValueError(f"{self.outcome.value} must carry only unresolved safe diagnostics")
        return self


def approved_source_retrieval_request_carried_sha256(
    request: ApprovedSourceRetrievalRequest,
) -> str:
    return _payload_sha256(
        request.model_dump(
            mode="json",
            exclude={"request_id", "request_sha256", "audit"},
            exclude_none=False,
        )
    )


def approved_source_retrieval_request_ref(
    request: ApprovedSourceRetrievalRequest,
) -> ObjectRef:
    return ObjectRef(
        object_type="approved-source-retrieval-request",
        object_id=request.request_id,
        object_version="r5-04",
        object_sha256=request.request_sha256,
    )


def approved_source_execution_result_carried_sha256(
    result: ApprovedSourceRetrievalExecutionResult,
) -> str:
    return _payload_sha256(
        result.model_dump(
            mode="json",
            exclude={
                "execution_result_id",
                "execution_result_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def approved_source_evidence_result_carried_sha256(
    result: ApprovedSourceEvidenceResult,
) -> str:
    return _payload_sha256(
        {
            "request_ref": _ref_payload(result.request_ref),
            "outcome": result.outcome.value,
            "source_evidence_set_ref": (
                None
                if result.source_evidence_set is None
                else {
                    "object_type": "source-evidence-set",
                    "object_id": (result.source_evidence_set.source_evidence_set_id),
                    "object_version": "v2",
                    "object_sha256": (result.source_evidence_set.source_evidence_set_sha256),
                }
            ),
            "unresolved_intent_ids": list(result.unresolved_intent_ids),
            "reasons": sorted(item.value for item in result.reasons),
            "policy_version": result.policy_version,
        }
    )


def _factory_ref(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef.model_validate(ref.model_dump(mode="python"))


def _parse_reason_set(
    value: object,
) -> frozenset[ApprovedSourceEvidenceReason]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise TypeError("reasons must be a collection")
    return frozenset(ApprovedSourceEvidenceReason(item) for item in value)


def _require_ref_type(
    ref: ObjectRef,
    expected_type: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type:
        raise ValueError(f"{field_name} must reference {expected_type}")


def _require_ref_version(
    ref: ObjectRef,
    expected_type: str,
    expected_version: str,
    field_name: str,
) -> None:
    _require_ref_type(ref, expected_type, field_name)
    if ref.object_version != expected_version:
        raise ValueError(f"{field_name} must reference {expected_type} {expected_version}")


def _require_unique(label: str, values: tuple[object, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
