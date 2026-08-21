import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, Field, StringConstraints, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2

FailureCode = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{1,127}$")]


class ModelRouteRejectionCodeV2(StrEnum):
    GOVERNANCE_AGENT_NOT_ALLOWED = "GOVERNANCE_AGENT_NOT_ALLOWED"
    GOVERNANCE_MODEL_NOT_ALLOWED = "GOVERNANCE_MODEL_NOT_ALLOWED"
    GOVERNANCE_DATA_CLASSIFICATION = "GOVERNANCE_DATA_CLASSIFICATION"
    GOVERNANCE_RESIDENCY = "GOVERNANCE_RESIDENCY"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    CONTEXT_LIMIT = "CONTEXT_LIMIT"
    OUTPUT_LIMIT = "OUTPUT_LIMIT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    QUALITY_BASELINE_MISSING = "QUALITY_BASELINE_MISSING"
    QUALITY_BELOW_MINIMUM = "QUALITY_BELOW_MINIMUM"
    HEALTH_SNAPSHOT_MISSING = "HEALTH_SNAPSHOT_MISSING"
    PRICE_SCHEDULE_MISSING = "PRICE_SCHEDULE_MISSING"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    GENERATOR_JUDGE_COLLISION = "GENERATOR_JUDGE_COLLISION"
    PREDECESSOR_MODEL_EXCLUDED = "PREDECESSOR_MODEL_EXCLUDED"


class GatewayInvocationStatusV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BACKPRESSURED = "BACKPRESSURED"
    FAILED = "FAILED"


class _GatewayObjectV2(ContractModelV2):
    object_id: Identifier
    object_sha256: Sha256
    audit: ContractAudit

    OBJECT_TYPE: ClassVar[str]
    OBJECT_VERSION: ClassVar[str] = "v2"

    @classmethod
    def create(cls, *, audit: ContractAudit, **values: object) -> Self:
        provisional = cls.model_construct(
            object_id=f"{cls.OBJECT_TYPE}://pending",
            object_sha256="0" * 64,
            audit=audit,
            **values,  # type: ignore[arg-type]
        )
        safe_audit = ContractAudit(
            created_at=audit.created_at,
            created_by=audit.created_by,
            governing_versions=audit.governing_versions,
            input_refs=_collect_refs(provisional),
        )
        provisional = cls.model_construct(
            object_id=f"{cls.OBJECT_TYPE}://pending",
            object_sha256="0" * 64,
            audit=safe_audit,
            **values,  # type: ignore[arg-type]
        )
        digest = _carried_sha256(provisional)
        return cls(
            object_id=f"{cls.OBJECT_TYPE}://sha256/{digest}",
            object_sha256=digest,
            audit=safe_audit,
            **values,
        )

    @model_validator(mode="after")
    def validate_derived_identity(self) -> Self:
        digest = _carried_sha256(self)
        if self.object_sha256 != digest or self.object_id != f"{self.OBJECT_TYPE}://sha256/{digest}":
            raise ValueError("Gateway object fields do not match the derived identity")
        if self.audit.input_refs != _collect_refs(self):
            raise ValueError("Gateway audit refs do not match object refs")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type=self.OBJECT_TYPE,
            object_id=self.object_id,
            object_version=self.OBJECT_VERSION,
            object_sha256=self.object_sha256,
        )


class ModelCapabilityProfileV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/model-capability-profile/v2"] = (
        "eval-factory/model-capability-profile/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "model-capability-profile"

    model_profile_id: Identifier
    provider_id: Identifier
    model_id: Identifier
    model_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    provider_model_profile_ref: ObjectRef | None = None
    capabilities: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    supported_data_classifications: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    supported_residencies: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    context_limit_tokens: int = Field(ge=1, le=10_000_000)
    output_limit_tokens: int = Field(ge=1, le=1_000_000)
    availability: Literal["AVAILABLE", "DEGRADED", "UNAVAILABLE"]
    catalog_version: Literal["model-catalog/v1"] = "model-catalog/v1"

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        for values, label in (
            (self.capabilities, "capabilities"),
            (self.supported_data_classifications, "supported_data_classifications"),
            (self.supported_residencies, "supported_residencies"),
        ):
            _require_sorted_unique(values, label)
        if self.output_limit_tokens > self.context_limit_tokens:
            raise ValueError("model output limit cannot exceed context limit")
        if self.provider_model_profile_ref is not None and (
            self.provider_model_profile_ref.object_type != "model-execution-profile"
            or self.provider_model_profile_ref.object_version != "v1"
        ):
            raise ValueError("provider model profile ref must remain facade-owned v1")
        return self


class PromptTemplateV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/prompt-template/v2"] = "eval-factory/prompt-template/v2"
    OBJECT_TYPE: ClassVar[str] = "prompt-template"

    prompt_template_id: Identifier
    agent_role: Identifier
    task_kind: Identifier
    system_template_ref: ObjectRef
    instruction_template_ref: ObjectRef
    required_input_object_types: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    output_schema_ref: ObjectRef
    allowed_tool_ids: tuple[Identifier, ...] = Field(default=(), max_length=256)
    required_model_capabilities: tuple[Identifier, ...] = Field(default=(), max_length=256)
    injection_policy_ref: ObjectRef
    template_version: int = Field(ge=1, le=1_000_000)
    predecessor_template_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_template(self) -> Self:
        _require_ref(self.system_template_ref, "prompt-template-content", "system_template_ref")
        _require_ref(
            self.instruction_template_ref,
            "prompt-template-content",
            "instruction_template_ref",
        )
        _require_ref(self.output_schema_ref, "json-schema", "output_schema_ref")
        _require_ref(
            self.injection_policy_ref,
            "prompt-injection-policy",
            "injection_policy_ref",
        )
        _require_sorted_unique(
            self.required_input_object_types,
            "required_input_object_types",
        )
        _require_sorted_unique(self.allowed_tool_ids, "allowed_tool_ids")
        _require_sorted_unique(
            self.required_model_capabilities,
            "required_model_capabilities",
        )
        if self.template_version == 1 and self.predecessor_template_ref is not None:
            raise ValueError("first prompt template cannot have a predecessor")
        if self.template_version > 1:
            if self.predecessor_template_ref is None:
                raise ValueError("successor prompt template requires a predecessor")
            _require_ref(
                self.predecessor_template_ref,
                "prompt-template",
                "predecessor_template_ref",
            )
        return self


class ModelQualityBaselineV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/model-quality-baseline/v2"] = (
        "eval-factory/model-quality-baseline/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "model-quality-baseline"

    baseline_id: Identifier
    model_profile_ref: ObjectRef
    task_kind: Identifier
    benchmark_ref: ObjectRef
    quality_basis_points: int = Field(ge=0, le=10_000)
    minimum_sample_count: int = Field(ge=1, le=10_000_000)

    @model_validator(mode="after")
    def validate_baseline(self) -> Self:
        _require_ref(
            self.model_profile_ref,
            "model-capability-profile",
            "model_profile_ref",
        )
        _require_ref(self.benchmark_ref, "model-quality-benchmark", "benchmark_ref")
        return self


class ModelHealthSnapshotV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/model-health-snapshot/v2"] = "eval-factory/model-health-snapshot/v2"
    OBJECT_TYPE: ClassVar[str] = "model-health-snapshot"

    health_snapshot_id: Identifier
    model_profile_ref: ObjectRef
    observed_at: datetime
    availability: Literal["HEALTHY", "DEGRADED", "UNAVAILABLE"]
    success_basis_points: int = Field(ge=0, le=10_000)
    latency_milliseconds: int = Field(ge=0, le=86_400_000)

    @model_validator(mode="after")
    def validate_health(self) -> Self:
        _require_ref(
            self.model_profile_ref,
            "model-capability-profile",
            "model_profile_ref",
        )
        _require_aware(self.observed_at, "observed_at")
        return self


class ModelPriceScheduleV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/model-price-schedule/v2"] = "eval-factory/model-price-schedule/v2"
    OBJECT_TYPE: ClassVar[str] = "model-price-schedule"

    price_schedule_id: Identifier
    model_profile_ref: ObjectRef
    input_micro_usd_per_million_tokens: int = Field(ge=0, le=10_000_000_000_000)
    output_micro_usd_per_million_tokens: int = Field(ge=0, le=10_000_000_000_000)
    effective_from: datetime
    effective_until: datetime | None = None

    @model_validator(mode="after")
    def validate_price(self) -> Self:
        _require_ref(
            self.model_profile_ref,
            "model-capability-profile",
            "model_profile_ref",
        )
        _require_aware(self.effective_from, "effective_from")
        if self.effective_until is not None:
            _require_aware(self.effective_until, "effective_until")
            if self.effective_until <= self.effective_from:
                raise ValueError("price schedule window is invalid")
        return self


class ModelRoutingPolicyV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/model-routing-policy/v2"] = "eval-factory/model-routing-policy/v2"
    OBJECT_TYPE: ClassVar[str] = "model-routing-policy"

    routing_policy_id: Identifier
    allowed_provider_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    allowed_agent_definition_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    allowed_model_profile_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    quality_weight: int = Field(ge=0, le=1_000_000)
    health_weight: int = Field(ge=0, le=1_000_000)
    latency_weight: int = Field(ge=0, le=1_000_000)
    cost_weight: int = Field(ge=0, le=1_000_000)
    policy_version: Literal["model-routing/v1"] = "model-routing/v1"

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_sorted_unique(self.allowed_provider_ids, "allowed_provider_ids")
        _require_sorted_unique_refs(
            self.allowed_agent_definition_refs,
            "allowed_agent_definition_refs",
        )
        if any(ref.object_type != "agent-definition" for ref in self.allowed_agent_definition_refs):
            raise ValueError("routing policy Agent allowlist contains an invalid ref")
        _require_sorted_unique_refs(
            self.allowed_model_profile_refs,
            "allowed_model_profile_refs",
        )
        if not any(
            (
                self.quality_weight,
                self.health_weight,
                self.latency_weight,
                self.cost_weight,
            )
        ):
            raise ValueError("routing policy requires at least one score weight")
        return self


class ModelRouteRequestV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/model-route-request/v2"] = "eval-factory/model-route-request/v2"
    OBJECT_TYPE: ClassVar[str] = "model-route-request"

    route_request_id: Identifier
    agent_task_ref: ObjectRef
    agent_definition_ref: ObjectRef
    task_kind: Identifier
    prompt_template_ref: ObjectRef
    required_capabilities: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    data_classification: Identifier
    residency: Identifier
    input_token_budget: int = Field(ge=0, le=10_000_000)
    output_token_budget: int = Field(ge=1, le=1_000_000)
    max_cost_micro_usd: int = Field(ge=0, le=10_000_000_000_000)
    minimum_quality_basis_points: int = Field(ge=0, le=10_000)
    allowed_model_profile_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    budget_reservation_ref: ObjectRef
    route_version: int = Field(ge=1, le=1_000_000)
    predecessor_route_ref: ObjectRef | None
    failed_receipt_ref: ObjectRef | None
    generator_model_profile_ref: ObjectRef | None

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_execution_subject_ref(self.agent_task_ref, "agent_task_ref")
        _require_ref(
            self.agent_definition_ref,
            "agent-definition",
            "agent_definition_ref",
        )
        _require_ref(
            self.prompt_template_ref,
            "prompt-template",
            "prompt_template_ref",
        )
        _require_sorted_unique(self.required_capabilities, "required_capabilities")
        _require_sorted_unique_refs(
            self.allowed_model_profile_refs,
            "allowed_model_profile_refs",
        )
        _require_ref(
            self.budget_reservation_ref,
            "work-model-reservation",
            "budget_reservation_ref",
        )
        if self.route_version == 1:
            if self.predecessor_route_ref is not None or self.failed_receipt_ref is not None:
                raise ValueError("first route cannot bind predecessor failure evidence")
        else:
            if self.predecessor_route_ref is None or self.failed_receipt_ref is None:
                raise ValueError("successor route requires predecessor and failure receipt")
            _require_ref(
                self.predecessor_route_ref,
                "model-route-decision",
                "predecessor_route_ref",
            )
            _require_ref(
                self.failed_receipt_ref,
                "gateway-receipt",
                "failed_receipt_ref",
            )
        if self.generator_model_profile_ref is not None:
            _require_ref(
                self.generator_model_profile_ref,
                "model-capability-profile",
                "generator_model_profile_ref",
            )
        return self


class ModelRouteCandidateV2(ContractModelV2):
    schema_version: Literal["eval-factory/model-route-candidate/v2"] = "eval-factory/model-route-candidate/v2"
    model_profile_ref: ObjectRef
    eligible: bool
    rejection_codes: tuple[ModelRouteRejectionCodeV2, ...]
    quality_basis_points: int | None = Field(default=None, ge=0, le=10_000)
    health_basis_points: int | None = Field(default=None, ge=0, le=10_000)
    latency_milliseconds: int | None = Field(default=None, ge=0, le=86_400_000)
    estimated_cost_micro_usd: int | None = Field(
        default=None,
        ge=0,
        le=10_000_000_000_000,
    )
    score: int | None

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        _require_ref(
            self.model_profile_ref,
            "model-capability-profile",
            "model_profile_ref",
        )
        _require_unique(
            tuple(code.value for code in self.rejection_codes),
            "candidate rejection codes",
        )
        facts = (
            self.quality_basis_points,
            self.health_basis_points,
            self.latency_milliseconds,
            self.estimated_cost_micro_usd,
            self.score,
        )
        if self.eligible:
            if self.rejection_codes or any(value is None for value in facts):
                raise ValueError("eligible candidate requires complete score facts")
        elif not self.rejection_codes or any(value is not None for value in facts):
            raise ValueError("rejected candidate requires closed reasons only")
        return self


class ModelRouteDecisionV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/model-route-decision/v2"] = "eval-factory/model-route-decision/v2"
    OBJECT_TYPE: ClassVar[str] = "model-route-decision"

    route_decision_id: Identifier
    request_ref: ObjectRef
    route_version: int = Field(ge=1, le=1_000_000)
    predecessor_route_ref: ObjectRef | None
    selected_model_profile_ref: ObjectRef
    candidates: tuple[ModelRouteCandidateV2, ...] = Field(min_length=1, max_length=1_000)
    routing_policy_ref: ObjectRef
    quality_baseline_ref: ObjectRef
    health_snapshot_ref: ObjectRef
    price_schedule_ref: ObjectRef
    budget_reservation_ref: ObjectRef
    no_silent_fallback: Literal[True] = True

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        _require_ref(self.request_ref, "model-route-request", "request_ref")
        if self.route_version == 1 and self.predecessor_route_ref is not None:
            raise ValueError("first route decision cannot have a predecessor")
        if self.route_version > 1:
            if self.predecessor_route_ref is None:
                raise ValueError("successor route decision requires a predecessor")
            _require_ref(
                self.predecessor_route_ref,
                "model-route-decision",
                "predecessor_route_ref",
            )
        for ref, object_type, label in (
            (self.routing_policy_ref, "model-routing-policy", "routing_policy_ref"),
            (
                self.quality_baseline_ref,
                "model-quality-baseline",
                "quality_baseline_ref",
            ),
            (
                self.health_snapshot_ref,
                "model-health-snapshot",
                "health_snapshot_ref",
            ),
            (
                self.price_schedule_ref,
                "model-price-schedule",
                "price_schedule_ref",
            ),
            (
                self.budget_reservation_ref,
                "work-model-reservation",
                "budget_reservation_ref",
            ),
        ):
            _require_ref(ref, object_type, label)
        keys = tuple(_ref_key(item.model_profile_ref) for item in self.candidates)
        if len(set(keys)) != len(keys):
            raise ValueError("route candidate model profiles must be unique")
        selected = [
            item for item in self.candidates if item.model_profile_ref == self.selected_model_profile_ref
        ]
        if len(selected) != 1 or not selected[0].eligible:
            raise ValueError("selected route must bind one eligible candidate")
        return self


class RAGSourcePolicyV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/rag-source-policy/v2"] = "eval-factory/rag-source-policy/v2"
    OBJECT_TYPE: ClassVar[str] = "rag-source-policy"

    source_policy_id: Identifier
    allowed_principal_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=1_000)
    allowed_purposes: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    allowed_source_classes: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    projection_policy_ref: ObjectRef
    max_results: int = Field(ge=1, le=1_000)
    max_bytes: int = Field(ge=1, le=100_000_000)
    max_characters: int = Field(ge=1, le=100_000_000)

    @model_validator(mode="after")
    def validate_source_policy(self) -> Self:
        for values, label in (
            (self.allowed_principal_ids, "allowed_principal_ids"),
            (self.allowed_purposes, "allowed_purposes"),
            (self.allowed_source_classes, "allowed_source_classes"),
        ):
            _require_sorted_unique(values, label)
        _require_ref(
            self.projection_policy_ref,
            "projection-policy",
            "projection_policy_ref",
        )
        return self


class RAGRequestV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/rag-request/v2"] = "eval-factory/rag-request/v2"
    OBJECT_TYPE: ClassVar[str] = "rag-request"

    rag_request_id: Identifier
    agent_task_ref: ObjectRef
    principal_id: Identifier
    purpose: Identifier
    query_ref: ObjectRef
    query_sha256: Sha256
    source_policy_ref: ObjectRef
    requested_source_classes: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    max_results: int = Field(ge=1, le=1_000)
    max_bytes: int = Field(ge=1, le=100_000_000)
    max_characters: int = Field(ge=1, le=100_000_000)

    @model_validator(mode="after")
    def validate_rag_request(self) -> Self:
        _require_ref(self.agent_task_ref, "agent-task", "agent_task_ref")
        _require_ref(self.query_ref, "rag-query-content", "query_ref")
        if self.query_ref.object_sha256 != self.query_sha256:
            raise ValueError("RAG query digest differs from private query ref")
        _require_ref(self.source_policy_ref, "rag-source-policy", "source_policy_ref")
        _require_sorted_unique(
            self.requested_source_classes,
            "requested_source_classes",
        )
        return self


class RAGResultV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/rag-result/v2"] = "eval-factory/rag-result/v2"
    OBJECT_TYPE: ClassVar[str] = "rag-result"

    rag_result_id: Identifier
    request_ref: ObjectRef
    approved_content_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=1_000)
    provenance_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=1_000)
    excluded_reason_codes: tuple[FailureCode, ...] = Field(default=(), max_length=256)
    result_count: int = Field(ge=0, le=1_000)
    total_bytes: int = Field(ge=0, le=100_000_000)
    total_characters: int = Field(ge=0, le=100_000_000)

    @model_validator(mode="after")
    def validate_rag_result(self) -> Self:
        _require_ref(self.request_ref, "rag-request", "request_ref")
        _require_sorted_unique_refs(self.approved_content_refs, "approved_content_refs")
        _require_sorted_unique_refs(self.provenance_refs, "provenance_refs")
        _require_sorted_unique(self.excluded_reason_codes, "excluded_reason_codes")
        if self.result_count != len(self.approved_content_refs):
            raise ValueError("RAG result count differs from approved content refs")
        if self.result_count != len(self.provenance_refs):
            raise ValueError("RAG result requires one provenance ref per content ref")
        if self.result_count == 0 and (self.total_bytes or self.total_characters):
            raise ValueError("empty RAG result cannot report content size")
        return self


class GatewayInvocationRequestV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/gateway-invocation-request/v2"] = (
        "eval-factory/gateway-invocation-request/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "gateway-invocation-request"

    invocation_request_id: Identifier
    agent_task_ref: ObjectRef
    route_decision_ref: ObjectRef
    prompt_template_ref: ObjectRef
    prompt_rendering_ref: ObjectRef
    output_schema_ref: ObjectRef
    rag_result_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=256)
    idempotency_key: Identifier

    @model_validator(mode="after")
    def validate_invocation_request(self) -> Self:
        _require_execution_subject_ref(self.agent_task_ref, "agent_task_ref")
        for ref, object_type, label in (
            (
                self.route_decision_ref,
                "model-route-decision",
                "route_decision_ref",
            ),
            (self.prompt_template_ref, "prompt-template", "prompt_template_ref"),
            (self.prompt_rendering_ref, "prompt-rendering", "prompt_rendering_ref"),
            (self.output_schema_ref, "json-schema", "output_schema_ref"),
        ):
            _require_ref(ref, object_type, label)
        _require_sorted_unique_refs(self.rag_result_refs, "rag_result_refs")
        if any(ref.object_type != "rag-result" for ref in self.rag_result_refs):
            raise ValueError("invocation RAG refs must reference rag-result/v2")
        return self


class GatewayUsageV2(ContractModelV2):
    schema_version: Literal["eval-factory/gateway-usage/v2"] = "eval-factory/gateway-usage/v2"
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)
    charged_tokens: int = Field(ge=0)
    reported_cost_micro_usd: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_usage(self) -> Self:
        expected = (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )
        if self.charged_tokens != expected:
            raise ValueError("charged tokens differ from normalized usage")
        return self


class GatewayReceiptV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/gateway-receipt/v2"] = "eval-factory/gateway-receipt/v2"
    OBJECT_TYPE: ClassVar[str] = "gateway-receipt"

    receipt_id: Identifier
    invocation_request_ref: ObjectRef
    route_decision_ref: ObjectRef
    status: GatewayInvocationStatusV2
    response_body_ref: ObjectRef | None
    usage: GatewayUsageV2
    failure_code: FailureCode | None
    completed_at: datetime

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        _require_ref(
            self.invocation_request_ref,
            "gateway-invocation-request",
            "invocation_request_ref",
        )
        _require_ref(
            self.route_decision_ref,
            "model-route-decision",
            "route_decision_ref",
        )
        _require_aware(self.completed_at, "completed_at")
        if self.status is GatewayInvocationStatusV2.SUCCEEDED:
            if self.response_body_ref is None or self.failure_code is not None:
                raise ValueError("successful receipt requires private response ref only")
            _require_ref(
                self.response_body_ref,
                "model-response-content",
                "response_body_ref",
            )
        elif self.response_body_ref is not None or self.failure_code is None:
            raise ValueError("failed receipt requires failure code and no response body ref")
        return self


class GatewayInvocationResultV2(_GatewayObjectV2):
    schema_version: Literal["eval-factory/gateway-invocation-result/v2"] = (
        "eval-factory/gateway-invocation-result/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "gateway-invocation-result"

    invocation_result_id: Identifier
    invocation_request_ref: ObjectRef
    route_decision_ref: ObjectRef
    receipt_ref: ObjectRef
    status: GatewayInvocationStatusV2
    output_ref: ObjectRef | None
    failure_code: FailureCode | None

    @model_validator(mode="after")
    def validate_invocation_result(self) -> Self:
        _require_ref(
            self.invocation_request_ref,
            "gateway-invocation-request",
            "invocation_request_ref",
        )
        _require_ref(
            self.route_decision_ref,
            "model-route-decision",
            "route_decision_ref",
        )
        _require_ref(self.receipt_ref, "gateway-receipt", "receipt_ref")
        if self.status is GatewayInvocationStatusV2.SUCCEEDED:
            if self.output_ref is None or self.failure_code is not None:
                raise ValueError("successful invocation result requires output only")
        elif self.output_ref is not None or self.failure_code is None:
            raise ValueError("failed invocation result requires failure code only")
        return self


def _carried_sha256(value: _GatewayObjectV2) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={"schema_version", "object_id", "object_sha256", "audit"},
    )
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _collect_refs(value: BaseModel) -> tuple[ObjectRef, ...]:
    refs: list[ObjectRef] = []

    def visit(item: object) -> None:
        if isinstance(item, ObjectRef):
            refs.append(item)
        elif isinstance(item, BaseModel):
            for field_name in type(item).model_fields:
                if field_name in {"audit", "object_id", "object_sha256"}:
                    continue
                visit(getattr(item, field_name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, tuple | list | set | frozenset):
            for child in item:
                visit(child)

    visit(value)
    unique: list[ObjectRef] = []
    seen: set[tuple[str, str, str, str]] = set()
    for ref in refs:
        key = _ref_key(ref)
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    return tuple(unique)


def _require_ref(value: ObjectRef, object_type: str, label: str) -> None:
    if value.object_type != object_type or value.object_version != "v2":
        raise ValueError(f"{label} must reference {object_type}/v2")


def _require_execution_subject_ref(value: ObjectRef, label: str) -> None:
    if (
        value.object_type,
        value.object_version,
    ) not in {
        ("agent-task", "v2"),
        ("harness-turn", "v1"),
    }:
        raise ValueError(f"{label} must reference agent-task/v2 or harness-turn/v1")


def _require_sorted_unique(values: tuple[str, ...], label: str) -> None:
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError(f"{label} must be sorted and unique")


def _require_unique(values: tuple[str, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _require_sorted_unique_refs(values: tuple[ObjectRef, ...], label: str) -> None:
    keys = tuple(_ref_key(ref) for ref in values)
    if tuple(sorted(keys)) != keys or len(set(keys)) != len(keys):
        raise ValueError(f"{label} must be sorted and unique")


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _require_aware(value: datetime, label: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


__all__ = [
    "GatewayInvocationRequestV2",
    "GatewayInvocationResultV2",
    "GatewayInvocationStatusV2",
    "GatewayReceiptV2",
    "GatewayUsageV2",
    "ModelCapabilityProfileV2",
    "ModelHealthSnapshotV2",
    "ModelPriceScheduleV2",
    "ModelQualityBaselineV2",
    "ModelRouteCandidateV2",
    "ModelRouteDecisionV2",
    "ModelRouteRejectionCodeV2",
    "ModelRouteRequestV2",
    "ModelRoutingPolicyV2",
    "PromptTemplateV2",
    "RAGRequestV2",
    "RAGResultV2",
    "RAGSourcePolicyV2",
]
