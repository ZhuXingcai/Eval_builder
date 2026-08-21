from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from env_mock_agent.facade import (
    AttachmentRouteDecisionV2,
    AttachmentRouteRequestV2,
    CapabilityToken,
    attachment_route_request_ref,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactBuildContractV2,
    ArtifactRoutingAggregateOutcomeV2,
    ArtifactRoutingPlanV2,
    artifact_build_contract_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import canonical_value_v2

ARTIFACT_ROUTING_POLICY_VERSION: Literal["artifact-routing/r5-05-v1"] = "artifact-routing/r5-05-v1"


class ArtifactRoutingPolicyError(RuntimeError):
    pass


class ArtifactRoutingExecutionOutcome(StrEnum):
    COMPLETED = "COMPLETED"
    NOT_REQUIRED = "NOT_REQUIRED"


class ArtifactBuildContractDefinition(ContractModel):
    schema_version: Literal["eval-factory/artifact-build-contract-definition/r5-05"] = (
        "eval-factory/artifact-build-contract-definition/r5-05"
    )
    definition_id: Identifier
    attachment_dependency_id: Identifier
    asset_type: CapabilityToken
    content_contract_ref: ObjectRef
    render_contract_ref: ObjectRef
    provider_payload_ref: ObjectRef | None = None
    source_evidence_set_ref: ObjectRef | None = None
    required_provider_capability_ids: tuple[Identifier, ...]
    required_runtime_tools: tuple[CapabilityToken, ...]
    runtime_role: Identifier
    runtime_resume_required: bool
    validator_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_definition(self) -> ArtifactBuildContractDefinition:
        _require_ref(
            self.content_contract_ref,
            "artifact-content-contract",
            "v2",
            "content_contract_ref",
        )
        _require_ref(
            self.render_contract_ref,
            "artifact-render-contract",
            "v2",
            "render_contract_ref",
        )
        if self.provider_payload_ref is not None:
            _require_ref(
                self.provider_payload_ref,
                "attachment-provider-payload",
                "v2",
                "provider_payload_ref",
            )
        if self.source_evidence_set_ref is not None:
            _require_ref(
                self.source_evidence_set_ref,
                "source-evidence-set",
                "v2",
                "source_evidence_set_ref",
            )
        for label, values in (
            (
                "required provider capability IDs",
                self.required_provider_capability_ids,
            ),
            ("required runtime tools", self.required_runtime_tools),
            ("validator IDs", self.validator_ids),
        ):
            _require_sorted_unique(label, values)
        return self


class ArtifactRoutingRequest(ContractModel):
    schema_version: Literal["eval-factory/artifact-routing-request/r5-05"] = (
        "eval-factory/artifact-routing-request/r5-05"
    )
    request_id: Identifier
    attachment_planning_context_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    artifact_evidence_matrix_ref: ObjectRef | None
    artifact_routing_policy_ref: ObjectRef
    build_contracts: tuple[ArtifactBuildContractV2, ...]
    facade_requests: tuple[AttachmentRouteRequestV2, ...]
    missing_contract_target_refs: tuple[ObjectRef, ...]
    blocked_mode_target_refs: tuple[ObjectRef, ...]
    policy_version: Literal["artifact-routing/r5-05-v1"] = ARTIFACT_ROUTING_POLICY_VERSION
    request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> ArtifactRoutingRequest:
        _require_ref(
            self.attachment_planning_context_ref,
            "attachment-planning-context",
            "v2",
            "attachment_planning_context_ref",
        )
        _require_ref(
            self.producer_task_view_ref,
            "producer-task-view",
            "v2",
            "producer_task_view_ref",
        )
        _require_ref(
            self.artifact_routing_policy_ref,
            "artifact-routing-policy",
            "v2",
            "artifact_routing_policy_ref",
        )
        if self.artifact_evidence_matrix_ref is None:
            if (
                self.build_contracts
                or self.facade_requests
                or self.missing_contract_target_refs
                or self.blocked_mode_target_refs
            ):
                raise ValueError("no-requirements routing request cannot contain matrix work")
            return self
        _require_ref(
            self.artifact_evidence_matrix_ref,
            "artifact-evidence-matrix",
            "v2",
            "artifact_evidence_matrix_ref",
        )
        contract_ids = tuple(item.artifact_build_contract_id for item in self.build_contracts)
        if len(contract_ids) != len(set(contract_ids)):
            raise ValueError("artifact build contract IDs must be unique")
        artifact_order = tuple((item.logical_path, item.artifact_id) for item in self.build_contracts)
        _require_sorted_unique(
            "artifact build contract path/artifact pairs",
            artifact_order,
        )
        request_ids = tuple(item.route_request_id for item in self.facade_requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("facade route request IDs must be unique")
        if len(self.build_contracts) != len(self.facade_requests):
            raise ValueError("facade requests must exactly cover artifact build contracts")
        for contract, facade_request in zip(
            self.build_contracts,
            self.facade_requests,
            strict=True,
        ):
            expected_ref = ObjectRef(
                object_type="artifact-build-contract",
                object_id=contract.artifact_build_contract_id,
                object_version="v2",
                object_sha256=contract.artifact_build_contract_sha256,
            )
            observed = facade_request.build_contract_ref
            if (
                observed.object_type,
                observed.object_id,
                observed.object_version,
                observed.object_sha256,
            ) != (
                expected_ref.object_type,
                expected_ref.object_id,
                expected_ref.object_version,
                expected_ref.object_sha256,
            ):
                raise ValueError("facade request must bind the exact artifact build contract")
        for label, refs in (
            (
                "missing contract target refs",
                self.missing_contract_target_refs,
            ),
            ("blocked mode target refs", self.blocked_mode_target_refs),
        ):
            keys = tuple(_ref_key(ref) for ref in refs)
            _require_sorted_unique(label, keys)
            for ref in refs:
                _require_ref(
                    ref,
                    "artifact-evidence-target",
                    "v2",
                    label,
                )
        if {_ref_key(ref) for ref in self.missing_contract_target_refs} & {
            _ref_key(ref) for ref in self.blocked_mode_target_refs
        }:
            raise ValueError("missing-contract and blocked-mode target refs must be disjoint")
        return self


class ArtifactRoutingExecution(ContractModel):
    schema_version: Literal["eval-factory/artifact-routing-execution/r5-05"] = (
        "eval-factory/artifact-routing-execution/r5-05"
    )
    facade_request: AttachmentRouteRequestV2
    facade_decision: AttachmentRouteDecisionV2

    @model_validator(mode="after")
    def validate_execution(self) -> ArtifactRoutingExecution:
        request_ref = attachment_route_request_ref(self.facade_request)
        decision_ref = self.facade_decision.route_request_ref
        if (
            request_ref.object_type,
            request_ref.object_id,
            request_ref.object_version,
            request_ref.object_sha256,
        ) != (
            decision_ref.object_type,
            decision_ref.object_id,
            decision_ref.object_version,
            decision_ref.object_sha256,
        ):
            raise ValueError("facade decision must bind the exact route request")
        return self


class ArtifactRoutingExecutionResult(ContractModel):
    schema_version: Literal["eval-factory/artifact-routing-execution-result/r5-05"] = (
        "eval-factory/artifact-routing-execution-result/r5-05"
    )
    result_id: Identifier
    request_ref: ObjectRef
    outcome: ArtifactRoutingExecutionOutcome
    executions: tuple[ArtifactRoutingExecution, ...]
    policy_version: Literal["artifact-routing/r5-05-v1"] = ARTIFACT_ROUTING_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> ArtifactRoutingExecutionOutcome:
        if isinstance(value, ArtifactRoutingExecutionOutcome):
            return value
        if isinstance(value, str):
            return ArtifactRoutingExecutionOutcome(value)
        raise TypeError("outcome must be an ArtifactRoutingExecutionOutcome")

    @model_validator(mode="after")
    def validate_result(self) -> ArtifactRoutingExecutionResult:
        _require_ref(
            self.request_ref,
            "artifact-routing-request",
            "r5-05",
            "request_ref",
        )
        request_ids = tuple(item.facade_request.route_request_id for item in self.executions)
        _require_sorted_unique("routing execution request IDs", request_ids)
        if self.outcome is ArtifactRoutingExecutionOutcome.NOT_REQUIRED and self.executions:
            raise ValueError("NOT_REQUIRED cannot contain routing executions")
        return self


class ArtifactRoutingCompilationResult(ContractModel):
    schema_version: Literal["eval-factory/artifact-routing-compilation-result/r5-05"] = (
        "eval-factory/artifact-routing-compilation-result/r5-05"
    )
    result_id: Identifier
    request_ref: ObjectRef
    execution_result_ref: ObjectRef
    outcome: ArtifactRoutingAggregateOutcomeV2
    routing_plan: ArtifactRoutingPlanV2 | None
    policy_version: Literal["artifact-routing/r5-05-v1"] = ARTIFACT_ROUTING_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> ArtifactRoutingAggregateOutcomeV2:
        if isinstance(value, ArtifactRoutingAggregateOutcomeV2):
            return value
        if isinstance(value, str):
            return ArtifactRoutingAggregateOutcomeV2(value)
        raise TypeError("outcome must be an ArtifactRoutingAggregateOutcomeV2")

    @model_validator(mode="after")
    def validate_result(self) -> ArtifactRoutingCompilationResult:
        _require_ref(
            self.request_ref,
            "artifact-routing-request",
            "r5-05",
            "request_ref",
        )
        _require_ref(
            self.execution_result_ref,
            "artifact-routing-execution-result",
            "r5-05",
            "execution_result_ref",
        )
        if self.outcome is ArtifactRoutingAggregateOutcomeV2.NOT_REQUIRED:
            if self.routing_plan is not None:
                raise ValueError("NOT_REQUIRED cannot carry a routing plan")
        elif self.routing_plan is None:
            raise ValueError("routing outcome requires a complete routing plan")
        elif self.routing_plan.aggregate_outcome is not self.outcome:
            raise ValueError("routing result outcome must match its plan")
        return self


def artifact_routing_request_carried_sha256(
    request: ArtifactRoutingRequest,
) -> str:
    payload = request.model_dump(
        mode="json",
        exclude={
            "request_id",
            "request_sha256",
            "audit",
            "build_contracts",
        },
        exclude_none=False,
    )
    payload["build_contract_refs"] = [
        artifact_build_contract_ref(contract).model_dump(
            mode="json",
            exclude_none=False,
        )
        for contract in request.build_contracts
    ]
    return _payload_sha256(payload)


def artifact_routing_request_ref(
    request: ArtifactRoutingRequest,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-routing-request",
        object_id=request.request_id,
        object_version="r5-05",
        object_sha256=request.request_sha256,
    )


def artifact_routing_execution_result_carried_sha256(
    result: ArtifactRoutingExecutionResult,
) -> str:
    return _payload_sha256(
        result.model_dump(
            mode="json",
            exclude={"result_id", "result_sha256", "audit"},
            exclude_none=False,
        )
    )


def artifact_routing_execution_result_ref(
    result: ArtifactRoutingExecutionResult,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-routing-execution-result",
        object_id=result.result_id,
        object_version="r5-05",
        object_sha256=result.result_sha256,
    )


def artifact_routing_compilation_result_carried_sha256(
    result: ArtifactRoutingCompilationResult,
) -> str:
    payload = result.model_dump(
        mode="json",
        exclude={
            "result_id",
            "result_sha256",
            "audit",
            "routing_plan",
        },
        exclude_none=False,
    )
    payload["routing_plan_ref"] = (
        None
        if result.routing_plan is None
        else {
            "object_type": "artifact-routing-plan",
            "object_id": result.routing_plan.artifact_routing_plan_id,
            "object_version": "v2",
            "object_sha256": result.routing_plan.artifact_routing_plan_sha256,
        }
    )
    return _payload_sha256(payload)


def artifact_routing_compilation_result_ref(
    result: ArtifactRoutingCompilationResult,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-routing-compilation-result",
        object_id=result.result_id,
        object_version="r5-05",
        object_sha256=result.result_sha256,
    )


def _require_ref(
    ref: ObjectRef,
    expected_type: str,
    expected_version: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type or ref.object_version != expected_version:
        raise ValueError(f"{field_name} must reference {expected_type} {expected_version}")


def _require_sorted_unique(
    field_name: str,
    values: tuple[object, ...],
) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be unique")
    if values != tuple(sorted(values, key=repr)):
        raise ValueError(f"{field_name} must be sorted")


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
