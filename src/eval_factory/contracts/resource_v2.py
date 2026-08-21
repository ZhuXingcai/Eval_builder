from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2

R6_RESOURCE_CONTROL_POLICY_VERSION: Literal["resource-control/r6-04-v1"] = "resource-control/r6-04-v1"


class ResourceKindV2(StrEnum):
    PROCESS = "PROCESS"
    RENDERER = "RENDERER"
    NETWORK = "NETWORK"
    STORAGE = "STORAGE"


class ResourceAdmissionOutcomeV2(StrEnum):
    ADMITTED = "ADMITTED"
    WAITING_CAPACITY = "WAITING_CAPACITY"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    UNSATISFIABLE_DEMAND = "UNSATISFIABLE_DEMAND"


class WorkResourceEventKindV2(StrEnum):
    COMPLETED_RELEASED = "COMPLETED_RELEASED"
    RELEASE_PENDING_TERMINATION = "RELEASE_PENDING_TERMINATION"
    TERMINATED_RELEASED = "TERMINATED_RELEASED"
    QUARANTINED = "QUARANTINED"


class WorkResourceReservationStateV2(StrEnum):
    ACTIVE = "ACTIVE"
    RELEASE_PENDING_TERMINATION = "RELEASE_PENDING_TERMINATION"
    RELEASED = "RELEASED"
    QUARANTINED = "QUARANTINED"


class WorkResourceTerminationReasonV2(StrEnum):
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    RESOURCE_OVERRUN = "RESOURCE_OVERRUN"


class WorkResourceTerminationOutcomeV2(StrEnum):
    TERMINATED = "TERMINATED"
    ALREADY_STOPPED = "ALREADY_STOPPED"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    FAILED = "FAILED"


class NonModelResourceVectorV2(ContractModelV2):
    schema_version: Literal["eval-factory/non-model-resource-vector/v2"] = (
        "eval-factory/non-model-resource-vector/v2"
    )
    processes: int = Field(ge=0)
    renderers: int = Field(ge=0)
    network_requests: int = Field(ge=0)
    storage_bytes: int = Field(ge=0)

    @classmethod
    def zero(cls) -> NonModelResourceVectorV2:
        return cls(
            processes=0,
            renderers=0,
            network_requests=0,
            storage_bytes=0,
        )

    def add(self, other: NonModelResourceVectorV2) -> NonModelResourceVectorV2:
        return type(self)(
            processes=self.processes + other.processes,
            renderers=self.renderers + other.renderers,
            network_requests=self.network_requests + other.network_requests,
            storage_bytes=self.storage_bytes + other.storage_bytes,
        )

    def subtract(self, other: NonModelResourceVectorV2) -> NonModelResourceVectorV2:
        values = {
            "processes": self.processes - other.processes,
            "renderers": self.renderers - other.renderers,
            "network_requests": self.network_requests - other.network_requests,
            "storage_bytes": self.storage_bytes - other.storage_bytes,
        }
        if any(value < 0 for value in values.values()):
            raise ValueError("resource subtraction cannot produce a negative value")
        return type(self)(
            processes=values["processes"],
            renderers=values["renderers"],
            network_requests=values["network_requests"],
            storage_bytes=values["storage_bytes"],
        )

    def exceeds(self, limit: NonModelResourceVectorV2) -> tuple[ResourceKindV2, ...]:
        exceeded: list[ResourceKindV2] = []
        if self.processes > limit.processes:
            exceeded.append(ResourceKindV2.PROCESS)
        if self.renderers > limit.renderers:
            exceeded.append(ResourceKindV2.RENDERER)
        if self.network_requests > limit.network_requests:
            exceeded.append(ResourceKindV2.NETWORK)
        if self.storage_bytes > limit.storage_bytes:
            exceeded.append(ResourceKindV2.STORAGE)
        return tuple(exceeded)


class ResourcePoolPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/resource-pool-policy/v2"] = "eval-factory/resource-pool-policy/v2"
    resource_pool_policy_id: Identifier
    resource_domain_ref: ObjectRef
    capacity: NonModelResourceVectorV2
    policy_version: Literal["resource-control/r6-04-v1"] = R6_RESOURCE_CONTROL_POLICY_VERSION
    resource_pool_policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_ref(self.resource_domain_ref, "resource-domain", "v1", "resource_domain_ref")
        _require_resource_audit(self.audit, (self.resource_domain_ref,))
        validate_resource_pool_policy_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        resource_domain_ref: ObjectRef,
        capacity: NonModelResourceVectorV2,
        audit: ContractAudit,
    ) -> ResourcePoolPolicyV2:
        value = cls(
            resource_pool_policy_id="resource-pool-policy://pending",
            resource_domain_ref=resource_domain_ref,
            capacity=capacity,
            resource_pool_policy_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value, "resource_pool_policy_id", "resource_pool_policy_sha256", "resource-pool-policy"
        )


class JobResourcePolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/job-resource-policy/v2"] = "eval-factory/job-resource-policy/v2"
    job_resource_policy_id: Identifier
    dataset_job_spec_ref: ObjectRef
    resolved_job_work_graph_ref: ObjectRef
    resource_pool_policy_ref: ObjectRef
    job_concurrency: NonModelResourceVectorV2
    job_budget: NonModelResourceVectorV2
    policy_version: Literal["resource-control/r6-04-v1"] = R6_RESOURCE_CONTROL_POLICY_VERSION
    job_resource_policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_ref(self.dataset_job_spec_ref, "dataset-job-spec", "v2", "dataset_job_spec_ref")
        _require_ref(
            self.resolved_job_work_graph_ref,
            "resolved-job-work-graph",
            "v2",
            "resolved_job_work_graph_ref",
        )
        _require_ref(
            self.resource_pool_policy_ref,
            "resource-pool-policy",
            "v2",
            "resource_pool_policy_ref",
        )
        refs = (
            self.dataset_job_spec_ref,
            self.resolved_job_work_graph_ref,
            self.resource_pool_policy_ref,
        )
        _require_resource_audit(self.audit, refs)
        validate_job_resource_policy_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        dataset_job_spec_ref: ObjectRef,
        resolved_job_work_graph_ref: ObjectRef,
        resource_pool_policy_ref: ObjectRef,
        job_concurrency: NonModelResourceVectorV2,
        job_budget: NonModelResourceVectorV2,
        audit: ContractAudit,
    ) -> JobResourcePolicyV2:
        value = cls(
            job_resource_policy_id="job-resource-policy://pending",
            dataset_job_spec_ref=dataset_job_spec_ref,
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            resource_pool_policy_ref=resource_pool_policy_ref,
            job_concurrency=job_concurrency,
            job_budget=job_budget,
            job_resource_policy_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(value, "job_resource_policy_id", "job_resource_policy_sha256", "job-resource-policy")


class WorkResourceDemandV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-resource-demand/v2"] = "eval-factory/work-resource-demand/v2"
    work_resource_demand_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    work_unit_ref: ObjectRef
    job_resource_policy_ref: ObjectRef
    execution_profile_ref: ObjectRef
    attempt: int = Field(ge=1)
    capacity_units: NonModelResourceVectorV2
    budget_allowance: NonModelResourceVectorV2
    termination_required: bool
    policy_version: Literal["resource-control/r6-04-v1"] = R6_RESOURCE_CONTROL_POLICY_VERSION
    work_resource_demand_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_demand(self) -> Self:
        _require_ref(
            self.resolved_job_work_graph_ref,
            "resolved-job-work-graph",
            "v2",
            "resolved_job_work_graph_ref",
        )
        _require_ref(self.work_unit_ref, "resolved-work-unit", "v2", "work_unit_ref")
        _require_ref(
            self.job_resource_policy_ref,
            "job-resource-policy",
            "v2",
            "job_resource_policy_ref",
        )
        _require_ref(
            self.execution_profile_ref,
            "resource-execution-profile",
            "v1",
            "execution_profile_ref",
        )
        if self.capacity_units.processes and not self.termination_required:
            raise ValueError("process resource demand requires termination support")
        if self.capacity_units.storage_bytes != self.budget_allowance.storage_bytes:
            raise ValueError("storage capacity and budget allowance must match")
        for capacity, allowance, label in (
            (self.capacity_units.processes, self.budget_allowance.processes, "process"),
            (self.capacity_units.renderers, self.budget_allowance.renderers, "renderer"),
            (
                self.capacity_units.network_requests,
                self.budget_allowance.network_requests,
                "network",
            ),
        ):
            if capacity > allowance:
                raise ValueError(f"{label} capacity cannot exceed its budget allowance")
            if allowance > 0 and capacity == 0:
                raise ValueError(f"{label} budget allowance requires active capacity")
        refs = (
            self.resolved_job_work_graph_ref,
            self.work_unit_ref,
            self.job_resource_policy_ref,
            self.execution_profile_ref,
        )
        _require_resource_audit(self.audit, refs)
        validate_work_resource_demand_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        resolved_job_work_graph_ref: ObjectRef,
        work_unit_ref: ObjectRef,
        job_resource_policy_ref: ObjectRef,
        execution_profile_ref: ObjectRef,
        attempt: int,
        capacity_units: NonModelResourceVectorV2,
        budget_allowance: NonModelResourceVectorV2,
        termination_required: bool,
        audit: ContractAudit,
    ) -> WorkResourceDemandV2:
        value = cls(
            work_resource_demand_id="work-resource-demand://pending",
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            work_unit_ref=work_unit_ref,
            job_resource_policy_ref=job_resource_policy_ref,
            execution_profile_ref=execution_profile_ref,
            attempt=attempt,
            capacity_units=capacity_units,
            budget_allowance=budget_allowance,
            termination_required=termination_required,
            work_resource_demand_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value, "work_resource_demand_id", "work_resource_demand_sha256", "work-resource-demand"
        )


class ResourceAdmissionDecisionV2(ContractModelV2):
    schema_version: Literal["eval-factory/resource-admission-decision/v2"] = (
        "eval-factory/resource-admission-decision/v2"
    )
    resource_admission_decision_id: Identifier
    job_resource_policy_ref: ObjectRef
    work_resource_demand_ref: ObjectRef
    outcome: ResourceAdmissionOutcomeV2
    constrained_resources: tuple[ResourceKindV2, ...]
    pool_head_version_before: int = Field(ge=0)
    job_head_version_before: int = Field(ge=0)
    decided_at: datetime
    policy_version: Literal["resource-control/r6-04-v1"] = R6_RESOURCE_CONTROL_POLICY_VERSION
    resource_admission_decision_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        _require_ref(
            self.job_resource_policy_ref,
            "job-resource-policy",
            "v2",
            "job_resource_policy_ref",
        )
        _require_ref(
            self.work_resource_demand_ref,
            "work-resource-demand",
            "v2",
            "work_resource_demand_ref",
        )
        expected = tuple(sorted(set(self.constrained_resources), key=lambda item: item.value))
        if self.constrained_resources != expected:
            raise ValueError("constrained_resources must be sorted and unique")
        if self.outcome is ResourceAdmissionOutcomeV2.ADMITTED:
            if self.constrained_resources:
                raise ValueError("ADMITTED decision cannot constrain resources")
        elif not self.constrained_resources:
            raise ValueError("negative admission decision requires constrained resources")
        refs = (self.job_resource_policy_ref, self.work_resource_demand_ref)
        _require_resource_audit(self.audit, refs)
        validate_resource_admission_decision_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        job_resource_policy_ref: ObjectRef,
        work_resource_demand_ref: ObjectRef,
        outcome: ResourceAdmissionOutcomeV2,
        constrained_resources: tuple[ResourceKindV2, ...],
        pool_head_version_before: int,
        job_head_version_before: int,
        decided_at: datetime,
        audit: ContractAudit,
    ) -> ResourceAdmissionDecisionV2:
        value = cls(
            resource_admission_decision_id="resource-admission-decision://pending",
            job_resource_policy_ref=job_resource_policy_ref,
            work_resource_demand_ref=work_resource_demand_ref,
            outcome=outcome,
            constrained_resources=tuple(sorted(set(constrained_resources), key=lambda item: item.value)),
            pool_head_version_before=pool_head_version_before,
            job_head_version_before=job_head_version_before,
            decided_at=decided_at,
            resource_admission_decision_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "resource_admission_decision_id",
            "resource_admission_decision_sha256",
            "resource-admission-decision",
        )


class WorkResourceReservationV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-resource-reservation/v2"] = (
        "eval-factory/work-resource-reservation/v2"
    )
    work_resource_reservation_id: Identifier
    resource_admission_decision_ref: ObjectRef
    admission_outcome: Literal[ResourceAdmissionOutcomeV2.ADMITTED] = ResourceAdmissionOutcomeV2.ADMITTED
    work_resource_demand_ref: ObjectRef
    work_lease_ref: ObjectRef
    work_dispatch_decision_ref: ObjectRef
    holder_ref: ObjectRef
    fencing_token: int = Field(ge=1)
    execution_handle_ref: ObjectRef
    capacity_units: NonModelResourceVectorV2
    budget_allowance: NonModelResourceVectorV2
    acquired_at: datetime
    policy_version: Literal["resource-control/r6-04-v1"] = R6_RESOURCE_CONTROL_POLICY_VERSION
    work_resource_reservation_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_reservation(self) -> Self:
        for ref, object_type, version, name in (
            (
                self.resource_admission_decision_ref,
                "resource-admission-decision",
                "v2",
                "resource_admission_decision_ref",
            ),
            (self.work_resource_demand_ref, "work-resource-demand", "v2", "work_resource_demand_ref"),
            (self.work_lease_ref, "work-lease", "v2", "work_lease_ref"),
            (
                self.work_dispatch_decision_ref,
                "work-dispatch-decision",
                "v2",
                "work_dispatch_decision_ref",
            ),
            (self.holder_ref, "worker-principal", "v1", "holder_ref"),
            (
                self.execution_handle_ref,
                "resource-execution-handle",
                "v1",
                "execution_handle_ref",
            ),
        ):
            _require_ref(ref, object_type, version, name)
        refs = (
            self.resource_admission_decision_ref,
            self.work_resource_demand_ref,
            self.work_lease_ref,
            self.work_dispatch_decision_ref,
            self.holder_ref,
            self.execution_handle_ref,
        )
        _require_resource_audit(self.audit, refs)
        validate_work_resource_reservation_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        resource_admission_decision_ref: ObjectRef,
        work_resource_demand_ref: ObjectRef,
        work_lease_ref: ObjectRef,
        work_dispatch_decision_ref: ObjectRef,
        holder_ref: ObjectRef,
        fencing_token: int,
        execution_handle_ref: ObjectRef,
        capacity_units: NonModelResourceVectorV2,
        budget_allowance: NonModelResourceVectorV2,
        acquired_at: datetime,
        audit: ContractAudit,
        admission_outcome: Literal[ResourceAdmissionOutcomeV2.ADMITTED] = (
            ResourceAdmissionOutcomeV2.ADMITTED
        ),
    ) -> WorkResourceReservationV2:
        value = cls(
            work_resource_reservation_id="work-resource-reservation://pending",
            resource_admission_decision_ref=resource_admission_decision_ref,
            admission_outcome=admission_outcome,
            work_resource_demand_ref=work_resource_demand_ref,
            work_lease_ref=work_lease_ref,
            work_dispatch_decision_ref=work_dispatch_decision_ref,
            holder_ref=holder_ref,
            fencing_token=fencing_token,
            execution_handle_ref=execution_handle_ref,
            capacity_units=capacity_units,
            budget_allowance=budget_allowance,
            acquired_at=acquired_at,
            work_resource_reservation_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "work_resource_reservation_id",
            "work_resource_reservation_sha256",
            "work-resource-reservation",
        )


class ResourceUsageV2(ContractModelV2):
    schema_version: Literal["eval-factory/resource-usage/v2"] = "eval-factory/resource-usage/v2"
    observed: NonModelResourceVectorV2
    usage_policy_version: Literal["resource-control/r6-04-v1"] = R6_RESOURCE_CONTROL_POLICY_VERSION


class WorkResourceTerminationRequestV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-resource-termination-request/v2"] = (
        "eval-factory/work-resource-termination-request/v2"
    )
    work_resource_termination_request_id: Identifier
    work_resource_reservation_ref: ObjectRef
    work_lease_event_ref: ObjectRef
    execution_handle_ref: ObjectRef
    fencing_token: int = Field(ge=1)
    reason: WorkResourceTerminationReasonV2
    requested_at: datetime
    policy_version: Literal["resource-control/r6-04-v1"] = R6_RESOURCE_CONTROL_POLICY_VERSION
    work_resource_termination_request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_ref(
            self.work_resource_reservation_ref,
            "work-resource-reservation",
            "v2",
            "work_resource_reservation_ref",
        )
        _require_ref(self.work_lease_event_ref, "work-lease-event", "v2", "work_lease_event_ref")
        _require_ref(
            self.execution_handle_ref,
            "resource-execution-handle",
            "v1",
            "execution_handle_ref",
        )
        refs = (
            self.work_resource_reservation_ref,
            self.work_lease_event_ref,
            self.execution_handle_ref,
        )
        _require_resource_audit(self.audit, refs)
        validate_work_resource_termination_request_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        work_resource_reservation_ref: ObjectRef,
        work_lease_event_ref: ObjectRef,
        execution_handle_ref: ObjectRef,
        fencing_token: int,
        reason: WorkResourceTerminationReasonV2,
        requested_at: datetime,
        audit: ContractAudit,
    ) -> WorkResourceTerminationRequestV2:
        value = cls(
            work_resource_termination_request_id="work-resource-termination-request://pending",
            work_resource_reservation_ref=work_resource_reservation_ref,
            work_lease_event_ref=work_lease_event_ref,
            execution_handle_ref=execution_handle_ref,
            fencing_token=fencing_token,
            reason=reason,
            requested_at=requested_at,
            work_resource_termination_request_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "work_resource_termination_request_id",
            "work_resource_termination_request_sha256",
            "work-resource-termination-request",
        )


class WorkResourceTerminationResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-resource-termination-result/v2"] = (
        "eval-factory/work-resource-termination-result/v2"
    )
    work_resource_termination_result_id: Identifier
    termination_request_ref: ObjectRef
    facade_termination_result_ref: ObjectRef
    outcome: WorkResourceTerminationOutcomeV2
    completed_at: datetime
    policy_version: Literal["resource-control/r6-04-v1"] = R6_RESOURCE_CONTROL_POLICY_VERSION
    work_resource_termination_result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.termination_request_ref,
            "work-resource-termination-request",
            "v2",
            "termination_request_ref",
        )
        _require_ref(
            self.facade_termination_result_ref,
            "attachment-resource-termination-result",
            "v2",
            "facade_termination_result_ref",
        )
        _require_resource_audit(
            self.audit,
            (
                self.termination_request_ref,
                self.facade_termination_result_ref,
            ),
        )
        validate_work_resource_termination_result_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        termination_request_ref: ObjectRef,
        facade_termination_result_ref: ObjectRef,
        outcome: WorkResourceTerminationOutcomeV2,
        completed_at: datetime,
        audit: ContractAudit,
    ) -> WorkResourceTerminationResultV2:
        value = cls(
            work_resource_termination_result_id="work-resource-termination-result://pending",
            termination_request_ref=termination_request_ref,
            facade_termination_result_ref=facade_termination_result_ref,
            outcome=outcome,
            completed_at=completed_at,
            work_resource_termination_result_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "work_resource_termination_result_id",
            "work_resource_termination_result_sha256",
            "work-resource-termination-result",
        )


class WorkResourceEventV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-resource-event/v2"] = "eval-factory/work-resource-event/v2"
    work_resource_event_id: Identifier
    work_resource_reservation_ref: ObjectRef
    work_lease_event_ref: ObjectRef
    event_kind: WorkResourceEventKindV2
    reservation_version: int = Field(ge=1)
    charged_usage: ResourceUsageV2
    released_capacity: NonModelResourceVectorV2
    resource_termination_request_ref: ObjectRef | None = None
    resource_termination_result_ref: ObjectRef | None = None
    policy_version: Literal["resource-control/r6-04-v1"] = R6_RESOURCE_CONTROL_POLICY_VERSION
    work_resource_event_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        _require_ref(
            self.work_resource_reservation_ref,
            "work-resource-reservation",
            "v2",
            "work_resource_reservation_ref",
        )
        _require_ref(self.work_lease_event_ref, "work-lease-event", "v2", "work_lease_event_ref")
        pending = self.event_kind is WorkResourceEventKindV2.RELEASE_PENDING_TERMINATION
        termination_complete = self.event_kind in {
            WorkResourceEventKindV2.TERMINATED_RELEASED,
            WorkResourceEventKindV2.QUARANTINED,
        }
        if pending != (self.resource_termination_request_ref is not None):
            raise ValueError("release-pending event requires exactly one termination request")
        if termination_complete != (self.resource_termination_result_ref is not None):
            raise ValueError("termination-complete event requires exactly one termination result")
        if pending and self.resource_termination_result_ref is not None:
            raise ValueError("release-pending event cannot contain a termination result")
        if termination_complete and self.resource_termination_request_ref is not None:
            raise ValueError("termination-complete event cannot contain a termination request")
        if self.resource_termination_request_ref is not None:
            _require_ref(
                self.resource_termination_request_ref,
                "work-resource-termination-request",
                "v2",
                "resource_termination_request_ref",
            )
        if self.resource_termination_result_ref is not None:
            _require_ref(
                self.resource_termination_result_ref,
                "work-resource-termination-result",
                "v2",
                "resource_termination_result_ref",
            )
        refs = (
            self.work_resource_reservation_ref,
            self.work_lease_event_ref,
            *(
                (self.resource_termination_request_ref,)
                if self.resource_termination_request_ref is not None
                else ()
            ),
            *(
                (self.resource_termination_result_ref,)
                if self.resource_termination_result_ref is not None
                else ()
            ),
        )
        _require_resource_audit(self.audit, refs)
        validate_work_resource_event_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        work_resource_reservation_ref: ObjectRef,
        work_lease_event_ref: ObjectRef,
        event_kind: WorkResourceEventKindV2,
        reservation_version: int,
        charged_usage: ResourceUsageV2,
        released_capacity: NonModelResourceVectorV2,
        resource_termination_request_ref: ObjectRef | None,
        audit: ContractAudit,
        resource_termination_result_ref: ObjectRef | None = None,
    ) -> WorkResourceEventV2:
        value = cls(
            work_resource_event_id="work-resource-event://pending",
            work_resource_reservation_ref=work_resource_reservation_ref,
            work_lease_event_ref=work_lease_event_ref,
            event_kind=event_kind,
            reservation_version=reservation_version,
            charged_usage=charged_usage,
            released_capacity=released_capacity,
            resource_termination_request_ref=resource_termination_request_ref,
            resource_termination_result_ref=resource_termination_result_ref,
            work_resource_event_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "work_resource_event_id",
            "work_resource_event_sha256",
            "work-resource-event",
        )


def _carried(value: ContractModelV2, id_field: str, sha_field: str) -> str:
    encoded = json.dumps(
        canonical_value_v2(value.model_dump(mode="python", exclude={id_field, sha_field, "audit"})),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finalize[ResourceModelT: ContractModelV2](
    value: ResourceModelT,
    id_field: str,
    sha_field: str,
    prefix: str,
) -> ResourceModelT:
    digest = _carried(value, id_field, sha_field)
    return value.model_copy(
        update={
            id_field: f"{prefix}://sha256/{digest}",
            sha_field: digest,
        }
    )


def _validate(value: ContractModelV2, id_field: str, sha_field: str, prefix: str) -> None:
    object_id = str(getattr(value, id_field))
    object_sha256 = str(getattr(value, sha_field))
    if object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    observed = _carried(value, id_field, sha_field)
    if object_id != f"{prefix}://sha256/{observed}" or object_sha256 != observed:
        raise ValueError(f"{prefix.replace('-', ' ')} identity is stale")


def _ref_for(value: ContractModelV2, object_type: str, id_field: str, sha_field: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=str(getattr(value, id_field)),
        object_version="v2",
        object_sha256=str(getattr(value, sha_field)),
    )


def resource_pool_policy_v2_ref(value: ResourcePoolPolicyV2) -> ObjectRef:
    validate_resource_pool_policy_v2_identity(value)
    return _ref_for(
        value,
        "resource-pool-policy",
        "resource_pool_policy_id",
        "resource_pool_policy_sha256",
    )


def job_resource_policy_v2_ref(value: JobResourcePolicyV2) -> ObjectRef:
    validate_job_resource_policy_v2_identity(value)
    return _ref_for(
        value,
        "job-resource-policy",
        "job_resource_policy_id",
        "job_resource_policy_sha256",
    )


def work_resource_demand_v2_ref(value: WorkResourceDemandV2) -> ObjectRef:
    validate_work_resource_demand_v2_identity(value)
    return _ref_for(
        value,
        "work-resource-demand",
        "work_resource_demand_id",
        "work_resource_demand_sha256",
    )


def resource_admission_decision_v2_ref(value: ResourceAdmissionDecisionV2) -> ObjectRef:
    validate_resource_admission_decision_v2_identity(value)
    return _ref_for(
        value,
        "resource-admission-decision",
        "resource_admission_decision_id",
        "resource_admission_decision_sha256",
    )


def work_resource_reservation_v2_ref(value: WorkResourceReservationV2) -> ObjectRef:
    validate_work_resource_reservation_v2_identity(value)
    return _ref_for(
        value,
        "work-resource-reservation",
        "work_resource_reservation_id",
        "work_resource_reservation_sha256",
    )


def work_resource_event_v2_ref(value: WorkResourceEventV2) -> ObjectRef:
    validate_work_resource_event_v2_identity(value)
    return _ref_for(
        value,
        "work-resource-event",
        "work_resource_event_id",
        "work_resource_event_sha256",
    )


def work_resource_termination_request_v2_ref(
    value: WorkResourceTerminationRequestV2,
) -> ObjectRef:
    validate_work_resource_termination_request_v2_identity(value)
    return _ref_for(
        value,
        "work-resource-termination-request",
        "work_resource_termination_request_id",
        "work_resource_termination_request_sha256",
    )


def work_resource_termination_result_v2_ref(
    value: WorkResourceTerminationResultV2,
) -> ObjectRef:
    validate_work_resource_termination_result_v2_identity(value)
    return _ref_for(
        value,
        "work-resource-termination-result",
        "work_resource_termination_result_id",
        "work_resource_termination_result_sha256",
    )


def validate_resource_pool_policy_v2_identity(value: ResourcePoolPolicyV2) -> None:
    _validate(
        value,
        "resource_pool_policy_id",
        "resource_pool_policy_sha256",
        "resource-pool-policy",
    )


def validate_job_resource_policy_v2_identity(value: JobResourcePolicyV2) -> None:
    _validate(
        value,
        "job_resource_policy_id",
        "job_resource_policy_sha256",
        "job-resource-policy",
    )


def validate_work_resource_demand_v2_identity(value: WorkResourceDemandV2) -> None:
    _validate(
        value,
        "work_resource_demand_id",
        "work_resource_demand_sha256",
        "work-resource-demand",
    )


def validate_resource_admission_decision_v2_identity(
    value: ResourceAdmissionDecisionV2,
) -> None:
    _validate(
        value,
        "resource_admission_decision_id",
        "resource_admission_decision_sha256",
        "resource-admission-decision",
    )


def validate_work_resource_reservation_v2_identity(
    value: WorkResourceReservationV2,
) -> None:
    _validate(
        value,
        "work_resource_reservation_id",
        "work_resource_reservation_sha256",
        "work-resource-reservation",
    )


def validate_work_resource_event_v2_identity(value: WorkResourceEventV2) -> None:
    _validate(
        value,
        "work_resource_event_id",
        "work_resource_event_sha256",
        "work-resource-event",
    )


def validate_work_resource_termination_request_v2_identity(
    value: WorkResourceTerminationRequestV2,
) -> None:
    _validate(
        value,
        "work_resource_termination_request_id",
        "work_resource_termination_request_sha256",
        "work-resource-termination-request",
    )


def validate_work_resource_termination_result_v2_identity(
    value: WorkResourceTerminationResultV2,
) -> None:
    _validate(
        value,
        "work_resource_termination_result_id",
        "work_resource_termination_result_sha256",
        "work-resource-termination-result",
    )


def _require_ref(ref: ObjectRef, object_type: str, version: str, name: str) -> None:
    if ref.object_type != object_type or ref.object_version != version:
        raise ValueError(f"{name} must reference {object_type} {version}")


def _require_resource_audit(audit: ContractAudit, refs: tuple[ObjectRef, ...]) -> None:
    expected = tuple(
        sorted(refs, key=lambda ref: (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256))
    )
    if audit.input_refs != expected:
        raise ValueError("resource control audit refs are incomplete")
    bindings = tuple(item for item in audit.governing_versions if item.component == "resource-control")
    if len(bindings) != 1 or bindings[0].version != R6_RESOURCE_CONTROL_POLICY_VERSION:
        raise ValueError("resource control audit is missing the current policy")
