from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.orchestration import (
    ItemStatus,
    JobStatus,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import DatasetJobSpecV2
from eval_factory.contracts.real_trace_stability_v2 import (
    REAL_TRACE_STABILITY_POLICY_VERSION,
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityCaseSummaryV2,
    RealTraceStabilityFaultPointV2,
    RealTraceStabilityPolicyV2,
    RealTraceStabilityReasonCodeV2,
    real_trace_stability_case_summary_v2_ref,
    validate_real_trace_stability_case_classification,
)
from eval_factory.contracts.trace import ParseQuality


class RealTraceStabilityMemberV1(ContractModelV2):
    schema_version: Literal["eval-factory/real-trace-stability-member/private-v1"] = (
        "eval-factory/real-trace-stability-member/private-v1"
    )
    case_key: Identifier
    instance_id: Identifier
    sid: Identifier
    business: Identifier
    category: str = Field(min_length=1, max_length=128)
    pool_id: str = Field(min_length=1, max_length=64)
    pool_category: str = Field(min_length=1, max_length=128)
    source_ref: Identifier
    raw_sha256: Sha256
    size_bytes: int = Field(ge=1, le=1_000_000_000)
    assigned_fault_point: RealTraceStabilityFaultPointV2

    @model_validator(mode="after")
    def validate_member(self) -> Self:
        if self.source_ref != f"raw_traj://{self.instance_id}":
            raise ValueError("stability member source ref differs from instance")
        observed = real_trace_stability_member_v1_carried_sha256(self)
        if self.case_key not in {
            "real-trace-stability-case://pending",
            f"real-trace-stability-case://sha256/{observed}",
        }:
            raise ValueError("stability member case key is stale")
        return self

    @classmethod
    def create(
        cls,
        *,
        instance_id: str,
        sid: str,
        business: str,
        category: str,
        pool_id: str,
        pool_category: str,
        raw_sha256: str,
        size_bytes: int,
        assigned_fault_point: RealTraceStabilityFaultPointV2,
    ) -> RealTraceStabilityMemberV1:
        value = cls(
            case_key="real-trace-stability-case://pending",
            instance_id=instance_id,
            sid=sid,
            business=business,
            category=category,
            pool_id=pool_id,
            pool_category=pool_category,
            source_ref=f"raw_traj://{instance_id}",
            raw_sha256=raw_sha256,
            size_bytes=size_bytes,
            assigned_fault_point=assigned_fault_point,
        )
        digest = real_trace_stability_member_v1_carried_sha256(value)
        return value.model_copy(update={"case_key": f"real-trace-stability-case://sha256/{digest}"})

    def to_ref(self) -> ObjectRef:
        return real_trace_stability_member_v1_ref(self)


class RealTraceStabilityInventoryV1(ContractModelV2):
    schema_version: Literal["eval-factory/real-trace-stability-inventory/private-v1"] = (
        "eval-factory/real-trace-stability-inventory/private-v1"
    )
    inventory_id: Identifier
    source_manifest_ref: ObjectRef
    members: tuple[RealTraceStabilityMemberV1, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    policy_version: Literal["real-trace-stability/r8-04-v1"] = REAL_TRACE_STABILITY_POLICY_VERSION
    inventory_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        _require_ref(
            self.source_manifest_ref,
            "real-trace-source-manifest",
            "csv/v1",
            "source_manifest_ref",
        )
        keys = tuple((item.instance_id, item.raw_sha256) for item in self.members)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("inventory members must be sorted and unique")
        for values, label in (
            ((item.instance_id for item in self.members), "instance"),
            ((item.sid for item in self.members), "SID"),
            ((item.raw_sha256 for item in self.members), "raw hash"),
            ((item.case_key for item in self.members), "case key"),
        ):
            observed = tuple(values)
            if len(observed) != len(set(observed)):
                raise ValueError(f"inventory contains duplicate {label}")
        expected_faults = tuple(RealTraceStabilityFaultPointV2)
        if tuple(item.assigned_fault_point for item in self.members) != tuple(
            expected_faults[index % len(expected_faults)] for index in range(len(self.members))
        ):
            raise ValueError("inventory fault assignment differs from canonical rotation")
        if self.audit.input_refs != (self.source_manifest_ref,):
            raise ValueError("inventory audit must bind exact source manifest")
        _validate_identity(
            self.inventory_id,
            self.inventory_sha256,
            "real-trace-stability-inventory",
            real_trace_stability_inventory_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        source_manifest_ref: ObjectRef,
        members: tuple[RealTraceStabilityMemberV1, ...],
        audit: ContractAudit,
    ) -> RealTraceStabilityInventoryV1:
        value = cls(
            inventory_id="real-trace-stability-inventory://pending",
            source_manifest_ref=source_manifest_ref,
            members=tuple(
                sorted(
                    members,
                    key=lambda item: (item.instance_id, item.raw_sha256),
                )
            ),
            inventory_sha256="0" * 64,
            audit=audit.model_copy(update={"input_refs": (source_manifest_ref,)}),
        )
        return _finalize(
            value,
            "inventory_id",
            "inventory_sha256",
            "real-trace-stability-inventory",
            real_trace_stability_inventory_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return real_trace_stability_inventory_v1_ref(self)


class RealTraceStabilityCaseResultV1(ContractModelV2):
    schema_version: Literal["eval-factory/real-trace-stability-case-result/private-v1"] = (
        "eval-factory/real-trace-stability-case-result/private-v1"
    )
    case_result_id: Identifier
    member_ref: ObjectRef
    dataset_job_spec_ref: ObjectRef
    source_trace_ref: ObjectRef
    job_id: Identifier
    job_status: JobStatus | None = None
    item_id: Identifier | None = None
    item_status: ItemStatus | None = None
    stage_run_refs: tuple[ObjectRef, ...] = ()
    stage_result_refs: tuple[ObjectRef, ...] = ()
    output_refs: tuple[ObjectRef, ...] = ()
    checkpoint_ref: ObjectRef | None = None
    stored_manifest_ref: ObjectRef | None = None
    completion_outbox_refs: tuple[ObjectRef, ...] = ()
    parse_quality: ParseQuality | None = None
    assigned_fault_point: RealTraceStabilityFaultPointV2
    fault_observed: bool
    resume_succeeded: bool
    replay_stable: bool
    attempt_count: int = Field(ge=0, le=1_000)
    unexpected_retry_count: int = Field(ge=0, le=1_000)
    resume_count: int = Field(ge=0, le=1_000)
    replay_count: int = Field(ge=0, le=1_000)
    outcome: RealTraceStabilityCaseOutcomeV2
    reason_code: RealTraceStabilityReasonCodeV2
    policy_version: Literal["real-trace-stability/r8-04-v1"] = REAL_TRACE_STABILITY_POLICY_VERSION
    case_result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        for ref, object_type, version, name in (
            (
                self.member_ref,
                "real-trace-stability-member",
                "private-v1",
                "member_ref",
            ),
            (
                self.dataset_job_spec_ref,
                "dataset-job-spec",
                "v2",
                "dataset_job_spec_ref",
            ),
            (
                self.source_trace_ref,
                "trace-source",
                "raw_traj_v1/1.0.0",
                "source_trace_ref",
            ),
        ):
            _require_ref(ref, object_type, version, name)
        _require_sorted_unique_refs(self.stage_run_refs, "stage_run_refs")
        _require_sorted_unique_refs(self.stage_result_refs, "stage_result_refs")
        _require_sorted_unique_refs(self.output_refs, "output_refs")
        _require_sorted_unique_refs(
            self.completion_outbox_refs,
            "completion_outbox_refs",
        )
        if self.unexpected_retry_count > self.attempt_count:
            raise ValueError("unexpected retries cannot exceed attempts")
        validate_real_trace_stability_case_classification(
            self.outcome,
            self.reason_code,
            replay_stable=self.replay_stable,
        )
        stable = self.outcome is RealTraceStabilityCaseOutcomeV2.STABLE
        if stable and (
            self.job_status is not JobStatus.SUCCEEDED
            or self.item_status is not ItemStatus.APPROVED
            or self.item_id is None
            or self.parse_quality is None
            or not self.fault_observed
            or not self.resume_succeeded
            or not self.replay_stable
            or self.attempt_count != 1
            or self.unexpected_retry_count != 0
            or self.resume_count != 1
            or self.replay_count != 1
            or len(self.stage_run_refs) != 1
            or len(self.stage_result_refs) != 1
            or len(self.output_refs) != 1
            or self.checkpoint_ref != self.output_refs[0]
            or self.stored_manifest_ref != self.output_refs[0]
            or len(self.completion_outbox_refs) != 1
        ):
            raise ValueError("stable private case result is incomplete")
        refs = _sorted_refs(
            (
                self.member_ref,
                self.dataset_job_spec_ref,
                self.source_trace_ref,
                *self.stage_run_refs,
                *self.stage_result_refs,
                *self.output_refs,
                *((self.checkpoint_ref,) if self.checkpoint_ref else ()),
                *((self.stored_manifest_ref,) if self.stored_manifest_ref else ()),
                *self.completion_outbox_refs,
            )
        )
        if self.audit.input_refs != refs:
            raise ValueError("private case result audit refs are incomplete")
        _validate_identity(
            self.case_result_id,
            self.case_result_sha256,
            "real-trace-stability-case-result",
            real_trace_stability_case_result_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        member_ref: ObjectRef,
        dataset_job_spec_ref: ObjectRef,
        source_trace_ref: ObjectRef,
        job_id: str,
        job_status: JobStatus | None,
        item_id: str | None,
        item_status: ItemStatus | None,
        stage_run_refs: tuple[ObjectRef, ...],
        stage_result_refs: tuple[ObjectRef, ...],
        output_refs: tuple[ObjectRef, ...],
        checkpoint_ref: ObjectRef | None,
        stored_manifest_ref: ObjectRef | None,
        completion_outbox_refs: tuple[ObjectRef, ...],
        parse_quality: ParseQuality | None,
        assigned_fault_point: RealTraceStabilityFaultPointV2,
        fault_observed: bool,
        resume_succeeded: bool,
        replay_stable: bool,
        attempt_count: int,
        unexpected_retry_count: int,
        resume_count: int,
        replay_count: int,
        outcome: RealTraceStabilityCaseOutcomeV2,
        reason_code: RealTraceStabilityReasonCodeV2,
        audit: ContractAudit,
    ) -> RealTraceStabilityCaseResultV1:
        stage_runs = _sorted_refs(stage_run_refs)
        stage_results = _sorted_refs(stage_result_refs)
        outputs = _sorted_refs(output_refs)
        outbox = _sorted_refs(completion_outbox_refs)
        refs = _sorted_refs(
            (
                member_ref,
                dataset_job_spec_ref,
                source_trace_ref,
                *stage_runs,
                *stage_results,
                *outputs,
                *((checkpoint_ref,) if checkpoint_ref else ()),
                *((stored_manifest_ref,) if stored_manifest_ref else ()),
                *outbox,
            )
        )
        value = cls(
            case_result_id="real-trace-stability-case-result://pending",
            member_ref=member_ref,
            dataset_job_spec_ref=dataset_job_spec_ref,
            source_trace_ref=source_trace_ref,
            job_id=job_id,
            job_status=job_status,
            item_id=item_id,
            item_status=item_status,
            stage_run_refs=stage_runs,
            stage_result_refs=stage_results,
            output_refs=outputs,
            checkpoint_ref=checkpoint_ref,
            stored_manifest_ref=stored_manifest_ref,
            completion_outbox_refs=outbox,
            parse_quality=parse_quality,
            assigned_fault_point=assigned_fault_point,
            fault_observed=fault_observed,
            resume_succeeded=resume_succeeded,
            replay_stable=replay_stable,
            attempt_count=attempt_count,
            unexpected_retry_count=unexpected_retry_count,
            resume_count=resume_count,
            replay_count=replay_count,
            outcome=outcome,
            reason_code=reason_code,
            case_result_sha256="0" * 64,
            audit=audit.model_copy(update={"input_refs": refs}),
        )
        return _finalize(
            value,
            "case_result_id",
            "case_result_sha256",
            "real-trace-stability-case-result",
            real_trace_stability_case_result_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return real_trace_stability_case_result_v1_ref(self)

    def to_public_summary(self) -> RealTraceStabilityCaseSummaryV2:
        return RealTraceStabilityCaseSummaryV2.create(
            private_case_result_ref=self.to_ref(),
            outcome=self.outcome,
            reason_code=self.reason_code,
            parse_quality=self.parse_quality,
            assigned_fault_point=self.assigned_fault_point,
            fault_observed=self.fault_observed,
            resume_succeeded=self.resume_succeeded,
            replay_stable=self.replay_stable,
            attempt_count=self.attempt_count,
            unexpected_retry_count=self.unexpected_retry_count,
            resume_count=self.resume_count,
            replay_count=self.replay_count,
            stage_result_count=len(self.stage_result_refs),
            completion_witness_count=len(self.completion_outbox_refs),
            audit=self.audit,
        )


class RealTraceStabilityResultSetV1(ContractModelV2):
    schema_version: Literal["eval-factory/real-trace-stability-result-set/private-v1"] = (
        "eval-factory/real-trace-stability-result-set/private-v1"
    )
    result_set_id: Identifier
    inventory_ref: ObjectRef
    policy_ref: ObjectRef
    case_result_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    case_summaries: tuple[RealTraceStabilityCaseSummaryV2, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    policy_version: Literal["real-trace-stability/r8-04-v1"] = REAL_TRACE_STABILITY_POLICY_VERSION
    result_set_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result_set(self) -> Self:
        _require_ref(
            self.inventory_ref,
            "real-trace-stability-inventory",
            "private-v1",
            "inventory_ref",
        )
        _require_ref(
            self.policy_ref,
            "real-trace-stability-policy",
            "v2",
            "policy_ref",
        )
        if len(self.case_result_refs) != len(set(self.case_result_refs)):
            raise ValueError("case_result_refs must be unique")
        if len(self.case_result_refs) != len(self.case_summaries):
            raise ValueError("result set case refs and summaries disagree")
        summary_refs = tuple(item.private_case_result_ref for item in self.case_summaries)
        if summary_refs != self.case_result_refs:
            raise ValueError("result set summaries do not bind exact private cases")
        if tuple(item.case_summary_id for item in self.case_summaries) != tuple(
            sorted(item.case_summary_id for item in self.case_summaries)
        ):
            raise ValueError("result set summaries must be sorted")
        refs = _sorted_refs((self.inventory_ref, self.policy_ref, *self.case_result_refs))
        if self.audit.input_refs != refs:
            raise ValueError("result set audit refs are incomplete")
        _validate_identity(
            self.result_set_id,
            self.result_set_sha256,
            "real-trace-stability-result-set",
            real_trace_stability_result_set_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        inventory_ref: ObjectRef,
        policy_ref: ObjectRef,
        case_results: tuple[RealTraceStabilityCaseResultV1, ...],
        audit: ContractAudit,
    ) -> RealTraceStabilityResultSetV1:
        pairs = tuple(
            sorted(
                ((result.to_ref(), result.to_public_summary()) for result in case_results),
                key=lambda pair: pair[1].case_summary_id,
            )
        )
        refs = tuple(pair[0] for pair in pairs)
        summaries = tuple(pair[1] for pair in pairs)
        audit_refs = _sorted_refs((inventory_ref, policy_ref, *refs))
        value = cls(
            result_set_id="real-trace-stability-result-set://pending",
            inventory_ref=inventory_ref,
            policy_ref=policy_ref,
            case_result_refs=refs,
            case_summaries=summaries,
            result_set_sha256="0" * 64,
            audit=audit.model_copy(update={"input_refs": audit_refs}),
        )
        return _finalize(
            value,
            "result_set_id",
            "result_set_sha256",
            "real-trace-stability-result-set",
            real_trace_stability_result_set_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return real_trace_stability_result_set_v1_ref(self)


class RealTraceStabilityLedgerCaseV1(ContractModelV2):
    member_ref: ObjectRef
    dataset_job_spec_ref: ObjectRef
    fault_point: RealTraceStabilityFaultPointV2


class RealTraceStabilityFaultObservationV1(ContractModelV2):
    schema_version: Literal["eval-factory/real-trace-stability-fault-observation/private-v1"] = (
        "eval-factory/real-trace-stability-fault-observation/private-v1"
    )
    member_ref: ObjectRef
    fault_point: RealTraceStabilityFaultPointV2
    observation_sha256: Sha256

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        _require_ref(
            self.member_ref,
            "real-trace-stability-member",
            "private-v1",
            "member_ref",
        )
        _validate_payload_hash(
            self,
            "observation_sha256",
            "fault observation",
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        member_ref: ObjectRef,
        fault_point: RealTraceStabilityFaultPointV2,
    ) -> RealTraceStabilityFaultObservationV1:
        value = cls(
            member_ref=member_ref,
            fault_point=fault_point,
            observation_sha256="0" * 64,
        )
        return value.model_copy(
            update={
                "observation_sha256": _model_payload_sha256(
                    value,
                    {"observation_sha256"},
                )
            }
        )


class RealTraceStabilityRunLedgerV1(ContractModelV2):
    schema_version: Literal["eval-factory/real-trace-stability-run-ledger/private-v1"] = (
        "eval-factory/real-trace-stability-run-ledger/private-v1"
    )
    inventory_ref: ObjectRef
    policy_ref: ObjectRef
    cases: tuple[RealTraceStabilityLedgerCaseV1, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    ledger_sha256: Sha256

    @model_validator(mode="after")
    def validate_ledger(self) -> Self:
        _require_ref(
            self.inventory_ref,
            "real-trace-stability-inventory",
            "private-v1",
            "inventory_ref",
        )
        _require_ref(
            self.policy_ref,
            "real-trace-stability-policy",
            "v2",
            "policy_ref",
        )
        keys = tuple(_ledger_case_key(item) for item in self.cases)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("run-ledger cases must be sorted and unique")
        _validate_payload_hash(self, "ledger_sha256", "run ledger")
        return self

    @classmethod
    def create(
        cls,
        *,
        inventory_ref: ObjectRef,
        policy_ref: ObjectRef,
        prepared: tuple[PreparedRealTraceStabilityCase, ...],
    ) -> RealTraceStabilityRunLedgerV1:
        cases = tuple(
            sorted(
                (
                    RealTraceStabilityLedgerCaseV1(
                        member_ref=item.member.to_ref(),
                        dataset_job_spec_ref=item.dataset_job_spec_ref,
                        fault_point=item.member.assigned_fault_point,
                    )
                    for item in prepared
                ),
                key=_ledger_case_key,
            )
        )
        value = cls(
            inventory_ref=inventory_ref,
            policy_ref=policy_ref,
            cases=cases,
            ledger_sha256="0" * 64,
        )
        return value.model_copy(update={"ledger_sha256": _model_payload_sha256(value, {"ledger_sha256"})})


@dataclass(frozen=True, slots=True)
class PreparedRealTraceStabilityCase:
    member: RealTraceStabilityMemberV1
    raw_path: Path
    trace: TraceSourceRef
    policy: RealTraceStabilityPolicyV2
    job_spec: DatasetJobSpecV2
    dataset_job_spec_ref: ObjectRef


def real_trace_stability_member_v1_carried_sha256(
    value: RealTraceStabilityMemberV1,
) -> str:
    return _model_payload_sha256(value, {"case_key"})


def real_trace_stability_member_v1_ref(
    value: RealTraceStabilityMemberV1,
) -> ObjectRef:
    digest = real_trace_stability_member_v1_carried_sha256(value)
    if value.case_key != f"real-trace-stability-case://sha256/{digest}":
        raise ValueError("stability member identity is stale")
    return ObjectRef(
        object_type="real-trace-stability-member",
        object_id=f"real-trace-stability-member://sha256/{digest}",
        object_version="private-v1",
        object_sha256=digest,
    )


def real_trace_stability_inventory_v1_carried_sha256(
    value: RealTraceStabilityInventoryV1,
) -> str:
    return _model_payload_sha256(
        value,
        {"inventory_id", "inventory_sha256", "audit"},
    )


def real_trace_stability_inventory_v1_ref(
    value: RealTraceStabilityInventoryV1,
) -> ObjectRef:
    _validate_identity(
        value.inventory_id,
        value.inventory_sha256,
        "real-trace-stability-inventory",
        real_trace_stability_inventory_v1_carried_sha256(value),
        allow_pending=False,
    )
    return ObjectRef(
        object_type="real-trace-stability-inventory",
        object_id=value.inventory_id,
        object_version="private-v1",
        object_sha256=value.inventory_sha256,
    )


def real_trace_stability_case_result_v1_carried_sha256(
    value: RealTraceStabilityCaseResultV1,
) -> str:
    return _model_payload_sha256(
        value,
        {"case_result_id", "case_result_sha256", "audit"},
    )


def real_trace_stability_case_result_v1_ref(
    value: RealTraceStabilityCaseResultV1,
) -> ObjectRef:
    _validate_identity(
        value.case_result_id,
        value.case_result_sha256,
        "real-trace-stability-case-result",
        real_trace_stability_case_result_v1_carried_sha256(value),
        allow_pending=False,
    )
    return ObjectRef(
        object_type="real-trace-stability-case-result",
        object_id=value.case_result_id,
        object_version="private-v1",
        object_sha256=value.case_result_sha256,
    )


def real_trace_stability_result_set_v1_carried_sha256(
    value: RealTraceStabilityResultSetV1,
) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={
            "result_set_id",
            "result_set_sha256",
            "audit",
            "case_summaries",
        },
        exclude_none=False,
    )
    payload["case_summary_refs"] = tuple(
        real_trace_stability_case_summary_v2_ref(summary) for summary in value.case_summaries
    )
    return _payload_sha256(payload)


def real_trace_stability_result_set_v1_ref(
    value: RealTraceStabilityResultSetV1,
) -> ObjectRef:
    _validate_identity(
        value.result_set_id,
        value.result_set_sha256,
        "real-trace-stability-result-set",
        real_trace_stability_result_set_v1_carried_sha256(value),
        allow_pending=False,
    )
    return ObjectRef(
        object_type="real-trace-stability-result-set",
        object_id=value.result_set_id,
        object_version="private-v1",
        object_sha256=value.result_set_sha256,
    )


def _ledger_case_key(
    value: RealTraceStabilityLedgerCaseV1,
) -> tuple[str, str, str]:
    return (
        value.member_ref.object_id,
        value.dataset_job_spec_ref.object_id,
        value.fault_point.value,
    )


def _validate_payload_hash(
    value: ContractModelV2,
    hash_field: str,
    label: str,
) -> None:
    observed = _model_payload_sha256(value, {hash_field})
    current = getattr(value, hash_field)
    if current != observed and current != "0" * 64:
        raise ValueError(f"{label} identity is stale")


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} has the wrong type or version")


def _require_sorted_unique_refs(
    values: tuple[ObjectRef, ...],
    field_name: str,
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must be sorted and unique")


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(values), key=_ref_key))


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _model_payload_sha256(
    value: ContractModelV2,
    exclude: set[str],
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
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


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
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


def _validate_identity(
    object_id: str,
    object_sha256: str,
    prefix: str,
    observed: str,
    *,
    allow_pending: bool = True,
) -> None:
    if allow_pending and object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    if object_id != f"{prefix}://sha256/{observed}" or object_sha256 != observed:
        raise ValueError(f"{prefix} identity is stale")


__all__ = [
    "PreparedRealTraceStabilityCase",
    "RealTraceStabilityCaseResultV1",
    "RealTraceStabilityFaultObservationV1",
    "RealTraceStabilityInventoryV1",
    "RealTraceStabilityLedgerCaseV1",
    "RealTraceStabilityMemberV1",
    "RealTraceStabilityResultSetV1",
    "RealTraceStabilityRunLedgerV1",
    "real_trace_stability_case_result_v1_carried_sha256",
    "real_trace_stability_case_result_v1_ref",
    "real_trace_stability_inventory_v1_carried_sha256",
    "real_trace_stability_inventory_v1_ref",
    "real_trace_stability_member_v1_carried_sha256",
    "real_trace_stability_member_v1_ref",
    "real_trace_stability_result_set_v1_carried_sha256",
    "real_trace_stability_result_set_v1_ref",
]
