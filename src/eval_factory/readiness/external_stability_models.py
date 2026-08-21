from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.external_stability_v2 import (
    EXTERNAL_REAL_TRACE_STABILITY_POLICY_VERSION,
)
from eval_factory.contracts.orchestration import ItemStatus, JobStatus
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityCaseSummaryV2,
    RealTraceStabilityFaultPointV2,
    RealTraceStabilityReasonCodeV2,
)
from eval_factory.contracts.trace import ParseQuality
from eval_factory.readiness.real_trace_stability_models import (
    RealTraceStabilityCaseResultV1,
)


class ExternalRealTraceStabilityCaseResultV1(ContractModelV2):
    schema_version: Literal["eval-factory/external-real-trace-stability-case-result/private-v1"] = (
        "eval-factory/external-real-trace-stability-case-result/private-v1"
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
    policy_version: Literal["external-real-trace-stability/r8-10-v1"] = (
        EXTERNAL_REAL_TRACE_STABILITY_POLICY_VERSION
    )
    case_result_sha256: Sha256
    audit: ContractAudit

    @classmethod
    def create(cls, **values: object) -> ExternalRealTraceStabilityCaseResultV1:
        audit = values.pop("audit")
        assert isinstance(audit, ContractAudit)
        refs = _sorted_refs(tuple(ref for ref in _collect_refs(values) if isinstance(ref, ObjectRef)))
        safe_audit = audit.model_copy(update={"input_refs": refs})
        provisional = cls.model_construct(
            case_result_id=("external-real-trace-stability-case-result://pending"),
            case_result_sha256="0" * 64,
            audit=safe_audit,
            **values,  # type: ignore[arg-type]
        )
        digest = _carried_sha256(provisional)
        return cls(
            case_result_id=(f"external-real-trace-stability-case-result://sha256/{digest}"),
            case_result_sha256=digest,
            audit=safe_audit,
            **values,  # type: ignore[arg-type]
        )

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.member_ref,
            "real-trace-stability-member",
            "private-v1",
            "member_ref",
        )
        _require_ref(
            self.dataset_job_spec_ref,
            "dataset-job-spec",
            "v2",
            "dataset_job_spec_ref",
        )
        if (
            self.source_trace_ref.object_type != "trace-source"
            or self.source_trace_ref.object_version
            not in {
                "curated_trajectory_v1/1.0.0",
                "runtime_snapshot_v1/1.0.0",
            }
        ):
            raise ValueError("source_trace_ref must reference an approved external trace adapter")
        stable = self.outcome is RealTraceStabilityCaseOutcomeV2.STABLE
        if stable and (
            self.reason_code is not RealTraceStabilityReasonCodeV2.NONE
            or self.job_status is not JobStatus.SUCCEEDED
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
            raise ValueError("stable external case result is incomplete")
        refs = _sorted_refs(_collect_refs(self, skip_audit=True))
        if self.audit.input_refs != refs:
            raise ValueError("external case-result audit refs are stale")
        digest = _carried_sha256(self)
        if self.case_result_sha256 != digest or self.case_result_id != (
            f"external-real-trace-stability-case-result://sha256/{digest}"
        ):
            raise ValueError("external case-result identity is stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="real-trace-stability-case-result",
            object_id=self.case_result_id,
            object_version="private-v1",
            object_sha256=self.case_result_sha256,
        )

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


class ExternalRealTraceStabilityResultSetV1(ContractModelV2):
    schema_version: Literal["eval-factory/external-real-trace-stability-result-set/private-v1"] = (
        "eval-factory/external-real-trace-stability-result-set/private-v1"
    )
    result_set_id: Identifier
    policy_ref: ObjectRef
    private_inventory_ref: ObjectRef
    case_result_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    case_summaries: tuple[RealTraceStabilityCaseSummaryV2, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    policy_version: Literal["external-real-trace-stability/r8-10-v1"] = (
        EXTERNAL_REAL_TRACE_STABILITY_POLICY_VERSION
    )
    result_set_sha256: Sha256
    audit: ContractAudit

    @classmethod
    def create(
        cls,
        *,
        policy_ref: ObjectRef,
        private_inventory_ref: ObjectRef,
        case_results: tuple[
            ExternalRealTraceStabilityCaseResultV1 | RealTraceStabilityCaseResultV1,
            ...,
        ],
        audit: ContractAudit,
    ) -> ExternalRealTraceStabilityResultSetV1:
        pairs = tuple(
            sorted(
                ((result.to_ref(), result.to_public_summary()) for result in case_results),
                key=lambda pair: pair[1].case_summary_id,
            )
        )
        refs = tuple(pair[0] for pair in pairs)
        summaries = tuple(pair[1] for pair in pairs)
        audit_refs = _sorted_refs((policy_ref, private_inventory_ref, *refs))
        safe_audit = audit.model_copy(update={"input_refs": audit_refs})
        value = cls.model_construct(
            result_set_id=("external-real-trace-stability-result-set://pending"),
            policy_ref=policy_ref,
            private_inventory_ref=private_inventory_ref,
            case_result_refs=refs,
            case_summaries=summaries,
            result_set_sha256="0" * 64,
            audit=safe_audit,
        )
        digest = _carried_sha256(value)
        return cls(
            result_set_id=(f"external-real-trace-stability-result-set://sha256/{digest}"),
            policy_ref=policy_ref,
            private_inventory_ref=private_inventory_ref,
            case_result_refs=refs,
            case_summaries=summaries,
            result_set_sha256=digest,
            audit=safe_audit,
        )

    @model_validator(mode="after")
    def validate_set(self) -> Self:
        _require_ref(
            self.policy_ref,
            "external-real-trace-stability-policy",
            "v2",
            "policy_ref",
        )
        _require_ref(
            self.private_inventory_ref,
            "external-corpus-inventory",
            "private-v1",
            "private_inventory_ref",
        )
        if (
            len(self.case_result_refs) != len(self.case_summaries)
            or tuple(summary.private_case_result_ref for summary in self.case_summaries)
            != self.case_result_refs
            or len(self.case_result_refs) != len(set(self.case_result_refs))
        ):
            raise ValueError("external stability result-set cases are stale")
        refs = _sorted_refs(
            (
                self.policy_ref,
                self.private_inventory_ref,
                *self.case_result_refs,
            )
        )
        if self.audit.input_refs != refs:
            raise ValueError("external stability result-set audit refs are stale")
        digest = _carried_sha256(self)
        if self.result_set_sha256 != digest or self.result_set_id != (
            f"external-real-trace-stability-result-set://sha256/{digest}"
        ):
            raise ValueError("external stability result-set identity is stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="external-real-trace-stability-result-set",
            object_id=self.result_set_id,
            object_version="private-v1",
            object_sha256=self.result_set_sha256,
        )


def _carried_sha256(value: ContractModelV2) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={
            "schema_version",
            "result_set_id",
            "result_set_sha256",
            "case_result_id",
            "case_result_sha256",
            "audit",
        },
    )
    canonical = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _collect_refs(
    value: object,
    *,
    skip_audit: bool = False,
) -> tuple[ObjectRef, ...]:
    refs: list[ObjectRef] = []

    def visit(item: object) -> None:
        if isinstance(item, ObjectRef):
            refs.append(item)
        elif isinstance(item, ContractModelV2):
            for field_name in type(item).model_fields:
                if skip_audit and field_name == "audit":
                    continue
                visit(getattr(item, field_name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, tuple | list):
            for child in item:
                visit(child)

    visit(value)
    return tuple(refs)


def _sorted_refs(refs: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(
        sorted(
            set(refs),
            key=lambda ref: (
                ref.object_type,
                ref.object_id,
                ref.object_version,
                ref.object_sha256,
            ),
        )
    )


def _require_ref(
    ref: ObjectRef,
    object_type: str,
    object_version: str,
    label: str,
) -> None:
    if ref.object_type != object_type or ref.object_version != object_version:
        raise ValueError(f"{label} must reference {object_type}/{object_version}")


__all__ = [
    "ExternalRealTraceStabilityCaseResultV1",
    "ExternalRealTraceStabilityResultSetV1",
]
