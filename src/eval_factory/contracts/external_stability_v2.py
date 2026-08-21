from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityCaseSummaryV2,
    validate_real_trace_stability_case_summary_v2_identity,
)

EXTERNAL_REAL_TRACE_STABILITY_POLICY_VERSION: Literal["external-real-trace-stability/r8-10-v1"] = (
    "external-real-trace-stability/r8-10-v1"
)


class ExternalRealTraceStabilityOutcomeV2(StrEnum):
    PASSED = "PASSED"
    STABILITY_THRESHOLD_NOT_MET = "STABILITY_THRESHOLD_NOT_MET"
    STATISTICAL_GATE_PENDING = "STATISTICAL_GATE_PENDING"
    INCOMPLETE = "INCOMPLETE"


class _ExternalStabilityObjectV2(ContractModelV2):
    object_id: Identifier
    object_sha256: Sha256
    audit: ContractAudit

    OBJECT_TYPE: ClassVar[str]

    @classmethod
    def _create(
        cls,
        *,
        audit: ContractAudit,
        values: dict[str, object],
    ) -> Self:
        refs = _sorted_refs(_collect_refs(values))
        safe_audit = audit.model_copy(update={"input_refs": refs})
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
    def validate_identity(self) -> Self:
        digest = _carried_sha256(self)
        if self.object_sha256 != digest or self.object_id != f"{self.OBJECT_TYPE}://sha256/{digest}":
            raise ValueError("external stability object identity is stale")
        refs = _sorted_refs(_collect_refs(self))
        if self.audit.input_refs != refs:
            raise ValueError("external stability audit refs are stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type=self.OBJECT_TYPE,
            object_id=self.object_id,
            object_version="v2",
            object_sha256=self.object_sha256,
        )


class ExternalRealTraceStabilityPolicyV2(_ExternalStabilityObjectV2):
    schema_version: Literal["eval-factory/external-real-trace-stability-policy/v2"] = (
        "eval-factory/external-real-trace-stability-policy/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "external-real-trace-stability-policy"

    package_manifest_ref: ObjectRef
    private_inventory_ref: ObjectRef
    source_count: int = Field(ge=1, le=100_000)
    inventory_sha256: Sha256
    required_unique_real_traces: Literal[100] = 100
    required_stable_rate_bps: Literal[10_000] = 10_000
    require_typed_result: Literal[True] = True
    require_expected_fault: Literal[True] = True
    require_resume_success: Literal[True] = True
    require_replay_stability: Literal[True] = True
    max_unexpected_retries: Literal[0] = 0
    required_completion_witnesses: Literal[1] = 1
    adapter_version: Literal[
        "curated_trajectory_v1/1.0.0",
        "runtime_snapshot_v1/1.0.0",
    ] = "runtime_snapshot_v1/1.0.0"
    trace_ir_version: Literal["v1"] = "v1"
    repair_policy_version: Literal["raw-traj-local-repair/v1"] = "raw-traj-local-repair/v1"
    recovery_policy_version: Literal["raw-traj-streaming-recovery/v1"] = "raw-traj-streaming-recovery/v1"
    normalization_policy_version: Literal["raw-traj-event-normalization/v1"] = (
        "raw-traj-event-normalization/v1"
    )
    tool_family_policy_version: Literal["raw-traj-tool-family/v1"] = "raw-traj-tool-family/v1"
    file_observation_policy_version: Literal["raw-traj-file-observation/v1"] = "raw-traj-file-observation/v1"
    segmentation_policy_version: Literal["deterministic-interaction-segments/v1"] = (
        "deterministic-interaction-segments/v1"
    )
    storage_policy_version: Literal["stored-manifest/v1"] = "stored-manifest/v1"
    max_source_bytes: int = Field(ge=1, le=1_000_000_000)
    max_total_source_bytes: int = Field(ge=1, le=1_000_000_000_000)
    max_private_bytes: int = Field(ge=2, le=5_000_000_000)
    max_report_bytes: int = Field(ge=2, le=100_000_000)
    policy_version: Literal["external-real-trace-stability/r8-10-v1"] = (
        EXTERNAL_REAL_TRACE_STABILITY_POLICY_VERSION
    )

    @classmethod
    def create(
        cls,
        *,
        package_manifest_ref: ObjectRef,
        private_inventory_ref: ObjectRef,
        source_count: int,
        inventory_sha256: str,
        max_source_bytes: int,
        max_total_source_bytes: int,
        max_private_bytes: int,
        max_report_bytes: int,
        adapter_version: Literal[
            "curated_trajectory_v1/1.0.0",
            "runtime_snapshot_v1/1.0.0",
        ] = "runtime_snapshot_v1/1.0.0",
        audit: ContractAudit,
    ) -> ExternalRealTraceStabilityPolicyV2:
        return cls._create(
            audit=audit,
            values={
                "package_manifest_ref": package_manifest_ref,
                "private_inventory_ref": private_inventory_ref,
                "source_count": source_count,
                "inventory_sha256": inventory_sha256,
                "max_source_bytes": max_source_bytes,
                "max_total_source_bytes": max_total_source_bytes,
                "max_private_bytes": max_private_bytes,
                "max_report_bytes": max_report_bytes,
                "adapter_version": adapter_version,
            },
        )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_ref(
            self.package_manifest_ref,
            "external-evidence-package-manifest",
            "v2",
            "package_manifest_ref",
        )
        _require_ref(
            self.private_inventory_ref,
            "external-corpus-inventory",
            "private-v1",
            "private_inventory_ref",
        )
        if (
            self.private_inventory_ref.object_sha256 != self.inventory_sha256
            or self.max_source_bytes > self.max_total_source_bytes
        ):
            raise ValueError("external stability inventory or limits are stale")
        return self


class ExternalRealTraceStabilityReportV2(_ExternalStabilityObjectV2):
    schema_version: Literal["eval-factory/external-real-trace-stability-report/v2"] = (
        "eval-factory/external-real-trace-stability-report/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "external-real-trace-stability-report"

    policy_ref: ObjectRef
    package_manifest_ref: ObjectRef
    private_inventory_ref: ObjectRef
    private_result_set_ref: ObjectRef
    unique_real_trace_count: int = Field(ge=1, le=100_000)
    case_summaries: tuple[RealTraceStabilityCaseSummaryV2, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    stable_case_count: int = Field(ge=0, le=100_000)
    unstable_case_count: int = Field(ge=0, le=100_000)
    infrastructure_error_case_count: int = Field(ge=0, le=100_000)
    incomplete_case_count: int = Field(ge=0, le=100_000)
    observed_threshold_met: bool
    outcome: ExternalRealTraceStabilityOutcomeV2
    satisfies_sc_011: bool
    authorizes_sc_010: Literal[False] = False
    authorizes_sc_012: Literal[False] = False
    authorizes_sc_013: Literal[False] = False
    authorizes_sc_014: Literal[False] = False
    authorizes_sc_015: Literal[False] = False
    authorizes_approval: Literal[False] = False
    authorizes_attestation: Literal[False] = False
    authorizes_production_release: Literal[False] = False
    claim_scope: Literal["REAL_TRACE_STABILITY_ONLY"] = "REAL_TRACE_STABILITY_ONLY"
    policy_version: Literal["external-real-trace-stability/r8-10-v1"] = (
        EXTERNAL_REAL_TRACE_STABILITY_POLICY_VERSION
    )

    @classmethod
    def create(
        cls,
        *,
        policy_ref: ObjectRef,
        package_manifest_ref: ObjectRef,
        private_inventory_ref: ObjectRef,
        private_result_set_ref: ObjectRef,
        unique_real_trace_count: int,
        case_summaries: tuple[RealTraceStabilityCaseSummaryV2, ...],
        audit: ContractAudit,
    ) -> ExternalRealTraceStabilityReportV2:
        summaries = tuple(
            sorted(
                case_summaries,
                key=lambda summary: summary.case_summary_id,
            )
        )
        fields = _derived_fields(unique_real_trace_count, summaries)
        return cls._create(
            audit=audit,
            values={
                "policy_ref": policy_ref,
                "package_manifest_ref": package_manifest_ref,
                "private_inventory_ref": private_inventory_ref,
                "private_result_set_ref": private_result_set_ref,
                "unique_real_trace_count": unique_real_trace_count,
                "case_summaries": summaries,
                **fields,
                "authorizes_sc_010": False,
                "authorizes_sc_012": False,
                "authorizes_sc_013": False,
                "authorizes_sc_014": False,
                "authorizes_sc_015": False,
                "authorizes_approval": False,
                "authorizes_attestation": False,
                "authorizes_production_release": False,
            },
        )

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_ref(
            self.policy_ref,
            "external-real-trace-stability-policy",
            "v2",
            "policy_ref",
        )
        _require_ref(
            self.package_manifest_ref,
            "external-evidence-package-manifest",
            "v2",
            "package_manifest_ref",
        )
        _require_ref(
            self.private_inventory_ref,
            "external-corpus-inventory",
            "private-v1",
            "private_inventory_ref",
        )
        _require_ref(
            self.private_result_set_ref,
            "external-real-trace-stability-result-set",
            "private-v1",
            "private_result_set_ref",
        )
        if len(self.case_summaries) != self.unique_real_trace_count:
            raise ValueError("external stability case count is not exact")
        for summary in self.case_summaries:
            validate_real_trace_stability_case_summary_v2_identity(summary)
        expected = _derived_fields(
            self.unique_real_trace_count,
            self.case_summaries,
        )
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("external stability derived report fields are stale")
        return self


def _derived_fields(
    unique_count: int,
    summaries: tuple[RealTraceStabilityCaseSummaryV2, ...],
) -> dict[str, object]:
    stable = sum(summary.outcome is RealTraceStabilityCaseOutcomeV2.STABLE for summary in summaries)
    unstable = sum(summary.outcome is RealTraceStabilityCaseOutcomeV2.UNSTABLE for summary in summaries)
    infrastructure = sum(
        summary.outcome is RealTraceStabilityCaseOutcomeV2.INFRASTRUCTURE_ERROR for summary in summaries
    )
    incomplete = sum(summary.outcome is RealTraceStabilityCaseOutcomeV2.INCOMPLETE for summary in summaries)
    threshold = stable == unique_count and all(summary.threshold_matched for summary in summaries)
    if infrastructure or incomplete:
        outcome = ExternalRealTraceStabilityOutcomeV2.INCOMPLETE
    elif unique_count < 100:
        outcome = ExternalRealTraceStabilityOutcomeV2.STATISTICAL_GATE_PENDING
    elif not threshold:
        outcome = ExternalRealTraceStabilityOutcomeV2.STABILITY_THRESHOLD_NOT_MET
    else:
        outcome = ExternalRealTraceStabilityOutcomeV2.PASSED
    return {
        "stable_case_count": stable,
        "unstable_case_count": unstable,
        "infrastructure_error_case_count": infrastructure,
        "incomplete_case_count": incomplete,
        "observed_threshold_met": threshold,
        "outcome": outcome,
        "satisfies_sc_011": (
            outcome is ExternalRealTraceStabilityOutcomeV2.PASSED and unique_count >= 100 and threshold
        ),
    }


def _carried_sha256(value: ContractModelV2) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={
            "schema_version",
            "object_id",
            "object_sha256",
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


def _collect_refs(value: object) -> tuple[ObjectRef, ...]:
    refs: list[ObjectRef] = []

    def visit(item: object) -> None:
        if isinstance(item, ObjectRef):
            refs.append(item)
        elif isinstance(item, ContractModelV2):
            for field_name in type(item).model_fields:
                if field_name == "audit":
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
    "EXTERNAL_REAL_TRACE_STABILITY_POLICY_VERSION",
    "ExternalRealTraceStabilityOutcomeV2",
    "ExternalRealTraceStabilityPolicyV2",
    "ExternalRealTraceStabilityReportV2",
]
