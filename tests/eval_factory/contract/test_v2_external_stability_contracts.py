from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.external_stability_v2 import (
    ExternalRealTraceStabilityOutcomeV2,
    ExternalRealTraceStabilityPolicyV2,
    ExternalRealTraceStabilityReportV2,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityCaseSummaryV2,
    RealTraceStabilityFaultPointV2,
    RealTraceStabilityReasonCodeV2,
)
from eval_factory.contracts.trace import ParseQuality


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ref(
    kind: str,
    suffix: str,
    *,
    version: str,
) -> ObjectRef:
    return ObjectRef(
        object_type=kind,
        object_id=f"{kind}://external-stability/{suffix}",
        object_version=version,
        object_sha256=_digest(f"{kind}:{suffix}:{version}"),
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
        created_by="external-stability-test",
        governing_versions=(
            VersionBinding(
                component="external-real-trace-stability",
                version="external-real-trace-stability/r8-10-v1",
            ),
        ),
    )


def _case_audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
        created_by="external-stability-kernel-test",
        governing_versions=(
            VersionBinding(
                component="real-trace-stability",
                version="real-trace-stability/r8-04-v1",
            ),
        ),
    )


def _policy() -> ExternalRealTraceStabilityPolicyV2:
    inventory_ref = _ref(
        "external-corpus-inventory",
        "inventory",
        version="private-v1",
    )
    return ExternalRealTraceStabilityPolicyV2.create(
        package_manifest_ref=_ref(
            "external-evidence-package-manifest",
            "package",
            version="v2",
        ),
        private_inventory_ref=inventory_ref,
        source_count=100,
        inventory_sha256=inventory_ref.object_sha256,
        max_source_bytes=1_000_000_000,
        max_total_source_bytes=1_000_000_000_000,
        max_private_bytes=1_000_000_000,
        max_report_bytes=100_000_000,
        audit=_audit(),
    )


def _summary(
    index: int,
    *,
    stable: bool = True,
) -> RealTraceStabilityCaseSummaryV2:
    return RealTraceStabilityCaseSummaryV2.create(
        private_case_result_ref=_ref(
            "real-trace-stability-case-result",
            f"case-{index:03d}",
            version="private-v1",
        ),
        outcome=(
            RealTraceStabilityCaseOutcomeV2.STABLE if stable else RealTraceStabilityCaseOutcomeV2.UNSTABLE
        ),
        reason_code=(
            RealTraceStabilityReasonCodeV2.NONE if stable else RealTraceStabilityReasonCodeV2.REPLAY_MISMATCH
        ),
        parse_quality=ParseQuality.STRICT,
        assigned_fault_point=tuple(RealTraceStabilityFaultPointV2)[index % 4],
        fault_observed=True,
        resume_succeeded=True,
        replay_stable=stable,
        attempt_count=1,
        unexpected_retry_count=0,
        resume_count=1,
        replay_count=1,
        stage_result_count=1,
        completion_witness_count=1,
        audit=_case_audit(),
    )


def _report(
    count: int,
    *,
    unstable_index: int | None = None,
) -> ExternalRealTraceStabilityReportV2:
    policy = _policy().model_copy(update={"source_count": count})
    policy = ExternalRealTraceStabilityPolicyV2.create(
        package_manifest_ref=policy.package_manifest_ref,
        private_inventory_ref=policy.private_inventory_ref,
        source_count=count,
        inventory_sha256=policy.inventory_sha256,
        max_source_bytes=policy.max_source_bytes,
        max_total_source_bytes=policy.max_total_source_bytes,
        max_private_bytes=policy.max_private_bytes,
        max_report_bytes=policy.max_report_bytes,
        audit=_audit(),
    )
    summaries = tuple(_summary(index, stable=index != unstable_index) for index in range(count))
    return ExternalRealTraceStabilityReportV2.create(
        policy_ref=policy.to_ref(),
        package_manifest_ref=policy.package_manifest_ref,
        private_inventory_ref=policy.private_inventory_ref,
        private_result_set_ref=_ref(
            "external-real-trace-stability-result-set",
            "results",
            version="private-v1",
        ),
        unique_real_trace_count=count,
        case_summaries=summaries,
        audit=_audit(),
    )


def test_external_stability_report_derives_sc_011_only() -> None:
    report = _report(100)

    assert report.outcome is ExternalRealTraceStabilityOutcomeV2.PASSED
    assert report.stable_case_count == 100
    assert report.observed_threshold_met is True
    assert report.satisfies_sc_011 is True
    assert report.authorizes_sc_010 is False
    assert report.authorizes_sc_012 is False
    assert report.authorizes_sc_013 is False
    assert report.authorizes_sc_014 is False
    assert report.authorizes_sc_015 is False
    assert report.authorizes_approval is False
    assert report.authorizes_attestation is False
    assert report.authorizes_production_release is False


def test_external_stability_pending_and_threshold_miss_are_distinct() -> None:
    pending = _report(99)
    unstable = _report(100, unstable_index=50)

    assert pending.outcome is (ExternalRealTraceStabilityOutcomeV2.STATISTICAL_GATE_PENDING)
    assert pending.satisfies_sc_011 is False
    assert unstable.outcome is (ExternalRealTraceStabilityOutcomeV2.STABILITY_THRESHOLD_NOT_MET)
    assert unstable.observed_threshold_met is False
    assert unstable.satisfies_sc_011 is False


def test_external_stability_contracts_are_strict_and_frozen() -> None:
    report = _report(100)
    with pytest.raises(ValidationError):
        report.satisfies_sc_011 = False
    payload = report.model_dump(mode="python")
    payload["satisfies_sc_011"] = False
    with pytest.raises(ValidationError):
        ExternalRealTraceStabilityReportV2.model_validate(payload)
    payload = report.model_dump(mode="python")
    payload["source_members"] = ["forbidden"]
    with pytest.raises(ValidationError):
        ExternalRealTraceStabilityReportV2.model_validate(payload)
