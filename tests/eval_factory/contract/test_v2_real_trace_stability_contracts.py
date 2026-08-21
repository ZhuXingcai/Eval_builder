from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.real_trace_stability_v2 import (
    REAL_TRACE_STABILITY_POLICY_VERSION,
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityCaseSummaryV2,
    RealTraceStabilityCategoryCountV2,
    RealTraceStabilityCorpusSummaryV2,
    RealTraceStabilityFaultPointV2,
    RealTraceStabilityOutcomeV2,
    RealTraceStabilityPolicyV2,
    RealTraceStabilityReasonCodeV2,
    RealTraceStabilityReportV2,
    RealTraceStabilitySampleStatusV2,
)
from eval_factory.contracts.trace import ParseQuality

NOW = datetime(2026, 8, 2, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[3]
GOLD = ROOT / "evals/golden/eval_factory/readiness/r8-04-real-trace-stability-v1.json"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ref(object_type: str, suffix: str, *, version: str = "v2") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r8-04/{suffix}",
        object_version=version,
        object_sha256=_digest(f"{object_type}:{suffix}:{version}"),
    )


def _manifest_ref() -> ObjectRef:
    return ObjectRef(
        object_type="real-trace-source-manifest",
        object_id="real-trace-source-manifest://manifest.csv",
        object_version="csv/v1",
        object_sha256="0d9df6c3935e00e5e15d486c9afe8cc2f581b63cf8f99ec42c1d3216cc74ee04",
    )


def _audit(*refs: ObjectRef, created_at: datetime = NOW) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="r8-04-contract-test",
        governing_versions=(
            VersionBinding(
                component="real-trace-stability",
                version=REAL_TRACE_STABILITY_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(sorted(set(refs), key=_ref_key)),
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _policy() -> RealTraceStabilityPolicyV2:
    return RealTraceStabilityPolicyV2.create(
        source_manifest_ref=_manifest_ref(),
        max_sources=10_000,
        max_source_bytes=100_000_000,
        max_total_source_bytes=10_000_000_000,
        max_private_bytes=32_000_000,
        max_report_bytes=8_000_000,
        audit=_audit(),
    )


def _corpus(count: int) -> RealTraceStabilityCorpusSummaryV2:
    policy = _policy()
    return RealTraceStabilityCorpusSummaryV2.create(
        source_manifest_ref=policy.source_manifest_ref,
        policy_ref=policy.to_ref(),
        private_inventory_ref=_ref(
            "real-trace-stability-inventory",
            f"inventory-{count}",
            version="private-v1",
        ),
        eligible_unique_trace_count=count,
        unique_instance_count=count,
        unique_raw_hash_count=count,
        total_raw_bytes=count * 1_000_000,
        minimum_raw_bytes=900_000,
        maximum_raw_bytes=1_100_000,
        category_counts=(RealTraceStabilityCategoryCountV2(category="science", count=count),),
        audit=_audit(),
    )


def _case(index: int, *, stable: bool = True) -> RealTraceStabilityCaseSummaryV2:
    outcome = RealTraceStabilityCaseOutcomeV2.STABLE if stable else RealTraceStabilityCaseOutcomeV2.UNSTABLE
    reason = RealTraceStabilityReasonCodeV2.NONE if stable else RealTraceStabilityReasonCodeV2.REPLAY_MISMATCH
    fault = tuple(RealTraceStabilityFaultPointV2)[index % 4]
    return RealTraceStabilityCaseSummaryV2.create(
        private_case_result_ref=_ref(
            "real-trace-stability-case-result",
            f"case-{index:03d}",
            version="private-v1",
        ),
        outcome=outcome,
        reason_code=reason,
        parse_quality=ParseQuality.STRICT,
        assigned_fault_point=fault,
        fault_observed=True,
        resume_succeeded=True,
        replay_stable=stable,
        attempt_count=1,
        unexpected_retry_count=0,
        resume_count=1,
        replay_count=1,
        stage_result_count=1,
        completion_witness_count=1,
        audit=_audit(),
    )


def _report(count: int, *, unstable_index: int | None = None) -> RealTraceStabilityReportV2:
    policy = _policy()
    corpus = _corpus(count)
    cases = tuple(_case(index, stable=index != unstable_index) for index in range(count))
    return RealTraceStabilityReportV2.create(
        policy_ref=policy.to_ref(),
        corpus_summary=corpus,
        private_result_set_ref=_ref(
            "real-trace-stability-result-set",
            f"set-{count}",
            version="private-v1",
        ),
        case_summaries=cases,
        audit=_audit(),
    )


def test_policy_corpus_and_report_are_strict_frozen_authorities() -> None:
    policy = _policy()
    report = _report(91)

    assert (
        policy.repair_policy_version,
        policy.recovery_policy_version,
        policy.normalization_policy_version,
        policy.tool_family_policy_version,
        policy.file_observation_policy_version,
        policy.segmentation_policy_version,
        policy.storage_policy_version,
    ) == (
        "raw-traj-local-repair/v1",
        "raw-traj-streaming-recovery/v1",
        "raw-traj-event-normalization/v1",
        "raw-traj-tool-family/v1",
        "raw-traj-file-observation/v1",
        "deterministic-interaction-segments/v1",
        "stored-manifest/v1",
    )
    assert report.outcome is RealTraceStabilityOutcomeV2.STATISTICAL_GATE_PENDING
    assert report.sample_status is RealTraceStabilitySampleStatusV2.INSUFFICIENT
    assert report.observed_threshold_met is True
    assert report.corpus_summary.shortfall_count == 9
    assert len(report.stable_case_refs) == 91
    assert sum(row.count for row in report.fault_point_counts) == 91
    with pytest.raises(ValidationError):
        report.report_sha256 = "f" * 64
    with pytest.raises(ValidationError):
        RealTraceStabilityReportV2.model_validate(
            {
                **report.model_dump(mode="python"),
                "source_trace_id": "forbidden",
            }
        )


def test_case_matrix_rejects_false_stability_and_private_surface() -> None:
    stable = _case(0)
    with pytest.raises(ValidationError, match="STABLE"):
        RealTraceStabilityCaseSummaryV2.model_validate(
            stable.model_copy(update={"replay_stable": False}).model_dump(mode="python")
        )
    with pytest.raises(ValidationError, match="outcome and reason"):
        RealTraceStabilityCaseSummaryV2.model_validate(
            stable.model_copy(
                update={
                    "outcome": RealTraceStabilityCaseOutcomeV2.UNSTABLE,
                    "reason_code": RealTraceStabilityReasonCodeV2.NONE,
                }
            ).model_dump(mode="python")
        )
    with pytest.raises(ValidationError, match="outcome and reason"):
        RealTraceStabilityCaseSummaryV2.model_validate(
            stable.model_copy(
                update={
                    "outcome": RealTraceStabilityCaseOutcomeV2.INFRASTRUCTURE_ERROR,
                    "reason_code": RealTraceStabilityReasonCodeV2.REPLAY_MISMATCH,
                    "threshold_matched": False,
                }
            ).model_dump(mode="python")
        )
    with pytest.raises(ValidationError, match="replay stability"):
        RealTraceStabilityCaseSummaryV2.model_validate(
            stable.model_copy(
                update={
                    "outcome": RealTraceStabilityCaseOutcomeV2.INFRASTRUCTURE_ERROR,
                    "reason_code": RealTraceStabilityReasonCodeV2.INTERNAL_ERROR,
                    "threshold_matched": False,
                }
            ).model_dump(mode="python")
        )
    serialized = stable.model_dump_json().casefold()
    for forbidden in (
        "instance_id",
        "source_trace",
        "raw_sha",
        "physical_path",
        "job_id",
        "stage_run",
        "payload",
        "final_output",
        "credential",
    ):
        assert forbidden not in serialized


def test_report_outcome_precedence_is_closed() -> None:
    pending_unstable = _report(91, unstable_index=0)
    sufficient_unstable = _report(100, unstable_index=0)
    passed = _report(100)

    assert pending_unstable.outcome is RealTraceStabilityOutcomeV2.STATISTICAL_GATE_PENDING
    assert pending_unstable.observed_threshold_met is False
    assert sufficient_unstable.outcome is RealTraceStabilityOutcomeV2.STABILITY_THRESHOLD_NOT_MET
    assert passed.outcome is RealTraceStabilityOutcomeV2.PASSED

    incomplete_case = RealTraceStabilityCaseSummaryV2.create(
        private_case_result_ref=_ref(
            "real-trace-stability-case-result",
            "incomplete",
            version="private-v1",
        ),
        outcome=RealTraceStabilityCaseOutcomeV2.INCOMPLETE,
        reason_code=RealTraceStabilityReasonCodeV2.CHILD_STATE_INCOMPLETE,
        parse_quality=None,
        assigned_fault_point=RealTraceStabilityFaultPointV2.BEFORE_SOURCE_REGISTRATION,
        fault_observed=False,
        resume_succeeded=False,
        replay_stable=False,
        attempt_count=0,
        unexpected_retry_count=0,
        resume_count=0,
        replay_count=0,
        stage_result_count=0,
        completion_witness_count=0,
        audit=_audit(),
    )
    base = _report(91)
    cases = list(base.case_summaries)
    replace_index = next(
        index
        for index, case in enumerate(cases)
        if case.assigned_fault_point is RealTraceStabilityFaultPointV2.BEFORE_SOURCE_REGISTRATION
    )
    cases[replace_index] = incomplete_case
    incomplete = RealTraceStabilityReportV2.create(
        policy_ref=base.policy_ref,
        corpus_summary=base.corpus_summary,
        private_result_set_ref=base.private_result_set_ref,
        case_summaries=tuple(cases),
        audit=_audit(),
    )
    assert incomplete.outcome is RealTraceStabilityOutcomeV2.INCOMPLETE


def test_behavior_identity_ignores_audit_actor_and_time() -> None:
    first = _report(91)
    later = RealTraceStabilityReportV2.create(
        policy_ref=first.policy_ref,
        corpus_summary=first.corpus_summary,
        private_result_set_ref=first.private_result_set_ref,
        case_summaries=first.case_summaries,
        audit=_audit(created_at=datetime(2026, 8, 3, tzinfo=UTC)),
    )

    assert first.to_ref() == later.to_ref()


def test_report_rejects_count_and_nested_ref_drift() -> None:
    report = _report(91)
    with pytest.raises(ValidationError, match="case"):
        RealTraceStabilityReportV2.model_validate(
            report.model_copy(update={"case_summaries": report.case_summaries[:-1]}).model_dump(mode="python")
        )
    with pytest.raises(ValidationError, match="corpus"):
        RealTraceStabilityReportV2.model_validate(
            report.model_copy(
                update={"corpus_summary_ref": _ref("real-trace-stability-corpus-summary", "wrong")}
            ).model_dump(mode="python")
        )
    original = _case(3)
    wrong_fault = RealTraceStabilityCaseSummaryV2.create(
        private_case_result_ref=original.private_case_result_ref,
        outcome=original.outcome,
        reason_code=original.reason_code,
        parse_quality=original.parse_quality,
        assigned_fault_point=RealTraceStabilityFaultPointV2.BEFORE_SOURCE_REGISTRATION,
        fault_observed=original.fault_observed,
        resume_succeeded=original.resume_succeeded,
        replay_stable=original.replay_stable,
        attempt_count=original.attempt_count,
        unexpected_retry_count=original.unexpected_retry_count,
        resume_count=original.resume_count,
        replay_count=original.replay_count,
        stage_result_count=original.stage_result_count,
        completion_witness_count=original.completion_witness_count,
        audit=_audit(),
    )
    with pytest.raises(ValueError, match="fault distribution"):
        RealTraceStabilityReportV2.create(
            policy_ref=_policy().to_ref(),
            corpus_summary=_corpus(4),
            private_result_set_ref=_ref(
                "real-trace-stability-result-set",
                "wrong-faults",
                version="private-v1",
            ),
            case_summaries=(_case(0), _case(1), _case(2), wrong_fault),
            audit=_audit(),
        )


def test_real_run_gold_is_content_free_pending_evidence() -> None:
    payload = json.loads(GOLD.read_text(encoding="utf-8"))

    assert payload["claim_scope"] == "REAL_TRACE_STABILITY_ONLY"
    assert payload["observed"] == {
        "outcome": "STATISTICAL_GATE_PENDING",
        "sample_status": "INSUFFICIENT",
        "observed_threshold_met": True,
        "stable_count": 91,
        "unstable_count": 0,
        "infrastructure_error_count": 0,
        "incomplete_count": 0,
        "parse_quality_counts": {"REPAIRED": 72, "STRICT": 19},
        "parse_quality_unavailable_count": 0,
        "fault_point_counts": {
            "after_source_registration": 23,
            "after_trace_store_persist": 23,
            "before_source_registration": 23,
            "before_stage_completion": 22,
        },
        "totals": {
            "attempts": 91,
            "unexpected_retries": 0,
            "resumes": 91,
            "replays": 91,
            "stage_results": 91,
            "completion_witnesses": 91,
        },
    }
    assert len(payload["case_summary_sha256"]) == len(set(payload["case_summary_sha256"])) == 91
    assert payload["evidence_partitions"] == {
        "real_execution": True,
        "replay_verification": True,
        "synthetic_load": False,
    }
    assert not any(payload["authorities"].values())
    serialized = GOLD.read_text(encoding="utf-8").casefold()
    for forbidden in (
        '"instance_id"',
        '"sid"',
        '"source_ref"',
        '"job_id"',
        '"item_id"',
        '"stage_run',
        '"checkpoint_ref"',
        '"output_ref',
        '"outbox',
        '"physical_path"',
        '"final_output"',
        '"credential',
        '"exception',
    ):
        assert forbidden not in serialized
