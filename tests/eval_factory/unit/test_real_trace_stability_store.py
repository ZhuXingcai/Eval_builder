from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_real_trace_stability import (
    _audit,
    _manifest_ref,
    _policy,
    _prepared,
)

from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityCaseSummaryV2,
    RealTraceStabilityCategoryCountV2,
    RealTraceStabilityCorpusSummaryV2,
    RealTraceStabilityFaultPointV2,
    RealTraceStabilityReasonCodeV2,
    RealTraceStabilityReportV2,
)
from eval_factory.readiness.real_trace_stability import _closed_case_result
from eval_factory.readiness.real_trace_stability_models import (
    RealTraceStabilityInventoryV1,
    RealTraceStabilityResultSetV1,
)
from eval_factory.readiness.real_trace_stability_store import (
    RealTraceStabilityMaterialStore,
    RealTraceStabilityReportStore,
    RealTraceStabilityStoreCodec,
    RealTraceStabilityStoreFaultPoint,
    RealTraceStabilityStoreInjectedCrash,
    RealTraceStabilityStoreIntegrityError,
    RealTraceStabilityStoreLimitError,
    RealTraceStabilityStoreTypeError,
    StaticRealTraceStabilityStoreFaultInjector,
)


def _values(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prepared = _prepared(
        tmp_path,
        RealTraceStabilityFaultPointV2.BEFORE_SOURCE_REGISTRATION,
    )
    inventory = RealTraceStabilityInventoryV1.create(
        source_manifest_ref=_manifest_ref(),
        members=(prepared.member,),
        audit=_audit(),
    )
    case = _closed_case_result(
        prepared,
        audit=_audit(),
        outcome=RealTraceStabilityCaseOutcomeV2.INCOMPLETE,
        reason=RealTraceStabilityReasonCodeV2.CHILD_STATE_INCOMPLETE,
        fault_observed=False,
    )
    result_set = RealTraceStabilityResultSetV1.create(
        inventory_ref=inventory.to_ref(),
        policy_ref=_policy().to_ref(),
        case_results=(case,),
        audit=_audit(),
    )
    corpus = RealTraceStabilityCorpusSummaryV2.create(
        source_manifest_ref=_manifest_ref(),
        policy_ref=_policy().to_ref(),
        private_inventory_ref=inventory.to_ref(),
        eligible_unique_trace_count=1,
        unique_instance_count=1,
        unique_raw_hash_count=1,
        total_raw_bytes=prepared.member.size_bytes,
        minimum_raw_bytes=prepared.member.size_bytes,
        maximum_raw_bytes=prepared.member.size_bytes,
        category_counts=(
            RealTraceStabilityCategoryCountV2(
                category="science",
                count=1,
            ),
        ),
        audit=_audit(),
    )
    report = RealTraceStabilityReportV2.create(
        policy_ref=_policy().to_ref(),
        corpus_summary=corpus,
        private_result_set_ref=result_set.to_ref(),
        case_summaries=result_set.case_summaries,
        audit=_audit(),
    )
    return inventory, case, result_set, report


def _material_store(
    root: Path,
    *,
    max_private_bytes: int = 10_000_000,
    fault_injector=None,
) -> RealTraceStabilityMaterialStore:
    return RealTraceStabilityMaterialStore(
        root,
        max_private_bytes=max_private_bytes,
        max_members=100,
        fault_injector=fault_injector,
    )


def _report_store(
    root: Path,
    *,
    max_report_bytes: int = 10_000_000,
    fault_injector=None,
) -> RealTraceStabilityReportStore:
    return RealTraceStabilityReportStore(
        root,
        max_report_bytes=max_report_bytes,
        max_cases=100,
        fault_injector=fault_injector,
    )


def test_private_material_and_public_report_round_trip(tmp_path: Path) -> None:
    inventory, case, result_set, report = _values(tmp_path / "fixture")
    material = _material_store(tmp_path / "material")
    reports = _report_store(tmp_path / "reports")

    assert material.put_inventory(inventory).written is True
    assert material.put_case_result(case).written is True
    assert material.put_result_set(result_set).written is True
    assert material.put_inventory(inventory).written is False
    assert material.get_inventory(inventory.to_ref()) == inventory
    assert material.get_case_result(case.to_ref()) == case
    assert material.get_result_set(result_set.to_ref()) == result_set
    assert reports.put(report).written is True
    assert reports.put(report).written is False
    assert reports.get(report.to_ref()) == report
    assert not hasattr(material, "root")
    assert not hasattr(reports, "root")


def test_result_set_identity_binds_public_case_summary_refs(
    tmp_path: Path,
) -> None:
    _inventory, case, result_set, _report = _values(tmp_path / "fixture")
    alternate = RealTraceStabilityCaseSummaryV2.create(
        private_case_result_ref=case.to_ref(),
        outcome=RealTraceStabilityCaseOutcomeV2.INFRASTRUCTURE_ERROR,
        reason_code=RealTraceStabilityReasonCodeV2.INTERNAL_ERROR,
        parse_quality=None,
        assigned_fault_point=case.assigned_fault_point,
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

    with pytest.raises(ValueError, match="identity is stale"):
        RealTraceStabilityResultSetV1.model_validate(
            result_set.model_copy(update={"case_summaries": (alternate,)}).model_dump(mode="python")
        )


def test_store_limits_types_and_unsafe_roots_fail_closed(
    tmp_path: Path,
) -> None:
    inventory, _case, _result_set, report = _values(tmp_path / "fixture")
    with pytest.raises(RealTraceStabilityStoreLimitError):
        RealTraceStabilityMaterialStore(
            tmp_path / "invalid",
            max_private_bytes=1,
            max_members=1,
        )
    with pytest.raises(RealTraceStabilityStoreLimitError, match="byte"):
        _material_store(
            tmp_path / "small",
            max_private_bytes=2,
        ).put_inventory(inventory)
    wrong = ObjectRef(
        object_type="wrong",
        object_id="wrong://r8-04/ref",
        object_version="v2",
        object_sha256="a" * 64,
    )
    with pytest.raises(RealTraceStabilityStoreTypeError):
        _material_store(tmp_path / "material").get_inventory(wrong)
    with pytest.raises(RealTraceStabilityStoreTypeError):
        _report_store(tmp_path / "reports").get(wrong)
    root_file = tmp_path / "file"
    root_file.write_text("x", encoding="utf-8")
    with pytest.raises(RealTraceStabilityStoreTypeError, match="non-symlink"):
        _report_store(root_file)
    root_link = tmp_path / "link"
    root_link.symlink_to(tmp_path / "reports", target_is_directory=True)
    with pytest.raises(RealTraceStabilityStoreTypeError, match="non-symlink"):
        _report_store(root_link)
    with pytest.raises(RealTraceStabilityStoreLimitError, match="case"):
        RealTraceStabilityReportStore(
            tmp_path / "few",
            max_report_bytes=10_000_000,
            max_cases=0,
        ).put(report)


def test_store_corruption_and_cas_orphan_are_not_visible(
    tmp_path: Path,
) -> None:
    inventory, _case, _result_set, report = _values(tmp_path / "fixture")
    material_root = tmp_path / "material"
    crashing = _material_store(
        material_root,
        fault_injector=StaticRealTraceStabilityStoreFaultInjector(
            crash_points=frozenset({RealTraceStabilityStoreFaultPoint.AFTER_CAS_WRITE})
        ),
    )
    with pytest.raises(RealTraceStabilityStoreInjectedCrash):
        crashing.put_inventory(inventory)
    clean = _material_store(material_root)
    with pytest.raises(RealTraceStabilityStoreIntegrityError, match="envelope"):
        clean.get_inventory(inventory.to_ref())
    retried = clean.put_inventory(inventory)
    assert retried.written is True
    assert retried.content_blob_written is False

    report_root = tmp_path / "reports"
    store = _report_store(report_root)
    write = store.put(report)
    envelope_path = (
        report_root
        / "envelopes"
        / RealTraceStabilityStoreCodec.REPORT.value
        / "sha256"
        / write.object_ref.object_sha256[:2]
        / f"{write.object_ref.object_sha256}.json"
    )
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    digest = envelope["content_blob_ref"]["object_sha256"]
    content_path = report_root / "cas" / "sha256" / digest[:2] / digest
    content_path.write_bytes(b"{}")
    with pytest.raises(RealTraceStabilityStoreIntegrityError, match="corrupt"):
        store.get(report.to_ref())


def test_post_envelope_crash_replays_first_authority(
    tmp_path: Path,
) -> None:
    _inventory, _case, _result_set, report = _values(tmp_path / "fixture")
    root = tmp_path / "reports"
    crashing = _report_store(
        root,
        fault_injector=StaticRealTraceStabilityStoreFaultInjector(
            crash_points=frozenset({RealTraceStabilityStoreFaultPoint.AFTER_ENVELOPE_WRITE})
        ),
    )
    with pytest.raises(RealTraceStabilityStoreInjectedCrash):
        crashing.put(report)

    clean = _report_store(root)
    assert clean.put(report).written is False
    assert clean.get(report.to_ref()) == report
