from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration_v2 import StageNameV2
from eval_factory.contracts.real_trace_stability_v2 import (
    REAL_TRACE_STABILITY_POLICY_VERSION,
    RealTraceStabilityFaultPointV2,
    RealTraceStabilityPolicyV2,
)
from eval_factory.readiness.real_trace_stability_builder import (
    RealTraceStabilityBuilder,
    RealTraceStabilityManifestError,
    RealTraceStabilitySourceError,
)
from eval_factory.readiness.real_trace_stability_models import (
    RealTraceStabilityInventoryV1,
    RealTraceStabilityMemberV1,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE_MANIFEST = ROOT.parent / "raw_traj/manifest.csv"
RAW_ROOT = ROOT.parent / "raw_traj"
NOW = datetime(2026, 8, 2, tzinfo=UTC)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r8-04-builder-test",
        governing_versions=(
            VersionBinding(
                component="real-trace-stability",
                version=REAL_TRACE_STABILITY_POLICY_VERSION,
            ),
        ),
    )


def _manifest_ref() -> ObjectRef:
    return ObjectRef(
        object_type="real-trace-source-manifest",
        object_id="real-trace-source-manifest://manifest.csv",
        object_version="csv/v1",
        object_sha256="0d9df6c3935e00e5e15d486c9afe8cc2f581b63cf8f99ec42c1d3216cc74ee04",
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


def test_builder_admits_exact_91_source_inventory_and_fault_rotation() -> None:
    compilation = RealTraceStabilityBuilder().compile_inventory(
        source_manifest=SOURCE_MANIFEST,
        raw_root=RAW_ROOT,
        policy=_policy(),
        audit=_audit(),
    )

    assert len(compilation.inventory.members) == 91
    assert compilation.corpus_summary.eligible_unique_trace_count == 91
    assert compilation.corpus_summary.shortfall_count == 9
    assert compilation.corpus_summary.total_raw_bytes == 215_687_856
    assert compilation.corpus_summary.minimum_raw_bytes == 861_548
    assert compilation.corpus_summary.maximum_raw_bytes == 4_181_501
    assert sum(row.count for row in compilation.corpus_summary.category_counts) == 91
    assert Counter(member.assigned_fault_point for member in compilation.inventory.members) == {
        RealTraceStabilityFaultPointV2.BEFORE_SOURCE_REGISTRATION: 23,
        RealTraceStabilityFaultPointV2.AFTER_SOURCE_REGISTRATION: 23,
        RealTraceStabilityFaultPointV2.AFTER_TRACE_STORE_PERSIST: 23,
        RealTraceStabilityFaultPointV2.BEFORE_STAGE_COMPLETION: 22,
    }
    assert all(compilation.path_for(member.instance_id).is_file() for member in compilation.inventory.members)


def test_builder_compiles_one_private_r1_only_job_spec() -> None:
    builder = RealTraceStabilityBuilder()
    compilation = builder.compile_inventory(
        source_manifest=SOURCE_MANIFEST,
        raw_root=RAW_ROOT,
        policy=_policy(),
        audit=_audit(),
    )

    prepared = builder.prepare_case(
        compilation.inventory.members[0],
        raw_path=compilation.path_for(compilation.inventory.members[0].instance_id),
        policy=_policy(),
        audit=_audit(),
    )

    assert prepared.job_spec.requested_stages == (StageNameV2.TRACE_INDEX,)
    assert len(prepared.job_spec.traces) == 1
    assert prepared.job_spec.model_profiles == ()
    assert prepared.job_spec.budget.max_model_requests == 0
    assert prepared.job_spec.budget.max_network_requests == 0
    assert prepared.job_spec.approval_mode.value == "NONE"
    assert prepared.job_spec.enabled_checkpoints == frozenset()
    assert prepared.trace.raw_sha256 == prepared.member.raw_sha256
    assert prepared.member.instance_id not in prepared.job_spec.job_id


def test_manifest_parser_rejects_schema_date_duplicate_and_limit() -> None:
    builder = RealTraceStabilityBuilder()
    with pytest.raises(RealTraceStabilityManifestError, match="header"):
        builder.parse_manifest(b"wrong,header\nx,y\n", max_sources=10)
    with pytest.raises(RealTraceStabilityManifestError, match="date"):
        builder.parse_manifest(
            (
                b"instance_id,sid,p_date,business,category,pool_id,pool_category\n"
                b"LH_001,sid-001,not-a-date,AgentPlan,science,1,science\n"
            ),
            max_sources=10,
        )
    duplicate = (
        b"instance_id,sid,p_date,business,category,pool_id,pool_category\n"
        b"LH_001,sid-001,2026-06-20,AgentPlan,science,1,science\n"
        b"LH_001,sid-002,2026-06-20,AgentPlan,science,2,science\n"
    )
    with pytest.raises(RealTraceStabilityManifestError, match="duplicate instance"):
        builder.parse_manifest(duplicate, max_sources=10)
    with pytest.raises(RealTraceStabilityManifestError, match="row limit"):
        builder.parse_manifest(duplicate, max_sources=1)


def test_builder_rejects_manifest_and_raw_root_drift(tmp_path: Path) -> None:
    changed = tmp_path / "manifest.csv"
    changed.write_bytes(SOURCE_MANIFEST.read_bytes() + b"\n")
    with pytest.raises(RealTraceStabilityManifestError, match="authority"):
        RealTraceStabilityBuilder().compile_inventory(
            source_manifest=changed,
            raw_root=RAW_ROOT,
            policy=_policy(),
            audit=_audit(),
        )
    with pytest.raises(RealTraceStabilitySourceError, match="inventory"):
        RealTraceStabilityBuilder().compile_inventory(
            source_manifest=SOURCE_MANIFEST,
            raw_root=tmp_path,
            policy=_policy(),
            audit=_audit(),
        )
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RealTraceStabilitySourceError, match="non-symlink"):
        RealTraceStabilityBuilder().compile_inventory(
            source_manifest=SOURCE_MANIFEST,
            raw_root=link,
            policy=_policy(),
            audit=_audit(),
        )


def test_private_inventory_rejects_raw_hash_alias() -> None:
    first = RealTraceStabilityMemberV1.create(
        instance_id="LH_001",
        sid="sid-001",
        business="AgentPlan",
        category="science",
        pool_id="1",
        pool_category="science",
        raw_sha256="a" * 64,
        size_bytes=100,
        assigned_fault_point=(RealTraceStabilityFaultPointV2.BEFORE_SOURCE_REGISTRATION),
    )
    second = RealTraceStabilityMemberV1.create(
        instance_id="LH_002",
        sid="sid-002",
        business="AgentPlan",
        category="science",
        pool_id="2",
        pool_category="science",
        raw_sha256="a" * 64,
        size_bytes=100,
        assigned_fault_point=(RealTraceStabilityFaultPointV2.AFTER_SOURCE_REGISTRATION),
    )

    with pytest.raises(ValueError, match="duplicate raw hash"):
        RealTraceStabilityInventoryV1.create(
            source_manifest_ref=_manifest_ref(),
            members=(first, second),
            audit=_audit(),
        )


def test_private_inventory_rejects_noncanonical_fault_rotation() -> None:
    members = (
        RealTraceStabilityMemberV1.create(
            instance_id="LH_001",
            sid="sid-001",
            business="AgentPlan",
            category="science",
            pool_id="1",
            pool_category="science",
            raw_sha256="a" * 64,
            size_bytes=100,
            assigned_fault_point=RealTraceStabilityFaultPointV2.BEFORE_SOURCE_REGISTRATION,
        ),
        RealTraceStabilityMemberV1.create(
            instance_id="LH_002",
            sid="sid-002",
            business="AgentPlan",
            category="science",
            pool_id="2",
            pool_category="science",
            raw_sha256="b" * 64,
            size_bytes=100,
            assigned_fault_point=RealTraceStabilityFaultPointV2.BEFORE_SOURCE_REGISTRATION,
        ),
    )

    with pytest.raises(ValueError, match="fault assignment"):
        RealTraceStabilityInventoryV1.create(
            source_manifest_ref=_manifest_ref(),
            members=members,
            audit=_audit(),
        )


def test_member_identity_is_stable_across_hash_seeds() -> None:
    script = """
from eval_factory.contracts.real_trace_stability_v2 import RealTraceStabilityFaultPointV2
from eval_factory.readiness.real_trace_stability_models import RealTraceStabilityMemberV1
value = RealTraceStabilityMemberV1.create(
    instance_id="LH_001",
    sid="sid-001",
    business="AgentPlan",
    category="science",
    pool_id="1",
    pool_category="science",
    raw_sha256="a" * 64,
    size_bytes=100,
    assigned_fault_point=RealTraceStabilityFaultPointV2.BEFORE_SOURCE_REGISTRATION,
)
print(value.to_ref().model_dump_json())
"""
    outputs = {
        subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        for seed in ("1", "321")
    }
    assert len(outputs) == 1
