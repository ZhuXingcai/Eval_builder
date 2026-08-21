from __future__ import annotations

from pathlib import Path

import pytest
from test_factory_dataset_store import _audit
from test_release_projection import _approval_policy

from eval_factory.agent_system.job_store_bridge import (
    FactoryJobStoreBridge,
    FactoryJobStoreBridgeError,
    FactoryJobStoreTemplate,
)
from eval_factory.contracts.approval import (
    ApprovalMode,
    FinalReviewScope,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ItemStatus,
    JobStatus,
    ResourceBudget,
)
from eval_factory.contracts.orchestration_v2 import (
    StageNameV2,
    dataset_item_id_v2,
)
from eval_factory.orchestration.job_store import JobStore


def _template() -> FactoryJobStoreTemplate:
    return FactoryJobStoreTemplate(
        requested_stages=(
            StageNameV2.TRACE_INDEX,
            StageNameV2.SAFETY,
            StageNameV2.LABEL,
            StageNameV2.TASK_AUTHORING,
            StageNameV2.ATTACHMENT,
            StageNameV2.ITEM_QUALITY,
            StageNameV2.BATCH_QUALITY,
            StageNameV2.RELEASE,
        ),
        privacy_profile="trusted-monitored-local",
        model_profiles=(),
        budget=ResourceBudget(
            max_model_requests=100,
            max_model_tokens=1_000_000,
            max_processes=4,
            max_renderers=4,
            max_network_requests=100,
            max_storage_bytes=10_000_000,
        ),
        concurrency=ConcurrencyLimit(
            model_requests=2,
            processes=2,
            renderers=2,
            network_requests=2,
            artifacts_per_item=4,
            items=4,
        ),
        approval_policy=_approval_policy(
            ApprovalMode.NONE,
            scope=FinalReviewScope.NONE,
        ),
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://factory-candidate",
        ),
    )


def _source(suffix: str) -> ObjectRef:
    return ObjectRef(
        object_type="trace-source",
        object_id=f"trace-source://{suffix}",
        object_version="1.0.0",
        object_sha256=("a" * 63 + ("1" if suffix == "alpha" else "2")),
    )


def _sources() -> tuple[ObjectRef, ...]:
    return (
        _source("alpha"),
        _source("beta"),
    )


def test_job_store_bridge_creates_and_replays_canonical_parent_graph(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "job.sqlite3")
    bridge = FactoryJobStoreBridge(
        job_store=store,
        template=_template(),
    )

    first = bridge.prepare(
        dataset_run_id="factory-run://dataset/job-bridge",
        source_trace_refs=_sources(),
        audit=_audit(),
    )
    replay = bridge.prepare(
        dataset_run_id="factory-run://dataset/job-bridge",
        source_trace_refs=tuple(reversed(_sources())),
        audit=_audit(),
    )

    assert replay == first
    assert first.job_spec.job_id == ("factory-run://dataset/job-bridge")
    assert first.graph.item_ids == tuple(
        dataset_item_id_v2(
            first.job_spec.job_id,
            source_ref,
        )
        for source_ref in first.graph.source_trace_refs
    )
    assert store.get_job(first.job_spec.job_id).status is JobStatus.RUNNING
    assert tuple(item.status for item in store.list_items(first.job_spec.job_id)) == (
        ItemStatus.RUNNING,
        ItemStatus.RUNNING,
    )


def test_job_store_bridge_rejects_source_and_template_drift(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "job.sqlite3")
    bridge = FactoryJobStoreBridge(
        job_store=store,
        template=_template(),
    )
    bridge.prepare(
        dataset_run_id="factory-run://dataset/job-bridge",
        source_trace_refs=_sources(),
        audit=_audit(),
    )

    with pytest.raises(
        FactoryJobStoreBridgeError,
        match="authority differs",
    ):
        bridge.prepare(
            dataset_run_id=("factory-run://dataset/job-bridge"),
            source_trace_refs=(
                _sources()[0],
                _source("changed"),
            ),
            audit=_audit(),
        )

    changed = FactoryJobStoreBridge(
        job_store=store,
        template=FactoryJobStoreTemplate(
            requested_stages=_template().requested_stages,
            privacy_profile="different-profile",
            model_profiles=(),
            budget=_template().budget,
            concurrency=_template().concurrency,
            approval_policy=_template().approval_policy,
            export_target=_template().export_target,
        ),
    )
    with pytest.raises(
        FactoryJobStoreBridgeError,
        match="authority differs",
    ):
        changed.prepare(
            dataset_run_id=("factory-run://dataset/job-bridge"),
            source_trace_refs=_sources(),
            audit=_audit(),
        )
