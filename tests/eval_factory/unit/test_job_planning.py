from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.approval import ApprovalCheckpoint, ApprovalMode
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedStageNodeV2,
    StageNameV2,
)
from eval_factory.orchestration.planning import (
    R6_STAGE_PLAN_POLICY_VERSION,
    DatasetJobPlanCompiler,
    JobPlanningPolicyError,
)

HASH = "a" * 64
NOW = datetime(2026, 7, 31, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = ROOT / "evals/golden/eval_factory/orchestration/r6-01-stage-dag-v1.json"

BASE_CHAIN = (
    StageNameV2.TRACE_INDEX,
    StageNameV2.SAFETY,
    StageNameV2.LABEL,
    StageNameV2.TASK_AUTHORING,
    StageNameV2.ATTACHMENT,
    StageNameV2.ITEM_QUALITY,
    StageNameV2.BATCH_QUALITY,
    StageNameV2.RELEASE,
)


def _ref(name: str) -> ObjectRef:
    return ObjectRef(
        object_type=name,
        object_id=f"{name}://r6-01/planner",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r6-planner-test",
        governing_versions=(VersionBinding(component="eval-factory-spec", version="approved-v2"),),
    )


def _spec(
    requested_stages: tuple[StageNameV2, ...],
    *,
    approval_mode: ApprovalMode = ApprovalMode.NONE,
    enabled_checkpoints: frozenset[ApprovalCheckpoint] = frozenset(),
    job_id: str = "job://r6-01/planner",
    idempotency_key: str = "create-job-r6-01-planner",
) -> DatasetJobSpecV2:
    return DatasetJobSpecV2(
        job_id=job_id,
        traces=(
            TraceSourceRef(
                source_trace_id="source-trace://r6-01/planner",
                source_uri="raw-traj://r6-01/planner",
                raw_sha256=HASH,
                adapter_name="raw-traj-v1",
                adapter_version="v1",
                processing_class="RESTRICTED_TRACE_RAW",
            ),
        ),
        requested_stages=requested_stages,
        privacy_profile="trusted-monitored-local",
        model_profiles=(),
        budget=ResourceBudget(
            max_model_requests=10,
            max_model_tokens=1000,
            max_processes=2,
            max_renderers=1,
            max_network_requests=5,
            max_storage_bytes=1024 * 1024,
        ),
        concurrency=ConcurrencyLimit(
            model_requests=1,
            processes=1,
            renderers=1,
            network_requests=1,
            artifacts_per_item=1,
            items=1,
        ),
        approval_policy_ref=_ref("user-approval-policy"),
        approval_mode=approval_mode,
        enabled_checkpoints=enabled_checkpoints,
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://canary",
        ),
        idempotency_key=idempotency_key,
        audit=_audit(),
    )


def _direct_chain(
    stages: tuple[StageNameV2, ...],
) -> tuple[tuple[StageNameV2, tuple[StageNameV2, ...]], ...]:
    return tuple((stage, () if index == 0 else (stages[index - 1],)) for index, stage in enumerate(stages))


def test_compile_preserves_complete_no_checkpoint_chain() -> None:
    spec = _spec(BASE_CHAIN)

    plan = DatasetJobPlanCompiler().compile(job_spec=spec, audit=_audit())

    assert plan.requested_stages == BASE_CHAIN
    assert plan.resolved_stages == BASE_CHAIN
    assert plan.auto_added_stages == ()
    assert tuple((node.stage, node.depends_on) for node in plan.nodes) == _direct_chain(BASE_CHAIN)
    assert plan.policy_version == R6_STAGE_PLAN_POLICY_VERSION


@pytest.mark.parametrize(
    ("requested", "resolved", "auto_added"),
    [
        (
            (StageNameV2.TRACE_INDEX, StageNameV2.LABEL),
            (
                StageNameV2.TRACE_INDEX,
                StageNameV2.SAFETY,
                StageNameV2.LABEL,
            ),
            (StageNameV2.SAFETY,),
        ),
        (
            (StageNameV2.TRACE_INDEX, StageNameV2.ITEM_QUALITY),
            (
                StageNameV2.TRACE_INDEX,
                StageNameV2.SAFETY,
                StageNameV2.LABEL,
                StageNameV2.TASK_AUTHORING,
                StageNameV2.ATTACHMENT,
                StageNameV2.ITEM_QUALITY,
            ),
            (
                StageNameV2.SAFETY,
                StageNameV2.LABEL,
                StageNameV2.TASK_AUTHORING,
                StageNameV2.ATTACHMENT,
            ),
        ),
        (
            (StageNameV2.TRACE_INDEX, StageNameV2.RELEASE),
            BASE_CHAIN,
            BASE_CHAIN[1:-1],
        ),
    ],
)
def test_compile_adds_exact_transitive_prerequisite_closure(
    requested: tuple[StageNameV2, ...],
    resolved: tuple[StageNameV2, ...],
    auto_added: tuple[StageNameV2, ...],
) -> None:
    plan = DatasetJobPlanCompiler().compile(
        job_spec=_spec(requested),
        audit=_audit(),
    )

    assert plan.requested_stages == requested
    assert plan.resolved_stages == resolved
    assert plan.auto_added_stages == auto_added
    assert tuple((node.stage, node.depends_on) for node in plan.nodes) == _direct_chain(resolved)


@pytest.mark.parametrize(
    ("approval_mode", "enabled_checkpoints", "requested", "expected"),
    [
        (
            ApprovalMode.NONE,
            frozenset(),
            (StageNameV2.TRACE_INDEX, StageNameV2.RELEASE),
            BASE_CHAIN,
        ),
        (
            ApprovalMode.PLAN_GATES,
            frozenset(
                {
                    ApprovalCheckpoint.LABEL_PLAN,
                    ApprovalCheckpoint.TASK_REWRITE_PLAN,
                    ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
                }
            ),
            (
                StageNameV2.TRACE_INDEX,
                StageNameV2.LABEL_PLAN,
                StageNameV2.TASK_REWRITE_PLAN,
                StageNameV2.ENVIRONMENT_STRATEGY,
                StageNameV2.RELEASE,
            ),
            (
                StageNameV2.TRACE_INDEX,
                StageNameV2.SAFETY,
                StageNameV2.LABEL_PLAN,
                StageNameV2.LABEL,
                StageNameV2.TASK_REWRITE_PLAN,
                StageNameV2.TASK_AUTHORING,
                StageNameV2.ENVIRONMENT_STRATEGY,
                StageNameV2.ATTACHMENT,
                StageNameV2.ITEM_QUALITY,
                StageNameV2.BATCH_QUALITY,
                StageNameV2.RELEASE,
            ),
        ),
        (
            ApprovalMode.FINAL_ONLY,
            frozenset({ApprovalCheckpoint.FINAL_DATASET_REVIEW}),
            (
                StageNameV2.TRACE_INDEX,
                StageNameV2.FINAL_DATASET_REVIEW,
                StageNameV2.RELEASE,
            ),
            (
                *BASE_CHAIN[:-1],
                StageNameV2.FINAL_DATASET_REVIEW,
                StageNameV2.RELEASE,
            ),
        ),
        (
            ApprovalMode.PLAN_AND_FINAL,
            frozenset(ApprovalCheckpoint),
            (
                StageNameV2.TRACE_INDEX,
                StageNameV2.LABEL_PLAN,
                StageNameV2.TASK_REWRITE_PLAN,
                StageNameV2.ENVIRONMENT_STRATEGY,
                StageNameV2.FINAL_DATASET_REVIEW,
                StageNameV2.RELEASE,
            ),
            tuple(StageNameV2),
        ),
        (
            ApprovalMode.CUSTOM,
            frozenset(
                {
                    ApprovalCheckpoint.LABEL_PLAN,
                    ApprovalCheckpoint.FINAL_DATASET_REVIEW,
                }
            ),
            (
                StageNameV2.TRACE_INDEX,
                StageNameV2.LABEL_PLAN,
                StageNameV2.FINAL_DATASET_REVIEW,
                StageNameV2.RELEASE,
            ),
            (
                StageNameV2.TRACE_INDEX,
                StageNameV2.SAFETY,
                StageNameV2.LABEL_PLAN,
                StageNameV2.LABEL,
                StageNameV2.TASK_AUTHORING,
                StageNameV2.ATTACHMENT,
                StageNameV2.ITEM_QUALITY,
                StageNameV2.BATCH_QUALITY,
                StageNameV2.FINAL_DATASET_REVIEW,
                StageNameV2.RELEASE,
            ),
        ),
    ],
)
def test_compile_inserts_only_enabled_checkpoint_stages_at_canonical_boundaries(
    approval_mode: ApprovalMode,
    enabled_checkpoints: frozenset[ApprovalCheckpoint],
    requested: tuple[StageNameV2, ...],
    expected: tuple[StageNameV2, ...],
) -> None:
    spec = _spec(
        requested,
        approval_mode=approval_mode,
        enabled_checkpoints=enabled_checkpoints,
    )

    plan = DatasetJobPlanCompiler().compile(job_spec=spec, audit=_audit())

    assert plan.resolved_stages == expected
    assert tuple((node.stage, node.depends_on) for node in plan.nodes) == _direct_chain(expected)


@pytest.mark.parametrize(
    "requested",
    [
        (StageNameV2.LABEL, StageNameV2.TRACE_INDEX),
        (StageNameV2.TRACE_INDEX, StageNameV2.LABEL, StageNameV2.SAFETY),
        (StageNameV2.TRACE_INDEX, StageNameV2.LABEL, StageNameV2.LABEL),
    ],
)
def test_invalid_requested_order_is_rejected_before_planning(
    requested: tuple[StageNameV2, ...],
) -> None:
    with pytest.raises(ValidationError):
        _spec(requested)


@pytest.mark.parametrize(
    "replacement",
    [
        ResolvedStageNodeV2.model_construct(
            stage=StageNameV2.SAFETY,
            depends_on=(StageNameV2.SAFETY,),
        ),
        ResolvedStageNodeV2.model_construct(
            stage=StageNameV2.SAFETY,
            depends_on=(StageNameV2.LABEL,),
        ),
        ResolvedStageNodeV2.model_construct(
            stage=StageNameV2.SAFETY,
            depends_on=(StageNameV2.TRACE_INDEX, StageNameV2.TRACE_INDEX),
        ),
    ],
)
def test_validate_current_fails_closed_for_self_forward_and_duplicate_edges(
    replacement: ResolvedStageNodeV2,
) -> None:
    spec = _spec((StageNameV2.TRACE_INDEX, StageNameV2.LABEL))
    compiler = DatasetJobPlanCompiler()
    plan = compiler.compile(job_spec=spec, audit=_audit())
    malformed = plan.model_copy(update={"nodes": (plan.nodes[0], replacement, plan.nodes[2])})

    with pytest.raises(JobPlanningPolicyError):
        compiler.validate_current(malformed, job_spec=spec)


def test_validate_current_fails_closed_for_unknown_stage() -> None:
    spec = _spec((StageNameV2.TRACE_INDEX, StageNameV2.LABEL))
    compiler = DatasetJobPlanCompiler()
    plan = compiler.compile(job_spec=spec, audit=_audit())
    unknown = ResolvedStageNodeV2.model_construct(
        stage="unknown_stage",  # type: ignore[arg-type]
        depends_on=(StageNameV2.TRACE_INDEX,),
    )
    malformed = plan.model_copy(update={"nodes": (plan.nodes[0], unknown, plan.nodes[2])})

    with (
        pytest.warns(UserWarning, match="serialized value may not be as expected"),
        pytest.raises(JobPlanningPolicyError),
    ):
        compiler.validate_current(malformed, job_spec=spec)


def test_validate_current_fails_closed_for_cyclic_root_edge() -> None:
    spec = _spec((StageNameV2.TRACE_INDEX, StageNameV2.LABEL))
    compiler = DatasetJobPlanCompiler()
    plan = compiler.compile(job_spec=spec, audit=_audit())
    malformed_root = ResolvedStageNodeV2.model_construct(
        stage=StageNameV2.TRACE_INDEX,
        depends_on=(StageNameV2.LABEL,),
    )
    malformed = plan.model_copy(update={"nodes": (malformed_root, *plan.nodes[1:])})

    with pytest.raises(JobPlanningPolicyError):
        compiler.validate_current(malformed, job_spec=spec)


def test_validate_current_rejects_stale_policy_and_cross_job_spec() -> None:
    compiler = DatasetJobPlanCompiler()
    spec = _spec((StageNameV2.TRACE_INDEX, StageNameV2.LABEL))
    plan = compiler.compile(job_spec=spec, audit=_audit())
    stale = plan.model_copy(update={"policy_version": "stage-policy/stale"})

    with pytest.raises(JobPlanningPolicyError, match="policy"):
        compiler.validate_current(stale, job_spec=spec)

    other = _spec(
        (StageNameV2.TRACE_INDEX, StageNameV2.LABEL),
        job_id="job://r6-01/other",
        idempotency_key="create-job-r6-01-other",
    )
    with pytest.raises(JobPlanningPolicyError, match="spec"):
        compiler.validate_current(plan, job_spec=other)


def test_stage_dag_gold_cases_are_executable_and_identity_bound() -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    assert gold["schema_version"] == "eval-factory/r6-01-stage-dag-gold/v1"
    assert gold["claim_scope"] == "CANONICAL_STAGE_DAG_AND_RESOLVED_PLAN_ONLY"
    assert len(gold["cases"]) == 4
    for index, case in enumerate(gold["cases"]):
        checkpoints = frozenset(ApprovalCheckpoint(value) for value in case["enabled_checkpoints"])
        requested = tuple(StageNameV2(value) for value in case["requested_stages"])
        spec = _spec(
            requested,
            approval_mode=ApprovalMode(case["approval_mode"]),
            enabled_checkpoints=checkpoints,
            job_id=f"job://r6-01/gold/{index}",
            idempotency_key=f"create-job-r6-01-gold-{index}",
        )

        plan = DatasetJobPlanCompiler().compile(job_spec=spec, audit=_audit())

        assert [stage.value for stage in plan.resolved_stages] == case["resolved_stages"]
        assert [stage.value for stage in plan.auto_added_stages] == case["auto_added_stages"]
        assert [
            {
                "stage": node.stage.value,
                "depends_on": [stage.value for stage in node.depends_on],
            }
            for node in plan.nodes
        ] == case["edges"]
        assert plan.policy_version == case["policy_version"]
        assert plan.resolved_job_plan_sha256 == case["resolved_job_plan_sha256"]


def test_stage_dag_identity_is_stable_across_python_hash_seeds() -> None:
    script = """
import runpy
ns = runpy.run_path("tests/eval_factory/unit/test_job_planning.py")
StageNameV2 = ns["StageNameV2"]
spec = ns["_spec"](
    (StageNameV2.TRACE_INDEX, StageNameV2.RELEASE),
    job_id="job://r6-01/gold/1",
    idempotency_key="create-job-r6-01-gold-1",
)
plan = ns["DatasetJobPlanCompiler"]().compile(job_spec=spec, audit=ns["_audit"]())
print(plan.resolved_job_plan_sha256)
"""
    observed = []
    for seed in ("1", "321"):
        process = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        observed.append(process.stdout.strip())

    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    expected = next(
        case["resolved_job_plan_sha256"]
        for case in gold["cases"]
        if case["case_id"] == "no-checkpoint-release"
    )
    assert observed == [expected, expected]
