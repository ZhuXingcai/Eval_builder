from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedDatasetJobPlanV2,
    ResolvedStageNodeV2,
    StageNameV2,
    dataset_job_spec_v2_ref,
    resolved_dataset_job_plan_v2_ref,
)
from eval_factory.orchestration.planning import DatasetJobPlanCompiler

HASH = "a" * 64
NOW = datetime(2026, 7, 31, tzinfo=UTC)


def _ref(name: str) -> ObjectRef:
    return ObjectRef(
        object_type=name,
        object_id=f"{name}://r6-01/contract",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit(*, created_at: datetime = NOW, created_by: str = "r6-contract-test") -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by=created_by,
        governing_versions=(VersionBinding(component="eval-factory-spec", version="approved-v2"),),
    )


def _spec(
    *,
    job_id: str = "job://r6-01/contract",
    requested_stages: tuple[StageNameV2, ...] = (
        StageNameV2.TRACE_INDEX,
        StageNameV2.LABEL,
    ),
) -> DatasetJobSpecV2:
    return DatasetJobSpecV2(
        job_id=job_id,
        traces=(
            TraceSourceRef(
                source_trace_id="source-trace://r6-01/contract",
                source_uri="raw-traj://r6-01/contract",
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
        approval_mode=ApprovalMode.NONE,
        enabled_checkpoints=frozenset(),
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://canary",
        ),
        idempotency_key=f"create-{job_id.removeprefix('job://').replace('/', '-')}",
        audit=_audit(),
    )


def _plan() -> ResolvedDatasetJobPlanV2:
    return DatasetJobPlanCompiler().compile(job_spec=_spec(), audit=_audit())


def test_resolved_plan_contract_is_strict_frozen_and_round_trips() -> None:
    plan = _plan()

    assert ResolvedDatasetJobPlanV2.model_validate_json(plan.model_dump_json()) == plan
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResolvedDatasetJobPlanV2.model_validate(
            {
                **plan.model_dump(mode="python"),
                "raw_trace": "must-not-cross-the-contract",
            }
        )
    with pytest.raises(ValidationError, match="frozen"):
        plan.policy_version = "changed"  # type: ignore[misc]


def test_job_spec_and_resolved_plan_refs_bind_exact_current_hashes() -> None:
    spec = _spec()
    plan = DatasetJobPlanCompiler().compile(job_spec=spec, audit=_audit())

    spec_ref = dataset_job_spec_v2_ref(spec)
    plan_ref = resolved_dataset_job_plan_v2_ref(plan)

    assert spec_ref.object_type == "dataset-job-spec"
    assert spec_ref.object_id == spec.job_id
    assert spec_ref.object_version == "v2"
    assert spec_ref.object_sha256 == spec.canonical_sha256()
    assert plan.dataset_job_spec_ref == spec_ref
    assert plan_ref.object_type == "resolved-dataset-job-plan"
    assert plan_ref.object_id == plan.resolved_job_plan_id
    assert plan_ref.object_version == "v2"
    assert plan_ref.object_sha256 == plan.resolved_job_plan_sha256


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("resolved_job_plan_sha256", "b" * 64, "identity is stale"),
        (
            "resolved_job_plan_id",
            "resolved-job-plan://sha256/" + "b" * 64,
            "identity is stale",
        ),
        (
            "dataset_job_spec_ref",
            ObjectRef(
                object_type="dataset-job-spec",
                object_id="job://r6-01/contract",
                object_version="v2",
                object_sha256="b" * 64,
            ),
            "audit refs are incomplete",
        ),
    ],
)
def test_resolved_plan_contract_rejects_stale_identity_and_source_ref(
    field: str,
    replacement: object,
    message: str,
) -> None:
    payload = _plan().model_dump(mode="python")
    payload[field] = replacement

    with pytest.raises(ValidationError, match=message):
        ResolvedDatasetJobPlanV2.model_validate(payload)


def test_resolved_plan_identity_excludes_audit_actor_and_time() -> None:
    spec = _spec()
    compiler = DatasetJobPlanCompiler()
    first = compiler.compile(job_spec=spec, audit=_audit())
    second = compiler.compile(
        job_spec=spec,
        audit=_audit(
            created_at=datetime(2030, 1, 1, tzinfo=UTC),
            created_by="different-r6-operator",
        ),
    )

    assert first.audit != second.audit
    assert first.resolved_job_plan_id == second.resolved_job_plan_id
    assert first.resolved_job_plan_sha256 == second.resolved_job_plan_sha256


def test_resolved_plan_contract_rejects_wrong_version_and_overlong_closure() -> None:
    plan = _plan()
    wrong_version = plan.model_dump(mode="python")
    wrong_version["schema_version"] = "eval-factory/resolved-dataset-job-plan/v1"
    with pytest.raises(ValidationError, match="literal_error"):
        ResolvedDatasetJobPlanV2.model_validate(wrong_version)

    overlong = plan.model_dump(mode="python")
    overlong.update(
        {
            "resolved_job_plan_id": "resolved-job-plan://pending",
            "resolved_stages": (
                *plan.resolved_stages,
                StageNameV2.TASK_AUTHORING,
            ),
            "auto_added_stages": (
                *plan.auto_added_stages,
                StageNameV2.TASK_AUTHORING,
            ),
            "nodes": (
                *plan.nodes,
                ResolvedStageNodeV2(
                    stage=StageNameV2.TASK_AUTHORING,
                    depends_on=(StageNameV2.LABEL,),
                ),
            ),
            "resolved_job_plan_sha256": "0" * 64,
        }
    )
    with pytest.raises(ValidationError, match="highest requested stage"):
        ResolvedDatasetJobPlanV2.model_validate(overlong)


def test_resolved_plan_contract_rejects_conflicting_policy_audit_bindings() -> None:
    plan = _plan()
    payload = plan.model_dump(mode="python")
    payload.update(
        {
            "resolved_job_plan_id": "resolved-job-plan://pending",
            "resolved_job_plan_sha256": "0" * 64,
            "audit": ContractAudit(
                created_at=plan.audit.created_at,
                created_by=plan.audit.created_by,
                governing_versions=(
                    *plan.audit.governing_versions,
                    VersionBinding(
                        component="dataset-job-stage-policy",
                        version="dataset-job-stage-policy/stale",
                    ),
                ),
                input_refs=plan.audit.input_refs,
            ),
        }
    )

    with pytest.raises(ValidationError, match="current stage policy"):
        ResolvedDatasetJobPlanV2.model_validate(payload)
