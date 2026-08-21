from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.canary_execution_v2 import (
    R6_CANARY_CLAIM_SCOPE,
    R6CanaryControlPlaneV2,
    R6CanaryDatasetResultV2,
    R6CanaryExecutionManifestV2,
    R6CanaryExpectedItemOutcomeV2,
    R6CanaryObjectCodecV2,
    R6CanarySeedObjectV2,
    R6CanaryTraceBindingV2,
    SemanticReviewFanoutV2,
    SemanticReviewRoundWorkV2,
)
from eval_factory.contracts.cli_v2 import PipelineControlConfigV2
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.observability_v2 import (
    BatchAuditOutcomeV2,
)
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    JobStatus,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedWorkUnitV2,
    StageNameV2,
    WorkDependencyJoinModeV2,
    WorkUnitScopeV2,
    trace_source_v2_ref,
)
from eval_factory.contracts.resource_v2 import (
    NonModelResourceVectorV2,
)
from eval_factory.contracts.review_v2 import (
    SemanticReviewRoundV2,
)
from eval_factory.orchestration.fanout import (
    DatasetJobWorkGraphCompiler,
)
from eval_factory.orchestration.planning import (
    DatasetJobPlanCompiler,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
HASH = "a" * 64
ROOT = Path(__file__).parents[3]
CANARY_STAGES = (
    StageNameV2.TRACE_INDEX,
    StageNameV2.SAFETY,
    StageNameV2.LABEL,
    StageNameV2.TASK_AUTHORING,
    StageNameV2.ATTACHMENT,
    StageNameV2.ITEM_QUALITY,
)


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-parent/{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit(
    *,
    created_by: str = "r6-parent-contract-test",
    created_at: datetime = NOW,
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by=created_by,
        governing_versions=(
            VersionBinding(
                component="eval-factory-spec",
                version="approved-v2",
            ),
            VersionBinding(
                component="artifact-execution",
                version="r5-06",
            ),
            VersionBinding(
                component="artifact-results",
                version="r5-07",
            ),
            VersionBinding(
                component="deterministic-validation",
                version="r5-08",
            ),
            VersionBinding(
                component="semantic-review",
                version="r5-09",
            ),
            VersionBinding(
                component="item-quality",
                version="r5-10",
            ),
        ),
    )


def _trace(suffix: str = "a") -> TraceSourceRef:
    return TraceSourceRef(
        source_trace_id=f"source-trace://r6-parent/{suffix}",
        source_uri=f"raw-traj://r6-parent/{suffix}",
        raw_sha256=HASH,
        adapter_name="raw-traj-v1",
        adapter_version="v1",
        processing_class="RESTRICTED_TRACE_RAW",
    )


def _spec(
    *,
    requested_stages: tuple[StageNameV2, ...] = CANARY_STAGES,
    channel: str = "CANARY",
) -> DatasetJobSpecV2:
    return DatasetJobSpecV2(
        job_id="job://r6-parent/canary",
        traces=(_trace(),),
        requested_stages=requested_stages,
        privacy_profile="trusted-monitored-local",
        model_profiles=(),
        budget=ResourceBudget(
            max_model_requests=0,
            max_model_tokens=0,
            max_processes=2,
            max_renderers=1,
            max_network_requests=2,
            max_storage_bytes=4096,
        ),
        concurrency=ConcurrencyLimit(
            model_requests=1,
            processes=1,
            renderers=1,
            network_requests=1,
            artifacts_per_item=1,
            items=1,
        ),
        approval_policy_ref=_ref(
            "user-approval-policy",
            "none",
        ),
        approval_mode=ApprovalMode.NONE,
        enabled_checkpoints=frozenset(),
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel=channel,  # type: ignore[arg-type]
            registry="registry://canary",
        ),
        idempotency_key="create-r6-parent-canary",
        audit=_audit(),
    )


def _control() -> PipelineControlConfigV2:
    return PipelineControlConfigV2(
        lease_duration_seconds=30,
        heartbeat_extension_seconds=20,
        max_attempts=2,
        retry_delay_seconds=(0,),
        retry_lease_expiry=True,
    )


def _control_plane() -> R6CanaryControlPlaneV2:
    return R6CanaryControlPlaneV2(
        resource_domain_ref=_ref(
            "resource-domain",
            "local",
            version="v1",
        ),
        resource_capacity=NonModelResourceVectorV2(
            processes=2,
            renderers=1,
            network_requests=2,
            storage_bytes=4096,
        ),
        worker_principal_ref=_ref(
            "worker-principal",
            "canary",
            version="v1",
        ),
        worker_version="r6-canary-worker/v1",
    )


def _seed() -> R6CanarySeedObjectV2:
    return R6CanarySeedObjectV2(
        object_ref=_ref("label-spec", "fixture"),
        codec=R6CanaryObjectCodecV2.LABEL_SPEC,
        payload_json="{}",
    )


def _fact_seed() -> R6CanarySeedObjectV2:
    return R6CanarySeedObjectV2(
        object_ref=_ref(
            "structured-fact-set",
            "fixture",
            version="r3-02",
        ),
        codec=R6CanaryObjectCodecV2.STRUCTURED_FACT_SET,
        payload_json="{}",
    )


def _binding(
    seed: R6CanarySeedObjectV2,
    fact_seed: R6CanarySeedObjectV2,
) -> R6CanaryTraceBindingV2:
    return R6CanaryTraceBindingV2(
        source_trace_ref=trace_source_v2_ref(_trace()),
        structured_fact_set_ref=fact_seed.object_ref,
        label_spec_refs=(seed.object_ref,),
        semantic_fixture_refs=(),
        task_fixture_refs=(),
        attachment_fixture_refs=(),
        expected_outcome=(R6CanaryExpectedItemOutcomeV2.SUCCEEDED),
    )


def _manifest(
    *,
    audit: ContractAudit | None = None,
) -> R6CanaryExecutionManifestV2:
    seed = _seed()
    fact_seed = _fact_seed()
    return R6CanaryExecutionManifestV2.create(
        job_spec=_spec(),
        control=_control(),
        control_plane=_control_plane(),
        seed_objects=(seed, fact_seed),
        trace_bindings=(_binding(seed, fact_seed),),
        audit=audit or _audit(),
    )


def test_manifest_is_strict_current_and_audit_time_independent() -> None:
    first = _manifest()
    second = _manifest(
        audit=_audit(
            created_by="another-operator",
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
        )
    )

    assert first.claim_scope == R6_CANARY_CLAIM_SCOPE
    assert first.job_spec.requested_stages == CANARY_STAGES
    assert first.manifest_id == second.manifest_id
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.audit != second.audit
    with pytest.raises(ValidationError, match="Extra inputs"):
        R6CanaryExecutionManifestV2.model_validate(
            {
                **first.model_dump(mode="python"),
                "private_reference": "forbidden",
            }
        )


def test_manifest_rejects_noncanary_stage_scope_and_missing_seed() -> None:
    value = _manifest()
    production = _spec(channel="PRODUCTION")

    with pytest.raises(ValidationError, match="stages"):
        R6CanaryExecutionManifestV2.create(
            job_spec=_spec(
                requested_stages=(
                    *CANARY_STAGES,
                    StageNameV2.BATCH_QUALITY,
                )
            ),
            control=_control(),
            control_plane=_control_plane(),
            seed_objects=value.seed_objects,
            trace_bindings=value.trace_bindings,
            audit=_audit(),
        )
    with pytest.raises(ValidationError, match="CANARY"):
        R6CanaryExecutionManifestV2.create(
            job_spec=production,
            control=_control(),
            control_plane=_control_plane(),
            seed_objects=value.seed_objects,
            trace_bindings=value.trace_bindings,
            audit=_audit(),
        )
    with pytest.raises(ValidationError, match="seed"):
        R6CanaryExecutionManifestV2.create(
            job_spec=_spec(),
            control=_control(),
            control_plane=_control_plane(),
            seed_objects=(_fact_seed(),),
            trace_bindings=value.trace_bindings,
            audit=_audit(),
        )
    with pytest.raises(
        ValidationError,
        match="required R5 governing versions",
    ):
        R6CanaryExecutionManifestV2.create(
            job_spec=_spec(),
            control=_control(),
            control_plane=_control_plane(),
            seed_objects=value.seed_objects,
            trace_bindings=value.trace_bindings,
            audit=ContractAudit(
                created_at=NOW,
                created_by="missing-stage-versions",
                governing_versions=(
                    VersionBinding(
                        component="eval-factory-spec",
                        version="approved-v2",
                    ),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("component", "version"),
    (
        ("batch-quality", "r7-03"),
        ("release-decision", "r7-08"),
        ("registry-publication", "r7-09"),
        ("production-readiness-attestation", "r8-08"),
        ("eval-factory-spec", "approved-r8"),
    ),
)
def test_manifest_rejects_post_r6_governing_markers(
    component: str,
    version: str,
) -> None:
    base = _audit()
    audit = base.model_copy(
        update={
            "governing_versions": (
                *base.governing_versions,
                VersionBinding(
                    component=component,
                    version=version,
                ),
            )
        }
    )

    with pytest.raises(
        ValidationError,
        match="post-R6 governing version",
    ):
        _manifest(audit=audit)


def test_manifest_bounds_seed_payloads_and_collection_sizes() -> None:
    with pytest.raises(ValidationError):
        R6CanarySeedObjectV2(
            object_ref=_ref("label-spec", "oversized"),
            codec=R6CanaryObjectCodecV2.LABEL_SPEC,
            payload_json=("{" + '"value":"' + ("x" * 1_048_576) + '"}'),
        )
    with pytest.raises(
        ValidationError,
        match="UTF-8 byte limit",
    ):
        R6CanarySeedObjectV2(
            object_ref=_ref("label-spec", "oversized-utf8"),
            codec=R6CanaryObjectCodecV2.LABEL_SPEC,
            payload_json=json.dumps(
                {"value": "\U0001f642" * 300_000},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    manifest = _manifest()
    with pytest.raises(ValidationError, match="at most 256"):
        R6CanaryExecutionManifestV2.model_validate(
            {
                **manifest.model_dump(mode="python"),
                "seed_objects": manifest.seed_objects * 129,
            }
        )
    with pytest.raises(ValidationError, match="at most 128"):
        R6CanaryExecutionManifestV2.model_validate(
            {
                **manifest.model_dump(mode="python"),
                "trace_bindings": manifest.trace_bindings * 129,
            }
        )


def test_cli_import_does_not_load_canary_physical_runtime() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import eval_factory.cli; "
                "assert 'eval_factory.orchestration.canary_driver' not in sys.modules; "
                "assert 'eval_factory.orchestration.canary_profile' not in sys.modules; "
                "assert 'env_mock_agent.facade.canary_v2' not in sys.modules; "
                "assert 'anthropic' not in sys.modules; "
                "assert not any(name.startswith('env_mock_agent.providers') "
                "for name in sys.modules); "
                "assert not any(name.startswith('env_mock_agent.runtimes') "
                "for name in sys.modules)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def _review_fanout() -> SemanticReviewFanoutV2:
    spec = _spec()
    plan = DatasetJobPlanCompiler().compile(
        job_spec=spec,
        audit=_audit(),
    )
    graph = DatasetJobWorkGraphCompiler().compile(
        job_spec=spec,
        resolved_plan=plan,
        audit=_audit(),
    )
    parent = next(unit for unit in graph.work_units if unit.stage is StageNameV2.ITEM_QUALITY)
    source_ref = parent.source_trace_ref
    assert source_ref is not None
    coverage = ResolvedWorkUnitV2.create(
        scope=WorkUnitScopeV2.ITEM,
        stage=StageNameV2.ITEM_QUALITY,
        job_id=parent.job_id,
        item_id=parent.item_id,
        source_trace_ref=source_ref,
        artifact_execution_group_ref=None,
        depends_on_work_unit_refs=(parent.depends_on_work_unit_refs),
        join_mode=WorkDependencyJoinModeV2.ALL_SUCCEEDED,
        semantic_review_round=(SemanticReviewRoundV2.COVERAGE_SOLVABILITY),
    )
    realism = ResolvedWorkUnitV2.create(
        scope=WorkUnitScopeV2.ITEM,
        stage=StageNameV2.ITEM_QUALITY,
        job_id=parent.job_id,
        item_id=parent.item_id,
        source_trace_ref=source_ref,
        artifact_execution_group_ref=None,
        depends_on_work_unit_refs=(_ref_for_unit(coverage),),
        join_mode=WorkDependencyJoinModeV2.ALL_SUCCEEDED,
        semantic_review_round=(SemanticReviewRoundV2.REALISM_CONSISTENCY),
    )
    leakage = ResolvedWorkUnitV2.create(
        scope=WorkUnitScopeV2.ITEM,
        stage=StageNameV2.ITEM_QUALITY,
        job_id=parent.job_id,
        item_id=parent.item_id,
        source_trace_ref=source_ref,
        artifact_execution_group_ref=None,
        depends_on_work_unit_refs=(_ref_for_unit(realism),),
        join_mode=WorkDependencyJoinModeV2.ALL_SUCCEEDED,
        semantic_review_round=(SemanticReviewRoundV2.LEAKAGE_EXECUTABILITY),
    )
    return SemanticReviewFanoutV2.create(
        resolved_job_work_graph_ref=_ref(
            "resolved-job-work-graph",
            graph.resolved_job_work_graph_id,
        ).model_copy(
            update={
                "object_id": graph.resolved_job_work_graph_id,
                "object_sha256": (graph.resolved_job_work_graph_sha256),
            }
        ),
        parent_item_quality_work_unit_ref=_ref_for_unit(parent),
        round_work=(
            SemanticReviewRoundWorkV2(
                round=SemanticReviewRoundV2.COVERAGE_SOLVABILITY,
                work_unit=coverage,
            ),
            SemanticReviewRoundWorkV2(
                round=SemanticReviewRoundV2.REALISM_CONSISTENCY,
                work_unit=realism,
            ),
            SemanticReviewRoundWorkV2(
                round=SemanticReviewRoundV2.LEAKAGE_EXECUTABILITY,
                work_unit=leakage,
            ),
        ),
        audit=_audit(),
    )


def _ref_for_unit(unit: ResolvedWorkUnitV2) -> ObjectRef:
    return ObjectRef(
        object_type="resolved-work-unit",
        object_id=unit.resolved_work_unit_id,
        object_version="v2",
        object_sha256=unit.resolved_work_unit_sha256,
    )


def test_semantic_review_fanout_binds_exact_order_and_parent() -> None:
    value = _review_fanout()

    assert tuple(item.round for item in value.round_work) == tuple(SemanticReviewRoundV2)
    assert value.round_work[1].work_unit.depends_on_work_unit_refs == (
        _ref_for_unit(value.round_work[0].work_unit),
    )
    with pytest.raises(ValidationError, match="round"):
        SemanticReviewFanoutV2.model_validate(
            {
                **value.model_dump(mode="python"),
                "round_work": tuple(reversed(value.round_work)),
            }
        )


def test_terminal_result_partitions_items_and_preserves_claim_scope() -> None:
    report_ref = _ref("batch-audit-report", "report")
    quality_ref = _ref(
        "item-quality-compilation-result",
        "quality",
    )
    package_ref = _ref(
        "final-package-manifest",
        "package",
    )
    result = R6CanaryDatasetResultV2(
        manifest_ref=_ref(
            "r6-canary-execution-manifest",
            "manifest",
        ),
        job_id="job://r6-parent/canary",
        job_status=JobStatus.SUCCEEDED,
        item_ids=(
            "item://r6-parent/blocked",
            "item://r6-parent/succeeded",
        ),
        succeeded_item_ids=("item://r6-parent/succeeded",),
        blocked_item_ids=("item://r6-parent/blocked",),
        failed_item_ids=(),
        quality_result_refs=(quality_ref,),
        package_manifest_refs=(package_ref,),
        audit_report_ref=report_ref,
        audit_outcome=BatchAuditOutcomeV2.COMPLETE,
    )

    assert result.claim_scope == R6_CANARY_CLAIM_SCOPE
    with pytest.raises(ValidationError, match="partition"):
        R6CanaryDatasetResultV2.model_validate(
            {
                **result.model_dump(mode="python"),
                "failed_item_ids": ("item://r6-parent/succeeded",),
            }
        )
    with pytest.raises(ValidationError, match="quality"):
        R6CanaryDatasetResultV2.model_validate(
            {
                **result.model_dump(mode="python"),
                "quality_result_refs": (),
            }
        )


def test_seed_codec_rejects_wrong_ref_type_and_sensitive_extra() -> None:
    with pytest.raises(ValidationError, match="label-spec"):
        R6CanarySeedObjectV2(
            object_ref=_ref("raw-trace", "forbidden"),
            codec=R6CanaryObjectCodecV2.LABEL_SPEC,
            payload_json="{}",
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        R6CanaryControlPlaneV2.model_validate(
            {
                **_control_plane().model_dump(mode="python"),
                "credential": "forbidden",
            }
        )
