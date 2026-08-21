from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import test_task_rewrite as task_fixture
from test_semantic_review import _policy, _sources
from test_structured_labeling import _label_spec, _predicate
from typer.testing import CliRunner

from env_mock_agent.providers import TextProvider
from eval_factory.attachment_planning import (
    ArtifactExecutionPlanningOutcome,
    ArtifactExecutionPlanningResult,
)
from eval_factory.cli import app
from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.canary_execution_v2 import (
    R6CanaryControlPlaneV2,
    R6CanaryExecutionManifestV2,
    R6CanaryExpectedItemOutcomeV2,
    R6CanaryObjectCodecV2,
    R6CanarySeedObjectV2,
    R6CanaryTraceBindingV2,
)
from eval_factory.contracts.cli_v2 import PipelineControlConfigV2
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.labeling_v2 import PredicateOperatorV2
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ItemStatus,
    JobStatus,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    StageNameV2,
    WorkLeaseEventKindV2,
)
from eval_factory.contracts.resource_v2 import (
    NonModelResourceVectorV2,
    ResourceAdmissionOutcomeV2,
    ResourceUsageV2,
)
from eval_factory.labeling.structured import StructuredFactSet
from eval_factory.orchestration.canary_driver import (
    CanaryDriverFaultInjector,
    CanaryDriverFaultPoint,
    CanaryDriverInjectedCrash,
    CanaryPipelineDriver,
    CanaryResourceAdmissionError,
    StaticCanaryDriverFaultInjector,
    create_canary_stage_store,
)
from eval_factory.orchestration.canary_profile import (
    CanaryProfileError,
    R6CanaryArtifactGroupOutput,
    R6CanaryAttachmentPreparationOutput,
    R6CanaryReviewFixture,
    R6CanaryTaskFixture,
    _artifact_group_outputs,
    _attachment_preparation,
    canary_profile_codec_registry,
)
from eval_factory.orchestration.job_store import (
    JobStore,
    StaleWorkLeaseError,
)
from eval_factory.orchestration.lifecycle import (
    PipelineLifecyclePolicyError,
)
from eval_factory.orchestration.runner import TraceIndexStageService
from eval_factory.orchestration.stage_object_store import (
    StageObjectInjectedCrash,
    StageObjectStore,
    StageObjectStoreFaultInjector,
    StageObjectStoreFaultPoint,
)
from eval_factory.trace.adapters import RawTrajV1Adapter
from eval_factory.trace.source_registry import TraceSourceRegistry
from eval_factory.trace.storage import TraceIndexStore

NOW = datetime(2026, 7, 31, tzinfo=UTC)
HASH = "a" * 64


class _ArtifactGroupFaultInjector:
    def __init__(
        self,
        store: JobStore,
        point: CanaryDriverFaultPoint,
    ) -> None:
        self._store = store
        self._point = point
        self.fired = False

    def maybe_raise(
        self,
        point: CanaryDriverFaultPoint,
        *,
        work_unit_id: str,
        attempt: int,
    ) -> None:
        del attempt
        if self.fired or point is not self._point:
            return
        units = self._store.list_work_units(job_id="job://r6-canary/driver")
        target = next(
            (unit for unit in units if unit.resolved_work_unit_id == work_unit_id),
            None,
        )
        if target is not None and target.scope.value == "ARTIFACT_GROUP":
            self.fired = True
            raise CanaryDriverInjectedCrash(f"injected artifact group crash at {point.value}")


class _ArtifactGroupOutputStoreFaultInjector:
    def __init__(self) -> None:
        self.fired = False

    def maybe_raise(
        self,
        point: StageObjectStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        if (
            not self.fired
            and point is StageObjectStoreFaultPoint.AFTER_CAS_WRITE
            and object_ref.object_type == "r6-canary-artifact-group-output"
        ):
            self.fired = True
            raise StageObjectInjectedCrash("injected group output persistence crash")


class _ProviderOperationStoreFaultInjector:
    def __init__(self) -> None:
        self.fired = False

    def maybe_raise(
        self,
        point: StageObjectStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        if (
            not self.fired
            and point is StageObjectStoreFaultPoint.AFTER_CAS_WRITE
            and object_ref.object_type == "r6-canary-provider-operation-output"
        ):
            self.fired = True
            raise StageObjectInjectedCrash("injected provider journal persistence crash")


class _TargetWorkUnitFaultInjector:
    def __init__(
        self,
        store: JobStore,
        *,
        stage: StageNameV2,
        source_trace_id: str | None = None,
        finalizer_only: bool = False,
    ) -> None:
        self._store = store
        self._stage = stage
        self._source_trace_id = source_trace_id
        self._finalizer_only = finalizer_only
        self.fired = False
        self.target_item_id: str | None = None

    def maybe_raise(
        self,
        point: CanaryDriverFaultPoint,
        *,
        work_unit_id: str,
        attempt: int,
    ) -> None:
        del attempt
        if self.fired or point is not CanaryDriverFaultPoint.AFTER_LEASE_COMPLETION:
            return
        target = next(
            (
                unit
                for unit in self._store.list_work_units(
                    job_id="job://r6-canary/driver",
                )
                if unit.resolved_work_unit_id == work_unit_id
            ),
            None,
        )
        if target is None or target.stage is not self._stage:
            return
        if self._finalizer_only and target.semantic_review_round is not None:
            return
        if self._source_trace_id is not None and (
            target.source_trace_ref is None or target.source_trace_ref.object_id != self._source_trace_id
        ):
            return
        self.fired = True
        self.target_item_id = target.item_id
        raise CanaryDriverInjectedCrash(f"injected targeted crash after {target.stage.value} completion")


def _source_chain_no_attachment() -> dict[str, object]:
    source = task_fixture._pending_draft()
    values = source.model_dump(mode="python")
    values.update(
        {
            "task_draft_id": "task-draft://pending",
            "visible_prompt": ("Explain the design constraints stated in this prompt."),
            "task_intent": "Evaluate a self-contained explanation.",
            "evaluation_claim": ("The contestant must explain the stated constraints."),
            "required_capabilities": ("reasoning",),
            "allowed_tools": (),
            "attachment_dependencies": (),
            "task_draft_sha256": "0" * 64,
        }
    )
    pending = task_fixture.TaskDraftV2.model_validate(values)
    digest = task_fixture.task_draft_carried_sha256(pending)
    pending = pending.model_copy(
        update={
            "task_draft_id": f"task-draft://sha256/{digest}",
            "task_draft_sha256": digest,
        }
    )
    task_draft, gate, reference_set = task_fixture._pass_prompt_safety(pending)
    base_criterion = task_fixture._criterion()
    reachability_values = base_criterion.reachability.model_dump(mode="python")
    reachability_values.update(
        {
            "reachability_id": "rubric-reachability://pending",
            "attachment_dependency_ids": (),
            "allowed_tool_ids": (),
            "reachability_sha256": "0" * 64,
        }
    )
    reachability = task_fixture.RubricReachabilityV2.model_validate(reachability_values)
    reachability_digest = task_fixture.rubric_reachability_carried_sha256(reachability)
    reachability = reachability.model_copy(
        update={
            "reachability_id": (f"rubric-reachability://sha256/{reachability_digest}"),
            "reachability_sha256": reachability_digest,
        }
    )
    criterion_values = base_criterion.model_dump(mode="python")
    criterion_values.update(
        {
            "criterion_id": "rubric-criterion://pending",
            "judged_object": task_fixture.RubricJudgedObjectV2(
                judged_object_id=("judged-object://r6-canary/response"),
                kind=(task_fixture.RubricJudgedObjectKindV2.CONTESTANT_RESPONSE),
                description="The contestant response.",
            ),
            "reachability": reachability,
            "criterion_sha256": "0" * 64,
        }
    )
    criterion = task_fixture.RubricCriterionV2.model_validate(criterion_values)
    criterion_digest = task_fixture.rubric_criterion_carried_sha256(criterion)
    criterion = criterion.model_copy(
        update={
            "criterion_id": (f"rubric-criterion://sha256/{criterion_digest}"),
            "criterion_sha256": criterion_digest,
        }
    )
    rubric_values = task_fixture._rubric_set(task_draft).model_dump(mode="python")
    rubric_values.update(
        {
            "rubric_set_id": "rubric-set://pending",
            "criteria": (criterion,),
            "total_weight": criterion.weight,
            "rubric_set_sha256": "0" * 64,
        }
    )
    rubric_set = task_fixture.RubricSetV2.model_validate(rubric_values)
    rubric_digest = task_fixture.rubric_set_carried_sha256(rubric_set)
    rubric_set = rubric_set.model_copy(
        update={
            "rubric_set_id": (f"rubric-set://sha256/{rubric_digest}"),
            "rubric_set_sha256": rubric_digest,
        }
    )
    evaluator_spec, reference_policy = task_fixture._evaluation_contract(rubric_set)
    catalog = task_fixture._tool_catalog()
    tool_policy, contestant_policy = task_fixture._tool_contract(
        task_draft,
        rubric_set,
        evaluator_spec,
        catalog,
    )
    producer_view_result, producer_bundle = task_fixture._producer_evidence()
    producer = task_fixture.ProducerTaskViewCompiler().compile(
        task_draft=task_draft,
        rubric_set=rubric_set,
        evaluator_spec=evaluator_spec,
        reference_policy=reference_policy,
        tool_policy=tool_policy,
        contestant_tool_policy=contestant_policy,
        producer_view_result=producer_view_result,
        producer_evidence_bundle=producer_bundle,
        audit=task_fixture._audit(),
    )
    assert producer.producer_task_view is not None
    assert producer.storage_authorization is not None
    return {
        "task_draft": task_draft,
        "gate": gate,
        "reference_set": reference_set,
        "rubric_set": rubric_set,
        "evaluator_spec": evaluator_spec,
        "reference_policy": reference_policy,
        "tool_policy": tool_policy,
        "contestant_policy": contestant_policy,
        "storage_authorization": producer.storage_authorization,
        "producer_task_view": producer.producer_task_view,
    }


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r6-canary-driver-test",
        governing_versions=(
            VersionBinding(
                component="r6-canary-driver",
                version="r6-parent-v1",
            ),
            VersionBinding(
                component="artifact-results",
                version="r5-07",
            ),
            VersionBinding(
                component="artifact-execution",
                version="r5-06",
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


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-canary/{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _raw_trace(
    path: Path,
    *,
    suffix: str = "driver",
) -> TraceSourceRef:
    request = {
        "messages": [
            {
                "role": "user",
                "content": "Inspect the provided input workspace.",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "read-1",
                        "name": "Read",
                        "input": {"file_path": "inputs/source.txt"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "read-1",
                        "content": "safe input",
                    }
                ],
            },
        ]
    }
    outer = {
        "account": "synthetic",
        "api_type": "chat",
        "business": "R6Canary",
        "endpoint": "offline",
        "event_time": "2026-08-01T00:00:00Z",
        "extra": '{"req_lost_number":0,"resp_lost_number":0}',
        "mm_urls": "",
        "model": "offline-model",
        "p_date": "2026-08-01",
        "request": json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "response": '{"content":[],"role":"assistant"}',
        "sid": f"r6-canary-{suffix}",
        "source": None,
    }
    path.write_text(
        json.dumps(outer, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    probe = RawTrajV1Adapter().probe(path)
    assert probe.raw_sha256 is not None
    return TraceSourceRef(
        source_trace_id=f"source-trace://r6-canary/{suffix}",
        source_uri=path.resolve().as_uri(),
        raw_sha256=probe.raw_sha256,
        adapter_name="raw_traj_v1",
        adapter_version="1.0.0",
        processing_class="RESTRICTED_TRACE_RAW",
    )


def _seed(
    codec: R6CanaryObjectCodecV2,
    value,
) -> R6CanarySeedObjectV2:
    registry = canary_profile_codec_registry()
    return R6CanarySeedObjectV2(
        object_ref=registry.reference(codec, value),
        codec=codec,
        payload_json=value.canonical_json().decode(),
    )


def _manifest(
    tmp_path: Path,
    *,
    with_attachment: bool = True,
) -> R6CanaryExecutionManifestV2:
    trace = _raw_trace(tmp_path / "raw.jsonl")
    audit = _audit()
    prep = TraceIndexStageService(
        source_registry=TraceSourceRegistry(tmp_path / "prep-source.sqlite3"),
        trace_store=TraceIndexStore(tmp_path / "prep-trace"),
    ).execute(
        trace=trace,
        audit=audit,
        job_id="job://r6-canary/prep",
        attempt=1,
    )
    fact_set = StructuredFactSet.from_stored_index(
        prep.stored_index,
        audit=audit,
    )
    fact_ref = canary_profile_codec_registry().reference(
        R6CanaryObjectCodecV2.STRUCTURED_FACT_SET,
        fact_set,
    )
    label_spec = _label_spec(
        positive=(
            _predicate(
                "predicate://r6-canary/file-read",
                field_path="tool_family",
                operator=PredicateOperatorV2.EQUALS,
                expected_value="file_read",
            ),
        )
    )
    chain = task_fixture._source_chain() if with_attachment else _source_chain_no_attachment()
    provider_fields = (
        {
            "producer_view_result": chain["producer_view_result"],
            "producer_evidence_bundle": chain["producer_bundle"],
            "text_provider_content": "safe input-state content",
            "attachment_relative_path": "inputs/source.txt",
        }
        if with_attachment
        else {}
    )
    task_canary_fixture = R6CanaryTaskFixture(
        fixture_id="r6-canary-task-fixture://driver",
        task_draft=chain["task_draft"],
        task_prompt_safety_gate=chain["gate"],
        rubric_set=chain["rubric_set"],
        evaluator_spec=chain["evaluator_spec"],
        reference_policy=chain["reference_policy"],
        tool_policy=chain["tool_policy"],
        contestant_tool_policy=chain["contestant_policy"],
        producer_storage_authorization=(chain["storage_authorization"]),
        producer_task_view=chain["producer_task_view"],
        leakage_reference_set=chain["reference_set"],
        **provider_fields,
    )
    review_fixture = R6CanaryReviewFixture(
        fixture_id="r6-canary-review-fixture://driver",
        review_policy=_policy(),
        context_sources=_sources(),
    )
    label_seed = _seed(
        R6CanaryObjectCodecV2.LABEL_SPEC,
        label_spec,
    )
    task_seed = _seed(
        R6CanaryObjectCodecV2.CANARY_ATTACHMENT_FIXTURE,
        task_canary_fixture,
    )
    review_seed = _seed(
        R6CanaryObjectCodecV2.CANARY_REVIEW_FIXTURE,
        review_fixture,
    )
    spec = DatasetJobSpecV2(
        job_id="job://r6-canary/driver",
        traces=(trace,),
        requested_stages=(
            StageNameV2.TRACE_INDEX,
            StageNameV2.SAFETY,
            StageNameV2.LABEL,
            StageNameV2.TASK_AUTHORING,
            StageNameV2.ATTACHMENT,
            StageNameV2.ITEM_QUALITY,
        ),
        privacy_profile="trusted-monitored-local",
        model_profiles=(),
        budget=ResourceBudget(
            max_model_requests=0,
            max_model_tokens=0,
            max_processes=2,
            max_renderers=0,
            max_network_requests=0,
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
        approval_policy_ref=_ref(
            "user-approval-policy",
            "none",
            version="v2",
        ),
        approval_mode=ApprovalMode.NONE,
        enabled_checkpoints=frozenset(),
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://canary",
        ),
        idempotency_key="create-r6-canary-driver",
        audit=audit,
    )
    binding = R6CanaryTraceBindingV2(
        source_trace_ref=ObjectRef(
            object_type="trace-source",
            object_id=trace.source_trace_id,
            object_version=trace.adapter_version,
            object_sha256=trace.raw_sha256,
        ),
        structured_fact_set_ref=fact_ref,
        label_spec_refs=(label_seed.object_ref,),
        semantic_fixture_refs=(),
        task_fixture_refs=(task_seed.object_ref,),
        attachment_fixture_refs=(review_seed.object_ref,),
        expected_outcome=R6CanaryExpectedItemOutcomeV2.SUCCEEDED,
    )
    return R6CanaryExecutionManifestV2.create(
        job_spec=spec,
        control=PipelineControlConfigV2(
            lease_duration_seconds=60,
            heartbeat_extension_seconds=30,
            max_attempts=1,
            retry_delay_seconds=(),
            retry_lease_expiry=False,
        ),
        control_plane=R6CanaryControlPlaneV2(
            resource_domain_ref=_ref(
                "resource-domain",
                "local",
            ),
            resource_capacity=NonModelResourceVectorV2(
                processes=2,
                renderers=0,
                network_requests=0,
                storage_bytes=1024 * 1024,
            ),
            worker_principal_ref=_ref(
                "worker-principal",
                "driver",
            ),
            worker_version="r6-canary-driver/v1",
        ),
        seed_objects=(label_seed, task_seed, review_seed),
        trace_bindings=(binding,),
        audit=audit,
    )


def _with_storage_limit(
    manifest: R6CanaryExecutionManifestV2,
    *,
    storage_bytes: int,
    job_id: str | None = None,
) -> R6CanaryExecutionManifestV2:
    resolved_job_id = job_id or manifest.job_spec.job_id
    spec_values = manifest.job_spec.model_dump(mode="python")
    spec_values.update(
        {
            "job_id": resolved_job_id,
            "idempotency_key": (f"create-{resolved_job_id}"),
            "budget": manifest.job_spec.budget.model_copy(update={"max_storage_bytes": storage_bytes}),
        }
    )
    job_spec = DatasetJobSpecV2.model_validate(spec_values)
    control_plane = manifest.control_plane.model_copy(
        update={
            "resource_capacity": (
                manifest.control_plane.resource_capacity.model_copy(update={"storage_bytes": storage_bytes})
            )
        }
    )
    return R6CanaryExecutionManifestV2.create(
        job_spec=job_spec,
        control=manifest.control,
        control_plane=control_plane,
        seed_objects=manifest.seed_objects,
        trace_bindings=manifest.trace_bindings,
        audit=manifest.audit,
    )


def _driver(
    tmp_path: Path,
    *,
    fault: CanaryDriverFaultPoint | None = None,
    stage_fault_injector: (StageObjectStoreFaultInjector | None) = None,
) -> CanaryPipelineDriver:
    return CanaryPipelineDriver(
        job_store=JobStore(
            tmp_path / "job.sqlite3",
            clock=lambda: NOW,
        ),
        source_registry=TraceSourceRegistry(tmp_path / "source.sqlite3"),
        trace_store=TraceIndexStore(tmp_path / "trace-store"),
        stage_store=StageObjectStore(
            tmp_path / "stage-store",
            registry=canary_profile_codec_registry(),
            fault_injector=stage_fault_injector,
        ),
        fault_injector=(
            StaticCanaryDriverFaultInjector(crash_points=frozenset({fault})) if fault is not None else None
        ),
    )


def _manifest_with_rejected_item(
    tmp_path: Path,
) -> R6CanaryExecutionManifestV2:
    base = _manifest(tmp_path)
    trace = _raw_trace(
        tmp_path / "raw-rejected.jsonl",
        suffix="rejected",
    )
    prep = TraceIndexStageService(
        source_registry=TraceSourceRegistry(tmp_path / "prep-rejected-source.sqlite3"),
        trace_store=TraceIndexStore(tmp_path / "prep-rejected-trace"),
    ).execute(
        trace=trace,
        audit=base.audit,
        job_id="job://r6-canary/prep-rejected",
        attempt=1,
    )
    fact_set = StructuredFactSet.from_stored_index(
        prep.stored_index,
        audit=base.audit,
    )
    fact_ref = canary_profile_codec_registry().reference(
        R6CanaryObjectCodecV2.STRUCTURED_FACT_SET,
        fact_set,
    )
    rejected_label = _label_spec(
        positive=(
            _predicate(
                "predicate://r6-canary/search",
                field_path="tool_family",
                operator=PredicateOperatorV2.EQUALS,
                expected_value="search",
            ),
        )
    )
    rejected_label = rejected_label.model_copy(
        update={
            "label_spec_id": "label-spec://r6-canary/rejected",
            "label_spec_sha256": "b" * 64,
        }
    )
    label_seed = _seed(
        R6CanaryObjectCodecV2.LABEL_SPEC,
        rejected_label,
    )
    shared_task = next(
        seed.object_ref
        for seed in base.seed_objects
        if seed.codec is R6CanaryObjectCodecV2.CANARY_ATTACHMENT_FIXTURE
    )
    shared_review = next(
        seed.object_ref
        for seed in base.seed_objects
        if seed.codec is R6CanaryObjectCodecV2.CANARY_REVIEW_FIXTURE
    )
    binding = R6CanaryTraceBindingV2(
        source_trace_ref=ObjectRef(
            object_type="trace-source",
            object_id=trace.source_trace_id,
            object_version=trace.adapter_version,
            object_sha256=trace.raw_sha256,
        ),
        structured_fact_set_ref=fact_ref,
        label_spec_refs=(label_seed.object_ref,),
        semantic_fixture_refs=(),
        task_fixture_refs=(shared_task,),
        attachment_fixture_refs=(shared_review,),
        expected_outcome=R6CanaryExpectedItemOutcomeV2.BLOCKED,
    )
    spec = DatasetJobSpecV2.model_validate(
        {
            **base.job_spec.model_dump(mode="python"),
            "traces": (*base.job_spec.traces, trace),
        }
    )
    return R6CanaryExecutionManifestV2.create(
        job_spec=spec,
        control=base.control,
        control_plane=base.control_plane,
        seed_objects=(*base.seed_objects, label_seed),
        trace_bindings=(*base.trace_bindings, binding),
        audit=base.audit,
    )


@pytest.mark.asyncio
async def test_canary_driver_runs_r1_to_item_quality(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    result = await _driver(tmp_path).run(manifest)

    assert result.job_status is JobStatus.SUCCEEDED
    assert len(result.succeeded_item_ids) == 1
    assert len(result.quality_result_refs) == 1
    assert len(result.package_manifest_refs) == 1
    assert result.blocked_item_ids == ()
    assert result.failed_item_ids == ()
    store = JobStore(
        tmp_path / "job.sqlite3",
        clock=lambda: NOW,
    )
    units = store.list_work_units(job_id=result.job_id)
    assert len(units) == 10
    assert len(store.list_work_leases(job_id=result.job_id)) == 10
    assert len(store.list_work_attempt_metrics(job_id=result.job_id)) == 10
    assert len(store.list_stage_runs(job_id=result.job_id)) == 9
    group = next(unit for unit in units if unit.scope.value == "ARTIFACT_GROUP")
    group_metrics = next(
        value
        for value in store.list_work_attempt_metrics(job_id=result.job_id)
        if value.work_unit_ref.object_id == group.resolved_work_unit_id
    )
    assert group_metrics.stage_run_ref is None
    assert group_metrics.resource_usage is not None
    assert group_metrics.resource_usage.observed.storage_bytes == len(b"safe input-state content\n")
    assert {ref.object_type for ref in group_metrics.result_refs} == {
        "artifact-execution-batch",
        "artifact-execution-receipt",
    }
    group_outputs = _artifact_group_outputs(
        create_canary_stage_store(tmp_path / "stage-store"),
        job_id=result.job_id,
        item_id=result.succeeded_item_ids[0],
    )
    invalid_group = group_outputs[0].model_dump(mode="python")
    invalid_group["work_unit_ref"] = _ref(
        "not-a-work-unit",
        "invalid-group",
        version="v2",
    )
    with pytest.raises(
        ValueError,
        match="one exact work unit receipt",
    ):
        R6CanaryArtifactGroupOutput.model_validate(invalid_group)
    assert all(
        store.get_stage_result_for_run(run.stage_run_id) is not None
        for run in store.list_stage_runs(job_id=result.job_id)
    )
    replay = await _driver(tmp_path).run(manifest)
    assert replay == result


@pytest.mark.asyncio
async def test_canary_driver_preserves_no_attachment_base_case(
    tmp_path: Path,
) -> None:
    result = await _driver(tmp_path).run(_manifest(tmp_path, with_attachment=False))

    assert result.job_status is JobStatus.SUCCEEDED
    store = JobStore(
        tmp_path / "job.sqlite3",
        clock=lambda: NOW,
    )
    units = store.list_work_units(job_id=result.job_id)
    assert len(units) == 9
    assert all(unit.scope.value != "ARTIFACT_GROUP" for unit in units)
    assert len(store.list_work_leases(job_id=result.job_id)) == 9
    assert len(store.list_stage_runs(job_id=result.job_id)) == 9
    assert len(store.list_work_attempt_metrics(job_id=result.job_id)) == 9


def test_canary_profile_rejects_incomplete_private_material(
    tmp_path: Path,
) -> None:
    attachment_root = tmp_path / "attachment"
    attachment_root.mkdir()
    attachment_manifest = _manifest(attachment_root)
    attachment_seed = next(
        seed
        for seed in attachment_manifest.seed_objects
        if seed.codec is R6CanaryObjectCodecV2.CANARY_ATTACHMENT_FIXTURE
    )
    incomplete = json.loads(attachment_seed.payload_json)
    incomplete["text_provider_content"] = None
    with pytest.raises(
        ValueError,
        match="complete attachment fixture",
    ):
        R6CanaryTaskFixture.model_validate_json(json.dumps(incomplete))

    no_attachment_root = tmp_path / "no-attachment"
    no_attachment_root.mkdir()
    no_attachment_manifest = _manifest(
        no_attachment_root,
        with_attachment=False,
    )
    no_attachment_seed = next(
        seed
        for seed in no_attachment_manifest.seed_objects
        if seed.codec is R6CanaryObjectCodecV2.CANARY_ATTACHMENT_FIXTURE
    )
    cross_mode = json.loads(no_attachment_seed.payload_json)
    cross_mode["text_provider_content"] = "unexpected"
    with pytest.raises(
        ValueError,
        match="cannot carry provider material",
    ):
        R6CanaryTaskFixture.model_validate_json(json.dumps(cross_mode))

    with pytest.raises(
        ValueError,
        match="exactly one execution group",
    ):
        R6CanaryAttachmentPreparationOutput(
            job_id="job://r6-canary/invalid-preparation",
            item_id="item://r6-canary/invalid-preparation",
            execution_result=ArtifactExecutionPlanningResult(
                outcome=(ArtifactExecutionPlanningOutcome.NOT_REQUIRED),
                world_ledger_snapshot=None,
                execution_plan=None,
                audit=_audit(),
            ),
        )
    with pytest.raises(
        CanaryProfileError,
        match="durable attachment preparation",
    ):
        _attachment_preparation(
            create_canary_stage_store(tmp_path / "empty-stage-store"),
            job_id="job://r6-canary/missing",
            item_id="item://r6-canary/missing",
        )


@pytest.mark.asyncio
async def test_canary_driver_surfaces_unsatisfiable_resource_demand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original_generate = TextProvider.generate

    def counted_generate(self, request):
        nonlocal calls
        calls += 1
        return original_generate(self, request)

    monkeypatch.setattr(
        TextProvider,
        "generate",
        counted_generate,
    )
    manifest = _with_storage_limit(
        _manifest(tmp_path),
        storage_bytes=1,
    )

    with pytest.raises(
        CanaryResourceAdmissionError,
    ) as error:
        await _driver(tmp_path).run(manifest)

    assert error.value.outcome is ResourceAdmissionOutcomeV2.UNSATISFIABLE_DEMAND
    assert calls == 0


@pytest.mark.asyncio
async def test_canary_driver_surfaces_shared_pool_capacity_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original_generate = TextProvider.generate

    def counted_generate(self, request):
        nonlocal calls
        calls += 1
        return original_generate(self, request)

    monkeypatch.setattr(
        TextProvider,
        "generate",
        counted_generate,
    )
    base = _manifest(tmp_path)
    holder_manifest = _with_storage_limit(
        base,
        storage_bytes=len(b"safe input-state content\n"),
    )
    holder = _driver(tmp_path)
    injector = _ArtifactGroupFaultInjector(
        holder.job_store,
        CanaryDriverFaultPoint.BEFORE_HANDLER,
    )
    holder.fault_injector = injector
    with pytest.raises(CanaryDriverInjectedCrash):
        await holder.run(holder_manifest)
    assert injector.fired

    waiting_manifest = _with_storage_limit(
        base,
        storage_bytes=len(b"safe input-state content\n"),
        job_id="job://r6-canary/waiting",
    )
    with pytest.raises(
        CanaryResourceAdmissionError,
    ) as error:
        await _driver(tmp_path).run(waiting_manifest)

    assert error.value.outcome is ResourceAdmissionOutcomeV2.WAITING_CAPACITY
    assert calls == 0


@pytest.mark.asyncio
async def test_rejected_label_does_not_block_unrelated_item(
    tmp_path: Path,
) -> None:
    result = await _driver(tmp_path).run(_manifest_with_rejected_item(tmp_path))

    assert result.job_status is JobStatus.SUCCEEDED
    assert len(result.succeeded_item_ids) == 1
    assert len(result.blocked_item_ids) == 1
    assert result.failed_item_ids == ()
    assert len(result.quality_result_refs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", tuple(CanaryDriverFaultPoint))
async def test_canary_driver_resumes_after_output_persistence(
    tmp_path: Path,
    fault: CanaryDriverFaultPoint,
) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(CanaryDriverInjectedCrash):
        await _driver(
            tmp_path,
            fault=fault,
        ).run(manifest)

    result = await _driver(tmp_path).run(manifest)
    assert result.job_status is JobStatus.SUCCEEDED
    assert len(result.quality_result_refs) == 1


@pytest.mark.asyncio
async def test_canary_driver_reconciles_finalizer_after_terminal_lease_crash(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    driver = _driver(tmp_path)
    injector = _TargetWorkUnitFaultInjector(
        driver.job_store,
        stage=StageNameV2.ITEM_QUALITY,
        finalizer_only=True,
    )
    driver.fault_injector = injector

    with pytest.raises(CanaryDriverInjectedCrash):
        await driver.run(manifest)

    assert injector.fired
    assert injector.target_item_id is not None
    assert driver.job_store.get_item(injector.target_item_id).status is ItemStatus.RUNNING

    result = await _driver(tmp_path).run(manifest)
    assert injector.target_item_id in result.succeeded_item_ids


@pytest.mark.asyncio
async def test_canary_driver_reconciles_rejected_label_after_terminal_lease_crash(
    tmp_path: Path,
) -> None:
    manifest = _manifest_with_rejected_item(tmp_path)
    rejected = next(
        binding
        for binding in manifest.trace_bindings
        if binding.expected_outcome is R6CanaryExpectedItemOutcomeV2.BLOCKED
    )
    driver = _driver(tmp_path)
    injector = _TargetWorkUnitFaultInjector(
        driver.job_store,
        stage=StageNameV2.LABEL,
        source_trace_id=rejected.source_trace_ref.object_id,
    )
    driver.fault_injector = injector

    with pytest.raises(CanaryDriverInjectedCrash):
        await driver.run(manifest)

    assert injector.fired
    assert injector.target_item_id is not None
    assert driver.job_store.get_item(injector.target_item_id).status is ItemStatus.RUNNING

    result = await _driver(tmp_path).run(manifest)
    assert injector.target_item_id in result.blocked_item_ids
    assert len(result.succeeded_item_ids) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", tuple(CanaryDriverFaultPoint))
async def test_artifact_group_resume_never_reinvokes_text_provider(
    tmp_path: Path,
    fault: CanaryDriverFaultPoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    calls = 0
    original_generate = TextProvider.generate

    def counted_generate(self, request):
        nonlocal calls
        calls += 1
        return original_generate(self, request)

    monkeypatch.setattr(
        TextProvider,
        "generate",
        counted_generate,
    )
    driver = _driver(tmp_path)
    injector: CanaryDriverFaultInjector = _ArtifactGroupFaultInjector(
        driver.job_store,
        fault,
    )
    driver.fault_injector = injector

    with pytest.raises(CanaryDriverInjectedCrash):
        await driver.run(manifest)
    assert isinstance(injector, _ArtifactGroupFaultInjector)
    assert injector.fired

    result = await _driver(tmp_path).run(manifest)
    assert result.job_status is JobStatus.SUCCEEDED
    assert calls == 1


@pytest.mark.asyncio
async def test_artifact_group_rejects_stale_completion_then_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    calls = 0
    original_generate = TextProvider.generate

    def counted_generate(self, request):
        nonlocal calls
        calls += 1
        return original_generate(self, request)

    monkeypatch.setattr(
        TextProvider,
        "generate",
        counted_generate,
    )
    driver = _driver(tmp_path)
    injector = _ArtifactGroupFaultInjector(
        driver.job_store,
        CanaryDriverFaultPoint.BEFORE_LEASE_COMPLETION,
    )
    driver.fault_injector = injector
    with pytest.raises(CanaryDriverInjectedCrash):
        await driver.run(manifest)

    units = {
        unit.resolved_work_unit_id: unit
        for unit in driver.job_store.list_work_units(job_id=manifest.job_spec.job_id)
    }
    lease = next(
        value
        for value in driver.job_store.list_work_leases(job_id=manifest.job_spec.job_id)
        if units[value.work_unit_ref.object_id].scope.value == "ARTIFACT_GROUP"
    )
    head = driver.job_store.get_work_lease_head(lease.work_lease_id)
    reservation = driver.resources.get_reservation_for_lease(lease.work_lease_id)
    policy = driver.job_store.get_work_control_policy(manifest.job_spec.job_id)
    with pytest.raises(StaleWorkLeaseError):
        driver.resources.complete_artifact_group(
            lease=lease,
            control_policy=policy,
            reservation=reservation,
            holder_ref=lease.holder_ref,
            expected_lease_version=head.lease_version + 1,
            event_kind=WorkLeaseEventKindV2.SUCCEEDED,
            result_refs=(),
            failure=None,
            usage=ResourceUsageV2(observed=NonModelResourceVectorV2.zero()),
            audit=manifest.audit,
            idempotency_key="r6-canary-stale-completion",
        )

    result = await _driver(tmp_path).run(manifest)
    assert result.job_status is JobStatus.SUCCEEDED
    assert calls == 1


@pytest.mark.asyncio
async def test_provider_operation_journal_survives_group_output_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    calls = 0
    original_generate = TextProvider.generate

    def counted_generate(self, request):
        nonlocal calls
        calls += 1
        return original_generate(self, request)

    monkeypatch.setattr(
        TextProvider,
        "generate",
        counted_generate,
    )
    injector = _ArtifactGroupOutputStoreFaultInjector()
    with pytest.raises(StageObjectInjectedCrash):
        await _driver(
            tmp_path,
            stage_fault_injector=injector,
        ).run(manifest)
    assert injector.fired

    stored = create_canary_stage_store(tmp_path / "stage-store").list_envelopes()
    assert sum(envelope.codec is R6CanaryObjectCodecV2.PROVIDER_OPERATION_OUTPUT for envelope in stored) == 1
    assert all(envelope.codec is not R6CanaryObjectCodecV2.ARTIFACT_GROUP_OUTPUT for envelope in stored)

    result = await _driver(tmp_path).run(manifest)
    assert result.job_status is JobStatus.SUCCEEDED
    assert calls == 1


@pytest.mark.asyncio
async def test_existing_provider_output_recovers_before_operation_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    calls = 0
    original_generate = TextProvider.generate

    def counted_generate(self, request):
        nonlocal calls
        calls += 1
        return original_generate(self, request)

    monkeypatch.setattr(
        TextProvider,
        "generate",
        counted_generate,
    )
    injector = _ProviderOperationStoreFaultInjector()
    with pytest.raises(StageObjectInjectedCrash):
        await _driver(
            tmp_path,
            stage_fault_injector=injector,
        ).run(manifest)
    assert injector.fired

    stored = create_canary_stage_store(tmp_path / "stage-store").list_envelopes()
    assert all(envelope.codec is not R6CanaryObjectCodecV2.PROVIDER_OPERATION_OUTPUT for envelope in stored)
    assert all(envelope.codec is not R6CanaryObjectCodecV2.ARTIFACT_GROUP_OUTPUT for envelope in stored)

    result = await _driver(tmp_path).run(manifest)
    assert result.job_status is JobStatus.SUCCEEDED
    assert calls == 1


@pytest.mark.asyncio
async def test_canary_resource_cancellation_stops_group_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    calls = 0
    original_generate = TextProvider.generate

    def counted_generate(self, request):
        nonlocal calls
        calls += 1
        return original_generate(self, request)

    monkeypatch.setattr(
        TextProvider,
        "generate",
        counted_generate,
    )
    driver = _driver(tmp_path)
    injector = _ArtifactGroupFaultInjector(
        driver.job_store,
        CanaryDriverFaultPoint.BEFORE_HANDLER,
    )
    driver.fault_injector = injector
    with pytest.raises(CanaryDriverInjectedCrash):
        await driver.run(manifest)

    cancellation = driver.lifecycle.cancel(
        job_id=manifest.job_spec.job_id,
        reason_code="operator_cancelled_canary",
        audit=manifest.audit,
        idempotency_key="r6-canary-cancel",
    )

    assert cancellation.mode.value == "RESOURCE"
    assert cancellation.pending_physical_acknowledgements == 0
    assert driver.job_store.get_job(manifest.job_spec.job_id).status is JobStatus.CANCELLED
    assert all(
        driver.job_store.get_work_lease_head(lease.work_lease_id).state.value != "ACTIVE"
        for lease in driver.job_store.list_work_leases(job_id=manifest.job_spec.job_id)
    )
    with pytest.raises(PipelineLifecyclePolicyError):
        await _driver(tmp_path).run(manifest)
    assert calls == 0


def test_public_canary_run_command_emits_bounded_result(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "pipeline",
            "canary-run",
            "--manifest",
            str(manifest_path),
            "--job-store",
            str(tmp_path / "cli-job.sqlite3"),
            "--source-registry",
            str(tmp_path / "cli-source.sqlite3"),
            "--trace-store",
            str(tmp_path / "cli-trace-store"),
            "--stage-store",
            str(tmp_path / "cli-stage-store"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["job_status"] == "SUCCEEDED"
    assert payload["claim_scope"] == "DEVELOPMENT_CANARY_ONLY"
    serialized = result.output.casefold()
    for forbidden in (
        "safe input",
        "visible_prompt",
        "payload_json",
        "private_reference",
        "physical_path",
    ):
        assert forbidden not in serialized


def test_public_canary_run_rejects_oversized_valid_manifest(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    encoded = manifest.model_dump_json().encode("utf-8")
    manifest_path = tmp_path / "oversized-manifest.json"
    manifest_path.write_bytes(encoded + (b" " * (8 * 1024 * 1024 + 1 - len(encoded))))

    result = CliRunner().invoke(
        app,
        [
            "pipeline",
            "canary-run",
            "--manifest",
            str(manifest_path),
            "--job-store",
            str(tmp_path / "oversized-job.sqlite3"),
            "--source-registry",
            str(tmp_path / "oversized-source.sqlite3"),
            "--trace-store",
            str(tmp_path / "oversized-trace-store"),
            "--stage-store",
            str(tmp_path / "oversized-stage-store"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "INVALID_INPUT"
    assert not (tmp_path / "oversized-job.sqlite3").exists()
