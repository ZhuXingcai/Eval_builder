from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from tests.eval_factory.integration.test_harness_agent_loop import (
    _components as _harness_components,
)
from tests.integration.test_evalfactory_agent_run_cli import (
    _approve_and_resume,
    _approve_review_batch,
    _args,
    _audit,
    _one_trace_root,
    _real_trace_root,
    _registry,
    _requirement,
    _two_trace_root,
)
from typer.testing import CliRunner

from eval_factory.agent_system.dataset_runtime import FactoryDatasetRuntime
from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.cli import app
from eval_factory.contracts.agent_system_v2 import (
    PlanKindV2,
    PlannerAssessmentActionV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetOutcomeV2,
)
from eval_factory.harness import (
    RequirementInterpretationProposalV1,
)
from eval_factory.packs import (
    build_generic_agent_trace_execution_pack,
)
from eval_factory.packs.generic_agent_trace.product_builder import (
    GenericAgentGraphRuntimeConfigV1,
)
from eval_factory.team import TeamStore


def _ref(object_type: str, suffix: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v1",
        object_version="v1",
        object_sha256="a" * 64,
    )


def _non_candidate_trace_root(
    tmp_path: Path,
) -> tuple[Path, str]:
    manifest, _manifest_sha256 = _real_trace_root(
        tmp_path,
        instance_ids=("LH_006",),
    )
    source_path = next(manifest.parent.glob("LH_006_*.jsonl"))
    envelope = json.loads(source_path.read_text(encoding="utf-8"))
    request = json.loads(envelope["request"])
    request["messages"] = [message for message in request["messages"] if message.get("role") != "user"]
    envelope["request"] = json.dumps(
        request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    source_path.write_text(
        json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _graph_cli_context(
    tmp_path: Path,
    *,
    manifest: Path,
    manifest_sha256: str,
    blocked_core_prompt_refs: tuple[ObjectRef, ...] = (),
) -> tuple[
    list[str],
    FactoryControlStore,
    PlanReviewService,
    GenericAgentGraphRuntimeConfigV1,
]:
    registration = build_generic_agent_trace_execution_pack(
        audit=_audit(),
    )
    proposal = RequirementInterpretationProposalV1(
        outcome="READY",
        assistant_message="Requirement is ready for Graph execution.",
        goals=("Build generic Agent evaluation data.",),
        source_expectations=("Use the admitted trace manifest.",),
        target_capabilities=("generic-agent-trace",),
        quality_intent="Auditable candidate data.",
        delivery_intent="Candidate JSON and JSONL.",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=10_000_000,
    )
    service, session_store, _agent, _provider = _harness_components(
        tmp_path / "harness-runtime",
        proposal,
    )
    main = service.create_session(
        session_id="graph-main-session",
        incarnation_id="graph-main-session-incarnation-1",
        composition_ref=registration.composition.to_ref(),
        created_by="graph-cli-test",
        idempotency_key="create-graph-main-session",
        audit=_audit(),
    )
    result = asyncio.run(
        service.post_message(
            session_id=main.session.session_id,
            expected_session_version=main.session.session_version,
            principal_ref=_ref("principal", "graph-cli"),
            content="Build the approved evaluation dataset.",
            artifact_envelope_refs=(),
            idempotency_key="ready-graph-main-session",
            audit=_audit(),
        )
    )
    assert result.requirement_policy_ref is not None
    member_sessions = {}
    for role in (
        "control",
        "coordinator",
        "quality",
        "requirement",
        "task",
        "trace",
    ):
        session_id = f"graph-member-{role}"
        session_store.create_session(
            session_id=session_id,
            incarnation_id=f"{session_id}-incarnation-1",
            composition_ref=registration.composition.to_ref(),
            created_by="graph-cli-test",
            idempotency_key=f"create-{session_id}",
            audit=_audit(),
        )
        member_sessions[role] = session_id
    graph_config = GenericAgentGraphRuntimeConfigV1(
        main_session_id=main.session.session_id,
        control_session_id=member_sessions["control"],
        coordinator_session_id=member_sessions["coordinator"],
        quality_session_id=member_sessions["quality"],
        requirement_session_id=member_sessions["requirement"],
        task_session_id=member_sessions["task"],
        trace_session_id=member_sessions["trace"],
        team_id="team-default-graph-cli",
        team_incarnation_id="team-default-graph-cli-incarnation-1",
        graph_binding_id="graph-binding-default-cli",
        thread_id="graph-thread-default-cli",
    )
    graph_config_path = tmp_path / "graph-runtime.json"
    graph_config_path.write_text(
        graph_config.model_dump_json(indent=2),
        encoding="utf-8",
    )
    arguments = _args(
        tmp_path,
        core_manifest=manifest,
        core_manifest_sha256=manifest_sha256,
        include_r4=True,
        include_attachment=True,
        include_specialists=True,
        include_release=True,
        blocked_core_prompt_refs=blocked_core_prompt_refs,
    )
    arguments.remove("--direct-runtime-compatibility")
    arguments.extend(
        (
            "--graph-runtime-config",
            str(graph_config_path),
            "--harness-store",
            str(session_store.path),
            "--team-store",
            str(tmp_path / "team.sqlite3"),
            "--capability-request-store",
            str(tmp_path / "capability-requests.sqlite3"),
            "--graph-journal",
            str(tmp_path / "graph-journal.sqlite3"),
            "--graph-checkpoint",
            str(tmp_path / "graph-checkpoint.sqlite3"),
        )
    )
    factory_store = FactoryControlStore(tmp_path / "factory.sqlite3")
    reviews = PlanReviewService(
        factory_store,
        compiler=DatasetBuildPlanCompiler(_registry()),
    )
    return arguments, factory_store, reviews, graph_config


def _forbid_direct_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reject_direct_runtime(
        self: FactoryDatasetRuntime,
        **values: object,
    ) -> object:
        del self, values
        raise AssertionError("default CLI invoked direct runtime")

    monkeypatch.setattr(
        FactoryDatasetRuntime,
        "advance",
        reject_direct_runtime,
    )


def _approve_item_reviews(
    *,
    factory_store: FactoryControlStore,
    reviews: PlanReviewService,
    payload: dict[str, object],
    plan_kind: PlanKindV2,
    suffix: str,
) -> None:
    bindings = factory_store.list_item_bindings(
        _requirement().run_id,
    )
    child_run_ids = {factory_store.get_item_run(binding).run_id for binding in bindings}
    _approve_review_batch(
        reviews,
        payload,
        plan_kind=plan_kind,
        expected_plan_refs={
            factory_store.get_domain_plan(
                run_id,
                plan_kind,
            ).plan_ref
            for run_id in child_run_ids
        },
        suffix=suffix,
    )


def test_agent_run_cli_uses_first_party_graph_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_direct_runtime(monkeypatch)
    manifest, manifest_sha256 = _two_trace_root(tmp_path)
    arguments, factory_store, reviews, graph_config = _graph_cli_context(
        tmp_path,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )

    waiting = CliRunner().invoke(app, arguments)

    assert waiting.exit_code == 0, waiting.stderr
    payload = json.loads(waiting.stdout)
    assert payload["status"] == "WAITING_REVIEW"
    assert len(payload["pending_review_refs"]) == 1
    assert (tmp_path / "team.sqlite3").is_file()
    assert (tmp_path / "graph-journal.sqlite3").is_file()
    assert (tmp_path / "graph-checkpoint.sqlite3").is_file()

    _approve_and_resume(
        reviews,
        payload["pending_review_refs"][0]["object_id"],
        suffix="graph-global",
    )

    after_global = CliRunner().invoke(app, arguments)

    assert after_global.exit_code == 0, after_global.stderr
    after_payload = json.loads(after_global.stdout)
    assert after_payload["status"] == "WAITING_REVIEW"
    assert len(after_payload["pending_review_refs"]) == 2
    assert after_payload["incomplete_count"] == 2
    team = TeamStore(tmp_path / "team.sqlite3").get_snapshot(
        graph_config.team_id,
    )
    assert team.graph.revision >= 1
    assert (
        len(
            factory_store.list_item_bindings(
                _requirement().run_id,
            )
        )
        == 2
    )
    bindings = factory_store.list_item_bindings(
        _requirement().run_id,
    )
    child_run_ids = {factory_store.get_item_run(binding).run_id for binding in bindings}
    _approve_review_batch(
        reviews,
        after_payload,
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        expected_plan_refs={
            factory_store.get_domain_plan(
                run_id,
                PlanKindV2.ATTACHMENT_GENERATION,
            ).plan_ref
            for run_id in child_run_ids
        },
        suffix="graph-attachment",
    )

    criteria = CliRunner().invoke(app, arguments)
    assert criteria.exit_code == 0, criteria.stderr
    criteria_payload = json.loads(criteria.stdout)
    _approve_review_batch(
        reviews,
        criteria_payload,
        plan_kind=PlanKindV2.CRITERIA_RUBRIC,
        expected_plan_refs={
            factory_store.get_domain_plan(
                run_id,
                PlanKindV2.CRITERIA_RUBRIC,
            ).plan_ref
            for run_id in child_run_ids
        },
        suffix="graph-criteria",
    )

    grading = CliRunner().invoke(app, arguments)
    assert grading.exit_code == 0, grading.stderr
    grading_payload = json.loads(grading.stdout)
    _approve_review_batch(
        reviews,
        grading_payload,
        plan_kind=PlanKindV2.GRADING_DESIGN,
        expected_plan_refs={
            factory_store.get_domain_plan(
                run_id,
                PlanKindV2.GRADING_DESIGN,
            ).plan_ref
            for run_id in child_run_ids
        },
        suffix="graph-grading",
    )

    delivery = CliRunner().invoke(app, arguments)
    assert delivery.exit_code == 0, delivery.stderr
    delivery_payload = json.loads(delivery.stdout)
    delivery_refs = delivery_payload["pending_review_refs"]
    assert len(delivery_refs) == 1
    delivery_review = reviews.show(
        delivery_refs[0]["object_id"],
    )
    assert delivery_review.request.plan_kind is PlanKindV2.FINAL_DELIVERY
    _approve_and_resume(
        reviews,
        delivery_review.request.review_request_id,
        suffix="graph-delivery",
    )

    completed = CliRunner().invoke(app, arguments)
    replay = CliRunner().invoke(app, arguments)
    assert completed.exit_code == 0, completed.stderr
    assert replay.exit_code == 0, replay.stderr
    completed_payload = json.loads(completed.stdout)
    assert json.loads(replay.stdout) == completed_payload
    assert completed_payload["status"] == "COMPLETED"
    assert completed_payload["candidate_count"] == 2
    assert completed_payload["delivery_manifest_ref"] is not None


def test_agent_run_cli_zero_candidate_stops_before_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_direct_runtime(monkeypatch)
    manifest, manifest_sha256 = _non_candidate_trace_root(tmp_path)
    arguments, factory_store, reviews, graph_config = _graph_cli_context(
        tmp_path,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    waiting = CliRunner().invoke(app, arguments)
    assert waiting.exit_code == 0, waiting.stderr
    waiting_payload = json.loads(waiting.stdout)
    _approve_and_resume(
        reviews,
        waiting_payload["pending_review_refs"][0]["object_id"],
        suffix="graph-zero-global",
    )

    blocked = CliRunner().invoke(app, arguments)
    assert blocked.exit_code == 0, blocked.stderr
    blocked_payload = json.loads(blocked.stdout)
    assessments_before_replay = factory_store.list_planner_assessments(
        _requirement().run_id,
    )
    replay = CliRunner().invoke(app, arguments)

    assert replay.exit_code == 0, replay.stderr
    assert json.loads(replay.stdout) == blocked_payload
    assert blocked_payload["status"] == "BLOCKED"
    assert blocked_payload["next_action"] == "NONE"
    assert blocked_payload["pending_review_refs"] == []
    assert blocked_payload["candidate_count"] == 0
    assert blocked_payload["delivery_manifest_ref"] is None
    aggregate = factory_store.get_dataset_aggregate(
        _requirement().run_id,
    )
    assert aggregate.outcome is CandidateDatasetOutcomeV2.NO_ELIGIBLE_ITEMS
    assert aggregate.reason_codes == ("NO_ELIGIBLE_ITEMS",)
    assert assessments_before_replay[-1].proposed_action is PlannerAssessmentActionV2.ESCALATE
    assert assessments_before_replay[-1].rationale_codes == ("TERMINAL_WORK_FAILED",)
    assert (
        factory_store.list_planner_assessments(
            _requirement().run_id,
        )
        == assessments_before_replay
    )
    work = TeamStore(tmp_path / "team.sqlite3").list_task_work(
        graph_config.team_id,
    )
    batch = tuple(value for value in work if value.task.task_kind == "batch-quality")
    assert len(batch) == 1
    assert batch[0].status.value == "FAILED"
    assert batch[0].attempt == batch[0].task.max_attempts


def test_agent_run_cli_single_candidate_blocks_cross_item_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_direct_runtime(monkeypatch)
    manifest, manifest_sha256 = _one_trace_root(tmp_path)
    arguments, factory_store, reviews, graph_config = _graph_cli_context(
        tmp_path,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    waiting = CliRunner().invoke(app, arguments)
    assert waiting.exit_code == 0, waiting.stderr
    waiting_payload = json.loads(waiting.stdout)
    _approve_and_resume(
        reviews,
        waiting_payload["pending_review_refs"][0]["object_id"],
        suffix="graph-single-global",
    )

    after_global = CliRunner().invoke(app, arguments)
    assert after_global.exit_code == 0, after_global.stderr
    payload = json.loads(after_global.stdout)
    for plan_kind, suffix in (
        (PlanKindV2.ATTACHMENT_GENERATION, "graph-single-attachment"),
        (PlanKindV2.CRITERIA_RUBRIC, "graph-single-criteria"),
        (PlanKindV2.GRADING_DESIGN, "graph-single-grading"),
    ):
        _approve_item_reviews(
            factory_store=factory_store,
            reviews=reviews,
            payload=payload,
            plan_kind=plan_kind,
            suffix=suffix,
        )
        result = CliRunner().invoke(app, arguments)
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)

    assessments_before_replay = factory_store.list_planner_assessments(
        _requirement().run_id,
    )
    replay = CliRunner().invoke(app, arguments)
    assert replay.exit_code == 0, replay.stderr
    assert json.loads(replay.stdout) == payload
    assert payload["status"] == "BLOCKED"
    assert payload["next_action"] == "NONE"
    assert payload["pending_review_refs"] == []
    assert payload["candidate_count"] == 0
    assert payload["blocked_count"] == 1
    assert payload["delivery_manifest_ref"] is None
    aggregate = factory_store.get_dataset_aggregate(
        _requirement().run_id,
    )
    assert aggregate.outcome is CandidateDatasetOutcomeV2.BLOCKED
    assert aggregate.reason_codes == ("INSUFFICIENT_BATCH_CANDIDATES",)
    assert assessments_before_replay[-1].proposed_action is PlannerAssessmentActionV2.ESCALATE
    assert assessments_before_replay[-1].rationale_codes == ("TERMINAL_WORK_FAILED",)
    assert (
        factory_store.list_planner_assessments(
            _requirement().run_id,
        )
        == assessments_before_replay
    )
    work = TeamStore(tmp_path / "team.sqlite3").list_task_work(
        graph_config.team_id,
    )
    batch = tuple(value for value in work if value.task.task_kind == "batch-quality")
    assert len(batch) == 1
    assert batch[0].status.value == "FAILED"
    assert batch[0].attempt == batch[0].task.max_attempts
