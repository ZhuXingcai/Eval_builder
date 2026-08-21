from __future__ import annotations

from pathlib import Path

import pytest
from test_support.team_runtime_fixtures import (
    audit,
    capability_runtime,
    ref,
    seed_envelope,
    team_fixture,
)

from eval_factory.harness.capability import CapabilityInvocationOutcomeV1
from eval_factory.harness.capability_runtime import (
    CapabilityProviderExecution,
    CapabilityRuntimeInputError,
)
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    RequirementPlanningCapabilityRequestV1,
    TraceIngestionCapabilityRequestV1,
)
from eval_factory.team import (
    TeamCapabilityRunner,
    TeamIdempotencyConflictError,
    TeamStore,
    TeamTaskStatusV1,
)


async def test_stage2_runtime_mount_publishes_heads_and_unblocks_join(
    tmp_path,
) -> None:
    fixture = team_fixture()
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    store.seed_artifact_head(
        team_ref=fixture.team.to_ref(),
        head_id="head-goal",
        envelope=seed_envelope(
            fixture,
            artifact_id="artifact-goal",
            semantic_role="evaluation-requirement",
            producer_capability_id="capability.requirement-planning",
        ),
        audit=audit(),
        idempotency_key="seed-goal",
    )
    registry, runtime, providers = capability_runtime(fixture)
    runner = TeamCapabilityRunner(
        store=store,
        runtime=runtime,
        registry=registry,
    )
    requirement = RequirementPlanningCapabilityRequestV1.create(
        plan_ref=ref("dataset-build-plan", "plan", version="v2"),
        policy_ref=ref("factory-run-policy", "policy", version="v2"),
        audit=audit(),
    )
    requirement_result = await runner.execute(
        team_id="team-stage3",
        task_id="task-requirement",
        request=requirement,
        audit=audit(),
    )
    assert requirement_result.result.canonical_result_ref is not None
    assert [head.head_id for head in requirement_result.artifact_heads] == [
        "head-plan",
    ]
    assert providers["capability.requirement-planning"].calls == 1
    assert (
        store.get_task_work(
            "team-stage3",
            "task-author",
        ).status
        is TeamTaskStatusV1.PENDING
    )

    trace = TraceIngestionCapabilityRequestV1.create(
        trace_source_ref=ref("trace-source", "source", version="v2"),
        job_id="job-stage3",
        attempt=1,
        audit=audit(),
    )
    trace_result = await runner.execute(
        team_id="team-stage3",
        task_id="task-trace",
        request=trace,
        audit=audit(),
    )
    assert [head.head_id for head in trace_result.artifact_heads] == [
        "head-trace",
    ]
    assert providers["capability.trace-ingestion"].calls == 1
    author = store.get_task_work("team-stage3", "task-author")
    assert author.status is TeamTaskStatusV1.READY
    snapshot = store.get_snapshot("team-stage3")
    author_grant = next(grant for grant in snapshot.authority.grants if grant.member_id == "member-task")
    assert (
        len(
            tuple(
                value for value in author_grant.data_scope_refs if value.object_type == "artifact-envelope"
            ),
        )
        == 1
    )
    context = store.projection_source("team-stage3", "member-task")
    assert {value.semantic_role for value in context.artifact_envelopes}.issuperset(
        {"compiled-build-plan", "trace-evidence"}
    )


@pytest.mark.parametrize(
    "fault_point",
    (
        "complete_task.after_call",
        "complete_task.after_result",
        "complete_task.after_usage",
        "complete_task.after_heads",
        "complete_task.after_task_head",
        "complete_task.after_response",
        "complete_task.after_outbox",
        "complete_task.after_idempotency",
    ),
)
async def test_runtime_exact_replay_does_not_repeat_owner_effect(
    tmp_path,
    fault_point: str,
) -> None:
    fixture = team_fixture()
    injected = False

    def fail_once(point: str) -> None:
        nonlocal injected
        if point == fault_point and not injected:
            injected = True
            raise RuntimeError("injected Team commit fault")

    store = TeamStore(
        tmp_path / "team.sqlite3",
        fault_injector=fail_once,
    )
    store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    store.seed_artifact_head(
        team_ref=fixture.team.to_ref(),
        head_id="head-goal",
        envelope=seed_envelope(
            fixture,
            artifact_id="artifact-goal",
            semantic_role="evaluation-requirement",
            producer_capability_id="capability.requirement-planning",
        ),
        audit=audit(),
        idempotency_key="seed-goal",
    )
    registry, runtime, providers = capability_runtime(fixture)
    runner = TeamCapabilityRunner(
        store=store,
        runtime=runtime,
        registry=registry,
    )
    request = RequirementPlanningCapabilityRequestV1.create(
        plan_ref=ref("dataset-build-plan", "plan", version="v2"),
        policy_ref=ref("factory-run-policy", "policy", version="v2"),
        audit=audit(),
    )
    with pytest.raises(RuntimeError, match="injected"):
        await runner.execute(
            team_id="team-stage3",
            task_id="task-requirement",
            request=request,
            audit=audit(),
        )
    assert providers["capability.requirement-planning"].calls == 1
    assert (
        store.get_task_work(
            "team-stage3",
            "task-requirement",
        ).status
        is TeamTaskStatusV1.ACTIVE
    )
    current = store.get_snapshot("team-stage3")
    store.seed_artifact_head(
        team_ref=current.team.to_ref(),
        head_id="head-attachment",
        envelope=seed_envelope(
            fixture,
            artifact_id="artifact-attachment",
            semantic_role="attachment-package",
            producer_capability_id="capability.attachment-reconstruction",
        ),
        audit=audit(),
        idempotency_key="seed-attachment-after-owner",
    )
    first = await runner.execute(
        team_id="team-stage3",
        task_id="task-requirement",
        request=request,
        audit=audit(),
    )
    assert providers["capability.requirement-planning"].calls == 1
    assert (
        store.get_task_work(
            "team-stage3",
            "task-requirement",
        ).status
        is TeamTaskStatusV1.COMPLETED
    )
    assert first.result.canonical_result_ref is not None


async def test_completion_replay_returns_original_historical_response(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = team_fixture()
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    store.seed_artifact_head(
        team_ref=fixture.team.to_ref(),
        head_id="head-goal",
        envelope=seed_envelope(
            fixture,
            artifact_id="artifact-goal",
            semantic_role="evaluation-requirement",
            producer_capability_id="capability.requirement-planning",
        ),
        audit=audit(),
        idempotency_key="seed-goal",
    )
    registry, runtime, _ = capability_runtime(fixture)
    captured = {}
    invoke = runtime.invoke

    async def record_invocation(**kwargs):
        invocation = await invoke(**kwargs)
        captured["invocation"] = invocation
        return invocation

    monkeypatch.setattr(runtime, "invoke", record_invocation)
    runner = TeamCapabilityRunner(
        store=store,
        runtime=runtime,
        registry=registry,
    )
    first = await runner.execute(
        team_id="team-stage3",
        task_id="task-requirement",
        request=RequirementPlanningCapabilityRequestV1.create(
            plan_ref=ref("dataset-build-plan", "plan", version="v2"),
            policy_ref=ref("factory-run-policy", "policy", version="v2"),
            audit=audit(),
        ),
        audit=audit(),
    )
    invocation = captured["invocation"]
    assert invocation.result.output_artifact_refs == first.result.output_artifact_refs
    lease = store.get_task_work(
        "team-stage3",
        "task-requirement",
    ).lease
    assert lease is not None

    current = store.get_snapshot("team-stage3")
    store.seed_artifact_head(
        team_ref=current.team.to_ref(),
        head_id="head-attachment",
        envelope=seed_envelope(
            fixture,
            artifact_id="artifact-attachment",
            semantic_role="attachment-package",
            producer_capability_id="capability.attachment-reconstruction",
        ),
        audit=audit(),
        idempotency_key="seed-attachment",
    )
    assert invocation.call.context.team_ref is not None
    replay = store.complete_task(
        team_ref=invocation.call.context.team_ref,
        graph_ref=fixture.graph.to_ref(),
        authority_ref=invocation.call.context.authority_ref,
        lease_ref=lease.to_ref(),
        fencing_token=lease.fencing_token,
        invocation=invocation,
        output_bindings=(("head-plan", invocation.output_artifacts[0]),),
        audit=audit(),
        idempotency_key=(f"{invocation.call.context.idempotency_key}.commit"),
    )

    assert replay == first
    runner_replay = await runner.execute(
        team_id="team-stage3",
        task_id="task-requirement",
        request=RequirementPlanningCapabilityRequestV1.create(
            plan_ref=ref("dataset-build-plan", "plan", version="v2"),
            policy_ref=ref("factory-run-policy", "policy", version="v2"),
            audit=audit(),
        ),
        audit=audit(),
    )
    assert runner_replay == first
    with pytest.raises(
        TeamIdempotencyConflictError,
        match="another Capability request",
    ):
        await runner.execute(
            team_id="team-stage3",
            task_id="task-requirement",
            request=RequirementPlanningCapabilityRequestV1.create(
                plan_ref=ref(
                    "dataset-build-plan",
                    "changed-plan",
                    version="v2",
                ),
                policy_ref=ref(
                    "factory-run-policy",
                    "policy",
                    version="v2",
                ),
                audit=audit(),
            ),
            audit=audit(),
        )


async def test_retry_preserves_capability_failure_reason(
    tmp_path: Path,
) -> None:
    fixture = team_fixture()
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    store.seed_artifact_head(
        team_ref=fixture.team.to_ref(),
        head_id="head-goal",
        envelope=seed_envelope(
            fixture,
            artifact_id="artifact-goal",
            semantic_role="evaluation-requirement",
            producer_capability_id="capability.requirement-planning",
        ),
        audit=audit(),
        idempotency_key="seed-goal",
    )
    registry, runtime, providers = capability_runtime(fixture)
    provider = providers["capability.requirement-planning"]

    async def require_approval(
        *args: object,
        **kwargs: object,
    ) -> CapabilityProviderExecution:
        del args, kwargs
        provider.calls += 1
        return CapabilityProviderExecution(
            outcome=CapabilityInvocationOutcomeV1.BLOCKED_POLICY,
            failure_code="USER_APPROVAL_REQUIRED",
        )

    provider.invoke = require_approval  # type: ignore[method-assign]
    runner = TeamCapabilityRunner(
        store=store,
        runtime=runtime,
        registry=registry,
    )

    completion = await runner.execute(
        team_id="team-stage3",
        task_id="task-requirement",
        request=RequirementPlanningCapabilityRequestV1.create(
            plan_ref=ref("dataset-build-plan", "plan", version="v2"),
            policy_ref=ref("factory-run-policy", "policy", version="v2"),
            audit=audit(),
        ),
        audit=audit(),
    )

    assert completion.event.reason_code == "USER_APPROVAL_REQUIRED"
    work = store.get_task_work("team-stage3", "task-requirement")
    assert work.status is TeamTaskStatusV1.READY
    assert work.latest_event is not None
    assert work.latest_event.reason_code == "USER_APPROVAL_REQUIRED"


async def test_task_budget_blocks_before_claim_or_owner_effect(tmp_path) -> None:
    fixture = team_fixture()
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=fixture.team,
        roster=fixture.roster,
        graph=fixture.graph,
        authority=fixture.authority,
        idempotency_key="create-team",
    )
    store.seed_artifact_head(
        team_ref=fixture.team.to_ref(),
        head_id="head-goal",
        envelope=seed_envelope(
            fixture,
            artifact_id="artifact-goal",
            semantic_role="evaluation-requirement",
            producer_capability_id="capability.requirement-planning",
        ),
        audit=audit(),
        idempotency_key="seed-goal",
    )
    registry, runtime, providers = capability_runtime(fixture)
    runner = TeamCapabilityRunner(
        store=store,
        runtime=runtime,
        registry=registry,
    )

    with pytest.raises(CapabilityRuntimeInputError, match="budget"):
        await runner.execute(
            team_id="team-stage3",
            task_id="task-requirement",
            request=RequirementPlanningCapabilityRequestV1.create(
                plan_ref=ref("dataset-build-plan", "plan", version="v2"),
                policy_ref=ref("factory-run-policy", "policy", version="v2"),
                audit=audit(),
            ),
            audit=audit(),
            model_requests_delta=3,
        )

    assert providers["capability.requirement-planning"].calls == 0
    assert (
        store.get_task_work(
            "team-stage3",
            "task-requirement",
        ).status
        is TeamTaskStatusV1.READY
    )
