from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from test_support.team_runtime_fixtures import (
    HASH,
    audit,
    ref,
    team_fixture,
)

from eval_factory.agent_system.graph import FactoryGraphDriftError
from eval_factory.agent_system.graph_authority import (
    FactoryGraphAuthorityResolver,
)
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalStore,
)
from eval_factory.blueprints import EvaluationBlueprintV1
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    FactoryRunStatusV2,
    FactoryRunV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
)
from eval_factory.harness import (
    ArtifactHeadV1,
    ExecutionAuthorityV1,
    HarnessGraphExecutionBindingV1,
    HarnessSessionIdentityV1,
    HarnessSessionProjectionV1,
    HarnessSessionStateV1,
    HarnessSessionStatusV1,
    MemberExecutionGrantV1,
)
from eval_factory.packs import (
    build_generic_agent_evaluation_blueprint,
)
from eval_factory.packs.generic_agent_trace.manifest import (
    generic_agent_trace_pack_registry,
)
from eval_factory.team import (
    TeamCheckpointV1,
    TeamSnapshot,
    TeamTaskGraphV1,
    TeamV1,
)


@dataclass
class _SessionSource:
    projection: HarnessSessionProjectionV1

    def get_projection_by_ref(
        self,
        reference: ObjectRef,
    ) -> HarnessSessionProjectionV1:
        assert reference == self.projection.session.session_ref
        return self.projection


@dataclass
class _FactorySource:
    run: FactoryRunV2
    current_run: FactoryRunV2
    policy: FactoryRunPolicyV2
    requirement: EvaluationRequirementSpecV2
    request: FactoryDatasetRunRequestV2

    def get_run_by_ref(self, reference: ObjectRef) -> FactoryRunV2:
        assert reference == self.run.to_ref()
        return self.run

    def get_run(self, run_id: str) -> FactoryRunV2:
        assert run_id == self.run.run_id
        return self.current_run

    def get_policy_by_ref(
        self,
        reference: ObjectRef,
    ) -> FactoryRunPolicyV2:
        assert reference == self.policy.to_ref()
        return self.policy

    def get_requirement_by_ref(
        self,
        reference: ObjectRef,
    ) -> EvaluationRequirementSpecV2:
        assert reference == self.requirement.to_ref()
        return self.requirement

    def get_dataset_request_by_ref(
        self,
        reference: ObjectRef,
    ) -> FactoryDatasetRunRequestV2:
        assert reference == self.request.to_ref()
        return self.request


@dataclass
class _TeamSource:
    snapshot: TeamSnapshot
    current_snapshot: TeamSnapshot
    checkpoint: TeamCheckpointV1 | None

    def get_snapshot_at(self, team_ref: ObjectRef) -> TeamSnapshot:
        assert team_ref == self.snapshot.team.to_ref()
        return self.snapshot

    def get_snapshot(self, team_id: str) -> TeamSnapshot:
        assert team_id == self.snapshot.team.team_id
        return self.current_snapshot

    def current_checkpoint(
        self,
        team_id: str,
    ) -> TeamCheckpointV1 | None:
        assert team_id == self.snapshot.team.team_id
        return self.checkpoint

    def current_artifact_heads(
        self,
        team_id: str,
    ) -> tuple[ArtifactHeadV1, ...]:
        assert team_id == self.snapshot.team.team_id
        return ()


@dataclass
class _Fixture:
    resolver: FactoryGraphAuthorityResolver
    binding: HarnessGraphExecutionBindingV1
    session_source: _SessionSource
    factory_source: _FactorySource
    team_source: _TeamSource
    blueprint: EvaluationBlueprintV1


def _build_fixture(tmp_path: Path) -> _Fixture:
    team_base = team_fixture()
    registration = team_base.registration
    policy = FactoryRunPolicyV2.create(
        policy_id="factory-policy.stage4",
        allowed_task_kinds=("dataset-production",),
        max_transitions=64,
        max_plan_revisions=4,
        max_agent_attempts=2,
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    requirement = EvaluationRequirementSpecV2.create(
        requirement_spec_id="requirement.stage4",
        run_id="factory-run.stage4",
        source_ref=ref(
            "evaluation-requirement-source",
            "stage4",
            version="v2",
        ),
        goals=("Build a generic Agent evaluation dataset.",),
        constraints=("Keep raw traces outside Graph state.",),
        assumptions=("Provider execution is a mechanism fixture.",),
        open_questions=(),
        requirement_version=1,
        audit=audit(),
    )
    run = FactoryRunV2.create(
        run_id=requirement.run_id,
        run_version=0,
        status=FactoryRunStatusV2.CREATED,
        policy_ref=policy.to_ref(),
        requirement_spec_ref=requirement.to_ref(),
        current_plan_ref=None,
        compiled_plan_ref=None,
        active_task_refs=(),
        result_refs=(),
        pending_review_ref=None,
        planner_assessment_ref=None,
        completion_ref=None,
        delivery_manifest_ref=None,
        transition_count=0,
        model_requests_used=0,
        model_tokens_used=0,
        cost_micro_usd_used=0,
        audit=audit(),
    )
    request = FactoryDatasetRunRequestV2.create(
        dataset_run_id=run.run_id,
        requirement_spec_ref=requirement.to_ref(),
        manifest_ref=ref("trace-manifest", "stage4", version="v2"),
        source_authorization_ref=ref(
            "trace-source-authorization",
            "stage4",
            version="v2",
        ),
        factory_policy_ref=policy.to_ref(),
        pipeline_policy_refs=(ref("trace-index-policy", "stage4", version="v2"),),
        gateway_registry_refs=(ref("agent-registry", "stage4", version="v2"),),
        output_target_ref=ref(
            "candidate-output-target",
            "stage4",
            version="v2",
        ),
        idempotency_key="dataset-run-stage4",
        max_transitions=64,
        audit=audit(),
    )

    team = TeamV1.create(
        team_id=team_base.team.team_id,
        team_incarnation_id=team_base.team.team_incarnation_id,
        team_version=team_base.team.team_version,
        goal_ref=requirement.to_ref(),
        composition_ref=team_base.team.composition_ref,
        roster_ref=team_base.team.roster_ref,
        task_graph_ref=team_base.team.task_graph_ref,
        authority_ref=team_base.team.authority_ref,
        coordinator_member_id=team_base.team.coordinator_member_id,
        lifecycle=team_base.team.lifecycle,
        audit=audit(),
    )
    snapshot = TeamSnapshot(
        team=team,
        roster=team_base.roster,
        graph=team_base.graph,
        authority=team_base.authority,
    )
    checkpoint = TeamCheckpointV1.create(
        checkpoint_id="team-stage4.checkpoint-1",
        team_ref=team.to_ref(),
        roster_ref=snapshot.roster.to_ref(),
        task_graph_ref=snapshot.graph.to_ref(),
        authority_ref=snapshot.authority.to_ref(),
        artifact_head_refs=(),
        subscription_refs=(),
        member_session_refs=tuple(member.independent_session_ref for member in snapshot.roster.members),
        message_cursor=0,
        audit=audit(),
    )
    blueprint = build_generic_agent_evaluation_blueprint(
        registration=registration,
        requirement_ref=requirement.to_ref(),
        team_ref=team.to_ref(),
        roster_ref=snapshot.roster.to_ref(),
        task_graph_ref=snapshot.graph.to_ref(),
        authority_ref=snapshot.authority.to_ref(),
        audit=audit(),
    )

    requirement_policy_ref = ref(
        "harness-requirement-policy",
        "stage4-ready",
    )
    identity = HarnessSessionIdentityV1.create(
        session_id="session-stage4",
        incarnation_id="session-stage4-incarnation-1",
        composition_ref=registration.composition.to_ref(),
        created_by="stage4-test",
        created_at=audit().created_at,
        audit=audit(),
    )
    session_state = HarnessSessionStateV1.create(
        identity=identity,
        session_version=3,
        status=HarnessSessionStatusV1.ACTIVE,
        last_event_sequence=3,
        current_interpretation_ref=ref(
            "requirement-interpretation",
            "stage4-ready",
        ),
        current_requirement_policy_ref=requirement_policy_ref,
        updated_at=audit().created_at,
    )
    projection = HarnessSessionProjectionV1(
        session=session_state,
        transcript=(),
        event_refs=(),
        current_interpretation_ref=(session_state.current_interpretation_ref),
        current_requirement_policy_ref=requirement_policy_ref,
        latest_gateway=None,
        pending_clarification_questions=(),
        source_fingerprint=HASH,
    )

    binding = HarnessGraphExecutionBindingV1.create(
        binding_id="graph-binding.stage4",
        session_ref=identity.to_ref(),
        requirement_ref=requirement.to_ref(),
        requirement_policy_ref=requirement_policy_ref,
        factory_run_ref=run.to_ref(),
        factory_request_ref=request.to_ref(),
        factory_policy_ref=policy.to_ref(),
        expected_factory_run_version=run.run_version,
        pack_manifest_ref=registration.manifest.to_ref(),
        composition_ref=registration.composition.to_ref(),
        blueprint_ref=blueprint.to_ref(),
        team_ref=team.to_ref(),
        roster_ref=snapshot.roster.to_ref(),
        task_graph_ref=snapshot.graph.to_ref(),
        execution_authority_ref=snapshot.authority.to_ref(),
        team_checkpoint_ref=checkpoint.to_ref(),
        expected_team_version=team.team_version,
        expected_task_graph_revision=snapshot.graph.revision,
        expected_authority_version=(snapshot.authority.authority_version),
        blackboard_head_refs=(),
        thread_id="thread-stage4",
        max_transitions=64,
        audit=audit(),
    )
    journal = FactoryGraphJournalStore(tmp_path / "graph-journal.sqlite3")
    journal.commit_binding(binding, idempotency_key="binding-stage4")
    session_source = _SessionSource(projection)
    factory_source = _FactorySource(
        run=run,
        current_run=run,
        policy=policy,
        requirement=requirement,
        request=request,
    )
    team_source = _TeamSource(
        snapshot=snapshot,
        current_snapshot=snapshot,
        checkpoint=checkpoint,
    )
    resolver = FactoryGraphAuthorityResolver(
        journal=journal,
        session_source=session_source,
        factory_source=factory_source,
        team_source=team_source,
        pack_registry=generic_agent_trace_pack_registry(audit=audit()),
        blueprints=(blueprint,),
    )
    return _Fixture(
        resolver=resolver,
        binding=binding,
        session_source=session_source,
        factory_source=factory_source,
        team_source=team_source,
        blueprint=blueprint,
    )


def test_graph_authority_accepts_non_widening_team_graph_successor(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    source = fixture.team_source.snapshot
    tasks = tuple(task for task in source.graph.tasks if task.task_kind != "delivery")
    graph = TeamTaskGraphV1.create(
        graph_id=source.graph.graph_id,
        team_id=source.team.team_id,
        revision=source.graph.revision + 1,
        predecessor_graph_ref=source.graph.to_ref(),
        roster_ref=source.roster.to_ref(),
        member_ids=source.graph.member_ids,
        tasks=tasks,
        audit=audit(),
    )
    retained_ids = {task.task_id for task in tasks}
    grants = tuple(
        MemberExecutionGrantV1(
            member_id=grant.member_id,
            principal_ref=grant.principal_ref,
            capability_definition_refs=(grant.capability_definition_refs),
            provider_binding_refs=grant.provider_binding_refs,
            task_ids=tuple(task_id for task_id in grant.task_ids if task_id in retained_ids),
            data_scope_refs=grant.data_scope_refs,
            data_purposes=grant.data_purposes,
            data_classifications=grant.data_classifications,
            allowed_side_effects=grant.allowed_side_effects,
        )
        for grant in source.authority.grants
    )
    authority = ExecutionAuthorityV1.create(
        authority_id=source.authority.authority_id,
        authority_version=source.authority.authority_version + 1,
        predecessor_authority_ref=source.authority.to_ref(),
        team_id=source.team.team_id,
        team_incarnation_id=source.team.team_incarnation_id,
        roster_ref=source.roster.to_ref(),
        task_graph_ref=graph.to_ref(),
        permission_policy_ref=(source.authority.permission_policy_ref),
        grants=grants,
        max_model_requests=source.authority.max_model_requests,
        max_model_tokens=source.authority.max_model_tokens,
        max_cost_micro_usd=source.authority.max_cost_micro_usd,
        used_model_requests=source.authority.used_model_requests,
        used_model_tokens=source.authority.used_model_tokens,
        used_cost_micro_usd=source.authority.used_cost_micro_usd,
        audit=audit(),
    )
    team = TeamV1.create(
        team_id=source.team.team_id,
        team_incarnation_id=source.team.team_incarnation_id,
        team_version=source.team.team_version + 1,
        goal_ref=source.team.goal_ref,
        composition_ref=source.team.composition_ref,
        roster_ref=source.roster.to_ref(),
        task_graph_ref=graph.to_ref(),
        authority_ref=authority.to_ref(),
        coordinator_member_id=source.team.coordinator_member_id,
        lifecycle=source.team.lifecycle,
        audit=audit(),
    )
    current = TeamSnapshot(
        team=team,
        roster=source.roster,
        graph=graph,
        authority=authority,
    )
    binding = fixture.binding.model_copy(
        update={
            "team_ref": team.to_ref(),
            "task_graph_ref": graph.to_ref(),
            "execution_authority_ref": authority.to_ref(),
            "expected_team_version": team.team_version,
            "expected_task_graph_revision": graph.revision,
            "expected_authority_version": (authority.authority_version),
        },
    )

    FactoryGraphAuthorityResolver._validate_blueprint(
        binding,
        blueprint=fixture.blueprint,
        source_snapshot=source,
        current_snapshot=current,
    )


def test_resolver_rebuilds_exact_cross_store_binding(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    assert fixture.resolver.resolve_current(fixture.binding.to_ref()) == fixture.binding
    assert (
        fixture.resolver.capture_current(
            fixture.binding.to_ref(),
            audit=audit(),
        )
        == fixture.binding
    )


def test_resolver_rejects_stale_factory_run(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    run = fixture.factory_source.run
    fixture.factory_source.current_run = FactoryRunV2.create(
        run_id=run.run_id,
        run_version=1,
        status=run.status,
        policy_ref=run.policy_ref,
        requirement_spec_ref=run.requirement_spec_ref,
        current_plan_ref=run.current_plan_ref,
        compiled_plan_ref=run.compiled_plan_ref,
        active_task_refs=run.active_task_refs,
        result_refs=run.result_refs,
        pending_review_ref=run.pending_review_ref,
        planner_assessment_ref=run.planner_assessment_ref,
        completion_ref=run.completion_ref,
        delivery_manifest_ref=run.delivery_manifest_ref,
        transition_count=1,
        model_requests_used=run.model_requests_used,
        model_tokens_used=run.model_tokens_used,
        cost_micro_usd_used=run.cost_micro_usd_used,
        audit=audit(),
    )
    with pytest.raises(FactoryGraphDriftError, match="not current"):
        fixture.resolver.resolve_current(fixture.binding.to_ref())


def test_resolver_rejects_session_without_ready_policy(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    session = fixture.session_source.projection.session.model_copy(
        update={"current_requirement_policy_ref": None},
    )
    fixture.session_source.projection = fixture.session_source.projection.model_copy(
        update={
            "session": session,
            "current_requirement_policy_ref": None,
        },
    )
    with pytest.raises(FactoryGraphDriftError, match="READY"):
        fixture.resolver.resolve_current(fixture.binding.to_ref())


def test_resolver_rejects_missing_current_team_checkpoint(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    fixture.team_source.checkpoint = None
    with pytest.raises(FactoryGraphDriftError, match="checkpoint"):
        fixture.resolver.resolve_current(fixture.binding.to_ref())
