from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from eval_factory.agent_system.graph import FactoryGraphDriftError
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalStore,
)
from eval_factory.blueprints import EvaluationBlueprintV1
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    FactoryRunV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
)
from eval_factory.harness.artifacts import ArtifactHeadV1
from eval_factory.harness.composition import StaticPackRegistry
from eval_factory.harness.contracts import ref_key, sorted_refs
from eval_factory.harness.graph_models import (
    HarnessGraphExecutionBindingV1,
)
from eval_factory.harness.runtime_models import (
    HarnessSessionProjectionV1,
    HarnessSessionStatusV1,
)
from eval_factory.packs.generic_agent_trace.graph_narrowing_rules import (
    generic_agent_authority_is_narrowing_successor,
    generic_agent_graph_is_narrowing_successor,
)
from eval_factory.team import TeamCheckpointV1, TeamSnapshot


class HarnessSessionAuthoritySource(Protocol):
    def get_projection_by_ref(
        self,
        reference: ObjectRef,
    ) -> HarnessSessionProjectionV1: ...


class FactoryControlAuthoritySource(Protocol):
    def get_run_by_ref(self, reference: ObjectRef) -> FactoryRunV2: ...

    def get_run(self, run_id: str) -> FactoryRunV2: ...

    def get_policy_by_ref(
        self,
        reference: ObjectRef,
    ) -> FactoryRunPolicyV2: ...

    def get_requirement_by_ref(
        self,
        reference: ObjectRef,
    ) -> EvaluationRequirementSpecV2: ...

    def get_dataset_request_by_ref(
        self,
        reference: ObjectRef,
    ) -> FactoryDatasetRunRequestV2: ...


class TeamAuthoritySource(Protocol):
    def get_snapshot_at(self, team_ref: ObjectRef) -> TeamSnapshot: ...

    def get_snapshot(self, team_id: str) -> TeamSnapshot: ...

    def current_checkpoint(
        self,
        team_id: str,
    ) -> TeamCheckpointV1 | None: ...

    def current_artifact_heads(
        self,
        team_id: str,
    ) -> tuple[ArtifactHeadV1, ...]: ...


class FactoryGraphAuthorityResolver:
    """Rebuild one graph binding from current cross-store authorities."""

    def __init__(
        self,
        *,
        journal: FactoryGraphJournalStore,
        session_source: HarnessSessionAuthoritySource,
        factory_source: FactoryControlAuthoritySource,
        team_source: TeamAuthoritySource,
        pack_registry: StaticPackRegistry,
        blueprints: Iterable[EvaluationBlueprintV1],
    ) -> None:
        self._journal = journal
        self._session_source = session_source
        self._factory_source = factory_source
        self._team_source = team_source
        self._pack_registry = pack_registry
        self._blueprints: dict[
            tuple[str, str, str, str],
            EvaluationBlueprintV1,
        ] = {}
        for blueprint in blueprints:
            key = ref_key(blueprint.to_ref())
            if key in self._blueprints:
                raise ValueError(
                    "duplicate Graph Blueprint authority",
                )
            self._blueprints[key] = blueprint
        if not self._blueprints:
            raise ValueError(
                "Graph authority resolver requires a Blueprint",
            )

    def resolve_current(
        self,
        reference: ObjectRef,
    ) -> HarnessGraphExecutionBindingV1:
        binding = self._journal.get_binding_by_ref(reference)
        current_binding = self._journal.get_binding(binding.binding_id)
        self._require(
            current_binding.to_ref() == reference,
            "graph execution binding is not current",
        )

        session = self._session_source.get_projection_by_ref(
            binding.session_ref,
        )
        self._validate_session(binding, session)

        run = self._factory_source.get_run_by_ref(
            binding.factory_run_ref,
        )
        current_run = self._factory_source.get_run(run.run_id)
        policy = self._factory_source.get_policy_by_ref(
            binding.factory_policy_ref,
        )
        requirement = self._factory_source.get_requirement_by_ref(
            binding.requirement_ref,
        )
        request = self._factory_source.get_dataset_request_by_ref(
            binding.factory_request_ref,
        )
        self._validate_factory(
            binding,
            run=run,
            current_run=current_run,
            policy=policy,
            requirement=requirement,
            request=request,
        )

        registration = self._pack_registry.resolve_manifest(
            binding.pack_manifest_ref,
        )
        self._require(
            registration.composition.to_ref() == binding.composition_ref,
            "Pack composition differs from graph binding authority",
        )
        blueprint = self._blueprints.get(ref_key(binding.blueprint_ref))
        self._require(
            blueprint is not None,
            "graph Blueprint authority was not registered",
        )
        assert blueprint is not None

        snapshot = self._team_source.get_snapshot_at(binding.team_ref)
        current_snapshot = self._team_source.get_snapshot(
            snapshot.team.team_id,
        )
        checkpoint = self._team_source.current_checkpoint(
            snapshot.team.team_id,
        )
        heads = self._team_source.current_artifact_heads(
            snapshot.team.team_id,
        )
        self._validate_team(
            binding,
            snapshot=snapshot,
            current_snapshot=current_snapshot,
            checkpoint=checkpoint,
            blackboard_head_refs=sorted_refs(head.to_ref() for head in heads),
        )
        self._validate_blueprint(
            binding,
            blueprint=blueprint,
            source_snapshot=self._team_source.get_snapshot_at(
                blueprint.team_ref,
            ),
            current_snapshot=current_snapshot,
        )
        return binding

    def capture_current(
        self,
        reference: ObjectRef,
        *,
        audit: ContractAudit,
    ) -> HarnessGraphExecutionBindingV1:
        prior = self._journal.get_binding_by_ref(reference)
        current_binding = self._journal.get_binding(prior.binding_id)
        self._require(
            current_binding.to_ref() == reference,
            "graph execution binding is not current",
        )
        session = self._session_source.get_projection_by_ref(
            prior.session_ref,
        )
        self._validate_session(prior, session)

        prior_run = self._factory_source.get_run_by_ref(
            prior.factory_run_ref,
        )
        run = self._factory_source.get_run(prior_run.run_id)
        policy = self._factory_source.get_policy_by_ref(
            prior.factory_policy_ref,
        )
        requirement = self._factory_source.get_requirement_by_ref(
            prior.requirement_ref,
        )
        request = self._factory_source.get_dataset_request_by_ref(
            prior.factory_request_ref,
        )

        registration = self._pack_registry.resolve_manifest(
            prior.pack_manifest_ref,
        )
        self._require(
            registration.composition.to_ref() == prior.composition_ref,
            "Pack composition differs from graph binding authority",
        )
        blueprint = self._blueprints.get(ref_key(prior.blueprint_ref))
        self._require(
            blueprint is not None,
            "graph Blueprint authority was not registered",
        )
        assert blueprint is not None

        prior_snapshot = self._team_source.get_snapshot_at(
            prior.team_ref,
        )
        snapshot = self._team_source.get_snapshot(
            prior_snapshot.team.team_id,
        )
        checkpoint = self._team_source.current_checkpoint(
            snapshot.team.team_id,
        )
        self._require(
            checkpoint is not None,
            "Team checkpoint authority is missing",
        )
        assert checkpoint is not None
        blackboard_head_refs = sorted_refs(
            head.to_ref()
            for head in self._team_source.current_artifact_heads(
                snapshot.team.team_id,
            )
        )
        candidate = HarnessGraphExecutionBindingV1.create(
            binding_id=prior.binding_id,
            session_ref=prior.session_ref,
            requirement_ref=prior.requirement_ref,
            requirement_policy_ref=prior.requirement_policy_ref,
            factory_run_ref=run.to_ref(),
            factory_request_ref=prior.factory_request_ref,
            factory_policy_ref=prior.factory_policy_ref,
            expected_factory_run_version=run.run_version,
            pack_manifest_ref=prior.pack_manifest_ref,
            composition_ref=prior.composition_ref,
            blueprint_ref=prior.blueprint_ref,
            team_ref=snapshot.team.to_ref(),
            roster_ref=snapshot.roster.to_ref(),
            task_graph_ref=snapshot.graph.to_ref(),
            execution_authority_ref=snapshot.authority.to_ref(),
            team_checkpoint_ref=checkpoint.to_ref(),
            expected_team_version=snapshot.team.team_version,
            expected_task_graph_revision=snapshot.graph.revision,
            expected_authority_version=(snapshot.authority.authority_version),
            blackboard_head_refs=blackboard_head_refs,
            thread_id=prior.thread_id,
            max_transitions=prior.max_transitions,
            audit=audit,
        )
        self._validate_factory(
            candidate,
            run=run,
            current_run=run,
            policy=policy,
            requirement=requirement,
            request=request,
        )
        self._validate_team(
            candidate,
            snapshot=snapshot,
            current_snapshot=snapshot,
            checkpoint=checkpoint,
            blackboard_head_refs=blackboard_head_refs,
        )
        self._validate_blueprint(
            candidate,
            blueprint=blueprint,
            source_snapshot=self._team_source.get_snapshot_at(
                blueprint.team_ref,
            ),
            current_snapshot=snapshot,
        )
        return candidate

    @staticmethod
    def _validate_session(
        binding: HarnessGraphExecutionBindingV1,
        session: HarnessSessionProjectionV1,
    ) -> None:
        state = session.session
        FactoryGraphAuthorityResolver._require(
            state.session_ref == binding.session_ref,
            "Harness session differs from graph binding authority",
        )
        FactoryGraphAuthorityResolver._require(
            state.status is HarnessSessionStatusV1.ACTIVE,
            "Harness session is not active",
        )
        FactoryGraphAuthorityResolver._require(
            state.composition_ref == binding.composition_ref,
            "Harness session composition drifted",
        )
        FactoryGraphAuthorityResolver._require(
            state.current_interpretation_ref is not None
            and state.current_requirement_policy_ref == binding.requirement_policy_ref,
            "Harness session has no current READY requirement policy",
        )

    @staticmethod
    def _validate_factory(
        binding: HarnessGraphExecutionBindingV1,
        *,
        run: FactoryRunV2,
        current_run: FactoryRunV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        request: FactoryDatasetRunRequestV2,
    ) -> None:
        FactoryGraphAuthorityResolver._require(
            current_run.to_ref() == binding.factory_run_ref and run.to_ref() == binding.factory_run_ref,
            "Factory run is not current",
        )
        FactoryGraphAuthorityResolver._require(
            run.run_version == binding.expected_factory_run_version,
            "Factory run version differs from graph binding authority",
        )
        FactoryGraphAuthorityResolver._require(
            policy.to_ref() == binding.factory_policy_ref and run.policy_ref == binding.factory_policy_ref,
            "Factory policy differs from graph binding authority",
        )
        FactoryGraphAuthorityResolver._require(
            requirement.to_ref() == binding.requirement_ref
            and run.requirement_spec_ref == binding.requirement_ref,
            "Factory requirement differs from graph binding authority",
        )
        FactoryGraphAuthorityResolver._require(
            request.to_ref() == binding.factory_request_ref
            and request.dataset_run_id == run.run_id
            and request.requirement_spec_ref == binding.requirement_ref
            and request.factory_policy_ref == binding.factory_policy_ref,
            "Factory dataset request differs from graph binding authority",
        )
        FactoryGraphAuthorityResolver._require(
            binding.max_transitions <= min(request.max_transitions, policy.max_transitions),
            "Graph transition budget exceeds source authority",
        )

    @staticmethod
    def _validate_team(
        binding: HarnessGraphExecutionBindingV1,
        *,
        snapshot: TeamSnapshot,
        current_snapshot: TeamSnapshot,
        checkpoint: TeamCheckpointV1 | None,
        blackboard_head_refs: tuple[ObjectRef, ...],
    ) -> None:
        FactoryGraphAuthorityResolver._require(
            snapshot.team.to_ref() == binding.team_ref and current_snapshot.team.to_ref() == binding.team_ref,
            "Team authority is not current",
        )
        FactoryGraphAuthorityResolver._require(
            snapshot.roster.to_ref() == binding.roster_ref
            and snapshot.graph.to_ref() == binding.task_graph_ref
            and snapshot.authority.to_ref() == binding.execution_authority_ref,
            "Team components differ from graph binding authority",
        )
        FactoryGraphAuthorityResolver._require(
            snapshot.team.roster_ref == binding.roster_ref
            and snapshot.team.task_graph_ref == binding.task_graph_ref
            and snapshot.team.authority_ref == binding.execution_authority_ref
            and snapshot.team.composition_ref == binding.composition_ref
            and snapshot.team.goal_ref == binding.requirement_ref,
            "Team binding fields drifted",
        )
        FactoryGraphAuthorityResolver._require(
            snapshot.team.team_version == binding.expected_team_version
            and snapshot.graph.revision == binding.expected_task_graph_revision
            and snapshot.authority.authority_version == binding.expected_authority_version,
            "Team versions differ from graph binding authority",
        )
        FactoryGraphAuthorityResolver._require(
            checkpoint is not None and checkpoint.to_ref() == binding.team_checkpoint_ref,
            "Team checkpoint differs from graph binding authority",
        )
        assert checkpoint is not None
        FactoryGraphAuthorityResolver._require(
            checkpoint.team_ref == binding.team_ref
            and checkpoint.roster_ref == binding.roster_ref
            and checkpoint.task_graph_ref == binding.task_graph_ref
            and checkpoint.authority_ref == binding.execution_authority_ref
            and checkpoint.artifact_head_refs == binding.blackboard_head_refs,
            "Team checkpoint source authority drifted",
        )
        FactoryGraphAuthorityResolver._require(
            blackboard_head_refs == binding.blackboard_head_refs,
            "Blackboard heads differ from graph binding authority",
        )

    @staticmethod
    def _validate_blueprint(
        binding: HarnessGraphExecutionBindingV1,
        *,
        blueprint: EvaluationBlueprintV1,
        source_snapshot: TeamSnapshot,
        current_snapshot: TeamSnapshot,
    ) -> None:
        FactoryGraphAuthorityResolver._require(
            blueprint.to_ref() == binding.blueprint_ref,
            "Blueprint differs from graph binding authority",
        )
        FactoryGraphAuthorityResolver._require(
            blueprint.requirement_ref == binding.requirement_ref
            and blueprint.composition_ref == binding.composition_ref
            and blueprint.pack_manifest_refs == (binding.pack_manifest_ref,)
            and source_snapshot.team.to_ref() == blueprint.team_ref
            and source_snapshot.roster.to_ref() == blueprint.roster_ref
            and source_snapshot.graph.to_ref() == blueprint.task_graph_ref
            and source_snapshot.authority.to_ref() == blueprint.authority_ref
            and current_snapshot.team.to_ref() == binding.team_ref
            and current_snapshot.roster.to_ref() == binding.roster_ref
            and current_snapshot.graph.to_ref() == binding.task_graph_ref
            and current_snapshot.authority.to_ref() == binding.execution_authority_ref
            and current_snapshot.team.team_id == source_snapshot.team.team_id
            and current_snapshot.team.team_incarnation_id == source_snapshot.team.team_incarnation_id
            and current_snapshot.team.team_version >= source_snapshot.team.team_version
            and current_snapshot.team.goal_ref == source_snapshot.team.goal_ref
            and current_snapshot.team.composition_ref == source_snapshot.team.composition_ref
            and current_snapshot.roster.to_ref() == source_snapshot.roster.to_ref()
            and generic_agent_graph_is_narrowing_successor(
                source=source_snapshot.graph,
                current=current_snapshot.graph,
            )
            and generic_agent_authority_is_narrowing_successor(
                source=source_snapshot.authority,
                current=current_snapshot.authority,
            ),
            "Blueprint source bindings drifted",
        )

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise FactoryGraphDriftError(message)


__all__ = [
    "FactoryControlAuthoritySource",
    "FactoryGraphAuthorityResolver",
    "HarnessSessionAuthoritySource",
    "TeamAuthoritySource",
]
