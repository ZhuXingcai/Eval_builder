from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eval_factory.agent_system.candidate_output import (
    CandidateDatasetOutputAssembler,
)
from eval_factory.agent_system.dataset_graph_commands import (
    FactoryDatasetGraphNodeCommands,
)
from eval_factory.agent_system.dataset_graph_terminal import (
    FactoryDatasetWorkflowTerminalCommands,
)
from eval_factory.agent_system.dataset_runtime import FactoryDatasetRuntime
from eval_factory.agent_system.graph_authority import (
    FactoryGraphAuthorityResolver,
)
from eval_factory.agent_system.graph_continuation import (
    FactoryGraphContinuationService,
)
from eval_factory.agent_system.graph_journal import FactoryGraphJournalStore
from eval_factory.agent_system.graph_projection import (
    FactoryGraphSessionReconciler,
)
from eval_factory.agent_system.graph_runtime import (
    FactoryDatasetGraphRuntime,
)
from eval_factory.agent_system.graph_transition import (
    FactoryGraphTransitionService,
)
from eval_factory.agent_system.planner_assessment_source import (
    FactoryPlannerAssessmentSource,
)
from eval_factory.agent_system.store import FactoryControlNotFoundError
from eval_factory.blueprints import EvaluationBlueprintV1
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
)
from eval_factory.harness.composition import (
    StaticPackRegistrationV1,
    StaticPackRegistry,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.harness.graph_models import (
    HarnessGraphExecutionBindingV1,
)
from eval_factory.harness.session_store import HarnessSessionStore
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflow,
)
from eval_factory.team import TeamStore


@dataclass(frozen=True, slots=True)
class FactoryDatasetGraphBootstrapResult:
    runtime: FactoryDatasetGraphRuntime
    binding: HarnessGraphExecutionBindingV1
    journal: FactoryGraphJournalStore


class FactoryDatasetGraphBootstrap:
    """Composes the default Graph facade from explicit owner authorities."""

    @staticmethod
    def create_binding(
        *,
        binding_id: str,
        thread_id: str,
        session_ref: ObjectRef,
        requirement_policy_ref: ObjectRef,
        dataset_runtime: FactoryDatasetRuntime,
        team_store: TeamStore,
        registration: StaticPackRegistrationV1,
        blueprint: EvaluationBlueprintV1,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        audit: ContractAudit,
    ) -> HarnessGraphExecutionBindingV1:
        try:
            stored_request = dataset_runtime.store.get_dataset_request(
                request.dataset_run_id,
            )
        except FactoryControlNotFoundError:
            dataset_runtime.admit_request(
                request=request,
                policy=policy,
                requirement=requirement,
                audit=audit,
            )
        else:
            if stored_request != request:
                raise ValueError(
                    "Factory Graph binding request differs from stored authority",
                )
        run = dataset_runtime.store.get_run(request.dataset_run_id)
        source_snapshot = team_store.get_snapshot_at(
            blueprint.team_ref,
        )
        snapshot = team_store.get_snapshot(
            source_snapshot.team.team_id,
        )
        checkpoint = team_store.current_checkpoint(
            snapshot.team.team_id,
        )
        if checkpoint is None:
            raise ValueError(
                "Factory Graph binding requires a current Team checkpoint",
            )
        if (
            blueprint.requirement_ref != requirement.to_ref()
            or blueprint.composition_ref != registration.composition.to_ref()
            or blueprint.pack_manifest_refs != (registration.manifest.to_ref(),)
            or blueprint.roster_ref != source_snapshot.roster.to_ref()
            or blueprint.task_graph_ref != source_snapshot.graph.to_ref()
            or blueprint.authority_ref != source_snapshot.authority.to_ref()
        ):
            raise ValueError(
                "Factory Graph binding inputs differ from current workflow authority",
            )
        return HarnessGraphExecutionBindingV1.create(
            binding_id=binding_id,
            session_ref=session_ref,
            requirement_ref=requirement.to_ref(),
            requirement_policy_ref=requirement_policy_ref,
            factory_run_ref=run.to_ref(),
            factory_request_ref=request.to_ref(),
            factory_policy_ref=policy.to_ref(),
            expected_factory_run_version=run.run_version,
            pack_manifest_ref=registration.manifest.to_ref(),
            composition_ref=registration.composition.to_ref(),
            blueprint_ref=blueprint.to_ref(),
            team_ref=snapshot.team.to_ref(),
            roster_ref=snapshot.roster.to_ref(),
            task_graph_ref=snapshot.graph.to_ref(),
            execution_authority_ref=snapshot.authority.to_ref(),
            team_checkpoint_ref=checkpoint.to_ref(),
            expected_team_version=snapshot.team.team_version,
            expected_task_graph_revision=snapshot.graph.revision,
            expected_authority_version=(snapshot.authority.authority_version),
            blackboard_head_refs=sorted_refs(
                value.to_ref()
                for value in team_store.current_artifact_heads(
                    snapshot.team.team_id,
                )
            ),
            thread_id=thread_id,
            max_transitions=request.max_transitions,
            audit=audit,
        )

    @staticmethod
    def build(
        *,
        dataset_runtime: FactoryDatasetRuntime,
        workflow: GenericAgentFactoryWorkflow,
        output_reader: CandidateDatasetOutputAssembler,
        session_store: HarnessSessionStore,
        team_store: TeamStore,
        registration: StaticPackRegistrationV1,
        blueprint: EvaluationBlueprintV1,
        binding: HarnessGraphExecutionBindingV1,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        journal_path: Path,
        checkpoint_path: Path,
        audit: ContractAudit,
    ) -> FactoryDatasetGraphBootstrapResult:
        roots = tuple(
            path.expanduser().resolve()
            for path in (
                dataset_runtime.store.path,
                session_store.path,
                team_store.path,
                journal_path,
                checkpoint_path,
            )
        )
        if len(set(roots)) != len(roots):
            raise ValueError(
                "Factory, Harness, Team, journal, and checkpoint paths must be distinct",
            )
        if (
            workflow.store is not team_store
            or workflow.blueprint != blueprint
            or workflow.runner.registry.registration != registration
            or output_reader.store is not dataset_runtime.store
            or request.to_ref() != binding.factory_request_ref
            or policy.to_ref() != binding.factory_policy_ref
            or requirement.to_ref() != binding.requirement_ref
            or registration.manifest.to_ref() != binding.pack_manifest_ref
            or registration.composition.to_ref() != binding.composition_ref
            or blueprint.to_ref() != binding.blueprint_ref
        ):
            raise ValueError(
                "Factory Graph bootstrap authority is inconsistent",
            )
        journal = FactoryGraphJournalStore(journal_path)
        authority = FactoryGraphAuthorityResolver(
            journal=journal,
            session_source=session_store,
            factory_source=dataset_runtime.store,
            team_source=team_store,
            pack_registry=StaticPackRegistry((registration,)),
            blueprints=(blueprint,),
        )
        transition = FactoryGraphTransitionService(
            journal=journal,
            authority=authority,
            audit=audit,
        )
        continuation = FactoryGraphContinuationService(
            journal=journal,
            authority=authority,
            factory_source=dataset_runtime.store,
            review_source=dataset_runtime.plan_reviews,
            audit=audit,
        )
        session_reconciler = FactoryGraphSessionReconciler(
            journal=journal,
            session_sink=session_store,
        )
        team_id = team_store.get_snapshot_at(
            binding.team_ref,
        ).team.team_id
        terminal = FactoryDatasetWorkflowTerminalCommands(
            runtime=dataset_runtime,
            workflow=workflow,
            output_reader=output_reader,
            team_id=team_id,
            request=request,
            audit=audit,
        )
        commands = FactoryDatasetGraphNodeCommands(
            runtime=dataset_runtime,
            workflow=workflow,
            terminal=terminal,
            team_id=team_id,
            request=request,
            policy=policy,
            requirement=requirement,
            audit=audit,
        )
        assessment = FactoryPlannerAssessmentSource(
            factory_store=dataset_runtime.store,
            team_store=team_store,
            team_id=team_id,
            audit=audit,
        )
        runtime = FactoryDatasetGraphRuntime(
            store=dataset_runtime.store,
            journal=journal,
            authority=authority,
            transition_service=transition,
            continuation_service=continuation,
            session_reconciler=session_reconciler,
            commands=commands,
            request=request,
            policy=policy,
            requirement=requirement,
            assessment_factory=assessment,
            checkpoint_path=checkpoint_path,
            audit=audit,
        )
        return FactoryDatasetGraphBootstrapResult(
            runtime=runtime,
            binding=binding,
            journal=journal,
        )


__all__ = [
    "FactoryDatasetGraphBootstrap",
    "FactoryDatasetGraphBootstrapResult",
]
