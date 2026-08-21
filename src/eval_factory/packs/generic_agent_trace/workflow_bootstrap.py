from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.agent_system.trace_candidate_store import (
    FactoryTraceCandidateMaterialStore,
)
from eval_factory.blueprints import EvaluationBlueprintV1
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness import (
    ArtifactEnvelopeV1,
    ArtifactModalityV1,
    HarnessCapabilityRuntime,
    StaticPackRegistrationV1,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentCapabilityRequestPreparation,
    GenericAgentFactoryWorkflow,
    GenericAgentFactoryWorkflowError,
    GenericAgentTaskGraphRefinement,
)
from eval_factory.packs.generic_agent_trace.manifest import (
    GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION,
    build_generic_agent_evaluation_blueprint,
)
from eval_factory.packs.generic_agent_trace.material_resolver import (
    CompositeCapabilityMaterialResolver,
)
from eval_factory.packs.generic_agent_trace.request_factories import (
    GenericAgentCapabilityPreparationService,
)
from eval_factory.packs.generic_agent_trace.request_preparation import (
    GenericAgentTaskRequestPreparer,
)
from eval_factory.packs.generic_agent_trace.request_store import (
    GenericAgentCapabilityRequestStore,
    StoredGenericAgentCapabilityRequestSource,
)
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterialization,
    GenericAgentTaskGraphMaterializer,
)
from eval_factory.packs.generic_agent_trace.trace_graph_refinement import (
    GenericAgentTraceGraphRefinement,
)
from eval_factory.team import (
    TeamCapabilityRunner,
    TeamCheckpointV1,
    TeamSnapshot,
    TeamStore,
    TeamTaskV1,
)


@dataclass(frozen=True, slots=True)
class GenericAgentWorkflowBootstrapResult:
    materialization: GenericAgentTaskGraphMaterialization
    snapshot: TeamSnapshot
    checkpoint: TeamCheckpointV1
    blueprint: EvaluationBlueprintV1
    workflow: GenericAgentFactoryWorkflow
    request_store: GenericAgentCapabilityRequestStore
    materials: CompositeCapabilityMaterialResolver
    preparation: GenericAgentCapabilityPreparationService


class GenericAgentWorkflowBootstrap:
    """Builds the current multi-instance Pack workflow from exact authorities."""

    @staticmethod
    def build(
        *,
        registration: StaticPackRegistrationV1,
        capability_runtime: HarnessCapabilityRuntime,
        materials: CompositeCapabilityMaterialResolver,
        private_store: FactoryPrivateObjectStore,
        request_store_path: Path,
        team_store: TeamStore,
        requirement: EvaluationRequirementSpecV2,
        source_refs: tuple[ObjectRef, ...],
        independent_session_refs: dict[str, ObjectRef],
        team_id: str,
        team_incarnation_id: str,
        max_model_requests: int,
        max_model_tokens: int,
        max_cost_micro_usd: int,
        audit: ContractAudit,
        request_preparation_factory: (
            Callable[
                [
                    GenericAgentTaskGraphMaterialization,
                    GenericAgentCapabilityPreparationService,
                ],
                GenericAgentCapabilityRequestPreparation,
            ]
            | None
        ) = None,
        graph_refinement_factory: (
            Callable[
                [GenericAgentTaskGraphMaterialization],
                GenericAgentTaskGraphRefinement,
            ]
            | None
        ) = None,
        trace_candidate_store: FactoryTraceCandidateMaterialStore | None = None,
    ) -> GenericAgentWorkflowBootstrapResult:
        if capability_runtime.registry.registration != registration:
            raise ValueError(
                "workflow bootstrap Runtime registration differs from Pack",
            )
        if request_store_path.expanduser().resolve() == team_store.path:
            raise ValueError(
                "Team and capability request stores must be distinct",
            )
        for provider in capability_runtime.registry.providers:
            provider_materials = getattr(provider, "materials", materials)
            if provider_materials is not materials:
                raise ValueError(
                    "workflow provider uses another material resolver",
                )
        execution_mode = registration.manifest.pack_version == GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION
        if graph_refinement_factory is not None and trace_candidate_store is not None:
            raise ValueError(
                "workflow graph refinement accepts one authority source",
            )
        if execution_mode and (graph_refinement_factory is None and trace_candidate_store is None):
            raise ValueError(
                "execution workflow requires Trace candidate graph refinement",
            )
        if not execution_mode and (graph_refinement_factory is not None or trace_candidate_store is not None):
            raise ValueError(
                "replay workflow cannot install execution graph refinement",
            )
        materialization = GenericAgentTaskGraphMaterializer().materialize(
            registration=registration,
            requirement_ref=requirement.to_ref(),
            source_refs=source_refs,
            independent_session_refs=independent_session_refs,
            team_id=team_id,
            team_incarnation_id=team_incarnation_id,
            max_model_requests=max_model_requests,
            max_model_tokens=max_model_tokens,
            max_cost_micro_usd=max_cost_micro_usd,
            audit=audit,
        )
        authority = materialization.authority
        team_store.create_team(
            team=authority.team,
            roster=authority.roster,
            graph=authority.graph,
            authority=authority.authority,
            idempotency_key=f"bootstrap-team:{team_id}",
        )
        root_task = tuple(
            task
            for task in authority.graph.tasks
            if materialization.instance(task.task_id).capability_id == "capability.requirement-planning"
        )
        if len(root_task) != 1 or len(root_task[0].input_artifact_head_ids) != 1:
            raise GenericAgentFactoryWorkflowError(
                "workflow requirement input is not uniquely materialized",
            )
        definition = next(
            value
            for value in registration.capability_definitions
            if value.capability_id == "capability.requirement-planning"
        )
        envelope = ArtifactEnvelopeV1.create(
            artifact_id=f"{team_id}.input-evaluation-requirement",
            subject_ref=requirement.to_ref(),
            schema_ref=definition.request_schema_ref,
            content_ref=requirement.to_ref(),
            media_type="application/json",
            modality=ArtifactModalityV1.DOCUMENT,
            domain_tags=("generic-agent-trace",),
            semantic_role="evaluation-requirement",
            purpose="evaluation-data-production",
            classification="INTERNAL",
            lineage_refs=(),
            producer_capability_ref=definition.to_ref(),
            producer_task_ref=None,
            validation_refs=(),
            revision=1,
            predecessor_envelope_ref=None,
            audit=audit,
        )
        team_store.seed_artifact_head(
            team_ref=authority.team.to_ref(),
            head_id=root_task[0].input_artifact_head_ids[0],
            envelope=envelope,
            audit=audit,
            idempotency_key=f"bootstrap-requirement:{team_id}",
        )
        snapshot = team_store.get_snapshot(team_id)
        heads = tuple(
            value.to_ref()
            for value in team_store.current_artifact_heads(
                team_id,
            )
        )
        checkpoint = team_store.current_checkpoint(team_id)
        if (
            checkpoint is None
            or checkpoint.team_ref != snapshot.team.to_ref()
            or checkpoint.roster_ref != snapshot.roster.to_ref()
            or checkpoint.task_graph_ref != snapshot.graph.to_ref()
            or checkpoint.authority_ref != snapshot.authority.to_ref()
            or checkpoint.artifact_head_refs != sorted_refs(heads)
        ):
            checkpoint = team_store.create_checkpoint(
                team_id,
                audit=audit,
                idempotency_key=(
                    "bootstrap-checkpoint:"
                    f"{snapshot.team.object_sha256}:"
                    f"{snapshot.authority.object_sha256}:"
                    f"{hashlib.sha256('|'.join(value.object_sha256 for value in sorted_refs(heads)).encode()).hexdigest()}"
                ),
            )
        blueprint = build_generic_agent_evaluation_blueprint(
            registration=registration,
            requirement_ref=requirement.to_ref(),
            team_ref=authority.team.to_ref(),
            roster_ref=authority.roster.to_ref(),
            task_graph_ref=authority.graph.to_ref(),
            authority_ref=authority.authority.to_ref(),
            audit=audit,
        )
        request_store = GenericAgentCapabilityRequestStore(
            request_store_path,
            private_store=private_store,
        )

        def current_task(task_id: str) -> TeamTaskV1:
            matches = tuple(
                task for task in team_store.get_snapshot(team_id).graph.tasks if task.task_id == task_id
            )
            if len(matches) != 1:
                raise GenericAgentFactoryWorkflowError(
                    "current Team task is not uniquely available",
                )
            return matches[0]

        preparer = GenericAgentTaskRequestPreparer(
            materialization=materialization,
            registry=capability_runtime.registry,
            store=request_store,
            current_task_resolver=current_task,
        )
        preparation = GenericAgentCapabilityPreparationService(
            preparer=preparer,
            materials=materials,
        )
        request_preparation = (
            request_preparation_factory(
                materialization,
                preparation,
            )
            if request_preparation_factory is not None
            else None
        )
        graph_refinement = (
            graph_refinement_factory(materialization)
            if graph_refinement_factory is not None
            else (
                GenericAgentTraceGraphRefinement(
                    dataset_run_id=requirement.run_id,
                    store=team_store,
                    candidate_store=trace_candidate_store,
                    materialization=materialization,
                )
                if trace_candidate_store is not None
                else None
            )
        )
        workflow = GenericAgentFactoryWorkflow(
            blueprint=blueprint,
            store=team_store,
            runner=TeamCapabilityRunner(
                store=team_store,
                runtime=capability_runtime,
                registry=capability_runtime.registry,
            ),
            request_source=StoredGenericAgentCapabilityRequestSource(
                store=request_store,
                registry=capability_runtime.registry,
            ),
            request_preparation=request_preparation,
            graph_refinement=graph_refinement,
        )
        workflow.validate_team(team_id)
        return GenericAgentWorkflowBootstrapResult(
            materialization=materialization,
            snapshot=snapshot,
            checkpoint=checkpoint,
            blueprint=blueprint,
            workflow=workflow,
            request_store=request_store,
            materials=materials,
            preparation=preparation,
        )


__all__ = [
    "GenericAgentWorkflowBootstrap",
    "GenericAgentWorkflowBootstrapResult",
]
