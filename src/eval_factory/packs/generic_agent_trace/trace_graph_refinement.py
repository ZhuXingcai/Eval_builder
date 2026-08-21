from __future__ import annotations

import hashlib

from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
)
from eval_factory.agent_system.trace_candidate_store import (
    FactoryTraceCandidateMaterialStore,
)
from eval_factory.contracts.agent_system_v2 import (
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.core import ContractAudit
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflowError,
)
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterialization,
)
from eval_factory.packs.generic_agent_trace.task_graph_narrowing import (
    GenericAgentTaskGraphNarrower,
    GenericAgentTaskGraphNarrowing,
)
from eval_factory.team import TeamSnapshot, TeamStore, TeamTaskStatusV1


class GenericAgentTraceGraphRefinement:
    """Commits one graph narrowing after all Trace dispositions exist."""

    def __init__(
        self,
        *,
        dataset_run_id: str,
        store: TeamStore,
        candidate_store: FactoryTraceCandidateMaterialStore,
        materialization: GenericAgentTaskGraphMaterialization,
        narrower: GenericAgentTaskGraphNarrower | None = None,
    ) -> None:
        self.dataset_run_id = dataset_run_id
        self.store = store
        self.candidate_store = candidate_store
        self.materialization = materialization
        self.narrower = narrower or GenericAgentTaskGraphNarrower()

    def refine(
        self,
        team_id: str,
        *,
        audit: ContractAudit,
    ) -> GenericAgentTaskGraphNarrowing | None:
        snapshot = self.store.get_snapshot(team_id)
        trace_instances = tuple(
            instance
            for instance in self.materialization.task_instances
            if instance.capability_id == "capability.trace-ingestion"
        )
        work = tuple(self.store.get_task_work(team_id, instance.task_id) for instance in trace_instances)
        if any(value.status is not TeamTaskStatusV1.COMPLETED for value in work):
            return None
        materials = []
        for instance in trace_instances:
            if instance.scope_ref is None:
                raise GenericAgentFactoryWorkflowError(
                    "Trace task instance has no source authority",
                )
            try:
                material = self.candidate_store.get(
                    run_id=self.dataset_run_id,
                    trace_source_ref=instance.scope_ref,
                )
            except FactoryControlNotFoundError as exc:
                raise GenericAgentFactoryWorkflowError(
                    "completed Trace task lacks candidate authority",
                ) from exc
            materials.append(material)
        candidate_refs = sorted_refs(
            material.authority.trace_source_ref
            for material in materials
            if (material.preparation.decision.disposition is TraceCandidateDispositionV2.CANDIDATE)
        )
        narrowing = self.narrower.narrow(
            snapshot=snapshot,
            materialization=self.materialization,
            candidate_source_refs=candidate_refs,
            audit=audit,
        )
        if not narrowing.changed:
            self._ensure_current_checkpoint(
                team_id,
                snapshot=narrowing.snapshot,
                audit=audit,
            )
            return narrowing
        digest = hashlib.sha256(
            "|".join(
                reference.object_sha256
                for reference in sorted_refs(material.authority.to_ref() for material in materials)
            ).encode()
        ).hexdigest()
        committed = self.store.promote_narrowed_graph(
            team_id,
            expected_team_ref=snapshot.team.to_ref(),
            expected_graph_ref=snapshot.graph.to_ref(),
            expected_authority_ref=snapshot.authority.to_ref(),
            successor=narrowing.snapshot,
            idempotency_key=(f"trace-disposition-narrowing.{digest}"),
        )
        self._ensure_current_checkpoint(
            team_id,
            snapshot=committed,
            audit=audit,
        )
        return GenericAgentTaskGraphNarrowing(
            snapshot=committed,
            task_instances=narrowing.task_instances,
            candidate_source_refs=(narrowing.candidate_source_refs),
            removed_task_ids=narrowing.removed_task_ids,
            changed=True,
        )

    def _ensure_current_checkpoint(
        self,
        team_id: str,
        *,
        snapshot: TeamSnapshot,
        audit: ContractAudit,
    ) -> None:
        artifact_head_refs = sorted_refs(head.to_ref() for head in self.store.current_artifact_heads(team_id))
        checkpoint = self.store.current_checkpoint(team_id)
        if (
            checkpoint is not None
            and checkpoint.team_ref == snapshot.team.to_ref()
            and checkpoint.roster_ref == snapshot.roster.to_ref()
            and checkpoint.task_graph_ref == snapshot.graph.to_ref()
            and checkpoint.authority_ref == snapshot.authority.to_ref()
            and checkpoint.artifact_head_refs == artifact_head_refs
        ):
            return
        digest = hashlib.sha256(
            "|".join(
                reference.object_sha256
                for reference in sorted_refs(
                    (
                        snapshot.team.to_ref(),
                        snapshot.roster.to_ref(),
                        snapshot.graph.to_ref(),
                        snapshot.authority.to_ref(),
                        *artifact_head_refs,
                    ),
                )
            ).encode()
        ).hexdigest()
        self.store.create_checkpoint(
            team_id,
            audit=audit,
            idempotency_key=(f"trace-disposition-narrowing-checkpoint.{digest}"),
        )


__all__ = ["GenericAgentTraceGraphRefinement"]
