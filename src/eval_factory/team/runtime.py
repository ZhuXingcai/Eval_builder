from __future__ import annotations

import hashlib
from dataclasses import replace

from pydantic import BaseModel

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.artifacts import ArtifactEnvelopeV1
from eval_factory.harness.capability import (
    CapabilityConsumerKindV1,
    CapabilityIdempotencyV1,
    CapabilityInvocationContextV1,
    CapabilityInvocationOutcomeV1,
    CapabilityProviderKindV1,
    CapabilityResultV1,
)
from eval_factory.harness.capability_runtime import (
    CapabilityPermissionScopeV1,
    CapabilityRuntimeInputError,
    CapabilityRuntimeInvocation,
    CapabilityRuntimeRegistry,
    HarnessCapabilityRuntime,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.harness.interaction import GraphMutationKindV1
from eval_factory.team.models import TeamTaskStatusV1
from eval_factory.team.store import (
    TeamConcurrencyError,
    TeamStore,
    TeamTaskCompletion,
    TeamTaskStateError,
)


class TeamCapabilityRunner:
    """Runs one current Team task through the shared Stage 2 Runtime."""

    def __init__(
        self,
        *,
        store: TeamStore,
        runtime: HarnessCapabilityRuntime,
        registry: CapabilityRuntimeRegistry,
    ) -> None:
        if runtime.registry is not registry:
            raise ValueError("Team runner Runtime and registry must be identical")
        self.store = store
        self.runtime = runtime
        self.registry = registry

    async def execute(
        self,
        *,
        team_id: str,
        task_id: str,
        request: BaseModel,
        audit: ContractAudit,
        lease_duration_seconds: int = 300,
        deterministic: bool = True,
        model_requests_delta: int = 0,
        model_tokens_delta: int = 0,
        cost_micro_usd_delta: int = 0,
    ) -> TeamTaskCompletion:
        snapshot = self.store.get_snapshot(team_id)
        if snapshot.team.composition_ref != self.registry.registration.composition.to_ref():
            raise CapabilityRuntimeInputError(
                "Team composition differs from the installed static Pack",
            )
        work = self.store.get_task_work(team_id, task_id)
        task = work.task
        if task.assigned_member_id is None:
            raise TeamTaskStateError("Team task has no assigned member")
        member = next(
            (value for value in snapshot.roster.members if value.member_id == task.assigned_member_id),
            None,
        )
        if member is None:
            raise TeamConcurrencyError("assigned Team member is stale")
        grant = next(
            (value for value in snapshot.authority.grants if value.member_id == member.member_id),
            None,
        )
        if grant is None:
            raise TeamConcurrencyError("assigned Team member grant is stale")
        definition = next(
            (
                value
                for value in self.registry.registration.capability_definitions
                if value.to_ref() == task.capability_definition_ref
            ),
            None,
        )
        if definition is None:
            raise CapabilityRuntimeInputError(
                "Team task Capability is not installed",
            )
        to_ref = getattr(request, "to_ref", None)
        if not callable(to_ref):
            raise CapabilityRuntimeInputError(
                "capability request lacks immutable reference authority",
            )
        request_ref = to_ref()
        if not isinstance(request_ref, ObjectRef):
            raise CapabilityRuntimeInputError(
                "capability request reference authority is invalid",
            )
        if (
            work.status is TeamTaskStatusV1.COMPLETED
            and work.lease is not None
            and work.fencing_token is not None
        ):
            operation = _operation_key(
                snapshot.team.team_incarnation_id,
                snapshot.graph.revision,
                task.task_id,
                work.attempt,
            )
            invocation_key = f"{operation}.attempt-{work.attempt}.fence-{work.fencing_token}"
            replay = self.store.replay_task_completion(
                team_id=team_id,
                task_id=task_id,
                invocation_key=invocation_key,
                request_ref=request_ref,
            )
            if replay is None:
                raise TeamTaskStateError(
                    "completed Team task is missing replay authority",
                )
            return replay
        used_budget = self.store.task_usage(team_id, task_id)
        requested_budget = (
            model_requests_delta,
            model_tokens_delta,
            cost_micro_usd_delta,
        )
        if any(
            used + requested > maximum
            for used, requested, maximum in zip(
                used_budget,
                requested_budget,
                (
                    task.max_model_requests,
                    task.max_model_tokens,
                    task.max_cost_micro_usd,
                ),
                strict=True,
            )
        ):
            raise CapabilityRuntimeInputError(
                "Team task Capability budget is exhausted",
            )
        if work.status is TeamTaskStatusV1.READY:
            attempt = work.attempt + 1
            operation = _operation_key(
                snapshot.team.team_incarnation_id,
                snapshot.graph.revision,
                task.task_id,
                attempt,
            )
            claim = self.store.claim_task(
                team_ref=snapshot.team.to_ref(),
                graph_ref=snapshot.graph.to_ref(),
                authority_ref=snapshot.authority.to_ref(),
                task_ref=task.to_ref(),
                member_ref=member.to_ref(),
                audit=audit,
                idempotency_key=f"{operation}.claim",
            )
            lease = self.store.acquire_lease(
                claim_ref=claim.to_ref(),
                lease_duration_seconds=lease_duration_seconds,
                audit=audit,
                idempotency_key=f"{operation}.lease",
            )
        elif work.status is TeamTaskStatusV1.ACTIVE and work.claim is not None and work.lease is not None:
            if definition.idempotency is CapabilityIdempotencyV1.NON_IDEMPOTENT:
                raise TeamTaskStateError(
                    "non-idempotent active work requires outcome verification",
                )
            attempt = work.attempt
            operation = _operation_key(
                snapshot.team.team_incarnation_id,
                snapshot.graph.revision,
                task.task_id,
                attempt,
            )
            claim = work.claim
            lease = work.lease
        else:
            raise TeamTaskStateError(
                "Team Capability runner requires ready or resumable active work",
            )
        leased_snapshot = self.store.get_snapshot(team_id)
        if leased_snapshot.team.to_ref() != snapshot.team.to_ref():
            raise TeamConcurrencyError("Team authority changed after task claim")
        inputs = self.store.task_input_envelopes(team_id, task_id)
        consumer = self.registry.bind_consumer(
            capability_id=definition.capability_id,
            consumer_id=f"team-agent-tool.{member.member_id}.{task.task_id}",
            consumer_version="v1",
            consumer_kind=CapabilityConsumerKindV1.AGENT_TOOL,
            projection_ref=None,
            audit=audit,
        )
        resolved = self.registry.resolve(consumer)
        local_execution = resolved.binding.provider_kind in {
            CapabilityProviderKindV1.LOCAL_DETERMINISTIC,
            CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
        }
        invocation_key = f"{operation}.attempt-{attempt}.fence-{lease.fencing_token}"
        context = CapabilityInvocationContextV1(
            session_ref=member.independent_session_ref,
            team_ref=leased_snapshot.team.to_ref(),
            member_ref=member.to_ref(),
            task_ref=task.to_ref(),
            authority_ref=leased_snapshot.authority.to_ref(),
            principal_ref=grant.principal_ref,
            data_purpose=grant.data_purposes[0],
            data_classification=grant.data_classifications[0],
            idempotency_key=invocation_key,
        )
        permission_scope = CapabilityPermissionScopeV1(
            member_id=member.member_id,
            task_id=task.task_id,
            data_scope_refs=sorted_refs(value.to_ref() for value in inputs),
            graph_mutation=GraphMutationKindV1.NONE,
            local_execution=local_execution,
            deterministic=deterministic,
            source_admission=False,
            external_execution=not local_execution,
            destructive=False,
            release=False,
            model_requests_delta=model_requests_delta,
            model_tokens_delta=model_tokens_delta,
            cost_micro_usd_delta=cost_micro_usd_delta,
        )

        # Provider/owner work is deliberately outside every TeamStore transaction.
        invocation = await self.runtime.invoke(
            consumer=consumer,
            context=context,
            permission_scope=permission_scope,
            authority=leased_snapshot.authority,
            request=request,
            input_artifacts=inputs,
            audit=audit,
        )
        invocation, bindings = self._bind_outputs(
            team_id=team_id,
            task_output_head_ids=task.output_artifact_head_ids,
            invocation=invocation,
            audit=audit,
        )
        return self.store.complete_task(
            team_ref=leased_snapshot.team.to_ref(),
            graph_ref=leased_snapshot.graph.to_ref(),
            authority_ref=leased_snapshot.authority.to_ref(),
            lease_ref=lease.to_ref(),
            fencing_token=lease.fencing_token,
            invocation=invocation,
            output_bindings=bindings,
            audit=audit,
            idempotency_key=f"{invocation_key}.commit",
        )

    def _bind_outputs(
        self,
        *,
        team_id: str,
        task_output_head_ids: tuple[str, ...],
        invocation: CapabilityRuntimeInvocation,
        audit: ContractAudit,
    ) -> tuple[
        CapabilityRuntimeInvocation,
        tuple[tuple[str, ArtifactEnvelopeV1], ...],
    ]:
        if invocation.result.outcome is not CapabilityInvocationOutcomeV1.SUCCEEDED:
            return invocation, ()
        outputs = tuple(
            sorted(
                invocation.output_artifacts,
                key=lambda value: (value.semantic_role, value.object_id),
            ),
        )
        head_ids = tuple(sorted(task_output_head_ids))
        if len(outputs) != len(head_ids):
            raise CapabilityRuntimeInputError(
                "Team task output heads differ from Runtime outputs",
            )
        current = {head.head_id: envelope for head, envelope in self.store.artifact_head_envelopes(team_id)}
        rebound: list[ArtifactEnvelopeV1] = []
        for head_id, output in zip(head_ids, outputs, strict=True):
            predecessor = current.get(head_id)
            envelope = ArtifactEnvelopeV1.create(
                artifact_id=(predecessor.artifact_id if predecessor is not None else output.artifact_id),
                subject_ref=output.subject_ref,
                schema_ref=output.schema_ref,
                content_ref=output.content_ref,
                media_type=output.media_type,
                modality=output.modality,
                domain_tags=output.domain_tags,
                semantic_role=output.semantic_role,
                purpose=output.purpose,
                classification=output.classification,
                lineage_refs=output.lineage_refs,
                producer_capability_ref=output.producer_capability_ref,
                producer_task_ref=output.producer_task_ref,
                validation_refs=output.validation_refs,
                revision=(predecessor.revision + 1 if predecessor is not None else 1),
                predecessor_envelope_ref=(predecessor.to_ref() if predecessor is not None else None),
                audit=audit,
            )
            rebound.append(envelope)
        rebound_refs = sorted_refs(value.to_ref() for value in rebound)
        if rebound_refs == invocation.result.output_artifact_refs:
            updated = invocation
        else:
            result = CapabilityResultV1.create(
                result_id=invocation.result.result_id,
                call_ref=invocation.call.to_ref(),
                outcome=invocation.result.outcome,
                canonical_result_ref=invocation.result.canonical_result_ref,
                output_artifact_refs=rebound_refs,
                validation_refs=invocation.result.validation_refs,
                failure_code=invocation.result.failure_code,
                audit=audit,
            )
            updated = replace(
                invocation,
                result=result,
                output_artifacts=tuple(rebound),
            )
        bindings = tuple(zip(head_ids, rebound, strict=True))
        return updated, bindings


def _operation_key(
    team_incarnation_id: str,
    graph_revision: int,
    task_id: str,
    attempt: int,
) -> str:
    raw = (f"{team_incarnation_id}:{graph_revision}:{task_id}:{attempt}").encode()
    return f"team-operation.{hashlib.sha256(raw).hexdigest()[:32]}"


__all__ = ["TeamCapabilityRunner"]
