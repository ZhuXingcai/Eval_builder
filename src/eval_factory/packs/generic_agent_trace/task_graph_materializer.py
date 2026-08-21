from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

from eval_factory.contracts.agent_system_v2 import PlanKindV2
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.harness import (
    CapabilityDefinitionV1,
    CapabilitySideEffectV1,
    ExecutionAuthorityV1,
    MemberExecutionGrantV1,
    StaticPackRegistrationV1,
)
from eval_factory.harness.contracts import sorted_refs, static_object_ref
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflowError,
    GenericAgentTeamAuthority,
)
from eval_factory.packs.generic_agent_trace.manifest import (
    GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION,
    GENERIC_AGENT_TRACE_PACK_ID,
    GENERIC_AGENT_TRACE_PACK_VERSION,
)
from eval_factory.team import (
    TeamLifecycleV1,
    TeamMemberStatusV1,
    TeamMemberV1,
    TeamRosterV1,
    TeamTaskGraphV1,
    TeamTaskStatusV1,
    TeamTaskV1,
    TeamV1,
)


class GenericAgentTaskScopeV1(StrEnum):
    DATASET = "DATASET"
    SOURCE = "SOURCE"


@dataclass(frozen=True, slots=True)
class GenericAgentTaskInstance:
    task_id: str
    capability_id: str
    scope: GenericAgentTaskScopeV1
    scope_ref: ObjectRef | None
    plan_kind: PlanKindV2 | None


@dataclass(frozen=True, slots=True)
class GenericAgentTaskGraphMaterialization:
    authority: GenericAgentTeamAuthority
    task_instances: tuple[GenericAgentTaskInstance, ...]

    def instance(self, task_id: str) -> GenericAgentTaskInstance:
        matches = tuple(value for value in self.task_instances if value.task_id == task_id)
        if len(matches) != 1:
            raise GenericAgentFactoryWorkflowError(
                "generic Agent task instance is not uniquely materialized",
            )
        return matches[0]


class GenericAgentTaskGraphMaterializer:
    """Expands static capability edges into bounded execution instances."""

    _EXPECTED_ROLES = frozenset(
        {
            "control",
            "coordinator",
            "quality",
            "requirement",
            "task",
            "trace",
        }
    )
    _EXPECTED_CAPABILITIES = frozenset(
        {
            "capability.requirement-planning",
            "capability.trace-ingestion",
            "capability.task-authoring",
            "capability.attachment-reconstruction",
            "capability.criteria-rubric",
            "capability.grading-design",
            "capability.quality-review",
            "capability.batch-quality",
            "capability.plan-review",
            "capability.delivery",
        }
    )

    def materialize(
        self,
        *,
        registration: StaticPackRegistrationV1,
        requirement_ref: ObjectRef,
        source_refs: tuple[ObjectRef, ...],
        independent_session_refs: dict[str, ObjectRef],
        team_id: str,
        team_incarnation_id: str,
        max_model_requests: int,
        max_model_tokens: int,
        max_cost_micro_usd: int,
        audit: ContractAudit,
    ) -> GenericAgentTaskGraphMaterialization:
        self._validate_inputs(
            registration=registration,
            source_refs=source_refs,
            independent_session_refs=independent_session_refs,
            team_id=team_id,
        )
        profiles = {value.role: value for value in registration.agent_profiles}
        members = tuple(
            TeamMemberV1.create(
                member_id=f"{team_id}.member-{role}",
                member_incarnation_id=(f"{team_incarnation_id}.member-{role}.1"),
                role=role,
                agent_profile_ref=profiles["coordinator" if role == "control" else role].to_ref(),
                independent_session_ref=independent_session_refs[role],
                capability_definition_refs=profiles[
                    "coordinator" if role == "control" else role
                ].capability_definition_refs,
                status=TeamMemberStatusV1.READY,
                is_coordinator=role == "coordinator",
                audit=audit,
            )
            for role in sorted(self._EXPECTED_ROLES)
        )
        roster = TeamRosterV1.create(
            roster_id=f"{team_id}.roster",
            team_id=team_id,
            revision=1,
            predecessor_roster_ref=None,
            members=members,
            audit=audit,
        )
        definitions = {value.capability_id: value for value in registration.capability_definitions}
        tasks, instances = self._tasks(
            team_id=team_id,
            registration=registration,
            definitions=definitions,
            source_refs=source_refs,
            max_model_requests=max_model_requests,
            max_model_tokens=max_model_tokens,
            max_cost_micro_usd=max_cost_micro_usd,
            audit=audit,
        )
        graph = TeamTaskGraphV1.create(
            graph_id=f"{team_id}.graph",
            team_id=team_id,
            revision=1,
            predecessor_graph_ref=None,
            roster_ref=roster.to_ref(),
            member_ids=tuple(value.member_id for value in members),
            tasks=tasks,
            audit=audit,
        )
        providers = {value.capability_definition_ref: value for value in registration.provider_bindings}
        grants = tuple(
            MemberExecutionGrantV1(
                member_id=member.member_id,
                principal_ref=static_object_ref(
                    object_type="principal",
                    object_id=f"principal://generic-agent/{member.member_id}",
                    object_version="v1",
                    payload={"member_id": member.member_id},
                ),
                capability_definition_refs=member.capability_definition_refs,
                provider_binding_refs=sorted_refs(
                    providers[reference].to_ref() for reference in member.capability_definition_refs
                ),
                task_ids=tuple(
                    sorted(task.task_id for task in tasks if task.assigned_member_id == member.member_id)
                ),
                data_scope_refs=(),
                data_purposes=("evaluation-data-production",),
                data_classifications=("INTERNAL", "RESTRICTED"),
                allowed_side_effects=tuple(
                    sorted(
                        CapabilitySideEffectV1,
                        key=lambda value: value.value,
                    )
                ),
            )
            for member in members
        )
        authority = ExecutionAuthorityV1.create(
            authority_id=f"{team_id}.authority",
            authority_version=1,
            predecessor_authority_ref=None,
            team_id=team_id,
            team_incarnation_id=team_incarnation_id,
            roster_ref=roster.to_ref(),
            task_graph_ref=graph.to_ref(),
            permission_policy_ref=registration.permission_policies[0].to_ref(),
            grants=grants,
            max_model_requests=max_model_requests,
            max_model_tokens=max_model_tokens,
            max_cost_micro_usd=max_cost_micro_usd,
            used_model_requests=0,
            used_model_tokens=0,
            used_cost_micro_usd=0,
            audit=audit,
        )
        team = TeamV1.create(
            team_id=team_id,
            team_incarnation_id=team_incarnation_id,
            team_version=1,
            goal_ref=requirement_ref,
            composition_ref=registration.composition.to_ref(),
            roster_ref=roster.to_ref(),
            task_graph_ref=graph.to_ref(),
            authority_ref=authority.to_ref(),
            coordinator_member_id=f"{team_id}.member-coordinator",
            lifecycle=TeamLifecycleV1.ACTIVE,
            audit=audit,
        )
        return GenericAgentTaskGraphMaterialization(
            authority=GenericAgentTeamAuthority(
                team=team,
                roster=roster,
                graph=graph,
                authority=authority,
            ),
            task_instances=tuple(sorted(instances, key=lambda value: value.task_id)),
        )

    def _tasks(
        self,
        *,
        team_id: str,
        registration: StaticPackRegistrationV1,
        definitions: dict[str, CapabilityDefinitionV1],
        source_refs: tuple[ObjectRef, ...],
        max_model_requests: int,
        max_model_tokens: int,
        max_cost_micro_usd: int,
        audit: ContractAudit,
    ) -> tuple[tuple[TeamTaskV1, ...], list[GenericAgentTaskInstance]]:
        tasks: list[TeamTaskV1] = []
        instances: list[GenericAgentTaskInstance] = []

        def head(value: str) -> str:
            return f"{team_id}.head-{value}"

        def task(value: str) -> str:
            return f"{team_id}.task-{value}"

        requirement_task = task("requirement-planning")
        compiled_head = head("compiled-build-plan")
        self._append_task(
            tasks,
            instances,
            registration=registration,
            definitions=definitions,
            task_id=requirement_task,
            task_kind="requirement-planning",
            capability_id="capability.requirement-planning",
            role="requirement",
            dependencies=(),
            input_heads=(head("evaluation-requirement"),),
            output_head=compiled_head,
            status=TeamTaskStatusV1.READY,
            scope=GenericAgentTaskScopeV1.DATASET,
            scope_ref=None,
            plan_kind=None,
            max_model_requests=max_model_requests,
            max_model_tokens=max_model_tokens,
            max_cost_micro_usd=max_cost_micro_usd,
            audit=audit,
        )
        global_review_task = task("plan-review-global")
        global_review_head = head("review-global")
        self._append_task(
            tasks,
            instances,
            registration=registration,
            definitions=definitions,
            task_id=global_review_task,
            task_kind="plan-review-global",
            capability_id="capability.plan-review",
            role="control",
            dependencies=(requirement_task,),
            input_heads=(compiled_head,),
            output_head=global_review_head,
            status=TeamTaskStatusV1.PENDING,
            scope=GenericAgentTaskScopeV1.DATASET,
            scope_ref=None,
            plan_kind=PlanKindV2.GLOBAL_BUILD,
            max_model_requests=max_model_requests,
            max_model_tokens=max_model_tokens,
            max_cost_micro_usd=max_cost_micro_usd,
            audit=audit,
        )

        source_heads: dict[str, list[str]] = {
            "trace": [],
            "task": [],
            "attachment": [],
            "quality": [],
            "criteria": [],
            "grading": [],
            "review": [global_review_head],
        }
        trace_tasks: list[str] = []
        fan_in_tasks: list[str] = []
        for source_ref in source_refs:
            suffix = _scope_digest(source_ref)
            trace_task = task(f"trace-{suffix}")
            trace_head = head(f"trace-{suffix}")
            author_task = task(f"task-authoring-{suffix}")
            author_head = head(f"task-{suffix}")
            attachment_review_task = task(f"plan-review-attachment-{suffix}")
            attachment_review_head = head(f"review-attachment-{suffix}")
            attachment_task = task(f"attachment-{suffix}")
            attachment_head = head(f"attachment-{suffix}")
            quality_task = task(f"quality-{suffix}")
            quality_head = head(f"quality-{suffix}")
            criteria_review_task = task(f"plan-review-criteria-{suffix}")
            criteria_review_head = head(f"review-criteria-{suffix}")
            criteria_task = task(f"criteria-{suffix}")
            criteria_head = head(f"criteria-{suffix}")
            grading_review_task = task(f"plan-review-grading-{suffix}")
            grading_review_head = head(f"review-grading-{suffix}")
            grading_task = task(f"grading-{suffix}")
            grading_head = head(f"grading-{suffix}")
            source_specs = (
                (
                    trace_task,
                    "trace-ingestion",
                    "capability.trace-ingestion",
                    "trace",
                    (global_review_task,),
                    (compiled_head,),
                    trace_head,
                    None,
                ),
                (
                    author_task,
                    "task-authoring",
                    "capability.task-authoring",
                    "task",
                    (trace_task,),
                    (trace_head,),
                    author_head,
                    None,
                ),
                (
                    attachment_review_task,
                    "plan-review-attachment",
                    "capability.plan-review",
                    "control",
                    (author_task,),
                    (compiled_head,),
                    attachment_review_head,
                    PlanKindV2.ATTACHMENT_GENERATION,
                ),
                (
                    attachment_task,
                    "attachment-reconstruction",
                    "capability.attachment-reconstruction",
                    "task",
                    (author_task, attachment_review_task),
                    (author_head,),
                    attachment_head,
                    None,
                ),
                (
                    quality_task,
                    "quality-review",
                    "capability.quality-review",
                    "quality",
                    (author_task, attachment_task),
                    (attachment_head, author_head),
                    quality_head,
                    None,
                ),
                (
                    criteria_review_task,
                    "plan-review-criteria",
                    "capability.plan-review",
                    "control",
                    (quality_task,),
                    (compiled_head,),
                    criteria_review_head,
                    PlanKindV2.CRITERIA_RUBRIC,
                ),
                (
                    criteria_task,
                    "criteria-rubric",
                    "capability.criteria-rubric",
                    "task",
                    (author_task, attachment_task, criteria_review_task),
                    (attachment_head, author_head),
                    criteria_head,
                    None,
                ),
                (
                    grading_review_task,
                    "plan-review-grading",
                    "capability.plan-review",
                    "control",
                    (criteria_task,),
                    (compiled_head,),
                    grading_review_head,
                    PlanKindV2.GRADING_DESIGN,
                ),
                (
                    grading_task,
                    "grading-design",
                    "capability.grading-design",
                    "task",
                    (criteria_task, grading_review_task),
                    (criteria_head,),
                    grading_head,
                    None,
                ),
            )
            for (
                task_id,
                task_kind,
                capability_id,
                role,
                dependencies,
                input_heads,
                output_head,
                plan_kind,
            ) in source_specs:
                self._append_task(
                    tasks,
                    instances,
                    registration=registration,
                    definitions=definitions,
                    task_id=task_id,
                    task_kind=task_kind,
                    capability_id=capability_id,
                    role=role,
                    dependencies=dependencies,
                    input_heads=input_heads,
                    output_head=output_head,
                    status=TeamTaskStatusV1.PENDING,
                    scope=GenericAgentTaskScopeV1.SOURCE,
                    scope_ref=source_ref,
                    plan_kind=plan_kind,
                    max_model_requests=max_model_requests,
                    max_model_tokens=max_model_tokens,
                    max_cost_micro_usd=max_cost_micro_usd,
                    audit=audit,
                )
            source_heads["task"].append(author_head)
            source_heads["trace"].append(trace_head)
            source_heads["attachment"].append(attachment_head)
            source_heads["quality"].append(quality_head)
            source_heads["criteria"].append(criteria_head)
            source_heads["grading"].append(grading_head)
            source_heads["review"].extend(
                (
                    attachment_review_head,
                    criteria_review_head,
                    grading_review_head,
                )
            )
            fan_in_tasks.extend(
                (author_task, quality_task, criteria_task, grading_task),
            )
            trace_tasks.append(trace_task)

        execution_mode = registration.manifest.pack_version == GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION
        batch_task = task("batch-quality")
        batch_head = head("batch-quality")
        self._append_task(
            tasks,
            instances,
            registration=registration,
            definitions=definitions,
            task_id=batch_task,
            task_kind="batch-quality",
            capability_id="capability.batch-quality",
            role="quality",
            dependencies=tuple((*trace_tasks, *fan_in_tasks) if execution_mode else fan_in_tasks),
            input_heads=(
                tuple(source_heads["trace"])
                if execution_mode
                else tuple(
                    value
                    for role in (
                        "criteria",
                        "grading",
                        "quality",
                        "task",
                    )
                    for value in source_heads[role]
                )
            ),
            output_head=batch_head,
            status=TeamTaskStatusV1.PENDING,
            scope=GenericAgentTaskScopeV1.DATASET,
            scope_ref=None,
            plan_kind=None,
            max_model_requests=max_model_requests,
            max_model_tokens=max_model_tokens,
            max_cost_micro_usd=max_cost_micro_usd,
            audit=audit,
        )
        final_review_task = task("plan-review-final-delivery")
        final_review_head = head("review-final-delivery")
        self._append_task(
            tasks,
            instances,
            registration=registration,
            definitions=definitions,
            task_id=final_review_task,
            task_kind="plan-review-final-delivery",
            capability_id="capability.plan-review",
            role="control",
            dependencies=(batch_task,),
            input_heads=(compiled_head,),
            output_head=final_review_head,
            status=TeamTaskStatusV1.PENDING,
            scope=GenericAgentTaskScopeV1.DATASET,
            scope_ref=None,
            plan_kind=PlanKindV2.FINAL_DELIVERY,
            max_model_requests=max_model_requests,
            max_model_tokens=max_model_tokens,
            max_cost_micro_usd=max_cost_micro_usd,
            audit=audit,
        )
        source_heads["review"].append(final_review_head)
        delivery_task = task("delivery")
        self._append_task(
            tasks,
            instances,
            registration=registration,
            definitions=definitions,
            task_id=delivery_task,
            task_kind="delivery",
            capability_id="capability.delivery",
            role="control",
            dependencies=(
                (batch_task, final_review_task)
                if execution_mode
                else (
                    batch_task,
                    final_review_task,
                    *fan_in_tasks,
                )
            ),
            input_heads=(
                (batch_head, final_review_head)
                if execution_mode
                else (
                    batch_head,
                    *source_heads["criteria"],
                    *source_heads["grading"],
                    *source_heads["quality"],
                    *source_heads["review"],
                    *source_heads["task"],
                )
            ),
            output_head=head("candidate-dataset-delivery"),
            status=TeamTaskStatusV1.PENDING,
            scope=GenericAgentTaskScopeV1.DATASET,
            scope_ref=None,
            plan_kind=None,
            max_model_requests=max_model_requests,
            max_model_tokens=max_model_tokens,
            max_cost_micro_usd=max_cost_micro_usd,
            audit=audit,
        )
        return tuple(tasks), instances

    @staticmethod
    def _append_task(
        tasks: list[TeamTaskV1],
        instances: list[GenericAgentTaskInstance],
        *,
        registration: StaticPackRegistrationV1,
        definitions: dict[str, CapabilityDefinitionV1],
        task_id: str,
        task_kind: str,
        capability_id: str,
        role: str,
        dependencies: tuple[str, ...],
        input_heads: tuple[str, ...],
        output_head: str,
        status: TeamTaskStatusV1,
        scope: GenericAgentTaskScopeV1,
        scope_ref: ObjectRef | None,
        plan_kind: PlanKindV2 | None,
        max_model_requests: int,
        max_model_tokens: int,
        max_cost_micro_usd: int,
        audit: ContractAudit,
    ) -> None:
        definition = definitions.get(capability_id)
        if definition is None:
            raise GenericAgentFactoryWorkflowError(
                "task instance capability is not installed",
            )
        acceptance = static_object_ref(
            object_type="acceptance-check",
            object_id=f"acceptance-check://generic-agent/{task_id}",
            object_version="v1",
            payload={
                "capability_id": capability_id,
                "pack_version": registration.manifest.pack_version,
                "plan_kind": plan_kind.value if plan_kind is not None else None,
                "scope_ref": scope_ref,
                "task_id": task_id,
            },
        )
        tasks.append(
            TeamTaskV1.create(
                task_id=task_id,
                task_kind=task_kind,
                dependency_task_ids=dependencies,
                assigned_member_id=f"{task_id.rsplit('.task-', 1)[0]}.member-{role}",
                capability_definition_ref=definition.to_ref(),
                input_artifact_head_ids=input_heads,
                output_artifact_head_ids=(output_head,),
                acceptance_check_refs=(acceptance,),
                status=status,
                max_attempts=3,
                max_model_requests=max_model_requests,
                max_model_tokens=max_model_tokens,
                max_cost_micro_usd=max_cost_micro_usd,
                audit=audit,
            )
        )
        instances.append(
            GenericAgentTaskInstance(
                task_id=task_id,
                capability_id=capability_id,
                scope=scope,
                scope_ref=scope_ref,
                plan_kind=plan_kind,
            )
        )

    def _validate_inputs(
        self,
        *,
        registration: StaticPackRegistrationV1,
        source_refs: tuple[ObjectRef, ...],
        independent_session_refs: dict[str, ObjectRef],
        team_id: str,
    ) -> None:
        if (
            registration.manifest.pack_id != GENERIC_AGENT_TRACE_PACK_ID
            or registration.manifest.pack_version
            not in {
                GENERIC_AGENT_TRACE_PACK_VERSION,
                GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION,
            }
            or {value.capability_id for value in registration.capability_definitions}
            != self._EXPECTED_CAPABILITIES
        ):
            raise GenericAgentFactoryWorkflowError(
                "task materializer requires the exact current generic Agent Pack",
            )
        if set(independent_session_refs) != self._EXPECTED_ROLES:
            raise GenericAgentFactoryWorkflowError(
                "task materializer requires exact independent sessions",
            )
        if ".task-" in team_id or len(team_id) > 120:
            raise ValueError("task materializer team ID is not safely expandable")
        if not 1 <= len(source_refs) <= 1_000:
            raise ValueError(
                "task materializer requires between 1 and 1000 sources",
            )
        if source_refs != sorted_refs(source_refs):
            raise ValueError(
                "task materializer source refs must be sorted and unique",
            )


def _scope_digest(reference: ObjectRef) -> str:
    payload = json.dumps(
        canonical_value_v2(reference),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


__all__ = [
    "GenericAgentTaskGraphMaterialization",
    "GenericAgentTaskGraphMaterializer",
    "GenericAgentTaskInstance",
    "GenericAgentTaskScopeV1",
]
