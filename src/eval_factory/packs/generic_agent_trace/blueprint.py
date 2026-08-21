from __future__ import annotations

from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from eval_factory.blueprints.models import (
    BlueprintBudgetV1,
    EvaluationBlueprintV1,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.capability_runtime import (
    CapabilityRuntimeRegistry,
)
from eval_factory.harness.composition import StaticPackRegistrationV1
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique_refs,
)
from eval_factory.harness.interaction import ExecutionAuthorityV1
from eval_factory.packs.generic_agent_trace.manifest import (
    GENERIC_AGENT_TRACE_PACK_ID,
    GENERIC_AGENT_TRACE_PACK_VERSION,
    build_generic_agent_evaluation_blueprint,
)


class GenericAgentBlueprintCompilerError(RuntimeError):
    """Raised when fixed Blueprint authority is missing or drifted."""


class GenericAgentBlueprintInputV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/blueprint-input/v1"] = (
        "generic-agent-trace/blueprint-input/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-agent-blueprint-input"

    requirement_ref: ObjectRef
    team_ref: ObjectRef
    roster_ref: ObjectRef
    task_graph_ref: ObjectRef
    authority: ExecutionAuthorityV1
    evaluator_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    review_gate_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    budget: BlueprintBudgetV1
    delivery_profile_ref: ObjectRef

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        require_ref(
            self.requirement_ref,
            "evaluation-requirement-spec",
            "requirement_ref",
            object_version="v2",
        )
        require_ref(self.team_ref, "agent-team", "team_ref")
        require_ref(self.roster_ref, "team-roster", "roster_ref")
        require_ref(
            self.task_graph_ref,
            "team-task-graph",
            "task_graph_ref",
        )
        require_sorted_unique_refs(
            self.evaluator_refs,
            "evaluator_refs",
        )
        require_sorted_unique_refs(
            self.review_gate_refs,
            "review_gate_refs",
        )
        require_ref(
            self.delivery_profile_ref,
            "delivery-profile",
            "delivery_profile_ref",
        )
        return self


class GenericAgentBlueprintCompiler:
    _REQUIRED_CAPABILITIES = frozenset(
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
        },
    )

    def __init__(
        self,
        *,
        registration: StaticPackRegistrationV1,
        runtime_registry: CapabilityRuntimeRegistry,
    ) -> None:
        self.registration = StaticPackRegistrationV1.model_validate(
            registration,
        )
        self.runtime_registry = runtime_registry

    def compile(
        self,
        blueprint_input: GenericAgentBlueprintInputV1,
        *,
        audit: ContractAudit,
    ) -> EvaluationBlueprintV1:
        self._validate_installed_authority(blueprint_input)
        blueprint = build_generic_agent_evaluation_blueprint(
            registration=self.registration,
            requirement_ref=blueprint_input.requirement_ref,
            team_ref=blueprint_input.team_ref,
            roster_ref=blueprint_input.roster_ref,
            task_graph_ref=blueprint_input.task_graph_ref,
            authority_ref=blueprint_input.authority.to_ref(),
            evaluator_refs=blueprint_input.evaluator_refs,
            review_gate_refs=blueprint_input.review_gate_refs,
            budget=blueprint_input.budget,
            delivery_profile_ref=(blueprint_input.delivery_profile_ref),
            audit=audit,
        )
        if (
            len(blueprint.capability_bindings) != 10
            or {value.capability_id for value in blueprint.capability_bindings} != self._REQUIRED_CAPABILITIES
            or blueprint.execution_provider_refs != self.registration.manifest.provider_binding_refs
        ):
            raise GenericAgentBlueprintCompilerError(
                "fixed generic Agent Blueprint is incomplete",
            )
        return blueprint

    def _validate_installed_authority(
        self,
        blueprint_input: GenericAgentBlueprintInputV1,
    ) -> None:
        registration = self.registration
        if (
            registration.manifest.pack_id != GENERIC_AGENT_TRACE_PACK_ID
            or registration.manifest.pack_version != GENERIC_AGENT_TRACE_PACK_VERSION
            or self.runtime_registry.registration != registration
        ):
            raise GenericAgentBlueprintCompilerError(
                "generic Agent Pack composition is not exact",
            )
        capabilities = {value.capability_id for value in registration.capability_definitions}
        if capabilities != self._REQUIRED_CAPABILITIES or len(self.runtime_registry.providers) != 10:
            raise GenericAgentBlueprintCompilerError(
                "generic Agent runtime provider set is incomplete",
            )
        authority = blueprint_input.authority
        if (
            authority.roster_ref != blueprint_input.roster_ref
            or authority.task_graph_ref != blueprint_input.task_graph_ref
            or authority.permission_policy_ref not in registration.manifest.permission_policy_refs
        ):
            raise GenericAgentBlueprintCompilerError(
                "Blueprint execution authority is stale",
            )
        granted_capabilities = {
            reference for grant in authority.grants for reference in grant.capability_definition_refs
        }
        granted_providers = {
            reference for grant in authority.grants for reference in grant.provider_binding_refs
        }
        if granted_capabilities != set(
            registration.manifest.capability_definition_refs
        ) or granted_providers != set(registration.manifest.provider_binding_refs):
            raise GenericAgentBlueprintCompilerError(
                "Blueprint execution grants differ from the Pack",
            )
        if blueprint_input.review_gate_refs != registration.manifest.permission_policy_refs:
            raise GenericAgentBlueprintCompilerError(
                "Blueprint review gates differ from the Pack",
            )
        budget = blueprint_input.budget
        if (
            budget.max_model_requests > authority.max_model_requests - authority.used_model_requests
            or budget.max_model_tokens > authority.max_model_tokens - authority.used_model_tokens
            or budget.max_cost_micro_usd > authority.max_cost_micro_usd - authority.used_cost_micro_usd
        ):
            raise GenericAgentBlueprintCompilerError(
                "Blueprint budget widens execution authority",
            )


__all__ = [
    "GenericAgentBlueprintCompiler",
    "GenericAgentBlueprintCompilerError",
    "GenericAgentBlueprintInputV1",
]
