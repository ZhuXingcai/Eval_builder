from __future__ import annotations

import hashlib
from dataclasses import dataclass

from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef


@dataclass(frozen=True, slots=True)
class GradingDesignAgentRegistryConfig:
    prompt_ref: ObjectRef
    model_policy_ref: ObjectRef


def build_grading_design_agent_registry(
    *,
    config: GradingDesignAgentRegistryConfig,
    audit: ContractAudit,
) -> AgentRegistry:
    capability = AgentCapabilityV2.create(
        capability_id="agent-capability://grading-design",
        task_kinds=("grading-design",),
        input_object_types=(
            "criteria-rubric-result",
            "evaluator-spec",
            "reference-policy",
            "rubric-set",
            "tool-policy",
        ),
        output_object_types=("grading-design-result",),
        model_capabilities=("reasoning", "structured-output"),
        tool_ids=("reference-grant-read",),
        data_purposes=("grading-design-authoring",),
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
        audit=audit,
    )
    definition = AgentDefinitionV2.create(
        agent_definition_id="agent-definition://grading-design",
        agent_role="grading-design-agent",
        agent_version="v1",
        capability_refs=(capability.to_ref(),),
        prompt_template_ref=config.prompt_ref,
        model_policy_ref=config.model_policy_ref,
        tool_ids=("reference-grant-read",),
        data_purpose="grading-design-authoring",
        allowed_data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
        validator_refs=(
            _stable_ref(
                "validator",
                "judge-design-validator",
            ),
        ),
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=16_000,
        max_cost_micro_usd=500_000,
        workspace_isolated=True,
        network_allowed=False,
        audit=audit,
    )
    return AgentRegistry(
        capabilities=(capability,),
        definitions=(definition,),
    )


def _stable_ref(
    object_type: str,
    seed: str,
) -> ObjectRef:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


__all__ = [
    "GradingDesignAgentRegistryConfig",
    "build_grading_design_agent_registry",
]
