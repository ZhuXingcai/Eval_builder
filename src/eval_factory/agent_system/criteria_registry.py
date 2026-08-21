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
class CriteriaRubricAgentRegistryConfig:
    prompt_ref: ObjectRef
    model_policy_ref: ObjectRef


def build_criteria_rubric_agent_registry(
    *,
    config: CriteriaRubricAgentRegistryConfig,
    audit: ContractAudit,
) -> AgentRegistry:
    capability = AgentCapabilityV2.create(
        capability_id="agent-capability://criteria-rubric",
        task_kinds=("criteria-rubric",),
        input_object_types=(
            "attachment-quality-assessment",
            "solvability-assessment",
            "task-draft",
        ),
        output_object_types=("criteria-rubric-result",),
        model_capabilities=("reasoning", "structured-output"),
        tool_ids=("rubric-candidate-read",),
        data_purposes=("criteria-rubric-authoring",),
        data_classifications=(
            "RESTRICTED_EVALUATOR_CONTROL",
            "RESTRICTED_TRACE_DERIVED",
        ),
        audit=audit,
    )
    definition = AgentDefinitionV2.create(
        agent_definition_id="agent-definition://criteria-rubric",
        agent_role="criteria-rubric-agent",
        agent_version="v1",
        capability_refs=(capability.to_ref(),),
        prompt_template_ref=config.prompt_ref,
        model_policy_ref=config.model_policy_ref,
        tool_ids=("rubric-candidate-read",),
        data_purpose="criteria-rubric-authoring",
        allowed_data_classifications=(
            "RESTRICTED_EVALUATOR_CONTROL",
            "RESTRICTED_TRACE_DERIVED",
        ),
        validator_refs=(
            _stable_ref(
                "validator",
                "criteria-rubric-chain-validator",
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
    "CriteriaRubricAgentRegistryConfig",
    "build_criteria_rubric_agent_registry",
]
