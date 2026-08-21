from __future__ import annotations

from dataclasses import dataclass

from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef


@dataclass(frozen=True, slots=True)
class AttachmentAgentRegistryConfig:
    mock_prompt_ref: ObjectRef
    quality_prompt_ref: ObjectRef
    solvability_prompt_ref: ObjectRef
    mock_model_policy_ref: ObjectRef
    quality_model_policy_ref: ObjectRef
    solvability_model_policy_ref: ObjectRef


def build_attachment_agent_registry(
    *,
    config: AttachmentAgentRegistryConfig,
    audit: ContractAudit,
) -> AgentRegistry:
    capabilities = (
        _capability(
            capability_id="agent-capability://attachment-mock",
            task_kind="attachment-mock",
            inputs=("attachment-planning-context",),
            outputs=("attachment-group-result",),
            tools=("attachment-execution",),
            purpose="attachment-production",
            model_capabilities=("structured-output",),
            audit=audit,
        ),
        _capability(
            capability_id="agent-capability://attachment-quality",
            task_kind="attachment-quality",
            inputs=(
                "attachment-subgraph-result",
                "item-quality-compilation-result",
            ),
            outputs=("attachment-quality-assessment",),
            tools=("item-quality-read",),
            purpose="attachment-quality",
            model_capabilities=(),
            audit=audit,
        ),
        _capability(
            capability_id="agent-capability://attachment-solvability",
            task_kind="attachment-solvability",
            inputs=(
                "attachment-quality-assessment",
                "solvability-safe-view",
            ),
            outputs=("solvability-assessment",),
            tools=("solvability-evidence-read",),
            purpose="attachment-solvability",
            model_capabilities=(
                "structured-output",
                "reasoning",
            ),
            audit=audit,
        ),
    )
    by_id = {capability.capability_id: capability for capability in capabilities}
    definitions = (
        _definition(
            definition_id="agent-definition://attachment-mock",
            role="attachment-mock-agent",
            capability=by_id["agent-capability://attachment-mock"],
            prompt_ref=config.mock_prompt_ref,
            model_policy_ref=config.mock_model_policy_ref,
            tools=("attachment-execution",),
            purpose="attachment-production",
            validators=("attachment-group-validator",),
            max_model_requests=0,
            audit=audit,
        ),
        _definition(
            definition_id="agent-definition://attachment-quality",
            role="attachment-quality-agent",
            capability=by_id["agent-capability://attachment-quality"],
            prompt_ref=config.quality_prompt_ref,
            model_policy_ref=config.quality_model_policy_ref,
            tools=("item-quality-read",),
            purpose="attachment-quality",
            validators=("attachment-quality-validator",),
            max_model_requests=0,
            audit=audit,
        ),
        _definition(
            definition_id=("agent-definition://attachment-solvability"),
            role="attachment-solvability-agent",
            capability=by_id["agent-capability://attachment-solvability"],
            prompt_ref=config.solvability_prompt_ref,
            model_policy_ref=config.solvability_model_policy_ref,
            tools=("solvability-evidence-read",),
            purpose="attachment-solvability",
            validators=("solvability-assessment-validator",),
            max_model_requests=1,
            audit=audit,
        ),
    )
    return AgentRegistry(
        capabilities=capabilities,
        definitions=definitions,
    )


def _capability(
    *,
    capability_id: str,
    task_kind: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    tools: tuple[str, ...],
    purpose: str,
    model_capabilities: tuple[str, ...],
    audit: ContractAudit,
) -> AgentCapabilityV2:
    return AgentCapabilityV2.create(
        capability_id=capability_id,
        task_kinds=(task_kind,),
        input_object_types=inputs,
        output_object_types=outputs,
        model_capabilities=model_capabilities,
        tool_ids=tools,
        data_purposes=(purpose,),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=audit,
    )


def _definition(
    *,
    definition_id: str,
    role: str,
    capability: AgentCapabilityV2,
    prompt_ref: ObjectRef,
    model_policy_ref: ObjectRef,
    tools: tuple[str, ...],
    purpose: str,
    validators: tuple[str, ...],
    max_model_requests: int,
    audit: ContractAudit,
) -> AgentDefinitionV2:
    return AgentDefinitionV2.create(
        agent_definition_id=definition_id,
        agent_role=role,
        agent_version="v1",
        capability_refs=(capability.to_ref(),),
        prompt_template_ref=prompt_ref,
        model_policy_ref=model_policy_ref,
        tool_ids=tools,
        data_purpose=purpose,
        allowed_data_classifications=("RESTRICTED_TRACE_DERIVED",),
        validator_refs=tuple(_stable_ref("validator", value) for value in validators),
        max_attempts=2,
        max_model_requests=max_model_requests,
        max_model_tokens=8192 if max_model_requests else 0,
        max_cost_micro_usd=(200_000 if max_model_requests else 0),
        workspace_isolated=True,
        network_allowed=False,
        audit=audit,
    )


def _stable_ref(object_type: str, seed: str) -> ObjectRef:
    import hashlib

    digest = hashlib.sha256(seed.encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


__all__ = [
    "AttachmentAgentRegistryConfig",
    "build_attachment_agent_registry",
]
