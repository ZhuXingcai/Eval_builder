from __future__ import annotations

from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
)
from eval_factory.contracts.core import ObjectRef


class AgentRegistryError(RuntimeError):
    pass


class AgentRegistry:
    def __init__(
        self,
        *,
        capabilities: tuple[AgentCapabilityV2, ...],
        definitions: tuple[AgentDefinitionV2, ...],
    ) -> None:
        self._capabilities: dict[tuple[str, str, str, str], AgentCapabilityV2] = {}
        self._capabilities_by_id: dict[str, AgentCapabilityV2] = {}
        for raw_capability in capabilities:
            capability = AgentCapabilityV2.model_validate_json(raw_capability.canonical_json())
            key = _ref_key(capability.to_ref())
            if key in self._capabilities or capability.capability_id in self._capabilities_by_id:
                raise AgentRegistryError("duplicate Agent capability authority")
            self._capabilities[key] = capability
            self._capabilities_by_id[capability.capability_id] = capability

        self._definitions: dict[tuple[str, str], AgentDefinitionV2] = {}
        write_owners: dict[tuple[str, str], str] = {}
        for raw_definition in definitions:
            definition = AgentDefinitionV2.model_validate_json(raw_definition.canonical_json())
            definition_key = (definition.agent_role, definition.agent_version)
            if definition_key in self._definitions:
                raise AgentRegistryError("duplicate Agent definition authority")
            resolved = self.capabilities_for(definition)
            supported_task_kinds = {
                task_kind for capability in resolved for task_kind in capability.task_kinds
            }
            supported_tools = {tool_id for capability in resolved for tool_id in capability.tool_ids}
            supported_purposes = {purpose for capability in resolved for purpose in capability.data_purposes}
            supported_classifications = {
                classification
                for capability in resolved
                for classification in capability.data_classifications
            }
            if not set(definition.tool_ids).issubset(supported_tools):
                raise AgentRegistryError("Agent definition requests an unsupported tool")
            if definition.data_purpose not in supported_purposes:
                raise AgentRegistryError("Agent definition requests an unsupported data purpose")
            if not set(definition.allowed_data_classifications).issubset(supported_classifications):
                raise AgentRegistryError("Agent definition widens data classification access")
            for capability in resolved:
                for task_kind in capability.task_kinds:
                    for output_type in capability.output_object_types:
                        owner_key = (task_kind, output_type)
                        prior = write_owners.get(owner_key)
                        if prior is not None and prior != definition.agent_role:
                            raise AgentRegistryError("Agent definitions have overlapping write ownership")
                        write_owners[owner_key] = definition.agent_role
            if not supported_task_kinds:
                raise AgentRegistryError("Agent definition has no supported task kind")
            self._definitions[definition_key] = definition

    @property
    def definitions(self) -> tuple[AgentDefinitionV2, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))

    @property
    def capabilities(self) -> tuple[AgentCapabilityV2, ...]:
        return tuple(self._capabilities[key] for key in sorted(self._capabilities))

    def capabilities_for(
        self,
        definition: AgentDefinitionV2,
    ) -> tuple[AgentCapabilityV2, ...]:
        values: list[AgentCapabilityV2] = []
        for reference in definition.capability_refs:
            capability = self._capabilities.get(_ref_key(reference))
            if capability is None:
                raise AgentRegistryError("Agent definition references an unknown capability")
            values.append(capability)
        return tuple(values)

    def capability_by_id(self, capability_id: str) -> AgentCapabilityV2:
        value = self._capabilities_by_id.get(capability_id)
        if value is None:
            raise AgentRegistryError("unknown Agent capability ID")
        return value

    def resolve(self, agent_role: str, task_kind: str) -> AgentDefinitionV2:
        matches = [
            definition
            for (role, _), definition in self._definitions.items()
            if role == agent_role
            and any(task_kind in capability.task_kinds for capability in self.capabilities_for(definition))
        ]
        if len(matches) != 1:
            raise AgentRegistryError("Agent role and task kind do not resolve uniquely")
        return matches[0]


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = ["AgentRegistry", "AgentRegistryError"]
