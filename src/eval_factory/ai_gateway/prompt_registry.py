from __future__ import annotations

from eval_factory.contracts.ai_gateway_v2 import PromptTemplateV2
from eval_factory.contracts.core import ObjectRef


class PromptRegistry:
    def __init__(self, templates: tuple[PromptTemplateV2, ...]) -> None:
        self._by_ref: dict[tuple[str, str, str, str], PromptTemplateV2] = {}
        self._by_role_task: dict[tuple[str, str], PromptTemplateV2] = {}
        for template in templates:
            ref_key = _ref_key(template.to_ref())
            role_key = (template.agent_role, template.task_kind)
            if ref_key in self._by_ref or role_key in self._by_role_task:
                raise ValueError("duplicate current prompt template authority")
            self._by_ref[ref_key] = template
            self._by_role_task[role_key] = template

    def get(self, agent_role: str, task_kind: str) -> PromptTemplateV2:
        value = self._by_role_task.get((agent_role, task_kind))
        if value is None:
            raise ValueError("unknown prompt role and task kind")
        return value

    def get_by_ref(self, reference: ObjectRef) -> PromptTemplateV2:
        value = self._by_ref.get(_ref_key(reference))
        if value is None:
            raise ValueError("unknown prompt template authority")
        return value

    def require_compatible(
        self,
        reference: ObjectRef,
        *,
        agent_role: str,
        task_kind: str,
        required_capabilities: tuple[str, ...],
    ) -> PromptTemplateV2:
        template = self.get_by_ref(reference)
        if template.agent_role != agent_role:
            raise ValueError("prompt template agent role is incompatible")
        if template.task_kind != task_kind:
            raise ValueError("prompt template task kind is incompatible")
        if not set(template.required_model_capabilities).issubset(required_capabilities):
            raise ValueError("prompt template model capabilities are incompatible")
        return template


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = ["PromptRegistry"]
