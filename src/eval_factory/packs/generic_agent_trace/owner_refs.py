from __future__ import annotations

from eval_factory.contracts.core import ObjectRef
from eval_factory.task_authoring import (
    ToolCapabilityCatalog,
    tool_capability_catalog_ref,
)


def generic_agent_tool_catalog_ref(
    catalog: ToolCapabilityCatalog,
) -> ObjectRef:
    return tool_capability_catalog_ref(catalog).model_copy(
        update={"object_version": "v2"},
    )


__all__ = ["generic_agent_tool_catalog_ref"]
