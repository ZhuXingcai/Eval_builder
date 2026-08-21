from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_support.team_runtime_fixtures import audit, ref

from eval_factory.packs.generic_agent_trace.capability_contracts import (
    AttachmentQualityCapabilityRequestV2,
)
from eval_factory.packs.generic_agent_trace.quality_execution import (
    GenericAgentAttachmentQualityExecution,
    GenericAgentAttachmentQualityExecutionError,
)


class _Store:
    def __init__(self) -> None:
        self.binding = SimpleNamespace(
            item_id="item-quality-execution",
            to_ref=lambda: ref(
                "factory-item-run-binding",
                "quality-execution",
                version="v2",
            ),
        )
        self.task_head = SimpleNamespace(
            to_ref=lambda: ref(
                "factory-item-stage-head",
                "quality-task",
                version="v2",
            ),
        )
        self.attachment_head = SimpleNamespace(
            to_ref=lambda: ref(
                "factory-item-stage-head",
                "quality-attachment",
                version="v2",
            ),
        )
        self.task_material_ref = ref(
            "task-authoring-material",
            "quality-execution",
            version="v2",
        )
        self.attachment_material_ref = ref(
            "attachment-execution-material",
            "quality-execution",
            version="v2",
        )

    def list_item_bindings(self, dataset_run_id: str):
        assert dataset_run_id == "dataset-quality-execution"
        return (self.binding,)

    def get_item_stage_head(self, item_id: str, stage: object):
        assert item_id == self.binding.item_id
        return self.task_head if str(stage).endswith("TASK_AUTHORING") else self.attachment_head

    def get_item_stage_material_ref(self, head_ref: object):
        return self.task_material_ref if head_ref == self.task_head.to_ref() else self.attachment_material_ref


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def advance(self, **kwargs: object):
        self.calls.append(kwargs)
        return SimpleNamespace(result=object())


@pytest.mark.asyncio
async def test_quality_execution_loads_current_owner_material() -> None:
    store = _Store()
    runtime = _Runtime()
    task_authoring = SimpleNamespace(
        producer_task_view_result=SimpleNamespace(
            producer_task_view=object(),
        ),
        leakage_reference_set=object(),
        task_contract_set=object(),
    )
    attachment = object()
    context = object()
    execution = GenericAgentAttachmentQualityExecution(
        dataset_run_id="dataset-quality-execution",
        store=cast(Any, store),
        task_authoring_materials=cast(
            Any,
            SimpleNamespace(get=lambda reference: task_authoring),
        ),
        attachment_materials=cast(
            Any,
            SimpleNamespace(get=lambda reference: attachment),
        ),
        runtime=cast(Any, runtime),
        contexts=cast(
            Any,
            SimpleNamespace(get=lambda item_id: context),
        ),
    )
    request = AttachmentQualityCapabilityRequestV2.create(
        item_binding_ref=store.binding.to_ref(),
        task_authoring_material_ref=store.task_material_ref,
        attachment_execution_material_ref=(store.attachment_material_ref),
        audit=audit(),
    )

    await execution.execute(request, audit=audit())

    assert len(runtime.calls) == 1
    assert runtime.calls[0]["supervised_execution"] is attachment
    assert runtime.calls[0]["context"] is context


@pytest.mark.asyncio
async def test_quality_execution_rejects_stale_material_ref() -> None:
    store = _Store()
    execution = GenericAgentAttachmentQualityExecution(
        dataset_run_id="dataset-quality-execution",
        store=cast(Any, store),
        task_authoring_materials=cast(Any, object()),
        attachment_materials=cast(Any, object()),
        runtime=cast(Any, object()),
        contexts=cast(Any, object()),
    )
    request = AttachmentQualityCapabilityRequestV2.create(
        item_binding_ref=store.binding.to_ref(),
        task_authoring_material_ref=ref(
            "task-authoring-material",
            "stale",
            version="v2",
        ),
        attachment_execution_material_ref=(store.attachment_material_ref),
        audit=audit(),
    )

    with pytest.raises(
        GenericAgentAttachmentQualityExecutionError,
        match="differs",
    ):
        await execution.execute(request, audit=audit())
