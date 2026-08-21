from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_support.team_runtime_fixtures import audit, ref

import eval_factory.packs.generic_agent_trace.capabilities as module
from eval_factory.contracts.agent_system_v2 import (
    AttachmentQualityOutcomeV2,
)
from eval_factory.harness import CapabilityInvocationOutcomeV1
from eval_factory.packs import build_generic_agent_trace_execution_pack
from eval_factory.packs.generic_agent_trace.capabilities import (
    AttachmentQualityWorkflowCapabilityProvider,
)
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    AttachmentQualityCapabilityRequestV2,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("quality_outcome", "provider_outcome"),
    (
        (
            AttachmentQualityOutcomeV2.PASSED,
            CapabilityInvocationOutcomeV1.SUCCEEDED,
        ),
        (
            AttachmentQualityOutcomeV2.REPAIR_REQUIRED,
            CapabilityInvocationOutcomeV1.FAILED,
        ),
    ),
)
async def test_quality_v2_provider_executes_complete_owner_workflow(
    monkeypatch: pytest.MonkeyPatch,
    quality_outcome: AttachmentQualityOutcomeV2,
    provider_outcome: CapabilityInvocationOutcomeV1,
) -> None:
    monkeypatch.setattr(
        module,
        "item_quality_compilation_result_ref",
        lambda value: ref(
            "item-quality-compilation-result",
            "quality-v2",
            version="v2",
        ),
    )
    monkeypatch.setattr(
        module,
        "deterministic_item_validation_result_ref",
        lambda value: ref(
            "deterministic-item-validation-result",
            "quality-v2",
            version="v2",
        ),
    )
    quality_ref = ref(
        "attachment-quality-assessment",
        "quality-v2",
        version="v2",
    )
    quality = SimpleNamespace(
        outcome=quality_outcome,
        reason_codes=(
            () if quality_outcome is AttachmentQualityOutcomeV2.PASSED else ("QUALITY_REPAIR_REQUIRED",)
        ),
        to_ref=lambda: quality_ref,
    )
    view = SimpleNamespace(
        item_quality_head_ref=ref(
            "factory-item-stage-head",
            "quality-v2",
            version="v2",
        ),
        material_ref=ref(
            "attachment-quality-material",
            "quality-v2",
            version="v2",
        ),
        result=SimpleNamespace(
            finalization=SimpleNamespace(
                attachment_quality=quality,
                item_quality=object(),
                source_validation=object(),
            ),
            solvability=SimpleNamespace(
                assessment=SimpleNamespace(
                    to_ref=lambda: ref(
                        "solvability-assessment",
                        "quality-v2",
                        version="v2",
                    ),
                ),
            ),
        ),
    )
    provider = AttachmentQualityWorkflowCapabilityProvider(
        build_generic_agent_trace_execution_pack(audit=audit()),
        execution=cast(
            Any,
            SimpleNamespace(execute=lambda *args, **kwargs: None),
        ),
    )

    async def execute(*args: object, **kwargs: object):
        del args, kwargs
        return view

    provider.execution = cast(
        Any,
        SimpleNamespace(execute=execute),
    )
    result = await provider.invoke(
        AttachmentQualityCapabilityRequestV2.create(
            item_binding_ref=ref(
                "factory-item-run-binding",
                "quality-v2",
                version="v2",
            ),
            task_authoring_material_ref=ref(
                "task-authoring-material",
                "quality-v2",
                version="v2",
            ),
            attachment_execution_material_ref=ref(
                "attachment-execution-material",
                "quality-v2",
                version="v2",
            ),
            audit=audit(),
        ),
        call=cast(Any, object()),
        audit=audit(),
    )

    assert result.outcome is provider_outcome
    assert (
        result.canonical_result_ref == quality_ref
        if provider_outcome is CapabilityInvocationOutcomeV1.SUCCEEDED
        else result.failure_code == "QUALITY_REPAIR_REQUIRED"
    )
