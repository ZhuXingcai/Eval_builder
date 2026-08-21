from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_attachment_supervised_execution import RUN_ID, _run
from test_support.team_runtime_fixtures import ref

from eval_factory.agent_system.attachment_execution_material import (
    FactoryAttachmentExecutionMaterialError,
    FactoryAttachmentExecutionMaterialStore,
)
from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.contracts.agent_system_v2 import (
    AttachmentGenerationPlanV2,
    CompiledAttachmentGenerationPlanV2,
    PlanKindV2,
)
from eval_factory.packs.generic_agent_trace.attachment_projection import (
    GenericAgentAttachmentProjection,
)
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    AttachmentReconstructionCapabilityRequestV1,
)


class _ProjectionStore:
    def __init__(self, execution: object) -> None:
        self.binding = SimpleNamespace(
            item_id="item-attachment-projection",
            item_run_ref=ref(
                "factory-run",
                "attachment-projection-item",
                version="v2",
            ),
            to_ref=lambda: ref(
                "factory-item-run-binding",
                "attachment-projection",
                version="v2",
            ),
        )
        self.execution = execution
        self.committed = None
        self.material_ref = None

    def list_item_bindings(self, dataset_run_id: str):
        assert dataset_run_id == "dataset-run-attachment-projection"
        return (self.binding,)

    def get_item_run(self, binding: object):
        assert binding is self.binding
        return SimpleNamespace(run_id=RUN_ID)

    def get_domain_result(self, run_id: str, plan_kind: object):
        del plan_kind
        assert run_id == RUN_ID
        return SimpleNamespace(
            result_ref=self.execution.subgraph_result.to_ref(),
        )

    def get_item_stage_head(self, item_id: str, stage: object):
        del stage
        assert item_id == self.binding.item_id
        return SimpleNamespace(
            result_ref=ref(
                "r4-task-contract-set",
                "attachment-projection",
                version="v2",
            ),
        )

    def commit_item_stage_head(
        self,
        stage_head: object,
        *,
        idempotency_key: str,
        material_ref: object,
    ):
        assert idempotency_key.endswith(".head")
        self.committed = stage_head
        self.material_ref = material_ref
        return stage_head


@pytest.mark.asyncio
async def test_attachment_projection_commits_recoverable_material(
    tmp_path: Path,
) -> None:
    execution, _facade, supervisor, _producer = await _run(
        tmp_path / "execution",
    )
    domain = supervisor.store.get_domain_plan(
        RUN_ID,
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
    )
    plan = AttachmentGenerationPlanV2.model_validate_json(
        domain.plan_record_json,
    )
    compiled = CompiledAttachmentGenerationPlanV2.model_validate_json(
        domain.compiled_plan_record_json,
    )
    request = AttachmentReconstructionCapabilityRequestV1.create(
        run_id=RUN_ID,
        plan_ref=plan.to_ref(),
        compiled_plan_ref=compiled.to_ref(),
        preparation_ref=ref(
            "attachment-r5-preparation",
            "attachment-projection",
            version="v2",
        ),
        prior_batch_ref=None,
        lease_duration_seconds=600,
        audit=plan.audit,
    )
    store = _ProjectionStore(execution)
    materials = FactoryAttachmentExecutionMaterialStore(
        FactoryPrivateObjectStore(tmp_path / "private"),
    )
    projection = GenericAgentAttachmentProjection(
        dataset_run_id="dataset-run-attachment-projection",
        store=cast(Any, store),
        materials=materials,
    )

    refs = projection.commit(
        request=request,
        execution=execution,
        audit=plan.audit,
        idempotency_key="attachment-projection",
    )

    assert {value.object_type for value in refs} == {
        "attachment-execution-material",
        "factory-item-stage-head",
    }
    assert store.material_ref is not None
    assert materials.get(store.material_ref) == execution
    with pytest.raises(
        FactoryAttachmentExecutionMaterialError,
        match="reference type",
    ):
        materials.get(
            store.material_ref.model_copy(
                update={"object_type": "wrong-material"},
            )
        )
