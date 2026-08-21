from __future__ import annotations

from pathlib import Path

import pytest
from grading_fixtures import audit, plan

from eval_factory.agent_system.grading_material_store import (
    GradingDesignMaterialError,
    GradingDesignMaterialStore,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
)
from eval_factory.contracts.agent_system_v2 import (
    GradingDesignPlanV2,
    JudgeDesignValidationOutcomeV2,
    JudgeDesignValidationV2,
)
from eval_factory.contracts.core import ObjectRef


def _validation() -> JudgeDesignValidationV2:
    source = plan()
    return JudgeDesignValidationV2.create(
        validation_id=("judge-design-validation://material-store"),
        plan_ref=source.to_ref(),
        design_spec_ref=None,
        criterion_ids=(),
        validated_output_fields=(),
        reference_access_result_refs=(),
        outcome=(JudgeDesignValidationOutcomeV2.BLOCKED_POLICY),
        reason_codes=("REFERENCE_ACCESS_DENIED",),
        audit=audit(),
    )


def test_material_store_binds_behavior_ref_to_private_cas_exactly(
    tmp_path: Path,
) -> None:
    store = GradingDesignMaterialStore(tmp_path)
    validation = _validation()
    reference = validation.to_ref()

    first = store.put_model(
        behavior_ref=reference,
        value=validation,
    )
    replay = store.put_model(
        behavior_ref=reference,
        value=validation,
    )

    assert replay == first
    assert store.get_model(reference, type(validation)) == validation
    assert not hasattr(store, "list")


def test_material_store_rejects_rebinding_binding_drift_and_wrong_schema(
    tmp_path: Path,
) -> None:
    store = GradingDesignMaterialStore(tmp_path)
    validation = _validation()
    reference = validation.to_ref()
    store.put_model(
        behavior_ref=reference,
        value=validation,
    )
    changed = reference.model_copy(
        update={"object_sha256": "f" * 64},
    )

    with pytest.raises(
        GradingDesignMaterialError,
        match="different material",
    ):
        store.put_model(
            behavior_ref=changed,
            value=validation,
        )
    with pytest.raises(
        GradingDesignMaterialError,
        match="binding drifted",
    ):
        store.get_model(
            changed,
            type(validation),
        )
    with pytest.raises(
        GradingDesignMaterialError,
        match="wrong schema",
    ):
        store.get_model(
            reference,
            GradingDesignPlanV2,
        )


def test_material_store_rejects_missing_binding_and_corrupt_cas(
    tmp_path: Path,
) -> None:
    store = GradingDesignMaterialStore(tmp_path)
    validation = _validation()
    reference = validation.to_ref()
    content_ref = store.put_model(
        behavior_ref=reference,
        value=validation,
    )
    path = store.private_store.root / "sha256" / content_ref.object_sha256[:2] / content_ref.object_sha256
    path.write_bytes(b"corrupt")

    with pytest.raises(
        FactoryPrivateObjectError,
        match="hash is corrupt",
    ):
        store.get_model(reference, type(validation))
    with pytest.raises(
        GradingDesignMaterialError,
        match="missing",
    ):
        store.get_model(
            ObjectRef(
                object_type="judge-design-validation",
                object_id=("judge-design-validation://missing"),
                object_version="v2",
                object_sha256="a" * 64,
            ),
            type(validation),
        )
