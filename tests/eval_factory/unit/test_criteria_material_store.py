from __future__ import annotations

from pathlib import Path

import pytest
from test_rubric_authoring import (
    _draft,
    _generated_proposal,
    _request,
)

from eval_factory.agent_system.criteria_material_store import (
    CriteriaRubricMaterialError,
    CriteriaRubricMaterialStore,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.task_v2 import (
    EvaluatorSpecV2,
    rubric_set_ref,
)
from eval_factory.task_authoring import RubricSetCompiler


def _rubric():
    draft = _draft()
    request = _request(draft)
    result = RubricSetCompiler().compile(
        request=request,
        proposal=_generated_proposal(request),
        task_draft=draft,
        audit=draft.audit,
    )
    assert result.rubric_set is not None
    return result.rubric_set


def test_material_store_binds_behavior_ref_to_private_cas_exactly(
    tmp_path: Path,
) -> None:
    store = CriteriaRubricMaterialStore(tmp_path)
    rubric = _rubric()
    reference = rubric_set_ref(rubric)

    first = store.put_model(
        behavior_ref=reference,
        value=rubric,
    )
    replay = store.put_model(
        behavior_ref=reference,
        value=rubric,
    )

    assert replay == first
    assert store.get_model(reference, type(rubric)) == rubric
    assert not hasattr(store, "list")


def test_material_store_rejects_behavior_rebinding_and_wrong_schema(
    tmp_path: Path,
) -> None:
    store = CriteriaRubricMaterialStore(tmp_path)
    rubric = _rubric()
    reference = rubric_set_ref(rubric)
    store.put_model(
        behavior_ref=reference,
        value=rubric,
    )
    changed = reference.model_copy(
        update={"object_sha256": "f" * 64},
    )

    with pytest.raises(
        CriteriaRubricMaterialError,
        match="different material",
    ):
        store.put_model(
            behavior_ref=changed,
            value=rubric,
        )
    with pytest.raises(
        CriteriaRubricMaterialError,
        match="wrong schema",
    ):
        store.get_model(reference, EvaluatorSpecV2)


def test_material_store_rejects_missing_binding_and_corrupt_cas(
    tmp_path: Path,
) -> None:
    store = CriteriaRubricMaterialStore(tmp_path)
    rubric = _rubric()
    reference = rubric_set_ref(rubric)
    content_ref = store.put_model(
        behavior_ref=reference,
        value=rubric,
    )
    path = store.private_store.root / "sha256" / content_ref.object_sha256[:2] / content_ref.object_sha256
    path.write_bytes(b"corrupt")

    with pytest.raises(
        FactoryPrivateObjectError,
        match="hash is corrupt",
    ):
        store.get_model(reference, type(rubric))
    with pytest.raises(
        CriteriaRubricMaterialError,
        match="missing",
    ):
        store.get_model(
            ObjectRef(
                object_type="rubric-set",
                object_id="rubric-set://missing",
                object_version="v2",
                object_sha256="a" * 64,
            ),
            type(rubric),
        )
