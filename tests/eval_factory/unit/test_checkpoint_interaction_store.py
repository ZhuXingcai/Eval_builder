from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from test_user_approval_requests import (
    _generation_policy,
    _job_spec,
)
from test_user_decisions import (
    _handling_policy,
    _label_request,
)

from eval_factory.approval.interaction_models import (
    UserCheckpointSourceContextV2,
    user_checkpoint_source_context_v2_ref,
)
from eval_factory.approval.interaction_store import (
    UserCheckpointMaterialIntegrityError,
    UserCheckpointMaterialLimitError,
    UserCheckpointMaterialStore,
    UserCheckpointMaterialTypeError,
)
from eval_factory.approval.requests import LabelPlanApprovalSource
from eval_factory.contracts.checkpoint_interaction_v2 import (
    user_checkpoint_presentation_v2_ref,
)


def _context() -> UserCheckpointSourceContextV2:
    policy, source, compilation, _ = _label_request()
    return UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(policy),
        approval_policy=policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )


def test_material_store_separates_private_context_and_public_presentation(
    tmp_path: Path,
) -> None:
    context = _context()
    presentation = context.presentations()[0]
    store = UserCheckpointMaterialStore(
        tmp_path / "checkpoint-store",
        max_source_context_bytes=2_000_000,
        max_presentation_bytes=1_000_000,
    )

    first_context = store.put_source_context(context)
    replay_context = store.put_source_context(context)
    first_presentation = store.put_presentation(presentation)

    assert first_context.written is True
    assert replay_context.written is False
    assert first_context.object_ref == user_checkpoint_source_context_v2_ref(context)
    assert first_presentation.object_ref == (user_checkpoint_presentation_v2_ref(presentation))
    assert store.get_source_context(first_context.object_ref) == context
    assert store.get_presentation(first_presentation.object_ref) == presentation
    with pytest.raises(UserCheckpointMaterialTypeError):
        store.get_presentation(first_context.object_ref)


def test_material_store_limits_missing_and_corruption_fail_closed(
    tmp_path: Path,
) -> None:
    context = _context()
    presentation = context.presentations()[0]
    with pytest.raises(UserCheckpointMaterialLimitError):
        UserCheckpointMaterialStore(
            tmp_path / "invalid",
            max_source_context_bytes=1,
            max_presentation_bytes=1,
        )
    limited = UserCheckpointMaterialStore(
        tmp_path / "limited",
        max_source_context_bytes=2,
        max_presentation_bytes=2,
    )
    with pytest.raises(UserCheckpointMaterialLimitError):
        limited.put_source_context(context)
    missing = UserCheckpointMaterialStore(
        tmp_path / "missing",
        max_source_context_bytes=2_000_000,
        max_presentation_bytes=1_000_000,
    )
    with pytest.raises(UserCheckpointMaterialIntegrityError, match="missing"):
        missing.get_presentation(user_checkpoint_presentation_v2_ref(presentation))

    root = tmp_path / "corrupt"
    corrupt = UserCheckpointMaterialStore(
        root,
        max_source_context_bytes=2_000_000,
        max_presentation_bytes=1_000_000,
    )
    write = corrupt.put_presentation(presentation)
    envelope_path = (
        root
        / "presentation"
        / "sha256"
        / write.object_ref.object_sha256[:2]
        / f"{write.object_ref.object_sha256}.json"
    )
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    content_digest = envelope["content_blob_ref"]["object_sha256"]
    content_path = root / "content" / "sha256" / content_digest[:2] / content_digest
    content_path.write_bytes(b"{}")
    with pytest.raises(UserCheckpointMaterialIntegrityError, match="corrupt"):
        corrupt.get_presentation(write.object_ref)

    envelope_path.write_text("{}", encoding="utf-8")
    with pytest.raises(UserCheckpointMaterialIntegrityError, match="invalid"):
        corrupt.get_presentation(write.object_ref)


def test_private_context_and_presentation_identity_ignore_nested_audit_actor_time(
    tmp_path: Path,
) -> None:
    context = _context()
    source_audit = context.label_plans[0].audit
    changed_audit = source_audit.model_copy(
        update={
            "created_at": source_audit.created_at + timedelta(days=1),
            "created_by": "another-r7-07-actor",
        }
    )
    changed_plan = context.label_plans[0].model_copy(update={"audit": changed_audit})
    changed_source = LabelPlanApprovalSource(
        label_spec=context.label_specs[0],
        label_plan=changed_plan,
    )
    changed_context = UserCheckpointSourceContextV2.create(
        job_spec=context.job_spec,
        approval_policy=context.approval_policy,
        generation_policy=context.generation_policy,
        handling_policy=context.handling_policy,
        request_compilation=context.request_compilation,
        requested_by=context.requested_by,
        sources=(changed_source,),
    )
    changed_presentation = changed_context.presentations()[0]

    assert user_checkpoint_source_context_v2_ref(changed_context) == (
        user_checkpoint_source_context_v2_ref(context)
    )
    assert user_checkpoint_presentation_v2_ref(changed_presentation) == (
        user_checkpoint_presentation_v2_ref(context.presentations()[0])
    )
    store = UserCheckpointMaterialStore(
        tmp_path / "audit-independent",
        max_source_context_bytes=2_000_000,
        max_presentation_bytes=1_000_000,
    )
    store.put_source_context(context)
    store.put_presentation(context.presentations()[0])

    replay_context = store.put_source_context(changed_context)
    replay_presentation = store.put_presentation(changed_presentation)

    assert replay_context.written is False
    assert replay_presentation.written is False
