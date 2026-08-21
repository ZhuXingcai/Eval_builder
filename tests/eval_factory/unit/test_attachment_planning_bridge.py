from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.attachment_planning import (
    ATTACHMENT_PLANNING_POLICY_VERSION,
    AttachmentPlanningBridge,
    AttachmentPlanningPolicyError,
)
from eval_factory.contracts import (
    attachment_planning_context_carried_sha256,
    producer_storage_authorization_carried_sha256,
    producer_storage_authorization_ref,
    producer_task_view_carried_sha256,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.task_v2 import (
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
)
from eval_factory.provenance import (
    evidence_bundle_carried_sha256,
    evidence_bundle_ref,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
FOURTH_HASH = "d" * 64
SOURCE_TRACE_ID = "source-trace://attachment-planning/r5-01"
TRACE_IR_VERSION_ID = "trace-ir://attachment-planning/r5-01"
PRODUCER_PRINCIPAL_ID = "principal://attachment-producer/r5-01"
ROOT = Path(__file__).resolve().parents[3]


def _audit(
    created_at: datetime = datetime(2026, 7, 27, tzinfo=UTC),
    *,
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="attachment-planning-test",
        governing_versions=(
            VersionBinding(
                component="attachment-planning",
                version="r5-01",
            ),
        ),
        input_refs=input_refs,
    )


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v1",
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _span(
    suffix: str,
    *,
    source_trace_id: str = SOURCE_TRACE_ID,
) -> SourceSpanRef:
    return SourceSpanRef(
        span_id=f"source-span://attachment-planning/{suffix}",
        source_trace_id=source_trace_id,
        raw_sha256=HASH,
    )


def _evidence(
    suffix: str,
    *,
    subject_ref: ObjectRef | None = None,
    source_trace_id: str = SOURCE_TRACE_ID,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://attachment-planning/{suffix}",
        subject_ref=subject_ref
        or _ref(
            "file-version-projection",
            suffix,
        ),
        source_spans=(
            _span(
                suffix,
                source_trace_id=source_trace_id,
            ),
        ),
        polarity=EvidencePolarity.POSITIVE,
        capability="attachment-planning",
        capability_complete=True,
    )


def _bundle(
    *,
    evidence: tuple[EvidenceRef, ...] | None = None,
    excluded_subject_refs: tuple[ObjectRef, ...] = (),
    projection_policy_ref: ObjectRef | None = None,
    consumer_stage: str = "attachment-producer",
    purpose: str = "attachment-production",
    source_trace_id: str = SOURCE_TRACE_ID,
    trace_ir_version_id: str = TRACE_IR_VERSION_ID,
    audit: ContractAudit | None = None,
) -> EvidenceBundle:
    values = (_evidence("input"),) if evidence is None else evidence
    bundle = EvidenceBundle(
        evidence_bundle_id="evidence-bundle://pending",
        source_trace_id=source_trace_id,
        trace_ir_version_id=trace_ir_version_id,
        consumer_stage=consumer_stage,
        purpose=purpose,
        projection_policy_ref=projection_policy_ref
        or _ref(
            "projection-policy",
            "attachment-producer",
            version="evidence-views/r2-05-v1",
        ),
        evidence=values,
        excluded_subject_refs=excluded_subject_refs,
        returned_characters=120 if values else 0,
        max_characters=1000,
        tainted_content_included=False,
        bundle_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = evidence_bundle_carried_sha256(bundle)
    return bundle.model_copy(
        update={
            "evidence_bundle_id": f"evidence-bundle://sha256/{digest}",
            "bundle_sha256": digest,
        }
    )


def _rehash_bundle(
    bundle: EvidenceBundle,
    **updates: object,
) -> EvidenceBundle:
    updated = bundle.model_copy(update=updates)
    digest = evidence_bundle_carried_sha256(updated)
    return updated.model_copy(
        update={
            "evidence_bundle_id": f"evidence-bundle://sha256/{digest}",
            "bundle_sha256": digest,
        }
    )


def _authorization(
    bundle: EvidenceBundle,
    *,
    authorized_subject_refs: tuple[ObjectRef, ...] | None = None,
    projection_policy_ref: ObjectRef | None = None,
    source_task_draft_sha256: str = OTHER_HASH,
    producer_principal_id: str = PRODUCER_PRINCIPAL_ID,
    audit: ContractAudit | None = None,
) -> ProducerStorageAuthorizationV2:
    subjects = (
        tuple(sorted((item.subject_ref for item in bundle.evidence), key=_ref_key))
        if authorized_subject_refs is None
        else authorized_subject_refs
    )
    authorization = ProducerStorageAuthorizationV2(
        authorization_id="producer-storage-authorization://pending",
        producer_principal_id=producer_principal_id,
        purpose="ATTACHMENT_PRODUCTION",
        source_task_draft_sha256=source_task_draft_sha256,
        projection_policy_ref=projection_policy_ref or bundle.projection_policy_ref,
        evidence_bundle_refs=(evidence_bundle_ref(bundle),),
        authorized_subject_refs=subjects,
        raw_store_access=False,
        canonical_store_access=False,
        quarantine_store_access=False,
        private_reference_store_access=False,
        credentials_issued=False,
        policy_version="producer-task-view/r4-08-v1",
        authorization_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = producer_storage_authorization_carried_sha256(authorization)
    return authorization.model_copy(
        update={
            "authorization_id": (f"producer-storage-authorization://sha256/{digest}"),
            "authorization_sha256": digest,
        }
    )


def _view(
    bundle: EvidenceBundle,
    authorization: ProducerStorageAuthorizationV2,
    *,
    safe_evidence_bundle_refs: tuple[ObjectRef, ...] | None = None,
    projection_policy_ref: ObjectRef | None = None,
    storage_authorization_ref: ObjectRef | None = None,
    source_task_draft_sha256: str = OTHER_HASH,
    source_contract_chain_sha256: str = THIRD_HASH,
    audit: ContractAudit | None = None,
) -> ProducerTaskViewV2:
    view = ProducerTaskViewV2(
        producer_task_view_id="producer-task-view://pending",
        producer_task_view_version=1,
        supersedes_producer_task_view_ref=None,
        query_instruction="Inspect the safe input-state workspace.",
        attachment_requirements=(),
        allowed_tools=(),
        safe_evidence_bundle_refs=safe_evidence_bundle_refs or (evidence_bundle_ref(bundle),),
        forbidden_outputs=("original final answer",),
        projection_policy_ref=projection_policy_ref or bundle.projection_policy_ref,
        storage_authorization_ref=storage_authorization_ref
        or producer_storage_authorization_ref(authorization),
        prompt_boundary_enforcement_ref=_ref(
            "prompt-boundary-enforcement",
            "attachment-producer",
            version="prompt-injection-as-data/r2-07-v1",
        ),
        source_task_draft_sha256=source_task_draft_sha256,
        source_contestant_tool_policy_sha256=FOURTH_HASH,
        source_contract_chain_sha256=source_contract_chain_sha256,
        policy_version="producer-task-view/r4-08-v1",
        producer_task_view_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = producer_task_view_carried_sha256(view)
    return view.model_copy(
        update={
            "producer_task_view_id": f"producer-task-view://sha256/{digest}",
            "producer_task_view_sha256": digest,
        }
    )


def _inputs(
    *,
    evidence: tuple[EvidenceRef, ...] | None = None,
    excluded_subject_refs: tuple[ObjectRef, ...] = (),
    audit: ContractAudit | None = None,
) -> tuple[
    ProducerTaskViewV2,
    ProducerStorageAuthorizationV2,
    EvidenceBundle,
]:
    bundle = _bundle(
        evidence=evidence,
        excluded_subject_refs=excluded_subject_refs,
        audit=audit,
    )
    authorization = _authorization(
        bundle,
        audit=audit,
    )
    view = _view(
        bundle,
        authorization,
        audit=audit,
    )
    return view, authorization, bundle


def test_bridge_compiles_minimal_exact_current_context() -> None:
    caller_ref = _ref(
        "private-reference",
        "caller-audit",
        digest=FOURTH_HASH,
    )
    view, authorization, bundle = _inputs()

    context = AttachmentPlanningBridge().compile(
        producer_task_view=view,
        storage_authorization=authorization,
        evidence_bundle=bundle,
        audit=_audit(input_refs=(caller_ref,)),
    )

    assert context.producer_task_view_ref.object_id == view.producer_task_view_id
    assert context.producer_storage_authorization_ref == (producer_storage_authorization_ref(authorization))
    assert context.safe_evidence_bundle_ref == evidence_bundle_ref(bundle)
    assert context.projection_policy_ref == bundle.projection_policy_ref
    assert context.source_trace_id == SOURCE_TRACE_ID
    assert context.trace_ir_version_id == TRACE_IR_VERSION_ID
    assert context.producer_principal_id == PRODUCER_PRINCIPAL_ID
    assert context.source_task_draft_sha256 == view.source_task_draft_sha256
    assert context.source_contract_chain_sha256 == (view.source_contract_chain_sha256)
    assert context.policy_version == ATTACHMENT_PLANNING_POLICY_VERSION
    assert context.attachment_planning_context_sha256 == (attachment_planning_context_carried_sha256(context))
    assert set(context.audit.input_refs) == {
        context.producer_task_view_ref,
        context.producer_storage_authorization_ref,
        context.safe_evidence_bundle_ref,
        context.projection_policy_ref,
    }
    assert caller_ref not in context.audit.input_refs

    serialized = str(context.model_dump(mode="json"))
    for forbidden in (
        "Inspect the safe input-state workspace.",
        "original final answer",
        "source-span://",
        "evidence-ref://",
        "file-version-projection://",
        "private-reference://caller-audit",
        "authorized_subject_refs",
        "query_instruction",
        "attachment_requirements",
        "allowed_tools",
        "forbidden_outputs",
    ):
        assert forbidden not in serialized


def test_bridge_accepts_empty_safe_bundle_without_fabricating_evidence() -> None:
    view, authorization, bundle = _inputs(evidence=())

    context = AttachmentPlanningBridge().compile(
        producer_task_view=view,
        storage_authorization=authorization,
        evidence_bundle=bundle,
        audit=_audit(),
    )

    assert bundle.evidence == ()
    assert authorization.authorized_subject_refs == ()
    assert context.safe_evidence_bundle_ref == evidence_bundle_ref(bundle)
    assert "source-span://" not in str(context.model_dump(mode="json"))


@pytest.mark.parametrize(
    "stale_input",
    ["view", "authorization", "bundle"],
)
def test_bridge_rejects_stale_source_identities(stale_input: str) -> None:
    view, authorization, bundle = _inputs()
    if stale_input == "view":
        view = view.model_copy(update={"producer_task_view_sha256": FOURTH_HASH})
    elif stale_input == "authorization":
        authorization = authorization.model_copy(update={"authorization_sha256": FOURTH_HASH})
    else:
        bundle = bundle.model_copy(update={"bundle_sha256": FOURTH_HASH})

    with pytest.raises(AttachmentPlanningPolicyError, match="identity"):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )


def test_bridge_rejects_cross_object_ref_and_policy_mismatches() -> None:
    _, authorization, bundle = _inputs()
    wrong_authorization_ref = _ref(
        "producer-storage-authorization",
        "wrong",
        version="v2",
        digest=FOURTH_HASH,
    )
    wrong_view = _view(
        bundle,
        authorization,
        storage_authorization_ref=wrong_authorization_ref,
    )
    with pytest.raises(AttachmentPlanningPolicyError, match="authorization"):
        AttachmentPlanningBridge().compile(
            producer_task_view=wrong_view,
            storage_authorization=authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )

    extra_bundle_ref = _ref(
        "evidence-bundle",
        "z-extra",
        digest=FOURTH_HASH,
    )
    wrong_view = _view(
        bundle,
        authorization,
        safe_evidence_bundle_refs=(
            evidence_bundle_ref(bundle),
            extra_bundle_ref,
        ),
    )
    with pytest.raises(AttachmentPlanningPolicyError, match="EvidenceBundle"):
        AttachmentPlanningBridge().compile(
            producer_task_view=wrong_view,
            storage_authorization=authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )

    wrong_policy = _ref(
        "projection-policy",
        "wrong",
        version="evidence-views/r2-05-v1",
        digest=FOURTH_HASH,
    )
    wrong_authorization = _authorization(
        bundle,
        projection_policy_ref=wrong_policy,
    )
    wrong_view = _view(
        bundle,
        wrong_authorization,
    )
    with pytest.raises(AttachmentPlanningPolicyError, match="projection policy"):
        AttachmentPlanningBridge().compile(
            producer_task_view=wrong_view,
            storage_authorization=wrong_authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )

    wrong_authorization = _authorization(
        bundle,
        source_task_draft_sha256=FOURTH_HASH,
    )
    wrong_view = _view(
        bundle,
        wrong_authorization,
    )
    with pytest.raises(AttachmentPlanningPolicyError, match="TaskDraft"):
        AttachmentPlanningBridge().compile(
            producer_task_view=wrong_view,
            storage_authorization=wrong_authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )


def test_bridge_requires_exact_bundle_subject_authorization() -> None:
    view, _, bundle = _inputs()
    missing_authorization = _authorization(
        bundle,
        authorized_subject_refs=(),
    )
    view = _view(
        bundle,
        missing_authorization,
    )

    with pytest.raises(AttachmentPlanningPolicyError, match="subject"):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=missing_authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )

    extra_subject = _ref(
        "file-version-projection",
        "z-extra",
        digest=FOURTH_HASH,
    )
    subjects = tuple(
        sorted(
            (
                bundle.evidence[0].subject_ref,
                extra_subject,
            ),
            key=_ref_key,
        )
    )
    extra_authorization = _authorization(
        bundle,
        authorized_subject_refs=subjects,
    )
    view = _view(
        bundle,
        extra_authorization,
    )
    with pytest.raises(AttachmentPlanningPolicyError, match="subject"):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=extra_authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )


def test_bridge_rejects_noncanonical_evidence_and_exclusion_order() -> None:
    evidence_a = _evidence(
        "a",
        subject_ref=_ref("file-version-projection", "a"),
    )
    evidence_b = _evidence(
        "b",
        subject_ref=_ref(
            "file-version-projection",
            "b",
            digest=OTHER_HASH,
        ),
    )
    view, authorization, bundle = _inputs(evidence=(evidence_b, evidence_a))
    with pytest.raises(AttachmentPlanningPolicyError, match=r"evidence.*sorted"):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )

    excluded_a = _ref("file-version", "a")
    excluded_b = _ref(
        "file-version",
        "b",
        digest=OTHER_HASH,
    )
    view, authorization, bundle = _inputs(
        evidence=(),
        excluded_subject_refs=(excluded_b, excluded_a),
    )
    with pytest.raises(AttachmentPlanningPolicyError, match=r"excluded.*sorted"):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )


def test_bridge_rejects_duplicate_evidence_and_cross_trace_spans() -> None:
    duplicate = _evidence("duplicate")
    bundle = _bundle(evidence=(duplicate, duplicate))
    authorization = _authorization(
        bundle,
        authorized_subject_refs=(duplicate.subject_ref,),
    )
    view = _view(bundle, authorization)
    with pytest.raises(AttachmentPlanningPolicyError, match="duplicate evidence"):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )

    shared_subject = _ref("file-version-projection", "shared")
    bundle = _bundle(
        evidence=(
            _evidence("a", subject_ref=shared_subject),
            _evidence("b", subject_ref=shared_subject),
        )
    )
    authorization = _authorization(
        bundle,
        authorized_subject_refs=(shared_subject,),
    )
    view = _view(bundle, authorization)
    with pytest.raises(AttachmentPlanningPolicyError, match="duplicate evidence subject"):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )

    cross_trace = _evidence(
        "cross-trace",
        source_trace_id="source-trace://other",
    )
    view, authorization, bundle = _inputs(evidence=(cross_trace,))
    with pytest.raises(AttachmentPlanningPolicyError, match="source trace"):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "raw_store_access",
        "canonical_store_access",
        "quarantine_store_access",
        "private_reference_store_access",
        "credentials_issued",
    ],
)
def test_bridge_rejects_direct_store_or_credential_bypass(
    field_name: str,
) -> None:
    _, authorization, bundle = _inputs()
    unsafe_authorization = authorization.model_copy(update={field_name: True})
    digest = producer_storage_authorization_carried_sha256(unsafe_authorization)
    unsafe_authorization = unsafe_authorization.model_copy(
        update={
            "authorization_id": (f"producer-storage-authorization://sha256/{digest}"),
            "authorization_sha256": digest,
        }
    )
    view = _view(bundle, unsafe_authorization)

    with pytest.raises(AttachmentPlanningPolicyError, match="store access or credentials"):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=unsafe_authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )


@pytest.mark.parametrize(
    ("consumer_stage", "purpose"),
    [
        pytest.param(
            "task-authoring",
            "attachment-production",
            id="wrong-stage",
        ),
        pytest.param(
            "attachment-producer",
            "evaluation",
            id="wrong-purpose",
        ),
    ],
)
def test_bridge_rejects_wrong_bundle_stage_or_purpose(
    consumer_stage: str,
    purpose: str,
) -> None:
    bundle = _bundle(
        consumer_stage=consumer_stage,
        purpose=purpose,
    )
    authorization = _authorization(bundle)
    view = _view(bundle, authorization)

    with pytest.raises(AttachmentPlanningPolicyError, match="stage or purpose"):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )


@pytest.mark.parametrize(
    ("bundle_updates", "message"),
    [
        pytest.param(
            {
                "returned_characters": 1001,
                "max_characters": 1000,
            },
            "budget",
            id="over-budget",
        ),
        pytest.param(
            {"tainted_content_included": True},
            "tainted",
            id="tainted-content",
        ),
    ],
)
def test_bridge_rejects_bundle_safety_invariant_bypass(
    bundle_updates: dict[str, object],
    message: str,
) -> None:
    _, _, original_bundle = _inputs()
    bundle = _rehash_bundle(original_bundle, **bundle_updates)
    authorization = _authorization(bundle)
    view = _view(bundle, authorization)

    with pytest.raises(AttachmentPlanningPolicyError, match=message):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )


def test_bridge_rejects_sensitive_projected_subject_bypass() -> None:
    unsafe_subject = _ref(
        "private-reference-projection",
        "unsafe",
        digest=FOURTH_HASH,
    )
    bundle = _bundle(
        evidence=(
            _evidence(
                "unsafe",
                subject_ref=unsafe_subject,
            ),
        )
    )
    safe_authorization = _authorization(
        bundle,
        authorized_subject_refs=(),
    )
    unsafe_authorization = safe_authorization.model_copy(
        update={"authorized_subject_refs": (unsafe_subject,)}
    )
    digest = producer_storage_authorization_carried_sha256(unsafe_authorization)
    unsafe_authorization = unsafe_authorization.model_copy(
        update={
            "authorization_id": (f"producer-storage-authorization://sha256/{digest}"),
            "authorization_sha256": digest,
        }
    )
    view = _view(bundle, unsafe_authorization)

    with pytest.raises(AttachmentPlanningPolicyError, match="safe projected"):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=unsafe_authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )


def test_bridge_rejects_duplicate_excluded_subject_bypass() -> None:
    excluded = _ref("file-version", "excluded")
    bundle = _bundle(
        evidence=(),
        excluded_subject_refs=(excluded, excluded),
    )
    authorization = _authorization(bundle)
    view = _view(bundle, authorization)

    with pytest.raises(AttachmentPlanningPolicyError, match="duplicate excluded subject"):
        AttachmentPlanningBridge().compile(
            producer_task_view=view,
            storage_authorization=authorization,
            evidence_bundle=bundle,
            audit=_audit(),
        )


def test_validate_current_accepts_exact_and_rejects_mutation() -> None:
    view, authorization, bundle = _inputs()
    bridge = AttachmentPlanningBridge()
    context = bridge.compile(
        producer_task_view=view,
        storage_authorization=authorization,
        evidence_bundle=bundle,
        audit=_audit(),
    )

    bridge.validate_current(
        producer_task_view=view,
        storage_authorization=authorization,
        evidence_bundle=bundle,
        attachment_planning_context=context,
    )

    mutated = context.model_copy(update={"source_contract_chain_sha256": FOURTH_HASH})
    digest = attachment_planning_context_carried_sha256(mutated)
    mutated = mutated.model_copy(
        update={
            "attachment_planning_context_id": (f"attachment-planning-context://sha256/{digest}"),
            "attachment_planning_context_sha256": digest,
        }
    )
    with pytest.raises(AttachmentPlanningPolicyError, match="current"):
        bridge.validate_current(
            producer_task_view=view,
            storage_authorization=authorization,
            evidence_bundle=bundle,
            attachment_planning_context=mutated,
        )


def test_context_identity_is_stable_across_audit_timestamps() -> None:
    first_inputs = _inputs(
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    second_inputs = _inputs(
        audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)),
    )
    bridge = AttachmentPlanningBridge()

    first = bridge.compile(
        producer_task_view=first_inputs[0],
        storage_authorization=first_inputs[1],
        evidence_bundle=first_inputs[2],
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    second = bridge.compile(
        producer_task_view=second_inputs[0],
        storage_authorization=second_inputs[1],
        evidence_bundle=second_inputs[2],
        audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)),
    )

    assert first.attachment_planning_context_id == (second.attachment_planning_context_id)
    assert first.attachment_planning_context_sha256 == (second.attachment_planning_context_sha256)
    assert first.canonical_sha256() != second.canonical_sha256()


def test_attachment_planning_identity_is_stable_across_python_hash_seed(
    tmp_path: Path,
) -> None:
    script = tmp_path / "check_attachment_planning_seed.py"
    script.write_text(
        """
from datetime import UTC, datetime
from eval_factory.contracts import (
    AttachmentPlanningContextV2,
    attachment_planning_context_carried_sha256,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

def ref(kind, suffix, version, digest):
    return ObjectRef(
        object_type=kind,
        object_id=f'{kind}://{suffix}',
        object_version=version,
        object_sha256=digest,
    )

audit = ContractAudit(
    created_at=datetime(2026, 7, 27, tzinfo=UTC),
    created_by='seed-test',
    governing_versions=(
        VersionBinding(component='attachment-planning', version='r5-01'),
    ),
)
context = AttachmentPlanningContextV2(
    attachment_planning_context_id='attachment-planning-context://pending',
    producer_task_view_ref=ref('producer-task-view', 'seed', 'v2', 'a' * 64),
    producer_storage_authorization_ref=ref(
        'producer-storage-authorization', 'seed', 'v2', 'b' * 64
    ),
    safe_evidence_bundle_ref=ref('evidence-bundle', 'seed', 'v1', 'c' * 64),
    projection_policy_ref=ref(
        'projection-policy', 'seed', 'evidence-views/r2-05-v1', 'd' * 64
    ),
    source_trace_id='source-trace://attachment-planning/seed',
    trace_ir_version_id='trace-ir://attachment-planning/seed',
    producer_principal_id='principal://attachment-producer/seed',
    source_task_draft_sha256='a' * 64,
    source_contract_chain_sha256='b' * 64,
    policy_version='attachment-planning/r5-01-v1',
    attachment_planning_context_sha256='c' * 64,
    audit=audit,
)
print(attachment_planning_context_carried_sha256(context))
""",
        encoding="utf-8",
    )
    outputs = []
    for seed in ("1", "99"):
        process = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONPATH": "src",
            },
            timeout=30,
        )
        assert process.returncode == 0, process.stderr
        outputs.append(process.stdout.strip())

    assert len(set(outputs)) == 1
