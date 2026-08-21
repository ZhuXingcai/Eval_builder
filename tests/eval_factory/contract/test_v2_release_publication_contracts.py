from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.release_export_v2 import (
    LHWorkspaceExportRequestV2,
    LHWorkspaceExportResultV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration import ItemStatus
from eval_factory.contracts.quality_v2 import final_package_content_sha256
from eval_factory.contracts.release import EvaluationItemComponents, ReleaseChannel
from eval_factory.contracts.release_projection_v2 import (
    EvaluationItemReleaseSubjectV2,
    ItemReleaseProjectionV2,
    ReleaseProjectionPhaseV2,
    ReleaseProjectionResultV2,
    evaluation_item_v2_carried_sha256,
    evaluation_item_v2_ref,
    query_spec_carried_sha256,
    query_spec_ref,
    release_decision_v2_carried_sha256,
    release_decision_v2_ref,
    release_projection_result_v2_ref,
)
from eval_factory.contracts.release_publication_v2 import (
    LHExportReceiptV2,
    LHReleaseItemManifestV2,
    NonProductionRegistryEntryV2,
    NonProductionReleaseManifestV2,
    PublishedItemProjectionV2,
    ReleasePublicationPolicyV2,
    ReleasePublicationResultV2,
    lh_export_receipt_v2_ref,
    release_publication_result_v2_ref,
    validate_release_publication_result_v2_identity,
)
from eval_factory.contracts.release_v2 import (
    EvaluationItemV2,
    ReleaseActionV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)
from eval_factory.contracts.task import QuerySpec

NOW = datetime(2026, 8, 2, tzinfo=UTC)
PUBLISHED_AT = datetime(2026, 8, 2, 1, tzinfo=UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-09/{suffix}",
        object_version=version,
        object_sha256=_digest(f"{object_type}:{suffix}:{version}"),
    )


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="release-publication-test",
        governing_versions=(
            VersionBinding(
                component="release-publication",
                version="release-publication/r7-09-v1",
            ),
        ),
        input_refs=tuple(
            sorted(
                set(refs),
                key=lambda value: (
                    value.object_type,
                    value.object_id,
                    value.object_version,
                    value.object_sha256,
                ),
            )
        ),
    )


def _facade_ref(value: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=value.object_type,
        object_id=value.object_id,
        object_version=value.object_version,
        object_sha256=value.object_sha256,
    )


def _query() -> QuerySpec:
    prompt = "Inspect the supplied input-state workspace."
    value = QuerySpec(
        query_spec_id="query-spec://pending",
        task_draft_ref=_ref("task-draft", "item-a"),
        prompt=prompt,
        attachment_dependency_ids=(),
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        audit=_audit(),
    )
    digest = query_spec_carried_sha256(value)
    return value.model_copy(update={"query_spec_id": f"query-spec://sha256/{digest}"})


def _decision(
    *,
    subject: EvaluationItemReleaseSubjectV2,
    action: ReleaseActionV2,
    state: ReleaseStateV2,
    previous_decision_ref: ObjectRef,
    item_version: str,
    decided_at: datetime,
    approved_item_ref: ObjectRef | None = None,
) -> ReleaseDecisionV2:
    value = ReleaseDecisionV2(
        release_decision_id="release-decision://pending",
        chain_id="release-chain://r7-09/item-a",
        previous_decision_ref=previous_decision_ref,
        item_id=subject.item_id,
        item_version=item_version,
        components=subject.components,
        release_subject_sha256=subject.release_subject_sha256,
        package_manifest_ref=subject.package_manifest_ref,
        package_sha256=subject.package_sha256,
        batch_quality_report_ref=subject.batch_quality_report_ref,
        automated_quality_passed=True,
        open_p0_count=0,
        open_p1_count=0,
        unresolved_non_waivable_count=0,
        user_approval_policy_ref=subject.user_approval_policy_ref,
        required_checkpoints=frozenset(),
        checkpoint_decisions=(),
        channel=subject.channel,
        registry=subject.registry,
        export_profile="LH",
        export_profile_version="v1",
        production_attestation_ref=None,
        action=action,
        state=state,
        idempotency_key=f"decision-{action.value.casefold()}",
        actor="release-publication-test",
        decided_at=decided_at,
        decision_sha256="0" * 64,
        audit=(
            _audit(
                subject.to_ref(),
                previous_decision_ref,
                approved_item_ref,
            )
            if approved_item_ref is not None
            else _audit()
        ),
    )
    digest = release_decision_v2_carried_sha256(value)
    return value.model_copy(
        update={
            "release_decision_id": f"release-decision://sha256/{digest}",
            "decision_sha256": digest,
        }
    )


def _item(
    *,
    subject: EvaluationItemReleaseSubjectV2,
    decision: ReleaseDecisionV2,
    approved_item_ref: ObjectRef | None = None,
) -> EvaluationItemV2:
    value = EvaluationItemV2(
        evaluation_item_id="evaluation-item://pending",
        item_version=decision.item_version,
        source_trace_refs=subject.source_trace_refs,
        label_decision_refs=subject.label_decision_refs,
        task_draft_ref=subject.task_draft_ref,
        attachment_reconstruction_result_ref=(subject.attachment_reconstruction_result_ref),
        components=subject.components,
        user_approval_policy_ref=subject.user_approval_policy_ref,
        user_decision_record_refs=subject.user_decision_record_refs,
        release_decision_ref=release_decision_v2_ref(decision),
        item_sha256="0" * 64,
        audit=(
            _audit(
                approved_item_ref,
                release_decision_v2_ref(decision),
            )
            if approved_item_ref is not None
            else _audit()
        ),
    )
    digest = evaluation_item_v2_carried_sha256(value)
    return value.model_copy(
        update={
            "evaluation_item_id": f"evaluation-item://sha256/{digest}",
            "item_sha256": digest,
        }
    )


def _approved_result() -> ReleaseProjectionResultV2:
    query = _query()
    components = EvaluationItemComponents(
        query_spec_ref=query_spec_ref(query),
        environment_spec_ref=_ref("environment-spec", "item-a"),
        rubric_set_ref=_ref("rubric-set", "item-a"),
        evaluator_spec_ref=_ref("evaluator-spec", "item-a"),
        reference_policy_ref=_ref("reference-policy", "item-a"),
        tool_policy_ref=_ref("tool-policy", "item-a"),
        provenance_manifest_ref=_ref("provenance-manifest", "item-a"),
        quality_report_ref=_ref("quality-report", "item-a"),
    )
    subject = EvaluationItemReleaseSubjectV2.create(
        job_id="job://r7-09/a",
        item_id="item://r7-09/a",
        source_trace_refs=(_ref("trace-source", "item-a", version="v1"),),
        label_decision_refs=(_ref("label-decision", "item-a"),),
        task_draft_ref=query.task_draft_ref,
        attachment_reconstruction_result_ref=_ref(
            "attachment-reconstruction-result",
            "item-a",
        ),
        components=components,
        item_quality_result_ref=_ref(
            "item-quality-compilation-result",
            "item-a",
        ),
        batch_quality_report_ref=_ref("batch-quality-report", "batch-a"),
        package_manifest_ref=_ref("final-package-manifest", "item-a"),
        package_sha256=final_package_content_sha256(()),
        user_approval_policy_ref=_ref("user-approval-policy", "job-a"),
        required_checkpoints=(),
        satisfied_checkpoint_bindings=(),
        relevant_revalidation_application_refs=(),
        revalidation_report_refs=(),
        current_head_refs=(),
        channel=ReleaseChannel.CANARY,
        registry="registry://r7-09/canary",
        export_profile="LH",
        export_profile_version="v1",
        policy_ref=_ref("release-projection-policy", "r7-08"),
        audit=_audit(),
    )
    approved = _decision(
        subject=subject,
        action=ReleaseActionV2.APPROVE,
        state=ReleaseStateV2.APPROVED,
        previous_decision_ref=_ref("release-decision", "candidate"),
        item_version="2",
        decided_at=NOW,
    )
    item = _item(subject=subject, decision=approved)
    projection = ItemReleaseProjectionV2.create(
        job_id=subject.job_id,
        item_id=subject.item_id,
        projection_revision=2,
        chain_id=approved.chain_id,
        previous_projection_ref=_ref("item-release-projection", "candidate"),
        release_subject_ref=subject.to_ref(),
        evaluation_item_ref=evaluation_item_v2_ref(item),
        release_decision_ref=release_decision_v2_ref(approved),
        release_state=ReleaseStateV2.APPROVED,
        pending_checkpoints=(),
        item_status=ItemStatus.APPROVED,
        audit=_audit(),
    )
    return ReleaseProjectionResultV2.create(
        phase=ReleaseProjectionPhaseV2.TERMINAL,
        release_subject=subject,
        query_spec=query,
        release_decision=approved,
        evaluation_item=item,
        item_projection=projection,
        previous_result_ref=_ref("release-projection-result", "candidate"),
        policy_ref=subject.policy_ref,
        audit=_audit(),
    )


def _publication() -> ReleasePublicationResultV2:
    source = _approved_result()
    policy = ReleasePublicationPolicyV2.create(
        max_items=10,
        max_workspace_members_per_item=100,
        max_workspace_bytes_per_item=1_000_000,
        max_manifest_refs=1_000,
        canary_registry_ids=frozenset({source.release_decision.registry}),
        internal_review_registry_ids=(frozenset({"registry://r7-09/internal"})),
        audit=_audit(),
    )
    item_manifest = LHReleaseItemManifestV2.create(
        job_id=source.release_subject.job_id,
        item_id=source.release_subject.item_id,
        approved_release_result_ref=release_projection_result_v2_ref(source),
        approved_evaluation_item_ref=source.evaluation_item_ref,
        approved_release_decision_ref=source.release_decision_ref,
        release_subject_ref=source.release_subject_ref,
        query_spec_ref=source.query_spec_ref,
        rubric_set_ref=source.release_subject.components.rubric_set_ref,
        environment_spec_ref=(source.release_subject.components.environment_spec_ref),
        provenance_manifest_ref=(source.release_subject.components.provenance_manifest_ref),
        quality_report_ref=source.release_subject.components.quality_report_ref,
        final_package_manifest_ref=source.release_subject.package_manifest_ref,
        package_sha256=source.release_subject.package_sha256,
        query_yaml_sha256=_digest("query"),
        rubrics_json_sha256=_digest("rubrics"),
        expected_workspace_members=(),
        channel=source.release_decision.channel,
        registry=source.release_decision.registry,
        audit=_audit(),
    )
    manifest = NonProductionReleaseManifestV2.create(
        job_id=source.release_subject.job_id,
        approved_item_ids=(source.release_subject.item_id,),
        rejected_item_ids=(),
        item_manifests=(item_manifest,),
        channel=source.release_decision.channel,
        registry=source.release_decision.registry,
        base_contract_manifest_ref=_ref(
            "contract-manifest",
            "v1",
            version="manifest/v1",
        ),
        overlay_contract_manifest_ref=_ref(
            "contract-manifest",
            "v2",
            version="manifest/v2",
        ),
        release_profile_decision_ref=_ref(
            "release-profile-decision",
            "lh-v1",
            version="decision/v1",
        ),
        policy_ref=policy.to_ref(),
        audit=_audit(),
    )
    workspace_request = LHWorkspaceExportRequestV2.create(
        item_id=source.release_subject.item_id,
        final_package_manifest_ref=_facade_ref(source.release_subject.package_manifest_ref),
        package_sha256=source.release_subject.package_sha256,
        output_refs=(),
        members=(),
        max_member_count=100,
        max_total_bytes=1_000_000,
    )
    workspace_result = LHWorkspaceExportResultV2.exported(
        request=workspace_request,
        observed_members=(),
        copied_output_refs=(),
    )
    receipt = LHExportReceiptV2.create(
        release_manifest_ref=manifest.to_ref(),
        item_manifest_ref=item_manifest.to_ref(),
        workspace_export_result=workspace_result,
        query_yaml_sha256=item_manifest.query_yaml_sha256,
        rubrics_json_sha256=item_manifest.rubrics_json_sha256,
        bundle_sha256=_digest("bundle"),
        bundle_file_count=3,
        bundle_total_bytes=512,
        policy_ref=policy.to_ref(),
        audit=_audit(),
    )
    published_decision = _decision(
        subject=source.release_subject,
        action=ReleaseActionV2.PUBLISH,
        state=ReleaseStateV2.RELEASED,
        previous_decision_ref=source.release_decision_ref,
        item_version="3",
        decided_at=PUBLISHED_AT,
        approved_item_ref=source.evaluation_item_ref,
    )
    published_item = _item(
        subject=source.release_subject,
        decision=published_decision,
        approved_item_ref=source.evaluation_item_ref,
    )
    registry = NonProductionRegistryEntryV2.create(
        release_manifest_ref=manifest.to_ref(),
        job_id=manifest.job_id,
        channel=manifest.channel,
        registry=manifest.registry,
        item_ids=manifest.approved_item_ids,
        item_manifest_refs=manifest.item_manifest_refs,
        export_receipt_refs=(receipt.to_ref(),),
        published_release_decision_refs=(release_decision_v2_ref(published_decision),),
        published_evaluation_item_refs=(evaluation_item_v2_ref(published_item),),
        bundle_sha256s=(receipt.bundle_sha256,),
        published_at=PUBLISHED_AT.isoformat(),
        policy_ref=policy.to_ref(),
        audit=_audit(),
    )
    projection = PublishedItemProjectionV2.create(
        job_id=manifest.job_id,
        item_id=source.release_subject.item_id,
        chain_id=published_decision.chain_id,
        approved_result_ref=release_projection_result_v2_ref(source),
        approved_projection_ref=source.item_projection_ref,
        release_manifest_ref=manifest.to_ref(),
        export_receipt_ref=receipt.to_ref(),
        registry_entry_ref=registry.to_ref(),
        publish_decision_ref=release_decision_v2_ref(published_decision),
        published_evaluation_item_ref=evaluation_item_v2_ref(published_item),
        audit=_audit(),
    )
    return ReleasePublicationResultV2.create(
        release_manifest=manifest,
        export_receipts=(receipt,),
        published_decisions=(published_decision,),
        published_items=(published_item,),
        item_projections=(projection,),
        registry_entry=registry,
        approved_source_result_refs=(release_projection_result_v2_ref(source),),
        policy_ref=policy.to_ref(),
        audit=_audit(),
    )


def test_publication_contracts_are_strict_frozen_and_content_addressed() -> None:
    result = _publication()

    validate_release_publication_result_v2_identity(result)
    assert result.registry_entry.channel is ReleaseChannel.CANARY
    assert result.published_decisions[0].action is ReleaseActionV2.PUBLISH
    assert (
        result.published_items[0].release_decision_ref
        == (result.registry_entry.published_release_decision_refs[0])
    )
    assert release_publication_result_v2_ref(result).object_sha256 == (result.result_sha256)

    with pytest.raises(ValidationError):
        ReleasePublicationResultV2.model_validate(
            {
                **result.model_dump(mode="python"),
                "unknown": True,
            }
        )
    with pytest.raises(ValidationError):
        result.phase = "CANDIDATE"  # type: ignore[assignment]


def test_publication_rejects_nested_identity_and_audit_drift() -> None:
    result = _publication()
    stale_item_manifest = result.release_manifest.item_manifests[0].model_copy(
        update={"registry": "registry://r7-09/other"}
    )
    stale_manifest = result.release_manifest.model_copy(update={"item_manifests": (stale_item_manifest,)})

    with pytest.raises(ValidationError, match="identity"):
        NonProductionReleaseManifestV2.model_validate(stale_manifest.model_dump(mode="python"))

    stale_audit = result.registry_entry.model_copy(update={"audit": _audit()})
    with pytest.raises(ValidationError, match="audit refs"):
        NonProductionRegistryEntryV2.model_validate(stale_audit.model_dump(mode="python"))


def test_publication_rejects_rehashed_receipt_binding_drift() -> None:
    result = _publication()
    receipt = result.export_receipts[0]
    stale_receipt = LHExportReceiptV2.create(
        release_manifest_ref=receipt.release_manifest_ref,
        item_manifest_ref=receipt.item_manifest_ref,
        workspace_export_result=receipt.workspace_export_result,
        query_yaml_sha256=_digest("different-query"),
        rubrics_json_sha256=receipt.rubrics_json_sha256,
        bundle_sha256=receipt.bundle_sha256,
        bundle_file_count=receipt.bundle_file_count,
        bundle_total_bytes=receipt.bundle_total_bytes,
        policy_ref=receipt.policy_ref,
        audit=_audit(),
    )
    assert lh_export_receipt_v2_ref(stale_receipt) != lh_export_receipt_v2_ref(receipt)
    registry = result.registry_entry
    rebound_registry = NonProductionRegistryEntryV2.create(
        release_manifest_ref=registry.release_manifest_ref,
        job_id=registry.job_id,
        channel=registry.channel,
        registry=registry.registry,
        item_ids=registry.item_ids,
        item_manifest_refs=registry.item_manifest_refs,
        export_receipt_refs=(stale_receipt.to_ref(),),
        published_release_decision_refs=(registry.published_release_decision_refs),
        published_evaluation_item_refs=(registry.published_evaluation_item_refs),
        bundle_sha256s=registry.bundle_sha256s,
        published_at=registry.published_at,
        policy_ref=registry.policy_ref,
        audit=_audit(),
    )

    with pytest.raises(ValidationError, match="receipt differs"):
        ReleasePublicationResultV2.model_validate(
            result.model_copy(
                update={
                    "export_receipts": (stale_receipt,),
                    "registry_entry": rebound_registry,
                }
            ).model_dump(mode="python")
        )


def test_publication_rejects_rehashed_publish_authority_drift() -> None:
    result = _publication()
    decision = result.published_decisions[0]
    changed_components = decision.components.model_copy(
        update={"quality_report_ref": _ref("quality-report", "changed")}
    )
    pending = decision.model_copy(
        update={
            "release_decision_id": "release-decision://pending",
            "components": changed_components,
            "decision_sha256": "0" * 64,
        }
    )
    digest = release_decision_v2_carried_sha256(pending)
    changed = pending.model_copy(
        update={
            "release_decision_id": f"release-decision://sha256/{digest}",
            "decision_sha256": digest,
        }
    )
    registry = result.registry_entry
    rebound_registry = NonProductionRegistryEntryV2.create(
        release_manifest_ref=registry.release_manifest_ref,
        job_id=registry.job_id,
        channel=registry.channel,
        registry=registry.registry,
        item_ids=registry.item_ids,
        item_manifest_refs=registry.item_manifest_refs,
        export_receipt_refs=registry.export_receipt_refs,
        published_release_decision_refs=(release_decision_v2_ref(changed),),
        published_evaluation_item_refs=(registry.published_evaluation_item_refs),
        bundle_sha256s=registry.bundle_sha256s,
        published_at=registry.published_at,
        policy_ref=registry.policy_ref,
        audit=_audit(),
    )

    with pytest.raises(ValidationError, match="approved authority"):
        ReleasePublicationResultV2.model_validate(
            result.model_copy(
                update={
                    "published_decisions": (changed,),
                    "registry_entry": rebound_registry,
                }
            ).model_dump(mode="python")
        )


def test_publication_policy_and_registry_have_no_production_surface() -> None:
    with pytest.raises(ValidationError, match="disjoint"):
        ReleasePublicationPolicyV2.create(
            max_items=1,
            max_workspace_members_per_item=1,
            max_workspace_bytes_per_item=1,
            max_manifest_refs=1,
            canary_registry_ids=frozenset({"registry://shared"}),
            internal_review_registry_ids=frozenset({"registry://shared"}),
            audit=_audit(),
        )
    with pytest.raises(ValidationError, match="production"):
        ReleasePublicationPolicyV2.create(
            max_items=1,
            max_workspace_members_per_item=1,
            max_workspace_bytes_per_item=1,
            max_manifest_refs=1,
            canary_registry_ids=frozenset({"registry://production"}),
            internal_review_registry_ids=frozenset({"registry://internal"}),
            audit=_audit(),
        )

    payload = _publication().model_dump_json()
    for forbidden in (
        "credential",
        "final_output",
        "grader_rule",
        "private_reference",
        "provider_payload",
        "raw_trace",
    ):
        assert forbidden not in payload
