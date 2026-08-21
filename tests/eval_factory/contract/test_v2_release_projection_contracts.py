from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.approval import ApprovalCheckpoint
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration import ItemStatus
from eval_factory.contracts.release import EvaluationItemComponents, ReleaseChannel
from eval_factory.contracts.release_projection_v2 import (
    EvaluationItemReleaseSubjectV2,
    ItemReleaseProjectionV2,
    ReleaseProjectionPhaseV2,
    ReleaseProjectionPolicyV2,
    ReleaseProjectionResultV2,
    evaluation_item_v2_carried_sha256,
    evaluation_item_v2_ref,
    query_spec_carried_sha256,
    query_spec_ref,
    release_decision_v2_carried_sha256,
    release_decision_v2_ref,
)
from eval_factory.contracts.release_v2 import (
    EvaluationItemV2,
    ReleaseActionV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)
from eval_factory.contracts.task import QuerySpec

NOW = datetime(2026, 8, 2, tzinfo=UTC)


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str | None = None,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-08/{suffix}",
        object_version=version,
        object_sha256=digest or (suffix[0] * 64),
    )


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="release-projection-contract-test",
        governing_versions=(
            VersionBinding(
                component="release-projection",
                version="release-projection/r7-08-v1",
            ),
        ),
        input_refs=tuple(sorted(set(refs), key=lambda value: value.object_id)),
    )


def _policy() -> ReleaseProjectionPolicyV2:
    return ReleaseProjectionPolicyV2.create(
        max_source_trace_refs=4,
        max_label_decision_refs=8,
        max_user_decision_refs=8,
        max_revalidation_reports=8,
        max_current_head_refs=64,
        max_chain_depth=8,
        allowed_channels=frozenset(
            {
                ReleaseChannel.CANARY,
                ReleaseChannel.INTERNAL_REVIEW,
            }
        ),
        allowed_export_profiles=frozenset({"LH"}),
        audit=_audit(),
    )


def _components() -> EvaluationItemComponents:
    return EvaluationItemComponents(
        query_spec_ref=_ref("query-spec", "a"),
        environment_spec_ref=_ref("environment-spec", "b"),
        rubric_set_ref=_ref("rubric-set", "c"),
        evaluator_spec_ref=_ref("evaluator-spec", "d"),
        reference_policy_ref=_ref("reference-policy", "e"),
        tool_policy_ref=_ref("tool-policy", "f"),
        provenance_manifest_ref=_ref("provenance-manifest", "1"),
        quality_report_ref=_ref("quality-report", "2"),
    )


def _subject(
    *,
    required: tuple[ApprovalCheckpoint, ...] = (ApprovalCheckpoint.FINAL_DATASET_REVIEW,),
    revalidation_report_refs: tuple[ObjectRef, ...] = (),
    relevant_application_refs: tuple[ObjectRef, ...] = (),
    current_head_refs: tuple[ObjectRef, ...] = (),
    query_ref: ObjectRef | None = None,
) -> EvaluationItemReleaseSubjectV2:
    components = _components()
    if query_ref is not None:
        components = components.model_copy(update={"query_spec_ref": query_ref})
    return EvaluationItemReleaseSubjectV2.create(
        job_id="job://r7-08/contracts",
        item_id="item://r7-08/contracts",
        source_trace_refs=(_ref("trace-source", "3", version="raw_traj_v1"),),
        label_decision_refs=(_ref("label-decision", "4"),),
        task_draft_ref=_ref("task-draft", "5"),
        attachment_reconstruction_result_ref=_ref(
            "attachment-reconstruction-result",
            "6",
        ),
        components=components,
        item_quality_result_ref=_ref("item-quality-compilation-result", "7"),
        batch_quality_report_ref=_ref("batch-quality-report", "8"),
        package_manifest_ref=_ref("final-package-manifest", "9"),
        package_sha256="a" * 64,
        user_approval_policy_ref=_ref("user-approval-policy", "b"),
        required_checkpoints=required,
        satisfied_checkpoint_bindings=(),
        relevant_revalidation_application_refs=relevant_application_refs,
        revalidation_report_refs=revalidation_report_refs,
        current_head_refs=current_head_refs,
        channel=ReleaseChannel.CANARY,
        registry="registry://r7-08/contracts",
        export_profile="LH",
        export_profile_version="v1",
        policy_ref=_policy().to_ref(),
        audit=_audit(),
    )


def _query(subject: EvaluationItemReleaseSubjectV2) -> QuerySpec:
    value = QuerySpec(
        query_spec_id="query-spec://pending",
        task_draft_ref=subject.task_draft_ref,
        prompt="Inspect the supplied workspace and report its structure.",
        attachment_dependency_ids=("attachment-dependency://r7-08/contracts",),
        prompt_sha256="0" * 64,
        audit=_audit(subject.task_draft_ref),
    )
    prompt_hash = __import__("hashlib").sha256(value.prompt.encode()).hexdigest()
    value = value.model_copy(update={"prompt_sha256": prompt_hash})
    digest = query_spec_carried_sha256(value)
    return value.model_copy(update={"query_spec_id": f"query-spec://sha256/{digest}"})


def _candidate_values() -> tuple[
    ReleaseProjectionPolicyV2,
    EvaluationItemReleaseSubjectV2,
    QuerySpec,
    ReleaseDecisionV2,
    EvaluationItemV2,
    ItemReleaseProjectionV2,
]:
    policy = _policy()
    subject = _subject()
    query = _query(subject)
    subject = _subject(query_ref=query_spec_ref(query))
    components = subject.components.model_copy(update={"query_spec_ref": query_spec_ref(query)})
    decision = ReleaseDecisionV2(
        release_decision_id="release-decision://pending",
        chain_id="release-chain://r7-08/contracts",
        previous_decision_ref=None,
        item_id=subject.item_id,
        item_version="1",
        components=components,
        release_subject_sha256=subject.release_subject_sha256,
        package_manifest_ref=subject.package_manifest_ref,
        package_sha256=subject.package_sha256,
        batch_quality_report_ref=subject.batch_quality_report_ref,
        automated_quality_passed=True,
        open_p0_count=0,
        open_p1_count=0,
        unresolved_non_waivable_count=0,
        user_approval_policy_ref=subject.user_approval_policy_ref,
        required_checkpoints=frozenset(subject.required_checkpoints),
        checkpoint_decisions=(),
        channel=subject.channel,
        registry=subject.registry,
        export_profile=subject.export_profile,
        export_profile_version=subject.export_profile_version,
        production_attestation_ref=None,
        action=ReleaseActionV2.REQUEST_RELEASE,
        state=ReleaseStateV2.CANDIDATE,
        idempotency_key="release-candidate-r7-08-contracts",
        actor="release-projection-contract-test",
        decided_at=NOW,
        decision_sha256="0" * 64,
        audit=_audit(subject.to_ref(), query_spec_ref(query)),
    )
    decision_digest = release_decision_v2_carried_sha256(decision)
    decision = decision.model_copy(
        update={
            "release_decision_id": f"release-decision://sha256/{decision_digest}",
            "decision_sha256": decision_digest,
        }
    )
    item = EvaluationItemV2(
        evaluation_item_id="evaluation-item://pending",
        item_version="1",
        source_trace_refs=subject.source_trace_refs,
        label_decision_refs=subject.label_decision_refs,
        task_draft_ref=subject.task_draft_ref,
        attachment_reconstruction_result_ref=(subject.attachment_reconstruction_result_ref),
        components=components,
        user_approval_policy_ref=subject.user_approval_policy_ref,
        user_decision_record_refs=(),
        release_decision_ref=release_decision_v2_ref(decision),
        item_sha256="0" * 64,
        audit=_audit(subject.to_ref(), release_decision_v2_ref(decision)),
    )
    item_digest = evaluation_item_v2_carried_sha256(item)
    item = item.model_copy(
        update={
            "evaluation_item_id": f"evaluation-item://sha256/{item_digest}",
            "item_sha256": item_digest,
        }
    )
    projection = ItemReleaseProjectionV2.create(
        job_id=subject.job_id,
        item_id=subject.item_id,
        projection_revision=1,
        chain_id=decision.chain_id,
        previous_projection_ref=None,
        release_subject_ref=subject.to_ref(),
        evaluation_item_ref=evaluation_item_v2_ref(item),
        release_decision_ref=release_decision_v2_ref(decision),
        release_state=ReleaseStateV2.CANDIDATE,
        pending_checkpoints=subject.required_checkpoints,
        audit=_audit(),
    )
    return policy, subject, query, decision, item, projection


def test_release_projection_contracts_are_strict_frozen_and_content_addressed() -> None:
    policy, subject, query, decision, item, projection = _candidate_values()
    result = ReleaseProjectionResultV2.create(
        phase=ReleaseProjectionPhaseV2.CANDIDATE,
        release_subject=subject,
        query_spec=query,
        release_decision=decision,
        evaluation_item=item,
        item_projection=projection,
        previous_result_ref=None,
        policy_ref=policy.to_ref(),
        audit=_audit(),
    )

    assert result.query_spec_ref == query_spec_ref(query)
    assert result.release_decision_ref == release_decision_v2_ref(decision)
    assert result.evaluation_item_ref == evaluation_item_v2_ref(item)
    assert result.release_subject_ref == subject.to_ref()

    with pytest.raises(ValidationError):
        ReleaseProjectionPolicyV2.model_validate({**policy.model_dump(mode="python"), "unknown": True})
    with pytest.raises(ValidationError):
        policy.max_chain_depth = 99  # type: ignore[misc]


def test_subject_requires_canonical_complete_revalidation_coverage() -> None:
    first_app = _ref("user-plan-application", "c")
    second_app = _ref("user-plan-application", "d")
    first_report = _ref("directed-revalidation-report", "e")
    second_report = _ref("directed-revalidation-report", "f")
    head = _ref("quality-report", "a")
    subject = _subject(
        relevant_application_refs=(first_app, second_app),
        revalidation_report_refs=(first_report, second_report),
        current_head_refs=(head,),
    )
    assert subject.revalidation_report_refs == (first_report, second_report)

    values = subject.model_dump(mode="python")
    values["revalidation_report_refs"] = (second_report, first_report)
    with pytest.raises(ValidationError, match="sorted"):
        EvaluationItemReleaseSubjectV2.model_validate(values)

    values = subject.model_dump(mode="python")
    values["revalidation_report_refs"] = (first_report,)
    with pytest.raises(ValidationError, match="coverage"):
        EvaluationItemReleaseSubjectV2.model_validate(values)


def test_projection_state_mapping_rejects_release_authority() -> None:
    _, subject, _, decision, item, _ = _candidate_values()
    with pytest.raises(ValidationError, match="pending"):
        ItemReleaseProjectionV2.create(
            job_id=subject.job_id,
            item_id=subject.item_id,
            projection_revision=1,
            chain_id=decision.chain_id,
            previous_projection_ref=None,
            release_subject_ref=subject.to_ref(),
            evaluation_item_ref=evaluation_item_v2_ref(item),
            release_decision_ref=release_decision_v2_ref(decision),
            release_state=ReleaseStateV2.CANDIDATE,
            pending_checkpoints=(),
            item_status=ItemStatus.NEEDS_REVIEW,
            audit=_audit(),
        )
    with pytest.raises((ValidationError, ValueError), match="R7-08"):
        ItemReleaseProjectionV2.create(
            job_id=subject.job_id,
            item_id=subject.item_id,
            projection_revision=1,
            chain_id=decision.chain_id,
            previous_projection_ref=None,
            release_subject_ref=subject.to_ref(),
            evaluation_item_ref=evaluation_item_v2_ref(item),
            release_decision_ref=release_decision_v2_ref(decision),
            release_state=ReleaseStateV2.RELEASED,
            pending_checkpoints=(),
            audit=_audit(),
        )


def test_result_revalidates_nested_query_and_release_identities() -> None:
    policy, subject, query, decision, item, projection = _candidate_values()
    tampered = query.model_copy(update={"prompt": "Caller-mutated prompt."})
    with pytest.raises((ValidationError, ValueError), match="QuerySpec"):
        ReleaseProjectionResultV2.create(
            phase=ReleaseProjectionPhaseV2.CANDIDATE,
            release_subject=subject,
            query_spec=tampered,
            release_decision=decision,
            evaluation_item=item,
            item_projection=projection,
            previous_result_ref=None,
            policy_ref=policy.to_ref(),
            audit=_audit(),
        )

    stale_decision = decision.model_copy(update={"release_subject_sha256": "f" * 64})
    with pytest.raises((ValidationError, ValueError), match="ReleaseDecision"):
        ReleaseProjectionResultV2.create(
            phase=ReleaseProjectionPhaseV2.CANDIDATE,
            release_subject=subject,
            query_spec=query,
            release_decision=stale_decision,
            evaluation_item=item,
            item_projection=projection,
            previous_result_ref=None,
            policy_ref=policy.to_ref(),
            audit=_audit(),
        )


def test_release_projection_schemas_exclude_sensitive_surfaces() -> None:
    schemas = (
        ReleaseProjectionPolicyV2.model_json_schema(),
        EvaluationItemReleaseSubjectV2.model_json_schema(),
        ItemReleaseProjectionV2.model_json_schema(),
        ReleaseProjectionResultV2.model_json_schema(),
    )
    serialized = str(schemas).casefold()
    for forbidden in (
        "raw_trace",
        "final_output",
        "private_reference",
        "grader_rule",
        "hidden_condition",
        "credential",
        "provider_payload",
        "exception_text",
        "registry_write",
    ):
        assert forbidden not in serialized
