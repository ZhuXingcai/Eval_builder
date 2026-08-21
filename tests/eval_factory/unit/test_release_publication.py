from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from test_duplicate_detection import _quality_result
from test_rubric_authoring import (
    _draft as _rubric_draft,
)
from test_rubric_authoring import (
    _generated_proposal as _rubric_proposal,
)
from test_rubric_authoring import (
    _request as _rubric_request,
)

from env_mock_agent.facade import LHWorkspaceExportResultV2
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    EnvironmentScopeDecision,
    EnvironmentStrategyChoice,
    QueryPackagingChoice,
    UserDecision,
    UserDecisionRecord,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserDecisionCommitOutcomeV2,
    UserDecisionCommitResultV2,
    user_decision_record_carried_sha256,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration import ItemStatus
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
    environment_spec_v2_ref,
    final_package_manifest_ref,
    item_quality_compilation_result_ref,
    provenance_manifest_v2_ref,
    quality_report_v2_ref,
)
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
)
from eval_factory.contracts.release_publication_v2 import (
    ReleasePublicationPolicyV2,
)
from eval_factory.contracts.release_v2 import (
    CheckpointDecisionBinding,
    EvaluationItemV2,
    ReleaseActionV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)
from eval_factory.contracts.task import QuerySpec
from eval_factory.contracts.task_v2 import (
    R4TaskContractSetV2,
    RubricSetV2,
    r4_task_contract_set_carried_sha256,
    rubric_set_ref,
)
from eval_factory.dataset.export import (
    ReleaseBundleFacts,
    ReleasePublicationCompiler,
    ReleasePublicationItemSource,
    ReleasePublicationManifestCompilation,
    ReleasePublicationPolicyError,
)
from eval_factory.task_authoring import RubricSetCompiler

NOW = datetime(2026, 8, 2, tzinfo=UTC)
PUBLISHED_AT = datetime(2026, 8, 2, 1, tzinfo=UTC)
GOLD_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals/golden/eval_factory/release"
    / "r7-09-lh-nonproduction-publication-v1.json"
)


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


def _rubric() -> RubricSetV2:
    draft = _rubric_draft(attachment_dependencies=())
    request = _rubric_request(draft)
    result = RubricSetCompiler().compile(
        request=request,
        proposal=_rubric_proposal(request),
        task_draft=draft,
        audit=_audit(),
    )
    assert result.rubric_set is not None
    return result.rubric_set


def _contract_set(rubric: RubricSetV2) -> R4TaskContractSetV2:
    value = R4TaskContractSetV2(
        contract_set_id="r4-task-contract-set://pending",
        task_draft_ref=rubric.task_draft_ref,
        task_prompt_safety_gate_ref=_ref("task-prompt-safety-gate", "item-a"),
        rubric_set_ref=rubric_set_ref(rubric),
        evaluator_spec_ref=_ref("evaluator-spec", "item-a"),
        reference_policy_ref=_ref("reference-policy", "item-a"),
        tool_policy_ref=_ref("tool-policy", "item-a"),
        contestant_tool_policy_ref=_ref("contestant-tool-policy", "item-a"),
        producer_storage_authorization_ref=_ref(
            "producer-storage-authorization",
            "item-a",
        ),
        producer_task_view_ref=_ref("producer-task-view", "item-a"),
        contract_set_sha256="0" * 64,
        audit=_audit(),
    )
    digest = r4_task_contract_set_carried_sha256(value)
    return value.model_copy(
        update={
            "contract_set_id": f"r4-task-contract-set://sha256/{digest}",
            "contract_set_sha256": digest,
        }
    )


def _quality(rubric: RubricSetV2) -> ItemQualityCompilationResultV2:
    return _quality_result(
        suffix="r7-09-item-a",
        contract_set=_contract_set(rubric),
        attachments=(),
    )


def _environment_commit(
    *,
    choice: QueryPackagingChoice,
) -> UserDecisionCommitResultV2:
    request_ref = _ref("user-approval-request", choice.value.casefold())
    approval_policy_ref = _ref("user-approval-policy", "job-a")
    plan_ref = _ref("environment-strategy", "job-a")
    record = UserDecisionRecord(
        decision_record_id="user-decision-record://pending",
        request_ref=request_ref,
        checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        approval_policy_ref=approval_policy_ref,
        subject_refs=(_ref("task-draft", "environment-subject"),),
        plan_ref=plan_ref,
        projection_ref=_ref("user-approval-projection", "environment"),
        authenticated_user="requesting-user",
        decision=UserDecision.ACCEPT,
        environment_decisions=(
            EnvironmentScopeDecision(
                requirement_id="environment-requirement://item-a",
                strategy=EnvironmentStrategyChoice.TRACE_FAITHFUL_MOCK,
            ),
        ),
        query_packaging=choice,
        reason="Approved environment strategy.",
        hard_gate_override_requested=False,
        idempotency_key=f"environment-{choice.value.casefold()}",
        decided_at=NOW,
        record_sha256="0" * 64,
    )
    digest = user_decision_record_carried_sha256(record)
    record = record.model_copy(
        update={
            "decision_record_id": f"user-decision-record://sha256/{digest}",
            "record_sha256": digest,
        }
    )
    return UserDecisionCommitResultV2.create(
        job_id="job://r7-09/a",
        dataset_job_spec_ref=_ref("dataset-job-spec", "job-a"),
        request_compilation_result_ref=_ref(
            "user-approval-request-compilation-result",
            "environment",
        ),
        request_ref=request_ref,
        decision_policy_ref=_ref(
            "user-decision-handling-policy",
            "environment",
        ),
        authentication_context_ref=_ref(
            "authenticated-user-context",
            "requesting-user",
        ),
        decision_record=record,
        adjustment_effect=None,
        request_revision=None,
        outcome=UserDecisionCommitOutcomeV2.ACCEPTED,
        audit=_audit(),
    )


def _query(rubric: RubricSetV2) -> QuerySpec:
    prompt = "Inspect the supplied input-state workspace."
    value = QuerySpec(
        query_spec_id="query-spec://pending",
        task_draft_ref=rubric.task_draft_ref,
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
    previous_decision_ref: ObjectRef | None,
    item_version: str,
    checkpoint_bindings: tuple[CheckpointDecisionBinding, ...],
    decided_at: datetime,
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
        required_checkpoints=frozenset(subject.required_checkpoints),
        checkpoint_decisions=checkpoint_bindings,
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
        audit=_audit(),
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
        audit=_audit(),
    )
    digest = evaluation_item_v2_carried_sha256(value)
    return value.model_copy(
        update={
            "evaluation_item_id": f"evaluation-item://sha256/{digest}",
            "item_sha256": digest,
        }
    )


def _approved_result(
    *,
    rubric: RubricSetV2,
    quality: ItemQualityCompilationResultV2,
    environment_commit: UserDecisionCommitResultV2 | None = None,
    release_projection_policy_ref: ObjectRef | None = None,
) -> ReleaseProjectionResultV2:
    query = _query(rubric)
    package = quality.final_package_manifest
    provenance = quality.provenance_manifest
    environment = quality.environment_spec
    assert package is not None
    assert provenance is not None
    assert environment is not None
    components = EvaluationItemComponents(
        query_spec_ref=query_spec_ref(query),
        environment_spec_ref=environment_spec_v2_ref(environment),
        rubric_set_ref=rubric_set_ref(rubric),
        evaluator_spec_ref=_ref("evaluator-spec", "item-a"),
        reference_policy_ref=_ref("reference-policy", "item-a"),
        tool_policy_ref=_ref("tool-policy", "item-a"),
        provenance_manifest_ref=provenance_manifest_v2_ref(provenance),
        quality_report_ref=quality_report_v2_ref(quality.quality_report),
    )
    bindings = (
        (
            CheckpointDecisionBinding(
                checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
                request_ref=environment_commit.request_ref,
                decision_record_ref=environment_commit.decision_record_ref,
            ),
        )
        if environment_commit is not None
        else ()
    )
    decision_refs = (environment_commit.decision_record_ref,) if environment_commit is not None else ()
    subject = EvaluationItemReleaseSubjectV2.create(
        job_id="job://r7-09/a",
        item_id="item://r7-09/a",
        source_trace_refs=(_ref("trace-source", "item-a", version="v1"),),
        label_decision_refs=(_ref("label-decision", "item-a"),),
        task_draft_ref=rubric.task_draft_ref,
        attachment_reconstruction_result_ref=_ref(
            "attachment-reconstruction-result",
            "item-a",
        ),
        components=components,
        item_quality_result_ref=item_quality_compilation_result_ref(quality),
        batch_quality_report_ref=_ref("batch-quality-report", "batch-a"),
        package_manifest_ref=final_package_manifest_ref(package),
        package_sha256=package.package_sha256,
        user_approval_policy_ref=_ref("user-approval-policy", "job-a"),
        required_checkpoints=(
            (ApprovalCheckpoint.ENVIRONMENT_STRATEGY,) if environment_commit is not None else ()
        ),
        satisfied_checkpoint_bindings=bindings,
        user_decision_record_refs=decision_refs,
        relevant_revalidation_application_refs=(),
        revalidation_report_refs=(),
        current_head_refs=(),
        channel=ReleaseChannel.CANARY,
        registry="registry://r7-09/canary",
        export_profile="LH",
        export_profile_version="v1",
        policy_ref=release_projection_policy_ref or _ref("release-projection-policy", "r7-08"),
        audit=_audit(),
    )
    approved = _decision(
        subject=subject,
        action=ReleaseActionV2.APPROVE,
        state=ReleaseStateV2.APPROVED,
        previous_decision_ref=_ref("release-decision", "candidate"),
        item_version="2",
        checkpoint_bindings=bindings,
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


def _policy(
    *,
    max_manifest_refs: int = 1_000,
) -> ReleasePublicationPolicyV2:
    return ReleasePublicationPolicyV2.create(
        max_items=10,
        max_workspace_members_per_item=100,
        max_workspace_bytes_per_item=1_000_000,
        max_manifest_refs=max_manifest_refs,
        canary_registry_ids=frozenset({"registry://r7-09/canary"}),
        internal_review_registry_ids=frozenset({"registry://r7-09/internal"}),
        audit=_audit(),
    )


def _source(
    *,
    choice: QueryPackagingChoice | None = None,
    release_projection_policy_ref: ObjectRef | None = None,
) -> tuple[ReleasePublicationItemSource, UserDecisionCommitResultV2 | None]:
    rubric = _rubric()
    quality = _quality(rubric)
    commit = _environment_commit(choice=choice) if choice is not None else None
    result = _approved_result(
        rubric=rubric,
        quality=quality,
        environment_commit=commit,
        release_projection_policy_ref=release_projection_policy_ref,
    )
    return (
        ReleasePublicationItemSource(
            approved_result=result,
            rubric_set=rubric,
            item_quality=quality,
            query_packaging_decision=commit,
        ),
        commit,
    )


def _compile_manifest(
    source: ReleasePublicationItemSource,
    *,
    policy: ReleasePublicationPolicyV2 | None = None,
    rejected_item_ids: tuple[str, ...] = ("item://r7-09/rejected",),
) -> ReleasePublicationManifestCompilation:
    return ReleasePublicationCompiler().compile_manifest(
        job_id=source.approved_result.release_subject.job_id,
        item_sources=(source,),
        rejected_item_ids=rejected_item_ids,
        policy=policy or _policy(),
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
        audit=_audit(),
    )


def test_compile_manifest_defaults_to_strict_lh_query_and_safe_serialization() -> None:
    source, _ = _source()

    compilation = _compile_manifest(source)

    assert compilation.release_manifest.approved_item_ids == (source.approved_result.release_subject.item_id,)
    assert compilation.rejected_item_ids == ("item://r7-09/rejected",)
    item = compilation.items[0]
    query = yaml.safe_load(item.query_yaml_bytes)
    rubrics = json.loads(item.rubrics_json_bytes)
    assert query["prompt"] == source.approved_result.query_spec.prompt
    assert query["profile"] == "LH"
    assert query["profile_version"] == "v1"
    assert rubrics["rubric_set_id"] == source.rubric_set.rubric_set_id
    assert rubrics["criteria"][0]["evaluator_binding"]
    serialized = item.query_yaml_bytes + item.rubrics_json_bytes
    for forbidden in (
        b"private_reference",
        b"grader_rule",
        b"raw_trace",
        b"source_span",
    ):
        assert forbidden not in serialized


def test_compile_manifest_recomputes_complete_rubric_identity() -> None:
    source, _ = _source()
    criterion = source.rubric_set.criteria[0].model_copy(update={"description": "Changed after approval."})
    stale_rubric = source.rubric_set.model_copy(update={"criteria": (criterion,)})

    with pytest.raises(ReleasePublicationPolicyError, match="stale or malformed"):
        _compile_manifest(
            ReleasePublicationItemSource(
                approved_result=source.approved_result,
                rubric_set=stale_rubric,
                item_quality=source.item_quality,
            )
        )


@pytest.mark.parametrize(
    "choice",
    (
        QueryPackagingChoice.OMIT_QUERY_YAML,
        QueryPackagingChoice.ASK_PER_TASK,
    ),
)
def test_strict_lh_blocks_omit_and_unresolved_query_choices(
    choice: QueryPackagingChoice,
) -> None:
    source, _ = _source(choice=choice)

    with pytest.raises(
        ReleasePublicationPolicyError,
        match="INCLUDE_QUERY_YAML",
    ):
        _compile_manifest(source)


def test_explicit_include_query_decision_is_admitted() -> None:
    source, commit = _source(choice=QueryPackagingChoice.INCLUDE_QUERY_YAML)

    compilation = _compile_manifest(source)

    assert commit is not None
    assert commit.decision_record_ref in (source.approved_result.evaluation_item.user_decision_record_refs)
    assert compilation.items[0].item_manifest.query_yaml_included is True


def test_manifest_ref_budget_is_preflighted() -> None:
    source, _ = _source()

    with pytest.raises(ReleasePublicationPolicyError, match="ref budget"):
        _compile_manifest(source, policy=_policy(max_manifest_refs=14))

    with pytest.raises(ReleasePublicationPolicyError, match="duplicates"):
        _compile_manifest(
            source,
            rejected_item_ids=(
                "item://r7-09/rejected",
                "item://r7-09/rejected",
            ),
        )


def test_compile_publication_creates_linked_publish_and_successor_item() -> None:
    source, _ = _source()
    manifest = _compile_manifest(source)
    workspace = LHWorkspaceExportResultV2.exported(
        request=manifest.items[0].workspace_request,
        observed_members=manifest.items[0].workspace_request.members,
        copied_output_refs=manifest.items[0].workspace_request.output_refs,
    )

    publication = (
        ReleasePublicationCompiler()
        .compile_publication(
            manifest=manifest,
            workspace_results=(workspace,),
            bundle_facts=(
                ReleaseBundleFacts(
                    bundle_sha256="a" * 64,
                    bundle_file_count=3,
                    bundle_total_bytes=512,
                ),
            ),
            policy=_policy(),
            published_at=PUBLISHED_AT,
            audit=_audit(),
        )
        .result
    )

    decision = publication.published_decisions[0]
    item = publication.published_items[0]
    assert decision.action is ReleaseActionV2.PUBLISH
    assert decision.state is ReleaseStateV2.RELEASED
    assert decision.previous_decision_ref == (source.approved_result.release_decision_ref)
    assert decision.components == source.approved_result.release_decision.components
    assert decision.decided_at == PUBLISHED_AT
    assert item.item_version == decision.item_version == "3"
    assert item.release_decision_ref == release_decision_v2_ref(decision)
    assert publication.item_projections[0].item_status is ItemStatus.RELEASED


def test_compile_publication_rejects_stale_workspace_and_naive_time() -> None:
    source, _ = _source()
    manifest = _compile_manifest(source)
    workspace = LHWorkspaceExportResultV2.exported(
        request=manifest.items[0].workspace_request,
        observed_members=manifest.items[0].workspace_request.members,
        copied_output_refs=manifest.items[0].workspace_request.output_refs,
    )
    stale = workspace.model_copy(update={"workspace_total_bytes": 1})

    with pytest.raises(ReleasePublicationPolicyError, match="workspace export"):
        ReleasePublicationCompiler().compile_publication(
            manifest=manifest,
            workspace_results=(stale,),
            bundle_facts=(
                ReleaseBundleFacts(
                    bundle_sha256="a" * 64,
                    bundle_file_count=3,
                    bundle_total_bytes=512,
                ),
            ),
            policy=_policy(),
            published_at=PUBLISHED_AT,
            audit=_audit(),
        )
    with pytest.raises(ReleasePublicationPolicyError, match="timezone"):
        ReleasePublicationCompiler().compile_publication(
            manifest=manifest,
            workspace_results=(workspace,),
            bundle_facts=(
                ReleaseBundleFacts(
                    bundle_sha256="a" * 64,
                    bundle_file_count=3,
                    bundle_total_bytes=512,
                ),
            ),
            policy=_policy(),
            published_at=datetime(2026, 8, 2),
            audit=_audit(),
        )


def test_release_publication_gold_is_content_free_and_scope_complete() -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    case_ids = {case["case_id"] for case in gold["cases"]}

    assert gold["claim_scope"] == "NONPRODUCTION_ONLY"
    assert {
        "lh-default-query-inclusion",
        "lh-explicit-query-inclusion",
        "lh-query-omit-blocked",
        "lh-query-ask-per-task-blocked",
        "mixed-approved-rejected-partition",
        "bundle-hash-drift",
        "exact-idempotent-replay",
        "changed-idempotency-request",
        "projection-rebuild",
        "generic-profile-blocked",
        "production-channel-blocked",
    }.issubset(case_ids)
    serialized = GOLD_PATH.read_text(encoding="utf-8").casefold()
    for forbidden in (
        "credential_value",
        "final_answer",
        "grader_prompt",
        "private_reference_body",
        "raw_trace_text",
    ):
        assert forbidden not in serialized
