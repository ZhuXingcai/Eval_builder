from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_artifact_execution import (
    _audit as _execution_audit,
)
from test_artifact_execution import (
    _ControlledExecutionFacade,
    _definition,
    _execution_plan,
    _routing_source,
)

from env_mock_agent.facade.validation_v2 import (
    ATTACHMENT_VALIDATION_POLICY_VERSION,
    AttachmentInventoryMemberTypeV2,
    AttachmentInventoryMemberV2,
    AttachmentValidationFacade,
    AttachmentValidationFailureCodeV2,
    AttachmentValidationFindingCategoryV2,
    AttachmentValidationFindingCodeV2,
    AttachmentValidationFindingV2,
    AttachmentValidationRequestV2,
    AttachmentValidationResultV2,
    AttachmentValidationSeverityV2,
    AttachmentValidationStatusV2,
    attachment_inventory_member_carried_sha256,
    attachment_validation_finding_carried_sha256,
    attachment_validation_request_ref,
    attachment_validation_result_carried_sha256,
    validate_attachment_validation_request_identity,
)
from eval_factory.attachment_planning import (
    ArtifactExecutionPlanningOutcome,
    ArtifactExecutionPlanningResult,
    ArtifactGroupExecutor,
    ArtifactResultCompiler,
    DeterministicValidationCompiler,
    DeterministicValidationPolicyError,
)
from eval_factory.contracts import (
    AttachmentReconstructionOutcomeV2,
    AttachmentReconstructionResultV2,
    DeterministicItemValidationOutcomeV2,
    ProducerAttachmentRequirementV2,
    ProducerTaskViewV2,
    attachment_reconstruction_result_v2_carried_sha256,
    producer_task_view_carried_sha256,
    producer_task_view_ref,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.task import AttachmentCriticality
from eval_factory.provenance import ConfiguredPiiRule
from eval_factory.task_authoring import PromptLeakageReferenceSetCompiler

HASH = "a" * 64
GOLD_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals/golden/eval_factory/attachment_validation"
    / "r5-08-deterministic-validation-v1.json"
)


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 7, 28, tzinfo=UTC),
        created_by="deterministic-validation-test",
        governing_versions=(
            VersionBinding(
                component="artifact-results",
                version="r5-07",
            ),
        ),
        input_refs=tuple(sorted(refs, key=lambda ref: ref.object_type)),
    )


def _producer_view(
    artifacts: tuple[tuple[str, str], ...] = (),
) -> ProducerTaskViewV2:
    value = ProducerTaskViewV2(
        producer_task_view_id="producer-task-view://pending",
        producer_task_view_version=1,
        supersedes_producer_task_view_ref=None,
        query_instruction="Inspect the provided input-state workspace.",
        attachment_requirements=tuple(
            ProducerAttachmentRequirementV2(
                dependency_id=(f"attachment-dependency://{artifact_id.rsplit('/', 1)[-1]}"),
                description=f"Provide safe input state at {path}.",
                criticality=AttachmentCriticality.REQUIRED,
            )
            for artifact_id, path in sorted(artifacts)
        ),
        allowed_tools=("file-read",),
        safe_evidence_bundle_refs=(_ref("evidence-bundle", "producer"),),
        forbidden_outputs=("original final answer",),
        projection_policy_ref=_ref(
            "projection-policy",
            "producer",
            version="v1",
        ),
        storage_authorization_ref=_ref(
            "producer-storage-authorization",
            "producer",
        ),
        prompt_boundary_enforcement_ref=_ref(
            "prompt-boundary-enforcement",
            "producer",
            version="r2-07",
        ),
        source_task_draft_sha256=HASH,
        source_contestant_tool_policy_sha256=HASH,
        source_contract_chain_sha256=HASH,
        policy_version="producer-task-view/r4-08-v1",
        producer_task_view_sha256=HASH,
        audit=_audit(),
    )
    digest = producer_task_view_carried_sha256(value)
    return value.model_copy(
        update={
            "producer_task_view_id": f"producer-task-view://sha256/{digest}",
            "producer_task_view_sha256": digest,
        }
    )


def _not_required_result(
    producer_view: ProducerTaskViewV2,
) -> AttachmentReconstructionResultV2:
    request_ref = _ref(
        "artifact-routing-request",
        "none",
        version="r5-05",
    )
    compilation_ref = _ref(
        "artifact-routing-compilation-result",
        "none",
        version="r5-05",
    )
    value = AttachmentReconstructionResultV2(
        attachment_reconstruction_result_v2_id=("attachment-reconstruction-result://pending"),
        routing_request_ref=request_ref,
        routing_compilation_result_ref=compilation_ref,
        attachment_planning_context_ref=_ref(
            "attachment-planning-context",
            "current",
        ),
        producer_task_view_ref=producer_task_view_ref(producer_view),
        evidence_matrix_ref=None,
        artifact_routing_plan_ref=None,
        artifact_execution_plan_ref=None,
        artifact_execution_batch_ref=None,
        artifact_results=(),
        artifact_result_refs=(),
        outcome=AttachmentReconstructionOutcomeV2.NOT_REQUIRED,
        candidate_output_refs=(),
        accepted_artifact_refs=(),
        failed_artifact_ids=(),
        resumable_artifact_ids=(),
        dependency_blocked_artifact_ids=(),
        required_incomplete_artifact_ids=(),
        optional_incomplete_artifact_ids=(),
        frozen_result=None,
        frozen_projection_gap="PRE_VALIDATION",
        environment_spec_ref=None,
        provenance_manifest_ref=None,
        quality_report_ref=None,
        package_sha256=None,
        input_state_only=None,
        policy_version="artifact-results/r5-07-v1",
        attachment_reconstruction_result_v2_sha256=HASH,
        audit=_audit(request_ref, compilation_ref),
    )
    digest = attachment_reconstruction_result_v2_carried_sha256(value)
    return value.model_copy(
        update={
            "attachment_reconstruction_result_v2_id": (f"attachment-reconstruction-result://sha256/{digest}"),
            "attachment_reconstruction_result_v2_sha256": digest,
        }
    )


class _NoCallFacade(AttachmentValidationFacade):
    async def validate(
        self,
        request: AttachmentValidationRequestV2,
    ):
        del request
        raise AssertionError("NOT_REQUIRED must not call validation facade")


class _FakeValidationFacade(AttachmentValidationFacade):
    def __init__(
        self,
        outcomes: dict[str, str] | None = None,
        logical_paths: dict[str, str] | None = None,
    ) -> None:
        self.outcomes = outcomes or {}
        self.logical_paths = logical_paths or {}
        self.calls: list[str] = []

    async def validate(
        self,
        request: AttachmentValidationRequestV2,
    ) -> AttachmentValidationResultV2:
        validate_attachment_validation_request_identity(request)
        self.calls.append(request.artifact_id)
        outcome = self.outcomes.get(request.artifact_id, "pass")
        required = (
            "common",
            "configured-pii",
            "metadata",
            "package-inventory",
            "restricted-fingerprint",
            "secrets",
            "text",
        )
        if outcome == "blocked":
            return _facade_result(
                request,
                status=AttachmentValidationStatusV2.BLOCKED,
                required=required,
                findings=(),
                inventory=(),
                failure_code=(AttachmentValidationFailureCodeV2.VALIDATOR_UNAVAILABLE),
            )
        member = _facade_member(
            request,
            logical_path=self.logical_paths.get(
                request.artifact_id,
                request.logical_path,
            ),
        )
        findings: tuple[AttachmentValidationFindingV2, ...] = ()
        if outcome == "secret":
            findings = (
                _facade_finding(
                    request,
                    member=member,
                    code=AttachmentValidationFindingCodeV2.SECRET_DETECTED,
                    category=(AttachmentValidationFindingCategoryV2.SECURITY),
                    non_waivable=True,
                    rule_id="builtin-secret/assigned-secret/v1",
                ),
            )
        elif outcome == "pii":
            findings = (
                _facade_finding(
                    request,
                    member=member,
                    code=(AttachmentValidationFindingCodeV2.CONFIGURED_PII_DETECTED),
                    category=(AttachmentValidationFindingCategoryV2.PRIVACY),
                    non_waivable=False,
                    rule_id="configured-pii/customer/v1",
                ),
            )
        return _facade_result(
            request,
            status=(AttachmentValidationStatusV2.FAILED if findings else AttachmentValidationStatusV2.PASSED),
            required=required,
            findings=findings,
            inventory=(member,),
            failure_code=None,
        )


def _facade_member(
    request: AttachmentValidationRequestV2,
    *,
    logical_path: str,
) -> AttachmentInventoryMemberV2:
    member = AttachmentInventoryMemberV2(
        inventory_member_id="attachment-inventory-member://pending",
        artifact_id=request.artifact_id,
        output_ref=request.output_ref,
        normalized_path=logical_path,
        member_type=AttachmentInventoryMemberTypeV2.FILE,
        media_type=request.media_type,
        size_bytes=32,
        content_sha256=request.output_sha256,
        container_ref=None,
        inventory_member_sha256=HASH,
    )
    digest = attachment_inventory_member_carried_sha256(member)
    return member.model_copy(
        update={
            "inventory_member_id": (f"attachment-inventory-member://sha256/{digest}"),
            "inventory_member_sha256": digest,
        }
    )


def _facade_finding(
    request: AttachmentValidationRequestV2,
    *,
    member: AttachmentInventoryMemberV2,
    code: AttachmentValidationFindingCodeV2,
    category: AttachmentValidationFindingCategoryV2,
    non_waivable: bool,
    rule_id: str,
) -> AttachmentValidationFindingV2:
    finding = AttachmentValidationFindingV2(
        finding_id="attachment-validation-finding://pending",
        artifact_id=request.artifact_id,
        subject_output_ref=request.output_ref,
        inventory_member_id=member.inventory_member_id,
        severity=AttachmentValidationSeverityV2.P0,
        category=category,
        code=code,
        rule_ids=(rule_id,),
        matched_fingerprint_ids=(),
        non_waivable=non_waivable,
        finding_sha256=HASH,
    )
    digest = attachment_validation_finding_carried_sha256(finding)
    return finding.model_copy(
        update={
            "finding_id": (f"attachment-validation-finding://sha256/{digest}"),
            "finding_sha256": digest,
        }
    )


def _facade_result(
    request: AttachmentValidationRequestV2,
    *,
    status: AttachmentValidationStatusV2,
    required: tuple[str, ...],
    findings: tuple[AttachmentValidationFindingV2, ...],
    inventory: tuple[AttachmentInventoryMemberV2, ...],
    failure_code: AttachmentValidationFailureCodeV2 | None,
) -> AttachmentValidationResultV2:
    result = AttachmentValidationResultV2(
        validation_result_id="attachment-validation-result://pending",
        validation_request_ref=attachment_validation_request_ref(request),
        artifact_id=request.artifact_id,
        output_ref=request.output_ref,
        output_sha256=request.output_sha256,
        status=status,
        executed_validator_ids=required if status is not AttachmentValidationStatusV2.BLOCKED else (),
        required_validator_ids=required,
        complete_leakage_categories=request.complete_leakage_categories,
        findings=findings,
        inventory_members=inventory,
        inventory_complete=status is not AttachmentValidationStatusV2.BLOCKED,
        scan_complete=status is not AttachmentValidationStatusV2.BLOCKED,
        failure_code=failure_code,
        policy_version=ATTACHMENT_VALIDATION_POLICY_VERSION,
        validation_result_sha256=HASH,
    )
    digest = attachment_validation_result_carried_sha256(result)
    return result.model_copy(
        update={
            "validation_result_id": (f"attachment-validation-result://sha256/{digest}"),
            "validation_result_sha256": digest,
        }
    )


async def _reconstruction_result(
    tmp_path: Path,
    *,
    view: ProducerTaskViewV2,
    artifacts: tuple[tuple[str, str], ...],
    fail_once: frozenset[str] = frozenset(),
) -> AttachmentReconstructionResultV2:
    view_ref = producer_task_view_ref(view)
    routing_request, routing_result = _routing_source(
        artifacts,
        producer_task_view_ref=view_ref,
    )
    execution_plan = await _execution_plan(
        tmp_path,
        artifacts,
        tuple(_definition(artifact_id) for artifact_id, _ in artifacts),
        producer_task_view_ref=view_ref,
    )
    execution_facade = _ControlledExecutionFacade(
        fail_once=fail_once,
        expected_parallelism=len(artifacts),
    )
    batch = await ArtifactGroupExecutor().run(
        execution_plan,
        facade=execution_facade,
        audit=_execution_audit(),
    )
    planning = ArtifactExecutionPlanningResult(
        outcome=ArtifactExecutionPlanningOutcome.PLANNED,
        world_ledger_snapshot=execution_plan.world_ledger_snapshot,
        execution_plan=execution_plan,
        audit=_execution_audit(),
    )
    return ArtifactResultCompiler().compile(
        routing_request=routing_request,
        routing_result=routing_result,
        execution_result=planning,
        execution_batch=batch,
        audit=_audit(),
    )


def _reference_set():
    return PromptLeakageReferenceSetCompiler().compile(
        trace_envelope_ref=_ref("trace-envelope", "current"),
        sources=(),
        complete_categories=frozenset(),
        audit=_audit(),
    )


@pytest.mark.asyncio
async def test_not_required_compiles_without_facade_call() -> None:
    view = _producer_view()
    reconstruction = _not_required_result(view)
    reference_set = _reference_set()

    result = await DeterministicValidationCompiler().compile(
        reconstruction_result=reconstruction,
        producer_task_view=view,
        leakage_reference_set=reference_set,
        configured_pii_rules=(),
        facade=_NoCallFacade(),
        audit=_audit(),
    )

    assert result.outcome is DeterministicItemValidationOutcomeV2.NOT_REQUIRED
    assert result.artifact_validation_results == ()
    assert result.candidate_inventory is None
    assert result.accepted_artifact_refs == ()
    assert result.input_state_only is None

    DeterministicValidationCompiler().validate_current(
        result,
        reconstruction_result=reconstruction,
        producer_task_view=view,
        leakage_reference_set=reference_set,
        configured_pii_rules=(),
    )


@pytest.mark.asyncio
async def test_mismatched_producer_view_fails_before_facade_call() -> None:
    view = _producer_view()
    reconstruction = _not_required_result(view).model_copy(
        update={
            "producer_task_view_ref": _ref("producer-task-view", "other"),
        }
    )
    reference_set = _reference_set()

    with pytest.raises(
        DeterministicValidationPolicyError,
        match="producer task view",
    ):
        await DeterministicValidationCompiler().compile(
            reconstruction_result=reconstruction,
            producer_task_view=view,
            leakage_reference_set=reference_set,
            configured_pii_rules=(
                ConfiguredPiiRule(
                    rule_id="configured-pii/customer/v1",
                    pattern=r"CUST-\d{5}",
                ),
            ),
            facade=_NoCallFacade(),
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_clean_candidate_compiles_pre_semantic_inventory(
    tmp_path: Path,
) -> None:
    artifacts = (("artifact://a", "inputs/a.txt"),)
    view = _producer_view(artifacts)
    reconstruction = await _reconstruction_result(
        tmp_path,
        view=view,
        artifacts=artifacts,
    )
    facade = _FakeValidationFacade()
    reference_set = _reference_set()
    pii_rules = (
        ConfiguredPiiRule(
            rule_id="configured-pii/customer/v1",
            pattern=r"CUST-\d{5}",
        ),
    )

    result = await DeterministicValidationCompiler().compile(
        reconstruction_result=reconstruction,
        producer_task_view=view,
        leakage_reference_set=reference_set,
        configured_pii_rules=pii_rules,
        facade=facade,
        audit=_audit(),
    )

    assert result.outcome is DeterministicItemValidationOutcomeV2.PASSED
    assert result.validated_artifact_ids == ("artifact://a",)
    assert result.finding_refs == ()
    assert result.candidate_inventory is not None
    assert result.candidate_inventory.final_package is False
    assert result.candidate_inventory.package_sha256 is None
    assert result.candidate_inventory.input_state_only is None
    assert result.accepted_artifact_refs == ()
    assert result.environment_spec_ref is None
    assert result.provenance_manifest_ref is None
    assert result.quality_report_ref is None
    assert facade.calls == ["artifact://a"]
    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    assert r"CUST-\d{5}" not in serialized

    DeterministicValidationCompiler().validate_current(
        result,
        reconstruction_result=reconstruction,
        producer_task_view=view,
        leakage_reference_set=reference_set,
        configured_pii_rules=pii_rules,
    )


@pytest.mark.parametrize(
    ("facade_outcome", "expected_outcome", "expected_inventory"),
    [
        (
            "pii",
            DeterministicItemValidationOutcomeV2.REQUIRES_REPAIR,
            True,
        ),
        (
            "secret",
            DeterministicItemValidationOutcomeV2.REJECTED,
            True,
        ),
        (
            "blocked",
            DeterministicItemValidationOutcomeV2.BLOCKED,
            False,
        ),
    ],
)
@pytest.mark.asyncio
async def test_candidate_findings_and_block_map_to_exact_aggregate(
    tmp_path: Path,
    facade_outcome: str,
    expected_outcome: DeterministicItemValidationOutcomeV2,
    expected_inventory: bool,
) -> None:
    artifacts = (("artifact://a", "inputs/a.txt"),)
    view = _producer_view(artifacts)
    reconstruction = await _reconstruction_result(
        tmp_path,
        view=view,
        artifacts=artifacts,
    )

    result = await DeterministicValidationCompiler().compile(
        reconstruction_result=reconstruction,
        producer_task_view=view,
        leakage_reference_set=_reference_set(),
        configured_pii_rules=(),
        facade=_FakeValidationFacade({"artifact://a": facade_outcome}),
        audit=_audit(),
    )

    assert result.outcome is expected_outcome
    assert (result.candidate_inventory is not None) is expected_inventory
    if facade_outcome == "pii":
        assert result.repair_required_artifact_ids == ("artifact://a",)
        assert result.artifact_validation_results[0].findings[0].non_waivable is False
    elif facade_outcome == "secret":
        assert result.rejected_artifact_ids == ("artifact://a",)
        assert result.artifact_validation_results[0].findings[0].non_waivable is True
    else:
        assert result.validation_blocked_artifact_ids == ("artifact://a",)


@pytest.mark.asyncio
async def test_partial_upstream_validates_only_available_candidate(
    tmp_path: Path,
) -> None:
    artifacts = (
        ("artifact://a", "inputs/a.txt"),
        ("artifact://b", "inputs/b.txt"),
    )
    view = _producer_view(artifacts)
    reconstruction = await _reconstruction_result(
        tmp_path,
        view=view,
        artifacts=artifacts,
        fail_once=frozenset({"artifact://b"}),
    )
    facade = _FakeValidationFacade()

    result = await DeterministicValidationCompiler().compile(
        reconstruction_result=reconstruction,
        producer_task_view=view,
        leakage_reference_set=_reference_set(),
        configured_pii_rules=(),
        facade=facade,
        audit=_audit(),
    )

    assert result.outcome is (DeterministicItemValidationOutcomeV2.UPSTREAM_INCOMPLETE)
    assert result.validated_artifact_ids == ("artifact://a",)
    assert result.upstream_incomplete_artifact_ids == ("artifact://b",)
    assert facade.calls == ["artifact://a"]


@pytest.mark.asyncio
async def test_candidate_inventory_casefold_collision_blocks_item(
    tmp_path: Path,
) -> None:
    artifacts = (
        ("artifact://a", "inputs/A.txt"),
        ("artifact://b", "inputs/a.txt"),
    )
    view = _producer_view(artifacts)
    reconstruction = await _reconstruction_result(
        tmp_path,
        view=view,
        artifacts=artifacts,
    )

    result = await DeterministicValidationCompiler().compile(
        reconstruction_result=reconstruction,
        producer_task_view=view,
        leakage_reference_set=_reference_set(),
        configured_pii_rules=(),
        facade=_FakeValidationFacade(),
        audit=_audit(),
    )

    assert result.outcome is DeterministicItemValidationOutcomeV2.BLOCKED
    assert result.candidate_inventory is None
    assert result.candidate_inventory_failure_code is (
        AttachmentValidationFailureCodeV2.PACKAGE_PATH_COLLISION
    )
    assert result.validated_artifact_ids == (
        "artifact://a",
        "artifact://b",
    )


@pytest.mark.asyncio
async def test_currentness_rejects_changed_configured_pii_policy(
    tmp_path: Path,
) -> None:
    artifacts = (("artifact://a", "inputs/a.txt"),)
    view = _producer_view(artifacts)
    reconstruction = await _reconstruction_result(
        tmp_path,
        view=view,
        artifacts=artifacts,
    )
    reference_set = _reference_set()
    first_rules = (
        ConfiguredPiiRule(
            rule_id="configured-pii/customer/v1",
            pattern=r"CUST-\d{5}",
        ),
    )
    result = await DeterministicValidationCompiler().compile(
        reconstruction_result=reconstruction,
        producer_task_view=view,
        leakage_reference_set=reference_set,
        configured_pii_rules=first_rules,
        facade=_FakeValidationFacade(),
        audit=_audit(),
    )

    with pytest.raises(
        DeterministicValidationPolicyError,
        match="source refs are stale",
    ):
        DeterministicValidationCompiler().validate_current(
            result,
            reconstruction_result=reconstruction,
            producer_task_view=view,
            leakage_reference_set=reference_set,
            configured_pii_rules=(
                ConfiguredPiiRule(
                    rule_id="configured-pii/customer/v1",
                    pattern=r"CLIENT-\d{5}",
                ),
            ),
        )


def test_r5_08_executable_gold_is_content_free_and_complete() -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    assert payload["gold_version"] == "r5-08-deterministic-validation-v1"
    assert {case["expected_outcome"] for case in payload["cases"]} == {
        "NOT_REQUIRED",
        "PASSED",
        "UPSTREAM_INCOMPLETE",
        "REQUIRES_REPAIR",
        "REJECTED",
        "BLOCKED",
    }
    assert payload["downstream_boundary"] == {
        "accepted_artifact_refs": [],
        "environment_spec_ref": None,
        "provenance_manifest_ref": None,
        "quality_report_ref": None,
        "package_sha256": None,
        "input_state_only": None,
    }
    serialized = json.dumps(payload, sort_keys=True).casefold()
    for denied in (
        "artifact_path",
        "extracted_text",
        "matched_text",
        "regex_pattern",
        "private_reference_content",
        "final_answer_content",
    ):
        assert denied not in serialized
