from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_deterministic_validation import (
    _FakeValidationFacade,
    _NoCallFacade,
    _not_required_result,
    _producer_view,
    _reference_set,
)

from env_mock_agent.facade import FacadeObjectRef
from env_mock_agent.facade.model_control_v2 import (
    ModelControlGrantV2,
    ModelControlReceiptOutcomeV2,
    ModelControlUsageSourceV2,
    ModelDemandModeFacadeV2,
    ModelInvocationDescriptorV2,
    model_invocation_descriptor_ref,
)
from env_mock_agent.facade.semantic_review_adapter import (
    ProviderAttachmentRepairContext,
    ProviderAttachmentRepairDecision,
    ProviderAttachmentRepairMaterial,
    ProviderSemanticFindingDecision,
    ProviderSemanticResolutionDecision,
    ProviderSemanticReviewContext,
    ProviderSemanticReviewDecision,
    ProviderSemanticReviewMaterial,
    RegistryAttachmentSemanticReviewFacade,
)
from env_mock_agent.facade.semantic_review_v2 import (
    AttachmentRepairOutcomeV2,
    AttachmentRepairRequestV2,
    AttachmentRepairSourceV2,
    AttachmentSemanticResolutionDispositionV2,
    AttachmentSemanticReviewFailureCodeV2,
    AttachmentSemanticReviewFindingCodeV2,
    AttachmentSemanticReviewOutcomeV2,
    AttachmentSemanticReviewRequestV2,
    AttachmentSemanticReviewRoundV2,
    attachment_repair_request_carried_sha256,
    attachment_repair_request_ref,
    attachment_semantic_review_request_ref,
)
from eval_factory.attachment_planning.review import (
    CandidateRevisionCompiler,
    CompleteRevisionValidator,
    IsolatedSemanticReviewOrchestrator,
    SemanticReviewContextSources,
    _review_request,
)
from eval_factory.attachment_planning.validation import (
    DeterministicValidationCompiler,
)
from eval_factory.contracts.approval import ApprovalMode
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ResourceBudget,
    StageRunStatus,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import DatasetJobSpecV2, StageNameV2
from eval_factory.contracts.review_v2 import (
    ROUND_ROLES,
    AttachmentCandidateRevisionV2,
    CandidateArtifactVersionV2,
    DeterministicRepairTargetV2,
    RevisionDeterministicValidationOutcomeV2,
    RevisionDeterministicValidationV2,
    SemanticReviewPolicyV2,
    SemanticReviewRoundV2,
    SemanticReviewWorkflowOutcomeV2,
    attachment_candidate_revision_ref,
    candidate_artifact_version_ref,
    semantic_finding_resolution_is_current,
)
from eval_factory.orchestration import JobStore

HASH = "a" * 64
NOW = datetime(2026, 7, 29, tzinfo=UTC)
GOLD_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals/golden/eval_factory/attachment_review"
    / "r5-09-isolated-semantic-review-v1.json"
)


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="semantic-review-test",
        governing_versions=(VersionBinding(component="semantic-review", version="r5-09"),),
        input_refs=tuple(
            sorted(
                set(refs),
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        ),
    )


def _facade_ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str = HASH,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _model_control(
    *,
    operation_ref: FacadeObjectRef,
    model_profile_ref: FacadeObjectRef,
    suffix: str,
) -> tuple[ModelInvocationDescriptorV2, ModelControlGrantV2]:
    descriptor = ModelInvocationDescriptorV2.create(
        operation_ref=operation_ref,
        model_profile_ref=model_profile_ref,
        mode=ModelDemandModeFacadeV2.DIRECT_REQUEST,
        request_allowance=1,
        input_token_allowance=100,
        output_token_allowance=200,
        cache_token_allowance=50,
        idempotency_key=f"semantic-model-control-{suffix}",
    )
    grant = ModelControlGrantV2.create(
        reservation_ref=_facade_ref("work-model-reservation", suffix),
        invocation_handle_ref=_facade_ref(
            "model-invocation-handle",
            suffix,
            version="v1",
        ),
        descriptor_ref=model_invocation_descriptor_ref(descriptor),
        model_profile_ref=model_profile_ref,
        fencing_token=1,
        request_allowance=descriptor.request_allowance,
        token_allowance=descriptor.token_allowance,
        output_token_limit=descriptor.output_token_allowance,
    )
    return descriptor, grant


def _job_spec() -> DatasetJobSpecV2:
    return DatasetJobSpecV2(
        job_id="job://semantic-review/test",
        traces=(
            TraceSourceRef(
                source_trace_id="source-trace://semantic-review",
                source_uri="file:///tmp/semantic-review.jsonl",
                raw_sha256=HASH,
                adapter_name="raw_traj_v1",
                adapter_version="1.0.0",
                processing_class="RESTRICTED_TRACE_RAW",
            ),
        ),
        requested_stages=(
            StageNameV2.TRACE_INDEX,
            StageNameV2.ITEM_QUALITY,
        ),
        privacy_profile="trusted-monitored-local",
        model_profiles=(),
        budget=ResourceBudget(
            max_model_requests=3,
            max_model_tokens=1000,
            max_processes=1,
            max_renderers=1,
            max_network_requests=0,
            max_storage_bytes=1024 * 1024,
        ),
        concurrency=ConcurrencyLimit(
            model_requests=1,
            processes=1,
            renderers=1,
            network_requests=1,
            artifacts_per_item=1,
            items=1,
        ),
        approval_policy_ref=_ref("user-approval-policy", "none"),
        approval_mode=ApprovalMode.NONE,
        enabled_checkpoints=frozenset(),
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://canary",
        ),
        idempotency_key="create-job-semantic-review",
        audit=_audit(),
    )


def _revision(*, output_digest: str = HASH) -> AttachmentCandidateRevisionV2:
    reconstruction_ref = _ref("attachment-reconstruction-result", "base")
    artifact = CandidateArtifactVersionV2.create_base(
        artifact_id="artifact://input",
        attachment_dependency_id="attachment-dependency://input",
        artifact_build_result_ref=_ref("artifact-build-result", "input"),
        build_spec_ref=_ref("artifact-build-spec", "input"),
        execution_request_ref=_ref("attachment-execution-request", "input"),
        execution_result_ref=_ref("attachment-execution-result", "input"),
        output_ref=_ref(
            "attachment-output",
            "input",
            digest=output_digest,
        ),
        logical_path="inputs/source.txt",
        media_type="text/plain",
        declared_validator_ids=("text-validator",),
        derivation_root_refs=(
            _ref("producer-task-view", "current"),
            _ref("artifact-build-spec", "input"),
        ),
    )
    return AttachmentCandidateRevisionV2.create_initial(
        base_reconstruction_result_ref=reconstruction_ref,
        artifact_versions=(artifact,),
        audit=_audit(
            reconstruction_ref,
            candidate_artifact_version_ref(artifact),
        ),
    )


def _validation(
    revision: AttachmentCandidateRevisionV2,
    outcome: RevisionDeterministicValidationOutcomeV2 = (RevisionDeterministicValidationOutcomeV2.PASSED),
    *,
    repair_targets: tuple[DeterministicRepairTargetV2, ...] = (),
) -> RevisionDeterministicValidationV2:
    revision_ref = attachment_candidate_revision_ref(revision)
    source_ref = _ref("deterministic-item-validation-result", "base")
    return RevisionDeterministicValidationV2.create(
        candidate_revision_ref=revision_ref,
        source_deterministic_validation_result_ref=source_ref,
        artifact_validation_result_refs=(_ref("artifact-deterministic-validation-result", "input"),),
        finding_refs=(),
        output_refs=revision.candidate_output_refs,
        outcome=outcome,
        audit=_audit(revision_ref, source_ref),
        repair_targets=repair_targets,
    )


def _policy() -> SemanticReviewPolicyV2:
    return SemanticReviewPolicyV2.create(
        model_profile_refs={
            round_: _ref(
                "model-profile",
                round_.value.lower(),
                version="v1",
            )
            for round_ in SemanticReviewRoundV2
        },
        prompt_versions={
            round_: f"semantic-review/{round_.value.lower()}/v1" for round_ in SemanticReviewRoundV2
        },
        projection_policy_refs={
            round_: _ref(
                "projection-policy",
                round_.value.lower(),
                version="v1",
            )
            for round_ in SemanticReviewRoundV2
        },
    )


def _sources() -> SemanticReviewContextSources:
    return SemanticReviewContextSources(
        query_public_view_ref=_ref("query-spec-public-view", "current"),
        candidate_manifest_ref=_ref("candidate-package-inventory", "current"),
        public_rubric_requirement_refs=(_ref("public-rubric-requirements", "current"),),
        safe_evidence_bundle_refs=(_ref("evidence-bundle", "coverage", version="v1"),),
        world_ledger_snapshot_ref=_ref("world-ledger-snapshot", "current"),
        approved_source_evidence_refs=(_ref("source-evidence", "current"),),
        taint_lineage_projection_refs=(_ref("taint-lineage-projection", "current"),),
        evaluator_contract_ref=_ref("evaluator-contract-projection", "current"),
        contestant_tool_policy_ref=_ref("contestant-tool-policy", "current"),
        leakage_reference_set_ref=_ref("prompt-leakage-reference-set", "current"),
        executability_result_refs=(_ref("deterministic-executability-result", "current"),),
    )


class _DynamicResolver:
    def __init__(self, source_output_bytes: bytes | None = None) -> None:
        self.source_output_bytes = source_output_bytes
        self.registered_output_refs: list[FacadeObjectRef] = []

    def resolve_review_material(
        self,
        request: AttachmentSemanticReviewRequestV2,
    ) -> ProviderSemanticReviewMaterial:
        return ProviderSemanticReviewMaterial(
            context_view_ref=request.context_view_ref,
            included_ref_inventory=tuple(
                sorted(
                    (
                        request.candidate_revision_ref,
                        request.deterministic_validation_result_ref,
                        request.context_view_ref,
                        *request.current_artifact_version_refs,
                        *request.current_output_refs,
                        *request.prior_finding_refs,
                        *request.prior_resolution_refs,
                        *request.prior_repair_plan_refs,
                        *request.prior_repair_result_refs,
                    ),
                    key=lambda ref: (
                        ref.object_type,
                        ref.object_id,
                        ref.object_version,
                        ref.object_sha256,
                    ),
                )
            ),
            denied_data_families=(
                "BUILD_TRANSCRIPT",
                "HIDDEN_REASONING",
                "RAW_PRIVATE_REFERENCE",
                "RAW_TRACE",
            ),
            allowed_evidence_ref_ids=frozenset(
                {
                    "semantic-evidence://coverage/input",
                    "semantic-evidence://coverage/item-dependency",
                    "semantic-evidence://executability/input",
                    "semantic-evidence://leakage/answer-bearing",
                    "semantic-evidence://leakage/resolution/1",
                    "semantic-evidence://leakage/resolution/2",
                    "semantic-evidence://realism/coverage-resolution",
                    "semantic-evidence://realism/world-fact",
                }
            ),
            safe_payload={"round": request.round.value},
        )

    def resolve_repair_material(
        self,
        request: AttachmentRepairRequestV2,
    ) -> ProviderAttachmentRepairMaterial:
        if self.source_output_bytes is None:
            raise AssertionError(f"unexpected repair: {request.repair_request_id}")
        return ProviderAttachmentRepairMaterial(
            source_output_ref=request.output_ref,
            source_output_bytes=self.source_output_bytes,
            logical_path="inputs/source.txt",
            media_type="text/plain",
        )

    def register_repaired_output(
        self,
        *,
        request: AttachmentRepairRequestV2,
        material: ProviderAttachmentRepairMaterial,
        output_ref: FacadeObjectRef,
        output_bytes: bytes,
    ) -> None:
        assert request.output_ref == material.source_output_ref
        assert hashlib.sha256(output_bytes).hexdigest() == output_ref.object_sha256
        self.registered_output_refs.append(output_ref)
        self.source_output_bytes = output_bytes


class _AcceptingBackend:
    def __init__(self) -> None:
        self.context_ids: list[str] = []

    async def review(
        self,
        context: ProviderSemanticReviewContext,
    ) -> ProviderSemanticReviewDecision:
        self.context_ids.append(context.context_id)
        return ProviderSemanticReviewDecision.accepted()


class _NoRepairBackend:
    async def repair(self, _context: object) -> object:
        raise AssertionError("clean semantic workflow must not repair")


class _NoRevalidation:
    async def validate(
        self,
        _revision: AttachmentCandidateRevisionV2,
    ) -> RevisionDeterministicValidationV2:
        raise AssertionError("clean semantic workflow must not revalidate")


class _RepairingReviewBackend:
    def __init__(self) -> None:
        self.rounds: list[AttachmentSemanticReviewRoundV2] = []
        self.old_subject_refs: tuple[FacadeObjectRef, ...] = ()

    async def review(
        self,
        context: ProviderSemanticReviewContext,
    ) -> ProviderSemanticReviewDecision:
        request = context.request
        self.rounds.append(request.round)
        if request.round is AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY:
            subjects = tuple(
                sorted(
                    (
                        request.candidate_revision_ref,
                        *request.current_artifact_version_refs,
                        *request.current_output_refs,
                    ),
                    key=lambda ref: (
                        ref.object_type,
                        ref.object_id,
                        ref.object_version,
                        ref.object_sha256,
                    ),
                )
            )
            self.old_subject_refs = subjects
            return ProviderSemanticReviewDecision.requires_repair(
                findings=(
                    ProviderSemanticFindingDecision(
                        code=(AttachmentSemanticReviewFindingCodeV2.REQUIRED_COVERAGE_MISSING),
                        subject_refs=subjects,
                        artifact_ids=("artifact://input",),
                        evidence_ref_ids=("semantic-evidence://coverage/input",),
                    ),
                )
            )
        if request.round is AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY:
            current_subjects = tuple(
                sorted(
                    (
                        request.candidate_revision_ref,
                        *request.current_artifact_version_refs,
                        *request.current_output_refs,
                    ),
                    key=lambda ref: (
                        ref.object_type,
                        ref.object_id,
                        ref.object_version,
                        ref.object_sha256,
                    ),
                )
            )
            return ProviderSemanticReviewDecision(
                outcome=AttachmentSemanticReviewOutcomeV2.ACCEPTED,
                resolutions=(
                    ProviderSemanticResolutionDecision(
                        prior_finding_ref=request.required_resolution_finding_refs[0],
                        old_subject_refs=self.old_subject_refs,
                        current_subject_refs=current_subjects,
                        repair_plan_ref=request.prior_repair_plan_refs[0],
                        repair_result_refs=request.prior_repair_result_refs,
                        disposition=(AttachmentSemanticResolutionDispositionV2.CONFIRMED_RESOLVED),
                        successor_finding_ref=None,
                        evidence_ref_ids=("semantic-evidence://realism/coverage-resolution",),
                    ),
                ),
            )
        return ProviderSemanticReviewDecision.accepted()


class _RoundThreeRepairBackend:
    async def review(
        self,
        context: ProviderSemanticReviewContext,
    ) -> ProviderSemanticReviewDecision:
        request = context.request
        if request.round is not AttachmentSemanticReviewRoundV2.LEAKAGE_EXECUTABILITY:
            return ProviderSemanticReviewDecision.accepted()
        subjects = tuple(
            sorted(
                (
                    request.candidate_revision_ref,
                    *request.current_artifact_version_refs,
                    *request.current_output_refs,
                ),
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        )
        return ProviderSemanticReviewDecision.requires_repair(
            findings=(
                ProviderSemanticFindingDecision(
                    code=(AttachmentSemanticReviewFindingCodeV2.EXECUTABILITY_FAILURE),
                    subject_refs=subjects,
                    artifact_ids=("artifact://input",),
                    evidence_ref_ids=("semantic-evidence://executability/input",),
                ),
            )
        )


class _RoundThreeRejectBackend:
    async def review(
        self,
        context: ProviderSemanticReviewContext,
    ) -> ProviderSemanticReviewDecision:
        request = context.request
        if request.round is not AttachmentSemanticReviewRoundV2.LEAKAGE_EXECUTABILITY:
            return ProviderSemanticReviewDecision.accepted()
        return ProviderSemanticReviewDecision.rejected(
            findings=(
                ProviderSemanticFindingDecision(
                    code=(AttachmentSemanticReviewFindingCodeV2.ANSWER_BEARING_CONTENT),
                    subject_refs=(request.candidate_revision_ref,),
                    artifact_ids=(),
                    evidence_ref_ids=("semantic-evidence://leakage/answer-bearing",),
                ),
            )
        )


class _AbstainingBackend:
    async def review(
        self,
        _context: ProviderSemanticReviewContext,
    ) -> ProviderSemanticReviewDecision:
        return ProviderSemanticReviewDecision(
            outcome=AttachmentSemanticReviewOutcomeV2.ABSTAINED,
            failure_code=(AttachmentSemanticReviewFailureCodeV2.INSUFFICIENT_EVIDENCE),
        )


class _ItemScopedRepairBackend:
    async def review(
        self,
        context: ProviderSemanticReviewContext,
    ) -> ProviderSemanticReviewDecision:
        request = context.request
        return ProviderSemanticReviewDecision.requires_repair(
            findings=(
                ProviderSemanticFindingDecision(
                    code=(AttachmentSemanticReviewFindingCodeV2.DEPENDENCY_NOT_SOLVABLE),
                    subject_refs=(request.candidate_revision_ref,),
                    artifact_ids=(),
                    evidence_ref_ids=("semantic-evidence://coverage/item-dependency",),
                ),
            )
        )


class _MissingResolutionBackend(_RepairingReviewBackend):
    async def review(
        self,
        context: ProviderSemanticReviewContext,
    ) -> ProviderSemanticReviewDecision:
        if context.request.round is AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY:
            self.rounds.append(context.request.round)
            return ProviderSemanticReviewDecision.accepted()
        return await super().review(context)


class _TwoRepairReviewBackend:
    def __init__(self) -> None:
        self.rounds: list[AttachmentSemanticReviewRoundV2] = []
        self.initial_subjects: tuple[FacadeObjectRef, ...] = ()
        self.intermediate_subjects: tuple[FacadeObjectRef, ...] = ()
        self.round_one_finding_ref: FacadeObjectRef | None = None
        self.round_one_plan_ref: FacadeObjectRef | None = None
        self.round_one_repair_result_refs: tuple[FacadeObjectRef, ...] = ()

    async def review(
        self,
        context: ProviderSemanticReviewContext,
    ) -> ProviderSemanticReviewDecision:
        request = context.request
        self.rounds.append(request.round)
        subjects = tuple(
            sorted(
                (
                    request.candidate_revision_ref,
                    *request.current_artifact_version_refs,
                    *request.current_output_refs,
                ),
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        )
        if request.round is AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY:
            self.initial_subjects = subjects
            return ProviderSemanticReviewDecision.requires_repair(
                findings=(
                    ProviderSemanticFindingDecision(
                        code=(AttachmentSemanticReviewFindingCodeV2.REQUIRED_COVERAGE_MISSING),
                        subject_refs=subjects,
                        artifact_ids=("artifact://input",),
                        evidence_ref_ids=("semantic-evidence://coverage/input",),
                    ),
                )
            )
        if request.round is AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY:
            self.intermediate_subjects = subjects
            self.round_one_finding_ref = request.required_resolution_finding_refs[0]
            self.round_one_plan_ref = request.prior_repair_plan_refs[0]
            self.round_one_repair_result_refs = request.prior_repair_result_refs
            return ProviderSemanticReviewDecision(
                outcome=AttachmentSemanticReviewOutcomeV2.REQUIRES_REPAIR,
                findings=(
                    ProviderSemanticFindingDecision(
                        code=(AttachmentSemanticReviewFindingCodeV2.WORLD_FACT_INCONSISTENT),
                        subject_refs=subjects,
                        artifact_ids=("artifact://input",),
                        evidence_ref_ids=("semantic-evidence://realism/world-fact",),
                    ),
                ),
                resolutions=(
                    ProviderSemanticResolutionDecision(
                        prior_finding_ref=self.round_one_finding_ref,
                        old_subject_refs=self.initial_subjects,
                        current_subject_refs=subjects,
                        repair_plan_ref=self.round_one_plan_ref,
                        repair_result_refs=self.round_one_repair_result_refs,
                        disposition=(AttachmentSemanticResolutionDispositionV2.CONFIRMED_RESOLVED),
                        successor_finding_ref=None,
                        evidence_ref_ids=("semantic-evidence://realism/coverage-resolution",),
                    ),
                ),
            )
        assert self.round_one_finding_ref is not None
        assert self.round_one_plan_ref is not None
        round_two_finding_ref = next(
            ref for ref in request.required_resolution_finding_refs if ref != self.round_one_finding_ref
        )
        round_two_plan_ref = next(
            ref for ref in request.prior_repair_plan_refs if ref != self.round_one_plan_ref
        )
        round_two_result_refs = tuple(
            ref for ref in request.prior_repair_result_refs if ref not in self.round_one_repair_result_refs
        )
        return ProviderSemanticReviewDecision(
            outcome=AttachmentSemanticReviewOutcomeV2.ACCEPTED,
            resolutions=tuple(
                ProviderSemanticResolutionDecision(
                    prior_finding_ref=finding_ref,
                    old_subject_refs=self.intermediate_subjects,
                    current_subject_refs=subjects,
                    repair_plan_ref=round_two_plan_ref,
                    repair_result_refs=round_two_result_refs,
                    disposition=(AttachmentSemanticResolutionDispositionV2.CONFIRMED_RESOLVED),
                    successor_finding_ref=None,
                    evidence_ref_ids=(f"semantic-evidence://leakage/resolution/{index}",),
                )
                for index, finding_ref in enumerate(
                    sorted(
                        (
                            self.round_one_finding_ref,
                            round_two_finding_ref,
                        ),
                        key=lambda ref: (
                            ref.object_type,
                            ref.object_id,
                            ref.object_version,
                            ref.object_sha256,
                        ),
                    ),
                    start=1,
                )
            ),
        )


class _SuccessfulRepairBackend:
    def __init__(self, output_bytes: bytes) -> None:
        self.output_bytes = output_bytes
        self.calls = 0

    async def repair(
        self,
        _context: ProviderAttachmentRepairContext,
    ) -> ProviderAttachmentRepairDecision:
        self.calls += 1
        return ProviderAttachmentRepairDecision.succeeded(
            output_bytes=self.output_bytes,
            worker_version="semantic-repair-fixture/v1",
        )


class _SequentialRepairBackend:
    def __init__(self, *outputs: bytes) -> None:
        self._outputs = outputs
        self.calls = 0

    async def repair(
        self,
        _context: ProviderAttachmentRepairContext,
    ) -> ProviderAttachmentRepairDecision:
        output = self._outputs[self.calls]
        self.calls += 1
        return ProviderAttachmentRepairDecision.succeeded(
            output_bytes=output,
            worker_version="semantic-repair-sequence/v1",
        )


class _PassingRevisionValidator:
    def __init__(self) -> None:
        self.revisions: list[AttachmentCandidateRevisionV2] = []
        self.results: list[RevisionDeterministicValidationV2] = []

    async def validate(
        self,
        revision: AttachmentCandidateRevisionV2,
    ) -> RevisionDeterministicValidationV2:
        self.revisions.append(revision)
        revision_ref = attachment_candidate_revision_ref(revision)
        source_ref = _ref(
            "deterministic-item-validation-result",
            f"revision-{revision.revision}",
        )
        result = RevisionDeterministicValidationV2.create(
            candidate_revision_ref=revision_ref,
            source_deterministic_validation_result_ref=source_ref,
            artifact_validation_result_refs=(
                _ref(
                    "artifact-deterministic-validation-result",
                    f"revision-{revision.revision}",
                ),
            ),
            finding_refs=(),
            output_refs=revision.candidate_output_refs,
            outcome=RevisionDeterministicValidationOutcomeV2.PASSED,
            audit=_audit(revision_ref, source_ref),
        )
        self.results.append(result)
        return result


class _BlockedRevisionValidator:
    def __init__(self) -> None:
        self.revisions: list[AttachmentCandidateRevisionV2] = []

    async def validate(
        self,
        revision: AttachmentCandidateRevisionV2,
    ) -> RevisionDeterministicValidationV2:
        self.revisions.append(revision)
        revision_ref = attachment_candidate_revision_ref(revision)
        source_ref = _ref(
            "deterministic-item-validation-result",
            f"blocked-revision-{revision.revision}",
        )
        return RevisionDeterministicValidationV2.create(
            candidate_revision_ref=revision_ref,
            source_deterministic_validation_result_ref=source_ref,
            artifact_validation_result_refs=(
                _ref(
                    "artifact-deterministic-validation-result",
                    f"blocked-revision-{revision.revision}",
                ),
            ),
            finding_refs=(),
            output_refs=revision.candidate_output_refs,
            outcome=RevisionDeterministicValidationOutcomeV2.BLOCKED,
            audit=_audit(revision_ref, source_ref),
        )


def _store(tmp_path: Path) -> tuple[JobStore, str, str]:
    store = JobStore(tmp_path / "job.sqlite3", clock=lambda: NOW)
    job = store.create_job(_job_spec())
    item = store.create_item(
        job.job_id,
        "item://semantic-review/test",
        idempotency_key="create-item-semantic-review",
    )
    return store, job.job_id, item.item_id


def test_r5_09_gold_is_content_free_and_covers_terminal_matrix() -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "eval-factory-r5-09-gold/v1"
    assert {item["scenario_id"] for item in payload["scenarios"]} == {
        "clean-three-round-pass",
        "pre-round-deterministic-repair",
        "round-one-repair-confirmed-by-round-two",
        "round-two-repair-confirmed-by-round-three",
        "round-three-non-waivable-reject",
        "item-scoped-requires-repair",
        "review-capability-blocked",
        "upstream-incomplete",
    }
    serialized = json.dumps(payload, sort_keys=True)
    for denied in (
        "artifact_content",
        "private_reference",
        "grader_rule",
        "hidden_reasoning",
        "runtime_transcript",
        "absolute_path",
    ):
        assert denied not in serialized


@pytest.mark.asyncio
async def test_candidate_revision_compiler_consumes_exact_r5_08_handoff() -> None:
    producer_view = _producer_view()
    reconstruction = _not_required_result(producer_view)
    deterministic = await DeterministicValidationCompiler().compile(
        reconstruction_result=reconstruction,
        producer_task_view=producer_view,
        leakage_reference_set=_reference_set(),
        configured_pii_rules=(),
        facade=_NoCallFacade(),
        audit=_audit(),
    )

    revision, wrapped = CandidateRevisionCompiler().compile_initial(
        reconstruction_result=reconstruction,
        deterministic_validation=deterministic,
        audit=_audit(),
    )

    assert revision.revision == 1
    assert revision.artifact_versions == ()
    assert revision.candidate_output_refs == ()
    assert wrapped.outcome is RevisionDeterministicValidationOutcomeV2.NOT_REQUIRED
    assert wrapped.output_refs == revision.candidate_output_refs


@pytest.mark.asyncio
async def test_complete_revision_validator_reuses_full_r5_08_compiler() -> None:
    artifacts = (("artifact://input", "inputs/source.txt"),)
    producer_view = _producer_view(artifacts)
    leakage_reference_set = _reference_set()
    revision = _revision()
    facade = _FakeValidationFacade()

    source = await DeterministicValidationCompiler().compile_revision(
        candidate_revision=revision,
        producer_task_view=producer_view,
        leakage_reference_set=leakage_reference_set,
        configured_pii_rules=(),
        facade=facade,
        audit=_audit(),
    )
    wrapped = await CompleteRevisionValidator(
        producer_task_view=producer_view,
        leakage_reference_set=leakage_reference_set,
        configured_pii_rules=(),
        facade=facade,
        audit=_audit(),
    ).validate(revision)

    assert source.attachment_reconstruction_result_ref == (attachment_candidate_revision_ref(revision))
    assert source.artifact_validation_results[0].artifact_build_result_ref == (
        revision.artifact_versions[0].base_artifact_build_result_ref
    )
    assert source.candidate_inventory is not None
    assert source.candidate_inventory.attachment_reconstruction_result_ref == (
        attachment_candidate_revision_ref(revision)
    )
    assert wrapped.outcome is RevisionDeterministicValidationOutcomeV2.PASSED
    assert wrapped.output_refs == revision.candidate_output_refs
    assert facade.calls == ["artifact://input", "artifact://input"]


@pytest.mark.asyncio
async def test_semantic_review_model_control_binds_grant_and_conservative_usage() -> None:
    revision = _revision()
    validation = _validation(revision)
    policy = _policy()
    round_ = SemanticReviewRoundV2.COVERAGE_SOLVABILITY
    request = _review_request(
        round_=round_,
        role=ROUND_ROLES[round_],
        stage_run_ref=_ref("stage-run", "model-controlled", version="identity/v1"),
        candidate_revision=revision,
        deterministic_validation=validation,
        context_view_ref=_ref("semantic-review-context-view", "model-controlled"),
        prior_round_result=None,
        prior_finding_refs=(),
        required_resolution_finding_refs=(),
        prior_resolution_refs=(),
        prior_repair_plan_refs=(),
        prior_repair_result_refs=(),
        policy=policy,
    )
    backend = _AcceptingBackend()
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends={request.reviewer_role: backend},
        repair_backend=None,
        clock=lambda: NOW,
    )
    wrong_descriptor, wrong_grant = _model_control(
        operation_ref=attachment_semantic_review_request_ref(request),
        model_profile_ref=_facade_ref(
            "model-profile",
            "wrong",
            version="v1",
        ),
        suffix="review-wrong",
    )
    with pytest.raises(ValueError, match="profile"):
        await facade.review_with_model_control(
            request=request,
            descriptor=wrong_descriptor,
            grant=wrong_grant,
        )
    assert backend.context_ids == []
    descriptor, grant = _model_control(
        operation_ref=attachment_semantic_review_request_ref(request),
        model_profile_ref=request.model_profile_ref,
        suffix="review",
    )

    result, receipt = await facade.review_with_model_control(
        request=request,
        descriptor=descriptor,
        grant=grant,
    )
    replay = await facade.review_with_model_control(
        request=request,
        descriptor=descriptor,
        grant=grant,
    )

    assert replay == (result, receipt)
    assert result is not None
    assert result.outcome is AttachmentSemanticReviewOutcomeV2.ACCEPTED
    assert receipt.outcome is ModelControlReceiptOutcomeV2.SUCCEEDED
    assert receipt.usage.source is ModelControlUsageSourceV2.CONSERVATIVE_ALLOWANCE
    assert receipt.usage.requests == grant.request_allowance
    assert receipt.usage.charged_tokens == grant.token_allowance
    assert len(backend.context_ids) == 1


@pytest.mark.asyncio
async def test_attachment_repair_model_control_blocks_before_missing_backend() -> None:
    source_bytes = b"source attachment"
    digest = hashlib.sha256(source_bytes).hexdigest()
    request = AttachmentRepairRequestV2(
        repair_request_id="attachment-repair-request://pending",
        repair_plan_ref=_facade_ref("targeted-repair-plan", "model-controlled"),
        source_round=AttachmentRepairSourceV2.DETERMINISTIC,
        candidate_revision_ref=_facade_ref(
            "attachment-candidate-revision",
            "model-controlled",
        ),
        artifact_id="artifact://model-controlled",
        artifact_version_ref=_facade_ref(
            "candidate-artifact-version",
            "model-controlled",
        ),
        build_spec_ref=_facade_ref("artifact-build-spec", "model-controlled"),
        execution_result_ref=_facade_ref(
            "attachment-execution-result",
            "model-controlled",
        ),
        output_ref=_facade_ref(
            "attachment-output",
            "model-controlled",
            digest=digest,
        ),
        output_sha256=digest,
        targeted_finding_refs=(_facade_ref("deterministic-validation-finding", "model-controlled"),),
        targeted_finding_codes=("FORMAT_INVALID",),
        attempt=1,
        prior_repair_result_ref=None,
        idempotency_key="repair-model-controlled",
        repair_request_sha256="0" * 64,
    )
    request_digest = attachment_repair_request_carried_sha256(request)
    request = request.model_copy(
        update={
            "repair_request_id": (f"attachment-repair-request://sha256/{request_digest}"),
            "repair_request_sha256": request_digest,
        }
    )
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(source_output_bytes=source_bytes),
        review_backends={},
        repair_backend=None,
        clock=lambda: NOW,
    )
    descriptor, grant = _model_control(
        operation_ref=attachment_repair_request_ref(request),
        model_profile_ref=_facade_ref(
            "model-profile",
            "repair",
            version="v1",
        ),
        suffix="repair",
    )

    result, receipt = await facade.repair_with_model_control(
        request=request,
        descriptor=descriptor,
        grant=grant,
    )

    assert result is not None
    assert result.outcome is AttachmentRepairOutcomeV2.BLOCKED_CAPABILITY
    assert receipt.outcome is ModelControlReceiptOutcomeV2.BLOCKED_CAPABILITY
    assert receipt.usage.charged_tokens == 0


@pytest.mark.asyncio
async def test_clean_workflow_runs_three_distinct_ordered_stage_runs(
    tmp_path: Path,
) -> None:
    store, job_id, item_id = _store(tmp_path)
    backend = _AcceptingBackend()
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends={role: backend for role in (_policy().roles_in_order)},
        repair_backend=_NoRepairBackend(),
    )
    revision = _revision()
    validation = _validation(revision)
    policy = _policy()
    orchestrator = IsolatedSemanticReviewOrchestrator()

    result = await orchestrator.run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=validation,
        review_policy=policy,
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=_NoRevalidation(),
        audit=_audit(),
    )

    assert result.outcome is SemanticReviewWorkflowOutcomeV2.PASSED
    assert [item.round for item in result.round_results] == list(SemanticReviewRoundV2)
    assert len(result.stage_result_refs) == 3
    assert len(set(backend.context_ids)) == 3
    stage_runs = store.list_stage_runs(
        job_id=job_id,
        item_id=item_id,
        stage=StageNameV2.ITEM_QUALITY,
    )
    assert len(stage_runs) == 3
    assert {item.attempt for item in stage_runs} == {1}
    assert all(item.retry_of_stage_run_id is None for item in stage_runs)
    assert all(item.status is StageRunStatus.SUCCEEDED for item in stage_runs)
    assert all(store.get_stage_result_for_run(item.stage_run_id) is not None for item in stage_runs)
    assert result.accepted_artifact_refs == ()
    assert result.quality_report_ref is None
    assert result.input_state_only is None
    orchestrator.validate_current(
        result,
        candidate_revision=revision,
        deterministic_validation=validation,
        review_policy=policy,
        job_store=store,
        job_id=job_id,
        item_id=item_id,
    )
    changed_policy = SemanticReviewPolicyV2.create(
        model_profile_refs=dict(
            zip(
                policy.rounds_in_order,
                policy.model_profile_refs,
                strict=True,
            )
        ),
        prompt_versions={
            round_: (
                "semantic-review/coverage_solvability/v2"
                if round_ is SemanticReviewRoundV2.COVERAGE_SOLVABILITY
                else policy.prompt_for(round_)
            )
            for round_ in policy.rounds_in_order
        },
        projection_policy_refs=dict(
            zip(
                policy.rounds_in_order,
                policy.projection_policy_refs,
                strict=True,
            )
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="candidate or validation is stale",
    ):
        orchestrator.validate_current(
            result,
            candidate_revision=revision,
            deterministic_validation=validation,
            review_policy=changed_policy,
            job_store=store,
            job_id=job_id,
            item_id=item_id,
        )


@pytest.mark.asyncio
async def test_clean_workflow_replay_reuses_stage_results_and_backend_results(
    tmp_path: Path,
) -> None:
    store, job_id, item_id = _store(tmp_path)
    backend = _AcceptingBackend()
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends={role: backend for role in _policy().roles_in_order},
        repair_backend=_NoRepairBackend(),
    )
    revision = _revision()
    validation = _validation(revision)
    orchestrator = IsolatedSemanticReviewOrchestrator()

    first = await orchestrator.run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=validation,
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=_NoRevalidation(),
        audit=_audit(),
    )
    outbox_count = len(store.list_outbox())
    replay = await orchestrator.run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=validation,
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=_NoRevalidation(),
        audit=_audit(),
    )

    assert replay == first
    assert len(backend.context_ids) == 3
    assert len(store.list_outbox()) == outbox_count
    assert (
        len(
            store.list_stage_runs(
                job_id=job_id,
                item_id=item_id,
                stage=StageNameV2.ITEM_QUALITY,
            )
        )
        == 3
    )


@pytest.mark.asyncio
async def test_item_scoped_requires_repair_stops_without_artifact_repair(
    tmp_path: Path,
) -> None:
    store, job_id, item_id = _store(tmp_path)
    repair_backend = _SuccessfulRepairBackend(b"must not be used")
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends={role: _ItemScopedRepairBackend() for role in _policy().roles_in_order},
        repair_backend=repair_backend,
    )
    revision = _revision()

    result = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=_validation(revision),
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=_NoRevalidation(),
        audit=_audit(),
    )

    assert result.outcome is SemanticReviewWorkflowOutcomeV2.REQUIRES_REPAIR
    assert len(result.round_results) == 1
    assert len(result.current_finding_refs) == 1
    assert result.repair_plan_refs == ()
    assert result.repair_result_refs == ()
    assert repair_backend.calls == 0


@pytest.mark.asyncio
async def test_model_unavailable_completes_blocked_capability_stage_result(
    tmp_path: Path,
) -> None:
    store, job_id, item_id = _store(tmp_path)
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends={},
        repair_backend=None,
    )
    revision = _revision()

    result = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=_validation(revision),
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=_NoRevalidation(),
        audit=_audit(),
    )

    assert result.outcome is SemanticReviewWorkflowOutcomeV2.BLOCKED
    stage = store.list_stage_runs(
        job_id=job_id,
        item_id=item_id,
        stage=StageNameV2.ITEM_QUALITY,
    )[0]
    persisted = store.get_stage_result_for_run(stage.stage_run_id)
    assert stage.status is StageRunStatus.BLOCKED_CAPABILITY
    assert persisted is not None
    assert persisted.status is StageRunStatus.BLOCKED_CAPABILITY
    assert persisted.failure is not None
    assert persisted.failure.code == "MODEL_UNAVAILABLE"


@pytest.mark.asyncio
async def test_abstained_review_blocks_progress_with_indeterminate_failure(
    tmp_path: Path,
) -> None:
    store, job_id, item_id = _store(tmp_path)
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends={role: _AbstainingBackend() for role in _policy().roles_in_order},
        repair_backend=None,
    )
    revision = _revision()

    result = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=_validation(revision),
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=_NoRevalidation(),
        audit=_audit(),
    )

    assert result.outcome is SemanticReviewWorkflowOutcomeV2.BLOCKED
    assert len(result.round_results) == 1
    stage = store.list_stage_runs(
        job_id=job_id,
        item_id=item_id,
        stage=StageNameV2.ITEM_QUALITY,
    )[0]
    persisted = store.get_stage_result_for_run(stage.stage_run_id)
    assert stage.status is StageRunStatus.BLOCKED_CAPABILITY
    assert persisted is not None and persisted.failure is not None
    assert persisted.failure.code == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_round_three_requires_repair_without_self_repair(
    tmp_path: Path,
) -> None:
    store, job_id, item_id = _store(tmp_path)
    repair_backend = _SuccessfulRepairBackend(b"must not be used")
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends={role: _RoundThreeRepairBackend() for role in _policy().roles_in_order},
        repair_backend=repair_backend,
    )
    revision = _revision()

    result = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=_validation(revision),
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=_NoRevalidation(),
        audit=_audit(),
    )

    assert result.outcome is SemanticReviewWorkflowOutcomeV2.REQUIRES_REPAIR
    assert [item.round for item in result.round_results] == list(SemanticReviewRoundV2)
    assert result.repair_plan_refs == ()
    assert result.repair_result_refs == ()
    assert len(result.current_finding_refs) == 1
    assert repair_backend.calls == 0
    assert (
        len(
            store.list_stage_runs(
                job_id=job_id,
                item_id=item_id,
                stage=StageNameV2.ITEM_QUALITY,
            )
        )
        == 3
    )


@pytest.mark.asyncio
async def test_round_three_non_waivable_finding_rejects_without_repair(
    tmp_path: Path,
) -> None:
    store, job_id, item_id = _store(tmp_path)
    repair_backend = _SuccessfulRepairBackend(b"must not be used")
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends={role: _RoundThreeRejectBackend() for role in _policy().roles_in_order},
        repair_backend=repair_backend,
    )
    revision = _revision()

    result = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=_validation(revision),
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=_NoRevalidation(),
        audit=_audit(),
    )

    assert result.outcome is SemanticReviewWorkflowOutcomeV2.REJECTED
    assert len(result.round_results) == 3
    assert len(result.current_finding_refs) == 1
    assert result.semantic_findings[0].non_waivable is True
    assert result.semantic_findings[0].severity.value == "P0"
    assert repair_backend.calls == 0
    assert all(
        item.status is StageRunStatus.SUCCEEDED
        for item in store.list_stage_runs(
            job_id=job_id,
            item_id=item_id,
            stage=StageNameV2.ITEM_QUALITY,
        )
    )


@pytest.mark.asyncio
async def test_round_one_repair_creates_successor_and_revalidates_before_round_two(
    tmp_path: Path,
) -> None:
    source_bytes = b"original incomplete input"
    repaired_bytes = b"repaired input with required coverage"
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    store, job_id, item_id = _store(tmp_path)
    review_backend = _RepairingReviewBackend()
    repair_backend = _SuccessfulRepairBackend(repaired_bytes)
    validator = _PassingRevisionValidator()
    resolver = _DynamicResolver(source_bytes)
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=resolver,
        review_backends={role: review_backend for role in _policy().roles_in_order},
        repair_backend=repair_backend,
        repaired_output_registrar=resolver,
    )
    revision = _revision(output_digest=source_digest)

    result = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=_validation(revision),
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=validator,
        audit=_audit(),
    )

    assert result.outcome is SemanticReviewWorkflowOutcomeV2.PASSED
    assert result.current_candidate_revision.revision == 2
    assert result.current_candidate_revision.changed_artifact_ids == ("artifact://input",)
    assert result.current_candidate_revision.candidate_output_refs[0].object_sha256 == (
        hashlib.sha256(repaired_bytes).hexdigest()
    )
    assert len(result.repair_plan_refs) == 1
    assert len(result.repair_result_refs) == 1
    assert len(result.deterministic_validation_result_refs) == 2
    assert len(result.stale_finding_refs) == 1
    assert result.current_finding_refs == ()
    assert repair_backend.calls == 1
    assert [item.revision for item in validator.revisions] == [2]
    assert review_backend.rounds == list(AttachmentSemanticReviewRoundV2)
    assert (
        len(
            store.list_stage_runs(
                job_id=job_id,
                item_id=item_id,
                stage=StageNameV2.ITEM_QUALITY,
            )
        )
        == 3
    )


@pytest.mark.asyncio
async def test_blocked_revalidation_stops_before_round_two(
    tmp_path: Path,
) -> None:
    source_bytes = b"original incomplete input"
    repaired_bytes = b"repaired input"
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    store, job_id, item_id = _store(tmp_path)
    review_backend = _RepairingReviewBackend()
    repair_backend = _SuccessfulRepairBackend(repaired_bytes)
    validator = _BlockedRevisionValidator()
    resolver = _DynamicResolver(source_bytes)
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=resolver,
        review_backends={role: review_backend for role in _policy().roles_in_order},
        repair_backend=repair_backend,
        repaired_output_registrar=resolver,
    )
    revision = _revision(output_digest=source_digest)

    result = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=_validation(revision),
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=validator,
        audit=_audit(),
    )

    assert result.outcome is SemanticReviewWorkflowOutcomeV2.BLOCKED
    assert result.current_candidate_revision.revision == 2
    assert len(result.round_results) == 1
    assert len(result.deterministic_validation_result_refs) == 2
    assert review_backend.rounds == [AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY]
    assert [item.revision for item in validator.revisions] == [2]


@pytest.mark.asyncio
async def test_round_two_repair_reconfirms_stale_prior_resolution(
    tmp_path: Path,
) -> None:
    source_bytes = b"original incomplete input"
    first_repair = b"coverage-complete but unrealistic input"
    second_repair = b"coverage-complete realistic input"
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    store, job_id, item_id = _store(tmp_path)
    review_backend = _TwoRepairReviewBackend()
    repair_backend = _SequentialRepairBackend(first_repair, second_repair)
    validator = _PassingRevisionValidator()
    resolver = _DynamicResolver(source_bytes)
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=resolver,
        review_backends={role: review_backend for role in _policy().roles_in_order},
        repair_backend=repair_backend,
        repaired_output_registrar=resolver,
    )
    revision = _revision(output_digest=source_digest)
    initial_validation = _validation(revision)
    policy = _policy()
    orchestrator = IsolatedSemanticReviewOrchestrator()

    result = await orchestrator.run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=initial_validation,
        review_policy=policy,
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=validator,
        audit=_audit(),
    )

    assert result.outcome is SemanticReviewWorkflowOutcomeV2.PASSED
    assert result.current_candidate_revision.revision == 3
    assert len(result.repair_plan_refs) == 2
    assert len(result.repair_result_refs) == 2
    assert len(result.semantic_findings) == 2
    assert len(result.semantic_resolutions) == 3
    assert len(result.stale_finding_refs) == 2
    assert result.current_finding_refs == ()
    assert (
        sum(
            semantic_finding_resolution_is_current(
                item,
                result.current_candidate_revision,
            )
            for item in result.semantic_resolutions
        )
        == 2
    )
    assert repair_backend.calls == 2
    assert [item.revision for item in validator.revisions] == [2, 3]
    assert review_backend.rounds == list(AttachmentSemanticReviewRoundV2)
    orchestrator.validate_current(
        result,
        candidate_revision=result.current_candidate_revision,
        deterministic_validation=validator.results[-1],
        review_policy=policy,
        job_store=store,
        job_id=job_id,
        item_id=item_id,
    )


@pytest.mark.asyncio
async def test_missing_later_resolution_blocks_after_round_one_repair(
    tmp_path: Path,
) -> None:
    source_bytes = b"original incomplete input"
    repaired_bytes = b"repaired input with required coverage"
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    store, job_id, item_id = _store(tmp_path)
    review_backend = _MissingResolutionBackend()
    repair_backend = _SuccessfulRepairBackend(repaired_bytes)
    validator = _PassingRevisionValidator()
    resolver = _DynamicResolver(source_bytes)
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=resolver,
        review_backends={role: review_backend for role in _policy().roles_in_order},
        repair_backend=repair_backend,
        repaired_output_registrar=resolver,
    )
    revision = _revision(output_digest=source_digest)

    result = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=_validation(revision),
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=validator,
        audit=_audit(),
    )

    assert result.outcome is SemanticReviewWorkflowOutcomeV2.BLOCKED
    stages = store.list_stage_runs(
        job_id=job_id,
        item_id=item_id,
        stage=StageNameV2.ITEM_QUALITY,
    )
    assert [item.status for item in stages] == [
        StageRunStatus.SUCCEEDED,
        StageRunStatus.BLOCKED_POLICY,
    ]
    blocked = store.get_stage_result_for_run(stages[-1].stage_run_id)
    assert blocked is not None and blocked.failure is not None
    assert blocked.failure.code == "BACKEND_OUTPUT_INVALID"


@pytest.mark.asyncio
async def test_repair_capability_block_returns_blocked_workflow_without_successor(
    tmp_path: Path,
) -> None:
    source_bytes = b"original incomplete input"
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    store, job_id, item_id = _store(tmp_path)
    review_backend = _RepairingReviewBackend()
    repair_backend = _SuccessfulRepairBackend(b"repaired input")
    validator = _PassingRevisionValidator()
    revision = _revision(output_digest=source_digest)
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(source_bytes),
        review_backends={role: review_backend for role in _policy().roles_in_order},
        repair_backend=repair_backend,
        repaired_output_registrar=None,
    )

    result = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=_validation(revision),
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=validator,
        audit=_audit(),
    )

    assert result.outcome is SemanticReviewWorkflowOutcomeV2.BLOCKED
    assert result.current_candidate_revision == revision
    assert len(result.round_results) == 1
    assert len(result.repair_plan_refs) == 1
    assert len(result.repair_result_refs) == 1
    assert result.repair_result_refs[0].object_type == "attachment-repair-result"
    assert len(result.current_finding_refs) == 1
    assert validator.revisions == []
    stages = store.list_stage_runs(
        job_id=job_id,
        item_id=item_id,
        stage=StageNameV2.ITEM_QUALITY,
    )
    assert len(stages) == 1
    assert stages[0].status is StageRunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_pre_round_deterministic_repair_revalidates_before_any_stage_run(
    tmp_path: Path,
) -> None:
    source_bytes = b"input with configured pii"
    repaired_bytes = b"redacted safe input"
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    store, job_id, item_id = _store(tmp_path)
    review_backend = _AcceptingBackend()
    repair_backend = _SuccessfulRepairBackend(repaired_bytes)
    validator = _PassingRevisionValidator()
    resolver = _DynamicResolver(source_bytes)
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=resolver,
        review_backends={role: review_backend for role in _policy().roles_in_order},
        repair_backend=repair_backend,
        repaired_output_registrar=resolver,
    )
    revision = _revision(output_digest=source_digest)
    repair_target = DeterministicRepairTargetV2(
        artifact_id="artifact://input",
        finding_refs=(_ref("deterministic-validation-finding", "pii"),),
        finding_codes=("CONFIGURED_PII_DETECTED",),
    )

    result = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=_validation(
            revision,
            RevisionDeterministicValidationOutcomeV2.REQUIRES_REPAIR,
            repair_targets=(repair_target,),
        ),
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=validator,
        audit=_audit(),
    )

    assert result.outcome is SemanticReviewWorkflowOutcomeV2.PASSED
    assert result.current_candidate_revision.revision == 2
    assert len(result.repair_plan_refs) == 1
    assert len(result.repair_result_refs) == 1
    assert len(result.deterministic_validation_result_refs) == 2
    assert result.stale_finding_refs == repair_target.finding_refs
    assert repair_backend.calls == 1
    assert [item.revision for item in validator.revisions] == [2]
    assert (
        len(
            store.list_stage_runs(
                job_id=job_id,
                item_id=item_id,
                stage=StageNameV2.ITEM_QUALITY,
            )
        )
        == 3
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (
            RevisionDeterministicValidationOutcomeV2.UPSTREAM_INCOMPLETE,
            SemanticReviewWorkflowOutcomeV2.UPSTREAM_INCOMPLETE,
        ),
        (
            RevisionDeterministicValidationOutcomeV2.REJECTED,
            SemanticReviewWorkflowOutcomeV2.REJECTED,
        ),
        (
            RevisionDeterministicValidationOutcomeV2.BLOCKED,
            SemanticReviewWorkflowOutcomeV2.BLOCKED,
        ),
    ],
)
async def test_non_admitted_validation_starts_no_semantic_stage(
    tmp_path: Path,
    outcome: RevisionDeterministicValidationOutcomeV2,
    expected: SemanticReviewWorkflowOutcomeV2,
) -> None:
    store, job_id, item_id = _store(tmp_path)
    revision = _revision()
    backend = _AcceptingBackend()
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends={role: backend for role in _policy().roles_in_order},
        repair_backend=_NoRepairBackend(),
    )

    result = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=revision,
        deterministic_validation=_validation(revision, outcome),
        review_policy=_policy(),
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=_NoRevalidation(),
        audit=_audit(),
    )

    assert result.outcome is expected
    assert result.round_results == ()
    assert backend.context_ids == []
    assert (
        store.list_stage_runs(
            job_id=job_id,
            item_id=item_id,
            stage=StageNameV2.ITEM_QUALITY,
        )
        == ()
    )
