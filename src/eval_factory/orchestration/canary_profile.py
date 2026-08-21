from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import Field, model_validator

from env_mock_agent.facade import (
    AttachmentExecutionFacade,
    AttachmentExecutionRequestV2,
    AttachmentExecutionResultV2,
    AttachmentRepairRequestV2,
    AttachmentRepairResultV2,
    AttachmentResourceGrantV2,
    AttachmentSemanticReviewOutcomeV2,
    AttachmentSemanticReviewRequestV2,
    AttachmentSemanticReviewResultV2,
    AttachmentValidationRequestV2,
    AttachmentValidationResultV2,
    CanaryTextAttachmentFacades,
    CanaryTextExecutionFixture,
    FacadeObjectRef,
    ResourceBoundAttachmentExecutionFacade,
    ResourceBoundAttachmentExecutionResultV2,
    SemanticCleanContextAttestationV2,
    WorldLedgerSnapshotRequestV2,
    WorldLedgerSnapshotV2,
    attachment_execution_request_ref,
    attachment_resource_grant_ref,
    attachment_semantic_review_request_ref,
    attachment_semantic_review_result_carried_sha256,
    create_canary_text_routing_facade,
)
from eval_factory.attachment_planning import (
    ARTIFACT_EVIDENCE_MODE_POLICY_VERSION,
    ArtifactBuildContractDefinition,
    ArtifactBuildSpecCompiler,
    ArtifactConsistencyDefinition,
    ArtifactEvidenceModeCompiler,
    ArtifactExecutionPlanCompiler,
    ArtifactExecutionPreparationBuilder,
    ArtifactGroupExecutor,
    ArtifactResultCompiler,
    ArtifactRoutingRequestBuilder,
    ArtifactRoutingRunner,
    AttachmentPlanningBridge,
    CandidateRevisionCompiler,
    DeterministicValidationCompiler,
    ItemQualityCompiler,
    SemanticReviewContextCompiler,
    SemanticReviewContextSources,
    SemanticReviewRoundExecutor,
    SemanticReviewWorkflowCompiler,
    WorldLedgerSnapshotRunner,
)
from eval_factory.attachment_planning.execution_models import (
    ArtifactExecutionPlanningOutcome,
    ArtifactExecutionPlanningResult,
)
from eval_factory.attachment_planning.routing_models import (
    ArtifactRoutingCompilationResult,
    ArtifactRoutingRequest,
    artifact_routing_compilation_result_carried_sha256,
    artifact_routing_request_carried_sha256,
    artifact_routing_request_ref,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactEvidenceTargetV2,
    ArtifactExecutionBatchV2,
    ArtifactRoutingAggregateOutcomeV2,
    ArtifactRoutingPolicyV2,
    AttachmentReconstructionResultV2,
    artifact_build_spec_v2_ref,
    artifact_evidence_target_carried_sha256,
    artifact_execution_batch_ref,
    artifact_execution_receipt_ref,
    artifact_routing_policy_carried_sha256,
)
from eval_factory.contracts.canary_execution_v2 import (
    R6CanaryExecutionManifestV2,
    R6CanaryObjectCodecV2,
    R6CanaryTraceBindingV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    ObjectRef,
)
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.labeling_v2 import LabelSpecV2
from eval_factory.contracts.orchestration import ItemStatus, StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    ArtifactGroupFanoutV2,
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    StageNameV2,
    WorkUnitScopeV2,
    resolved_work_unit_v2_ref,
)
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.resource_v2 import (
    NonModelResourceVectorV2,
    WorkResourceReservationV2,
)
from eval_factory.contracts.review_v2 import (
    AttachmentCandidateRevisionV2,
    RevisionDeterministicValidationV2,
    SemanticReviewPolicyV2,
    SemanticReviewRoundResultV2,
    SemanticReviewRoundV2,
    SemanticReviewWorkflowOutcomeV2,
)
from eval_factory.contracts.safety import (
    EvidenceBundle,
    OriginClass,
    ProvenanceDecision,
    Visibility,
)
from eval_factory.contracts.task import AttachmentDependency
from eval_factory.contracts.task_v2 import (
    ContestantToolPolicyV2,
    EvaluatorSpecV2,
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
    PromptLeakageReferenceSetV2,
    R4TaskContractSetV2,
    ReferencePolicyV2,
    RubricSetV2,
    TaskDraftV2,
    TaskPromptSafetyGateV2,
    ToolPolicyV2,
)
from eval_factory.contracts.validation_v2 import (
    DeterministicItemValidationResultV2,
)
from eval_factory.labeling.decision import (
    LabelDecisionMerger,
    LabelDecisionMergeRequest,
    LabelDecisionMergeResult,
)
from eval_factory.labeling.structured import (
    StructuredFactSet,
    StructuredLabelResult,
    StructuredPredicateCompiler,
)
from eval_factory.orchestration.canary_errors import (
    CanaryProfileError,
)
from eval_factory.orchestration.fanout import ArtifactGroupFanoutCompiler
from eval_factory.orchestration.job_store import (
    JobStore,
    RecordNotFoundError,
)
from eval_factory.orchestration.models import StageRunRecord
from eval_factory.orchestration.resources import (
    ResourceControlService,
)
from eval_factory.orchestration.runner import (
    TraceIndexStageService,
)
from eval_factory.orchestration.stage_object_store import (
    StageObjectCodecDefinition,
    StageObjectCodecRegistry,
    StageObjectStore,
    canary_stage_object_codec_registry,
)
from eval_factory.provenance.decisions import (
    ProvenanceDecisionInput,
    ProvenanceDecisionTable,
)
from eval_factory.provenance.views import EvidenceViewResult
from eval_factory.task_authoring.rewrites import (
    R4TaskContractSetCompiler,
)


class R6CanaryTaskFixture(ContractModelV2):
    schema_version: str = "eval-factory/r6-canary-task-fixture/v1"
    fixture_id: str
    task_draft: TaskDraftV2
    task_prompt_safety_gate: TaskPromptSafetyGateV2
    rubric_set: RubricSetV2
    evaluator_spec: EvaluatorSpecV2
    reference_policy: ReferencePolicyV2
    tool_policy: ToolPolicyV2
    contestant_tool_policy: ContestantToolPolicyV2
    producer_storage_authorization: ProducerStorageAuthorizationV2
    producer_task_view: ProducerTaskViewV2
    leakage_reference_set: PromptLeakageReferenceSetV2
    producer_view_result: EvidenceViewResult | None = None
    producer_evidence_bundle: EvidenceBundle | None = None
    text_provider_content: str | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
    )
    attachment_relative_path: str | None = Field(
        default=None,
        pattern=r"^inputs/[A-Za-z0-9._ -]+\.txt$",
    )

    @model_validator(mode="after")
    def validate_attachment_fixture(self) -> R6CanaryTaskFixture:
        has_requirements = bool(self.producer_task_view.attachment_requirements)
        values = (
            self.producer_view_result,
            self.producer_evidence_bundle,
            self.text_provider_content,
            self.attachment_relative_path,
        )
        if has_requirements:
            if (
                len(self.producer_task_view.attachment_requirements) != 1
                or len(self.task_draft.attachment_dependencies) != 1
                or any(value is None for value in values)
            ):
                raise ValueError("text canary requires one complete attachment fixture")
        elif any(value is not None for value in values):
            raise ValueError("no-attachment canary cannot carry provider material")
        return self


class R6CanaryReviewFixture(ContractModelV2):
    schema_version: str = "eval-factory/r6-canary-review-fixture/v1"
    fixture_id: str
    review_policy: SemanticReviewPolicyV2
    context_sources: SemanticReviewContextSources


class R6CanaryTraceIndexOutput(ContractModel):
    schema_version: str = "eval-factory/r6-canary-trace-index-output/v1"
    item_id: str
    structured_fact_set: StructuredFactSet


class R6CanarySafetyOutput(ContractModel):
    schema_version: str = "eval-factory/r6-canary-safety-output/v1"
    item_id: str
    provenance_decision: ProvenanceDecision


class R6CanaryLabelOutput(ContractModel):
    schema_version: str = "eval-factory/r6-canary-label-output/v1"
    item_id: str
    structured_result: StructuredLabelResult
    merge_result: LabelDecisionMergeResult


class R6CanaryTaskOutput(ContractModel):
    schema_version: str = "eval-factory/r6-canary-task-output/v1"
    item_id: str
    task_contract_set: R4TaskContractSetV2
    producer_task_view: ProducerTaskViewV2
    leakage_reference_set: PromptLeakageReferenceSetV2
    task_draft: TaskDraftV2
    producer_storage_authorization: ProducerStorageAuthorizationV2
    producer_view_result: EvidenceViewResult | None = None
    producer_evidence_bundle: EvidenceBundle | None = None
    text_provider_content: str | None = None
    attachment_relative_path: str | None = None


class R6CanaryAttachmentPreparationOutput(ContractModel):
    schema_version: str = "eval-factory/r6-canary-attachment-preparation-output/v1"
    job_id: str
    item_id: str
    execution_result: ArtifactExecutionPlanningResult

    @model_validator(mode="after")
    def validate_preparation(
        self,
    ) -> R6CanaryAttachmentPreparationOutput:
        plan = self.execution_result.execution_plan
        if (
            self.execution_result.outcome is not ArtifactExecutionPlanningOutcome.PLANNED
            or plan is None
            or len(plan.groups) != 1
        ):
            raise ValueError("text canary preparation requires exactly one execution group")
        return self


class R6CanaryProviderOperationOutput(ContractModel):
    schema_version: str = "eval-factory/r6-canary-provider-operation-output/v1"
    job_id: str
    item_id: str
    work_unit_ref: ObjectRef
    execution_request_ref: ObjectRef
    resource_result: ResourceBoundAttachmentExecutionResultV2

    @model_validator(mode="after")
    def validate_operation(
        self,
    ) -> R6CanaryProviderOperationOutput:
        if (
            self.work_unit_ref.object_type != "resolved-work-unit"
            or self.work_unit_ref.object_version != "v2"
            or self.execution_request_ref
            != _object_ref(self.resource_result.execution_result.execution_request_ref)
        ):
            raise ValueError("provider operation journal bindings are invalid")
        return self


class R6CanaryArtifactGroupOutput(ContractModel):
    schema_version: str = "eval-factory/r6-canary-artifact-group-output/v1"
    job_id: str
    item_id: str
    work_unit_ref: ObjectRef
    execution_batch: ArtifactExecutionBatchV2
    resource_usage: NonModelResourceVectorV2

    @model_validator(mode="after")
    def validate_group_output(self) -> R6CanaryArtifactGroupOutput:
        if (
            self.work_unit_ref.object_type != "resolved-work-unit"
            or self.work_unit_ref.object_version != "v2"
            or len(self.execution_batch.receipts) != 1
        ):
            raise ValueError("text canary group output requires one exact work unit receipt")
        return self


class R6CanaryAttachmentOutput(ContractModel):
    schema_version: str = "eval-factory/r6-canary-attachment-output/v1"
    item_id: str
    reconstruction_result: AttachmentReconstructionResultV2
    source_deterministic_validation: DeterministicItemValidationResultV2
    candidate_revision: AttachmentCandidateRevisionV2
    deterministic_validation: RevisionDeterministicValidationV2


@dataclass(frozen=True)
class _TextAttachmentResult:
    reconstruction: AttachmentReconstructionResultV2
    source_validation: DeterministicItemValidationResultV2


@dataclass(frozen=True)
class _TextAttachmentPlan:
    routing_request: ArtifactRoutingRequest
    routing_result: ArtifactRoutingCompilationResult
    execution_result: ArtifactExecutionPlanningResult
    physical: CanaryTextAttachmentFacades


@dataclass(frozen=True)
class CanaryStageExecution:
    output_refs: tuple[ObjectRef, ...]
    checkpoint_ref: ObjectRef | None = None
    item_terminal_status: ItemStatus | None = None
    resource_usage: NonModelResourceVectorV2 | None = None
    deferred: bool = False


@dataclass(frozen=True)
class CanaryHandlerContext:
    manifest: R6CanaryExecutionManifestV2
    binding: R6CanaryTraceBindingV2
    graph: ResolvedJobWorkGraphV2
    work_unit: ResolvedWorkUnitV2
    stage_run: StageRunRecord | None
    stage_store: StageObjectStore
    job_store: JobStore
    trace_index_service: TraceIndexStageService
    dependency_output_refs: tuple[ObjectRef, ...]
    item_result_refs: tuple[ObjectRef, ...]
    resource_control: ResourceControlService | None = None
    resource_reservation: WorkResourceReservationV2 | None = None
    artifact_group_fanout: ArtifactGroupFanoutV2 | None = None
    private_root: Path | None = None

    def seed[T: ContractModel](
        self,
        codec: R6CanaryObjectCodecV2,
        refs: tuple[ObjectRef, ...],
        model_type: type[T],
    ) -> T:
        seeds = {seed.object_ref: seed for seed in self.manifest.seed_objects}
        matches = tuple(ref for ref in refs if ref in seeds and seeds[ref].codec is codec)
        if len(matches) != 1:
            raise CanaryProfileError(f"expected exactly one {codec.value} seed")
        return self.stage_store.get_as(
            matches[0],
            codec=codec,
            model_type=model_type,
        )

    def dependency[T: ContractModel](
        self,
        codec: R6CanaryObjectCodecV2,
        model_type: type[T],
    ) -> T:
        expected_type = self.stage_store.registry.definition(codec).object_type
        matches = tuple(ref for ref in self.dependency_output_refs if ref.object_type == expected_type)
        if len(matches) != 1:
            raise CanaryProfileError(f"expected exactly one {codec.value} dependency")
        return self.stage_store.get_as(
            matches[0],
            codec=codec,
            model_type=model_type,
        )


class CanaryStageHandler(Protocol):
    async def execute(
        self,
        context: CanaryHandlerContext,
    ) -> CanaryStageExecution: ...


class R6CanaryStageHandler:
    async def execute(
        self,
        context: CanaryHandlerContext,
    ) -> CanaryStageExecution:
        if context.work_unit.scope is WorkUnitScopeV2.ARTIFACT_GROUP:
            return await self._artifact_group(context)
        round_ = context.work_unit.semantic_review_round
        if round_ is not None:
            return await self._review_round(context, round_)
        if context.work_unit.stage is StageNameV2.ATTACHMENT:
            return await self._attachment(context)
        return {
            StageNameV2.TRACE_INDEX: self._trace_index,
            StageNameV2.SAFETY: self._safety,
            StageNameV2.LABEL: self._label,
            StageNameV2.TASK_AUTHORING: self._task_authoring,
            StageNameV2.ITEM_QUALITY: self._item_quality,
        }[context.work_unit.stage](context)

    def _trace_index(
        self,
        context: CanaryHandlerContext,
    ) -> CanaryStageExecution:
        trace = next(
            trace
            for trace in context.manifest.job_spec.traces
            if trace.source_trace_id == context.binding.source_trace_ref.object_id
        )
        execution = context.trace_index_service.execute(
            trace=trace,
            audit=context.manifest.audit,
            job_id=context.work_unit.job_id,
            attempt=_running_stage(context).attempt,
        )
        value = R6CanaryTraceIndexOutput(
            item_id=_item_id(context),
            structured_fact_set=StructuredFactSet.from_stored_index(
                execution.stored_index,
                audit=context.manifest.audit,
            ),
        )
        observed_fact_ref = context.stage_store.registry.reference(
            R6CanaryObjectCodecV2.STRUCTURED_FACT_SET,
            value.structured_fact_set,
        )
        if observed_fact_ref != (context.binding.structured_fact_set_ref):
            raise CanaryProfileError("TRACE_INDEX output differs from manifest structured fact binding")
        write = context.stage_store.put(
            R6CanaryObjectCodecV2.TRACE_INDEX_OUTPUT,
            value,
        )
        return CanaryStageExecution(
            output_refs=(write.object_ref,),
            checkpoint_ref=execution.trace_envelope_ref,
        )

    def _safety(
        self,
        context: CanaryHandlerContext,
    ) -> CanaryStageExecution:
        trace_output = context.dependency(
            R6CanaryObjectCodecV2.TRACE_INDEX_OUTPUT,
            R6CanaryTraceIndexOutput,
        )
        subject = trace_output.structured_fact_set.trace_envelope_ref
        decision = ProvenanceDecisionTable().decide(
            ProvenanceDecisionInput(
                subject_ref=subject,
                origin_class=OriginClass.SYSTEM_OR_HARNESS_CONTEXT,
                visibility=Visibility.STAGE_PROJECTION,
                rule_ids=("r6-canary-safe-trace-envelope",),
                confidence=1.0,
                audit=context.manifest.audit,
            )
        )
        write = context.stage_store.put(
            R6CanaryObjectCodecV2.SAFETY_OUTPUT,
            R6CanarySafetyOutput(
                item_id=_item_id(context),
                provenance_decision=decision,
            ),
        )
        return CanaryStageExecution(output_refs=(write.object_ref,))

    def _label(
        self,
        context: CanaryHandlerContext,
    ) -> CanaryStageExecution:
        trace_output = _item_object(
            context,
            R6CanaryObjectCodecV2.TRACE_INDEX_OUTPUT,
            R6CanaryTraceIndexOutput,
        )
        label_spec = context.seed(
            R6CanaryObjectCodecV2.LABEL_SPEC,
            context.binding.label_spec_refs,
            LabelSpecV2,
        )
        structured = StructuredPredicateCompiler().compile(
            label_spec=label_spec,
            fact_set=trace_output.structured_fact_set,
            audit=context.manifest.audit,
        )
        merged = LabelDecisionMerger().merge(
            LabelDecisionMergeRequest(
                label_spec=label_spec,
                structured_result=structured,
                audit=context.manifest.audit,
            )
        )
        write = context.stage_store.put(
            R6CanaryObjectCodecV2.LABEL_OUTPUT,
            R6CanaryLabelOutput(
                item_id=_item_id(context),
                structured_result=structured,
                merge_result=merged,
            ),
        )
        return CanaryStageExecution(
            output_refs=(write.object_ref,),
            item_terminal_status=(
                None if merged.label_decision.decision.value == "MATCH" else ItemStatus.REJECTED
            ),
        )

    def _task_authoring(
        self,
        context: CanaryHandlerContext,
    ) -> CanaryStageExecution:
        fixture = context.seed(
            R6CanaryObjectCodecV2.CANARY_ATTACHMENT_FIXTURE,
            context.binding.task_fixture_refs,
            R6CanaryTaskFixture,
        )
        contract_set = R4TaskContractSetCompiler().compile(
            task_draft=fixture.task_draft,
            task_prompt_safety_gate=(fixture.task_prompt_safety_gate),
            rubric_set=fixture.rubric_set,
            evaluator_spec=fixture.evaluator_spec,
            reference_policy=fixture.reference_policy,
            tool_policy=fixture.tool_policy,
            contestant_tool_policy=(fixture.contestant_tool_policy),
            producer_storage_authorization=(fixture.producer_storage_authorization),
            producer_task_view=fixture.producer_task_view,
            audit=context.manifest.audit,
        )
        write = context.stage_store.put(
            R6CanaryObjectCodecV2.TASK_AUTHORING_OUTPUT,
            R6CanaryTaskOutput(
                item_id=_item_id(context),
                task_contract_set=contract_set,
                producer_task_view=fixture.producer_task_view,
                leakage_reference_set=(fixture.leakage_reference_set),
                task_draft=fixture.task_draft,
                producer_storage_authorization=(fixture.producer_storage_authorization),
                producer_view_result=fixture.producer_view_result,
                producer_evidence_bundle=(fixture.producer_evidence_bundle),
                text_provider_content=fixture.text_provider_content,
                attachment_relative_path=(fixture.attachment_relative_path),
            ),
        )
        return CanaryStageExecution(output_refs=(write.object_ref,))

    async def _attachment(
        self,
        context: CanaryHandlerContext,
    ) -> CanaryStageExecution:
        task = context.dependency(
            R6CanaryObjectCodecV2.TASK_AUTHORING_OUTPUT,
            R6CanaryTaskOutput,
        )
        if task.producer_task_view.attachment_requirements:
            text_plan = await _prepare_text_attachment(
                context,
                task,
            )
            preparation = R6CanaryAttachmentPreparationOutput(
                job_id=context.work_unit.job_id,
                item_id=_item_id(context),
                execution_result=text_plan.execution_result,
            )
            preparation_write = context.stage_store.put(
                R6CanaryObjectCodecV2.ATTACHMENT_PREPARATION_OUTPUT,
                preparation,
            )
            try:
                fanout = context.job_store.get_artifact_group_fanout(context.work_unit.resolved_work_unit_id)
                ArtifactGroupFanoutCompiler().validate_current(
                    fanout,
                    graph=context.graph,
                    parent_attachment_work_unit=context.work_unit,
                    artifact_execution_planning_result=(preparation.execution_result),
                )
            except RecordNotFoundError:
                fanout = ArtifactGroupFanoutCompiler().compile(
                    graph=context.graph,
                    parent_attachment_work_unit=context.work_unit,
                    artifact_execution_planning_result=(preparation.execution_result),
                    audit=context.manifest.audit,
                )
                context.job_store.create_artifact_group_fanout(
                    fanout,
                    idempotency_key=(
                        f"r6-canary-artifact-fanout:{context.work_unit.resolved_work_unit_sha256}"
                    ),
                )
            group_outputs = _artifact_group_outputs(
                context.stage_store,
                job_id=context.work_unit.job_id,
                item_id=_item_id(context),
            )
            if not group_outputs:
                return CanaryStageExecution(
                    output_refs=(preparation_write.object_ref,),
                    deferred=True,
                )
            if len(group_outputs) != 1:
                raise CanaryProfileError("text canary requires one artifact group output")
            group_output = group_outputs[0]
            expected_group_ref = resolved_work_unit_v2_ref(fanout.group_work_units[0])
            if group_output.work_unit_ref != expected_group_ref:
                raise CanaryProfileError("artifact group output does not bind the fanout")
            text_result = await _finalize_text_attachment(
                task=task,
                text_plan=text_plan,
                batch=group_output.execution_batch,
                audit=context.manifest.audit,
            )
            reconstruction = text_result.reconstruction
            source_validation = text_result.source_validation
        else:
            request, routing, execution = _not_required_sources(
                task.producer_task_view,
                context.manifest.audit,
            )
            reconstruction = ArtifactResultCompiler().compile(
                routing_request=request,
                routing_result=routing,
                execution_result=execution,
                execution_batch=None,
                audit=context.manifest.audit,
            )
            source_validation = await _run_not_required_validation(
                reconstruction,
                task,
                context.manifest.audit,
            )
        revision, validation = CandidateRevisionCompiler().compile_initial(
            reconstruction_result=reconstruction,
            deterministic_validation=source_validation,
            audit=context.manifest.audit,
        )
        write = context.stage_store.put(
            R6CanaryObjectCodecV2.ATTACHMENT_OUTPUT,
            R6CanaryAttachmentOutput(
                item_id=_item_id(context),
                reconstruction_result=reconstruction,
                source_deterministic_validation=source_validation,
                candidate_revision=revision,
                deterministic_validation=validation,
            ),
        )
        return CanaryStageExecution(
            output_refs=(write.object_ref,),
            resource_usage=NonModelResourceVectorV2.zero(),
        )

    async def _artifact_group(
        self,
        context: CanaryHandlerContext,
    ) -> CanaryStageExecution:
        if (
            context.artifact_group_fanout is None
            or context.resource_control is None
            or context.resource_reservation is None
        ):
            raise CanaryProfileError("artifact group execution is missing fanout or resource state")
        task = _item_object(
            context,
            R6CanaryObjectCodecV2.TASK_AUTHORING_OUTPUT,
            R6CanaryTaskOutput,
        )
        text_plan = await _prepare_text_attachment(context, task)
        preparation = _attachment_preparation(
            context.stage_store,
            job_id=context.work_unit.job_id,
            item_id=_item_id(context),
        )
        if preparation.execution_result != text_plan.execution_result:
            raise CanaryProfileError("artifact group preparation differs from durable plan")
        plan = preparation.execution_result.execution_plan
        if (
            plan is None
            or len(plan.groups) != 1
            or context.work_unit.artifact_execution_group_ref
            != context.artifact_group_fanout.group_work_units[0].artifact_execution_group_ref
        ):
            raise CanaryProfileError("text canary group does not bind its one-group plan")
        prior = _artifact_group_outputs(
            context.stage_store,
            job_id=context.work_unit.job_id,
            item_id=_item_id(context),
        )
        if prior:
            if len(prior) != 1:
                raise CanaryProfileError("text canary has ambiguous artifact group journals")
            output = prior[0]
            if output.work_unit_ref != resolved_work_unit_v2_ref(context.work_unit):
                raise CanaryProfileError("artifact group journal belongs to another work unit")
            ArtifactGroupExecutor().validate_current(
                plan,
                output.execution_batch,
            )
            return _group_stage_execution(output)
        resource_facade = _ResourceExecutionFacade(
            delegate=text_plan.physical.execution,
            resources=context.resource_control,
            reservation=context.resource_reservation,
            operation_store=context.stage_store,
            job_id=context.work_unit.job_id,
            item_id=_item_id(context),
            work_unit_ref=resolved_work_unit_v2_ref(context.work_unit),
        )
        batch = await ArtifactGroupExecutor().run(
            plan,
            facade=resource_facade,
            audit=context.manifest.audit,
        )
        if batch.succeeded_artifact_ids != plan.routed_artifact_ids:
            raise CanaryProfileError("text canary artifact group did not succeed")
        output = R6CanaryArtifactGroupOutput(
            job_id=context.work_unit.job_id,
            item_id=_item_id(context),
            work_unit_ref=resolved_work_unit_v2_ref(context.work_unit),
            execution_batch=batch,
            resource_usage=resource_facade.observed,
        )
        context.stage_store.put(
            R6CanaryObjectCodecV2.ARTIFACT_GROUP_OUTPUT,
            output,
        )
        return _group_stage_execution(output)

    async def _review_round(
        self,
        context: CanaryHandlerContext,
        round_: SemanticReviewRoundV2,
    ) -> CanaryStageExecution:
        attachment = _item_object(
            context,
            R6CanaryObjectCodecV2.ATTACHMENT_OUTPUT,
            R6CanaryAttachmentOutput,
        )
        fixture = context.seed(
            R6CanaryObjectCodecV2.CANARY_REVIEW_FIXTURE,
            context.binding.attachment_fixture_refs,
            R6CanaryReviewFixture,
        )
        prior = _prior_round(context)
        view = SemanticReviewContextCompiler().compile(
            round_=round_,
            policy=fixture.review_policy,
            candidate_revision=attachment.candidate_revision,
            deterministic_validation=(attachment.deterministic_validation),
            sources=fixture.context_sources,
            prior_round_result=prior,
            prior_finding_refs=(),
            prior_resolution_refs=(),
            prior_repair_plan_refs=(),
            prior_repair_result_refs=(),
            audit=context.manifest.audit,
        )
        execution = await SemanticReviewRoundExecutor().execute(
            running_stage=_running_stage(context),
            round_=round_,
            role=fixture.review_policy.roles_in_order[fixture.review_policy.rounds_in_order.index(round_)],
            candidate_revision=attachment.candidate_revision,
            deterministic_validation=(attachment.deterministic_validation),
            context_view=view,
            predecessor=prior,
            prior_factory_findings=(),
            prior_view_finding_refs=(),
            prior_facade_finding_refs=(),
            pending_resolutions=(),
            prior_facade_resolution_refs=(),
            prior_view_repair_plan_refs=(),
            prior_view_repair_result_refs=(),
            prior_facade_repair_result_refs=(),
            review_policy=fixture.review_policy,
            review_facade=_AcceptingSemanticReviewFacade(),
            audit=context.manifest.audit,
        )
        write = context.stage_store.put(
            R6CanaryObjectCodecV2.SEMANTIC_REVIEW_ROUND_OUTPUT,
            execution.round_result,
        )
        return CanaryStageExecution(output_refs=(write.object_ref,))

    def _item_quality(
        self,
        context: CanaryHandlerContext,
    ) -> CanaryStageExecution:
        task = _item_object(
            context,
            R6CanaryObjectCodecV2.TASK_AUTHORING_OUTPUT,
            R6CanaryTaskOutput,
        )
        attachment = _item_object(
            context,
            R6CanaryObjectCodecV2.ATTACHMENT_OUTPUT,
            R6CanaryAttachmentOutput,
        )
        fixture = context.seed(
            R6CanaryObjectCodecV2.CANARY_REVIEW_FIXTURE,
            context.binding.attachment_fixture_refs,
            R6CanaryReviewFixture,
        )
        rounds = _all_rounds(context)
        stage_result_refs = tuple(
            _stage_result_ref_for_round(
                context.job_store,
                context.work_unit.item_id,
                item.stage_run_ref.object_id,
            )
            for item in rounds
        )
        workflow = SemanticReviewWorkflowCompiler().compile(
            initial_revision=attachment.candidate_revision,
            initial_validation=attachment.deterministic_validation,
            review_policy=fixture.review_policy,
            current_revision=attachment.candidate_revision,
            validation_refs=(_revision_validation_ref(attachment.deterministic_validation),),
            repair_plan_refs=(),
            repair_result_refs=(),
            round_results=rounds,
            stage_result_refs=stage_result_refs,
            semantic_findings=(),
            semantic_resolutions=(),
            current_finding_refs=(attachment.deterministic_validation.finding_refs),
            stale_finding_refs=(),
            outcome=SemanticReviewWorkflowOutcomeV2.PASSED,
            audit=context.manifest.audit,
        )
        quality = ItemQualityCompiler().compile(
            task_contract_set=task.task_contract_set,
            review_policy=fixture.review_policy,
            semantic_workflow=workflow,
            candidate_revision=attachment.candidate_revision,
            deterministic_validation=(attachment.deterministic_validation),
            source_deterministic_validation=(attachment.source_deterministic_validation),
            job_store=context.job_store,
            job_id=context.work_unit.job_id,
            item_id=_item_id(context),
            audit=context.manifest.audit,
        )
        context.stage_store.put(
            R6CanaryObjectCodecV2.ITEM_QUALITY_OUTPUT,
            quality,
        )
        return CanaryStageExecution(output_refs=(item_quality_compilation_result_ref(quality),))


def canary_profile_codec_registry() -> StageObjectCodecRegistry:
    base = canary_stage_object_codec_registry()
    return base.extend(
        (
            _definition(
                R6CanaryObjectCodecV2.CANARY_ATTACHMENT_FIXTURE,
                R6CanaryTaskFixture,
                "r6-canary-attachment-fixture",
            ),
            _definition(
                R6CanaryObjectCodecV2.CANARY_REVIEW_FIXTURE,
                R6CanaryReviewFixture,
                "r6-canary-review-fixture",
            ),
            _definition(
                R6CanaryObjectCodecV2.TRACE_INDEX_OUTPUT,
                R6CanaryTraceIndexOutput,
                "r6-canary-trace-index-output",
            ),
            _definition(
                R6CanaryObjectCodecV2.SAFETY_OUTPUT,
                R6CanarySafetyOutput,
                "r6-canary-safety-output",
            ),
            _definition(
                R6CanaryObjectCodecV2.LABEL_OUTPUT,
                R6CanaryLabelOutput,
                "r6-canary-label-output",
            ),
            _definition(
                R6CanaryObjectCodecV2.TASK_AUTHORING_OUTPUT,
                R6CanaryTaskOutput,
                "r6-canary-task-authoring-output",
            ),
            _definition(
                R6CanaryObjectCodecV2.ATTACHMENT_PREPARATION_OUTPUT,
                R6CanaryAttachmentPreparationOutput,
                "r6-canary-attachment-preparation-output",
            ),
            _definition(
                R6CanaryObjectCodecV2.PROVIDER_OPERATION_OUTPUT,
                R6CanaryProviderOperationOutput,
                "r6-canary-provider-operation-output",
            ),
            _definition(
                R6CanaryObjectCodecV2.ARTIFACT_GROUP_OUTPUT,
                R6CanaryArtifactGroupOutput,
                "r6-canary-artifact-group-output",
            ),
            _definition(
                R6CanaryObjectCodecV2.ATTACHMENT_OUTPUT,
                R6CanaryAttachmentOutput,
                "r6-canary-attachment-output",
            ),
            StageObjectCodecDefinition(
                codec=(R6CanaryObjectCodecV2.SEMANTIC_REVIEW_ROUND_OUTPUT),
                model_type=SemanticReviewRoundResultV2,
                object_type="semantic-review-round-result",
                fixed_object_version="v2",
                object_id_field=("semantic_review_round_result_id"),
                object_sha256_field="round_result_sha256",
            ),
            StageObjectCodecDefinition(
                codec=R6CanaryObjectCodecV2.ITEM_QUALITY_OUTPUT,
                model_type=ItemQualityCompilationResultV2,
                object_type="item-quality-compilation-result",
                fixed_object_version="v2",
                object_id_field="item_quality_compilation_result_id",
                object_sha256_field="result_sha256",
            ),
        )
    )


def _definition(
    codec: R6CanaryObjectCodecV2,
    model_type: type[ContractModel],
    object_type: str,
) -> StageObjectCodecDefinition:
    return StageObjectCodecDefinition(
        codec=codec,
        model_type=model_type,
        object_type=object_type,
        fixed_object_version="v1",
    )


def _item_id(context: CanaryHandlerContext) -> str:
    if context.work_unit.item_id is None:
        raise CanaryProfileError("canary stage requires Item work")
    return context.work_unit.item_id


def _running_stage(
    context: CanaryHandlerContext,
) -> StageRunRecord:
    if context.stage_run is None:
        raise CanaryProfileError("ITEM-scoped canary work requires a StageRun")
    return context.stage_run


def _attachment_preparation(
    store: StageObjectStore,
    *,
    job_id: str,
    item_id: str,
) -> R6CanaryAttachmentPreparationOutput:
    matches = tuple(
        value
        for envelope in store.list_envelopes()
        if envelope.codec is R6CanaryObjectCodecV2.ATTACHMENT_PREPARATION_OUTPUT
        for value in (
            store.get_as(
                envelope.object_ref,
                codec=(R6CanaryObjectCodecV2.ATTACHMENT_PREPARATION_OUTPUT),
                model_type=R6CanaryAttachmentPreparationOutput,
            ),
        )
        if value.job_id == job_id and value.item_id == item_id
    )
    if len(matches) != 1:
        raise CanaryProfileError("expected one durable attachment preparation")
    return matches[0]


def _artifact_group_outputs(
    store: StageObjectStore,
    *,
    job_id: str,
    item_id: str,
) -> tuple[R6CanaryArtifactGroupOutput, ...]:
    return tuple(
        value
        for envelope in store.list_envelopes()
        if envelope.codec is R6CanaryObjectCodecV2.ARTIFACT_GROUP_OUTPUT
        for value in (
            store.get_as(
                envelope.object_ref,
                codec=R6CanaryObjectCodecV2.ARTIFACT_GROUP_OUTPUT,
                model_type=R6CanaryArtifactGroupOutput,
            ),
        )
        if value.job_id == job_id and value.item_id == item_id
    )


def _provider_operation_outputs(
    store: StageObjectStore,
    *,
    job_id: str,
    item_id: str,
    work_unit_ref: ObjectRef,
) -> tuple[R6CanaryProviderOperationOutput, ...]:
    return tuple(
        value
        for envelope in store.list_envelopes()
        if envelope.codec is R6CanaryObjectCodecV2.PROVIDER_OPERATION_OUTPUT
        for value in (
            store.get_as(
                envelope.object_ref,
                codec=(R6CanaryObjectCodecV2.PROVIDER_OPERATION_OUTPUT),
                model_type=R6CanaryProviderOperationOutput,
            ),
        )
        if (value.job_id == job_id and value.item_id == item_id and value.work_unit_ref == work_unit_ref)
    )


def _group_stage_execution(
    output: R6CanaryArtifactGroupOutput,
) -> CanaryStageExecution:
    batch = output.execution_batch
    return CanaryStageExecution(
        output_refs=(
            artifact_execution_batch_ref(batch),
            *(artifact_execution_receipt_ref(receipt) for receipt in batch.receipts),
        ),
        resource_usage=output.resource_usage,
    )


def _item_object[T: ContractModel](
    context: CanaryHandlerContext,
    codec: R6CanaryObjectCodecV2,
    model_type: type[T],
) -> T:
    object_type = context.stage_store.registry.definition(codec).object_type
    matches = tuple(ref for ref in context.item_result_refs if ref.object_type == object_type)
    if len(matches) != 1:
        raise CanaryProfileError(f"expected one current {codec.value} item output")
    return context.stage_store.get_as(
        matches[0],
        codec=codec,
        model_type=model_type,
    )


def _not_required_sources(
    producer_view: ProducerTaskViewV2,
    audit: ContractAudit,
) -> tuple[
    ArtifactRoutingRequest,
    ArtifactRoutingCompilationResult,
    ArtifactExecutionPlanningResult,
]:
    producer_ref = ObjectRef(
        object_type="producer-task-view",
        object_id=producer_view.producer_task_view_id,
        object_version="v2",
        object_sha256=producer_view.producer_task_view_sha256,
    )
    request = ArtifactRoutingRequest(
        request_id="artifact-routing-request://pending",
        attachment_planning_context_ref=_opaque_ref("attachment-planning-context"),
        producer_task_view_ref=producer_ref,
        artifact_evidence_matrix_ref=None,
        artifact_routing_policy_ref=_opaque_ref("artifact-routing-policy"),
        build_contracts=(),
        facade_requests=(),
        missing_contract_target_refs=(),
        blocked_mode_target_refs=(),
        policy_version="artifact-routing/r5-05-v1",
        request_sha256="0" * 64,
        audit=audit,
    )
    digest = artifact_routing_request_carried_sha256(request)
    request = request.model_copy(
        update={
            "request_id": (f"artifact-routing-request://sha256/{digest}"),
            "request_sha256": digest,
        }
    )
    routing = ArtifactRoutingCompilationResult(
        result_id="artifact-routing-compilation-result://pending",
        request_ref=artifact_routing_request_ref(request),
        execution_result_ref=_opaque_ref(
            "artifact-routing-execution-result",
            version="r5-05",
        ),
        outcome=ArtifactRoutingAggregateOutcomeV2.NOT_REQUIRED,
        routing_plan=None,
        policy_version="artifact-routing/r5-05-v1",
        result_sha256="0" * 64,
        audit=audit,
    )
    result_digest = artifact_routing_compilation_result_carried_sha256(routing)
    routing = routing.model_copy(
        update={
            "result_id": (f"artifact-routing-compilation-result://sha256/{result_digest}"),
            "result_sha256": result_digest,
        }
    )
    return (
        request,
        routing,
        ArtifactExecutionPlanningResult(
            outcome=ArtifactExecutionPlanningOutcome.NOT_REQUIRED,
            world_ledger_snapshot=None,
            execution_plan=None,
            audit=audit,
        ),
    )


async def _prepare_text_attachment(
    context: CanaryHandlerContext,
    task: R6CanaryTaskOutput,
) -> _TextAttachmentPlan:
    if (
        task.producer_view_result is None
        or task.producer_evidence_bundle is None
        or task.text_provider_content is None
        or task.attachment_relative_path is None
        or context.private_root is None
    ):
        raise CanaryProfileError("text attachment preparation is missing fixture state")
    requirement = task.producer_task_view.attachment_requirements[0]
    dependency = task.task_draft.attachment_dependencies[0]
    if requirement.dependency_id != dependency.dependency_id:
        raise CanaryProfileError("task and producer attachment dependency differ")
    planning_context = AttachmentPlanningBridge().compile(
        producer_task_view=task.producer_task_view,
        storage_authorization=(task.producer_storage_authorization),
        evidence_bundle=task.producer_evidence_bundle,
        audit=context.manifest.audit,
    )
    target = _text_target(
        planning_context_ref=ObjectRef(
            object_type="attachment-planning-context",
            object_id=planning_context.attachment_planning_context_id,
            object_version="v2",
            object_sha256=(planning_context.attachment_planning_context_sha256),
        ),
        dependency=dependency,
        relative_path=task.attachment_relative_path,
        audit=context.manifest.audit,
    )
    mode_result = ArtifactEvidenceModeCompiler().compile(
        attachment_planning_context=planning_context,
        producer_task_view=task.producer_task_view,
        storage_authorization=(task.producer_storage_authorization),
        evidence_bundle=task.producer_evidence_bundle,
        evidence_view_result=task.producer_view_result,
        timelines=(),
        targets=(target,),
        audit=context.manifest.audit,
    )
    matrix = mode_result.artifact_evidence_matrix
    if matrix is None:
        raise CanaryProfileError("text attachment mode did not produce a matrix")
    definition = _text_build_definition(dependency.dependency_id)
    policy = _text_routing_policy(context.manifest.audit)
    request = ArtifactRoutingRequestBuilder().build(
        attachment_planning_context=planning_context,
        producer_task_view=task.producer_task_view,
        storage_authorization=(task.producer_storage_authorization),
        producer_view_result=task.producer_view_result,
        producer_evidence_bundle=task.producer_evidence_bundle,
        artifact_evidence_matrix=matrix,
        artifact_targets=(target,),
        routing_policy=policy,
        contract_definitions=(definition,),
        audit=context.manifest.audit,
    )
    route_execution = await ArtifactRoutingRunner().run(
        request,
        facade=create_canary_text_routing_facade(clock=lambda: context.manifest.audit.created_at),
        audit=context.manifest.audit,
    )
    routing = ArtifactBuildSpecCompiler().compile(
        request=request,
        execution_result=route_execution,
        attachment_planning_context=planning_context,
        producer_task_view=task.producer_task_view,
        storage_authorization=(task.producer_storage_authorization),
        producer_view_result=task.producer_view_result,
        producer_evidence_bundle=task.producer_evidence_bundle,
        artifact_evidence_matrix=matrix,
        artifact_targets=(target,),
        routing_policy=policy,
        audit=context.manifest.audit,
    )
    if routing.routing_plan is None:
        raise CanaryProfileError("text attachment routing produced no plan")
    route_entry = routing.routing_plan.entries[0]
    build_spec = route_entry.build_spec
    contract = request.build_contracts[0]
    if build_spec is None or contract.provider_payload_ref is None:
        raise CanaryProfileError("text attachment route lacks provider material")
    physical = CanaryTextAttachmentFacades(
        fixture=CanaryTextExecutionFixture(
            artifact_id=target.artifact_id,
            dependency_id=dependency.dependency_id,
            relative_path=task.attachment_relative_path,
            content=task.text_provider_content,
            validators=build_spec.build_spec.validator_ids,
            producer_task_view_ref=_facade_ref(build_spec.build_spec.producer_task_view_ref),
            build_spec_ref=_facade_ref(artifact_build_spec_v2_ref(build_spec)),
            content_contract_ref=_facade_ref(build_spec.build_spec.content_contract_ref),
            render_contract_ref=_facade_ref(build_spec.build_spec.render_contract_ref),
            provider_payload_ref=_facade_ref(contract.provider_payload_ref),
            authorized_evidence_ref_ids=tuple(
                item.evidence_ref_id for item in build_spec.build_spec.authorized_evidence_refs
            ),
        ),
        staging_root=context.private_root / "provider-staging",
        clock=lambda: context.manifest.audit.created_at,
    )
    preparation = ArtifactExecutionPreparationBuilder().build(
        routing_request=request,
        routing_result=routing,
        definitions=(
            ArtifactConsistencyDefinition(
                artifact_id=target.artifact_id,
                dependency_artifact_ids=(),
                locked_fact_ids=(),
            ),
        ),
        world_ledger_ref=physical.world_ledger_ref,
        audit=context.manifest.audit,
    )
    snapshot = await WorldLedgerSnapshotRunner().run(
        preparation,
        facade=physical.execution,
    )
    execution_result = ArtifactExecutionPlanCompiler().compile(
        routing_request=request,
        routing_result=routing,
        preparation=preparation,
        world_ledger_snapshot=snapshot,
        audit=context.manifest.audit,
    )
    plan = execution_result.execution_plan
    if plan is None:
        raise CanaryProfileError("text attachment execution produced no plan")
    return _TextAttachmentPlan(
        routing_request=request,
        routing_result=routing,
        execution_result=execution_result,
        physical=physical,
    )


async def _finalize_text_attachment(
    *,
    task: R6CanaryTaskOutput,
    text_plan: _TextAttachmentPlan,
    batch: ArtifactExecutionBatchV2,
    audit: ContractAudit,
) -> _TextAttachmentResult:
    plan = text_plan.execution_result.execution_plan
    if plan is None:
        raise CanaryProfileError("text attachment finalization has no execution plan")
    ArtifactGroupExecutor().validate_current(plan, batch)
    reconstruction = ArtifactResultCompiler().compile(
        routing_request=text_plan.routing_request,
        routing_result=text_plan.routing_result,
        execution_result=text_plan.execution_result,
        execution_batch=batch,
        audit=audit,
    )
    requests = tuple(
        receipt.facade_request for receipt in batch.receipts if receipt.facade_request is not None
    )
    source_validation = await DeterministicValidationCompiler().compile(
        reconstruction_result=reconstruction,
        producer_task_view=task.producer_task_view,
        leakage_reference_set=task.leakage_reference_set,
        configured_pii_rules=(),
        facade=text_plan.physical.validation_facade(requests),
        audit=audit,
    )
    return _TextAttachmentResult(
        reconstruction=reconstruction,
        source_validation=source_validation,
    )


class _CanaryExecutionFacade(
    AttachmentExecutionFacade,
    ResourceBoundAttachmentExecutionFacade,
    Protocol,
):
    def recover_existing_output(
        self,
        request: AttachmentExecutionRequestV2,
        grant: AttachmentResourceGrantV2,
    ) -> ResourceBoundAttachmentExecutionResultV2 | None: ...


class _ResourceExecutionFacade(AttachmentExecutionFacade):
    def __init__(
        self,
        *,
        delegate: _CanaryExecutionFacade,
        resources: ResourceControlService,
        reservation: WorkResourceReservationV2,
        operation_store: StageObjectStore,
        job_id: str,
        item_id: str,
        work_unit_ref: ObjectRef,
    ) -> None:
        self._delegate = delegate
        self._resources = resources
        self._reservation = reservation
        self._operation_store = operation_store
        self._job_id = job_id
        self._item_id = item_id
        self._work_unit_ref = work_unit_ref
        self.observed = NonModelResourceVectorV2.zero()

    async def snapshot_world(
        self,
        request: WorldLedgerSnapshotRequestV2,
    ) -> WorldLedgerSnapshotV2:
        return await self._delegate.snapshot_world(request)

    async def execute(
        self,
        request: AttachmentExecutionRequestV2,
    ) -> AttachmentExecutionResultV2:
        grant = self._resources.to_facade_grant(
            self._reservation,
            operation_ref=attachment_execution_request_ref(request),
        )
        request_ref = _object_ref(attachment_execution_request_ref(request))
        prior = _provider_operation_outputs(
            self._operation_store,
            job_id=self._job_id,
            item_id=self._item_id,
            work_unit_ref=self._work_unit_ref,
        )
        if len(prior) > 1:
            raise CanaryProfileError("provider operation journal is ambiguous")
        if prior:
            operation = prior[0]
            if (
                operation.execution_request_ref != request_ref
                or operation.resource_result.resource_grant_ref != attachment_resource_grant_ref(grant)
            ):
                raise CanaryProfileError("provider operation journal is stale")
            result = operation.resource_result
        else:
            recovered = self._delegate.recover_existing_output(
                request,
                grant,
            )
            if recovered is None:
                result = await self._delegate.execute_with_resources(
                    request,
                    grant,
                )
            else:
                result = recovered
            self._operation_store.put(
                R6CanaryObjectCodecV2.PROVIDER_OPERATION_OUTPUT,
                R6CanaryProviderOperationOutput(
                    job_id=self._job_id,
                    item_id=self._item_id,
                    work_unit_ref=self._work_unit_ref,
                    execution_request_ref=request_ref,
                    resource_result=result,
                ),
            )
        usage = result.usage
        self.observed = self.observed.add(
            NonModelResourceVectorV2(
                processes=usage.process_starts,
                renderers=usage.renderer_operations,
                network_requests=usage.network_requests,
                storage_bytes=usage.retained_storage_bytes,
            )
        )
        return result.execution_result


def _text_target(
    *,
    planning_context_ref: ObjectRef,
    dependency: AttachmentDependency,
    relative_path: str,
    audit: ContractAudit,
) -> ArtifactEvidenceTargetV2:
    value = ArtifactEvidenceTargetV2(
        artifact_evidence_target_id=("artifact-evidence-target://pending"),
        attachment_planning_context_ref=planning_context_ref,
        attachment_dependency_id=dependency.dependency_id,
        artifact_id="artifact://r6-canary/text-input",
        logical_path=relative_path,
        media_type="text/plain",
        criticality=dependency.criticality,
        requirement_evidence=dependency.evidence,
        candidate_source_refs=(),
        policy_version=ARTIFACT_EVIDENCE_MODE_POLICY_VERSION,
        artifact_evidence_target_sha256="0" * 64,
        audit=audit,
    )
    digest = artifact_evidence_target_carried_sha256(value)
    return value.model_copy(
        update={
            "artifact_evidence_target_id": (f"artifact-evidence-target://sha256/{digest}"),
            "artifact_evidence_target_sha256": digest,
        }
    )


def _text_build_definition(
    dependency_id: str,
) -> ArtifactBuildContractDefinition:
    return ArtifactBuildContractDefinition(
        definition_id="artifact-build-contract-definition://r6-canary/text",
        attachment_dependency_id=dependency_id,
        asset_type="txt",
        content_contract_ref=_opaque_ref("artifact-content-contract"),
        render_contract_ref=_opaque_ref("artifact-render-contract"),
        provider_payload_ref=_opaque_ref("attachment-provider-payload"),
        source_evidence_set_ref=None,
        required_provider_capability_ids=("attachment-provider/generate/txt/v1",),
        required_runtime_tools=("write",),
        runtime_role="attachment-writer",
        runtime_resume_required=True,
        validator_ids=(
            "secret-validator",
            "text-validator",
        ),
    )


def _text_routing_policy(
    audit: ContractAudit,
) -> ArtifactRoutingPolicyV2:
    model_refs = (
        _opaque_ref("model-profile", suffix="sdk"),
        _opaque_ref("model-profile", suffix="cli"),
        _opaque_ref("model-profile", suffix="pi"),
    )
    policy_refs = (
        _opaque_ref("provider-capability-policy"),
        _opaque_ref("runtime-capability-policy"),
        _opaque_ref("validator-policy"),
    )
    value = ArtifactRoutingPolicyV2(
        artifact_routing_policy_id=("artifact-routing-policy://pending"),
        deterministic_provider_first=True,
        approved_provider_ids=("text",),
        runtime_order=(
            "claude_agent_sdk",
            "claude_code_cli",
            "pi_rpc",
        ),
        runtime_model_profile_refs=model_refs,
        provider_capability_policy_ref=policy_refs[0],
        runtime_capability_policy_ref=policy_refs[1],
        validator_policy_ref=policy_refs[2],
        silent_degradation_allowed=False,
        policy_version="artifact-routing/r5-05-v1",
        artifact_routing_policy_sha256="0" * 64,
        audit=audit.model_copy(
            update={
                "input_refs": tuple(
                    sorted(
                        (*model_refs, *policy_refs),
                        key=_ref_key,
                    )
                )
            }
        ),
    )
    digest = artifact_routing_policy_carried_sha256(value)
    return value.model_copy(
        update={
            "artifact_routing_policy_id": (f"artifact-routing-policy://sha256/{digest}"),
            "artifact_routing_policy_sha256": digest,
        }
    )


class _NoCallValidationFacade:
    async def validate(
        self,
        request: AttachmentValidationRequestV2,
    ) -> AttachmentValidationResultV2:
        del request
        raise CanaryProfileError("NOT_REQUIRED validation invoked a facade")


async def _run_not_required_validation(
    reconstruction: AttachmentReconstructionResultV2,
    task: R6CanaryTaskOutput,
    audit: ContractAudit,
) -> DeterministicItemValidationResultV2:
    return await DeterministicValidationCompiler().compile(
        reconstruction_result=reconstruction,
        producer_task_view=task.producer_task_view,
        leakage_reference_set=task.leakage_reference_set,
        configured_pii_rules=(),
        facade=_NoCallValidationFacade(),
        audit=audit,
    )


class _AcceptingSemanticReviewFacade:
    async def review(
        self,
        request: AttachmentSemanticReviewRequestV2,
    ) -> AttachmentSemanticReviewResultV2:
        request_ref = attachment_semantic_review_request_ref(request)
        included = (
            request.candidate_revision_ref,
            request.deterministic_validation_result_ref,
            request.context_view_ref,
            *request.current_artifact_version_refs,
            *request.current_output_refs,
            *request.prior_finding_refs,
            *request.prior_resolution_refs,
            *request.prior_repair_plan_refs,
            *request.prior_repair_result_refs,
        )
        attestation = SemanticCleanContextAttestationV2.create(
            semantic_review_request_ref=request_ref,
            stage_run_ref=request.stage_run_ref,
            round=request.round,
            reviewer_role=request.reviewer_role,
            context_view_ref=request.context_view_ref,
            included_ref_inventory=included,
        )
        result = AttachmentSemanticReviewResultV2(
            semantic_review_result_id=("attachment-semantic-review-result://pending"),
            semantic_review_request_ref=request_ref,
            candidate_revision_ref=request.candidate_revision_ref,
            deterministic_validation_result_ref=(request.deterministic_validation_result_ref),
            round=request.round,
            reviewer_role=request.reviewer_role,
            outcome=AttachmentSemanticReviewOutcomeV2.ACCEPTED,
            clean_context_attestation=attestation,
            findings=(),
            resolutions=(),
            failure_code=None,
            semantic_review_result_sha256="0" * 64,
        )
        digest = attachment_semantic_review_result_carried_sha256(result)
        return result.model_copy(
            update={
                "semantic_review_result_id": (f"attachment-semantic-review-result://sha256/{digest}"),
                "semantic_review_result_sha256": digest,
            }
        )

    async def repair(
        self,
        request: AttachmentRepairRequestV2,
    ) -> AttachmentRepairResultV2:
        del request
        raise CanaryProfileError("accepting canary review cannot repair")


def _prior_round(
    context: CanaryHandlerContext,
) -> SemanticReviewRoundResultV2 | None:
    expected = context.stage_store.registry.definition(
        R6CanaryObjectCodecV2.SEMANTIC_REVIEW_ROUND_OUTPUT
    ).object_type
    refs = tuple(ref for ref in context.dependency_output_refs if ref.object_type == expected)
    if not refs:
        return None
    if len(refs) != 1:
        raise CanaryProfileError("semantic review round has ambiguous predecessor")
    return context.stage_store.get_as(
        refs[0],
        codec=(R6CanaryObjectCodecV2.SEMANTIC_REVIEW_ROUND_OUTPUT),
        model_type=SemanticReviewRoundResultV2,
    )


def _all_rounds(
    context: CanaryHandlerContext,
) -> tuple[SemanticReviewRoundResultV2, ...]:
    object_type = context.stage_store.registry.definition(
        R6CanaryObjectCodecV2.SEMANTIC_REVIEW_ROUND_OUTPUT
    ).object_type
    refs = tuple(ref for ref in context.item_result_refs if ref.object_type == object_type)
    values = tuple(
        context.stage_store.get_as(
            ref,
            codec=(R6CanaryObjectCodecV2.SEMANTIC_REVIEW_ROUND_OUTPUT),
            model_type=SemanticReviewRoundResultV2,
        )
        for ref in refs
    )
    by_round = {value.round: value for value in values}
    if set(by_round) != set(SemanticReviewRoundV2):
        raise CanaryProfileError("item quality requires all semantic review rounds")
    return tuple(by_round[round_] for round_ in SemanticReviewRoundV2)


def _stage_result_ref_for_round(
    store: JobStore,
    item_id: str | None,
    stage_run_id: str,
) -> ObjectRef:
    if item_id is None:
        raise CanaryProfileError("semantic review has no Item")
    result = store.get_stage_result_for_run(stage_run_id)
    if result is None or result.status is not StageRunStatus.SUCCEEDED:
        raise CanaryProfileError("semantic review StageResult is missing")
    from eval_factory.orchestration.models import (
        stage_result_record_ref,
    )

    return stage_result_record_ref(result)


def _revision_validation_ref(
    value: RevisionDeterministicValidationV2,
) -> ObjectRef:
    from eval_factory.contracts.review_v2 import (
        revision_deterministic_validation_ref,
    )

    return revision_deterministic_validation_ref(value)


def _opaque_ref(
    object_type: str,
    *,
    version: str = "v2",
    suffix: str = "not-required",
) -> ObjectRef:
    digest = "a" * 64
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-canary/{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _ref_key(
    ref: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _facade_ref(ref: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _object_ref(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )
