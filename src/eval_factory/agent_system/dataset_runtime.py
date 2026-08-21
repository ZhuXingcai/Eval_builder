from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from eval_factory.agent_system.attachment_item_runtime import (
    FactoryAttachmentItemRuntime,
    FactoryAttachmentItemState,
    FactoryAttachmentItemView,
)
from eval_factory.agent_system.batch_quality_material import (
    FactoryBatchQualityResult,
)
from eval_factory.agent_system.batch_quality_runtime import (
    FactoryBatchQualityContext,
    FactoryBatchQualityRuntime,
)
from eval_factory.agent_system.core_material import (
    FactoryCoreMaterialStore,
)
from eval_factory.agent_system.core_vertical import (
    CoreVerticalExecution,
    CoreVerticalRunner,
)
from eval_factory.agent_system.delivery_runtime import (
    FactoryDeliveryWaitingView,
)
from eval_factory.agent_system.item_run_materializer import (
    FactoryItemRunMaterializer,
)
from eval_factory.agent_system.item_specialist_runtime import (
    FactoryItemSpecialistContext,
    FactoryItemSpecialistInput,
    FactoryItemSpecialistRuntime,
    FactoryItemSpecialistState,
    FactoryItemSpecialistView,
)
from eval_factory.agent_system.job_store_bridge import (
    FactoryJobStoreAuthority,
    FactoryJobStoreBridge,
)
from eval_factory.agent_system.job_store_witness_bridge import (
    FactoryItemStageWitnessSource,
    FactoryJobStoreWitnessBridge,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewService,
)
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.release_runtime import (
    FactoryDatasetReleaseContext,
)
from eval_factory.agent_system.release_source_builder import (
    FactoryReleaseSourceItem,
)
from eval_factory.agent_system.specialists import (
    GatewayRequirementPlannerAgent,
)
from eval_factory.agent_system.store import (
    FactoryControlIntegrityError,
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.agent_system.task_authoring_bridge import (
    FactoryTaskAuthoringBridge,
    FactoryTaskAuthoringBridgeResult,
)
from eval_factory.agent_system.task_authoring_commit import (
    FactoryTaskAuthoringCommitter,
)
from eval_factory.agent_system.task_authoring_material import (
    FactoryTaskAuthoringMaterialStore,
)
from eval_factory.agent_system.trace_candidate import (
    TraceCandidatePreparation,
)
from eval_factory.batch_quality.reports import BatchQualityItemSource
from eval_factory.contracts.agent_system_v2 import (
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    FactoryRunStatusV2,
    FactoryRunV2,
    PlanKindV2,
    PlanReviewStateV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.batch_quality_v2 import (
    batch_quality_report_v2_ref,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetNextActionV2,
    FactoryDatasetPlanningAuthorityV2,
    FactoryDatasetRunRequestV2,
    FactoryDatasetRunViewV2,
    FactoryItemRunBindingV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.labeling_v2 import label_decision_ref
from eval_factory.contracts.quality_v2 import (
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.task_v2 import (
    r4_task_contract_set_ref,
    task_prompt_safety_gate_ref,
)


class FactoryDatasetRuntimeError(RuntimeError):
    pass


class FactoryItemSpecialistContextFactory(Protocol):
    def build(
        self,
        *,
        binding: FactoryItemRunBindingV2,
        job_authority: FactoryJobStoreAuthority,
    ) -> FactoryItemSpecialistContext: ...


class FactoryBatchQualityContextFactory(Protocol):
    def build(
        self,
        *,
        job_authority: FactoryJobStoreAuthority,
    ) -> FactoryBatchQualityContext: ...


class FactoryDatasetReleaseContextFactory(Protocol):
    def build(
        self,
        *,
        job_authority: FactoryJobStoreAuthority,
        item_ids: tuple[str, ...],
    ) -> FactoryDatasetReleaseContext: ...


@dataclass(frozen=True, slots=True)
class FactoryDatasetCoreInput:
    manifest_path: Path
    raw_root: Path
    expected_manifest_sha256: str

    def __post_init__(self) -> None:
        manifest = self.manifest_path.expanduser().resolve()
        raw_root = self.raw_root.expanduser().resolve()
        if (
            self.manifest_path.is_symlink()
            or not manifest.is_file()
            or self.raw_root.is_symlink()
            or not raw_root.is_dir()
            or manifest.parent != raw_root
        ):
            raise ValueError("core input requires one safe manifest under raw root")
        object.__setattr__(self, "manifest_path", manifest)
        object.__setattr__(self, "raw_root", raw_root)


class FactoryDatasetRuntime:
    def __init__(
        self,
        *,
        store: FactoryControlStore,
        planner: GatewayRequirementPlannerAgent,
        compiler: DatasetBuildPlanCompiler,
        plan_reviews: PlanReviewService,
        requested_by: str,
        core_runner: CoreVerticalRunner | None = None,
        core_materials: FactoryCoreMaterialStore | None = None,
        task_authoring_bridge: (FactoryTaskAuthoringBridge | None) = None,
        task_authoring_materials: (FactoryTaskAuthoringMaterialStore | None) = None,
        attachment_runtime: FactoryAttachmentItemRuntime | None = None,
        item_specialist_runtime: FactoryItemSpecialistRuntime | None = None,
        item_specialist_contexts: (Mapping[str, FactoryItemSpecialistContext] | None) = None,
        item_specialist_context_factory: (FactoryItemSpecialistContextFactory | None) = None,
        batch_quality_runtime: FactoryBatchQualityRuntime | None = None,
        batch_quality_context: FactoryBatchQualityContext | None = None,
        batch_quality_context_factory: (FactoryBatchQualityContextFactory | None) = None,
        release_context: FactoryDatasetReleaseContext | None = None,
        release_context_factory: (FactoryDatasetReleaseContextFactory | None) = None,
        job_store_bridge: FactoryJobStoreBridge | None = None,
        job_store_witness_bridge: (FactoryJobStoreWitnessBridge | None) = None,
    ) -> None:
        self.store = store
        self.planner = planner
        self.compiler = compiler
        self.plan_reviews = plan_reviews
        self.requested_by = requested_by
        self.core_runner = core_runner
        self.core_materials = core_materials
        self.task_authoring_bridge = task_authoring_bridge
        self.task_authoring_materials = task_authoring_materials
        self.attachment_runtime = attachment_runtime
        self.item_specialist_runtime = item_specialist_runtime
        self.item_specialist_contexts = dict(item_specialist_contexts or {})
        self.item_specialist_context_factory = item_specialist_context_factory
        self.batch_quality_runtime = batch_quality_runtime
        self.batch_quality_context = batch_quality_context
        self.batch_quality_context_factory = batch_quality_context_factory
        self.release_context = release_context
        self.release_context_factory = release_context_factory
        self.job_store_bridge = job_store_bridge
        self.job_store_witness_bridge = job_store_witness_bridge
        if (core_runner is None) != (core_materials is None):
            raise ValueError("core runner and material store must appear together")
        if (task_authoring_bridge is None) != (task_authoring_materials is None):
            raise ValueError("task authoring bridge and material store must appear together")
        if item_specialist_runtime is not None and attachment_runtime is None:
            raise ValueError("item specialist runtime requires attachment runtime")
        if item_specialist_runtime is None and item_specialist_contexts is not None:
            raise ValueError("item specialist contexts require specialist runtime")
        if item_specialist_context_factory is not None and item_specialist_runtime is None:
            raise ValueError("item specialist context factory requires specialist runtime")
        if batch_quality_runtime is None and (
            batch_quality_context is not None or batch_quality_context_factory is not None
        ):
            raise ValueError("batch quality context requires its runtime")
        if batch_quality_runtime is not None and (
            (batch_quality_context is None) == (batch_quality_context_factory is None)
        ):
            raise ValueError("batch quality runtime requires one context source")
        if batch_quality_runtime is not None and item_specialist_runtime is None:
            raise ValueError("batch quality runtime requires item specialists")
        if (
            release_context is not None or release_context_factory is not None
        ) and batch_quality_runtime is None:
            raise ValueError("dataset release requires batch quality runtime")
        if release_context is not None and release_context_factory is not None:
            raise ValueError("dataset release requires one context source")
        if job_store_bridge is not None and task_authoring_bridge is None:
            raise ValueError("JobStore bridge requires task authoring")
        if item_specialist_context_factory is not None and job_store_bridge is None:
            raise ValueError("item specialist context factory requires JobStore bridge")
        if (
            batch_quality_context_factory is not None
            or release_context_factory is not None
            or job_store_witness_bridge is not None
        ) and job_store_bridge is None:
            raise ValueError("dynamic batch/release authority requires JobStore bridge")
        if (
            job_store_witness_bridge is not None
            and job_store_bridge is not None
            and job_store_witness_bridge.job_store.path != job_store_bridge.job_store.path
        ):
            raise ValueError("JobStore witness bridge must share the canonical JobStore")
        if release_context is not None and (
            release_context.source_builder.store is not store
            or release_context.runtime.candidates.store is not store
            or release_context.source_builder.job_store.path
            != release_context.runtime.candidates.persistence.store.path
        ):
            raise ValueError("dataset release context must share Factory and Job stores")

    def admit_request(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2:
        self._validate_inputs(
            request=request,
            policy=policy,
            requirement=requirement,
        )
        run = self._ensure_run(
            request=request,
            policy=policy,
            requirement=requirement,
        )
        stored_request = self.store.commit_dataset_request(
            request,
            idempotency_key=(f"{request.idempotency_key}:commit-request"),
        )
        if stored_request != request:
            raise FactoryDatasetRuntimeError(
                "stored dataset request differs from current input",
            )
        return self.current_view(
            request=request,
            audit=audit,
            run=run,
        )

    async def plan_requirement(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2:
        self.admit_request(
            request=request,
            policy=policy,
            requirement=requirement,
            audit=audit,
        )
        run = self.store.get_run(request.dataset_run_id)
        if run.status in {
            FactoryRunStatusV2.FAILED,
            FactoryRunStatusV2.CANCELLED,
            FactoryRunStatusV2.BLOCKED,
            FactoryRunStatusV2.WAITING_REVIEW,
        }:
            return self.current_view(
                request=request,
                audit=audit,
                run=run,
            )
        if run.current_plan_ref is None:
            await self._plan(
                request=request,
                run=run,
                policy=policy,
                requirement=requirement,
                audit=audit,
            )
        return self.current_view(
            request=request,
            audit=audit,
        )

    def review_global_plan(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2:
        run = self.store.get_run(request.dataset_run_id)
        if run.current_plan_ref is None:
            raise FactoryDatasetRuntimeError(
                "global plan must be committed before review",
            )
        plan, _compiled = self.store.get_plan(run.run_id)
        if self._has_resumed_global_plan(
            run_id=run.run_id,
            plan=plan,
        ):
            return self.current_view(
                request=request,
                audit=audit,
                run=self.store.get_run(run.run_id),
            )
        if run.status is FactoryRunStatusV2.WAITING_REVIEW:
            if run.pending_review_ref is None:
                raise FactoryControlIntegrityError(
                    "waiting dataset run has no review authority",
                )
            self.plan_reviews.show(
                run.pending_review_ref.object_id,
            )
            return self.current_view(
                request=request,
                audit=audit,
                run=run,
            )
        review = self.plan_reviews.open_global_plan(
            run_id=run.run_id,
            title="Dataset build plan",
            summary_lines=(
                f"Goals: {len(plan.goals)}",
                f"Stages: {len(plan.stage_order)}",
                f"Tasks: {len(plan.tasks)}",
            ),
            editable_paths=(
                "assumptions",
                "goals",
                "tasks",
                "unresolved_questions",
                "user_constraints",
            ),
            warning_codes=(("UNRESOLVED_QUESTIONS",) if plan.unresolved_questions else ()),
            requested_by=self.requested_by,
            idempotency_key=(f"{request.idempotency_key}:open-global-review:{plan.object_sha256}"),
            audit=audit,
        )
        return self._view(
            request=request,
            run=self.store.get_run(run.run_id),
            pending_review_refs=(review.request.to_ref(),),
            next_action=FactoryDatasetNextActionV2.REVIEW_PLAN,
            audit=audit,
        )

    def verify_compiled_global_plan(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2:
        run = self.store.get_run(request.dataset_run_id)
        plan, compiled = self.store.get_plan(run.run_id)
        if (
            run.current_plan_ref != plan.to_ref()
            or run.compiled_plan_ref != compiled.to_ref()
            or not self._has_resumed_global_plan(
                run_id=run.run_id,
                plan=plan,
            )
        ):
            raise FactoryDatasetRuntimeError(
                "compiled global plan lacks current resumed authority",
            )
        return self.current_view(
            request=request,
            audit=audit,
            run=run,
        )

    def current_view(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        audit: ContractAudit,
        run: FactoryRunV2 | None = None,
    ) -> FactoryDatasetRunViewV2:
        current = run or self.store.get_run(
            request.dataset_run_id,
        )
        pending = (current.pending_review_ref,) if current.pending_review_ref is not None else ()
        if current.status is FactoryRunStatusV2.WAITING_REVIEW:
            next_action = FactoryDatasetNextActionV2.REVIEW_PLAN
        elif current.status in {
            FactoryRunStatusV2.FAILED,
            FactoryRunStatusV2.CANCELLED,
            FactoryRunStatusV2.BLOCKED,
        }:
            next_action = FactoryDatasetNextActionV2.NONE
        else:
            next_action = FactoryDatasetNextActionV2.ADVANCE
        return self._view(
            request=request,
            run=current,
            pending_review_refs=pending,
            next_action=next_action,
            audit=audit,
        )

    async def advance(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        core_input: FactoryDatasetCoreInput | None = None,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2:
        self._validate_inputs(
            request=request,
            policy=policy,
            requirement=requirement,
        )
        run = self._ensure_run(
            request=request,
            policy=policy,
            requirement=requirement,
        )
        stored_request = self.store.commit_dataset_request(
            request,
            idempotency_key=(f"{request.idempotency_key}:commit-request"),
        )
        if stored_request != request:
            raise FactoryDatasetRuntimeError("stored dataset request differs from current input")
        if run.status in {
            FactoryRunStatusV2.FAILED,
            FactoryRunStatusV2.CANCELLED,
            FactoryRunStatusV2.BLOCKED,
        }:
            return self._view(
                request=request,
                run=self.store.get_run(run.run_id),
                pending_review_refs=(),
                next_action=FactoryDatasetNextActionV2.NONE,
                audit=audit,
            )
        if run.status is FactoryRunStatusV2.WAITING_REVIEW:
            if run.pending_review_ref is None:
                raise FactoryControlIntegrityError("waiting dataset run has no review authority")
            self.plan_reviews.show(
                run.pending_review_ref.object_id,
            )
            return self._view(
                request=request,
                run=run,
                pending_review_refs=(run.pending_review_ref,),
                next_action=FactoryDatasetNextActionV2.REVIEW_PLAN,
                audit=audit,
            )
        if run.current_plan_ref is None:
            run = await self._plan(
                request=request,
                run=run,
                policy=policy,
                requirement=requirement,
                audit=audit,
            )
        plan, _compiled = self.store.get_plan(run.run_id)
        if not self._has_resumed_global_plan(
            run_id=run.run_id,
            plan=plan,
        ):
            review = self.plan_reviews.open_global_plan(
                run_id=run.run_id,
                title="Dataset build plan",
                summary_lines=(
                    f"Goals: {len(plan.goals)}",
                    f"Stages: {len(plan.stage_order)}",
                    f"Tasks: {len(plan.tasks)}",
                ),
                editable_paths=(
                    "assumptions",
                    "goals",
                    "tasks",
                    "unresolved_questions",
                    "user_constraints",
                ),
                warning_codes=(("UNRESOLVED_QUESTIONS",) if plan.unresolved_questions else ()),
                requested_by=self.requested_by,
                idempotency_key=(f"{request.idempotency_key}:open-global-review:{plan.object_sha256}"),
                audit=audit,
            )
            return self._view(
                request=request,
                run=self.store.get_run(run.run_id),
                pending_review_refs=(review.request.to_ref(),),
                next_action=FactoryDatasetNextActionV2.REVIEW_PLAN,
                audit=audit,
            )
        if core_input is not None:
            if core_input.expected_manifest_sha256 != request.manifest_ref.object_sha256:
                raise ValueError("core input manifest hash differs from request")
            execution = await self._advance_core(
                request=request,
                policy=policy,
                requirement=requirement,
                core_input=core_input,
                audit=audit,
            )
            if self.task_authoring_bridge is not None:
                task_authoring = await self._advance_task_authoring(
                    request=request,
                    requirement=requirement,
                    execution=execution,
                    audit=audit,
                )
                job_authority = self._prepare_job_store(
                    request=request,
                    task_authoring=task_authoring,
                    audit=audit,
                )
                if job_authority is not None:
                    self._build_item_specialist_contexts(
                        authority=job_authority,
                        bindings=tuple(binding for binding, _result in task_authoring),
                    )
                    self._build_batch_release_contexts(
                        authority=job_authority,
                    )
                    self._validate_job_contexts(
                        authority=job_authority,
                        bindings=tuple(binding for binding, _result in task_authoring),
                    )
                if self.attachment_runtime is not None:
                    attachment_views = tuple(
                        [
                            await self.attachment_runtime.advance(
                                binding=binding,
                                task_authoring=result,
                                policy=policy,
                                audit=audit,
                            )
                            for binding, result in task_authoring
                        ]
                    )
                    pending_reviews = tuple(
                        value.review_ref
                        for value in attachment_views
                        if value.state is FactoryAttachmentItemState.WAITING_REVIEW
                    )
                    if pending_reviews:
                        return self._view(
                            request=request,
                            run=self.store.get_run(run.run_id),
                            pending_review_refs=pending_reviews,
                            next_action=(FactoryDatasetNextActionV2.REVIEW_PLAN),
                            status=(FactoryRunStatusV2.WAITING_REVIEW),
                            audit=audit,
                        )
                    if self.item_specialist_runtime is not None:
                        specialist_views = []
                        for (
                            binding,
                            task_authoring_result,
                        ), attachment_view in zip(
                            task_authoring,
                            attachment_views,
                            strict=True,
                        ):
                            context = self.item_specialist_contexts.get(binding.item_id)
                            producer_view = task_authoring_result.producer_task_view_result.producer_task_view
                            if context is None or producer_view is None:
                                raise FactoryDatasetRuntimeError("item specialist context is unavailable")
                            specialist_views.append(
                                await self.item_specialist_runtime.advance(
                                    binding=binding,
                                    source=FactoryItemSpecialistInput(
                                        producer_task_view=producer_view,
                                        leakage_reference_set=(task_authoring_result.leakage_reference_set),
                                        task_contract_set=(task_authoring_result.task_contract_set),
                                        task_draft=(task_authoring_result.task_draft),
                                    ),
                                    attachment=attachment_view,
                                    policy=policy,
                                    context=context,
                                    audit=audit,
                                )
                            )
                        pending_specialist_reviews = tuple(
                            review_ref for view in specialist_views for review_ref in view.pending_review_refs
                        )
                        if pending_specialist_reviews:
                            return self._view(
                                request=request,
                                run=self.store.get_run(run.run_id),
                                pending_review_refs=(pending_specialist_reviews),
                                next_action=(FactoryDatasetNextActionV2.REVIEW_PLAN),
                                status=(FactoryRunStatusV2.WAITING_REVIEW),
                                audit=audit,
                            )
                        if self.batch_quality_runtime is not None and self.batch_quality_context is not None:
                            if self.job_store_witness_bridge is not None and job_authority is not None:
                                witness_sources = tuple(
                                    _item_witness_source(
                                        binding=binding,
                                        task_authoring=task_authoring_result,
                                        attachment=attachment_view,
                                        specialist=specialist_view,
                                    )
                                    for (
                                        binding,
                                        task_authoring_result,
                                    ), attachment_view, specialist_view in zip(
                                        task_authoring,
                                        attachment_views,
                                        specialist_views,
                                        strict=True,
                                    )
                                    if specialist_view.state is not FactoryItemSpecialistState.BLOCKED_QUALITY
                                )
                                if len(witness_sources) != len(specialist_views):
                                    raise FactoryDatasetRuntimeError(
                                        "non-approvable item quality cannot enter Batch QA"
                                    )
                                self.job_store_witness_bridge.commit_items(
                                    graph=job_authority.graph,
                                    sources=witness_sources,
                                    audit=audit,
                                )
                            candidate_sources = []
                            blocked_binding_refs = []
                            for (
                                binding,
                                task_authoring_result,
                            ), specialist_view in zip(
                                task_authoring,
                                specialist_views,
                                strict=True,
                            ):
                                if specialist_view.state is not FactoryItemSpecialistState.BLOCKED_QUALITY:
                                    candidate_sources.append(
                                        _batch_quality_source(
                                            binding=binding,
                                            task_authoring=(task_authoring_result),
                                            specialist=(specialist_view),
                                        )
                                    )
                                else:
                                    blocked_binding_refs.append(binding.to_ref())
                            batch_view = await self.batch_quality_runtime.advance(
                                dataset_run_id=request.dataset_run_id,
                                core_vertical_result_ref=(execution.result.to_ref()),
                                sources=tuple(candidate_sources),
                                blocked_binding_refs=tuple(blocked_binding_refs),
                                context=self.batch_quality_context,
                                audit=audit,
                            )
                            if (
                                self.job_store_witness_bridge is not None
                                and job_authority is not None
                                and isinstance(
                                    batch_view.result,
                                    FactoryBatchQualityResult,
                                )
                            ):
                                self.job_store_witness_bridge.commit_batch(
                                    graph=job_authority.graph,
                                    batch_quality_ref=(batch_quality_report_v2_ref(batch_view.result.report)),
                                    audit=audit,
                                )
                            if batch_view.aggregate.candidate_binding_refs:
                                if not isinstance(
                                    batch_view.result,
                                    FactoryBatchQualityResult,
                                ):
                                    raise FactoryDatasetRuntimeError(
                                        "candidate aggregate lacks batch quality material"
                                    )
                                candidate_refs = set(batch_view.aggregate.candidate_binding_refs)
                                release_items = tuple(
                                    _release_source_item(
                                        binding=binding,
                                        task_authoring=task_authoring_result,
                                        attachment=attachment_view,
                                        specialist=specialist_view,
                                    )
                                    for (
                                        binding,
                                        task_authoring_result,
                                    ), attachment_view, specialist_view in zip(
                                        task_authoring,
                                        attachment_views,
                                        specialist_views,
                                        strict=True,
                                    )
                                    if binding.to_ref() in candidate_refs
                                )
                                if job_authority is not None:
                                    self._build_release_context(
                                        authority=job_authority,
                                        item_ids=tuple(value.binding.item_id for value in release_items),
                                    )
                                if self.release_context is not None:
                                    release = self.release_context.advance(
                                        dataset_run_id=request.dataset_run_id,
                                        aggregate=batch_view.aggregate,
                                        items=release_items,
                                        batch_quality=batch_view.result.report,
                                        factory_policy=policy,
                                        audit=audit,
                                    )
                                    if isinstance(
                                        release.delivery,
                                        FactoryDeliveryWaitingView,
                                    ):
                                        return self._view(
                                            request=request,
                                            run=self.store.get_run(run.run_id),
                                            pending_review_refs=(release.delivery.review_ref,),
                                            next_action=(FactoryDatasetNextActionV2.REVIEW_PLAN),
                                            status=(FactoryRunStatusV2.WAITING_REVIEW),
                                            audit=audit,
                                        )
        return self._view(
            request=request,
            run=self.store.get_run(run.run_id),
            pending_review_refs=(),
            next_action=FactoryDatasetNextActionV2.ADVANCE,
            audit=audit,
        )

    async def _advance_core(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        core_input: FactoryDatasetCoreInput,
        audit: ContractAudit,
    ) -> CoreVerticalExecution:
        if self.core_runner is None or self.core_materials is None:
            raise FactoryDatasetRuntimeError("core execution capability is unavailable")
        plan, compiled = self.store.get_plan(request.dataset_run_id)
        try:
            execution = self.core_materials.get(request.dataset_run_id)
        except FactoryControlNotFoundError:
            planning = self.store.get_planning_authority(request.dataset_run_id)
            if planning.plan_ref != plan.to_ref():
                raise FactoryDatasetRuntimeError("planning route does not bind current plan") from None
            execution = await self.core_runner.run(
                manifest_path=core_input.manifest_path,
                raw_root=core_input.raw_root,
                expected_manifest_sha256=(core_input.expected_manifest_sha256),
                requirement=requirement,
                planning_route_ref=planning.route_decision_ref,
                audit=audit,
            )
            execution = self.core_materials.commit(
                run_id=request.dataset_run_id,
                compiled_plan_ref=compiled.to_ref(),
                execution=execution,
                idempotency_key=(f"{request.idempotency_key}:core:{execution.result.object_sha256}"),
            )
        self._materialize_item_runs(
            request=request,
            policy=policy,
            requirement=requirement,
            execution=execution,
            audit=audit,
        )
        return execution

    def _materialize_item_runs(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        execution: CoreVerticalExecution,
        audit: ContractAudit,
    ) -> None:
        materializer = FactoryItemRunMaterializer(self.store)
        prompts_by_source = {
            prompt.source_spans[0].source_trace_id: prompt for prompt in execution.extracted_prompts
        }
        intents_by_prompt = {intent.extracted_prompt_ref: intent for intent in execution.inferred_intents}
        rewrites_by_prompt = {
            rewrite.extracted_prompt_ref: rewrite for rewrite in execution.rewrite_candidates
        }
        for decision in execution.decisions:
            if decision.disposition is not TraceCandidateDispositionV2.CANDIDATE:
                continue
            if self.core_runner is None or decision.cleaned_trace_ref is None:
                raise FactoryDatasetRuntimeError("candidate decision has no current cleaned trace")
            prompt = prompts_by_source.get(decision.source_trace_id)
            if prompt is None:
                raise FactoryDatasetRuntimeError("candidate decision has no extracted prompt")
            intent = intents_by_prompt.get(prompt.to_ref())
            rewrite = rewrites_by_prompt.get(prompt.to_ref())
            if intent is None or rewrite is None or rewrite.inferred_intent_ref != intent.to_ref():
                raise FactoryDatasetRuntimeError("candidate semantic material is incomplete")
            manifest = self.core_runner.load_manifest(
                decision.cleaned_trace_ref.object_id,
            )
            source_trace_ref = ObjectRef(
                object_type="trace-source",
                object_id=decision.source_ref.object_id,
                object_version=manifest.adapter_version,
                object_sha256=(decision.source_ref.object_sha256),
            )
            materializer.materialize(
                request=request,
                policy=policy,
                requirement=requirement,
                source_trace_ref=source_trace_ref,
                candidate=TraceCandidatePreparation(
                    decision=decision,
                    extracted_prompt=prompt,
                    inferred_intent=intent,
                    rewrite_candidate=rewrite,
                    route_refs=(),
                ),
                audit=audit,
            )

    async def _advance_task_authoring(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        requirement: EvaluationRequirementSpecV2,
        execution: CoreVerticalExecution,
        audit: ContractAudit,
    ) -> tuple[
        tuple[
            FactoryItemRunBindingV2,
            FactoryTaskAuthoringBridgeResult,
        ],
        ...,
    ]:
        if self.task_authoring_bridge is None or self.task_authoring_materials is None:
            raise FactoryDatasetRuntimeError("task authoring capability is unavailable")
        decisions = {value.to_ref(): value for value in execution.decisions}
        prompts = {value.to_ref(): value for value in execution.extracted_prompts}
        intents = {value.to_ref(): value for value in execution.inferred_intents}
        rewrites = {value.to_ref(): value for value in execution.rewrite_candidates}
        committer = FactoryTaskAuthoringCommitter(
            store=self.store,
            materials=self.task_authoring_materials,
        )
        results: list[
            tuple[
                FactoryItemRunBindingV2,
                FactoryTaskAuthoringBridgeResult,
            ]
        ] = []
        for binding in self.store.list_item_bindings(request.dataset_run_id):
            decision = decisions.get(binding.trace_candidate_decision_ref)
            prompt = prompts.get(binding.extracted_prompt_ref)
            intent = intents.get(binding.inferred_intent_ref)
            rewrite = rewrites.get(binding.rewrite_candidate_ref)
            if decision is None or prompt is None or intent is None or rewrite is None:
                raise FactoryDatasetRuntimeError("item binding material is unavailable")
            try:
                task_head = self.store.get_item_stage_head(
                    binding.item_id,
                    FactoryItemStageV2.TASK_AUTHORING,
                )
            except FactoryControlNotFoundError:
                result = await self.task_authoring_bridge.run(
                    decision=decision,
                    extracted_prompt=prompt,
                    intent=intent,
                    rewrite=rewrite,
                    requirement=requirement,
                    audit=audit,
                )
            else:
                material_ref = self.store.get_item_stage_material_ref(
                    task_head.to_ref(),
                )
                result = self.task_authoring_materials.get(material_ref)
            committed = committer.commit(
                dataset_run_id=request.dataset_run_id,
                binding=binding,
                result=result,
                audit=audit,
                idempotency_key=(
                    f"{request.idempotency_key}:"
                    "task-authoring-head:"
                    f"{binding.object_sha256}:"
                    f"{result.task_contract_set.contract_set_sha256}"
                ),
            )
            results.append((binding, committed.result))
        return tuple(results)

    def _prepare_job_store(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        task_authoring: tuple[
            tuple[
                FactoryItemRunBindingV2,
                FactoryTaskAuthoringBridgeResult,
            ],
            ...,
        ],
        audit: ContractAudit,
    ) -> FactoryJobStoreAuthority | None:
        if self.job_store_bridge is None:
            return None
        authority = self.job_store_bridge.prepare(
            dataset_run_id=request.dataset_run_id,
            source_trace_refs=tuple(result.source_trace_ref for _binding, result in task_authoring),
            audit=audit,
        )
        binding_item_ids = tuple(binding.item_id for binding, _result in task_authoring)
        if len(set(binding_item_ids)) != len(binding_item_ids) or set(binding_item_ids) != set(
            authority.graph.item_ids
        ):
            raise FactoryDatasetRuntimeError("Factory bindings differ from canonical JobStore Items")
        return authority

    def _validate_job_contexts(
        self,
        *,
        authority: FactoryJobStoreAuthority,
        bindings: tuple[
            FactoryItemRunBindingV2,
            ...,
        ],
    ) -> None:
        bridge = self.job_store_bridge
        if bridge is None:
            raise FactoryDatasetRuntimeError("JobStore authority lacks its bridge")
        expected_path = bridge.job_store.path
        for binding in bindings:
            context = self.item_specialist_contexts.get(binding.item_id)
            if context is None:
                continue
            if (
                context.quality.job_store.path != expected_path
                or context.quality.job_id != authority.job_spec.job_id
                or context.quality.item_id != binding.item_id
            ):
                raise FactoryDatasetRuntimeError(
                    "item specialist context differs from canonical JobStore authority"
                )
        if (
            self.batch_quality_context is not None
            and self.batch_quality_context.resolved_job_work_graph != authority.graph
        ):
            raise FactoryDatasetRuntimeError("batch context differs from canonical JobStore graph")
        if (
            self.release_context is not None
            and self.release_context.source_builder.job_store.path != expected_path
        ):
            raise FactoryDatasetRuntimeError("release context differs from canonical JobStore")

    def _build_item_specialist_contexts(
        self,
        *,
        authority: FactoryJobStoreAuthority,
        bindings: tuple[
            FactoryItemRunBindingV2,
            ...,
        ],
    ) -> None:
        factory = self.item_specialist_context_factory
        if factory is None:
            return
        for binding in bindings:
            current = self.item_specialist_contexts.get(binding.item_id)
            built = factory.build(
                binding=binding,
                job_authority=authority,
            )
            if current is not None and current != built:
                raise FactoryDatasetRuntimeError("item specialist context factory drifted")
            self.item_specialist_contexts[binding.item_id] = built

    def _build_batch_release_contexts(
        self,
        *,
        authority: FactoryJobStoreAuthority,
    ) -> None:
        if self.batch_quality_context_factory is not None:
            built_batch = self.batch_quality_context_factory.build(
                job_authority=authority,
            )
            if self.batch_quality_context is not None and self.batch_quality_context != built_batch:
                raise FactoryDatasetRuntimeError("batch quality context factory drifted")
            self.batch_quality_context = built_batch

    def _build_release_context(
        self,
        *,
        authority: FactoryJobStoreAuthority,
        item_ids: tuple[str, ...],
    ) -> None:
        if self.release_context_factory is None:
            return
        built_release = self.release_context_factory.build(
            job_authority=authority,
            item_ids=item_ids,
        )
        if self.release_context is not None and self.release_context != built_release:
            raise FactoryDatasetRuntimeError("release context factory drifted")
        self.release_context = built_release

    @staticmethod
    def _validate_inputs(
        *,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
    ) -> None:
        if (
            request.dataset_run_id != requirement.run_id
            or request.requirement_spec_ref != requirement.to_ref()
        ):
            raise ValueError("request requirement authority does not match input")
        if request.factory_policy_ref != policy.to_ref():
            raise ValueError("request policy authority does not match input")
        if request.max_transitions > policy.max_transitions:
            raise ValueError("request transition budget exceeds run policy")

    def _ensure_run(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
    ) -> FactoryRunV2:
        try:
            run = self.store.get_run(request.dataset_run_id)
        except FactoryControlNotFoundError:
            return self.store.create_run(
                policy=policy,
                requirement=requirement,
                idempotency_key=(f"{request.idempotency_key}:create-run"),
            )
        if run.policy_ref != policy.to_ref() or run.requirement_spec_ref != requirement.to_ref():
            raise FactoryDatasetRuntimeError("existing dataset run input authority changed")
        return run

    async def _plan(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        run: FactoryRunV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        audit: ContractAudit,
    ) -> FactoryRunV2:
        result = await self.planner.propose(
            task_ref=_stable_ref(
                "agent-task",
                f"{request.object_sha256}:global-planning",
            ),
            run_ref=run.to_ref(),
            requirement=requirement,
            audit=audit,
        )
        compiled = self.compiler.compile(
            plan=result.plan,
            policy=policy,
            audit=audit,
        )
        authority = FactoryDatasetPlanningAuthorityV2.create(
            dataset_run_ref=run.to_ref(),
            planning_version=1,
            predecessor_authority_ref=None,
            plan_ref=result.plan.to_ref(),
            route_decision_ref=result.route.to_ref(),
            invocation_result_ref=result.invocation_result_ref,
            audit=audit,
        )
        self.store.commit_planning_authority(
            authority,
            idempotency_key=(f"{request.idempotency_key}:planning-authority:{authority.object_sha256}"),
        )
        return self.store.commit_plan(
            run_id=run.run_id,
            expected_run_version=run.run_version,
            plan=result.plan,
            compiled_plan=compiled,
            audit=audit,
            idempotency_key=(f"{request.idempotency_key}:commit-global-plan:{result.plan.object_sha256}"),
        )

    def _has_resumed_global_plan(
        self,
        *,
        run_id: str,
        plan: DatasetBuildPlanV2,
    ) -> bool:
        page = self.plan_reviews.list_reviews(
            run_id=run_id,
            state=PlanReviewStateV2.RESUMED,
            limit=500,
        )
        matches = tuple(
            item
            for item in page.items
            if (item.request.plan_kind is PlanKindV2.GLOBAL_BUILD and item.plan.to_ref() == plan.to_ref())
        )
        if len(matches) > 1:
            raise FactoryControlIntegrityError("dataset run has multiple resumed global plans")
        return len(matches) == 1

    def _view(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        run: FactoryRunV2,
        pending_review_refs: tuple[ObjectRef, ...],
        next_action: FactoryDatasetNextActionV2,
        status: FactoryRunStatusV2 | None = None,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2:
        bindings = self.store.list_item_bindings(run.run_id)
        try:
            aggregate = self.store.get_dataset_aggregate(run.run_id)
        except FactoryControlNotFoundError:
            candidate_count = 0
            rejected_count = 0
            blocked_count = 0
            incomplete_count = len(bindings)
            aggregate_ref = None
        else:
            candidate_count = aggregate.candidate_count
            rejected_count = aggregate.rejected_count
            blocked_count = aggregate.blocked_count
            incomplete_count = len(bindings) - aggregate.item_count
            aggregate_ref = aggregate.to_ref()
        observed_status = status or run.status
        if observed_status is FactoryRunStatusV2.COMPLETED:
            next_action = FactoryDatasetNextActionV2.INSPECT_OUTPUT
        return FactoryDatasetRunViewV2.create(
            request_ref=request.to_ref(),
            dataset_run_ref=run.to_ref(),
            status=observed_status,
            pending_review_refs=pending_review_refs,
            item_binding_refs=tuple(binding.to_ref() for binding in bindings),
            candidate_count=candidate_count,
            rejected_count=rejected_count,
            blocked_count=blocked_count,
            incomplete_count=incomplete_count,
            aggregate_result_ref=aggregate_ref,
            delivery_manifest_ref=run.delivery_manifest_ref,
            next_action=next_action,
            audit=audit,
        )


def _batch_quality_source(
    *,
    binding: FactoryItemRunBindingV2,
    task_authoring: FactoryTaskAuthoringBridgeResult,
    specialist: FactoryItemSpecialistView,
) -> BatchQualityItemSource:
    gate = task_authoring.prompt_safety_result.task_prompt_safety_gate
    if gate is None:
        raise FactoryDatasetRuntimeError("batch quality requires prompt safety authority")
    criteria_authority = specialist.criteria.authority if specialist.criteria is not None else None
    task_contract_set = task_authoring.task_contract_set
    item_quality = specialist.quality.result.finalization.item_quality
    if (
        criteria_authority is not None
        and criteria_authority.reviewed_task_contract_set is not None
        and criteria_authority.reviewed_item_quality is not None
    ):
        task_contract_set = criteria_authority.reviewed_task_contract_set
        item_quality = criteria_authority.reviewed_item_quality
    return BatchQualityItemSource(
        item_id=binding.item_id,
        source_trace_ref=task_authoring.source_trace_ref,
        trace_envelope=task_authoring.trace_envelope,
        label_decisions=(task_authoring.label_decision,),
        selection_context=task_authoring.selection_context,
        task_draft=task_authoring.task_draft,
        task_prompt_safety_gate=gate,
        leakage_reference_set=(task_authoring.leakage_reference_set),
        task_contract_set=task_contract_set,
        item_quality=item_quality,
    )


def _release_source_item(
    *,
    binding: FactoryItemRunBindingV2,
    task_authoring: FactoryTaskAuthoringBridgeResult,
    attachment: FactoryAttachmentItemView,
    specialist: FactoryItemSpecialistView,
) -> FactoryReleaseSourceItem:
    gate = task_authoring.prompt_safety_result.task_prompt_safety_gate
    criteria = specialist.criteria
    grading = specialist.grading
    if (
        gate is None
        or attachment.supervised_execution is None
        or specialist.state is not FactoryItemSpecialistState.EXECUTED
        or criteria is None
        or criteria.authority is None
        or criteria.stage_head_ref is None
        or criteria.material_ref is None
        or grading is None
        or grading.authority is None
        or grading.stage_head_ref is None
    ):
        raise FactoryDatasetRuntimeError("release source requires complete item authority")
    reviewed_contract_set = criteria.authority.reviewed_task_contract_set
    reviewed_item_quality = criteria.authority.reviewed_item_quality
    if reviewed_contract_set is None or reviewed_item_quality is None:
        raise FactoryDatasetRuntimeError("release source lacks reviewed Criteria authority")
    return FactoryReleaseSourceItem(
        binding=binding,
        source_trace_ref=task_authoring.source_trace_ref,
        label_decisions=(task_authoring.label_decision,),
        task_draft=task_authoring.task_draft,
        task_prompt_safety_gate=gate,
        base_task_contract_set=(task_authoring.task_contract_set),
        task_contract_set=reviewed_contract_set,
        attachment_result=(attachment.supervised_execution.r5_execution.reconstruction_result),
        attachment_item_quality_ref=(
            item_quality_compilation_result_ref(specialist.quality.result.finalization.item_quality)
        ),
        item_quality=reviewed_item_quality,
        criteria_result_ref=criteria.authority.execution.result.to_ref(),
        criteria_material_ref=criteria.material_ref,
        grading_result_ref=grading.authority.execution.result.to_ref(),
    )


def _item_witness_source(
    *,
    binding: FactoryItemRunBindingV2,
    task_authoring: FactoryTaskAuthoringBridgeResult,
    attachment: FactoryAttachmentItemView,
    specialist: FactoryItemSpecialistView,
) -> FactoryItemStageWitnessSource:
    gate = task_authoring.prompt_safety_result.task_prompt_safety_gate
    if (
        gate is None
        or attachment.supervised_execution is None
        or specialist.state is FactoryItemSpecialistState.BLOCKED_QUALITY
    ):
        raise FactoryDatasetRuntimeError("JobStore witness requires complete item authority")
    return FactoryItemStageWitnessSource(
        item_id=binding.item_id,
        trace_index_ref=task_authoring.source_trace_ref,
        safety_ref=task_prompt_safety_gate_ref(gate),
        label_ref=label_decision_ref(task_authoring.label_decision),
        task_authoring_ref=r4_task_contract_set_ref(task_authoring.task_contract_set),
        attachment_ref=(attachment.supervised_execution.subgraph_result.to_ref()),
        item_quality_ref=(
            item_quality_compilation_result_ref(
                specialist.criteria.authority.reviewed_item_quality
                if (
                    specialist.criteria is not None
                    and specialist.criteria.authority is not None
                    and specialist.criteria.authority.reviewed_item_quality is not None
                )
                else specialist.quality.result.finalization.item_quality
            )
        ),
    )


def _stable_ref(
    object_type: str,
    seed: str,
) -> ObjectRef:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


__all__ = [
    "FactoryBatchQualityContextFactory",
    "FactoryDatasetCoreInput",
    "FactoryDatasetReleaseContext",
    "FactoryDatasetReleaseContextFactory",
    "FactoryDatasetRuntime",
    "FactoryDatasetRuntimeError",
    "FactoryItemSpecialistContextFactory",
]
