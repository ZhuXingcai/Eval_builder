from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

from eval_factory.agent_system.core_vertical import CoreVerticalRunner
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.agent_system.r4_authoring_agent import (
    GatewayR4AuthoringAgent,
)
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    ExtractedUserPromptV2,
    InferredUserIntentV2,
    TaskRewriteCandidateV2,
    TraceCandidateDecisionV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    ObjectRef,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelSpecV2,
    PredicateOperatorV2,
    SelectionContextV2,
    StructuredPredicateV2,
)
from eval_factory.contracts.safety import (
    ContentRiskLabel,
    EvidenceBundle,
    OriginClass,
    TaintLabel,
    Visibility,
)
from eval_factory.contracts.task import (
    ReferenceMode,
)
from eval_factory.contracts.task_v2 import (
    TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2,
    EvaluatorExecutionModeV2,
    PromptLeakageCategoryV2,
    PromptLeakageReferenceSetV2,
    R4TaskContractSetV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
)
from eval_factory.contracts.trace import (
    ParseQuality,
    ToolFamily,
    TraceEnvelope,
    TraceEventType,
)
from eval_factory.labeling import (
    LabelDecisionMerger,
    LabelDecisionMergeRequest,
    StructuredFactSet,
    StructuredPredicateCompiler,
)
from eval_factory.provenance import (
    EvidenceBundleCompiler,
    EvidenceBundleCompileRequest,
    EvidenceViewEngine,
    EvidenceViewPrincipal,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewRequest,
    EvidenceViewResult,
    EvidenceViewSubject,
    ProjectionEvidenceBinding,
    ProvenanceDecisionInput,
    ProvenanceDecisionTable,
)
from eval_factory.task_authoring import (
    EvaluationContractCompiler,
    EvaluationContractOutcome,
    EvaluatorBindingDefinition,
    ProducerTaskViewCompiler,
    ProducerTaskViewProjectionOutcome,
    PromptLeakageReferenceSetCompiler,
    R4TaskContractSetCompiler,
    RestrictedPromptLeakageSource,
    RubricAuthoringOutcome,
    RubricCandidateRequestBuilder,
    RubricSetCompiler,
    SelectionContextFirewall,
    SelectionContextFirewallRequest,
    TaskDraftAuthoringRequestBuilder,
    TaskDraftCompiler,
    TaskEpisodeCompiler,
    TaskEpisodeGroupingRequestBuilder,
    TaskPromptSafetyCompiler,
    TaskPromptSafetyRequestBuilder,
    ToolCapabilityCatalog,
    ToolCapabilityDefinition,
    ToolPolicyCompilationOutcome,
    ToolPolicyCompiler,
    tool_capability_catalog_carried_sha256,
    tool_capability_definition_carried_sha256,
)
from eval_factory.task_authoring.draft_models import (
    TaskDraftAuthoringResult,
)
from eval_factory.task_authoring.evaluation_models import (
    EvaluationContractCompileResult,
)
from eval_factory.task_authoring.models import (
    TaskEpisodeGroupingResult,
)
from eval_factory.task_authoring.producer_models import (
    ProducerTaskViewProjectionResult,
)
from eval_factory.task_authoring.prompt_safety_models import (
    TaskPromptSafetyResult,
)
from eval_factory.task_authoring.rubric_models import (
    RubricAuthoringResult,
)
from eval_factory.task_authoring.tool_models import (
    ToolPolicyCompilationResult,
)
from eval_factory.trace.storage.models import StoredTraceIndex

_FORBIDDEN_OUTPUTS = (
    "grader rules or hidden pass conditions",
    "original agent final answer",
    "pre-completed requested deliverable",
    "private reference material",
)
_MAX_TASK_VISIBLE_PROMPT_CHARACTERS = 20_000
_EVALUATOR_BINDING_ID = "evaluator-binding://factory-dataset/default"


class FactoryTaskAuthoringBridgeError(RuntimeError):
    pass


class FactoryTaskAuthoringBlockReason(StrEnum):
    VISIBLE_PROMPT_CHAR_LIMIT_EXCEEDED = "VISIBLE_PROMPT_CHAR_LIMIT_EXCEEDED"


class FactoryTaskAuthoringCapabilityError(FactoryTaskAuthoringBridgeError):
    def __init__(
        self,
        *,
        reason: FactoryTaskAuthoringBlockReason,
        observed_characters: int,
        limit_characters: int,
    ) -> None:
        self.reason = reason
        self.observed_characters = observed_characters
        self.limit_characters = limit_characters
        super().__init__(f"task authoring capability blocked: {reason.value}")


@dataclass(frozen=True, slots=True)
class FactoryTaskAuthoringBridgeResult:
    source_trace_ref: ObjectRef
    trace_envelope: TraceEnvelope
    label_spec: LabelSpecV2
    label_decision: LabelDecisionV2
    selection_context: SelectionContextV2
    task_episode_result: TaskEpisodeGroupingResult
    task_draft_result: TaskDraftAuthoringResult
    prompt_safety_result: TaskPromptSafetyResult
    task_draft: TaskDraftV2
    rubric_result: RubricAuthoringResult
    evaluation_contract_result: EvaluationContractCompileResult
    tool_policy_result: ToolPolicyCompilationResult
    producer_evidence_view: EvidenceViewResult
    producer_evidence_bundle: EvidenceBundle
    leakage_reference_set: PromptLeakageReferenceSetV2
    producer_task_view_result: ProducerTaskViewProjectionResult
    task_contract_set: R4TaskContractSetV2


class FactoryTaskAuthoringBridge:
    def __init__(
        self,
        *,
        core_runner: CoreVerticalRunner,
        private_store: FactoryPrivateObjectStore,
        agent: GatewayR4AuthoringAgent,
    ) -> None:
        self.core_runner = core_runner
        self.private_store = private_store
        self.agent = agent

    async def run(
        self,
        *,
        decision: TraceCandidateDecisionV2,
        extracted_prompt: ExtractedUserPromptV2,
        intent: InferredUserIntentV2,
        rewrite: TaskRewriteCandidateV2,
        requirement: EvaluationRequirementSpecV2,
        audit: ContractAudit,
    ) -> FactoryTaskAuthoringBridgeResult:
        if (
            decision.cleaned_trace_ref is None
            or rewrite.extracted_prompt_ref != extracted_prompt.to_ref()
            or rewrite.inferred_intent_ref != intent.to_ref()
        ):
            raise FactoryTaskAuthoringBridgeError("task authoring inputs do not form one candidate")
        stored = self.core_runner.load_index(decision.cleaned_trace_ref.object_id)
        original_prompt = self.private_store.get_text(extracted_prompt.content_ref)
        rewritten_prompt = self.private_store.get_text(rewrite.rewritten_prompt_ref).strip()
        if len(rewritten_prompt) > _MAX_TASK_VISIBLE_PROMPT_CHARACTERS:
            raise FactoryTaskAuthoringCapabilityError(
                reason=(FactoryTaskAuthoringBlockReason.VISIBLE_PROMPT_CHAR_LIMIT_EXCEEDED),
                observed_characters=len(rewritten_prompt),
                limit_characters=(_MAX_TASK_VISIBLE_PROMPT_CHARACTERS),
            )
        (
            label_spec,
            label_decision,
            fact_set,
            trace_envelope,
        ) = self._label(
            stored=stored,
            extracted_prompt=extracted_prompt,
            requirement=requirement,
            audit=audit,
        )
        view = self._task_author_view(
            extracted_prompt=extracted_prompt,
            prompt_text=original_prompt,
            audit=audit,
        )
        selection_bundle = self._bundle(
            stored=stored,
            extracted_prompt=extracted_prompt,
            view=view,
            purpose="task-episode-grouping",
            audit=audit,
        )
        authoring_bundle = self._bundle(
            stored=stored,
            extracted_prompt=extracted_prompt,
            view=view,
            purpose="task-draft-authoring",
            audit=audit,
        )
        selection = SelectionContextFirewall().apply(
            SelectionContextFirewallRequest(
                decisions=(label_decision,),
                evidence_bundle=selection_bundle,
                task_authoring_note_refs=(),
                restricted_signal_refs=(),
                audit=audit,
            )
        )
        if selection.selection_context is None:
            raise FactoryTaskAuthoringBridgeError("selection context firewall blocked candidate")
        episode_request = TaskEpisodeGroupingRequestBuilder().build(
            selection_context=selection.selection_context,
            trace_envelope_ref=fact_set.trace_envelope_ref,
            segments=stored.interaction_segments,
            evidence_bundle=selection_bundle,
            model_profile="internal-task-episode-grouper-v1",
            prompt_version="task-episode-grouping-prompt/v1",
            abstain_conditions=(
                "ambiguous task boundary",
                "missing safe evidence",
            ),
            audit=audit,
        )
        episode_proposal = await self.agent.group_episode(
            task_ref=_stable_ref(
                "agent-task",
                f"{rewrite.object_id}:episode",
            ),
            request=episode_request,
            audit=audit,
        )
        episode_result = TaskEpisodeCompiler().compile(
            request=episode_request,
            proposal=episode_proposal,
            selection_context=selection.selection_context,
            segments=stored.interaction_segments,
            evidence_bundle=selection_bundle,
            audit=audit,
        )
        if not episode_result.episodes:
            raise FactoryTaskAuthoringBridgeError("task episode Agent produced no current episode")
        draft_request = TaskDraftAuthoringRequestBuilder().build(
            selection_context=selection.selection_context,
            task_episodes=episode_result.episodes,
            selection_evidence_bundle=selection_bundle,
            authoring_view_result=view,
            authoring_evidence_bundle=authoring_bundle,
            allowed_capability_ids=("instruction-following",),
            allowed_tool_ids=(),
            required_forbidden_outputs=_FORBIDDEN_OUTPUTS,
            abstain_conditions=(
                "ambiguous task intent",
                "missing safe evidence",
            ),
            model_profile="internal-task-author-v1",
            prompt_version="task-draft-authoring-prompt/v1",
            audit=audit,
        )
        intent_summary = " ".join(claim.summary for claim in intent.claims)
        draft_proposal = await self.agent.author_draft(
            task_ref=_stable_ref(
                "agent-task",
                f"{rewrite.object_id}:draft",
            ),
            request=draft_request,
            rewritten_prompt=rewritten_prompt,
            task_intent=intent_summary,
            evaluation_claim=requirement.goals[0],
            audit=audit,
        )
        draft_result = TaskDraftCompiler().compile(
            request=draft_request,
            proposal=draft_proposal,
            selection_context=selection.selection_context,
            task_episodes=episode_result.episodes,
            selection_evidence_bundle=selection_bundle,
            authoring_view_result=view,
            authoring_evidence_bundle=authoring_bundle,
            audit=audit,
        )
        if draft_result.task_draft is None:
            raise FactoryTaskAuthoringBridgeError("task author Agent produced no TaskDraft")
        reference_set = PromptLeakageReferenceSetCompiler().compile(
            trace_envelope_ref=fact_set.trace_envelope_ref,
            sources=self._restricted_sources(
                stored=stored,
                audit=audit,
            ),
            complete_categories=(TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2),
            audit=audit,
        )
        safety_request = TaskPromptSafetyRequestBuilder().build(
            task_draft=draft_result.task_draft,
            leakage_reference_set=reference_set,
            model_profile="internal-task-prompt-safety-v1",
            prompt_version="task-prompt-safety-prompt/v1",
            audit=audit,
        )
        safety_proposal = await self.agent.review_prompt(
            task_ref=_stable_ref(
                "agent-task",
                f"{rewrite.object_id}:prompt-safety",
            ),
            request=safety_request,
            audit=audit,
        )
        safety_result = TaskPromptSafetyCompiler().compile(
            request=safety_request,
            proposal=safety_proposal,
            task_draft=draft_result.task_draft,
            leakage_reference_set=reference_set,
            audit=audit,
        )
        task_draft = safety_result.task_draft
        gate = safety_result.task_prompt_safety_gate
        if (
            task_draft is None
            or gate is None
            or task_draft.prompt_safety_status is not TaskDraftPromptSafetyStatusV2.PASSED
        ):
            raise FactoryTaskAuthoringBridgeError("task prompt safety did not produce a passed TaskDraft")
        rubric_request = RubricCandidateRequestBuilder().build(
            task_draft=task_draft,
            allowed_evaluator_bindings=(_EVALUATOR_BINDING_ID,),
            audit=audit,
        )
        rubric_proposal = await self.agent.author_rubric(
            task_ref=_stable_ref(
                "agent-task",
                f"{rewrite.object_id}:rubric",
            ),
            request=rubric_request,
            audit=audit,
        )
        rubric_result = RubricSetCompiler().compile(
            request=rubric_request,
            proposal=rubric_proposal,
            task_draft=task_draft,
            audit=audit,
        )
        if rubric_result.outcome is not RubricAuthoringOutcome.COMPILED or rubric_result.rubric_set is None:
            raise FactoryTaskAuthoringBridgeError("rubric Agent produced no reachable RubricSet")
        evaluation_result = EvaluationContractCompiler().compile(
            rubric_set=rubric_result.rubric_set,
            binding_definitions=(
                EvaluatorBindingDefinition(
                    evaluator_binding_id=_EVALUATOR_BINDING_ID,
                    evaluator_type="contract-evaluator",
                    execution_mode=(EvaluatorExecutionModeV2.DETERMINISTIC),
                    evaluator_version="factory-dataset/r4-bridge-v1",
                    input_contract_ref=_stable_ref(
                        "evaluator-input-contract",
                        "factory-dataset/r4-bridge-v1",
                    ),
                    output_contract_ref=_stable_ref(
                        "evaluator-output-contract",
                        "factory-dataset/r4-bridge-v1",
                    ),
                    evaluator_principal_id=("principal://evaluator/factory-dataset"),
                    reference_refs=(),
                    timeout_seconds=300,
                ),
            ),
            reference_mode=ReferenceMode.NONE,
            audit=audit,
        )
        if (
            evaluation_result.outcome is not EvaluationContractOutcome.COMPILED
            or evaluation_result.evaluator_spec is None
            or evaluation_result.reference_policy is None
        ):
            raise FactoryTaskAuthoringBridgeError("evaluation contract did not compile")
        tool_result = ToolPolicyCompiler().compile(
            task_draft=task_draft,
            rubric_set=rubric_result.rubric_set,
            evaluator_spec=evaluation_result.evaluator_spec,
            catalog=self._tool_catalog(audit),
            audit=audit,
        )
        if (
            tool_result.outcome is not ToolPolicyCompilationOutcome.COMPILED
            or tool_result.tool_policy is None
            or tool_result.contestant_projection is None
        ):
            raise FactoryTaskAuthoringBridgeError("tool policy did not compile")
        producer_view = self._producer_view(
            extracted_prompt=extracted_prompt,
            prompt_text=original_prompt,
            audit=audit,
        )
        producer_bundle = self._bundle(
            stored=stored,
            extracted_prompt=extracted_prompt,
            view=producer_view,
            consumer_stage="attachment-producer",
            purpose="attachment-production",
            capability="attachment-production",
            audit=audit,
        )
        producer_result = ProducerTaskViewCompiler().compile(
            task_draft=task_draft,
            rubric_set=rubric_result.rubric_set,
            evaluator_spec=evaluation_result.evaluator_spec,
            reference_policy=evaluation_result.reference_policy,
            tool_policy=tool_result.tool_policy,
            contestant_tool_policy=(tool_result.contestant_projection),
            producer_view_result=producer_view,
            producer_evidence_bundle=producer_bundle,
            audit=audit,
        )
        if (
            producer_result.outcome is not ProducerTaskViewProjectionOutcome.PROJECTED
            or producer_result.producer_task_view is None
            or producer_result.storage_authorization is None
        ):
            raise FactoryTaskAuthoringBridgeError("producer task view did not project")
        task_contract_set = R4TaskContractSetCompiler().compile(
            task_draft=task_draft,
            task_prompt_safety_gate=gate,
            rubric_set=rubric_result.rubric_set,
            evaluator_spec=evaluation_result.evaluator_spec,
            reference_policy=evaluation_result.reference_policy,
            tool_policy=tool_result.tool_policy,
            contestant_tool_policy=(tool_result.contestant_projection),
            producer_storage_authorization=(producer_result.storage_authorization),
            producer_task_view=producer_result.producer_task_view,
            audit=audit,
        )
        return FactoryTaskAuthoringBridgeResult(
            source_trace_ref=ObjectRef(
                object_type="trace-source",
                object_id=decision.source_ref.object_id,
                object_version=(stored.manifest.adapter_version),
                object_sha256=decision.source_ref.object_sha256,
            ),
            trace_envelope=trace_envelope,
            label_spec=label_spec,
            label_decision=label_decision,
            selection_context=selection.selection_context,
            task_episode_result=episode_result,
            task_draft_result=draft_result,
            prompt_safety_result=safety_result,
            task_draft=task_draft,
            rubric_result=rubric_result,
            evaluation_contract_result=evaluation_result,
            tool_policy_result=tool_result,
            producer_evidence_view=producer_view,
            producer_evidence_bundle=producer_bundle,
            leakage_reference_set=reference_set,
            producer_task_view_result=producer_result,
            task_contract_set=task_contract_set,
        )

    @staticmethod
    def _label(
        *,
        stored: StoredTraceIndex,
        extracted_prompt: ExtractedUserPromptV2,
        requirement: EvaluationRequirementSpecV2,
        audit: ContractAudit,
    ) -> tuple[
        LabelSpecV2,
        LabelDecisionV2,
        StructuredFactSet,
        TraceEnvelope,
    ]:
        predicate = StructuredPredicateV2(
            predicate_id=("predicate://factory-dataset/user-turn-present/v1"),
            fact_type="interaction-segment",
            field_path="boundary_method",
            operator=PredicateOperatorV2.EQUALS,
            expected_value="user_turn",
            window_events=None,
            required_capability="conversation_events",
            rule_version="factory-dataset-selection/v1",
        )
        label_seed = {
            "requirement": requirement.to_ref().model_dump(mode="json"),
            "predicate": predicate.model_dump(mode="json"),
        }
        label_digest = _hash_json(label_seed)
        label_spec = LabelSpecV2(
            label_spec_id=(f"label-spec://factory-dataset/{label_digest}"),
            label_version="v2",
            name="requirement_candidate",
            requirement=(
                "Candidate contains one source-bound user turn admitted by the reviewed dataset requirement."
            ),
            prerequisite_predicates=(),
            positive_predicates=(predicate,),
            negative_predicates=(),
            semantic_residual=None,
            decision_threshold=1.0,
            review_threshold=1.0,
            label_plan_ref=None,
            policy_version="factory-dataset-selection/v1",
            label_spec_sha256=label_digest,
            audit=audit,
        )
        fact_set = StructuredFactSet.from_stored_index(
            stored,
            audit=audit,
        )
        trace_envelope = _trace_envelope(
            stored,
            fact_set=fact_set,
            audit=audit,
        )
        fact_set = StructuredFactSet.model_validate(
            fact_set.model_copy(
                update={"trace_envelope_ref": _trace_envelope_ref(trace_envelope)}
            ).model_dump(mode="python")
        )
        structured = StructuredPredicateCompiler().compile(
            label_spec=label_spec,
            fact_set=fact_set,
            audit=audit,
        )
        merged = LabelDecisionMerger().merge(
            LabelDecisionMergeRequest(
                label_spec=label_spec,
                structured_result=structured,
                audit=audit,
            )
        )
        if (
            not merged.label_decision.positive_evidence
            or extracted_prompt.source_spans[0].source_trace_id != stored.manifest.source_trace_id
        ):
            raise FactoryTaskAuthoringBridgeError("structured candidate label lacks source evidence")
        return (
            label_spec,
            merged.label_decision,
            fact_set,
            trace_envelope,
        )

    @staticmethod
    def _task_author_view(
        *,
        extracted_prompt: ExtractedUserPromptV2,
        prompt_text: str,
        audit: ContractAudit,
    ) -> EvidenceViewResult:
        return FactoryTaskAuthoringBridge._evidence_view(
            extracted_prompt=extracted_prompt,
            prompt_text=prompt_text,
            principal_id="principal://task-author/factory-dataset",
            principal_type=EvidenceViewPrincipalType.TASK_AUTHOR,
            purpose=EvidenceViewPurpose.TASK_AUTHORING,
            audit=audit,
        )

    @staticmethod
    def _producer_view(
        *,
        extracted_prompt: ExtractedUserPromptV2,
        prompt_text: str,
        audit: ContractAudit,
    ) -> EvidenceViewResult:
        return FactoryTaskAuthoringBridge._evidence_view(
            extracted_prompt=extracted_prompt,
            prompt_text=prompt_text,
            principal_id=("principal://attachment-producer/factory-dataset"),
            principal_type=(EvidenceViewPrincipalType.ATTACHMENT_PRODUCER),
            purpose=EvidenceViewPurpose.ATTACHMENT_PRODUCTION,
            audit=audit,
        )

    @staticmethod
    def _evidence_view(
        *,
        extracted_prompt: ExtractedUserPromptV2,
        prompt_text: str,
        principal_id: str,
        principal_type: EvidenceViewPrincipalType,
        purpose: EvidenceViewPurpose,
        audit: ContractAudit,
    ) -> EvidenceViewResult:
        decision = ProvenanceDecisionTable().decide(
            ProvenanceDecisionInput(
                subject_ref=extracted_prompt.content_ref,
                origin_class=OriginClass.USER_SUPPLIED_INPUT,
                visibility=Visibility.STAGE_PROJECTION,
                rule_ids=("factory-dataset-user-prompt/v1",),
                confidence=1.0,
                audit=audit,
            )
        )
        return EvidenceViewEngine().project(
            EvidenceViewRequest(
                principal=EvidenceViewPrincipal(
                    principal_id=principal_id,
                    principal_type=principal_type,
                    allowed_purposes=frozenset({purpose}),
                    max_subjects=4,
                    max_characters=1_000_000,
                ),
                purpose=purpose,
                subjects=(
                    EvidenceViewSubject(
                        subject_ref=extracted_prompt.content_ref,
                        decision=decision,
                        projection_text=prompt_text,
                    ),
                ),
                max_characters=1_000_000,
                audit=audit,
            )
        )

    @staticmethod
    def _bundle(
        *,
        stored: StoredTraceIndex,
        extracted_prompt: ExtractedUserPromptV2,
        view: EvidenceViewResult,
        consumer_stage: str = "task-authoring",
        purpose: str,
        capability: str = "task-intent-evidence",
        audit: ContractAudit,
    ) -> EvidenceBundle:
        projection = view.included_items[0]
        return (
            EvidenceBundleCompiler()
            .compile(
                EvidenceBundleCompileRequest(
                    source_trace_id=stored.manifest.source_trace_id,
                    trace_ir_version_id=stored.trace_ir_version_id,
                    consumer_stage=consumer_stage,
                    purpose=purpose,
                    view_result=view,
                    bindings=(
                        ProjectionEvidenceBinding(
                            projection_item_id=(projection.projection_item_id),
                            source_spans=(extracted_prompt.source_spans),
                            polarity=EvidencePolarity.POSITIVE,
                            capability=capability,
                            capability_complete=True,
                        ),
                    ),
                    max_characters=1_000_000,
                    audit=audit,
                )
            )
            .evidence_bundle
        )

    @staticmethod
    def _tool_catalog(
        audit: ContractAudit,
    ) -> ToolCapabilityCatalog:
        definition = ToolCapabilityDefinition(
            definition_id="tool-capability-definition://pending",
            tool_id="file-read",
            tool_family=ToolFamily.FILE_READ,
            capability_ref=_stable_ref(
                "tool-capability",
                "factory-dataset/file-read",
            ),
            enforcement_profile_ref=_stable_ref(
                "tool-enforcement-profile",
                "factory-dataset/file-read",
            ),
            constraint_profile_ref=_stable_ref(
                "tool-constraint-profile",
                "factory-dataset/file-read",
            ),
            contestant_descriptor_ref=_stable_ref(
                "contestant-tool-descriptor",
                "factory-dataset/file-read",
            ),
            contestant_constraint_profile_ref=_stable_ref(
                "contestant-tool-constraint-profile",
                "factory-dataset/file-read",
            ),
            contestant_eligible=True,
            definition_sha256="0" * 64,
        )
        definition_digest = tool_capability_definition_carried_sha256(definition)
        definition = definition.model_copy(
            update={
                "definition_id": (f"tool-capability-definition://sha256/{definition_digest}"),
                "definition_sha256": definition_digest,
            }
        )
        catalog = ToolCapabilityCatalog(
            catalog_id="tool-capability-catalog://pending",
            catalog_version=("tool-capability-catalog/factory-dataset-v1"),
            definitions=(definition,),
            catalog_sha256="0" * 64,
            audit=audit,
        )
        catalog_digest = tool_capability_catalog_carried_sha256(catalog)
        return catalog.model_copy(
            update={
                "catalog_id": (f"tool-capability-catalog://sha256/{catalog_digest}"),
                "catalog_sha256": catalog_digest,
            }
        )

    @staticmethod
    def _restricted_sources(
        *,
        stored: StoredTraceIndex,
        audit: ContractAudit,
    ) -> tuple[RestrictedPromptLeakageSource, ...]:
        blobs = {blob.content_ref.object_id: blob for blob in stored.content_blobs}
        values: list[RestrictedPromptLeakageSource] = []
        for event in stored.events:
            if event.event_type is not TraceEventType.ASSISTANT_TEXT or event.content_ref is None:
                continue
            blob = blobs.get(event.content_ref.object_id)
            if blob is None:
                raise FactoryTaskAuthoringBridgeError("assistant output blob is unresolved")
            text = blob.canonical_bytes.decode("utf-8").strip()
            if not text:
                continue
            content_digest = hashlib.sha256(text.encode()).hexdigest()
            subject_id_digest = hashlib.sha256(f"{event.event_id}:{content_digest}".encode()).hexdigest()
            subject_ref = ObjectRef(
                object_type="final-output",
                object_id=(f"final-output://sha256/{subject_id_digest}"),
                object_version="v1",
                object_sha256=content_digest,
            )
            decision = ProvenanceDecisionTable().decide(
                ProvenanceDecisionInput(
                    subject_ref=subject_ref,
                    origin_class=OriginClass.AGENT_GENERATED_FINAL,
                    visibility=Visibility.PRIVATE_STORE,
                    taint_labels=frozenset({TaintLabel.FINAL_OUTPUT_DERIVED}),
                    content_risk_labels=frozenset({ContentRiskLabel.ANSWER_BEARING}),
                    source_event_refs=(
                        ObjectRef(
                            object_type="trace-event",
                            object_id=event.event_id,
                            object_version="trace-ir/v1",
                            object_sha256=event.canonical_sha256(),
                        ),
                    ),
                    rule_ids=("factory-dataset-assistant-output/v1",),
                    confidence=1.0,
                    audit=audit,
                )
            )
            values.append(
                RestrictedPromptLeakageSource(
                    source_id=(
                        "restricted-prompt-leakage-source://"
                        f"{hashlib.sha256(event.event_id.encode()).hexdigest()}"
                    ),
                    category=PromptLeakageCategoryV2.FINAL_ANSWER,
                    source_subject_ref=subject_ref,
                    source_provenance_decision=decision,
                    classification_evidence_ref=_stable_ref(
                        "safety-classification",
                        event.event_id,
                    ),
                    source_text=text,
                    logical_paths=(),
                    content_sha256=content_digest,
                    capability_complete=True,
                )
            )
        return tuple(values)


def _trace_envelope(
    stored: StoredTraceIndex,
    *,
    fact_set: StructuredFactSet,
    audit: ContractAudit,
) -> TraceEnvelope:
    manifest = stored.manifest
    if manifest.trace_ir_schema_version != "v1":
        raise FactoryTaskAuthoringBridgeError("trace envelope schema version is unsupported")
    return TraceEnvelope(
        source_trace_id=manifest.source_trace_id,
        trace_ir_version_id=manifest.trace_ir_version_id,
        source_uri=manifest.source_uri,
        raw_sha256=manifest.raw_sha256,
        adapter_name=manifest.adapter_name,
        adapter_version=manifest.adapter_version,
        trace_ir_schema_version="v1",
        repair_policy_version=manifest.repair_policy_version,
        segmentation_policy_version=(manifest.segmentation_policy_version),
        parse_quality=ParseQuality(manifest.parse_quality),
        capabilities=fact_set.capabilities,
        event_refs=_refs_of_type(
            manifest.object_refs,
            "trace-event",
        ),
        tool_call_refs=_refs_of_type(
            manifest.object_refs,
            "tool-call-record",
        ),
        file_observation_refs=_refs_of_type(
            manifest.object_refs,
            "file-observation",
        ),
        segment_refs=_refs_of_type(
            manifest.object_refs,
            "interaction-segment",
        ),
        repair_map_refs=_refs_of_type(
            manifest.object_refs,
            "repair-map",
        ),
        unrecoverable_span_refs=(),
        audit=audit,
    )


def _trace_envelope_ref(
    envelope: TraceEnvelope,
) -> ObjectRef:
    return ObjectRef(
        object_type="trace-envelope",
        object_id=envelope.trace_ir_version_id,
        object_version="stored-manifest/v1",
        object_sha256=envelope.canonical_sha256(),
    )


def _refs_of_type(
    values: tuple[ObjectRef, ...],
    object_type: str,
) -> tuple[ObjectRef, ...]:
    return tuple(value for value in values if value.object_type == object_type)


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


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
    "FactoryTaskAuthoringBlockReason",
    "FactoryTaskAuthoringBridge",
    "FactoryTaskAuthoringBridgeError",
    "FactoryTaskAuthoringBridgeResult",
    "FactoryTaskAuthoringCapabilityError",
]
