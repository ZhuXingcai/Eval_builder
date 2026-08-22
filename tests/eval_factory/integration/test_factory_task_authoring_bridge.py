from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
from test_core_vertical_real_traces import (
    RAW_ROOT,
    _audit,
    _gateway,
    _ref,
    _ref_key,
)

from eval_factory.agent_system.core_vertical import CoreVerticalRunner
from eval_factory.agent_system.r4_authoring_agent import (
    GatewayR4AuthoringAgent,
    R4AuthoringAgentConfig,
    RubricAgentInputV1,
    TaskDraftAgentInputV1,
    TaskEpisodeAgentInputV1,
    TaskPromptSafetyAgentInputV1,
)
from eval_factory.agent_system.specialists import (
    GatewayIntentRewriteAgent,
    SemanticAgentConfig,
)
from eval_factory.agent_system.task_authoring_bridge import (
    FactoryTaskAuthoringBlockReason,
    FactoryTaskAuthoringBridge,
    FactoryTaskAuthoringCapabilityError,
)
from eval_factory.agent_system.task_authoring_material import (
    FactoryTaskAuthoringMaterialError,
    FactoryTaskAuthoringMaterialStore,
)
from eval_factory.ai_gateway.invocation import (
    EmbeddedAIGateway,
    ProviderInvocationResult,
)
from eval_factory.ai_gateway.model_catalog import ModelCatalog
from eval_factory.ai_gateway.prompt_registry import PromptRegistry
from eval_factory.ai_gateway.receipts import GatewayRecordStore
from eval_factory.ai_gateway.routing import ModelRouter
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    TaskRewriteCandidateV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    GatewayUsageV2,
    ModelCapabilityProfileV2,
    ModelHealthSnapshotV2,
    ModelPriceScheduleV2,
    ModelQualityBaselineV2,
    ModelRoutingPolicyV2,
    PromptTemplateV2,
)
from eval_factory.contracts.task import (
    AttachmentCriticality,
    EvidencePriority,
    RequirementConflict,
    RubricVisibility,
)
from eval_factory.contracts.task_v2 import (
    RubricJudgedObjectKindV2,
    TaskDraftPromptSafetyStatusV2,
)
from eval_factory.task_authoring import (
    FakeRubricGenerationFixture,
    FakeRubricGenerationRunner,
    FakeTaskDraftAuthoringFixture,
    FakeTaskDraftAuthoringRunner,
    FakeTaskEpisodeGroupingFixture,
    FakeTaskEpisodeGroupingRunner,
    FakeTaskPromptSafetyFixture,
    FakeTaskPromptSafetyRunner,
    RubricAuthoringOutcome,
    RubricCriterionSelection,
    TaskDraftAttachmentFixture,
    TaskDraftAuthoringOutcome,
    TaskDraftContentFixture,
    TaskDraftRequirementFixture,
    TaskEpisodeGroupFixture,
    TaskEpisodeGroupingOutcome,
    TaskEpisodeSegmentSelection,
    TaskPromptSafetyOutcome,
)

pytestmark = pytest.mark.skipif(
    not (RAW_ROOT / "manifest.csv").is_file(),
    reason="private 91-trace corpus is not installed",
)


def _one_trace_root(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "raw"
    root.mkdir()
    with (RAW_ROOT / "manifest.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        row = next(value for value in reader if value["instance_id"] == "LH_077")
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    manifest = root / "manifest.csv"
    with manifest.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    source_name = f"{row['instance_id']}_{row['sid']}.jsonl"
    (root / source_name).write_bytes((RAW_ROOT / source_name).read_bytes())
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _profile(task_kind: str) -> ModelCapabilityProfileV2:
    return ModelCapabilityProfileV2.create(
        model_profile_id=f"model-profile://r4/{task_kind}",
        provider_id="provider://offline-r4",
        model_id=f"model://r4/{task_kind}",
        model_version="v1",
        capabilities=tuple(sorted((task_kind, "structured-output"))),
        supported_data_classifications=("RESTRICTED_TRACE_DERIVED",),
        supported_residencies=("LOCAL",),
        context_limit_tokens=200_000,
        output_limit_tokens=16_000,
        availability="AVAILABLE",
        audit=_audit(),
    )


def _prompt(task_kind: str) -> PromptTemplateV2:
    return PromptTemplateV2.create(
        prompt_template_id=f"prompt-template://r4/{task_kind}",
        agent_role=f"{task_kind}-agent",
        task_kind=task_kind,
        system_template_ref=_ref(
            "prompt-template-content",
            f"{task_kind}-system",
        ),
        instruction_template_ref=_ref(
            "prompt-template-content",
            f"{task_kind}-instruction",
        ),
        required_input_object_types=("r4-authoring-input",),
        output_schema_ref=_ref("json-schema", task_kind),
        allowed_tool_ids=(),
        required_model_capabilities=tuple(sorted((task_kind, "structured-output"))),
        injection_policy_ref=_ref("prompt-injection-policy"),
        template_version=1,
        audit=_audit(),
    )


class _R4Provider:
    def __init__(self, private_store) -> None:
        self.private_store = private_store
        self.calls = 0

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        self.calls += 1
        return self._invoke(request, model_profile)

    def _invoke(
        self,
        request: GatewayInvocationRequestV2,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        capability = next(value for value in model_profile.capabilities if value != "structured-output")
        if capability == "task-episode-grouping":
            rendering = self.private_store.get_model(
                request.prompt_rendering_ref,
                TaskEpisodeAgentInputV1,
            )
            user_segment = next(
                (segment for segment in rendering.request.segments if segment.boundary_method == "user_turn"),
                rendering.request.segments[0],
            )
            proposal = FakeTaskEpisodeGroupingRunner().run(
                rendering.request,
                fixture=FakeTaskEpisodeGroupingFixture(
                    fixture_id="r4-provider-fixture://episode",
                    outcome=TaskEpisodeGroupingOutcome.GROUPED,
                    groups=(
                        TaskEpisodeGroupFixture(
                            group_id="task-episode-group://r4-provider",
                            selections=(
                                TaskEpisodeSegmentSelection(
                                    segment_id=user_segment.segment_id,
                                    evidence_ref_ids=(rendering.request.evidence_refs[0].evidence_ref_id,),
                                ),
                            ),
                            rationale_ref=_ref(
                                "task-episode-rationale",
                                "r4-provider",
                            ),
                        ),
                    ),
                    unresolved_reasons=frozenset(),
                ),
                audit=_audit(),
            )
        elif capability == "task-draft-authoring":
            rendering = self.private_store.get_model(
                request.prompt_rendering_ref,
                TaskDraftAgentInputV1,
            )
            evidence_id = rendering.request.evidence_views[0].evidence_ref_id
            episode_id = rendering.request.task_episode_refs[0].object_id
            requirement_id = "requirement://r4-provider/primary"
            proposal = FakeTaskDraftAuthoringRunner().run(
                rendering.request,
                fixture=FakeTaskDraftAuthoringFixture(
                    fixture_id="r4-provider-fixture://draft",
                    outcome=TaskDraftAuthoringOutcome.DRAFTED,
                    content=TaskDraftContentFixture(
                        visible_prompt=rendering.rewritten_prompt,
                        task_intent=rendering.task_intent,
                        evaluation_claim=(rendering.evaluation_claim),
                        required_capability_ids=("instruction-following",),
                        allowed_tool_ids=(),
                        forbidden_outputs=(rendering.request.required_forbidden_outputs),
                        attachment_dependencies=(
                            TaskDraftAttachmentFixture(
                                dependency_id=("attachment-dependency://r4-provider/context"),
                                description=("Use inputs/context.txt."),
                                criticality=(AttachmentCriticality.REQUIRED),
                                evidence_priority=(EvidencePriority.EXPLICIT_REQUIREMENT),
                                evidence_ref_ids=(evidence_id,),
                            ),
                        ),
                        requirements=(
                            TaskDraftRequirementFixture(
                                requirement_id=requirement_id,
                                statement=(rendering.evaluation_claim),
                                criticality=(AttachmentCriticality.CRITICAL),
                                evidence_priority=(EvidencePriority.EXPLICIT_REQUIREMENT),
                                evidence_ref_ids=(evidence_id,),
                                task_episode_ids=(episode_id,),
                                conflict_status=(RequirementConflict.NONE),
                            ),
                        ),
                        prompt_requirement_ids=(requirement_id,),
                        uncertainties=(),
                    ),
                    unresolved_reasons=frozenset(),
                ),
                audit=_audit(),
            )
        elif capability == "rubric-authoring":
            rendering = self.private_store.get_model(
                request.prompt_rendering_ref,
                RubricAgentInputV1,
            )
            proposal = FakeRubricGenerationRunner().run(
                rendering.request,
                fixture=FakeRubricGenerationFixture(
                    fixture_id="r4-provider-fixture://rubric",
                    outcome=RubricAuthoringOutcome.COMPILED,
                    criteria=(
                        RubricCriterionSelection(
                            selection_id=("rubric-criterion-selection://r4-provider/response"),
                            judged_object_id=("judged-object://r4-provider/contestant-response"),
                            judged_object_kind=(RubricJudgedObjectKindV2.CONTESTANT_RESPONSE),
                            judged_object_description=("The contestant response."),
                            description=(rendering.request.evaluation_claim),
                            weight=1.0,
                            prompt_requirement_ids=tuple(
                                item.requirement_id for item in rendering.request.prompt_requirements
                            ),
                            attachment_dependency_ids=(),
                            allowed_tool_ids=(),
                            visibility=(RubricVisibility.EVALUATOR_ONLY),
                            evaluator_binding=("evaluator-binding://factory-dataset/default"),
                        ),
                    ),
                    unresolved_reasons=frozenset(),
                    model_available=True,
                ),
                model_profile="internal-rubric-author-v1",
                prompt_version="rubric-authoring-prompt/v1",
                audit=_audit(),
            )
        else:
            rendering = self.private_store.get_model(
                request.prompt_rendering_ref,
                TaskPromptSafetyAgentInputV1,
            )
            proposal = FakeTaskPromptSafetyRunner().run(
                rendering.request,
                fixture=FakeTaskPromptSafetyFixture(
                    fixture_id="r4-provider-fixture://safety",
                    outcome=TaskPromptSafetyOutcome.PASSED,
                    findings=(),
                    unresolved_reasons=frozenset(),
                    model_available=True,
                ),
                audit=_audit(),
            )
        output_ref = self.private_store.put_model(
            object_type={
                "task-episode-grouping": ("task-episode-grouping-proposal"),
                "task-draft-authoring": ("task-draft-authoring-proposal"),
                "task-prompt-safety": ("task-prompt-safety-proposal"),
                "rubric-authoring": ("rubric-candidate-proposal"),
            }[capability],
            value=proposal,
        )
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=self.private_store.put_text(
                object_type="model-response-content",
                text=output_ref.object_id,
            ),
            output_ref=output_ref,
            usage=GatewayUsageV2(
                input_tokens=100,
                output_tokens=50,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                charged_tokens=150,
                reported_cost_micro_usd=100,
            ),
            failure_code=None,
        )


def _r4_agent(tmp_path: Path, private_store):
    task_kinds = (
        "task-episode-grouping",
        "task-draft-authoring",
        "task-prompt-safety",
        "rubric-authoring",
    )
    profiles = tuple(_profile(value) for value in task_kinds)
    prompts = tuple(_prompt(value) for value in task_kinds)
    baselines = tuple(
        ModelQualityBaselineV2.create(
            baseline_id=f"model-quality-baseline://r4/{kind}",
            model_profile_ref=profile.to_ref(),
            task_kind=kind,
            benchmark_ref=_ref(
                "model-quality-benchmark",
                kind,
            ),
            quality_basis_points=9_000,
            minimum_sample_count=100,
            audit=_audit(),
        )
        for kind, profile in zip(
            task_kinds,
            profiles,
            strict=True,
        )
    )
    health = tuple(
        ModelHealthSnapshotV2.create(
            health_snapshot_id=f"model-health-snapshot://r4/{kind}",
            model_profile_ref=profile.to_ref(),
            observed_at=_audit().created_at,
            availability="HEALTHY",
            success_basis_points=9_900,
            latency_milliseconds=10,
            audit=_audit(),
        )
        for kind, profile in zip(
            task_kinds,
            profiles,
            strict=True,
        )
    )
    prices = tuple(
        ModelPriceScheduleV2.create(
            price_schedule_id=f"model-price-schedule://r4/{kind}",
            model_profile_ref=profile.to_ref(),
            input_micro_usd_per_million_tokens=1,
            output_micro_usd_per_million_tokens=1,
            effective_from=_audit().created_at,
            audit=_audit(),
        )
        for kind, profile in zip(
            task_kinds,
            profiles,
            strict=True,
        )
    )
    agent_refs = tuple(
        sorted(
            (_ref("agent-definition", value) for value in task_kinds),
            key=_ref_key,
        )
    )
    catalog = ModelCatalog(
        profiles=profiles,
        quality_baselines=baselines,
        health_snapshots=health,
        price_schedules=prices,
    )
    provider = _R4Provider(private_store)
    gateway = EmbeddedAIGateway(
        router=ModelRouter(
            catalog,
            ModelRoutingPolicyV2.create(
                routing_policy_id="model-routing-policy://r4",
                allowed_provider_ids=("provider://offline-r4",),
                allowed_agent_definition_refs=agent_refs,
                allowed_model_profile_refs=tuple(
                    sorted(
                        (profile.to_ref() for profile in profiles),
                        key=_ref_key,
                    )
                ),
                quality_weight=100,
                health_weight=10,
                latency_weight=1,
                cost_weight=1,
                audit=_audit(),
            ),
        ),
        catalog=catalog,
        prompts=PromptRegistry(prompts),
        records=GatewayRecordStore(tmp_path / "r4-gateway.sqlite3"),
        provider=provider,
    )
    by_kind = {value.task_kind: value for value in prompts}
    agent_ref_by_kind = {kind: _ref("agent-definition", kind) for kind in task_kinds}
    return (
        GatewayR4AuthoringAgent(
            gateway=gateway,
            private_store=private_store,
            config=R4AuthoringAgentConfig(
                episode_prompt=by_kind["task-episode-grouping"],
                draft_prompt=by_kind["task-draft-authoring"],
                safety_prompt=by_kind["task-prompt-safety"],
                rubric_prompt=by_kind["rubric-authoring"],
                episode_agent_definition_ref=agent_ref_by_kind["task-episode-grouping"],
                draft_agent_definition_ref=agent_ref_by_kind["task-draft-authoring"],
                safety_agent_definition_ref=agent_ref_by_kind["task-prompt-safety"],
                rubric_agent_definition_ref=agent_ref_by_kind["rubric-authoring"],
                allowed_model_profile_refs=tuple(profile.to_ref() for profile in profiles),
                episode_budget_reservation_ref=_ref(
                    "work-model-reservation",
                    "episode",
                ),
                draft_budget_reservation_ref=_ref(
                    "work-model-reservation",
                    "draft",
                ),
                safety_budget_reservation_ref=_ref(
                    "work-model-reservation",
                    "safety",
                ),
                rubric_budget_reservation_ref=_ref(
                    "work-model-reservation",
                    "rubric",
                ),
            ),
        ),
        provider,
    )


@pytest.mark.asyncio
async def test_real_trace_compiles_r3_r4_authority_and_replays(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha = _one_trace_root(tmp_path)
    gateway, private_store, _core_provider, values = _gateway(tmp_path / "core")
    profiles = values["profiles"]
    prompts = values["prompts"]
    assert isinstance(profiles, dict)
    assert isinstance(prompts, dict)
    profile_refs = tuple(
        sorted(
            (profile.to_ref() for profile in profiles.values()),
            key=_ref_key,
        )
    )
    core_runner = CoreVerticalRunner(
        workspace=tmp_path / "core-workspace",
        private_store=private_store,
        semantic_agent=GatewayIntentRewriteAgent(
            gateway=gateway,
            private_store=private_store,
            config=SemanticAgentConfig(
                intent_prompt=prompts["extraction"],
                rewrite_prompt=prompts["rewrite"],
                intent_agent_definition_ref=_ref(
                    "agent-definition",
                    "extraction",
                ),
                rewrite_agent_definition_ref=_ref(
                    "agent-definition",
                    "rewrite",
                ),
                allowed_model_profile_refs=profile_refs,
                intent_budget_reservation_ref=_ref(
                    "work-model-reservation",
                    "extraction",
                ),
                rewrite_budget_reservation_ref=_ref(
                    "work-model-reservation",
                    "rewrite",
                ),
            ),
        ),
    )
    requirement = EvaluationRequirementSpecV2.create(
        requirement_spec_id="evaluation-requirement-spec://r4-bridge",
        run_id="factory-run://r4-bridge",
        source_ref=_ref(
            "evaluation-requirement-source",
            "r4-bridge",
        ),
        goals=("Evaluate source-grounded instruction following.",),
        constraints=("Do not expose original final output.",),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )
    core = await core_runner.run(
        manifest_path=manifest,
        raw_root=manifest.parent,
        expected_manifest_sha256=manifest_sha,
        requirement=requirement,
        planning_route_ref=_ref(
            "model-route-decision",
            "planning",
        ),
        audit=_audit(),
    )
    r4_agent, provider = _r4_agent(
        tmp_path,
        private_store,
    )
    bridge = FactoryTaskAuthoringBridge(
        core_runner=core_runner,
        private_store=private_store,
        agent=r4_agent,
    )
    first = await bridge.run(
        decision=core.decisions[0],
        extracted_prompt=core.extracted_prompts[0],
        intent=core.inferred_intents[0],
        rewrite=core.rewrite_candidates[0],
        requirement=requirement,
        audit=_audit(),
    )
    calls = provider.calls
    replay = await bridge.run(
        decision=core.decisions[0],
        extracted_prompt=core.extracted_prompts[0],
        intent=core.inferred_intents[0],
        rewrite=core.rewrite_candidates[0],
        requirement=requirement,
        audit=_audit(),
    )

    assert replay == first
    assert provider.calls == calls == 4
    assert first.label_decision.decision.value == "MATCH"
    assert first.task_episode_result.episodes
    assert first.task_draft_result.task_draft is not None
    assert first.task_draft_result.task_draft.prompt_safety_status is TaskDraftPromptSafetyStatusV2.PENDING
    assert first.prompt_safety_result.task_draft is not None
    assert first.prompt_safety_result.task_draft.prompt_safety_status is TaskDraftPromptSafetyStatusV2.PASSED
    assert (
        first.task_contract_set.task_draft_ref.object_id
        == first.prompt_safety_result.task_draft.task_draft_id
    )
    assert first.prompt_safety_result.task_draft.attachment_dependencies
    assert first.producer_evidence_bundle.purpose == "attachment-production"
    assert first.producer_evidence_view.included_items
    assert first.leakage_reference_set.complete_categories
    materials = FactoryTaskAuthoringMaterialStore(private_store)
    material_ref = materials.put(first)
    assert materials.get(material_ref) == first
    with pytest.raises(
        FactoryTaskAuthoringMaterialError,
        match="reference type",
    ):
        materials.get(material_ref.model_copy(update={"object_type": "wrong-material"}))
    public = first.prompt_safety_result.model_dump_json().casefold()
    for marker in (
        "credential",
        "grader-rule",
        "hidden-pass-condition",
        "private-reference",
        "raw-trace",
    ):
        assert marker not in public

    source_rewrite = core.rewrite_candidates[0]
    oversized_rewrite = TaskRewriteCandidateV2.create(
        candidate_id="task-rewrite-candidate://r4-bridge/oversized",
        extracted_prompt_ref=source_rewrite.extracted_prompt_ref,
        inferred_intent_ref=source_rewrite.inferred_intent_ref,
        rewrite_plan_ref=source_rewrite.rewrite_plan_ref,
        rewritten_prompt_ref=private_store.put_text(
            object_type="rewritten-prompt-content",
            text="x" * 20_001,
        ),
        evidence_refs=source_rewrite.evidence_refs,
        audit=_audit(),
    )
    with pytest.raises(FactoryTaskAuthoringCapabilityError) as blocked:
        await bridge.run(
            decision=core.decisions[0],
            extracted_prompt=core.extracted_prompts[0],
            intent=core.inferred_intents[0],
            rewrite=oversized_rewrite,
            requirement=requirement,
            audit=_audit(),
        )

    assert blocked.value.reason is FactoryTaskAuthoringBlockReason.VISIBLE_PROMPT_CHAR_LIMIT_EXCEEDED
    assert blocked.value.observed_characters == 20_001
    assert blocked.value.limit_characters == 20_000
    assert provider.calls == calls
