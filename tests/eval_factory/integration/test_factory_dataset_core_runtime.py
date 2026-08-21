from __future__ import annotations

from pathlib import Path

import pytest
from test_core_vertical_real_traces import (
    MANIFEST_SHA256,
    RAW_ROOT,
    _audit,
    _gateway,
    _ref,
    _ref_key,
)

from eval_factory.agent_system.core_material import (
    FactoryCoreMaterialStore,
)
from eval_factory.agent_system.core_vertical import CoreVerticalRunner
from eval_factory.agent_system.dataset_runtime import (
    FactoryDatasetCoreInput,
    FactoryDatasetRuntime,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.agent_system.specialists import (
    GatewayIntentRewriteAgent,
    GatewayRequirementPlannerAgent,
    RequirementPlannerConfig,
    SemanticAgentConfig,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    PlanDecisionKindV2,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
    FactoryItemStageV2,
)

USER = "user://dataset-core-owner"

pytestmark = pytest.mark.skipif(
    not (RAW_ROOT / "manifest.csv").is_file(),
    reason="private 91-trace corpus is not installed",
)


def _registry() -> AgentRegistry:
    extract = AgentCapabilityV2.create(
        capability_id="agent-capability://trace-extraction",
        task_kinds=("trace-extraction",),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("extracted-user-prompt",),
        model_capabilities=("structured-output",),
        tool_ids=("trace-query",),
        data_purposes=("evaluation-dataset-construction",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=_audit(),
    )
    rewrite = AgentCapabilityV2.create(
        capability_id="agent-capability://task-rewrite",
        task_kinds=("task-rewrite",),
        input_object_types=("extracted-user-prompt",),
        output_object_types=("task-rewrite-candidate",),
        model_capabilities=("structured-output",),
        tool_ids=("task-rewrite",),
        data_purposes=("evaluation-dataset-construction",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=_audit(),
    )
    definitions = tuple(
        AgentDefinitionV2.create(
            agent_definition_id=f"agent-definition://{role}",
            agent_role=role,
            agent_version="v1",
            capability_refs=(capability.to_ref(),),
            prompt_template_ref=_ref("prompt-template", role),
            model_policy_ref=_ref("model-routing-policy"),
            tool_ids=capability.tool_ids,
            data_purpose="evaluation-dataset-construction",
            allowed_data_classifications=("RESTRICTED_TRACE_DERIVED",),
            validator_refs=(_ref("validator", role),),
            max_attempts=2,
            max_model_requests=2,
            max_model_tokens=4_096,
            max_cost_micro_usd=100_000,
            workspace_isolated=True,
            network_allowed=False,
            audit=_audit(),
        )
        for role, capability in (
            ("trace-extraction-agent", extract),
            ("task-rewrite-agent", rewrite),
        )
    )
    return AgentRegistry(
        capabilities=(extract, rewrite),
        definitions=definitions,
    )


def _requirement() -> EvaluationRequirementSpecV2:
    return EvaluationRequirementSpecV2.create(
        requirement_spec_id="evaluation-requirement-spec://runtime-real-91",
        run_id="factory-run://runtime-real-91",
        source_ref=_ref(
            "evaluation-requirement-source",
            "runtime-real-91",
        ),
        goals=("Select traces that can become source-grounded evaluation tasks.",),
        constraints=("Keep original user prompts immutable and source-bound.",),
        assumptions=("This 91-trace cohort validates closure, not quality.",),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )


def _policy() -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://runtime-real-91",
        allowed_task_kinds=(
            "task-rewrite",
            "trace-extraction",
        ),
        max_transitions=2_048,
        max_plan_revisions=8,
        max_agent_attempts=2,
        max_model_requests=10_000,
        max_model_tokens=20_000_000,
        max_cost_micro_usd=250_000_000,
        audit=_audit(),
    )


def _request() -> FactoryDatasetRunRequestV2:
    return FactoryDatasetRunRequestV2.create(
        dataset_run_id=_requirement().run_id,
        requirement_spec_ref=_requirement().to_ref(),
        manifest_ref=_ref(
            "trace-manifest",
            MANIFEST_SHA256,
        ).model_copy(
            update={
                "object_id": (f"trace-manifest://sha256/{MANIFEST_SHA256}"),
                "object_sha256": MANIFEST_SHA256,
            }
        ),
        source_authorization_ref=_ref(
            "trace-source-authorization",
            "runtime-real-91",
        ),
        factory_policy_ref=_policy().to_ref(),
        pipeline_policy_refs=(
            _ref("batch-quality-policy"),
            _ref("release-projection-policy"),
        ),
        gateway_registry_refs=(
            _ref("agent-registry"),
            _ref("model-catalog"),
            _ref("prompt-registry"),
        ),
        output_target_ref=_ref("candidate-output-target"),
        idempotency_key="runtime-real-91",
        max_transitions=2_048,
        audit=_audit(),
    )


def _runtime(
    tmp_path: Path,
) -> tuple[
    FactoryDatasetRuntime,
    PlanReviewService,
    object,
    FactoryControlStore,
    FactoryCoreMaterialStore,
]:
    gateway, private_store, provider, values = _gateway(
        tmp_path,
    )
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
    semantic_agent = GatewayIntentRewriteAgent(
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
    )
    planner = GatewayRequirementPlannerAgent(
        gateway=gateway,
        private_store=private_store,
        config=RequirementPlannerConfig(
            prompt=prompts["planning"],
            agent_definition_ref=_ref(
                "agent-definition",
                "planning",
            ),
            allowed_model_profile_refs=profile_refs,
            budget_reservation_ref=_ref(
                "work-model-reservation",
                "planning",
            ),
        ),
    )
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    compiler = DatasetBuildPlanCompiler(_registry())
    reviews = PlanReviewService(
        store,
        compiler=compiler,
    )
    materials = FactoryCoreMaterialStore(store)
    runtime = FactoryDatasetRuntime(
        store=store,
        planner=planner,
        compiler=compiler,
        plan_reviews=reviews,
        requested_by=USER,
        core_runner=CoreVerticalRunner(
            workspace=tmp_path / "core-workspace",
            private_store=private_store,
            semantic_agent=semantic_agent,
        ),
        core_materials=materials,
    )
    return runtime, reviews, provider, store, materials


@pytest.mark.asyncio
async def test_runtime_executes_real_core_and_materializes_child_runs(
    tmp_path: Path,
) -> None:
    runtime, reviews, provider, store, materials = _runtime(
        tmp_path,
    )
    waiting = await runtime.advance(
        request=_request(),
        policy=_policy(),
        requirement=_requirement(),
        audit=_audit(),
    )
    review_id = waiting.pending_review_refs[0].object_id
    reviews.decide(
        review_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by=USER,
            reason_code="GLOBAL_PLAN_APPROVED",
            idempotency_key="approve-real-core-runtime",
        ),
        audit=_audit(),
    )
    reviews.resume(
        review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-real-core-runtime",
        ),
        audit=_audit(),
    )
    core_input = FactoryDatasetCoreInput(
        manifest_path=RAW_ROOT / "manifest.csv",
        raw_root=RAW_ROOT,
        expected_manifest_sha256=MANIFEST_SHA256,
    )

    completed_core = await runtime.advance(
        request=_request(),
        policy=_policy(),
        requirement=_requirement(),
        core_input=core_input,
        audit=_audit(),
    )
    call_count = provider.calls
    replay = await runtime.advance(
        request=_request(),
        policy=_policy(),
        requirement=_requirement(),
        core_input=core_input,
        audit=_audit(),
    )

    assert completed_core == replay
    assert len(completed_core.item_binding_refs) == 91
    assert completed_core.incomplete_count == 91
    assert provider.calls == call_count == 183
    execution = materials.get(_requirement().run_id)
    assert execution.result.source_count == 91
    assert len(execution.rewrite_candidates) == 91
    bindings = store.list_item_bindings(_requirement().run_id)
    assert len(bindings) == 91
    assert all(binding.item_id.startswith("item://r6-02/sha256/") for binding in bindings)
    for binding in bindings:
        head = store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.CORE_SELECTION,
        )
        assert head.item_binding_ref == binding.to_ref()
        assert head.result_ref == binding.rewrite_candidate_ref
