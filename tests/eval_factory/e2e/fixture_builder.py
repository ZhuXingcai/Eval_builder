from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eval_factory.agent_system.attachment_planning import (
    AttachmentGenerationPlanCompiler,
)
from eval_factory.agent_system.attachment_registry import (
    AttachmentAgentRegistryConfig,
    build_attachment_agent_registry,
)
from eval_factory.agent_system.criteria_planning import (
    CriteriaRubricPlanCompiler,
)
from eval_factory.agent_system.criteria_registry import (
    CriteriaRubricAgentRegistryConfig,
    build_criteria_rubric_agent_registry,
)
from eval_factory.agent_system.grading_planning import (
    GradingDesignPlanCompiler,
)
from eval_factory.agent_system.grading_registry import (
    GradingDesignAgentRegistryConfig,
    build_grading_design_agent_registry,
)
from eval_factory.agent_system.plan_adapters import (
    AttachmentGenerationPlanAdapter,
    CriteriaRubricPlanAdapter,
    GlobalBuildPlanAdapter,
    GradingDesignPlanAdapter,
    ReviewablePlanAdapterRegistry,
)
from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
    AttachmentGenerationPlanV2,
    AttachmentMockWorkV2,
    CriteriaRubricGoalV2,
    CriteriaRubricPlanV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    GradingDesignPlanV2,
    JudgeAggregationModeV2,
    JudgeTaskMappingV2,
    PlanKindV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.task import ReferenceMode
from eval_factory.contracts.task_v2 import (
    RubricJudgedObjectKindV2,
)

NOW = datetime(2026, 8, 7, tzinfo=UTC)
USER = "user://plan-owner"


@dataclass(frozen=True, slots=True)
class BrowserFixture:
    store_path: Path
    registry_path: Path
    reviews: dict[str, str]
    run_ids: dict[str, str]


def build_browser_fixture(root: Path) -> BrowserFixture:
    root.mkdir(parents=True, exist_ok=True)
    store_path = root / "factory-control.sqlite3"
    store = FactoryControlStore(store_path)
    registry = _registry()
    global_compiler = DatasetBuildPlanCompiler(registry)
    attachment_compiler = AttachmentGenerationPlanCompiler(registry)
    criteria_compiler = CriteriaRubricPlanCompiler(registry)
    grading_compiler = GradingDesignPlanCompiler(registry)
    service = PlanReviewService(
        store,
        compiler=global_compiler,
        adapter_registry=ReviewablePlanAdapterRegistry(
            (
                GlobalBuildPlanAdapter(global_compiler),
                AttachmentGenerationPlanAdapter(attachment_compiler),
                CriteriaRubricPlanAdapter(criteria_compiler),
                GradingDesignPlanAdapter(grading_compiler),
            )
        ),
        clock=lambda: NOW,
    )
    reviews: dict[str, str] = {}
    run_ids: dict[str, str] = {}
    scenarios = (
        "edit",
        "approve",
        "reject",
        "defer",
        "request-more",
        "conflict",
        "reload",
        "export",
        "invalid",
    )
    for kind in (
        PlanKindV2.ATTACHMENT_GENERATION,
        PlanKindV2.CRITERIA_RUBRIC,
        PlanKindV2.GRADING_DESIGN,
    ):
        for scenario in scenarios:
            key = f"{kind.value.casefold()}-{scenario}"
            run_id = f"factory-run://browser-e2e/{key}"
            run = store.create_run(
                policy=_policy(key),
                requirement=_requirement(run_id, key),
                idempotency_key=f"create-{key}",
            )
            global_plan = _global_plan(
                run.to_ref(),
                key,
                registry,
            )
            compiled_global = global_compiler.compile(
                plan=global_plan,
                policy=_policy(key),
                audit=_audit(),
            )
            store.commit_plan(
                run_id=run_id,
                expected_run_version=run.run_version,
                plan=global_plan,
                compiled_plan=compiled_global,
                audit=_audit(),
                idempotency_key=f"commit-global-{key}",
            )
            current = store.get_run(run_id)
            if kind is PlanKindV2.ATTACHMENT_GENERATION:
                plan = _attachment_plan(
                    current.to_ref(),
                    key,
                    registry,
                )
                compiled = attachment_compiler.compile(
                    plan=plan,
                    policy=_policy(key),
                    audit=_audit(),
                )
            elif kind is PlanKindV2.CRITERIA_RUBRIC:
                plan = _criteria_plan(
                    current.to_ref(),
                    key,
                    registry,
                )
                compiled = criteria_compiler.compile(
                    plan=plan,
                    policy=_policy(key),
                    audit=_audit(),
                )
            else:
                plan = _grading_plan(
                    current.to_ref(),
                    key,
                    registry,
                )
                compiled = grading_compiler.compile(
                    plan=plan,
                    policy=_policy(key),
                    audit=_audit(),
                )
            store.commit_domain_plan(
                run_id=run_id,
                expected_run_version=current.run_version,
                plan_kind=kind,
                plan=plan,
                compiled_plan=compiled,
                audit=_audit(),
                idempotency_key=f"commit-domain-{key}",
            )
            opened = service.open_plan(
                run_id=run_id,
                plan_kind=kind,
                requested_by=USER,
                idempotency_key=f"open-{key}",
                audit=_audit(),
            )
            reviews[key] = opened.request.review_request_id
            run_ids[key] = run_id
    registry_path = root / "agent-registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "capabilities": [value.model_dump(mode="json") for value in registry.capabilities],
                "definitions": [value.model_dump(mode="json") for value in registry.definitions],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return BrowserFixture(
        store_path=store_path,
        registry_path=registry_path,
        reviews=reviews,
        run_ids=run_ids,
    )


def _registry() -> AgentRegistry:
    trace_capability = AgentCapabilityV2.create(
        capability_id="agent-capability://browser-trace",
        task_kinds=("trace-extraction",),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("extracted-user-prompt",),
        model_capabilities=("structured-output",),
        tool_ids=("trace-query",),
        data_purposes=("evaluation-dataset-construction",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=_audit(),
    )
    trace_definition = AgentDefinitionV2.create(
        agent_definition_id="agent-definition://browser-trace",
        agent_role="trace-extraction-agent",
        agent_version="v1",
        capability_refs=(trace_capability.to_ref(),),
        prompt_template_ref=_ref(
            "prompt-template",
            "browser-trace",
        ),
        model_policy_ref=_ref(
            "model-routing-policy",
            "browser-trace",
        ),
        tool_ids=("trace-query",),
        data_purpose="evaluation-dataset-construction",
        allowed_data_classifications=("RESTRICTED_TRACE_DERIVED",),
        validator_refs=(_ref("validator", "browser-trace"),),
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
        workspace_isolated=True,
        network_allowed=False,
        audit=_audit(),
    )
    attachment = build_attachment_agent_registry(
        config=AttachmentAgentRegistryConfig(
            mock_prompt_ref=_ref(
                "prompt-template",
                "attachment-mock",
            ),
            quality_prompt_ref=_ref(
                "prompt-template",
                "attachment-quality",
            ),
            solvability_prompt_ref=_ref(
                "prompt-template",
                "attachment-solvability",
            ),
            mock_model_policy_ref=_ref(
                "model-routing-policy",
                "attachment-mock",
            ),
            quality_model_policy_ref=_ref(
                "model-routing-policy",
                "attachment-quality",
            ),
            solvability_model_policy_ref=_ref(
                "model-routing-policy",
                "attachment-solvability",
            ),
        ),
        audit=_audit(),
    )
    criteria = build_criteria_rubric_agent_registry(
        config=CriteriaRubricAgentRegistryConfig(
            prompt_ref=_ref(
                "prompt-template",
                "criteria-rubric",
            ),
            model_policy_ref=_ref(
                "model-routing-policy",
                "criteria-rubric",
            ),
        ),
        audit=_audit(),
    )
    grading = build_grading_design_agent_registry(
        config=GradingDesignAgentRegistryConfig(
            prompt_ref=_ref(
                "prompt-template",
                "grading-design",
            ),
            model_policy_ref=_ref(
                "model-routing-policy",
                "grading-design",
            ),
        ),
        audit=_audit(),
    )
    return AgentRegistry(
        capabilities=(
            trace_capability,
            *attachment.capabilities,
            *criteria.capabilities,
            *grading.capabilities,
        ),
        definitions=(
            trace_definition,
            *attachment.definitions,
            *criteria.definitions,
            *grading.definitions,
        ),
    )


def _policy(seed: str) -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id=f"factory-run-policy://browser-e2e/{seed}",
        allowed_task_kinds=(
            "attachment-mock",
            "criteria-rubric",
            "grading-design",
            "trace-extraction",
        ),
        max_transitions=128,
        max_plan_revisions=8,
        max_agent_attempts=2,
        max_model_requests=20,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
        audit=_audit(),
    )


def _requirement(
    run_id: str,
    seed: str,
) -> EvaluationRequirementSpecV2:
    return EvaluationRequirementSpecV2.create(
        requirement_spec_id=(f"evaluation-requirement-spec://browser-e2e/{seed}"),
        run_id=run_id,
        source_ref=_ref(
            "evaluation-requirement-source",
            seed,
        ),
        goals=("Review safe evaluation construction plans.",),
        constraints=("Do not expose private evaluator material.",),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )


def _global_plan(
    run_ref: ObjectRef,
    seed: str,
    registry: AgentRegistry,
) -> DatasetBuildPlanV2:
    definition = registry.resolve(
        "trace-extraction-agent",
        "trace-extraction",
    )
    task = DatasetBuildPlanTaskV2(
        task_key="extract",
        stage="core",
        task_kind="trace-extraction",
        agent_role=definition.agent_role,
        dependency_task_keys=(),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("extracted-user-prompt",),
        required_capability_ids=("agent-capability://browser-trace",),
        acceptance_check_refs=definition.validator_refs,
        plan_review_kind=PlanKindV2.GLOBAL_BUILD,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
    )
    return DatasetBuildPlanV2.create(
        plan_id=f"dataset-build-plan://browser-e2e/{seed}",
        run_ref=run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        goals=("Build a safe browser E2E fixture.",),
        user_constraints=("No private content.",),
        assumptions=(),
        unresolved_questions=(),
        stage_order=("core",),
        tasks=(task,),
        required_review_kinds=(PlanKindV2.GLOBAL_BUILD,),
        total_model_requests=1,
        total_model_tokens=4096,
        total_cost_micro_usd=100_000,
        audit=_audit(),
    )


def _attachment_plan(
    run_ref: ObjectRef,
    seed: str,
    registry: AgentRegistry,
) -> AttachmentGenerationPlanV2:
    definition = registry.resolve(
        "attachment-mock-agent",
        "attachment-mock",
    )
    work = AttachmentMockWorkV2(
        work_key="work-main",
        artifact_group_ref=_ref(
            "artifact-execution-group",
            seed,
        ),
        artifact_ids=(f"artifact://browser-e2e/{seed}",),
        agent_role=definition.agent_role,
        dependency_work_keys=(),
        input_object_types=("attachment-planning-context",),
        output_object_types=("attachment-group-result",),
        required_capability_ids=("agent-capability://attachment-mock",),
        allowed_tool_ids=("attachment-execution",),
        data_purposes=("attachment-production",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        workspace_policy_ref=_ref(
            "agent-workspace-policy",
            "isolated",
        ),
        acceptance_check_refs=(_ref("acceptance-check", seed),),
        max_attempts=2,
        max_model_requests=0,
        max_model_tokens=0,
        max_cost_micro_usd=0,
    )
    return AttachmentGenerationPlanV2.create(
        plan_id=(f"attachment-generation-plan://browser-e2e/{seed}"),
        run_ref=run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        producer_task_view_ref=_ref(
            "producer-task-view",
            seed,
        ),
        evidence_bundle_ref=_ref(
            "evidence-bundle",
            seed,
            version="v1",
        ),
        attachment_planning_context_ref=_ref(
            "attachment-planning-context",
            seed,
        ),
        works=(work,),
        max_parallel_groups=1,
        quality_policy_ref=_ref(
            "attachment-quality-policy",
            seed,
        ),
        solvability_policy_ref=_ref(
            "solvability-policy",
            seed,
        ),
        total_model_requests=0,
        total_model_tokens=0,
        total_cost_micro_usd=0,
        audit=_audit(),
    )


def _criteria_plan(
    run_ref: ObjectRef,
    seed: str,
    registry: AgentRegistry,
) -> CriteriaRubricPlanV2:
    definition = registry.resolve(
        "criteria-rubric-agent",
        "criteria-rubric",
    )
    return CriteriaRubricPlanV2.create(
        plan_id=f"criteria-rubric-plan://browser-e2e/{seed}",
        run_ref=run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        task_draft_ref=_ref("task-draft", seed),
        attachment_quality_ref=_ref(
            "attachment-quality-assessment",
            seed,
        ),
        solvability_ref=_ref(
            "solvability-assessment",
            seed,
        ),
        allowed_prompt_requirement_ids=("requirement://visible",),
        required_prompt_requirement_ids=("requirement://visible",),
        allowed_attachment_dependency_ids=(),
        required_attachment_dependency_ids=(),
        allowed_task_tool_ids=(),
        required_task_tool_ids=(),
        criterion_goals=(
            CriteriaRubricGoalV2(
                goal_id=f"criteria-goal://browser-e2e/{seed}",
                goal_summary="Judge only the visible response.",
                judged_object_kind=(RubricJudgedObjectKindV2.CONTESTANT_RESPONSE),
                prompt_requirement_ids=("requirement://visible",),
                evaluator_binding_id=("evaluator-binding://browser-e2e"),
                weight_basis_points=10_000,
            ),
        ),
        allowed_evaluator_binding_ids=("evaluator-binding://browser-e2e",),
        allowed_reference_modes=(ReferenceMode.NONE,),
        selected_reference_mode=ReferenceMode.NONE,
        tool_catalog_ref=_ref(
            "tool-capability-catalog",
            seed,
            version="tool-capability-catalog/r4-07-v1",
        ),
        agent_role=definition.agent_role,
        required_capability_ids=("agent-capability://criteria-rubric",),
        specialist_tool_ids=("rubric-candidate-read",),
        data_classifications=(
            "RESTRICTED_EVALUATOR_CONTROL",
            "RESTRICTED_TRACE_DERIVED",
        ),
        prompt_template_ref=definition.prompt_template_ref,
        model_policy_ref=definition.model_policy_ref,
        acceptance_check_refs=definition.validator_refs,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=16_000,
        max_cost_micro_usd=500_000,
        audit=_audit(),
    )


def _grading_plan(
    run_ref: ObjectRef,
    seed: str,
    registry: AgentRegistry,
) -> GradingDesignPlanV2:
    definition = registry.resolve(
        "grading-design-agent",
        "grading-design",
    )
    return GradingDesignPlanV2.create(
        plan_id=f"grading-design-plan://browser-e2e/{seed}",
        run_ref=run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        criteria_rubric_result_ref=_ref(
            "criteria-rubric-result",
            seed,
        ),
        rubric_set_ref=_ref("rubric-set", seed),
        evaluator_spec_ref=_ref("evaluator-spec", seed),
        reference_policy_ref=_ref(
            "reference-policy",
            seed,
        ),
        tool_policy_ref=_ref("tool-policy", seed),
        generator_model_profile_ref=_ref(
            "model-capability-profile",
            f"{seed}-generator",
        ),
        judge_tasks=(
            JudgeTaskMappingV2(
                task_key="judge-main",
                criterion_ids=("rubric-criterion://browser-e2e",),
                evaluator_binding_id=("evaluator-binding://browser-e2e"),
                score_weight_basis_points=10_000,
            ),
        ),
        aggregation_mode=JudgeAggregationModeV2.WEIGHTED_SUM,
        passing_score_basis_points=7000,
        minimum_confidence_basis_points=8000,
        escalate_on_reference_unavailable=True,
        judge_input_schema_ref=_ref(
            "json-schema",
            "judge-input",
        ),
        judge_output_schema_ref=_ref(
            "json-schema",
            "judge-output",
        ),
        required_output_fields=(
            "ABSTAIN",
            "EVIDENCE_IDS",
            "FAILURE_CLASS",
            "SCORE_BASIS_POINTS",
        ),
        allowed_judge_model_profile_refs=(
            _ref(
                "model-capability-profile",
                f"{seed}-judge",
            ),
        ),
        judge_prompt_template_ref=(definition.prompt_template_ref),
        agent_role=definition.agent_role,
        required_capability_ids=("agent-capability://grading-design",),
        specialist_tool_ids=("reference-grant-read",),
        data_classifications=("RESTRICTED_EVALUATOR_CONTROL",),
        model_policy_ref=definition.model_policy_ref,
        acceptance_check_refs=definition.validator_refs,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=16_000,
        max_cost_micro_usd=500_000,
        audit=_audit(),
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="browser-e2e-fixture",
        governing_versions=(
            VersionBinding(
                component="graph-console-browser-e2e",
                version="v1",
                sha256="a" * 64,
            ),
        ),
    )


def _ref(
    object_type: str,
    seed: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    digest = hashlib.sha256(f"{object_type}:{seed}:{version}".encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version=version,
        object_sha256=digest,
    )


__all__ = [
    "USER",
    "BrowserFixture",
    "build_browser_fixture",
]
