from __future__ import annotations

from datetime import UTC, datetime

import pytest

from eval_factory.agent_system.attachment_planning import (
    AttachmentGenerationPlanCompiler,
    AttachmentGenerationPlanCompilerError,
)
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
    AttachmentGenerationPlanV2,
    AttachmentMockWorkV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str = "example",
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 6, tzinfo=UTC),
        created_by="attachment-agent-planning-test",
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="attachment-plan-v1",
                sha256=HASH,
            ),
        ),
    )


def _registry() -> AgentRegistry:
    capability = AgentCapabilityV2.create(
        capability_id="agent-capability://attachment-mock",
        task_kinds=("attachment-mock",),
        input_object_types=("attachment-planning-context",),
        output_object_types=("attachment-group-result",),
        model_capabilities=("structured-output",),
        tool_ids=("attachment-execution",),
        data_purposes=("attachment-production",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=_audit(),
    )
    definition = AgentDefinitionV2.create(
        agent_definition_id="agent-definition://attachment-mock",
        agent_role="attachment-mock-agent",
        agent_version="v1",
        capability_refs=(capability.to_ref(),),
        prompt_template_ref=_ref("prompt-template", "attachment-mock"),
        model_policy_ref=_ref("model-routing-policy"),
        tool_ids=("attachment-execution",),
        data_purpose="attachment-production",
        allowed_data_classifications=("RESTRICTED_TRACE_DERIVED",),
        validator_refs=(_ref("validator", "attachment-group"),),
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
        workspace_isolated=True,
        network_allowed=False,
        audit=_audit(),
    )
    return AgentRegistry(
        capabilities=(capability,),
        definitions=(definition,),
    )


def _work(
    key: str,
    *,
    dependencies: tuple[str, ...] = (),
    tool_ids: tuple[str, ...] = ("attachment-execution",),
) -> AttachmentMockWorkV2:
    return AttachmentMockWorkV2(
        work_key=key,
        artifact_group_ref=_ref("artifact-execution-group", key),
        artifact_ids=(f"artifact-{key}",),
        agent_role="attachment-mock-agent",
        dependency_work_keys=dependencies,
        input_object_types=("attachment-planning-context",),
        output_object_types=("attachment-group-result",),
        required_capability_ids=("agent-capability://attachment-mock",),
        allowed_tool_ids=tool_ids,
        data_purposes=("attachment-production",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        workspace_policy_ref=_ref("agent-workspace-policy"),
        acceptance_check_refs=(_ref("acceptance-check", key),),
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
    )


def _plan(
    *,
    works: tuple[AttachmentMockWorkV2, ...] | None = None,
    max_parallel_groups: int = 2,
) -> AttachmentGenerationPlanV2:
    values = works or (_work("work-a"), _work("work-b"))
    return AttachmentGenerationPlanV2.create(
        plan_id="attachment-generation-plan://example",
        run_ref=_ref("factory-run"),
        plan_version=1,
        predecessor_plan_ref=None,
        producer_task_view_ref=_ref("producer-task-view"),
        evidence_bundle_ref=_ref("evidence-bundle", version="v1"),
        attachment_planning_context_ref=_ref("attachment-planning-context"),
        works=values,
        max_parallel_groups=max_parallel_groups,
        quality_policy_ref=_ref("attachment-quality-policy"),
        solvability_policy_ref=_ref("solvability-policy"),
        total_model_requests=sum(value.max_model_requests for value in values),
        total_model_tokens=sum(value.max_model_tokens for value in values),
        total_cost_micro_usd=sum(value.max_cost_micro_usd for value in values),
        audit=_audit(),
    )


def _policy(
    *,
    allowed_task_kinds: tuple[str, ...] = ("attachment-mock",),
    max_agent_attempts: int = 2,
    max_model_requests: int = 10,
) -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://attachment",
        allowed_task_kinds=allowed_task_kinds,
        max_transitions=128,
        max_plan_revisions=8,
        max_agent_attempts=max_agent_attempts,
        max_model_requests=max_model_requests,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
        audit=_audit(),
    )


def test_compiler_resolves_agents_and_canonical_topological_order() -> None:
    plan = _plan(
        works=(
            _work("work-b", dependencies=("work-a",)),
            _work("work-a"),
        ),
    )
    compiled = AttachmentGenerationPlanCompiler(_registry()).compile(
        plan=plan,
        policy=_policy(),
        audit=_audit(),
    )

    assert compiled.topological_work_keys == ("work-a", "work-b")
    assert tuple(value.work_key for value in compiled.assignments) == (
        "work-a",
        "work-b",
    )
    assert all(value.agent_definition_ref.object_type == "agent-definition" for value in compiled.assignments)
    assert compiled.max_parallel_groups == 2


def test_compiler_rejects_policy_budget_and_scope_widening() -> None:
    over_budget = _policy(max_model_requests=1)
    with pytest.raises(
        AttachmentGenerationPlanCompilerError,
        match="budget",
    ):
        AttachmentGenerationPlanCompiler(_registry()).compile(
            plan=_plan(),
            policy=over_budget,
            audit=_audit(),
        )

    widened_tool = _plan(
        works=(_work("work-a", tool_ids=("attachment-execution", "shell")),),
        max_parallel_groups=1,
    )
    with pytest.raises(
        AttachmentGenerationPlanCompilerError,
        match="tool",
    ):
        AttachmentGenerationPlanCompiler(_registry()).compile(
            plan=widened_tool,
            policy=_policy(),
            audit=_audit(),
        )


def test_compiler_rejects_parallelism_and_schema_mismatch() -> None:
    with pytest.raises(
        AttachmentGenerationPlanCompilerError,
        match="parallel",
    ):
        AttachmentGenerationPlanCompiler(_registry()).compile(
            plan=_plan(max_parallel_groups=2),
            policy=_policy(),
            audit=_audit(),
            max_parallel_groups=1,
        )

    invalid = _work("work-a").model_copy(
        update={"output_object_types": ("unexpected-output",)},
    )
    invalid_plan = _plan(works=(invalid,), max_parallel_groups=1)
    with pytest.raises(
        AttachmentGenerationPlanCompilerError,
        match="output schema",
    ):
        AttachmentGenerationPlanCompiler(_registry()).compile(
            plan=invalid_plan,
            policy=_policy(),
            audit=_audit(),
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"agent_role": "missing-attachment-agent"}, "resolution"),
        (
            {"required_capability_ids": ("agent-capability://unsupported",)},
            "capability",
        ),
        ({"input_object_types": ("unsupported-input",)}, "input schema"),
        ({"data_purposes": ("unsupported-purpose",)}, "data purpose"),
        (
            {"data_classifications": ("PUBLIC",)},
            "data classification",
        ),
        ({"max_model_requests": 3}, "definition budget"),
    ),
)
def test_compiler_rejects_agent_scope_widening(
    updates: dict[str, object],
    message: str,
) -> None:
    invalid = _work("work-a").model_copy(update=updates)

    with pytest.raises(
        AttachmentGenerationPlanCompilerError,
        match=message,
    ):
        AttachmentGenerationPlanCompiler(_registry()).compile(
            plan=_plan(works=(invalid,), max_parallel_groups=1),
            policy=_policy(),
            audit=_audit(),
        )


@pytest.mark.parametrize(
    ("policy", "message"),
    (
        (
            _policy(allowed_task_kinds=("trace-screening",)),
            "task kind",
        ),
        (
            _policy(max_agent_attempts=1),
            "attempt policy",
        ),
    ),
)
def test_compiler_rejects_run_policy_widening(
    policy: FactoryRunPolicyV2,
    message: str,
) -> None:
    with pytest.raises(
        AttachmentGenerationPlanCompilerError,
        match=message,
    ):
        AttachmentGenerationPlanCompiler(_registry()).compile(
            plan=_plan(),
            policy=policy,
            audit=_audit(),
        )
