from __future__ import annotations

from pathlib import Path

import pytest
from attachment_gateway_fixtures import (
    DeterministicAttachmentProvider,
    audit,
    build_gateway,
    profile,
    prompt,
    ref,
)
from pydantic import ValidationError

from eval_factory.agent_system.attachment_planner import (
    AttachmentPlanningAgentConfig,
    AttachmentPlanningAgentError,
    AttachmentPlanningInputV1,
    GatewayAttachmentPlanningAgent,
)
from eval_factory.agent_system.attachment_planning import (
    AttachmentGenerationPlanCompiler,
    AttachmentGenerationPlanCompilerError,
)
from eval_factory.agent_system.attachment_registry import (
    AttachmentAgentRegistryConfig,
    build_attachment_agent_registry,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.ai_gateway.routing import ModelRouteBlockedError
from eval_factory.contracts.agent_system_v2 import (
    AttachmentGenerationPlanV2,
    AttachmentMockWorkV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    PromptTemplateV2,
)
from eval_factory.contracts.core import ObjectRef


def _work(
    key: str,
    *,
    artifact_id: str | None = None,
    dependencies: tuple[str, ...] = (),
) -> AttachmentMockWorkV2:
    return AttachmentMockWorkV2(
        work_key=key,
        artifact_group_ref=ref(
            "artifact-execution-group",
            key,
        ),
        artifact_ids=(artifact_id or f"artifact://{key}",),
        agent_role="attachment-mock-agent",
        dependency_work_keys=dependencies,
        input_object_types=("attachment-planning-context",),
        output_object_types=("attachment-group-result",),
        required_capability_ids=("agent-capability://attachment-mock",),
        allowed_tool_ids=("attachment-execution",),
        data_purposes=("attachment-production",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        workspace_policy_ref=ref(
            "agent-workspace-policy",
            "isolated",
        ),
        acceptance_check_refs=(ref("acceptance-check", key),),
        max_attempts=2,
        max_model_requests=0,
        max_model_tokens=0,
        max_cost_micro_usd=0,
    )


def _plan(
    *,
    run_ref: ObjectRef | None = None,
    works: tuple[AttachmentMockWorkV2, ...] | None = None,
) -> AttachmentGenerationPlanV2:
    values = works or (
        _work("work-a"),
        _work("work-b", dependencies=("work-a",)),
    )
    return AttachmentGenerationPlanV2.create(
        plan_id="attachment-generation-plan://gateway-proposal",
        run_ref=run_ref or ref("factory-run", "current"),
        plan_version=1,
        predecessor_plan_ref=None,
        producer_task_view_ref=ref(
            "producer-task-view",
            "current",
        ),
        evidence_bundle_ref=ref(
            "evidence-bundle",
            "current",
            version="v1",
        ),
        attachment_planning_context_ref=ref(
            "attachment-planning-context",
            "current",
        ),
        works=values,
        max_parallel_groups=min(2, len(values)),
        quality_policy_ref=ref(
            "attachment-quality-policy",
            "current",
        ),
        solvability_policy_ref=ref(
            "solvability-policy",
            "current",
        ),
        total_model_requests=sum(value.max_model_requests for value in values),
        total_model_tokens=sum(value.max_model_tokens for value in values),
        total_cost_micro_usd=sum(value.max_cost_micro_usd for value in values),
        audit=audit(),
    )


def _policy(
    *,
    max_model_requests: int = 10,
    max_agent_attempts: int = 2,
) -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://attachment-planner",
        allowed_task_kinds=("attachment-mock",),
        max_transitions=128,
        max_plan_revisions=8,
        max_agent_attempts=max_agent_attempts,
        max_model_requests=max_model_requests,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
        audit=audit(),
    )


def _planning_prompt() -> PromptTemplateV2:
    return prompt(
        task_kind="attachment-planning",
        agent_role="attachment-planning-agent",
        input_type="attachment-planning-input",
        output_type="attachment-generation-plan",
    )


def _registry():
    return build_attachment_agent_registry(
        config=AttachmentAgentRegistryConfig(
            mock_prompt_ref=ref(
                "prompt-template",
                "attachment-mock",
            ),
            quality_prompt_ref=ref(
                "prompt-template",
                "attachment-quality",
            ),
            solvability_prompt_ref=ref(
                "prompt-template",
                "attachment-solvability",
            ),
            mock_model_policy_ref=ref(
                "model-routing-policy",
                "attachment-mock",
            ),
            quality_model_policy_ref=ref(
                "model-routing-policy",
                "attachment-quality",
            ),
            solvability_model_policy_ref=ref(
                "model-routing-policy",
                "attachment-solvability",
            ),
        ),
        audit=audit(),
    )


def _agent(
    tmp_path: Path,
    *,
    proposed_plan: AttachmentGenerationPlanV2,
    failed: bool = False,
    model_capabilities: tuple[str, ...] = (
        "reasoning",
        "structured-output",
    ),
):
    private_store = FactoryPrivateObjectStore(tmp_path / "private-cas")
    planning_prompt = _planning_prompt()
    output_ref = private_store.put_model(
        object_type="attachment-generation-plan",
        value=proposed_plan,
    )
    provider = DeterministicAttachmentProvider(
        {
            _ref_key(planning_prompt.to_ref()): output_ref,
        },
        failed_prompt_refs=((planning_prompt.to_ref(),) if failed else ()),
    )
    planning_profile = profile(
        "attachment-planning",
        capabilities=model_capabilities,
    )
    agent_definition_ref = ref(
        "agent-definition",
        "attachment-planning",
    )
    gateway = build_gateway(
        tmp_path,
        prompts=(planning_prompt,),
        profiles=(planning_profile,),
        agent_definition_refs=(agent_definition_ref,),
        provider=provider,
    )
    agent = GatewayAttachmentPlanningAgent(
        gateway=gateway,
        private_store=private_store,
        compiler=AttachmentGenerationPlanCompiler(_registry()),
        config=AttachmentPlanningAgentConfig(
            prompt=planning_prompt,
            agent_definition_ref=agent_definition_ref,
            allowed_model_profile_refs=(planning_profile.to_ref(),),
            budget_reservation_ref=ref(
                "work-model-reservation",
                "attachment-planning",
            ),
        ),
    )
    return agent, private_store, provider, planning_profile


def _proposal_kwargs(
    plan: AttachmentGenerationPlanV2,
) -> dict[str, object]:
    return {
        "task_ref": ref("agent-task", "attachment-planning"),
        "run_ref": plan.run_ref,
        "producer_task_view_ref": plan.producer_task_view_ref,
        "evidence_bundle_ref": plan.evidence_bundle_ref,
        "attachment_planning_context_ref": (plan.attachment_planning_context_ref),
        "work_templates": tuple(reversed(plan.works)),
        "quality_policy_ref": plan.quality_policy_ref,
        "solvability_policy_ref": plan.solvability_policy_ref,
        "policy": _policy(),
        "audit": audit(),
    }


@pytest.mark.asyncio
async def test_attachment_planner_routes_compiles_and_replays(
    tmp_path: Path,
) -> None:
    plan = _plan()
    agent, private_store, provider, planning_profile = _agent(
        tmp_path,
        proposed_plan=plan,
    )

    first = await agent.propose(**_proposal_kwargs(plan))
    second = await agent.propose(**_proposal_kwargs(plan))

    assert first.plan == plan
    assert first.compiled_plan.topological_work_keys == (
        "work-a",
        "work-b",
    )
    assert first.route.selected_model_profile_ref == (planning_profile.to_ref())
    assert second == first
    assert len(provider.calls) == 1

    rendering_ref = provider.calls[0][0].prompt_rendering_ref
    rendering = private_store.get_model(
        rendering_ref,
        AttachmentPlanningInputV1,
    )
    assert tuple(value.work_key for value in rendering.work_templates) == (
        "work-a",
        "work-b",
    )


@pytest.mark.asyncio
async def test_attachment_planner_persists_failed_receipt_without_fallback(
    tmp_path: Path,
) -> None:
    plan = _plan()
    agent, _, provider, _ = _agent(
        tmp_path,
        proposed_plan=plan,
        failed=True,
    )

    with pytest.raises(
        AttachmentPlanningAgentError,
        match="did not succeed",
    ):
        await agent.propose(**_proposal_kwargs(plan))

    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_attachment_planner_blocks_unavailable_model_capability(
    tmp_path: Path,
) -> None:
    plan = _plan()
    agent, _, provider, _ = _agent(
        tmp_path,
        proposed_plan=plan,
        model_capabilities=("structured-output",),
    )

    with pytest.raises(ModelRouteBlockedError):
        await agent.propose(**_proposal_kwargs(plan))

    assert provider.calls == []


@pytest.mark.asyncio
async def test_attachment_planner_rejects_source_and_artifact_widening(
    tmp_path: Path,
) -> None:
    source = _plan()
    source_kwargs = _proposal_kwargs(source)

    changed_run = _plan(run_ref=ref("factory-run", "other"))
    run_agent, _, _, _ = _agent(
        tmp_path / "run",
        proposed_plan=changed_run,
    )
    with pytest.raises(
        AttachmentPlanningAgentError,
        match="source authority",
    ):
        await run_agent.propose(**source_kwargs)

    changed_artifact = _plan(
        works=(
            _work("work-a", artifact_id="artifact://other"),
            _work("work-b", dependencies=("work-a",)),
        )
    )
    artifact_agent, _, _, _ = _agent(
        tmp_path / "artifact",
        proposed_plan=changed_artifact,
    )
    with pytest.raises(
        AttachmentPlanningAgentError,
        match="artifact ownership",
    ):
        await artifact_agent.propose(**source_kwargs)


@pytest.mark.asyncio
async def test_attachment_planner_keeps_deterministic_compiler_authority(
    tmp_path: Path,
) -> None:
    plan = _plan()
    agent, _, provider, _ = _agent(
        tmp_path,
        proposed_plan=plan,
    )
    values = _proposal_kwargs(plan)
    values["policy"] = _policy(max_agent_attempts=1)

    with pytest.raises(
        AttachmentGenerationPlanCompilerError,
        match="attempt",
    ):
        await agent.propose(**values)

    assert len(provider.calls) == 1


def test_attachment_planning_input_rejects_invalid_source_refs() -> None:
    plan = _plan()
    values = {
        "run_ref": plan.run_ref,
        "producer_task_view_ref": plan.producer_task_view_ref,
        "evidence_bundle_ref": plan.evidence_bundle_ref,
        "attachment_planning_context_ref": (plan.attachment_planning_context_ref),
        "work_templates": plan.works,
        "quality_policy_ref": plan.quality_policy_ref,
        "solvability_policy_ref": plan.solvability_policy_ref,
    }

    with pytest.raises(ValidationError, match="invalid source refs"):
        AttachmentPlanningInputV1(
            **(
                values
                | {
                    "producer_task_view_ref": ref(
                        "task-draft",
                        "private",
                    )
                }
            )
        )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )
