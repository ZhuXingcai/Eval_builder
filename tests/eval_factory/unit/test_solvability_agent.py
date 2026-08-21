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
from test_attachment_quality import (
    _item_quality,
    _subgraph_authority,
    _subgraph_result,
)

from eval_factory.agent_system.attachment_quality import (
    AttachmentQualityAgent,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.agent_system.solvability import (
    GatewaySolvabilityAgent,
    SolvabilityAgentConfig,
    SolvabilityAgentError,
    SolvabilityProposalV1,
    SolvabilitySafeViewV1,
)
from eval_factory.ai_gateway.routing import (
    ModelRouteBlockedError,
)
from eval_factory.contracts.agent_system_v2 import (
    AttachmentQualityAssessmentV2,
    AttachmentSubgraphOutcomeV2,
    AttachmentSubgraphResultV2,
    SolvabilityOutcomeV2,
)
from eval_factory.contracts.ai_gateway_v2 import PromptTemplateV2
from eval_factory.contracts.core import ObjectRef


def _solvability_prompt() -> PromptTemplateV2:
    return prompt(
        task_kind="attachment-solvability",
        agent_role="attachment-solvability-agent",
        input_type="solvability-safe-view",
        output_type="solvability-assessment",
    )


def _agent(
    tmp_path: Path,
    *,
    proposal_evidence_refs: tuple[ObjectRef, ...],
    outcome: SolvabilityOutcomeV2 = SolvabilityOutcomeV2.SOLVABLE,
    failed: bool = False,
    generator_only: bool = False,
):
    private_store = FactoryPrivateObjectStore(tmp_path / "private-cas")
    solvability_prompt = _solvability_prompt()
    proposal = SolvabilityProposalV1(
        outcome=outcome,
        evidence_refs=proposal_evidence_refs,
        reason_codes=(() if outcome is SolvabilityOutcomeV2.SOLVABLE else ("NOT_SOLVABLE",)),
    )
    output_ref = private_store.put_model(
        object_type="solvability-proposal",
        value=proposal,
    )
    provider = DeterministicAttachmentProvider(
        {
            _ref_key(solvability_prompt.to_ref()): output_ref,
        },
        failed_prompt_refs=((solvability_prompt.to_ref(),) if failed else ()),
    )
    generator_profile = profile("attachment-generator")
    judge_profile = profile("attachment-solvability-judge")
    agent_definition_ref = ref(
        "agent-definition",
        "attachment-solvability",
    )
    gateway = build_gateway(
        tmp_path,
        prompts=(solvability_prompt,),
        profiles=(generator_profile, judge_profile),
        agent_definition_refs=(agent_definition_ref,),
        provider=provider,
    )
    allowed_refs = (
        (generator_profile.to_ref(),)
        if generator_only
        else tuple(
            sorted(
                (
                    generator_profile.to_ref(),
                    judge_profile.to_ref(),
                ),
                key=_ref_key,
            )
        )
    )
    agent = GatewaySolvabilityAgent(
        gateway=gateway,
        private_store=private_store,
        config=SolvabilityAgentConfig(
            prompt=solvability_prompt,
            agent_definition_ref=agent_definition_ref,
            allowed_model_profile_refs=allowed_refs,
            budget_reservation_ref=ref(
                "work-model-reservation",
                "attachment-solvability",
            ),
            generator_model_profile_ref=(generator_profile.to_ref()),
        ),
    )
    return agent, private_store, provider, generator_profile, judge_profile


async def _quality_authority(
    tmp_path: Path,
    *,
    suffix: str,
) -> tuple[
    AttachmentSubgraphResultV2,
    AttachmentQualityAssessmentV2,
]:
    item_quality, source_validation = await _item_quality(
        tmp_path / "item-quality",
    )
    subgraph, group_results, envelopes = _subgraph_authority(
        AttachmentSubgraphOutcomeV2.SUCCEEDED,
        reconstruction_result_ref=(source_validation.attachment_reconstruction_result_ref),
        suffix=suffix,
    )
    quality = AttachmentQualityAgent().assess(
        subgraph_result=subgraph,
        group_results=group_results,
        work_envelopes=envelopes,
        source_validation=source_validation,
        item_quality=item_quality,
        audit=audit(),
    )
    return subgraph, quality


@pytest.mark.asyncio
async def test_solvability_agent_uses_independent_judge_and_replays(
    tmp_path: Path,
) -> None:
    subgraph, quality = await _quality_authority(
        tmp_path,
        suffix="current",
    )
    evidence_refs = (quality.validator_result_refs[0],)
    agent, private_store, provider, generator, judge = _agent(
        tmp_path,
        proposal_evidence_refs=evidence_refs,
    )

    first = await agent.assess(
        task_ref=ref("agent-task", "attachment-solvability"),
        subgraph_result=subgraph,
        quality=quality,
        evidence_refs=evidence_refs,
        audit=audit(),
    )
    second = await agent.assess(
        task_ref=ref("agent-task", "attachment-solvability"),
        subgraph_result=subgraph,
        quality=quality,
        evidence_refs=evidence_refs,
        audit=audit(),
    )

    assert first.assessment.outcome is SolvabilityOutcomeV2.SOLVABLE
    assert first.route.selected_model_profile_ref == judge.to_ref()
    generator_candidate = next(
        candidate for candidate in first.route.candidates if candidate.model_profile_ref == generator.to_ref()
    )
    assert tuple(value.value for value in generator_candidate.rejection_codes) == (
        "GENERATOR_JUDGE_COLLISION",
    )
    assert second == first
    assert len(provider.calls) == 1

    rendering = private_store.get_model(
        provider.calls[0][0].prompt_rendering_ref,
        SolvabilitySafeViewV1,
    )
    assert rendering.attachment_subgraph_result_ref == subgraph.to_ref()
    assert rendering.quality_assessment_ref == quality.to_ref()
    assert rendering.evidence_refs == evidence_refs


@pytest.mark.asyncio
async def test_solvability_agent_persists_failed_receipt_without_fallback(
    tmp_path: Path,
) -> None:
    subgraph, quality = await _quality_authority(
        tmp_path,
        suffix="failed",
    )
    evidence_refs = (quality.validator_result_refs[0],)
    agent, _, provider, _, _ = _agent(
        tmp_path,
        proposal_evidence_refs=evidence_refs,
        failed=True,
    )

    with pytest.raises(
        SolvabilityAgentError,
        match="did not succeed",
    ):
        await agent.assess(
            task_ref=ref("agent-task", "attachment-solvability-failed"),
            subgraph_result=subgraph,
            quality=quality,
            evidence_refs=evidence_refs,
            audit=audit(),
        )

    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_solvability_agent_blocks_generator_only_route(
    tmp_path: Path,
) -> None:
    subgraph, quality = await _quality_authority(
        tmp_path,
        suffix="blocked-route",
    )
    evidence_refs = (quality.validator_result_refs[0],)
    agent, _, provider, _, _ = _agent(
        tmp_path,
        proposal_evidence_refs=evidence_refs,
        generator_only=True,
    )

    with pytest.raises(ModelRouteBlockedError):
        await agent.assess(
            task_ref=ref("agent-task", "attachment-solvability-blocked"),
            subgraph_result=subgraph,
            quality=quality,
            evidence_refs=evidence_refs,
            audit=audit(),
        )

    assert provider.calls == []


@pytest.mark.asyncio
async def test_solvability_agent_rejects_quality_from_another_subgraph(
    tmp_path: Path,
) -> None:
    _, quality = await _quality_authority(
        tmp_path,
        suffix="first",
    )
    current_subgraph = _subgraph_result(
        AttachmentSubgraphOutcomeV2.SUCCEEDED,
        reconstruction_result_ref=ref(
            "attachment-reconstruction-result",
            "second",
        ),
        suffix="second",
    )
    evidence_refs = (quality.validator_result_refs[0],)
    agent, _, provider, _, _ = _agent(
        tmp_path,
        proposal_evidence_refs=evidence_refs,
    )

    with pytest.raises(
        SolvabilityAgentError,
        match="current attachment subgraph",
    ):
        await agent.assess(
            task_ref=ref("agent-task", "attachment-solvability-cross"),
            subgraph_result=current_subgraph,
            quality=quality,
            evidence_refs=evidence_refs,
            audit=audit(),
        )

    assert provider.calls == []


@pytest.mark.asyncio
async def test_solvability_agent_rejects_evidence_outside_quality_authority(
    tmp_path: Path,
) -> None:
    subgraph, quality = await _quality_authority(
        tmp_path,
        suffix="private-evidence",
    )
    private_ref = ref(
        "model-response-content",
        "private-solvability-material",
    )
    agent, _, provider, _, _ = _agent(
        tmp_path,
        proposal_evidence_refs=(quality.validator_result_refs[0],),
    )

    with pytest.raises(
        SolvabilityAgentError,
        match="quality validator authority",
    ):
        await agent.assess(
            task_ref=ref("agent-task", "attachment-solvability-private"),
            subgraph_result=subgraph,
            quality=quality,
            evidence_refs=(private_ref,),
            audit=audit(),
        )

    assert provider.calls == []


def test_solvability_private_contracts_are_closed_and_safe() -> None:
    subgraph_ref = ref("attachment-subgraph-result", "current")
    quality_ref = ref("attachment-quality-assessment", "current")
    evidence_ref = ref(
        "revision-deterministic-validation",
        "current",
    )

    safe_view = SolvabilitySafeViewV1(
        attachment_subgraph_result_ref=subgraph_ref,
        quality_assessment_ref=quality_ref,
        evidence_refs=(evidence_ref,),
    )
    assert safe_view.evidence_refs == (evidence_ref,)

    with pytest.raises(
        ValidationError,
        match="current attachment refs",
    ):
        SolvabilitySafeViewV1(
            attachment_subgraph_result_ref=ref(
                "attachment-generation-plan",
                "wrong",
            ),
            quality_assessment_ref=quality_ref,
            evidence_refs=(evidence_ref,),
        )
    with pytest.raises(
        ValidationError,
        match="private material",
    ):
        SolvabilitySafeViewV1(
            attachment_subgraph_result_ref=subgraph_ref,
            quality_assessment_ref=quality_ref,
            evidence_refs=(ref("grader-rule", "private"),),
        )
    with pytest.raises(
        ValidationError,
        match="requires reasons",
    ):
        SolvabilityProposalV1(
            outcome=SolvabilityOutcomeV2.BLOCKED,
            evidence_refs=(evidence_ref,),
            reason_codes=(),
        )
    with pytest.raises(
        ValidationError,
        match="cannot retain reasons",
    ):
        SolvabilityProposalV1(
            outcome=SolvabilityOutcomeV2.SOLVABLE,
            evidence_refs=(evidence_ref,),
            reason_codes=("UNEXPECTED_REASON",),
        )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )
