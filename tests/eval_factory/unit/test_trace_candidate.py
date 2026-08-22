from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_support.team_runtime_fixtures import audit, ref

from eval_factory.agent_system.core_vertical import CoreVerticalRunner
from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.agent_system.trace_candidate import (
    TraceCandidatePreparationError,
    TraceCandidatePreparationService,
)
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    InferredUserIntentV2,
    IntentClaimV2,
    TaskRewriteCandidateV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.orchestration.runner import TraceIndexStageService
from eval_factory.trace import TraceSourceRegistry
from eval_factory.trace.storage import TraceIndexStore

ROOT = Path(__file__).resolve().parents[4]
RAW_ROOT = ROOT / "raw_traj"
SID = "0217819691644661fad3a044f20103f8e4b726dec7c2fa4ebc6fa"
SOURCE_PATH = RAW_ROOT / f"LH_002_{SID}.jsonl"

pytestmark = pytest.mark.skipif(
    not SOURCE_PATH.is_file(),
    reason="private 91-trace corpus is not installed",
)


@dataclass(frozen=True)
class _RefValue:
    reference: ObjectRef

    def to_ref(self) -> ObjectRef:
        return self.reference


class _SemanticAgent:
    def __init__(self, private_store: FactoryPrivateObjectStore) -> None:
        self.private_store = private_store
        self.calls: list[str] = []

    async def infer_intent(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append("intent")
        prompt = kwargs["extracted_prompt"]
        intent = InferredUserIntentV2.create(
            inferred_intent_id="inferred-user-intent://trace-candidate",
            extracted_prompt_ref=prompt.to_ref(),  # type: ignore[union-attr]
            claims=(
                IntentClaimV2(
                    claim_id="intent-claim://trace-candidate",
                    summary="Complete the source-grounded task.",
                    evidence_refs=(
                        prompt.to_ref(),  # type: ignore[union-attr]
                    ),
                    confidence_basis_points=9_000,
                    uncertain=False,
                ),
            ),
            unresolved_requirements=(),
            abstained=False,
            audit=audit(),
        )
        return SimpleNamespace(
            intent=intent,
            route=_RefValue(
                ref(
                    "model-route-decision",
                    "trace-candidate-intent",
                    version="v2",
                ),
            ),
        )

    async def rewrite(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append("rewrite")
        prompt = kwargs["extracted_prompt"]
        intent = kwargs["intent"]
        rewritten = self.private_store.put_text(
            object_type="rewritten-prompt-content",
            text="Prepare a source-grounded evaluation task.",
        )
        candidate = TaskRewriteCandidateV2.create(
            candidate_id="task-rewrite-candidate://trace-candidate",
            extracted_prompt_ref=prompt.to_ref(),  # type: ignore[union-attr]
            inferred_intent_ref=intent.to_ref(),  # type: ignore[union-attr]
            rewrite_plan_ref=ref(
                "task-rewrite-plan",
                "trace-candidate",
                version="v2",
            ),
            rewritten_prompt_ref=rewritten,
            evidence_refs=(
                prompt.to_ref(),  # type: ignore[union-attr]
                intent.to_ref(),  # type: ignore[union-attr]
            ),
            audit=audit(),
        )
        return SimpleNamespace(
            candidate=candidate,
            route=_RefValue(
                ref(
                    "model-route-decision",
                    "trace-candidate-rewrite",
                    version="v2",
                ),
            ),
        )


def _indexed(tmp_path: Path):
    raw_sha256 = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    source_trace_id = "source-trace://LH_002/0217819691644661fad3a044f20103f8e4b726dec7c2fa4ebc6fa"
    registry = TraceSourceRegistry(tmp_path / "sources.sqlite3")
    trace = registry.register(
        SOURCE_PATH,
        source_trace_id=source_trace_id,
        source_uri=SOURCE_PATH.as_uri(),
    ).source
    execution = TraceIndexStageService(
        source_registry=registry,
        trace_store=TraceIndexStore(tmp_path / "trace-store"),
    ).execute(
        trace=trace,
        audit=audit(),
        job_id="trace-candidate-test",
        attempt=1,
    )
    source_ref = ObjectRef(
        object_type="trace-source",
        object_id=trace.source_trace_id,
        object_version="v2",
        object_sha256=raw_sha256,
    )
    return execution.stored_index, source_ref


@pytest.mark.asyncio
async def test_trace_candidate_preparation_reuses_indexed_authority(
    tmp_path: Path,
) -> None:
    indexed, source_ref = _indexed(tmp_path)
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    semantic = _SemanticAgent(private_store)
    service = TraceCandidatePreparationService(
        private_store=private_store,
        semantic_agent=semantic,  # type: ignore[arg-type]
    )

    result = await service.prepare(
        indexed=indexed,
        source_ref=source_ref,
        audit=audit(),
    )

    assert result.decision.disposition is (TraceCandidateDispositionV2.CANDIDATE)
    assert result.extracted_prompt is not None
    assert result.inferred_intent is not None
    assert result.rewrite_candidate is not None
    assert semantic.calls == ["intent", "rewrite"]
    assert {
        "trace-candidate-decision",
        "extracted-user-prompt",
        "inferred-user-intent",
        "task-rewrite-candidate",
        "model-route-decision",
    }.issubset(
        {value.object_type for value in result.validation_refs()},
    )


@pytest.mark.asyncio
async def test_trace_candidate_preparation_handles_non_candidate_and_drift(
    tmp_path: Path,
) -> None:
    indexed, source_ref = _indexed(tmp_path)
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    semantic = _SemanticAgent(private_store)
    service = TraceCandidatePreparationService(
        private_store=private_store,
        semantic_agent=semantic,  # type: ignore[arg-type]
    )
    no_user = replace(
        indexed,
        events=tuple(event for event in indexed.events if event.event_type.value != "user_text"),
    )

    result = await service.prepare(
        indexed=no_user,
        source_ref=source_ref,
        audit=audit(),
    )

    assert result.decision.disposition is (TraceCandidateDispositionV2.NON_CANDIDATE)
    assert result.extracted_prompt is None
    assert semantic.calls == []
    with pytest.raises(
        TraceCandidatePreparationError,
        match="source authority",
    ):
        await service.prepare(
            indexed=indexed,
            source_ref=source_ref.model_copy(
                update={"object_sha256": "b" * 64},
            ),
            audit=audit(),
        )


@pytest.mark.asyncio
async def test_core_vertical_reuses_per_source_candidate_service(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / f"LH_002_{SID}.jsonl").write_bytes(
        SOURCE_PATH.read_bytes(),
    )
    manifest = raw_root / "manifest.csv"
    manifest.write_text(
        "instance_id,sid,p_date,business,category,pool_id,"
        "pool_category\n"
        f"LH_002,{SID},2026-06-20,AgentPlan,education,4,"
        "education\n",
        encoding="utf-8",
    )
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    semantic = _SemanticAgent(private_store)
    runner = CoreVerticalRunner(
        workspace=tmp_path / "vertical",
        private_store=private_store,
        semantic_agent=semantic,  # type: ignore[arg-type]
    )
    requirement = EvaluationRequirementSpecV2.create(
        requirement_spec_id="requirement.trace-candidate",
        run_id="factory-run.trace-candidate",
        source_ref=ref(
            "evaluation-requirement-source",
            "trace-candidate",
            version="v2",
        ),
        goals=("Build one source-grounded task.",),
        constraints=(),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=audit(),
    )

    result = await runner.run(
        manifest_path=manifest,
        raw_root=raw_root,
        expected_manifest_sha256=hashlib.sha256(
            manifest.read_bytes(),
        ).hexdigest(),
        requirement=requirement,
        planning_route_ref=ref(
            "model-route-decision",
            "planning",
            version="v2",
        ),
        audit=audit(),
    )

    assert len(result.decisions) == 1
    assert result.decisions[0].disposition is (TraceCandidateDispositionV2.CANDIDATE)
    assert len(result.extracted_prompts) == 1
    assert semantic.calls == ["intent", "rewrite"]
