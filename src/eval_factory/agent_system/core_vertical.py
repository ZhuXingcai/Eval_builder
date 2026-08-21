from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.agent_system.specialists import GatewayIntentRewriteAgent
from eval_factory.agent_system.trace_candidate import (
    TraceCandidatePreparationService,
)
from eval_factory.contracts.agent_system_v2 import (
    CoreVerticalResultV2,
    EvaluationRequirementSpecV2,
    ExtractedUserPromptV2,
    InferredUserIntentV2,
    TaskRewriteCandidateV2,
    TraceCandidateDecisionV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.trace import (
    RawTrajRecovery,
    RawTrajV1Normalizer,
    RawTrajV1Parser,
    TraceIndexBuilder,
    TraceSourceRegistry,
)
from eval_factory.trace.indexing.models import TraceIndexResult
from eval_factory.trace.parsing.raw_traj_v1 import TraceParseError
from eval_factory.trace.source_registry import TraceSourceRegistryError
from eval_factory.trace.storage import TraceIndexStore
from eval_factory.trace.storage.models import (
    StoredTraceIndex,
    StoredTraceManifest,
    TraceObjectNotFoundError,
)


class CoreVerticalError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CoreVerticalExecution:
    result: CoreVerticalResultV2
    decisions: tuple[TraceCandidateDecisionV2, ...]
    extracted_prompts: tuple[ExtractedUserPromptV2, ...]
    inferred_intents: tuple[InferredUserIntentV2, ...]
    rewrite_candidates: tuple[TaskRewriteCandidateV2, ...]


@dataclass(frozen=True, slots=True)
class _CoreVerticalRunBinding:
    manifest_sha256: str
    requirement_ref: ObjectRef
    planning_route_ref: ObjectRef
    source_refs: tuple[ObjectRef, ...]
    audit_sha256: str


class CoreVerticalRunner:
    def __init__(
        self,
        *,
        workspace: Path,
        private_store: FactoryPrivateObjectStore,
        semantic_agent: GatewayIntentRewriteAgent,
    ) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.private_store = private_store
        self.semantic_agent = semantic_agent
        self.candidate_preparer = TraceCandidatePreparationService(
            private_store=private_store,
            semantic_agent=semantic_agent,
        )
        self.trace_store = TraceIndexStore(
            self.workspace / "trace-store",
        )
        self._versioned_trace_store_root = self.workspace / "trace-store" / "by-version" / "sha256"
        self._last_binding: _CoreVerticalRunBinding | None = None
        self._last_execution: CoreVerticalExecution | None = None

    def load_index(
        self,
        trace_ir_version_id: str,
    ) -> StoredTraceIndex:
        versioned_root = self._versioned_trace_store_path(
            trace_ir_version_id,
        )
        if versioned_root.exists():
            try:
                return TraceIndexStore(
                    versioned_root,
                ).load(trace_ir_version_id)
            except TraceObjectNotFoundError:
                pass
        return self.trace_store.load(trace_ir_version_id)

    def load_manifest(
        self,
        trace_ir_version_id: str,
    ) -> StoredTraceManifest:
        versioned_root = self._versioned_trace_store_path(
            trace_ir_version_id,
        )
        if versioned_root.exists():
            try:
                return TraceIndexStore(
                    versioned_root,
                ).load_manifest(trace_ir_version_id)
            except TraceObjectNotFoundError:
                pass
        return self.trace_store.load_manifest(trace_ir_version_id)

    async def run(
        self,
        *,
        manifest_path: Path,
        raw_root: Path,
        expected_manifest_sha256: str,
        requirement: EvaluationRequirementSpecV2,
        planning_route_ref: ObjectRef,
        audit: ContractAudit,
    ) -> CoreVerticalExecution:
        manifest_bytes = manifest_path.read_bytes()
        observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if observed_manifest_sha256 != expected_manifest_sha256:
            raise CoreVerticalError("trace manifest digest differs from approved input")
        entries = _read_manifest(manifest_path)
        if not entries:
            raise CoreVerticalError("trace manifest is empty")
        sources = tuple(
            (
                instance_id,
                sid,
                raw_root / f"{instance_id}_{sid}.jsonl",
                _source_ref(
                    f"source-trace://{instance_id}/{sid}",
                    raw_root / f"{instance_id}_{sid}.jsonl",
                ),
            )
            for instance_id, sid in entries
        )
        binding = _CoreVerticalRunBinding(
            manifest_sha256=observed_manifest_sha256,
            requirement_ref=requirement.to_ref(),
            planning_route_ref=planning_route_ref,
            source_refs=tuple(source_ref for _, _, _, source_ref in sources),
            audit_sha256=hashlib.sha256(audit.canonical_json()).hexdigest(),
        )
        if self._last_binding == binding and self._last_execution is not None:
            return self._last_execution
        registry = TraceSourceRegistry(self.workspace / "trace-source-registry.sqlite3")
        decisions: list[TraceCandidateDecisionV2] = []
        prompts: list[ExtractedUserPromptV2] = []
        intents: list[InferredUserIntentV2] = []
        rewrites: list[TaskRewriteCandidateV2] = []
        route_refs: list[ObjectRef] = [planning_route_ref]

        for instance_id, sid, source_path, source_ref in sources:
            source_trace_id = f"source-trace://{instance_id}/{sid}"
            try:
                indexed = _index_trace(
                    source_path,
                    source_trace_id=source_trace_id,
                    registry=registry,
                    audit=audit,
                )
                versioned_store = self._versioned_trace_store(
                    indexed.normalization_result.trace_ir_version_id,
                )
                materialized = versioned_store.materialize(indexed)
                try:
                    stored = versioned_store.load(
                        indexed.normalization_result.trace_ir_version_id,
                    )
                except TraceObjectNotFoundError:
                    _commit, stored = versioned_store.persist_and_materialize(
                        indexed,
                        audit=audit,
                    )
                if stored != materialized:
                    raise CoreVerticalError(
                        "stored trace index differs from current source",
                    )
                prepared = await self.candidate_preparer.prepare(
                    indexed=stored,
                    source_ref=source_ref,
                    audit=audit,
                )
                decisions.append(prepared.decision)
                if prepared.extracted_prompt is not None:
                    prompts.append(prepared.extracted_prompt)
                if prepared.inferred_intent is not None:
                    intents.append(prepared.inferred_intent)
                if prepared.rewrite_candidate is not None:
                    rewrites.append(prepared.rewrite_candidate)
                route_refs.extend(prepared.route_refs)
            except (
                CoreVerticalError,
                TraceParseError,
                TraceSourceRegistryError,
                UnicodeDecodeError,
                ValueError,
                RuntimeError,
            ):
                decisions.append(
                    TraceCandidateDecisionV2.create(
                        decision_id=f"trace-candidate-decision://{instance_id}/blocked",
                        source_trace_id=source_trace_id,
                        source_ref=source_ref,
                        disposition=TraceCandidateDispositionV2.BLOCKED,
                        reason_codes=("TRACE_PROCESSING_FAILED",),
                        cleaned_trace_ref=None,
                        audit=audit,
                    )
                )

        by_disposition = {
            disposition: tuple(
                sorted(
                    (decision for decision in decisions if decision.disposition is disposition),
                    key=lambda value: value.source_trace_id,
                )
            )
            for disposition in TraceCandidateDispositionV2
        }
        manifest_ref = ObjectRef(
            object_type="trace-manifest",
            object_id=f"trace-manifest://sha256/{observed_manifest_sha256}",
            object_version="v2",
            object_sha256=observed_manifest_sha256,
        )
        result = CoreVerticalResultV2.create(
            result_id=f"core-vertical-result://sha256/{observed_manifest_sha256}",
            manifest_ref=manifest_ref,
            requirement_spec_ref=requirement.to_ref(),
            candidate_decision_refs=_sorted_refs(
                tuple(decision.to_ref() for decision in by_disposition[TraceCandidateDispositionV2.CANDIDATE])
            ),
            non_candidate_decision_refs=_sorted_refs(
                tuple(
                    decision.to_ref()
                    for decision in by_disposition[TraceCandidateDispositionV2.NON_CANDIDATE]
                )
            ),
            blocked_decision_refs=_sorted_refs(
                tuple(decision.to_ref() for decision in by_disposition[TraceCandidateDispositionV2.BLOCKED])
            ),
            extracted_prompt_refs=_sorted_refs(tuple(value.to_ref() for value in prompts)),
            inferred_intent_refs=_sorted_refs(tuple(value.to_ref() for value in intents)),
            rewrite_candidate_refs=_sorted_refs(tuple(value.to_ref() for value in rewrites)),
            route_decision_refs=_sorted_refs(tuple(route_refs)),
            source_count=len(entries),
            audit=audit,
        )
        execution = CoreVerticalExecution(
            result=result,
            decisions=tuple(sorted(decisions, key=lambda value: value.source_trace_id)),
            extracted_prompts=tuple(sorted(prompts, key=lambda value: value.object_id)),
            inferred_intents=tuple(sorted(intents, key=lambda value: value.object_id)),
            rewrite_candidates=tuple(sorted(rewrites, key=lambda value: value.object_id)),
        )
        self._last_binding = binding
        self._last_execution = execution
        return execution

    def _versioned_trace_store(
        self,
        trace_ir_version_id: str,
    ) -> TraceIndexStore:
        return TraceIndexStore(
            self._versioned_trace_store_path(
                trace_ir_version_id,
            )
        )

    def _versioned_trace_store_path(
        self,
        trace_ir_version_id: str,
    ) -> Path:
        digest = hashlib.sha256(
            trace_ir_version_id.encode(),
        ).hexdigest()
        return self._versioned_trace_store_root / digest[:2] / digest


def _index_trace(
    source_path: Path,
    *,
    source_trace_id: str,
    registry: TraceSourceRegistry,
    audit: ContractAudit,
) -> TraceIndexResult:
    registered = registry.register(
        source_path,
        source_trace_id=source_trace_id,
        source_uri=f"raw-traj://{_suffix(source_trace_id)}",
    )
    parsed = RawTrajV1Parser().parse(
        source_path,
        registered_source=registered,
        audit=audit,
    )
    recovered = RawTrajRecovery().recover(
        source_path,
        parse_result=parsed,
        audit=audit,
    )
    normalized = RawTrajV1Normalizer().normalize(
        source_path,
        recovery_result=recovered,
        audit=audit,
    )
    return TraceIndexBuilder().build(
        normalization_result=normalized,
        audit=audit,
    )


def _read_manifest(path: Path) -> tuple[tuple[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [
            "instance_id",
            "sid",
            "p_date",
            "business",
            "category",
            "pool_id",
            "pool_category",
        ]:
            raise CoreVerticalError("trace manifest columns are not canonical")
        values = tuple((str(row["instance_id"]), str(row["sid"])) for row in reader)
    if len(set(values)) != len(values):
        raise CoreVerticalError("trace manifest contains duplicate members")
    return values


def _source_ref(source_trace_id: str, source_path: Path) -> ObjectRef:
    if not source_path.is_file() or source_path.is_symlink():
        raise CoreVerticalError("trace source file is missing or unsafe")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return ObjectRef(
        object_type="trace-source",
        object_id=source_trace_id,
        object_version="v2",
        object_sha256=digest,
    )


def _suffix(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda value: (
                value.object_type,
                value.object_id,
                value.object_version,
                value.object_sha256,
            ),
        )
    )


__all__ = [
    "CoreVerticalError",
    "CoreVerticalExecution",
    "CoreVerticalRunner",
]
