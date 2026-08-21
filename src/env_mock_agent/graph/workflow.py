from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from env_mock_agent.graph.idempotency import operation_key
from env_mock_agent.graph.state import GraphState
from env_mock_agent.profiles import get_profile
from env_mock_agent.providers import (
    ProviderRegistry,
    ProviderRequest,
    preflight_dependencies,
)
from env_mock_agent.providers.helpers import artifact_result, sha256_path
from env_mock_agent.runtimes import RuntimeRegistry
from env_mock_agent.schemas import (
    ArtifactPlan,
    ArtifactResult,
    ArtifactStatus,
    Criticality,
    DependencySpec,
    FindingCategory,
    ModelProfile,
    Problem,
    ProblemLedger,
    ProblemStatus,
    ReviewRoundResult,
    RunStatus,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeName,
    RuntimeRequest,
    Severity,
    SourceEvidence,
    SourceType,
    TaskSpec,
    ValidationFinding,
    WorldLedger,
)
from env_mock_agent.store import RunStore
from env_mock_agent.validators import ValidationRequest, ValidatorRegistry
from env_mock_agent.world import build_world_ledger

ROUND_CATEGORIES = {
    1: {
        FindingCategory.STRUCTURE,
        FindingCategory.COVERAGE,
        FindingCategory.SOLVABILITY,
        FindingCategory.TRACEABILITY,
        FindingCategory.CAPABILITY,
    },
    2: {
        FindingCategory.AUTHENTICITY,
        FindingCategory.CONSISTENCY,
        FindingCategory.METADATA,
    },
    3: {
        FindingCategory.ANSWER_LEAKAGE,
        FindingCategory.SECURITY,
        FindingCategory.EXECUTABILITY,
    },
}

ROUND_NAMES = {
    1: "coverage_and_solvability",
    2: "realism_and_consistency",
    3: "leakage_and_executability",
}


def _int_value(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


class EnvironmentMockWorkflow:
    def __init__(
        self,
        *,
        providers: ProviderRegistry | None = None,
        validators: ValidatorRegistry | None = None,
        runtimes: RuntimeRegistry | None = None,
    ) -> None:
        self.providers = providers or ProviderRegistry.default()
        self.validators = validators or ValidatorRegistry.default()
        self.runtimes = runtimes or RuntimeRegistry.default()

    def compile(
        self,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        *,
        interrupt_after: list[str] | None = None,
    ) -> object:
        graph = StateGraph(GraphState)
        graph.add_node("normalize", self.normalize)
        graph.add_node("classify", self.classify)
        graph.add_node("capability_preflight", self.capability_preflight)
        graph.add_node("retrieve_evidence", self.retrieve_evidence)
        graph.add_node("build_world_ledger", self.world_ledger)
        graph.add_node("plan_artifacts", self.plan_artifacts)
        graph.add_node("build_artifacts", self.build_artifacts)
        graph.add_node("validate_artifacts", self.validate_artifacts)
        graph.add_node("review_round_1", lambda state: self.review(state, 1))
        graph.add_node("repair_round_1", lambda state: self.repair(state, 1))
        graph.add_node("review_round_2", lambda state: self.review(state, 2))
        graph.add_node("repair_round_2", lambda state: self.repair(state, 2))
        graph.add_node("review_round_3", lambda state: self.review(state, 3))
        graph.add_node("final_gate", self.final_gate)
        graph.add_node("assemble", self.assemble)
        graph.add_node("export", self.export)

        graph.add_edge(START, "normalize")
        graph.add_edge("normalize", "classify")
        graph.add_edge("classify", "capability_preflight")
        graph.add_edge("capability_preflight", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "build_world_ledger")
        graph.add_edge("build_world_ledger", "plan_artifacts")
        graph.add_edge("plan_artifacts", "build_artifacts")
        graph.add_edge("build_artifacts", "validate_artifacts")
        graph.add_edge("validate_artifacts", "review_round_1")
        graph.add_edge("review_round_1", "repair_round_1")
        graph.add_edge("repair_round_1", "review_round_2")
        graph.add_edge("review_round_2", "repair_round_2")
        graph.add_edge("repair_round_2", "review_round_3")
        graph.add_edge("review_round_3", "final_gate")
        graph.add_conditional_edges(
            "final_gate",
            lambda state: "assemble" if state.get("release_allowed") else "end",
            {"assemble": "assemble", "end": END},
        )
        graph.add_edge("assemble", "export")
        graph.add_edge("export", END)
        return graph.compile(
            checkpointer=checkpointer,
            interrupt_after=interrupt_after,
            name="environment_mock_workflow",
        )

    @staticmethod
    def _store(state: GraphState) -> RunStore:
        store = RunStore.open(Path(state["runs_root"]), state["run_id"])
        if store.read_record().status == RunStatus.CANCELLED:
            raise RuntimeError(f"run is cancelled: {state['run_id']}")
        return store

    def normalize(self, state: GraphState) -> dict[str, object]:
        store = self._store(state)
        task = TaskSpec.model_validate(state["task"])
        store.update_status(RunStatus.PLANNING, current_node="normalize")
        return {
            "task": task.model_dump(mode="json"),
            "profile": state.get("profile", task.profile),
            "repair_attempts": state.get("repair_attempts", {}),
            "route_history": state.get("route_history", []),
        }

    def classify(self, state: GraphState) -> dict[str, object]:
        task = TaskSpec.model_validate(state["task"])
        self._store(state).update_status(RunStatus.PLANNING, current_node="classify")
        return {
            "dependencies": [dependency.model_dump(mode="json") for dependency in task.dependencies],
            "forbidden_outputs": [item.model_dump(mode="json") for item in task.forbidden_outputs],
        }

    def capability_preflight(self, state: GraphState) -> dict[str, object]:
        dependencies = [DependencySpec.model_validate(item) for item in state["dependencies"]]
        decisions = preflight_dependencies(dependencies, self.providers)
        self._store(state).update_status(
            RunStatus.PLANNING,
            current_node="capability_preflight",
        )
        return {"capabilities": [item.model_dump(mode="json") for item in decisions]}

    def retrieve_evidence(self, state: GraphState) -> dict[str, object]:
        store = self._store(state)
        store.update_status(RunStatus.RETRIEVING, current_node="retrieve_evidence")
        key = operation_key(
            state["run_id"],
            "retrieve_evidence",
            inputs=state["dependencies"],
        )
        cached = store.read_operation(key)
        if isinstance(cached, dict) and isinstance(cached.get("evidence"), list):
            return {"evidence": cached["evidence"]}
        task = TaskSpec.model_validate(state["task"])
        source_workspace = self._source_workspace(task)
        evidence: list[SourceEvidence] = []
        if source_workspace is not None:
            for dependency in task.dependencies:
                source = source_workspace / dependency.path
                if not source.exists():
                    continue
                evidence.append(
                    SourceEvidence(
                        source_id=f"local-{dependency.dependency_id}",
                        local_path=str(source.resolve()),
                        title=source.name,
                        publisher="task fixture",
                        source_type=SourceType.DIRECT_FILE,
                        sha256=sha256_path(source),
                        used_by=[dependency.dependency_id],
                        supported_dependencies=[dependency.dependency_id],
                        usage_basis="copy from source task workspace",
                    )
                )
        serialized = [item.model_dump(mode="json") for item in evidence]
        for item in evidence:
            store.append_evidence(item)
        store.write_operation(key, {"evidence": serialized})
        return {"evidence": serialized}

    def world_ledger(self, state: GraphState) -> dict[str, object]:
        store = self._store(state)
        evidence = [SourceEvidence.model_validate(item) for item in state.get("evidence", [])]
        task = TaskSpec.model_validate(state["task"])
        raw_locked = task.metadata.get("locked_facts")
        locked_facts = raw_locked if isinstance(raw_locked, dict) else {}
        ledger = build_world_ledger(evidence, locked_facts=locked_facts)
        store.write_model("world/ledger.json", ledger)
        return {"world_ledger": ledger.model_dump(mode="json")}

    def plan_artifacts(self, state: GraphState) -> dict[str, object]:
        task = TaskSpec.model_validate(state["task"])
        capabilities = {str(item["dependency_id"]): item for item in state["capabilities"]}
        force_runtime = task.metadata.get("force_runtime") is True
        plans: list[ArtifactPlan] = []
        for index, dependency in enumerate(task.dependencies, start=1):
            capability = capabilities[dependency.dependency_id]
            plans.append(
                ArtifactPlan(
                    artifact_id=f"A-{index:03d}",
                    dependency_id=dependency.dependency_id,
                    relative_path=dependency.path.rstrip("/"),
                    asset_type=dependency.asset_type,
                    provider=(
                        str(capability.get("provider"))
                        if capability.get("available") and not force_runtime
                        else None
                    ),
                    runtime_role="builder",
                    content_contract=self._content_contract(dependency),
                    validators=self._validator_names(dependency.asset_type),
                    source_evidence_ids=[
                        str(item["source_id"])
                        for item in state.get("evidence", [])
                        if dependency.dependency_id
                        in cast(list[object], item.get("supported_dependencies", []))
                    ],
                    seed=index,
                )
            )
        self._store(state).write_json(
            "task/artifact_plans.json",
            {"artifacts": [item.model_dump(mode="json") for item in plans]},
        )
        return {"artifact_plans": [item.model_dump(mode="json") for item in plans]}

    def build_artifacts(self, state: GraphState) -> dict[str, object]:
        store = self._store(state)
        store.update_status(RunStatus.BUILDING, current_node="build_artifacts")
        task = TaskSpec.model_validate(state["task"])
        dependencies = {item.dependency_id: item for item in task.dependencies}
        evidence = [SourceEvidence.model_validate(item) for item in state.get("evidence", [])]
        ledger = WorldLedger.model_validate(state.get("world_ledger", {}))
        results = dict(state.get("artifact_results", {}))
        route_history = list(state.get("route_history", []))
        for raw_plan in state["artifact_plans"]:
            plan = ArtifactPlan.model_validate(raw_plan)
            key = operation_key(
                state["run_id"],
                "build_artifact",
                artifact_id=plan.artifact_id,
                inputs=plan.model_dump(mode="json"),
            )
            cached = store.read_operation(key)
            if isinstance(cached, dict) and isinstance(cached.get("result"), dict):
                results[plan.artifact_id] = cast(dict[str, object], cached["result"])
                continue
            result, route = self._build_one(
                store,
                task,
                dependencies[plan.dependency_id],
                plan,
                evidence,
                ledger,
            )
            serialized = result.model_dump(mode="json")
            results[plan.artifact_id] = serialized
            route_history.append(route)
            store.write_json(
                f"staging/{plan.artifact_id}/result.json",
                serialized,
            )
            store.write_operation(key, {"result": serialized, "route": route})
        return {
            "artifact_results": results,
            "route_history": route_history,
        }

    def validate_artifacts(self, state: GraphState) -> dict[str, object]:
        self._store(state).update_status(
            RunStatus.VALIDATING,
            current_node="validate_artifacts",
        )
        return self._validate(state)

    def review(self, state: GraphState, round_number: int) -> dict[str, object]:
        store = self._store(state)
        store.update_status(
            RunStatus.REVIEWING,
            current_node=f"review_round_{round_number}",
        )
        findings = [ValidationFinding.model_validate(item) for item in state.get("findings", [])]
        round_findings = [item for item in findings if item.category in ROUND_CATEGORIES[round_number]]
        blockers = [item for item in round_findings if item.severity in {Severity.P0, Severity.P1}]
        score = max(
            0.0,
            100.0
            - sum(
                40 if item.severity == Severity.P0 else 20 if item.severity == Severity.P1 else 5
                for item in round_findings
            ),
        )
        result = ReviewRoundResult(
            round_number=round_number,
            name=ROUND_NAMES[round_number],
            findings=round_findings,
            score=score,
            accepted=not blockers,
            summary=(
                "deterministic review passed"
                if not blockers
                else f"{len(blockers)} release-blocking findings"
            ),
        )
        reviews = list(state.get("reviews", []))
        reviews.append(result.model_dump(mode="json"))
        problems = self._merge_problems(
            state.get("problems", []),
            round_findings,
            round_number,
        )
        ledger = ProblemLedger(problems=problems)
        store.write_model(f"validation/round_{round_number}.json", result)
        store.write_model("validation/problem_ledger.json", ledger)
        return {
            "reviews": reviews,
            "problems": [item.model_dump(mode="json") for item in problems],
        }

    def repair(self, state: GraphState, round_number: int) -> dict[str, object]:
        store = self._store(state)
        store.update_status(
            RunStatus.REPAIRING,
            current_node=f"repair_round_{round_number}",
        )
        findings = [ValidationFinding.model_validate(item) for item in state.get("findings", [])]
        target_ids = {
            item.artifact_id
            for item in findings
            if item.artifact_id
            and item.category in ROUND_CATEGORIES[round_number]
            and item.severity in {Severity.P0, Severity.P1}
        }
        if not target_ids:
            return {}
        attempts = dict(state.get("repair_attempts", {}))
        task = TaskSpec.model_validate(state["task"])
        dependencies = {item.dependency_id: item for item in task.dependencies}
        evidence = [SourceEvidence.model_validate(item) for item in state.get("evidence", [])]
        ledger = WorldLedger.model_validate(state.get("world_ledger", {}))
        results = dict(state["artifact_results"])
        route_history = list(state.get("route_history", []))
        for raw_plan in state["artifact_plans"]:
            plan = ArtifactPlan.model_validate(raw_plan)
            if plan.artifact_id not in target_ids or not plan.provider:
                continue
            attempts[plan.artifact_id] = attempts.get(plan.artifact_id, 0) + 1
            if attempts[plan.artifact_id] > 2:
                continue
            plan.seed += attempts[plan.artifact_id]
            result, route = self._build_one(
                store,
                task,
                dependencies[plan.dependency_id],
                plan,
                evidence,
                ledger,
            )
            results[plan.artifact_id] = result.model_dump(mode="json")
            route_history.append({**route, "repair_round": round_number})
        updated_state = cast(GraphState, {**state, "artifact_results": results})
        validation = self._validate(updated_state)
        current_ids = {
            str(item["finding_id"]) for item in cast(list[dict[str, object]], validation["findings"])
        }
        problems = [Problem.model_validate(item) for item in state.get("problems", [])]
        for problem in problems:
            if (
                problem.problem_id not in current_ids
                and problem.status == ProblemStatus.OPEN
                and problem.category in ROUND_CATEGORIES[round_number]
            ):
                problem.status = ProblemStatus.FIXED
                problem.fixed_round = round_number
        store.write_model("validation/problem_ledger.json", ProblemLedger(problems=problems))
        return {
            "artifact_results": results,
            "route_history": route_history,
            "repair_attempts": attempts,
            "findings": validation["findings"],
            "problems": [item.model_dump(mode="json") for item in problems],
        }

    def final_gate(self, state: GraphState) -> dict[str, object]:
        store = self._store(state)
        problems = ProblemLedger(
            problems=[Problem.model_validate(item) for item in state.get("problems", [])]
        )
        results = [ArtifactResult.model_validate(item) for item in state.get("artifact_results", {}).values()]
        blocked_capability = any(item.status == ArtifactStatus.BLOCKED_CAPABILITY for item in results)
        release_allowed = not problems.has_release_blocker and not blocked_capability
        if release_allowed:
            store.update_status(RunStatus.ASSEMBLING, current_node="final_gate")
            return {"release_allowed": True}
        status = RunStatus.BLOCKED_CAPABILITY if blocked_capability else RunStatus.FAILED
        store.update_status(
            status,
            current_node="final_gate",
            error="release gate rejected package",
        )
        return {
            "release_allowed": False,
            "final_status": status.value,
        }

    def assemble(self, state: GraphState) -> dict[str, object]:
        store = self._store(state)
        store.update_status(RunStatus.ASSEMBLING, current_node="assemble")
        results = [ArtifactResult.model_validate(item) for item in state["artifact_results"].values()]
        key = operation_key(
            state["run_id"],
            "assemble",
            inputs=[item.sha256 for item in results],
        )
        cached = store.read_operation(key)
        if isinstance(cached, dict) and isinstance(cached.get("package_path"), str):
            return {"package_path": cached["package_path"]}
        temporary = store.path / "package.tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        for result in results:
            if result.status not in {ArtifactStatus.BUILT, ArtifactStatus.VALIDATED}:
                continue
            plan = next(
                ArtifactPlan.model_validate(item)
                for item in state["artifact_plans"]
                if item["artifact_id"] == result.artifact_id
            )
            source = Path(result.path or "")
            destination = temporary / plan.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        package = store.path / "package"
        shutil.rmtree(package, ignore_errors=True)
        temporary.replace(package)
        record = store.update_status(RunStatus.PACKAGED, current_node="assemble")
        record.package_path = str(package)
        store.write_model("run.json", record)
        store.write_operation(key, {"package_path": str(package)})
        return {
            "package_path": str(package),
            "final_status": RunStatus.PACKAGED.value,
        }

    def export(self, state: GraphState) -> dict[str, object]:
        destination = state.get("export_path")
        if not destination:
            return {}
        store = self._store(state)
        task = TaskSpec.model_validate(state["task"])
        exported = get_profile(state["profile"]).export(
            store,
            task,
            Path(destination),
            overwrite=state.get("overwrite", False),
        )
        record = store.update_status(RunStatus.EXPORTED, current_node="export")
        record.export_path = str(exported)
        store.write_model("run.json", record)
        return {
            "final_status": RunStatus.EXPORTED.value,
            "export_path": str(exported),
        }

    def _validate(self, state: GraphState) -> dict[str, object]:
        store = self._store(state)
        task = TaskSpec.model_validate(state["task"])
        dependency_by_id = {item.dependency_id: item for item in task.dependencies}
        evidence = [SourceEvidence.model_validate(item) for item in state.get("evidence", [])]
        ledger = WorldLedger.model_validate(state.get("world_ledger", {}))
        results = dict(state["artifact_results"])
        findings: list[ValidationFinding] = []
        for raw_plan in state["artifact_plans"]:
            plan = ArtifactPlan.model_validate(raw_plan)
            result = ArtifactResult.model_validate(results[plan.artifact_id])
            dependency = dependency_by_id[plan.dependency_id]
            if result.status == ArtifactStatus.BLOCKED_CAPABILITY:
                findings.append(
                    ValidationFinding(
                        finding_id=f"capability-{plan.artifact_id}",
                        severity=(
                            Severity.P1
                            if dependency.criticality in {Criticality.CRITICAL, Criticality.HIGH}
                            else Severity.P2
                        ),
                        category=FindingCategory.CAPABILITY,
                        description=result.warnings[0]
                        if result.warnings
                        else "artifact capability is blocked",
                        artifact_id=plan.artifact_id,
                        path=plan.relative_path,
                        repair_action="configure a capable provider/runtime or keep the run blocked",
                    )
                )
                continue
            if result.status == ArtifactStatus.FAILED:
                findings.append(
                    ValidationFinding(
                        finding_id=f"runtime-failure-{plan.artifact_id}",
                        severity=Severity.P1,
                        category=FindingCategory.CAPABILITY,
                        description=(result.warnings[0] if result.warnings else "artifact generation failed"),
                        artifact_id=plan.artifact_id,
                        path=plan.relative_path,
                        repair_action="retry with a capable provider/runtime",
                    )
                )
                continue
            request = ValidationRequest(
                plan=plan,
                result=result,
                dependency=dependency,
                evidence=evidence,
                forbidden_outputs=task.forbidden_outputs,
                world_ledger=ledger,
            )
            artifact_findings = self.validators.validate(request, plan.validators)
            findings.extend(artifact_findings)
            has_blocker = any(item.severity in {Severity.P0, Severity.P1} for item in artifact_findings)
            result.status = ArtifactStatus.FAILED if has_blocker else ArtifactStatus.VALIDATED
            results[plan.artifact_id] = result.model_dump(mode="json")
        serialized = [item.model_dump(mode="json") for item in findings]
        store.write_json("validation/deterministic.json", {"findings": serialized})
        return {"artifact_results": results, "findings": serialized}

    def _build_one(
        self,
        store: RunStore,
        task: TaskSpec,
        dependency: DependencySpec,
        plan: ArtifactPlan,
        evidence: list[SourceEvidence],
        ledger: WorldLedger,
    ) -> tuple[ArtifactResult, dict[str, object]]:
        staging = store.staging_path(plan.artifact_id)
        source_workspace = self._source_workspace(task)
        source = source_workspace / dependency.path if source_workspace else None
        if source is not None and source.exists():
            destination = staging / plan.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            return (
                artifact_result(plan.artifact_id, destination, "copy_from_source"),
                {
                    "artifact_id": plan.artifact_id,
                    "selected": "copy_from_source",
                    "reason": "dependency exists in source task workspace",
                },
            )
        provider = self.providers.get(plan.provider) if plan.provider else None
        if provider is not None:
            result = provider.generate(
                ProviderRequest(
                    plan=plan,
                    staging_root=staging,
                    evidence=evidence,
                    world_ledger=ledger,
                )
            )
            return (
                result,
                {
                    "artifact_id": plan.artifact_id,
                    "selected": f"provider:{provider.name}",
                    "reason": "deterministic provider available",
                },
            )
        runtime_result = self._build_with_runtime(store, task, plan, staging)
        if runtime_result is not None:
            return runtime_result
        return (
            ArtifactResult(
                artifact_id=plan.artifact_id,
                status=ArtifactStatus.BLOCKED_CAPABILITY,
                attempts=0,
                warnings=[f"no provider or configured runtime can build {plan.asset_type}"],
            ),
            {
                "artifact_id": plan.artifact_id,
                "selected": None,
                "reason": "BLOCKED_CAPABILITY",
            },
        )

    def _build_with_runtime(
        self,
        store: RunStore,
        task: TaskSpec,
        plan: ArtifactPlan,
        staging: Path,
    ) -> tuple[ArtifactResult, dict[str, object]] | None:
        runtime_value = task.metadata.get("runtime_name")
        profile_value = task.metadata.get("model_profile")
        if not isinstance(runtime_value, str) or not isinstance(profile_value, dict):
            return None
        try:
            runtime_name = RuntimeName(runtime_value)
            runtime = self.runtimes.get(runtime_name)
            model_profile = ModelProfile.model_validate(profile_value)
        except (ValueError, KeyError):
            return None
        request = RuntimeRequest(
            run_id=store.run_id,
            artifact_id=plan.artifact_id,
            role=plan.runtime_role or "builder",
            workspace=str(staging),
            prompt=(
                f"Create only the input-state artifact at {plan.relative_path}. "
                f"Content contract: {plan.content_contract!r}. "
                "Do not create the task's final deliverable or conclusions."
            ),
            system_policy=("Operate only in the workspace. Generate input material, never final answers."),
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            model_profile=model_profile,
            max_turns=_int_value(task.metadata.get("max_turns"), 50),
            max_tool_events=_int_value(task.metadata.get("max_tool_events"), 300),
            timeout_seconds=_int_value(task.metadata.get("timeout_seconds"), 1800),
        )

        async def collect() -> list[RuntimeEvent]:
            return [event async for event in runtime.run(request)]

        events = asyncio.run(collect())
        for event in events:
            store.events.append(event)
        target = staging / plan.relative_path
        terminal = events[-1] if events else None
        if (
            terminal is None
            or terminal.event_type != RuntimeEventType.RUNTIME_FINISHED
            or not target.exists()
        ):
            return (
                ArtifactResult(
                    artifact_id=plan.artifact_id,
                    status=ArtifactStatus.FAILED,
                    attempts=1,
                    runtime=runtime_name.value,
                    model=model_profile.model,
                    warnings=[terminal.message if terminal is not None else "runtime emitted no events"],
                ),
                {
                    "artifact_id": plan.artifact_id,
                    "selected": runtime_name.value,
                    "reason": "runtime failed or did not write the contracted artifact",
                },
            )
        return (
            artifact_result(
                plan.artifact_id,
                target,
                runtime_name.value,
                metadata={"model": model_profile.model},
            ),
            {
                "artifact_id": plan.artifact_id,
                "selected": runtime_name.value,
                "reason": "configured open-ended runtime",
                "model": model_profile.model,
            },
        )

    @staticmethod
    def _source_workspace(task: TaskSpec) -> Path | None:
        if task.metadata.get("ignore_source_workspace") is True:
            return None
        value = task.metadata.get("workspace_path")
        if not isinstance(value, str):
            return None
        path = Path(value).expanduser().resolve()
        return path if path.is_dir() else None

    @staticmethod
    def _content_contract(dependency: DependencySpec) -> dict[str, object]:
        if isinstance(dependency.expected_content, dict):
            return dict(dependency.expected_content)
        description = dependency.expected_content or f"Input material for {dependency.path}"
        asset_type = dependency.asset_type.lower().removeprefix("file_")
        if asset_type in {"txt", "text", "md", "markdown", "html", "rtf"}:
            return {"content": description, "min_characters": 1}
        if asset_type in {"json", "yaml", "yml"}:
            return {"data": {"description": description}}
        if asset_type == "xlsx":
            return {
                "sheets": [
                    {
                        "name": "Data",
                        "rows": [["Description"], [description]],
                    }
                ]
            }
        if asset_type == "docx":
            return {
                "title": Path(dependency.path).stem,
                "paragraphs": [description],
            }
        if asset_type == "pdf":
            return {
                "title": Path(dependency.path).stem,
                "paragraphs": [description],
            }
        if asset_type in {"code", "project", "directory"}:
            return {
                "files": {"README.md": f"# Input Project\n\n{description}\n"},
                "required_files": ["README.md"],
            }
        return {"description": description}

    @staticmethod
    def _validator_names(asset_type: str) -> list[str]:
        normalized = asset_type.lower().removeprefix("file_")
        format_validator = {
            "txt": "text",
            "text": "text",
            "md": "text",
            "markdown": "text",
            "html": "text",
            "rtf": "text",
            "json": "structured",
            "yaml": "structured",
            "yml": "structured",
            "docx": "docx",
            "xlsx": "xlsx",
            "pdf": "pdf",
            "code": "code_project",
            "project": "code_project",
            "directory": "code_project",
        }.get(normalized)
        names = ["common"]
        if format_validator:
            names.append(format_validator)
        names.extend(["metadata", "secrets", "leakage", "traceability"])
        return names

    @staticmethod
    def _merge_problems(
        existing: list[dict[str, object]],
        findings: list[ValidationFinding],
        round_number: int,
    ) -> list[Problem]:
        problems = {item.problem_id: item for item in (Problem.model_validate(raw) for raw in existing)}
        for item in findings:
            if item.finding_id in problems:
                problem = problems[item.finding_id]
                problem.severity = item.severity
                problem.description = item.description
                problem.evidence = item.evidence
                problem.repair_action = item.repair_action
                problem.status = ProblemStatus.OPEN
                continue
            problems[item.finding_id] = Problem(
                problem_id=item.finding_id,
                severity=item.severity,
                category=item.category,
                description=item.description,
                evidence=item.evidence,
                repair_action=item.repair_action,
                introduced_round=round_number,
            )
        return list(problems.values())
