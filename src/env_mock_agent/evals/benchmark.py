from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, Field

from env_mock_agent.adapters import load_tasks
from env_mock_agent.graph import EnvironmentMockWorkflow
from env_mock_agent.runtimes import RuntimeRegistry
from env_mock_agent.schemas import (
    InputAdapterType,
    ModelProfile,
    RunRecord,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeName,
    RuntimeRequest,
)
from env_mock_agent.store import RunStore, sqlite_checkpointer


class BenchmarkCase(BaseModel):
    task_id: str
    path: str


class BenchmarkRuntime(BaseModel):
    name: str
    runtime: RuntimeName
    model_profile: ModelProfile
    required_env: list[str] = Field(default_factory=list)


class BenchmarkSuite(BaseModel):
    suite_id: str
    cases: list[BenchmarkCase]
    runtimes: list[BenchmarkRuntime]
    repetitions: int = Field(default=3, ge=1)
    timeout_seconds: int = Field(default=5400, ge=30)
    max_tool_events: int = Field(default=300, ge=1)
    include_ark_subset: bool = True

    @classmethod
    def load(cls, path: Path) -> BenchmarkSuite:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(value)


class PreflightResult(BaseModel):
    runtime_name: str
    ready: bool
    reason: str = ""
    evidence: dict[str, object] = Field(default_factory=dict)


class TrialResult(BaseModel):
    trial_id: str
    suite_id: str
    task_id: str
    runtime_name: str
    repetition: int
    status: str
    reason: str = ""
    run_id: str | None = None
    duration_seconds: float = 0
    final_status: str | None = None
    package_path: str | None = None
    p0_count: int = 0
    p1_count: int = 0
    cost_usd: float = 0
    tool_calls: int = 0
    preflight_evidence: dict[str, object] = Field(default_factory=dict)


class BenchmarkRunner:
    def __init__(
        self,
        suite: BenchmarkSuite,
        output_root: Path,
        *,
        runtime_registry: RuntimeRegistry | None = None,
    ) -> None:
        self.suite = suite
        self.output_root = output_root.expanduser().resolve()
        self.runtime_registry = runtime_registry or RuntimeRegistry.default()
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        preflight = {runtime.name: self._preflight(runtime) for runtime in self.suite.runtimes}
        trials: list[TrialResult] = []
        for case in self.suite.cases:
            for runtime in self.suite.runtimes:
                readiness = preflight[runtime.name]
                for repetition in range(1, self.suite.repetitions + 1):
                    if not readiness.ready:
                        trials.append(
                            self._blocked_trial(
                                case,
                                runtime,
                                repetition,
                                readiness,
                            )
                        )
                        continue
                    trials.append(self._run_trial(case, runtime, repetition, readiness))
        if self.suite.include_ark_subset:
            ark_readiness, ark_trials = self._ark_subset_trials()
            preflight[ark_readiness.runtime_name] = ark_readiness
            trials.extend(ark_trials)
        self._write_results(preflight, trials)
        return self.output_root

    def _preflight(self, config: BenchmarkRuntime) -> PreflightResult:
        missing = [name for name in config.required_env if not os.environ.get(name)]
        if missing:
            return PreflightResult(
                runtime_name=config.name,
                ready=False,
                reason=f"missing required credentials: {', '.join(missing)}",
                evidence={"missing_env": missing},
            )
        runtime = self.runtime_registry.get(config.runtime)
        capabilities = asyncio_run(runtime.probe())
        if not capabilities.available:
            return PreflightResult(
                runtime_name=config.name,
                ready=False,
                reason=capabilities.reason or "runtime unavailable",
                evidence=capabilities.model_dump(mode="json"),
            )
        with tempfile.TemporaryDirectory(prefix=f"envmock-preflight-{config.name}-") as temporary:
            request = RuntimeRequest(
                run_id=f"preflight-{config.name}",
                artifact_id="credential-check",
                role="preflight",
                workspace=temporary,
                prompt="Reply with READY only. Do not create files or use tools.",
                allowed_tools=[],
                disallowed_tools=["Bash", "Write", "Edit", "WebSearch", "WebFetch"],
                model_profile=config.model_profile,
                max_turns=2,
                max_tool_events=2,
                timeout_seconds=30,
            )
            events = asyncio_run(collect_events(runtime, request))
        terminal = events[-1] if events else None
        if terminal is None or terminal.event_type != RuntimeEventType.RUNTIME_FINISHED:
            return PreflightResult(
                runtime_name=config.name,
                ready=False,
                reason=terminal.message if terminal is not None else "runtime emitted no terminal event",
                evidence=(terminal.model_dump(mode="json") if terminal is not None else {"events": 0}),
            )
        return PreflightResult(
            runtime_name=config.name,
            ready=True,
            evidence={
                "runtime_version": capabilities.version,
                "session": (
                    terminal.session.model_dump(mode="json") if terminal.session is not None else None
                ),
            },
        )

    def _run_trial(
        self,
        case: BenchmarkCase,
        runtime: BenchmarkRuntime,
        repetition: int,
        readiness: PreflightResult,
        *,
        dependency_ids: set[str] | None = None,
    ) -> TrialResult:
        suffix = "-subset" if dependency_ids is not None else ""
        trial_id = f"{case.task_id}-{runtime.name}-r{repetition}{suffix}"
        started = time.monotonic()
        task = load_tasks(InputAdapterType.LH, Path(case.path))[0]
        if dependency_ids is not None:
            task.dependencies = [item for item in task.dependencies if item.dependency_id in dependency_ids]
        task.metadata["runtime_name"] = runtime.runtime.value
        task.metadata["model_profile"] = runtime.model_profile.model_dump(mode="json")
        task.metadata["force_runtime"] = True
        task.metadata["ignore_source_workspace"] = True
        task.metadata["timeout_seconds"] = self.suite.timeout_seconds
        task.metadata["max_tool_events"] = self.suite.max_tool_events
        runs_root = self.output_root / "runs"
        store = RunStore(runs_root, trial_id)
        store.initialize(
            RunRecord(
                run_id=trial_id,
                task_id=task.task_id,
                profile="lh",
            ),
            task,
        )
        state: dict[str, object] = {
            "run_id": trial_id,
            "runs_root": str(runs_root),
            "profile": "lh",
            "task": task.model_dump(mode="json"),
            "overwrite": False,
        }
        try:
            with sqlite_checkpointer(runs_root / ".checkpoints.sqlite") as checkpointer:
                graph = cast(Any, EnvironmentMockWorkflow().compile(checkpointer))
                result = graph.invoke(
                    state,
                    {"configurable": {"thread_id": trial_id}},
                )
            return self._completed_trial(
                trial_id,
                case,
                runtime,
                repetition,
                readiness,
                store,
                result,
                time.monotonic() - started,
            )
        except Exception as exc:
            return TrialResult(
                trial_id=trial_id,
                suite_id=self.suite.suite_id,
                task_id=case.task_id,
                runtime_name=runtime.name,
                repetition=repetition,
                status="failed",
                reason=str(exc),
                run_id=trial_id,
                duration_seconds=time.monotonic() - started,
                final_status=store.read_record().status.value,
                preflight_evidence=readiness.evidence,
            )

    def _completed_trial(
        self,
        trial_id: str,
        case: BenchmarkCase,
        runtime: BenchmarkRuntime,
        repetition: int,
        readiness: PreflightResult,
        store: RunStore,
        result: dict[str, object],
        duration: float,
    ) -> TrialResult:
        problems_raw = store.read_json("validation/problem_ledger.json")
        problems = problems_raw.get("problems", []) if isinstance(problems_raw, dict) else []
        p0_count = sum(
            isinstance(item, dict) and item.get("severity") == "P0" and item.get("status") == "open"
            for item in problems
        )
        p1_count = sum(
            isinstance(item, dict) and item.get("severity") == "P1" and item.get("status") == "open"
            for item in problems
        )
        usage = [
            item for item in store.events.read() if item.get("event_type") == RuntimeEventType.USAGE.value
        ]
        return TrialResult(
            trial_id=trial_id,
            suite_id=self.suite.suite_id,
            task_id=case.task_id,
            runtime_name=runtime.name,
            repetition=repetition,
            status="completed",
            run_id=trial_id,
            duration_seconds=duration,
            final_status=str(result.get("final_status") or store.read_record().status.value),
            package_path=cast(str | None, result.get("package_path")),
            p0_count=p0_count,
            p1_count=p1_count,
            cost_usd=sum(self._usage_float(item, "cost_usd") for item in usage),
            tool_calls=sum(self._usage_int(item, "tool_calls") for item in usage),
            preflight_evidence=readiness.evidence,
        )

    def _blocked_trial(
        self,
        case: BenchmarkCase,
        runtime: BenchmarkRuntime,
        repetition: int,
        readiness: PreflightResult,
    ) -> TrialResult:
        return TrialResult(
            trial_id=f"{case.task_id}-{runtime.name}-r{repetition}",
            suite_id=self.suite.suite_id,
            task_id=case.task_id,
            runtime_name=runtime.name,
            repetition=repetition,
            status="blocked_preflight",
            reason=readiness.reason,
            preflight_evidence=readiness.evidence,
        )

    def _ark_subset_trials(self) -> tuple[PreflightResult, list[TrialResult]]:
        runtime = BenchmarkRuntime(
            name="pi_ark_glm",
            runtime=RuntimeName.PI_RPC,
            model_profile=ModelProfile(
                provider="ark",
                model="glm-5-2-260617",
                thinking="low",
            ),
            required_env=["ARK_API_KEY"],
        )
        readiness = self._preflight(runtime)
        dependency_ids = {
            "LH_023": {"D-001", "D-002", "D-003"},
            "LH_067": {"D-002", "D-003", "D-005"},
        }
        trials: list[TrialResult] = []
        for case in self.suite.cases:
            if not readiness.ready:
                trials.append(self._blocked_trial(case, runtime, 1, readiness))
                continue
            trials.append(
                self._run_trial(
                    case,
                    runtime,
                    1,
                    readiness,
                    dependency_ids=dependency_ids.get(case.task_id, set()),
                )
            )
        return readiness, trials

    def _write_results(
        self,
        preflight: dict[str, PreflightResult],
        trials: list[TrialResult],
    ) -> None:
        manifest = {
            "suite": self.suite.model_dump(mode="json"),
            "generated_at": datetime.now(UTC).isoformat(),
            "preflight": {name: result.model_dump(mode="json") for name, result in preflight.items()},
            "trials": [item.model_dump(mode="json") for item in trials],
        }
        (self.output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (self.output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            for item in trials:
                handle.write(item.model_dump_json() + "\n")
        completed = [item for item in trials if item.status == "completed"]
        blocked = [item for item in trials if item.status == "blocked_preflight"]
        report = [
            f"# Benchmark Report: {self.suite.suite_id}",
            "",
            f"- Generated: {datetime.now(UTC).isoformat()}",
            f"- Planned main trials: {len(self.suite.cases) * len(self.suite.runtimes) * self.suite.repetitions}",
            f"- Completed: {len(completed)}",
            f"- Blocked in preflight: {len(blocked)}",
            "",
            "## Preflight",
            "",
            "| Runtime | Ready | Reason |",
            "|---|---:|---|",
            *[
                f"| {name} | {'yes' if result.ready else 'no'} | {result.reason or '-'} |"
                for name, result in preflight.items()
            ],
            "",
            "## Interpretation",
            "",
            (
                "No harness-quality conclusion is valid until all compared runtimes pass credential "
                "preflight with the same model. Blocked trials are explicit failure evidence, not "
                "successful benchmark runs."
                if not completed
                else "Completed trials contain package, quality, usage, and duration evidence."
            ),
            "",
        ]
        (self.output_root / "report.md").write_text(
            "\n".join(report),
            encoding="utf-8",
        )

    @staticmethod
    def _usage_float(item: dict[str, object], key: str) -> float:
        usage = item.get("usage")
        if not isinstance(usage, dict):
            return 0.0
        value = usage.get(key)
        return float(value) if isinstance(value, int | float) else 0.0

    @staticmethod
    def _usage_int(item: dict[str, object], key: str) -> int:
        usage = item.get("usage")
        if not isinstance(usage, dict):
            return 0
        value = usage.get(key)
        return int(value) if isinstance(value, int | float) else 0


async def collect_events(runtime: Any, request: RuntimeRequest) -> list[RuntimeEvent]:
    return [event async for event in runtime.run(request)]


def asyncio_run(value: Any) -> Any:
    import asyncio

    return asyncio.run(value)
