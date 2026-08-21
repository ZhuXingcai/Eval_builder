import inspect
import sqlite3
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
)
from contextlib import asynccontextmanager, contextmanager
from enum import StrEnum
from pathlib import Path
from typing import NotRequired, Protocol, Required, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import FactoryRunStatusV2
from eval_factory.contracts.core import ObjectRef
from eval_factory.harness.graph_models import (
    HarnessGraphExecutionBindingV1,
)


class FactoryGraphError(RuntimeError):
    pass


class FactoryGraphDriftError(FactoryGraphError):
    pass


class FactoryGraphTransitionLimitError(FactoryGraphError):
    pass


class FactoryGraphHandlerError(FactoryGraphError):
    pass


class FactoryGraphBindingResolver(Protocol):
    def resolve_current(
        self,
        reference: ObjectRef,
    ) -> HarnessGraphExecutionBindingV1: ...


class FactoryGraphSignalV2(StrEnum):
    CONTINUE = "CONTINUE"
    WAIT = "WAIT"
    RETRY = "RETRY"
    REPLAN = "REPLAN"
    ESCALATE = "ESCALATE"
    FINISH = "FINISH"
    BLOCKED = "BLOCKED"


class FactoryGraphState(TypedDict):
    factory_run_id: Required[str]
    expected_run_version: Required[int]
    transition_count: Required[int]
    signal: Required[FactoryGraphSignalV2]
    run_ref: NotRequired[ObjectRef]
    run_status: NotRequired[FactoryRunStatusV2]
    current_plan_ref: NotRequired[ObjectRef | None]
    compiled_plan_ref: NotRequired[ObjectRef | None]
    pending_review_ref: NotRequired[ObjectRef | None]
    planner_assessment_ref: NotRequired[ObjectRef | None]
    completion_ref: NotRequired[ObjectRef | None]
    delivery_manifest_ref: NotRequired[ObjectRef | None]
    graph_binding_ref: NotRequired[ObjectRef]
    graph_checkpoint_ref: NotRequired[ObjectRef | None]
    session_ref: NotRequired[ObjectRef]
    team_ref: NotRequired[ObjectRef]
    team_checkpoint_ref: NotRequired[ObjectRef]
    blueprint_ref: NotRequired[ObjectRef]
    composition_ref: NotRequired[ObjectRef]
    expected_team_version: NotRequired[int]
    expected_task_graph_revision: NotRequired[int]
    expected_authority_version: NotRequired[int]
    current_node: NotRequired[str]
    error_code: NotRequired[str]


FactoryGraphNodeUpdate = dict[str, object]
FactoryGraphNodeHandler = Callable[
    [FactoryGraphState],
    FactoryGraphNodeUpdate | Awaitable[FactoryGraphNodeUpdate],
]


class FactoryGraphTransitionJournal(Protocol):
    def begin_transition(
        self,
        *,
        node: str,
        state: FactoryGraphState,
    ) -> ObjectRef: ...

    def complete_transition(
        self,
        *,
        node: str,
        state: FactoryGraphState,
        pre_checkpoint_ref: ObjectRef,
    ) -> FactoryGraphNodeUpdate: ...


_HANDLER_UPDATE_FIELDS = frozenset(
    {
        "compiled_plan_ref",
        "completion_ref",
        "current_plan_ref",
        "delivery_manifest_ref",
        "error_code",
        "expected_authority_version",
        "expected_run_version",
        "expected_task_graph_revision",
        "expected_team_version",
        "graph_binding_ref",
        "graph_checkpoint_ref",
        "pending_review_ref",
        "planner_assessment_ref",
        "blueprint_ref",
        "composition_ref",
        "run_ref",
        "run_status",
        "session_ref",
        "signal",
        "team_checkpoint_ref",
        "team_ref",
    }
)


class FactoryControlGraph:
    def __init__(
        self,
        store: FactoryControlStore,
        *,
        handlers: Mapping[str, FactoryGraphNodeHandler] | None = None,
        transition_limit: int = 256,
        binding_resolver: FactoryGraphBindingResolver | None = None,
        transition_journal: FactoryGraphTransitionJournal | None = None,
    ) -> None:
        if transition_limit < 1 or transition_limit > 100_000:
            raise ValueError("factory graph transition limit must be between 1 and 100000")
        self._store = store
        self._handlers = dict(handlers or {})
        self._transition_limit = transition_limit
        self._binding_resolver = binding_resolver
        self._transition_journal = transition_journal
        unknown_handlers = set(self._handlers) - {
            "requirement_planner",
            "review_global_plan",
            "compile_global_plan",
            "dispatch_ready_work",
            "validate_core_vertical",
            "planner_assessment",
            "assemble_core_output",
            "review_final_delivery",
            "export_core_output",
        }
        if unknown_handlers:
            raise ValueError("factory graph contains handlers for unknown nodes")

    def compile(
        self,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        *,
        interrupt_after: list[str] | None = None,
    ) -> CompiledStateGraph[FactoryGraphState, None, FactoryGraphState, FactoryGraphState]:
        graph: StateGraph[
            FactoryGraphState,
            None,
            FactoryGraphState,
            FactoryGraphState,
        ] = StateGraph(FactoryGraphState)
        graph.add_node("load_authority", self._load_authority)
        graph.add_node("admit_request", self._admit_request)
        graph.add_node(
            "requirement_planner",
            self._handler_node("requirement_planner"),
        )
        graph.add_node(
            "review_global_plan",
            self._handler_node("review_global_plan"),
        )
        graph.add_node(
            "compile_global_plan",
            self._handler_node("compile_global_plan"),
        )
        graph.add_node(
            "dispatch_ready_work",
            self._handler_node("dispatch_ready_work"),
        )
        graph.add_node(
            "validate_core_vertical",
            self._handler_node("validate_core_vertical"),
        )
        graph.add_node(
            "planner_assessment",
            self._handler_node("planner_assessment"),
        )
        graph.add_node(
            "assemble_core_output",
            self._handler_node("assemble_core_output"),
        )
        graph.add_node(
            "review_final_delivery",
            self._handler_node("review_final_delivery"),
        )
        graph.add_node(
            "export_core_output",
            self._handler_node("export_core_output"),
        )
        graph.add_node("blocked", self._blocked)

        graph.add_edge(START, "load_authority")
        graph.add_conditional_edges(
            "load_authority",
            self._route_after_load,
            {
                "admit": "admit_request",
                "review": "review_global_plan",
                "dispatch": "dispatch_ready_work",
                "blocked": "blocked",
                "end": END,
            },
        )
        graph.add_edge("admit_request", "requirement_planner")
        graph.add_conditional_edges(
            "requirement_planner",
            self._route_standard,
            {
                "continue": "review_global_plan",
                "replan": "requirement_planner",
                "blocked": "blocked",
                "wait": END,
            },
        )
        graph.add_conditional_edges(
            "review_global_plan",
            self._route_standard,
            {
                "continue": "compile_global_plan",
                "replan": "requirement_planner",
                "blocked": "blocked",
                "wait": END,
            },
        )
        graph.add_conditional_edges(
            "compile_global_plan",
            self._route_standard,
            {
                "continue": "dispatch_ready_work",
                "replan": "requirement_planner",
                "blocked": "blocked",
                "wait": END,
            },
        )
        graph.add_conditional_edges(
            "dispatch_ready_work",
            self._route_standard,
            {
                "continue": "validate_core_vertical",
                "replan": "requirement_planner",
                "blocked": "blocked",
                "wait": END,
            },
        )
        graph.add_conditional_edges(
            "validate_core_vertical",
            self._route_validation,
            {
                "continue": "planner_assessment",
                "retry": "dispatch_ready_work",
                "replan": "requirement_planner",
                "blocked": "blocked",
                "wait": END,
            },
        )
        graph.add_conditional_edges(
            "planner_assessment",
            self._route_assessment,
            {
                "continue": "dispatch_ready_work",
                "retry": "dispatch_ready_work",
                "replan": "requirement_planner",
                "escalate": END,
                "finish": "assemble_core_output",
                "blocked": "blocked",
                "wait": END,
            },
        )
        graph.add_conditional_edges(
            "assemble_core_output",
            self._route_standard,
            {
                "continue": "review_final_delivery",
                "replan": "requirement_planner",
                "blocked": "blocked",
                "wait": END,
            },
        )
        graph.add_conditional_edges(
            "review_final_delivery",
            self._route_standard,
            {
                "continue": "export_core_output",
                "replan": "assemble_core_output",
                "blocked": "blocked",
                "wait": END,
            },
        )
        graph.add_edge("export_core_output", END)
        graph.add_edge("blocked", END)
        return graph.compile(
            checkpointer=checkpointer,
            interrupt_after=interrupt_after,
            name="eval_factory_control_graph",
        )

    def _load_authority(self, state: FactoryGraphState) -> FactoryGraphNodeUpdate:
        update = self._synchronize(state, node="load_authority")
        update["signal"] = FactoryGraphSignalV2.CONTINUE
        return update

    def _admit_request(self, state: FactoryGraphState) -> FactoryGraphNodeUpdate:
        update = self._synchronize(state, node="admit_request")
        update["signal"] = FactoryGraphSignalV2.CONTINUE
        return update

    def _controlled_node(
        self,
        node: str,
        state: FactoryGraphState,
    ) -> FactoryGraphNodeUpdate:
        synchronized = self._synchronize(state, node=node)
        handler = self._handlers.get(node)
        if handler is None:
            synchronized["signal"] = FactoryGraphSignalV2.WAIT
            return synchronized
        pre_checkpoint_ref = self._begin_transition(
            node=node,
            state=synchronized,
        )
        handler_state = dict(state)
        handler_state.update(synchronized)
        update = handler(handler_state)  # type: ignore[arg-type]
        if inspect.isawaitable(update):
            if inspect.iscoroutine(update):
                update.close()
            raise FactoryGraphHandlerError("async factory graph handler requires ainvoke")
        merged = self._merge_handler_update(
            synchronized,
            update,
        )
        return self._complete_transition(
            node=node,
            state=merged,
            pre_checkpoint_ref=pre_checkpoint_ref,
        )

    async def _controlled_node_async(
        self,
        node: str,
        state: FactoryGraphState,
    ) -> FactoryGraphNodeUpdate:
        synchronized = self._synchronize(state, node=node)
        handler = self._handlers.get(node)
        if handler is None:
            synchronized["signal"] = FactoryGraphSignalV2.WAIT
            return synchronized
        pre_checkpoint_ref = self._begin_transition(
            node=node,
            state=synchronized,
        )
        handler_state = dict(state)
        handler_state.update(synchronized)
        update = handler(handler_state)  # type: ignore[arg-type]
        if inspect.isawaitable(update):
            update = await update
        merged = self._merge_handler_update(
            synchronized,
            update,
        )
        return self._complete_transition(
            node=node,
            state=merged,
            pre_checkpoint_ref=pre_checkpoint_ref,
        )

    def _handler_node(
        self,
        node: str,
    ) -> RunnableLambda[
        FactoryGraphState,
        FactoryGraphNodeUpdate,
    ]:
        async def invoke_async(
            state: FactoryGraphState,
        ) -> FactoryGraphNodeUpdate:
            return await self._controlled_node_async(
                node,
                state,
            )

        def invoke_sync(
            state: FactoryGraphState,
        ) -> FactoryGraphNodeUpdate:
            return self._controlled_node(node, state)

        return RunnableLambda(
            invoke_sync,
            afunc=invoke_async,
        )

    @staticmethod
    def _merge_handler_update(
        synchronized: FactoryGraphNodeUpdate,
        update: FactoryGraphNodeUpdate,
    ) -> FactoryGraphNodeUpdate:
        unknown_fields = set(update) - _HANDLER_UPDATE_FIELDS
        if unknown_fields:
            raise FactoryGraphHandlerError("factory graph handler returned forbidden state fields")
        signal = update.get("signal")
        if not isinstance(signal, FactoryGraphSignalV2):
            raise FactoryGraphHandlerError("factory graph handler must return a closed signal")
        synchronized.update(update)
        return synchronized

    def _begin_transition(
        self,
        *,
        node: str,
        state: FactoryGraphNodeUpdate,
    ) -> ObjectRef | None:
        if self._transition_journal is None:
            return None
        return self._transition_journal.begin_transition(
            node=node,
            state=state,  # type: ignore[arg-type]
        )

    def _complete_transition(
        self,
        *,
        node: str,
        state: FactoryGraphNodeUpdate,
        pre_checkpoint_ref: ObjectRef | None,
    ) -> FactoryGraphNodeUpdate:
        if self._transition_journal is None:
            return state
        if pre_checkpoint_ref is None:
            raise FactoryGraphHandlerError(
                "Graph transition journal has no pre checkpoint",
            )
        journal_update = self._transition_journal.complete_transition(
            node=node,
            state=state,  # type: ignore[arg-type]
            pre_checkpoint_ref=pre_checkpoint_ref,
        )
        unknown_fields = set(journal_update) - _HANDLER_UPDATE_FIELDS
        if unknown_fields:
            raise FactoryGraphHandlerError(
                "Graph transition journal returned forbidden state fields",
            )
        state.update(journal_update)
        return state

    def _blocked(self, state: FactoryGraphState) -> FactoryGraphNodeUpdate:
        update = self._synchronize(state, node="blocked")
        update["signal"] = FactoryGraphSignalV2.WAIT
        return update

    def _synchronize(
        self,
        state: FactoryGraphState,
        *,
        node: str,
    ) -> FactoryGraphNodeUpdate:
        transition_count = state["transition_count"] + 1
        if transition_count > self._transition_limit:
            raise FactoryGraphTransitionLimitError("factory graph transition limit exceeded")
        run = self._store.get_run(state["factory_run_id"])
        if run.run_version != state["expected_run_version"]:
            raise FactoryGraphDriftError(
                f"factory run version drift: expected {state['expected_run_version']}, "
                f"observed {run.run_version}"
            )
        if "run_ref" in state and state["run_ref"] != run.to_ref():
            raise FactoryGraphDriftError("factory run ref differs from store authority")
        binding_update = self._synchronize_binding(
            state,
            run_ref=run.to_ref(),
            run_version=run.run_version,
        )
        state_values: Mapping[str, object] = state
        for field_name, observed in (
            ("current_plan_ref", run.current_plan_ref),
            ("compiled_plan_ref", run.compiled_plan_ref),
            ("pending_review_ref", run.pending_review_ref),
            ("planner_assessment_ref", run.planner_assessment_ref),
            ("completion_ref", run.completion_ref),
            ("delivery_manifest_ref", run.delivery_manifest_ref),
        ):
            if field_name in state_values and state_values[field_name] != observed:
                raise FactoryGraphDriftError(f"{field_name} differs from factory control store authority")
        return {
            **binding_update,
            "compiled_plan_ref": run.compiled_plan_ref,
            "completion_ref": run.completion_ref,
            "current_node": node,
            "current_plan_ref": run.current_plan_ref,
            "delivery_manifest_ref": run.delivery_manifest_ref,
            "expected_run_version": run.run_version,
            "pending_review_ref": run.pending_review_ref,
            "planner_assessment_ref": run.planner_assessment_ref,
            "run_ref": run.to_ref(),
            "run_status": run.status,
            "transition_count": transition_count,
        }

    def _synchronize_binding(
        self,
        state: FactoryGraphState,
        *,
        run_ref: ObjectRef,
        run_version: int,
    ) -> FactoryGraphNodeUpdate:
        reference = state.get("graph_binding_ref")
        if reference is None:
            return {}
        if self._binding_resolver is None:
            raise FactoryGraphDriftError(
                "graph binding state requires an authority resolver",
            )
        binding = self._binding_resolver.resolve_current(reference)
        if binding.to_ref() != reference:
            raise FactoryGraphDriftError(
                "graph execution binding is not current",
            )
        expected = (
            ("session_ref", binding.session_ref),
            ("team_ref", binding.team_ref),
            ("team_checkpoint_ref", binding.team_checkpoint_ref),
            ("blueprint_ref", binding.blueprint_ref),
            ("composition_ref", binding.composition_ref),
            (
                "expected_team_version",
                binding.expected_team_version,
            ),
            (
                "expected_task_graph_revision",
                binding.expected_task_graph_revision,
            ),
            (
                "expected_authority_version",
                binding.expected_authority_version,
            ),
        )
        state_values: Mapping[str, object] = state
        for field_name, observed in expected:
            if field_name in state_values and state_values[field_name] != observed:
                raise FactoryGraphDriftError(
                    f"{field_name} differs from graph binding authority",
                )
        if binding.factory_run_ref != run_ref or binding.expected_factory_run_version != run_version:
            raise FactoryGraphDriftError(
                "factory run differs from graph binding authority",
            )
        return {
            "blueprint_ref": binding.blueprint_ref,
            "composition_ref": binding.composition_ref,
            "expected_authority_version": (binding.expected_authority_version),
            "expected_task_graph_revision": (binding.expected_task_graph_revision),
            "expected_team_version": binding.expected_team_version,
            "graph_binding_ref": binding.to_ref(),
            "session_ref": binding.session_ref,
            "team_checkpoint_ref": binding.team_checkpoint_ref,
            "team_ref": binding.team_ref,
        }

    @staticmethod
    def _route_after_load(state: FactoryGraphState) -> str:
        status = state["run_status"]
        if status in {
            FactoryRunStatusV2.COMPLETED,
            FactoryRunStatusV2.FAILED,
            FactoryRunStatusV2.CANCELLED,
        }:
            return "end"
        if status is FactoryRunStatusV2.BLOCKED:
            return "blocked"
        if status is FactoryRunStatusV2.WAITING_REVIEW:
            return "review"
        if status is FactoryRunStatusV2.RUNNING:
            return "dispatch"
        return "review" if state.get("compiled_plan_ref") is not None else "admit"

    @staticmethod
    def _route_standard(state: FactoryGraphState) -> str:
        signal = state["signal"]
        if signal is FactoryGraphSignalV2.CONTINUE:
            return "continue"
        if signal is FactoryGraphSignalV2.REPLAN:
            return "replan"
        if signal is FactoryGraphSignalV2.BLOCKED:
            return "blocked"
        return "wait"

    @staticmethod
    def _route_validation(state: FactoryGraphState) -> str:
        if state["signal"] is FactoryGraphSignalV2.RETRY:
            return "retry"
        return FactoryControlGraph._route_standard(state)

    @staticmethod
    def _route_assessment(state: FactoryGraphState) -> str:
        signal = state["signal"]
        routes = {
            FactoryGraphSignalV2.CONTINUE: "continue",
            FactoryGraphSignalV2.RETRY: "retry",
            FactoryGraphSignalV2.REPLAN: "replan",
            FactoryGraphSignalV2.ESCALATE: "escalate",
            FactoryGraphSignalV2.FINISH: "finish",
            FactoryGraphSignalV2.BLOCKED: "blocked",
            FactoryGraphSignalV2.WAIT: "wait",
        }
        return routes[signal]


@contextmanager
def factory_graph_checkpointer(
    path: Path,
    *,
    control_store_path: Path,
) -> Iterator[SqliteSaver]:
    checkpoint_path = path.expanduser().resolve()
    authority_path = control_store_path.expanduser().resolve()
    if checkpoint_path == authority_path:
        raise ValueError("LangGraph checkpoint and FactoryControlStore require separate paths")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    try:
        yield SqliteSaver(connection)
    finally:
        connection.close()


@asynccontextmanager
async def async_factory_graph_checkpointer(
    path: Path,
    *,
    control_store_path: Path,
) -> AsyncIterator[AsyncSqliteSaver]:
    checkpoint_path = path.expanduser().resolve()
    authority_path = control_store_path.expanduser().resolve()
    if checkpoint_path == authority_path:
        raise ValueError(
            "LangGraph checkpoint and FactoryControlStore require separate paths",
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(
        str(checkpoint_path),
    ) as saver:
        yield saver


__all__ = [
    "FactoryControlGraph",
    "FactoryGraphBindingResolver",
    "FactoryGraphDriftError",
    "FactoryGraphError",
    "FactoryGraphHandlerError",
    "FactoryGraphNodeHandler",
    "FactoryGraphSignalV2",
    "FactoryGraphState",
    "FactoryGraphTransitionJournal",
    "FactoryGraphTransitionLimitError",
    "async_factory_graph_checkpointer",
    "factory_graph_checkpointer",
]
