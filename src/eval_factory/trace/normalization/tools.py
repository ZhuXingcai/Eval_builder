from __future__ import annotations

from dataclasses import dataclass

from eval_factory.contracts.trace import ToolFamily
from eval_factory.trace.normalization.models import stable_id

MAX_EXPLICIT_ID_LENGTH = 255

_ALIASES: dict[str, ToolFamily] = {
    "agent": ToolFamily.SUBAGENT,
    "ask": ToolFamily.USER_INTERACTION,
    "askuserquestion": ToolFamily.USER_INTERACTION,
    "bash": ToolFamily.SHELL,
    "complete_step": ToolFamily.TASK_MANAGEMENT,
    "docragindex": ToolFamily.SEARCH,
    "edit": ToolFamily.FILE_EDIT,
    "edit_file": ToolFamily.FILE_EDIT,
    "exitplanmode": ToolFamily.USER_INTERACTION,
    "glob": ToolFamily.FILE_LIST,
    "grep": ToolFamily.SEARCH,
    "historysearch": ToolFamily.SEARCH,
    "ls": ToolFamily.FILE_LIST,
    "memorysearch": ToolFamily.SEARCH,
    "multi_edit": ToolFamily.FILE_EDIT,
    "powershell": ToolFamily.SHELL,
    "read": ToolFamily.FILE_READ,
    "read_file": ToolFamily.FILE_READ,
    "taskcreate": ToolFamily.TASK_MANAGEMENT,
    "tasklist": ToolFamily.TASK_MANAGEMENT,
    "taskoutput": ToolFamily.SUBAGENT,
    "taskstop": ToolFamily.SUBAGENT,
    "taskupdate": ToolFamily.TASK_MANAGEMENT,
    "todo_write": ToolFamily.TASK_MANAGEMENT,
    "todowrite": ToolFamily.TASK_MANAGEMENT,
    "webfetch": ToolFamily.FETCH,
    "websearch": ToolFamily.SEARCH,
    "write": ToolFamily.FILE_WRITE,
    "write_file": ToolFamily.FILE_WRITE,
}


@dataclass(frozen=True)
class ExplicitId:
    value: str
    valid: bool
    synthetic: bool
    reason: str | None = None


@dataclass(frozen=True)
class ResultIdFields:
    relation_id: ExplicitId
    ambiguous: bool


def classify_tool_family(name: object) -> ToolFamily:
    if not isinstance(name, str):
        return ToolFamily.UNKNOWN
    key = name.strip().casefold()
    return _ALIASES.get(key, ToolFamily.UNKNOWN)


def normalized_tool_name(name: object) -> str:
    if not isinstance(name, str):
        return "unknown"
    stripped = name.strip()
    if not stripped or len(stripped) > 255:
        return "unknown"
    return stripped


def extract_call_id(value: object, *, fallback_seed: dict[str, object]) -> ExplicitId:
    return _explicit_or_synthetic(value, fallback_seed=fallback_seed, missing_reason="missing-call-id")


def extract_result_id(
    *,
    tool_use_id: object,
    tool_call_id: object,
    fallback_seed: dict[str, object],
) -> ResultIdFields:
    first = _explicit_or_none(tool_use_id)
    second = _explicit_or_none(tool_call_id)
    if first is not None and second is not None and first.value != second.value:
        synthetic = _synthetic_id(fallback_seed)
        return ResultIdFields(
            relation_id=ExplicitId(
                value=synthetic,
                valid=False,
                synthetic=True,
                reason="conflicting-result-id-fields",
            ),
            ambiguous=True,
        )
    selected = first if first is not None else second
    if selected is not None:
        return ResultIdFields(relation_id=selected, ambiguous=False)
    return ResultIdFields(
        relation_id=_explicit_or_synthetic(
            tool_use_id if tool_use_id is not None else tool_call_id,
            fallback_seed=fallback_seed,
            missing_reason="missing-result-id",
        ),
        ambiguous=False,
    )


def _explicit_or_synthetic(
    value: object,
    *,
    fallback_seed: dict[str, object],
    missing_reason: str,
) -> ExplicitId:
    explicit = _explicit_or_none(value)
    if explicit is not None:
        return explicit
    reason = missing_reason
    if value is not None:
        reason = "invalid-explicit-id"
    return ExplicitId(
        value=_synthetic_id(fallback_seed),
        valid=False,
        synthetic=True,
        reason=reason,
    )


def _explicit_or_none(value: object) -> ExplicitId | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or len(stripped) > MAX_EXPLICIT_ID_LENGTH:
        return None
    return ExplicitId(value=stripped, valid=True, synthetic=False)


def _synthetic_id(seed: dict[str, object]) -> str:
    return stable_id("synthetic", seed)
