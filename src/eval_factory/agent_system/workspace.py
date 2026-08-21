from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from eval_factory.contracts.agent_system_v2 import (
    AgentWorkspaceReceiptV2,
    AgentWorkspaceStateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import canonical_value_v2

_MARKER = ".agent-workspace-authority.json"


class WorkspaceLease(Protocol):
    @property
    def agent_task_ref(self) -> ObjectRef: ...

    @property
    def agent_definition_ref(self) -> ObjectRef: ...

    @property
    def attempt(self) -> int: ...

    @property
    def fencing_token(self) -> int: ...


class AgentWorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AgentWorkspaceHandle:
    path: Path
    receipt: AgentWorkspaceReceiptV2


class AgentWorkspaceManager:
    def __init__(self, root: Path) -> None:
        candidate = root.expanduser().absolute()
        if candidate.is_symlink():
            raise AgentWorkspaceError("workspace root cannot be a symlink")
        if candidate.exists() and not candidate.is_dir():
            raise AgentWorkspaceError("workspace root must be a directory")
        candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = candidate.resolve()
        self._workspaces = self.root / "workspaces"
        self._quarantine = self.root / "quarantine"
        self._receipts = self.root / ".authority" / "receipts"
        self._namespace_heads = self.root / ".authority" / "namespaces"
        for directory in (
            self._workspaces,
            self._quarantine,
            self._receipts,
            self._namespace_heads,
        ):
            if directory.is_symlink():
                raise AgentWorkspaceError("workspace authority directory cannot be a symlink")
            directory.mkdir(
                parents=True,
                exist_ok=True,
                mode=0o700,
            )

    def open(
        self,
        *,
        run_ref: ObjectRef,
        lease: WorkspaceLease,
        audit: ContractAudit,
    ) -> AgentWorkspaceHandle:
        namespace_sha256 = _namespace_sha256(
            run_ref,
            lease,
        )
        terminal = self._terminal_receipt(namespace_sha256)
        if terminal is not None:
            raise AgentWorkspaceError("workspace namespace is already terminal")
        path = self._workspace_path(run_ref, lease)
        active = AgentWorkspaceReceiptV2.create(
            workspace_receipt_id=(f"agent-workspace-receipt://{namespace_sha256}/active"),
            run_ref=run_ref,
            task_ref=lease.agent_task_ref,
            agent_definition_ref=lease.agent_definition_ref,
            attempt=lease.attempt,
            fencing_token=lease.fencing_token,
            namespace_sha256=namespace_sha256,
            state=AgentWorkspaceStateV2.ACTIVE,
            predecessor_workspace_receipt_ref=None,
            inventory_sha256=None,
            reason_code=None,
            audit=audit,
        )
        marker = path / _MARKER
        if path.exists():
            if path.is_symlink() or not path.is_dir():
                raise AgentWorkspaceError("workspace namespace path is unsafe")
            stored = self._read_receipt(marker)
            if stored.to_ref() != active.to_ref() or stored.state is not AgentWorkspaceStateV2.ACTIVE:
                raise AgentWorkspaceError("workspace namespace authority conflicts")
            return AgentWorkspaceHandle(
                path=path,
                receipt=stored,
            )
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
        self._assert_under_root(path)
        self._persist_receipt(active)
        _write_once(marker, active.canonical_json())
        return AgentWorkspaceHandle(path=path, receipt=active)

    def commit(
        self,
        handle: AgentWorkspaceHandle,
        *,
        audit: ContractAudit,
    ) -> AgentWorkspaceReceiptV2:
        terminal = self._terminal_receipt(handle.receipt.namespace_sha256)
        if terminal is not None:
            if terminal.state is AgentWorkspaceStateV2.COMMITTED:
                return terminal
            raise AgentWorkspaceError("quarantined workspace cannot be committed")
        self._validate_active_handle(handle)
        inventory_sha256 = self._inventory_sha256(handle.path)
        committed = AgentWorkspaceReceiptV2.create(
            workspace_receipt_id=(f"agent-workspace-receipt://{handle.receipt.namespace_sha256}/committed"),
            run_ref=handle.receipt.run_ref,
            task_ref=handle.receipt.task_ref,
            agent_definition_ref=(handle.receipt.agent_definition_ref),
            attempt=handle.receipt.attempt,
            fencing_token=handle.receipt.fencing_token,
            namespace_sha256=handle.receipt.namespace_sha256,
            state=AgentWorkspaceStateV2.COMMITTED,
            predecessor_workspace_receipt_ref=(handle.receipt.to_ref()),
            inventory_sha256=inventory_sha256,
            reason_code=None,
            audit=audit,
        )
        self._persist_terminal(committed)
        return committed

    def quarantine(
        self,
        handle: AgentWorkspaceHandle,
        *,
        reason_code: str,
        audit: ContractAudit,
    ) -> AgentWorkspaceReceiptV2:
        terminal = self._terminal_receipt(handle.receipt.namespace_sha256)
        if terminal is not None:
            if terminal.state is AgentWorkspaceStateV2.QUARANTINED:
                return terminal
            raise AgentWorkspaceError("committed workspace cannot be quarantined")
        self._validate_active_receipt(handle.receipt)
        destination = self._quarantine / handle.receipt.namespace_sha256
        if handle.path.exists():
            if handle.path.is_symlink() or not handle.path.is_dir():
                raise AgentWorkspaceError("workspace namespace path is unsafe")
            if destination.exists():
                raise AgentWorkspaceError("workspace quarantine destination conflicts")
            os.replace(handle.path, destination)
        elif not destination.is_dir():
            raise AgentWorkspaceError("workspace namespace is missing")
        quarantined = AgentWorkspaceReceiptV2.create(
            workspace_receipt_id=(f"agent-workspace-receipt://{handle.receipt.namespace_sha256}/quarantined"),
            run_ref=handle.receipt.run_ref,
            task_ref=handle.receipt.task_ref,
            agent_definition_ref=(handle.receipt.agent_definition_ref),
            attempt=handle.receipt.attempt,
            fencing_token=handle.receipt.fencing_token,
            namespace_sha256=handle.receipt.namespace_sha256,
            state=AgentWorkspaceStateV2.QUARANTINED,
            predecessor_workspace_receipt_ref=(handle.receipt.to_ref()),
            inventory_sha256=None,
            reason_code=reason_code,
            audit=audit,
        )
        self._persist_terminal(quarantined)
        return quarantined

    def quarantine_lease(
        self,
        *,
        run_ref: ObjectRef,
        lease: WorkspaceLease,
        reason_code: str,
        audit: ContractAudit,
    ) -> AgentWorkspaceReceiptV2:
        namespace_sha256 = _namespace_sha256(run_ref, lease)
        terminal = self._terminal_receipt(namespace_sha256)
        if terminal is not None:
            if (
                terminal.state is AgentWorkspaceStateV2.QUARANTINED
                and terminal.run_ref == run_ref
                and terminal.task_ref == lease.agent_task_ref
                and terminal.agent_definition_ref == lease.agent_definition_ref
                and terminal.attempt == lease.attempt
                and terminal.fencing_token == lease.fencing_token
                and terminal.reason_code == reason_code
            ):
                return terminal
            raise AgentWorkspaceError("workspace terminal authority conflicts")
        path = self._workspace_path(run_ref, lease)
        if path.is_dir() and not path.is_symlink():
            active = self._read_receipt(path / _MARKER)
            handle = AgentWorkspaceHandle(
                path=path,
                receipt=active,
            )
        else:
            handle = self.open(
                run_ref=run_ref,
                lease=lease,
                audit=audit,
            )
        return self.quarantine(
            handle,
            reason_code=reason_code,
            audit=audit,
        )

    def resolve(
        self,
        reference: ObjectRef,
    ) -> AgentWorkspaceReceiptV2:
        if reference.object_type != "agent-workspace-receipt" or reference.object_version != "v2":
            raise AgentWorkspaceError("workspace receipt ref type is invalid")
        path = self._receipt_path(reference.object_sha256)
        receipt = self._read_receipt(path)
        if receipt.to_ref() != reference:
            raise AgentWorkspaceError("workspace receipt ref drifted")
        return receipt

    def recover(
        self,
        *,
        run_ref: ObjectRef,
        lease: WorkspaceLease,
        audit: ContractAudit,
    ) -> AgentWorkspaceHandle:
        namespace_sha256 = _namespace_sha256(
            run_ref,
            lease,
        )
        terminal = self._terminal_receipt(namespace_sha256)
        if terminal is None:
            return self.open(
                run_ref=run_ref,
                lease=lease,
                audit=audit,
            )
        if (
            terminal.state is not AgentWorkspaceStateV2.COMMITTED
            or terminal.run_ref != run_ref
            or terminal.task_ref != lease.agent_task_ref
            or terminal.agent_definition_ref != lease.agent_definition_ref
            or terminal.attempt != lease.attempt
            or terminal.fencing_token != lease.fencing_token
        ):
            raise AgentWorkspaceError("workspace terminal authority conflicts")
        path = self._workspace_path(run_ref, lease)
        self._assert_under_root(path)
        if path.is_symlink() or not path.is_dir():
            raise AgentWorkspaceError("committed workspace namespace is missing")
        marker = self._read_receipt(path / _MARKER)
        if (
            marker.state is not AgentWorkspaceStateV2.ACTIVE
            or marker.run_ref != terminal.run_ref
            or marker.task_ref != terminal.task_ref
            or marker.agent_definition_ref != terminal.agent_definition_ref
            or marker.attempt != terminal.attempt
            or marker.fencing_token != terminal.fencing_token
            or terminal.predecessor_workspace_receipt_ref != marker.to_ref()
        ):
            raise AgentWorkspaceError("committed workspace marker drifted")
        if self._inventory_sha256(path) != terminal.inventory_sha256:
            raise AgentWorkspaceError("committed workspace inventory drifted")
        return AgentWorkspaceHandle(
            path=path,
            receipt=terminal,
        )

    def _workspace_path(
        self,
        run_ref: ObjectRef,
        lease: WorkspaceLease,
    ) -> Path:
        run_hash = hashlib.sha256(run_ref.object_id.encode()).hexdigest()
        task_hash = hashlib.sha256(lease.agent_task_ref.object_id.encode()).hexdigest()
        return self._workspaces / run_hash / task_hash / str(lease.attempt) / str(lease.fencing_token)

    def _validate_active_handle(
        self,
        handle: AgentWorkspaceHandle,
    ) -> None:
        self._validate_active_receipt(handle.receipt)
        self._assert_under_root(handle.path)
        if handle.path.is_symlink() or not handle.path.is_dir():
            raise AgentWorkspaceError("workspace namespace path is unsafe")
        marker = self._read_receipt(handle.path / _MARKER)
        if marker != handle.receipt:
            raise AgentWorkspaceError("workspace active marker drifted")

    @staticmethod
    def _validate_active_receipt(
        receipt: AgentWorkspaceReceiptV2,
    ) -> None:
        if receipt.state is not AgentWorkspaceStateV2.ACTIVE:
            raise AgentWorkspaceError("workspace handle is not active")

    def _inventory_sha256(self, root: Path) -> str:
        members: list[dict[str, object]] = []
        pending = [root]
        while pending:
            directory = pending.pop()
            for entry in sorted(
                os.scandir(directory),
                key=lambda value: value.name,
            ):
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise AgentWorkspaceError("workspace contains a symlink")
                if stat.S_ISDIR(mode):
                    pending.append(path)
                    continue
                if not stat.S_ISREG(mode):
                    raise AgentWorkspaceError("workspace contains a special file")
                if relative == _MARKER:
                    continue
                content = path.read_bytes()
                members.append(
                    {
                        "relative_path": relative,
                        "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
        encoded = json.dumps(
            canonical_value_v2(
                sorted(
                    members,
                    key=lambda value: str(value["relative_path"]),
                )
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _persist_terminal(
        self,
        receipt: AgentWorkspaceReceiptV2,
    ) -> None:
        self._persist_receipt(receipt)
        path = self._namespace_heads / f"{receipt.namespace_sha256}.json"
        _write_once(path, receipt.canonical_json())

    def _terminal_receipt(
        self,
        namespace_sha256: str,
    ) -> AgentWorkspaceReceiptV2 | None:
        path = self._namespace_heads / f"{namespace_sha256}.json"
        if not path.exists():
            return None
        return self._read_receipt(path)

    def _persist_receipt(
        self,
        receipt: AgentWorkspaceReceiptV2,
    ) -> None:
        _write_once(
            self._receipt_path(receipt.object_sha256),
            receipt.canonical_json(),
        )

    def _receipt_path(self, object_sha256: str) -> Path:
        return self._receipts / f"{object_sha256}.json"

    @staticmethod
    def _read_receipt(path: Path) -> AgentWorkspaceReceiptV2:
        if path.is_symlink() or not path.is_file():
            raise AgentWorkspaceError("workspace receipt material is missing or unsafe")
        try:
            receipt = AgentWorkspaceReceiptV2.model_validate_json(path.read_bytes())
        except ValidationError as exc:
            raise AgentWorkspaceError("workspace receipt material is invalid") from exc
        if receipt.canonical_json() != path.read_bytes():
            raise AgentWorkspaceError("workspace receipt material is not canonical")
        return receipt

    def _assert_under_root(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise AgentWorkspaceError("workspace path escapes private root")


def _namespace_sha256(
    run_ref: ObjectRef,
    lease: WorkspaceLease,
) -> str:
    payload = {
        "run_ref": run_ref,
        "task_ref": lease.agent_task_ref,
        "attempt": lease.attempt,
        "fencing_token": lease.fencing_token,
    }
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise AgentWorkspaceError("workspace immutable material conflicts")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != content:
                raise AgentWorkspaceError("workspace immutable material raced") from None
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "AgentWorkspaceError",
    "AgentWorkspaceHandle",
    "AgentWorkspaceManager",
]
