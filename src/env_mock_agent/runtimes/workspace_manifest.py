from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

RUNTIME_DIRECTORIES = {
    ".envmock-sessions",
    ".pi-home",
    ".runtime-cache",
    ".runtime-home",
    ".runtime-tmp",
}


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    relative_path: str
    sha256: str
    size: int
    mtime_ns: int

    def as_payload(self) -> dict[str, object]:
        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }


WorkspaceManifest = dict[str, ManifestEntry]


def snapshot_workspace(root: Path) -> WorkspaceManifest:
    root = root.expanduser().resolve()
    if not root.exists():
        return {}
    entries: WorkspaceManifest = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in RUNTIME_DIRECTORIES:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        relative_path = relative.as_posix()
        entries[relative_path] = ManifestEntry(
            relative_path=relative_path,
            sha256=digest.hexdigest(),
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
    return entries


def changed_files(before: WorkspaceManifest, after: WorkspaceManifest) -> list[ManifestEntry]:
    return [entry for relative_path, entry in sorted(after.items()) if before.get(relative_path) != entry]
