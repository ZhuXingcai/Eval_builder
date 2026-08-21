from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ContentCache:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, source: Path) -> tuple[str, Path]:
        digest = sha256_file(source)
        target = self.root / digest[:2] / digest
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return digest, target

    def get(self, digest: str) -> Path | None:
        path = self.root / digest[:2] / digest
        return path if path.is_file() else None
