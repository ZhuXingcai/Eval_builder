from __future__ import annotations

import hashlib
import json
from pathlib import Path

from env_mock_agent.schemas import FetchResult


class RetrievalCache:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def get(self, url: str) -> FetchResult | None:
        record = self.root / f"{self.key(url)}.json"
        if not record.is_file():
            return None
        result = FetchResult.model_validate_json(record.read_text(encoding="utf-8"))
        return result if Path(result.content_path).is_file() else None

    def put(self, result: FetchResult) -> Path:
        record = self.root / f"{self.key(result.url)}.json"
        record.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return record
