from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from threading import Lock

from env_mock_agent.schemas import RuntimeEvent


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def append(self, event: RuntimeEvent | dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = event.model_dump(mode="json") if isinstance(event, RuntimeEvent) else event
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()

    def read(self) -> Iterable[dict[str, object]]:
        if not self.path.exists():
            return []
        events: list[dict[str, object]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        events.append(value)
        return events
