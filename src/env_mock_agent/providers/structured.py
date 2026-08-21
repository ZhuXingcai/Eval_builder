from __future__ import annotations

import json

import yaml

from env_mock_agent.providers.base import ArtifactProvider, ProviderRequest
from env_mock_agent.providers.helpers import artifact_result
from env_mock_agent.schemas import ArtifactResult


class StructuredDataProvider(ArtifactProvider):
    name = "structured"
    asset_types = ("json", "yaml", "yml")

    def generate(self, request: ProviderRequest) -> ArtifactResult:
        target = request.destination()
        data = request.plan.content_contract.get("data", {})
        if target.suffix.lower() == ".json":
            target.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            target.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=True),
                encoding="utf-8",
            )
        return artifact_result(request.plan.artifact_id, target, self.name)
