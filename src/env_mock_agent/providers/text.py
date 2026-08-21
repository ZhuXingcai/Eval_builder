from __future__ import annotations

from env_mock_agent.providers.base import ArtifactProvider, ProviderRequest
from env_mock_agent.providers.helpers import artifact_result, as_string_list
from env_mock_agent.schemas import ArtifactResult


class TextProvider(ArtifactProvider):
    name = "text"
    asset_types = ("txt", "text", "md", "markdown", "html", "rtf")

    def generate(self, request: ProviderRequest) -> ArtifactResult:
        target = request.destination()
        contract = request.plan.content_contract
        content = contract.get("content")
        if content is None:
            lines = as_string_list(contract.get("lines"))
            sections = contract.get("sections")
            if isinstance(sections, list):
                for raw_section in sections:
                    if not isinstance(raw_section, dict):
                        continue
                    heading = str(raw_section.get("heading") or "")
                    body = str(raw_section.get("content") or "")
                    if heading:
                        prefix = "## " if target.suffix.lower() in {".md", ".markdown"} else ""
                        lines.append(f"{prefix}{heading}")
                    if body:
                        lines.append(body)
            content = "\n\n".join(lines)
        target.write_text(str(content).rstrip() + "\n", encoding="utf-8")
        return artifact_result(request.plan.artifact_id, target, self.name)
