from __future__ import annotations

from env_mock_agent.providers.base import ArtifactProvider, ProviderRequest
from env_mock_agent.providers.helpers import artifact_result
from env_mock_agent.runtimes.security import is_path_inside
from env_mock_agent.schemas import ArtifactResult


class CodeProjectProvider(ArtifactProvider):
    name = "code_project"
    asset_types = ("code", "project", "directory")

    def generate(self, request: ProviderRequest) -> ArtifactResult:
        target = request.destination()
        target.mkdir(parents=True, exist_ok=True)
        files = request.plan.content_contract.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("code project content_contract.files must be a non-empty mapping")
        for relative_path, raw_content in sorted(files.items(), key=lambda item: str(item[0])):
            destination = (target / str(relative_path)).resolve()
            if not is_path_inside(target, destination):
                raise ValueError(f"code project path escapes root: {relative_path}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(str(raw_content), encoding="utf-8")
        return artifact_result(
            request.plan.artifact_id,
            target,
            self.name,
            metadata={
                "files": sorted(
                    path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()
                )
            },
        )
