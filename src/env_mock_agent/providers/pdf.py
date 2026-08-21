from __future__ import annotations

import textwrap

import pymupdf
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

from env_mock_agent.providers.base import ArtifactProvider, ProviderRequest
from env_mock_agent.providers.helpers import artifact_result, as_string_list
from env_mock_agent.schemas import ArtifactResult


class PdfProvider(ArtifactProvider):
    name = "pdf"
    asset_types = ("pdf",)

    def generate(self, request: ProviderRequest) -> ArtifactResult:
        target = request.destination()
        contract = request.plan.content_contract
        lines = self._content_lines(contract)
        font_name = "Helvetica"
        if any(any(ord(character) > 127 for character in line) for line in lines):
            font_name = "STSong-Light"
            pdfmetrics.registerFont(UnicodeCIDFont(font_name))
        document = canvas.Canvas(
            str(target),
            pagesize=A4,
            pageCompression=1,
            invariant=1,
        )
        document.setTitle(str(contract.get("title") or ""))
        document.setAuthor(str(contract.get("author") or ""))
        document.setCreator("")
        document.setProducer("")
        _, height = A4
        text = document.beginText(54, height - 54)
        text.setFont(font_name, 10)
        for line in lines:
            wrapped = textwrap.wrap(
                line,
                width=72 if font_name == "Helvetica" else 36,
                replace_whitespace=False,
                drop_whitespace=False,
            ) or [""]
            for part in wrapped:
                if text.getY() < 54:
                    document.drawText(text)
                    document.showPage()
                    text = document.beginText(54, height - 54)
                    text.setFont(font_name, 10)
                text.textLine(part)
            text.textLine("")
        document.drawText(text)
        document.save()
        with pymupdf.open(target) as parsed:  # type: ignore[no-untyped-call]
            page_count = parsed.page_count
        return artifact_result(
            request.plan.artifact_id,
            target,
            self.name,
            metadata={"pages": page_count},
        )

    @staticmethod
    def _content_lines(contract: dict[str, object]) -> list[str]:
        lines: list[str] = []
        title = str(contract.get("title") or "")
        if title:
            lines.append(title)
        lines.extend(as_string_list(contract.get("paragraphs")))
        sections = contract.get("sections")
        if isinstance(sections, list):
            for raw_section in sections:
                if not isinstance(raw_section, dict):
                    continue
                heading = str(raw_section.get("heading") or "")
                if heading:
                    lines.append(heading)
                lines.extend(as_string_list(raw_section.get("paragraphs", raw_section.get("content"))))
        tables = contract.get("tables")
        if isinstance(tables, list):
            for raw_table in tables:
                if not isinstance(raw_table, dict):
                    continue
                for row in raw_table.get("rows", []):
                    if isinstance(row, list):
                        lines.append(" | ".join(str(value) for value in row))
        return lines or [str(contract.get("content") or "Input material")]
