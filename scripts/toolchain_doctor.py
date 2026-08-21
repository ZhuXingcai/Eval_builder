from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from env_mock_agent.providers import ProviderRegistry

TOOLS = {
    "libreoffice": ["libreoffice", "--version"],
    "ffmpeg": ["ffmpeg", "-version"],
    "ffprobe": ["ffprobe", "-version"],
    "pandoc": ["pandoc", "--version"],
    "pdftotext": ["pdftotext", "-v"],
    "pdfinfo": ["pdfinfo", "-v"],
    "tesseract": ["tesseract", "--version"],
    "qpdf": ["qpdf", "--version"],
    "ghostscript": ["gs", "--version"],
    "imagemagick": ["magick", "--version"],
}


def audit() -> dict[str, object]:
    results: dict[str, object] = {}
    for name, command in TOOLS.items():
        executable = shutil.which(command[0])
        if executable is None:
            results[name] = {"status": "missing"}
            continue
        process = subprocess.run(command, capture_output=True, check=False, text=True, timeout=15)
        output = (process.stdout or process.stderr).strip().splitlines()
        results[name] = {
            "status": "ok" if process.returncode == 0 else "error",
            "path": executable,
            "version": output[0] if output else "",
        }
    providers = [
        capability.model_dump(mode="json") for capability in ProviderRegistry.default().capabilities()
    ]
    return {"schema_version": 1, "tools": results, "providers": providers}


def main() -> None:
    report = audit()
    report_path = Path("docs/quality/toolchain-audit.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
