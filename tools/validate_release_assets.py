#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "MODEL_CARD.md",
    "STATUS.md",
    "pipeline-manifest.json",
    "pyproject.toml",
    "src/dit_document_classification_pipeline/__init__.py",
    "src/dit_document_classification_pipeline/pipeline.py",
    "tools/convert_weights.py",
    "docs/WEIGHTS.md",
]


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"missing release assets: {missing}")
    manifest = json.loads((ROOT / "pipeline-manifest.json").read_text(encoding="utf-8"))
    if manifest["model"]["id"] != "microsoft/dit-base-finetuned-rvlcdip":
        raise SystemExit("pipeline-manifest model identity mismatch")
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    if "HOLD" not in status or "licen" not in status.lower():
        raise SystemExit("STATUS.md must retain the licence/redistribution HOLD")
    print("release assets: static validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
