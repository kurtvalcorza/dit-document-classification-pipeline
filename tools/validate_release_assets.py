#!/usr/bin/env python
"""Static release-asset validation: files, model card (spec 1.2), notebook metadata (spec 2.2).

This is a static check. It is not notebook execution evidence (Notebook Specification REL8).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "microsoft/dit-base-finetuned-rvlcdip"
MODEL_REVISION = "23f8b03d130fb66bbc9a15df3c75d753e49240eb"
MODEL_CARD_SPEC = "1.2"
NOTEBOOK_SPEC = "2.2"
NOTEBOOK = "tutorials/DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb"
REQUIRED = [
    "LICENSE",
    "README.md",
    "MODEL_CARD.md",
    "STATUS.md",
    "pipeline-manifest.json",
    "pyproject.toml",
    "src/dit_document_classification_pipeline/__init__.py",
    "src/dit_document_classification_pipeline/pipeline.py",
    "tools/convert_weights.py",
    "docs/WEIGHTS.md",
    "docs/release-verification.md",
    "tutorials/README.md",
    NOTEBOOK,
]

# Model Card Specification 1.2, section 4: heading and level, in order.
CARD_SECTIONS = [
    ("Description", 4),
    ("Intended Use and Limitations", 4),
    ("Primary Intended Uses", 6),
    ("Primary Intended Users", 6),
    ("Out-of-scope use cases", 6),
    ("Factors", 4),
    ("Groups", 6),
    ("Instrumentation", 6),
    ("Environment", 6),
    ("Metrics", 4),
    ("Performance Measures", 6),
    ("Decision thresholds", 6),
    ("Approaches to uncertainty and variability", 6),
    ("Ethical considerations and biases", 4),
    ("Data", 6),
    ("Human Life", 6),
    ("Mitigations", 6),
    ("Risks and harms", 6),
    ("Use cases", 6),
]
CONTAINERS = {"intended use and limitations", "factors", "metrics", "ethical considerations and biases"}
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|Insert text here|Tooltip:", re.I)
BARE_NA = re.compile(r"^\s*(N/?A|None|Not applicable)\.?\s*$", re.I)
HEADING = re.compile(r"(#{1,6})\s+(.*)")


def fail(message: str) -> None:
    raise SystemExit(f"release asset validation failed: {message}")


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        fail("MODEL_CARD.md must open with a YAML front-matter block (G1)")
    end = text.index("\n---", 4)
    fields = {}
    for line in text[4:end].splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip('"')
    return fields, text[end + 4 :]


def card_sections(body: str) -> list[tuple[int, str, str]]:
    """Return (level, name, content) for every heading, content running to the next heading."""
    lines = body.splitlines()
    marks = [(i, m) for i, line in enumerate(lines) if (m := HEADING.match(line))]
    sections = []
    for n, (i, match) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        name = match.group(2).strip().strip("*").strip()
        sections.append((len(match.group(1)), name, "\n".join(lines[i + 1 : end]).strip()))
    return sections


def check_model_card(text: str) -> None:
    front, body = split_front_matter(text)
    for key in ("license", "model_card_spec", "pipeline_tag", "base_model", "date_published"):
        if not front.get(key):
            fail(f"MODEL_CARD.md front matter must declare {key} (G1)")
    if front["model_card_spec"] != MODEL_CARD_SPEC:
        fail(f'MODEL_CARD.md model_card_spec must be "{MODEL_CARD_SPEC}"')
    if front["base_model"] != MODEL_ID:
        fail("MODEL_CARD.md base_model does not name the pinned upstream model")
    if not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?|null", front["date_published"]):
        fail("MODEL_CARD.md date_published must be YYYY, YYYY-MM, YYYY-MM-DD or null (G2)")

    sections = card_sections(body)
    if sum(level == 1 for level, _, _ in sections) != 1:
        fail("MODEL_CARD.md must carry exactly one level-1 heading (G3)")
    preamble = body.split("> [!WARNING]", 1)[0]
    if "kurtvalcorza/dit-document-classification-pipeline" in preamble:
        fail("MODEL_CARD.md badge row must not link to this repository (G8)")

    wanted = [name.lower() for name, _ in CARD_SECTIONS]
    positions = [n for n, (_, name, _) in enumerate(sections) if name.lower() in wanted]
    found = [sections[n] for n in positions]
    if [name.lower() for _, name, _ in found] != wanted:
        fail(f"MODEL_CARD.md required sections missing or out of order (G4, G6): {[n for _, n, _ in found]}")
    if positions != list(range(positions[0], positions[0] + len(wanted))):
        fail("MODEL_CARD.md required sections must form one contiguous block (G6)")
    for (level, name, content), (_, want_level) in zip(found, CARD_SECTIONS, strict=True):
        if level != want_level:
            fail(f"MODEL_CARD.md section {name!r} must be a level-{want_level} heading (G5)")
        if name.lower() not in CONTAINERS and (not content or BARE_NA.match(content)):
            fail(f"MODEL_CARD.md section {name!r} is unanswered or a bare N/A (G9, G10)")
    if PLACEHOLDER.search(body) or "<!--" in body:
        fail("MODEL_CARD.md contains a placeholder, tooltip or HTML comment (G9, G11)")
    if MODEL_REVISION not in body:
        fail("MODEL_CARD.md must state the pinned upstream revision")


def check_notebook(notebook: dict, registry: str) -> None:
    meta = notebook["metadata"].get("dimer", {})
    expected = {
        "notebook_spec": NOTEBOOK_SPEC,
        "notebook_profile": "TASK-INFERENCE",
        "notebook_mode": "WORKSHOP",
        "standalone": True,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            fail(f"{NOTEBOOK} metadata.dimer.{key} must be {value!r}, found {meta.get(key)!r}")
    model = meta.get("model", {})
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        fail(f"{NOTEBOOK} metadata must pin {MODEL_ID}@{MODEL_REVISION}")
    opening = "".join(notebook["cells"][0]["source"])
    for needle in ("`TASK-INFERENCE`", "`WORKSHOP`", f"`{NOTEBOOK_SPEC}`"):
        if needle not in opening:
            fail(f"{NOTEBOOK} opening cell must declare {needle} (Notebook Specification 3.4)")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and (cell.get("outputs") or cell.get("execution_count")):
            fail(f"{NOTEBOOK} must be committed without outputs (SRC4)")
    row = next((line for line in registry.splitlines() if Path(NOTEBOOK).name in line), None)
    if row is None:
        fail("tutorials/README.md must register the notebook")
    if "release-grade" in row.lower():
        fail("tutorials/README.md must not mark the notebook release-grade while licence review is pending")


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        fail(f"missing release assets: {missing}")
    manifest = json.loads((ROOT / "pipeline-manifest.json").read_text(encoding="utf-8"))
    if manifest["model"]["id"] != MODEL_ID or manifest["model"]["revision"] != MODEL_REVISION:
        fail("pipeline-manifest model identity mismatch")
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    if "HOLD" not in status or "licen" not in status.lower():
        fail("STATUS.md must retain the licence/redistribution HOLD")
    check_model_card((ROOT / "MODEL_CARD.md").read_text(encoding="utf-8"))
    check_notebook(
        json.loads((ROOT / NOTEBOOK).read_text(encoding="utf-8")),
        (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8"),
    )
    print(f"release assets: static validation passed (card {MODEL_CARD_SPEC}, notebook {NOTEBOOK_SPEC})")
    print("NOTE: static source validation only; not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
