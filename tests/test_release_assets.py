"""The release validator accepts the committed card and notebook and rejects spec violations."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validator", ROOT / "tools" / "validate_release_assets.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)

CARD = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
NOTEBOOK = json.loads((ROOT / validator.NOTEBOOK).read_text(encoding="utf-8"))
REGISTRY = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")


def test_committed_assets_pass():
    assert validator.main() == 0


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c.replace('model_card_spec: "1.2"', 'model_card_spec: "1.1"'),
        lambda c: c.replace('date_published: "2022-03-07"\n', ""),
        lambda c: c.replace("###### Human Life", "#### Human Life"),
        lambda c: c.replace("###### Groups\n", "###### Groups\n\n<!-- Tooltip: groups -->\n"),
        lambda c: c.replace("#### Metrics\n", "## Aside\n\ntext\n\n#### Metrics\n"),
        lambda c: c.replace("###### Use cases", "###### Other"),
        lambda c: c.replace("\n# DiT", "\n# DiT\n\n# Second title"),
    ],
)
def test_model_card_violations_are_rejected(edit):
    changed = edit(CARD)
    assert changed != CARD
    with pytest.raises(SystemExit):
        validator.check_model_card(changed)


def test_bare_na_section_is_rejected():
    start = CARD.index("###### Human Life")
    end = CARD.index("###### Mitigations")
    with pytest.raises(SystemExit):
        validator.check_model_card(CARD[:start] + "###### Human Life\n\nN/A\n\n" + CARD[end:])


@pytest.mark.parametrize(
    "key,value", [("notebook_spec", "2.1"), ("notebook_profile", "E2E"), ("standalone", False)]
)
def test_notebook_metadata_violations_are_rejected(key, value):
    notebook = json.loads(json.dumps(NOTEBOOK))
    notebook["metadata"]["dimer"][key] = value
    with pytest.raises(SystemExit):
        validator.check_notebook(notebook, REGISTRY)


def test_notebook_with_outputs_is_rejected():
    notebook = json.loads(json.dumps(NOTEBOOK))
    cell = next(c for c in notebook["cells"] if c["cell_type"] == "code")
    cell["execution_count"] = 1
    with pytest.raises(SystemExit):
        validator.check_notebook(notebook, REGISTRY)
