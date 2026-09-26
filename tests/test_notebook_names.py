"""Every name the notebook's code cells use must be defined somewhere in the notebook."""
import ast
import builtins
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "tutorials" / "DIMER_Document_Type_Classification_RVL_CDIP_Workshop.ipynb"


def test_no_undefined_names_across_code_cells():
    # `math` was used by the gallery cell but never imported; the Colab T4 run stopped there.
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    tree = ast.parse("\n\n".join(cells))
    bound = set(dir(builtins)) | {"display", "get_ipython"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert not used - bound, f"names used but never defined: {sorted(used - bound)}"
