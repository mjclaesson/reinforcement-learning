"""Generate demo/warehouse.py from the notebook, and save its figure outputs.

The notebook stays the source of truth. Run this again after the notebook changes:

    py -3.12 demo/extract_from_notebook.py

Cells are picked by their first line (not by index), so moving cells around is fine.
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "multi_agent_nstep.ipynb"
OUT_MODULE = ROOT / "demo" / "warehouse.py"
OUT_FIGURES = ROOT / "demo" / "figures"
REPORT_FIGURES = ROOT / "figures"

# first line of every code cell we want, in order
WANTED_FIRST_LINES = [
    "from collections import deque",                      # imports
    "def make_layout(",                                   # layout generator
    "class WarehouseGrid:",                               # grid
    "STEP_REWARD = -1",                                   # TwoAgentWarehouse + constants
    "from collections import defaultdict",                # choose_action, train
    "# kör en episod utan slump och visa vad agenterna lärt sig",  # run_greedy
    "# uppdatera den äldsta posten i bufferten med en n-stegs retur",  # nstep_update, train_nstep
    "# kör en girig episod tyst och returnera (steg tills båda i mål, antal krockar)",  # greedy_result
    "# kör en girig episod tyst: (steg när A kom i mål, steg när B kom i mål, krockar)",  # greedy_finish_times
    "PICK_REWARD = 5",                                    # PickupWarehouse, optimal_steps
    "# som choose_action, men slumpar bland alla handlingar inklusive PICK",  # train_pickup, run_greedy_pickup
    "class TwoAgentPickupWarehouse:",                     # + lower_bound_steps
    "# n-stegs Q-learning för två plockande agenter, en Q-tabell per agent på obs(i)",  # train_two_pickup
]

# png outputs we want, keyed by the first line of the producing cell -> file name
WANTED_FIGURES = {
    "import os": "two_agent_pickup_learning_curve.png",
    "params = [(\"alpha\", \"α\"), (\"gamma\", \"γ\"), (\"epsilon\", \"ε\"), (\"n\", \"n\")]": "tuning_effect.png",
    "fig, ax = plt.subplots(figsize=(6.5, 4.5))": "n_comparison_steps.png",
    "success = n_comparison[\"success_rate\"] * 100": "n_comparison_success.png",
}


def definitions_only(src):
    """Keep top-level imports, def, class and CONSTANT = ... ; drop experiment code."""
    import ast

    tree = ast.parse(src)
    kept = []
    for node in tree.body:
        keep = isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            keep = bool(names) and all(n.isupper() for n in names)
        if keep:
            kept.append(ast.get_source_segment(src, node))
    return "\n\n".join(kept)


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    by_first_line = {}
    for c in code_cells:
        src = "".join(c["source"])
        first = src.strip().splitlines()[0] if src.strip() else ""
        by_first_line[first] = (src, c.get("outputs", []))

    parts = [
        '"""Environments and training code, generated from multi_agent_nstep.ipynb.',
        "",
        "Do not edit by hand: run demo/extract_from_notebook.py instead.",
        '"""',
        "import numpy as np",
        "",
    ]
    def find(prefix):
        hits = [k for k in by_first_line if k.startswith(prefix)]
        if len(hits) != 1:
            raise SystemExit(f"expected one cell starting with {prefix!r}, found {len(hits)}")
        return by_first_line[hits[0]]

    for first in WANTED_FIRST_LINES:
        src, _ = find(first)
        # the module must not run anything on import: only keep definitions
        parts.append(f"# ---- from notebook cell starting with: {first[:60]}")
        parts.append(definitions_only(src))
        parts.append("")
    OUT_MODULE.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT_MODULE)

    OUT_FIGURES.mkdir(exist_ok=True)
    REPORT_FIGURES.mkdir(exist_ok=True)
    for first, name in WANTED_FIGURES.items():
        try:
            _, outputs = find(first)
        except SystemExit as e:
            print(e)
            continue
        pngs = [o["data"]["image/png"] for o in outputs if "image/png" in o.get("data", {})]
        if not pngs:
            print("no png output in cell:", first[:50])
            continue
        data = base64.b64decode(pngs[0])
        (OUT_FIGURES / name).write_bytes(data)
        target = REPORT_FIGURES / name
        if not target.exists():
            target.write_bytes(data)
        print("saved", name)


if __name__ == "__main__":
    main()
