# demo/ — presentation app (prototype)

Built 2026-10-19 for the recorded presentation. Throw-away quality: no tests, hard-coded result tables.

## What it proved
- The notebook's environments and training code can be lifted into a module automatically
  (`extract_from_notebook.py`, picks cells by first line, keeps only definitions).
- Pre-trained greedy runs saved as frames (`models/runs.json`) replay instantly in Streamlit;
  no training at presentation time except the 3-second live demo on the corridor.
- Side-by-side replay of selfish vs cooperative is the clearest way to show the result.

## Files
- `app.py` + `app_pages/` — one page per presentation section.
- `viz.py` — drawing, player (slider + play), learning-curve chart.
- `warehouse.py` — generated, do not edit.
- `train_models.py` — produces `models/runs.json` (seed 0 for every run).
- `figures/` — pngs pulled from the notebook outputs.

## What to throw away / known gaps
- Result tables in `app_pages/results.py` are typed in by hand from the report; they do not read the notebook.
- Play is a blocking loop with `time.sleep`; fine for recording, not for a shared server.
- `screens/` and `*.log` are scratch output.
