"""Shared helpers for the demo app: load runs, draw the warehouse, replay frames."""
import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Circle, Rectangle

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

AGENT_COLORS = ["#1f77b4", "#d62728"]        # A blue, B red
ITEM_COLORS = ["#aec7e8", "#ff9896"]         # lighter versions for the items


@st.cache_data
def load_runs():
    path = HERE / "models" / "runs.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def figure_path(name):
    """A figure saved by extract_from_notebook.py or the notebook itself, or None."""
    for folder in (HERE / "figures", HERE.parent / "figures"):
        p = folder / name
        if p.exists():
            return str(p)
    return None


def item_owner(run, item):
    for i, items in enumerate(run.get("items_per_agent", [])):
        if item in items:
            return i
    return 0


def draw_frame(run, step, ax=None, trail=True, cell=0.55):
    """Draw one frame of a run. Returns the matplotlib figure."""
    rows = run["grid"]
    n_rows, n_cols = len(rows), len(rows[0])
    frames = run["frames"]
    step = max(0, min(step, len(frames) - 1))
    frame = frames[step]
    n_agents = len(frame["pos"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(n_cols * cell, n_rows * cell))
    else:
        fig = ax.figure
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # floor and shelves
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            color = "#9e9e9e" if ch == "#" else "#fafafa"
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#dddddd", linewidth=0.8))

    # drop-off
    if "dropoff" in run:
        r, c = run["dropoff"]
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#ffe08a", edgecolor="#dddddd"))
        ax.text(c + 0.5, r + 0.5, "D", ha="center", va="center", fontsize=11, fontweight="bold")

    # in the pickup task the start is also the goal: mark it with a thin colored frame
    if "goals" not in run:
        for i, s in enumerate(run.get("starts", [])):
            ax.add_patch(Rectangle((s[1] + 0.05, s[0] + 0.05), 0.9, 0.9, facecolor="none",
                                   edgecolor=AGENT_COLORS[i], linewidth=1.5, linestyle="--"))
    for i, g in enumerate(run.get("goals", [])):
        ax.text(g[1] + 0.5, g[0] + 0.5, "ab"[i], ha="center", va="center", fontsize=11,
                color=AGENT_COLORS[i], fontweight="bold")

    # items on shelves: colored by owner, crossed out once they have left the shelf
    for item, (r, c) in run.get("items", {}).items():
        item = int(item)
        owner = item_owner(run, item)
        bit = run["items_per_agent"][owner].index(item)
        gone = bool(frame.get("mask", [0] * n_agents)[owner] & (1 << bit))
        if gone:
            ax.text(c + 0.5, r + 0.5, "×", ha="center", va="center", fontsize=10, color="#555555")
        else:
            ax.add_patch(Rectangle((c + 0.15, r + 0.15), 0.7, 0.7, facecolor=ITEM_COLORS[owner],
                                   edgecolor="#333333", linewidth=0.6))
            ax.text(c + 0.5, r + 0.5, str(item), ha="center", va="center", fontsize=8)

    # trail: where each agent has been
    if trail and step > 0:
        for i in range(n_agents):
            ys = [f["pos"][i][0] + 0.5 for f in frames[: step + 1]]
            xs = [f["pos"][i][1] + 0.5 for f in frames[: step + 1]]
            ax.plot(xs, ys, color=AGENT_COLORS[i], alpha=0.35, linewidth=2)

    # agents
    for i in range(n_agents):
        r, c = frame["pos"][i]
        ax.add_patch(Circle((c + 0.5, r + 0.5), 0.36, facecolor=AGENT_COLORS[i], edgecolor="white", linewidth=1.5))
        ax.text(c + 0.5, r + 0.5, "AB"[i], ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if frame.get("carrying", [False] * n_agents)[i]:
            ax.add_patch(Circle((c + 0.5, r + 0.5), 0.46, facecolor="none", edgecolor="black", linewidth=2))

    fig.tight_layout(pad=0.1)
    return fig


def frame_caption(run, step):
    frames = run["frames"]
    step = max(0, min(step, len(frames) - 1))
    f = frames[step]
    if f["actions"] is None:
        return "step 0: start"
    names = "AB"[: len(f["actions"])]
    acts = ", ".join(f"{n} {a}" for n, a in zip(names, f["actions"]))
    text = f"step {step}: {acts}"
    if f["events"]:
        text += "  ·  " + "; ".join(f["events"])
    return text


def summary_line(run):
    s = run["summary"]
    parts = []
    if "A_done" in s:
        parts.append(f"A done: {s['A_done']}")
    if "B_done" in s:
        parts.append(f"B done: {s['B_done']}")
    if "collisions" in s:
        parts.append(f"collisions: {s['collisions']}")
    if "delivered" in s:
        parts.append(f"delivered: {s['delivered']}")
    if "lower_bound" in run:
        parts.append(f"lower bound alone: A {run['lower_bound'][0]}, B {run['lower_bound'][1]}")
    elif "optimal" in run:
        parts.append(f"optimum: {run['optimal']}")
    return "  ·  ".join(parts)


def player(runs, key, fig_width=320):
    """Step slider + play button for one or several runs shown side by side."""
    n_steps = max(len(r["frames"]) for r in runs) - 1
    state_key = f"{key}_step"
    pending_key = f"{key}_pending"   # where Play leaves the slider; applied before the widget is created
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    def go_to_start():
        st.session_state[state_key] = 0

    @st.fragment
    def _player():
        if pending_key in st.session_state:
            st.session_state[state_key] = st.session_state.pop(pending_key)
        row = st.container(horizontal=True, vertical_alignment="bottom")
        with row:
            step = st.slider("Step", 0, n_steps, key=state_key, width=400)
            speed = st.select_slider("Speed", options=["slow", "normal", "fast"], value="normal",
                                     key=f"{key}_speed", width=200)
            play = st.button("Play", icon=":material/play_arrow:", key=f"{key}_play")
            st.button("Reset", icon=":material/replay:", key=f"{key}_reset", on_click=go_to_start)

        cols = st.columns(len(runs))
        slots = []
        for col, run in zip(cols, runs):
            with col:
                st.markdown(f"**{run['title']}**")
                st.caption(run["settings"])
                slots.append((st.empty(), st.empty()))

        def show(s):
            for (img, cap), run in zip(slots, runs):
                fig = draw_frame(run, s)
                img.pyplot(fig, width=fig_width, alt=f"{run['title']} at step {s}")
                plt.close(fig)
                cap.caption(frame_caption(run, s))

        if play:
            delay = {"slow": 0.6, "normal": 0.3, "fast": 0.12}[speed]
            start = st.session_state[state_key]
            for s in range(start, n_steps + 1):
                show(s)
                time.sleep(delay)
            st.session_state[pending_key] = n_steps
            st.rerun()
        else:
            show(step)

        for run in runs:
            st.caption(f"{run['title']}: {summary_line(run)}")

    _player()


def learning_curve_chart(runs, title="Mean steps per episode"):
    """Altair line chart of the learning curves of several runs."""
    import altair as alt
    import pandas as pd

    rows = []
    for run in runs:
        block = run["curve_block"]
        for i, v in enumerate(run["curve"]):
            rows.append({"episode": (i + 1) * block, "steps": v, "run": run["title"]})
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("episode:Q", title="episode"),
            y=alt.Y("steps:Q", title=f"mean steps per {block} episodes"),
            color=alt.Color("run:N", title=None),
            tooltip=["run", "episode", alt.Tooltip("steps", format=".1f")],
        )
        .properties(title=title, height=300)
    )
    st.altair_chart(chart, width="stretch")
