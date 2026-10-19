import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import warehouse as w
from viz import draw_frame, frame_caption

st.title("Problem formulation")

tab_env, tab_sar, tab_rules = st.tabs(["Environment", "States, actions, rewards", "Collision rules, step by step"])

with tab_env:
    st.markdown(
        """
* The warehouse is a grid of **floor** (`.`) and **shelf** (`#`) cells. Agents walk on floor cells only.
* Items lie on shelves and are picked from a floor cell next to the shelf (a *pick spot*). Shelves can be picked from both sides.
* Items are placed on random shelves **once** and then stay fixed, so the layout is static during training.
* A layout is generated from four numbers (shelf rows, shelves per row, shelf width and depth) or written by hand as text,
  which is how the narrow corridor was made.
* A breadth-first search checks that every floor cell is reachable and that every shelf has a pick spot, otherwise the layout is rejected.
* Moves are **deterministic** and both agents move at the same time.
"""
    )
    st.code("####.####\n.........", language="text")
    st.caption("The narrow corridor: two agents must pass each other, and the only place to give way is the pocket in the middle.")

with tab_sar:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("States")
        st.table(
            pd.DataFrame(
                {
                    "Version": ["Two agents, navigation", "Pickup (one agent)", "Two agents, pickup"],
                    "What an agent sees": [
                        "both positions ((r₁,c₁),(r₂,c₂))",
                        "(position, mask, carrying)",
                        "own (position, mask, carrying) + the other agent's position",
                    ],
                    "Size": ["~57² ≈ 3 000", "57 × 8 × 2 ≈ 900", "57 × 4 × 2 × 57 ≈ 26 000 per agent"],
                }
            ).set_index("Version")
        )
        st.caption(
            "mask = one bit per item, set when the item has left its shelf. In the two-agent pickup task an agent does "
            "not see what the other one has picked: that would multiply the table by 8 and make it too sparse. "
            "The two-agent task is therefore a partially observable Markov game, not a single MDP."
        )
        st.subheader("Actions")
        st.markdown(
            "`UP`, `DOWN`, `LEFT`, `RIGHT`, `WAIT` and, in the pickup versions, `PICK`. "
            "`WAIT` exists so an agent can let the other pass. Delivery happens automatically when a carrying agent steps on D."
        )
    with c2:
        st.subheader("Rewards")
        st.table(
            pd.DataFrame(
                {
                    "Event": [
                        "every step",
                        "try to walk into the other agent (blocked)",
                        "pick an item",
                        "deliver an item at drop-off",
                        "PICK with nothing to pick, or hands full",
                    ],
                    "Reward": ["−1", "−5 extra", "+5", "+10", "−2 extra"],
                }
            ).set_index("Event")
        )
        st.markdown(
            """
**Two ways of giving the step reward with two agents**

* **selfish**: −1 per step until the agent reaches *its own* goal, then 0.
* **cooperative**: both get −1 per step until *both* are at their goals. The fast agent pays for the slow one,
  so it should learn to let the other pass instead of blocking.

**Objective:** finish the task in as few steps as possible, i.e. maximise the sum of rewards per episode.
"""
        )

with tab_rules:
    st.markdown(
        """
Both agents choose an action at the same time, then:
1. walking into a shelf or outside the grid → the agent stays (still costs −1);
2. both want the same cell → **both stay**;
3. they try to swap places → **both stay**;
4. the agent that wanted to move but was stopped by the other one gets **−5 extra**.
"""
    )

    # the same test as in the notebook, on a five-cell corridor
    tests = [
        (("RIGHT", "LEFT"), "both take one step"),
        (("RIGHT", "LEFT"), "both want the middle cell → collision, both stay"),
        (("RIGHT", "WAIT"), "B waits, A takes the middle cell"),
        (("RIGHT", "LEFT"), "they try to swap places → collision, both stay"),
        (("UP", "WAIT"), "A walks into the wall: stays, but no collision penalty"),
    ]

    @st.cache_data
    def collision_run():
        env = w.TwoAgentWarehouse(w.WarehouseGrid("....."), starts=[(0, 0), (0, 4)], goals=[(0, 4), (0, 0)])
        state = env.reset()
        frames = [{"pos": [list(p) for p in state], "actions": None, "rewards": None, "events": []}]
        for actions, text in tests:
            state, rewards, done = env.step(actions)
            frames.append({"pos": [list(p) for p in state], "actions": list(actions), "rewards": rewards,
                           "events": [text, f"rewards {rewards}"]})
        return {"title": "Collision test", "grid": ["....."], "starts": env.starts, "goals": env.goals,
                "frames": frames, "settings": "", "summary": {}}

    run = collision_run()
    step = st.slider("Step", 0, len(run["frames"]) - 1, 0, key="collision_step")
    fig = draw_frame(run, step, trail=False, cell=0.9)
    st.pyplot(fig, width=450, alt=f"Collision test, step {step}")
    plt.close(fig)
    st.markdown(f"**{frame_caption(run, step)}**")
