import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from viz import draw_frame, load_runs

st.title("Multi-agent order picking in a simulated warehouse")
st.caption("Joakim Möll · Filip Sjögren · Jesper Claesson · Sebastian Andersson")

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("The problem")
    st.markdown(
        """
Robots pick orders in a warehouse: walk to a shelf, pick the item, carry it to the
drop-off point, come back for the next one. Companies like Amazon and Ocado run
hundreds of such robots in the same aisles.

**Why reinforcement learning?** We never tell the robot the route. It only gets a
reward signal, and has to learn a policy that finishes the order in as few steps as possible.
"""
    )

    st.subheader("Goals of the project")
    st.markdown(
        """
1. **Build a warehouse environment** that we control ourselves (grid, shelves, items, drop-off).
2. **Train agents with tabular methods** (Q-learning and n-step Q-learning) to finish an order in as few steps as possible.
3. **Put two agents in the same warehouse** and study how collisions and two different reward
   functions, selfish and cooperative, change their behaviour.
"""
    )

    st.subheader("Three environments, built one on top of the other")
    st.table(
        pd.DataFrame(
            {
                "Version": ["Two agents, navigation", "Pickup", "Two agents, pickup"],
                "Agents": [2, 1, 2],
                "Task": [
                    "reach own goal without colliding",
                    "pick 3 items, deliver at drop-off, return to start",
                    "pick 2 items each, shared drop-off, return to own start",
                ],
            }
        ).set_index("Version")
    )

    st.info(
        "Scope: a small static 9×9 warehouse with 2–3 items per agent, so that tabular methods work. "
        "Generalising to new layouts would need function approximation (DQN) and is left as future work.",
        icon=":material/crop_square:",
    )

with right:
    st.subheader("The 9×9 warehouse")
    runs = load_runs()
    if runs is None:
        st.warning("No trained runs found. Run `python demo/train_models.py` first.")
    else:
        run = runs["two_pickup_selfish"]
        fig = draw_frame(run, 0, trail=False, cell=0.6)
        st.pyplot(fig, width=380, alt="The 9x9 warehouse with shelves, four items and two agents")
        plt.close(fig)
        st.caption(
            "Grey = shelves, D = shared drop-off. Agent A (blue) starts top-left and owns items 0 and 1, "
            "agent B (red) starts top-right and owns items 2 and 3. Items are picked from the floor cell next to the shelf."
        )
