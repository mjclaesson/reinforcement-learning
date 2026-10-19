import pandas as pd
import streamlit as st

from viz import figure_path, learning_curve_chart, load_runs, player

st.title("Results")

runs = load_runs()
if runs is None:
    st.error("No trained runs found. Run `python demo/train_models.py` first (about 5 minutes).")
    st.stop()

experiment = st.segmented_control(
    "Experiment",
    ["Two agents, open map", "Narrow corridor", "n-step", "One agent, pickup", "Two agents, pickup", "Hyperparameter tuning"],
    default="Two agents, pickup",
)

if experiment == "Two agents, open map":
    st.markdown(
        "Two agents cross the default 7×10 map diagonally. Independent Q-learning, 10 000 episodes. "
        "Both find the shortest route (15 steps) and never collide."
    )
    player([runs["nav_open"]], key="nav_open", fig_width=520)
    learning_curve_chart([runs["nav_open"]])

elif experiment == "Narrow corridor":
    st.markdown(
        "The corridor has one pocket. One agent must step aside and wait: optimum 10 steps until both are home. "
        "Below: the same seed trained with the selfish and the cooperative reward."
    )
    player([runs["narrow_selfish"], runs["narrow_cooperative"]], key="narrow", fig_width=480)
    st.subheader("Selfish vs cooperative, 10 seeds each")
    st.table(
        pd.DataFrame(
            {
                "Reward mode": ["selfish", "cooperative"],
                "Finished": ["8/10", "9/10"],
                "Optimal (10 steps)": ["0/10", "0/10"],
                "Mean steps until both done": [12.6, 12.7],
                "Collisions": [0, 0],
            }
        ).set_index("Reward mode")
    )
    st.markdown(
        "* Collision-free every time, optimal almost never: the agents settle on a safe but slow way of passing.\n"
        "* The result depends on the seed. On two of ten selfish seeds the greedy agents block each other and never finish.\n"
        "* On this small map the two reward modes are indistinguishable."
    )
    st.subheader("Different settings, same map")
    st.table(
        pd.DataFrame(
            {
                "Setting": ["seed 0", "seed 1", "seed 2", "50 000 episodes, α = 0.2"],
                "A done": ["–", "14", "13", "11"],
                "B done": ["–", "11", "10", "10"],
                "Optimal": ["10", "10", "10", "10"],
                "Collisions": ["0", "0", "0", "0"],
            }
        ).set_index("Setting")
    )

elif experiment == "n-step":
    st.markdown("n-step Q-learning on the narrow corridor, 5 seeds per n, 10 000 episodes, α = 0.5.")
    st.table(
        pd.DataFrame(
            {
                "n": [1, 3, 5, 10],
                "Finished": ["4/5", "5/5", "4/5", "4/5"],
                "Optimal": ["0/5", "0/5", "0/5", "0/5"],
                "Mean steps": [13.0, 13.4, 14.0, 15.5],
                "Collisions": [0, 0, 0, 0],
            }
        ).set_index("n")
    )
    st.markdown(
        "* Longer returns did **not** help here: the greedy policy gets slower as n grows.\n"
        "* The n rewards include random ε-steps, so a long return is a noisier target. Episodes are short, so one-step bootstrapping is enough.\n"
        "* With n = 5 the single-agent pickup task also needed a lower α (0.1) to stay stable."
    )

elif experiment == "One agent, pickup":
    st.markdown(
        "One agent, three items, 9×9 map. The optimum (51 steps) is computed with breadth-first search over all "
        "(position, mask, carrying) states, so no policy can be shorter."
    )
    player([runs["pickup_one"]], key="pickup_one", fig_width=420)
    st.table(
        pd.DataFrame(
            {
                "Setting": ["n = 1, α = 0.5", "n = 5, α = 0.1"],
                "Steps (greedy, mean of 5 seeds)": [51.0, 51.8],
                "Optimum (BFS)": [51, 51],
                "Items delivered": [3, 3],
                "Useless PICK": [0, 0],
            }
        ).set_index("Setting")
    )
    img = figure_path("pickup_learning_curve.png")
    if img:
        st.image(img, width=650, alt="Learning curves for the single-agent pickup task")
    else:
        learning_curve_chart([runs["pickup_one"]])

elif experiment == "Two agents, pickup":
    st.markdown(
        "Two agents, two items each, shared drop-off, back to own start. Lower bound = each agent alone in the warehouse: "
        "**A 42, B 30**. The gap to it is the price of sharing the warehouse. "
        "Each mode is shown with its own best hyperparameters from the grid search."
    )
    sel, coop = runs["two_pickup_selfish"], runs["two_pickup_cooperative"]
    m = st.columns(5)
    m[0].metric("Lower bound (A / B)", "42 / 30")
    m[1].metric("Selfish, A done", sel["summary"]["A_done"], delta=f"{sel['summary']['A_done'] - 42:+d} vs bound", delta_color="inverse")
    m[2].metric("Selfish, B done", sel["summary"]["B_done"], delta=f"{sel['summary']['B_done'] - 30:+d} vs bound", delta_color="inverse")
    m[3].metric("Cooperative, A done", coop["summary"]["A_done"], delta=f"{coop['summary']['A_done'] - 42:+d} vs bound", delta_color="inverse")
    m[4].metric("Cooperative, B done", coop["summary"]["B_done"], delta=f"{coop['summary']['B_done'] - 30:+d} vs bound", delta_color="inverse")
    st.caption("Greedy run for seed 0. Black ring = the agent is carrying an item. × = item has left its shelf.")
    player([sel, coop], key="two_pickup", fig_width=420)
    learning_curve_chart([sel, coop])

    st.subheader("Final comparison, 10 seeds, 30 000 episodes")
    st.table(
        pd.DataFrame(
            {
                "": ["Lower bound", "selfish", "cooperative"],
                "Finished": ["", "10/10", "10/10"],
                "A done": ["42", "51.6", "47.2"],
                "B done": ["30", "34.0", "32.2"],
                "Both done": ["42", "51.6 ± 4.4", "47.2 ± 1.3"],
                "Collisions": ["", "0", "0"],
            }
        ).set_index("")
    )
    st.markdown(
        "* Both modes finish on every seed, with **zero collisions**.\n"
        "* Cooperative finishes the order **4.4 steps faster** on average and is much more consistent (45–50 steps against 44–59).\n"
        "* Both agents gain: A is 4.4 steps faster and B 1.8 steps faster."
    )

    st.subheader("What the initial Q-value does (3 seeds, α = 0.5, n = 1)")
    st.table(
        pd.DataFrame(
            {
                "Reward mode": ["selfish", "cooperative", "selfish", "cooperative", "selfish", "cooperative"],
                "Q init": [0, 0, -30, -30, -100, -100],
                "Finished": ["3/3", "0/3", "3/3", "3/3", "3/3", "3/3"],
                "A done (gap)": ["43.3 (1.3)", "–", "49.3 (7.3)", "51.0 (9.0)", "51.3 (9.3)", "55.3 (13.3)"],
                "B done (gap)": ["30.0 (0.0)", "–", "33.7 (3.7)", "33.3 (3.3)", "30.0 (0.0)", "31.3 (1.3)"],
            }
        ).set_index(["Reward mode", "Q init"])
    )
    st.markdown(
        "* With Q init 0 the cooperative agents never finish: every unvisited state looks better than finishing, so they wander.\n"
        "* A negative start fixes it. Too negative (−100) and the agents settle for the first route they find."
    )

elif experiment == "Hyperparameter tuning":
    st.markdown(
        "Grid search on the two-agent pickup task: α ∈ {0.1, 0.3, 0.5}, γ ∈ {0.95, 0.99}, final ε ∈ {0.05, 0.1, 0.2}, "
        "n ∈ {1, 3, 5}. 54 configurations × 3 seeds × 10 000 episodes, once per reward mode, Q init −30."
    )
    st.table(
        pd.DataFrame(
            {
                "Reward mode": ["selfish", "cooperative"],
                "α": [0.1, 0.1],
                "γ": [0.99, 0.99],
                "final ε": [0.05, 0.2],
                "n": [1, 3],
                "Mean steps (both done)": [49.7, 44.7],
            }
        ).set_index("Reward mode")
    )
    st.markdown(
        "* **γ = 0.99** is necessary: 64 % of the runs finish, against 30 % with γ = 0.95.\n"
        "* **α = 0.1** finishes most often (69 %, against 26 % with α = 0.5).\n"
        "* Selfish prefers a **low final ε and n = 1**. Cooperative prefers **n = 3**: its reward depends on what the other agent does later, "
        "and a longer return carries that information back.\n"
        "* The comparison between modes depends on the hyperparameters: under the selfish optimum, selfish wins (49.7 vs 53.3); "
        "under the cooperative optimum, cooperative wins (44.7 vs 52.3)."
    )
    img = figure_path("tuning_effect.png")
    if img:
        st.image(img, width="stretch", alt="Effect of each hyperparameter on steps and success rate, selfish mode")
    c1, c2 = st.columns(2)
    img = figure_path("n_comparison_steps.png")
    if img:
        c1.image(img, alt="Completion time for n = 1, 3, 5 with 10 seeds")
    img = figure_path("n_comparison_success.png")
    if img:
        c2.image(img, alt="Success rate for n = 1, 3, 5 with 10 seeds")
    st.table(
        pd.DataFrame(
            {"n": [1, 3, 5], "Finished": ["10/10", "9/10", "10/10"], "Mean steps (both done)": ["49.6 ± 1.2", "55.2 ± 2.0", "53.9 ± 2.8"]}
        ).set_index("n")
    )
    st.caption("Re-check of n for the selfish configuration with 10 seeds: n = 1 is both fastest and most consistent.")
