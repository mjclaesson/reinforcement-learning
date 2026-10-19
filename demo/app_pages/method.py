import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import warehouse as w
from viz import player

st.title("Method")

tab_q, tab_eps, tab_live = st.tabs(["Q-learning and n-step", "Exploration schedule", "Train live"])

with tab_q:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Why tabular methods")
        st.markdown(
            """
* The state spaces are small (a few hundred to ~26 000 states per agent), so we can store a Q-value for every state–action pair.
* Tables are easy to inspect and reproduce, and match the course material.
* Value iteration needs a fixed model. With two agents learning at the same time, what happens to one agent depends on
  the other one's current policy, which changes during training. So we learn from experience instead.
"""
        )
        st.subheader("Independent Q-learning")
        st.markdown("Each agent has its **own Q-table** and learns as if the other one were part of the environment:")
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\left(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right)")
        st.markdown(
            "Consequence: from one agent's point of view the environment is **non-stationary**, "
            "because the other agent keeps changing its behaviour. We come back to this in the discussion."
        )
    with c2:
        st.subheader("n-step Q-learning")
        st.markdown("Wait *n* steps, use the real rewards for those steps, then bootstrap:")
        st.latex(r"G = r_0 + \gamma r_1 + \dots + \gamma^{n-1} r_{n-1} + \gamma^{n} \max_{a} Q(s_n, a)")
        st.latex(r"Q(s_0,a_0) \leftarrow Q(s_0,a_0) + \alpha\,(G - Q(s_0,a_0))")
        st.markdown(
            """
* Idea: a collision penalty (−5) reaches back to the action that caused it, not only to the last step.
* *n* = 1 is ordinary Q-learning; we checked that both implementations give identical results.
* Based on Sutton & Barto ch. 7 with the Sarsa target replaced by the max, as in Mnih et al. (2016).
* The *n* rewards follow the ε-greedy behaviour policy, so for *n* > 1 the update is not fully off-policy.
  We use the uncorrected version; ε shrinks during training so the difference matters less towards the end.
"""
        )
        st.subheader("Initial Q-value")
        st.markdown(
            "All rewards in the cooperative pickup task are negative until the episode ends, so a table that starts at 0 is "
            "**optimistic**: unvisited states look better than visited ones and the greedy agent keeps wandering. "
            "We therefore start the pickup Q-tables at −30."
        )

with tab_eps:
    st.subheader("ε-greedy with exponential decay")
    st.latex(
        r"\text{decay} = \left(\frac{\varepsilon_{\text{floor}}}{\varepsilon_{\text{start}}}\right)^{\frac{1}{f N}},"
        r"\qquad \varepsilon \leftarrow \max(\varepsilon_{\text{floor}},\ \varepsilon \cdot \text{decay})"
    )
    st.markdown(
        "ε reaches the floor after a fraction *f* of the *N* episodes and stays there. "
        "Large ε early finds the pick and deliver rewards, small ε late exploits what was learned. "
        "Greedy evaluation afterwards uses ε = 0."
    )
    row = st.container(horizontal=True)
    with row:
        eps_start = st.slider("ε start", 0.1, 1.0, 1.0, 0.05, width=220)
        eps_floor = st.select_slider("ε floor", options=[0.01, 0.05, 0.1, 0.2], value=0.05, width=220)
        fraction = st.slider("fraction f", 0.2, 1.0, 0.8, 0.1, width=220)
        episodes = st.select_slider("episodes N", options=[1000, 5000, 10000, 30000], value=5000, width=220)
    decay = (eps_floor / eps_start) ** (1 / (fraction * episodes))
    eps = eps_start
    values = []
    for _ in range(episodes):
        values.append(eps)
        eps = max(eps_floor, eps * decay)
    df = pd.DataFrame({"episode": np.arange(episodes), "epsilon": values})
    chart = (
        alt.Chart(df.iloc[:: max(1, episodes // 500)])
        .mark_line()
        .encode(x="episode:Q", y=alt.Y("epsilon:Q", scale=alt.Scale(domain=[0, 1])))
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(f"decay factor per episode ≈ {decay:.5f}; ε after 1000 episodes ≈ {values[min(999, episodes - 1)]:.2f}")

with tab_live:
    st.subheader("Train two agents live on the narrow corridor")
    st.markdown(
        "Optimum: 10 steps until both are at their goals (one agent goes straight, the other steps into the pocket and waits). "
        "Training takes a few seconds. Try a different seed or reward mode and look at the greedy policy afterwards."
    )
    with st.form("live_train"):
        row = st.container(horizontal=True, vertical_alignment="bottom")
        with row:
            mode = st.segmented_control("Reward mode", ["selfish", "cooperative"], default="selfish")
            n = st.segmented_control("n", [1, 3, 5], default=1)
            alpha = st.select_slider("α", options=[0.1, 0.2, 0.3, 0.5], value=0.5, width=160)
            eps_end = st.select_slider("final ε", options=[0.05, 0.1, 0.2], value=0.05, width=160)
            n_episodes = st.select_slider("episodes", options=[1000, 3000, 5000, 10000], value=5000, width=160)
            seed = st.number_input("seed", 0, 99, 1, width=120)
        submitted = st.form_submit_button("Train", icon=":material/model_training:")

    if submitted:
        env = w.TwoAgentWarehouse(w.WarehouseGrid("####.####\n........."), starts=[(1, 0), (1, 8)],
                                  goals=[(1, 8), (1, 0)], reward_mode=mode or "selfish")
        t0 = time.time()
        with st.spinner("Training..."):
            Qs, steps = w.train_nstep(env, n=int(n or 1), episodes=int(n_episodes), alpha=float(alpha),
                                      epsilon_end=float(eps_end), seed=int(seed))
        elapsed = time.time() - t0

        # replay the greedy policy and record frames in the same format as the saved runs
        rng = np.random.default_rng(0)
        state = env.reset()
        frames = [{"pos": [list(p) for p in state], "actions": None, "rewards": None, "events": []}]
        finished = [None, None]
        crashes = 0
        for t in range(1, 51):
            a = [w.choose_action(Qs[i], state, 0.0, rng) for i in range(2)]
            names = [w.ACTION_NAMES[a[0]], w.ACTION_NAMES[a[1]]]
            state, rewards, done = env.step(tuple(names))
            events = []
            for i in range(2):
                if rewards[i] < w.STEP_REWARD:
                    events.append(f"{'AB'[i]} blocked (-5)")
                    crashes += 1
                if finished[i] is None and state[i] == env.goals[i]:
                    finished[i] = t
                    events.append(f"{'AB'[i]} reached goal")
            frames.append({"pos": [list(p) for p in state], "actions": names, "rewards": rewards, "events": events})
            if done:
                break
        block = max(1, int(n_episodes) // 20)
        st.session_state["live_run"] = {
            "title": f"Live: {mode}, n={n}", "kind": "two_nav", "reward_mode": mode,
            "grid": ["####.####", "........."], "starts": env.starts, "goals": env.goals,
            "frames": frames, "summary": {"A_done": finished[0], "B_done": finished[1], "collisions": crashes},
            "curve": [float(np.mean(steps[i:i + block])) for i in range(0, len(steps), block)],
            "curve_block": block, "optimal": 10,
            "settings": f"α {alpha}, final ε {eps_end}, {n_episodes} episodes, seed {seed}, trained in {elapsed:.1f} s",
        }

    live = st.session_state.get("live_run")
    if live:
        s = live["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("A at goal after", s["A_done"] if s["A_done"] is not None else "never")
        m2.metric("B at goal after", s["B_done"] if s["B_done"] is not None else "never")
        m3.metric("Collisions", s["collisions"])
        m4.metric("Optimum", 10)
        c1, c2 = st.columns([3, 2])
        with c1:
            player([live], key="live_player", fig_width=520)
        with c2:
            df = pd.DataFrame({"episode": [(i + 1) * live["curve_block"] for i in range(len(live["curve"]))],
                               "steps": live["curve"]})
            st.altair_chart(
                alt.Chart(df).mark_line(point=True).encode(x="episode:Q", y=alt.Y("steps:Q", title="mean steps"))
                .properties(title="Learning curve", height=300),
                width="stretch",
            )
