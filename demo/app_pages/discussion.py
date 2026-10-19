import altair as alt
import pandas as pd
import streamlit as st

st.title("Discussion and conclusion")

tab_what, tab_why, tab_limits, tab_conclusion = st.tabs(
    ["What the numbers say", "Why it looks like this", "Limitations and related work", "Conclusion"]
)

with tab_what:
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.subheader("Three things we saw")
        st.markdown(
            """
1. **Collision-free is easy, optimal is hard.** Every trained policy, on every map, has zero collisions.
   But two agents that learn at the same time almost never find the joint optimum.
2. **The reward design matters more than the algorithm.** Cooperative reward: 47.2 steps, ±1.3.
   Selfish: 51.6 steps, ±4.4. n-step, α and ε moved the numbers by a few steps; the reward mode changed the behaviour.
3. **Small details decide whether learning happens at all.** With Q initialised to 0 the cooperative agents never finish.
   With −30 they do. Nothing else changed.
"""
        )
    with c2:
        df = pd.DataFrame(
            {
                "setting": ["Alone (bound)", "Selfish", "Cooperative"] * 2,
                "agent": ["A"] * 3 + ["B"] * 3,
                "steps": [42, 51.6, 47.2, 30, 34.0, 32.2],
            }
        )
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("steps:Q", title="steps until done (mean of 10 seeds)"),
                y=alt.Y("setting:N", title=None, sort=["Alone (bound)", "Selfish", "Cooperative"]),
                color=alt.Color("agent:N", scale=alt.Scale(domain=["A", "B"], range=["#1f77b4", "#d62728"]), title="agent"),
                row=alt.Row("agent:N", title=None),
                tooltip=["setting", "agent", "steps"],
            )
            .properties(height=90, title="The price of sharing the warehouse")
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Time until each agent is done, best hyperparameters per mode, 10 seeds, 30 000 episodes. "
                   "The order is finished when the slower agent (A) is done.")

with tab_why:
    st.subheader("Non-stationarity")
    st.markdown(
        """
* Each agent learns with its own Q-table and treats the other agent as part of the environment.
  But the other agent is learning too, so the transition dynamics seen by one agent **drift during training**:
  the problem is a Markov game, not a fixed MDP, and Q-learning's convergence guarantee does not apply.
* This is why the narrow corridor is so **seed-dependent**: which agent gives way is decided early by chance,
  then both agents lock the convention in. On two of ten selfish seeds the greedy agents block each other and never finish.
* It also explains why the selfish mode **varies more between seeds** (±4.4) than the cooperative one (±1.3):
  with a shared reward both agents optimise the same number, so their conventions agree more often.
"""
    )
    st.subheader("Why a gap of five steps remains")
    st.markdown(
        """
* The best cooperative policy finishes in 47 steps against a lower bound of 42. Only about two of those steps are actual giving way.
* The rest is **hesitation in rarely visited states**: the observation contains the other agent's exact position,
  so the table has ~26 000 rows per agent and some rows are visited a handful of times. Q-values for `UP` and `WAIT`
  end up almost equal and the greedy agent dithers.
* Training three times longer (100 000 episodes) did not change the greedy result. The gap is the observation, not the time.
* A quick test with a **coarser observation** ("other agent within two cells: yes/no") removed most of the gap
  and shrank the table ~28×, but was not run with the final code, so it is future work rather than a result.
"""
    )
    st.subheader("Optimistic initialisation, by accident")
    st.markdown(
        """
* In the cooperative pickup task every reward is negative until the episode ends. A table that starts at 0 is therefore
  **optimistic**: an unvisited state always looks better than a visited one, and the greedy policy chases new states instead of finishing.
* This is the textbook "optimistic initial values" exploration trick (Sutton & Barto §2.6) working against us.
  Starting at −30 removes it. Starting at −100 is pessimistic: the agents stop exploring and keep the first route they find.
"""
    )
    st.subheader("n-step: it depends on the reward")
    st.markdown(
        """
* On the corridor and for the selfish pickup task, n = 1 was best. The n rewards follow the ε-greedy behaviour policy,
  so a longer return is a **noisier target** (no importance-sampling correction), and episodes are short enough for one-step bootstrapping.
* For the cooperative pickup task n = 3 was best (44.7 vs 53.3 steps for n = 1). Here an agent's reward depends on
  **what the other agent does several steps later**; a longer return carries that information back to the action that mattered.
* Consequence: the selfish-vs-cooperative comparison is **not hyperparameter-free**. Under the selfish optimum, selfish wins.
  Under the cooperative optimum, cooperative wins by more. We report best against best.
"""
    )

with tab_limits:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Limitations")
        st.markdown(
            """
* **Static warehouse.** Items are on the same shelves every episode; the policy does not transfer to a new layout.
  That would need function approximation (DQN) and a different observation.
* **Small.** 9×9 grid, two agents, two items each. Tabular Q scales badly: the observation already has ~26 000 values per agent.
* **Deterministic and synchronous.** Both agents move at the same time with perfect moves. Real robots have delays and slip.
* **Lower bound, not optimum.** For two agents we compare with each agent alone. The true joint optimum is somewhere between 42 and 47 steps.
* **Few seeds in places.** The Q-init table uses 3 seeds per row; the final comparison uses 10.
"""
        )
    with c2:
        st.subheader("Related work")
        st.markdown(
            """
* **Independent Q-learning** (Tan 1993; Claus & Boutilier 1998): the approach we use. Known to work in small cooperative
  games but without convergence guarantees, because of exactly the non-stationarity we saw.
* **Multi-agent RL surveys** (Buşoniu, Babuška & De Schutter 2008; Hernandez-Leal et al. 2017) list non-stationarity
  and credit assignment as the two central problems. Our cooperative reward is the simplest possible shared-credit scheme.
* **Multi-agent path finding** (Sharon et al. 2015, CBS; Stern et al. 2019) solves the routing part optimally with a
  central planner and full information. We solve it decentralised, by learning, and land ~12 % above the single-agent bound.
* **Warehouse robots** (Wurman, D'Andrea & Mountz 2008, Kiva/Amazon) coordinate hundreds of robots with central planning.
  Learning-based coordination is interesting where the planner's model is wrong or unavailable.
"""
        )
    st.subheader("Practical impact")
    st.markdown(
        "In a real warehouse what matters is when the **whole order** is done, not when each robot is done. "
        "Our result says the reward should be defined on that quantity: the cooperative agents finished the order faster "
        "and more predictably. The environment is far simpler than a real warehouse, so this is an example of how reward "
        "design shapes coordination, not a deployable controller."
    )

with tab_conclusion:
    st.subheader("What we built")
    st.markdown(
        "A warehouse environment with three difficulty levels, independent tabular Q-learning and n-step Q-learning for two agents "
        "with collision rules, two reward designs, and a 54-configuration grid search per reward mode."
    )
    st.subheader("What we found")
    st.markdown(
        """
* Avoiding collisions is learned quickly and reliably; reaching the joint optimum is not, because both agents learn at once.
* A **shared, cooperative reward** gave the fastest and most consistent order completion: 47.2 ± 1.3 steps against 51.6 ± 4.4,
  with both agents gaining.
* The **initial Q-value** decides whether the cooperative agents learn at all, and n-step returns help only when the reward
  depends on the other agent's future actions.
"""
    )
    st.subheader("What we would do next")
    st.markdown(
        """
1. **Coarser observation** of the other agent (near/far instead of exact position) to remove the hesitation and shrink the table.
2. **More agents and items**, where the shared drop-off becomes a real bottleneck.
3. **Random layouts per episode**, which needs function approximation (DQN) and a map-based observation.
4. **Richer collision model**, e.g. a collision drops the item.
"""
    )
    st.success("Collision-free is easy. Optimal is hard. The reward you choose decides which one you get.", icon=":material/check_circle:")
