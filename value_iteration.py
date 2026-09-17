import numpy as np
from gridworld import GridWorld, Action


def value_iteration(env: GridWorld, gamma=0.95, theta=1e-6, max_iters=1000):
    V = {s: 0.0 for s in env.states}

    for _ in range(max_iters):
        delta = 0.0
        new_V = V.copy()

        for s in env.states:
            if env.is_terminal(s):
                continue

            action_values = []
            for a in env.actions:
                probs = env.transition_probs(s, a)
                q_sa = sum(
                    p * (env.reward(s, a, s_next) + gamma * V[s_next])
                    for s_next, p in probs.items()
                )
                action_values.append(q_sa)

            new_V[s] = max(action_values)
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        if delta < theta:
            break

    policy = extract_policy(env, V, gamma)
    return V, policy


def extract_policy(env: GridWorld, V, gamma=0.95):
    policy = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = None
            continue

        best_a, best_q = None, -float("inf")
        for a in env.actions:
            probs = env.transition_probs(s, a)
            q_sa = sum(
                p * (env.reward(s, a, s_next) + gamma * V[s_next])
                for s_next, p in probs.items()
            )
            if q_sa > best_q:
                best_q, best_a = q_sa, a
        policy[s] = best_a
    return policy


def print_policy(env: GridWorld, policy):
    arrows = {
        Action.UP: "^",
        Action.DOWN: "v",
        Action.LEFT: "<",
        Action.RIGHT: ">",
        None: " ",
    }
    for r in range(env.n_rows):
        row = ""
        for c in range(env.n_cols):
            s = (r, c)
            if s == env.goal:
                row += " G "
            elif s in env.obstacles:
                row += " X "
            else:
                row += f" {arrows[policy[s]]} "
        print(row)
