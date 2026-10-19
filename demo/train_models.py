"""Train every model the demo app shows, once, and save greedy runs as JSON.

    .venv\\Scripts\\python demo\\train_models.py        (about 5 minutes)

Output: demo/models/runs.json. The app only replays these frames; it never trains
(except the small live demo on the narrow corridor).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import warehouse as w  # noqa: E402

OUT = HERE / "models" / "runs.json"


def grid_rows(grid):
    rows = []
    for r in range(grid.n_rows):
        rows.append("".join("#" if (r, c) in grid.shelf_cells else "." for c in range(grid.n_cols)))
    return rows


def curve(steps, block):
    return [float(np.mean(steps[i:i + block])) for i in range(0, len(steps), block)]


# ---------- greedy runs that record every frame ----------

def replay_two_nav(env, Qs, max_steps=50):
    rng = np.random.default_rng(0)
    state = env.reset()
    frames = [{"pos": [list(p) for p in state], "actions": None, "rewards": None, "events": []}]
    finished = [None, None]
    crashes = 0
    for t in range(1, max_steps + 1):
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
    return frames, {"A_done": finished[0], "B_done": finished[1], "collisions": crashes}


def replay_pickup(env, Q, max_steps=150):
    rng = np.random.default_rng(0)
    state = env.reset()
    frames = [{"pos": [list(env.pos)], "mask": [env.mask], "carrying": [env.carrying],
               "actions": None, "rewards": None, "events": []}]
    finished = None
    for t in range(1, max_steps + 1):
        a = w.choose_action_pick(Q, state, 0.0, rng)
        picked_before, delivered_before = bin(env.mask).count("1"), env.n_delivered()
        state, reward, done = env.step(w.ACTIONS_PICK[a])
        events = []
        if bin(env.mask).count("1") > picked_before:
            events.append("picked an item (+5)")
        elif env.n_delivered() > delivered_before:
            events.append("delivered (+10)")
        elif w.ACTIONS_PICK[a] == "PICK":
            events.append("useless PICK (-2)")
        frames.append({"pos": [list(env.pos)], "mask": [env.mask], "carrying": [env.carrying],
                       "actions": [w.ACTIONS_PICK[a]], "rewards": [reward], "events": events})
        if done:
            finished = t
            break
    return frames, {"A_done": finished, "delivered": env.n_delivered()}


def replay_two_pickup(env, Qs, max_steps=200):
    rng = np.random.default_rng(0)
    env.reset()
    frames = [{"pos": [list(p) for p in env.pos], "mask": list(env.mask), "carrying": list(env.carrying),
               "actions": None, "rewards": None, "events": []}]
    finished = [None, None]
    crashes = 0
    for t in range(1, max_steps + 1):
        a = [w.choose_action_pick(Qs[i], env.obs(i), 0.0, rng) for i in range(2)]
        names = [w.ACTIONS_PICK[a[0]], w.ACTIONS_PICK[a[1]]]
        picked_before = [bin(m).count("1") for m in env.mask]
        delivered_before = [env.n_delivered(0), env.n_delivered(1)]
        _, rewards, done = env.step(tuple(names))
        events = []
        for i in range(2):
            name = "AB"[i]
            if rewards[i] <= w.STEP_REWARD + w.CRASH_REWARD:
                events.append(f"{name} blocked (-5)")
                crashes += 1
            if bin(env.mask[i]).count("1") > picked_before[i]:
                events.append(f"{name} picked an item (+5)")
            elif env.n_delivered(i) > delivered_before[i]:
                events.append(f"{name} delivered (+10)")
            elif names[i] == "PICK":
                events.append(f"{name} useless PICK (-2)")
            if finished[i] is None and env.finished(i):
                finished[i] = t
                events.append(f"{name} finished")
        frames.append({"pos": [list(p) for p in env.pos], "mask": list(env.mask), "carrying": list(env.carrying),
                       "actions": names, "rewards": rewards, "events": events})
        if done:
            break
    return frames, {"A_done": finished[0], "B_done": finished[1], "collisions": crashes,
                    "delivered": [env.n_delivered(0), env.n_delivered(1)]}


# ---------- the runs ----------

def main():
    runs = {}
    t0 = time.time()

    # 1. two agents, open map (step 3 in the notebook)
    env = w.TwoAgentWarehouse(w.WarehouseGrid(), starts=[(0, 0), (0, 9)], goals=[(6, 9), (6, 0)])
    Qs, steps = w.train(env)
    frames, summary = replay_two_nav(env, Qs)
    runs["nav_open"] = {
        "title": "Two agents, open map", "kind": "two_nav", "reward_mode": "selfish",
        "grid": grid_rows(env.grid), "starts": env.starts, "goals": env.goals,
        "frames": frames, "summary": summary, "curve": curve(steps, 500), "curve_block": 500,
        "optimal": 15, "settings": "Q-learning, 10 000 episodes, alpha 0.5, gamma 0.99, seed 0",
    }
    print("nav_open", summary, f"{time.time() - t0:.0f}s")

    # 2. narrow corridor, selfish and cooperative (steps 4 and 5b), seed 1
    layout = "####.####\n........."
    for mode in ["selfish", "cooperative"]:
        env = w.TwoAgentWarehouse(w.WarehouseGrid(layout), starts=[(1, 0), (1, 8)], goals=[(1, 8), (1, 0)],
                                  reward_mode=mode)
        Qs, steps = w.train(env, seed=1)
        frames, summary = replay_two_nav(env, Qs)
        runs[f"narrow_{mode}"] = {
            "title": f"Narrow corridor, {mode}", "kind": "two_nav", "reward_mode": mode,
            "grid": grid_rows(env.grid), "starts": env.starts, "goals": env.goals,
            "frames": frames, "summary": summary, "curve": curve(steps, 500), "curve_block": 500,
            "optimal": 10, "settings": "Q-learning, 10 000 episodes, alpha 0.5, gamma 0.99, seed 1",
        }
        print(f"narrow_{mode}", summary, f"{time.time() - t0:.0f}s")

    # 3. one agent, pickup (step 6)
    grid = w.WarehouseGrid(w.make_layout(n_shelf_rows=4, shelves_per_row=2, shelf_width=3, shelf_depth=1))
    grid.place_items_randomly([0, 1, 2], seed=1)
    env = w.PickupWarehouse(grid, items=[0, 1, 2], finish=(0, 0))
    best = w.optimal_steps(env)
    Q, steps = w.train_pickup(env, n=1, alpha=0.5, episodes=5000, seed=0)
    frames, summary = replay_pickup(env, Q)
    runs["pickup_one"] = {
        "title": "One agent, pickup", "kind": "pickup", "reward_mode": "selfish",
        "grid": grid_rows(grid), "items": {str(k): list(v) for k, v in grid.item_to_shelf.items()},
        "items_per_agent": [[0, 1, 2]], "starts": [env.start], "dropoff": list(env.dropoff),
        "frames": frames, "summary": summary, "curve": curve(steps, 250), "curve_block": 250,
        "optimal": best, "settings": "Q-learning (n=1), 5 000 episodes, alpha 0.5, gamma 0.99, seed 0",
    }
    print("pickup_one", summary, "optimum", best, f"{time.time() - t0:.0f}s")

    # 4. two agents, pickup, best hyperparameters per reward mode (step 7 + tuning)
    grid2 = w.WarehouseGrid(w.make_layout(n_shelf_rows=4, shelves_per_row=2, shelf_width=3, shelf_depth=1))
    grid2.place_items_randomly([0, 1, 2, 3], seed=1)
    configs = {
        "selfish": dict(n=1, alpha=0.1, gamma=0.99, epsilon_end=0.05),
        "cooperative": dict(n=3, alpha=0.1, gamma=0.99, epsilon_end=0.2),
    }
    for mode, cfg in configs.items():
        env = w.TwoAgentPickupWarehouse(grid2, items_per_agent=[[0, 1], [2, 3]], starts=[(0, 0), (0, 8)],
                                        reward_mode=mode)
        bounds = w.lower_bound_steps(env)
        Qs, steps = w.train_two_pickup(env, q_init=-30, seed=0, episodes=30000, **cfg)
        frames, summary = replay_two_pickup(env, Qs)
        runs[f"two_pickup_{mode}"] = {
            "title": f"Two agents, pickup, {mode}", "kind": "two_pickup", "reward_mode": mode,
            "grid": grid_rows(grid2), "items": {str(k): list(v) for k, v in grid2.item_to_shelf.items()},
            "items_per_agent": [[0, 1], [2, 3]], "starts": env.starts, "dropoff": list(env.dropoff),
            "frames": frames, "summary": summary, "curve": curve(steps, 1000), "curve_block": 1000,
            "lower_bound": list(bounds),
            "settings": (f"n-step Q-learning, n={cfg['n']}, alpha {cfg['alpha']}, gamma {cfg['gamma']}, "
                         f"final epsilon {cfg['epsilon_end']}, Q init -30, 30 000 episodes, seed 0"),
        }
        print(f"two_pickup_{mode}", summary, "bounds", bounds, f"{time.time() - t0:.0f}s")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(runs), encoding="utf-8")
    print("wrote", OUT, f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
