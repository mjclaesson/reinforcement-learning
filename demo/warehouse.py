"""Environments and training code, generated from multi_agent_nstep.ipynb.

Do not edit by hand: run demo/extract_from_notebook.py instead.
"""
import numpy as np

# ---- from notebook cell starting with: from collections import deque
from collections import deque

import numpy as np

import itertools

import pandas as pd

# ---- from notebook cell starting with: def make_layout(
def make_layout(
    n_shelf_rows=2,
    shelves_per_row=3,
    shelf_width=2,
    shelf_depth=2,
):
    # en gång mer än antalet hyllor, åt båda hållen
    n_rows = n_shelf_rows * shelf_depth + (n_shelf_rows + 1)
    n_cols = shelves_per_row * shelf_width + (shelves_per_row + 1)

    # börja med golv överallt
    grid = np.full((n_rows, n_cols), ".")

    # lägg in hyllorna
    for i in range(n_shelf_rows):
        for j in range(shelves_per_row):
            top = 1 + i * (shelf_depth + 1)
            left = 1 + j * (shelf_width + 1)
            grid[top:top + shelf_depth, left:left + shelf_width] = "#"

    # start uppe till vänster, mål nere till höger
    grid[0, 0] = "S"
    grid[-1, -1] = "G"

    return "\n".join("".join(row) for row in grid)

# ---- from notebook cell starting with: class WarehouseGrid:
class WarehouseGrid:
    def __init__(self, layout=None):
        # standardkartan om ingen annan skickas in
        if layout is None:
            layout = make_layout()

        rows = layout.strip().splitlines()
        self.n_rows = len(rows)
        self.n_cols = len(rows[0])

        # vilket tecken som står i varje cell
        symbols = {(r, c): symbol
                   for r, row in enumerate(rows)
                   for c, symbol in enumerate(row)}

        # sortera cellerna efter tecken
        self.shelf_cells = {cell for cell, s in symbols.items() if s == "#"}
        self.walkable_cells = {cell for cell, s in symbols.items() if s != "#"}
        self.start = next((cell for cell, s in symbols.items() if s == "S"), None)
        self.goal = next((cell for cell, s in symbols.items() if s == "G"), None)

        # plockplatser: golvcellerna bredvid varje hyllcell
        self.pick_spots = {
            shelf: self.walkable_neighbors(shelf) for shelf in self.shelf_cells
        }

        # vilken vara som ligger på vilken hylla
        self.item_to_shelf = {}

        self.check_all_cells_reachable()

    # golvcellerna ovanför, under, till vänster och till höger
    def walkable_neighbors(self, cell):
        r, c = cell
        around = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [n for n in around if n in self.walkable_cells]

    # felmeddelande om världen är uppdelad eller en hylla inte går att nå
    def check_all_cells_reachable(self):
        start = next(iter(self.walkable_cells))
        visited = {start}
        queue = deque([start])
        while queue:
            for n in self.walkable_neighbors(queue.popleft()):
                if n not in visited:
                    visited.add(n)
                    queue.append(n)

        # celler som aldrig nåddes betyder att världen är uppdelad
        if visited != self.walkable_cells:
            raise ValueError(f"Här kan du inte gå för fan! {sorted(self.walkable_cells - visited)}")

        # varje hylla måste ha minst en plockplats
        for shelf, spots in self.pick_spots.items():
            if not spots:
                raise ValueError(f"Hyllan {shelf} har ingen plockplats.")

    # lägg en vara på en hylla
    def add_item(self, item, shelf):
        if shelf not in self.shelf_cells:
            raise ValueError(f"{shelf} är ingen hylla!")
        self.item_to_shelf[item] = shelf

    # lägg varje vara på en slumpad hylla, max en vara per hylla
    def place_items_randomly(self, items, seed=None):
        rng = np.random.default_rng(seed)
        shelves = sorted(self.shelf_cells)
        chosen = rng.choice(len(shelves), size=len(items), replace=False)
        for item, i in zip(items, chosen):
            self.add_item(item, shelves[i])

    # var man ska stå för att plocka en vara
    def pick_spots_for_item(self, item):
        return self.pick_spots[self.item_to_shelf[item]]

    # rita världen, lager för lager
    def render(self, show_pick_spots=False):
        grid = np.full((self.n_rows, self.n_cols), ".")

        if show_pick_spots:
            for spots in self.pick_spots.values():
                for cell in spots:
                    grid[cell] = "p"

        for cell in self.shelf_cells:
            grid[cell] = "#"

        if self.start is not None:
            grid[self.start] = "S"
        if self.goal is not None:
            grid[self.goal] = "G"

        for row in grid:
            print(" ".join(row))
        print()

# ---- from notebook cell starting with: STEP_REWARD = -1
STEP_REWARD = -1

CRASH_REWARD = -5

ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
    "WAIT": (0, 0),
}

class TwoAgentWarehouse:
    def __init__(self, grid, starts, goals, reward_mode="selfish"):
        self.grid = grid        # Jockes WarehouseGrid, hyllor och golv
        self.starts = starts    # en start per agent, t.ex. [(0, 0), (0, 9)]
        self.goals = goals      # ett mål per agent
        # "selfish": 0 när man själv är i mål. "cooperative": -1 tills båda är i mål
        if reward_mode not in ("selfish", "cooperative"):
            raise ValueError(f"Okänt reward_mode: {reward_mode}")
        self.reward_mode = reward_mode
        self.reset()

    # ställ båda agenterna på sin start igen
    def reset(self):
        self.positions = list(self.starts)
        return tuple(self.positions)

    # rita världen med agenterna A och B, och deras mål a och b
    def render(self):
        grid = np.full((self.grid.n_rows, self.grid.n_cols), ".")

        for cell in self.grid.shelf_cells:
            grid[cell] = "#"

        for goal, name in zip(self.goals, "ab"):
            grid[goal] = name
        for pos, name in zip(self.positions, "AB"):
            grid[pos] = name

        for row in grid:
            print(" ".join(row))
        print()

    # flytta en agent ett steg, stå kvar om det är hylla eller utanför kartan
    def move(self, pos, action):
        dr, dc = ACTIONS[action]
        new = (pos[0] + dr, pos[1] + dc)
        if new in self.grid.walkable_cells:
            return new
        return pos

    # båda agenterna tar en handling samtidigt, t.ex. actions = ("RIGHT", "WAIT")
    def step(self, actions):
        old = list(self.positions)

        # vart vill agenterna gå? en agent som redan är i mål står still
        wanted = []
        for i in range(2):
            if old[i] == self.goals[i]:
                wanted.append(old[i])
            else:
                wanted.append(self.move(old[i], actions[i]))

        # samma ruta eller byta plats med varandra -> båda står kvar
        same_cell = wanted[0] == wanted[1]
        swap = wanted[0] == old[1] and wanted[1] == old[0]
        if same_cell or swap:
            new = old
        else:
            new = wanted

        done = all(p == g for p, g in zip(new, self.goals))

        # rewards, en per agent
        rewards = []
        for i in range(2):
            if old[i] == self.goals[i]:
                if self.reward_mode == "selfish" or done:
                    rewards.append(0)  # redan klar, inget mer minus
                else:
                    rewards.append(STEP_REWARD)  # klar, men väntar på den andra
                continue
            reward = STEP_REWARD
            # ville gå men blev stoppad av den andra agenten
            if wanted[i] != old[i] and new[i] == old[i]:
                reward += CRASH_REWARD
            rewards.append(reward)

        self.positions = list(new)
        return tuple(self.positions), rewards, done

# ---- from notebook cell starting with: from collections import defaultdict
from collections import defaultdict

ACTION_NAMES = list(ACTIONS)

def choose_action(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(len(ACTION_NAMES))
    q = Q[state]
    best = np.flatnonzero(q == q.max())  # om flera är lika bra, slumpa bland dem
    return rng.choice(best)

def train(env, episodes=10000, alpha=0.5, gamma=0.99, epsilon=1.0, max_steps=200, seed=0, epsilon_end=0.05, decay_fraction=0.8):
    rng = np.random.default_rng(seed)
    Qs = [defaultdict(lambda: np.zeros(len(ACTION_NAMES))) for _ in range(2)]
    steps_per_episode = []
    decay = (epsilon_end / epsilon) ** (1 / (decay_fraction * episodes))

    for episode in range(episodes):
        state = env.reset()
        for t in range(max_steps):
            a = [choose_action(Qs[i], state, epsilon, rng) for i in range(2)]
            next_state, rewards, done = env.step((ACTION_NAMES[a[0]], ACTION_NAMES[a[1]]))

            for i in range(2):
                if done:
                    target = rewards[i]  # episoden slut, ingen framtid kvar
                else:
                    target = rewards[i] + gamma * Qs[i][next_state].max()
                Qs[i][state][a[i]] += alpha * (target - Qs[i][state][a[i]])

            state = next_state
            if done:
                break
        steps_per_episode.append(t + 1)
        epsilon = max(epsilon_end, epsilon * decay)

    return Qs, steps_per_episode

# ---- from notebook cell starting with: # kör en episod utan slump och visa vad agenterna lärt sig
def run_greedy(env, Qs, max_steps=50, show=True):
    rng = np.random.default_rng(0)
    state = env.reset()
    finished = [None, None]  # vilket steg varje agent kom i mål
    crashes = 0
    if show:
        env.render()

    for t in range(1, max_steps + 1):
        a = [choose_action(Qs[i], state, 0.0, rng) for i in range(2)]
        state, rewards, done = env.step((ACTION_NAMES[a[0]], ACTION_NAMES[a[1]]))
        crashes += sum(r < STEP_REWARD for r in rewards)
        for i in range(2):
            if finished[i] is None and state[i] == env.goals[i]:
                finished[i] = t
        if show:
            print(f"steg {t}: A {ACTION_NAMES[a[0]]}, B {ACTION_NAMES[a[1]]}")
            env.render()
        if done:
            break

    print(f"A i mål efter {finished[0]} steg, B efter {finished[1]} steg, {crashes} krockar")

# ---- from notebook cell starting with: # uppdatera den äldsta posten i bufferten med en n-stegs ret
def nstep_update(Q, buffer, bootstrap_state, alpha, gamma):
    G = 0.0
    if bootstrap_state is not None:
        G = Q[bootstrap_state].max()
    # räkna bakifrån: G = r + gamma * G, så gamma^len hamnar på lånet
    for _, _, r in reversed(buffer):
        G = r + gamma * G
    s, a, _ = buffer[0]
    Q[s][a] += alpha * (G - Q[s][a])

def train_nstep(env, n=5, episodes=10000, alpha=0.5, gamma=0.99, epsilon=1.0, max_steps=200, seed=0, epsilon_end=0.05, decay_fraction=0.8):
    rng = np.random.default_rng(seed)
    Qs = [defaultdict(lambda: np.zeros(len(ACTION_NAMES))) for _ in range(2)]
    steps_per_episode = []
    decay = (epsilon_end / epsilon) ** (1 / (decay_fraction * episodes))

    for episode in range(episodes):
        state = env.reset()
        buffers = [[], []]  # varje agents senaste (state, action, reward)

        for t in range(max_steps):
            a = [choose_action(Qs[i], state, epsilon, rng) for i in range(2)]
            next_state, rewards, done = env.step((ACTION_NAMES[a[0]], ACTION_NAMES[a[1]]))

            for i in range(2):
                buffers[i].append((state, a[i], rewards[i]))

                if done:
                    # episoden slut: uppdatera alla kvarvarande steg utan att låna
                    while buffers[i]:
                        nstep_update(Qs[i], buffers[i], None, alpha, gamma)
                        buffers[i].pop(0)
                elif len(buffers[i]) == n:
                    # bufferten full: uppdatera äldsta steget och släpp det
                    nstep_update(Qs[i], buffers[i], next_state, alpha, gamma)
                    buffers[i].pop(0)

            state = next_state
            if done:
                break

        # tidsgränsen nåddes: lån från sista staten för det som är kvar
        for i in range(2):
            while buffers[i]:
                nstep_update(Qs[i], buffers[i], state, alpha, gamma)
                buffers[i].pop(0)

        steps_per_episode.append(t + 1)
        epsilon = max(epsilon_end, epsilon * decay)

    return Qs, steps_per_episode

# ---- from notebook cell starting with: # kör en girig episod tyst och returnera (steg tills båda i 
def greedy_result(env, Qs, max_steps=50):
    rng = np.random.default_rng(0)
    state = env.reset()
    crashes = 0
    for t in range(1, max_steps + 1):
        a = [choose_action(Qs[i], state, 0.0, rng) for i in range(2)]
        state, rewards, done = env.step((ACTION_NAMES[a[0]], ACTION_NAMES[a[1]]))
        crashes += sum(r < STEP_REWARD for r in rewards)
        if done:
            return t, crashes
    return None, crashes

# ---- from notebook cell starting with: # kör en girig episod tyst: (steg när A kom i mål, steg när 
def greedy_finish_times(env, Qs, max_steps=50):
    rng = np.random.default_rng(0)
    state = env.reset()
    finished = [None, None]
    crashes = 0
    for t in range(1, max_steps + 1):
        a = [choose_action(Qs[i], state, 0.0, rng) for i in range(2)]
        state, rewards, done = env.step((ACTION_NAMES[a[0]], ACTION_NAMES[a[1]]))
        crashes += sum(r < STEP_REWARD for r in rewards)
        for i in range(2):
            if finished[i] is None and state[i] == env.goals[i]:
                finished[i] = t
        if done:
            break
    return finished[0], finished[1], crashes

# ---- from notebook cell starting with: PICK_REWARD = 5
PICK_REWARD = 5

DELIVER_REWARD = 10

BAD_PICK_PENALTY = -2

ACTIONS_PICK = list(ACTIONS) + ["PICK"]

class PickupWarehouse:
    def __init__(self, grid, items, start=None, dropoff=None, finish=None):
        self.grid = grid                # WarehouseGrid med varorna utplacerade
        self.items = list(items)        # t.ex. [0, 1, 2], plats i listan = bit i masken
        self.start = grid.start if start is None else start
        # lämningsplats: nedre mitten om inget annat anges
        self.dropoff = (grid.n_rows - 1, grid.n_cols // 2) if dropoff is None else dropoff
        self.finish = finish            # None = klart direkt när sista varan är levererad
        self.full_mask = (1 << len(self.items)) - 1

        for item in self.items:
            if item not in grid.item_to_shelf:
                raise ValueError(f"Vara {item} ligger inte på någon hylla.")
        if self.dropoff not in grid.walkable_cells:
            raise ValueError(f"Lämningsplatsen {self.dropoff} är inte en golvruta.")
        self.reset()

    def reset(self):
        self.pos = self.start
        self.mask = 0              # varor som lämnat hyllan
        self.carrying = False      # bär agenten på en vara?
        return (self.pos, self.mask, self.carrying)

    # räkna ut vad en handling gör, utan att ändra något
    def transition(self, pos, mask, carrying, action):
        reward = STEP_REWARD
        if action == "PICK":
            picked = False
            if not carrying:
                for i, item in enumerate(self.items):
                    already_gone = mask & (1 << i)
                    if not already_gone and pos in self.grid.pick_spots_for_item(item):
                        mask |= 1 << i
                        carrying = True
                        picked = True
                        reward += PICK_REWARD
                        break      # en vara per PICK
            if not picked:
                reward += BAD_PICK_PENALTY   # ingen vara här, eller händerna är fulla
        else:
            dr, dc = ACTIONS[action]
            new = (pos[0] + dr, pos[1] + dc)
            if new in self.grid.walkable_cells:   # hylla eller utanför -> står kvar
                pos = new
            if carrying and pos == self.dropoff:  # leverera automatiskt
                carrying = False
                reward += DELIVER_REWARD

        all_delivered = mask == self.full_mask and not carrying
        done = all_delivered and (self.finish is None or pos == self.finish)
        return pos, mask, carrying, reward, done

    def step(self, action):
        self.pos, self.mask, self.carrying, reward, done = self.transition(
            self.pos, self.mask, self.carrying, action)
        return (self.pos, self.mask, self.carrying), reward, done

    def n_delivered(self):
        return bin(self.mask).count("1") - int(self.carrying)

    # rita världen: S = start, D = lämningsplats, A = agenten (C när den bär en vara),
    # siffra = vara kvar på hyllan, x = vara som lämnat hyllan
    def render(self):
        grid = np.full((self.grid.n_rows, self.grid.n_cols), ".")
        for cell in self.grid.shelf_cells:
            grid[cell] = "#"
        for i, item in enumerate(self.items):
            shelf = self.grid.item_to_shelf[item]
            grid[shelf] = "x" if self.mask & (1 << i) else str(item)
        grid[self.start] = "S"
        grid[self.dropoff] = "D"
        grid[self.pos] = "C" if self.carrying else "A"
        for row in grid:
            print(" ".join(row))
        print()

def optimal_steps(env):
    first = (env.start, 0, False)
    seen = {first}
    queue = deque([(first, 0)])
    while queue:
        (pos, mask, carrying), d = queue.popleft()
        for action in ACTIONS_PICK:
            new_pos, new_mask, new_carrying, _, done = env.transition(pos, mask, carrying, action)
            if done:
                return d + 1
            nxt = (new_pos, new_mask, new_carrying)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, d + 1))
    return None

# ---- from notebook cell starting with: # som choose_action, men slumpar bland alla handlingar inklu
def choose_action_pick(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(len(ACTIONS_PICK))
    q = Q[state]
    best = np.flatnonzero(q == q.max())
    return rng.choice(best)

def train_pickup(env, n=1, episodes=3000, alpha=0.5, gamma=0.99, epsilon=1.0, max_steps=300, seed=0, epsilon_end=0.05, decay_fraction=0.8):
    rng = np.random.default_rng(seed)
    Q = defaultdict(lambda: np.zeros(len(ACTIONS_PICK)))
    steps_per_episode = []
    decay = (epsilon_end / epsilon) ** (1 / (decay_fraction * episodes))

    for episode in range(episodes):
        state = env.reset()
        buffer = []

        for t in range(max_steps):
            a = choose_action_pick(Q, state, epsilon, rng)
            next_state, reward, done = env.step(ACTIONS_PICK[a])
            buffer.append((state, a, reward))

            if done:
                # klar: uppdatera allt som är kvar utan att låna
                while buffer:
                    nstep_update(Q, buffer, None, alpha, gamma)
                    buffer.pop(0)
                break
            if len(buffer) == n:
                nstep_update(Q, buffer, next_state, alpha, gamma)
                buffer.pop(0)
            state = next_state

        # tidsgränsen nåddes: lån från sista staten
        while buffer:
            nstep_update(Q, buffer, state, alpha, gamma)
            buffer.pop(0)

        steps_per_episode.append(t + 1)
        epsilon = max(epsilon_end, epsilon * decay)

    return Q, steps_per_episode

def run_greedy_pickup(env, Q, max_steps=150, show=True):
    rng = np.random.default_rng(0)
    state = env.reset()
    if show:
        env.render()
    bad_picks = 0
    for t in range(1, max_steps + 1):
        a = choose_action_pick(Q, state, 0.0, rng)
        picked_before, delivered_before = bin(env.mask).count("1"), env.n_delivered()
        state, reward, done = env.step(ACTIONS_PICK[a])
        note = ""
        if bin(env.mask).count("1") > picked_before:
            note = "  <- plockade en vara"
        elif env.n_delivered() > delivered_before:
            note = "  <- levererade"
        elif ACTIONS_PICK[a] == "PICK":
            note = "  <- onödig PICK (straff)"
            bad_picks += 1
        if show:
            print(f"steg {t:2d}: {ACTIONS_PICK[a]:5s} -> {state[0]}{note}")
        if done:
            break
    if show:
        print()
        env.render()
    return (t if done else None), env.n_delivered(), bad_picks

# ---- from notebook cell starting with: class TwoAgentPickupWarehouse:
class TwoAgentPickupWarehouse:
    """Två agenter i PickupWarehouse-reglerna. Varje agent har egna varor, egen start (= mål)
    och de delar lämningsplatsen D. Krockregler som i TwoAgentWarehouse."""

    def __init__(self, grid, items_per_agent, starts, dropoff=None, reward_mode="selfish"):
        self.grid = grid
        self.items = [list(items) for items in items_per_agent]   # t.ex. [[0, 1], [2, 3]]
        self.starts = list(starts)                                 # en start per agent, också mål
        self.dropoff = (grid.n_rows - 1, grid.n_cols // 2) if dropoff is None else dropoff
        self.full_masks = [(1 << len(items)) - 1 for items in self.items]
        if reward_mode not in ("selfish", "cooperative"):
            raise ValueError(f"Okänt reward_mode: {reward_mode}")
        self.reward_mode = reward_mode

        for items in self.items:
            for item in items:
                if item not in grid.item_to_shelf:
                    raise ValueError(f"Vara {item} ligger inte på någon hylla.")
        if self.dropoff not in grid.walkable_cells:
            raise ValueError(f"Lämningsplatsen {self.dropoff} är inte en golvruta.")
        self.reset()

    def reset(self):
        self.pos = list(self.starts)
        self.mask = [0, 0]
        self.carrying = [False, False]
        return self.state()

    # hela tillståndet, används av render och done
    def state(self):
        return tuple((self.pos[i], self.mask[i], self.carrying[i]) for i in range(2))

    # det agent i ser: sitt eget tillstånd plus var den andra står
    def obs(self, i):
        return (self.pos[i], self.mask[i], self.carrying[i], self.pos[1 - i])

    # är agent i helt klar? allt levererat och tillbaka på sin start
    def finished(self, i):
        return (self.mask[i] == self.full_masks[i]
                and not self.carrying[i]
                and self.pos[i] == self.starts[i])

    def step(self, actions):
        old = list(self.pos)

        # 1. vart vill agenterna gå? PICK och WAIT = stå kvar, en klar agent står kvar
        wanted = []
        for i in range(2):
            if self.finished(i) or actions[i] == "PICK":
                wanted.append(old[i])
            else:
                dr, dc = ACTIONS[actions[i]]
                new = (old[i][0] + dr, old[i][1] + dc)
                wanted.append(new if new in self.grid.walkable_cells else old[i])

        # 2. samma ruta eller byta plats -> båda står kvar
        same_cell = wanted[0] == wanted[1]
        swap = wanted[0] == old[1] and wanted[1] == old[0]
        new = old if (same_cell or swap) else wanted

        # 3. belöning per agent: steg, krock, plock, leverans
        rewards = []
        for i in range(2):
            if self.finished(i):
                rewards.append(0)
                continue
            reward = STEP_REWARD
            if wanted[i] != old[i] and new[i] == old[i]:
                reward += CRASH_REWARD           # blev stoppad av den andra

            if actions[i] == "PICK":
                picked = False
                if not self.carrying[i]:
                    for b, item in enumerate(self.items[i]):
                        already_gone = self.mask[i] & (1 << b)
                        if not already_gone and old[i] in self.grid.pick_spots_for_item(item):
                            self.mask[i] |= 1 << b
                            self.carrying[i] = True
                            picked = True
                            reward += PICK_REWARD
                            break
                if not picked:
                    reward += BAD_PICK_PENALTY
            elif self.carrying[i] and new[i] == self.dropoff:
                self.carrying[i] = False
                reward += DELIVER_REWARD
            rewards.append(reward)

        self.pos = list(new)
        done = self.finished(0) and self.finished(1)

        # cooperative: den som är klar får ändå -1 tills båda är klara
        if self.reward_mode == "cooperative" and not done:
            for i in range(2):
                if self.finished(i) and rewards[i] == 0:
                    rewards[i] = STEP_REWARD
        return self.state(), rewards, done

    def n_delivered(self, i):
        return bin(self.mask[i]).count("1") - int(self.carrying[i])

    # S/T = starter, D = lämningsplats, A/B = agenterna (a/b när de bär), siffra = vara kvar, x = plockad
    def render(self):
        grid = np.full((self.grid.n_rows, self.grid.n_cols), ".")
        for cell in self.grid.shelf_cells:
            grid[cell] = "#"
        for i in range(2):
            for b, item in enumerate(self.items[i]):
                grid[self.grid.item_to_shelf[item]] = "x" if self.mask[i] & (1 << b) else str(item)
        for start, name in zip(self.starts, "ST"):
            grid[start] = name
        grid[self.dropoff] = "D"
        for i, name in enumerate("AB"):
            grid[self.pos[i]] = name.lower() if self.carrying[i] else name
        for row in grid:
            print(" ".join(row))
        print()

def lower_bound_steps(env):
    bounds = []
    for i in range(2):
        solo = PickupWarehouse(env.grid, items=env.items[i], start=env.starts[i],
                               dropoff=env.dropoff, finish=env.starts[i])
        bounds.append(optimal_steps(solo))
    return bounds

# ---- from notebook cell starting with: # n-stegs Q-learning för två plockande agenter, en Q-tabell 
def train_two_pickup(env, n=1, episodes=30000, alpha=0.5, gamma=0.99, epsilon=1.0,
                     max_steps=300, seed=0, q_init=0.0, epsilon_end=0.05, decay_fraction=0.8):
    rng = np.random.default_rng(seed)
    # float(q_init): annars blir Q en heltalsarray och alla uppdateringar trunkeras
    Qs = [defaultdict(lambda: np.full(len(ACTIONS_PICK), float(q_init))) for _ in range(2)]
    steps_per_episode = []
    decay = (epsilon_end / epsilon) ** (1 / (decay_fraction * episodes))

    for episode in range(episodes):
        env.reset()
        obs = [env.obs(0), env.obs(1)]
        buffers = [[], []]

        for t in range(max_steps):
            a = [choose_action_pick(Qs[i], obs[i], epsilon, rng) for i in range(2)]
            _, rewards, done = env.step((ACTIONS_PICK[a[0]], ACTIONS_PICK[a[1]]))
            next_obs = [env.obs(0), env.obs(1)]

            for i in range(2):
                buffers[i].append((obs[i], a[i], rewards[i]))
                if done:
                    while buffers[i]:
                        nstep_update(Qs[i], buffers[i], None, alpha, gamma)
                        buffers[i].pop(0)
                elif len(buffers[i]) == n:
                    nstep_update(Qs[i], buffers[i], next_obs[i], alpha, gamma)
                    buffers[i].pop(0)

            obs = next_obs
            if done:
                break

        for i in range(2):
            while buffers[i]:
                nstep_update(Qs[i], buffers[i], obs[i], alpha, gamma)
                buffers[i].pop(0)
        steps_per_episode.append(t + 1)
        epsilon = max(epsilon_end, epsilon * decay)

    return Qs, steps_per_episode

def run_greedy_two_pickup(env, Qs, max_steps=200, show=False):
    rng = np.random.default_rng(0)
    env.reset()
    finished = [None, None]
    crashes = 0
    if show:
        env.render()
    for t in range(1, max_steps + 1):
        a = [choose_action_pick(Qs[i], env.obs(i), 0.0, rng) for i in range(2)]
        _, rewards, done = env.step((ACTIONS_PICK[a[0]], ACTIONS_PICK[a[1]]))
        crashes += sum(r <= STEP_REWARD + CRASH_REWARD for r in rewards)
        for i in range(2):
            if finished[i] is None and env.finished(i):
                finished[i] = t
        if show:
            print(f"steg {t}: A {ACTIONS_PICK[a[0]]}, B {ACTIONS_PICK[a[1]]}  rewards {rewards}")
            env.render()
        if done:
            break
    return finished[0], finished[1], crashes, [env.n_delivered(0), env.n_delivered(1)]
