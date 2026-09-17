"""
MDP definition
--------------
States (S):
    Every cell (row, col) in an R x C grid is a state, EXCEPT cells that
    are permanently walled off (not used here, but easy to add).
    Obstacle cells and the goal cell ARE states the agent can occupy;
    obstacles are just states that give a negative reward and end the
    episode, rather than being "unreachable" squares.

Actions (A):
    UP, DOWN, LEFT, RIGHT  -> move one cell in the corresponding direction.

Transition probabilities P(s' | s, a):
    - With probability `success_prob` the agent moves in the intended
      direction.
    - With probability (1 - success_prob) the agent "slips" and instead
      moves in one of the two directions perpendicular to the intended
      one (split evenly).
    - If the resulting move would take the agent off the grid, the agent
      stays in place instead (edge case).
    - The goal and every obstacle are TERMINAL states: any action taken
      from them deterministically stays in that same state with 0
      reward (the episode is over, so this is just a formal fix-up so
      the MDP stays well defined).
    - Set success_prob=1.0 for a fully deterministic grid world.

Reward function R(s, a, s'):
    - +1.0  for transitioning INTO the goal state
    - -1.0  for transitioning INTO an obstacle state
    - -0.01 "living cost" for every other move
"""

from enum import Enum
import numpy as np


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# (row_delta, col_delta) for each action
ACTION_DELTAS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}

# perpendicular actions used for "slip" transitions
PERPENDICULAR = {
    Action.UP: (Action.LEFT, Action.RIGHT),
    Action.DOWN: (Action.LEFT, Action.RIGHT),
    Action.LEFT: (Action.UP, Action.DOWN),
    Action.RIGHT: (Action.UP, Action.DOWN),
}


class GridWorld:
    def __init__(
        self,
        n_rows,
        n_cols,
        start,
        goal,
        obstacles=None,
        success_prob=1.0,
        step_reward=-0.01,
        goal_reward=1.0,
        obstacle_reward=-1.0,
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles) if obstacles else set()
        self.success_prob = success_prob
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.obstacle_reward = obstacle_reward

        self.actions = list(Action)
        self.states = [
            (r, c) for r in range(n_rows) for c in range(n_cols)
        ]

        self._agent_pos = start

    def is_terminal(self, state):
        return state == self.goal or state in self.obstacles

    def _move(self, state, action):
        """Apply an action to a state, handling grid edges. No stochasticity here."""
        dr, dc = ACTION_DELTAS[action]
        r, c = state
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
            return (nr, nc)
        return state  # off-grid move -> stay in place

    def transition_probs(self, state, action):
        """
        Return a dict {next_state: probability} for taking `action` in `state`.
        """
        if self.is_terminal(state):
            # terminal states are absorbing
            return {state: 1.0}

        probs = {}

        def add(s, p):
            probs[s] = probs.get(s, 0.0) + p

        # intended direction
        add(self._move(state, action), self.success_prob)

        # slip directions
        slip_prob = 1.0 - self.success_prob
        if slip_prob > 0:
            left_a, right_a = PERPENDICULAR[action]
            add(self._move(state, left_a), slip_prob / 2)
            add(self._move(state, right_a), slip_prob / 2)

        return probs

    def reward(self, state, action, next_state):
        if self.is_terminal(state):
            return 0.0
        if next_state == self.goal:
            return self.goal_reward
        if next_state in self.obstacles:
            return self.obstacle_reward
        return self.step_reward

    def reset(self):
        self._agent_pos = self.start
        return self._agent_pos

    def step(self, action):
        """Sample a transition and return (next_state, reward, done)."""
        probs = self.transition_probs(self._agent_pos, action)
        next_states = list(probs.keys())
        p = list(probs.values())
        next_state = next_states[np.random.choice(len(next_states), p=p)]

        r = self.reward(self._agent_pos, action, next_state)
        self._agent_pos = next_state
        done = self.is_terminal(next_state)
        return next_state, r, done

    def render(self):
        grid = ""
        for r in range(self.n_rows):
            row = ""
            for c in range(self.n_cols):
                cell = (r, c)
                if cell == self._agent_pos:
                    row += " A "
                elif cell == self.goal:
                    row += " G "
                elif cell in self.obstacles:
                    row += " X "
                else:
                    row += " . "
            grid += row + "\n"
        print(grid)
