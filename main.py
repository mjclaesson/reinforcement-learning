from gridworld import GridWorld
from value_iteration import value_iteration, print_policy


def main():
    env = GridWorld(
        n_rows=4,
        n_cols=4,
        start=(3, 0),
        goal=(0, 3),
        obstacles=[(1, 1), (2, 2)],
        success_prob=1.0,   # 0.8 intended move, 0.2 split between the two slip directions
        step_reward=-0.01,
        goal_reward=1.0,
        obstacle_reward=-1.0,
    )

    V, policy = value_iteration(env, gamma=0.95)

    print("Learned policy (^ v < > = greedy action, G = goal, X = obstacle):")
    print_policy(env, policy)

    print("\nState values:")
    for r in range(env.n_rows):
        print(["{:6.2f}".format(V[(r, c)]) for c in range(env.n_cols)])

    print("\nRunning one episode with the learned policy:")
    state = env.reset()
    env.render()
    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < 50:
        action = policy[state]
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        env.render()

    print(f"Episode finished in {steps} steps, total reward = {total_reward:.2f}")


if __name__ == "__main__":
    main()