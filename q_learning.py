import numpy as np
import matplotlib.pyplot as plt
import os

# Define all games as payoff matrices
games = {
    "stag_hunt": {
        "payoffs_p1": np.array([[1, 0], [2/3, 2/3]]),
        "payoffs_p2": np.array([[1, 2/3], [0, 2/3]])
    },
    "subsidy_game": {
        "payoffs_p1": np.array([[12, 0], [11, 10]]),
        "payoffs_p2": np.array([[12, 11], [0, 10]])
    },
    "matching_pennies": {
        "payoffs_p1": np.array([[0, 1], [1, 0]]),
        "payoffs_p2": np.array([[1, 0], [0, 1]])
    },
    "prisoners_dilemma": {
        "payoffs_p1": np.array([[3, 0], [5, 1]]),
        "payoffs_p2": np.array([[3, 5], [0, 1]])
    }
}

# Utility functions
def softmax(q_values, tau):
    exp_q = np.exp(q_values / tau)
    return exp_q / np.sum(exp_q)

# Base Q-learning class
class QLearner:
    def __init__(self, n_actions, lr=0.1, gamma=0.95):
        self.q = np.zeros(n_actions)
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.policy_record = []

    def update(self, a, reward):
        raise NotImplementedError

    def select_action(self):
        raise NotImplementedError

# ε-greedy Q-learning
class EpsilonGreedyQLearner(QLearner):
    def __init__(self, n_actions, epsilon=0.1, **kwargs):
        super().__init__(n_actions, **kwargs)
        self.epsilon = epsilon

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        return np.argmax(self.q)

    def update(self, a, reward):
        self.q[a] += self.lr * (reward - self.q[a])
        self.policy_record.append(self.q.copy())

# Boltzmann Q-learning
class BoltzmannQLearner(QLearner):
    def __init__(self, n_actions, tau=1.0, **kwargs):
        super().__init__(n_actions, **kwargs)
        self.tau = tau

    def select_action(self):
        probs = softmax(self.q, self.tau)
        return np.random.choice(self.n_actions, p=probs)

    def update(self, a, reward):
        self.q[a] += self.lr * (reward - self.q[a])
        self.policy_record.append(softmax(self.q, self.tau))

# Lenient Boltzmann Q-learning
class LenientBoltzmannQLearner(BoltzmannQLearner):
    def __init__(self, n_actions, initial_leniency=1.0, decay=0.999, **kwargs):
        super().__init__(n_actions, **kwargs)
        self.leniency = initial_leniency
        self.decay = decay
        self.visits = np.zeros(n_actions)

    def update(self, a, reward):
        self.visits[a] += 1
        leniency = self.leniency ** self.visits[a]
        optimistic_reward = max(self.q[a], reward)
        self.q[a] += self.lr * (optimistic_reward - self.q[a]) * leniency
        self.policy_record.append(softmax(self.q, self.tau))

# Simulation runner
def run_simulation(game_name, learner_class, steps=1000, **kwargs):
    env = games[game_name]
    p1 = learner_class(n_actions=2, **kwargs)
    p2 = learner_class(n_actions=2, **kwargs)

    for _ in range(steps):
        a1 = p1.select_action()
        a2 = p2.select_action()
        r1 = env["payoffs_p1"][a1, a2]
        r2 = env["payoffs_p2"][a1, a2]
        p1.update(a1, r1)
        p2.update(a2, r2)

    return p1.policy_record, p2.policy_record

# Plotting utility
def plot_and_save(policy1, policy2, game_name, algo_name):
    os.makedirs("plots1000", exist_ok=True)
    policy1 = np.array(policy1)
    policy2 = np.array(policy2)

    plt.figure(figsize=(10, 5))
    plt.plot(policy1[:, 0], label='P1: Action 0')
    plt.plot(policy1[:, 1], label='P1: Action 1')
    plt.plot(policy2[:, 0], '--', label='P2: Action 0')
    plt.plot(policy2[:, 1], '--', label='P2: Action 1')
    plt.title(f"{game_name} - {algo_name}")
    plt.xlabel("Episodes")
    plt.ylabel("Policy Probability / Q-values")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots1000/{game_name}_{algo_name}.png")
    plt.close()

# Main runner
algos = {
    "epsilon_greedy": (EpsilonGreedyQLearner, {"epsilon": 0.1}),
    "boltzmann": (BoltzmannQLearner, {"tau": 0.5}),
    "lenient_boltzmann": (LenientBoltzmannQLearner, {"tau": 0.5, "initial_leniency": 0.99, "decay": 0.999})
}

for game in games:
    for algo_name, (cls, params) in algos.items():
        p1_pol, p2_pol = run_simulation(game, cls, **params)
        plot_and_save(p1_pol, p2_pol, game, algo_name)

print("✅ Done! Check the 'plots' folder for output.")
