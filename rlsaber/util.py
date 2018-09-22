import numpy as np
import scipy.signal


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def compute_v_and_adv(rewards, values, bootstrapped_value, gamma, lam=1.0):
    rewards = np.array(rewards)
    values = np.array(list(values) + [bootstrapped_value])
    v = discount(np.array(list(rewards) + [bootstrapped_value]), gamma)[:-1]
    delta = rewards + gamma * values[1:] - values[:-1]
    adv = discount(delta, gamma * lam)
    return v, adv

def compute_returns(rewards, bootstrap_value, terminals, gamma):
    # (N, T) -> (T, N)
    rewards = np.transpose(rewards, [1, 0])
    terminals = np.transpose(terminals, [1, 0])
    returns = []
    R = bootstrap_value
    for i in reversed(range(rewards.shape[0])):
        R = rewards[i] + (1.0 - terminals[i]) * gamma * R
        returns.append(R)
    returns = reversed(returns)
    # (T, N) -> (N, T)
    returns = np.transpose(list(returns), [1, 0])
    return returns

def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    # (N, T) -> (T, N)
    rewards = np.transpose(rewards, [1, 0])
    values = np.transpose(values, [1, 0])
    values = np.vstack((values, [bootstrap_values]))
    terminals = np.transpose(terminals, [1, 0])
    # compute delta
    deltas = []
    for i in reversed(range(rewards.shape[0])):
        V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))
    # compute gae
    A = deltas[-1,:]
    advantages = [A]
    for i in reversed(range(deltas.shape[0] - 1)):
        A = deltas[i] + (1.0 - terminals[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)
    # (T, N) -> (N, T)
    advantages = np.transpose(list(advantages), [1, 0])
    return advantages

class Rollout:
    def __init__(self):
        self.flush()

    def add(self, state, action, reward, value, terminal=False, feature=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.terminals.append(terminal)
        self.features.append(feature)

    def flush(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.terminals = []
        self.features = []
