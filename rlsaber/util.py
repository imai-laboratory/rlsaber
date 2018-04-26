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
