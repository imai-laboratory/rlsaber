import numpy as np
import random


class ConstantExplorer:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def select_action(self, t, greedy_action, num_actions):
        if random.random() < self.epsilon: 
            return np.random.choice(num_actions)
        return greedy_action

class LinearDecayExplorer:
    def __init__(self, final_exploration_step=10**6,
                start_epsilon=1.0, final_epsilon=0.1):
        self.final_exploration_step = final_exploration_step
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.base_epsilon = self.start_epsilon - self.final_epsilon

    def select_action(self, t, greedy_action, num_actions):
        factor = 1 - float(t) / self.final_exploration_step
        if factor < 0:
            factor = 0
        eps = self.base_epsilon * factor + self.final_epsilon
        if random.random() < eps: 
            return np.random.choice(num_actions)
        return greedy_action
