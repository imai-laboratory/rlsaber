import gym
import gym_pull
from gym.spaces import Discrete

gym_pull.pull('github.com/ppaquette/gym-super-mario')

from ppaquette_gym_super_mario import wrappers

# manual controller mapping to discrete actions
# beucase gym-super-mario uses DiscreteToMultiDiscrete
# which is removed in the latest gym
action_mapping = {
    0:  [0, 0, 0, 0, 0, 0],  # NOOP
    1:  [1, 0, 0, 0, 0, 0],  # Up
    2:  [0, 0, 1, 0, 0, 0],  # Down
    3:  [0, 1, 0, 0, 0, 0],  # Left
    4:  [0, 1, 0, 0, 1, 0],  # Left + A
    5:  [0, 1, 0, 0, 0, 1],  # Left + B
    6:  [0, 1, 0, 0, 1, 1],  # Left + A + B
    7:  [0, 0, 0, 1, 0, 0],  # Right
    8:  [0, 0, 0, 1, 1, 0],  # Right + A
    9:  [0, 0, 0, 1, 0, 1],  # Right + B
    10: [0, 0, 0, 1, 1, 1],  # Right + A + B
    11: [0, 0, 0, 0, 1, 0],  # A
    12: [0, 0, 0, 0, 0, 1],  # B
    13: [0, 0, 0, 0, 1, 1],  # A + B
}

modewrapper = wrappers.SetPlayingMode('algo')

class MarioEnv:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = None
        self.action_space = Discrete(len(action_mapping.keys()))

    def reset(self):
        if self.env is not None:
            self.env.close()
        self.env = modewrapper(gym.make(self.env_name))
        return self.env.reset()

    def step(self, action):
        controller_action = action_mapping[action]
        return self.env.step(controller_action)

def make(env_name):
    return MarioEnv(env_name)
