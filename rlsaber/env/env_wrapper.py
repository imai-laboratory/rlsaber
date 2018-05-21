from collections import deque
import numpy as np
import copy


class EnvWrapper:
    def __init__(self, env, r_preprocess=None, s_preprocess=None):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.r_preprocess = r_preprocess
        self.s_preprocess = s_preprocess
        self.results = {
            'rewards': 0
        }

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.results['rewards'] += reward
        # preprocess reward
        if self.r_preprocess is not None:
            reward = self.r_preprocess(reward)
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        self.results['rewards'] = 0
        return state

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def get_results(self):
        return self.results

class ActionRepeatEnvWrapper(EnvWrapper):
    def __init__(self, env, r_preprocess=None, s_preprocess=None, repeat=4):
        super().__init__(env, r_preprocess, s_preprocess)
        self.repeat = repeat
        self.states = deque(maxlen=repeat)

    def step(self, action):
        sum_of_reward = 0
        done = False
        for i in range(self.repeat):
            if done:
                state = np.zeros_like(np.array(state))
                reward = 0
            else:
                state, reward, done, info = super().step(action)
            sum_of_reward += reward
            self.states.append(state)
        return np.array(list(self.states)), sum_of_reward, done, info

    def reset(self):
        state = super().reset()
        for i in range(self.repeat):
            init_state = np.zeros_like(np.array(state))
            self.states.append(init_state)
        self.states.append(state)
        return np.array(list(self.states))

class BatchEnvWrapper:
    def __init__(self, envs):
        self.envs = envs
        self.running = [True for _ in range(len(envs))]
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        state_shape = envs[0].observation_space.shape
        self.zero_state = np.zeros_like(self.reset(0), dtype=np.float32)

    def step(self, actions):
        states = []
        rewards = []
        dones = []
        infos = []
        for i, env in enumerate(self.envs):
            if self.running[i]:
                state, reward, done, info = env.step(actions[i])
            else:
                state = copy.deepcopy(self.zero_state)
                reward = 0
                done = True
                info = None
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            self.running[i] = not done
        return states, rewards, dones, infos

    def reset(self, index):
        self.running[index] = True
        return self.envs[index].reset()

    def render(self, mode='human'):
        return self.envs[0].render(mode=mode)

    def get_num_of_envs(self):
        return len(self.envs)

    def get_results(self):
        return list(map(lambda e: e.get_results(), self.envs))
