from collections import deque
import numpy as np
import copy
import os
import cv2

class Evaluator:
    def __init__(self,
                 env,
                 state_shape=[84, 84],
                 state_window=1,
                 eval_episodes=10,
                 render=False,
                 recorder=None,
                 record_episodes=3):
        self.env = env
        self.state_shape = state_shape
        self.eval_episodes = eval_episodes
        self.render = render
        self.recorder = recorder
        self.record_episodes = record_episodes

        self.init_states = deque(
            np.zeros([state_window] + state_shape, dtype=np.float32).tolist(),
            maxlen=state_window)

    def start(self, agent, trainer_step, trainer_episode):
        episode = 0
        rewards = []
        if self.recorder is not None:
            recorded_episodes = np.random.choice(
                self.eval_episodes, self.record_episodes, replace=False)
            recorders = {i: copy.deepcopy(self.recorder) for i in recorded_episodes}
        while True:
            sum_of_rewards = 0
            reward = 0
            done = False
            state = self.env.reset()
            states = copy.deepcopy(self.init_states)
            while True:
                states.append(state.tolist())
                nd_states = np.array(list(states))

                if self.render:
                    self.env.render()

                if self.recorder is not None and episode in recorders:
                    recorders[episode].append(self.env.render(mode='rgb_array'))

                # episode reaches the end
                if done:
                    episode += 1
                    rewards.append(sum_of_rewards)
                    agent.stop_episode(nd_states, reward, False)
                    break

                action = agent.act(nd_states, reward, False)
                state, reward, done, info = self.env.step(action)

                sum_of_rewards += reward

            if episode == self.eval_episodes:
                break

        if self.recorder is not None:
            for index, recorder in recorders.items():
                recorder.save_mp4('{}_{}.mp4'.format(trainer_step, index))
                recorder.flush()

        return rewards

class Recorder:
    def __init__(self, outdir, bgr=True):
        self.outdir = outdir
        self.images = []
        self.bgr = bgr

    def append(self, image):
        self.images.append(image)

    def save_mp4(self, file_name):
        path = os.path.join(self.outdir, file_name)
        save_video(path, self.images, bgr=self.bgr)

    def flush(self):
        self.images = []

def save_video(path, images, frame_rate=30.0, bgr=True):
    fourcc = cv2.VideoWriter_fourcc(*'MP4S')
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(path, fourcc, frame_rate, (width, height), True)
    for image in images:
        if bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        writer.write(image)
    writer.release()
