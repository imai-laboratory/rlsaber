from collections import deque
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import copy
import time
import threading
import copy


class AgentInterface:
    def act(self, state, reward, training):
        raise NotImplementedError()

    def stop_episode(state, reward, training):
        raise NotImplementedError()

class Trainer:
    def __init__(self,
                 env,
                 agent,
                 state_shape=[84, 84],
                 final_step=1e7,
                 state_window=1,
                 training=True,
                 render=False,
                 debug=True,
                 progress_bar=True,
                 before_action=None,
                 after_action=None,
                 end_episode=None,
                 is_finished=None,
                 evaluator=None,
                 end_eval=None,
                 should_eval=lambda s, e: s % 10 ** 5 == 0):
        self.env = env
        self.final_step = final_step
        self.init_states = deque(
            np.zeros(
                [state_window] + state_shape,
                dtype=np.float32
            ).tolist(),
            maxlen=state_window
        )
        self.agent = agent
        self.training = training
        self.render = render
        self.debug = debug
        self.progress_bar = progress_bar
        self.before_action = before_action
        self.after_action = after_action
        self.end_episode = end_episode
        self.is_finished = is_finished
        self.evaluator = evaluator
        self.end_eval = end_eval
        self.should_eval = should_eval

        # counters
        self.global_step = 0
        self.local_step = 0
        self.episode = 0
        self.sum_of_rewards = 0
        self.pause = True

        # for multithreading
        self.resume_event = threading.Event()
        self.resume_event.set()

    def move_to_next(self, states, reward, done):
        states = np.array(list(states))
        # take next action
        action = self.agent.act(
            states,
            reward,
            self.training
        )
        state, reward, done, info = self.env.step(action)
        # render environment
        if self.render:
            self.env.render()
        return state, reward, done, info

    def finish_episode(self, states, reward):
        states = np.array(list(states))
        self.agent.stop_episode(
            states,
            reward,
            self.training
        )

    def start(self):
        if self.progress_bar:
            pbar = tqdm(total=self.final_step, dynamic_ncols=True)
        while True:
            self.local_step = 0
            self.sum_of_rewards = 0
            reward = 0
            done = False
            state = self.env.reset()
            states = copy.deepcopy(self.init_states)
            while True:
                # to stop trainer from outside
                self.resume_event.wait()

                states.append(state.tolist())

                # episode reaches the end
                if done:
                    raw_reward = self.env.get_results()['rewards']
                    self.episode += 1
                    if self.progress_bar:
                        pbar.update(self.local_step)
                        msg = 'step: {}, episode: {}, reward: {}'.format(
                            self.global_step, self.episode, raw_reward)
                        pbar.set_description(msg)
                    self.end_episode_callback(
                        raw_reward, self.global_step, self.episode)
                    self.finish_episode(states, reward)
                    break

                self.before_action_callback(
                    states, self.global_step, self.local_step)

                state, reward, done, info = self.move_to_next(
                    states, reward, done)

                self.after_action_callback(
                    states, reward, self.global_step, self.local_step)

                self.sum_of_rewards += reward
                self.global_step += 1
                self.local_step += 1

                if self.evaluator is not None:
                    self.evaluate()

            if self.is_training_finished():
                if self.progress_bar:
                    pbar.close()
                return

    def before_action_callback(self, states, global_step, local_step):
        if self.before_action is not None:
            self.before_action(
                states,
                global_step,
                local_step
            )

    def after_action_callback(self, states, reward, global_step, local_step):
        if self.after_action is not None:
            self.after_action(
                states,
                reward,
                global_step,
                local_step
            )

    def end_episode_callback(self, reward, global_step, episode):
        if self.end_episode is not None:
            self.end_episode(
                reward,
                global_step,
                episode
            )

    def is_training_finished(self):
        if self.is_finished is not None:
            return self.is_finished(self.global_step)
        return self.global_step > self.final_step

    def evaluate(self):
        should_eval = self.should_eval(self.global_step, self.episode)
        if should_eval:
            print('evaluation starts')
            agent = copy.copy(self.agent)
            agent.stop_episode(copy.deepcopy(self.init_states), 0, False)
            eval_rewards = self.evaluator.start(
                agent, self.global_step, self.episode)
            if self.end_eval is not None:
                self.end_eval(self.global_step, self.episode, eval_rewards)
            if self.debug:
                msg = '[eval] step: {}, episode: {}, reward: {}'
                print(msg.format(
                    self.global_step, self.episode, np.mean(eval_rewards)))

    def stop(self):
        self.resume_event.clear()

    def resume(self):
        self.resume_event.set()

class BatchTrainer(Trainer):
    def __init__(self,
                env, # BatchEnvWrapper
                agent,
                state_shape=[84, 84],
                final_step=1e7,
                state_window=1,
                training=True,
                render=False,
                debug=True,
                time_horizon=20,
                batch_size=None,
                before_action=None,
                after_action=None,
                end_episode=None):
        super().__init__(
            env=env,
            agent=agent,
            state_shape=state_shape,
            final_step=final_step,
            state_window=state_window,
            training=training,
            render=render,
            debug=debug,
            before_action=before_action,
            after_action=after_action,
            end_episode=end_episode
        )

        # overwrite global_step
        self.global_step = 0
        self.time_horizon = time_horizon
        self.batch_size = time_horizon if batch_size is None else batch_size

    # TODO: Remove this overwrite
    def move_to_next(self, states, reward, done):
        # take next action
        action = self.agent.act(
            states,
            reward,
            done, # overwrite line this
            self.training
        )
        state, reward, done, info = self.env.step(action)
        # render environment
        if self.render:
            self.env.render()
        return state, reward, done, info

    # overwrite
    def start(self):
        to_ndarray = lambda q: np.array(list(map(lambda s: list(s), copy.deepcopy(q))))

        # values for the number of n environment
        n_envs = self.env.get_num_of_envs()
        self.local_step = [0 for _ in range(n_envs)]
        self.sum_of_rewards = [0 for _ in range(n_envs)]
        rewards = [0 for _ in range(n_envs)]
        dones = [False for _ in range(n_envs)]
        states = [self.env.reset(i) for i in range(n_envs)]
        queue_states = [copy.deepcopy(self.init_states) for _ in range(n_envs)]
        for i, state in enumerate(states):
            queue_states[i].append(state.tolist())
        t = 0
        pbar = tqdm(total=self.final_step, dynamic_ncols=True)

        # training loop
        while True:
            for i in range(n_envs):
                self.before_action_callback(
                    states[i],
                    self.global_step,
                    self.local_step[i]
                )

            # backup episode status
            prev_dones = dones
            states, rewards, dones, infos = self.move_to_next(
                to_ndarray(queue_states), rewards, prev_dones)

            for i in range(n_envs):
                self.after_action_callback(
                    states[i],
                    rewards[i],
                    self.global_step,
                    self.local_step[i]
                )

            # add state to queue
            for i, (state, done) in enumerate(zip(states, dones)):
                if done:
                    raw_reward = self.env.get_results()[i]['rewards']
                    self.episode += 1
                    global_step = self.global_step - (n_envs - i - 1)
                    msg = 'step: {}, episode: {}, reward: {}'
                    pbar.update(self.local_step[i])
                    pbar.set_description(
                        msg.format(global_step, self.episode, raw_reward))
                    # callback at the end of episode
                    self.end_episode(raw_reward, global_step, self.episode)
                    queue_states[i] = copy.deepcopy(self.init_states)
                    self.sum_of_rewards[i] = 0
                    self.local_step[i] = 0
                queue_states[i].append(state)

            for i in range(n_envs):
                self.sum_of_rewards[i] += rewards[i]
                if not dones[i]:
                    self.global_step += 1
                    self.local_step[i] += 1

            t += 1

            # pass transitions and update network
            should_update = t > 0 and t % self.time_horizon == 0
            self.agent.receive_next(to_ndarray(queue_states), rewards,
                                    dones, should_update and self.training)

            if self.is_training_finished():
                pbar.close()
                return

class AsyncTrainer:
    def __init__(self,
                envs,
                agents,
                state_shape=[84, 84],
                final_step=1e7,
                state_window=1,
                training=True,
                render=False,
                debug=True,
                progress_bar=True,
                before_action=None,
                after_action=None,
                end_episode=None,
                n_threads=10,
                evaluator=None,
                end_eval=None,
                should_eval=None):
        # meta data shared by all threads
        self.meta_data = {
            'shared_step': 0,
            'shared_episode': 0,
            'last_eval_step': 0,
            'last_eval_episode': 0
        }
        if progress_bar:
            pbar = tqdm(total=final_step, dynamic_ncols=True)

        # inserted callbacks
        def _before_action(state, global_step, local_step):
            shared_step = self.meta_data['shared_step']
            if before_action is not None:
                before_action(state, shared_step, global_step, local_step)

        def _after_action(state, reward, global_step, local_step):
            self.meta_data['shared_step'] += 1
            shared_step = self.meta_data['shared_step']
            if after_action is not None:
                after_action(state, reward, shared_step, global_step, local_step)

        def _end_episode(i):
            def func(reward, global_step, episode):
                shared_step = self.meta_data['shared_step']
                self.meta_data['shared_episode'] += 1
                shared_episode = self.meta_data['shared_episode']
                if end_episode is not None:
                    end_episode(
                        reward,
                        shared_step,
                        global_step,
                        shared_episode,
                        episode
                    )
                if progress_bar:
                    pbar.update(self.trainers[i].local_step)
                    msg = 'step: {}, episode: {}, reward: {}'.format(
                        shared_step, shared_episode, reward)
                    pbar.set_description(msg)
            return func

        def _end_eval(step, episode, rewards):
            shared_step = self.meta_data['shared_step']
            shared_episode = self.meta_data['shared_episode']
            for trainer in self.trainers:
                trainer.resume()
            if debug:
                msg = '[eval] step: {}, episode: {}, reward: {}'
                print(msg.format(shared_step, shared_episode, np.mean(rewards)))
            if end_eval is not None:
                end_eval(shared_step, shared_episode, step, episode, rewards)

        def _should_eval(step, episode):
            shared_step = self.meta_data['shared_step']
            shared_episode = self.meta_data['shared_episode']
            last_eval_step = self.meta_data['last_eval_step']
            last_eval_episode = self.meta_data['last_eval_episode']
            if should_eval is not None:
                is_eval = should_eval(
                    last_eval_step, last_eval_episode,
                    shared_step, shared_episode, step, episode)
                if is_eval:
                    for trainer in self.trainers:
                        trainer.stop()
                    self.meta_data['last_eval_step'] = shared_step
                    self.meta_data['last_eval_episode'] = shared_episode
            return is_eval

        self.trainers = []
        for i in range(n_threads):
            env = envs[i]
            agent = agents[i]
            trainer = Trainer(
                env=env,
                agent=agent,
                state_shape=state_shape,
                final_step=final_step,
                state_window=state_window,
                training=training,
                render=i == 0 and render,
                debug=False,
                progress_bar=False,
                before_action=_before_action,
                after_action=_after_action,
                end_episode=_end_episode(i),
                is_finished=lambda s: self.meta_data['shared_step'] > final_step,
                evaluator=evaluator if i == 0 else None,
                should_eval=_should_eval if i == 0 else None,
                end_eval=_end_eval if i == 0 else None
            )
            self.trainers.append(trainer)

    def start(self):
        sess = tf.get_default_session()
        coord = tf.train.Coordinator()
        # gym renderer is only available on the main thread
        render_trainer = self.trainers[0]
        threads = []
        for i in range(len(self.trainers) - 1):
            def run(index):
                with sess.as_default():
                    self.trainers[index + 1].start()
            thread = threading.Thread(target=run, args=(i,))
            thread.start()
            threads.append(thread)
            time.sleep(0.1)
        render_trainer.start()
        coord.join(threads)
