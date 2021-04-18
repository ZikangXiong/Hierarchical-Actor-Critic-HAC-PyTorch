import numpy as np
import torch as th

from .HAC import HAC as HAC_
from gym.spaces import Box, Discrete

import os
import pickle

from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common import logger
from auto_rl.learning.utils import get_violation_count


class HAC(HAC_):
    def __init__(self,
                 env,
                 goal_state,
                 k_level,
                 H,
                 threshold,
                 lr,
                 lambda_,
                 gamma,
                 exploration_action_noise,
                 exploration_state_noise,
                 verbose=0,
                 device="cpu",
                 render=False,
                 seed=None):
        assert isinstance(env.observation_space, Box) and \
               isinstance(env.observation_space, Box), "Working on discrete space"

        self.env = env
        self.goal_state = goal_state  # final goal
        self.verbose = verbose

        # for save / load
        self._init_params = (goal_state, k_level, H, threshold, lr, lambda_, gamma, exploration_action_noise,
                             exploration_state_noise, verbose, device, render, seed)

        action_offset = (env.action_space.low + env.action_space.high) / 2
        action_bounds = env.action_space.high - action_offset
        state_offset = (env.observation_space.low + env.observation_space.high) / 2
        state_bounds = env.observation_space.high - state_offset

        action_offset = th.FloatTensor(action_offset.reshape(1, -1)).to(device)
        action_bounds = th.FloatTensor(action_bounds.reshape(1, -1)).to(device)
        state_bounds = th.FloatTensor(state_bounds.reshape(1, -1)).to(device)
        state_offset = th.FloatTensor(state_offset.reshape(1, -1)).to(device)

        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]

        action_clip_low = np.array([-1.0 * action_bounds])
        action_clip_high = np.array([action_bounds])
        state_clip_low = env.observation_space.low
        state_clip_high = env.observation_space.high

        if seed:
            env.seed(seed)
            th.manual_seed(seed)
            np.random.seed(seed)

        super(HAC, self).__init__(k_level, H, state_dim, action_dim, render, threshold,
                                  action_bounds, action_offset, state_bounds, state_offset, lr, device)

        super(HAC, self).set_parameters(lambda_, gamma, action_clip_low, action_clip_high,
                                        state_clip_low, state_clip_high, exploration_action_noise,
                                        exploration_state_noise)

    def learn(self, total_timesteps, n_steps, n_iter, batch_size, save_path, tb_log_path=None):
        configure_logger(verbose=self.verbose, tensorboard_log=tb_log_path, tb_log_name="HAC", reset_num_timesteps=True)

        step_count = 0
        i_episode = 1
        while step_count <= total_timesteps:
            self.reward = 0
            self.timestep = 0

            state = self.env.reset()
            # collecting experience in environment
            last_state, done, _step_count = self.run_HAC(self.env, self.k_level - 1, state, self.goal_state,
                                                         is_subgoal_test=False)
            step_count += _step_count

            # updating with collected data
            if step_count > n_steps * i_episode:
                vio_num = get_violation_count(self.env)
                if vio_num is not None:
                    logger.record("rollout/violation", vio_num)
                logger.record(f"rollout/ep_rew_mean", self.reward)

                self.update(n_iter, batch_size)
                i_episode += 1

                logger.dump(step_count)

        self.save(save_path)
        return self

    def predict(self, state: np.ndarray, deterministic=True) -> np.ndarray:
        # Upper level action is the goal applied to lower level
        action = th.from_numpy(self.goal_state).float()
        state = th.from_numpy(state).float()

        for pi in self.HAC[::-1]:
            action = pi.select_action(state, action)

        return action

    def save(self, path, **kwargs):
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(f"{path}/params.pkl", "wb") as f:
            pickle.dump(self._init_params, f)
        super(HAC, self).save(directory=path, name="model")

    @classmethod
    def load(cls, path, env):
        with open(f"{path}/params.pkl", "rb") as f:
            params = pickle.load(f)

        model = cls(env, *params)
        super(HAC, model).load(directory=path, name="model")

        return model
