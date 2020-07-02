# This file was based on
# https://github.com/openai/baselines/blob/edb52c22a5e14324304a491edc0f91b6cc07453b/baselines/common/atari_wrappers.py
# its license:
#
# The MIT License
#
# Copyright (c) 2017 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces

from matplotlib import pyplot as plt

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max = 30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)

        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)

        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

 
class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = np.clip(reward, -1.0, 1.0)
        return obs, reward, done, info


class ResizeFrameEnv(gym.Wrapper):
    def __init__(self, env, width = 96, height = 96, frame_stacking = 4):
        gym.Wrapper.__init__(self, env)

        self.width  = width
        self.height = height
        self.frame_stacking = frame_stacking    
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(self.frame_stacking, self.height, self.width), dtype=float)
    
    def reset(self):
        obs = self.env.reset()

        self.slices = np.zeros((self.frame_stacking, self.height, self.width))
       
        return self._parse_state(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        return self._parse_state(obs), reward, done, info

    def _parse_state(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        for i in reversed(range(self.frame_stacking-1)):
            self.slices[i+1] = self.slices[i].copy()
        
        self.slices[0] = np.array(frame).copy()/255.0

        return self.slices


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done    = True
            reward  = -1.0
        if lives == 0 and self.inital_lives > 0:
            reward = -1.0

        self.lives = lives
        return obs, reward, done, info

    def _reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        self.inital_lives = self.env.unwrapped.ale.lives()
        return obs



def AtariWrapper(env, width = 96, height = 96, frame_stacking = 4):
    env = NoopResetEnv(env)
    env = FireResetEnv(env)
    env = SkipEnv(env, 4)
    env = ClipRewardEnv(env)
    env = ResizeFrameEnv(env, width, height, frame_stacking)
    env = EpisodicLifeEnv(env)
    
    return env
    

