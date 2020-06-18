import gym
import numpy
import agents

import models.lunar_lander_a2c.src.model
import models.lunar_lander_a2c.src.config

import time

from common.TrainingLog import *


path = "models/lunar_lander_a2c/"

paralel_envs_count = 8


class SetRewardRange(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        reward = reward / 10.0

        if reward < -1.0:
            reward = -1.0

        if reward > 1.0:
            reward = 1.0

        return obs, reward, done, info

envs = []

for i in range(paralel_envs_count):
    env = gym.make("LunarLander-v2")

    #env = SetRewardRange(env)
    obs = env.reset()
    env.seed(i)

    envs.append(env)

obs             = envs[0].observation_space
actions_count   = envs[0].action_space.n


log = TrainingLog(path + "result/result.log")


model  = models.lunar_lander_a2c.src.model
config = models.lunar_lander_a2c.src.config.Config()
 
agent = agents.AgentA2C(envs, model, config)

while log.episodes < 2000:
    reward, done = agent.main()
    log.add(reward, done)


'''
agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    envs[0].render()
    time.sleep(0.01)
'''