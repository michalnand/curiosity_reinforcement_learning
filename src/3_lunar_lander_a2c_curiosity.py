import gym
import numpy
import agents

import models.lunar_lander_a2c_curiosity.src.model_a2c
import models.lunar_lander_a2c_curiosity.src.model_curiosity
import models.lunar_lander_a2c_curiosity.src.config

import time


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

        return obs, reward, [done, done], info


envs = [] 

for i in range(paralel_envs_count):
    env = gym.make("LunarLander-v2")

    env = SetRewardRange(env)
    obs = env.reset()
    env.seed(i)

    envs.append(env)

obs             = envs[0].observation_space
actions_count   = envs[0].action_space.n




model_a2c           = models.lunar_lander_a2c_curiosity.src.model_a2c
model_curiosity     = models.lunar_lander_a2c_curiosity.src.model_curiosity
config = models.lunar_lander_a2c_curiosity.src.config.Config()
 
agent = agents.AgentA2CCuriosity(envs, model_a2c, model_curiosity, config)

score = 0
while agent.iterations < 100000:
    reward = agent.main()
    score+= reward

    if agent.iterations%256 == 0:
        envs[0].render()
        print(agent.iterations, score)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    envs[0].render()
    time.sleep(0.01)
