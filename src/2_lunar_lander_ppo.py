import gym
import numpy
import agents

import models.lunar_lander_ppo.src.model   as Model
import models.lunar_lander_ppo.src.config  as Config

import time

from common.Training import *

path = "models/lunar_lander_ppo/"


envs_count = 8


class LunarLanderWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = numpy.clip(reward / 10.0, -1.0, 1.0)
        return obs, reward, done, info


envs = []
for i in range(envs_count):
    env = gym.make("LunarLander-v2")
    env = LunarLanderWrapper(env)
    envs.append(env)


obs             = envs[0].observation_space
actions_count   = envs[0].action_space.n


 
agent = agents.AgentPPO(envs, Model, Config)

max_iterations = 100000

trainig = TrainingIterations(envs, agent, max_iterations, path, 1000)
trainig.run()

agent.load(path)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)

