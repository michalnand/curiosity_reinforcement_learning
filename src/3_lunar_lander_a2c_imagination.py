import gym
import numpy
import agents



import models.lunar_lander_a2c_imagination.src.model        as Model
import models.lunar_lander_a2c_imagination.src.model_env    as ModelEnv
import models.lunar_lander_a2c_imagination.src.config       as Config

import time

from common.Training import *

path = "models/lunar_lander_a2c_imagination/"

class LunarLanderWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = numpy.clip(reward / 10.0, -1.0, 1.0)
        return obs, reward, done, info

env = gym.make("LunarLander-v2")
env = LunarLanderWrapper(env)

agent = agents.AgentA2CImagination(env, Model, ModelEnv, Config)

max_iterations = 100000

trainig = TrainingIterations(env, agent, max_iterations, path, 100)
trainig.run()

agent.load(path)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    envs[0].render()
    time.sleep(0.01)

