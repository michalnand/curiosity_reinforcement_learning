import sys
sys.path.insert(0, '..')
import gym
import numpy
import time


import agents


import models.a2c_continuous.src.model        as Model
import models.a2c_continuous.src.config       as Config
from common.Training import *

path = "models/a2c_continuous/"

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
    env = gym.make("LunarLanderContinuous-v2")
    env = LunarLanderWrapper(env)
    envs.append(env)


 
agent = agents.AgentA2CContinuous(envs, Model, Config)

max_iterations = 100000

trainig = TrainingIterations(envs, agent, max_iterations, path, 1000)
trainig.run()

agent.load(path)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    envs[0].render()
    time.sleep(0.01)

