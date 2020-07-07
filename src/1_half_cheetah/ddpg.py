import sys
sys.path.insert(0, '..')
import gym
import pybulletgym
import numpy
import time


import agents


import models.ddpg.src.model_critic     as ModelCritic
import models.ddpg.src.model_actor      as ModelActor
import models.ddpg.src.config           as Config
from common.Training import *

path = "models/ddpg/"


class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = numpy.clip(reward*0.1, -1.0, 1.0)
        return obs, reward, done, info



env = gym.make("HalfCheetahPyBulletEnv-v0")
env = EnvWrapper(env)
env.render()


 
agent = agents.AgentDDPG(env, ModelCritic, ModelActor, Config)

max_iterations = 1000000

#trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
#trainig.run() 


agent.load(path)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)

