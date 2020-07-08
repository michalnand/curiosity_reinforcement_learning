import sys
sys.path.insert(0, '..')
import gym
import numpy
import time


import agents


import models.ddpg.src.model_critic     as ModelCritic
import models.ddpg.src.model_actor      as ModelActor
import models.ddpg.src.config           as Config
from common.Training import *

path = "models/ddpg/"


class LunarLanderWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = reward/100.0
        return obs, reward, done, info

env = gym.make("LunarLanderContinuous-v2")
env = LunarLanderWrapper(env)


agent = agents.AgentDDPG(env, ModelCritic, ModelActor, Config)

trainig = TrainingEpisodes(env, agent, episodes_count=4000, episode_max_length=1000, saving_path=path, logging_iterations=1000)
trainig.run()


'''
agent.load(path)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
'''