import sys
sys.path.insert(0, '..')
import gym
import numpy
import time


import agents


import models.dqn.src.model     as Model
import models.dqn.src.config           as Config
from common.Training import *

path = "models/dqn/"

class LunarLanderWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = reward/100.0
        return obs, reward, done, info

env = gym.make("LunarLander-v2")
env = LunarLanderWrapper(env)

agent = agents.AgentDQN(env, Model, Config)

#trainig = TrainingEpisodes(env, agent, episodes_count=2500, episode_max_length=1000, saving_path=path, logging_iterations=1000)
#trainig.run()


agent.load(path)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
