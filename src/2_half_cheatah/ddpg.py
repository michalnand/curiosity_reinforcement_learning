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

env = gym.make("HalfCheetahPyBulletEnv-v0")
#env.render()

agent = agents.AgentDDPG(env, ModelCritic, ModelActor, Config)

trainig = TrainingEpisodes(env, agent, episodes_count=2000, episode_max_length=1000, saving_path=path, logging_iterations=1000)
trainig.run()

'''
agent.load(path)

agent.disable_training()
agent.iterations = 0
while True:
    reward, done = agent.main()
    env.render()
    time.sleep(0.01)
'''