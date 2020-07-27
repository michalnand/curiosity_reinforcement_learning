import sys
sys.path.insert(0, '..')
import gym
import pybulletgym
import numpy
import time


import agents


import models.sdpg.src.model_critic     as ModelCritic
import models.sdpg.src.model_actor      as ModelActor
import models.sdpg.src.config           as Config
from common.Training import *

path = "models/sdpg/"

env = gym.make("AntPyBulletEnv-v0")
#env.render()

agent = agents.AgentSDPG(env, ModelCritic, ModelActor, Config)

trainig = TrainingEpisodes(env, agent, episodes_count=1000, episode_max_length=1000, saving_path=path, logging_iterations=1000)
trainig.run()

'''
agent.load(path)
agent.disable_training()
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
'''