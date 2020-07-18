import sys
sys.path.insert(0, '..')
import gym
import pybulletgym
import numpy
import time

 
import agents


import models.ddpg_curiosity_multihead.src.model_critic     as ModelCritic
import models.ddpg_curiosity_multihead.src.model_actor      as ModelActor
import models.ddpg_curiosity_multihead.src.model_env        as ModelEnv
import models.ddpg_curiosity_multihead.src.config           as Config
from common.Training import *

path = "models/ddpg_curiosity_multihead/"

env = gym.make("HalfCheetahPyBulletEnv-v0")
#env.render()

agent = agents.AgentDDPGCuriosity(env, ModelCritic, ModelActor, ModelEnv, Config)

max_iterations = 5000000
trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run()

'''
agent.load(path)
agent.disable_training()
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
'''