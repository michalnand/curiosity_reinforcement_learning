import sys
sys.path.insert(0, '..')
import gym
import pybulletgym
import numpy
import time


import agents

import models.ddpg_imagination_multihead.src.model_critic     as ModelCritic
import models.ddpg_imagination_multihead.src.model_actor      as ModelActor
import models.ddpg_imagination_multihead.src.model_env        as ModelEnv
import models.ddpg_imagination_multihead.src.config           as Config
from common.Training import *

path = "models/ddpg_imagination_multihead/"

env = gym.make("AntPyBulletEnv-v0")
env.render()

agent = agents.AgentDDPGImagination(env, ModelCritic, ModelActor, ModelEnv, Config)

max_iterations = 10**6
#trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
#trainig.run()


agent.load(path)
agent.disable_training()
while True:
    agent.main()
    time.sleep(0.01)
