import sys
sys.path.insert(0, '..')

import agents
import time
import gym
import numpy

import models.dqn_curiosity.src.model            as Model
import models.dqn_curiosity.src.model_env        as ModelEnv
import models.dqn_curiosity.src.config           as Config


from common.Training import *
from common.atari_wrapper import *


path = "models/dqn_curiosity/"

env = gym.make("MsPacmanNoFrameskip-v4")
env = AtariWrapper(env)
env.reset()


agent = agents.AgentDQNCuriosity(env, Model, ModelEnv, Config)

max_iterations = 10000000

trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run()

'''
#agent.load(path)

agent.disable_training()
agent.iterations = 0
while True:
    reward, done = agent.main()

    env.render()
    time.sleep(0.01)
'''