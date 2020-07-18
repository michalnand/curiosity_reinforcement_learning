import sys
sys.path.insert(0, '..')

import agents
import time
import gym
import numpy

import models.pong_dqn_curiosity_multihead.src.model            as Model
import models.pong_dqn_curiosity_multihead.src.model_env        as ModelEnv
import models.pong_dqn_curiosity_multihead.src.config           as Config


from common.Training import *
from common.atari_wrapper import *


path = "models/pong_dqn_curiosity_multihead/"

env = gym.make("PongNoFrameskip-v4")
env = AtariWrapper(env)
env.reset()


agent = agents.AgentDQNCuriosity(env, Model, ModelEnv, Config)

max_iterations = 10*(10**6)

trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run()
 
'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()

    env.render()
    time.sleep(0.01)
'''