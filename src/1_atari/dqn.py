import sys
sys.path.insert(0, '..')

import agents
import time
import gym
import numpy

import models.dqn.src.model            as Model
import models.dqn.src.config           as Config


from common.Training import *
from common.atari_wrapper import *


path = "models/dqn/"

env = gym.make("MsPacmanNoFrameskip-v4")
env = AtariWrapper(env)
env.reset()


agent = agents.AgentDQN(env, Model, Config)

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