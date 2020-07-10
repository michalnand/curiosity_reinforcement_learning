import sys
sys.path.insert(0, '..')

import agents
import time
import gym
import numpy

import models.pacman_dqn.src.model            as Model
import models.pacman_dqn.src.config           as Config


from common.Training import *
from common.atari_wrapper import *


path = "models/pacman_dqn/"

#env = gym.make("PongNoFrameskip-v4")
env = gym.make("MsPacmanNoFrameskip-v4")

env = AtariWrapper(env)
env.reset()


agent = agents.AgentDQN(env, Model, Config)

max_iterations = 10*(10**6)

trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
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
