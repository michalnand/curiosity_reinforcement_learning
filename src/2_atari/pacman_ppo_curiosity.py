import sys
sys.path.insert(0, '..')
import gym
import numpy
import time

import agents

import models.pacman_ppo_curiosity.src.model       as Model
import models.pacman_ppo_curiosity.src.model_env   as ModelEnv
import models.pacman_ppo_curiosity.src.config      as Config

from common.atari_wrapper import *
from common.Training import *

path = "models/pacman_ppo_curiosity/"


envs_count = 8

envs = []
for i in range(envs_count):
    env = gym.make("MsPacmanNoFrameskip-v4")
    env = AtariWrapper(env)
    envs.append(env)


obs             = envs[0].observation_space
actions_count   = envs[0].action_space.n

agent = agents.AgentPPOCuriosity(envs, Model, ModelEnv, Config)

max_iterations = 10**6

trainig = TrainingIterations(envs, agent, max_iterations, path, 1000)
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
