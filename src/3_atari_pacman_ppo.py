import agents
import time
import gym
import numpy
from common.atari_wrapper import *

import models.atari_pacman_ppo.src.model
import models.atari_pacman_ppo.src.config

from common.Training import *

path = "models/atari_pacman_ppo/"

env = gym.make("MsPacmanNoFrameskip-v4")

env = atari_wrapper(env)

obs             = env.observation_space
actions_count   = env.action_space.n


model  = models.atari_pacman_ppo.src.model
config = models.atari_pacman_ppo.src.config.Config()
 
agent = agents.AgentPPO(env, model, config)


max_iterations = 10*(10**6)

trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
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