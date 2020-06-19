import agents
import time
import gym
import numpy
import common.atari_wrapper

import models.atari_ppo.src.model
import models.atari_ppo.src.config

from common.Training import *

path = "models/atari_ppo/"

#env = gym.make("PongNoFrameskip-v4")
env = gym.make("BreakoutNoFrameskip-v4")
#env = gym.make("MsPacmanNoFrameskip-v4")

env = common.atari_wrapper.Create(env)

obs             = env.observation_space
actions_count   = env.action_space.n


model  = models.atari_ppo.src.model
config = models.atari_ppo.src.config.Config()
 
agent = agents.AgentPPO(env, model, config)


max_iterations = 10*(10**6)


trainig = TrainingIterations(env, agent, max_iterations, path, 500)
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