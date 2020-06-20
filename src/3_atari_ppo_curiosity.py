import agents
import time
import gym
import numpy
from common.atari_wrapper import *

import models.atari_ppo_curiosity.src.model
import models.atari_ppo_curiosity.src.model_curiosity
import models.atari_ppo_curiosity.src.config

from common.Training import *

path = "models/atari_ppo_curiosity/"

env = gym.make("PongNoFrameskip-v4")
#env = gym.make("BreakoutNoFrameskip-v4")
#env = gym.make("MsPacmanNoFrameskip-v4")

env = atari_wrapper(env)

obs             = env.observation_space
actions_count   = env.action_space.n
 

model  = models.atari_ppo_curiosity.src.model
model_curiosity  = models.atari_ppo_curiosity.src.model_curiosity
config = models.atari_ppo_curiosity.src.config.Config()
 
agent = agents.AgentPPOCuriosity(env, model, model_curiosity, config)


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