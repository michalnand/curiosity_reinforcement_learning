import gym
import numpy
import agents

import models.lunar_lander_ppo_curiosity.src.model
import models.lunar_lander_ppo_curiosity.src.model_curiosity
import models.lunar_lander_ppo_curiosity.src.config

import time

from common.Training import *

path = "models/lunar_lander_ppo_curiosity/"
env = gym.make("LunarLander-v2")

obs             = env.observation_space
actions_count   = env.action_space.n


model  = models.lunar_lander_ppo_curiosity.src.model
model_curiosity  = models.lunar_lander_ppo_curiosity.src.model_curiosity

config = models.lunar_lander_ppo_curiosity.src.config.Config()
 
agent = agents.AgentPPOCuriosity(env, model, model_curiosity, config)


max_iterations = 200000

#trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
#trainig.run()

agent.load(path)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)

