import gym
import numpy
import agents

import models.lunar_lander_ppo.src.model   as Model
import models.lunar_lander_ppo.src.config  as Config

import time

from common.Training import *

path = "models/lunar_lander_ppo/"
env = gym.make("LunarLander-v2")

obs             = env.observation_space
actions_count   = env.action_space.n


 
agent = agents.AgentPPO(env, Model, Config)

max_iterations = 100000

trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
trainig.run()

agent.load(path)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)

