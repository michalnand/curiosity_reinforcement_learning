import gym
import numpy
import agents

import models.lunar_lander_ppo.src.model
import models.lunar_lander_ppo.src.config

import time

from common.Training import *

path = "models/lunar_lander_ppo/"
env = gym.make("LunarLander-v2")

obs             = env.observation_space
actions_count   = env.action_space.n


model  = models.lunar_lander_ppo.src.model
config = models.lunar_lander_ppo.src.config.Config()
 
agent = agents.AgentPPO(env, model, config)


max_episodes = 2000
max_episode_steps = 300

#trainig = TrainingEpisodes(env, agent, max_episodes, max_episode_steps, path, 500)
#trainig.run()

agent.load(path)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)

