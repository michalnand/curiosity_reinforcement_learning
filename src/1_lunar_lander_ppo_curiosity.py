import gym
import numpy
import agents

import models.lunar_lander_ppo_curiosity.src.model
import models.lunar_lander_ppo_curiosity.src.model_curiosity
import models.lunar_lander_ppo_curiosity.src.config

import time

from common.TrainingLog import *

path = "models/lunar_lander_ppo_curiosity/"
env = gym.make("LunarLander-v2")


obs             = env.observation_space
actions_count   = env.action_space.n


model  = models.lunar_lander_ppo_curiosity.src.model
model_curiosity  = models.lunar_lander_ppo_curiosity.src.model_curiosity

config = models.lunar_lander_ppo_curiosity.src.config.Config()
 
agent = agents.AgentPPOCuriosity(env, model, model_curiosity, config)


max_episodes      = 2000
max_episode_steps = 300 

log = TrainingLog(path + "result/result.log")
best = False

'''
for episode in range(max_episodes):

    env.reset()
    steps = 0
    while True:
        reward, done = agent.main()
        
        steps+= 1

        if steps >= max_episode_steps:
            log.add(reward, True)
            break

        if done:
            log.add(reward, True)
            break 

        log.add(reward, False)

        if log.is_best:
            best = True

    if episode%100 == 0 and best == True:
        best = False 
        print("\n\n")
        print("saving new best with score = ", log.episode_score_best)
        agent.save(path)
        print("\n\n")

agent.save(path)
'''

agent.load(path)

agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
