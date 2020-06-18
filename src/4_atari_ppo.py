import agents
import time
import gym
import numpy
import common.atari_wrapper

import models.atari_ppo.src.model
import models.atari_ppo.src.config

from common.TrainingLog import *

path = "models/atari_ppo/"
log = TrainingLog(path + "result/result.log", episode_skip_log = 1)

#env = gym.make("PongNoFrameskip-v4")
#env = gym.make("BreakoutNoFrameskip-v4")
env = gym.make("MsPacmanNoFrameskip-v4")

env = common.atari_wrapper.Create(env)

env.reset()

obs             = env.observation_space
actions_count   = env.action_space.n



model  = models.atari_ppo.src.model
config = models.atari_ppo.src.config.Config()
 
agent = agents.AgentPPO(env, model, config)


max_iterations = 10*(10**6)

for iteration in range(max_iterations):
    reward, done = agent.main()
    log.add(reward, done)

    if iteration%1000 == 0 and log.is_best:
        agent.save(path)

if log.is_best:
    agent.save(path)

'''
agent.load(path)

agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
'''