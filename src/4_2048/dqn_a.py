import sys
sys.path.insert(0, '..')

import agents
import time
import gym_2048
import gym
import numpy

import models.dqn_a.src.model            as Model
import models.dqn_a.src.config           as Config


from common.Training import *
from common.Game2048Wrapper import *

path = "models/dqn_a/"

env = gym.make("2048-v0")
#env = gym.make("Tiny2048-v0")
env = Game2048Wrapper(env, 4)
env.reset()


agent = agents.AgentDQN(env, Model, Config)


#trainig = TrainingEpisodes(env, agent, episodes_count=200000, episode_max_length=2048, saving_path=path, logging_iterations=1000)
#trainig.run()



agent.load(path)


agent.disable_training()
agent.iterations = 0
while True:
    reward, done = agent.main()

    if done:
        print(env.stats)
        print(env.stats_norm)
        print("\n")
      