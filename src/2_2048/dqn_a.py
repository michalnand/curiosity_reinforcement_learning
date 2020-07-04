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

max_iterations = 20*(10**6)

#trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
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

    '''
    env.render()
    print("\n")
    if done:
        print("\n\n\n\n\n\n\n")
        time.sleep(1.0)
    '''
