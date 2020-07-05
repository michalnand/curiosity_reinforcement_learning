import sys
sys.path.insert(0, '..')
import gym
import pybulletgym
import numpy
import time


import agents


import models.a2c_continuous.src.model        as Model
import models.a2c_continuous.src.config       as Config
from common.Training import *

path = "models/a2c_continuous/"

envs_count = 1

envs = []
for i in range(envs_count):
    env = gym.make("AntPyBulletEnv-v0")
    #env.render()
    env.reset()
    envs.append(env)


 
agent = agents.AgentA2CContinuous(envs, Model, Config)

max_iterations = 1000000

trainig = TrainingIterations(envs, agent, max_iterations, path, 1000)
trainig.run()

'''
agent.load(path)

agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    envs[0].render()
    time.sleep(0.01)
    print("step")
'''
