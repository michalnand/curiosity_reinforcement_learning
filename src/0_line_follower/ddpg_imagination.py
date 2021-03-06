import sys
sys.path.insert(0, '..')
import gym
import gym_line_follower
import numpy
import time

 
import agents


import models.ddpg_imagination.src.model_critic     as ModelCritic
import models.ddpg_imagination.src.model_actor      as ModelActor
import models.ddpg_imagination.src.model_env        as ModelEnv
import models.ddpg_imagination.src.config           as Config
from common.Training import *

path = "models/ddpg_imagination/"

class Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        
    def observation(self, state):
        state_np = numpy.array(state).astype(numpy.float32)
        return state_np


env = gym.make("LineFollower-v0", gui = True)
env = Wrapper(env)

agent = agents.AgentDDPGImagination(env, ModelCritic, ModelActor, ModelEnv, Config)

max_iterations = (10**5)
#trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
#trainig.run()


agent.load(path)
agent.disable_training()
while True:
    agent.main()
    time.sleep(0.01)
