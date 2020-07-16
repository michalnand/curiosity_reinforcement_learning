import sys
sys.path.insert(0, '..')
import gym
import pybulletgym
import gym_line_follower 


import numpy
import time


import agents


import models.ddpg.src.model_critic     as ModelCritic
import models.ddpg.src.model_actor      as ModelActor
import models.ddpg.src.config           as Config
from common.Training import *

path = "models/ddpg/"


class Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        
    def observation(self, obs):
        return numpy.array(obs).astype(numpy.float32)

env = gym.make("LineFollower-v0", gui = False)
env = Wrapper(env)


 
agent = agents.AgentDDPG(env, ModelCritic, ModelActor, Config)


trainig = TrainingEpisodes(env, agent, episodes_count=2000, episode_max_length=1000, saving_path=path, logging_iterations=1000)
trainig.run()

'''
agent.load(path)
agent.disable_training()
agent.iterations = 0
while True:
    reward, done = agent.main()
    env.render()
    time.sleep(0.01)
'''