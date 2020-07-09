import sys
sys.path.insert(0, '..')
import gym
import pybulletgym
import numpy
import time

 
import agents


import models.ddpg_curiosity_sparse.src.model_critic     as ModelCritic
import models.ddpg_curiosity_sparse.src.model_actor      as ModelActor
import models.ddpg_curiosity_sparse.src.model_env        as ModelEnv
import models.ddpg_curiosity_sparse.src.config           as Config
from common.Training import *

path = "models/ddpg_curiosity_sparse/"

class SparseRewards(gym.Wrapper):
    def __init__(self, env, sparsity = 0.9):
        gym.Wrapper.__init__(self, env)
        self.sparsity = sparsity
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action) 
        if numpy.random.rand() < self.sparsity:
            reward = 0.0
        return obs, reward, done, info


env = gym.make("AntPyBulletEnv-v0")
env = SparseRewards(env)
#env.render()


 
agent = agents.AgentDDPGCuriosity(env, ModelCritic, ModelActor, ModelEnv, Config)

trainig = TrainingEpisodes(env, agent, episodes_count=2000, episode_max_length=1000, saving_path=path, logging_iterations=1000)
trainig.run()

'''
agent.load(path)
agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
'''
