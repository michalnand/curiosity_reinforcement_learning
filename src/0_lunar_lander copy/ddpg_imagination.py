import sys
sys.path.insert(0, '..')
import gym
import numpy
import time

 
import agents


import models.ddpg_imagination.src.model_critic     as ModelCritic
import models.ddpg_imagination.src.model_actor      as ModelActor
import models.ddpg_imagination.src.model_env        as ModelEnv
import models.ddpg_imagination.src.config           as Config
from common.Training import *

path = "models/ddpg_imagination/"

class Wrapper(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        reward = reward / 10.0

        if reward < -1.0: 
            reward = -1.0

        if reward > 1.0:
            reward = 1.0

        return obs, reward, done, info


env = gym.make("LunarLanderContinuous-v2")
env = Wrapper(env)
env.reset()

agent = agents.AgentDDPGImagination(env, ModelCritic, ModelActor, ModelEnv, Config)

max_iterations = (10**6)
#trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
#trainig.run()


agent.load(path)
agent.disable_training()
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
