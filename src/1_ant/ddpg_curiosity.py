import sys
sys.path.insert(0, '..')
import gym
import pybulletgym
import numpy
import time

 
import agents


import models.ddpg_curiosity.src.model_critic     as ModelCritic
import models.ddpg_curiosity.src.model_actor      as ModelActor
import models.ddpg_curiosity.src.model_env        as ModelEnv
import models.ddpg_curiosity.src.config           as Config
from common.Training import *

path = "models/ddpg_curiosity/"


env = gym.make("AntPyBulletEnv-v0")
env.render()


 
agent = agents.AgentDDPGCuriosity(env, ModelCritic, ModelActor, ModelEnv, Config)

#trainig = TrainingEpisodes(env, agent, episodes_count=4000, episode_max_length=1000, saving_path=path, logging_iterations=1000)
#trainig.run()


agent.load(path)
agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
