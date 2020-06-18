import gym
import numpy
import agents

import models.mountain_car_dqn_curiosity.src.model_dqn
import models.mountain_car_dqn_curiosity.src.model_curiosity
import models.mountain_car_dqn_curiosity.src.config


import common.dqn_experiment


gym.envs.register(
    id='MountainCarCustom-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=4096      # MountainCar-v0 uses 200
)

env = gym.make("MountainCarCustom-v0")


class SetRewardRange(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward < 0:
            reward = -0.001

        if done: 
            reward = 1.0
        
        return obs, reward, done, info


env = SetRewardRange(env)
env.reset()


ModelDQN        = models.mountain_car_dqn_curiosity.src.model_dqn
ModelCuriosity  = models.mountain_car_dqn_curiosity.src.model_curiosity
Config          = models.mountain_car_dqn_curiosity.src.config.Config

Agent           = agents.AgentDQNCuriosity

experiment = common.dqn_experiment.DQNExperiment(env, ModelDQN, ModelCuriosity, Config, Agent, rounds=20, training_iterations=100000)

experiment.process("mountain_car_curiosity_dqn.log")