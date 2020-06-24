import gym
import numpy
import agents

import models.lunar_lander_dqn_imagination.src.model        as ModelDQN
import models.lunar_lander_dqn_imagination.src.model_env    as ModelEnv
import models.lunar_lander_dqn_imagination.src.config       as Config

import time

from common.Training import * 

path = "models/lunar_lander_dqn_imagination/"


env = gym.make("LunarLander-v2")

class LunarLanderWrapper(gym.Wrapper):
    def __init__(self, env, frame_skip = 4):
        gym.Wrapper.__init__(self, env)
        self.frame_skip = frame_skip

    def step(self, action):
        
        reward = 0.0
        for i in range(self.frame_skip):
            obs, reward_, done, info = self.env.step(action)
            reward+= reward_
            if done:
                break

        reward = numpy.clip(reward / 10.0, -1.0, 1.0)
 
        
        return obs, reward, done, info

env = LunarLanderWrapper(env)



obs             = env.observation_space
actions_count   = env.action_space.n

agent = agents.AgentDQNImagination(env, ModelDQN, ModelEnv, Config)


max_iterations = 100000

trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
trainig.run()

agent.load(path)


agent.disable_training()
agent.iterations = 0
while True:
    agent.main()
    env.render()
    time.sleep(0.01)

