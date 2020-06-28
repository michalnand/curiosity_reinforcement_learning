import gym_2048
import gym
import numpy
import agents


import models.game2048_dqn.src.model        as Model
import models.game2048_dqn.src.model_tiny   as ModelTiny
import models.game2048_dqn.src.config       as Config

import time

from common.Training import *

path = "models/game2048_dqn/"

class Game2048Wrapper(gym.Wrapper):
    def __init__(self, env, size):
        gym.Wrapper.__init__(self, env)
        self.size = size

        self.channels = self.size*self.size + 1 + 1
        self.height   = self.size
        self.width    = self.size

        self.observation_space.shape = (self.channels, self.height, self.width)
        self.score    = 0
        self.max_tile = 0
        self.stats    = numpy.zeros(self.size*self.size*2)

    def reset(self):

        #print("score    = ", self.score)
        #print("max_tile = ", self.max_tile)

        obs = self.env.reset()
        self.score, self.max_tile = self._update_score(obs)
        return self._parse_state(obs)
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.score, self.max_tile = self._update_score(obs)

        if reward < 1.0:
            reward = -1.0
        else:
            reward = numpy.log2(reward)

        return self._parse_state(obs), reward, done, info 

    def _parse_state(self, state):

        state_norm = numpy.log2(numpy.clip(state, 1, 2**self.channels)).astype(int)


        state_ = numpy.rollaxis(numpy.eye(self.channels)[state_norm], 2, 0)
    
        return state_

    def _update_score(self, obs):

        max_tile  = numpy.max(obs)
        sum_tiles = numpy.sum(obs)

        max_tile_idx = int(numpy.log2(max_tile)) - 1
        self.stats[max_tile_idx]+= 1

        return sum_tiles, max_tile

env = gym.make("2048-v0")
#env = gym.make("Tiny2048-v0")
env = Game2048Wrapper(env, 4)

env.reset()


agent = agents.AgentDQN(env, Model, Config)

max_iterations = 1000000

#trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
#trainig.run()

agent.load(path)


agent.disable_training()
agent.iterations = 0
while True:
    reward, done = agent.main()

    env.render()
    print("\n")
    if done:
        print("\n\n\n\n\n\n\n")
        time.sleep(1.0)

