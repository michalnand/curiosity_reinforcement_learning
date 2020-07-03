import sys
sys.path.insert(0, '..')

import gym_2048
import gym
import numpy
import agents

import torch

from common.atari_wrapper import *
import models.env_conv_model_b.model as Model
from common.Game2048Wrapper import *


env = gym.make("2048-v0")
#env = gym.make("Tiny2048-v0")
env = Game2048Wrapper(env, 4)
env.reset()



state = env.reset()

actions_count   = env.action_space.n

cm =  agents.CuriosityModule(Model, state.shape, actions_count, learning_rate = 0.001, buffer_size = 4096)


for i in range(100000):
    action = numpy.random.randint(actions_count)
    state_, reward, done, _ = env.step(action)


    cm.add(state, action, reward, done)

    loss = cm.train()
     
    if loss is not None:
        print("iteration = ", i, " loss = ", loss, state.shape)

       
        curiosity, reward_prediction = cm.eval_np(state, state_, action)

        print(curiosity)
        print(reward, reward_prediction)

        print("\n\n\n")

    if done:
        state = env.reset()
    else:
        state = state_.copy()