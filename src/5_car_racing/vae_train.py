import sys
sys.path.insert(0, '..')

import gym
import numpy
from PIL import Image
import agents
import common.common_frame_wrapper

from UnsupervisedDataset import *


env = gym.make("CarRacing-v0")
env = common.common_frame_wrapper.CommonFrameWrapper(env, height=64, width=64, frame_stacking=1, frame_skipping=4)
obs = env.reset()

agent = agents.AgentRandomContinuous(env)

dataset = UnsupervisedDataset(2048)


max_iterations = (10**6)
for iterations in range(max_iterations):
    reward, done = agent.main()

    if done:
        env.reset()

    state = env.state

    dataset.add(state)
    if dataset.is_full():

        for b in range(10):
            batch = dataset.get_random_batch()
            print(batch.shape)
        #vae.train()
        dataset.clear()
        print("training")
        