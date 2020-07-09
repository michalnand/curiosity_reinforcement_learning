import sys
sys.path.insert(0, '..')
import gym
import numpy
import time

import agents

import common.atari_wrapper




#env = gym.make("PongNoFrameskip-v4")
#env = gym.make("BreakoutNoFrameskip-v4")
env = gym.make("MsPacmanNoFrameskip-v4")
env = common.atari_wrapper.AtariWrapper(env)

obs = env.reset()

agent = agents.AgentRandom(env)

print(obs)
print(obs.shape)

k = 0.1
fps = 0
while True:
    time_start = time.time()
    reward, done = agent.main()
    time_stop  = time.time()
    env.render()

    fps = (1.0-k)*fps + k*1.0/(time_stop - time_start)

    print(fps)

    if reward != 0:
        print("reward = ", reward)
    if done:
        print("DONE \n\n")
    
    time.sleep(0.01)
