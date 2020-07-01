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
env = common.atari_wrapper.Create(env)

obs = env.reset()

agent = agents.AgentRandom(env)

print(obs)
print(obs.shape)


while True:
    reward, done = agent.main()
    env.render()

    if reward != 0:
        print("reward = ", reward)
    if done:
        print("DONE \n\n")
    
    time.sleep(0.01)
