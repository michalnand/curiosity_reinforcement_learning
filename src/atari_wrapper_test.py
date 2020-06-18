import agents
import time
import gym
import common.atari_wrapper


from common.TrainingLog import *

path = "models/atari_ppo/"
log = TrainingLog(path + "result/result.log", episode_skip_log = 1)

#env = gym.make("PongNoFrameskip-v4")
#env = gym.make("BreakoutNoFrameskip-v4")
env = gym.make("MsPacmanNoFrameskip-v4")
env = common.atari_wrapper.Create(env)

env.reset()

agent = agents.AgentRandom(env)


while True:
    reward, done = agent.main()
    env.render()

    if done:
        print("DONE ", reward)
    time.sleep(0.01)
