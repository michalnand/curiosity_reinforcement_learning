import agents
import time
import gym
from common.atari_wrapper import *


from common.TrainingLog import *

path = "models/atari_ppo/"
log = TrainingLog(path + "result/result.log", episode_skip_log = 100)

env = gym.make("PongNoFrameskip-v4")
#env = gym.make("BreakoutNoFrameskip-v4")
#env = gym.make("MsPacmanNoFrameskip-v4")
env = atari_wrapper(env)

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
    
    #time.sleep(0.01)
