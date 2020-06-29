import agents
import time
import gym_2048
import gym
import numpy

class Game2048Wrapper(gym.Wrapper):
    def __init__(self, env, size):
        gym.Wrapper.__init__(self, env)

      
        self.observation_space.shape = (1, size, size)
        self.score    = 0
        self.max_tile = 0
        self.max_value = 15

    def reset(self):

        '''
        print("score    = ", self.score)
        print("max_tile = ", self.max_tile)
       
        print("score    = ", self.score)
        print("max_tile = ", self.max_tile)
        for i in range(len(self.stats)):
            print(2**(i+1), end="\t ")
        print("\n")
        for i in range(len(self.stats)):
            print(self.stats[i], end="\t ")
        print("\n")
        '''


        obs = self.env.reset()
        self.score, self.max_tile = self._update_score(obs)
        return self._parse_state(obs)
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.score, self.max_tile = self._update_score(obs)

        if reward > 1.0:
            reward = numpy.log2(reward)/self.max_value
        elif done:
            reward = -1.0
        else:
            reward = 0.0
        
        return self._parse_state(obs), reward, done, info 

    def _parse_state(self, state):
        state_norm = numpy.log2(numpy.clip(state, 1, 2**self.max_value))/self.max_value
        state_norm = numpy.expand_dims(state_norm, 0)
        return state_norm

    def _update_score(self, obs):

        max_tile  = numpy.max(obs)
        sum_tiles = numpy.sum(obs)

        return sum_tiles, max_tile

env = gym.make("2048-v0")
#env = gym.make("Tiny2048-v0")
env = Game2048Wrapper(env, 4)

env.reset()

obs_shape       =   env.observation_space.shape
actions_count   =   env.action_space.n


print("observation shape ", obs_shape)
print("actions count ", actions_count)
print("observation\n", env.observation_space)


 
agent = agents.AgentRandom(env)

while True:
    agent.main()
    env.render()
    print("\n\n")
    time.sleep(0.01)
