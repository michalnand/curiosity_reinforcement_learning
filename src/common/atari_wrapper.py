import gym
import numpy
from PIL import Image

class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip = 4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            state, reward, done, info = self.env.step(action)
            total_reward+= reward
            if done:
                break

        return state, total_reward, done, info


class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, height = 96, width = 96, frame_stacking = 4):
        super(ResizeEnv, self).__init__(env)
        self.height = height
        self.width  = width
        self.frame_stacking = frame_stacking

        state_shape = (self.frame_stacking, self.height, self.width)
        self.dtype = numpy.float32

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=self.dtype)
        self.state = numpy.zeros(state_shape, dtype=self.dtype)

    def observation(self, state):
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize((self.height, self.width))

        for i in reversed(range(self.frame_stacking-1)):
            self.state[i+1] = self.state[i].copy()
        self.state[0] = numpy.array(img).astype(self.dtype)/255.0

        return self.state

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)

        return obs

def AtariWrapper(env, height = 96, width = 96, frame_stacking=4, frame_skipping=4):
    env = SkipEnv(env, frame_skipping)
    env = ResizeEnv(env, height, width, frame_stacking)
    env = FireResetEnv(env) 

    return env