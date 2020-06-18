import numpy



class AgentRandom:
    def __init__(self, env):
        self.env = env
        self.actions_count = self.env.action_space.n
        self.iterations = 0

    def main(self):
        self.iterations+= 1
        action = numpy.random.randint(self.actions_count)

        state, reward, done, info = self.env.step(action)

        if done:
            state = self.env.reset()

        return reward, done
    
   
