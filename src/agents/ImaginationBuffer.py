import numpy
import collections
import torch

Transition = collections.namedtuple("Transition", ("state", "action", "reward"))

class ImaginationBuffer:

    def __init__(self, size):
        self.size   = size
       
        self.clear()

    def clear(self):
        self.ptr    = 0 
        self.buffer = []

    def length(self):
        return len(self.buffer)

    def is_full(self):
        if self.length() == self.size:
            return True
            
        return False

    def add(self, state, action, reward):

        item = Transition(state.copy(), action.copy(), reward)

        if self.length() < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.ptr] = item
            self.ptr = (self.ptr + 1)%self.length()

    def sample(self, batch_size, device):
        
        state_shape     = (batch_size, ) + self.buffer[0].state.shape[0:]
        action_shape    = (batch_size, ) + self.buffer[0].action.shape[0:]
      

        state_t         = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        action_t        = torch.zeros(action_shape, dtype=torch.float32).to(device)
        state_next_t    = torch.zeros(state_shape,  dtype=torch.float32).to(device)
      
        reward_t        = torch.zeros(batch_size)

        
        for i in range(0, batch_size):
            n  = numpy.random.randint(self.length() - 1)
            state_t[i]      = torch.from_numpy(self.buffer[n].state).to(device)
            action_t[i]     = torch.from_numpy(self.buffer[n].action).to(device)
            state_next_t[i] = torch.from_numpy(self.buffer[n+1].state).to(device)
            reward_t[i]     = self.buffer[n].reward
      
        return state_t, action_t, state_next_t, reward_t