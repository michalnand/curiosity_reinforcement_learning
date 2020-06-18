import numpy
import collections
import torch

Transition = collections.namedtuple("Transition", ("state", "action"))

class CusiosityBuffer:

    def __init__(self, size):
        self.size   = size
       
        self.ptr    = 0 
        self.buffer = []

    def length(self):
        return len(self.buffer)

    def is_full(self):
        if self.length() == self.size:
            return True
            
        return False

    def add(self, state, action):

        item = Transition(state.copy(), action.copy())

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
      
        
        for i in range(0, batch_size):
            n  = numpy.random.randint(self.length() - 1)
            state_t[i]      = torch.from_numpy(self.buffer[n].state).to(device)
            action_t[i]     = torch.from_numpy(self.buffer[n].action).to(device)
            state_next_t[i] = torch.from_numpy(self.buffer[n+1].state).to(device)
      
        return state_t, action_t, state_next_t