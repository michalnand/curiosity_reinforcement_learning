import torch
import numpy

class PolicyBuffer:
    def __init__(self, envs_count, buffer_size, state_shape, actions_count, device):
        self.envs_count     = envs_count
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_count  = actions_count
        self.device         = device

        self.clear()
  
    def clear(self):
        self.states_prev_b      = torch.zeros((self.envs_count, self.buffer_size, ) + self.state_shape).to(self.device)
        self.states_b           = torch.zeros((self.envs_count, self.buffer_size, ) + self.state_shape).to(self.device)

        self.logits_b           = torch.zeros((self.envs_count, self.buffer_size, self.actions_count)).to(self.device)
        self.values_b           = torch.zeros((self.envs_count, self.buffer_size, 1)).to(self.device)
        self.actions_b          = torch.zeros((self.envs_count, self.buffer_size), dtype=int)
        self.rewards_b          = numpy.zeros((self.envs_count, self.buffer_size)) 
        self.dones_b            = numpy.zeros((self.envs_count, self.buffer_size), dtype=bool)

        self.discounted_rewards_b = numpy.zeros((self.envs_count, self.buffer_size, 1))
        
        self.idxs = numpy.zeros(self.envs_count, dtype=int)

    def add(self, env_id, state, logits, value, action, reward, done):
        idx = int(self.idxs[env_id])

        self.states_b[env_id][idx]    = state

        self.states_prev_b[env_id][idx] = self.states_b[env_id][idx].clone()
        self.states_b[env_id][idx]      = state.clone()
        

        self.logits_b[env_id][idx]    = logits
        self.values_b[env_id][idx]    = value
        self.actions_b[env_id][idx]   = action
        self.rewards_b[env_id][idx]   = reward
        self.dones_b[env_id][idx]     = done

        self.idxs[env_id] = int(self.idxs[env_id] + 1)

    def size(self):
        return self.idxs[0]

    
    def calc_discounted_reward(self, gamma):

        self.discounted_rewards = numpy.zeros((self.envs_count, self.buffer_size, 1))

        for env_id in range(self.envs_count):
            q = 0.0
            for n in reversed(range(self.buffer_size)):
                if self.dones_b[env_id][n]:
                    gamma_ = 0.0
                else:
                    gamma_ = gamma

                q = self.rewards_b[env_id][n] + gamma_*q
                self.discounted_rewards[env_id][n][0] = q
        

            


    def sample(self, env_id, batch_size):
        states         = []
        logits         = []
        values         = []
        actions        = []
        rewards        = []
        dones          = []
        discounted_rewards = []

        for i in range(batch_size):
            idx = numpy.random.randint(self.size())

            states.append(self.states_b[env_id][idx])
            logits.append(self.logits_b[env_id][idx])
            values.append(self.values_b[env_id][idx])
            actions.append(self.actions_b[env_id][idx])
            rewards.append(self.rewards_b[env_id][idx])
            dones.append(self.dones_b[env_id][idx])
            discounted_rewards.append(self.discounted_rewards_b[env_id][idx])


        return states, logits, values, actions, rewards, dones, discounted_rewards

    