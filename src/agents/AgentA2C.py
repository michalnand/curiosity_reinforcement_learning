import numpy
import torch

from torch.distributions import Categorical

class AgentA2C():
    def __init__(self, envs, model, config):
        self.envs = envs

        self.envs_count = len(self.envs)
 
        self.gamma          = config.gamma
        self.entropy_beta   = config.entropy_beta
        self.batch_size     = config.batch_size
       
        self.observation_shape = self.envs[0].observation_space.shape
        self.actions_count     = self.envs[0].action_space.n

        
        self.model          = model.Model(self.observation_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.observations = []

        for env in self.envs:
            self.observations.append(env.reset())

        self.enable_training()

        self.iterations = 0
        self._init_buffer()


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False


    def _init_buffer(self):
        self.logits_b           = torch.zeros((self.envs_count, self.batch_size, self.actions_count)).to(self.model.device)
        self.values_b           = torch.zeros((self.envs_count, self.batch_size, 1)).to(self.model.device)
        self.action_b           = torch.zeros((self.envs_count, self.batch_size), dtype=int)
        self.rewards_b          = numpy.zeros((self.envs_count, self.batch_size))
        self.done_b             = numpy.zeros((self.envs_count, self.batch_size), dtype=bool)

        self.idx = 0

        
    def process_env(self, env_id = 0):
        observation_t   = torch.tensor(self.observations[env_id], dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)
        logits, value   = self.model.forward(observation_t)

        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
            
        self.observations[env_id], reward, done, _ = self.envs[env_id].step(action_t.item())
        

        if self.enabled_training:
            self.logits_b[env_id][self.idx]     = logits.squeeze(0)
            self.values_b[env_id][self.idx]     = value.squeeze(0)
            self.action_b[env_id][self.idx]     = action_t.item()
            self.rewards_b[env_id][self.idx]    = reward
            self.done_b[env_id][self.idx]       = done

        if done:
            self.envs[env_id].reset()

        return reward, done
        
    def compute_loss(self, env_id = 0):
        target_values_b = self._calc_q_values(self.rewards_b[env_id], self.values_b[env_id].detach().cpu().numpy(), self.done_b[env_id])

        target_values_b = torch.FloatTensor(target_values_b).to(self.model.device)

        probs     = torch.nn.functional.softmax(self.logits_b[env_id], dim = 1)
        log_probs = torch.nn.functional.log_softmax(self.logits_b[env_id], dim = 1)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        loss_value = (target_values_b - self.values_b[env_id])**2
        loss_value = loss_value.mean()


        '''
        compute actor loss 
        L = log(pi(s, a))*(T - V(s)) = log(pi(s, a))*A
        '''
        advantage   = (target_values_b - self.values_b[env_id]).detach()
        loss_policy = -log_probs[range(len(log_probs)), self.action_b[env_id]]*advantage
        loss_policy = loss_policy.mean()

        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs*log_probs).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        #train network, with gradient cliping
        loss = loss_value + loss_policy + loss_entropy
        return loss


    
    def main(self):
        reward = 0
        done = False
        for env_id in range(self.envs_count):
            tmp, tmp_done = self.process_env(env_id)
            if env_id == 0:
                reward = tmp
                done = tmp_done

        if self.enabled_training:
            self.idx+= 1
        
        if self.idx > self.batch_size-1:   

            self.rewards_b = (self.rewards_b - self.rewards_b.mean())/self.rewards_b.std()   
            
            loss = 0
            for env_id in range(self.envs_count):
                loss+= self.compute_loss(env_id)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step() 

            #clear batch buffer
            self._init_buffer()

            '''
            print("loss_value = ", loss_value.detach().cpu().numpy())
            print("loss_policy = ", loss_policy.detach().cpu().numpy())
            print("loss_entropy = ", loss_entropy.detach().cpu().numpy())
            print("\n\n\n")
            '''

        self.iterations+= 1

        return reward, done
            
    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
    

    def _calc_q_values(self, rewards, critic_value, done):
        size = len(rewards)
        result = numpy.zeros((size, 1))

        q = 0.0
        for n in reversed(range(size)):
            if done[n]:
                gamma = 0.0
            else:
                gamma = self.gamma

            q = rewards[n] + gamma*q
            result[n][0] = q

        return result
   