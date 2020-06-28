import numpy
import torch

from torch.distributions import Categorical
from .ImaginationModule import *

class AgentA2CImagination():
    def __init__(self, env, Model, ModelImagination, Config):
        self.env = env

        config = Config.Config()
 
        self.gamma          = config.gamma
        self.entropy_beta   = config.entropy_beta
        
        self.batch_size     = config.batch_size
        self.rollouts       = config.rollouts

        self.state_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        
        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.imagination_update_steps = config.imagination_update_steps
        self.imagination_module = ImaginationModule(ModelImagination, self.state_shape, self.actions_count, config.imagination_learning_rate, config.imagination_buffer_size)

        self.state = self.env.reset()

        self.enable_training()

        self.iterations = 0
        self._init_buffer()


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False


    def _init_buffer(self):
        self.logits_b           = torch.zeros((self.rollouts, self.batch_size, self.actions_count)).to(self.model.device)
        self.values_b           = torch.zeros((self.rollouts, self.batch_size, 1)).to(self.model.device)
        self.action_b           = torch.zeros((self.rollouts, self.batch_size), dtype=int)
        self.rewards_b          = numpy.zeros((self.rollouts, self.batch_size))
        self.done_b             = numpy.zeros((self.rollouts, self.batch_size), dtype=bool)

        self.idx = 0

        
        
    def compute_loss(self, idx):
        target_values_b = self._calc_q_values(self.rewards_b[idx], self.values_b[idx].detach().cpu().numpy(), self.done_b[idx])

        target_values_b = torch.FloatTensor(target_values_b).to(self.model.device)

        probs     = torch.nn.functional.softmax(self.logits_b[idx], dim = 1)
        log_probs = torch.nn.functional.log_softmax(self.logits_b[idx], dim = 1)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        loss_value = (target_values_b - self.values_b[idx])**2
        loss_value = loss_value.mean()


        '''
        compute actor loss 
        L = log(pi(s, a))*(T - V(s)) = log(pi(s, a))*A
        '''
        advantage   = (target_values_b - self.values_b[idx]).detach()
        loss_policy = -log_probs[range(len(log_probs)), self.action_b[idx]]*advantage
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

    def process_imagination(self):

        state_t   = self.imagination_module.get_state(self.rollouts).clone()
        

        for rollout in range(self.rollouts):
            
            state  = state_t[rollout]
            
            for n in range(self.batch_size):
                
                logits, value   = self.model.forward(state)
                action, action_t = self._select_action(logits)

                state_, reward = self.imagination_module.eval_np(state, action)

                if n == self.batch_size-1:
                    done = True
                else:
                    done = False

                self.logits_b[rollout][self.idx]     = logits.squeeze(0)
                self.values_b[rollout][self.idx]     = value.squeeze(0)
                self.action_b[rollout][self.idx]     = action_t.item()
                self.rewards_b[rollout][self.idx]    = reward
                self.done_b[rollout][self.idx]       = done

                state = torch.from_numpy(state_)

        loss = 0
        for rollout in range(self.rollouts):
            loss+= self.compute_loss(rollout)


        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step() 

        self._init_buffer()

    
    def main(self):

        state_t   = torch.tensor(self.state, dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)
    
        logits, value   = self.model.forward(state_t)
        action, _ = self._select_action(logits)

        state_, reward, done, _ = self.env.step(action)

        if self.enabled_training:
            self.imagination_module.add(self.state, action, reward)

            if self.iterations%self.imagination_update_steps == 0:
                loss = self.imagination_module.train()
                print("imagination loss = ", loss)

            if self.iterations > 2*4096:
                self.process_imagination()

    
        if done:
            self.state = self.env.reset()
        else:
            self.state = state_.copy()

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
   
    def _select_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t.item(), action_t
            