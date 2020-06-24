import numpy
import torch

from torch.distributions import Categorical

from .ImaginationModule import *

class AgentA2CImagination():
    def __init__(self, env, Model, ModelEnv, Config):
        self.env = env
        self.state = self.env.reset()

        config = Config.Config()

        self.gamma          = config.gamma
        self.entropy_beta   = config.entropy_beta
        self.rollouts       = config.rollouts
        self.forward_ahead_steps       = config.forward_ahead_steps
       
        self.actions_count     = self.env.action_space.n

        self.model          = Model.Model(self.state.shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.imagination_module = ImaginationModule(ModelEnv, self.state.shape, self.actions_count, config.imagination_learning_rate, config.imagination_buffer_size)

        self.iterations = 0

        self.enable_training()


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):

        state_t   = torch.tensor(self.state, dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)
        logits, value   = self.model(state_t)

        action = self._sample_action(logits.detach())
            
        state, reward, done, _ = self.env.step(action)
        
        if self.enabled_training:
           
            self.imagination_module.add(self.state, action, reward, done)

            if self.iterations > 4096 and self.iterations%8 == 0:
                loss_imagination = self.imagination_module.train()
                loss_policy = self.train_policy(state, self.rollouts, self.forward_ahead_steps)

                print("loss_imagination = ", loss_imagination)
                print("loss_policy = ", loss_policy)

        if done:
            self.state = self.env.reset()
        else:
            self.state = state.copy()

        self.iterations+= 1
        
        return reward, done
            
    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
    

    

    def train_policy(self, initial_state, rollouts = 32, forward_ahead_steps = 64):
        buffer_size = rollouts*forward_ahead_steps

        self.logits_b   = torch.zeros((buffer_size, self.actions_count)).to(self.model.device)
        self.values_b   = torch.zeros((buffer_size, 1)).to(self.model.device)
        self.action_b   = torch.zeros((buffer_size), dtype=int)
        self.rewards_b  = numpy.zeros((buffer_size))
        self.done_b     = numpy.zeros((buffer_size), dtype=bool)
        
        state_initial_t = self.imagination_module.get_state(rollouts)

        idx = 0 
        for r in range(rollouts):

            state_t   = (state_initial_t[r].clone()).unsqueeze(0)

            for n in range(forward_ahead_steps):                
                logits, value   = self.model.forward(state_t)

                action = self._sample_action(logits)
            
                state_new, reward = self.imagination_module.eval(state_t, action)

                state_t   = state_new.clone()
                reward    = reward.detach()[0][0]

                self.logits_b[idx]     = logits.squeeze(0)
                self.values_b[idx]     = value.squeeze(0)
                self.action_b[idx]     = action
                self.rewards_b[idx]    = reward

                if n == forward_ahead_steps-1:
                    self.done_b[idx] = True
                else:
                    self.done_b[idx]  = False

                idx+= 1


        loss = self.compute_loss()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step() 

        return loss.to("cpu").detach().numpy()


    def compute_loss(self): 
        target_values_b = self._calc_q_values(self.rewards_b, self.values_b.detach().cpu().numpy(), self.done_b)
        target_values_b = (target_values_b - target_values_b.mean())/target_values_b.std()

 
        target_values_b = torch.FloatTensor(target_values_b).to(self.model.device)

        probs     = torch.nn.functional.softmax(self.logits_b, dim = 1)
        log_probs = torch.nn.functional.log_softmax(self.logits_b, dim = 1)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        loss_value = (target_values_b - self.values_b)**2
        loss_value = loss_value.mean()
        
        '''
        compute actor loss 
        L = log(pi(s, a))*(T - V(s)) = log(pi(s, a))*A
        '''
        advantage   = (target_values_b - self.values_b).detach()
        loss_policy = -log_probs[range(len(log_probs)), self.action_b]*advantage
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
   
    def _sample_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action                = action_distribution_t.sample().item()

        return action
