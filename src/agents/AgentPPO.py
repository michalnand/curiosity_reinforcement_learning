import numpy
import torch

from torch.distributions import Categorical

from .PolicyBuffer import *

class AgentPPO():
    def __init__(self, envs, Model, Config):
        self.envs = envs

        config = Config.Config()

        self.envs_count = len(self.envs)
 
        self.gamma          = config.gamma
        self.entropy_beta   = config.entropy_beta

        self.buffer_size        = config.buffer_size
        self.batch_size         = config.batch_size 
        self.training_epochs    = config.training_epochs 

       
        self.state_shape = self.envs[0].observation_space.shape
        self.actions_count     = self.envs[0].action_space.n

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_old      = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.states = []

        self.buffer = PolicyBuffer(self.envs_count, self.batch_size, self.state_shape, self.actions_count, self.model.device)

        for env in self.envs:
            self.states.append(env.reset())

        self.enable_training()

        self.iterations = 0

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

   
    def process_env(self, env_id = 0):
        state_t   = torch.tensor(self.states[env_id], dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)

        logits, value   = self.model_old.forward(state_t)

        action_t, _ = self._get_action(logits)
            
        self.states[env_id], reward, done, _ = self.envs[env_id].step(action_t.item())
        
        if self.enabled_training:
            self.buffer.add(env_id, state_t.squeeze(0), logits.squeeze(0), value.squeeze(0), action_t.item(), reward, done)
           
        if done:
            self.states[env_id] = self.envs[env_id].reset()


        return reward, done
        
    
    
    def main(self):
        reward = 0
        done = False
        for env_id in range(self.envs_count):
            tmp, tmp_done = self.process_env(env_id)
            if env_id == 0:
                reward = tmp
                done = tmp_done
        

        if self.buffer.size() > self.batch_size-1:  

            self.buffer.calc_discounted_reward(self.gamma)

            for epoch in range(self.training_epochs):
                loss = 0
                for env_id in range(self.envs_count):
                    loss+= self._compute_loss(env_id)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step() 

            #clear batch buffer
            self.buffer.clear()
            

        self.iterations+= 1

        return reward, done
            
    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
    
   

    def _get_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution   = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution.sample()

        entropy               = action_distribution.entropy()

        return action_t, entropy
  

    def _compute_loss(self, env_id):
        
        batch_count = self.buffer_size//self.batch_size
        for n in range(batch_count)
            states, logits_t, values, actions, rewards_t, dones, discounted_rewards_t = self.buffer.sample(env_id, self.batch_size)

            log_probs_t = torch.nn.functional.log_softmax(logits_t, dim = 1)

            logits, value_t = self.model.forward(states)

            probs     = torch.nn.functional.softmax(logits, dim = 1)
            log_probs = torch.nn.functional.log_softmax(logits, dim = 1)

            dist_entropy = self._get_action(logits)


                
            #compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(log_probs - log_probs_t.detach())
            
            #compute loss
            advantage = discounted_rewards_t - value_t.detach()
            
            loss1 = ratio*advantage
            loss2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantage
                
            loss_policy = -torch.min(loss1, loss2)
            loss_policy = loss_policy.mean()

            loss_value = ((discounted_rewards_t - value_t)**2) 
            loss_value = loss_value.mean()
            
            loss_entropy = -self.entropy_beta*dist_entropy
            loss_entropy = loss_entropy.mean()

            loss = loss_policy + loss_value + loss_entropy

        return loss










