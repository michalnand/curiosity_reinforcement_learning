import numpy
import torch
from .ExperienceBufferContinuous import *


class AgentSDPG():
    def __init__(self, env, ModelCritic, ModelActor, Config):
        self.env = env

        config = Config.Config()

        self.batch_size     = config.batch_size
        self.gamma          = config.gamma
        self.update_frequency = config.update_frequency
        self.tau                =  config.tau

        self.sample_size    = config.sample_size

        self.exploration    = config.exploration
    
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        self.experience_replay = ExperienceBufferContinuous(config.experience_replay_size)

        self.model_actor            = ModelActor.Model(self.state_shape, self.actions_count)
        self.model_actor_target     = ModelActor.Model(self.state_shape, self.actions_count)

        self.model_critic           = ModelCritic.Model(self.state_shape, self.actions_count)
        self.model_critic_target    = ModelCritic.Model(self.state_shape, self.actions_count)

        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(), lr= config.actor_learning_rate)
        self.optimizer_critic   = torch.optim.Adam(self.model_critic.parameters(), lr= config.critic_learning_rate, weight_decay=config.critic_learning_rate*0.01)

        self.state          = env.reset()

        self.iterations     = 0

        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
       
        state_t     = torch.from_numpy(self.state).to(self.model_actor.device).unsqueeze(0).float()

        action_t    = self.model_actor(state_t)
        action      = action_t.squeeze(0).detach().to("cpu").numpy()

        noise  = numpy.random.normal(loc = 0.0, scale = epsilon, size = self.actions_count)
        action = action + epsilon*noise

        action = numpy.clip(action, -1.0, 1.0)

        state_new, self.reward, done, self.info = self.env.step(action)

        if self.enabled_training:
            self.experience_replay.add(self.state, action, self.reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        self.iterations+= 1

        return self.reward, done
        
        
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model_critic.device)
        
        reward_t = reward_t.unsqueeze(-1)
        done_t   = (1.0 - done_t).unsqueeze(-1)

        value_target    = torch.zeros(self.sample_size, self.batch_size, 1).to(self.model_critic.device)
        value_predicted = torch.zeros(self.sample_size, self.batch_size, 1).to(self.model_critic.device)

        for j in range(self.sample_size):
            std = 0.1
            state_noised        = state_t.clone()       + std*torch.randn(state_t.shape)
            state_next_noise_t  = state_next_t.clone()  + std*torch.randn(state_t.shape)

            action_next_t   = self.model_actor_target.forward(state_next_noise_t).detach()
            value_next_t    = self.model_critic_target.forward(state_next_noise_t, action_next_t).detach()

            #critic loss
            value_target[j]    = reward_t + self.gamma*done_t*value_next_t
            value_predicted[j] = self.model_critic.forward(state_noised, action_t)

        #wasserstein optimal transport loss - approximaiton, eq 12
        value_target, _    = torch.sort(value_target, 1)
        value_predicted, _ = torch.sort(value_predicted, 1)
        #compute loss, pairwise matrix
        critic_loss     = ((value_target - value_predicted)**2) 
        critic_loss     = critic_loss.mean()

        '''
        #wasserstein optimal transport loss
        value_target        = value_target.transpose(0, 1)          #switch shape (sample, batch, 1) to (batch, sample, 1)
        value_predicted     = value_predicted.transpose(0, 1)       #switch shape (sample, batch, 1) to (batch, sample, 1)
        value_predicted, _  = torch.sort(value_predicted, 1)     #sort predicted values over sample exis, ascending
        #compute loss, pairwise matrix
        critic_loss     = ((value_target - value_predicted.transpose(1, 2))**2) 
        critic_loss     = critic_loss.mean()
        '''

        #update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward() 
        self.optimizer_critic.step()

        #actor loss
        actor_loss      = -self.model_critic.forward(state_t, self.model_actor.forward(state_t))
        actor_loss      = actor_loss.mean()

        #update actor
        self.optimizer_actor.zero_grad()       
        actor_loss.backward()
        self.optimizer_actor.step()

        # update target networks 
        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)
       
        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)


    def save(self, save_path):
        self.model_critic.save(save_path)
        self.model_actor.save(save_path)

    def load(self, save_path):
        self.model_critic.load(save_path)
        self.model_actor.load(save_path)
    