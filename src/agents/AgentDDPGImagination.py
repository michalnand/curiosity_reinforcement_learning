import numpy
import torch
from .ExperienceBufferContinuous import *

from .ImaginationModule import *


class AgentDDPGImagination():
    def __init__(self, env, ModelCritic, ModelActor, ModelImagination, Config):
        self.env = env

        config = Config.Config()

        self.batch_size     = config.batch_size
        self.gamma          = config.gamma
        self.update_frequency = config.update_frequency
        self.tau                =  config.tau

        self.exploration    = config.exploration

       
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        self.experience_replay = ExperienceBufferContinuous(config.experience_replay_size)



        self.model_actor    = ModelActor.Model(self.state_shape, self.actions_count)
        self.model_actor_target    = ModelActor.Model(self.state_shape, self.actions_count)

        self.model_critic   = ModelCritic.Model(self.state_shape, self.actions_count)
        self.model_critic_target   = ModelCritic.Model(self.state_shape, self.actions_count)

        
        

        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(), lr= config.actor_learning_rate)
        self.optimizer_critic   = torch.optim.Adam(self.model_critic.parameters(), lr= config.critic_learning_rate, weight_decay=0.0001)

        self.imagination_rollouts = config.imagination_rollouts
        self.imagination_steps  = config.imagination_steps
        self.imagination_module = ImaginationModule(ModelImagination, self.state_shape, self.actions_count, config.imagination_learning_rate, config.imagination_buffer_size, True)

        self.state    = env.reset()

        self.iterations     = 0

        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    

    def main(self):
        
        if self.enabled_training:
            self.exploration.process()

        state_t     = torch.from_numpy(self.state).to(self.model_actor.device).unsqueeze(0).float()

        rewards, actions = self.sample_imagination(self.state)

        action_best_idx = numpy.argmax(rewards)

        action_t = actions[action_best_idx][0]

        action = action_t.squeeze(0).detach().to("cpu").numpy()
        
        state_new, self.reward, done, self.info = self.env.step(action)

        if self.enabled_training:
            self.imagination_module.add(self.state, action, self.reward, done)
            self.experience_replay.add(self.state, action, self.reward + rewards[action_best_idx], done)

        if self.enabled_training and self.iterations%self.update_frequency == 0:
            self.imagination_module.train()
        
        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()

        self.state = state_new
            
        if done:
            self.env.reset()

        self.iterations+= 1

        return self.reward, done

    def sample_imagination(self, state_initial):
        
        actions = []
        rewards = []
        for m in range(self.imagination_rollouts):
            state_t     = torch.from_numpy(state_initial).to(self.model_actor.device).unsqueeze(0).float()

            reward_sum = 0.0
            actions_rollout = []
            for n in range(self.imagination_steps):           
                action_t = self._sample_action(state_t)
                actions_rollout.append(action_t.clone())

                state_next_t, reward = self.imagination_module.eval(state_t, action_t)
                reward      =  reward.squeeze(0).detach().to("cpu").numpy()

                state_t = state_next_t.detach().clone()
                reward_sum+= reward

            actions.append(actions_rollout)

            rewards.append(reward_sum/self.imagination_steps)


        return rewards, actions

            
        
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model_critic.device)
        
        reward_t = reward_t.unsqueeze(-1)
        done_t   = (1.0 - done_t).unsqueeze(-1)

        action_next_t   = self.model_actor_target.forward(state_next_t).detach()
        value_next_t    = self.model_critic_target.forward(state_next_t, action_next_t).detach()

        #critic loss
        value_target    = reward_t + self.gamma*done_t*value_next_t
        value_predicted = self.model_critic.forward(state_t, action_t)

        critic_loss     = ((value_target - value_predicted)**2)
        critic_loss     = critic_loss.mean()
     
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
        self.imagination_module.save(save_path) 

    def load(self, save_path):
        self.model_critic.load(save_path)
        self.model_actor.load(save_path)
        self.imagination_module.load(save_path)     


    def _sample_action(self, state_t):
        
        if self.enabled_training:
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
       
        action_t    = self.model_actor(state_t)

        noise  = torch.randn(state_t.shape[0], self.actions_count).to(self.model_actor.device)
        action_t = action_t + epsilon*noise
        action_t = torch.clamp(action_t, -1.0, 1.0)
    

        return action_t
