import numpy
import torch
from .experience_replay import *

from .CuriosityModule import *


class AgentDQNCuriosity():
    def __init__(self, env, ModelDQN, ModelCuriosity, Config):
        self.env = env

        self.action = 0

        config = Config()

        self.batch_size     = config.batch_size

        self.exploration    = config.exploration
        self.gamma          = config.gamma
        self.curiosity_beta = config.curiosity_beta

        self.iterations     = 0

        self.update_frequency = config.update_frequency


       
        self.observation_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        self.experience_replay = Buffer(config.experience_replay_size)

        self.model_dqn      = ModelDQN.Model(self.observation_shape, self.actions_count)
        self.optimizer_dqn  = torch.optim.Adam(self.model_dqn.parameters(), lr= config.learning_rate)

        self.curiosity_module = CuriosityModule(ModelCuriosity, self.observation_shape, self.actions_count, config.curiosity_learning_rate)

        self.observation    = env.reset()
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
        
        q_values = self.model_dqn.get_q_values(self.observation)
        self.action = self.choose_action_e_greedy(q_values, epsilon)

        observation_new, self.reward, done, self.info = self.env.step(self.action)

        if self.enabled_training:
            self.experience_replay.add(self.observation, self.action, self.reward, done)
            self.curiosity_module.add(self.observation, self.action)

        if self.enabled_training and self.iterations%self.update_frequency == 0:
            self.curiosity_module.train()
        
        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()

        
        self.observation = observation_new
            
        if done:
            self.env.reset()


        self.iterations+= 1
        return self.reward, done
        
        
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model_dqn.device)            

        #compute curiosity
        curiosity_t  = self.curiosity_module.eval(state_t, action_t, state_next_t)
        curiosity_t  = numpy.clip(self.curiosity_beta*curiosity_t, 0.0, 1.0)   

        #q values, state now, state next
        q_predicted      = self.model_dqn.forward(state_t)
        q_predicted_next = self.model_dqn.forward(state_next_t)

        #compute target, Q learning
        q_target         = q_predicted.clone()
        for i in range(self.batch_size):
            action_idx    = action_t[i]
            q_target[i][action_idx]   = curiosity_t[i] + reward_t[i] + self.gamma*torch.max(q_predicted_next[i])*(1 - done_t[i])

        #train DQN model
        loss = ((q_target.detach() - q_predicted)**2).mean() 
        self.optimizer_dqn.zero_grad()
        loss.backward()
        for param in self.model_dqn.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer_dqn.step()


        
    def choose_action_e_greedy(self, q_values, epsilon):
        result = numpy.argmax(q_values)
        
        if numpy.random.random() < epsilon:
            result = numpy.random.randint(len(q_values))
        
        return result

    def save(self, save_path):
        self.model_dqn.save(save_path + "DQN_") 
        self.model_curiosity.save(save_path + "Curiosity_") 

    def load(self, save_path):
        self.model_dqn.load(save_path + "DQN_") 
        self.model_curiosity.load(save_path + "Curiosity_")     