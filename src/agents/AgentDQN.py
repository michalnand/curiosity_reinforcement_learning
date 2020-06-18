import numpy
import torch
from .experience_replay import *


class AgentDQN():
    def __init__(self, env, Model, Config):
        self.env = env

        self.action = 0

        config  = Config()

        self.batch_size     = config.batch_size

        self.exploration    = config.exploration
        self.gamma          = config.gamma

        self.iterations     = 0

        self.update_frequency = config.update_frequency

       
        self.observation_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        self.experience_replay = Buffer(config.experience_replay_size)

        self.model      = Model.Model(self.observation_shape, self.actions_count)
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

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

        
        q_values = self.model.get_q_values(self.observation)
        self.action = self.choose_action_e_greedy(q_values, epsilon)

        observation_new, self.reward, done, self.info = self.env.step(self.action)
 
        if self.enabled_training:
            self.experience_replay.add(self.observation, self.action, self.reward, done)


        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()

        self.observation = observation_new
            
        if done:
            self.env.reset()

        self.iterations+= 1

        return self.reward, done
        
        
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model.device)
        
        #q values, state now, state next
        q_predicted      = self.model.forward(state_t)
        q_predicted_next = self.model.forward(state_next_t)

        #compute target, Q learning
        q_target         = q_predicted.clone()
        for i in range(self.batch_size):
            action_idx    = action_t[i]
            q_target[i][action_idx]   = reward_t[i] + self.gamma*torch.max(q_predicted_next[i])*(1 - done_t[i])

        #train DQN model
        loss = ((q_target.detach() - q_predicted)**2).mean() 
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()


        
    def choose_action_e_greedy(self, q_values, epsilon):
        result = numpy.argmax(q_values)
        
        if numpy.random.random() < epsilon:
            result = numpy.random.randint(len(q_values))
        
        return result

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
    