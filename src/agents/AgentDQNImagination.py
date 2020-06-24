import numpy
import torch
from .ExperienceBuffer import *
from .ImaginationModule import *


class AgentDQNImagination():
    def __init__(self, env, Model, ModelEnv, Config):
        self.env = env
        self.state    = env.reset()

        config = Config.Config()
        
        self.update_frequency   = config.update_frequency
        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma


        self.rollouts               = config.rollouts
        self.forward_ahead_steps    = config.forward_ahead_steps
       
       
        self.state_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        self.experience_replay = ExperienceBuffer(config.experience_replay_size)

        self.model      = Model.Model(self.state_shape, self.actions_count)
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        self.imagination_module = ImaginationModule(ModelEnv, self.state.shape, self.actions_count, config.imagination_learning_rate, config.imagination_buffer_size)


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

        q_values = self.model.get_q_values(self.state)

        action = self.choose_action_e_greedy(q_values, epsilon)

        state_new, reward, done, self.info = self.env.step(action)
        

        if self.enabled_training:
            
            self.imagination_module.add(self.state, action, reward)
            loss = self.imagination_module.train()

            if loss is not None:
                self.process_imagination()

            if loss is not None:
                print("imagination loss = ", loss)

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        self.iterations+= 1

        return reward, done

    def process_imagination(self):
        iterations = 0
        for rollout in range(self.rollouts):
            state  = self.state.copy()
            
            for n in range(self.forward_ahead_steps):
                #obtain Q values
                q_values = self.model.get_q_values(state)

                #choose action
                epsilon = self.exploration.get()
                action = self.choose_action_e_greedy(q_values, epsilon)

                #process action in model
                state_new, reward = self.imagination_module.eval_np(state, action)

                if n >= self.forward_ahead_steps-1:
                    done = True
                else:
                    done = False
    
                self.experience_replay.add(state, action, reward, done)

                state = state_new.copy()

                #train dqn
                if iterations%self.update_frequency == 0 and self.experience_replay.is_full():
                    self.dqn_train()

                iterations+= 1

        
        
    def dqn_train(self):
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
            param.grad.data.clamp_(-0.1, 0.1)
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
    