import torch
from .ExperienceBuffer import *


class MIMModule:
    def __init__(self, model, state_shape, actions_count, learning_rate = 0.001, buffer_size = 1024):

        self.actions_count = actions_count

        self.model          = model.Model(state_shape, actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= learning_rate)

        self.buffer = ExperienceBuffer(buffer_size)


    def add(self, state, action, reward, done = False):
        self.buffer.add(state, action, reward, done)
        
    
    def train(self, batch_size = 64):
        if self.buffer.is_full() == False:
            return None

        state_t, action_t, reward_t, state_next_t, done_t = self.buffer.sample(batch_size, self.model.device)

        reward_t = reward_t.unsqueeze(1)
        action_t = self._one_hot_encoding(action_t)

        features = self.model.forward(state_t, state_next_t, action_t)

        loss = 0.5*features - torch.log(0.5*torch.exp(features))
        loss = -loss.mean()


        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()


        return loss.detach().to("cpu").numpy()

    def eval(self, state_t, state_next_t, action):
        batch_size = state_t.shape[0]
        action_t = torch.zeros((batch_size, self.actions_count))
        action_t[range(batch_size), action] = 1.0
        action_t.to(self.model.device)

        state_next_prediction_t, reward_prediction_t = self.model.forward(state_t, action_t)

        curiosity = ((state_next_t - state_next_prediction_t)**2.0)
        curiosity = curiosity.view(curiosity.size(0), -1).sum(dim = 1)
        
        curiosity = curiosity.detach()
        reward_prediction_t   = reward_prediction_t[0].detach()

        return curiosity, reward_prediction_t


    def eval_np(self, state, state_next, action):

        state_t         = torch.tensor(state, dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)
        state_next_t    = torch.tensor(state_next, dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)

        curiosity, reward_prediction = self.eval(state_t, state_next_t, action)

        curiosity  = curiosity[0].to("cpu").numpy()
        reward_prediction = reward_prediction[0].to("cpu").numpy()

        return curiosity, reward_prediction

    def get_state(self, batch_size):
        state_t, _, _, _, _ = self.buffer.sample(batch_size, self.model.device)
        return state_t


    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

    def _one_hot_encoding(self, input):

        size = len(input)
        result = torch.zeros((size, self.actions_count))
        
        result[range(size), input] = 1.0
        
        return result.to(self.model.device)

  