import torch
from .CusiosityBuffer import *


class CuriosityModule:
    def __init__(self, model, state_shape, actions_count, learning_rate = 0.001, buffer_size = 1024):

        self.actions_count = actions_count

        self.model          = model.Model(state_shape, actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= learning_rate)

        self.buffer = CusiosityBuffer(buffer_size)


    def add(self, state, action):
        if hasattr(action, "shape") and len(action.shape) > 1:
            self.buffer.add(state, action)
        else:
            action_ = numpy.zeros(self.actions_count)
            action_[action] = 1.0
            self.buffer.add(state, action_)
        
    def train(self, batch_size = 32):
        if self.buffer.is_full() == False:
            return None

        batch_count = self.buffer.length()//batch_size

        loss_sum = 0.0
        for i in range(batch_count):
            state_t, action_t, state_next_t = self.buffer.sample(batch_size, self.model.device)

            state_next_prediction_t = self.model.forward(state_t, action_t)

            loss = ((state_next_t - state_next_prediction_t)**2.0).mean()

            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            loss_sum+= loss.detach().to("cpu").numpy()

        self.buffer.clear()

        loss_sum = loss_sum/batch_size

        return loss_sum

    def eval(self, state, action, state_next):
        state_t         = state.clone().to(self.model.device)
        state_next_t    = state_next.clone().to(self.model.device)

        if len(action.shape) > 1:
            action_t    = torch.tensor(action, dtype=torch.float32).to(self.model.device)
        else:
            batch_size = state.shape[0]
            action_t = torch.zeros((batch_size, self.actions_count))
            action_t[range(batch_size), action] = 1.0
            action_t.to(self.model.device)

        state_next_prediction_t = self.model.forward(state_t, action_t)

        curiosity = ((state_next_t - state_next_prediction_t)**2.0).mean(dim = 1)
        curiosity = curiosity.to("cpu").detach().numpy()

        return curiosity

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

    def _one_hot_action(self, action, batch_size):
        action_t_one_hot = torch.zeros((batch_size, self.actions_count))
        action_t_one_hot[range(batch_size), action] = 1.0
        action_t_one_hot.to(self.model.device)

        return action_t_one_hot
