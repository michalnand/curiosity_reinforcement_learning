import torch
import torch.nn as nn
from torchviz import make_dot

import numpy

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class NoiseLayer(torch.nn.Module):
    def __init__(self, inputs_count, init_range = 0.1):
        super(NoiseLayer, self).__init__()
        
        self.inputs_count   = inputs_count
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        w_initial   = init_range*torch.rand(self.inputs_count, device = self.device)
        
        self.w      = torch.nn.Parameter(w_initial, requires_grad = True)     
        self.distribution = torch.distributions.normal.Normal(0.0, 1.0)
 
    def forward(self, x):
        noise =  self.distribution.sample((self.inputs_count, )).detach().to(self.device)
        return x + self.w*noise


class Head(torch.nn.Module):

    def __init__(self, inputs_count, outputs_count, fc_count = 128):
        super(Head, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.model_value = nn.Sequential(
                                            nn.Linear(inputs_count, 128),
                                            nn.ReLU(),                      
                                            nn.Linear(128, 1) 
        )

        self.model_advantage = nn.Sequential(
                                                nn.Linear(inputs_count, 128),
                                                nn.ReLU(),                      
                                                nn.Linear(128, outputs_count)
        )

        self.model_value.to(self.device)
        self.model_advantage.to(self.device)

        print(self.model_value, "\n")
        print(self.model_advantage, "\n")

    def forward(self, input):
        value       = self.model_value(input)
        advantage   = self.model_advantage(input)

        result = value + advantage - advantage.mean()
        return result

class ListModules(nn.Module):
    def __init__(self, *args):
        super(ListModules, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, fc_count = 128, n_heads = 4):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape = input_shape
        self.n_heads = n_heads
        self.outputs_count = outputs_count
        
        input_channels  = self.input_shape[0]
        fc_input_height = self.input_shape[1]
        fc_input_width  = self.input_shape[2]    

        ratio           = 2**4
        fc_inputs_count = 64*((fc_input_width)//ratio)*((fc_input_height)//ratio)
 

        self.model_features = nn.Sequential(
                                            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(), 

                                            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(),
                    
                                            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(),
                                
                                            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(),
                                            
                                            Flatten(),
                                            NoiseLayer(fc_inputs_count, 0.001)
        )
                      
        self.model_attention = nn.Sequential( 
                                                nn.Linear(fc_inputs_count, fc_count),
                                                nn.ReLU(),
                                                nn.Linear(fc_count, self.n_heads),
                                                nn.Softmax(dim=1)
        )

        heads = []
        for i in range(n_heads):
            heads.append(Head(fc_inputs_count, outputs_count, fc_count))
        self.heads = ListModules(*heads)

        self.model_features.to(self.device)
        self.model_attention.to(self.device)
        self.heads.to(self.device)

        print(self.model_features, "\n")
        print(self.model_attention, "\n")

    def forward(self, state):
        batch_size = state.shape[0]
        features = self.model_features(state)

        attention = self.model_attention(features)

        heads_output  = torch.zeros((self.n_heads, batch_size, self.outputs_count)).to(self.device)

        for i in range(self.n_heads):    
            heads_output[i] = self.heads[i].forward(features)

        heads_output    = heads_output.transpose(0, 1)
        attention       = attention.unsqueeze(-1).repeat((1, 1, self.outputs_count))
        
        result  = torch.sum(attention*heads_output, dim = 1)
        return result

    def save(self, path):
        torch.save(self.model_features.state_dict(), path + "trained/model_features.pt")
        torch.save(self.model_attention.state_dict(), path + "trained/model_attention.pt")
        for i in range(self.n_heads):
            torch.save(self.heads[i].state_dict(), path + "trained/model_head_" + str(i) + ".pt")



    def load(self, path):
        self.model_features.load_state_dict(torch.load( path + "trained/model_features.pt", map_location = self.device))
        self.model_features.eval()

        self.model_attention.load_state_dict(torch.load( path + "trained/model_attention.pt", map_location = self.device))
        self.model_attention.eval() 
     
        for i in range(self.n_heads):
            self.heads[i].load_state_dict(torch.load( path + "trained/model_head_" + str(i) + ".pt", map_location = self.device))
            self.heads[i].eval()




if __name__ == "__main__":
    batch_size      = 1
    input_shape     = (4, 96, 96)
    actions_count   = 5

    model = Model(input_shape, actions_count, n_heads = 4)

    state       = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2])

    q_values = model.forward(state)
    
    make_dot(q_values).render("graph", format="png")

