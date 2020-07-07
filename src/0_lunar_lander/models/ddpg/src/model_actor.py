import torch
import torch.nn as nn


class NoiseLayer(torch.nn.Module):
    def __init__(self, inputs_count, init_range = 0.1):
        super(NoiseLayer, self).__init__()
        
        self.inputs_count   = inputs_count
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        w_initial   = init_range*torch.ones(self.inputs_count, device = self.device)
        
        self.w      = torch.nn.Parameter(w_initial, requires_grad = True)     
        self.distribution = torch.distributions.normal.Normal(0.0, 1.0)
 
    def forward(self, x):
        noise =  self.distribution.sample((self.inputs_count, )).detach().to(self.device)
        return x + self.w*noise


class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 64):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
         
        self.layers = [ 
                                    nn.Linear(input_shape[0], hidden_count),
                                    nn.ReLU(),           
                                    nn.Linear(hidden_count, hidden_count),
                                    nn.ReLU(),    
                                    nn.Linear(hidden_count, outputs_count),
                                    nn.Tanh()                                    
        ]

        '''
        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.uniform_(self.layers[i].weight, -0.003, 0.003)
        '''

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)
       

    def forward(self, state):
        return self.model(state)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "trained/model_actor.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "trained/model_actor.pt", map_location = self.device))
        self.model.eval()  
    