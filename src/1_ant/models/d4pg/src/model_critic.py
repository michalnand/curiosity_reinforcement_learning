import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 256, n_atoms = 21, v_min = -1.0, v_max = 1.0):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [ 
                        nn.Linear(input_shape[0] + outputs_count, hidden_count),
                        nn.ReLU(),
                        nn.Linear(hidden_count, hidden_count//2),
                        nn.ReLU(),            
                        nn.Linear(hidden_count//2, n_atoms)           
        ] 

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        torch.nn.init.uniform_(self.layers[4].weight, -0.003, 0.003)
 
        self.model = nn.Sequential(*self.layers) 
        self.model.to(self.device)


        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))

        print(self.model)
       

    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1)
        return self.model(x)

    def compute_q_values(self, distribution):
        weights = torch.functional.softmax(distribution, dim=1)*self.supports
        res = weights.sum(dim=1)

        return res.unsqueeze(dim=-1)

    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "trained/model_critic.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "trained/model_critic.pt", map_location = self.device))
        self.model.eval()  
    
