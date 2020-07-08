import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 64):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.layers_features = [    nn.Linear(input_shape[0], hidden_count),
                                    nn.ReLU()
        ]
         
        self.layers = [ nn.Linear(hidden_count + outputs_count, hidden_count),
                        nn.ReLU(),            
                        nn.Linear(hidden_count, 1)           
        ] 

        torch.nn.init.xavier_uniform_(self.layers_features[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.uniform_(self.layers[2].weight, -0.003, 0.003)
 
        self.model_features = nn.Sequential(*self.layers_features) 
        self.model_features.to(self.device)

        self.model = nn.Sequential(*self.layers) 
        self.model.to(self.device)

        print(self.model_features)
        print(self.model)
        print("\n")
       

    def forward(self, state, action):
        features = self.model_features(state)
        x = torch.cat([features, action], dim = 1)
        return self.model(x)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model_features.state_dict(), path + "trained/model_critic_features.pt")
        torch.save(self.model.state_dict(), path + "trained/model_critic.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model_features.load_state_dict(torch.load(path + "trained/model_critic_features.pt", map_location = self.device))
        self.model.load_state_dict(torch.load(path + "trained/model_critic.pt", map_location = self.device))
        self.model.eval()  
    
