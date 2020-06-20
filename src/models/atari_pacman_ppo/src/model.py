import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, hidden_count = 64):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        fc_input_height = self.input_shape[1]
        fc_input_width  = self.input_shape[2]    

        ratio           = 2**4

        hidden_count = 256
        fc_inputs_count = ((fc_input_width)//ratio)*((fc_input_height)//ratio)
 
        self.layers_features = [ 
                                nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=2),
                                nn.ReLU(), 

                                nn.Conv2d(32, 32, kernel_size=3, stride=3, padding=2),
                                nn.ReLU(),
        
                                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
                                nn.ReLU(),
                    
                                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=2),
                                nn.ReLU()
                            ]   


        self.layers_policy = [
                                Flatten(), 
                                nn.Linear(fc_inputs_count*64, hidden_count),
                                nn.ReLU(),

                                nn.Linear(hidden_count, outputs_count),
                                nn.Softmax(dim=-1)
                            ]

        self.layers_value = [
                                Flatten(), 
                                nn.Linear(fc_inputs_count*64, hidden_count),
                                nn.ReLU(),

                                nn.Linear(hidden_count, 1)
                            ]
      
        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)

        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device) 

        self.model_policy = nn.Sequential(*self.layers_policy)
        self.model_policy.to(self.device) 

        self.model_value = nn.Sequential(*self.layers_value)
        self.model_value.to(self.device)

        print(self.model_features)
        print(self.model_policy)
        print(self.model_value)

    def forward(self, state):
        features = self.model_features(state)

        policy = self.model_policy(features)
        value  = self.model_value(features)

        return policy, value

   
    def save(self, path):
        print("saving to ", path)

        torch.save(self.model_features.state_dict(), path + "trained/model_features.pt")
        torch.save(self.model_policy.state_dict(), path + "trained/model_policy.pt")
        torch.save(self.model_value.state_dict(), path + "trained/model_value.pt")

    def load(self, path):       
        print("loading from ", path)

        self.model_features.load_state_dict(torch.load(path + "trained/model_features.pt", map_location = self.device))
        self.model_policy.load_state_dict(torch.load(path + "trained/model_policy.pt", map_location = self.device))
        self.model_value.load_state_dict(torch.load(path + "trained/model_value.pt", map_location = self.device))
        
        self.model_features.eval() 
        self.model_policy.eval() 
        self.model_value.eval()  
    