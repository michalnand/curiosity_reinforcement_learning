import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = []
        
        self.layers.append(nn.Conv2d(channels, channels, kernel_size=2, stride=1, padding=0))
        self.layers.append(nn.BatchNorm2d(channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=1, padding=0))
        self.layers.append(nn.BatchNorm2d(channels))
           
        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

    def forward(self, x):

        y = self.model(x) 
        return y + x

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, features_count = 16, n_layers = 4, hidden_count = 256):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.input_shape    = (1, ) + input_shape
        self.outputs_count  = outputs_count 
        
        
        self.features_layers = []

        self.features_layers.append(nn.Conv2d(input_shape[0], features_count, kernel_size=1, stride=1, padding=0))
        self.features_layers.append(nn.ReLU())

        for n in range(n_layers):
            self.features_layers.append(ResidualBlock(features_count))
                                    
        self.features_layers.append(Flatten())


        self.layers_policy = [
                                nn.Linear(features_count*16, hidden_count),
                                nn.ReLU(),   

                                nn.Linear(hidden_count, outputs_count)
                            ]

        self.layers_critic = [          
                                nn.Linear(features_count*16, hidden_count),
                                nn.ReLU(),   

                                nn.Linear(hidden_count, 1)
                            ]


        for i in range(len(self.features_layers)):
            if hasattr(self.features_layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.features_layers[i].weight)


        self.model_features = nn.Sequential(*self.features_layers)
        self.model_features.to(self.device)

        self.model_policy = nn.Sequential(*self.layers_policy)
        self.model_policy.to(self.device)

        self.model_critic = nn.Sequential(*self.layers_critic)
        self.model_critic.to(self.device)


        print(self.model_features)
        print(self.model_policy)
        print(self.model_critic)


    def forward(self, state):
        features_output =  self.model_features(state)

        policy_output = self.model_policy(features_output)
        critic_output = self.model_critic(features_output)

      
        return policy_output, critic_output
    
    def save(self, path):
        print("saving to ", path)

        torch.save(self.model_features.state_dict(), path + "trained/model_features.pt")
        torch.save(self.model_policy.state_dict(), path + "trained/model_policy.pt")
        torch.save(self.model_critic.state_dict(), path + "trained/model_critic.pt")

    def load(self, path):       
        print("loading from ", path)

        self.model_features.load_state_dict(torch.load(path + "trained/model_features.pt", map_location = self.device))
        self.model_policy.load_state_dict(torch.load(path + "trained/model_policy.pt", map_location = self.device))
        self.model_critic.load_state_dict(torch.load(path + "trained/model_critic.pt", map_location = self.device))
    
        self.model_features.eval() 
        self.model_policy.eval() 
        self.model_critic.eval()  
    