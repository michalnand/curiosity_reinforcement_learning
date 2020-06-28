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
        
        self.layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(channels))
        
        self.activation = nn.ReLU()


        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

    def forward(self, x):

        y = self.model(x) 
        return self.activation(y + x)

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, features_count = 32, n_stages = 2, hidden_count = 256):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.input_shape    = input_shape
        self.outputs_count  = outputs_count 
        
        
        self.layers = []

        self.layers.append(nn.Conv2d(input_shape[0], features_count, kernel_size=1, stride=1, padding=0))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.ZeroPad2d((0, 1, 0, 1)))

        for n in range(n_stages):
            self.layers.append(ResidualBlock(features_count))
                                    
        self.layers.append(Flatten())

        self.layers.append(nn.Linear(features_count*25, hidden_count))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_count, outputs_count))

      

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)


        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

    
        print(self.model)
      

    def forward(self, state):
        return self.model(state)

    def get_q_values(self, state):
        with torch.no_grad():
            state_dev       = torch.tensor(state, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
            network_output  = self.forward(state_dev)

            return network_output[0].to("cpu").detach().numpy()
    
    def save(self, path):
        name = path + "trained/model_dqn.pt"
        print("saving", name)
        torch.save(self.model.state_dict(), name)

    def load(self, path):
        name = path + "trained/model_dqn.pt"
        print("loading", name) 

        self.model.load_state_dict(torch.load(name))
        self.model.eval() 
     