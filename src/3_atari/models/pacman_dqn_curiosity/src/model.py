import torch
import torch.nn as nn

import numpy

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_channels  = input_shape[0]
        fc_input_height = input_shape[1]//16
        fc_input_width  = input_shape[2]//16


        self.layers = [
                        nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(), 
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
 
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                        

                        Flatten(),

                        nn.Linear(fc_input_height*fc_input_width*64, 512),
                        nn.ReLU(),
                        
                        nn.Linear(512, outputs_count)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)
      
    def forward(self, state):
        return self.model.forward(state)

    def save(self, path):
        name = path + "trained/model_dqn.pt"
        print("saving", name)
        torch.save(self.model.state_dict(), name)

    def load(self, path):
        name = path + "trained/model_dqn.pt"
        print("loading", name) 

        self.model.load_state_dict(torch.load(name))
        self.model.eval() 
     

