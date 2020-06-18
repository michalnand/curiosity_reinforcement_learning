import torch
import torch.nn as nn

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, hidden_count = 64):
        super(Model, self).__init__()

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu" 
         
        self.layers = [ 
                        nn.Linear(input_shape[0] + outputs_count, hidden_count),
                        nn.ReLU(),   

                        nn.Linear(hidden_count, hidden_count),
                        nn.ReLU(),                      

                        nn.Linear(hidden_count, input_shape[0])
                    ]


        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)
        

    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1)
        y = self.model(x)# + state
        return y

    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/model_curiosity.pt")
        

    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "trained/model_curiosity.pt", map_location = self.device))
        self.model.eval() 
        