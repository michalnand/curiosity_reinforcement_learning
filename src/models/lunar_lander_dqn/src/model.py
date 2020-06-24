import torch
import torch.nn as nn



class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, hidden_count = 64):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.model = nn.Sequential(
                                    nn.Linear(input_shape[0], hidden_count),
                                    nn.ReLU(),
                                    
                                    nn.Linear(hidden_count, hidden_count),
                                    nn.ReLU(),

                                    nn.Linear(hidden_count, outputs_count),
        )

        self.model.to(self.device)

        print(self.model)
      
    def forward(self, state):
        return self.model.forward(state)

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
     