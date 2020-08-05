import torch
import torch.nn as nn
from torchviz import make_dot


class CriticHead(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, hidden_count = 256):
        super(CriticHead, self).__init__()

        self.device = "cpu"

        self.layers = [ 
                        nn.Linear(input_shape[0] + outputs_count, hidden_count),
                        nn.ReLU(),
                        nn.Linear(hidden_count, hidden_count//2),
                        nn.ReLU(),            
                        nn.Linear(hidden_count//2, 1)           
        ] 

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        torch.nn.init.uniform_(self.layers[4].weight, -0.003, 0.003)
 
        self.model = nn.Sequential(*self.layers) 
        self.model.to(self.device)

        print(self.model, "\n")
       
    def forward(self, input):
        return self.model(input) 



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

    def __init__(self, input_shape, outputs_count, hidden_count = 256, n_heads = 4):
        super(Model, self).__init__()

        self.device = "cpu"

        self.n_heads = n_heads

        heads = []
        for i in range(n_heads):
            heads.append(CriticHead(input_shape, outputs_count, hidden_count))
        
        self.heads = ListModules(*heads)

        self.input_shape = input_shape

        self.model_attention = nn.Sequential(
                                                nn.Linear(input_shape[0] + outputs_count, hidden_count),
                                                nn.ReLU(),
                                                nn.Linear(hidden_count, hidden_count//2),
                                                nn.ReLU(),
                                                nn.Linear(hidden_count//2, self.n_heads),
                                                nn.Softmax(dim=1)
        )


        self.model_attention.to(self.device)
        print(self.model_attention)
        print("\n\n\n")

    def forward(self, state, action):
        batch_size = state.shape[0]
        x = torch.cat([state, action], dim = 1)

        attention = self.model_attention.forward(x)
       
        heads_output  = torch.zeros(self.n_heads, batch_size, 1).to(self.device)

        for i in range(self.n_heads):            
            heads_output[i] = self.heads[i].forward(x)

        heads_output  = heads_output.transpose(0, 1)

        attention = attention.unsqueeze(-1) 

        result = torch.sum(attention*heads_output, dim = 1)

        
        return result

    def save(self, path):
        torch.save(self.model_attention.state_dict(), path + "trained/model_attention.pt")
        for i in range(self.n_heads):
            torch.save(self.heads[i].state_dict(), path + "trained/model_head_" + str(i) + ".pt")

        

    def load(self, path):       
        self.model_attention.load_state_dict(torch.load(path + "trained/model_attention.pt", map_location = self.device))
        self.model_attention.eval() 

        for i in range(self.n_heads):
            self.heads[i].load_state_dict(torch.load(path + "trained/model_head_" + str(i) + ".pt", map_location = self.device))
            self.heads[i].eval()
        

if __name__ == "__main__":
    batch_size      = 32
    input_shape     = (26, )
    actions_count   = 5

    model = Model(input_shape, actions_count, hidden_count=128, n_heads = 4)

    state       = torch.randn(batch_size, input_shape[0])
    action      = torch.randn(batch_size, actions_count)

    critic_output = model.forward(state, action)
    
    make_dot(critic_output).render("graph", format="png")
