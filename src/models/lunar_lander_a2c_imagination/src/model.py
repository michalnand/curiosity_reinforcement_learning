import torch
import torch.nn as nn

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, hidden_count = 64):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
         
        self.model_features = nn.Sequential(
                                    nn.Linear(input_shape[0], hidden_count),
                                    nn.ReLU()                      
        )

        self.model_policy = nn.Sequential(
                                nn.Linear(hidden_count, hidden_count),
                                nn.ReLU(),   

                                nn.Linear(hidden_count, outputs_count)
        )

        self.model_critic = nn.Sequential(          
                                nn.Linear(hidden_count, hidden_count),
                                nn.ReLU(),   

                                nn.Linear(hidden_count, 1)
        )


      
        self.model_features.to(self.device)
        self.model_policy.to(self.device)
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
    