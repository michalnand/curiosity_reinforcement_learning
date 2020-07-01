import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, kernels_count = 64, hidden_count = 64):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape 
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]   

    
        self.conv0 = nn.Sequential( 
                                    nn.Conv2d(input_channels + outputs_count, kernels_count, kernel_size=2, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.Conv2d(kernels_count, kernels_count, kernel_size=2, stride=1, padding=0),
                                    nn.ReLU()
        )

       
        self.deconv0 = nn.Sequential(
                                        nn.ConvTranspose2d(kernels_count, kernels_count, kernel_size=2, stride=1, padding=0),
                                        nn.ReLU(), 

                                        nn.ConvTranspose2d(kernels_count, kernels_count, kernel_size=2, stride=1, padding=0),
                                        nn.ReLU(), 

                                        nn.Conv2d(kernels_count, input_channels, kernel_size=1, stride=1, padding=0)
                                    )

        self.reward_model = nn.Sequential(
                                            Flatten(),

                                            nn.Linear(kernels_count*4, hidden_count),
                                            nn.ReLU(),

                                            nn.Linear(hidden_count, 1)
        )

        self.conv0.to(self.device)
        self.deconv0.to(self.device)
        self.reward_model.to(self.device)

                                    

    def forward(self, state, action):
        action_ = action.unsqueeze(1).unsqueeze(1).transpose(3, 1).repeat((1, 1, self.input_shape[1], self.input_shape[2])).to(self.device)
        
        model_input      = torch.cat([state, action_], dim = 1)

        conv0_output     = self.conv0(model_input)


        observation_prediction = self.deconv0(conv0_output)

        reward_prediction = self.reward_model(conv0_output)

        
        return observation_prediction, reward_prediction

    def save(self, path):
        print("saving ", path)

        torch.save(self.conv0.state_dict(), path + "trained/model_curiosity_conv0.pt")
        torch.save(self.deconv0.state_dict(), path + "trained/model_curiosity_deconv0.pt")
        torch.save(self.reward_model.state_dict(), path + "trained/model_curiosity_reward_model.pt")


    def load(self, path):
        print("loading ", path, " device = ", self.device) 

        self.conv0.load_state_dict(torch.load(path + "trained/model_curiosity_conv0.pt", map_location = self.device))
        self.deconv0.load_state_dict(torch.load(path + "trained/model_curiosity_deconv0.pt", map_location = self.device))
        self.reward_model.load_state_dict(torch.load(path + "trained/model_curiosity_reward_model.pt", map_location = self.device))

        self.conv0.eval() 
        self.deconv0.eval() 
        self.reward_model.eval() 


    
    def _layers_to_model(self, layers):

        for i in range(len(layers)):
            if isinstance(layers[i], nn.Conv2d) or isinstance(layers[i], nn.Linear):
                torch.nn.init.xavier_uniform_(layers[i].weight)

        model = nn.Sequential(*layers)
        model.to(self.device)

        return model


if __name__ == "__main__":
    batch_size = 8

    channels = 1
    height   = 4
    width    = 4

    actions_count = 4


    state   = torch.rand((batch_size, channels, height, width))
    action  = torch.rand((batch_size, actions_count))


    model = Model((channels, height, width), actions_count)


    y, r = model.forward(state, action)

    print(y.shape, r.shape)
