import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape 
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]   

        kernels_count = 32

        fc_input_height = input_shape[1]//(4*2*2)
        fc_input_width  = input_shape[2]//(4*2*2)

        self.conv0 = nn.Sequential( 
                                    nn.Conv2d(input_channels + outputs_count, kernels_count, kernel_size=4, stride=4, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(kernels_count, kernels_count, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
                                    nn.Conv2d(kernels_count, kernels_count, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
        )

        self.deconv0 = nn.Sequential(
                                        nn.ConvTranspose2d(kernels_count, input_channels, kernel_size=4, stride=4, padding=0),
                                    )

        self.reward = nn.Sequential(
                                            nn.Conv2d(kernels_count, kernels_count, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(),

                                            nn.Conv2d(kernels_count, kernels_count, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(),

                                            Flatten(),
                                            nn.Linear(fc_input_height*fc_input_width*kernels_count, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 1)
        ) 

        self.conv0.to(self.device)
        self.conv1.to(self.device) 
        self.deconv0.to(self.device) 
        self.reward.to(self.device) 

        print(self.conv0)
        print(self.conv1)
        print(self.deconv0)  
        print(self.reward)                      

    def forward(self, state, action):
        action_ = action.unsqueeze(1).unsqueeze(1).transpose(3, 1).repeat((1, 1, self.input_shape[1], self.input_shape[2])).to(self.device)

        model_input      = torch.cat([state, action_], dim = 1)
        conv0_output     = self.conv0(model_input)
        conv1_output     = self.conv1(conv0_output)

        tmp = conv0_output + conv1_output

        observation_prediction = self.deconv0(tmp)
        reward_prediction      = self.reward(tmp)
        
        return observation_prediction + state, reward_prediction

    def save(self, path):
        print("saving ", path)

        torch.save(self.conv0.state_dict(), path + "trained/model_curiosity_conv0.pt")
        torch.save(self.conv1.state_dict(), path + "trained/model_curiosity_conv1.pt")
        torch.save(self.deconv0.state_dict(), path + "trained/model_curiosity_deconv0.pt")
        torch.save(self.reward.state_dict(), path + "trained/model_curiosity_reward.pt")


    def load(self, path):
        print("loading ", path, " device = ", self.device) 

        self.conv0.load_state_dict(torch.load(path + "trained/model_curiosity_conv0.pt", map_location = self.device))
        self.conv1.load_state_dict(torch.load(path + "trained/model_curiosity_conv1.pt", map_location = self.device))
        self.deconv0.load_state_dict(torch.load(path + "trained/model_curiosity_deconv0.pt", map_location = self.device))
        self.reward.load_state_dict(torch.load(path + "trained/model_curiosity_reward.pt", map_location = self.device))

        self.conv0.eval() 
        self.conv1.eval() 
        self.conv2.eval() 
        self.reward.eval() 


    
    def _layers_to_model(self, layers):

        for i in range(len(layers)):
            if isinstance(layers[i], nn.Conv2d) or isinstance(layers[i], nn.Linear):
                torch.nn.init.xavier_uniform_(layers[i].weight)

        model = nn.Sequential(*layers)
        model.to(self.device)

        return model


if __name__ == "__main__":
    batch_size = 8

    channels = 4
    height   = 96
    width    = 96

    actions_count = 7


    state   = torch.rand((batch_size, channels, height, width))
    action  = torch.rand((batch_size, actions_count))


    model = Model((channels, height, width), actions_count)


    y, r = model.forward(state, action)

    print(y.shape)
    print(r.shape)
