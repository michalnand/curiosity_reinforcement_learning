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


        layer_0_kernels_count = 32
        layer_1_kernels_count = 32
        layer_2_kernels_count = 32 
        layer_3_kernels_count = 32

        hidden_count = 16**2

        self.scale_ratio           = 2**4

        fc_inputs_count = ((input_width)//self.scale_ratio)*((input_height)//self.scale_ratio)*layer_3_kernels_count
        
        layers_encoder = [ 
                            nn.Conv2d(input_channels, layer_0_kernels_count, kernel_size=3, stride=2, padding=1),
                            nn.ReLU(), 

                            nn.Conv2d(layer_0_kernels_count, layer_1_kernels_count, kernel_size=3, stride=2, padding=1),
                            nn.ReLU(), 

                            nn.Conv2d(layer_1_kernels_count, layer_2_kernels_count, kernel_size=3, stride=2, padding=1),
                            nn.ReLU(), 

                            nn.Conv2d(layer_2_kernels_count, layer_3_kernels_count, kernel_size=3, stride=2, padding=1),
                            nn.ReLU(),                         
                        ]


        layers_decoder = [
                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.Conv2d(layer_3_kernels_count + outputs_count, layer_3_kernels_count, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),

                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.Conv2d(layer_3_kernels_count, layer_2_kernels_count, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(), 

                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.Conv2d(layer_2_kernels_count, layer_1_kernels_count, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(), 

                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.Conv2d(layer_1_kernels_count, layer_0_kernels_count, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(), 

                            nn.Conv2d(layer_0_kernels_count, input_channels, kernel_size=1, stride=1, padding=0)
                        ]

        self.model_encoder = self._layers_to_model(layers_encoder)
        self.model_decoder = self._layers_to_model(layers_decoder)

      
        print(self.model_encoder)
        print(self.model_decoder)

    def forward(self, state, action):
        action_ = action.unsqueeze(1).unsqueeze(1).transpose(3, 1).repeat((1, 1, self.input_shape[1]//self.scale_ratio, self.input_shape[2]//self.scale_ratio))

        encoder_output     = self.model_encoder(state)

        decoder_input      = torch.cat([encoder_output, action_], dim = 1)

        
        return self.model_decoder(decoder_input)

    def save(self, path):
        print("saving ", path)

        torch.save(self.model_encoder.state_dict(), path + "trained/model_curiosity_encoder.pt")
        torch.save(self.model_decoder.state_dict(), path + "trained/model_curiosity_decoder.pt")

    def load(self, path):
        print("loading ", path, " device = ", self.device) 

        self.model_encoder.load_state_dict(torch.load(path + "trained/model_curiosity_encoder.pt", map_location = self.device))
        self.model_decoder.load_state_dict(torch.load(path + "trained/model_curiosity_decoder.pt", map_location = self.device))
        
        self.model_encoder.eval() 
        self.model_decoder.eval() 


    
    def _layers_to_model(self, layers):

        for i in range(len(layers)):
            if isinstance(layers[i], nn.Conv2d) or isinstance(layers[i], nn.Linear):
                torch.nn.init.xavier_uniform_(layers[i].weight)

        model = nn.Sequential(*layers)
        model.to(self.device)

        return model


if __name__ == "__main__":
    batch_size = 1

    channels = 4
    height   = 96
    width    = 96

    actions_count = 7


    state   = torch.rand((batch_size, channels, height, width))
    action  = torch.rand((batch_size, actions_count))


    model = Model((channels, height, width), actions_count)


    y = model.forward(state, action)

    print(y.shape)
