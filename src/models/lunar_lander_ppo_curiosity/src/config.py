class Config():

    def __init__(self):        
        self.gamma          = 0.99

        self.update_iterations  = 2048
        self.eps_clip           = 0.2
        self.entropy_beta       = 0.01
        self.training_epochs    = 4

        self.learning_rate      = 0.002

        self.curiosity_learning_rate = 0.00001
        self.curiosity_scale         = 10.0
        self.curiosity_buffer_size   = 1024
        


