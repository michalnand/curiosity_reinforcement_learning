class Config():

    def __init__(self):        
        self.gamma          = 0.99
        self.learning_rate      = 0.001

        self.eps_clip           = 0.2
        self.entropy_beta       = 0.01

        self.batch_size         = 256
        self.buffer_size        = 1024
        self.training_epochs    = 4


        self.curiosity_update_steps     = 8
        self.curiosity_learning_rate    = 0.0001
        self.curiosity_buffer_size      = 4096
        self.curiosity_beta             = 100.0
        


