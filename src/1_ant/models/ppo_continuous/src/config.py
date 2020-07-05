class Config():

    def __init__(self):        
        self.gamma          = 0.99
        self.learning_rate      = 0.0002

        self.eps_clip           = 0.2
        self.entropy_beta       = 0.01

        self.batch_size         = 64
        self.buffer_size        = 1024
        self.training_epochs    = 4
        


