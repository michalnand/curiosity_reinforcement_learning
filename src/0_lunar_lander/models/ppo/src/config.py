class Config():

    def __init__(self):        
        self.gamma          = 0.99
        self.learning_rate      = 0.001

        self.eps_clip           = 0.2
        self.entropy_beta       = 0.01

        self.batch_size         = 64
        self.training_epochs    = 4
        


