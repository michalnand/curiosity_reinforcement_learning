class Config():

    def __init__(self):        
        self.gamma          = 0.99

        self.eps_clip           = 0.2
        self.entropy_beta       = 0.01

        self.buffer_size        = 1024
        self.batch_size         = 64 
        self.training_epochs    = 4 

        self.learning_rate  = 0.00025
        


