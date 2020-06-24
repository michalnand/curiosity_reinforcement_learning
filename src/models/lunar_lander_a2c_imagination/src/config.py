class Config():

    def __init__(self):        
        self.gamma          = 0.99
        self.learning_rate  = 0.001

        self.entropy_beta   = 0.01

        self.rollouts               = 16
        self.forward_ahead_steps    = 32

        self.imagination_learning_rate = 0.001
        self.imagination_buffer_size   = 1024
        
