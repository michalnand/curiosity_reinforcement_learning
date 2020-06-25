class Config():

    def __init__(self):        
        self.gamma          = 0.99
        self.learning_rate  = 0.001

        self.entropy_beta   = 0.01

        self.batch_size     = 64
        
        self.curiosity_update_steps = 8
        self.curiosity_learning_rate = 0.001
        self.curiosity_buffer_size   = 4096
        self.curiosity_beta     = 1.0