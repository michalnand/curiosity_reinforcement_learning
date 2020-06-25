class Config():

    def __init__(self):        
        self.gamma          = 0.99
        self.learning_rate  = 0.001

        self.entropy_beta   = 0.01

        self.batch_size     = 16
        self.rollouts       = 32
        
        self.imagination_update_steps = 8
        self.imagination_learning_rate = 0.001
        self.imagination_buffer_size   = 4096
