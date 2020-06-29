class Config():

    def __init__(self):        
        self.gamma          = 0.99
        self.learning_rate  = 0.001

        self.entropy_beta   = 0.01

        self.batch_size     = 8
        self.rollouts       = 128
        
        self.model_env_update_steps = 8
        self.model_env_learning_rate = 0.001
        self.model_env_buffer_size   = 4096
