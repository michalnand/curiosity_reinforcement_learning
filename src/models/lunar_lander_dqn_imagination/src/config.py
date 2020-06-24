import common.decay

class Config(): 

    def __init__(self):
        self.gamma = 0.99

        self.update_frequency = 4
        self.batch_size     = 32 
        self.exploration    = common.decay.Linear(60000, 1.0, 0.1, 0.1)
        self.learning_rate  = 0.0001

        self.experience_replay_size = 8192

        self.rollouts              = 8
        self.forward_ahead_steps   = 512
       
        self.imagination_learning_rate = 0.001 
        self.imagination_buffer_size   = 1024
 

