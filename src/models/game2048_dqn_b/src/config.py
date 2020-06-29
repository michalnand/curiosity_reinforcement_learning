import common.decay

class Config(): 

    def __init__(self):
        self.gamma = 0.99

        self.update_frequency = 4

        self.batch_size     = 32 
        self.learning_rate  = 0.0001
        
        self.exploration = common.decay.Exponential(0.999999, 1.0, 0.02, 0.02)

        self.experience_replay_size = 8192
 

