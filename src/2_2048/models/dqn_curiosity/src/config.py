import common.decay

class Config(): 

    def __init__(self):
        self.gamma = 0.99

        self.update_frequency = 16

        self.batch_size     = 32 
        self.learning_rate  = 0.0001

        self.exploration = common.decay.Exponential(0.999999, 1.0, 0.02, 0.02)

        self.experience_replay_size = 8192


        self.curiosity_update_steps     = 8
        self.curiosity_learning_rate    = 0.0001
        self.curiosity_buffer_size      = 4096
        self.curiosity_beta             = 1.0
 

