import common.decay

class Config(): 

    def __init__(self):
        self.gamma = 0.99
        self.update_frequency = 4
        self.update_target_frequency = 10000

        self.batch_size     = 64 
        self.learning_rate  = 0.0001
        self.bellman_steps  = 4
        
        #self.exploration    = common.decay.Linear(1000000, 1.0, 0.05, 0.02)
        self.exploration     = common.decay.Exponential(0.999999, 1.0, 0.1, 0.02)
        
        self.experience_replay_size = 16384

        self.curiosity_beta             = 100.0
        self.curiosity_learning_rate    = 0.0001 

