import common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.99
        self.learning_rate          = 0.0001

        self.batch_size          = 64 
        self.update_frequency    = 4

        self.exploration   = common.decay.Linear(200000, 1.0, 0.1, 0.05)
        #self.exploration   = common.decay.Exponential(0.99998, 1.0, 0.05, 0.05)
  
        self.experience_replay_size = 8192