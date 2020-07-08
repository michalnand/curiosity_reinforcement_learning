import common.decay

class Config():

    def __init__(self):        
        self.gamma          = 0.99
        self.critic_learning_rate = 0.0002
        self.actor_learning_rate  = 0.0001
        self.tau = 0.001

        self.batch_size          = 64
        self.update_frequency    = 4

        self.exploration   = common.decay.Linear(200000, 1.0, 0.5, 0.2)
  
        self.experience_replay_size = 16384

        
        self.curiosity_beta = 0.1
        self.curiosity_learning_rate = 0.0001 
        self.curiosity_buffer_size = 4069