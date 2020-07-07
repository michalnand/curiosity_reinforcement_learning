import common.decay

class Config():

    def __init__(self):        
        self.gamma          = 0.99
        self.critic_learning_rate = 0.001
        self.actor_learning_rate  = 0.0001
        self.tau = 0.02

        self.batch_size         = 64
        self.update_frequency    = 4

        self.exploration   = common.decay.Exponential(0.99995, 1.0, 0.1, 0.02)
 
        
        self.experience_replay_size = 8192