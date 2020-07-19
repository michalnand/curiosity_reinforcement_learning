import common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.9
        self.critic_learning_rate   = 0.0002
        self.actor_learning_rate    = 0.0001
        self.tau                    = 0.001

        self.batch_size             = 64
        self.update_frequency       = 4
        self.sample_size            = 16

        self.exploration   = common.decay.Linear(1000000, 1.0, 0.2, 0.2)

        self.experience_replay_size = 256 #16384