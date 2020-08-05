import common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.9
        self.critic_learning_rate   = 0.0005
        self.actor_learning_rate    = 0.00025
        self.tau                    = 0.001

        self.batch_size          = 64
        self.update_frequency    = 4

        self.exploration   = common.decay.Linear(1000000, 1.0, 0.3, 0.3)

        self.experience_replay_size = 16384