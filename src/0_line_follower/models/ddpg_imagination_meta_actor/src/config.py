import common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.9
        self.critic_learning_rate   = 0.001
        self.actor_learning_rate    = 0.0005
        self.tau                    = 0.001

        self.batch_size          = 64
        self.update_frequency    = 4

        self.exploration   = common.decay.Linear(50000, 1.0, 0.2, 0.2)
  
        self.experience_replay_size = 16384

        self.imagination_beta             = 1.0
        self.entropy_beta                 = 1.0
        self.imagination_rollouts         = 4
        self.imagination_steps            = 4
        self.imagination_learning_rate    = 0.001 
