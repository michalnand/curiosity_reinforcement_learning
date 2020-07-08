



class ServoEnv:
    def __init__(self):

        self.k_min = 0.1
        self.k_max = 10.0

        self.a_min = 0.7
        self.a_max = 0.98

        self.time_steps = 8

    def reset(self):
        self.position        = 0.0
        self.velocity        = 0.0

        r = numpy.random.rand()
        self.k     = r*self.k_min + (1.0 - r)(self.k_max)
        
        r = numpy.random.rand()
        self.a     = r*self.a_min + (1.0 - r)(self.a_max)

        self.target_position = numpy.random.rand()*2.0 - 1.0

        self.state = numpy.zeros((self.time_steps, 2))


    def step(self, action):
        force = action[0]

        self.velocity = self.a*self.velocity + (1.0 - self.a)*self.k*force
        self.position = self.position + self.velocity*0.01

        if self.position > 1.0:
            self.position = 1.0
        
        if self.position < -1.0:
            self.position = -1.0

        dif = numpy.abs(self.target_position - self.position)
        reward = -0.001
        reward+= dif
        
        if dif < 0.02 and numpy.abs(self.velocity) < 0.1:
            done = True
        else:
            done = False


    def _get_state(self):
        state[0][0] = self.position
        state[0][1] = self.velocity

        return state
         


