import numpy
import time

class TrainingLog:

    def __init__(self, file_name = None, episode_skip_log = 10):

        self.iterations         = 0
        self.episodes           = 0
        self.episode_score_sum = 0.0
        self.episode_iterations_sum = 0.0
        self.total_score        = 0.0

        self.episode_score_sum_filtered = 0.0

        self.episode_score_best = -10**6

        self.episode_time_prev = time.time()
        self.episode_time_now  = time.time()
        self.episode_time_filtered = 0.0

        self.is_best = False

        self.episode_skip_log   = episode_skip_log

        self.file_name = file_name

        if self.file_name != None:
            f = open(self.file_name,"w+")
            f.close()

    def add(self, reward, done):

        self.total_score+= reward
        self.episode_score_sum+= reward
        self.episode_iterations_sum+= 1

        self.iterations+= 1

        self.is_best = False

        if done:
            self.episodes+= 1

            k = 0.1
            self.episode_score_sum_filtered = (1.0 - k)*self.episode_score_sum_filtered + k*self.episode_score_sum
            
            self.episode_time_prev = self.episode_time_now
            self.episode_time_now  = time.time()
            self.episode_time_filtered = (1.0 - k)*self.episode_time_filtered + k*(self.episode_time_now - self.episode_time_prev)

            if self.episodes > 20:
                if self.episode_score_sum_filtered > self.episode_score_best:
                    self.episode_score_best = self.episode_score_sum_filtered
                    self.is_best = True



            dp = 3

            log_str = ""
            log_str+= str(self.iterations) + " "
            log_str+= str(self.episodes) + " "
            log_str+= str(round(self.episode_iterations_sum, dp)) + " "
            log_str+= str(round(self.total_score, dp)) + " "
            log_str+= str(round(self.episode_score_sum, dp)) + " "
            log_str+= str(round(self.episode_score_sum_filtered, dp)) + " "
            log_str+= str(round(self.episode_time_filtered, 4)) + " "

            self.episode_score_sum = 0
            self.episode_iterations_sum = 0
            

            if self.episodes%self.episode_skip_log == 0:
                print(log_str)

                if self.file_name != None:
                    f = open(self.file_name,"a+")
                    f.write(log_str+"\n")
                    f.close()




