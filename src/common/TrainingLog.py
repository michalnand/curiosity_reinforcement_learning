import numpy
import time

class TrainingLog:

    def __init__(self, file_name, episodes_count, episode_skip_log = 10, iterations_skip_mode = False):

        self.iterations         = 0
        self.episodes           = 0
        self.episode_score_sum = 0.0
        self.episode_iterations_sum = 0.0
        self.total_score        = 0.0
        self.episodes_count     = episodes_count

        self.iterations_skip_mode = iterations_skip_mode

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


                tmp = self.episodes_count//10
                if self.episodes%tmp == 0:
                    eta_time = (self.episodes_count - self.episodes)*self.episode_time_filtered
                    print("ETA time ", round(eta_time, 1), "[s]")
                    print("ETA time ", round(eta_time/3600, 1), "[hours]")
                    print("\n\n")

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
            
            log_result = False
            if self.iterations_skip_mode:
                if self.iterations%self.episode_skip_log == 0:
                    log_result = True
                elif self.episodes%self.episode_skip_log == 0:
                    log_result = True

            if log_result:
                print(log_str)

                if self.file_name != None:
                    f = open(self.file_name,"a+")
                    f.write(log_str+"\n")
                    f.close()




