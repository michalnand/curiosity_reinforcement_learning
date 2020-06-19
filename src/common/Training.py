import numpy
from .TrainingLog import *


class TrainingEpisodes:
    def __init__(self, env, agent, episodes_count, episode_max_length, saving_path, saving_period_episode = 500):
        self.env = env
        self.agent = agent
        self.episodes_count = episodes_count
        self.episode_max_length = episode_max_length
        self.saving_path = saving_path
        self.saving_period_episode = saving_period_episode

    def run(self):
        
        log = TrainingLog(self.saving_path + "result/result.log", self.saving_period_episode)
        new_best = False

        for episode in range(self.episodes_count):

            self.env.reset()
            steps = 0
            while True:
                reward, done = self.agent.main()
                
                steps+= 1

                if log.is_best:
                    new_best = True

                if steps >= self.episode_max_length:
                    log.add(reward, True)
                    break

                if done:
                    log.add(reward, True)
                    break 

                log.add(reward, False)
                
            if episode%self.saving_period_episode == 0 and new_best == True:
                new_best = False 
                print("\n\n")
                print("saving new best with score = ", log.episode_score_best)
                self.agent.save(self.saving_path)
                print("\n\n")

            

        if new_best == True: 
            new_best = False 
            print("\n\n")
            print("saving new best with score = ", log.episode_score_best)
            self.agent.save(self.saving_path)
            print("\n\n")






class TrainingIterations:
    def __init__(self, env, agent, iterations_count, saving_path, saving_period_iterations = 10000):
        self.env = env
        self.agent = agent

        self.iterations_count = iterations_count
     
        self.saving_path = saving_path
        self.saving_period_iterations = saving_period_iterations

    def run(self):
        
        log = TrainingLog(self.saving_path + "result/result.log", self.iterations_count, True)
        new_best = False

        for iteration in range(self.iterations_count):
            reward, done = self.agent.main()
            log.add(reward, done)
            
            if log.is_best:
                new_best = True

            if iteration%self.saving_period_iterations == 0 and new_best == True:
                new_best = False 
                print("\n\n")
                print("saving new best with score = ", log.episode_score_best)
                self.agent.save(self.saving_path)
                print("\n\n")

            
        if new_best == True: 
            new_best = False 
            print("\n\n")
            print("saving new best with score = ", log.episode_score_best)
            self.agent.save(self.saving_path)
            print("\n\n")
