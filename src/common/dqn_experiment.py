import numpy
from .RLStats import *

class DQNExperiment:
    def __init__(self, env, ModelDQN, ModelCuriosity, Config, Agent, rounds = 100, training_iterations = 200000):
        
        self.env = env

        self.ModelDQN = ModelDQN
        self.ModelCuriosity = ModelCuriosity
        self.Config = Config
        self.Agent = Agent

        self.rounds = rounds
        self.training_iterations = training_iterations


        
    def process(self, result_file_name = "./experiment.log"):
        self.saving_iterations = 100

        rl_stats = RLStats()

        for r in range(self.rounds):
            print("playing round ", r, " done = ", round(100.0*r/self.rounds, 2), "%")
            iterations_, episodes_, score_result_, score_per_episode_result_ = self.process_episode()

            rl_stats.add(iterations_, episodes_, score_result_, score_per_episode_result_ )

        rl_stats.save(result_file_name)
        
        
    def process_episode(self):
        self.env.reset()

        obs             = self.env.observation_space
        actions_count   = self.env.action_space.n

        agent = self.Agent(self.env, self.ModelDQN, self.ModelCuriosity, self.Config)

        score   = 0
        episode = 0

        score_game_prev = 0
        score_game_now  = 0
        score_per_game_filtered = 0

        iterations_total = []
        episodes_total = []
        score_result = []
        score_per_game_result = []

        while agent.iterations < self.training_iterations:
            reward, done = agent.main()

            score+= reward

            if done:
                self.env.reset()
                episode+= 1

                score_game_prev = score_game_now
                score_game_now  = score

                score_per_game  = score_game_now - score_game_prev

                k = 0.2
                score_per_game_filtered = (1.0 - k)*score_per_game_filtered + k*score_per_game

                print("\t\t episode = ", episode, score_per_game_filtered, round(agent.iterations*100.0/self.training_iterations, 2))


            if agent.iterations%self.saving_iterations == 0:
                iterations_total.append(agent.iterations)
                episodes_total.append(episode)
                score_result.append(score)
                score_per_game_result.append(score_per_game_filtered)


        return iterations_total, episodes_total, score_result, score_per_game_result
 