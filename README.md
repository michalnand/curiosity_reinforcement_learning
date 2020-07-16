# curiosity_reinforcement_learning

some experiments with curiosity learning


# lunar lander results

![](images/lunar_lander_ppo.gif)

![](results/training_score_per_iterations.png)
![](results/training_score_per_episode.png)

* note : A2C was running in 8 paralel environments, total number of iterations (games) need to be multiplied by 8



# ANT  results

![](images/ant.gif)

![](src/1_ant/results/training_score_per_episode.png)

* DDPG : common ddpg
* DDPG + curiosity : simple environment forward model added
* DDPG + multihead curiosity : multiple environment forward models, controlled by attention mechanism