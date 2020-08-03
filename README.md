# curiosity_reinforcement_learning

some experiments with curiosity and imagination learning

playing with DQN and DDPG, with mutlihead critic or multihead curiosity module
![](images/graph.png)


# Line follower

![](images/line_follower.gif)

![](src/0_line_follower/results/training_score_per_episode.png)

* DDPG : common ddpg
* DDPG + imagination : DDPG imagination (4 rollouts + 4 steps) and bonus reward from imagination
* DDPG multihead + imagination : DDPG with multihead critic and imagination (4 rollouts + 4 steps) and bonus reward from imagination,



# pybullet Ant walking

![](images/ant.gif)

![](src/1_ant/results/training_score_per_episode.png)

* DDPG : common ddpg
* DDPG + imagination : DDPG imagination (4 rollouts + 4 steps) and bonus reward from imagination
* DDPG multihead + imagination : DDPG with multihead critic and imagination (4 rollouts + 4 steps) and bonus reward from imagination,
