# Project 2 - Continuous Control in the Reacher Environment

<center>
	<img src="https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png" alt="drawing" width="480"/>
</center>

## Introduction

This project aims to train two tennis players agents at the same time capable of bounce the ball back and forth to each other using the Unity Tennis environment and getting as many rewards as possible. In this environment, the agent only collects a positive reward when it hits the ball back to the opponent otherwise it receives a negative reward. 

The agents are both rackets that can move and jump to hit the ball. The state vector in each time for each agent has 8 elements representing positions and velocities of both the racket and the ball. In each time step, the agent should take an action that has 2 inputs which move the racket and jump.

Here, we present the solution the environment using Multi-agent Deep Deterministic Policy Gradients (MADDPG) algorithm. Please, for a more detailed information about the states, actions and the details of the algorithm used to train the agent refer to the report.pdf file.

## Files in project

|  File | Description | 
|-------|-------------|
| Tennis.ipynb  | A notebook where we visualize the environment, train the agent and assess its performance. | 
| agents.py  | A python script where we define the modified DDPG agent. | 
| models.py  | A python script where we define the neural networks (Actor and Critic) used by the agent to estimate the action-value function. |
| report.pdf  | A more complete report of the environment, data and results |
| trained_models | Saved weights of the agent in the single arm environment for both actor and critic | 

## Installation and requirements

Plese, visit the Deep Reinforcement Learning [repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) maintained by Udacity in order to install all dependencies to work with the code used here. All the steps presented there can be break down into the steps in file requirements.txt:

```console
youruse@yourcomputer:~$ conda env create -f environment.yml
youruse@yourcomputer:~$ python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

Finally we should download and extract the reacher environment in the same folder of your project. Please, refer to the list below to download it:

- Linux: [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## The agent

Our agent uses a modified version of the Multi-Agent Deep Deterministic Policy Gradients (MADDPG) algorithm presented in the paper [Multi-Agent Actor Critic for Mixed Cooperative-Competitive Environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf) to train the agents on the environment and collect the highest reward during every single step continuously improving them in each episode.

### Training

The agents are trained in the Tennis.ipynb notebook running the cells presented in its training section. A typical training routine yields a result as the one below in which the agent with the parameters given in the parameters_dict variable in the notebook is capable of reaching an average score of 0.5 at the end of the last 100 episodes (the reward in each episode is given by the highest reward between the two agents). 

![The maximum rewards obtained by the agents at the end of each episode as well as the average of the last 100 episodes and a moving average with step size equal to 3.](https://github.com/joseilberto/project3_multi_agent_udacity/blob/master/images/scores.png)

## Report file

The report.pdf file is a more detailed document in which the detailed information of the agent and the neural networks used are presented. We also discuss further the results found and what improvements can be performed. 




