"""
@author: Jamie Kang
"""

import numpy as np
from bandit import ContextualBandit
from agents import Agent, Agent_LASSO
from sklearn import linear_model

np.random.seed(213)

## Hyperparameters for the contextual bandit model
k = 5 # number of arms
p = 5 # covariate dimension
n = 1000 # number of data

## Hyperparameters for the bandit agent
h = 0.1

## Initialize bandit model
bandit = ContextualBandit(n,p,k)
X = bandit.covariates
rewards = bandit.rewards
betas = bandit.betas

## Initialize agent
agent = Agent_LASSO(n=n, h=h, k=k, lam=0.01)


## Run agent
for t in range(n):
	if  t%10 == 0:
		progress = int(t/n * 100)
		print(progress, "%","done!")
	x_t = X[t,]
	if agent.FS_schedule[t] == -1: # Greedy
		arm_t = agent._GS_decision(x_t, FS_filter=True)
		y_t = bandit.get_rewards(x_t)[arm_t]
		agent._update_GS_data(x_t, y_t, arm_t)

	else: # Forced sampling
		arm_t = agent.FS_schedule[t]
		y_t = bandit.get_rewards(x_t)[arm_t]
		agent._update_FS_data(x_t, y_t, arm_t)

	agent._update_history(x_t, y_t, arm_t)


# print("Predicted rewards: ", agent.history[(n-10):n,1])
print("Pulled arms: ", agent.history[(n-10):n,2])
true_arms = np.argmax(rewards[(n-10):n, :], axis=1)
print("True arms: ", true_arms)

