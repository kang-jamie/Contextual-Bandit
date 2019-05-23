"""
author: Jamie Kang
jamiekang@stanford.edu

"""

import numpy as np
from bandit import ContextualBandit
from agents import Agent, Agent_LASSO
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

## Set random seed
np.random.seed(312)

## Hyperparameters for the contextual bandit model
k = 2 # number of arms
p = 5 # covariate dimension
n = 4000 # number of data

## Hyperparameters for the bandit agent
h = 0.5

## Initialize bandit model
bandit = ContextualBandit(n,p,k)

X = bandit.covariates
rewards = bandit.rewards
betas = bandit.betas

## Initialize agent
agent = Agent_LASSO(n=n, h=h, k=k, greedy_only=True, lam=0.01)
# agent = Agent_LASSO(n=n, h=h, k=k, greedy_only=True, lam=0.01)

## Run agent
cum_reward = np.zeros(n)
cum_true_reward = np.zeros(n)
cum_regret = np.zeros(n)
true_rewards = np.zeros(n)

prev_reward = 0
prev_true_reward = 0
prev_cum_regret = 0

for t in range(n):
	if  t%100 == 0:
		progress = int(t/n * 100)
		print(progress, "%","done!")
	x_t = X[t,]
	if agent.FS_schedule[t] == -1: # Greedy
		arm_t = agent._GS_decision(x_t, FS_filter=True)
		y_t = bandit.get_rewards(x_t)[arm_t]
		agent._update_GS_data(x_t, y_t, arm_t)

	else: # Forced sampling
		arm_t = int(agent.FS_schedule[t])
		y_t = bandit.get_rewards(x_t)[arm_t]
		agent._update_FS_data(x_t, y_t, arm_t)

	# Log the history
	agent._update_history(x_t, y_t, arm_t)
	reward = y_t
	true_reward = bandit.get_true_arm_reward(x_t)[1]
	cum_reward[t] = prev_reward + reward
	cum_true_reward[t] = prev_true_reward + true_reward
	cum_regret[t] = prev_cum_regret + true_reward - reward
	prev_reward = reward
	prev_true_reward = true_reward
	# prev_regret = true_reward - reward
	prev_cum_regret = prev_cum_regret + true_reward - reward
	true_rewards[t] = true_reward

## Sanity check
print("Predicted rewards: ", agent.history[(n-10):n,1])
print("Pulled arms: ", agent.history[(n-10):n,2])
true_arms = np.argmax(rewards[(n-10):n, :], axis=1)
print("True rewards: ", true_rewards[(n-10):n])
print("True arms: ", true_arms)

## Plot the results
roll_mean_cum_regret = pd.DataFrame(cum_regret).rolling(1).mean() #to plot moving average for less noise
plt.figure()
plt.plot(roll_mean_cum_regret, label=str(agent))
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.ylim([0, None])
# plt.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, -0.25))
plt.show()

