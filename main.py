"""
author: Jamie Kang
jamiekang@stanford.edu

"""

import numpy as np
from bandit import ContextualBandit
from agents import Agent, Agent_LASSO, Agent_OLS
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

## Set random seed
np.random.seed(111)

## Hyperparameters for the contextual bandit model
k = 2 # number of arms
p = 3 # covariate dimension
n = 10000 # number of data

## Hyperparameters for the bandit agent
h = 0.25

## Initialize bandit model
bandit = ContextualBandit(n,p,k, diversity=False)

X = bandit.covariates
rewards = bandit.rewards
betas = bandit.betas

## Initialize agent
agentList = []
# agentList.append(Agent_LASSO(n=n, h=h, k=k, greedy_only=True, lam=0.01, name= "Greedy_LASSO"))
agentList.append(Agent_OLS(n=n, h=h, k=k, greedy_only=True, name= "Greedy_OLS"))
agentList.append(Agent_OLS(n=n, h=h, k=k, q=20, greedy_only=False, name= "OLS with q=20"))
agentList.append(Agent_OLS(n=n, h=h, k=k, q=50, greedy_only=False, name= "OLS with q=50"))

plt.figure()

for agent in agentList:
	## Run agent
	cum_reward = np.zeros(n)
	cum_true_reward = np.zeros(n)
	cum_regret = np.zeros(n)
	true_rewards = np.zeros(n)
	avg_regret_per_period = np.zeros(n)
	prev_reward = 0
	prev_true_reward = 0
	prev_cum_regret = 0

	for t in range(n):
		if  t%100 == 0:
			progress = int(t/n * 100)
			print(progress, "%","done!")
		x_t = X[t,]
		if agent.FS_schedule[t] == -1: # Greedy
			if agent.greedy_only:
				arm_t = agent._GS_decision(x_t, FS_filter=False)
			else:
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
		prev_cum_regret = prev_cum_regret + true_reward - reward
		true_rewards[t] = true_reward
		if t>= 1:
			avg_regret_per_period[t] = cum_regret[t] / t

	## Sanity check
	# print("Predicted rewards: ", agent.history[(n-10):n,1])
	# print("Pulled arms: ", agent.history[(n-10):n,2])
	# true_arms = np.argmax(rewards[(n-10):n, :], axis=1)
	# print("True rewards: ", true_rewards[(n-10):n])
	# print("True arms: ", true_arms)

	## Plot the results
	avg_regret_per_period[0] = avg_regret_per_period[1]
	roll_avg_regret_per_period = pd.DataFrame(avg_regret_per_period).rolling(50).mean() #to plot moving average for less noise
	# plt.plot(roll_avg_regret_per_period, label="roll_"+str(agent))
	# plt.plot(avg_regret_per_period, label=str(agent))
	plt.plot(cum_regret, label=str(agent))

plt.legend(frameon=False)
plt.xlabel("t")
# plt.ylabel("Avg regret per period")
plt.ylabel("Cumulative regret")
plt.show()

