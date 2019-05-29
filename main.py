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
# p = 3 # covariate dimension
p = 100 # covariate dimension
n = 10000 # number of data

## Hyperparameters for the bandit agent
h = 0.25

## Initialize bandit model
bandit = ContextualBandit(n,p,k, diversity=True)

X = bandit.covariates
rewards = bandit.rewards
betas = bandit.betas

## Initialize agent
agentList = []
agentList.append(Agent_OLS(n=n, h=h, k=k, greedy_only=True, name= "Greedy_OLS"))
agentList.append(Agent_OLS(n=n, h=h, k=k, q=20, greedy_only=False, name= "OLS with q=20"))
# agentList.append(Agent_OLS(n=n, h=h, k=k, q=50, greedy_only=False, name= "OLS with q=50"))

# agentList.append(Agent_LASSO(n=n, h=h, k=k, greedy_only=False, lam=0.01, name = "LASSO"))

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for agent in agentList:
	## Run agent
	optimal_rewards = np.zeros(n) 
	rewards = np.zeros(n) 

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
		optimal_rewards[t] = bandit.get_true_arm_reward(x_t)[1] 
		rewards[t] = reward 

	## Sanity check
	# print("Predicted rewards: ", agent.history[(n-10):n,1])
	# print("Pulled arms: ", agent.history[(n-10):n,2])
	# true_arms = np.argmax(rewards[(n-10):n, :], axis=1)
	# print("True rewards: ", true_rewards[(n-10):n])
	# print("True arms: ", true_arms)

	regrets = optimal_rewards - rewards
	cum_regret = np.cumsum(regrets)
	avg_regret = cum_regret / (np.array(range(n))+1)
	avg_regret[0] = avg_regret[1]

	## Plot the results
	ax1.plot(cum_regret, label=str(agent))
	ax2.plot(avg_regret, label=str(agent))

ax1.legend(frameon=False)
ax1.set_xlabel("t")
ax1.set_ylabel("Cumulative regret")
ax2.legend(frameon=False)
ax2.set_xlabel("t")
ax2.set_ylabel("Average regret")

plt.show()
