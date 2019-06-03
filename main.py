"""
author: Jamie Kang
jamiekang@stanford.edu

"""

import numpy as np
from bandit import ContextualBandit
from agents import Agent, Agent_LASSO, Agent_OLS, Agent_MARS, Agent_MARS2
from sklearn import linear_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

## Set random seed
np.random.seed(123)

## Hyperparameters for the contextual bandit model
k = 2 # number of arms
p = 30 # covariate dimension
# p = 100 # covariate dimension
n = 1000 # number of data

## Hyperparameters for the bandit agent
h = 5

## Initialize bandit model
bandit = ContextualBandit(n,p,k, diversity=True, reward_type=4)
print("True params:",)

X = bandit.covariates
rewards = bandit.rewards
betas = bandit.betas

## Initialize agent: Uncomment the lines that correspond to agents in use
agentList = []
# agentList.append(Agent_OLS(n=n, h=h, k=k, greedy_only=True, name= "Greedy_OLS"))
# agentList.append(Agent_OLS(n=n, h=h, k=k, greedy_only=False, name= "OLS"))
# agentList.append(Agent_OLS(n=n, h=h, k=k, p=p, greedy_only=False, basis_expansion=True, name= "OLS_BE"))
# agentList.append(Agent_LASSO(n=n, h=h, k=k, greedy_only=False, lam= 0.05, name= "LASSO"))
agentList.append(Agent_LASSO(n=n, h=h, k=k, p=p, greedy_only=False, basis_expansion=True, lam= 0.05, name= "LASSO_BE"))
agentList.append(Agent_MARS2(n=n, h=h, k=k, greedy_only=False, name="MARS2"))
# agentList.append(Agent_MARS(n=n, h=h, k=k, greedy_only=False, name="MARS"))

# agentList.append(Agent_LASSO(n=n, h=h, k=k, greedy_only=False, lam=0.01, name = "LASSO"))

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for agent in agentList:
	## Run agent
	optimal_rewards = np.zeros(n) 
	rewards = np.zeros(n) 

	for t in range(n):
		if  t%10 == 0:
			progress = int(t/n * 100)
			print(progress, "%","done!")
		x_t = X[t,]
		if agent.FS_schedule[t] == -1: # Greedy
			if agent.greedy_only:
				arm_t = agent._GS_decision(x_t, FS_filter=False)
			else:
				arm_t = agent._GS_decision(x_t, FS_filter=True)
			y_t = bandit.rewards[t,arm_t]
			agent._update_GS_data(x_t, y_t, arm_t)

		else: # Forced sampling
			arm_t = int(agent.FS_schedule[t])
			y_t = bandit.rewards[t,arm_t]
			agent._update_FS_data(x_t, y_t, arm_t)

		# Log the history
		agent._update_history(x_t, y_t, arm_t)
		reward = y_t		
		optimal_rewards[t] = bandit.rewards[t,:].max()
		# optimal_rewards[t] = bandit.get_true_arm_reward(x_t)[1] 
		rewards[t] = reward 

	## Sanity check
	# print("Arm: ", arm_t)
	# print("Predicted params: ", agent.params)
	# print("True params: ", bandit.betas)

	# print("Predicted rewards: ", agent.history[(n-10):n,1])
	# print("Pulled arms: ", agent.history[(n-10):n,2])
	# true_arms = np.argmax(rewards[(n-10):n, :], axis=1)
	# print("True rewards: ", optimal_rewards[(n-10):n])
	# print("True arms: ", true_arms)

	regrets = optimal_rewards - rewards
	cum_regret = np.cumsum(regrets) 
	avg_regret = cum_regret[0:n] / np.array(range(1,n+1))
	cum_name = "p30_reward4_"+str(agent)+"_cum.csv"
	avg_name = "p30_reward4_"+str(agent)+"_avg.csv"
	np.savetxt(cum_name, cum_regret,delimiter=",")
	np.savetxt(avg_name, avg_regret,delimiter=",")

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
