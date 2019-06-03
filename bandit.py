"""
author: Jamie Kang
jamiekang@stanford.edu

"""

import numpy as np

class ContextualBandit(object):
	def __init__(self, n, p, k, diversity=True, reward_type=0):
		self.n = n
		self.p = p
		self.k = k
		self.diversity = diversity
		self.reward_type = reward_type
		self.set_covariates()
		self.set_rewards()
	def __str__(self):
		pass

	def set_covariates(self):
		if self.diversity:
			self.covariates = np.random.uniform(low=-1,high=1,size=(self.n, self.p))
			# self.covariates = np.random.normal(loc=0,scale=1,size=(self.n, self.p))
		else:
			self.covariates = np.random.uniform(size=(self.n, self.p))
		# To satisfy covariate diversity, MUST contain the origin

	def set_rewards(self):

		if self.reward_type == -1:
			self.betas = np.random.uniform(size=(self.p,self.k))
		else:
			self.betas = np.random.uniform(size=(5, self.k)) #NOTE: Change this
		self.rewards = self.compute_rewards(self.covariates, self.betas)

	def compute_rewards(self, x, betas):
		if self.reward_type == 0: # Sparse linear reward
			x_used = x[:,0:5]
		elif self.reward_type == 1: # y = x0^2 + x1^2 + x2^2 + x3^2 + x4^2
			x_used = x[:,0:5]**2
		elif self.reward_type == 2:	# y = x0 + x1 + x2 + x0^2 + x1x2
			x_used0 = np.concatenate((x[:,0:3], np.reshape(x[:,0] ** 2,(-1,1))), axis=1)
			x_used = np.concatenate((x_used0, np.reshape(x[:,1] * x[:,2],(-1,1))), axis=1)
		elif self.reward_type == 3: # y = x0 + x1 + x2 + x0x1 + x1x2
			x_used0 = np.concatenate((x[:,0:3], np.reshape(x[:,0] * x[:,1],(-1,1))), axis=1)
			x_used = np.concatenate((x_used0, np.reshape(x[:,1] * x[:,2],(-1,1))), axis=1)
		elif self.reward_type == 4: # y = x0x1 + x1x2 + x2x3 + x3x4 + x4x5
			x_used1 = np.concatenate((np.reshape(x[:,0] * x[:,1],(-1,1)), np.reshape(x[:,1] * x[:,2],(-1,1))), axis=1)
			x_used2 = np.concatenate((x_used1, np.reshape(x[:,2] * x[:,3],(-1,1))), axis=1)
			x_used3 = np.concatenate((x_used2, np.reshape(x[:,3] * x[:,4],(-1,1))), axis=1)
			x_used = np.concatenate((x_used3, np.reshape(x[:,4] * x[:,5],(-1,1))), axis=1)
		else: # Non-sparse linear reward
			x_used = x
		rewards = np.dot(x_used, betas)
		return(rewards)


	def get_covariate(self, t):
		this_covaraite = self.covariates[t,]
		return this_covaraite

	def get_rewards(self, x):
		this_reward = self.compute_rewards(x, self.betas)
		# this_reward = np.dot(x, self.betas)
		return this_reward

	def get_true_arm_reward(self, x):
		this_reward = self.compute_rewards(x, self.betas)
		arm = np.argmax(this_reward)
		reward = this_reward.max()
		return [arm, reward]