"""
author: Jamie Kang
jamiekang@stanford.edu

"""

import numpy as np

class ContextualBandit(object):
	def __init__(self, n, p, k, diversity=True):
		self.n = n
		self.p = p
		self.k = k
		self.diversity = diversity

		self.set_covariates()
		self.set_rewards()
	def __str__(self):
		pass

	def set_covariates(self):
		if self.diversity:
			self.covariates = np.random.uniform(low=-1,high=1,size=(self.n, self.p))
			# To satisfy covariate diversity, MUST contain the origin

		else:
			self.covariates = np.random.uniform(size=(self.n, self.p))

	def set_rewards(self):
		self.betas = np.random.uniform(size=(self.p, self.k)) #NOTE: Change this
		self.rewards = self.compute_rewards(self.covariates, self.betas)
		# self.rewards = np.dot(self.covariates, self.betas)

	def compute_rewards(self, x, betas):
		rewards = np.dot(x,betas)
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