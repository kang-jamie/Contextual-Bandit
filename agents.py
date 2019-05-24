"""
author: Jamie Kang
jamiekang@stanford.edu

"""

import numpy as np
from sklearn import linear_model

class Agent(object):
	def __init__(self, n, h, k, q=50, greedy_only=False, name=None, **kwargs):
		self.name = name
		self.h = h
		self.k = k
		self.n = n
		self.FS = np.array([])
		self.GS = np.array([])
		self.history = np.array([])
		self.greedy_only = greedy_only
		self.q = q
		if greedy_only:
			self.FS_schedule = np.zeros((self.n)) - 1
		else:
			self._FS_schedule()	

	def __str__(self):
		return self.name
		# pass

	def _feature_map(self, x):
		#TODO
		# z = 
		z = np.array(x)
		return x

	def _update_FS_data(self, x, y, arm):
		data = [x, y, arm]
		data = np.array(data)
		if self.FS.size==0:
			self.FS = data.copy()
			self.FS = np.reshape(self.FS, (-1,3)) #reshape to a 3-column row vector
		else:
			self.FS = np.vstack((self.FS, data))

	def _update_GS_data(self, x, y, arm):
		data = [x, y, arm]
		data = np.array(data)
		if self.GS.size==0:
			self.GS = data.copy()
			self.GS = np.reshape(self.GS, (-1,3)) 
		else:
			self.GS = np.vstack((self.GS, data))

	def _update_history(self, x, y, arm):
		data = [x, y, arm]
		data = np.array(data)
		if self.history.size==0:
			self.history = data.copy()
			self.history = np.reshape(self.history, (-1,3))
		else:
			self.history = np.vstack((self.history,data))

	def _FS_schedule(self):
		q = self.q
		FS_num = 0
		self.FS_schedule = np.zeros((self.n)) - 1 # initialize to all-greedy schedule (i.e. -1)
		for i in range(self.k):
			l = 0
			while ((2**l -1)*self.n*q + q*(i-1)+1) < self.n:
				for j in range((q*(i-1)+1),(q*i+1)):
					Ti = (2**l -1)*self.n*q + j
					if Ti < self.n:
						self.FS_schedule[Ti] = i
						FS_num = FS_num + 1
					else:
						break
				l = l+1
		# print("FS_num: ", FS_num)

	def _FS_decision(self, x):
		rewards = self._predict_rewards(self.FS, x)
		arm = self._pick_arm(rewards)
		return arm

	def _GS_decision(self, x, FS_filter = True):
		GS_rewards = self._predict_rewards(self.GS, x)
		if (self.FS.size!=0) & (FS_filter): # if using FS filtering
			FS_rewards = self._FS_decision(x)
			FS_filter = FS_rewards.max() - self.h/2
			GS_rewards[FS_rewards < FS_filter] == GS_rewards.min() - 10 # filter out not-top-arms
		arm = self._pick_arm(GS_rewards)
		return arm

	def _random_argmax(self, vector):
		max_ind = np.random.choice(np.flatnonzero(vector == vector.max()))
		return max_ind

	def _pick_arm(self, rewards):
		arm = self._random_argmax(rewards)
		return arm

	def _predict_rewards(self, data, x):
		if data.size == 0: # if the data is empty, set all rewards to 0
			rewards = np.zeros((self.k))
		else:
			z = self._feature_map(x)
			rewards = np.empty((self.k)) # create an empty reward vector
			for arm in range(self.k):
				data_i = data[data[:,2]==arm,:] # data corresponding to the chosen arm
				if data_i.size==0: # No data with the specified arm pulled
					rewards[arm] = 0 #NOTE: some arbitrarily small number or 0? 
				else: # if the arm data exists
					x_i = data_i[:,0].tolist() # convert array of array to array
					y_i = data_i[:,1]
					z_i = self._feature_map(x_i)
					rewards[arm] = self.estimate_reward(z_i, y_i, z) # estimate reward for new covariate using the arm data
		return rewards

	def estimate_reward(self, z_train, y_train, z):
		# Each agent subclass will have its own estimate_reward 
		pass
	
class Agent_LASSO(Agent):
	def __init__(self, lam, **kwargs):
		Agent.__init__(self, **kwargs)
		self.lam = lam

	def __str__(self):
		# return "AgentLASSO(lambda={})".format(self.lam)
		return self.name

	def estimate_reward(self, z_train, y_train, z):
		lasso_model = linear_model.Lasso(alpha=self.lam)
		lasso_model.fit(z_train, y_train)
		reward = lasso_model.predict([z])
		return reward

class Agent_OLS(Agent):
	def __init__(self, **kwargs):
		Agent.__init__(self, **kwargs)

	def __str__(self):
		# return "AgentLASSO(lambda={})".format(self.lam)
		return self.name

	def estimate_reward(self, z_train, y_train, z):
		ols_model = linear_model.LinearRegression()
		ols_model.fit(z_train, y_train)
		reward = ols_model.predict([z])
		return reward

# class Agent_KNN(Agent):

# class Agent_RF(Agent):

