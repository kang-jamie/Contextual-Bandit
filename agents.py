"""
author: Jamie Kang
jamiekang@stanford.edu

"""

import numpy as np
from sklearn import linear_model

class Agent(object):
	def __init__(self, n, h, k, greedy_only=False, **kwargs):
		self.h = h
		self.k = k
		self.n = n
		self.FS = np.array([])
		self.GS = np.array([])
		self.history = np.array([])
		if greedy_only:
			self.FS_schedule = np.zeros((self.n)) - 1
		else:
			self._FS_schedule()	

	def __str__(self):
		pass

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
		#TODO
		self.FS_schedule = np.zeros((self.n)) - 1 # initialize to all-greedy schedule (i.e. -1)
		FS_num = int(np.log(self.n) * 50) #TODO: change this 30 to sth else
		FS_t = np.random.choice(a=range(int(self.n/2)), size=FS_num, replace=False) #TODO: change /2 
		self.FS_schedule[FS_t] = np.random.randint(low=0, high=self.k, size=FS_num, dtype='I')
		print("FS_num: ", FS_num) #TODO: remove this

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
		return "AgentLASSO(lambda={})".format(self.lam)

	def estimate_reward(self, z_train, y_train, z):
		lasso_model = linear_model.Lasso(alpha=self.lam)
		lasso_model.fit(z_train, y_train)
		reward = lasso_model.predict([z])
		return reward



# class Agent_KNN(Agent):

# class Agent_RF(Agent):