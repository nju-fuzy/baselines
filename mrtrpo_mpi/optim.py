import numpy as np

def get_coefficient(G,S,method = 1):
	num_reward = G.shape[0]
	coe = np.ones((num_reward))
	return coe/num_reward
