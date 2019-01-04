import numpy as np

def get_coefficient(G,S,method = 1):
	num_reward = G.shape[0]
	coe = np.ones((num_reward))
	H = np.dot(S, G.T)
	SS = np.dot(S, S.T)
	GG = np.dot(G, G.T)
	return coe/num_reward
