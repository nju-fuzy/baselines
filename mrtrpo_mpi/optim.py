import numpy as np
#from pylab import *
import math
from cvxpy import *
from qcqp import *
import pdb
def get_coefficient(G,S,method = 1):
    num_reward = G.shape[0]
    coe = np.ones((num_reward))
    H = np.dot(S, G.T)
    SS = np.dot(S, S.T)
    if method == 1:
        return l2_norm_constraint(H,num_reward)
    elif method == 2:
        return max_update(SS,H,num_reward)
    else:
        return coe/num_reward

def l2_norm_constraint(H,num_reward):
    #print(H)
    x = Variable(num_reward)
    #print(H)
    #vals, vecs = np.linalg.eigh(H)
    #max_eigenvalue = np.max(vals)
    #A = (max_eigenvalue + 1)* np.eye(num_reward)
    objective = Maximize(quad_form(x, H))
    constraints = [ x >= 0, x.T * x == 1]

    prob = Problem(objective, constraints)
    qcqp = QCQP(prob)

    # SDR SPECTRAL
    qcqp.suggest(SPECTRAL)
    #print("SDR lower bound: %.3f" % qcqp.sdr_bound)

    # Attempt to improve the starting point given by the suggest method
    f_cd, v_cd = qcqp.improve(COORD_DESCENT)
    #primal_result = prob.solve()
    return np.array(x.value).reshape((num_reward))

def max_update(SS,H,num_reward):
    x = Variable(num_reward)
    objective = Minimize(quad_form(x,H))
    constraints = [ x >= 0,
              x.T * SS * x == 1]

    prob = Problem(objective, constraints)
    qcqp = QCQP(prob)

    # SDR SPECTRAL
    qcqp.suggest(SDR)
    #print("SDR lower bound: %.3f" % qcqp.sdr_bound)

    # Attempt to improve the starting point given by the suggest method
    f_cd, v_cd = qcqp.improve(COORD_DESCENT)

    return np.array(x.value).reshape((num_reward))