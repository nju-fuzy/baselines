import numpy as np
#from pylab import *
import math
from cvxpy import *
from qcqp import *
import pdb
def get_coefficient(G,S,method = 4):
    num_reward = G.shape[0]
    coe = np.ones((num_reward))
    H = np.dot(S, G.T)
    SS = np.dot(S, S.T)
    if method == 1:
        return l1_norm_constraint(H,num_reward)
    elif method == 2:
        return l2_norm_constraint(H,num_reward)
    elif method == 3:
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
def l1_norm_constraint(H,num_reward):
    vals, vecs = np.linalg.eigh(H)
    x = vecs[:,-1]
    sum1 = np.sum(x > 0)
    if sum1 < 0.5:
        x = -x
    x = np.maximum(x,0)
    x = x / np.sum(x)
    for i in range(num_reward):
        #print(np.sum(x * np.dot(H,x)))
        index1 = np.argmax(x)
        x[index1] = -1
        index2 = np.argmax(x)
        x[index2] = 0
        lower = 0
        upper = - np.sum(x)
        a = H[index1,index1] + H[index2,index2] - H[index1,index2] - H[index2,index1]
        b = upper * (-2 * H[index2,index2] + H[index1,index2] + H[index2,index1])
        if a < 1e-6:
            op = 0 if b < 0 else upper
        else:
            middle = b/(-2 * a)
            if middle < upper/2:
        	    op = upper
            else:
        	    op = 0
        x[index1] = op
        x[index2] = upper - op
    return x

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

def test_get_coe():
    n = 3
    A = np.random.rand(n,n)
    H = np.dot(A,A.transpose())
    coe = l1_norm_constraint(H,n)
    return coe


if __name__ == '__main__':
    for i in range(100):
        coe = test_get_coe()
        print(coe)
