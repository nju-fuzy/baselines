import numpy as np
#from pylab import *
import math
import scipy.linalg
#from cvxpy import *
#from qcqp import *
import pdb
def get_coefficient(G,S,method = 2):
    try:
        num_reward = G.shape[0]
        coe = np.ones((num_reward))
        H = np.dot(S, G.T)
        SS = np.dot(S, S.T)
        if method == 1:
            return max_update_project(SS,H,num_reward)
        elif method == 2:
            return max_update_corrd(SS,H,num_reward)
        else:
            return coe/num_reward
    except:
        num_reward = G.shape[0]
        coe = np.ones((num_reward))
        return coe/num_reward

def max_update_corrd(SS,H,num_reward):
    if np.abs(np.linalg.det(SS)) < 1e-6:
        SS = SS + 0.001 * np.eye(num_reward)
    A = scipy.linalg.sqrtm(np.linalg.inv(SS))
    modify_H = np.dot(np.dot(A.T,H),A)
    eig, eigvec = np.linalg.eig(modify_H)
    index = np.argmax(eig)
    vec = eigvec[:,index]
    vec = np.dot(A,vec)
    sum1 = np.sum(vec > 0)
    if sum1 < 0.5:
        vec = -vec
    if np.sum(vec >= 0) == num_reward:
        return vec / np.linalg.norm(vec,2)
    else:
        vec = np.maximum(vec,0)
        vec = vec / m_norm(SS,vec)
        initialobj = np.dot(np.dot(H,vec),vec)
        alpha = 0.01
        direct = - np.ones((num_reward,num_reward)) / (num_reward - 1)
        for i in range(num_reward):
            direct[i,i] = 1
        record = np.ones((num_reward))
        for i in range(1000):
            #print(vec,np.linalg.norm(vec,2))
            #print('improve',np.dot(np.dot(H,vec),vec),initialobj)
            index = i % num_reward
            objold = np.dot(np.dot(H,vec),vec)
            vec1 = vec + alpha * direct[index,:]
            vec2 = vec - alpha * direct[index,:]   
            vec1 = np.maximum(vec1,0)
            vec2 = np.maximum(vec2,0)
            vec1 = vec1 / m_norm(SS,vec1)
            vec2 = vec2 / m_norm(SS,vec2)

            obj1 = np.dot(np.dot(H,vec1),vec1)
            obj2 = np.dot(np.dot(H,vec2),vec2)
            if obj1 > objold and obj1 >= obj2:
                vec = vec1
            elif obj2 > objold and obj2 >= obj1:
                vec = vec2
            objnew = np.dot(np.dot(H,vec),vec)
            record[index] = objnew-objold
            if np.sum(record) <= 1e-8:
                alpha = alpha / 2
                record = np.ones((num_reward))
            if alpha < 0.0001:
                break
        vec = vec / np.linalg.norm(vec,2)
        return vec
def m_norm(M,x):
    return np.sqrt(np.dot(np.dot(M,x),x))
def max_update_project(SS,H,num_reward):
    if np.abs(np.linalg.det(SS)) < 1e-6:
        SS = SS + 0.001 * np.eye(num_reward)
    A = scipy.linalg.sqrtm(np.linalg.inv(SS))
    modify_H = np.dot(np.dot(A.T,H),A)
    eig, eigvec = np.linalg.eig(modify_H)
    index = np.argmax(eig)
    vec = eigvec[:,index]
    vec = np.dot(A,vec)
    sum1 = np.sum(vec > 0)
    if sum1 < 0.5:
        vec = -vec
    if np.sum(vec >= 0) == num_reward:
        return vec / np.linalg.norm(vec,2)
    else:
        vec = np.maximum(vec,0)
        vec = vec / m_norm(SS,vec)
        initialobj = np.dot(np.dot(H,vec),vec)
        alpha = 0.01
        record = np.ones((num_reward))
        for i in range(1000):
            #print(vec,np.linalg.norm(vec,2))
            #print('improve',np.dot(np.dot(H,vec),vec),initialobj)
            index = i % num_reward
            objold = np.dot(np.dot(H,vec),vec)
            direct = np.dot(H,vec)
            vec1 = vec + alpha * direct
            vec2 = vec - alpha * direct 
            vec1 = np.maximum(vec1,0)
            vec2 = np.maximum(vec2,0)
            vec1 = vec1 / m_norm(SS,vec1)
            vec2 = vec2 / m_norm(SS,vec2)

            obj1 = np.dot(np.dot(H,vec1),vec1)
            obj2 = np.dot(np.dot(H,vec2),vec2)
            if obj1 > objold and obj1 >= obj2:
                vec = vec1
            elif obj2 > objold and obj2 >= obj1:
                vec = vec2
            objnew = np.dot(np.dot(H,vec),vec)
            record[index] = objnew-objold
            if np.sum(record) <= 1e-8:
                alpha = alpha / 2
                record = np.ones((num_reward))
            if alpha < 0.0001:
                break
        vec = vec / np.linalg.norm(vec,2)
        return vec

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
