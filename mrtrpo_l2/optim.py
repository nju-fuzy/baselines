import numpy as np
#from pylab import *
import math
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
            return l2_norm_constraint_project(H,num_reward)
        elif method == 2:
            return l2_norm_constraint_coord(H,num_reward)
        else:
            return coe/num_reward
    except:
        num_reward = G.shape[0]
        coe = np.ones((num_reward))
        return coe/num_reward

def l2_norm_constraint_coord(H,num_reward):
    eig, eigvec = np.linalg.eig(H)
    index = np.argmax(eig)
    vec = eigvec[:,index]
    sum1 = np.sum(vec > 0)
    if sum1 < 0.5:
        vec = -vec
    if np.sum(vec >= 0) == num_reward:
        return vec
    else:
        vec = np.maximum(vec,0)
        vec = vec / np.linalg.norm(vec,2)
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
            vec1 = vec1 / np.linalg.norm(vec1,2)
            vec2 = vec2 / np.linalg.norm(vec2,2)

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
        return vec

def l2_norm_constraint_project(H,num_reward):
    eig, eigvec = np.linalg.eig(H)
    index = np.argmax(eig)
    vec = eigvec[:,index]
    sum1 = np.sum(vec > 0)
    if sum1 < 0.5:
        vec = -vec
    if np.sum(vec >= 0) == num_reward:
        return vec
    else:
        vec = np.maximum(vec,0)
        vec = vec / np.linalg.norm(vec,2)
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
            direct = np.dot(H,vec)
            vec1 = vec + alpha * direct
            vec2 = vec - alpha * direct
            vec1 = np.maximum(vec1,0)
            vec2 = np.maximum(vec2,0)
            vec1 = vec1 / np.linalg.norm(vec1,2)
            vec2 = vec2 / np.linalg.norm(vec2,2)

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
        return vec

def test_get_coe():
    n = 3
    A = np.random.rand(n,n)
    H = np.dot(A,A.transpose())
    coe = l2_norm_constraint(H,n)
    eig, eigvec = np.linalg.eig(H)
    index = np.argmax(eig)
    vec = eigvec[index]
    vec = np.abs(vec)
    vec = vec / np.linalg.norm(vec,2)
    print(vec)
    print(np.dot(np.dot(H,vec),vec))
    print(np.dot(np.dot(H,coe),coe))
    return coe


if __name__ == '__main__':
    for i in range(100):
        coe = test_get_coe()
        print(coe)
        print("======================================")