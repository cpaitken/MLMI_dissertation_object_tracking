import numpy as np
from scipy.stats import norm
from scipy.linalg import solve, cholesky
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,PathPatch
from matplotlib.path import Path
from copy import deepcopy as dc
from Models.functions import iSE

#####################
# Extended Goal State
#####################

def gise1_pred(t,m,v,s2,l,sigma_p=0.0):
    #get prior covariance
    C = iSE(t,t,s2,l)

     # compute Fk
    d = m.shape[0] - 1 #To account for the goal dimension in the state
    #print("D is currently:", d)
    ftw = solve(C[1:,1:],C[1:,0])
    F = np.eye(d-1,k=-1)
    F[0,:] = ftw
    F_aug = np.eye(d)
    F_aug[:-1,:-1] = F
    
    # compute Pk
    ptw = C[0,0] - (C[0,1:] * ftw).sum()
    P = np.zeros([d,d])
    P[0,0] = ptw

    #Create extended transition matrix F_goal and extended P_k
    F_goal = np.eye(d+1)
    F_goal[:d, :d] = F_aug

    P_goal = np.zeros((d+1, d+1))
    P_goal[:d, :d] = P
    P_goal[d,d] = sigma_p

    #Compute predicted mean and covariance
    m_pred = F_goal @ m
    v_pred = F_goal @ v @ F_goal.T + P_goal  

    return m_pred, v_pred

def augmented_update(y, m, v, sy):
    #Mapping matrix H for the observation model that includes the latent goal
    H = np.zeros((1, m.shape[0]))
    H[0,0] = 1 #Include most recent position
    H[0, -1] = 1 #Include latent goal

    #Calculate Kalman Gain
    Hv = H @ v
    HvHs = Hv @ H.T + sy
    KG = (v @ H.T)/HvHs

    #Calculate innovation
    y_in = (y - (H @ m).flatten()).reshape(1,-1)

    #Update state
    m_up = m + KG @ y_in
    v_up = v - KG @ v[[0,-1], :].sum(0).reshape([1,-1])

    return m_up, v_up