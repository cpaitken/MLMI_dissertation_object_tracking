import numpy as np
from scipy.stats import norm
from scipy.linalg import solve, cholesky
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,PathPatch
from matplotlib.path import Path
from copy import deepcopy as dc
from Models.functions import iSE, SE

#####################
# Extended Goal State
#####################

def gise1_pred(t,m,v,s2,l,sigma_p=0.0):
    #get prior covariance
    C = iSE(t,t,s2,l)

     # compute Fk
    d = m.shape[0] - 1 #To account for the goal dimension in the state
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

    ##Experiment to see if positions lead to goal
    beta = 0.1
    for i in range(1):
        F_goal[i,-1] = beta
    ##End of experimental section

    P_goal = np.zeros((d+1, d+1))
    P_goal[:d, :d] = P
    P_goal[d,d] = sigma_p

    #Compute predicted mean and covariance
    m_pred = F_goal @ m
    v_pred = F_goal @ v @ F_goal.T + P_goal  

    return m_pred, v_pred, F_goal

def augmented_update(y, m, v, sy):
    #Mapping matrix H for the observation model that includes the latent goal
    H = np.zeros((1, m.shape[0]))
    H[0,0] = 1 #Include most recent position
    #Just experimenting with the measurement model
    #H[0, -1] = 1 #Include latent goal
    #End of experimental section
    H[0, -2] = 1 #Include the initial position (as is done in the iSE-1 model)


    #Calculate Kalman Gain
    Hv = H @ v
    HvHs = Hv @ H.T + sy
    KG = (v @ H.T)/HvHs

    #Experimental section to zero out KG for goal
    #KG[-1, :] = 0
    #End of experimental section

    #Calculate innovation
    y_in = (y - (H @ m).flatten()).reshape(1,-1)

    #Update state
    m_up = m + KG @ y_in
    v_up = v - KG @ H @ v

    return m_up, v_up, KG, y_in

#####################
# Extended Goal State for normal GP (SE) model
#####################

#Prediction function -- based on se_pred function
def g_se_pred(t,m,v,s2,l,sigma_p=0.0):

    C = SE(t,t,s2,l)

    #Compute Fk
    d = m.shape[0] - 1
    ftw = solve(C[1:,1:],C[1:,0])
    F = np.eye(d,k=-1)
    F[0,:-1] = ftw

    #Extend Transition Matrix to F_goal
    F_goal = np.eye(d+1)
    F_goal[:d, :d] = F

    #Compute Pk
    ptw = C[0,0] - (C[0,1:] * ftw).sum()
    P = np.zeros([d,d])
    P[0,0] = ptw

    #Extend Pk to P_goal
    P_goal = np.zeros((d+1, d+1))
    P_goal[:d, :d] = P
    P_goal[d,d] = sigma_p

    #Compute predicted mean and covariance
    m_pred = F_goal @ m
    v_pred = F_goal @ v @ F_goal.T + P_goal

    return m_pred, v_pred, F_goal

def g_update(y, m, v, sy):
    #Attempting to make this in the same style as original update function
    # y_in = (y-m[0,:] - m[-1,:]).reshape([1,-1]) #The m[-1,:] is the goal

    # Kgain = (v[:,0] + v[-1,-1]).reshape([-1,1]) / (v[0,0]+v[-1,-1]+ (2*v[0,-1])+sy)

    # m_upd = m + Kgain @ y_in
    # v_upd = v- Kgain @ (v[0,:] +v[-1,:]).reshape([1,-1]) #Second part v[-1,:] is for the goal
    #End of try 1

    #Try 2
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
    v_up = v - KG @ H @ v

    return m_up, v_up, KG, y_in

## Generating Goal Driven Track ##
def gen_goal_driven_track(Tmax,d,s2,l, goal, sigma_p=0.0, dim=2, dt=1, first_is_last=False):
    #Created similarly to gen_SE_track but with adding the goal state to match the measurement model
    x = np.zeros([Tmax,dim])

    #Prior variance
    t = dt * np.arange(d,0,-1)
    C = SE(t,t,s2,l)

    #Sample over initial window - SAME
    sqrt_C = cholesky(C)
    x[-d:,:] = sqrt_C.T @ norm.rvs(size=[d,dim])

    #Common quantities - DIFFERENT
    ftw = solve(C[1:,1:],C[1:,0])
    ptw = C[0,0] - (C[0,1:] * ftw).sum()

    for k in range(d, Tmax):
        mean = ftw.T @ x[-k:d-k-1,:]

        x[-k-1,:] = norm.rvs(mean,ptw**0.5)

    if not first_is_last:
        x = x[::-1,:]
    
    x = x+goal
    return x

## NOT WRITTEN BY ME, JUST TO CHECK ##
def gen_gp_traj_with_goal_mean(Tmax, d,s2, l, goal, dt=1):
    t = dt * np.arange(Tmax)
    traj = np.zeros((Tmax, 2))
    for dim in range(2):
        # Build SE covariance
        C = SE(t,t,s2,l)
        # Sample from zero-mean GP
        sqrt_C = cholesky(C)
        gp_sample = sqrt_C @ norm.rvs(size=Tmax)
        # Add the goal mean
        traj[:, dim] = gp_sample + goal[dim]
    return traj

def gen_gp_bridge(Tmax, s2, l, goal, dt=1, start=None):
    t = dt * np.arange(Tmax)
    traj = np.zeros((Tmax, 2))
    for dim in range(2):
        # Build SE covariance
        C = s2 * np.exp(-0.5 * ((t[:, None] - t[None, :]) / l) ** 2)
        obs_idx = [0, Tmax-1]
        rest_idx = np.arange(1, Tmax-1)
        C_obs = C[np.ix_(obs_idx, obs_idx)]
        C_rest = C[np.ix_(rest_idx, rest_idx)]
        C_cross = C[np.ix_(rest_idx, obs_idx)]
        # Choose start value
        if start is not None:
            start_val = start[dim]
        else:
            start_val = np.random.normal(0, np.sqrt(s2))
        y_obs = np.array([start_val, goal[dim]])
        mu = np.zeros(Tmax)
        mu_obs = mu[obs_idx]
        mu_rest = mu[rest_idx]
        # Conditional mean and covariance
        cond_mean = mu_rest + C_cross @ np.linalg.inv(C_obs) @ (y_obs - mu_obs)
        cond_cov = C_rest - C_cross @ np.linalg.inv(C_obs) @ C_cross.T
        # Sample the interior points
        traj[rest_idx, dim] = np.random.multivariate_normal(cond_mean, cond_cov)
        traj[0, dim] = start_val
        traj[-1, dim] = goal[dim]
    return traj