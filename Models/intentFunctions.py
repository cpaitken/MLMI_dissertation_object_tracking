import numpy as np
from scipy.stats import norm
from scipy.linalg import solve, cholesky
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,PathPatch
from matplotlib.path import Path
from copy import deepcopy as dc
from Models.functions import iSE, SE
from scipy.stats import multivariate_normal

#####################
# Extended Goal State
#####################

def gise1_pred(t,m,v,s2,l,sigma_g=0.0):
    #get prior covariance
    C = iSE(t,t,s2,l)

     # compute Fk
    d = m.shape[0] - 1 #To account for the goal dimension in the state
    ftw = solve(C[1:,1:],C[1:,0])
    F = np.eye(d-1,k=-1)

    #To account for iSE difference
    F[0,:] = ftw
    F_aug = np.eye(d)
    F_aug[:-1,:-1] = F

    #To account for extra goal dimension
    F_goal = np.eye(d+1)
    F_goal[:-1, :-1] = F_aug

    # compute Pk
    ptw = C[0,0] - (C[0,1:] * ftw).sum()
    P = np.zeros([d,d])
    P[0,0] = ptw

    P_goal = np.zeros((d+1, d+1))
    P_goal[:-1, :-1] = P
    P_goal[-1,-1] = sigma_g

    #Compute predicted mean and covariance
    m_pred = F_goal @ m
    v_pred = F_goal @ v @ F_goal.T + P_goal  

    return m_pred, v_pred, F_goal, P_goal

def augmented_update(y, m, v, sy):
    #Mapping matrix H for the observation model that includes the latent goal
    H = np.zeros((1, m.shape[0]))
    H[0,0] = 1 #Include most recent position
    #Just experimenting with the measurement model
    #H[0, -1] = 1 #Include latent goal
    #End of experimental section
    H[0, -2] = 1 #Include the initial position (as is done in the iSE-1 model)
    H[0, -1] = 1 #Include the goal


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
    ftw = solve(C[1:,1:],C[1:,0]) ## This is the part I am unsure about -- but the SE function doesnt include the state anyway
    #ftw = solve(C[1:-1,1:-1],C[1:-1,0:-1]) ## This is changed to exclude goal state for the initial thing
    F = np.eye(d,k=-1)
    F[0,:-1] = ftw

    #Extend Transition Matrix to F_goal
    F_goal = np.eye(d+1)
    F_goal[:d, :d] = F

    #Compute Pk
    ptw = C[0,0] - (C[0,1:] * ftw).sum()
    P = np.zeros([d,d])
    P[0,0] = ptw

    ##Experiment to see if positions lead to goal
    # beta = 0.7
    # for i in range(1):
    #     F_goal[i,-1] = beta
    ##End of experimental section

    #Extend Pk to P_goal
    P_goal = np.zeros((d+1, d+1))
    P_goal[:d, :d] = P
    P_goal[d,d] = sigma_p

    #Compute predicted mean and covariance
    m_pred = F_goal @ m
    v_pred = F_goal @ v @ F_goal.T + P_goal

    return m_pred, v_pred, F_goal, P_goal

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

#####################
# Extended Goal and Convergence State for iSE model
#####################
def converging_ise_pred(t,m,v,s2,l,sigma_g=0.0, sigma_c=0.0):
    C = iSE(t,t,s2,l)

     # compute Fk
    d = m.shape[0] - 1 #To account for the goal and convergence dimension in the state
    ftw = solve(C[1:,1:],C[1:,0])
    F = np.eye(d-1,k=-1)
    F[0,:] = ftw
    F_aug = np.eye(d)
    F_aug[:-1,:-1] = F
    
    # compute Pk
    ptw = C[0,0] - (C[0,1:] * ftw).sum()
    P = np.zeros([d,d])
    P[0,0] = ptw

    #Create extended transition matrix F_goal and extended P_k for goal dimension
    F_goal = np.eye(d+1)
    F_goal[:d, :d] = F_aug

    P_goal = np.zeros((d+1, d+1))
    P_goal[:d, :d] = P
    P_goal[d,d] = sigma_g

    #Create extended transition matrix F_conv and P_conv for convergence dimension
    # F_conv = np.eye(d+2)
    # F_conv[:d+1, :d+1] = F_goal

    # P_conv = np.zeros((d+2, d+2))
    # P_conv[:d+1, :d+1] = P_goal
    # P_conv[d+1,d+1] = sigma_c

    #Compute predicted mean and covariance
    m_pred = F_goal @ m
    v_pred = F_goal @ v @ F_goal.T + P_goal  

    return m_pred, v_pred, F_goal, P_goal

def conv_ise_measure(m, lambda_val,t):
    #Just returning the expected location from the state mean
    current_lambda = lambda_val
    decay_rate = np.exp(-1*current_lambda*t)

    H = np.zeros((1, m.shape[0]))
    H[0,0] = decay_rate
    H[0,-2] = decay_rate #New index of initial position
    H[0, -1] = (1 - decay_rate) #New index of goal
    expected_loc = H @ m
    return expected_loc

def lambda_kf_update(y, m, v, sy, t):
    #Mapping matrix H for the observation model that includes the latent goal
    m_x = m[:, 0]
    print("Shape of m_x:", m_x.shape)
    m_y = m[:, 1]
    cur_lambda = m[-1]
    x_lambda = cur_lambda[0]
    y_lambda = cur_lambda[1]

    H_x = np.zeros((1, 7))
    print("Shape of H_x:", H_x.shape)
    H_y = np.zeros((1, 7))
    H_x[0, 0] = np.exp(-1*x_lambda*t)
    H_x[0, -2] = (1 - np.exp(-1 * x_lambda * t))
    H_y[0, 0] = np.exp(-1*y_lambda*t)
    H_y[0, -2] = (1 - np.exp(-1 * y_lambda * t))

    S_x = H_x @ v @ H_x.T + sy
    KG_x = (v @ H_x.T) / S_x

    S_y = H_y @ v @ H_y.T + sy
    KG_y = (v @ H_y.T) / S_y

    y_in_x = (y[0] - (H_x @ m_x).flatten())
    y_in_y = (y[1] - (H_y @ m_y).flatten())

    m_up_x = m_x + KG_x @ y_in_x
    m_up_y = m_y + KG_y @ y_in_y

    v_up_x = v - KG_x @ H_x @ v
    v_up_y = v - KG_y @ H_y @ v

    #Combine m and v updates
    print("Shape of m_up_x:", m_up_x.shape)
    print("Shape of m_up_y:", m_up_y.shape)
    print("Shape of v_up_x:", v_up_x.shape)
    print("Shape of v_up_y:", v_up_y.shape)
    m_up = np.column_stack((m_up_x, m_up_y))
    v_up = np.vstack((v_up_x, v_up_y))

    return m_up, v_up, KG_x, KG_y

def fixed_lambda_kf_update(y, m, v, sy, t, lambda_val):
    H = np.zeros((1, m.shape[0]))
    decay_rate = np.exp(-1*lambda_val*t)
    H[0,0] = decay_rate
    H[0,-2] = decay_rate
    H[0, -1] = (1 - decay_rate)

    Hv = H @ v
    predicted_loc_updateStep = H @ m.copy()
    HvHs = Hv @ H.T + sy
    KG = (v @ H.T)/HvHs

    y_in = (y - (H @ m).flatten()).reshape(1,-1)

    m_up = m + KG @ y_in
    v_up = v - KG @ H @ v

    return m_up, v_up, KG, decay_rate, predicted_loc_updateStep

def fixed_lambda_kf_update_for_PF(y, m, v, sy, t, lambda_val):
    H = np.zeros((1, m.shape[0]))
    decay_rate = np.exp(-1*lambda_val*t)
    H[0,0] = decay_rate
    H[0,-2] = decay_rate
    H[0, -1] = (1 - decay_rate)

    Hv = H @ v
    predicted_loc_updateStep = H @ m.copy()
    HvHs = Hv @ H.T + sy
    KG = (v @ H.T)/HvHs

    #Get observation likelihood
    R = np.eye(2) * sy**2
    S_k = H @ v @ H.T + R 
    #print("Shape of predicted_loc_updateStep:", predicted_loc_updateStep.shape)
    #print("Predicted location update step is:", predicted_loc_updateStep)
    cur_lambda_state_dist = multivariate_normal(mean=predicted_loc_updateStep[0], cov=S_k)
    obs_likelihood = cur_lambda_state_dist.logpdf(y)
    #End of observation likelihood

    y_in = (y - (H @ m).flatten()).reshape(1,-1)

    m_up = m + KG @ y_in
    v_up = v - KG @ H @ v

    return m_up, v_up, KG, decay_rate, obs_likelihood

def systematic_resample_particles(weights):
    num_particles = len(weights)
    cumulative_weights = np.cumsum(weights)
    random_start = np.random.uniform(0, 1/num_particles)
    U_i = random_start + (np.arange(num_particles) / num_particles)

    indices = np.searchsorted(cumulative_weights, U_i)

    return indices


#####################
## Generating Goal Driven Track ##
#####################
def gen_goal_driven_track(Tmax,d,s2,l, goal, sigma_g=0.0, dim=2, dt=1, first_is_last=False):
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

## Generating Goal Driven Track ##
def gen_iSE_driven_track(Tmax,d,s2,l, goal, sigma_p=0.0, dim=2, dt=1, first_is_last=False):
    #Created similarly to gen_SE_track but with adding the goal state to match the measurement model
    x = np.zeros([Tmax,dim])

    #Prior variance
    t = dt * np.arange(d,0,-1)
    C = iSE(t,t,s2,l)

    #Sample over initial window - SAME
    sqrt_C = cholesky(C)
    x[-d:,:] = sqrt_C.T @ norm.rvs(size=[d,dim])

    #Common quantities - DIFFERENT
    ftw = solve(C[1:,1:],C[1:,0])
    ptw = C[0,0] - (C[0,1:] * ftw).sum()

    for k in range(d, Tmax):
        # prepare next sample
        x_star = x[d-k-1,:]
        mean = x_star + ftw.T @ (x[-k:d-k-1,:] - x_star)
        
        # sample next step
        x[-k-1,:] = norm.rvs(mean,ptw**0.5)

    if not first_is_last:
        x = x[::-1,:]
    
    x = x+goal
    return x

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
            start_val = start[dim] - goal[dim]
        else:
            start_val = np.random.normal(0, np.sqrt(s2)) - goal[dim]
        y_obs = np.array([start_val, 0])
        mu = np.zeros(Tmax)
        mu_obs = mu[obs_idx]
        mu_rest = mu[rest_idx]
        # Conditional mean and covariance
        cond_mean = mu_rest + C_cross @ np.linalg.inv(C_obs) @ (y_obs - mu_obs)
        cond_cov = C_rest - C_cross @ np.linalg.inv(C_obs) @ C_cross.T
        # Sample the interior points
        traj[rest_idx, dim] = np.random.multivariate_normal(cond_mean, cond_cov)
        traj[0, dim] = start_val
        traj[-1, dim] = 0

        #Add goal back
        traj[:, dim] += goal[dim]
    return traj

def gen_iSE_track_goal_converging(Tmax, d, s2, l, dim=2, dt=1, first_is_last=False, lambda_val=0.05, goal=np.array([50,50]), varying_lambda=False, lambda_values=None):
    # initiate track object
    x = np.zeros([Tmax,dim])
    
    # get prior variance
    t = dt * np.arange(d,0,-1)
    C = iSE(t,t,s2,l)
    
    # sample over initial window
    sqrt_C = cholesky(C)
    initial_window = sqrt_C.T @ norm.rvs(size=[d,dim])
    x[-d:,:] = initial_window
    
    # common quantities
    g = solve(C[1:,1:],C[1:,0])
    q = C[0,0] - C[0,1:] @ g

    iSE_contributions = []
    iSE_contributions_final = []
    decay_rate = []
    goal_contributions = []

    x_final = np.zeros([Tmax,dim])
    x_final[-d:,:] = initial_window

    for k in range(d,Tmax):
        
        # prepare next sample
        x_star = x[d-k-1,:]
        mean = x_star + g.T @ (x[-k:d-k-1,:] - x_star)
        
        # sample next step
        cur_contribution = norm.rvs(mean,q**0.5)
        x[-k-1,:] = cur_contribution
        iSE_contributions.append(cur_contribution)

    if varying_lambda:
        for k in range(d, Tmax):
            cur_lambda = lambda_values[k]
            print("Current lambda is:", cur_lambda)
            print("Current time is:", k)
            decay_rate.append(np.exp(-1*cur_lambda*(Tmax-k)))
            iSE_contribution = np.exp(-1*cur_lambda*(Tmax-k))*x[-k-1,:]
            print("Amount of iSE contribution is:", np.exp(-1*cur_lambda*(Tmax-k)))
            print("ise Contribution at time", k, "is:", iSE_contribution)
            iSE_contributions_final.append(iSE_contribution)
            goal_contribution = (1-np.exp(-1*cur_lambda*(Tmax-k)))*goal
            print("Amount of goal contribution is:", (1-np.exp(-1*cur_lambda*(Tmax-k))))
            print("Goal contribution at time", k, "is:", goal_contribution)
            x_final[-k-1,:] = iSE_contribution + goal_contribution
            print("Final contribution at time", k, "is:", x_final[-k-1,:])
            goal_contributions.append(goal_contribution)
    else:
        for k in range(d,Tmax):
            decay_rate.append(np.exp(-1*lambda_val*(Tmax-k)))
            iSE_contribution = np.exp(-1*lambda_val*(Tmax-k))*x[-k-1,:]
            iSE_contributions_final.append(iSE_contribution)
            goal_contribution = (1-np.exp(-1*lambda_val*(Tmax-k)))*goal
            x_final[-k-1,:] = iSE_contribution + goal_contribution
            goal_contributions.append(goal_contribution)

    ##Remove last d points from x_final as they are the initial window
    x_final = x_final[:-d,:]

    return x_final, iSE_contributions, goal_contributions, decay_rate, iSE_contributions_final
    
    if not first_is_last:
        x = x[::-1,:]
    
    return x