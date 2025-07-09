import numpy as np
from scipy.stats import norm
from scipy.linalg import solve, cholesky
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,PathPatch
from matplotlib.path import Path
from copy import deepcopy as dc
from Models.functions import iSE, SE
from Models.intentFunctions import converging_ise_pred, conv_ise_measure

##This file is for clarity on steps of particle filtering sequential inference##

#Method to sample particles
def sample_particles(m,v,N):
    new_particles = np.zeros((N, m.shape[0], m.shape[1]))
    for dim in range(m.shape[1]):
        new_particles[:,:,dim] = np.random.multivariate_normal(m[:,dim], v, N)
    return new_particles

#Method to propagate particles given the prediction step
def propagate_particles(particles, part_cov, t, s2, l, sigma_g, sigma_c):
    N = particles.shape[0]
    state_dim = particles.shape[1]
    coordinate_dimension = particles.shape[2]

    #Arrays for new
    new_particles = np.zeros((N, state_dim, coordinate_dimension))
    new_part_cov = np.zeros_like(part_cov)
    all_F_conv = []
    all_P_conv = []

    #Propagate each particle
    for i in range(N):
        m_pred, v_pred, F_conv, P_conv = converging_ise_pred(t, particles[i,:], part_cov[i,:], s2, l, sigma_g, sigma_c)
        #print("V pred is:", v_pred)
        #Adding process noise part
        for dim in range(coordinate_dimension):
             new_particles[i,:,dim] = np.random.multivariate_normal(m_pred[:,dim], v_pred)
             #Truncate lambda to be positive
             new_particles[i,:,dim][-1] = max(new_particles[i,:,dim][-1], 0.0)
        #new_particles[i,:] = m_pred #This is to try with no noise
        #print("New particles[i,:] is:", new_particles[i,:])
        new_part_cov[i,:] = v_pred
        all_F_conv.append(F_conv)
        all_P_conv.append(P_conv)

    
    return new_particles, new_part_cov, all_F_conv, all_P_conv

def computeNorm_weights(particles, observation, sy, t):
    N = particles.shape[0]
    weights = np.zeros(N)
    meas_and_obv = []

    for i in range(N):
        expected_measurement = conv_ise_measure(particles[i,:], sy, t)
        #Determine likelihood
        diff = observation - expected_measurement
        #Assuming measurement noise independent in each dimension (2)
        R = (sy) * np.eye(2)
        #Computing the likelihood using PDF multivar Gaussian
        exp_likelihood = -0.5 * diff.T @ np.linalg.inv(R) @ diff
        norm_const = 1/ np.sqrt((2*np.pi)**2 * np.linalg.det(R))
        weights[i] = norm_const * np.exp(exp_likelihood)
        ##DEBUGGING##
        pair = np.vstack([observation, expected_measurement])
        meas_and_obv.append(pair)
    
    #Normalize weights
    weights += 1e-300
    weights = weights / np.sum(weights)
    ##DEBUGGING##
    meas_and_obv = np.array(meas_and_obv)
    return weights, meas_and_obv

def computeNorm_weights_lambdaPF(particles, observation, sy):
    N = particles.shape[0]
    weights = np.zeros(N)
    meas_and_obv = []

    for i in range(N):
        expected_measurement = particles[i, -1]
        #Determine likelihood
        diff = observation - expected_measurement
        #Assuming measurement noise independent in each dimension (2)
        R = (sy) * np.eye(2)
        #Computing the likelihood using PDF multivar Gaussian
        exp_likelihood = -0.5 * diff.T @ np.linalg.inv(R) @ diff
        norm_const = 1/ np.sqrt((2*np.pi)**2 * np.linalg.det(R))
        weights[i] = norm_const * np.exp(exp_likelihood)
        ##DEBUGGING##
        pair = np.vstack([observation, expected_measurement])
        meas_and_obv.append(pair)
    
    #Normalize weights
    weights += 1e-300
    weights = weights / np.sum(weights)
    ##DEBUGGING##
    meas_and_obv = np.array(meas_and_obv)
    return weights, meas_and_obv

def resample_particles(particles, weights):
    N = particles.shape[0]
    indices = np.random.choice(N, size=N, p=weights)
    return particles[indices,:]







        
