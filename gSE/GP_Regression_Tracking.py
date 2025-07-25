import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from dataFunctions import make_groundtruth
import matplotlib.pyplot as plt
import Models.intentFunctions as iF
import Models.functions as f
from dataFunctions import get_model_rmse

#This has s2=100 and ls=3
groundtruth = make_groundtruth("Data/goalMeanGP_track.txt", UZH=False)

Tmax = groundtruth.shape[0]

#Create noisy observation data
sy = 1
noisy_data = [groundtruth[k] + np.random.normal(0, sy, 2) for k in range(Tmax)]


#Inputs for the GP
t1 = np.arange(Tmax).reshape(-1,1) #Time steps
Y = np.array([arr.flatten() for arr in noisy_data])
s2=100
ls=3
d=5



#Final Objects
gpr_list = []
predicted_x = []
x_stds = []

#GP Regression for 2D
for dim in range(2):
    kernel = RBF(length_scale=3)
    #kernel = SE(t,t,10,3)
    gpr = GaussianProcessRegressor(kernel=kernel).fit(t1, Y[:,dim])

    x_pred, x_std = gpr.predict(t1, return_std=True)
    gpr_list.append(gpr)
    predicted_x.append(x_pred)
    x_stds.append(x_std)

##BEgin debugging just the SE model
##Initialization
dt=1
t = dt * np.arange(d,0,-1)
mk_normal = [groundtruth[0,:]*np.ones([d,2])]
vk_normal = [f.SE(t,t,s2/10,ls)]
G_prior = np.array([10,10])
G_var = 1

#Initialization for goal model
mk_goal = [groundtruth[0,:]*np.ones([d,2])]
mk_goal[0] = np.vstack((mk_goal[0], G_prior))
#Making covariance matrix for goal model
vk_goal = [np.eye(d+1)]
vk_goal[0][:-1,:-1] = f.SE(t,t,s2/10,ls)
vk_goal[0][-1,-1] = G_var

X_normal = np.zeros([Tmax,2])
S_normal = np.zeros([Tmax]) #Uncertainty at each time step

X_goal = np.zeros([Tmax,2])
S_goal = np.zeros([Tmax]) #Uncertainty at each time step for goal model
G_goal = np.zeros([Tmax,2])

for k in range(Tmax):
    #Prediction step
    m_predN, v_predN, F_augN = f.se_pred(t,mk_normal[-1],vk_normal[-1],s2,ls)

    #Skip association step for now
    y = noisy_data[k]
    datum = y

    #Update step
    m_upN, v_upN, KGN, y_inN = f.update(datum, m_predN, v_predN, sy)

    mk_normal.append(m_upN)
    vk_normal.append(v_upN)

    #X_normal[k,:] = m_upN[0,:] + m_upN[-1,:]
    X_normal[k,:] = m_upN[0,:]
    #S_normal[k] = v_upN[0,0] + v_upN[-1,-1] 
    S_normal[k] = v_upN[0,0]

    # normal_F_aug.append(F_augN.copy())
    # normal_predicted_means.append(m_predN.copy())
    # normal_updated_means.append(m_upN.copy())
for k in range(Tmax):
    m_pred, v_pred, F_goal = iF.g_se_pred(t,mk_goal[-1],vk_goal[-1],s2,ls)

    y = noisy_data[k]
    datum = y

    m_up, v_up, KGN, y_in = iF.g_update(datum, m_pred, v_pred, sy)

    mk_goal.append(m_up)
    vk_goal.append(v_up)
    #X_goal[k,:] = m_up[0,:] + m_up[-2,:] #Measurement model dependent only on initial location
    #X_goal[k,:] = m_up[0,:] + m_up[-2,:] + m_up[-1,:]  #Measurement model dependent on goal
    X_goal[k,:] = m_up[0,:] + m_up[-1,:]
    #S_goal[k] = v_up[0,0] + v_up[-2,-2] 
    S_goal[k] = v_up[0,0] + v_up[-1,-1]

    G_goal[k,:] = m_up[-1,:]

    # goal_F_aug.append(F_goal.copy())
    # goal_predicted_means.append(m_pred.copy())
    # goal_updated_means.append(m_up.copy())

#Plotting 1D
# for dim in range(2):
#     plt.figure()
#     plt.plot(t, Y[:,dim], label="Noisy Data")
#     plt.plot(t, groundtruth[:,dim], label="Groundtruth")
#     plt.plot(t, predicted_x[dim], 'b-', label='GP Mean')
#     plt.fill_between(
#         t.ravel(),
#         predicted_x[dim] - 2 * x_stds[dim],
#         predicted_x[dim] + 2 * x_stds[dim],
#         color='blue', alpha=0.2, label='GP ±2 std'
#     )
#     plt.legend()
#     plt.xlabel('Time')
#     plt.ylabel(f'Dimension {dim}')
#     plt.show()

plt.figure(figsize=(8, 6))
plt.plot(Y[:, 0], Y[:, 1], 'kx', label='Noisy Data', alpha=0.5)
plt.plot(groundtruth[:, 0], groundtruth[:, 1], 'g-', label='Groundtruth', linewidth=2)
plt.plot(predicted_x[0], predicted_x[1], 'b-', label='GP Mean', linewidth=2)
plt.plot(X_normal[:, 0], X_normal[:, 1], 'r-', label='Kalman Filter', linewidth=2)
plt.plot(X_goal[:, 0], X_goal[:, 1], 'p-', label='Kalman Filter Goal Aware', linewidth=2)
# for k in range(len(S_normal)):
#     ellipse = ellipse(
#         (X_normal[k, 0], X_normal[k, 1]),  # Center
#         width=2*S_normal[k], height=2*S_normal[k],  # 2*std for 95% coverage
#         edgecolor='none',
#         facecolor='red',
#         alpha=0.1
#     )
#     plt.gca().add_patch(ellipse)
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Trajectory')
plt.axis('equal')
plt.show()


print(f"RMSE of SE model:  {get_model_rmse(X_normal, groundtruth):.4f}")
print(f"RMSE of goal model:  {get_model_rmse(X_goal, groundtruth):.4f}")

predicted_x_arr = np.column_stack(predicted_x)  # shape (T, 2)

# RMSE per dimension
rmse_per_dim = np.sqrt(np.mean((predicted_x_arr - groundtruth)**2, axis=0))

# Overall RMSE (all dimensions together)
rmse_overall = np.sqrt(np.mean((predicted_x_arr - groundtruth)**2))
print("Overall RMSE:", rmse_overall)












