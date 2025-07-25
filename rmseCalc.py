from dataFunctions import print_rmse_summary, analyze_convergence_by_parameters, save_convergence_matrices_to_txt
import numpy as np

# print("S2:100_LS:5")
# print_rmse_summary("Debugging/gSE_Params/s2Ls/S2:100_LS:5_G_var:10000_G_prior:Truth")
# print("S2:1000_LS:5")
# print_rmse_summary("Debugging/gSE_Params/s2Ls/S2:1000_LS:5_G_var:10000_G_prior:Truth")
# print("S2:10000_LS:5")
# print_rmse_summary("Debugging/gSE_Params/s2Ls/S2:10000_LS:5_G_var:10000_G_prior:Truth")

# print("S2:100_LS:7")
# print_rmse_summary("Debugging/gSE_Params/s2Ls/S2:100_LS:7_G_var:10000_G_prior:Truth")
# print("S2:1000_LS:7")
# print_rmse_summary("Debugging/gSE_Params/s2Ls/S2:1000_LS:7_G_var:10000_G_prior:Truth")
# print("S2:10000_LS:7")
# print_rmse_summary("Debugging/gSE_Params/s2Ls/S2:10000_LS:7_G_var:10000_G_prior:Truth")

# print("S2:100_LS:10")
# print_rmse_summary("Debugging/gSE_Params/s2Ls/S2:100_LS:10_G_var:10000_G_prior:Truth")
# print("S2:1000_LS:10")
# print_rmse_summary("Debugging/gSE_Params/s2Ls/S2:1000_LS:10_G_var:10000_G_prior:Truth")
# print("S2:10000_LS:10")
# print_rmse_summary("Debugging/gSE_Params/s2Ls/S2:10000_LS:10_G_var:10000_G_prior:Truth")

# print("S2:100_LS:20")
# print_rmse_summary("Debugging/gSE_Params/s2Ls/S2:100_LS:20_G_var:10000_G_prior:Truth")
# print("S2:1000_LS:20")
# print_rmse_summary("Debugging/gSE_Params/s2Ls/S2:1000_LS:20_G_var:10000_G_prior:Truth")
# print("S2:10000_LS:20")
# print_rmse_summary("Debugging/gSE_Params/s2Ls/S2:10000_LS:20_G_var:10000_G_prior:Truth")

quad_goal_options = [np.array([-20,20]), np.array([20,20]), np.array([20,-20]), np.array([-20,-20])]
quad_debugging_folder_base = "Debugging/SE_MultTargets/Quad_PED/"
Tmax = 100
quad_sigma_g_values = [0.0, 0.1, 0.5, 1.0, 2.0]

csg_goal_options = [np.array([50,100]), np.array([35,115]), np.array([65,115]), np.array([35, 90]), 
np.array([65, 90]), np.array([50, 85]), np.array([35, 65]), np.array([65, 65]),
np.array([50, 60]), np.array([35, 45]), np.array([65, 45])]
csg_debugging_folder_base = "Debugging/SE_MultTargets/CSG_PED/"
Tmax = 150
csg_sigma_g_values = [0.0, 0.1, 0.2]

csg = True
separate_tracking = True


if csg:
    goal_options = csg_goal_options
    num_goal_options = 11
    debugging_folder_base = csg_debugging_folder_base
    Tmax = 150
    sigma_g_values = csg_sigma_g_values
else:
    goal_options = quad_goal_options
    num_goal_options = 4
    debugging_folder_base = quad_debugging_folder_base
    Tmax = 100
    sigma_g_values = quad_sigma_g_values

map_goal_idx = {tuple(goal): idx for idx, goal in enumerate(goal_options)}
G_var_values = [0, 1, 2, 5, 10]

end_dict = analyze_convergence_by_parameters(debugging_folder_base, map_goal_idx=map_goal_idx, sigma_options=sigma_g_values, gvar_options=G_var_values, num_goal_options=num_goal_options, Tmax=Tmax, separate_tracking=separate_tracking)
save_convergence_matrices_to_txt(end_dict, filename="Convergence_statistics.txt", folder=debugging_folder_base, separate_tracking=separate_tracking)