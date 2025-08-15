import numpy as np
from tqdm import tqdm
# from src.optimize.optimizer import tpbvp_objective_and_jacobian

# def generate_guess(
#         time_span                    : np.ndarray                     ,
#         boundary_condition_pos_vec_o : np.ndarray                     ,
#         boundary_condition_vel_vec_o : np.ndarray                     ,
#         boundary_condition_pos_vec_f : np.ndarray                     ,
#         boundary_condition_vel_vec_f : np.ndarray                     ,
#         min_type                     : str        = 'energy'          ,
#         mass_o                       : np.float64 = np.float64(1.0e+3),
#         use_thrust_acc_limits        : bool       = True              ,
#         thrust_acc_min               : np.float64 = np.float64(0.0e+0),
#         thrust_acc_max               : np.float64 = np.float64(1.0e+1),
#         use_thrust_limits            : bool       = False             ,
#         thrust_min                   : np.float64 = np.float64(0.0e+0),
#         thrust_max                   : np.float64 = np.float64(1.0e+1),
#         k_steepness                  : np.float64 = np.float64(0.0e+0),
#         init_guess_steps             : int        = 3000              ,
#     ):
#     """
#     Generates a robust initial guess for the co-states: copos_vec, covel_vec
#     """
#     print("\nInitial Guess Process")

#     # Loop through random guesses for the costates
#     print("  Random Initial Guess Generation")
#     error_mag_min = np.Inf
#     for idx in tqdm(range(init_guess_steps), desc="Processing", leave=False, total=init_guess_steps):
#         copos_vec_o        = np.random.uniform(low=-1, high=1, size=2)
#         covel_vec_o        = np.random.uniform(low=-1, high=1, size=2)
#         decision_state_idx = np.hstack([copos_vec_o, covel_vec_o])
        
#         error_idx = \
#             tpbvp_objective_and_jacobian(
#                 decision_state_idx                                  ,
#                 time_span                                           ,
#                 boundary_condition_pos_vec_o                        ,
#                 boundary_condition_vel_vec_o                        ,
#                 boundary_condition_pos_vec_f                        ,
#                 boundary_condition_vel_vec_f                        ,
#                 min_type                     = min_type             ,
#                 mass_o                       = mass_o               ,
#                 use_thrust_acc_limits        = use_thrust_acc_limits,
#                 use_thrust_acc_smoothing     = True                 ,
#                 thrust_acc_min               = thrust_acc_min       ,
#                 thrust_acc_max               = thrust_acc_max       ,
#                 use_thrust_limits            = use_thrust_limits    ,
#                 use_thrust_smoothing         = True                 ,
#                 thrust_min                   = thrust_min           ,
#                 thrust_max                   = thrust_max           ,
#                 k_steepness                  = k_steepness          ,
#                 include_jacobian             = False                ,
#             )

#         error_mag_idx = np.linalg.norm(error_idx)
#         if error_mag_idx < error_mag_min:
#             idx_min            = idx
#             error_mag_min      = error_mag_idx
#             decision_state_min = decision_state_idx
#             integ_state_min    = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decision_state_min])
#             if idx==0:
#                 tqdm.write(f"                               {'Fixed':>14s} {'Fixed':>14s} {'Fixed':>14s} {'Fixed':>14s} {'Free':>14s} {'Free':>14s} {'Free':>14s} {'Free':>14s}")
#                 tqdm.write(f"          {'Step':>5s} {'Error-Mag':>14s} {'Pos-Xo':>14s} {'Pos-Yo':>14s} {'Vel-Xo':>14s} {'Vel-Yo':>14s} {'Co-Pos-Xo':>14s} {'Co-Pos-Yo':>14s} {'Co-Vel-Xo':>14s} {'Co-Vel-Yo':>14s}")
#             integ_state_min_str = ' '.join(f"{x:>14.6e}" for x in integ_state_min)
#             tqdm.write(f"     {idx_min:>5d}/{init_guess_steps:>4d} {error_mag_min:>14.6e} {integ_state_min_str}")

#     # Pack up and print solution
#     costate_o_guess = decision_state_min
#     return costate_o_guess