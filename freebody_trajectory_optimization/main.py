# Imports
import random
import numpy as np
from src.load.processor     import optimal_trajectory_input
from src.optimize.optimizer import optimal_trajectory_solve

np.random.seed(42)


# Main
def main():

    # Start optimization trajectory program
    print("\n==========================")
    print(  "OPTIMAL TRAJECTORY PROGRAM")
    print(  "==========================")

    # Optimal trajectory input
    # (
    #     min_type                    ,
    #     time_span                   ,
    #     boundary_condition_pos_vec_o,
    #     boundary_condition_vel_vec_o,
    #     boundary_condition_pos_vec_f,
    #     boundary_condition_vel_vec_f,
    #     use_thrust_acc_limits       ,
    #     thrust_acc_min              ,
    #     thrust_acc_max              ,
    #     use_thrust_limits           ,
    #     thrust_min                  ,
    #     thrust_max                  ,
    #     k_idxinitguess              ,
    #     k_idxfinsoln                ,
    #     k_idxdivs                   ,
    #     init_guess_steps            ,
    #     mass_o                      ,
    #     input_filepath              ,
    #     output_folderpath           ,
    # ) = \
    #     optimal_trajectory_input()
    files_folders_params = optimal_trajectory_input()

    # Optimal trajectory solve
    optimal_trajectory_solve(files_folders_params)
    # optimal_trajectory_solve(
    #     time_span                                           ,
    #     boundary_condition_pos_vec_o                        ,
    #     boundary_condition_vel_vec_o                        ,
    #     boundary_condition_pos_vec_f                        ,
    #     boundary_condition_vel_vec_f                        ,
    #     min_type                     = min_type             ,
    #     use_thrust_acc_limits        = use_thrust_acc_limits,
    #     thrust_acc_min               = thrust_acc_min       ,
    #     thrust_acc_max               = thrust_acc_max       ,
    #     use_thrust_limits            = use_thrust_limits    ,
    #     thrust_min                   = thrust_min           ,
    #     thrust_max                   = thrust_max           ,
    #     k_idxinitguess               = k_idxinitguess       ,
    #     k_idxfinsoln                 = k_idxfinsoln         , 
    #     k_idxdivs                    = k_idxdivs            ,
    #     init_guess_steps             = init_guess_steps     ,
    #     mass_o                       = mass_o               ,
    #     input_filepath               = input_filepath       ,
    #     output_folderpath            = output_folderpath    ,
    # )

    # End optimization trajectory program
    print()


if __name__ == '__main__':
    main()

