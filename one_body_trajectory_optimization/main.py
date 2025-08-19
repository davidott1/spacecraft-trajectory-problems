# Imports
import random
import numpy as np
from src.load.processor     import optimal_trajectory_input
from src.optimize.optimizer import optimal_trajectory_solve

np.random.seed(42)


# Main
def main():

    # Start optimization trajectory program
    print(2*"\n"+"==========================")
    print(       "OPTIMAL TRAJECTORY PROGRAM")
    print(       "==========================")

    # Optimal trajectory input
    (
        files_folders_parameters    ,
        system_parameters           ,
        optimization_parameters     ,
        integration_state_parameters,
        equality_parameters         ,
        inequality_parameters       ,
    ) = optimal_trajectory_input()
    
    # Optimal trajectory solve
    optimal_trajectory_solve(
        files_folders_parameters    ,
        system_parameters           ,
        optimization_parameters     ,
        integration_state_parameters,
        equality_parameters         ,
        inequality_parameters       ,
    )

    # End optimization trajectory program
    print()
    return True


if __name__ == '__main__':
    main()

