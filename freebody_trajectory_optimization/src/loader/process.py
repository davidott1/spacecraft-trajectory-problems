import numpy as np
import astropy.units as u
from pathlib import Path


def process_input(
        input_files_params,
    ):

    # Print
    print("\nInput Parameters")

    # Unpack
    input_filename = Path(input_files_params['input_filename']).stem

    # Create parameters dictionary and print to screen
    max_parameter_length = max([len(parameter) for parameter in input_files_params['input_parameters'].keys()])
    max_value_length     = 14
    print(f"  {'Variable':<{max_parameter_length}s} : {'Value':>{2*max_value_length+2}s} {'Unit':<{max_value_length}s}")
    parameters_with_units = {}
    for parameter, value_unit in input_files_params['input_parameters'].items():
        if isinstance(value_unit, dict):
            # Handle value_unit as dictionary
            
            # Unpack
            value    = value_unit['value']
            unit_str = value_unit['unit']

            # Handle parameters with unit
            if unit_str not in ("None", None):
                parameters_with_units[parameter] = value * u.Unit(unit_str)
                if np.asarray(value_unit['value']).ndim == 0: # type: ignore
                    value_str = str(f"{value_unit['value']:>{max_value_length}.6e}") # type: ignore
                else:
                    value_str = ', '.join([str(f"{val:>{max_value_length}.6e}") for val in value_unit['value']])

            # Handle parameters with None unit
            else:
                if isinstance(value, (str, bool)):
                    parameters_with_units[parameter] = value
                    value_str = str(value)
                    unit_str  = ""
                elif isinstance(value, (int, float)): 
                    parameters_with_units[parameter] = value * u.one
                    value_str = str(value)
                    unit_str  = ""
            
            # Print row: variable, value, and unit
            print(f"  {parameter:<{max_parameter_length}s} : {value_str:>{2*max_value_length+2}s} {unit_str:<{max_value_length}s}") # type: ignore

        elif isinstance(value_unit, str):
            # Handle value_unit as string

            # Assign parameter value
            parameters_with_units[parameter] = value_unit

            # Print row: variable, value
            print(f"  {parameter:<14s} : {value_unit:>{2*max_value_length+2}s}")

    # Convert to standard units: seconds, meters, kilograms, one
    min_type              = parameters_with_units.get(             'min_type', 'energy'                            )
    time_span             = parameters_with_units.get(            'time_span', [ 0.0e+0, 1.0e+1 ] * u.s            ).to_value(u.s            ) # type: ignore
    pos_vec_o             = parameters_with_units.get(            'pos_vec_o', [ 0.0e+0, 0.0e+0 ] * u.m            ).to_value(u.m            ) # type: ignore
    vel_vec_o             = parameters_with_units.get(            'vel_vec_o', [ 0.0e+0, 0.0e+0 ] * u.m/u.s        ).to_value(u.m/u.s        ) # type: ignore
    pos_vec_f             = parameters_with_units.get(            'pos_vec_f', [ 1.0e+1, 1.0e+1 ] * u.m            ).to_value(u.m            ) # type: ignore
    vel_vec_f             = parameters_with_units.get(            'vel_vec_f', [ 1.0e+0, 1.0e+0 ] * u.m/u.s        ).to_value(u.m/u.s        ) # type: ignore
    mass_o                = parameters_with_units.get(               'mass_o', 1.0e+3             * u.kg           ).to_value(u.kg           ) # type: ignore
    use_thrust_acc_limits = parameters_with_units.get('use_thrust_acc_limits', False                               )
    thrust_acc_min        = parameters_with_units.get(       'thrust_acc_min', 0.0e+0             * u.m/u.s**2     ).to_value(u.m/u.s**2     ) # type: ignore
    thrust_acc_max        = parameters_with_units.get(       'thrust_acc_max', 1.0e+0             * u.m/u.s**2     ).to_value(u.m/u.s**2     ) # type: ignore
    use_thrust_limits     = parameters_with_units.get(    'use_thrust_limits', False                               )
    thrust_min            = parameters_with_units.get(           'thrust_min', 0.0e+0             * u.kg*u.m/u.s**2).to_value(u.kg*u.m/u.s**2) # type: ignore
    thrust_max            = parameters_with_units.get(           'thrust_max', 1.0e+0             * u.kg*u.m/u.s**2).to_value(u.kg*u.m/u.s**2) # type: ignore
    k_idxinitguess        = parameters_with_units.get(       'k_idxinitguess', None                                )
    k_idxfinsoln          = parameters_with_units.get(         'k_idxfinsoln', None                                )
    k_idxdivs             = parameters_with_units.get(            'k_idxdivs', 10                 * u.one          ).to_value(u.one          ) # type: ignore
    init_guess_steps      = parameters_with_units.get(     'init_guess_steps', 3000               * u.one          ).to_value(u.one          ) # type: ignore

    # Create boundary conditions
    boundary_condition_pos_vec_o = pos_vec_o
    boundary_condition_vel_vec_o = vel_vec_o
    boundary_condition_pos_vec_f = pos_vec_f
    boundary_condition_vel_vec_f = vel_vec_f

    # Validate input
    print("\nValidate Input")

    # Enforce types
    k_idxdivs        = int(k_idxdivs)
    init_guess_steps = int(init_guess_steps)

    # Check if both thrust and thrust-acc constraints are set
    print("  Check thrust and thrust-acc constraints")
    if use_thrust_acc_limits and use_thrust_limits:
        use_thrust_acc_limits = True
        use_thrust_limits     = False
        print(
            "    Warning: Cannot use both thrust acceleration limits and thrust limits."
            + f" Choosing use_thrust_acc_limits = {use_thrust_acc_limits} and use_thrust_limits = {use_thrust_limits}."
        )
    
    # Check if min-type fuel is set but no thrust or thrust-acc constraint
    if (
            min_type=='fuel' 
            and use_thrust_acc_limits is False and use_thrust_limits is False
        ):
        use_thrust_acc_limits = True
        use_thrust_limits     = False
        print(
            "    Warning: Min type is fuel, but no thrust or thrust-acc constraint is set."
            + f"   Choosing use_thrust_acc_limits = {use_thrust_acc_limits} and use_thrust_limits = {use_thrust_limits}."
        )

    # Determine k values
    print("  Compute k-continuation parameters for thrust smoothing")

    # Check if k_idxdivs is valid
    if min_type == 'energy' and not use_thrust_acc_limits and not use_thrust_limits:
        if k_idxdivs != 1:
            print(
                f"    Warning: Thrust smoothing is not needed using a k-continuation method:"
                + f" min_type is {min_type},"
                + f" use_thrust_acc_limits is {use_thrust_acc_limits},"
                + f" and use_thrust_limits is {use_thrust_limits}."
                + f" \n             k_idxdivs = {k_idxdivs} is not valid. Setting k_idxdivs = 1."
            )
            k_idxdivs = 1 # k has no purpose, but the loop needs to run once

    # Determine the first k value based on thrust or thrust-acc constraints if not an input
    if k_idxinitguess is None:
        if min_type == 'fuel':
            k_idxinitguess = np.float64(4.0e+0)
        else: # assume min_type energy
            if use_thrust_limits:
                k_idxinitguess = np.float64(4.0e+0 / (thrust_max     - thrust_min     + 1.0e-9))
            elif use_thrust_acc_limits:
                k_idxinitguess = np.float64(4.0e+0 / (thrust_acc_max - thrust_acc_min + 1.0e-9))
            else:
                k_idxinitguess = np.float64(4.0e+0)
        print(f"    Initial k-steepness : {k_idxinitguess:<{max_value_length}.6e}")
    else:
        k_idxinitguess = np.float64(k_idxinitguess) # type: ignore

    # Determine last k value
    if k_idxfinsoln is None:
        k_idxfinsoln = np.float64(1.0e+1 * k_idxinitguess)
        print(f"    Final k-steepness   : {k_idxfinsoln:<{max_value_length}.6e}")
    else:
        k_idxfinsoln = np.float64(k_idxfinsoln) # type: ignore

    # Pack up variable input
    return (
        min_type                    ,
        time_span                   ,
        boundary_condition_pos_vec_o,
        boundary_condition_vel_vec_o,
        boundary_condition_pos_vec_f,
        boundary_condition_vel_vec_f,
        use_thrust_acc_limits       ,
        thrust_acc_min              ,
        thrust_acc_max              ,
        use_thrust_limits           ,
        thrust_min                  ,
        thrust_max                  ,
        k_idxinitguess              ,
        k_idxfinsoln                , 
        k_idxdivs                   ,
        init_guess_steps            ,
        mass_o                      ,
        input_filename              ,
    )