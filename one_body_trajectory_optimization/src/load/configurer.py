import numpy as np
import astropy.units as u
from pathlib import Path


# Create parameters dictionary and print to screen
def _configure_parameters(
        input_params      : dict     ,
        system_parameters : dict     ,
    ) -> dict:
    """
    Parses parameters from the input dict, handles units, and logs them to the console.
    """

    # Unpack
    max_value_length = system_parameters['max_value_length']

    # Loop initialization
    print("\n  Input Parameters")
    max_parameter_length = max([len(parameter) for parameter in input_params['input_parameters'].keys()])
    print(f"    {'Variable':<{max_parameter_length}s} : {'Value':>{2*max_value_length+2}s} {'Unit':<{max_value_length}s}")
    parameters_with_units = {}
    
    # Loop
    for parameter, value_unit in input_params['input_parameters'].items():
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
            print(f"    {parameter:<{max_parameter_length}s} : {value_str:>{2*max_value_length+2}s} {unit_str:<{max_value_length}s}") # type: ignore

        elif isinstance(value_unit, str):
            # Handle value_unit as string

            # Assign parameter value
            parameters_with_units[parameter] = value_unit

            # Print row: variable, value
            print(f"  {parameter:<14s} : {value_unit:>{2*max_value_length+2}s}")
            
    return parameters_with_units


# Convert to standard units: seconds, meters, kilograms, one
def _convert_parameters_to_standard_units(
        parameters_with_units: dict
    ) -> dict:
    
    parameters_with_units_defaults = {
        'min_type'              : (           'energy', None      , str   ),
        'time_span'             : ( [ 0.0e+0, 1.0e+1 ], u.s       , float ),
        'pos_vec_o'             : ( [ 0.0e+0, 0.0e+0 ], u.m       , float ), # type: ignore
        'vel_vec_o'             : ( [ 0.0e+0, 0.0e+0 ], u.m/u.s   , float ), # type: ignore
        'pos_vec_f'             : ( [ 1.0e+1, 1.0e+1 ], u.m       , float ), # type: ignore
        'vel_vec_f'             : ( [ 1.0e+0, 1.0e+0 ], u.m/u.s   , float ), # type: ignore
        'mass_o'                : (             1.0e+3, u.kg      , float ), # type: ignore
        'use_thrust_acc_limits' : (              False, None      , bool  ),
        'thrust_acc_min'        : (             0.0e+0, u.m/u.s**2, float ), # type: ignore
        'thrust_acc_max'        : (             1.0e+0, u.m/u.s**2, float ), # type: ignore
        'use_thrust_limits'     : (              False, None      , bool  ),
        'thrust_min'            : (             0.0e+0, u.N       , float ), # type: ignore
        'thrust_max'            : (             1.0e+0, u.N       , float ), # type: ignore
        'k_idxinitguess'        : (               None, None      , int   ),
        'k_idxfinsoln'          : (               None, None      , int   ),
        'k_idxdivs'             : (                 10, u.one     , int   ),
        'init_guess_steps'      : (               3000, u.one     , int   ),
    }

    # Build the parameter-without-units dictionary by applying defaults and converting units
    parameters_without_units = {}
    standard_units = {
        'one'      : u.one,
        'time'     : u.s  , # type: ignore
        'distance' : u.m  , # type: ignore
        'mass'     : u.kg , # type: ignore
    }
    for param, (val_default, val_unit, val_type) in parameters_with_units_defaults.items():
        if isinstance(val_default, str):
            parameters_without_units[param] = parameters_with_units.get(param, val_default)
        elif isinstance(val_default, bool):
            parameters_without_units[param] = parameters_with_units.get(param, val_default)
        elif val_default is None:
            if parameters_with_units.get(param, val_default) is None:
                parameters_without_units[param] = parameters_with_units.get(param, val_default)
            elif parameters_with_units.get(param, val_default).unit == u.one:
                parameters_without_units[param] = parameters_with_units.get(param, val_default).to_value(standard_units['one'])
            else:
                parameters_without_units[param] = parameters_with_units.get(param, val_default)
        else:
            if val_unit in (u.one, None):
                parameters_without_units[param] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['one'])
            elif val_unit in (u.s,): # type: ignore
                parameters_without_units[param] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['time'])
            elif val_unit in (u.m, u.km): # type: ignore
                parameters_without_units[param] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['distance'])
            elif val_unit in (u.m/u.s, u.km/u.s): # type: ignore
                parameters_without_units[param] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['distance']/standard_units['time'])
            elif val_unit in (u.kg,): # type: ignore
                parameters_without_units[param] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['mass'])
            elif val_unit in (u.N, u.kN, u.kg*u.m/u.s**2, u.kg*u.km/u.s**2): # type: ignore
                parameters_without_units[param] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['mass']*standard_units['distance']/standard_units['time']**2)
            elif val_unit in (u.N/u.kg, u.kN/u.kg, u.m/u.s**2, u.km/u.s**2): # type: ignore
                parameters_without_units[param] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['distance']/standard_units['time']**2)
        
        # Enforce types
        if val_type == int and parameters_without_units[param] is not None:
            parameters_without_units[param] = int(parameters_without_units[param])
    return parameters_without_units


# Validate input
def _validate_input(
    system_parameters       : dict,
    optimization_parameters : dict,
    inequality_parameters   : dict,
):
    # Print validation process
    print("\n  Validate Input")

    # Check if both thrust and thrust-acc constraints are set
    print("    Check thrust and thrust-acc constraints")
    if inequality_parameters['use_thrust_acc_limits'] and inequality_parameters['use_thrust_limits']:
        inequality_parameters['use_thrust_acc_limits'] = True
        inequality_parameters['use_thrust_limits']     = False
        print(
            "      Warning: Cannot use both thrust acceleration limits and thrust limits."
            + f" Choosing use_thrust_acc_limits = {inequality_parameters['use_thrust_acc_limits']} and use_thrust_limits = {inequality_parameters['use_thrust_limits']}."
        )
    
    # Check if min-type fuel is set but no thrust or thrust-acc constraint
    if (
            optimization_parameters['min_type'] == 'fuel'
            and inequality_parameters['use_thrust_acc_limits'] is False and inequality_parameters['use_thrust_limits'] is False
        ):
        inequality_parameters['use_thrust_acc_limits'] = True
        inequality_parameters['use_thrust_limits']     = False
        print(
            "      Warning: Min type is fuel, but no thrust or thrust-acc constraint is set."
            + f" Choosing use_thrust_acc_limits = {inequality_parameters['use_thrust_acc_limits']} and use_thrust_limits = {inequality_parameters['use_thrust_limits']}."
        )

    # Determine k values
    print("    Compute k-continuation parameters for thrust smoothing")

    # Check if k_idxdivs is valid
    if optimization_parameters['min_type'] == 'energy' and not inequality_parameters['use_thrust_acc_limits'] and not inequality_parameters['use_thrust_limits']:
        if inequality_parameters['k_idxdivs'] != 1:
            print(
                f"      Warning: Thrust smoothing is not needed using a k-continuation method:"
                + f" min_type is {optimization_parameters['min_type']},"
                + f" use_thrust_acc_limits is {inequality_parameters['use_thrust_acc_limits']},"
                + f" and use_thrust_limits is {inequality_parameters['use_thrust_limits']}."
                + f" \n               k_idxdivs = {inequality_parameters['k_idxdivs']} is not valid. Setting k_idxdivs = 1."
            )
            inequality_parameters['k_idxdivs'] = 1 # k has no purpose, but the loop needs to run once

    # Determine the first k value based on thrust or thrust-acc constraints if not an input
    if inequality_parameters['k_idxinitguess'] is None:
        if optimization_parameters['min_type'] == 'fuel':
            inequality_parameters['k_idxinitguess'] = np.float64(4.0e+0)
        else: # assume min_type energy
            if inequality_parameters['use_thrust_limits']:
                inequality_parameters['k_idxinitguess'] = np.float64(4.0e+0 / (inequality_parameters['thrust_max']     - inequality_parameters['thrust_min']     + 1.0e-9))
            elif inequality_parameters['use_thrust_acc_limits']:
                inequality_parameters['k_idxinitguess'] = np.float64(4.0e+0 / (inequality_parameters['thrust_acc_max'] - inequality_parameters['thrust_acc_min'] + 1.0e-9))
            else:
                inequality_parameters['k_idxinitguess'] = np.float64(4.0e+0)
        print(f"      Initial k-steepness : {inequality_parameters['k_idxinitguess']:<{system_parameters['max_value_length']}.6e}")
    else:
        inequality_parameters['k_idxinitguess'] = np.float64(inequality_parameters['k_idxinitguess']) # type: ignore

    # Determine last k value
    if inequality_parameters['k_idxfinsoln'] is None:
        inequality_parameters['k_idxfinsoln'] = np.float64(1.0e+1 * inequality_parameters['k_idxinitguess'])
        print(f"      Final k-steepness   : {inequality_parameters['k_idxfinsoln']:<{system_parameters['max_value_length']}.6e}")
    else:
        inequality_parameters['k_idxfinsoln'] = np.float64(inequality_parameters['k_idxfinsoln']) # type: ignore


def configure_validate_input(
        input_files_params : dict,
    ):

    # Unpack filepaths and folderpaths
    input_filepath    = Path(input_files_params[   'input_filepath'])
    output_folderpath = Path(input_files_params['output_folderpath'])
    files_folders_parameters = {
        'input_filepath'    : input_filepath   ,
        'output_folderpath' : output_folderpath,
    }

    # Set print parameters
    system_parameters = {
        'plot_show'        : True,
        'plot_save'        : True,
        'max_value_length' : 14,
    }

    # Create parameters dictionary and print to screen
    all_parameters_with_units = \
        _configure_parameters(
            input_files_params,
            system_parameters ,
        )

    # Convert to standard units: seconds, meters, kilograms, one
    all_parameters_without_units = \
        _convert_parameters_to_standard_units(
            all_parameters_with_units,
        )

    # Organize parameters into dictionaries: system parameters, integration-state parameters, equality parameters, and inequality parameters
    optimization_parameters = {
        'min_type'         : all_parameters_without_units['min_type'        ],
        'init_guess_steps' : all_parameters_without_units['init_guess_steps'],
        'include_jacobian' : False,
    }
    integration_state_parameters = {
        'time_span'             : all_parameters_without_units['time_span'],
        'pos_vec_o'             : all_parameters_without_units['pos_vec_o'], # type: ignore
        'vel_vec_o'             : all_parameters_without_units['vel_vec_o'], # type: ignore
        'pos_vec_f'             : all_parameters_without_units['pos_vec_f'], # type: ignore
        'vel_vec_f'             : all_parameters_without_units['vel_vec_f'], # type: ignore
        'mass_o'                : all_parameters_without_units['mass_o'   ], # type: ignore
        'post_process'          : False, # type: ignore
    }
    equality_parameters = {
          'pos_vec_o_mns' : all_parameters_without_units['pos_vec_o'],
          'vel_vec_o_mns' : all_parameters_without_units['vel_vec_o'],
          'pos_vec_o_pls' : all_parameters_without_units['pos_vec_o'], # by design
          'vel_vec_o_pls' : all_parameters_without_units['vel_vec_o'], # by design
        'copos_vec_o_mns' :                      np.array([0.0, 0.0]),
        'covel_vec_o_mns' :                      np.array([0.0, 0.0]),
        'copos_vec_o_pls' :                      np.array([0.0, 0.0]),
        'covel_vec_o_pls' :                      np.array([0.0, 0.0]),
          'pos_vec_f_mns' :                      np.array([0.0, 0.0]),
          'vel_vec_f_mns' :                      np.array([0.0, 0.0]),
          'pos_vec_f_pls' : all_parameters_without_units['pos_vec_f'],
          'vel_vec_f_pls' : all_parameters_without_units['vel_vec_f'],
        'copos_vec_f_mns' :                      np.array([0.0, 0.0]),
        'covel_vec_f_mns' :                      np.array([0.0, 0.0]),
        'copos_vec_f_pls' :                      np.array([0.0, 0.0]),
        'covel_vec_f_pls' :                      np.array([0.0, 0.0]),
    }
    inequality_parameters = {
        'use_thrust_acc_limits'    : all_parameters_without_units['use_thrust_acc_limits'],
        'use_thrust_acc_smoothing' : False                                                ,
        'thrust_acc_min'           : all_parameters_without_units['thrust_acc_min'       ], # type: ignore
        'thrust_acc_max'           : all_parameters_without_units['thrust_acc_max'       ], # type: ignore
        'use_thrust_limits'        : all_parameters_without_units['use_thrust_limits'    ],
        'use_thrust_smoothing'     : False                                                ,
        'thrust_min'               : all_parameters_without_units['thrust_min'           ], # type: ignore
        'thrust_max'               : all_parameters_without_units['thrust_max'           ], # type: ignore
        'k_idxinitguess'           : all_parameters_without_units['k_idxinitguess'       ],
        'k_idxfinsoln'             : all_parameters_without_units['k_idxfinsoln'         ],
        'k_idxdivs'                : all_parameters_without_units['k_idxdivs'            ],
        'k_steepness'              : all_parameters_without_units['k_idxinitguess'       ],  
    }

    # Validate input
    _validate_input(
        system_parameters      ,
        optimization_parameters,
        inequality_parameters  ,
    )

    return (
        files_folders_parameters    ,
        system_parameters           ,
        optimization_parameters     ,
        integration_state_parameters,
        equality_parameters         ,
        inequality_parameters       ,
    )


def configure_output(
        output_folderpath_input: str,
    ):
    output_folderpath = Path(output_folderpath_input)
    output_folderpath.mkdir(parents=True, exist_ok=True)
    print(f"    Output Folderpath : {output_folderpath}")
    return str(output_folderpath)


