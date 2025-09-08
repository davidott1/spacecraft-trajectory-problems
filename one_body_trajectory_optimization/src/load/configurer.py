import numpy as np
import astropy.units as u
from pathlib import Path
from typing import Any, Dict


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
    print(f"    {'Variable':<{max_parameter_length}s} : {'Value':>{2*max_value_length+2}s} {'Unit':<8s}  {'Mode':>5s}")
    parameters_with_units = {}
    
    # Loop
    for parameter, value_unit_mode in input_params['input_parameters'].items():
        if isinstance(value_unit_mode, dict):
            # Handle value_unit as dictionary
            
            # Unpack
            value    = value_unit_mode['value']
            unit_str = value_unit_mode['unit']
            mode_str = value_unit_mode.get('mode', '')

            # Handle parameters with unit
            if unit_str not in ("None", None):
                if unit_str in (
                        's', 
                        'm', 'km',
                        'kg',  
                        'm/s', 'km/s',
                        'm/s^2', 'N/kg', 'km/s^2',
                        'm/s^3', 'km/s^3',
                        'm^2/s^2', 'km^2/s^2',
                        'm^2/s^4', 'km^2/s^4',
                        'N', 'kg*m/s^2'
                    ):
                    u_unit_unit_str = u.Unit(unit_str)
                else:
                    # unit not recognized, assume it is unit one
                    u_unit_unit_str = u.one
                parameters_with_units[parameter] = value * u_unit_unit_str
                if np.asarray(value_unit_mode['value']).ndim == 0: # type: ignore
                    value_str = str(f"{value_unit_mode['value']:>{max_value_length}.6e}") # type: ignore
                else:
                    value_str = ', '.join([str(f"{val:>{max_value_length}.6e}") for val in value_unit_mode['value']])

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
            print(f"    {parameter:<{max_parameter_length}s} : {value_str:>{2*max_value_length+2}s} {unit_str:<8s}  {mode_str:>5s}") # type: ignore

        elif isinstance(value_unit_mode, str):
            # Handle value_unit as string

            # Assign parameter value
            parameters_with_units[parameter] = value_unit_mode

            # Print row: variable, value
            print(f"  {parameter:<14s} : {value_unit_mode:>{2*max_value_length+2}s}")

    return parameters_with_units


# Convert to standard units: seconds, meters, kilograms, one
def _convert_parameters_to_standard_units(
        parameters_with_units: dict
    ) -> dict:
    parameters_with_units_defaults = {
        'min_type'              : [           'energy', None      , str   ],
        'time_f'                : [           1.0e+1  , u.s       , float ],
        'pos_vec_o'             : [ [ 0.0e+0, 0.0e+0 ], u.m       , float ], # type: ignore
        'vel_vec_o'             : [ [ 0.0e+0, 0.0e+0 ], u.m/u.s   , float ], # type: ignore
        'copos_vec_o'           : [ [ 0.0e+0, 0.0e+0 ], None      , float ], # type: ignore
        'covel_vec_o'           : [ [ 0.0e+0, 0.0e+0 ], None      , float ], # type: ignore
        'pos_vec_f'             : [ [ 1.0e+1, 1.0e+1 ], u.m       , float ], # type: ignore
        'vel_vec_f'             : [ [ 1.0e+0, 1.0e+0 ], u.m/u.s   , float ], # type: ignore
        'copos_vec_f'           : [ [ 0.0e+0, 0.0e+0 ], None      , float ], # type: ignore
        'covel_vec_f'           : [ [ 0.0e+0, 0.0e+0 ], None      , float ], # type: ignore
        'mass_o'                : [           1.0e+3  , u.kg      , float ], # type: ignore
        'exhaust_velocity'      : [           3.0e+3  , u.m/u.s   , float ], # type: ignore
        'constant_gravity'      : [           0.0e+0  , u.m/u.s**2, float ], # type: ignore
        'use_thrust_acc_limits' : [              False, None      , bool  ],
        'thrust_acc_min'        : [           0.0e+0  , u.m/u.s**2, float ], # type: ignore
        'thrust_acc_max'        : [           1.0e+0  , u.m/u.s**2, float ], # type: ignore
        'use_thrust_limits'     : [              False, None      , bool  ],
        'thrust_min'            : [           0.0e+0  , u.N       , float ], # type: ignore
        'thrust_max'            : [           1.0e+0  , u.N       , float ], # type: ignore
        'k_idxinitguess'        : [               None, None      , int   ],
        'k_idxfinsoln'          : [               None, None      , int   ],
        'k_idxdivs'             : [                 10, u.one     , int   ],
        'init_guess_steps'      : [               3000, u.one     , int   ],
    }

    # Dynamically set the default units for co-state
    if parameters_with_units_defaults['min_type'][0] == 'fuel':
        parameters_with_units_defaults['copos_vec_o'][1] = 1.0  /u.s
        parameters_with_units_defaults['covel_vec_o'][1] = u.one
        parameters_with_units_defaults['copos_vec_f'][1] = 1.0  /u.s
        parameters_with_units_defaults['covel_vec_f'][1] = u.one
    else: # assume min_type == 'energy'
        parameters_with_units_defaults['copos_vec_o'][1] = u.m/u.s**3 # type: ignore
        parameters_with_units_defaults['covel_vec_o'][1] = u.m/u.s**2 # type: ignore
        parameters_with_units_defaults['copos_vec_f'][1] = u.m/u.s**3 # type: ignore
        parameters_with_units_defaults['covel_vec_f'][1] = u.m/u.s**2 # type: ignore

    # Build the parameter-standard-units dictionary by applying defaults and converting units
    parameters_standard_units = {}
    standard_units = {
        'one'      : u.one,
        'time'     : u.s  , # type: ignore
        'distance' : u.m  , # type: ignore
        'mass'     : u.kg , # type: ignore
    }
    for param, (val_default, val_unit, val_type) in parameters_with_units_defaults.items():
        parameters_standard_units[param] = {}
        if isinstance(val_default, str):
            parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default)
            parameters_standard_units[param]['unit' ] = "None"
        elif isinstance(val_default, bool):
            parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default)
            parameters_standard_units[param]['unit' ] = "None"
        elif val_default is None:
            if parameters_with_units.get(param, val_default) is None:
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default)
                parameters_standard_units[param]['unit' ] = "None"
            elif parameters_with_units.get(param, val_default).unit == u.one:
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default).to_value(standard_units['one'])
                parameters_standard_units[param]['unit' ] = str(standard_units['one'])
            else:
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default)
                parameters_standard_units[param]['unit' ] = str(val_unit)
        else:
            if val_unit in (u.one, None):
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['one'])
                parameters_standard_units[param]['unit' ] = str(standard_units['one'])
            elif val_unit in (u.s,): # type: ignore
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['time'])
                parameters_standard_units[param]['unit' ] = str(standard_units['time'])
            elif val_unit in (u.m, u.km): # type: ignore
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['distance'])
                parameters_standard_units[param]['unit' ] = str(standard_units['distance'])
            elif val_unit in (u.m/u.s, u.km/u.s): # type: ignore
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['distance']/standard_units['time'])
                parameters_standard_units[param]['unit' ] = str(standard_units['distance']/standard_units['time'])
            elif val_unit in (u.kg,): # type: ignore
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['mass'])
                parameters_standard_units[param]['unit' ] = str(standard_units['mass'])
            elif val_unit in (u.N, u.kN, u.kg*u.m/u.s**2, u.kg*u.km/u.s**2): # type: ignore
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['mass']*standard_units['distance']/standard_units['time']**2)
                parameters_standard_units[param]['unit' ] = str(standard_units['mass']*standard_units['distance']/standard_units['time']**2)
            elif val_unit in (u.N/u.kg, u.kN/u.kg, u.m/u.s**2, u.km/u.s**2): # type: ignore
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['distance']/standard_units['time']**2)
                parameters_standard_units[param]['unit' ] = str(standard_units['distance']/standard_units['time']**2)
            elif val_unit in (1.0/u.s,): # type: ignore
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default * val_unit).to_value(1.0/standard_units['time'])
                parameters_standard_units[param]['unit' ] = str(1.0/standard_units['time'])
            elif val_unit in (u.m/u.s**3, u.km/u.s**3): # type: ignore
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['distance']/standard_units['time']**3)
                parameters_standard_units[param]['unit' ] = str(standard_units['distance']/standard_units['time']**3)
            elif val_unit in (u.m**2/u.s**4, u.km**2/u.s**4): # type: ignore
                parameters_standard_units[param]['value'] = parameters_with_units.get(param, val_default * val_unit).to_value(standard_units['distance']**2/standard_units['time']**4)
                parameters_standard_units[param]['unit' ] = str(standard_units['distance']**2/standard_units['time']**4)

        # Enforce types
        if val_type == int and parameters_standard_units[param]['value'] is not None:
            parameters_standard_units[param]['value'] = int(parameters_standard_units[param]['value'])

    return parameters_standard_units


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
    all_parameters_variable_units = \
        _configure_parameters(
            input_files_params,
            system_parameters ,
        )
    
    # Convert to standard units: seconds, meters, kilograms, one
    all_parameters_standard_units = \
        _convert_parameters_to_standard_units(
            all_parameters_variable_units,
        )
    
    # Determine if variable is free or fixed
    pos_vec_o_params = input_files_params['input_parameters'].get('pos_vec_o')
    pos_vec_o_mode = 'free' if pos_vec_o_params is None else pos_vec_o_params.get('mode', 'fixed')

    pos_vec_f_params = input_files_params['input_parameters'].get('pos_vec_f')
    pos_vec_f_mode = 'free' if pos_vec_f_params is None else pos_vec_f_params.get('mode', 'fixed')
    
    vel_vec_o_params = input_files_params['input_parameters'].get('vel_vec_o')
    vel_vec_o_mode = 'free' if vel_vec_o_params is None else vel_vec_o_params.get('mode', 'fixed')

    vel_vec_f_params = input_files_params['input_parameters'].get('vel_vec_f')
    vel_vec_f_mode = 'free' if vel_vec_f_params is None else vel_vec_f_params.get('mode', 'fixed')
    
    copos_vec_o_params = input_files_params['input_parameters'].get('copos_vec_o')
    copos_vec_o_mode = 'free' if copos_vec_o_params is None else copos_vec_o_params.get('mode', 'fixed')

    copos_vec_f_params = input_files_params['input_parameters'].get('copos_vec_f')
    copos_vec_f_mode = 'free' if copos_vec_f_params is None else copos_vec_f_params.get('mode', 'fixed')
    
    covel_vec_o_params = input_files_params['input_parameters'].get('covel_vec_o')
    covel_vec_o_mode = 'free' if covel_vec_o_params is None else covel_vec_o_params.get('mode', 'fixed')

    covel_vec_f_params = input_files_params['input_parameters'].get('covel_vec_f')
    covel_vec_f_mode = 'free' if covel_vec_f_params is None else covel_vec_f_params.get('mode', 'fixed')

    # Time or hamiltonian: free vs. fixed
    if input_files_params['input_parameters']['time_f']['mode'] == 'free':
        ham_f_mode = 'fixed'
        ham_f_mns  = 0.0
        ham_f_pls  = 0.0
    else:
        ham_f_mode = 'free'
        ham_f_mns  = 0.0  # dummy value
        ham_f_pls  = 0.0  # dummy value

    # Hamiltonian unit
    if all_parameters_standard_units['min_type'        ]['value'] == 'fuel':
        ham_unit = 'm/s'
    elif all_parameters_standard_units['min_type'        ]['value'] == 'energy':
        ham_unit = 'm^2/s^4'
    else:
        ham_unit = ''

    # Organize parameters into dictionaries: system parameters, integration-state parameters, equality parameters, and inequality parameters
    optimization_parameters = {
        'min_type'         : all_parameters_standard_units['min_type'        ]['value'],
        'init_guess_steps' : all_parameters_standard_units['init_guess_steps']['value'],
        'include_jacobian' : False                                           ,
    }
    integration_state_parameters = {
        'time_o'           : 0.0                                                       ,
        'time_f'           : all_parameters_standard_units['time_f']                   ,
        'delta_time_of'    : all_parameters_standard_units['time_f']['value'] - 0.0    ,
        'pos_vec_o'        : all_parameters_standard_units['pos_vec_o']['value']       , # type: ignore
        'vel_vec_o'        : all_parameters_standard_units['vel_vec_o']['value']       , # type: ignore
        'pos_vec_f'        : all_parameters_standard_units['pos_vec_f']['value']       , # type: ignore
        'vel_vec_f'        : all_parameters_standard_units['vel_vec_f']['value']       , # type: ignore
        'mass_o'           : all_parameters_standard_units['mass_o']['value']          , # type: ignore
        'exhaust_velocity' : all_parameters_standard_units['exhaust_velocity']['value'], # type: ignore
        'constant_gravity' : all_parameters_standard_units['constant_gravity']['value'], # type: ignore
        'opt_ctrl_obj_o'   : np.float64(0.0)                                           , # type: ignore
        'post_process'     : False                                                     ,
        'include_scstm'    : False                                                     ,
    }
    equality_parameters: Dict[str, Any]  = {
        'time': {
            'o': {
                'mode' : 'fixed',
                'unit' : 's'    ,
                'mns'  : 0.0    ,
                'pls'  : 0.0
            },
            'f': {
                'mode' : input_files_params['input_parameters']['time_f']['mode'],
                'unit' : all_parameters_standard_units['time_f']['unit' ]        ,
                'mns'  : all_parameters_standard_units['time_f']['value']        ,
                'pls'  : all_parameters_standard_units['time_f']['value']
            }
        },
        'pos_vec': {
            'o': {
                'mode' : pos_vec_o_mode                                     ,
                'unit' : all_parameters_standard_units['pos_vec_o']['unit' ],
                'mns'  : all_parameters_standard_units['pos_vec_o']['value'],
                'pls'  : all_parameters_standard_units['pos_vec_o']['value']
            },
            'f': {
                'mode' : pos_vec_f_mode                                     ,
                'unit' : all_parameters_standard_units['pos_vec_f']['unit' ],
                'mns'  : all_parameters_standard_units['pos_vec_f']['value'],
                'pls'  : all_parameters_standard_units['pos_vec_f']['value']
            }
        },
        'vel_vec': {
            'o': {
                'mode' : vel_vec_o_mode                                     ,
                'unit' : all_parameters_standard_units['vel_vec_o']['unit' ],
                'mns'  : all_parameters_standard_units['vel_vec_o']['value'],
                'pls'  : all_parameters_standard_units['vel_vec_o']['value']
            },
            'f': {
                'mode' : vel_vec_f_mode                                     ,
                'unit' : all_parameters_standard_units['vel_vec_f']['unit' ],
                'mns'  : all_parameters_standard_units['vel_vec_f']['value'],
                'pls'  : all_parameters_standard_units['vel_vec_f']['value']
            }
        },
        'copos_vec': {
            'o': {
                'mode' : copos_vec_o_mode                                     ,
                'unit' : all_parameters_standard_units['copos_vec_o']['unit' ],
                'mns'  : all_parameters_standard_units['copos_vec_o']['value'],
                'pls'  : all_parameters_standard_units['copos_vec_o']['value']
            },
            'f': {
                'mode' : copos_vec_f_mode                                     ,
                'unit' : all_parameters_standard_units['copos_vec_f']['unit' ],
                'mns'  : all_parameters_standard_units['copos_vec_f']['value'],
                'pls'  : all_parameters_standard_units['copos_vec_f']['value']
            }
        },
        'covel_vec': {
            'o': {
                'mode' : covel_vec_o_mode                                     ,
                'unit' : all_parameters_standard_units['covel_vec_o']['unit' ],
                'mns'  : all_parameters_standard_units['covel_vec_o']['value'],
                'pls'  : all_parameters_standard_units['covel_vec_o']['value']
            },
            'f': {
                'mode' : covel_vec_f_mode                                     ,
                'unit' : all_parameters_standard_units['covel_vec_f']['unit' ],
                'mns'  : all_parameters_standard_units['covel_vec_f']['value'],
                'pls'  : all_parameters_standard_units['covel_vec_f']['value']
            }
        },
        'ham': {
            'o': {
                'mode' : 'fixed',
                'unit' : ham_unit,
                'mns'  : 0.0, # dummy value
                'pls'  : 0.0  # dummy value
            },
            'f': {
                'mode' : ham_f_mode,
                'unit' : ham_unit,
                'mns'  : ham_f_mns,
                'pls'  : ham_f_pls
            }
        }
    }
    inequality_parameters = {
        'use_thrust_acc_limits'    : all_parameters_standard_units['use_thrust_acc_limits']['value'],
        'use_thrust_acc_smoothing' : False                                                          ,
        'thrust_acc_min'           : all_parameters_standard_units['thrust_acc_min'       ]['value'],
        'thrust_acc_max'           : all_parameters_standard_units['thrust_acc_max'       ]['value'],
        'use_thrust_limits'        : all_parameters_standard_units['use_thrust_limits'    ]['value'],
        'use_thrust_smoothing'     : False                                                          ,
        'thrust_min'               : all_parameters_standard_units['thrust_min'           ]['value'],
        'thrust_max'               : all_parameters_standard_units['thrust_max'           ]['value'],
        'k_idxinitguess'           : all_parameters_standard_units['k_idxinitguess'       ]['value'],
        'k_idxfinsoln'             : all_parameters_standard_units['k_idxfinsoln'         ]['value'],
        'k_idxdivs'                : all_parameters_standard_units['k_idxdivs'            ]['value'],
        'k_steepness'              : all_parameters_standard_units['k_idxinitguess'       ]['value'],
        'alpha'                    : 1.0
    }
    
    # Validate input
    #   Need to check "squareness" of boundary-value problem: n free = m fixed
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


