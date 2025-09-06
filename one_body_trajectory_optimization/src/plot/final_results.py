import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as mplt
import matplotlib.axes   as maxes
from matplotlib.widgets import Button
from src.utility.bounding_functions import bounded_smooth_func
from pathlib import Path


def uppercase_first(s: str) -> str:
    return s[:1].upper() + s[1:]


def plot_final_results(
        results_finsoln             ,
        files_folders_parameters    ,
        system_parameters           ,
        optimization_parameters     ,
        equality_parameters         ,
        inequality_parameters       ,
    ):
    """
    Calculates and plots all relevant results for the final trajectory solution.
    """

    # Unpack parameters
    min_type                 =  optimization_parameters['min_type'                ]
    use_thrust_acc_limits    =    inequality_parameters['use_thrust_acc_limits'   ]
    use_thrust_acc_smoothing =    inequality_parameters['use_thrust_acc_smoothing']
    thrust_acc_min           =    inequality_parameters['thrust_acc_min'          ]
    thrust_acc_max           =    inequality_parameters['thrust_acc_max'          ]
    use_thrust_limits        =    inequality_parameters['use_thrust_limits'       ]
    use_thrust_smoothing     =    inequality_parameters['use_thrust_smoothing'    ]
    thrust_min               =    inequality_parameters['thrust_min'              ]
    thrust_max               =    inequality_parameters['thrust_max'              ]
    k_steepness              =    inequality_parameters['k_steepness'             ]
    alpha                    =    inequality_parameters['alpha'                   ]
    input_filepath           = files_folders_parameters['input_filepath'          ]
    output_folderpath        = files_folders_parameters['output_folderpath'       ]
    plot_show                =        system_parameters['plot_show'               ]
    plot_save                =        system_parameters['plot_save'               ]

    # Unpack state and costate histories
    time_t                                     = results_finsoln.t
    state_t                                    = results_finsoln.y
    pos_x_t, pos_y_t, vel_x_t, vel_y_t         = state_t[0:4]
    copos_x_t, copos_y_t, covel_x_t, covel_y_t = state_t[4:8]
    mass_t                                     = state_t[-2]
    opt_ctrl_obj_t                             = state_t[-1]

    # Double check terminal states if variable is free
    if equality_parameters[  'pos_vec']['o']['mode'] == 'free': equality_parameters[  'pos_vec']['o']['mns'] = np.array([   pos_x_t[ 0],   pos_y_t[ 0] ])
    if equality_parameters[  'vel_vec']['o']['mode'] == 'free': equality_parameters[  'vel_vec']['o']['mns'] = np.array([   vel_x_t[ 0],   vel_y_t[ 0] ])
    if equality_parameters['copos_vec']['o']['mode'] == 'free': equality_parameters['copos_vec']['o']['mns'] = np.array([ copos_x_t[ 0], copos_y_t[ 0] ])
    if equality_parameters['covel_vec']['o']['mode'] == 'free': equality_parameters['covel_vec']['o']['mns'] = np.array([ covel_x_t[ 0], covel_y_t[ 0] ])
    if equality_parameters[  'pos_vec']['f']['mode'] == 'free': equality_parameters[  'pos_vec']['f']['pls'] = np.array([   pos_x_t[-1],   pos_y_t[-1] ])
    if equality_parameters[  'vel_vec']['f']['mode'] == 'free': equality_parameters[  'vel_vec']['f']['pls'] = np.array([   vel_x_t[-1],   vel_y_t[-1] ])
    if equality_parameters['copos_vec']['f']['mode'] == 'free': equality_parameters['copos_vec']['f']['pls'] = np.array([ copos_x_t[-1], copos_y_t[-1] ])
    if equality_parameters['covel_vec']['f']['mode'] == 'free': equality_parameters['covel_vec']['f']['pls'] = np.array([ covel_x_t[-1], covel_y_t[-1] ])

    pos_vec_o_mns   = equality_parameters[  'pos_vec']['o']['mns']
    vel_vec_o_mns   = equality_parameters[  'vel_vec']['o']['mns']
    copos_vec_o_mns = equality_parameters['copos_vec']['o']['mns']
    covel_vec_o_mns = equality_parameters['covel_vec']['o']['mns']
    pos_vec_f_pls   = equality_parameters[  'pos_vec']['f']['pls']
    vel_vec_f_pls   = equality_parameters[  'vel_vec']['f']['pls']
    copos_vec_f_pls = equality_parameters['copos_vec']['f']['pls']
    covel_vec_f_pls = equality_parameters['covel_vec']['f']['pls']

    # Recalculate the thrust profile to match the dynamics function
    if min_type == 'fuel':
        epsilon          = np.float64(1.0e-6)
        covel_mag_t      = np.sqrt(covel_x_t**2 + covel_y_t**2 + epsilon**2)
        switching_func_t = covel_mag_t - 1.0
        thrust_acc_mag_t = np.zeros_like(switching_func_t)
        for idx, switching_func_t_value in enumerate(switching_func_t):
            if use_thrust_limits:
                thrust_acc_min = thrust_min / mass_t[idx]
                thrust_acc_max = thrust_max / mass_t[idx]
            if switching_func_t_value > 0:
                thrust_acc_mag_t[idx] = thrust_acc_max
            elif switching_func_t_value < 0:
                thrust_acc_mag_t[idx] = thrust_acc_min
            else:
                thrust_acc_mag_t[idx] = thrust_acc_min
        thrust_acc_dir_t = np.array([ -covel_x_t/covel_mag_t, -covel_y_t/covel_mag_t ])
        thrust_acc_vec_t = thrust_acc_mag_t * thrust_acc_dir_t
    elif min_type == 'energyfuel':
        covel_mag_t      = np.sqrt(covel_x_t**2 + covel_y_t**2)
        switching_func_t = covel_mag_t - (1.0 - alpha)
        thrust_acc_mag_t = np.zeros_like(switching_func_t)
        for idx, switching_func_t_value in enumerate(switching_func_t):
            if use_thrust_limits:
                thrust_acc_min = thrust_min / mass_t[idx]
                thrust_acc_max = thrust_max / mass_t[idx]
            if switching_func_t_value > 0:
                # Thrust on
                thrust_acc_mag_t[idx] = (covel_mag_t[idx] - (1 - alpha)) / alpha
                thrust_acc_mag_t[idx] = bounded_smooth_func(thrust_acc_mag_t[idx], thrust_acc_min, thrust_acc_max, k_steepness)
            else:
                # Thrust off
                thrust_acc_mag_t[idx] = 0.0
        thrust_acc_x_dir_t = -covel_x_t / covel_mag_t
        thrust_acc_y_dir_t = -covel_y_t / covel_mag_t
        thrust_acc_dir_t   = np.vstack([thrust_acc_x_dir_t, thrust_acc_y_dir_t])
        thrust_acc_vec_t = thrust_acc_mag_t * thrust_acc_dir_t
    else: # assumes energy
        if use_thrust_limits:
            covel_mag_t = np.sqrt(covel_x_t**2 + covel_y_t**2)
            if use_thrust_smoothing:
                # Thrust or thrust-acc constraint smoothing
                thrust_acc_mag_t = covel_mag_t
                thrust_acc_mag_t = bounded_smooth_func(thrust_acc_mag_t, thrust_min / mass_t, thrust_max / mass_t, k_steepness)
            else:
                # No thrust or thrust-acc constraint smoothing
                thrust_acc_mag_t = covel_mag_t
                thrust_acc_mag_t = np.minimum( thrust_acc_mag_t, thrust_max / mass_t ) # max thrust-acc constraint
                thrust_acc_mag_t = np.maximum( thrust_acc_mag_t, thrust_min / mass_t ) # min thrust-acc constraint
            thrust_acc_x_dir_t = -covel_x_t / covel_mag_t
            thrust_acc_y_dir_t = -covel_y_t / covel_mag_t
            thrust_acc_dir_t   = np.vstack([thrust_acc_x_dir_t, thrust_acc_y_dir_t])
            thrust_acc_x_t     = thrust_acc_mag_t * thrust_acc_x_dir_t
            thrust_acc_y_t     = thrust_acc_mag_t * thrust_acc_y_dir_t
            thrust_acc_vec_t   = np.vstack([thrust_acc_x_t, thrust_acc_y_t])
        elif use_thrust_acc_limits:
            covel_mag_t = np.sqrt(covel_x_t**2 + covel_y_t**2)
            if use_thrust_acc_smoothing:
                # Thrust or thrust-acc constraint smoothing
                thrust_acc_mag_t = covel_mag_t
                thrust_acc_mag_t = bounded_smooth_func(thrust_acc_mag_t, thrust_acc_min, thrust_acc_max, k_steepness)
            else:
                # No thrust or thrust-acc constraint smoothing
                thrust_acc_mag_t = covel_mag_t
                thrust_acc_mag_t = np.minimum( thrust_acc_mag_t, thrust_acc_max ) # max thrust-acc constraint
                thrust_acc_mag_t = np.maximum( thrust_acc_mag_t, thrust_acc_min ) # min thrust-acc constraint
            thrust_acc_x_dir_t = -covel_x_t / covel_mag_t
            thrust_acc_y_dir_t = -covel_y_t / covel_mag_t
            thrust_acc_dir_t   = np.vstack([thrust_acc_x_dir_t, thrust_acc_y_dir_t])
            thrust_acc_x_t     = thrust_acc_mag_t * thrust_acc_x_dir_t
            thrust_acc_y_t     = thrust_acc_mag_t * thrust_acc_y_dir_t
            thrust_acc_vec_t   = np.vstack([thrust_acc_x_t, thrust_acc_y_t])
        else: # assume no thrust nor thrust-acc constraints
            # Thrust or thrust-acc constraints
            thrust_acc_vec_t = np.array([ -covel_x_t, -covel_y_t ])
            thrust_acc_dir_t = thrust_acc_vec_t / np.linalg.norm(thrust_acc_vec_t, axis=0, keepdims=True)
            thrust_acc_mag_t = np.sqrt( thrust_acc_vec_t[0]**2 + thrust_acc_vec_t[1]**2 )
    thrust_mag_t = mass_t * thrust_acc_mag_t
    thrust_dir_t = thrust_acc_dir_t # same
    thrust_vec_t = thrust_mag_t * thrust_dir_t

    # Thrust plotting parameters
    if use_thrust_limits:
        color_thrust      = mcolors.CSS4_COLORS['blue']
        label_thrust_name = 'Thrust Mag'
        label_thrust_unit = '[kg$\cdot$m/s$^2$]'
        title_thrust      = 'Thrust Max'
    else: # assume use_thrust_acc_limits
        color_thrust      = mcolors.CSS4_COLORS['red']
        label_thrust_name = 'Thrust Acc Mag'
        label_thrust_unit = '[m/s$^2$]'
        title_thrust      = 'Thrust Acceleration Max'

    # Create trajectory figure
    mplt.style.use('seaborn-v0_8-whitegrid')
    fig = mplt.figure(figsize=(15,8))
    gs = fig.add_gridspec(5, 2, width_ratios=[8, 7])

    # Configure figure
    if min_type == 'fuel':
        title_min_type = "Minimum Fuel"
    elif min_type == 'energy':
        title_min_type = "Minimum Energy"
    elif min_type == 'energyfuel':
        title_min_type = "Minimum Energy to Fuel"
    else: # assume energy
        title_min_type = ""
    fig.suptitle(
        f"OPTIMAL TRAJECTORY: {title_min_type}"
        "\nOne-Body Dynamics"
        f"\n{uppercase_first(equality_parameters['time']['f']['mode'])} Time-f |"
        f" {uppercase_first(equality_parameters['pos_vec']['o']['mode'])} Initial-Position"
        f", {uppercase_first(equality_parameters['vel_vec']['o']['mode'])} Initial-Velocity"
        f" to {uppercase_first(equality_parameters['pos_vec']['f']['mode'])} Final-Position"
        f", {uppercase_first(equality_parameters['vel_vec']['f']['mode'])} Final-Velocity"
        f"\n{title_thrust}",
        fontsize   = 16      ,
        fontweight = 'normal',
    )

    # 2D position path: pos-x vs. pos-y
    fig_w, fig_h  = fig.get_size_inches()
    ax_height     = 0.75
    ax_width      = ax_height * (fig_h / fig_w)
    square_coords = [0.07, 0.1, ax_width, ax_height]
    ax1_pos       = fig.add_axes(square_coords) # type: ignore
    def _plot_thrust_on_position_space(
            ax1_pos               : maxes.Axes                             ,
            pos_x_t               : np.ndarray                             ,
            pos_y_t               : np.ndarray                             ,
            thrust_acc_vec_t      : np.ndarray                             ,
            thrust_acc_mag_t      : np.ndarray                             ,
            thrust_vec_t          : np.ndarray                             ,
            thrust_mag_t          : np.ndarray                             ,
            use_thrust_acc_limits : bool       = True                      ,
            thrust_acc_min        : np.float64 = np.float64(0.0e+0)        ,
            thrust_acc_max        : np.float64 = np.float64(1.0e+1)        ,
            use_thrust_limits     : bool       = False                     ,
            thrust_min            : np.float64 = np.float64(0.0e+0)        ,
            thrust_max            : np.float64 = np.float64(1.0e+1)        ,
            color_thrust                       = mcolors.CSS4_COLORS['red'],
        ):

        min_pos = min(min(pos_x_t), min(pos_y_t))
        max_pos = max(max(pos_x_t), max(pos_y_t))
        if use_thrust_limits:
            plot_thrust_max  = max(thrust_mag_t)
            thrust_vec_scale = 0.2 * (max_pos - min_pos) / plot_thrust_max
            end_x            = pos_x_t + thrust_vec_t[0] * thrust_vec_scale
            end_y            = pos_y_t + thrust_vec_t[1] * thrust_vec_scale
        else: # assume min-fuel or min-energy is using thrust-acc and possibley thrust-acc limits:
            plot_thrust_acc_max  = max(thrust_acc_mag_t)
            thrust_acc_vec_scale = 0.2 * (max_pos - min_pos) / plot_thrust_acc_max
            end_x                = pos_x_t + thrust_acc_vec_t[0] * thrust_acc_vec_scale
            end_y                = pos_y_t + thrust_acc_vec_t[1] * thrust_acc_vec_scale

        # Find contiguous segments where thrust is active
        is_thrust_on = thrust_acc_mag_t > 1.0e-9

        # Find the start and end indices of each 'True' block
        padded = np.concatenate(([False], is_thrust_on, [False]))
        diffs  = np.diff(padded.astype(int))
        starts = np.where(diffs == +1)[0]
        stops  = np.where(diffs == -1)[0]

        # Loop through each segment and draw a separate polygon
        for idx, (start_idx, stop_idx) in enumerate(zip(starts, stops)):

            # Slice the data arrays to get just the points for this segment
            segment_pos_x = pos_x_t[start_idx:stop_idx]
            segment_pos_y = pos_y_t[start_idx:stop_idx]
            segment_end_x = end_x  [start_idx:stop_idx]
            segment_end_y = end_y  [start_idx:stop_idx]

            # Construct the polygon for this segment
            poly_x = np.concatenate([segment_pos_x, segment_end_x[::-1]])
            poly_y = np.concatenate([segment_pos_y, segment_end_y[::-1]])

            # Draw the polygon for the current segment
            ax1_pos.fill(
                poly_x                  ,
                poly_y                  , 
                facecolor = color_thrust,
                alpha     = 0.5         ,
                edgecolor = 'none'      ,
            )

        # Draw some thrust or thrust-acc vectors
        for idx in np.linspace(0,len(pos_x_t)-1,20).astype(int):
            ax1_pos.plot(
                [pos_x_t[idx], end_x[idx]],
                [pos_y_t[idx], end_y[idx]],
                color     = color_thrust  ,
                linewidth = 2.0           ,
            )
    _plot_thrust_on_position_space(
        ax1_pos               = ax1_pos              ,
        pos_x_t               = pos_x_t              ,
        pos_y_t               = pos_y_t              ,
        thrust_acc_vec_t      = thrust_acc_vec_t     ,
        thrust_acc_mag_t      = thrust_acc_mag_t     ,
        thrust_vec_t          = thrust_vec_t         ,
        thrust_mag_t          = thrust_mag_t         ,
        use_thrust_acc_limits = use_thrust_acc_limits,
        thrust_acc_min        = thrust_acc_min       ,
        thrust_acc_max        = thrust_acc_max       ,
        use_thrust_limits     = use_thrust_limits    ,
        thrust_min            = thrust_min           ,
        thrust_max            = thrust_max           ,
        color_thrust          = color_thrust         ,
    )
    ax1_pos.plot(pos_vec_o_mns[ 0], pos_vec_o_mns[ 1], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='>', markersize=20, markerfacecolor=mcolors.CSS4_COLORS['white'], markeredgecolor=mcolors.CSS4_COLORS['black']                                       )
    ax1_pos.plot(pos_vec_f_pls[ 0], pos_vec_f_pls[ 1], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='s', markersize=16, markerfacecolor=mcolors.CSS4_COLORS['white'], markeredgecolor=mcolors.CSS4_COLORS['black']                                       )
    ax1_pos.plot(      pos_x_t[ 0],       pos_y_t[ 0], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='>', markersize=10, markerfacecolor=mcolors.CSS4_COLORS['black'], markeredgecolor=mcolors.CSS4_COLORS['black'], linestyle='None', label='Start'      )
    ax1_pos.plot(      pos_x_t[-1],       pos_y_t[-1], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='s', markersize=10, markerfacecolor=mcolors.CSS4_COLORS['black'], markeredgecolor=mcolors.CSS4_COLORS['black'], linestyle='None', label='End'        )
    ax1_pos.plot(      pos_x_t    ,       pos_y_t    , color=mcolors.CSS4_COLORS['black'], linewidth=2.0,                                                                                                                                          label='Trajectory' )
    ax1_pos.set_xlabel('Position X [m]')
    ax1_pos.set_ylabel('Position Y [m]')
    ax1_pos.grid(True)
    ax1_pos.axis('equal')
    ax1_pos.legend(loc='upper left')

    # 2D velocity path: vel-x vs. vel-y
    ax1_vel = fig.add_axes(square_coords) # type: ignore
    ax1_vel.set_visible(False)
    def _plot_thrust_on_velocity_space(
            ax                    : maxes.Axes                             ,
            vel_x_t               : np.ndarray                             ,
            vel_y_t               : np.ndarray                             ,
            thrust_acc_vec_t      : np.ndarray                             ,
            thrust_acc_mag_t      : np.ndarray                             ,
            thrust_vec_t          : np.ndarray                             ,
            thrust_mag_t          : np.ndarray                             ,
            use_thrust_acc_limits : bool       = True                      ,
            thrust_acc_min        : np.float64 = np.float64(0.0e+0)        ,
            thrust_acc_max        : np.float64 = np.float64(1.0e+1)        ,
            use_thrust_limits     : bool       = False                     ,
            thrust_min            : np.float64 = np.float64(0.0e+0)        ,
            thrust_max            : np.float64 = np.float64(1.0e+1)        ,
            color_thrust                       = mcolors.CSS4_COLORS['red'],
        ):

        min_vel = min(min(vel_x_t), min(vel_y_t))
        max_vel = max(max(vel_x_t), max(vel_y_t))
        if use_thrust_limits:
            plot_thrust_max  = max(thrust_mag_t)
            thrust_vec_scale = 1.0 * (max_vel - min_vel) / plot_thrust_max
            end_x            = vel_x_t + thrust_vec_t[0] * thrust_vec_scale
            end_y            = vel_y_t + thrust_vec_t[1] * thrust_vec_scale
        else: # assume min-fuel or min-energy is using thrust-acc and possibley thrust-acc limits:
            plot_thrust_acc_max  = max(thrust_acc_mag_t)
            thrust_acc_vec_scale = 1.0 * (max_vel - min_vel) / plot_thrust_acc_max
            end_x                = vel_x_t + thrust_acc_vec_t[0] * thrust_acc_vec_scale
            end_y                = vel_y_t + thrust_acc_vec_t[1] * thrust_acc_vec_scale

        # Find contiguous segments where thrust is active
        is_thrust_on = thrust_acc_mag_t > 1.0e-9

        # Find the start and end indices of each 'True' block
        padded = np.concatenate(([False], is_thrust_on, [False]))
        diffs  = np.diff(padded.astype(int))
        starts = np.where(diffs == +1)[0]
        stops  = np.where(diffs == -1)[0]

        # Loop through each segment and draw a separate polygon
        for idx, (start_idx, stop_idx) in enumerate(zip(starts, stops)):

            # Slice the data arrays to get just the points for this segment
            segment_vel_x = vel_x_t[start_idx:stop_idx]
            segment_vel_y = vel_y_t[start_idx:stop_idx]
            segment_end_x = end_x  [start_idx:stop_idx]
            segment_end_y = end_y  [start_idx:stop_idx]

            # Construct the polygon for this segment
            poly_x = np.concatenate([segment_vel_x, segment_end_x[::-1]])
            poly_y = np.concatenate([segment_vel_y, segment_end_y[::-1]])

            # Draw the polygon for the current segment
            ax.fill(
                poly_x,
                poly_y, 
                facecolor = color_thrust,
                alpha     = 0.5,
                edgecolor = 'none',
            )

        # Draw some thrust or thrust-acc vectors
        for idx in np.linspace(0,len(vel_x_t)-1,20).astype(int):
            ax.plot(
                [vel_x_t[idx], end_x[idx]],
                [vel_y_t[idx], end_y[idx]],
                color     = color_thrust  ,
                linewidth = 2.0           ,
            )
    _plot_thrust_on_velocity_space(
        ax                    = ax1_vel              ,
        vel_x_t               = vel_x_t              ,
        vel_y_t               = vel_y_t              ,
        thrust_acc_vec_t      = thrust_acc_vec_t     ,
        thrust_acc_mag_t      = thrust_acc_mag_t     ,
        thrust_vec_t          = thrust_vec_t         ,
        thrust_mag_t          = thrust_mag_t         ,
        use_thrust_acc_limits = use_thrust_acc_limits,
        thrust_acc_min        = thrust_acc_min       ,
        thrust_acc_max        = thrust_acc_max       ,
        use_thrust_limits     = use_thrust_limits    ,
        thrust_min            = thrust_min           ,
        thrust_max            = thrust_max           ,
        color_thrust          = color_thrust         ,
    )
    ax1_vel.plot(vel_vec_o_mns[ 0], vel_vec_o_mns[ 1], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='>', markersize=20, markerfacecolor=mcolors.CSS4_COLORS['white'], markeredgecolor=mcolors.CSS4_COLORS['black']                                       )
    ax1_vel.plot(vel_vec_f_pls[ 0], vel_vec_f_pls[ 1], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='s', markersize=16, markerfacecolor=mcolors.CSS4_COLORS['white'], markeredgecolor=mcolors.CSS4_COLORS['black']                                       )
    ax1_vel.plot(      vel_x_t[ 0],       vel_y_t[ 0], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='>', markersize=10, markerfacecolor=mcolors.CSS4_COLORS['black'], markeredgecolor=mcolors.CSS4_COLORS['black'], linestyle='None', label='Start'      )
    ax1_vel.plot(      vel_x_t[-1],       vel_y_t[-1], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='s', markersize=10, markerfacecolor=mcolors.CSS4_COLORS['black'], markeredgecolor=mcolors.CSS4_COLORS['black'], linestyle='None', label='End'        )
    ax1_vel.plot(      vel_x_t    ,       vel_y_t    , color=mcolors.CSS4_COLORS['black'], linewidth=2.0,                                                                                                                                          label='Trajectory' )
    ax1_vel.set_xlabel("Velocity X [m/s]", labelpad=2)
    ax1_vel.set_ylabel("Velocity Y [m/s]", labelpad=10)
    ax1_vel.grid(True)
    ax1_vel.axis('equal')
    ax1_vel.legend(loc='upper left')
    
    # Create a button to swap pos and vel plots
    ax_button_posxposy_vs_velxvely = fig.add_axes([0.025, 0.026, 0.03, 0.05]) # [left, bottom, width, height] # type: ignore
    button_posxposy_vs_velxvely = Button(ax_button_posxposy_vs_velxvely, "Swap", color=mcolors.CSS4_COLORS['darkgrey'], hovercolor='0.975')
    def _swap_posxposy_vs_velxvely(event):
        # Switch visibility of the position and velocity axes
        if ax1_pos.get_visible():
            ax1_pos.set_visible(False)
            ax1_vel.set_visible(True)
        else:
            ax1_pos.set_visible(True)
            ax1_vel.set_visible(False)

        # Redraw the canvas to show the changes
        fig.canvas.draw_idle()
    button_posxposy_vs_velxvely.on_clicked(_swap_posxposy_vs_velxvely)
    fig._button_posxposy_vs_velxvely = button_posxposy_vs_velxvely # type: ignore

    # Optimal Control Objective vs. Time
    ax2 = fig.add_subplot(gs[0,1])
    ax2.plot(time_t, opt_ctrl_obj_t, color=mcolors.CSS4_COLORS['black'], linewidth=2.0, label='Mass')
    ax2.set_xticklabels([])
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True, useOffset=False)
    if min_type == 'fuel':
        obj_label_unit = '[m/s]'
    elif min_type == 'energyfuel':
        obj_label_unit = ''
    elif min_type == 'energy':
        obj_label_unit = '[m$^2$/$s^3$]'
    else:
        obj_label_unit = ''
    ax2.set_ylabel(f'Objective\n{obj_label_unit}')

    # Thrust-Acc or Thrust Profile
    ax3 = fig.add_subplot(gs[1,1])
    if use_thrust_limits:
        ax3.axhline(y=float(thrust_min), color=mcolors.CSS4_COLORS['black'], linestyle=':', linewidth=2.0, label=f'Thrust Min')
        ax3.axhline(y=float(thrust_max), color=mcolors.CSS4_COLORS['black'], linestyle=':', linewidth=2.0, label=f'Thrust Max')
    elif use_thrust_acc_limits:
        ax3.axhline(y=float(thrust_acc_min), color=mcolors.CSS4_COLORS['black'], linestyle=':', linewidth=2.0, label=f'Thrust Acc Min')
        ax3.axhline(y=float(thrust_acc_max), color=mcolors.CSS4_COLORS['black'], linestyle=':', linewidth=2.0, label=f'Thrust Acc Max')
    if use_thrust_limits:
        ax3.fill_between(
            time_t                             ,
            thrust_mag_t                       ,
            where        = (thrust_mag_t > 0.0), # type: ignore
            facecolor    = color_thrust        ,
            edgecolor    = 'none'              ,
            alpha        = 0.5                 ,
        )
    else: # assume use_thrust_acc_limits
        ax3.fill_between(
            time_t                                     ,
            thrust_acc_mag_t                           ,
            where            = (thrust_acc_mag_t > 0.0), # type: ignore
            facecolor        = color_thrust            ,
            edgecolor        = 'none'                  ,
            alpha            = 0.5                     ,
        )
    ax3.set_xticklabels([])
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True, useOffset=False)
    ax3.set_ylabel(label_thrust_name + '\n' + label_thrust_unit)
    if use_thrust_limits:
        plot_thrust_min = 0.0
        plot_thrust_max = max(thrust_mag_t)
        ax3.set_ylim(
            plot_thrust_min - (plot_thrust_max - plot_thrust_min) * 0.1,
            plot_thrust_max + (plot_thrust_max - plot_thrust_min) * 0.1,
        )
    else: # assume use_thrust_acc_limits
        plot_thrust_acc_min = 0.0
        plot_thrust_acc_max = max(thrust_acc_mag_t)
        ax3.set_ylim(
            plot_thrust_acc_min - (plot_thrust_acc_max - plot_thrust_acc_min) * 0.1,
            plot_thrust_acc_max + (plot_thrust_acc_max - plot_thrust_acc_min) * 0.1,
        )
        # plot_thrust_min = 0.0
        # plot_thrust_max = max(thrust_mag_t)
        # ax3.set_ylim(
        #     plot_thrust_min - (plot_thrust_max - plot_thrust_min) * 0.1,
        #     plot_thrust_max + (plot_thrust_max - plot_thrust_min) * 0.1,
        # )
    def _plot_thrust_vectors_on_thrust_vs_time(
            ax3                   : maxes.Axes        ,
            time_t                : np.ndarray        ,
            thrust_acc_mag_t      : np.ndarray        ,
            thrust_mag_t          : np.ndarray        ,
            use_thrust_acc_limits : bool       = False,
            use_thrust_limits     : bool       = False,
        ):

        if use_thrust_limits:
            end_t = thrust_mag_t
        elif use_thrust_acc_limits:
            end_t = thrust_acc_mag_t
        else: # assume min_type 'energy' with no thrust max or thrust-acc max
            end_t = thrust_acc_mag_t

        for idx in np.linspace(0,len(time_t)-1,20).astype(int):
            ax3.plot(
                [time_t[idx], time_t[idx]],
                [        0.0,  end_t[idx]], # type: ignore
                color     = color_thrust  ,
                linewidth = 2.0           ,
            )
    _plot_thrust_vectors_on_thrust_vs_time(
        ax3                   = ax3                  ,
        time_t                = time_t               ,
        thrust_acc_mag_t      = thrust_acc_mag_t     ,
        thrust_mag_t          = thrust_mag_t         ,
        use_thrust_acc_limits = use_thrust_acc_limits,
        use_thrust_limits     = use_thrust_limits    ,
    )

    # Mass vs. Time
    ax4 = fig.add_subplot(gs[2,1])
    ax4.plot(time_t, mass_t, color=mcolors.CSS4_COLORS['black'], linewidth=2.0, label='Mass')
    ax4.set_xticklabels([])
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True, useOffset=False)
    ax4.set_ylabel('Mass\n[kg]')

    # Position vs. Time
    ax5_timepos = fig.add_subplot(gs[3,1])
    ax5_timepos.plot(time_t[ 0], pos_vec_o_mns[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax5_timepos.plot(time_t[-1], pos_vec_f_pls[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 16, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax5_timepos.plot(time_t    ,       pos_x_t    , color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, label='X' )
    ax5_timepos.plot(time_t[ 0],       pos_x_t[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax5_timepos.plot(time_t[-1],       pos_x_t[-1], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax5_timepos.plot(time_t[ 0], pos_vec_o_mns[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax5_timepos.plot(time_t[-1], pos_vec_f_pls[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 16, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax5_timepos.plot(time_t    ,       pos_y_t    , color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, label='Y' )
    ax5_timepos.plot(time_t[ 0],       pos_y_t[ 0], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax5_timepos.plot(time_t[-1],       pos_y_t[-1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax5_timepos.set_xticklabels([])
    ax5_timepos.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True, useOffset=False)
    ax5_timepos.set_ylabel('Position\n[m]')
    ax5_timepos.legend()
    ax5_timepos.grid(True)
    min_ylim = min(min(pos_x_t), min(pos_y_t))
    max_ylim = max(max(pos_x_t), max(pos_y_t))
    ax5_timepos.set_ylim(
        min_ylim - (max_ylim - min_ylim) * 0.2,
        max_ylim + (max_ylim - min_ylim) * 0.2,
    )

    # Co-Position vs. Time
    ax5_timecopos = fig.add_subplot(gs[3,1])
    ax5_timecopos.set_visible(False)
    ax5_timecopos.plot(time_t[ 0], copos_vec_o_mns[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax5_timecopos.plot(time_t[-1], copos_vec_f_pls[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 16, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax5_timecopos.plot(time_t    ,       copos_x_t    , color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, label='X' )
    ax5_timecopos.plot(time_t[ 0],       copos_x_t[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax5_timecopos.plot(time_t[-1],       copos_x_t[-1], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax5_timecopos.plot(time_t[ 0], copos_vec_o_mns[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax5_timecopos.plot(time_t[-1], copos_vec_f_pls[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 16, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax5_timecopos.plot(time_t    ,       copos_y_t    , color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, label='Y' )
    ax5_timecopos.plot(time_t[ 0],       copos_y_t[ 0], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax5_timecopos.plot(time_t[-1],       copos_y_t[-1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax5_timecopos.set_xticklabels([])
    ax5_timecopos.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True, useOffset=False)
    units_label = '[1/s]' if min_type=='fuel' else '[m/s$^3$]'
    ax5_timecopos.set_ylabel('Co-Position\n'+units_label)
    ax5_timecopos.legend()
    ax5_timecopos.grid(True)
    min_ylim = min(min(copos_x_t), min(copos_y_t))
    max_ylim = max(max(copos_x_t), max(copos_y_t))
    ax5_timecopos.set_ylim(
        min_ylim - (max_ylim - min_ylim) * 0.2,
        max_ylim + (max_ylim - min_ylim) * 0.2,
    )

    # Create a button to swap pos and copos
    ax_button_timepos_vs_timecopos = fig.add_axes([0.960, 0.295, 0.03, 0.05]) # [left, bottom, width, height] # type: ignore
    button_timepos_vs_timecopos = Button(ax_button_timepos_vs_timecopos, "Swap", color=mcolors.CSS4_COLORS['darkgrey'], hovercolor='0.975')
    def _swap_timepos_vs_timecopos(event):
        # Switch visibility of the position and velocity axes
        if ax5_timepos.get_visible():
            ax5_timepos.set_visible(False)
            ax5_timecopos.set_visible(True)
        else:
            ax5_timepos.set_visible(True)
            ax5_timecopos.set_visible(False)

        # Redraw the canvas to show the changes
        fig.canvas.draw_idle()
    button_timepos_vs_timecopos.on_clicked(_swap_timepos_vs_timecopos)
    fig._button_timepos_vs_timecopos = button_timepos_vs_timecopos # type: ignore

    # Velocity vs. Time
    ax6_timevel = fig.add_subplot(gs[4,1])
    ax6_timevel.plot(time_t[ 0], vel_vec_o_mns[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax6_timevel.plot(time_t[-1], vel_vec_f_pls[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 16, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax6_timevel.plot(time_t    ,       vel_x_t    , color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, label='X' )
    ax6_timevel.plot(time_t[ 0],       vel_x_t[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax6_timevel.plot(time_t[-1],       vel_x_t[-1], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax6_timevel.plot(time_t[ 0], vel_vec_o_mns[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax6_timevel.plot(time_t[-1], vel_vec_f_pls[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 16, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax6_timevel.plot(time_t    ,       vel_y_t    , color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, label='Y' )
    ax6_timevel.plot(time_t[ 0],       vel_y_t[ 0], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax6_timevel.plot(time_t[-1],       vel_y_t[-1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax6_timevel.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True, useOffset=False)
    ax6_timevel.set_xlabel('Time [s]')
    ax6_timevel.set_ylabel('Velocity\n[m/s]')
    ax6_timevel.legend()
    ax6_timevel.grid(True)
    min_ylim = min(min(vel_x_t), min(vel_y_t))
    max_ylim = max(max(vel_x_t), max(vel_y_t))
    ax6_timevel.set_ylim(
        min_ylim - (max_ylim - min_ylim) * 0.2,
        max_ylim + (max_ylim - min_ylim) * 0.2,
    )

    # Co-Velocity vs. Time
    ax6_timecovel = fig.add_subplot(gs[4,1])
    ax6_timecovel.set_visible(False)
    ax6_timecovel.plot(time_t[ 0], covel_vec_o_mns[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax6_timecovel.plot(time_t[-1], covel_vec_f_pls[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 16, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax6_timecovel.plot(time_t    ,       covel_x_t    , color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, label='X' )
    ax6_timecovel.plot(time_t[ 0],       covel_x_t[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax6_timecovel.plot(time_t[-1],       covel_x_t[-1], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax6_timecovel.plot(time_t[ 0], covel_vec_o_mns[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax6_timecovel.plot(time_t[-1], covel_vec_f_pls[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 16, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax6_timecovel.plot(time_t    ,       covel_y_t    , color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, label='Y' )
    ax6_timecovel.plot(time_t[ 0],       covel_y_t[ 0], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax6_timecovel.plot(time_t[-1],       covel_y_t[-1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax6_timecovel.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True, useOffset=False)
    ax6_timecovel.set_xlabel('Time [s]')
    units_label = '[1]' if min_type=='fuel' else '[m/s$^2$]'
    ax6_timecovel.set_ylabel('Co-Velocity\n'+units_label)
    ax6_timecovel.legend()
    ax6_timecovel.grid(True)
    min_ylim = min(min(covel_x_t), min(covel_y_t))
    max_ylim = max(max(covel_x_t), max(covel_y_t))
    ax6_timecovel.set_ylim(
        min_ylim - (max_ylim - min_ylim) * 0.2,
        max_ylim + (max_ylim - min_ylim) * 0.2,
    )

    # Create a button to swap vel and covel
    ax_button_timevel_vs_timecovel = fig.add_axes([0.960, 0.145, 0.03, 0.05]) # [left, bottom, width, height] # type: ignore
    button_timevel_vs_timecovel = Button(ax_button_timevel_vs_timecovel, "Swap", color=mcolors.CSS4_COLORS['darkgrey'], hovercolor='0.975')
    def _swap_timevel_vs_timecovel(event):
        # Switch visibility of the position and velocity axes
        if ax6_timevel.get_visible():
            ax6_timevel.set_visible(False)
            ax6_timecovel.set_visible(True)
        else:
            ax6_timevel.set_visible(True)
            ax6_timecovel.set_visible(False)

        # Redraw the canvas to show the changes
        fig.canvas.draw_idle()
    button_timevel_vs_timecovel.on_clicked(_swap_timevel_vs_timecovel)
    fig._button_timevel_vs_timecovel = button_timevel_vs_timecovel # type: ignore

    # Configure figure
    fig.subplots_adjust(
        left   = 0.05,
        right  = 0.95,
        top    = 0.83,
        hspace = 0.25,
        wspace = 0.15,
    )
    fig.align_ylabels()

    if plot_save:
        fig_filename = f"{input_filepath.stem}_optimal_trajectory.png".lower()
        if "example" in fig_filename:
            filepath = output_folderpath / f"{input_filepath.stem}_optimal_trajectory.png"
        else:
            filepath = output_folderpath / f"example_{input_filepath.stem}_optimal_trajectory.png"
        fig.savefig(filepath)

    if plot_show:
        mplt.show()