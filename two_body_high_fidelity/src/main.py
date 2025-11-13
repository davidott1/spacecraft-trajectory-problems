import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.model.constants import CONVERTER
from two_body_high_fidelity.src.model.two_body import TwoBodyDynamics, PHYSICALCONSTANTS, EquationsOfMotion
from plot.trajectory import plot_3d_trajectories, plot_time_series, plot_3d_error, plot_time_series_error
from src.propagation import propagate_orbit
from initialization import initial_guess
from two_body_high_fidelity.src.propagation.tle_propagator import propagate_tle, get_tle_initial_state
from typing import Optional

def main():
    """
    Main function to set up and propagate a spacecraft orbit for one day.
    
    SGP4 vs SDP4 (Deep Space) Selection:
    ------------------------------------
    The SGP4 library automatically selects between SGP4 and SDP4 (deep space) 
    propagators based on the orbital period:
    
    - SGP4: Used for orbits with period < 225 minutes (orbital radius < ~6.6 R_E or ~42,164 km)
    - SDP4: Used for orbits with period >= 225 minutes (typically GEO and higher)
    
    For circular orbits:s
    - LEO  (   ~500 km): Period   ~95 min -> SGP4
    - MEO  (~20,000 km): Period  ~718 min -> SDP4  
    - GEO  (~35,786 km): Period ~1436 min -> SDP4
    
    The transition happens at approximately 6.6 Earth radii from Earth's center,
    which corresponds to an altitude of about 10,000-10,200 km.
    """
    #### INPUT ####

    # Time
    time_o = 0.0                                  # initial time [s]
    time_f = time_o + 1 * CONVERTER.SEC_PER_DAY   # final time [s]

    # Example TLEs:
    
    # LEO: NOAA-20 (~824 km altitude, sun-synchronous)
    tle_line1_leo = "1 43013U 17073A   24204.50704861  .00000201  00000+0  20499-4 0  9993"
    tle_line2_leo = "2 43013  98.7119 296.0139 0001458  83.3997 276.7258 14.19554887355539"
    
    # MEO: GPS satellite NAVSTAR 76 (~20,200 km altitude, Period ~718 min)
    tle_line1_meo = "1 41019U 15062A   24204.50000000 -.00000023  00000+0  00000+0 0  9992"
    tle_line2_meo = "2 41019  54.9887 201.2345 0004321  45.6789 314.4321  2.00564756 65432"
    
    # GEO: GOES-16 (~35,786 km altitude, geostationary, Period ~1436 min)
    tle_line1_geo = "1 41866U 16071A   24204.50000000 -.00000266  00000+0  00000+0 0  9999"
    tle_line2_geo = "2 41866   0.0392 267.8642 0000631 189.5432 313.2156  1.00271798 28956"
    
    # Select which TLE to use
    tle_selection = 'geo'  # Options: 'leo', 'meo', 'geo'
    
    if tle_selection == 'leo':
        tle_line1 = tle_line1_leo
        tle_line2 = tle_line2_leo
        enable_third_body = False  # Not significant for LEO
        print("\n>>> Using LEO TLE (NOAA-20, ~824 km) - SGP4 propagator <<<")
    elif tle_selection == 'meo':
        tle_line1 = tle_line1_meo
        tle_line2 = tle_line2_meo
        enable_third_body = True  # Important for MEO
        print("\n>>> Using MEO TLE (GPS NAVSTAR 76, ~20,200 km) - SDP4 propagator <<<")
    elif tle_selection == 'geo':
        tle_line1 = tle_line1_geo
        tle_line2 = tle_line2_geo
        enable_third_body = True  # Critical for GEO
        print("\n>>> Using GEO TLE (GOES-16, ~35,786 km) - SDP4 propagator <<<")
    else:
        raise ValueError(f"Invalid tle_selection: {tle_selection}")
    
    use_tle = True  # Set to True to use TLE initial conditions

    # Spacecraft properties
    cd   = 2.2          # drag coefficient [-]
    area = 10.0         # cross-sectional area [m^2]
    mass = 2294.0       # spacecraft mass [kg] (e.g., NOAA-20)
    
    disable_drag_sgp4 = False  # enable drag in SGP4 for comparison

    #### END INPUT ####

    # Spacecraft initial state
    if use_tle:
        # If modeling drag, derive parameters from TLE B* term for consistency
        if not disable_drag_sgp4:
            print("\nDeriving drag parameters from TLE B* term...")
            # B* is in characters 54-61 of TLE line 1. Format is -.XXXXX+X
            bstar_str = tle_line1[53:61]
            # The format is 'SXXXXXEY' where S is sign, XXXXX is mantissa, E is sign of exponent, Y is exponent
            # e.g., ' 12345-3' -> 0.12345 * 10^-3. The sign is often a space.
            # The given TLE has '00000-0', which means 0.0.
            # A more robust parsing is needed.
            sign         = 1 if bstar_str[0] in ' +' else -1
            base_str     = bstar_str[1:6]
            exponent_str = bstar_str[6:8].replace('-', '-').strip()
            
            base     = sign * float(f"0.{base_str}")
            exponent = int(exponent_str)
            
            bstar = base * (10**exponent) # B* drag term [1/earth_radii]
            
            # Convert B* to Cd*A/m. B* = (Cd*A/m) * rho_0 / 2, where rho_0 is a reference density.
            # The sgp4 library uses a reference density of 0.1570 kg / (m * R_E^2)
            # and R_E = 6378135.0 m. This gives rho_0 = 3.844e-12 kg/m^3.
            # and R_E = 6378135.0 m. This gives rho_0 = 3.844e-12 kg/m^3.
            rho_0 = 0.1570 / (6378135.0**2) # Reference density in kg/m^3
            earth_radii_to_m = 6378135.0
            
            cd_area_over_mass = 2 * bstar / earth_radii_to_m / rho_0 # m^2/kg
            
            # Set cd and area to 1.0 and calculate mass to match the ratio
            cd   = cd_area_over_mass
            area = 1.0
            mass = 1.0
            
            print(f"  B* = {bstar:.4e} 1/R_E")
            print(f"  Derived Cd*A/m = {cd_area_over_mass:.4e} m^2/kg")
            print(f"  Using Cd={cd}, A={area} m^2, m={mass:.2f} kg for high-fidelity model.")

        # Propagate TLE for 10 minutes to get a new initial state
        time_offset = 10 * 60.0  # seconds
        print(f"\nPropagating TLE for {time_offset/60.0} minutes to get new initial state...")
        
        state_at_offset = propagate_tle(
          tle_line1=tle_line1, tle_line2=tle_line2,
          time_o=time_offset, time_f=time_offset, num_points=1,
          disable_drag=disable_drag_sgp4, to_j2000=True
        )
        if not state_at_offset['success']:
            raise RuntimeError(f"Failed to get state at {time_offset}s: {state_at_offset['message']}")
        
        initial_state = state_at_offset['state'][:, 0]
        time_o = time_offset  # Update the start time for high-fidelity propagation
        
        print(f"New initial state obtained for t = {time_o}s.")
    else:
        igs = 'elliptical'                 # initial guess selection: circular elliptical
        alt = 500e3                        # altitude [m]
        ecc = 0.2                          # eccentricity [-]
        inc = 95.0 * CONVERTER.RAD_PER_DEG # inclination [rad]
        
        initial_state = initial_guess.get_initial_state(
            initial_guess_selection = igs,
            alt                     = alt,
            inc                     = inc,
            ecc                     = ecc,
        )

    # Set up dynamics model for Earth with perturbations
    two_body_dynamics = TwoBodyDynamics(
        gp                   = PHYSICALCONSTANTS.EARTH.GP,
        time_o               = time_o,
        j_2                  = PHYSICALCONSTANTS.EARTH.J_2,
        j_3                  = 0.0*PHYSICALCONSTANTS.EARTH.J_3,
        j_4                  = 0.0*PHYSICALCONSTANTS.EARTH.J_4,
        pos_ref              = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR,
        cd                   = cd,
        area                 = area,
        mass                 = mass,
        enable_third_body    = enable_third_body,
        third_body_use_spice = True,
        third_body_bodies    = ['SUN', 'MOON'],
    )
    
    # Propagate the orbit
    result = propagate_orbit(
        initial_state       = initial_state,
        time_o              = time_o,
        time_f              = time_f,
        dynamics            = two_body_dynamics,
        get_coe_time_series = True,
    )

    # Propagate TLE with SGP4
    if use_tle:
        print("\nPropagating TLE with SGP4...")
        result_tle = propagate_tle(
            tle_line1    = tle_line1,
            tle_line2    = tle_line2,
            time_o       = time_o,
            time_f       = time_f,
            num_points   = 1000,
            disable_drag = disable_drag_sgp4,
            to_j2000     = True,  # Transform TEME to J2000
        )
        
        if result_tle['success']:
            print(f"SGP4 propagation successful!")
        else:
            print(f"SGP4 propagation failed: {result_tle['message']}")

    # Display results
    if result['success']:
        print(f"\nHigh-fidelity propagation successful!")
        print(f"Status: {result['message']}")
        print(f"Number of time steps: {len(result['time'])}")
        
        # Create plots
        print("\nGenerating plots...")
        
        # High-fidelity plots
        fig1 = plot_3d_trajectories(result)
        fig1.suptitle('High-Fidelity Propagation', fontsize=16)
        
        fig2 = plot_time_series(result)
        fig2.suptitle('High-Fidelity Propagation - Time Series', fontsize=16)
        
        # TLE/SGP4 plots
        if use_tle and result_tle['success']:
            fig3 = plot_3d_trajectories(result_tle)
            fig3.suptitle('SGP4 Propagation', fontsize=16)
            
            fig4 = plot_time_series(result_tle)
            fig4.suptitle('SGP4 Propagation - Time Series', fontsize=16)
            
            # Error plots (SGP4 as reference)
            fig5 = plot_3d_error(result_tle, result, 
                                 title='Error: SGP4 vs High-Fidelity')
            
            fig6 = plot_time_series_error(result_tle, result,
                                          title='Time Series Error: SGP4 vs High-Fidelity')
        
        plt.show()
    else:
        print(f"\nPropagation failed!")
        print(f"Status: {result['message']}")
    
    return result


if __name__ == "__main__":
    main()