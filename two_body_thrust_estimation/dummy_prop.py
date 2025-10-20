from datetime import datetime, timedelta
import numpy as np
from sgp4.api import Satrec, jday
from dynamics import Dynamics
from sgp4_helper import SGP4Helper
import matplotlib.pyplot as plt
from pathlib import Path

# Original TLE (example ISS)
line1 = '1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005'
line2 = '2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.54225995 12345'
sat = Satrec.twoline2rv(line1, line2)

print("Dummy propagation: SGP4 -> rebuild TLE from state -> continue")

start_time = datetime(2024, 1, 1, 12, 0, 0)
total_minutes = 180    # total propagation span
retle_minute  = 60     # minute at which we rebuild a TLE
print(f"Total minutes: {total_minutes}")
print(f"Re-TLE at minute: {retle_minute}")

# Containers
time_min = []
sma_km   = []
ecc      = []
inc_deg  = []
retle_flag = []
# NEW: position containers (km)
pos_x = []
pos_y = []
pos_z = []

dynamics = Dynamics()

# Capture initial elements
init_el = SGP4Helper.get_orbital_elements(sat)
print("Initial elements:")
print(f"  SMA  (km): {init_el['sma']/1000.0:.3f}")
print(f"  ECC       : {init_el['ecc']:.6e}")
print(f"  INC  (deg): {np.rad2deg(init_el['inc']):.6f}")

retle_done = False
retle_el_before = None
retle_el_after  = None

for m in range(total_minutes + 1):
    current_time = start_time + timedelta(minutes=m)
    jd, fr = jday(current_time.year, current_time.month, current_time.day,
                  current_time.hour, current_time.minute, current_time.second)
    err, r_eci_km, v_eci_kms = sat.sgp4(jd, fr)
    if err != 0:
        print(f"SGP4 error code {err} at minute {m}")
        continue

    # Record current orbital elements from position/velocity (convert to meters)
    r_m = np.array(r_eci_km) * 1000.0
    v_m = np.array(v_eci_kms) * 1000.0
    coe = dynamics.rv2coe(r_m, v_m)
    time_min.append(m)
    sma_km.append(coe['sma']/1000.0)
    ecc.append(coe['ecc'])
    inc_deg.append(np.rad2deg(coe['inc']))
    retle_flag.append(0 if not retle_done else 1)

    # NEW: store ECI position (km)
    pos_x.append(r_eci_km[0])
    pos_y.append(r_eci_km[1])
    pos_z.append(r_eci_km[2])

    # Rebuild TLE at specified minute (once)
    if (m == retle_minute) and (not retle_done):
        print(f"\nRebuilding TLE at minute {m}...")
        retle_el_before = SGP4Helper.get_orbital_elements(sat)
        # Assume osculating == mean; directly create new Satrec
        sat = SGP4Helper.create_satrec_from_state(r_m, v_m, current_time, bstar=0.0)
        retle_el_after = SGP4Helper.get_orbital_elements(sat)
        print("Pre-rebuild elements:")
        print(f"  SMA (km): {retle_el_before['sma']/1000.0:.3f}")
        print(f"  ECC     : {retle_el_before['ecc']:.6e}")
        print(f"  INC (deg): {np.rad2deg(retle_el_before['inc']):.6f}")
        print("Post-rebuild elements:")
        print(f"  SMA (km): {retle_el_after['sma']/1000.0:.3f}")
        print(f"  ECC     : {retle_el_after['ecc']:.6e}")
        print(f"  INC (deg): {np.rad2deg(retle_el_after['inc']):.6f}\n")
        retle_done = True

# Convert lists
time_min  = np.array(time_min)
sma_km    = np.array(sma_km)
ecc       = np.array(ecc)
inc_deg   = np.array(inc_deg)
retle_flag= np.array(retle_flag)
# NEW: convert position arrays
pos_x = np.array(pos_x)
pos_y = np.array(pos_y)
pos_z = np.array(pos_z)

# REPLACED plotting section with 3D + elements
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(3, 2)

# 3D trajectory (left column spans rows)
ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax3d.plot(pos_x, pos_y, pos_z, 'k-', linewidth=1.2)
ax3d.scatter(pos_x[0], pos_y[0], pos_z[0], marker='^', c='g', s=70, label='Start')
ax3d.scatter(pos_x[-1], pos_y[-1], pos_z[-1], marker='s', c='k', s=60, label='End')
if retle_done and retle_minute < len(pos_x):
    ax3d.scatter(pos_x[retle_minute], pos_y[retle_minute], pos_z[retle_minute],
                 marker='*', c='r', s=140, label='Re-TLE')
ax3d.set_xlabel('X (km)')
ax3d.set_ylabel('Y (km)')
ax3d.set_zlabel('Z (km)')
ax3d.set_title('ECI Trajectory')
ax3d.grid(True, alpha=0.3)
ax3d.legend(loc='best')

# Orbital elements (right column)
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(time_min, sma_km, 'k-')
ax1.set_ylabel('SMA (km)')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
ax2.plot(time_min, ecc, 'k-')
ax2.set_ylabel('Ecc')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
ax3.plot(time_min, inc_deg, 'k-')
ax3.set_ylabel('Inc (deg)')
ax3.set_xlabel('Time (min)')
ax3.grid(True, alpha=0.3)

if retle_done:
    for a in (ax1, ax2, ax3):
        a.axvline(retle_minute, color='r', linestyle='--', alpha=0.6)

plt.tight_layout()
out = Path.cwd() / "dummy_prop.png"
plt.savefig(out, dpi=150)
print(f"Done. Points: {len(time_min)}. Plot: {out}")
plt.show()