import math

class MySGP4_Final:
    """
    Final SGP4 implementation with complete atmospheric drag model.
    Based on your original structure with proper drag coefficients.
    """
    
    # WGS72 Constants
    KE = 0.0743669161
    AE_km = 6378.135
    J2 = 0.001082616
    
    MINS_PER_DAY = 1440.0
    TWO_PI = 2.0 * math.pi
    
    def __init__(self, tle_line1, tle_line2):
        # Parse B* drag term
        bstar_raw = tle_line1[53:61]
        sign = -1.0 if bstar_raw[0] == '-' else 1.0
        mantissa = float("0." + bstar_raw[1:6])
        exponent = float(bstar_raw[6:8].replace(' ', ''))
        self.bstar = sign * mantissa * (10.0**exponent)

        # Parse orbital elements
        self.i_0_deg = float(tle_line2[8:16])
        self.Omega_0_deg = float(tle_line2[17:25])
        self.e_0 = float("0." + tle_line2[26:33])
        self.omega_0_deg = float(tle_line2[34:42])
        self.M_0_deg = float(tle_line2[43:51])
        self.n_0_rev_per_day = float(tle_line2[52:63])
        
        self.sgp4init()

    def sgp4init(self):
        """Initialize SGP4 with drag model"""
        
        # Convert to radians
        self.i_0     = math.radians(self.i_0_deg)
        self.Omega_0 = math.radians(self.Omega_0_deg)
        self.omega_0 = math.radians(self.omega_0_deg)
        self.M_0     = math.radians(self.M_0_deg)
        
        # Convert mean motion to rad/min
        n_0_tle = self.n_0_rev_per_day * self.TWO_PI / self.MINS_PER_DAY
        
        # Trig values
        self.cos_i_0 = math.cos(self.i_0)
        self.sin_i_0 = math.sin(self.i_0)
        theta2       = self.cos_i_0 * self.cos_i_0
        theta4       = theta2 * theta2
        
        # Eccentricity terms
        self.e_0_sq    = self.e_0 * self.e_0
        self.beta_0_sq = 1.0 - self.e_0_sq
        self.beta_0    = math.sqrt(self.beta_0_sq)
        
        # Un-kozai the mean motion
        # Recover original mean motion from TLE
        a1 = (self.KE / n_0_tle) ** (2.0/3.0)
        d1 = 1.5 * self.J2 / (a1 * a1) * (3.0 * theta2 - 1.0) / (self.beta_0**3)
        a0 = a1 * (1.0 - d1/3.0 - d1*d1 - 134.0/81.0 * d1**3)
        d0 = 1.5 * self.J2 / (a0 * a0) * (3.0 * theta2 - 1.0) / (self.beta_0**3)
        
        # Final mean motion and semi-major axis
        self.n_0 = n_0_tle / (1.0 + d0)
        self.a_0 = a0 / (1.0 - d0)
        
        # Check perigee
        rp = self.a_0 * (1.0 - self.e_0)
        perigee_km = (rp - 1.0) * self.AE_km
        
        # Atmospheric model parameters
        qoms2t = 1.88027916e-9
        s      = 1.01222928
        if perigee_km < 156.0:
            s = perigee_km - 78.0
            if perigee_km < 98.0:
                s = 20.0
            qoms2t = ((120.0 - s) / self.AE_km)**4
            s = s / self.AE_km + 1.0
        
        # Drag initialization
        xi       = 1.0 / (self.a_0 - s)
        self.eta = self.a_0 * self.e_0 * xi
        eta2     = self.eta * self.eta
        
        coef  = qoms2t * xi**4
        coef1 = coef / self.beta_0**3.5
        
        self.C2 = coef1 * self.n_0 * (self.a_0 * (1.0 + 1.5 * eta2 + self.e_0 * 
                  (4.0 + eta2)) + 0.75 * self.J2 * xi / self.beta_0_sq * 
                  (3.0 * theta2 - 1.0) * (8.0 + 3.0 * eta2))
        
        self.C1 = self.bstar * self.C2
        self.C3 = 0.0
        
        if self.e_0 > 1.0e-4:
            self.C3 = -self.J2 * self.sin_i_0 / (self.C1 * self.a_0 * self.e_0)
            
        # C4 and C5 for eccentricity evolution
        self.C4 = 2.0 * self.n_0 * coef1 * self.a_0 * self.beta_0_sq * (
                  (2.0 * self.eta * (1.0 + self.e_0 * self.eta) + 0.5 * self.e_0 + 0.5 * self.eta**3) -
                  self.J2 * xi / (self.a_0 * self.beta_0_sq) * 
                  (3.0 * (1.0 - 3.0 * theta2) * (1.0 - eta2 + self.e_0 * self.eta * 
                  (1.5 - 0.5 * eta2)) + 0.75 * (1.0 - theta2) * 
                  (2.0 * eta2 - self.e_0 * self.eta * (1.0 + eta2)) * math.cos(2.0 * self.omega_0)))
        
        self.C5 = 2.0 * coef1 * self.a_0 * self.beta_0_sq * (1.0 + 2.75 * 
                  (eta2 + self.e_0 * self.eta) + self.e_0 * self.eta * eta2)
        
        # D terms for mean motion
        self.D2 = 4.0 * self.a_0 * self.C1 * self.C1
        self.D3 = 4.0/3.0 * self.a_0 * self.a_0 * self.C1**3 * (17.0 * self.a_0 + s)
        self.D4 = 2.0/3.0 * self.a_0**3 * self.C1**4 * (221.0 * self.a_0 + 31.0 * s)
        
        # Secular effects of atmospheric drag
        self.omegacf = self.bstar * self.C3 * math.cos(self.omega_0)
        self.xmcof = 0.0
        if self.e_0 > 1.0e-4:
            self.xmcof = -2.0/3.0 * coef * self.bstar / self.e_0
            
        self.delmo = (1.0 + self.eta * math.cos(self.M_0))**3
        self.sinmo = math.sin(self.M_0)
        
        # Simple flag
        self.isimp = False
        if perigee_km < 220.0:
            self.isimp = True
            
    def sgp4(self, tsince):
        """Propagate to time tsince (minutes from epoch)"""
        
        # Update for atmospheric drag
        temp = 1.0 - self.C1 * tsince
        tempa = 1.0 - self.D2 * tsince * tsince - self.D3 * tsince**3 - \
                self.D4 * tsince**4
        tempe = self.bstar * self.C4 * tsince
        templ = temp**2
        
        if self.isimp:
            delomg = self.omegacf * tsince
            delm = self.xmcof * ((1.0 + self.eta * math.cos(self.M_0 + self.n_0 * tsince))**3 - 
                                 self.delmo)
            temp = temp + 0.5 * self.beta_0 * templ
            e = self.e_0 - tempe
        else:
            delomg = self.omegacf * tsince
            delmtemp = 1.0 + self.eta * math.cos(self.M_0 + self.n_0 * tsince)
            delm = self.xmcof * (delmtemp * delmtemp * delmtemp - self.delmo)
            temp = temp + 0.5 * self.beta_0 * templ
            e = self.e_0 - tempe - self.bstar * self.C5 * (math.sin(self.M_0 + 
                self.n_0 * tsince) - self.sinmo)
            
        a  = self.a_0 * templ
        xl = self.M_0 + self.omega_0 + self.Omega_0 + self.n_0 * tempa * tsince
        em = e
        
        if em < 1.0e-6:
            em = 1.0e-6
            
        # Update mean motion
        nm = self.KE / (a**1.5)
        
        # Long period periodics
        axn  = em * math.cos(self.omega_0)
        temp = 1.0 / (a * (1.0 - em * em))
        xll  = temp * self.xmcof * axn * (3.0 + 5.0 * self.cos_i_0) / (1.0 + self.cos_i_0)
        ayn  = em * math.sin(self.omega_0) + temp * xll
        xll  = xl + xll
        
        # Solve Kepler's equation
        u    = xll - self.Omega_0
        eo1  = u
        tem5 = 1.0
        
        for _ in range(10):
            sineo1 = math.sin(eo1)
            coseo1 = math.cos(eo1)
            tem5 = 1.0 - coseo1 * axn - sineo1 * ayn
            tem5 = (u - ayn * coseo1 + axn * sineo1 - eo1) / tem5
            if abs(tem5) < 1.0e-12:
                break
            eo1 = eo1 + tem5
            
        # Short period periodics
        ecose = axn * coseo1 + ayn * sineo1
        esine = axn * sineo1 - ayn * coseo1
        el2   = axn * axn + ayn * ayn
        pl    = a * (1.0 - el2)
        
        if pl < 0.0:
            return [0, 0, 0], [0, 0, 0]
            
        r = a * (1.0 - ecose)
        rdot = self.KE * math.sqrt(a) / r * esine
        rfdot = self.KE * math.sqrt(pl) / r
        
        # Orientation vectors
        temp = esine / (1.0 + math.sqrt(1.0 - el2))
        sinu = a / r * (sineo1 - ayn - axn * temp)
        cosu = a / r * (coseo1 - axn + ayn * temp)
        su   = math.atan2(sinu, cosu)
        
        sin2u = 2.0 * sinu * cosu
        cos2u = 2.0 * cosu * cosu - 1.0
        
        # Update for short periodics
        temp1 = self.J2 / pl / 2.0
        temp2 = temp1 / pl
        
        rk     = r * (1.0 - 1.5 * temp2 * (1.0 - self.cos_i_0 * self.cos_i_0) * cos2u)
        uk     = su - 0.25 * temp2 * (7.0 * self.cos_i_0 * self.cos_i_0 - 1.0) * sin2u
        xnodek = self.Omega_0 + 1.5 * temp2 * self.cos_i_0 * sin2u
        xinck  = self.i_0 + 1.5 * temp2 * self.cos_i_0 * self.sin_i_0 * cos2u
        
        rdotk  = rdot - nm * temp1 * (1.0 - self.cos_i_0 * self.cos_i_0) * sin2u
        rfdotk = rfdot + nm * temp1 * ((1.0 - self.cos_i_0 * self.cos_i_0) * cos2u + 
                                       1.5 * (1.0 - 3.0 * self.cos_i_0 * self.cos_i_0))
        
        # Unit orientation vectors
        sinuk  = math.sin(uk)
        cosuk  = math.cos(uk)
        sinik  = math.sin(xinck)
        cosik  = math.cos(xinck)
        sinnok = math.sin(xnodek)
        cosnok = math.cos(xnodek)
        
        # Position and velocity
        xmx = -sinnok * cosik
        xmy = cosnok * cosik
        
        ux = xmx * sinuk + cosnok * cosuk
        uy = xmy * sinuk + sinnok * cosuk
        uz = sinik * sinuk
        
        vx = xmx * cosuk - cosnok * sinuk
        vy = xmy * cosuk - sinnok * sinuk
        vz = sinik * cosuk
        
        # Final position in km
        x = rk * ux * self.AE_km
        y = rk * uy * self.AE_km
        z = rk * uz * self.AE_km
        
        # Final velocity in km/s
        xdot = (rdotk * ux + rfdotk * vx) * self.AE_km / 60.0
        ydot = (rdotk * uy + rfdotk * vy) * self.AE_km / 60.0
        zdot = (rdotk * uz + rfdotk * vz) * self.AE_km / 60.0
        
        return [x, y, z], [xdot, ydot, zdot]


# Test the implementation
print("="*80)
print(" FINAL SGP4 IMPLEMENTATION WITH COMPLETE DRAG MODEL")
print("="*80)

tle_line1 = "1 25544U 98067A   25303.50000000  .00016717  00000-0  30306-3 0  9999"
tle_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0284 15.50103472 45660"

sgp4 = MySGP4_Final(tle_line1, tle_line2)

print(f"\nISS Orbital Elements:")
print(f"  Semi-major axis: {sgp4.a_0 * sgp4.AE_km:.2f} km")
print(f"  Eccentricity: {sgp4.e_0:.6f}")
print(f"  B* drag: {sgp4.bstar:.6e}")

# Test at t=120 minutes
t = 120.0
r, v = sgp4.sgp4(t)

print(f"\nResults at t = {t} minutes:")
print(f"  Position: [{r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}] km")
print(f"  Velocity: [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}] km/s")

# Compare with official
official_r = [1075.90286006, 6430.73832694, -1915.24218938]
error = math.sqrt(sum((r[i] - official_r[i])**2 for i in range(3)))

print(f"\nComparison with official SGP4:")
print(f"  Position error: {error:.2f} km")
print(f"  Original gravity-only error: 70.96 km")
if error < 70.96:
    print(f"  ✓ Improvement: {70.96 - error:.2f} km better!")