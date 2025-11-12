class PropagatorValidator:
    """
    Comprehensive validation suite for orbital propagator
    """
    
    def __init__(self, propagator):
        self.propagator = propagator
        
    def test_energy_conservation(self, initial_state, time_span):
        """Conservative force energy drift < 1e-10"""
        
    def test_j2_secular_rates(self, initial_coe):
        """Compare RAAN/argp rates with analytical J2 theory"""
        
    def test_gto_transfer(self):
        """GTO to GEO circularization - known ΔV budget"""
        
    def test_moon_gravity_pull(self):
        """High lunar orbit - validate 3rd body effects"""
        
    def compare_with_sp3(self, sp3_file, satellite_prn):
        """Compare with GPS precise ephemeris (best accuracy)"""
