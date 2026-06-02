import numpy as np
from scipy.integrate import solve_ivp

# Physical Constants and Conversions to Geometric Units (G = c = M_sun = 1)
# Length unit: G * M_sun / c^2 = 1.476625061 km
# Mass unit: M_sun = 1.9884e33 g
# Density unit: M_sun / (Length unit)^3 = 6.17718e17 g/cm^3
# Pressure unit: Density unit * c^2 = 5.55169e38 dyn/cm^2

CGS_DENSITY_TO_GEO = 1.0 / 6.177180424564887e17
CGS_PRESSURE_TO_GEO = 1.0 / 5.551689230559092e38
GEO_RADIUS_TO_KM = 1.476625061
GEO_MASS_TO_MSUN = 1.0

class PiecewisePolytropicEOS:
    """
    Thermodynamically consistent Equation of State (EOS) combining a crust with 
    a 3-region piecewise polytropic core. Integrates analytical derivatives 
    to guarantee a noise-free speed of sound.
    """
    def __init__(self, log10_P1_cgs, gamma1, gamma2, gamma3):
        # Physical boundary rest-mass densities (cgs)
        # crust-core transition: rho_0 = 2.7e14 g/cm^3
        # core transition 1: rho_1 = 10^14.7 g/cm^3
        # core transition 2: rho_2 = 10^15.0 g/cm^3
        self.rho_cgs = np.array([2.7e14, 10**14.7, 10**15.0])
        self.rho_geo = self.rho_cgs * CGS_DENSITY_TO_GEO
        
        # Crust polytropic index
        self.gamma0 = 1.3569
        self.gammas = np.array([self.gamma0, gamma1, gamma2, gamma3])
        
        # Pressure P1 in geometric units (pressure at transition density rho_1)
        self.P1_geo = (10**log10_P1_cgs) * CGS_PRESSURE_TO_GEO
        
        # Determine polytropic pressure constants K_i
        # Zone 1: P = K1 * rho^gamma1 => K1 = P1_geo / rho_1^gamma1
        self.Ks = np.zeros(4)
        self.Ks[1] = self.P1_geo / (self.rho_geo[1] ** self.gammas[1])
        
        # Zone 2: P = K2 * rho^gamma2 => continuity at rho_1
        self.Ks[2] = self.Ks[1] * (self.rho_geo[1] ** (self.gammas[1] - self.gammas[2]))
        
        # Zone 3: P = K3 * rho^gamma3 => continuity at rho_2
        self.Ks[3] = self.Ks[2] * (self.rho_geo[2] ** (self.gammas[2] - self.gammas[3]))
        
        # Zone 0 (crust): P = K0 * rho^gamma0 => continuity at rho_0
        self.Ks[0] = self.Ks[1] * (self.rho_geo[0] ** (self.gammas[1] - self.gammas[0]))
        
        # Determine thermodynamic integration constants a_i for energy density epsilon
        # epsilon_i = (1 + a_i)*rho + K_i / (gamma_i - 1) * rho^gamma_i
        self.as_ = np.zeros(4)
        self.as_[0] = 0.0  # Normalized crust constant
        
        # Recursive calculation of core constants via boundary continuity
        for i in range(1, 4):
            rho_bnd = self.rho_geo[i-1]
            term_prev = (self.Ks[i-1] / (self.gammas[i-1] - 1.0)) * (rho_bnd ** (self.gammas[i-1] - 1.0))
            term_curr = (self.Ks[i] / (self.gammas[i] - 1.0)) * (rho_bnd ** (self.gammas[i] - 1.0))
            self.as_[i] = self.as_[i-1] + term_prev - term_curr
            
        # Determine the pressure boundaries corresponding to the density boundaries
        self.P_bnd = np.zeros(3)
        self.P_bnd[0] = self.Ks[0] * (self.rho_geo[0] ** self.gammas[0]) # Crust-Core
        self.P_bnd[1] = self.Ks[1] * (self.rho_geo[1] ** self.gammas[1]) # Core 1-2 (which equals P1_geo)
        self.P_bnd[2] = self.Ks[2] * (self.rho_geo[2] ** self.gammas[2]) # Core 2-3

    def _get_zone(self, p):
        """Helper to determine the EOS zone for a given pressure in geometric units."""
        if p < self.P_bnd[0]:
            return 0
        elif p < self.P_bnd[1]:
            return 1
        elif p < self.P_bnd[2]:
            return 2
        else:
            return 3

    def density_from_pressure(self, p):
        """Rest-mass density rho in geometric units given pressure p in geometric units."""
        if p <= 0.0:
            return 0.0
        zone = self._get_zone(p)
        return (p / self.Ks[zone]) ** (1.0 / self.gammas[zone])

    def energy_density_from_pressure(self, p):
        """Energy density epsilon in geometric units given pressure p in geometric units."""
        if p <= 0.0:
            return 0.0
        zone = self._get_zone(p)
        rho = (p / self.Ks[zone]) ** (1.0 / self.gammas[zone])
        return (1.0 + self.as_[zone]) * rho + p / (self.gammas[zone] - 1.0)

    def sound_speed_sq_from_pressure(self, p):
        """Analytical thermodynamically consistent sound speed squared c_s^2 = dp/de."""
        if p <= 0.0:
            return 0.0
        zone = self._get_zone(p)
        eps = self.energy_density_from_pressure(p)
        gamma = self.gammas[zone]
        # cs^2 = gamma * P / (epsilon + P)
        cs2 = gamma * p / (eps + p)
        return cs2

def tov_equations(r, U, eos):
    """
    Coupled general relativistic TOV structure and Regge-Wheeler l=2 metric perturbation equations.
    State vector: U = [m, p, y]
    """
    m, p, y = U
    
    # Boundary / exterior check
    if p <= 0.0:
        return [0.0, 0.0, 0.0]
        
    eps = eos.energy_density_from_pressure(p)
    cs2 = eos.sound_speed_sq_from_pressure(p)
    
    # 1. TOV mass equation
    dmdr = 4.0 * np.pi * r**2 * eps
    
    # Singularity safety at center
    if r == 0.0:
        return [0.0, 0.0, 0.0]
        
    # 2. TOV pressure equation
    denom = r * (r - 2.0 * m)
    if denom <= 0.0:
        # Physical singularity (black hole limit reached, though shouldn't happen in stable stars)
        return [0.0, 0.0, 0.0]
        
    dpdr = - (eps + p) * (m + 4.0 * np.pi * r**3 * p) / denom
    
    # 3. Tidal perturbation y(r) equation
    # F(r) coefficient
    F = (1.0 - 4.0 * np.pi * r**2 * (eps - p)) / (1.0 - 2.0 * m / r)
    
    # Q(r) coefficient
    if cs2 <= 0.0:
        # Treat sound speed boundary physically
        term_cs2 = 0.0
    else:
        term_cs2 = (eps + p) / cs2
        
    term_Q1 = (4.0 * np.pi * r**2 * (5.0 * eps + 9.0 * p + term_cs2)) / (1.0 - 2.0 * m / r)
    term_Q2 = 6.0 / (1.0 - 2.0 * m / r)
    term_Q3 = 4.0 * ((m + 4.0 * np.pi * r**3 * p) / (r * (1.0 - 2.0 * m / r)))**2
    
    r2Q = term_Q1 - term_Q2 - term_Q3
    
    dydr = - (y**2 + y * F + r2Q) / r
    
    return [dmdr, dpdr, dydr]

def solve_star(eos, central_pressure_cgs, rtol=1e-8, atol=1e-8):
    """
    Solves the stellar structure equations for a given central pressure and EOS.
    Returns stellar mass M (M_sun), radius R (km), Love number k2, and tidal deformability Lambda.
    """
    P_c = central_pressure_cgs * CGS_PRESSURE_TO_GEO
    eps_c = eos.energy_density_from_pressure(P_c)
    
    # Start integration at small non-zero radius to avoid singularity at r=0
    r_start = 1e-5 / GEO_RADIUS_TO_KM  # 10^-5 km converted to geometric units
    
    # Analytical Taylor expansions for boundary initialization
    m_start = (4.0 / 3.0) * np.pi * r_start**3 * eps_c
    p_start = P_c - 2.0 * np.pi * r_start**2 * (eps_c + P_c) * (P_c + eps_c / 3.0)
    y_start = 2.0
    
    U_start = [m_start, p_start, y_start]
    
    # Define termination event at surface (p = 0)
    P_min = 1e-15 * P_c
    def surface_reached(r, U, eos):
        return U[1] - P_min
    surface_reached.terminal = True
    
    # Max integration radius (50 km in geometric units)
    r_max = 50.0 / GEO_RADIUS_TO_KM
    
    # Solve system using adaptive step Scipy solver
    sol = solve_ivp(
        fun=tov_equations,
        t_span=[r_start, r_max],
        y0=U_start,
        args=(eos,),
        events=surface_reached,
        rtol=rtol,
        atol=atol,
        method='RK45'
    )
    
    # Check for successful integration
    if sol.status == -1 or len(sol.t) < 2:
        return None
        
    # Interpolate exact surface properties to prevent boundary step bias
    # Last step before crossing P_min
    r_prev = sol.t[-2]
    m_prev, p_prev, y_prev = sol.y[:, -2]
    
    # Final step
    r_final = sol.t[-1]
    m_final, p_final, y_final = sol.y[:, -1]
    
    # Linear interpolation factor
    frac = p_prev / (p_prev - p_final + 1e-30)
    frac = np.clip(frac, 0.0, 1.0)
    
    R_geo = r_prev + frac * (r_final - r_prev)
    M_geo = m_prev + frac * (m_final - m_prev)
    y_R = y_prev + frac * (y_final - y_prev)
    
    # Converts physical observables
    R_km = R_geo * GEO_RADIUS_TO_KM
    M_msun = M_geo * GEO_MASS_TO_MSUN
    
    # Compute compactness C
    C = M_geo / R_geo
    
    # Calculate Love number k2 and tidal deformability Lambda
    # Standard Hinderer expression
    ln_1_2C = np.log(1.0 - 2.0 * C + 1e-30)
    
    num_term = 2.0 + 2.0 * C * (y_R - 1.0) - y_R
    denom_term_1 = 2.0 * C * (6.0 - 3.0 * y_R + 3.0 * C * (5.0 * y_R - 8.0))
    denom_term_2 = 4.0 * C**3 * (13.0 - 11.0 * y_R + C * (3.0 * y_R - 2.0) + 2.0 * C**2 * (1.0 + y_R))
    denom_term_3 = 3.0 * (1.0 - 2.0 * C)**2 * (2.0 - y_R + 2.0 * C * (y_R - 1.0)) * ln_1_2C
    
    denom = denom_term_1 + denom_term_2 + denom_term_3
    
    if abs(denom) < 1e-15:
        k2 = 0.0
        Lambda = 0.0
    else:
        k2 = (8.0 / 5.0) * (C**5) * ((1.0 - 2.0 * C)**2) * num_term / denom
        Lambda = (2.0 / 3.0) * k2 * (C**-5)
        
    return {
        "M": M_msun,
        "R": R_km,
        "k2": k2,
        "Lambda": Lambda,
        "C": C
    }

if __name__ == "__main__":
    # Quick physics check when run directly
    print("Testing TOV Solver with SLy-like piecewise polytrope...")
    # Standard SLy parameterization core values (Read et al. 2009)
    # log10_P1 = 34.384, gamma1 = 3.005, gamma2 = 2.988, gamma3 = 2.851
    sly_eos = PiecewisePolytropicEOS(log10_P1_cgs=34.384, gamma1=3.005, gamma2=2.988, gamma3=2.851)
    
    # Try to find a ~1.4 M_sun star by sweeping central pressures
    p_center_sweep = np.logspace(34.2, 35.5, 30)
    best_star = None
    min_diff = 999.0
    
    for p_c in p_center_sweep:
        res = solve_star(sly_eos, p_c)
        if res is not None:
            diff = abs(res["M"] - 1.4)
            if diff < min_diff:
                min_diff = diff
                best_star = res
                
    if best_star is not None:
        print(f"Successfully solved standard SLy EOS star close to 1.4 M_sun:")
        print(f"  Mass:  {best_star['M']:.4f} M_sun (Target: 1.4)")
        print(f"  Radius: {best_star['R']:.4f} km (Expected: ~11.5 - 12.0 km)")
        print(f"  Lambda: {best_star['Lambda']:.1f} (Expected: ~300 - 450)")
        print(f"  Love k2: {best_star['k2']:.4f}")
        print(f"  Compactness: {best_star['C']:.4f}")
    else:
        print("Failed to solve any stars.")
