"""
flight_math.py - Core Flight Dynamics & Performance Engine
Formulas and notation based on J.D. Anderson, Introduction to Flight.
All inputs and outputs are in SI units (m, s, kg, N, W, rad, Pa, K).
"""

import numpy as np
from scipy.optimize import minimize_scalar

# -----------------------------------------------------------------------------
# 1. ATMOSPHERIC MODEL (ISA with optional Indian/hot-day temperature offset)
# -----------------------------------------------------------------------------
def isa_atmosphere(h_m, delta_T=0.0):
    """
    Computes ambient temperature, pressure, and density at geometric altitude h (m).
    Anderson Section 3.3.
    delta_T: Temperature deviation from standard ISA in Kelvin/Celsius.
    """
    # Sea-level standard constants
    T0 = 288.16      # Sea level temperature (K)
    p0 = 101325.0    # Sea level pressure (Pa)
    R = 287.058      # Specific gas constant for air (J/(kg*K))
    g0 = 9.80665     # Acceleration due to gravity (m/s^2)
    L = 0.0065       # Standard temperature lapse rate (K/m) in troposphere
    h_trop = 11000.0 # Troposphere boundary (m)

    h = min(h_m, 20000.0) # Valid through lower stratosphere

    if h <= h_trop:
        T_isa = T0 - L * h
        p = p0 * (1.0 - (L * h) / T0) ** (g0 / (R * L))
    else:
        T_trop = T0 - L * h_trop
        p_trop = p0 * (1.0 - (L * h_trop) / T0) ** (g0 / (R * L))
        T_isa = T_trop
        p = p_trop * np.exp(-g0 * (h - h_trop) / (R * T_trop))

    # Apply India / Hot Day temperature offset: T_actual = T_isa + delta_T
    T_actual = T_isa + delta_T
    rho = p / (R * T_actual)

    return T_actual, p, rho


# -----------------------------------------------------------------------------
# 2. FUNDAMENTAL SPEEDS & DRAG POLAR (Anderson Chapter 5 & 6)
# -----------------------------------------------------------------------------
def get_k(AR, e):
    """Induced drag factor: k = 1 / (pi * e * AR)"""
    return 1.0 / (np.pi * e * AR)

def stall_speed(W, S, rho, CL_max):
    """V_stall = sqrt( 2*W / (rho * S * CL_max) )"""
    return np.sqrt((2.0 * W) / (rho * S * CL_max))

def speed_L_over_D_max(W, S, rho, CD0, k):
    """
    Velocity for (L/D)_max: Minimum thrust required condition.
    CL_md = sqrt(CD0 / k)
    V_md = sqrt( (2*W) / (rho * S) * sqrt(k / CD0) )
    """
    CL_md = np.sqrt(CD0 / k)
    V_md = np.sqrt((2.0 * W) / (rho * S * CL_md))
    return V_md, CL_md

def speed_min_power(W, S, rho, CD0, k):
    """
    Velocity for minimum power required (max endurance for prop):
    CL_mp = sqrt(3 * CD0 / k) = sqrt(3) * CL_md
    V_mp = V_md / (3^(1/4)) ~= 0.7598 * V_md
    """
    CL_mp = np.sqrt(3.0 * CD0 / k)
    V_mp = np.sqrt((2.0 * W) / (rho * S * CL_mp))
    return V_mp, CL_mp


# -----------------------------------------------------------------------------
# 3. THRUST AND POWER REQUIRED
# -----------------------------------------------------------------------------
def thrust_required(V, W, S, rho, CD0, k):
    """
    T_R = q * S * CD0 + (k * W^2) / (q * S)
    where q = 0.5 * rho * V^2
    """
    q = 0.5 * rho * (V ** 2)
    T_R = q * S * CD0 + (k * (W ** 2)) / (q * S)
    return T_R

def power_required(V, W, S, rho, CD0, k):
    """P_R = T_R * V"""
    return thrust_required(V, W, S, rho, CD0, k) * V


# -----------------------------------------------------------------------------
# 4. CLIMB & CEILING PERFORMANCE (Anderson Section 6.7)
# -----------------------------------------------------------------------------
def rate_of_climb(V, W, S, rho, CD0, k, prop_type="Jet", T_avail=0.0, P_avail=0.0):
    """
    R/C = (Power Available - Power Required) / W
    For Jet: P_avail = T_avail * V
    For Prop: P_avail is directly supplied
    """
    PR = power_required(V, W, S, rho, CD0, k)
    if prop_type == "Jet":
        PA = T_avail * V
    else:
        PA = P_avail

    excess_power = PA - PR
    RC = excess_power / W
    return RC

def find_ceilings(W, S, AR, e, CD0, prop_type="Jet", T_SL=0.0, P_SL=0.0, delta_T=0.0):
    """
    Computes Absolute Ceiling (RC_max = 0 m/s) and Service Ceiling (RC_max = 0.5 m/s or 100 ft/min).
    Standard atmospheric density variation with altitude: T(h) = T_SL * (rho/rho0)^1.0 for jet,
    P(h) = P_SL * (rho/rho0)^1.0 for un-turbocharged prop.
    """
    k = get_k(AR, e)
    _, _, rho0 = isa_atmosphere(0.0, delta_T)

    altitudes = np.linspace(0, 16000, 161)
    max_rc_list = []

    for h in altitudes:
        _, _, rho = isa_atmosphere(h, delta_T)
        density_ratio = rho / rho0

        if prop_type == "Jet":
            T_avail_h = T_SL * density_ratio
            P_avail_h = 0.0
            # Jet best climb occurs near (L/D)_max speed
            V_test = np.linspace(speed_min_power(W, S, rho, CD0, k)[0], 
                                 speed_L_over_D_max(W, S, rho, CD0, k)[0] * 1.5, 40)
        else:
            P_avail_h = P_SL * density_ratio
            T_avail_h = 0.0
            # Prop best climb occurs near min power speed
            V_test = np.linspace(speed_min_power(W, S, rho, CD0, k)[0] * 0.8, 
                                 speed_L_over_D_max(W, S, rho, CD0, k)[0] * 1.3, 40)

        rc_vals = [rate_of_climb(v, W, S, rho, CD0, k, prop_type, T_avail_h, P_avail_h) for v in V_test]
        max_rc_list.append(max(rc_vals))

    max_rc_list = np.array(max_rc_list)

    # Find service ceiling (RC = 0.5 m/s ~= 100 ft/min)
    service_ceiling = np.interp(0.5, max_rc_list[::-1], altitudes[::-1])
    # Find absolute ceiling (RC = 0 m/s)
    abs_ceiling = np.interp(0.0, max_rc_list[::-1], altitudes[::-1])

    return max(0.0, float(service_ceiling)), max(0.0, float(abs_ceiling))


# -----------------------------------------------------------------------------
# 5. TURNING FLIGHT / MANEUVERS (Anderson Section 6.13)
# -----------------------------------------------------------------------------
def turning_performance(V, W, S, rho, CD0, k, n_load_factor, CL_max):
    """
    Level coordinated turn:
    n = 1 / cos(phi)  ==>  phi = arccos(1/n)
    Turn radius: R = V^2 / (g * sqrt(n^2 - 1))
    Turn rate: omega = (g * sqrt(n^2 - 1)) / V
    """
    g = 9.80665

    # Check if load factor is physically possible without stalling at this speed:
    CL_turn = (2.0 * W * n_load_factor) / (rho * (V ** 2) * S)
    is_stalled = CL_turn > CL_max

    if n_load_factor <= 1.0:
        return 0.0, np.inf, 0.0, is_stalled

    bank_angle_deg = np.degrees(np.arccos(1.0 / n_load_factor))
    turn_radius = (V ** 2) / (g * np.sqrt(n_load_factor ** 2 - 1.0))
    turn_rate_deg_s = np.degrees((g * np.sqrt(n_load_factor ** 2 - 1.0)) / V)

    return bank_angle_deg, turn_radius, turn_rate_deg_s, is_stalled


# -----------------------------------------------------------------------------
# 6. OPTIMIZATION ROUTINES (Phase 1.4 Objectives)
# -----------------------------------------------------------------------------
def optimize_range_and_endurance_speeds(W, S, rho, CD0, k, prop_type="Jet"):
    """
    Calculates analytical optimal velocities for maximum range and endurance.
    From Anderson Chapter 6 Breguet relations:
      - Propeller:
          Max Range -> (L/D)_max:  CL = sqrt(CD0/k)
          Max Endurance -> (CL^(3/2)/CD)_max: CL = sqrt(3*CD0/k)
      - Jet:
          Max Range -> (CL^(1/2)/CD)_max:  CL = sqrt(CD0 / (3*k))
          Max Endurance -> (L/D)_max: CL = sqrt(CD0/k)
    """
    V_L_over_D_max, CL_L_over_D_max = speed_L_over_D_max(W, S, rho, CD0, k)
    V_min_P, CL_min_P = speed_min_power(W, S, rho, CD0, k)

    if prop_type == "Propeller":
        V_opt_range = V_L_over_D_max
        CL_opt_range = CL_L_over_D_max
        V_opt_endurance = V_min_P
        CL_opt_endurance = CL_min_P
    else: # Jet
        CL_opt_range = np.sqrt(CD0 / (3.0 * k))
        V_opt_range = np.sqrt((2.0 * W) / (rho * S * CL_opt_range))
        V_opt_endurance = V_L_over_D_max
        CL_opt_endurance = CL_L_over_D_max

    return {
        "V_opt_range": float(V_opt_range),
        "CL_opt_range": float(CL_opt_range),
        "V_opt_endurance": float(V_opt_endurance),
        "CL_opt_endurance": float(CL_opt_endurance),
        "V_LD_max": float(V_L_over_D_max),
        "V_min_power": float(V_min_P)
    }

def optimize_wing_loading(target_V_stall, rho, CL_max):
    """
    Finds the optimal wing loading (W/S) to guarantee a maximum stall speed requirement:
    W/S = 0.5 * rho * (V_stall_target^2) * CL_max
    """
    optimal_WS = 0.5 * rho * (target_V_stall ** 2) * CL_max
    return float(optimal_WS)


# -----------------------------------------------------------------------------
# 7. V-n DIAGRAM (FLIGHT ENVELOPE)
# -----------------------------------------------------------------------------
def vn_diagram_boundaries(W, S, rho, CL_max, n_pos_limit=3.8, n_neg_limit=-1.52, V_max_level=100.0):
    """
    Computes aerodynamic lift limit curves and maneuver speed V_A.
    Anderson Section 6.13 / FAR Part 23 envelope.
    """
    # Corner speed / Maneuvering speed (V_A) where n_stall = n_pos_limit
    V_A = np.sqrt((2.0 * W * n_pos_limit) / (rho * S * CL_max))
    V_dive = 1.25 * V_max_level

    # Positive aerodynamic stall limit: n = 0.5 * rho * V^2 * S * CL_max / W
    V_pos_curve = np.linspace(0.0, V_A, 100)
    n_pos_curve = (0.5 * rho * (V_pos_curve ** 2) * S * CL_max) / W

    # Negative aerodynamic stall limit (assuming CL_min ~= -0.6 * CL_max)
    CL_min = -0.6 * CL_max
    V_neg_limit_speed = np.sqrt((2.0 * W * abs(n_neg_limit)) / (rho * S * abs(CL_min)))
    V_neg_curve = np.linspace(0.0, V_neg_limit_speed, 100)
    n_neg_curve = (0.5 * rho * (V_neg_curve ** 2) * S * CL_min) / W

    return {
        "V_A": float(V_A),
        "V_dive": float(V_dive),
        "V_pos_curve": V_pos_curve,
        "n_pos_curve": n_pos_curve,
        "V_neg_curve": V_neg_curve,
        "n_neg_curve": n_neg_curve,
        "V_neg_limit_speed": float(V_neg_limit_speed)
    }

# -----------------------------------------------------------------------------
# 8. EXPANDED MANEUVER DYNAMICS (CLIMB, DESCENT & GLIDE) - Anderson Sec 6.7 & 6.9
# -----------------------------------------------------------------------------
def climb_performance(V, W, S, rho, CD0, k, prop_type="Jet", T_avail=0.0, P_avail=0.0):
    """
    Computes steady climb rate (R/C) and climb angle gamma.
    sin(gamma) = (T - D)/W = (P_A - P_R) / (W * V)
    R/C = V * sin(gamma)
    """
    TR = thrust_required(V, W, S, rho, CD0, k)
    PR = power_required(V, W, S, rho, CD0, k)

    if prop_type == "Jet":
        PA = T_avail * V
        excess_thrust = T_avail - TR
        sin_gamma = excess_thrust / W
    else:
        PA = P_avail
        excess_power = PA - PR
        sin_gamma = excess_power / (W * V)

    # Clamp sin_gamma to physically realistic limits [-1, 1]
    sin_gamma = np.clip(sin_gamma, -1.0, 1.0)
    gamma_deg = np.degrees(np.arcsin(sin_gamma))
    RC = V * sin_gamma

    return RC, gamma_deg

def glide_and_descent_performance(V, W, S, rho, CD0, k):
    """
    Unpowered descent / glide performance (Anderson Section 6.9).
    tan(theta_glide) = 1 / (L/D) = D / L
    Sink rate: V_sink = V * sin(theta_glide) ~= V / (L/D)
    """
    q = 0.5 * rho * (V ** 2)
    CL = W / (q * S)
    CD = CD0 + k * (CL ** 2)
    LD = CL / CD

    # Glide angle theta (downward path angle)
    theta_rad = np.arctan(1.0 / LD)
    theta_deg = np.degrees(theta_rad)
    sink_rate = V * np.sin(theta_rad)

    return LD, theta_deg, sink_rate


# -----------------------------------------------------------------------------
# 9. TAKEOFF AND LANDING PERFORMANCE (Anderson Section 6.11)
# -----------------------------------------------------------------------------
def takeoff_ground_roll(W, S, rho, CL_max, CD0, k, prop_type="Jet", T_SL=0.0, P_SL=0.0, mu=0.04):
    """
    Anderson Section 6.11 ground roll formulation.
    V_LO = 1.2 * V_stall
    Average acceleration evaluated at representative speed V_rep = 0.707 * V_LO.
    mu: Runway rolling friction coefficient (typical dry concrete = 0.04, grass = 0.07).
    """
    g = 9.80665
    V_stall_to = np.sqrt((2.0 * W) / (rho * S * CL_max))
    V_LO = 1.2 * V_stall_to
    V_rep = 0.707 * V_LO

    q_rep = 0.5 * rho * (V_rep ** 2)

    # During ground roll prior to rotation: CL_ground ~= 0.8 * (mu / (2*k)) for min drag
    # Anderson approximation: lift on ground is modest, CL_g ~= 0.1 to 0.2
    CL_g = 0.15
    CD_g = CD0 + k * (CL_g ** 2)

    L = q_rep * S * CL_g
    D = q_rep * S * CD_g

    if prop_type == "Jet":
        T = T_SL
    else:
        # P = T * V -> at representative ground speed
        T = P_SL / max(V_rep, 5.0)

    # Net force F = T - D - mu*(W - L)
    net_force = T - D - mu * max(0.0, W - L)
    if net_force <= 0:
        return np.inf, V_LO, V_stall_to  # Aircraft cannot accelerate

    a_avg = (net_force / W) * g
    s_ground_roll = (V_LO ** 2) / (2.0 * a_avg)

    return float(s_ground_roll), float(V_LO), float(V_stall_to)

def landing_ground_roll(W, S, rho, CL_max, CD0, k, mu_braking=0.40):
    """
    Anderson Section 6.11 landing roll.
    Approach speed: V_A = 1.3 * V_stall
    Touchdown speed: V_TD = 1.15 * V_stall
    Braking friction mu_braking ~= 0.4 for typical anti-skid brakes.
    """
    g = 9.80665
    V_stall_land = np.sqrt((2.0 * W) / (rho * S * CL_max))
    V_TD = 1.15 * V_stall_land
    V_rep = 0.707 * V_TD

    q_rep = 0.5 * rho * (V_rep ** 2)
    CL_g = 0.10
    CD_g = CD0 + k * (CL_g ** 2)

    L = q_rep * S * CL_g
    D = q_rep * S * CD_g

    # Retarding force: F_ret = D + mu*(W - L) (assuming idle thrust T ~= 0)
    retarding_force = D + mu_braking * max(0.0, W - L)
    a_decel = (retarding_force / W) * g
    s_landing = (V_TD ** 2) / (2.0 * a_decel)

    return float(s_landing), float(V_TD), float(V_stall_land)