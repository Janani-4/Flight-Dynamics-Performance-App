"""
app.py - Flight Performance, Optimization & Flight Envelope Tool
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import flight_math as fm

st.set_page_config(page_title="Aircraft Performance & Optimization Tool", layout="wide")

st.title("✈️Aircraft Performance and Optimization✈️")
st.markdown(
    "Tool for performance estimation, numerical flight envelopes and mission optimization. "
    "Formulae are based on the textbook **J.D. Anderson: *Introduction to Flight***."
)
st.write("---")

# -----------------------------------------------------------------------------
# 1. PRESET AIRCRAFT PROFILES
# -----------------------------------------------------------------------------
st.sidebar.header("1. Aircraft Selection and Presets")

PRESETS = {
    "Custom Aircraft": {
        "prop_type": "Jet", "W": 10000.0, "S": 16.0, "AR": 8.0, "e": 0.80,
        "CD0": 0.0250, "CL_max": 1.50, "T_SL": 3000.0, "P_SL": 0.0, "n_max": 3.8, "n_min": -1.52
    },
    "Cessna 172 Skyhawk (Piston-Prop)": {
        "prop_type": "Propeller", "W": 11120.0, "S": 16.17, "AR": 7.38, "e": 0.78,
        "CD0": 0.0270, "CL_max": 1.60, "T_SL": 0.0, "P_SL": 119300.0, "n_max": 3.8, "n_min": -1.52
    },
    "Business Jet (Learjet-Class)": {
        "prop_type": "Jet", "W": 53000.0, "S": 21.50, "AR": 6.80, "e": 0.82,
        "CD0": 0.0220, "CL_max": 1.45, "T_SL": 19500.0, "P_SL": 0.0, "n_max": 3.4, "n_min": -1.36
    },
    "Fixed-Wing Surveillance UAV": {
        "prop_type": "Propeller", "W": 250.0, "S": 1.20, "AR": 10.0, "e": 0.85,
        "CD0": 0.0280, "CL_max": 1.35, "T_SL": 0.0, "P_SL": 1800.0, "n_max": 4.5, "n_min": -2.0
    }
}

preset_name = st.sidebar.selectbox("Load Aircraft Preset", list(PRESETS.keys()))
p = PRESETS[preset_name]

# Allow overriding or selecting propulsion type
prop_type = st.sidebar.radio(
    "Propulsion Type", ["Jet", "Propeller"],
    index=0 if p["prop_type"] == "Jet" else 1
)

st.sidebar.subheader("Geometry and Aerodynamics")
W = st.sidebar.number_input("Weight W (N)", value=float(p["W"]), step=500.0)
S = st.sidebar.number_input("Wing Area S (m²)", value=float(p["S"]), step=0.5)
AR = st.sidebar.number_input("Aspect Ratio AR", value=float(p["AR"]), step=0.5)
e = st.sidebar.number_input("Oswald Efficiency e", value=float(p["e"]), min_value=0.5, max_value=1.0, step=0.02)
CD0 = st.sidebar.number_input("Zero-Lift Drag CD0", value=float(p["CD0"]), step=0.002, format="%.4f")
CL_max = st.sidebar.number_input("Max Lift Coeff CL_max", value=float(p["CL_max"]), step=0.1)

st.sidebar.subheader("Installed Propulsion (Sea Level)")
if prop_type == "Jet":
    T_SL = st.sidebar.number_input("Max Sea Level Thrust T_max (N)", value=float(p["T_SL"] if p["T_SL"] > 0 else 3000.0), step=250.0)
    P_SL = 0.0
else:
    P_SL = st.sidebar.number_input("Max Sea Level Power P_max (W)", value=float(p["P_SL"] if p["P_SL"] > 0 else 120000.0), step=5000.0)
    T_SL = 0.0

st.sidebar.subheader("Atmosphere and Maneuvers")
h = st.sidebar.number_input("Flight Altitude h (m)", value=2000.0, step=500.0)
delta_T = st.sidebar.slider("ISA Temp Offset ΔT (°C)", min_value=-10.0, max_value=30.0, value=5.0, step=1.0)
n_load = st.sidebar.number_input("Pull-up Maneuver Load Factor n (g's) for Case 3 and V-n diagram", value=2.0, min_value=1.05, max_value=6.0, step=0.1)

# Derived flight conditions
k = fm.get_k(AR, e)
_, _, rho0 = fm.isa_atmosphere(0.0, delta_T)
T_amb, p_amb, rho = fm.isa_atmosphere(h, delta_T)
density_ratio = rho / rho0

V_stall = fm.stall_speed(W, S, rho, CL_max)
V_md, CL_md = fm.speed_L_over_D_max(W, S, rho, CD0, k)
V_mp, CL_mp = fm.speed_min_power(W, S, rho, CD0, k)
opt_speeds = fm.optimize_range_and_endurance_speeds(W, S, rho, CD0, k, prop_type=prop_type)

# Pre-compute takeoff and landing rolls for the dataset
s_to, v_lo, v_stall_to = fm.takeoff_ground_roll(
    W, S, rho0, CL_max, CD0, k, prop_type=prop_type, T_SL=T_SL, P_SL=P_SL, mu=0.04
)
s_lnd, v_td, v_stall_lnd = fm.landing_ground_roll(
    W, S, rho0, CL_max, CD0, k, mu_braking=0.40
)

# -----------------------------------------------------------------------------
# TOP BANNER
# -----------------------------------------------------------------------------
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Air Density ρ", f"{rho:.3f} kg/m³")
col_b.metric("Induced Drag Factor k", f"{k:.5f}")
col_c.metric("Stall Speed V_stall", f"{V_stall:.2f} m/s")
col_d.metric("Min Drag Speed V_(L/D)max", f"{V_md:.2f} m/s")

st.write("---")

# -----------------------------------------------------------------------------
# CASE SELECTION BUTTONS
# -----------------------------------------------------------------------------
# Initialize active case in session state if not already set
if "selected_case" not in st.session_state:
    st.session_state.selected_case = "case1"

# Define the 4 columns for rectangular buttons
b_col1, b_col2, b_col3, b_col4 = st.columns(4)

with b_col1:
    if st.button(
        "Speed Optimization for Range and Endurance",
        type="primary" if st.session_state.selected_case == "case1" else "secondary",
        width="stretch"
    ):
        st.session_state.selected_case = "case1"
        st.rerun()

with b_col2:
    if st.button(
        "Optimal Wing Loading Design",
        type="primary" if st.session_state.selected_case == "case2" else "secondary",
        width="stretch"
    ):
        st.session_state.selected_case = "case2"
        st.rerun()

with b_col3:
    if st.button(
        "Maneuver Performance and Ceilings",
        type="primary" if st.session_state.selected_case == "case3" else "secondary",
        width="stretch"
    ):
        st.session_state.selected_case = "case3"
        st.rerun()

with b_col4:
    if st.button(
        "Takeoff and Landing Distances",
        type="primary" if st.session_state.selected_case == "case4" else "secondary",
        width="stretch"
    ):
        st.session_state.selected_case = "case4"
        st.rerun()

st.write("---")

# -----------------------------------------------------------------------------
# CONDITIONAL RENDERING OF THE SELECTED CASE
# -----------------------------------------------------------------------------
if st.session_state.selected_case == "case1":
    st.subheader(f"Velocity Optimization for {prop_type} Aircraft")
    c1, c2 = st.columns(2)
    with c1:
        st.info("### Maximum Range Condition")
        st.metric("Optimal Range Speed V_opt", f"{opt_speeds['V_opt_range']:.2f} m/s")
        st.write(f"Optimal Lift Coefficient: **$C_L$ = {opt_speeds['CL_opt_range']:.3f}**")
        st.caption("Propeller: $(L/D)_{\\max}$ | Jet: $(C_L^{1/2}/C_D)_{\\max}$")
    with c2:
        st.info("### Maximum Endurance Condition")
        st.metric("Optimal Endurance Speed V_opt", f"{opt_speeds['V_opt_endurance']:.2f} m/s")
        st.write(f"Optimal Lift Coefficient: **$C_L$ = {opt_speeds['CL_opt_endurance']:.3f}**")
        st.caption("Propeller: $(C_L^{3/2}/C_D)_{\\max}$ (Min Power) | Jet: $(L/D)_{\\max}$ (Min Thrust)")

    st.write("---")
    st.subheader("Performance Curves: Required vs. Available")

    V_vals = np.linspace(V_stall * 1.01, V_md * 2.4, 200)
    TR_vals = [float(fm.thrust_required(v, W, S, rho, CD0, k)) for v in V_vals]
    PR_vals = [float(fm.power_required(v, W, S, rho, CD0, k)) / 1000.0 for v in V_vals]

    fig_perf = go.Figure()

    if prop_type == "Jet":
        T_avail_h = float(T_SL * density_ratio)
        TA_vals = [T_avail_h] * len(V_vals)

        fig_perf.add_trace(go.Scatter(x=V_vals, y=TR_vals, mode='lines', name='Thrust Required (T_R)', line=dict(color='#1f77b4', width=3)))
        fig_perf.add_trace(go.Scatter(x=V_vals, y=TA_vals, mode='lines', name='Thrust Available (T_A)', line=dict(color='#2ca02c', width=3, dash='dash')))

        fig_perf.add_trace(go.Scatter(
            x=[V_md], y=[fm.thrust_required(V_md, W, S, rho, CD0, k)],
            mode='markers', name='(L/D)_max / Min T_R',
            marker=dict(color='red', size=12, symbol='diamond'),
            hovertemplate="Speed: %{x:.2f} m/s<br>Min T_R: %{y:.1f} N<extra></extra>"
        ))
        fig_perf.add_trace(go.Scatter(
            x=[opt_speeds['V_opt_range']], y=[fm.thrust_required(opt_speeds['V_opt_range'], W, S, rho, CD0, k)],
            mode='markers', name='Max Range Speed',
            marker=dict(color='orange', size=14, symbol='star'),
            hovertemplate="Speed: %{x:.2f} m/s<br>T_R: %{y:.1f} N<extra></extra>"
        ))

        diff = np.array(TA_vals) - np.array(TR_vals)
        crossings = np.where(np.diff(np.sign(diff)))[0]
        if len(crossings) > 0:
            V_max_est = float(V_vals[crossings[-1]])
            fig_perf.add_vline(x=V_max_est, line_width=1.5, line_dash="dot", line_color="green", annotation_text=f"V_max ≈ {V_max_est:.1f} m/s")

        fig_perf.update_layout(
            title=f"Thrust Required & Available at h = {h:.0f} m (ΔT = {delta_T:+.0f}°C)",
            xaxis_title="True Airspeed V (m/s)", yaxis_title="Thrust (N)",
            hovermode="x unified"
        )

    else:
        P_avail_h = float((P_SL * density_ratio) / 1000.0)  # kW
        PA_vals = [P_avail_h] * len(V_vals)

        fig_perf.add_trace(go.Scatter(x=V_vals, y=PR_vals, mode='lines', name='Power Required (P_R)', line=dict(color='#ff7f0e', width=3)))
        fig_perf.add_trace(go.Scatter(x=V_vals, y=PA_vals, mode='lines', name='Power Available (P_A)', line=dict(color='#2ca02c', width=3, dash='dash')))

        fig_perf.add_trace(go.Scatter(
            x=[V_mp], y=[fm.power_required(V_mp, W, S, rho, CD0, k) / 1000.0],
            mode='markers', name='Min Power / Max Endurance',
            marker=dict(color='red', size=12, symbol='diamond'),
            hovertemplate="Speed: %{x:.2f} m/s<br>Min P_R: %{y:.1f} kW<extra></extra>"
        ))
        fig_perf.add_trace(go.Scatter(
            x=[opt_speeds['V_opt_range']], y=[fm.power_required(opt_speeds['V_opt_range'], W, S, rho, CD0, k) / 1000.0],
            mode='markers', name='Max Range Speed',
            marker=dict(color='purple', size=14, symbol='star'),
            hovertemplate="Speed: %{x:.2f} m/s<br>P_R: %{y:.1f} kW<extra></extra>"
        ))

        diff = np.array(PA_vals) - np.array(PR_vals)
        crossings = np.where(np.diff(np.sign(diff)))[0]
        if len(crossings) > 0:
            V_max_est = float(V_vals[crossings[-1]])
            fig_perf.add_vline(x=V_max_est, line_width=1.5, line_dash="dot", line_color="green", annotation_text=f"V_max ≈ {V_max_est:.1f} m/s")

        fig_perf.update_layout(
            title=f"Power Required & Available at h = {h:.0f} m (ΔT = {delta_T:+.0f}°C)",
            xaxis_title="True Airspeed V (m/s)", yaxis_title="Power (kW)",
            hovermode="x unified"
        )

    fig_perf.add_vline(x=V_stall, line_width=1.5, line_dash="dash", line_color="red", annotation_text=f"V_stall = {V_stall:.1f} m/s")
    st.plotly_chart(fig_perf, width="stretch")


elif st.session_state.selected_case == "case2":
    st.subheader("Optimal Wing Loading (W/S) Design Tool")
    v_target = st.number_input("Target Max Stall Speed (m/s)", value=25.0, min_value=10.0, max_value=80.0, step=1.0)
    opt_WS = fm.optimize_wing_loading(v_target, rho, CL_max)
    required_S = W / opt_WS
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.metric("Maximum Wing Loading (W/S)", f"{opt_WS:.2f} N/m²")
    with col_w2:
        st.metric("Minimum Required Wing Area", f"{required_S:.2f} m²")
        if required_S > S:
            st.warning(f"⚠️ Current wing area ({S:.1f} m²) exceeds stall limit! Increase to at least {required_S:.2f} m².")
        else:
            st.success(f"✅ Current wing area ({S:.1f} m²) safely satisfies the stall constraint.")

elif st.session_state.selected_case == "case3":
    st.subheader("Maneuver Dynamics and Ceilings")

    if st.button("Calculate Service & Absolute Ceilings"):
        with st.spinner("Evaluating climb excess power envelope across altitudes..."):
            serv_ceil, abs_ceil = fm.find_ceilings(
                W, S, AR, e, CD0, prop_type=prop_type, T_SL=T_SL, P_SL=P_SL, delta_T=delta_T
            )
            col_c1, col_c2 = st.columns(2)
            col_c1.metric("Service Ceiling (R/C = 0.5 m/s)", f"{serv_ceil:.0f} m")
            col_c2.metric("Absolute Ceiling (R/C = 0 m/s)", f"{abs_ceil:.0f} m")

    st.write("---")
    V_maneuver = st.slider(
        "Maneuver Airspeed V (m/s) for Climb/Glide/Turn Analysis",
        min_value=float(np.round(V_stall * 1.05, 1)),
        max_value=float(np.round(V_md * 2.2, 1)),
        value=float(np.round(opt_speeds['V_opt_range'], 1)),
        step=1.0
    )

    st.markdown("#### Climb and Unpowered Descent Performance")
    T_avail_current = T_SL * density_ratio if prop_type == "Jet" else 0.0
    P_avail_current = P_SL * density_ratio if prop_type == "Propeller" else 0.0

    rc_val, gamma_climb = fm.climb_performance(
        V_maneuver, W, S, rho, CD0, k, prop_type=prop_type,
        T_avail=T_avail_current, P_avail=P_avail_current
    )
    ld_glide, theta_glide, sink_rate = fm.glide_and_descent_performance(
        V_maneuver, W, S, rho, CD0, k
    )

    climb_col1, climb_col2, glide_col1, glide_col2 = st.columns(4)
    climb_col1.metric("Rate of Climb (R/C)", f"{rc_val:.2f} m/s")
    climb_col2.metric("Climb Angle γ", f"{gamma_climb:.2f}°")
    glide_col1.metric("Glide Ratio (L/D)", f"{ld_glide:.2f}:1")
    glide_col2.metric("Glide Sink Rate", f"{sink_rate:.2f} m/s")

    st.write("---")
    st.markdown(f"#### Coordinated Level Turn Dynamics ($n = {n_load:.1f}$ g)")
    bank_deg, radius_m, turn_rate, is_stalled = fm.turning_performance(
        V_maneuver, W, S, rho, CD0, k, n_load, CL_max
    )

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Bank Angle φ", f"{bank_deg:.1f}°")
    tc2.metric("Turn Radius R", f"{radius_m:.1f} m")
    tc3.metric("Turn Rate ω", f"{turn_rate:.2f} °/s")

    if is_stalled:
        st.error(f"🚨 Aerodynamic Stall: At V = {V_maneuver:.1f} m/s, pulling {n_load:.1f}g demands $C_L > C_{{L\\max}}$!")
    else:
        st.success("✅ Turn load factor is safely within aerodynamic lift limits.")

    st.write("---")
    st.subheader("Flight Envelope: Maneuvering V-n Diagram")
    st.markdown("Shows maximum aerodynamically attainable and structurally allowable load factors across airspeed.")

    V_max_guess = V_md * 2.0
    vn = fm.vn_diagram_boundaries(W, S, rho, CL_max, n_pos_limit=p["n_max"], n_neg_limit=p["n_min"], V_max_level=V_max_guess)

    fig_vn = go.Figure()

    fig_vn.add_trace(go.Scatter(x=vn["V_pos_curve"], y=vn["n_pos_curve"], mode='lines', name='+CL_max Stall Boundary', line=dict(color='blue', width=2.5)))
    fig_vn.add_trace(go.Scatter(x=[vn["V_A"], vn["V_dive"]], y=[p["n_max"], p["n_max"]], mode='lines', name='+Structural Limit', line=dict(color='red', width=2.5, dash='dash')))

    fig_vn.add_trace(go.Scatter(x=vn["V_neg_curve"], y=vn["n_neg_curve"], mode='lines', name='-CL_min Stall Boundary', line=dict(color='blue', width=2.5)))
    fig_vn.add_trace(go.Scatter(x=[vn["V_neg_limit_speed"], vn["V_dive"]], y=[p["n_min"], p["n_min"]], mode='lines', name='-Structural Limit', line=dict(color='red', width=2.5, dash='dash')))

    fig_vn.add_trace(go.Scatter(x=[vn["V_dive"], vn["V_dive"]], y=[p["n_min"], p["n_max"]], mode='lines', name='Dive Speed Limit (V_D)', line=dict(color='darkred', width=2.5)))

    fig_vn.add_trace(go.Scatter(
        x=[opt_speeds['V_opt_range']], y=[n_load], mode='markers', name='Your Maneuver Input',
        marker=dict(color='gold', size=14, symbol='circle', line=dict(color='black', width=2)),
        hovertemplate="Airspeed: %{x:.2f} m/s<br>Load Factor: %{y:.2f} g<extra></extra>"
    ))

    fig_vn.update_layout(
        title=f"V-n Diagram at Altitude h = {h:.0f} m (Corner Speed V_A = {vn['V_A']:.1f} m/s)",
        xaxis_title="Airspeed V (m/s)", yaxis_title="Load Factor n (g)",
        hovermode="closest",
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    )

    st.plotly_chart(fig_vn, width="stretch")

elif st.session_state.selected_case == "case4":
    st.subheader("Runway Ground Roll and Transition Performance (JD Anderson Sec 6.11)")
    st.markdown("Ground run distance over runway surface until liftoff ($V_{LO}$) and landing stop ($V_{TD}$).")

    col_to_params, col_lnd_params = st.columns(2)
    with col_to_params:
        st.markdown("##### Takeoff Configuration")
        mu_takeoff = st.selectbox(
            "Runway Surface Friction (Takeoff μ)", [0.04, 0.07, 0.10], index=0,
            format_func=lambda x: f"Dry Concrete/Asphalt (μ={x})" if x==0.04 else (f"Hard Turf/Grass (μ={x})" if x==0.07 else f"Soft Turf (μ={x})")
        )
        s_to, v_lo, v_stall_to = fm.takeoff_ground_roll(
            W, S, rho0, CL_max, CD0, k, prop_type=prop_type, T_SL=T_SL, P_SL=P_SL, mu=mu_takeoff
        )
        st.metric("Ground Roll Distance ($S_{LO}$)", f"{s_to:.1f} m")
        st.write(f"• Liftoff Speed ($V_{{LO}} = 1.2 V_{{stall}}$): **{v_lo:.1f} m/s**")
        st.write(f"• Ground Stall Speed: **{v_stall_to:.1f} m/s**")

    with col_lnd_params:
        st.markdown("##### Landing Configuration")
        mu_braking = st.slider("Braking Friction Coefficient ($μ_b$)", min_value=0.20, max_value=0.50, value=0.40, step=0.05)
        s_lnd, v_td, v_stall_lnd = fm.landing_ground_roll(
            W, S, rho0, CL_max, CD0, k, mu_braking=mu_braking
        )
        st.metric("Landing Ground Roll ($S_{TD}$)", f"{s_lnd:.1f} m")
        st.write(f"• Touchdown Speed ($V_{{TD}} = 1.15 V_{{stall}}$): **{v_td:.1f} m/s**")
        st.write(f"• Approach Speed ($V_A = 1.3 V_{{stall}}$): **{v_stall_lnd * 1.3:.1f} m/s**")

#-- Data Export Section --
# -----------------------------------------------------------------------------
# DATA EXPORT SECTION (EXCEL / CSV COMPATIBLE)
# -----------------------------------------------------------------------------
import pandas as pd

# Ensure summary metrics dictionary has clean numbers
summary_data = {
    "Parameter": [
        "Aircraft Weight W (N)",
        "Wing Area S (m²)",
        "Flight Altitude h (m)",
        "Ambient Air Density ρ (kg/m³)",
        "Stall Speed V_stall (m/s)",
        "Min Drag Speed V_(L/D)max (m/s)",
        "Max Range Optimal Speed (m/s)",
        "Max Endurance Optimal Speed (m/s)",
        "Takeoff Ground Roll (m)",
        "Landing Ground Roll (m)"
    ],
    "Value": [
        float(W),
        float(S),
        float(h),
        round(float(rho), 3),
        round(float(V_stall), 2),
        round(float(V_md), 2),
        round(float(opt_speeds['V_opt_range']), 2),
        round(float(opt_speeds['V_opt_endurance']), 2),
        round(float(s_to), 1),
        round(float(s_lnd), 1)
    ],
    "Unit": ["N", "m²", "m", "kg/m³", "m/s", "m/s", "m/s", "m/s", "m", "m"]
}

df_summary = pd.DataFrame(summary_data)

# Convert DataFrame to utf-8 encoded CSV bytes for Excel
csv_bytes = df_summary.to_csv(index=False).encode("utf-8")

st.write("---")
st.subheader("Performance Summary")

# Optional preview of the table right on the webpage
st.dataframe(df_summary, use_container_width=True)

st.download_button(
    label="Download Performance Summary (.csv)",
    data=csv_bytes,
    file_name="aircraft_performance_summary.csv",
    mime="text/csv"
)
