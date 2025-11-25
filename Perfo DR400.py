import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from data import LFCS_RUNWAYS, DB_TAKEOFF, DB_LANDING

# --- CONSTANTS ---
LANDING_SAFETY_FACTOR = 1.3
GRASS_SURFACE_CORRECTION = 1.15
OBSTACLE_CLEARANCE_HEIGHT_M = 15
CLIMB_PROJECTION_DISTANCE_M = 300
MIN_WEIGHT_KG = 700
MAX_WEIGHT_KG = 900
A_A_FREQ = "119.000"
LFCS_ELEV_FT = 192

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="DR420 Perf 3D - LFCS",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('style.css')

# --- MOTEUR DE CALCUL MATHÉMATIQUE ---
def get_interpolated_values(db, zp_input, temp_input, weight_input):
    """
    Performs a 2D interpolation on performance data.

    Args:
        db (list): The performance database (takeoff or landing).
        zp_input (int): The pressure altitude in feet.
        temp_input (int): The temperature in degrees Celsius.
        weight_input (int): The aircraft weight in kg.

    Returns:
        tuple: A tuple containing the interpolated roll and distance.
    """
    df = pd.DataFrame(db, columns=['zp', 'temp', 'r900', 'd900', 'r700', 'd700'])
    
    min_temp = df['temp'].min()
    max_temp = df['temp'].max()
    if not (min_temp <= temp_input <= max_temp):
        st.sidebar.warning(f"Température hors plage ({min_temp}°C - {max_temp}°C). Les résultats peuvent être imprécis.")

    unique_zps = sorted(df['zp'].unique())
    zp_low = max([z for z in unique_zps if z <= zp_input], default=unique_zps[0])
    zp_high = min([z for z in unique_zps if z >= zp_input], default=unique_zps[-1])

    def interp_temp_at_zp(target_zp, target_t):
        sub_df = df[df['zp'] == target_zp].sort_values('temp')
        if sub_df.empty: return {'r900':0, 'd900':0, 'r700':0, 'd700':0}
        
        target_t = np.clip(target_t, sub_df['temp'].min(), sub_df['temp'].max())
        
        row_low = sub_df[sub_df['temp'] <= target_t].iloc[-1]
        row_high = sub_df[sub_df['temp'] >= target_t].iloc[0]
            
        denom = row_high['temp'] - row_low['temp']
        ratio_t = (target_t - row_low['temp']) / denom if denom != 0 else 0
        res = {}
        for col in ['r900', 'd900', 'r700', 'd700']:
            res[col] = row_low[col] + (row_high[col] - row_low[col]) * ratio_t
        return res

    vals_low = interp_temp_at_zp(zp_low, temp_input)
    vals_high = interp_temp_at_zp(zp_high, temp_input)
    
    denom_zp = zp_high - zp_low
    ratio_zp = (zp_input - zp_low) / denom_zp if denom_zp != 0 else 0
    
    final_base = {k: vals_low[k] + (vals_high[k] - vals_low[k]) * ratio_zp for k in vals_low.keys()}
    
    w_clamped = max(MIN_WEIGHT_KG, min(MAX_WEIGHT_KG, weight_input)) 
    ratio_w = (w_clamped - MIN_WEIGHT_KG) / (MAX_WEIGHT_KG - MIN_WEIGHT_KG)
    roll = final_base['r700'] + (final_base['r900'] - final_base['r700']) * ratio_w
    dist = final_base['d700'] + (final_base['d900'] - final_base['d700']) * ratio_w
    return roll, dist

def apply_corrections(roll, dist, wind, is_grass, context="takeoff"):
    """
    Applies corrections for wind and runway surface.

    Args:
        roll (float): The ground roll distance.
        dist (float): The total distance.
        wind (float): The headwind component in knots.
        is_grass (bool): True if the runway surface is grass.
        context (str): "takeoff" or "landing".

    Returns:
        tuple: A tuple containing the corrected roll and distance.
    """
    w_factor = 1.0
    if wind >= 0: 
        if context == "takeoff": xp, fp = [0, 10, 20, 30, 50], [1.0, 0.85, 0.65, 0.55, 0.45]
        else: xp, fp = [0, 10, 20, 30, 50], [1.0, 0.78, 0.63, 0.52, 0.40]
        w_factor = np.interp(wind, xp, fp)
    else: # Tailwind
        w_factor = 1.0 + (abs(wind) / 2) * 0.10

    s_factor = GRASS_SURFACE_CORRECTION if is_grass else 1.0
    return roll * w_factor * s_factor, dist * w_factor * s_factor

def calculate_climb_gradient(zp, headwind_kt):
    """
    Calculates the climb gradient.

    Args:
        zp (int): The pressure altitude in feet.
        headwind_kt (float): The headwind component in knots.

    Returns:
        tuple: A tuple containing the climb gradient in percent and the rate of climb in ft/min.
    """
    vz_fpm = 570 - (43 * (zp / 1000))
    vz_ms = max(0, vz_fpm * 0.00508) 
    
    tas_kmh = 140 * (1 + (zp/1000)*0.02) 
    tas_ms = tas_kmh / 3.6
    wind_ms = headwind_kt * 0.5144
    ground_speed_ms = max(10, tas_ms - wind_ms)
    
    gradient_pct = (vz_ms / ground_speed_ms) * 100
    return gradient_pct, vz_fpm

# --- MOTEUR 3D & GRAPHIQUE ---

def get_coordinates(distance, heading_deg):
    """Converts distance and heading to cartesian coordinates."""
    rad = math.radians(heading_deg)
    x = math.sin(rad) * distance
    y = math.cos(rad) * distance
    return x, y

def create_runway_mesh(rwy_len, width, qfu, is_grass):
    """Creates a 3D mesh for the runway."""
    rad = math.radians(qfu)
    dir_x, dir_y = math.sin(rad), math.cos(rad)
    perp_x, perp_y = math.cos(rad), -math.sin(rad)
    half_w = width / 2
    
    c1_x, c1_y = 0 - (perp_x * half_w), 0 - (perp_y * half_w)
    c2_x, c2_y = 0 + (perp_x * half_w), 0 + (perp_y * half_w)
    c3_x, c3_y = (dir_x * rwy_len) + (perp_x * half_w), (dir_y * rwy_len) + (perp_y * half_w)
    c4_x, c4_y = (dir_x * rwy_len) - (perp_x * half_w), (dir_y * rwy_len) - (perp_y * half_w)
    
    color = '#2d6a4f' if is_grass else '#6c757d'
    return go.Mesh3d(x=[c1_x, c2_x, c3_x, c4_x], y=[c1_y, c2_y, c3_y, c4_y], z=[0, 0, 0, 0],
                     color=color, opacity=1, i=[0, 0], j=[1, 2], k=[2, 3], name='Surface Piste', hoverinfo='none')

def create_3d_visualization(rwy_data, roll_dist, total_dist, gradient_pct=0, is_takeoff=True):
    """Creates the 3D visualization of the performance."""
    qfu = rwy_data["qfu"]
    rwy_len = rwy_data["len"]
    available_len = rwy_data["tora"] if is_takeoff else rwy_data["lda"]
    width = rwy_data["width"]
    is_grass = rwy_data["surface"] == "grass"

    fig = go.Figure()

    # Runway
    fig.add_trace(create_runway_mesh(rwy_len, width, qfu, is_grass))
    
    avail_x, avail_y = get_coordinates(available_len, qfu)
    perp_x, perp_y = math.cos(math.radians(qfu)), -math.sin(math.radians(qfu))
    
    # Threshold & Limit
    fig.add_trace(go.Scatter3d(x=[-perp_x*width/2, perp_x*width/2], y=[-perp_y*width/2, perp_y*width/2], 
                               z=[0.1, 0.1], mode='lines', line=dict(color='white', width=5), name='Seuil'))
    fig.add_trace(go.Scatter3d(x=[avail_x - perp_x*width/2, avail_x + perp_x*width/2], y=[avail_y - perp_y*width/2, avail_y + perp_y*width/2], 
                               z=[0.1, 0.1], mode='lines', line=dict(color='red', width=5, dash='solid'), name='Limite Piste'))

    if is_takeoff:
        roll_x, roll_y = get_coordinates(roll_dist, qfu)
        obst_x, obst_y = get_coordinates(total_dist, qfu)

        # Ground roll
        fig.add_trace(go.Scatter3d(x=[0, roll_x], y=[0, roll_y], z=[0.5, 0.5], mode='lines', line=dict(color='#3b82f6', width=6), name='Roulement'))
        fig.add_trace(go.Scatter3d(x=[roll_x], y=[roll_y], z=[0.5], mode='markers', marker=dict(size=5, color='#3b82f6'), name='Rotation'))
        
        # 15m climb
        fig.add_trace(go.Scatter3d(x=[roll_x, obst_x], y=[roll_y, obst_y], z=[0.5, OBSTACLE_CLEARANCE_HEIGHT_M], mode='lines', line=dict(color='#60a5fa', width=6), name='Montée 15m'))
        col = 'red' if total_dist > available_len else '#10b981'
        fig.add_trace(go.Scatter3d(x=[obst_x], y=[obst_y], z=[OBSTACLE_CLEARANCE_HEIGHT_M], mode='markers+text', marker=dict(size=8, color=col), 
                                   text=[f"Obst. {OBSTACLE_CLEARANCE_HEIGHT_M}m ({int(total_dist)}m)"], textposition="top center", name='Passage 50ft'))
        
        # Established climb
        if gradient_pct > 0:
            ext_x, ext_y = get_coordinates(total_dist + CLIMB_PROJECTION_DISTANCE_M, qfu)
            ext_z = OBSTACLE_CLEARANCE_HEIGHT_M + (CLIMB_PROJECTION_DISTANCE_M * (gradient_pct / 100)) 
            
            fig.add_trace(go.Scatter3d(x=[obst_x, ext_x], y=[obst_y, ext_y], z=[OBSTACLE_CLEARANCE_HEIGHT_M, ext_z], 
                                       mode='lines', line=dict(color='#22c55e', width=5, dash='dash'), name='Montée Vy'))
            fig.add_trace(go.Scatter3d(x=[ext_x], y=[ext_y], z=[ext_z], mode='markers+text', marker=dict(size=4, color='#22c55e'), 
                                       text=[f"Alt à +{CLIMB_PROJECTION_DISTANCE_M}m: {int(ext_z)}m"], textposition="top center", showlegend=False))
            
        title = f"Décollage - QFU {qfu}° - Pente {gradient_pct:.1f}%"
    else: # Landing
        air_dist_horiz = total_dist - roll_dist
        touch_x, touch_y = get_coordinates(air_dist_horiz, qfu)
        stop_x, stop_y = get_coordinates(total_dist, qfu)
        
        fig.add_trace(go.Scatter3d(x=[0, touch_x], y=[0, touch_y], z=[OBSTACLE_CLEARANCE_HEIGHT_M, 0.5], mode='lines', line=dict(color='#f97316', width=6), name='Finale'))
        fig.add_trace(go.Scatter3d(x=[touch_x], y=[touch_y], z=[0.5], mode='markers', marker=dict(size=5, color='#f97316'), name='Toucher'))
        fig.add_trace(go.Scatter3d(x=[touch_x, stop_x], y=[touch_y, stop_y], z=[0.5, 0.5], mode='lines', line=dict(color='#fb923c', width=6), name='Freinage'))
        
        col = 'red' if total_dist > available_len else '#10b981'
        fig.add_trace(go.Scatter3d(x=[stop_x], y=[stop_y], z=[0.5], mode='markers+text', marker=dict(size=8, color=col, symbol='square'), 
                                   text=[f"Arrêt ({int(total_dist)}m)"], textposition="top center", name='Arrêt'))
        title = f"Atterrissage - QFU {qfu}°"

    # Smart Camera
    angle_rad = math.radians(-qfu)
    base_x, base_y = 0, -1.8 
    rot_x = base_x * math.cos(angle_rad) - base_y * math.sin(angle_rad)
    rot_y = base_x * math.sin(angle_rad) + base_y * math.cos(angle_rad)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(title="Alt (m)", range=[0, 150]),
            aspectmode='data',
            camera=dict(eye=dict(x=rot_x, y=rot_y, z=0.6), center=dict(x=0, y=0, z=0))
        ),
        margin=dict(l=0, r=0, b=0, t=30), height=600
    )
    return fig

# --- UI SIDEBAR ---
with st.sidebar:
    st.title("📍 LFCS - Config")
    
    selected_rwy_name = st.selectbox("Piste en service", list(LFCS_RUNWAYS.keys()))
    rwy_data = LFCS_RUNWAYS[selected_rwy_name]
    
    # Dynamic VAC Info
    st.markdown("---")
    st.markdown("**Informations VAC**")
    st.info(f"""
    **Freq A/A:** {A_A_FREQ}
    **Elev:** {LFCS_ELEV_FT} ft
    **QFU:** {rwy_data['qfu']}°
    """)
    if rwy_data['qfu'] == 33:
        st.warning("⚠️ **03:** Virage à gauche INTERDIT après décollage.")
    if rwy_data['qfu'] == 213:
        st.success("✅ **21:** QFU Préférentiel.")

    st.markdown("---")
    st.subheader("Avion & Météo")
    weight = st.slider("Masse (kg)", MIN_WEIGHT_KG, MAX_WEIGHT_KG, MAX_WEIGHT_KG)
    zp = st.number_input("Alt. Pression Zp (ft)", 0, 10000, LFCS_ELEV_FT, step=50)
    temp = st.number_input("Température (°C)", -30, 40, 20)
    
    wind_speed = st.slider("Vitesse du vent (kt)", 0, 30, 5)
    wind_dir = st.number_input("Direction du vent (°)", 0, 360, rwy_data['qfu'], step=10)
    
    wind_angle_rad = math.radians(wind_dir - rwy_data['qfu'])
    headwind_component = wind_speed * math.cos(wind_angle_rad)
    crosswind_component = wind_speed * math.sin(wind_angle_rad)
    
    st.metric("Vent Effectif", f"{int(headwind_component)} kt", 
              delta="Vent de Face" if headwind_component >= 0 else "Vent Arrière",
              delta_color="normal" if headwind_component >= 0 else "inverse")
    st.metric("Vent de Travers", f"{int(abs(crosswind_component))} kt")

# --- MAIN CALCULATIONS ---
is_grass = rwy_data["surface"] == "grass"

# Takeoff
roll_to_raw, dist_to_raw = get_interpolated_values(DB_TAKEOFF, zp, temp, weight)
roll_to, dist_to = apply_corrections(roll_to_raw, dist_to_raw, headwind_component, is_grass, "takeoff")

# Landing
roll_ldg_raw, dist_ldg_raw = get_interpolated_values(DB_LANDING, zp, temp, weight)
roll_ldg, dist_ldg = apply_corrections(roll_ldg_raw, dist_ldg_raw, headwind_component, is_grass, "landing")
dist_ldg_safe = dist_ldg * LANDING_SAFETY_FACTOR

# Climb
gradient_pct, vz_fpm = calculate_climb_gradient(zp, headwind_component)

# Distance to pattern altitude
dist_to_pattern = dist_to + ( (305 - OBSTACLE_CLEARANCE_HEIGHT_M) / (gradient_pct/100) ) if gradient_pct > 0 else 9999

# --- MAIN TABS ---
st.title(f"✈️ Performances DR420 à Saucats (LFCS)")
st.caption(f"Altitude terrain prise en compte : {zp} ft (VAC: {LFCS_ELEV_FT} ft) ")

tab1, tab2 = st.tabs(["🛫 Décollage & Montée", "🛬 Atterrissage"])

with tab1:
    # Takeoff Metrics
    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="metric-card"><div class="metric-label">Roulement Sol</div><div class="metric-value">{int(roll_to)} m</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><div class="metric-label">Franchissement {OBSTACLE_CLEARANCE_HEIGHT_M}m</div><div class="metric-value">{int(dist_to)} m</div></div>', unsafe_allow_html=True)
    
    margin = rwy_data['tora'] - dist_to
    color_m = "#22c55e" if margin > 0 else "#ef4444"
    col3.markdown(f'<div class="metric-card" style="border-left: 5px solid {color_m}"><div class="metric-label">Marge Restante</div><div class="metric-value" style="color:{color_m}">{int(margin)} m</div></div>', unsafe_allow_html=True)

    st.write("")
    
    # Climb Metrics
    st.subheader("Performances de Montée")
    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="metric-card" style="border-left: 5px solid #8b5cf6"><div class="metric-label">Pente Sol</div><div class="metric-value">{gradient_pct:.1f} %</div></div>', unsafe_allow_html=True)
    
    alt_at_proj = OBSTACLE_CLEARANCE_HEIGHT_M + (CLIMB_PROJECTION_DISTANCE_M * (gradient_pct/100)) if gradient_pct > 0 else OBSTACLE_CLEARANCE_HEIGHT_M
    m2.markdown(f'<div class="metric-card" style="border-left: 5px solid #0ea5e9"><div class="metric-label">Altitude à +{CLIMB_PROJECTION_DISTANCE_M}m (Dist)</div><div class="metric-value">{int(alt_at_proj)} m</div></div>', unsafe_allow_html=True)
    
    m3.markdown(f'<div class="metric-card"><div class="metric-label">Dist. pour 1000ft (TDP)</div><div class="metric-value">{int(dist_to_pattern/1000)} km</div></div>', unsafe_allow_html=True)

    st.write("")
    if dist_to > rwy_data['tora']:
        st.markdown(f'<div class="safety-warning">⛔ <b>DANGER :</b> La distance de décollage dépasse la TORA !</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="safety-ok">✅ Décollage possible (TORA OK).</div>', unsafe_allow_html=True)

    # 3D CHART
    fig_to = create_3d_visualization(rwy_data, roll_to, dist_to, gradient_pct, is_takeoff=True)
    st.plotly_chart(fig_to, use_container_width=True)

with tab2:
    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="metric-card" style="border-left: 5px solid #f97316"><div class="metric-label">Distance Totale</div><div class="metric-value">{int(dist_ldg)} m</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card" style="border-left: 5px solid #f97316"><div class="metric-label">Avec Majoration x{LANDING_SAFETY_FACTOR}</div><div class="metric-value">{int(dist_ldg_safe)} m</div></div>', unsafe_allow_html=True)
    
    margin_l = rwy_data['lda'] - dist_ldg_safe
    color_ml = "#22c55e" if margin_l > 0 else "#ef4444"
    col3.markdown(f'<div class="metric-card" style="border-left: 5px solid {color_ml}"><div class="metric-label">Marge LDA (Sécu)</div><div class="metric-value" style="color:{color_ml}">{int(margin_l)} m</div></div>', unsafe_allow_html=True)

    st.write("")
    if dist_ldg > rwy_data['lda']:
        st.markdown(f'<div class="safety-warning">⛔ <b>DANGER :</b> La piste est trop courte pour atterrir !</div>', unsafe_allow_html=True)
    elif dist_ldg_safe > rwy_data['lda']:
        st.markdown(f'<div class="safety-warning">⚠️ <b>ATTENTION :</b> Atterrissage possible mais marge de sécurité x{LANDING_SAFETY_FACTOR} non respectée.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="safety-ok">✅ Atterrissage possible avec marge de sécurité.</div>', unsafe_allow_html=True)

    # 3D CHART
    fig_ldg = create_3d_visualization(rwy_data, roll_ldg, dist_ldg, is_takeoff=False)
    st.plotly_chart(fig_ldg, use_container_width=True)