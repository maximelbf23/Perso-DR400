import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
# On importe les données depuis votre fichier data.py existant
from data import LFCS_RUNWAYS, DB_TAKEOFF, DB_LANDING

# ==========================================
# 1. CONSTANTES & CONFIGURATION
# ==========================================
LANDING_SAFETY_FACTOR = 1.3
GRASS_SURFACE_CORRECTION = 1.15
WET_SURFACE_CORRECTION = 1.15  # Nouveau facteur sécurité
OBSTACLE_CLEARANCE_HEIGHT_M = 15
CLIMB_PROJECTION_DISTANCE_M = 400
MIN_WEIGHT_KG = 700
MAX_WEIGHT_KG = 900
A_A_FREQ = "119.000"
LFCS_ELEV_FT = 192
MAX_DEMONSTRATED_CROSSWIND = 22  # Limite vent de travers DR400

st.set_page_config(
    page_title="DR420 Perf 3D - LFCS",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nous intégrons le CSS directement ici pour éviter les problèmes de chargement de fichier
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
.main { background: linear-gradient(to right, #f8f9fa, #e9ecef); }

.metric-container {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    text-align: center;
    border-left: 5px solid #cbd5e1;
    height: 100%;
}
.metric-value { font-size: 24px; font-weight: 700; color: #1e293b; }
.metric-sub { font-size: 14px; color: #64748b; margin-top: -5px; }
.metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }

/* Status Colors */
.status-ok { border-left-color: #22c55e !important; }
.status-warn { border-left-color: #f59e0b !important; }
.status-danger { border-left-color: #ef4444 !important; }

.alert-box { padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; margin: 10px 0; }
.alert-danger { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
.alert-warning { background-color: #fef3c7; color: #92400e; border: 1px solid #fbbf24; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# 2. MOTEUR PHYSIQUE AVANCÉ
# ==========================================

def calculate_density_altitude(zp, temp):
    """Calcule l'altitude densité : crucial en été."""
    isa_temp = 15 - (2 * zp / 1000)
    return zp + 118.6 * (temp - isa_temp)

def get_interpolated_values(db, zp_input, temp_input, weight_input):
    """Interpolation robuste avec protection des bornes."""
    df = pd.DataFrame(db, columns=['zp', 'temp', 'r900', 'd900', 'r700', 'd700'])
    
    # Sécurisation des entrées
    t_min, t_max = df['temp'].min(), df['temp'].max()
    t_safe = np.clip(temp_input, t_min, t_max)
    
    unique_zps = sorted(df['zp'].unique())
    zp_low = max([z for z in unique_zps if z <= zp_input], default=unique_zps[0])
    zp_high = min([z for z in unique_zps if z >= zp_input], default=unique_zps[-1])

    def interp_temp(target_z, target_t):
        sub = df[df['zp'] == target_z].sort_values('temp')
        if sub.empty: return {'r900':0, 'd900':0, 'r700':0, 'd700':0}
        
        row_low = sub[sub['temp'] <= target_t].iloc[-1] if not sub[sub['temp'] <= target_t].empty else sub.iloc[0]
        row_high = sub[sub['temp'] >= target_t].iloc[0] if not sub[sub['temp'] >= target_t].empty else sub.iloc[-1]
        
        ratio = 0 if row_high['temp'] == row_low['temp'] else (target_t - row_low['temp']) / (row_high['temp'] - row_low['temp'])
        return {col: row_low[col] + (row_high[col] - row_low[col]) * ratio for col in ['r900', 'd900', 'r700', 'd700']}

    v_low = interp_temp(zp_low, t_safe)
    v_high = interp_temp(zp_high, t_safe)
    
    ratio_z = 0 if zp_high == zp_low else (zp_input - zp_low) / (zp_high - zp_low)
    base = {k: v_low[k] + (v_high[k] - v_low[k]) * ratio_z for k in v_low.keys()}
    
    # Interpolation Masse
    ratio_w = (weight_input - MIN_WEIGHT_KG) / (MAX_WEIGHT_KG - MIN_WEIGHT_KG)
    roll = base['r700'] + (base['r900'] - base['r700']) * ratio_w
    dist = base['d700'] + (base['d900'] - base['d700']) * ratio_w
    
    return roll, dist

def apply_corrections(roll, dist, wind, is_grass, is_wet, context="takeoff"):
    """Applique les facteurs vent, herbe ET piste mouillée."""
    # Vent
    if wind >= 0: # Face
        xp, fp = ([0, 10, 20, 30, 50], [1.0, 0.85, 0.65, 0.55, 0.45]) if context == "takeoff" else ([0, 10, 20, 30, 50], [1.0, 0.78, 0.63, 0.52, 0.40])
        w_factor = np.interp(wind, xp, fp)
    else: # Arrière
        w_factor = 1.0 + (abs(wind) / 2) * 0.10

    # Surface
    s_factor = 1.0
    if is_grass: s_factor *= GRASS_SURFACE_CORRECTION
    if is_wet: s_factor *= WET_SURFACE_CORRECTION
    
    return roll * w_factor * s_factor, dist * w_factor * s_factor

def calculate_climb_perf(zp, temp, headwind_kt):
    """Calcule la pente en tenant compte de la température (Densité)."""
    # Vz Standard corrigée température (-1% par deg > ISA approx)
    isa = 15 - (2 * zp/1000)
    delta_isa = temp - isa
    perf_factor = 1.0 - (delta_isa * 0.01) if delta_isa > 0 else 1.0
    
    vz_fpm = (570 - (43 * (zp / 1000))) * perf_factor
    vz_ms = max(0, vz_fpm * 0.00508)
    
    # TAS Réelle (augmente avec l'altitude et la chaleur)
    dens_ratio = (288.15 / (temp + 273.15)) * (1 - (0.0000225577 * zp))**4.25588
    tas_kmh = 140 / math.sqrt(dens_ratio if dens_ratio > 0.5 else 0.5)
    
    ground_speed_ms = max(10, (tas_kmh / 3.6) - (headwind_kt * 0.5144))
    gradient_pct = (vz_ms / ground_speed_ms) * 100
    
    return gradient_pct, vz_fpm, tas_kmh

# ==========================================
# 3. VISUALISATION 3D
# ==========================================
def get_coords(dist, qfu):
    rad = math.radians(qfu)
    return math.sin(rad) * dist, math.cos(rad) * dist

def create_3d_chart(rwy_data, roll, dist, grad_pct=0, is_takeoff=True):
    qfu, rlen, width = rwy_data["qfu"], rwy_data["len"], rwy_data["width"]
    avail = rwy_data["tora"] if is_takeoff else rwy_data["lda"]
    is_grass = rwy_data["surface"] == "grass"
    
    fig = go.Figure()
    
    # Piste
    cx, cy = get_coords(rlen, qfu)
    px, py = math.cos(math.radians(qfu)) * width/2, -math.sin(math.radians(qfu)) * width/2
    fig.add_trace(go.Mesh3d(x=[-px, px, cx+px, cx-px], y=[-py, py, cy+py, cy-py], z=[0,0,0,0], color='#3f6212' if is_grass else '#475569', opacity=1, name='Piste'))
    
    # Seuils
    ax, ay = get_coords(avail, qfu)
    fig.add_trace(go.Scatter3d(x=[ax-px, ax+px], y=[ay-py, ay+py], z=[0.1, 0.1], mode='lines', line=dict(color='red', width=4), name='Limite'))

    # Trajectoire
    roll_x, roll_y = get_coords(roll, qfu)
    obst_x, obst_y = get_coords(dist, qfu)
    line_col = '#3b82f6' if is_takeoff else '#f97316'

    if is_takeoff:
        fig.add_trace(go.Scatter3d(x=[0, roll_x], y=[0, roll_y], z=[0.5, 0.5], mode='lines', line=dict(color=line_col, width=6), name='Roulement'))
        fig.add_trace(go.Scatter3d(x=[roll_x, obst_x], y=[roll_y, obst_y], z=[0.5, 15], mode='lines', line=dict(color='#60a5fa', width=5), name='Montée 15m'))
        
        if grad_pct > 0: # Projection Montée
            px, py = get_coords(dist + CLIMB_PROJECTION_DISTANCE_M, qfu)
            pz = 15 + (CLIMB_PROJECTION_DISTANCE_M * grad_pct/100)
            fig.add_trace(go.Scatter3d(x=[obst_x, px], y=[obst_y, py], z=[15, pz], mode='lines', line=dict(color='#22c55e', width=4, dash='dash'), name='Montée Initiale'))

    else: # Landing
        air_d = dist - roll
        touch_x, touch_y = get_coords(air_d, qfu)
        stop_x, stop_y = get_coords(dist, qfu)
        # Axe approche
        app_x, app_y = get_coords(-300, qfu)
        fig.add_trace(go.Scatter3d(x=[app_x, 0], y=[app_y, 0], z=[15, 15], mode='lines', line=dict(color='gray', width=2, dash='dot'), name='Axe Approche'))
        fig.add_trace(go.Scatter3d(x=[0, touch_x], y=[0, touch_y], z=[15, 0.5], mode='lines', line=dict(color=line_col, width=6), name='Finale'))
        fig.add_trace(go.Scatter3d(x=[touch_x, stop_x], y=[touch_y, stop_y], z=[0.5, 0.5], mode='lines', line=dict(color='#fb923c', width=6), name='Freinage'))

    # Caméra
    rad_cam = math.radians(-qfu - 90)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=500, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(range=[0, 100]), aspectratio=dict(x=1, y=1, z=0.3), camera=dict(eye=dict(x=1.5*math.cos(rad_cam), y=1.5*math.sin(rad_cam), z=0.8))))
    return fig

# ==========================================
# 4. INTERFACE UTILISATEUR
# ==========================================
with st.sidebar:
    st.header("1. Situation")
    rwy_name = st.selectbox("Piste", list(LFCS_RUNWAYS.keys()))
    rwy = LFCS_RUNWAYS[rwy_name]
    c1, c2 = st.columns(2)
    c1.metric("QFU", f"{rwy['qfu']}°")
    c2.metric("Elev", f"{LFCS_ELEV_FT} ft")
    
    st.markdown("---")
    st.header("2. Météo & Avion")
    
    # SECTION MODIFIÉE : QNH & Zp
    with st.expander("Paramètres de vol", expanded=True):
        weight = st.slider("Masse (kg)", MIN_WEIGHT_KG, MAX_WEIGHT_KG, MAX_WEIGHT_KG)
        
        # Saisie du QNH au lieu de Zp direct
        qnh = st.number_input("QNH (hPa)", 950, 1050, 1013, help="Pression atmosphérique du jour")
        temp = st.number_input("Température (°C)", -20, 45, 20)

        # Calcul automatique de Zp
        # Zp = Elevation + (1013 - QNH) * 27
        zp_calc = LFCS_ELEV_FT + (1013 - qnh) * 27
        zp = max(0, int(zp_calc))
        
        st.caption(f"ℹ️ Altitude Pression (Zp) : **{zp} ft**")
    
    with st.expander("Vent & État Piste", expanded=True):
        wind_speed = st.slider("Vent (kt)", 0, 35, 5)
        wind_dir = st.number_input("Dir Vent (°)", 0, 360, rwy['qfu'], step=10)
        is_wet = st.checkbox("Piste Mouillée 💧", help="Facteur x1.15 supplémentaire")

    # Calculs Vent
    w_rad = math.radians(wind_dir - rwy['qfu'])
    hw, xw = wind_speed * math.cos(w_rad), wind_speed * math.sin(w_rad)
    da = calculate_density_altitude(zp, temp)
    
    st.markdown("---")
    col_w1, col_w2 = st.columns(2)
    col_w1.metric("Vent Face", f"{int(hw)} kt", delta_color="normal" if hw >= 0 else "inverse")
    col_w2.metric("Travers", f"{int(abs(xw))} kt", delta_color="inverse" if abs(xw) > MAX_DEMONSTRATED_CROSSWIND else "normal")
    st.info(f"🌫️ **Alt. Densité:** {int(da)} ft")

# Logique Principale
is_grass = rwy["surface"] == "grass"
r_to, d_to = apply_corrections(*get_interpolated_values(DB_TAKEOFF, zp, temp, weight), hw, is_grass, is_wet, "takeoff")
r_ldg, d_ldg = apply_corrections(*get_interpolated_values(DB_LANDING, zp, temp, weight), hw, is_grass, is_wet, "landing")
d_ldg_safe = d_ldg * LANDING_SAFETY_FACTOR
grad_pct, vz_fpm, tas_kmh = calculate_climb_perf(zp, temp, hw)

# Affichage Principal
st.title(f"✈️ Perf DR420 - LFCS {rwy_name}")

if abs(xw) > MAX_DEMONSTRATED_CROSSWIND:
    st.markdown(f'<div class="alert-box alert-danger">⚠️ VENT DE TRAVERS EXCESSIF ({int(abs(xw))} kt)</div>', unsafe_allow_html=True)
if is_wet and is_grass:
    st.markdown('<div class="alert-box alert-warning">⚠️ ATTENTION: HERBE MOUILLÉE</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🛫 DÉCOLLAGE", "🛬 ATTERRISSAGE"])

with tab1:
    m_to = rwy['tora'] - d_to
    status = "status-ok" if m_to > 100 else "status-warn" if m_to > 0 else "status-danger"
    
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-container status-ok"><div class="metric-label">Roulement</div><div class="metric-value">{int(r_to)} m</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-container status-ok"><div class="metric-label">Passage 15m</div><div class="metric-value">{int(d_to)} m</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-container {status}"><div class="metric-label">Marge Restante</div><div class="metric-value">{int(m_to)} m</div></div>', unsafe_allow_html=True)
    
    st.write("")
    st.progress(min(1.0, d_to / rwy['tora']))
    
    st.markdown("##### 📈 Montée Initiale")
    k1, k2, k3 = st.columns(3)
    k1.metric("Pente Sol", f"{grad_pct:.1f} %")
    k2.metric("Vz Estimée", f"{int(vz_fpm)} ft/min")
    k3.metric("Vitesse Sol", f"{int(tas_kmh - (hw * 1.852))} km/h")
    
    st.plotly_chart(create_3d_chart(rwy, r_to, d_to, grad_pct, True), use_container_width=True)

with tab2:
    m_ldg = rwy['lda'] - d_ldg_safe
    status_l = "status-ok" if m_ldg > 50 else "status-warn" if m_ldg > 0 else "status-danger"
    
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-container status-ok"><div class="metric-label">Roulement</div><div class="metric-value">{int(r_ldg)} m</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-container status-warn"><div class="metric-label">Total Majioré (x{LANDING_SAFETY_FACTOR})</div><div class="metric-value">{int(d_ldg_safe)} m</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-container {status_l}"><div class="metric-label">Marge Sécu</div><div class="metric-value">{int(m_ldg)} m</div></div>', unsafe_allow_html=True)
    
    st.write("")
    st.progress(min(1.0, d_ldg_safe / rwy['lda']))
    st.plotly_chart(create_3d_chart(rwy, r_ldg, d_ldg, 0, False), use_container_width=True)