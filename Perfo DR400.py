import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="DR420 Perf 3D - LFCS",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background-color: white; padding: 15px; border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; border-left: 5px solid #3b82f6;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #1e293b; }
    .metric-label { font-size: 12px; color: #64748b; text-transform: uppercase; }
    .safety-warning { background-color: #fef2f2; border: 1px solid #ef4444; color: #b91c1c; padding: 10px; border-radius: 5px; font-weight: bold;}
    .safety-ok { background-color: #f0fdf4; border: 1px solid #22c55e; color: #15803d; padding: 10px; border-radius: 5px; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# --- DONNÉES LFCS (SAUCATS) ---
LFCS_RUNWAYS = {
    "03 Revêtue": {"qfu": 33, "len": 800, "width": 20, "surface": "paved", "tora": 800, "lda": 800},
    "21 Revêtue": {"qfu": 213, "len": 800, "width": 20, "surface": "paved", "tora": 800, "lda": 740},
    "03R Herbe":  {"qfu": 33, "len": 774, "width": 80, "surface": "grass", "tora": 774, "lda": 774},
    "21L Herbe":  {"qfu": 213, "len": 774, "width": 80, "surface": "grass", "tora": 774, "lda": 774},
}

# --- BASE DE DONNÉES PERF (DR420) ---
DB_TAKEOFF = [
    (0, -5, 245, 460, 120, 225), (0, 15, 285, 535, 140, 260), (0, 35, 325, 610, 160, 300),
    (2500, -10, 300, 560, 145, 275), (2500, 10, 350, 655, 170, 320), (2500, 30, 405, 760, 195, 370),
    (5000, -15, 370, 695, 180, 340), (5000, 5, 435, 820, 215, 400), (5000, 25, 505, 950, 250, 465),
    (8000, -21, 490, 920, 240, 450), (8000, -1, 575, 1080, 280, 525), (8000, 19, 670, 1260, 330, 620),
]
DB_LANDING = [
    (0, -5, 185, 435, 145, 365), (0, 15, 200, 460, 155, 385), (0, 35, 210, 485, 165, 400),
    (4000, -13, 205, 475, 160, 395), (4000, 7, 225, 505, 175, 420), (4000, 27, 240, 535, 185, 440),
    (8000, -21, 235, 525, 180, 430), (8000, -1, 250, 555, 195, 460), (8000, 19, 270, 590, 210, 485),
]

# --- MOTEUR DE CALCUL ---
def get_interpolated_values(db, zp_input, temp_input, weight_input):
    df = pd.DataFrame(db, columns=['zp', 'temp', 'r900', 'd900', 'r700', 'd700'])
    unique_zps = sorted(df['zp'].unique())
    zp_low = max([z for z in unique_zps if z <= zp_input], default=unique_zps[0])
    zp_high = min([z for z in unique_zps if z >= zp_input], default=unique_zps[-1])

    def interp_temp_at_zp(target_zp, target_t):
        sub_df = df[df['zp'] == target_zp].sort_values('temp')
        if sub_df.empty: return {'r900':0, 'd900':0, 'r700':0, 'd700':0}
        
        # Bornage sécurisé
        if target_t <= sub_df['temp'].min():
            row_low = row_high = sub_df.iloc[0]
        elif target_t >= sub_df['temp'].max():
            row_low = row_high = sub_df.iloc[-1]
        else:
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
    
    w_clamped = max(700, min(900, weight_input)) 
    ratio_w = (w_clamped - 700) / (200)
    roll = final_base['r700'] + (final_base['r900'] - final_base['r700']) * ratio_w
    dist = final_base['d700'] + (final_base['d900'] - final_base['d700']) * ratio_w
    return roll, dist

def apply_corrections(roll, dist, wind, is_grass, context="takeoff"):
    w_factor = 1.0
    # Vent Face (wind >= 0)
    if wind >= 0: 
        if context == "takeoff":
            xp, fp = [0, 10, 20, 30, 50], [1.0, 0.85, 0.65, 0.55, 0.45]
        else:
            xp, fp = [0, 10, 20, 30, 50], [1.0, 0.78, 0.63, 0.52, 0.40]
        w_factor = np.interp(wind, xp, fp)
    else: # Vent arrière
        w_factor = 1.0 + (abs(wind) / 2) * 0.10

    s_factor = 1.15 if is_grass else 1.0
    return roll * w_factor * s_factor, dist * w_factor * s_factor

# --- MOTEUR 3D AVANCÉ ---

def get_coordinates(distance, heading_deg):
    """Calcule point X,Y pour une distance et un cap."""
    rad = math.radians(heading_deg)
    # Axe Nord = Y, Axe Est = X
    x = math.sin(rad) * distance
    y = math.cos(rad) * distance
    return x, y

def create_runway_mesh(rwy_len, width, qfu, is_grass):
    """Crée un rectangle parfait pour représenter la piste"""
    # Vecteur direction piste (Unitaire)
    rad = math.radians(qfu)
    dir_x = math.sin(rad)
    dir_y = math.cos(rad)
    
    # Vecteur perpendiculaire (Largeur) - Tourne de 90 deg
    perp_x = math.cos(rad) 
    perp_y = -math.sin(rad)
    
    half_w = width / 2
    
    # 4 Coins du rectangle (Seuil = 0,0)
    # Coin Arrière Gauche
    c1_x = 0 - (perp_x * half_w)
    c1_y = 0 - (perp_y * half_w)
    # Coin Arrière Droit
    c2_x = 0 + (perp_x * half_w)
    c2_y = 0 + (perp_y * half_w)
    # Coin Avant Droit
    c3_x = (dir_x * rwy_len) + (perp_x * half_w)
    c3_y = (dir_y * rwy_len) + (perp_y * half_w)
    # Coin Avant Gauche
    c4_x = (dir_x * rwy_len) - (perp_x * half_w)
    c4_y = (dir_y * rwy_len) - (perp_y * half_w)
    
    color = '#2d6a4f' if is_grass else '#6c757d'
    
    # Mesh3D (2 triangles pour faire un rectangle)
    return go.Mesh3d(
        x=[c1_x, c2_x, c3_x, c4_x],
        y=[c1_y, c2_y, c3_y, c4_y],
        z=[0, 0, 0, 0],
        color=color,
        opacity=1,
        i=[0, 0], j=[1, 2], k=[2, 3], # Définition des faces triangulaires
        name='Surface Piste', hoverinfo='none'
    )

def create_3d_visualization(rwy_data, roll_dist, total_dist, is_takeoff=True):
    qfu = rwy_data["qfu"]
    rwy_len = rwy_data["len"]
    available_len = rwy_data["tora"] if is_takeoff else rwy_data["lda"]
    width = rwy_data["width"]
    is_grass = rwy_data["surface"] == "grass"

    fig = go.Figure()

    # 1. SURFACE DE PISTE (Mesh3D solide)
    fig.add_trace(create_runway_mesh(rwy_len, width, qfu, is_grass))

    # Marqueurs Limites
    avail_x, avail_y = get_coordinates(available_len, qfu)
    
    # Ligne de seuil
    perp_x, perp_y = math.cos(math.radians(qfu)), -math.sin(math.radians(qfu))
    fig.add_trace(go.Scatter3d(
        x=[-perp_x*width/2, perp_x*width/2], 
        y=[-perp_y*width/2, perp_y*width/2], 
        z=[0.1, 0.1], mode='lines', line=dict(color='white', width=5), name='Seuil'
    ))
    
    # Ligne de fin TORA/LDA
    fig.add_trace(go.Scatter3d(
        x=[avail_x - perp_x*width/2, avail_x + perp_x*width/2], 
        y=[avail_y - perp_y*width/2, avail_y + perp_y*width/2], 
        z=[0.1, 0.1], mode='lines', line=dict(color='red', width=5, dash='solid'), name='Limite Piste'
    ))

    # Calculs des trajectoires (Axe central)
    if is_takeoff:
        roll_x, roll_y = get_coordinates(roll_dist, qfu)
        obst_x, obst_y = get_coordinates(total_dist, qfu)

        # Roulement
        fig.add_trace(go.Scatter3d(
            x=[0, roll_x], y=[0, roll_y], z=[0.5, 0.5], # Un peu surélevé pour visibilité
            mode='lines', line=dict(color='#3b82f6', width=6), name='Roulement'
        ))
        # Rotation
        fig.add_trace(go.Scatter3d(x=[roll_x], y=[roll_y], z=[0.5], mode='markers', marker=dict(size=5, color='#3b82f6')))
        
        # Montée
        fig.add_trace(go.Scatter3d(
            x=[roll_x, obst_x], y=[roll_y, obst_y], z=[0.5, 15],
            mode='lines', line=dict(color='#60a5fa', width=6), name='Montée'
        ))
        # Obstacle
        col = 'red' if total_dist > available_len else '#10b981'
        fig.add_trace(go.Scatter3d(
            x=[obst_x], y=[obst_y], z=[15], mode='markers+text',
            marker=dict(size=8, color=col), text=[f"15m ({int(total_dist)}m)"], textposition="top center"
        ))
        title = f"Décollage - QFU {qfu}°"
    else:
        # Atterrissage
        air_dist_horiz = total_dist - roll_dist
        touch_x, touch_y = get_coordinates(air_dist_horiz, qfu)
        stop_x, stop_y = get_coordinates(total_dist, qfu)
        
        # Descente
        fig.add_trace(go.Scatter3d(
            x=[0, touch_x], y=[0, touch_y], z=[15, 0.5],
            mode='lines', line=dict(color='#f97316', width=6), name='Approche'
        ))
        # Toucher
        fig.add_trace(go.Scatter3d(x=[touch_x], y=[touch_y], z=[0.5], mode='markers', marker=dict(size=5, color='#f97316')))
        
        # Freinage
        fig.add_trace(go.Scatter3d(
            x=[touch_x, stop_x], y=[touch_y, stop_y], z=[0.5, 0.5],
            mode='lines', line=dict(color='#fb923c', width=6), name='Freinage'
        ))
        # Arrêt
        col = 'red' if total_dist > available_len else '#10b981'
        fig.add_trace(go.Scatter3d(
            x=[stop_x], y=[stop_y], z=[0.5], mode='markers+text',
            marker=dict(size=8, color=col, symbol='square'), text=[f"Arrêt ({int(total_dist)}m)"], textposition="top center"
        ))
        title = f"Atterrissage - QFU {qfu}°"

    # --- CAMÉRA INTELLIGENTE ---
    # On place la caméra "derrière" le seuil (0,0) en fonction du QFU
    # Vecteur unitaire inverse
    rad_cam = math.radians(qfu + 180) # Opposé au QFU
    cam_dist = max(rwy_len, 1000) * 0.8 # Distance recul caméra
    
    # Position caméra (x, y, z)
    # En Plotly 3D, les coordonnées camera.eye sont relatives à la taille de la scène.
    # On triche un peu : on force une vue alignée.
    
    # Pour simplifier l'UX, on calcule les bounds pour l'axe
    max_dim = max(rwy_len, total_dist) * 1.2
    mid_x, mid_y = get_coordinates(max_dim/2, qfu)
    
    # Vecteur directeur pour la caméra (Eye position relative to center)
    # Si QFU = 33 (Nord Est) -> Caméra doit être au Sud Ouest (X neg, Y neg)
    # Si QFU = 213 (Sud Ouest) -> Caméra doit être au Nord Est (X pos, Y pos)
    # Eye coordinates: x,y,z are proportional to the bounding box diagonal.
    
    # Astuce Math: Rotation de l'œil en fonction du QFU
    # Base view (looking North): x=0, y=-1.5, z=0.5
    # Rotation autour de Z
    angle_rad = math.radians(-qfu) # Rotation inverse
    base_x, base_y = 0, -1.8
    
    rot_x = base_x * math.cos(angle_rad) - base_y * math.sin(angle_rad)
    rot_y = base_x * math.sin(angle_rad) + base_y * math.cos(angle_rad)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title="Alt (m)", range=[0, 100]),
            aspectmode='data', # Important pour ne pas déformer la piste
            camera=dict(
                eye=dict(x=rot_x, y=rot_y, z=0.5), # Hauteur caméra
                center=dict(x=0, y=0, z=0) # Regarde le Seuil
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=500
    )
    return fig

# --- INTERFACE UTILISATEUR ---
with st.sidebar:
    st.title("📍 LFCS - Config")
    
    st.subheader("Choix de la Piste")
    selected_rwy_name = st.selectbox("Piste en service", list(LFCS_RUNWAYS.keys()))
    rwy_data = LFCS_RUNWAYS[selected_rwy_name]
    
    # Affichage infos piste
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("QFU", f"{rwy_data['qfu']}°")
    col_p2.metric("Surface", "Herbe" if rwy_data['surface']=='grass' else "Dur")
    col_p3, col_p4 = st.columns(2)
    col_p3.metric("TORA", f"{rwy_data['tora']} m")
    col_p4.metric("LDA", f"{rwy_data['lda']} m")

    st.markdown("---")
    st.subheader("Avion & Météo")
    weight = st.slider("Masse (kg)", 700, 900, 900)
    zp = st.number_input("Alt. Pression Zp (ft)", 0, 10000, 1000, step=100)
    temp = st.number_input("Température (°C)", -30, 40, 20)
    
    # Calcul du vent effectif
    wind_speed = st.slider("Vitesse du vent (kt)", 0, 30, 5)
    wind_dir = st.number_input("Direction du vent (°)", 0, 360, rwy_data['qfu'], step=10)
    
    wind_angle_rad = math.radians(wind_dir - rwy_data['qfu'])
    headwind_component = wind_speed * math.cos(wind_angle_rad)
    crosswind_component = wind_speed * math.sin(wind_angle_rad)

    st.metric("Vent Effectif", f"{int(headwind_component)} kt", 
              delta="Vent de Face" if headwind_component >= 0 else "Vent Arrière",
              help="Positif = Face, Négatif = Arrière")
    if abs(crosswind_component) > 15: st.warning(f"⚠️ Vent de travers fort : {int(abs(crosswind_component))} kt")

# --- CALCULS ---
is_grass = rwy_data["surface"] == "grass"
# Décollage
roll_to_raw, dist_to_raw = get_interpolated_values(DB_TAKEOFF, zp, temp, weight)
roll_to, dist_to = apply_corrections(roll_to_raw, dist_to_raw, headwind_component, is_grass, "takeoff")
# Atterrissage
roll_ldg_raw, dist_ldg_raw = get_interpolated_values(DB_LANDING, zp, temp, weight)
roll_ldg, dist_ldg = apply_corrections(roll_ldg_raw, dist_ldg_raw, headwind_component, is_grass, "landing")
dist_ldg_safe = dist_ldg * 1.3

# --- MAIN ---
st.title(f"✈️ Performances DR420 à Saucats (LFCS)")
st.subheader(f"Piste {selected_rwy_name}")

tab1, tab2 = st.tabs(["🛫 Décollage 3D", "🛬 Atterrissage 3D"])

with tab1:
    # Métriques
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Roulement</div><div class="metric-value">{int(roll_to)} m</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">Passage 15m</div><div class="metric-value">{int(dist_to)} m</div></div>', unsafe_allow_html=True)
    
    margin = rwy_data['tora'] - dist_to
    margin_color = "#22c55e" if margin > 0 else "#ef4444"
    c3.markdown(f'<div class="metric-card" style="border-left: 5px solid {margin_color}"><div class="metric-label">Marge TORA</div><div class="metric-value" style="color:{margin_color}">{int(margin)} m</div></div>', unsafe_allow_html=True)
    
    st.write("")
    # Alerte Sécurité
    if dist_to > rwy_data['tora']:
         st.markdown(f'<div class="safety-warning">⛔ DANGER : La distance de décollage ({int(dist_to)}m) dépasse la TORA ({rwy_data["tora"]}m) !</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="safety-ok">✅ Décollage possible sur {selected_rwy_name}.</div>', unsafe_allow_html=True)

    # VISUALISATION 3D DÉCOLLAGE
    fig_to = create_3d_visualization(rwy_data, roll_to, dist_to, is_takeoff=True)
    st.plotly_chart(fig_to, use_container_width=True)

with tab2:
    # Métriques
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card" style="border-left: 5px solid #f97316"><div class="metric-label">Distance Totale (Manuel)</div><div class="metric-value">{int(dist_ldg)} m</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card" style="border-left: 5px solid #f97316"><div class="metric-label">Distance + Majoration 30%</div><div class="metric-value">{int(dist_ldg_safe)} m</div></div>', unsafe_allow_html=True)
    
    margin_ldg = rwy_data['lda'] - dist_ldg_safe
    margin_color_ldg = "#22c55e" if margin_ldg > 0 else "#ef4444"
    c3.markdown(f'<div class="metric-card" style="border-left: 5px solid {margin_color_ldg}"><div class="metric-label">Marge LDA (sur dist. majorée)</div><div class="metric-value" style="color:{margin_color_ldg}">{int(margin_ldg)} m</div></div>', unsafe_allow_html=True)

    st.write("")
    # Alerte Sécurité Atterrissage
    if dist_ldg > rwy_data['lda']:
         st.markdown(f'<div class="safety-warning">⛔ DANGER : La distance manuelle ({int(dist_ldg)}m) dépasse déjà la LDA ({rwy_data["lda"]}m) !</div>', unsafe_allow_html=True)
    elif dist_ldg_safe > rwy_data['lda']:
         st.markdown(f'<div class="safety-warning">⚠️ ATTENTION : Atterrissage court. La distance majorée dépasse la LDA. Marge de sécurité réduite.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="safety-ok">✅ Atterrissage possible avec marge de sécurité standard (x1.3).</div>', unsafe_allow_html=True)

    # VISUALISATION 3D ATTERRISSAGE
    fig_ldg = create_3d_visualization(rwy_data, roll_ldg, dist_ldg, is_takeoff=False)
    st.plotly_chart(fig_ldg, use_container_width=True)