import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURACIÓN E IDENTIDAD ---
st.set_page_config(layout="wide", page_title="IPC Intelligence Tool", page_icon=":material/rate_review:")

COLORS = {
    "bg_dark": "#0e1117",
    "card_bg": "#1e2130",
    "primary": "#00d4ff",
    "secondary": "#1a73e8",
    "accent_green": "#00f2a1",
    "palette": px.colors.qualitative.Plotly
}

st.markdown(f"""
    <style>
    .stApp {{ background-color: {COLORS['bg_dark']}; }}
    .network-wrapper {{
        display: flex; justify-content: center; align-items: center;
        background-color: #000; border: 1px solid #333;
        border-radius: 12px; height: 500px; width: 100%; overflow: hidden;
    }}
    .metric-container {{
        background-color: {COLORS['card_bg']}; padding: 20px;
        border-radius: 10px; border-left: 5px solid {COLORS['primary']};
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARGA Y PROCESAMIENTO DE DATOS DESDE EXCEL ---
@st.cache_data
def load_and_process_data():
    # Ruta apuntando al directorio data
    excel_path = 'data/ipc.xlsx'
    
    try:
        # Leemos las pestañas específicas directamente del archivo maestro Excel
        df_var = pd.read_excel(excel_path, sheet_name='Variación mensual aperturas', skiprows=4)
        df_weights = pd.read_excel(excel_path, sheet_name='Ponderaciones', skiprows=3)
    except Exception as e:
        st.error(f":material/warning: Error al cargar los datos: {e}. Verifica que el archivo exista en 'data/ipc.xlsx' y que las hojas se llamen 'Variación mensual aperturas' y 'Ponderaciones'.")
        return None, None, None

    # Limpieza de Ponderaciones
    weights = df_weights.dropna(subset=['Divisiones']).copy()
    weights['Divisiones'] = weights['Divisiones'].str.strip()
    
    # Limpieza de Variaciones
    df_var.rename(columns={df_var.columns[0]: 'Apertura'}, inplace=True)
    df_var = df_var.dropna(subset=['Apertura'])
    
    # Mapeo de Regiones
    regions = {'GBA': 'GBA', 'Pampeana': 'Pampeana', 'Noroeste': 'NOA', 'Noreste': 'NEA', 'Cuyo': 'Cuyo', 'Patagonia': 'Patagonia'}
    main_divs = ['Alimentos y bebidas no alcohólicas', 'Bebidas alcohólicas y tabaco', 'Prendas de vestir y calzado', 
                'Vivienda, agua, electricidad, gas y otros combustibles', 'Equipamiento y mantenimiento del hogar', 
                'Salud', 'Transporte', 'Comunicación', 'Recreación y cultura', 'Educación', 'Restaurantes y hoteles', 'Bienes y servicios varios']

    processed = []
    date_cols = df_var.columns[1:]
    
    for r_name, r_code in regions.items():
        region_mask = df_var['Apertura'].str.contains(f'Región {r_name}', na=False)
        region_df = df_var[region_mask].copy()
        region_df['Apertura'] = region_df['Apertura'].str.replace(f'Región {r_name}-', '', regex=False).str.strip()
        
        r_weights = weights.set_index('Divisiones')[r_code].to_dict()
        
        for _, row in region_df.iterrows():
            apert = row['Apertura']
            if apert == 'Nivel general' or apert in main_divs:
                for date in date_cols:
                    val = pd.to_numeric(str(row[date]).replace('///', 'NaN'), errors='coerce')
                    processed.append({
                        'Region': r_name, 'Apertura': apert, 'Fecha': date,
                        'Variacion': val, 'Peso': r_weights.get(apert, 0) if apert != 'Nivel general' else 100
                    })

    full_df = pd.DataFrame(processed)
    full_df['Fecha'] = pd.to_datetime(full_df['Fecha'])
    latest = full_df['Fecha'].max()
    df_24m = full_df[full_df['Fecha'] >= (latest - pd.DateOffset(months=23))]
    
    return full_df, df_24m, latest

df_all, df_24m, latest_date = load_and_process_data()

if df_all is not None:
    # --- 3. SELECTOR DE REGIÓN ---
    selected_region = st.sidebar.selectbox("Seleccionar Región de Análisis", df_all['Region'].unique())
    
    # --- 4. SECCIÓN 1: CAUSALIDAD (Basada en Incidencia Real) ---
    st.header(f"1. Causalidad del Impacto - Región {selected_region}")
    
    # Cálculo de Incidencia para el último mes
    df_latest = df_all[(df_all['Region'] == selected_region) & (df_all['Fecha'] == latest_date)].copy()
    ng_val = df_latest[df_latest['Apertura'] == 'Nivel general']['Variacion'].values[0]
    
    df_inc = df_latest[df_latest['Apertura'] != 'Nivel general'].copy()
    df_inc['Incidencia'] = (df_inc['Variacion'] * df_inc['Peso']) / 100
    df_inc['Contribucion'] = (df_inc['Incidencia'] / ng_val) * 100
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("Composición Actual")
        st.write(f"Inflación Mensual: **{ng_val}%**")
        fig_pie = px.pie(df_inc, names='Apertura', values='Contribucion', hole=0.4, 
                        color_discrete_sequence=COLORS['palette'])
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False, height=350)
        st.plotly_chart(fig_pie, width='stretch')

    with col_b:
        st.subheader("Histórico de Causalidad (Últimos 6 meses)")
        h_dates = sorted(df_all['Fecha'].unique())[-6:]
        df_h = df_all[(df_all['Region'] == selected_region) & (df_all['Fecha'].isin(h_dates))].copy()
        
        hist_inc = []
        for d in h_dates:
            d_df = df_h[df_h['Fecha'] == d].copy()
            total = d_df[d_df['Apertura'] == 'Nivel general']['Variacion'].values[0]
            divs = d_df[d_df['Apertura'] != 'Nivel general'].copy()
            divs['Cont_Pct'] = ((divs['Variacion'] * divs['Peso']) / 100 / total) * 100
            hist_inc.append(divs)
        
        df_hist_plot = pd.concat(hist_inc)
        fig_area = px.area(df_hist_plot, x='Fecha', y='Cont_Pct', color='Apertura',
                            color_discrete_sequence=COLORS['palette'])
        fig_area.update_layout(height=350, margin=dict(t=20, b=20), xaxis_title=None, yaxis_title="% Contribución")
        st.plotly_chart(fig_area, width='stretch')

    # --- 5. SECCIÓN 2: SERIE HISTÓRICA 24M ---
    st.divider()
    st.subheader("2. Evolución Nivel General (Ventana 24 Meses)")
    df_ng_24 = df_24m[(df_24m['Region'] == selected_region) & (df_24m['Apertura'] == 'Nivel general')]
    fig_line = px.line(df_ng_24, x='Fecha', y='Variacion', markers=True)
    fig_line.update_traces(line_color=COLORS['primary'], line_width=4)
    fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis_title="% Mensual")
    st.plotly_chart(fig_line, width='stretch')
# --- 6. SECCIÓN 3: RED DE NODOS  (Rubros con más impacto) ---
    st.divider()
    st.subheader("3. Red de Propagación de Impactos")

    top_divs = df_inc.nlargest(5, 'Peso')

    fig_net = go.Figure()

    # Centro: Nivel General (Letra y nodo más grandes)
    fig_net.add_trace(go.Scatter(
        x=[0.5], y=[0.5], 
        mode='markers+text', 
        text=['<b>IPC</b>'], 
        marker=dict(size=80, color=COLORS['primary']), # Aumentado de 60 a 80
        textposition="middle center",
        textfont=dict(size=18, color="white"), # Fuente interna destacada
        hoverinfo='none'
    ))

    # Nodos periféricos
    radius = 0.25 # Aumentado ligeramente para dar espacio a nodos más grandes
    angles = np.linspace(0, 2*np.pi, 6)[:-1]

    for i, (idx, row) in enumerate(top_divs.iterrows()):
        x, y = 0.5 + radius*np.cos(angles[i]), 0.5 + radius*np.sin(angles[i])
        
        # Línea de conexión más gruesa
        fig_net.add_trace(go.Scatter(
            x=[0.5, x], y=[0.5, y], 
            mode='lines', 
            line=dict(color='#555', width=2),
            hoverinfo='none'
        ))
        
        display_text = row['Apertura'].replace(' y ', ' y<br>').replace(' de ', ' de<br>').replace(' para ', ' para<br>')
        
        # Escalado de tamaño de nodo aumentado
        # Multiplicador sube de 150 a 250
        node_size = min(max(row['Peso'] * 250, 40), 80) 
        
        fig_net.add_trace(go.Scatter(
            x=[x], y=[y], 
            mode='markers+text', 
            text=[f"<b>{display_text}</b>"], 
            marker=dict(
                size=node_size, 
                color=COLORS['secondary'], 
                line=dict(width=3, color='#fff')
            ), 
            textposition="bottom center",
            textfont=dict(size=13), # Fuente de etiquetas más grande
            hoverinfo='text',
            hovertext=f"{row['Apertura']}: {row['Peso']}%"
        ))

    # Ajustes de diseño
    fig_net.update_layout(
        xaxis=dict(range=[0.05, 0.95], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[0.05, 0.95], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        margin=dict(l=50, r=50, t=50, b=50), 
        height=600, # Aumentamos la altura del lienzo para acomodar el tamaño extra
        showlegend=False, 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color="white") # Tamaño base de fuente aumentado
    )

    st.plotly_chart(fig_net, width='stretch', config={'displayModeBar': False})

    # --- 7. SECCIÓN 4: TABLA Y PROYECCIÓN ---
    st.divider()
    c_tab, c_met = st.columns([2, 1])
    
    with c_tab:
        st.subheader("4. Detalle de Variaciones por Rubro")
        df_table = df_latest[['Apertura', 'Variacion', 'Peso']].sort_values('Variacion', ascending=False)
        st.dataframe(df_table, width='stretch', height=400)
        
    with c_met:
        st.subheader("Métrica Predictiva")
        avg_6m = df_all[(df_all['Region'] == selected_region) & (df_all['Apertura'] == 'Nivel general')].tail(6)['Variacion'].mean()
        forecast = avg_6m * 0.95
        
        st.markdown(f"""
        <div class="metric-container">
            <p style="margin:0; opacity:0.7;">PROMEDIO PROYECTADO 2026</p>
            <h2 style="margin:0; color:{COLORS['primary']};">{forecast:.2f}%</h2>
            <p style="color:{COLORS['accent_green']};">Tendencia: Desaceleración suave</p>
        </div>
        """, unsafe_allow_html=True)