import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Predicción IPC - IA Robusta", page_icon=":material/analytics:")

# --- INICIALIZACIÓN DE ESTADOS (Para evitar que los gráficos se borren) ---
if 'res_ia_robusta' not in st.session_state:
    st.session_state.res_ia_robusta = None
if 'res_ia_entrenamiento' not in st.session_state:
    st.session_state.res_ia_entrenamiento = None

# --- 1. CARGA Y TRANSFORMACIÓN DE DATOS ---
def force_to_date(value):
    try:
        dt = pd.to_datetime(value, dayfirst=True, errors='coerce')
        if pd.notna(dt): return dt.replace(day=1)
        serial = float(value)
        dt = pd.to_datetime('1899-12-30') + pd.to_timedelta(serial, 'D')
        return dt.replace(day=1)
    except:
        return pd.NaT

@st.cache_data
def load_and_transform_ipc(file_path):
    if not os.path.exists(file_path): return None
    try:
        raw_df = pd.read_excel(file_path, sheet_name="Variación mensual aperturas", engine='openpyxl')
        header_row_index = 0
        for i, row in raw_df.iterrows():
            potential_dates = [force_to_date(val) for val in row if pd.notna(force_to_date(val))]
            if len(potential_dates) > 3:
                header_row_index = i + 1
                break
        df = pd.read_excel(file_path, sheet_name="Variación mensual aperturas", skiprows=header_row_index, engine='openpyxl')
        df = df.rename(columns={df.columns[0]: 'Rubro'})
        df['Rubro'] = df['Rubro'].astype(str).str.strip()
        date_mapping = {col: force_to_date(col) for col in df.columns[1:] if pd.notna(force_to_date(col))}
        df_long = df.melt(id_vars=['Rubro'], value_vars=list(date_mapping.keys()), var_name='Original', value_name='Variacion')
        df_long['Fecha'] = df_long['Original'].map(date_mapping)
        df_long['Variacion'] = pd.to_numeric(df_long['Variacion'], errors='coerce')
        df_clean = df_long.dropna(subset=['Variacion', 'Fecha'])
        df_clean = df_clean.groupby(['Rubro', 'Fecha'])['Variacion'].mean().reset_index()
        return df_clean.sort_values(['Rubro', 'Fecha'])
    except Exception as e:
        st.error(f"Error al procesar el Excel: {e}")
        return None

def get_physics_metrics(series):
    accel = series.diff().iloc[-1] if len(series) > 1 else 0
    counts = np.histogram(series, bins=10)[0]
    ent = entropy(counts) if np.sum(counts) > 0 else 0
    return accel, ent

# --- 2. ARQUITECTURAS DE REDES NEURONALES ---

# Modelo 1: Robusto con Dropout para Monte Carlo
class RobustLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.linear(out)

# Modelo 2: Entrenamiento vs Ajuste
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

# --- 3. FUNCIONES DE PROCESAMIENTO ---

def run_mc_model(ts_data, periods=6, seq_len=12):
    values = ts_data.values.reshape(-1, 1).astype(float)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)
    X, y = [], []
    for i in range(len(scaled)-seq_len):
        X.append(scaled[i:i+seq_len]); y.append(scaled[i+seq_len])
    X, y = torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
    
    model = RobustLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.SmoothL1Loss()
    
    for _ in range(100):
        model.train(); opt.zero_grad(); loss_fn(model(X), y).backward(); opt.step()
    
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'): m.train()
    
    sims = []
    for _ in range(30):
        curr = scaled[-seq_len:].tolist()
        path = []
        for _ in range(periods):
            with torch.no_grad():
                nxt = model(torch.FloatTensor([curr[-seq_len:]])).item()
                path.append(nxt); curr.append([nxt])
        sims.append(path)
    
    sims = scaler.inverse_transform(np.array(sims).reshape(-1, 1)).reshape(30, periods)
    return np.mean(sims, axis=0), np.percentile(sims, 10, axis=0), np.percentile(sims, 90, axis=0)

def run_train_fit_model(ts_data, epochs=50, seq_len=12, periods=6):
    values = ts_data.values.reshape(-1, 1).astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    X, y = [], []
    for i in range(len(scaled)-seq_len):
        X.append(scaled[i:i+seq_len]); y.append(scaled[i+seq_len])
    X, y = torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
    
    model = SimpleLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        model.train(); opt.zero_grad(); nn.MSELoss()(model(X), y).backward(); opt.step()
    
    model.eval()
    with torch.no_grad():
        fit = scaler.inverse_transform(model(X).numpy())
    
    curr = scaled[-seq_len:].tolist()
    preds = []
    for _ in range(periods):
        with torch.no_grad():
            nxt = model(torch.FloatTensor([curr[-seq_len:]])).item()
            preds.append(nxt); curr.append([nxt])
    
    return fit, scaler.inverse_transform(np.array(preds).reshape(-1, 1))

# --- 4. INTERFAZ DE USUARIO ---
st.title("Proyección de Inflación con Inteligencia Artificial")

file_path = os.path.join(os.getcwd(), "data", "ipc.xlsx")
df_ipc = load_and_transform_ipc(file_path)

if df_ipc is not None:
    rubros = [r for r in sorted(df_ipc['Rubro'].unique()) if len(r) > 3]
    sel_rubro = st.sidebar.selectbox("Rubro:", rubros)
    meses_f = st.sidebar.slider("Meses proyección:", 3, 6, 6)
    epocas_sel = st.sidebar.select_slider("Épocas Entrenamiento:", options=[30, 50, 100], value=50)

    ts_data = df_ipc[df_ipc['Rubro'] == sel_rubro].groupby('Fecha')['Variacion'].mean().asfreq('MS').ffill()

    # Métricas superiores
    if len(ts_data) > 12:
        accel, ent = get_physics_metrics(ts_data)
        c1, c2, c3 = st.columns(3)
        c1.metric("Último IPC", f"{ts_data.iloc[-1]:.2f}%")
        c2.metric("Aceleración", f"{accel:.2f} pp")
        c3.metric("Entropía (Caos)", f"{ent:.2f}")

        # BOTONES DE ACCIÓN
        col_btn1, col_btn2 = st.columns(2)
        
        if col_btn1.button("Ejecutar IA Robusta (Monte Carlo)", type="primary", width='stretch'):
            m, l, h = run_mc_model(ts_data, meses_f)
            st.session_state.res_ia_robusta = (m, l, h)

        if col_btn2.button(f"Ejecutar Entrenamiento ({epocas_sel} Épocas)", width='stretch'):
            fit, fut = run_train_fit_model(ts_data, epocas_sel, 12, meses_f)
            st.session_state.res_ia_entrenamiento = (fit, fut)

        st.divider()

        # --- DESPLIEGUE DE GRÁFICOS (Persistentes) ---

        # 1. Gráfico de IA Robusta
        if st.session_state.res_ia_robusta:
            st.subheader("Análisis de Incertidumbre (Monte Carlo)")
            m, l, h = st.session_state.res_ia_robusta
            f_dates = pd.date_range(ts_data.index[-1] + pd.offsets.MonthBegin(1), periods=meses_f, freq='MS')
            
            fig1 = go.Figure()
            hist = ts_data.tail(24)
            xr = [hist.index[-1]] + list(f_dates)
            
            fig1.add_trace(go.Scatter(x=xr+xr[::-1], y=list(h)+list(l)[::-1], fill='toself', fillcolor='rgba(255,0,127,0.1)', line=dict(color='rgba(0,0,0,0)'), name="Rango 80%"))
            fig1.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Real", line=dict(color='#00d1ff')))
            fig1.add_trace(go.Scatter(x=xr, y=[hist.values[-1]]+list(m), name="IA Robusta", line=dict(dash='dot', color='#ff007f')))
            fig1.update_layout(template="plotly_dark", height=400, margin=dict(t=20, b=20))
            st.plotly_chart(fig1, width='stretch')

        # 2. Gráfico de Entrenamiento vs Predicción
        if st.session_state.res_ia_entrenamiento:
            st.subheader(f"Ajuste de Entrenamiento vs Proyección ({epocas_sel} épocas)")
            fit, fut = st.session_state.res_ia_entrenamiento
            f_dates = pd.date_range(ts_data.index[-1] + pd.offsets.MonthBegin(1), periods=meses_f, freq='MS')
            
            fig2 = go.Figure()
            # Datos de entrenamiento (se saltan los primeros 12 meses por la ventana de tiempo)
            fit_dates = ts_data.index[12:]
            
            fig2.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, name="Histórico Real", line=dict(color='#00d1ff', width=1)))
            fig2.add_trace(go.Scatter(x=fit_dates, y=fit.flatten(), name="Ajuste del Modelo (Fit)", line=dict(color='#ffa500', dash='dot')))
            
            xr2 = [ts_data.index[-1]] + list(f_dates)
            yr2 = [ts_data.values[-1]] + list(fut.flatten())
            fig2.add_trace(go.Scatter(x=xr2, y=yr2, name="Predicción", line=dict(color='#ff007f', width=3)))
            
            fig2.update_layout(template="plotly_dark", height=400, margin=dict(t=20, b=20))
            st.plotly_chart(fig2, width='stretch')
else:
    st.error("Archivo no encontrado en /data/ipc.xlsx")