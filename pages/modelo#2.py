import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import entropy
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide", page_title="Analytics-IPC-model Torch-entropy", page_icon=":material/analytics:")

# --- PROCESAMIENTO DE DATOS ---

def force_to_date(value):
    try:
        dt = pd.to_datetime(value, dayfirst=True, errors='coerce')
        if pd.notna(dt):
            return dt.replace(day=1)
        serial = float(value)
        dt = pd.to_datetime('1899-12-30') + pd.to_timedelta(serial, 'D')
        return dt.replace(day=1)
    except:
        return pd.NaT

@st.cache_data
def load_and_transform_ipc(file_path):
    if not os.path.exists(file_path):
        st.error(f"Archivo no encontrado: {file_path}")
        return None
    try:
        raw_df = pd.read_excel(file_path, sheet_name="Variación mensual aperturas", engine='openpyxl')
        header_row_index = 0
        for i, row in raw_df.iterrows():
            potential_dates = [force_to_date(val) for val in row if pd.notna(force_to_date(val))]
            if len(potential_dates) > 3:
                header_row_index = i + 1
                break
        
        df = pd.read_excel(file_path, sheet_name="Variación mensual aperturas", 
                        skiprows=header_row_index, engine='openpyxl')
        df = df.rename(columns={df.columns[0]: 'Rubro'})
        df['Rubro'] = df['Rubro'].astype(str).str.strip()

        date_mapping = {col: force_to_date(col) for col in df.columns[1:] if pd.notna(force_to_date(col))}
        
        df_long = df.melt(id_vars=['Rubro'], value_vars=list(date_mapping.keys()), 
                        var_name='Original', value_name='Variacion')
        df_long['Fecha'] = df_long['Original'].map(date_mapping)
        df_long['Variacion'] = pd.to_numeric(df_long['Variacion'], errors='coerce')
        
        # Limpieza de nulos
        df_clean = df_long.dropna(subset=['Variacion', 'Fecha'])
        
        # ELIMINAR DUPLICADOS AQUÍ TAMBIÉN (Por si el Excel tiene filas repetidas)
        df_clean = df_clean.groupby(['Rubro', 'Fecha'])['Variacion'].mean().reset_index()
        
        return df_clean.sort_values(['Rubro', 'Fecha'])
    except Exception as e:
        st.error(f"Error en carga: {e}")
        return None

# --- MODELOS Y MÉTRICAS ---

def get_physics_metrics(series):
    # Aceleración: Delta de la variación mensual (derivada segunda)
    accel = series.diff().iloc[-1] if len(series) > 1 else 0
    # Entropía: Mide el desorden/incertidumbre
    counts = np.histogram(series, bins=10)[0]
    ent = entropy(counts) if np.sum(counts) > 0 else 0
    return accel, ent

def run_predictions(series, months=6):
    # Forzamos frecuencia mensual. Al haber limpiado duplicados antes, ya no fallará.
    series = series.asfreq('MS')
    preds = {}
    
    # SARIMA
    try:
        model_s = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False).fit(disp=False)
        preds['SARIMA'] = model_s.get_forecast(steps=months).predicted_mean
    except:
        preds['SARIMA'] = pd.Series([series.mean()]*months, index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq='MS'))
    
    # HOLT-WINTERS
    try:
        model_hw = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12).fit()
        preds['Holt-Winters'] = model_hw.forecast(months)
    except:
        preds['Holt-Winters'] = pd.Series([series.iloc[-1]]*months, index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq='MS'))
        
    return preds

# --- INTERFAZ ---

st.subheader("Dashboard Analisis Variación-IPC-Modelos SARIMA HOLT-WINTERS")

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "../data", "ipc.xlsx")

data = load_and_transform_ipc(file_path)

if data is not None:
    st.sidebar.subheader("Variación IPC Modelos SARMIA- HOLT WINTERS")
    rubros = [r for r in sorted(data['Rubro'].unique()) if len(r) > 3 and "Fuente" not in r]
    selected = st.sidebar.selectbox("Rubro / Región", rubros)
    
    subset = data[data['Rubro'] == selected]
    
    # SEGURO CONTRA DUPLICADOS: Agrupamos por fecha antes de convertir en Serie
    ts_data = subset.groupby('Fecha')['Variacion'].mean()

    if not ts_data.empty:
        accel, ent = get_physics_metrics(ts_data)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Último Mes", f"{ts_data.iloc[-1]:.1f}%")
        m2.metric("Aceleración", f"{accel:.2f} pp", delta=f"{accel:.2f}", delta_color="inverse")
        m3.metric("Entropía (Caos)", f"{ent:.2f}")
        m4.metric("Volatilidad Hist.", f"{ts_data.std():.2f}")

        # Predicciones
        predictions = run_predictions(ts_data)
        f_dates = pd.date_range(ts_data.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq='MS')

        # Gráfico Comparativo
        fig = go.Figure()
        hist = ts_data.tail(24)
        fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Histórico Real", line=dict(color='white', width=2)))
        fig.add_trace(go.Scatter(x=f_dates, y=predictions['SARIMA'], name="SARIMA (Estructural)", 
                                line=dict(color='#00d1b2', dash='dot')))
        fig.add_trace(go.Scatter(x=f_dates, y=predictions['Holt-Winters'], name="Holt-Winters (Tendencia)", 
                                line=dict(color='#ff3860')))
        st.subheader(f"Proyección 6 Meses: {selected}")
        fig.update_layout(template="plotly_dark", title=f"Proyección 6 Meses: {selected}", 
                        hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width='stretch')
        
        with st.expander("Detalle de Valores Proyectados"):
            df_comp = pd.DataFrame({
                "Mes": f_dates.strftime('%m/%Y'),
                "SARIMA (%)": predictions['SARIMA'].values,
                "Holt-Winters (%)": predictions['Holt-Winters'].values
            }).round(2)
            st.dataframe(df_comp, width='stretch')

#---------------------------------------------------------------------------------------------
# usando modelo predicitivo con Torch
#st.set_page_config(layout="wide", page_title="Fintech Analytics - IPC Predictor")
#st.divider()
st.markdown("---")
# --- PROCESAMIENTO DE DATOS (VERSION FINAL SIN DUPLICADOS) ---

def force_to_date(value):
    try:
        dt = pd.to_datetime(value, errors='coerce')
        if pd.notna(dt): return dt.replace(day=1)
        serial = float(value)
        dt = pd.to_datetime('1899-12-30') + pd.to_timedelta(serial, 'D')
        return dt.replace(day=1)
    except: return pd.NaT

@st.cache_data
def load_and_transform_ipc(file_path):
    if not os.path.exists(file_path): return None
    try:
        raw_df = pd.read_excel(file_path, sheet_name="Variación mensual aperturas", header=None, engine='openpyxl')
        
        # Buscar fila de fechas
        header_idx = 0
        for i in range(len(raw_df)):
            if pd.notna(force_to_date(raw_df.iloc[i, 1])):
                header_idx = i
                break
        
        dates_row = raw_df.iloc[header_idx, 1:]
        parsed_dates = [force_to_date(d) for d in dates_row]
        
        data_rows = []
        unique_names = []
        current_region = "Nivel Nacional"

        # Lógica para evitar duplicados combinando Región + Categoría
        for i in range(header_idx + 1, len(raw_df)):
            name = str(raw_df.iloc[i, 0]).strip()
            if name == "nan" or "Fuente:" in name: continue
            
            if "Región" in name:
                current_region = name
                display_name = name # Es la cabecera de la región
            else:
                # Si es una categoría (como Nivel General), le pegamos la región
                display_name = f"{current_region} - {name}"
            
            unique_names.append(display_name)
            data_rows.append(raw_df.iloc[i, 1:].values)
        
        final_df = pd.DataFrame(data_rows, index=unique_names, columns=parsed_dates).T
        final_df.index = pd.to_datetime(final_df.index)
        return final_df.sort_index().apply(pd.to_numeric, errors='coerce').dropna(how='all')
    except Exception as e:
        st.error(f"Error en carga: {e}")
        return None

# --- MODELO PYTORCH LSTM ---

class PreciosLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1, :])

def train_and_predict_lstm(ts_data, periods=6, seq_length=12):
    # Aseguramos que ts_data sea una Serie de números
    values = ts_data.values.astype(float)
    if len(values) <= seq_length:
        return np.array([values[-1]] * periods)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i+seq_length])
        y.append(data_scaled[i+seq_length])
    
    X_tensor = torch.FloatTensor(np.array(X))
    y_tensor = torch.FloatTensor(np.array(y))

    model = PreciosLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(50): # Epochs optimizados para velocidad
        optimizer.zero_grad()
        loss = loss_fn(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    last_seq = data_scaled[-seq_length:].tolist()
    preds = []
    for _ in range(periods):
        seq = torch.FloatTensor(last_seq[-seq_length:]).unsqueeze(0)
        with torch.no_grad():
            nxt = model(seq).item()
            preds.append(nxt)
            last_seq.append([nxt])
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# --- INTERFAZ ---

st.subheader("Dashboard: Predictor de variación de IPC Inflación (IA + Stats)")

# Ruta automática
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/ipc.xlsx")
df_ipc = load_and_transform_ipc(file_path)

if df_ipc is not None and not df_ipc.empty:
    st.sidebar.subheader("Predictor de Variación IPC (IA-Stats)")
    selected = st.sidebar.selectbox("Seleccionar Serie (Región - Categoría)", df_ipc.columns)
    
    # IMPORTANTE: .squeeze() asegura que si hay un duplicado, tome solo uno, pero con la lógica de arriba ya no habrá.
    ts_data = df_ipc[selected].dropna()
    if isinstance(ts_data, pd.DataFrame):
        ts_data = ts_data.iloc[:, 0] # Si por algo sigue siendo DataFrame, toma la primera columna

    if len(ts_data) > 2:
        m1, m2, m3, m4 = st.columns(4)
        
        # Ahora last_val es garantizado un FLOAT
        last_val = float(ts_data.iloc[-1])
        prev_val = float(ts_data.iloc[-2])
        accel = last_val - prev_val
        
        m1.metric("Última Variación", f"{last_val:.2f}%")
        m2.metric("Aceleración", f"{accel:.2f} pp", delta=f"{accel:.2f}", delta_color="inverse")
        m3.metric("Entropía", f"{entropy(np.histogram(ts_data, bins=10)[0]):.2f}")
        m4.metric("Volatilidad", f"{ts_data.std():.2f}")

        with st.spinner('Entrenando Red Neuronal...'):
            # Predicción IA
            lstm_preds = train_and_predict_lstm(ts_data)
            
            # Predicción Estadística Simple (Holt-Winters)
            hw_model = ExponentialSmoothing(ts_data, seasonal='add', seasonal_periods=12).fit()
            hw_preds = hw_model.forecast(6)
            
            f_dates = pd.date_range(ts_data.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq='MS')

        # Gráfico
        st.subheader(f"Proyección 6 Meses: {selected}")
        fig = go.Figure()
        hist = ts_data.tail(24)
        fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Histórico", line=dict(color='white', width=3)))
        fig.add_trace(go.Scatter(x=f_dates, y=hw_preds, name="Estadística (H-W)", line=dict(dash='dot', color='cyan')))
        fig.add_trace(go.Scatter(x=f_dates, y=lstm_preds, name="IA (PyTorch LSTM)", line=dict(color='#b83280', width=4)))
        
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, width='stretch')
        
        # Tabla
        st.write("**Proyección de IA para los próximos 6 meses:**")
        res_df = pd.DataFrame({"Mes": f_dates.strftime('%Y-%m'), "Predicción IA (%)": lstm_preds})
        st.dataframe(res_df.T)

else:
    st.error("No se encontró 'ipc.xlsx'. Verifica el nombre del archivo.")

st.write("   ")

st.divider()    