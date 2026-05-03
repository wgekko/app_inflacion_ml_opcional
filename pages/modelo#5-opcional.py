#--------------------------------------------------------
# primer modelo de prediccion 

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import os
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import MinMaxScaler
# from scipy.stats import entropy

# # --- CONFIGURACIÓN ---
# st.set_page_config(layout="wide", page_title="Predicción IPC-IA", page_icon=":material/memory:")

# # --- CARGA DE DATOS ---
# def force_to_date(value):
#     try:
#         dt = pd.to_datetime(value, dayfirst=True, errors='coerce')
#         if pd.notna(dt): return dt.replace(day=1)
#         serial = float(value)
#         dt = pd.to_datetime('1899-12-30') + pd.to_timedelta(serial, 'D')
#         return dt.replace(day=1)
#     except:
#         return pd.NaT

# @st.cache_data
# def load_and_transform_ipc(file_path):
#     # Carga y limpieza adaptada de tu código base[cite: 3]
#     if not os.path.exists(file_path): return None
#     try:
#         raw_df = pd.read_excel(file_path, sheet_name="Variación mensual aperturas", engine='openpyxl')
#         header_row_index = 0
#         for i, row in raw_df.iterrows():
#             potential_dates = [force_to_date(val) for val in row if pd.notna(force_to_date(val))]
#             if len(potential_dates) > 3:
#                 header_row_index = i + 1
#                 break
        
#         df = pd.read_excel(file_path, sheet_name="Variación mensual aperturas", skiprows=header_row_index, engine='openpyxl')
#         df = df.rename(columns={df.columns[0]: 'Rubro'})
#         df['Rubro'] = df['Rubro'].astype(str).str.strip()

#         date_mapping = {col: force_to_date(col) for col in df.columns[1:] if pd.notna(force_to_date(col))}
#         df_long = df.melt(id_vars=['Rubro'], value_vars=list(date_mapping.keys()), var_name='Original', value_name='Variacion')
#         df_long['Fecha'] = df_long['Original'].map(date_mapping)
#         df_long['Variacion'] = pd.to_numeric(df_long['Variacion'], errors='coerce')
        
#         df_clean = df_long.dropna(subset=['Variacion', 'Fecha'])
#         df_clean = df_clean.groupby(['Rubro', 'Fecha'])['Variacion'].mean().reset_index()
#         return df_clean.sort_values(['Rubro', 'Fecha'])
#     except Exception as e:
#         return None

# def get_physics_metrics(series):
#     # Cálculo de métricas extraídas de tu análisis previo[cite: 3]
#     accel = series.diff().iloc[-1] if len(series) > 1 else 0
#     counts = np.histogram(series, bins=10)[0]
#     ent = entropy(counts) if np.sum(counts) > 0 else 0
#     return accel, ent

# # --- ARQUITECTURA ROBUSTA LSTM ---
# class RobustLSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=128, num_layers=2, output_size=1, dropout=0.2):
#         super().__init__()
#         # Múltiples capas LSTM con Dropout para evitar overfitting
#         self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, 
#                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
#         self.linear = nn.Linear(hidden_layer_size, output_size)

#     def forward(self, input_seq):
#         lstm_out, _ = self.lstm(input_seq)
#         return self.linear(lstm_out[:, -1, :])

# def train_and_predict_robust_lstm(ts_data, periods=6, seq_length=12):
#     values = ts_data.values.astype(float)
#     if len(values) <= seq_length:
#         return np.array([values[-1]] * periods)

#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     data_scaled = scaler.fit_transform(values.reshape(-1, 1))

#     X, y = [], []
#     for i in range(len(data_scaled) - seq_length):
#         X.append(data_scaled[i:i+seq_length])
#         y.append(data_scaled[i+seq_length])
    
#     X_tensor = torch.FloatTensor(np.array(X))
#     y_tensor = torch.FloatTensor(np.array(y))

#     # Inicializar modelo robusto
#     model = RobustLSTM(num_layers=2, hidden_layer_size=64, dropout=0.2)
    
#     # Adam optimizador con Weight Decay (L2 Penalty)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
#     # Huber Loss (SmoothL1Loss) tolera mejor los outliers inflacionarios
#     loss_fn = nn.SmoothL1Loss()

#     # Entrenamiento
#     epochs = 150
#     progress_bar = st.progress(0, text="Entrenando Red Neuronal...")
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         predictions = model(X_tensor)
#         loss = loss_fn(predictions, y_tensor)
#         loss.backward()
#         optimizer.step()
        
#         if epoch % 15 == 0:
#             progress_bar.progress(epoch / epochs, text=f"Entrenando Red Neuronal... (Época {epoch}/{epochs})")
            
#     progress_bar.empty()

#     # Predicción Autorregresiva
#     model.eval()
#     last_seq = data_scaled[-seq_length:].tolist()
#     preds = []
    
#     for _ in range(periods):
#         seq = torch.FloatTensor(last_seq[-seq_length:]).unsqueeze(0)
#         with torch.no_grad():
#             nxt = model(seq).item()
#             preds.append(nxt)
#             last_seq.append([nxt])
            
#     return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# # --- INTERFAZ DE USUARIO ---
# st.title("Proyección de Inflación mediante Deep Learning")

# # Guía de usuario no experto (UX)
# with st.expander(":material/article: Guía rápida para interpretar este panel (Para nuevos usuarios)"):
#     st.markdown("""
#     * **¿Qué es la Aceleración?** Imagina la inflación como un vehículo. Si los precios suben de 4% a 6%, el vehículo está *acelerando*. Si baja de 6% a 5%, sigue avanzando (los precios suben), pero está *frenando* (desaceleración).
#     * **¿Qué es la Entropía?** Mide la imprevisibilidad o el "caos" en los datos. Una entropía alta significa que los precios saltan bruscamente sin un patrón claro, lo que hace el mercado más incierto.
#     * **Red Neuronal LSTM:** Funciona leyendo la "historia" de los precios de los últimos meses, aprendiendo qué recordar y qué olvidar, para deducir de forma inteligente el comportamiento del próximo semestre.
#     """)

# root_dir = os.getcwd() 
# file_path = os.path.join(root_dir, "data", "ipc.xlsx")
# df_ipc = load_and_transform_ipc(file_path)

# if df_ipc is not None and not df_ipc.empty:
#     rubros = [r for r in sorted(df_ipc['Rubro'].unique()) if len(r) > 3]
#     selected = st.sidebar.selectbox("Seleccionar Rubro a Analizar:", rubros)
#     meses_pred = st.sidebar.slider("Meses a proyectar:", 3, 6, 6)
    
#     ts_data = df_ipc[df_ipc['Rubro'] == selected].groupby('Fecha')['Variacion'].mean().asfreq('MS').ffill()

#     if len(ts_data) > 12:
#         accel, ent = get_physics_metrics(ts_data)
        
#         st.subheader(f"Análisis Actual: {selected}")
#         m1, m2, m3, m4 = st.columns(4)
#         m1.metric("Última Variación IPC", f"{ts_data.iloc[-1]:.2f}%")
#         m2.metric("Aceleración", f"{accel:.2f} pp", delta=f"{accel:.2f}", delta_color="inverse")
#         m3.metric("Entropía (Caos)", f"{ent:.2f}")
#         m4.metric("Volatilidad (σ)", f"{ts_data.std():.2f}")

#         st.divider()

#         if st.button("Generar Proyección con IA", type="primary"):
#             lstm_preds = train_and_predict_robust_lstm(ts_data, periods=meses_pred)
#             f_dates = pd.date_range(ts_data.index[-1] + pd.offsets.MonthBegin(1), periods=meses_pred, freq='MS')

#             st.success("¡Modelo entrenado exitosamente!")
            
#             # Gráfico
#             fig = go.Figure()
#             hist = ts_data.tail(24)
#             fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Histórico Real", line=dict(color='#00d1ff', width=3)))
            
#             # Conectar la línea de predicción con el último dato real
#             x_pred = [hist.index[-1]] + list(f_dates)
#             y_pred = [hist.values[-1]] + list(lstm_preds)
            
#             fig.add_trace(go.Scatter(x=x_pred, y=y_pred, name="Predicción LSTM (Robusta)", 
#                                      line=dict(dash='dot', color='#ff007f', width=4),
#                                      mode='lines+markers+text',
#                                      text=[""] + [f"{p:.1f}%" for p in lstm_preds],
#                                      textposition="top center"))
            
#             fig.update_layout(template="plotly_dark", height=500, hovermode="x unified")
#             st.plotly_chart(fig, width='stretch')

# else:
#     st.error("No se pudo procesar el archivo 'ipc.xlsx'. Verifica que esté en la carpeta 'data/'.")


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
st.set_page_config(layout="wide", page_title="Predicción IPC-IA", page_icon=":material/analytics:")

# --- 1. CARGA Y TRANSFORMACIÓN DE DATOS ---
def force_to_date(value):
    """Convierte valores de Excel (fechas o números seriales) a objetos datetime de Python."""
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
    """
    Carga el archivo Excel de IPC y lo transforma a formato largo (long format) para el modelo[cite: 3].
    Se enfoca en la pestaña 'Variación mensual aperturas'.
    """
    if not os.path.exists(file_path): 
        return None
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

# --- 2. MÉTRICAS ANALÍTICAS (FÍSICA APLICADA) ---
def get_physics_metrics(series):
    """Calcula la aceleración y la entropía de la serie de tiempo para medir caos y tendencia[cite: 3]."""
    accel = series.diff().iloc[-1] if len(series) > 1 else 0
    # Cálculo de entropía basado en la distribución de las variaciones
    counts = np.histogram(series, bins=10)[0]
    ent = entropy(counts) if np.sum(counts) > 0 else 0
    return accel, ent

# --- 3. MODELADO DE DEEP LEARNING (LSTM + MC DROPOUT) ---
class RobustLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        # Stacked LSTM para capturar patrones complejos[cite: 3]
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        # Aplicamos dropout antes de la capa final para la técnica Monte Carlo
        out = self.dropout(lstm_out[:, -1, :])
        return self.linear(out)

def enable_dropout(model):
    """Mantiene activa la capa de Dropout durante la fase de predicción."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def train_and_predict_mc_dropout(ts_data, periods=6, seq_length=12, n_simulations=50):
    """
    Entrena la red y genera múltiples simulaciones para obtener escenarios optimistas y pesimistas.
    """
    values = ts_data.values.astype(float)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i+seq_length])
        y.append(data_scaled[i+seq_length])
    
    X_tensor = torch.FloatTensor(np.array(X))
    y_tensor = torch.FloatTensor(np.array(y))

    model = RobustLSTM(num_layers=2, hidden_layer_size=64, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss() # Huber Loss para mayor robustez ante outliers[cite: 1]

    # Entrenamiento rápido
    epochs = 120
    progress_bar = st.progress(0, text="Entrenando Red Neuronal...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 12 == 0:
            progress_bar.progress(epoch / epochs)
    progress_bar.empty()

    # Inferencia Monte Carlo
    model.eval()
    enable_dropout(model)
    
    all_sims = []
    for _ in range(n_simulations):
        last_seq = data_scaled[-seq_length:].tolist()
        path = []
        for _ in range(periods):
            seq = torch.FloatTensor(last_seq[-seq_length:]).unsqueeze(0)
            with torch.no_grad():
                nxt = model(seq).item()
                path.append(nxt)
                last_seq.append([nxt])
        all_sims.append(path)

    all_sims = np.array(all_sims)
    # Desescalar resultados
    sims_unscaled = scaler.inverse_transform(all_sims.reshape(-1, 1)).reshape(n_simulations, periods)
    
    return (np.mean(sims_unscaled, axis=0), 
            np.percentile(sims_unscaled, 10, axis=0), 
            np.percentile(sims_unscaled, 90, axis=0))

# --- 4. INTERFAZ DE USUARIO (UX) ---
st.title("Proyección de Inflación con Inteligencia Artificial")

# Sección de ayuda para usuarios no técnicos
with st.expander("ℹ:material/article: ¿Cómo interpretar estos indicadores analíticos?"):
    st.markdown("""
    *   **Aceleración:** Indica si la inflación está tomando impulso o perdiendo fuerza respecto al mes anterior. Si es positiva, el ritmo de aumento crece.
    *   **Entropía (Caos):** Cuantifica qué tan desordenados o impredecibles son los cambios de precios en este rubro. A mayor entropía, más difícil es predecir con exactitud.
    *   **Escenarios (Sombreado):** Dado que el futuro es incierto, el modelo no da un solo número. El área sombreada muestra dónde es más probable que se ubique la inflación (Escenario Base vs. Optimista/Pesimista).
    """)

# Carga de datos
root_dir = os.getcwd() 
file_path = os.path.join(root_dir, "data", "ipc.xlsx")
df_ipc = load_and_transform_ipc(file_path)

if df_ipc is not None and not df_ipc.empty:
    # Sidebar
    rubros = [r for r in sorted(df_ipc['Rubro'].unique()) if len(r) > 3]
    selected_rubro = st.sidebar.selectbox("Seleccione el Rubro:", rubros)
    meses_a_proyectar = st.sidebar.slider("Meses a proyectar:", 3, 6, 6)
    
    # Preparar Serie
    ts_data = df_ipc[df_ipc['Rubro'] == selected_rubro].groupby('Fecha')['Variacion'].mean().asfreq('MS').ffill()

    if len(ts_data) > 12:
        # Dashboard de Métricas
        accel, ent = get_physics_metrics(ts_data)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Último Registro", f"{ts_data.iloc[-1]:.2f}%")
        c2.metric("Aceleración", f"{accel:.2f} pp", delta=f"{accel:.2f}", delta_color="inverse")
        c3.metric("Entropía (Caos)", f"{ent:.2f}")
        c4.metric("Volatilidad Anual", f"{ts_data.tail(12).std():.2f}")

        st.divider()

        if st.button("Ejecutar Modelo Predictivo LSTM", type="primary"):
            # Predicción
            mean_p, low_p, high_p = train_and_predict_mc_dropout(ts_data, periods=meses_a_proyectar)
            f_dates = pd.date_range(ts_data.index[-1] + pd.offsets.MonthBegin(1), periods=meses_a_proyectar, freq='MS')

            # --- Visualización con Plotly[cite: 4] ---
            fig = go.Figure()
            hist = ts_data.tail(24) # Mostrar últimos 2 años para contexto

            # 1. Área de Confianza (Escenarios)
            x_range = [hist.index[-1]] + list(f_dates)
            y_upper = [hist.values[-1]] + list(high_p)
            y_lower = [hist.values[-1]] + list(low_p)
            y_mean  = [hist.values[-1]] + list(mean_p)

            fig.add_trace(go.Scatter(
                x=x_range + x_range[::-1],
                y=y_upper + y_lower[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 127, 0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                name="Rango de Probabilidad (80%)",
                hoverinfo="skip"
            ))

            # 2. Líneas de datos
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Datos Reales", line=dict(color='#00d1ff', width=3)))
            fig.add_trace(go.Scatter(x=x_range, y=y_mean, name="Proyección Base IA", 
                                     line=dict(dash='dot', color='#ff007f', width=4),
                                     mode='lines+markers+text',
                                     text=[""] + [f"{p:.1f}%" for p in mean_p],
                                     textposition="top center"))

            fig.update_layout(
                template="plotly_dark", 
                height=550, 
                hovermode="x unified",
                title=f"Proyección de Inflación: {selected_rubro}",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, width='stretch')

            # Detalle de Escenarios
            with st.expander(":material/table: Ver Tabla de Escenarios Proyectados"):
                df_res = pd.DataFrame({
                    "Fecha": f_dates.strftime('%m/%Y'),
                    "Optimista (Baja)": low_p,
                    "Base (Esperado)": mean_p,
                    "Pesimista (Alta)": high_p
                }).set_index("Fecha")
                st.table(df_res.style.format("{:.2f}%"))
    else:
        st.warning("Se requieren al menos 12 meses de datos históricos para entrenar el modelo.")
else:
    st.error("No se encontró el archivo 'ipc.xlsx' en la carpeta '/data/'. Por favor, cárgalo para continuar.")