import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os



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

# --- MODELO LSTM ESTÁNDAR ---
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# --- FUNCIÓN DE PROCESAMIENTO Y ENTRENAMIENTO ---
def run_train_test_model(ts_data, epochs=50, look_back=12, future_periods=6):
    dataset = ts_data.values.reshape(-1, 1).astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)

    # Crear secuencias para entrenamiento
    X, y = [], []
    for i in range(len(dataset_scaled) - look_back):
        X.append(dataset_scaled[i : i + look_back])
        y.append(dataset_scaled[i + look_back])
    
    X = torch.tensor(np.array(X))
    y = torch.tensor(np.array(y))

    model = SimpleLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Bucle de Entrenamiento
    progress_bar = st.progress(0)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        progress_bar.progress((epoch + 1) / epochs, text=f"Loss: {loss.item():.6f}")

    # 1. "Training Fit": Predicciones sobre los mismos datos de entrenamiento
    model.eval()
    with torch.no_grad():
        train_fit_scaled = model(X).numpy()
    train_fit = scaler.inverse_transform(train_fit_scaled)

    # 2. "Future Prediction": Proyectar hacia adelante
    last_window = dataset_scaled[-look_back:].tolist()
    future_preds_scaled = []
    
    for _ in range(future_periods):
        seq = torch.tensor([last_window[-look_back:]])
        with torch.no_grad():
            nxt = model(seq).item()
            future_preds_scaled.append(nxt)
            last_window.append([nxt])
            
    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))

    return train_fit, future_preds, look_back

# --- INTERFAZ STREAMLIT ---
st.subheader("Modelo de Entrenamiento vs. Predicción")

# Configuración en el Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros del Nuevo Modelo")
num_epochs = st.sidebar.select_slider("Número de Épocas:", options=[30, 50, 100], value=50)
pred_months = st.sidebar.slider("Meses a predecir:", 1, 12, 6)

if 'ts_data' in locals() or 'ts_data' in globals():
    if st.button(f"Entrenar con {num_epochs} Épocas"):
        train_fit, future_preds, lb = run_train_test_model(ts_data, epochs=num_epochs, future_periods=pred_months)
        
        # Preparar fechas
        history_dates = ts_data.index
        train_fit_dates = ts_data.index[lb:]
        future_dates = pd.date_range(ts_data.index[-1] + pd.offsets.MonthBegin(1), periods=pred_months, freq='MS')

        # --- GRÁFICO COMPARATIVO ---
        fig = go.Figure()

        # Datos Reales
        fig.add_trace(go.Scatter(x=history_dates, y=ts_data.values, name="Datos Reales (Histórico)", 
                                 line=dict(color='#00d1ff', width=2)))

        # Ajuste de Entrenamiento (Training Fit)
        fig.add_trace(go.Scatter(x=train_fit_dates, y=train_fit.flatten(), name="Ajuste de Entrenamiento (Fit)", 
                                 line=dict(color='#ffa500', dash='dot')))

        # Predicción Futura
        # Conectamos con el último punto real para continuidad visual
        x_future = [history_dates[-1]] + list(future_dates)
        y_future = [ts_data.values[-1]] + list(future_preds.flatten())
        
        fig.add_trace(go.Scatter(x=x_future, y=y_future, name="Valores Predictivos", 
                                 line=dict(color='#ff007f', width=4)))

        fig.update_layout(
            title=f"Ajuste de Red Neuronal y Proyección ({num_epochs} épocas)",
            template="plotly_dark",
            hovermode="x unified",
            xaxis_title="Fecha",
            yaxis_title="Variación IPC (%)"
        )

        st.plotly_chart(fig, width='stretch')
        
        # Mensaje de UX
        st.success(f"Modelo completado. La línea naranja muestra qué tan bien aprendió la IA de tus datos pasados.")
else:
    st.info("Carga el archivo IPC en la sección anterior para habilitar este análisis.")