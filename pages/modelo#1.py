import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# --- CONFIGURACIÓN ---
st.set_page_config(layout="wide", page_title="Analytics|Forecast Inflación", page_icon=":material/search_insights:")

@st.cache_data
def load_data_optimized(file_source):
    """Carga el Excel asegurando que la serie llegue hasta 2026."""
    try:
        xl = pd.ExcelFile(file_source, engine="openpyxl")
        sheet = "Variación mensual aperturas"
        
        if sheet not in xl.sheet_names:
            return f"ERROR: No se encontró la pestaña '{sheet}'"
            
        preview = xl.parse(sheet, nrows=20, header=None)
        header_row = 0
        for i, row in preview.iterrows():
            row_values = [str(val) for val in row.values if pd.notna(val)]
            if any("Nivel general" in s for s in row_values) or any("/" in s for s in row_values) or any("-" in s for s in row_values):
                header_row = i
                break
        
        df = xl.parse(sheet, skiprows=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        
        df = df.rename(columns={df.columns[0]: "Rubro"})
        df["Rubro"] = df["Rubro"].astype(str).str.strip()
        
        date_cols = []
        for col in df.columns:
            dt = pd.to_datetime(col, errors='coerce')
            if pd.notna(dt):
                date_cols.append(col)

        if not date_cols:
            return "NO_DATES"

        df_melted = df.melt(
            id_vars=["Rubro"],
            value_vars=date_cols,
            var_name="Fecha",
            value_name="Variacion"
        )
        
        df_melted["Fecha"] = pd.to_datetime(df_melted["Fecha"], errors="coerce")
        df_melted["Variacion"] = pd.to_numeric(df_melted["Variacion"], errors="coerce")
        
        return df_melted.dropna(subset=["Fecha", "Variacion"]).sort_values(["Rubro", "Fecha"])

    except Exception as e:
        return f"ERROR: {str(e)}"

def run_forecast_v2(series, steps=6):
    """Modelo Estadístico SARIMA"""
    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        forecast = res.get_forecast(steps=steps)
        mean = forecast.predicted_mean
        conf = forecast.conf_int(alpha=0.1)
        return mean, conf
    except:
        last_val = series.iloc[-1]
        mean = pd.Series([last_val] * steps)
        conf = pd.DataFrame({'lower Variacion': [last_val*0.8]*steps, 'upper Variacion': [last_val*1.2]*steps})
        return mean, conf

# def run_lstm_forecast(series, steps=6, lookback=12):
#     """
#     Modelo Univariado - Unistep con Redes LSTM (Deep Learning).
#     Entrena al vuelo con los datos del rubro seleccionado.
#     """
#     if len(series) <= lookback:
#         return pd.Series([series.iloc[-1]] * steps)
        
#     # 1. Preprocesamiento: Las LSTM requieren datos escalados
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
#     # 2. Creación de secuencias (Ej: usamos 12 meses para predecir 1)
#     X, y = [], []
#     for i in range(lookback, len(scaled_data)):
#         X.append(scaled_data[i-lookback:i, 0])
#         y.append(scaled_data[i, 0])
        
#     X, y = np.array(X), np.array(y)
    
#     # Reshape para keras: [samples, time steps, features]
#     X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
#     # 3. Arquitectura del Modelo
#     model = Sequential()
#     # Capa LSTM recibiendo la secuencia
#     model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
#     # Capa densa de salida con 1 neurona (Predicción Unistep)
#     model.add(Dense(1))
    
#     model.compile(optimizer='adam', loss='mean_squared_error')
    
#     # 4. Entrenamiento 
#     # (Usamos 20 epochs para que el dashboard no se congele mucho tiempo)
#     model.fit(X, y, batch_size=1, epochs=20, verbose=0)
    
#     # 5. Predicción Iterativa (Proyección a N meses)
#     last_sequence = scaled_data[-lookback:]
#     predictions = []
    
#     current_seq = last_sequence.copy()
    
#     for _ in range(steps):
#         curr_X = np.reshape(current_seq, (1, lookback, 1))
#         pred = model.predict(curr_X, verbose=0)
#         predictions.append(pred[0, 0])
        
#         # Desplazamos la ventana: quitamos el mes más viejo, agregamos la predicción
#         current_seq = np.append(current_seq[1:], pred, axis=0)
        
#     # 6. Invertimos el escalado para volver a % de inflación
#     predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
#     return pd.Series(predictions.flatten())

def run_lstm_forecast(series, steps=6, lookback=12):
    """
    Modelo Univariado - Unistep con Redes LSTM (Deep Learning).
    Entrena al vuelo con los datos del rubro seleccionado.
    """
    if len(series) <= lookback:
        return pd.Series([series.iloc[-1]] * steps)
        
    # 1. Preprocesamiento: Las LSTM requieren datos escalados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    # 2. Creación de secuencias (Ej: usamos 12 meses para predecir 1)
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # Reshape para keras: [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # 3. Arquitectura del Modelo (Actualizada para evitar warnings)
    model = Sequential()
    # Definimos la entrada de forma explícita
    model.add(Input(shape=(X.shape[1], 1)))
    # Capa LSTM recibiendo la secuencia procesada por la capa Input
    model.add(LSTM(50, return_sequences=False))
    # Capa densa de salida con 1 neurona (Predicción Unistep)
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 4. Entrenamiento 
    # (Usamos 20 epochs para que el dashboard no se congele mucho tiempo)
    model.fit(X, y, batch_size=1, epochs=20, verbose=0)
    
    # 5. Predicción Iterativa (Proyección a N meses)
    last_sequence = scaled_data[-lookback:]
    predictions = []
    
    current_seq = last_sequence.copy()
    
    for _ in range(steps):
        curr_X = np.reshape(current_seq, (1, lookback, 1))
        pred = model.predict(curr_X, verbose=0)
        predictions.append(pred[0, 0])
        
        # Desplazamos la ventana: quitamos el mes más viejo, agregamos la predicción
        # Corregimos el append para asegurar la forma (lookback, 1)
        current_seq = np.append(current_seq[1:], pred).reshape(-1, 1)
        
    # 6. Invertimos el escalado para volver a % de inflación
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return pd.Series(predictions.flatten())


# --- LÓGICA DE INTERFAZ ---

st.subheader("Dashboard Predictivo | Inflación - modelos SARIMA-RED LSTM unistep")

current_dir = os.path.dirname(os.path.abspath(__file__))
local_file_path = os.path.join(current_dir, "../data", "ipc.xlsx")

df_clean = None

if os.path.exists(local_file_path):
    df_clean = load_data_optimized(local_file_path)
    if isinstance(df_clean, str):
        st.sidebar.error(f"Error en archivo local: {df_clean}")
        df_clean = None
    else:
        st.sidebar.success(":material/check_circle: Datos cargados desde archivo local")

if df_clean is None:
    st.info(":material/folder_check: Sube el archivo ipc.xlsx para continuar")
    uploaded_file = st.file_uploader("", type=["xlsx"])
    if uploaded_file is not None:
        df_clean = load_data_optimized(uploaded_file)
        if isinstance(df_clean, str):
            st.error(df_clean)
            df_clean = None

if df_clean is not None:
    rubros = sorted(df_clean['Rubro'].unique())
    selected = st.sidebar.selectbox("Seleccione Región o Rubro", rubros)
    
    ts = df_clean[df_clean['Rubro'] == selected].set_index('Fecha')['Variacion']
    #ts = ts.asfreq('MS')
    #ts = ts.fillna(method='ffill')
    ts = ts.ffill()
    col1, col2, col3 = st.columns(3)
    col1.metric("Último Registro", f"{ts.iloc[-1]:.1f}%", f"{ts.iloc[-1] - ts.iloc[-2]:.1f}%")
    col2.metric("Promedio Anual", f"{ts.tail(12).mean():.1f}%")
    col3.metric("Máximo Histórico", f"{ts.max():.1f}%")

    st.subheader(f"Comparativa de Modelos a 6 Meses: {selected}")
    
    # Generar fechas futuras
    f_dates = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq='MS')
    
    # Procesamiento SARIMA
    f_mean_sarima, f_conf = run_forecast_v2(ts)
    
    # Procesamiento LSTM (Con un spinner porque tarda unos segundos en entrenar)
    with st.spinner('Entrenando Red Neuronal LSTM...'):
        f_lstm = run_lstm_forecast(ts)

    # Gráfico Plotly Comparativo
    fig = go.Figure()
    
    # Histórico
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Real", line=dict(color="#00d1ff", width=2)))
    
    # Predicción SARIMA (Estadística)
    fig.add_trace(go.Scatter(x=f_dates, y=f_mean_sarima, name="SARIMA (Estadística)", line=dict(color="yellow", width=2, dash="dash")))
    
    # Predicción LSTM (Deep Learning)
    fig.add_trace(go.Scatter(x=f_dates, y=f_lstm, name="Red LSTM (Unistep)", line=dict(color="#ff007f", width=2.5, dash="dot")))

    # Rango SARIMA
    fig.add_trace(go.Scatter(
        x=np.concatenate([f_dates, f_dates[::-1]]),
        y=np.concatenate([f_conf.iloc[:, 1], f_conf.iloc[:, 0][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 255, 0, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name="Rango Confianza (SARIMA)"
    ))

    fig.update_layout(template="plotly_dark", hovermode="x unified", height=600)
    st.plotly_chart(fig, width='stretch')

    with st.expander("Valores Proyectados (Variación Mensual %)"):

    # --- TABLA DE PREDICCIONES Y MÉTRICAS ---
        st.subheader(":material/fact_check: Detalle de Proyecciones (Variación Mensual %)")

        # Consolidamos las predicciones en un DataFrame
        df_tabla = pd.DataFrame({
            "Mes": f_dates.strftime('%B %Y'),
            "SARIMA (Base)": f_mean_sarima.values,
            "SARIMA (Mín)": f_conf.iloc[:, 0].values,
            "SARIMA (Máx)": f_conf.iloc[:, 1].values,
            "Red LSTM (Unistep)": f_lstm.values
        })

        # Establecemos el mes como índice
        df_tabla.set_index("Mes", inplace=True)

        # Mostramos la tabla con formato profesional
        st.dataframe(
            df_tabla.style.format("{:.2f}%")
            .highlight_max(subset=["Red LSTM (Unistep)", "SARIMA (Base)"], color="#3d0000")
            .highlight_min(subset=["Red LSTM (Unistep)", "SARIMA (Base)"], color="#002b1a"),
            width='stretch'
        )

        # Botón de exportación para tus reportes
        col_exp1, col_exp2 = st.columns([1, 4])
        with col_exp1:
            csv = df_tabla.to_csv().encode('utf-8')
            st.download_button(
                label=":material/export_notes: Exportar a CSV",
                data=csv,
                file_name=f"proyeccion_{selected.replace(' ', '_')}.csv",
                mime="text/csv",
            )

else:
    st.warning("Aguardando carga de datos...")


st.divider()

