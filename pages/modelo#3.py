import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import entropy
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Analytics-IPC AI", page_icon=":material/query_stats:")

# --- FUNCIONES DE SOPORTE ---

def force_to_date(value):
    """Convierte valores de Excel (seriales o strings) a objetos datetime."""
    try:
        # Intenta conversión estándar
        dt = pd.to_datetime(value, dayfirst=True, errors='coerce')
        if pd.notna(dt):
            return dt.replace(day=1)
        # Intenta conversión de serial de Excel
        serial = float(value)
        dt = pd.to_datetime('1899-12-30') + pd.to_timedelta(serial, 'D')
        return dt.replace(day=1)
    except:
        return pd.NaT

@st.cache_data
def load_and_transform_ipc(file_path):
    """Carga, limpia y transforma el archivo del INDEC."""
    if not os.path.exists(file_path):
        return None
    try:
        # 1. Escaneo de cabecera dinámica
        raw_df = pd.read_excel(file_path, sheet_name="Variación mensual aperturas", engine='openpyxl')
        header_row_index = 0
        for i, row in raw_df.iterrows():
            potential_dates = [force_to_date(v) for v in row if pd.notna(force_to_date(v))]
            if len(potential_dates) > 3:
                header_row_index = i + 1
                break
        
        # 2. Carga con cabecera correcta
        df = pd.read_excel(file_path, sheet_name="Variación mensual aperturas", 
                            skiprows=header_row_index, engine='openpyxl')
        
        # 3. Limpieza de columnas
        df = df.rename(columns={df.columns[0]: 'Rubro'})
        df['Rubro'] = df['Rubro'].astype(str).str.strip()
        
        # 4. Mapeo de fechas
        date_mapping = {col: force_to_date(col) for col in df.columns[1:] if pd.notna(force_to_date(col))}
        
        # 5. Transformación a formato largo (Melt)
        df_long = df.melt(id_vars=['Rubro'], value_vars=list(date_mapping.keys()), 
                            var_name='Original', value_name='Variacion')
        df_long['Fecha'] = df_long['Original'].map(date_mapping)
        df_long['Variacion'] = pd.to_numeric(df_long['Variacion'], errors='coerce')
        
        # 6. Eliminación de duplicados y nulos
        df_clean = df_long.dropna(subset=['Variacion', 'Fecha'])
        # Agrupamos por si el mismo rubro aparece repetido en el Excel
        return df_clean.groupby(['Rubro', 'Fecha'])['Variacion'].mean().reset_index()
    
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None

def run_ml_forecast(series, months=6):
    """Predicción mediante Random Forest (Machine Learning)."""
    df_ml = series.reset_index()
    df_ml['Month_Num'] = np.arange(len(df_ml))
    
    X = df_ml[['Month_Num']].values
    y = df_ml['Variacion'].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    future_X = np.arange(len(df_ml), len(df_ml) + months).reshape(-1, 1)
    return model.predict(future_X)

# --- CUERPO DE LA APLICACIÓN ---

st.subheader("Dashboard Predictivo | Inflación-(Predictive AI)")
st.markdown("---")

# Ruta del archivo
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "../data", "ipc.xlsx")

data = load_and_transform_ipc(file_path)

if data is not None:
    # Sidebar: Selección de rubro
    rubros = [r for r in sorted(data['Rubro'].unique()) if len(r) > 3 and "Fuente" not in r]
    selected = st.sidebar.selectbox("Seleccionar Rubro o Región", rubros)
    
    # Preparación de la Serie Temporal
    subset = data[data['Rubro'] == selected]
    # .asfreq('MS') asegura que no falten meses y .ffill() completa huecos
    ts_data = subset.groupby('Fecha')['Variacion'].mean().asfreq('MS').ffill()

    if not ts_data.empty:
        # --- SECCIÓN 1: MÉTRICAS DE IMPACTO ---
        m1, m2, m3, m4 = st.columns(4)
        
        last_val = ts_data.iloc[-1]
        accel = ts_data.diff().iloc[-1] if len(ts_data) > 1 else 0
        
        # Cálculo de entropía (caos en la serie)
        counts = np.histogram(ts_data, bins=10)[0]
        ent = entropy(counts) if np.sum(counts) > 0 else 0

        m1.metric("Variación Último Mes", f"{last_val:.1f}%")
        m2.metric("Aceleración", f"{accel:.2f} pp", delta=f"{accel:.2f}", delta_color="inverse")
        m3.metric("Entropía (Incertidumbre)", f"{ent:.2f}")
        m4.metric("Volatilidad Anual (σ)", f"{ts_data.tail(12).std():.2f}")

        # --- SECCIÓN 2: DESCOMPOSICIÓN ESTRUCTURAL ---
        st.subheader(":material/account_tree: Estructura Interna del Precio (Insight)")
        st.subheader(f"Rubro o Región : {selected}")
        try:
            # La descomposición separa la tendencia del ruido mensual
            decomp = seasonal_decompose(ts_data, model='additive', period=12)
            
            fig_dec = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    subplot_titles=("Tendencia (Dirección de Fondo)", "Estacionalidad (Efecto Calendario)"))
            
            fig_dec.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, name="Tendencia", 
                                        line=dict(color='#00d1b2', width=3)), row=1, col=1)
            fig_dec.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, name="Estacional", 
                                        line=dict(color='#ffdd57')), row=2, col=1)
            
            fig_dec.update_layout(height=450, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_dec, width='stretch')
        except:
            st.info("Se requieren más datos históricos para realizar la descomposición estacional.")

        # --- SECCIÓN 3: PREDICCIONES MULTI-MODELO ---
        st.markdown("---")

        st.subheader(":material/analytics: Proyecciones de IA y Econometría (Próximos 6 Meses)")
        st.subheader(f"Rubro o Región : {selected}")
        f_dates = pd.date_range(ts_data.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq='MS')
        
        # 1. Modelo SARIMA (Estadístico)
        try:
            model_s = SARIMAX(ts_data, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            p_sarima = model_s.get_forecast(steps=6).predicted_mean
        except:
            p_sarima = pd.Series([ts_data.mean()]*6, index=f_dates)

        # 2. Modelo Random Forest (ML)
        p_rf = run_ml_forecast(ts_data.to_frame(), 6)

        # Gráfico de Predicciones
        fig_pred = go.Figure()
        
        # Histórico reciente (18 meses)
        hist = ts_data.tail(18)
        fig_pred.add_trace(go.Scatter(x=hist.index, y=hist, name="Histórico Real", 
                                    line=dict(color='white', width=2)))
        
        # Línea SARIMA
        fig_pred.add_trace(go.Scatter(x=f_dates, y=p_sarima, name="Modelo SARIMA", 
                                    line=dict(dash='dot', color='#00d1b2')))
        
        # Línea Random Forest
        fig_pred.add_trace(go.Scatter(x=f_dates, y=p_rf, name="Modelo Random Forest (IA)", 
                                    line=dict(dash='dash', color='#ff3860')))
        
        fig_pred.update_layout(template="plotly_dark", hovermode="x unified", 
                            legend=dict(orientation="h", y=1.1), height=500)
        st.plotly_chart(fig_pred, width='stretch')

        # --- SECCIÓN 4: DATOS CRUDOS ---
        with st.expander("Ver detalle de valores proyectados"):
            df_comp = pd.DataFrame({
                "Mes Proyectado": f_dates.strftime('%b %Y'),
                "SARIMA (%)": p_sarima.values,
                "Random Forest (%)": p_rf
            }).round(2)
            st.table(df_comp)

else:
    st.error(":material/warning: No se pudo cargar el archivo 'ipc.xlsx'. Verifica que esté en la carpeta 'data/'.")

st.write("   ")

st.divider()    