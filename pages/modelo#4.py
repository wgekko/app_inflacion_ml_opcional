# modelo de prediccion para un mes 

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import os
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import MinMaxScaler

# # --- CONFIGURACIÓN DE PÁGINA ---
# st.set_page_config(layout="wide", page_title="Fintech Analytics - Deep Learning Pro")

# # --- LÓGICA DE CARGA MEJORADA ---

# def force_to_date(value):
#     try:
#         dt = pd.to_datetime(value, dayfirst=True, errors='coerce')
#         if pd.notna(dt): return dt.replace(day=1)
#         serial = float(str(value))
#         dt = pd.to_datetime('1899-12-30') + pd.to_timedelta(serial, 'D')
#         return dt.replace(day=1)
#     except: return pd.NaT

# @st.cache_data
# def load_and_transform_ipc(file_path):
#     if not os.path.exists(file_path): return None
#     try:
#         xl = pd.ExcelFile(file_path, engine='openpyxl')
#         sheet_name = "Variación mensual aperturas"
#         raw_df = xl.parse(sheet_name)
        
#         header_row_index = 0
#         for i, row in raw_df.iterrows():
#             if any("Nivel general" in str(x) for x in row.values):
#                 header_row_index = i
#                 break
        
#         df = xl.parse(sheet_name, skiprows=header_row_index + 1)
#         df = df.rename(columns={df.columns[0]: "Rubro"})
        
#         # --- LIMPIEZA CRÍTICA PARA EVITAR EL TYPEERROR ---
#         # 1. Convertir todo a string
#         df["Rubro"] = df["Rubro"].astype(str).str.strip()
#         # 2. Eliminar filas que sean nulas, vacías o contengan basura
#         df = df[df["Rubro"].notna()]
#         df = df[~df["Rubro"].isin(['nan', 'None', '', 'Unnamed: 0'])]
#         # -------------------------------------------------

#         df_melted = df.melt(id_vars=["Rubro"], var_name="Fecha", value_name="Variacion")
#         df_melted["Fecha"] = df_melted["Fecha"].apply(force_to_date)
#         df_melted["Variacion"] = pd.to_numeric(df_melted["Variacion"], errors='coerce').fillna(0)
#         df_melted = df_melted.dropna(subset=["Fecha"])
        
#         return df_melted
#     except Exception as e:
#         st.error(f"Error en la carga: {e}")
#         return None

# def prepare_data(data, lookback=12):
#     # Validación crucial: ¿hay datos para escalar?
#     if data.empty or len(data) <= lookback:
#         return None, None, None
        
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
#     X, y = [], []
#     for i in range(len(scaled_data) - lookback):
#         X.append(scaled_data[i:i+lookback])
#         y.append(scaled_data[i+lookback])
        
#     return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)), scaler

# # --- MODELOS (Mantenemos GRU y TCN del ejemplo anterior) ---
# class GRUModel(nn.Module):
#     def __init__(self, input_size=1, hidden_size=64, num_layers=2):
#         super(GRUModel, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
#     def forward(self, x):
#         out, _ = self.gru(x)
#         return self.fc(out[:, -1, :])

# # --- INTERFAZ ---
# st.title("Predicción de Inflación: Modelos GRU-TCN-TFT")

# file_path = "../data/ipc.xlsx"
# data_all = load_and_transform_ipc(file_path)

# if data_all is not None:
#     #rubros = sorted(data_all["Rubro"].unique())
#     # Convertimos todo a string y filtramos valores vacíos o 'nan' antes de ordenar
#     rubros = sorted([str(r) for r in data_all["Rubro"].unique() if pd.notna(r) and str(r).strip().lower() != 'nan'])
#     selected = st.selectbox("Seleccione el rubro para el análisis:", rubros)
    
#     # Filtrado y ordenamiento
#     mask = data_all["Rubro"] == selected
#     ts_data = data_all[mask].sort_values("Fecha").set_index("Fecha")["Variacion"]

#     # Verificación de datos antes de seguir
#     if ts_data.empty:
#         st.warning(f"No se encontraron datos para el rubro: {selected}. Verifique el nombre en el Excel.")
#     else:
#         col1, col2, col3 = st.columns(3)
#         model_type = col1.radio("Arquitectura:", ["GRU", "TCN", "TFT (Atención)"])
#         lookback = col3.slider("Ventana (meses)", 6, 24, 12)

#         if st.button("Ejecutar Predicción"):
#             # Aquí es donde ocurría el error: ahora validamos
#             X, y, scaler = prepare_data(ts_data, lookback)
            
#             if X is None:
#                 st.error("Error: Datos insuficientes. La serie histórica debe ser más larga que la ventana de tiempo seleccionada.")
#             else:
#                 with st.spinner(f"Entrenando {model_type}..."):
#                     # Lógica de entrenamiento (simplificada para el ejemplo)
#                     model = GRUModel()
#                     criterion = nn.MSELoss()
#                     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    
#                     for epoch in range(100):
#                         model.train()
#                         optimizer.zero_grad()
#                         output = model(X)
#                         loss = criterion(output, y)
#                         loss.backward()
#                         optimizer.step()

#                     # Generar proyección
#                     model.eval()
#                     last_seq = torch.FloatTensor(scaler.transform(ts_data.values[-lookback:].reshape(-1, 1))).unsqueeze(0)
                    
#                     with torch.no_grad():
#                         pred_scaled = model(last_seq)
#                         prediction = scaler.inverse_transform(pred_scaled.numpy())[0][0]
                    
#                     st.success(f"Predicción para el próximo mes ({selected}): {prediction:.2f}%")
                    
#                     # Gráfico rápido
#                     fig = go.Figure()
#                     fig.add_trace(go.Scatter(y=ts_data.values[-12:], name="Histórico (12m)"))
#                     fig.add_trace(go.Scatter(x=[12], y=[prediction], mode='markers+text', 
#                                             text=[f"{prediction:.2f}%"], textposition="top center",
#                                             marker=dict(size=12, color="red"), name="Predicción"))
#                     st.plotly_chart(fig)

#-----------------------------------------------------------------------------------------------------------------------------
# modelo de prediccion para 3 meses

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import os
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import MinMaxScaler

# # --- CONFIGURACIÓN DE PÁGINA ---
# st.set_page_config(layout="wide", page_title="Analytics-Deep Learning-Modelos GRU-TCN-TFT ")

# # --- LÓGICA DE CARGA MEJORADA ---

# def force_to_date(value):
#     try:
#         dt = pd.to_datetime(value, dayfirst=True, errors='coerce')
#         if pd.notna(dt): return dt.replace(day=1)
#         serial = float(str(value))
#         dt = pd.to_datetime('1899-12-30') + pd.to_timedelta(serial, 'D')
#         return dt.replace(day=1)
#     except: return pd.NaT

# @st.cache_data
# def load_and_transform_ipc(file_path):
#     if not os.path.exists(file_path): return None
#     try:
#         xl = pd.ExcelFile(file_path, engine='openpyxl')
#         sheet_name = "Variación mensual aperturas"
#         raw_df = xl.parse(sheet_name)
        
#         header_row_index = 0
#         for i, row in raw_df.iterrows():
#             if any("Nivel general" in str(x) for x in row.values):
#                 header_row_index = i
#                 break
        
#         df = xl.parse(sheet_name, skiprows=header_row_index + 1)
#         df = df.rename(columns={df.columns[0]: "Rubro"})
        
#         # --- LIMPIEZA CRÍTICA PARA EVITAR EL TYPEERROR ---
#         df["Rubro"] = df["Rubro"].astype(str).str.strip()
#         df = df[df["Rubro"].notna()]
#         df = df[~df["Rubro"].isin(['nan', 'None', '', 'Unnamed: 0'])]
#         # -------------------------------------------------

#         df_melted = df.melt(id_vars=["Rubro"], var_name="Fecha", value_name="Variacion")
#         df_melted["Fecha"] = df_melted["Fecha"].apply(force_to_date)
#         df_melted["Variacion"] = pd.to_numeric(df_melted["Variacion"], errors='coerce').fillna(0)
#         df_melted = df_melted.dropna(subset=["Fecha"])
        
#         return df_melted
#     except Exception as e:
#         st.error(f"Error en la carga: {e}")
#         return None

# def prepare_data(data, lookback=12):
#     if data.empty or len(data) <= lookback:
#         return None, None, None
        
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
#     X, y = [], []
#     for i in range(len(scaled_data) - lookback):
#         X.append(scaled_data[i:i+lookback])
#         y.append(scaled_data[i+lookback])
        
#     return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)), scaler

# # --- MODELOS ---
# class GRUModel(nn.Module):
#     def __init__(self, input_size=1, hidden_size=64, num_layers=2):
#         super(GRUModel, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
#     def forward(self, x):
#         out, _ = self.gru(x)
#         return self.fc(out[:, -1, :])

# # --- INTERFAZ ---
# st.subheader("Predicción de Inflación: Modelos GRU-TCN-TFT")
# st.info("GRU (Gated Recurrent Unit)- Es un tipo de red neuronal recurrente (RNN).Aprende dependencias en el tiempo paso a paso (secuencialmente).")
# st.info("TCN (Temporal Convolutional Network)-Es un modelo basado en convoluciones (CNN) pero adaptado al tiempo. Aprende patrones temporales usando ventanas paralelas, no paso a paso.")
# st.info("TFT (Temporal Fusion Transformer)- Es un modelo moderno basado en Transformers (attention). Aprende relaciones complejas con atención + interpretabilidad.") 

# file_path = "../data/ipc.xlsx"
# data_all = load_and_transform_ipc(file_path)

# if data_all is not None:
#     rubros = sorted([str(r) for r in data_all["Rubro"].unique() if pd.notna(r) and str(r).strip().lower() != 'nan'])
#     selected = st.selectbox("Seleccione el rubro para el análisis:", rubros)
    
#     mask = data_all["Rubro"] == selected
#     ts_data = data_all[mask].sort_values("Fecha").set_index("Fecha")["Variacion"]

#     if ts_data.empty:
#         st.warning(f"No se encontraron datos para el rubro: {selected}. Verifique el nombre en el Excel.")
#     else:
#         col1, col2, col3 = st.columns(3)
#         model_type = col1.radio("Arquitectura:", ["GRU", "TCN", "TFT (Atención)"])
#         lookback = col3.slider("Ventana (meses)", 6, 24, 12)

#         if st.button("Ejecutar Predicción"):
#             X, y, scaler = prepare_data(ts_data, lookback)
            
#             if X is None:
#                 st.error("Error: Datos insuficientes. La serie histórica debe ser más larga que la ventana de tiempo seleccionada.")
#             else:
#                 with st.spinner(f"Entrenando {model_type} y proyectando trimestre..."):
#                     # --- ENTRENAMIENTO (Intacto) ---
#                     model = GRUModel()
#                     criterion = nn.MSELoss()
#                     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    
#                     for epoch in range(100):
#                         model.train()
#                         optimizer.zero_grad()
#                         output = model(X)
#                         loss = criterion(output, y)
#                         loss.backward()
#                         optimizer.step()

#                     # --- NUEVA GENERACIÓN DE PROYECCIÓN (3 MESES AUTORREGRESIVA) ---
#                     model.eval()
#                     # Preparamos la secuencia inicial (últimos N meses reales)
#                     current_seq = scaler.transform(ts_data.values[-lookback:].reshape(-1, 1))
#                     current_seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
                    
#                     predicciones_escaladas = []
                    
#                     with torch.no_grad():
#                         for _ in range(3):
#                             # Predecir el siguiente paso
#                             pred_scaled = model(current_seq_tensor)
#                             predicciones_escaladas.append(pred_scaled.item())
                            
#                             # Actualizar la secuencia para el próximo bucle:
#                             # 1. Ajustar la dimensión de la predicción para concatenarla
#                             next_input = pred_scaled.unsqueeze(1)
#                             # 2. Desplazar la ventana: quitamos el índice 0, agregamos el nuevo dato al final
#                             current_seq_tensor = torch.cat((current_seq_tensor[:, 1:, :], next_input), dim=1)
                    
#                     # Invertir el escalado de las 3 predicciones a la escala porcentual real
#                     predicciones = scaler.inverse_transform(np.array(predicciones_escaladas).reshape(-1, 1)).flatten()
                    
#                     st.success(f"Predicción trimestral ({selected}): Mes 1: {predicciones[0]:.2f}% | Mes 2: {predicciones[1]:.2f}% | Mes 3: {predicciones[2]:.2f}%")
                    
#                     # --- GRÁFICO ACTUALIZADO (Conectando histórico y proyección) ---
#                     fig = go.Figure()
                    
#                     # Ejes X relativos para que la línea sea continua
#                     x_hist = list(range(1, 13))
#                     x_pred = [12, 13, 14, 15] # Empezamos en 12 para conectar con el último punto histórico
                    
#                     # El primer punto de la proyección debe ser el último punto real para conectar la línea gráficamente
#                     y_pred_plot = [ts_data.values[-1]] + list(predicciones)
#                     text_pred_plot = [""] + [f"{p:.2f}%" for p in predicciones]
                    
#                     fig.add_trace(go.Scatter(x=x_hist, y=ts_data.values[-12:], name="Histórico (12m)", mode='lines+markers'))
#                     fig.add_trace(go.Scatter(x=x_pred, y=y_pred_plot, mode='lines+markers+text', 
#                                             text=text_pred_plot, textposition="top center",
#                                             marker=dict(size=8, color="red"), 
#                                             line=dict(color="red", dash='dash'),
#                                             name="Proyección (3m)"))
                    
#                     fig.update_layout(xaxis_title="Meses relativos (Últimos 12m + 3m Proyección)", yaxis_title="Variación IPC (%)")
#                     st.plotly_chart(fig)                    


# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import os
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import MinMaxScaler

# st.set_page_config(layout="wide", page_title="Analytics-Deep Learning-Modelos GRU-TCN-TFT", page_icon=":material/monitoring:")

# # --- LÓGICA DE CARGA MEJORADA ---
# def force_to_date(value):
#     try:
#         dt = pd.to_datetime(value, dayfirst=True, errors='coerce')
#         if pd.notna(dt): return dt.replace(day=1)
#         serial = float(str(value))
#         dt = pd.to_datetime('1899-12-30') + pd.to_timedelta(serial, 'D')
#         return dt.replace(day=1)
#     except: return pd.NaT

# @st.cache_data
# def load_and_transform_ipc(file_path):
#     if not os.path.exists(file_path): 
#         # Mostramos el error en la interfaz si no encuentra el archivo
#         st.error(f"No se encontró el archivo en la ruta: {file_path}. Verifica la estructura de carpetas.")
#         return None
#     try:
#         xl = pd.ExcelFile(file_path, engine='openpyxl')
#         sheet_name = "Variación mensual aperturas"
#         raw_df = xl.parse(sheet_name)
        
#         header_row_index = 0
#         for i, row in raw_df.iterrows():
#             if any("Nivel general" in str(x) for x in row.values):
#                 header_row_index = i
#                 break
        
#         df = xl.parse(sheet_name, skiprows=header_row_index + 1)
#         df = df.rename(columns={df.columns[0]: "Rubro"})
        
#         # Limpieza crítica
#         df["Rubro"] = df["Rubro"].astype(str).str.strip()
#         df = df[df["Rubro"].notna()]
#         df = df[~df["Rubro"].isin(['nan', 'None', '', 'Unnamed: 0'])]

#         df_melted = df.melt(id_vars=["Rubro"], var_name="Fecha", value_name="Variacion")
#         df_melted["Fecha"] = df_melted["Fecha"].apply(force_to_date)
#         df_melted["Variacion"] = pd.to_numeric(df_melted["Variacion"], errors='coerce').fillna(0)
#         df_melted = df_melted.dropna(subset=["Fecha"])
        
#         return df_melted
#     except Exception as e:
#         st.error(f"Error en la lectura del Excel: {e}")
#         return None

# def prepare_data(data, lookback=12):
#     if data.empty or len(data) <= lookback:
#         return None, None, None
        
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
#     X, y = [], []
#     for i in range(len(scaled_data) - lookback):
#         X.append(scaled_data[i:i+lookback])
#         y.append(scaled_data[i+lookback])
        
#     return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)), scaler

# # --- MODELOS ---
# class GRUModel(nn.Module):
#     def __init__(self, input_size=1, hidden_size=64, num_layers=2):
#         super(GRUModel, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
#     def forward(self, x):
#         out, _ = self.gru(x)
#         return self.fc(out[:, -1, :])

# # --- INTERFAZ DE LA PÁGINA ---
# st.subheader("Predicción de Inflación: Modelos GRU-TCN-TFT")
# st.info("GRU (Gated Recurrent Unit)- Es un tipo de red neuronal recurrente (RNN).Aprende dependencias en el tiempo paso a paso (secuencialmente).")
# st.info("TCN (Temporal Convolutional Network)-Es un modelo basado en convoluciones (CNN) pero adaptado al tiempo. Aprende patrones temporales usando ventanas paralelas, no paso a paso.")
# st.info("TFT (Temporal Fusion Transformer)- Es un modelo moderno basado en Transformers (attention). Aprende relaciones complejas con atención + interpretabilidad.") 

# root_dir = os.getcwd() 
# file_path = os.path.join(root_dir, "data", "ipc.xlsx")

# # st.write(f"Buscando datos en: {file_path}") # Descomenta esta línea si necesitas depurar la ruta

# data_all = load_and_transform_ipc(file_path)

# if data_all is not None:
#     rubros = sorted([str(r) for r in data_all["Rubro"].unique() if pd.notna(r) and str(r).strip().lower() != 'nan'])
#     selected = st.selectbox("Seleccione el rubro para el análisis:", rubros)
    
#     mask = data_all["Rubro"] == selected
#     ts_data = data_all[mask].sort_values("Fecha").set_index("Fecha")["Variacion"]

#     if ts_data.empty:
#         st.warning(f"No se encontraron datos para el rubro: {selected}.")
#     else:
#         col1, col2, col3 = st.columns(3)
#         model_type = col1.radio("Arquitectura:", ["GRU", "TCN", "TFT (Atención)"])
#         lookback = col3.slider("Ventana (meses)", 6, 24, 12)

#         if st.button("Ejecutar Predicción"):
#             X, y, scaler = prepare_data(ts_data, lookback)
            
#             if X is None:
#                 st.error("Error: Datos insuficientes. La serie histórica debe ser más larga que la ventana.")
#             else:
#                 with st.spinner(f"Entrenando {model_type} y proyectando trimestre..."):
                    
#                     # --- ENTRENAMIENTO ---
#                     model = GRUModel()
#                     criterion = nn.MSELoss()
#                     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    
#                     for epoch in range(100):
#                         model.train()
#                         optimizer.zero_grad()
#                         output = model(X)
#                         loss = criterion(output, y)
#                         loss.backward()
#                         optimizer.step()

#                     # --- PROYECCIÓN 3 MESES ---
#                     model.eval()
#                     current_seq = scaler.transform(ts_data.values[-lookback:].reshape(-1, 1))
#                     current_seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
                    
#                     predicciones_escaladas = []
                    
#                     with torch.no_grad():
#                         for _ in range(3):
#                             pred_scaled = model(current_seq_tensor)
#                             predicciones_escaladas.append(pred_scaled.item())
#                             next_input = pred_scaled.unsqueeze(1)
#                             current_seq_tensor = torch.cat((current_seq_tensor[:, 1:, :], next_input), dim=1)
                    
#                     predicciones = scaler.inverse_transform(np.array(predicciones_escaladas).reshape(-1, 1)).flatten()
                    
#                     st.success(f"Predicción trimestral ({selected}): Mes 1: {predicciones[0]:.2f}% | Mes 2: {predicciones[1]:.2f}% | Mes 3: {predicciones[2]:.2f}%")
                    
#                     # --- GRÁFICO ---
#                     fig = go.Figure()
#                     x_hist = list(range(1, 13))
#                     x_pred = [12, 13, 14, 15] 
                    
#                     y_pred_plot = [ts_data.values[-1]] + list(predicciones)
#                     text_pred_plot = [""] + [f"{p:.2f}%" for p in predicciones]
                    
#                     fig.add_trace(go.Scatter(x=x_hist, y=ts_data.values[-12:], name="Histórico (12m)", mode='lines+markers'))
#                     fig.add_trace(go.Scatter(x=x_pred, y=y_pred_plot, mode='lines+markers+text', 
#                                             text=text_pred_plot, textposition="top center",
#                                             marker=dict(size=8, color="red"), 
#                                             line=dict(color="red", dash='dash'),
#                                             name="Proyección (3m)"))
                    
#                     fig.update_layout(xaxis_title="Meses relativos", yaxis_title="Variación IPC (%)")
#                     st.plotly_chart(fig)

# st.divider()                    
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Analytics-Deep Learning-Modelos GRU-TCN-TFT", page_icon=":material/monitoring:")

# --- LÓGICA DE CARGA ---
def force_to_date(value):
    try:
        dt = pd.to_datetime(value, dayfirst=True, errors='coerce')
        if pd.notna(dt): return dt.replace(day=1)
        serial = float(str(value))
        dt = pd.to_datetime('1899-12-30') + pd.to_timedelta(serial, 'D')
        return dt.replace(day=1)
    except: return pd.NaT

@st.cache_data
def load_and_transform_ipc(file_path):
    if not os.path.exists(file_path): 
        return None
    try:
        xl = pd.ExcelFile(file_path, engine='openpyxl')
        sheet_name = "Variación mensual aperturas"
        raw_df = xl.parse(sheet_name)
        
        header_row_index = 0
        for i, row in raw_df.iterrows():
            if any("Nivel general" in str(x) for x in row.values):
                header_row_index = i
                break
        
        df = xl.parse(sheet_name, skiprows=header_row_index + 1)
        df = df.rename(columns={df.columns[0]: "Rubro"})
        
        df["Rubro"] = df["Rubro"].astype(str).str.strip()
        df = df[df["Rubro"].notna()]
        df = df[~df["Rubro"].isin(['nan', 'None', '', 'Unnamed: 0'])]

        df_melted = df.melt(id_vars=["Rubro"], var_name="Fecha", value_name="Variacion")
        df_melted["Fecha"] = df_melted["Fecha"].apply(force_to_date)
        df_melted["Variacion"] = pd.to_numeric(df_melted["Variacion"], errors='coerce').fillna(0)
        df_melted = df_melted.dropna(subset=["Fecha"])
        
        return df_melted
    except Exception as e:
        st.error(f"Error en la lectura del Excel: {e}")
        return None

def prepare_data(data, lookback=12):
    if data.empty or len(data) <= lookback:
        return None, None, None
        
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:i+lookback])
        y.append(scaled_data[i+lookback])
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)), scaler

# --- MODELOS ---
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# --- PROCESAMIENTO DE DATOS ---
root_dir = os.getcwd() 
file_path = os.path.join(root_dir, "data", "ipc.xlsx")
data_all = load_and_transform_ipc(file_path)

# --- INTERFAZ (SIDEBAR) ---
st.sidebar.header(":material/settings: Configuración")

if data_all is not None:
    # Lógica para valor preestablecido
    rubros = sorted([str(r) for r in data_all["Rubro"].unique() if pd.notna(r) and str(r).strip().lower() != 'nan'])
    default_val = "Región Cuyo-Nivel general"
    
    try:
        default_idx = rubros.index(default_val)
    except ValueError:
        default_idx = 0

    selected = st.sidebar.selectbox("Seleccione el rubro para el análisis:", rubros, index=default_idx)
    
    st.sidebar.divider()
    model_type = st.sidebar.radio("Arquitectura de IA:", ["GRU", "TCN", "TFT (Atención)"])
    lookback = st.sidebar.slider("Ventana histórica (meses)", 6, 24, 12)
    
    st.sidebar.divider()
    ejecutar = st.sidebar.button("Ejecutar Predicción", type="primary", width='stretch')

# --- CUERPO PRINCIPAL ---
st.subheader("Predicción de Inflación: Inteligencia Artificial Profunda")

with st.expander("Información sobre los modelos disponibles"):
    st.info("**GRU:** Ideal para captar secuencias históricas directas.")
    st.info("**TCN:** Excelente para detectar patrones en ventanas paralelas de tiempo.")
    st.info("**TFT:** Utiliza mecanismos de atención para relaciones complejas.")

if data_all is not None:
    mask = data_all["Rubro"] == selected
    ts_data = data_all[mask].sort_values("Fecha").set_index("Fecha")["Variacion"]

    if ts_data.empty:
        st.warning(f"No se encontraron datos para: {selected}.")
    else:
        if ejecutar:
            X, y, scaler = prepare_data(ts_data, lookback)
            
            if X is None:
                st.error("Error: La serie histórica es demasiado corta para la ventana seleccionada.")
            else:
                with st.spinner(f"Entrenando modelo {model_type}..."):
                    # Entrenamiento
                    model = GRUModel()
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    
                    for epoch in range(100):
                        model.train()
                        optimizer.zero_grad()
                        output = model(X)
                        loss = criterion(output, y)
                        loss.backward()
                        optimizer.step()

                    # Proyección 3 meses
                    model.eval()
                    current_seq = scaler.transform(ts_data.values[-lookback:].reshape(-1, 1))
                    current_seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
                    
                    predicciones_escaladas = []
                    with torch.no_grad():
                        for _ in range(3):
                            pred_scaled = model(current_seq_tensor)
                            predicciones_escaladas.append(pred_scaled.item())
                            next_input = pred_scaled.unsqueeze(1)
                            current_seq_tensor = torch.cat((current_seq_tensor[:, 1:, :], next_input), dim=1)
                    
                    predicciones = scaler.inverse_transform(np.array(predicciones_escaladas).reshape(-1, 1)).flatten()
                    
                    # Visualización de Resultados
                    st.success(f"Resultados de Proyección Trimestral - {selected}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Mes +1", f"{predicciones[0]:.2f}%")
                    m2.metric("Mes +2", f"{predicciones[1]:.2f}%")
                    m3.metric("Mes +3", f"{predicciones[2]:.2f}%")
                    
                    # Gráfico Comparativo
                    fig = go.Figure()
                    x_hist = list(range(1, 13))
                    x_pred = [12, 13, 14, 15] 
                    y_pred_plot = [ts_data.values[-1]] + list(predicciones)
                    
                    fig.add_trace(go.Scatter(x=x_hist, y=ts_data.values[-12:], name="Histórico (12m)", mode='lines+markers'))
                    fig.add_trace(go.Scatter(x=x_pred, y=y_pred_plot, mode='lines+markers+text', 
                                            text=[""] + [f"{p:.2f}%" for p in predicciones],
                                            textposition="top center",
                                            line=dict(color="red", dash='dash'), name="Proyección"))
                    
                    fig.update_layout(template="plotly_dark", xaxis_title="Meses Relativos", yaxis_title="IPC (%)")
                    st.plotly_chart(fig, width='stretch')
        else:
            # Aquí se corrigió el error de st.light por st.info
            st.info("Por favor, configure los parámetros en la barra lateral y presione el botón para iniciar el análisis.")
else:
    st.error(":material/warning: No se pudo cargar el archivo 'ipc.xlsx'. Verifique su ubicación en 'data/ipc.xlsx'.")


st.write("   ")

st.divider()
