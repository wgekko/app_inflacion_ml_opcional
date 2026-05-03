import streamlit as st
import os
import base64

st.set_page_config(layout="wide", page_title="Predicciones IPC-inflacion", page_icon=":material/rate_review:")

hide_sidebar_style = """
    <style>
        /* Oculta la barra lateral completa */
        [data-testid="stSidebar"] {
            display: none;
        }
        /* Ajusta el área principal para usar todo el ancho */
        [data-testid="stAppViewContainer"] {
            margin-left: 0px;
        }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)



def load_component():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "assets")
    
    try:
        with open(os.path.join(base_path, "card.html"), "r", encoding="utf-8") as f:
            html_content = f.read()
        with open(os.path.join(base_path, "card.css"), "r", encoding="utf-8") as f:
            css_content = f.read()
        with open(os.path.join(base_path, "card.js"), "r", encoding="utf-8") as f:
            js_content = f.read()

        full_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            <style>{css_content}</style>
        </head>
        <body>
            {html_content}
            <script>{js_content}</script>
        </body>
        </html>
        """
        
        b64_html = base64.b64encode(full_code.encode("utf-8")).decode("utf-8")
        return f"data:text/html;base64,{b64_html}"

    except FileNotFoundError as e:
        st.error(f"No se encontró el archivo: {e.filename}")
        return None

def load_folder_logo():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "assets")
    
    try:
        with open(os.path.join(base_path, "folder.html"), "r", encoding="utf-8") as f:
            html_content = f.read()
        with open(os.path.join(base_path, "folder.css"), "r", encoding="utf-8") as f:
            css_content = f.read()

        full_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>{css_content}</style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        b64_html = base64.b64encode(full_code.encode("utf-8")).decode("utf-8")
        return f"data:text/html;base64,{b64_html}"

    except FileNotFoundError as e:
        st.error(f"No se encontró el archivo del logo: {e.filename}")
        return None

# Estilo para limpiar la interfaz de Streamlit
st.markdown("""
    <style>
        .block-container { padding: 0rem; max-width: 100%; }
    </style>
""", unsafe_allow_html=True)

# Obtener las URLs de datos
data_url_cards = load_component()
data_url_logo = load_folder_logo()

# --- FILA SUPERIOR: Logo arriba a la derecha ---
# Proporción: 75% vacío a la izquierda, 25% para el logo a la derecha
col_vacia, col_logo = st.columns([8, 2])

with col_logo:
    if data_url_logo:
        # Mostramos el logo con su fondo original. Height 300 permite ver toda la animación
        st.iframe(src=data_url_logo, height=250)

# --- FILA PRINCIPAL: Contenido Central ---
# Una columna central (6) con márgenes laterales (2 cada uno)
col_margen_izq, col_central, col_margen_der = st.columns([2, 6, 2])

with col_central:
    if data_url_cards:
        st.iframe(src=data_url_cards, height=500)

    # Espaciado
    st.write("") 
    
    # Inyectar CSS global desde archivo
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    # Contenedor de botones
    with st.container(border=True):
#        c1, c2, c3 = st.columns([1,2,1])
#        with c2:
#            st.subheader("Opciones de Modelos de Análisis", anchor=False)

        st.markdown(
            "<h3 style='text-align: center;'>Opciones de Modelos de Análisis</h3>",
            unsafe_allow_html=True
        )
        #st.subheader("Panel de Control Financiero 2026", anchor=False) # Quité text_alignment='center' para evitar warnings en versiones nuevas de Streamlit
        
        b1, b2, b3, b4 = st.columns(4)
        
        with b1:
            if st.button("Modelo #1", key="acceso", width='stretch'):
                st.switch_page("pages/modelo#1.py")
        with b2:
            if st.button("Modelo #2", key="acceso1", width='stretch'):
                st.switch_page("pages/modelo#2.py")
        with b3:
            if st.button("Modelo #3", key="acceso2", width='stretch'):
                st.switch_page("pages/modelo#3.py")
        with b4:
            if st.button("Modelo #4", key="acceso3", width='stretch'):
                st.switch_page("pages/modelo#4.py")

st.write("   ")

st.divider()                