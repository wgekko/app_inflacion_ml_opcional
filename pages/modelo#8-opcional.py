import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from pathlib import Path

# =========================================================
# CONTROL DE DEPENDENCIAS
# =========================================================

try:
    import shap
    import xgboost as xgb
    HAS_XAI = True
except ImportError:
    HAS_XAI = False

try:
    from pyvis.network import Network
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    HAS_AGGRID = True
except ImportError:
    HAS_AGGRID = False

try:
    from streamlit_echarts import st_echarts
    HAS_ECHARTS = True
except ImportError:
    HAS_ECHARTS = False

# =========================================================
# CONFIG STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Panel Financiero Vanguard",
    layout="wide",
    page_icon=":material/rate_review:"
)

# =========================================================
# CSS GLOBAL
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0e1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.metric-card {
    background-color: #161b22;
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #30363d;
}

iframe {
    border-radius: 12px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATOS SINTÉTICOS
# =========================================================

@st.cache_data
def generar_datos():

    fechas = pd.date_range(
        start="2017-01-01",
        end="2026-03-01",
        freq="MS"
    )

    regiones = [
        "AMBA",
        "NOA",
        "NEA",
        "Cuyo",
        "Patagonia",
        "Pampeana"
    ]

    np.random.seed(42)

    lista = []

    for region in regiones:

        n = len(fechas)

        energia = np.random.normal(0.03, 0.01, n)

        transporte = (
            energia * 0.8 +
            np.random.normal(0.01, 0.005, n)
        )

        alimentos = (
            transporte * 0.5 +
            np.random.normal(0.03, 0.015, n)
        )

        factor = np.random.uniform(0.92, 1.08)

        energia *= factor
        transporte *= factor
        alimentos *= factor

        nivel_general = (
            energia * 0.15 +
            transporte * 0.25 +
            alimentos * 0.60
        )

        temp = pd.DataFrame({
            "Fecha": fechas,
            "Region": region,
            "Energia": energia * 100,
            "Transporte": transporte * 100,
            "Alimentos": alimentos * 100,
            "Nivel_General": nivel_general * 100
        })

        lista.append(temp)

    return pd.concat(lista)

df = generar_datos()

# =========================================================
# BASE NACIONAL
# =========================================================

df_nacional = (
    df.groupby("Fecha")[[
        "Energia",
        "Transporte",
        "Alimentos",
        "Nivel_General"
    ]]
    .mean()
)

# =========================================================
# HEADER
# =========================================================

st.title("Panel de Control Financiero de Vanguardia")

st.markdown("""
Integración de XAI, Grafos, UI Premium y Modelado Bayesiano
""")

# =========================================================
# KPIs
# =========================================================

ultimo = df_nacional["Nivel_General"].iloc[-1]

inflacion_yoy = (
    (
        ultimo /
        df_nacional["Nivel_General"].iloc[-12]
    ) - 1
) * 100

volatilidad = df_nacional["Nivel_General"].std()

momentum = (
    df_nacional["Nivel_General"]
    .pct_change(3)
    .iloc[-1]
) * 100

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Inflación Actual",
        f"{ultimo:.2f}%"
    )

with k2:
    st.metric(
        "Inflación Interanual",
        f"{inflacion_yoy:.2f}%"
    )

with k3:
    st.metric(
        "Volatilidad",
        f"{volatilidad:.2f}"
    )

with k4:
    st.metric(
        "Momentum 3M",
        f"{momentum:.2f}%"
    )

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. XAI (SHAP)",
    "2. SOTA & Bayesiano",
    "3. Redes (Grafos)",
    "4. Narrativa (LLM)",
    "5. UI Premium"
])

# =========================================================
# FUNCION SHAP TEMPORAL
# =========================================================

@st.cache_data
def calcular_shap_temporal(df_base):

    if not HAS_XAI:
        return None

    resultados = []

    for i in range(12, len(df_base)):

        temp = df_base.iloc[:i]

        X = temp[[
            "Energia",
            "Transporte",
            "Alimentos"
        ]]

        y = temp["Nivel_General"]

        model = xgb.XGBRegressor(
            n_estimators=50,
            random_state=42
        )

        model.fit(X, y)

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X)

        impacto = np.abs(shap_values[-1])

        resultados.append({
            "Fecha": temp.index[-1],
            "Energia": impacto[0],
            "Transporte": impacto[1],
            "Alimentos": impacto[2]
        })

    return pd.DataFrame(resultados)

# =========================================================
# TAB 1 - SHAP
# =========================================================

with tab1:

    st.subheader("Análisis de Causalidad")

    if HAS_XAI:

        X = df_nacional[[
            "Energia",
            "Transporte",
            "Alimentos"
        ]]

        y = df_nacional["Nivel_General"]

        model = xgb.XGBRegressor(
            n_estimators=50,
            random_state=42
        )

        model.fit(X, y)

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X)

        shap_sum = np.abs(shap_values).mean(axis=0)

        df_shap = pd.DataFrame({
            "Rubro": X.columns,
            "Impacto SHAP": shap_sum
        }).sort_values(
            by="Impacto SHAP",
            ascending=True
        )

        fig_shap = go.Figure()

        fig_shap.add_trace(
            go.Bar(
                x=df_shap["Impacto SHAP"],
                y=df_shap["Rubro"],
                orientation="h",
                marker=dict(
                    color=[
                        "#ffdd57",
                        "#ff3860",
                        "#00d1b2"
                    ]
                )
            )
        )

        fig_shap.update_layout(
            template="plotly_dark",
            height=450,
            title="Importancia de Variables"
        )

        st.plotly_chart(
            fig_shap,
            width='stretch'
        )

        st.markdown("### Evolución Histórica")

        df_hist = calcular_shap_temporal(df_nacional)

        if df_hist is not None:

            fig_hist = go.Figure()

            colores = {
                "Energia": "#ff3860",
                "Transporte": "#ffdd57",
                "Alimentos": "#00d1b2"
            }

            for col in [
                "Energia",
                "Transporte",
                "Alimentos"
            ]:

                fig_hist.add_trace(
                    go.Scatter(
                        x=df_hist["Fecha"],
                        y=df_hist[col],
                        stackgroup="one",
                        mode="lines",
                        name=col,
                        line=dict(
                            width=2,
                            color=colores[col]
                        )
                    )
                )

            fig_hist.update_layout(
                template="plotly_dark",
                height=500,
                hovermode="x unified",
                title="Persistencia Inflacionaria"
            )

            st.plotly_chart(
                fig_hist,
                width='stretch'
            )

    else:
        st.warning("Instalar shap y xgboost")

# =========================================================
# TAB 2 - BAYESIANO
# =========================================================

with tab2:

    st.subheader("Forecast Bayesiano")

    historico = df_nacional["Nivel_General"].tail(24)

    ultimo_valor = historico.iloc[-1]

    meses_pred = 6

    fechas_pred = pd.date_range(
        historico.index[-1] + pd.offsets.MonthBegin(1),
        periods=meses_pred,
        freq="MS"
    )

    simulaciones = np.zeros((1000, meses_pred))

    simulaciones[:, 0] = (
        ultimo_valor *
        np.random.normal(1.02, 0.05, 1000)
    )

    for i in range(1, meses_pred):

        simulaciones[:, i] = (
            simulaciones[:, i-1] *
            np.random.normal(1.01, 0.08, 1000)
        )

    mean_pred = np.mean(simulaciones, axis=0)

    p10 = np.percentile(
        simulaciones,
        10,
        axis=0
    )

    p90 = np.percentile(
        simulaciones,
        90,
        axis=0
    )

    fig_bayes = go.Figure()

    fig_bayes.add_trace(
        go.Scatter(
            x=historico.index,
            y=historico.values,
            mode="lines",
            name="Histórico",
            line=dict(
                color="white",
                width=3
            )
        )
    )

    fig_bayes.add_trace(
        go.Scatter(
            x=list(fechas_pred) + list(fechas_pred)[::-1],
            y=list(p90) + list(p10)[::-1],
            fill="toself",
            fillcolor="rgba(0,209,178,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Credibilidad"
        )
    )

    fig_bayes.add_trace(
        go.Scatter(
            x=fechas_pred,
            y=mean_pred,
            mode="lines",
            line=dict(
                color="#00d1b2",
                width=4
            ),
            name="Forecast"
        )
    )

    fig_bayes.update_layout(
        template="plotly_dark",
        height=600,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_bayes,
        width='stretch'
    )

# =========================================================
# TAB 3 - GRAFOS
# =========================================================

with tab3:

    st.subheader("Propagación del Impacto Inflacionario")

    if HAS_GRAPH:

        net = Network(
            height="650px",
            width="100%",
            bgcolor="#0e1117",
            font_color="white",
            directed=True
        )

        net.barnes_hut()

        # Nodos
        net.add_node(
            "Combustibles",
            size=35,
            color="#ff3860"
        )

        net.add_node(
            "Transporte",
            size=30,
            color="#ffdd57"
        )

        net.add_node(
            "Logística",
            size=25,
            color="#ffdd57"
        )

        net.add_node(
            "Alimentos",
            size=45,
            color="#00d1b2"
        )

        net.add_node(
            "IPC General",
            size=55,
            color="#ffffff"
        )

        # Aristas
        net.add_edge(
            "Combustibles",
            "Transporte",
            value=8
        )

        net.add_edge(
            "Combustibles",
            "Logística",
            value=6
        )

        net.add_edge(
            "Transporte",
            "Alimentos",
            value=7
        )

        net.add_edge(
            "Logística",
            "Alimentos",
            value=5
        )

        net.add_edge(
            "Alimentos",
            "IPC General",
            value=10
        )
        net.set_options("""
            var options = {

            "autoResize": true,

            "layout": {
                "randomSeed": 42,
                "improvedLayout": true
            },

            "physics": {
                "enabled": true,

                "barnesHut": {
                "gravitationalConstant": -5000,
                "centralGravity": 0.35,
                "springLength": 180,
                "springConstant": 0.04,
                "damping": 0.12,
                "avoidOverlap": 0.5
                },

                "stabilization": {
                "enabled": true,
                "iterations": 300,
                "fit": true
                }
            },

            "interaction": {
                "hover": true,
                "dragNodes": true,
                "dragView": true,
                "zoomView": true
            }
            }
            """)


        try:

            path = Path("/tmp/grafo.html")

            net.save_graph(str(path))

            html = path.read_text(
                encoding="utf-8"
            )

            js_fix = """
<script>

window.addEventListener("load", function () {

    try {

        if (typeof network !== "undefined") {

            setTimeout(function () {

                network.fit({
                    animation: {
                        duration: 800,
                        easingFunction: "easeInOutQuad"
                    }
                });

            }, 300);

            network.once(
                "stabilizationIterationsDone",
                function () {

                    network.fit({
                        animation: {
                            duration: 800,
                            easingFunction: "easeInOutQuad"
                        }
                    });

                    network.setOptions({
                        physics: {
                            enabled: false
                        }
                    });

                }
            );
        }

    } catch (e) {
        console.log(e);
    }

});

window.addEventListener("resize", function () {

    try {

        if (typeof network !== "undefined") {

            network.fit({
                animation: false
            });

        }

    } catch (e) {
        console.log(e);
    }

});

</script>

</body>
"""

            html = html.replace(
                "</body>",
                js_fix
            )

            st.components.v1.html(
                html,
                height=720,
                scrolling=False
            )

        except Exception as e:

            st.error(f"Error grafo: {e}")

    else:
        st.warning("Instalar pyvis")

# =========================================================
# TAB 4 - NARRATIVA
# =========================================================

with tab4:

    st.subheader("Narrativa Financiera")

    tendencia = "aceleración"

    if momentum < 0:
        tendencia = "desaceleración"

    st.markdown(f"""
> ⚠️ El sistema detecta una tendencia de **{tendencia}** inflacionaria.

> 📈 El componente con mayor persistencia continúa siendo **Alimentos**.

> 🔍 El análisis SHAP temporal evidencia cambios estructurales en los motores inflacionarios.

> 🛡️ El modelo Bayesiano proyecta un rango probabilístico con 80% de credibilidad.

> 📊 Momentum actual: **{momentum:.2f}%**
""")

# =========================================================
# TAB 5 - UI PREMIUM
# =========================================================

with tab5:

    st.subheader("Experiencia Empresarial")

    c1, c2 = st.columns([2, 1])

    with c1:

        st.markdown("### Tabla Regional")

        regiones = st.multiselect(
            "Regiones",
            options=df["Region"].unique(),
            default=df["Region"].unique()
        )

        df_filtrado = df[
            df["Region"].isin(regiones)
        ]

        if HAS_AGGRID:

            gb = GridOptionsBuilder.from_dataframe(
                df_filtrado.round(2)
            )

            gb.configure_pagination(
                paginationAutoPageSize=False,
                paginationPageSize=20
            )

            gb.configure_side_bar()

            gb.configure_default_column(
                sortable=True,
                filterable=True,
                resizable=True
            )

            grid_options = gb.build()

            AgGrid(
                df_filtrado.round(2),
                gridOptions=grid_options,
                theme="alpine",
                height=500
            )

        else:

            st.dataframe(
                df_filtrado,
                width='stretch'
            )

    with c2:

        st.markdown("### Métrica Dinámica")

        ultimos_6 = (
            df_nacional["Nivel_General"]
            .tail(6)
            .round(2)
            .tolist()
        )

        forecast_6 = (
            pd.Series(mean_pred)
            .round(2)
            .tolist()
        )

        if HAS_ECHARTS:

            options = {

                "backgroundColor": "#0e1117",

                "tooltip": {
                    "trigger": "axis"
                },

                "xAxis": {
                    "type": "category",
                    "data": [
                        "M-5",
                        "M-4",
                        "M-3",
                        "M-2",
                        "M-1",
                        "Actual",
                        "F+1",
                        "F+2",
                        "F+3",
                        "F+4",
                        "F+5",
                        "F+6"
                    ]
                },

                "yAxis": {
                    "type": "value"
                },

                "series": [

                    {
                        "type": "line",
                        "smooth": True,
                        "data": ultimos_6 + forecast_6,
                        "lineStyle": {
                            "width": 4
                        },
                        "areaStyle": {}
                    }

                ]
            }

            st_echarts(
                options=options,
                height="400px"
            )

            st.metric(
                "Inflación Actual",
                f"{ultimo:.2f}%"
            )

            st.metric(
                "Forecast Promedio",
                f"{np.mean(mean_pred):.2f}%"
            )

        else:
            st.warning("Instalar streamlit-echarts")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
Panel Financiero Vanguard • XAI • Forecast • Redes • UX Enterprise
""")