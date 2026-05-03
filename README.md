# app_inflacion_ml_opcional
analisis de IPC inflacion con modelos LSTM
# Reporte Técnico de Arquitectura: Analytics-IPC AI

    ## 1. Stack Tecnológico y Lenguajes
    El proyecto ha sido desarrollado bajo un ecosistema de **Python 3.12+**, seleccionado por su madurez en el manejo de series temporales y redes neuronales.

    - **Frontend & UI**: Se utiliza **Streamlit** para la creación de interfaces reactivas, permitiendo una visualización de datos en tiempo real sin la necesidad de un backend desacoplado complejo.
    - **Motores de Inferencia**:
    - **PyTorch**: Implementado para modelos GRU y TCN por su flexibilidad en el manejo de tensores dinámicos.
    - **TensorFlow/Keras**: Utilizado en arquitecturas LSTM por su eficiencia en capas recurrentes unistep. 
    - **Análisis Estadístico**: **Statsmodels** para el despliegue de modelos SARIMA y descomposiciones estacionales (Trend, Seasonal, Resid).
    - **Machine Learning**: **Scikit-learn** para el preprocesamiento de datos (MinMaxScaler) y modelos de ensamble (Random Forest).

    ## 2. Lógica de Ingeniería de Datos
    Se implementó un motor de **parsing dinámico** para archivos Excel del INDEC. Dada la variabilidad en los formatos gubernamentales, el sistema localiza automáticamente la fila de cabecera mediante escaneo de palabras clave ("Nivel general") y realiza una limpieza de datos en caliente utilizando `pandas`, asegurando que los tipos de datos sean compatibles con los modelos predictivos.

    ## 3. Innovación en Métricas (Física Financiera)
    El software no se limita a proyecciones lineales; integra conceptos avanzados de análisis de señales:
    - **Entropía de Shannon**: Aplicada para medir la incertidumbre o el "caos" en la variación de precios. Una entropía alta indica una menor confiabilidad en la predicción debido a shocks externos.
    - **Aceleración e Inercia**: Cálculo de derivadas de segundo orden sobre la serie temporal para identificar cambios bruscos en la tendencia inflacionaria antes de que se reflejen en la media móvil.

    ## 4. Ventajas de la Aplicación
    ### Desde el Análisis de Datos:
    - **Reducción de Sesgo**: Al comparar modelos estadísticos (SARIMA) con redes neuronales, se obtiene una visión balanceada que evita el sobreajuste (overfitting).
    - **Descomposición Granular**: Capacidad de aislar el ruido estacional para entender la inflación núcleo (Core Inflation).

    ### Desde la Perspectiva Financiera:
    - **Toma de Decisiones Estratégicas**: Herramienta clave para la proyección de flujos de caja y ajustes presupuestarios.
    - **Mitigación de Riesgos**: La métrica de aceleración sirve como un sistema de alerta temprana para la renegociación de contratos o coberturas financieras.

    # Predicciones IPC - Fintech Analytics Dashboard
    Una plataforma analítica avanzada desarrollada para proyectar la variación mensual de la inflación (IPC) utilizando un enfoque multimodelo que combina estadística clásica, Machine Learning y Deep Learning. 

    Este proyecto está diseñado como una solución de Business Intelligence orientada a finanzas, permitiendo transformar datos crudos en insights accionables mediante una interfaz interactiva y de alta fidelidad.

    ## Características Principales
    * **Arquitectura Multimodelo:** Implementación y comparación en tiempo real de algoritmos SARIMA, Holt-Winters, Random Forest Regressor, y arquitecturas de redes neuronales (LSTM unistep, GRU y TCN).
    * **Física Financiera:** Cálculo de métricas avanzadas como la *Aceleración* de precios y la *Entropía* (caos del mercado) para evaluar la volatilidad.
    * **Procesamiento Dinámico de Datos:** Pipeline de ETL integrado que lee, limpia y transforma dinámicamente archivos `.xlsx` complejos del INDEC.
    * **Diseño Centrado en UX:** Interfaz limpia sin barras laterales intrusivas, con componentes HTML/CSS/JS inyectados para una experiencia fluida.

    ## Stack Tecnológico

    * **Lenguaje Core:** Python
    * **Frontend & Data App:** Streamlit
    * **Deep Learning:** PyTorch, TensorFlow / Keras
    * **Machine Learning & Estadística:** Scikit-learn, Statsmodels, SciPy
    * **Visualización:** Plotly Graph Objects

    ## ⚙️ Instalación y Uso


con modelos opcionales adicionados 
    1. Clonar el repositorio:
    git clone [https://github.com/wgekko/
    cd tu-repositorio
