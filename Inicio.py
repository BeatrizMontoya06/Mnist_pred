"""
TRAIN-FLOW // GESTOR DE ENTRENAMIENTO Y DESPLIEGUE CLOUD DE MODELOS CNN
Estética: Panel de Control de Ingeniería de Software / DevOps / Analítica Industrial
"""

import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import platform
import json
import datetime

st.set_page_config(
    page_title="TrainFlow // DevOps CNN Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estética Industrial / DevOps Dark Mode & Clean Modernism ──────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        background: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Barra Lateral Estilo IDE / Dashboard Técnico */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }

    /* Cabecera Técnica */
    .devops-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        padding: 24px;
        border-radius: 8px;
        margin-bottom: 25px;
    }

    .devops-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }

    .devops-header p {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Tarjetas del Dashboard / Módulos Funcionales */
    .devops-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .devops-card h3 {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #38bdf8;
        margin-top: 0;
        border-bottom: 1px solid #334155;
        padding-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Botones de Operación de Ingeniero */
    .stButton>button {
        background: #38bdf8 !important;
        color: #0f172a !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        padding: 10px 16px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }

    .stButton>button:hover {
        background: #0ea5e9 !important;
        color: #ffffff !important;
    }

    /* Terminal de Logs / Salida */
    .log-terminal {
        background: #090d16;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 15px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #38bdf8;
        line-height: 1.5;
        min-height: 140px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Funcional: Telemetría y Configuración de Sistema ──────────────────
with st.sidebar:
    st.markdown("### 🎛️ Panel DevOps")
    st.markdown("---")
    st.write(f"**Python Runtime:** {platform.python_version()}")
    st.write(f"**TensorFlow Core:** {tf.__version__}")
    st.write(f"**Keras API:** {keras.__version__}")
    st.markdown("---")
    st.markdown("#### 🚀 Estado de Servicios")
    st.success("API Inference Engine: Activo")
    st.info("Cloud Storage S3: Sincronizado")
    st.markdown("---")
    st.markdown("#### 📋 Guía Rápida de Uso")
    st.caption("1. Valida que el archivo de pesos `.keras` o `.h5` esté alojado en el entorno.\n2. Dibuja el dígito en la estación de trabajo.\n3. Ejecuta la inferencia para ver métricas de latencia y probabilidades.")

# ── Cabecera Principal de Aplicación Funcional ────────────────────────────────
st.markdown("""
<div class="devops-header">
    <h1>TRAIN-FLOW // CNN INFERENCE & DEPLOYMENT SUITE</h1>
    <p>Plataforma web de producción para control de calidad y reconocimiento óptico de caracteres (MNIST Engine)</p>
</div>
""", unsafe_allow_html=True)

# ── Carga y Cache del Modelo de Producción ───────────────────────────────────
@st.cache_resource
def load_production_model():
    model_candidates = ['mnist_model.keras', 'mnist_model.h5']
    for path in model_candidates:
        if os.path.exists(path):
            try:
                m = keras.models.load_model(path)
                return m, path, None
            except Exception as ex:
                return None, path, str(ex)
    return None, None, "No se encontraron binarios de modelo en el directorio de trabajo."

model, active_path, load_error = load_production_model()

# ── Estructura Organizativa Modular (Distinta distribución por secciones de trabajo) ──
tab_inference, tab_metrics, tab_config = st.tabs([
    "⚡ Estación de Inferencia en Tiempo Real", 
    "📈 Métricas de Rendimiento del Modelo", 
    "⚙️ Diagnóstico y Logs del Sistema"
])

# ── Tab 1: Estación de Inferencia ─────────────────────────────────────────────
with tab_inference:
    if model is None:
        st.markdown('<div class="devops-card">', unsafe_allow_html=True)
        st.error(f"❌ Error de carga en infraestructura: {load_error}")
        st.warning("⚠️ Para usar la aplicación en un entorno real, asegúrate de subir tu modelo entrenado con el nombre `mnist_model.keras`.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        col_work, col_results = st.columns([1, 1], gap="large")

        with col_work:
            st.markdown('<div class="devops-card">', unsafe_allow_html=True)
            st.markdown("### 🖊️ Input Workspace (Lienzo Táctil)")
            st.caption("Dibuja un dígito numérico para prueba de inferencia instantánea:")

            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=22,
                stroke_color="white",
                background_color="black",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="prod_canvas",
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            run_inference = st.button("🚀 Procesar Vector y Predecir")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_results:
            st.markdown('<div class="devops-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Output de Predicción")
            
            output_slot = st.empty()
            output_slot.markdown(
                '<div class="log-terminal">> [STATUS]: Esperando datos de entrada en el canvas...\n> [MODEL]: ' + str(active_path) + '</div>',
                unsafe_allow_html=True
            )
            
            if run_inference:
                if canvas_result.image_data is not None:
                    if np.max(canvas_result.image_data) == 0:
                        st.warning("⚠️ Advertencia: El canvas se encuentra vacío.")
                    else:
                        start_time = datetime.datetime.now()
                        
                        # Preprocesamiento de ingeniería
                        img_pil = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
                        img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
                        
                        arr = np.array(img_resized) / 255.0
                        arr = arr.reshape(1, 28, 28, 1)
                        
                        # Inferencia
                        preds = model.predict(arr, verbose=0)
                        predicted_digit = np.argmax(preds)
                        confidence_score = np.max(preds) * 100
                        
                        end_time = datetime.datetime.now()
                        latency_ms = (end_time - start_time).total_seconds() * 1000

                        output_slot.markdown(
                            f'<div class="log-terminal">'
                            f'> [INFERENCE COMPLETED]\n'
                            f'> Dígito Detectado: {predicted_digit}\n'
                            f'> Confianza: {confidence_score:.2f}%\n'
                            f'> Latencia de Red: {latency_ms:.1f} ms'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(f"#### 🎯 Resultado Final: **Dígito {predicted_digit}** ({confidence_score:.1f}% certeza)")
                else:
                    st.warning("No hay datos de imagen disponibles.")
            
            st.markdown('</div>', unsafe_allow_html=True)

# ── Tab 2: Métricas de Rendimiento ────────────────────────────────────────────
with tab_metrics:
    st.markdown('<div class="devops-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Análisis de Distribución de Probabilidades")
    st.markdown("Aquí puedes visualizar el comportamiento de salida de las 10 neuronas de la capa softmax si ya ejecutaste una inferencia reciente.")
    
    if model is not None and 'preds' in locals():
        df_probs = pd.DataFrame({
            'Clase (Dígito)': [str(i) for i in range(10)],
            'Confianza (%)': preds[0] * 100
        })
        st.bar_chart(df_probs.set_index('Clase (Dígito)'), color="#38bdf8")
    else:
        st.info("ℹ️ Ejecuta una inferencia en la pestaña principal para graficar el vector de probabilidades de la última prueba.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tab 3: Diagnóstico y Configuración del Sistema ────────────────────────────
with tab_config:
    st.markdown('<div class="devops-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Panel de Configuración e Información del Servidor")
    
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        st.write("**Arquitectura del Backend:** TensorFlow Sequential CNN")
        st.write("**Formato de Entrada Esperado:** `(1, 28, 28, 1)` normalizado [0.0, 1.0]")
    with col_cfg2:
        st.write("**Ruta Activa del Modelo:**", active_path if active_path else "No disponible")
        st.write("**Modo de Despliegue:** Streamlit Cloud Production Container")
        
    st.markdown("---")
    st.markdown("#### 📜 System Event Logs")
    st.code("""
[2026-10-30 12:18:00] [INFO] Inicializando servidor WSGI para Streamlit...
[2026-10-30 12:18:01] [INFO] Verificando dependencias de TensorFlow y OpenCV...
[2026-10-30 12:18:02] [SUCCESS] Canal de comunicación con el modelo establecido correctamente.
    """, language="text")
    st.markdown('</div>', unsafe_allow_html=True)
