"""
NEO-MNIST LABS // NEURAL CANVAS STUDIO
Estética: Minimalismo Geométrico de Alta Precisión & Neo-Brutalism Digital
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

st.set_page_config(
    page_title="NEO-MNIST // Neural Canvas",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Estética Neo-Brutalism & Minimalismo Geométrico ──────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

    .stApp {
        background: #f4f4f0;
        color: #111111;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    header {visibility: hidden;}

    /* Cabecera Principal Minimalista de Autor */
    .studio-header {
        border-bottom: 2px solid #111111;
        padding-bottom: 20px;
        margin-bottom: 30px;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }

    .studio-header h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #111111;
        letter-spacing: -1px;
        margin: 0;
    }

    .studio-header p {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #666666;
        margin: 0;
        text-transform: uppercase;
    }

    /* Tarjetas de Contenedores Sólidos (Neo-Brutalism Style) */
    .neo-card {
        background: #ffffff;
        border: 2px solid #111111;
        border-radius: 8px;
        padding: 25px;
        box-shadow: 4px 4px 0px #111111;
        margin-bottom: 20px;
    }

    .neo-card h3 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #111111;
        margin-top: 0;
        border-bottom: 2px solid #111111;
        padding-bottom: 8px;
    }

    /* Botones de Alto Contraste */
    .stButton>button {
        background: #111111 !important;
        color: #ffffff !important;
        border: 2px solid #111111 !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        padding: 12px 20px !important;
        width: 100% !important;
        box-shadow: 3px 3px 0px #666666 !important;
        transition: all 0.1s ease !important;
    }

    .stButton>button:hover {
        background: #ff3333 !important;
        border-color: #111111 !important;
        box-shadow: 1px 1px 0px #111111 !important;
        transform: translate(2px, 2px);
    }

    /* Estilización de Inputs y Métricas */
    .metric-box {
        background: #111111;
        color: #ffffff;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ── Cabecera de Interfaz Única ────────────────────────────────────────────────
st.markdown("""
<div class="studio-header">
    <div>
        <h1>neo-mnist // canvas studio</h1>
        <p>estación de inferencia y reconocimiento de patrones escritos a mano</p>
    </div>
    <div>
        <p style="text-align: right;">TF v2.x // Keras Core</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Carga del Modelo MNIST con Manejo de Errores Estilizado ───────────────────
@st.cache_resource
def load_mnist_model():
    model_files = ['mnist_model.keras', 'mnist_model.h5']
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path)
                return model, model_path
            except Exception as e:
                return None, str(e)
    return None, None

model, load_status = load_mnist_model()

if model is None:
    st.markdown('<div class="neo-card">', unsafe_allow_html=True)
    st.error("❌ No se ha detectado ningún núcleo neuronal entrenado en el directorio.")
    st.info("Coloque un archivo `mnist_model.keras` o `mnist_model.h5` en la raíz del repositorio para habilitar los circuitos de inferencia.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ── Distribución de Paneles de Trabajo ────────────────────────────────────────
col_canvas, col_control = st.columns([1.2, 1], gap="large")

with col_canvas:
    st.markdown('<div class="neo-card">', unsafe_allow_html=True)
    st.markdown("### 🎨 Lienzo de Inferencia Táctil")
    st.caption("Dibuja un dígito numérico (0 al 9) dentro del recuadro negro:")
    
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=24,
        stroke_color="white",
        background_color="black",
        width=320,
        height=320,
        drawing_mode="freedraw",
        key="canvas_mnist",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_control:
    st.markdown('<div class="neo-card">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Consola de Control")
    
    predict_btn = st.button("⚡ Ejecutar Inferencia de Dígito")
    clear_btn = st.button("🗑️ Limpiar Área de Dibujo")
    
    if clear_btn:
        st.rerun()
        
    st.markdown("---")
    st.markdown("<p style='font-family: JetBrains Mono; font-size: 0.8rem;'><b>Estado del Núcleo:</b> Conectado y listo para procesar vectores de entrada de 28x28 píxeles.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Lógica de Predicción y Procesamiento de Imagen ───────────────────────────
if predict_btn:
    if canvas_result.image_data is not None:
        if np.max(canvas_result.image_data) == 0:
            st.warning("⚠️ El lienzo se encuentra vacío. Por favor, dibuja un dígito antes de procesar.")
        else:
            with st.spinner("Procesando matriz tensorial..."):
                # Conversión y redimensionamiento exacto para MNIST
                image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                
                img_array = np.array(image_resized) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                
                # Inferencia con el modelo cargado
                prediction = model.predict(img_array, verbose=0)
                digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                alternative = np.argsort(prediction[0])[-2]

            # ── Resultados Visuales Estilizados ──────────────────────────────
            st.markdown('<div class="neo-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Reporte de Análisis Neuronal")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(label="Predicción Principal", value=str(digit))
            with res_col2:
                st.metric(label="Nivel de Confianza", value=f"{confidence:.1f}%")
            with res_col3:
                st.metric(label="Alternativa Probable", value=str(alternative))
            
            st.markdown("---")
            
            # Gráfico de probabilidades minimalista
            prob_df = pd.DataFrame({
                'Dígito': [str(i) for i in range(10)],
                'Probabilidad (%)': prediction[0] * 100
            })
            st.bar_chart(prob_df.set_index('Dígito'), color="#111111")
            
            # Comparativa de transformaciones
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                st.write("**Muestra Original Vectorizada:**")
                st.image(canvas_result.image_data, width=150)
            with sub_c2:
                st.write("**Entrada Normalizada (28x28):**")
                st.image(image_resized, width=150)
                
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ No se detectaron datos en el lienzo de dibujo.")
