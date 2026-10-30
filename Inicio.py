"""
ASTRA-VIZ // ORÁCULO DE PATRONES SIDERALES
Estética: Alquimia Astronómica / Laboratorio Estelar & Manuscrito Esotérico Digital
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
    page_title="Astra-Viz // Oráculo Sideral",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Estética Alquimia Estelar & Astronomía Oscura ─────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Share+Tech+Mono&display=swap');

    .stApp {
        background: #070913;
        background-image: 
            radial-gradient(circle at 50% 20%, #171c30 0%, #070913 80%),
            linear-gradient(rgba(23, 28, 48, 0.2) 1px, transparent 1px);
        background-size: 100% 100%, 100% 20px;
        color: #d1d8e0;
        font-family: 'Cormorant Garamond', serif;
    }

    header {visibility: hidden;}

    /* Cabecera Esotérica / Observatorio */
    .astral-header {
        border-top: 1px solid #485460;
        border-bottom: 1px solid #485460;
        background: rgba(11, 15, 26, 0.85);
        padding: 30px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0,0,0,0.8), inset 0 0 20px rgba(74, 144, 226, 0.1);
        margin-bottom: 35px;
    }

    .astral-header h1 {
        font-family: 'Cinzel', serif;
        font-size: 2.6rem;
        color: #f1f2f6;
        letter-spacing: 4px;
        margin: 0;
        text-transform: uppercase;
        text-shadow: 0 0 15px rgba(209, 216, 224, 0.3);
    }

    .astral-header p {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.85rem;
        color: #809bce;
        letter-spacing: 3px;
        margin-top: 10px;
        text-transform: uppercase;
    }

    /* Paneles Estilo Pergamino Astral */
    .astral-panel {
        background: #0b0f1a;
        border: 1px solid #1e2736;
        border-top: 2px solid #485460;
        border-radius: 6px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        margin-bottom: 25px;
    }

    .astral-panel h3 {
        font-family: 'Cinzel', serif;
        font-size: 1rem;
        color: #809bce;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0;
        border-bottom: 1px dashed #1e2736;
        padding-bottom: 10px;
    }

    /* Botones de Invocación Astral */
    .stButton>button {
        background: transparent !important;
        color: #f1f2f6 !important;
        border: 1px solid #485460 !important;
        border-radius: 2px !important;
        font-family: 'Cinzel', serif !important;
        font-size: 0.8rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        padding: 14px 20px !important;
        width: 100% !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }

    .stButton>button:hover {
        background: #485460 !important;
        color: #ffffff !important;
        border-color: #809bce !important;
        box-shadow: 0 0 15px rgba(128, 155, 206, 0.4);
    }

    /* Consola de Revelación */
    .revelation-box {
        background: #06080e;
        border: 1px solid #485460;
        border-left: 3px solid #f39c12;
        padding: 25px;
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.3rem;
        line-height: 1.8;
        color: #f1f2f6;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.9);
        border-radius: 2px;
    }

    .stTextInput input {
        background: #06080e !important;
        border: 1px solid #1e2736 !important;
        color: #d1d8e0 !important;
        font-family: 'Share Tech Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Cabecera del Observatorio ─────────────────────────────────────────────────
st.markdown("""
<div class="astral-header">
    <h1>astra-viz // oráculo sideral</h1>
    <p>espejo de inferencia estelar y decodificación de geometría sagrada</p>
</div>
""", unsafe_allow_html=True)

# ── Vinculación del Núcleo Neuronal con Lectura Esotérica ─────────────────────
@st.cache_resource
def load_astral_model():
    model_files = ['mnist_model.keras', 'mnist_model.h5']
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path)
                return model, model_path
            except Exception as e:
                return None, str(e)
    return None, None

model, load_status = load_astral_model()

if model is None:
    st.markdown('<div class="astral-panel">', unsafe_allow_html=True)
    st.error("⚠️ El observatorio está ciego: No se detecta el pergamino del modelo (`mnist_model.keras`).")
    st.info("Coloque los pesos del modelo en la raíz del sistema para abrir el portal de adivinación.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ── Estructura de Trabajo en Dos Altares (Lienzo y Consola) ───────────────────
altar_left, altar_right = st.columns([1.1, 1], gap="large")

with altar_left:
    st.markdown('<div class="astral-panel">', unsafe_allow_html=True)
    st.markdown("### 🌌 El Espejo de Trazos Siderales")
    st.caption("Dibuja un símbolo numérico (0 al 9) sobre el firmamento oscuro:")
    
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=25,
        stroke_color="white",
        background_color="black",
        width=320,
        height=320,
        drawing_mode="freedraw",
        key="canvas_astral_oracle",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with altar_right:
    st.markdown('<div class="astral-panel">', unsafe_allow_html=True)
    st.markdown("### 📜 Ritos y Mandatos")
    
    divine_btn = st.button("🔮 Consultar al Oráculo Sideral")
    purge_btn = st.button("🧹 Disipar los Trazos del Lienzo")
    
    if purge_btn:
        st.rerun()
        
    st.markdown("---")
    st.markdown("""
    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: #485460; line-height: 1.6;">
    [ESTADO: EL ESPEJO ESTelar ESTÁ CALIBRADO]<br>
    Los vectores latentes traducirán la energía de tu pulso en una constelación matemática definitiva.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Ceremonia de Inferencia y Revelación ─────────────────────────────────────
if divine_btn:
    if canvas_result.image_data is not None:
        if np.max(canvas_result.image_data) == 0:
            st.warning("⚠️ El firmamento está completamente vacío. Dibuja un símbolo para invocar la respuesta.")
        else:
            with st.spinner("✨ Leyendo las alineaciones y frecuencias tensoriales..."):
                image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                
                img_array = np.array(image_resized) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                
                prediction = model.predict(img_array, verbose=0)
                digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                alternative = np.argsort(prediction[0])[-2]

            # ── Resultados en Formato de Profecía / Manuscrito ──────────────
            st.markdown('<div class="astral-panel">', unsafe_allow_html=True)
            st.markdown("### 👁️ El Decreto del Oráculo")
            
            st.markdown(
                f'<div class="revelation-box">'
                f'Las corrientes cósmicas han convergido. El símbolo manifestado en el altar corresponde al dígito sagrado <b>{digit}</b>, '
                f'con una resonancia y certeza del <b>{confidence:.1f}%</b>. La segunda lectura en importancia apunta hacia el arquetipo <b>{alternative}</b>.'
                f'</div>',
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(label="Símbolo Primario", value=str(digit))
            with m_col2:
                st.metric(label="Certeza Astral", value=f"{confidence:.1f}%")
            with m_col3:
                st.metric(label="Eco Alternativo", value=str(alternative))
            
            st.markdown("---")
            st.markdown("#### 📊 Espectro de Energías Numéricas (Probabilidades)")
            
            prob_df = pd.DataFrame({
                'Símbolo': [str(i) for i in range(10)],
                'Resonancia (%)': prediction[0] * 100
            })
            st.bar_chart(prob_df.set_index('Símbolo'), color="#809bce")
            
            view_col1, view_col2 = st.columns(2)
            with view_col1:
                st.write("**Constelación Original Registrada:**")
                st.image(canvas_result.image_data, width=160)
            with view_col2:
                st.write("**Esencia Reducida (28x28):**")
                st.image(image_resized, width=160)
                
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ No se detectó ninguna forma geométrica en el lienzo.")
