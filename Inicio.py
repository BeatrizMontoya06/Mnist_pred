"""
📖 EL BOSQUE MÁGICO DE LOS NÚMEROS // CUENTO INTERACTIVO INFANTIL
Estética: Ilustración de Cuento de Hadas / Acuarela Mágica y Fantasía Infantil
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
    page_title="El Bosque Mágico de los Números",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Estética de Cuento Infantil, Acuarela y Fantasía Mágica ────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Quicksand:wght@500;700&display=swap');

    .stApp {
        background: linear-gradient(180deg, #e0f7fa 0%, #f1f8e9 50%, #fffde7 100%);
        color: #2c3e50;
        font-family: 'Quicksand', sans-serif;
    }

    header {visibility: hidden;}

    /* Cabecera estilo Libro de Cuentos Ilustrado */
    .storybook-header {
        background: rgba(255, 255, 255, 0.85);
        border: 4px dashed #81c784;
        border-radius: 24px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(46, 125, 50, 0.15);
        margin-bottom: 30px;
    }

    .storybook-header h1 {
        font-family: 'Fredoka One', cursive;
        font-size: 2.8rem;
        color: #2e7d32;
        margin: 0;
        text-shadow: 2px 2px 0px #c8e6c9;
    }

    .storybook-header p {
        font-family: 'Quicksand', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #558b2f;
        margin-top: 10px;
    }

    /* Tarjetas estilo Cabaña del Bosque / Pergamino Mágico */
    .fairy-card {
        background: rgba(255, 255, 255, 0.92);
        border: 3px solid #aed581;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(139, 195, 74, 0.2);
        margin-bottom: 20px;
    }

    .fairy-card h3 {
        font-family: 'Fredoka One', cursive;
        font-size: 1.3rem;
        color: #33691e;
        margin-top: 0;
        border-bottom: 2px dotted #c8e6c9;
        padding-bottom: 10px;
    }

    /* Botones Mágicos Interactivos */
    .stButton>button {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        font-family: 'Fredoka One', cursive !important;
        font-size: 1rem !important;
        letter-spacing: 1px !important;
        padding: 14px 24px !important;
        width: 100% !important;
        box-shadow: 0 6px 15px rgba(67, 160, 71, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #81c784 0%, #66bb6a 100%) !important;
        transform: scale(1.03);
        box-shadow: 0 8px 20px rgba(67, 160, 71, 0.6) !important;
    }

    /* Tarjeta de narración del cuento */
    .story-bubble {
        background: #f1f8e9;
        border: 2px dashed #7cb342;
        border-radius: 16px;
        padding: 20px;
        font-family: 'Quicksand', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #33691e;
        line-height: 1.6;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ── Cabecera del Cuento Mágico ────────────────────────────────────────────────
st.markdown("""
<div class="storybook-header">
    <h1>🍄 El Bosque Mágico de los Números 🌳</h1>
    <p>¡Ayuda al Duendecillo Sabio a adivinar el número secreto que dibujes con tu varita mágica!</p>
</div>
""", unsafe_allow_html=True)

# ── Carga del Modelo MNIST con Mensajes Temáticos de Fantasía ─────────────────
@st.cache_resource
def load_fairy_model():
    model_files = ['mnist_model.keras', 'mnist_model.h5']
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path)
                return model, model_path
            except Exception as e:
                return None, str(e)
    return None, None

model, load_status = load_fairy_model()

if model is None:
    st.markdown('<div class="fairy-card">', unsafe_allow_html=True)
    st.error("🧚‍♂️ ¡Oh no! El espíritu del bosque no encuentra el pergamino del modelo (`mnist_model.keras`).")
    st.info("Coloca tu archivo de entrenamiento en la raíz del bosque para despertar la magia de la predicción.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ── Distribución en el Bosque (Lienzo y Consola del Duende) ───────────────────
col_art, col_quest = st.columns([1.2, 1], gap="large")

with col_art:
    st.markdown('<div class="fairy-card">', unsafe_allow_html=True)
    st.markdown("### 🎨 El Lienzo Encantado")
    st.caption("Dibuja un número mágico del 0 al 9 usando tu ratón o dedo como varita:")
    
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=26,
        stroke_color="white",
        background_color="black",
        width=300,
        height=300,
        drawing_mode="freedraw",
        key="canvas_fairy_forest",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_quest:
    st.markdown('<div class="fairy-card">', unsafe_allow_html=True)
    st.markdown("### 🪄 Hechizos y Acciones")
    
    predict_btn = st.button("✨ ¡Consultar al Duende Sabio!")
    clear_btn = st.button("🧹 Borrar el Pizarrón Mágico")
    
    if clear_btn:
        st.rerun()
        
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.9rem; color: #558b2f; line-height: 1.5;">
    <b>Leyenda del Bosque:</b><br>
    Cada trazo que dibujas emite un destello de luz que las hadas recolectan para adivinar el mensaje secreto del número.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Lógica Mágica de Inferencia y Narrativa de Cuento ────────────────────────
if predict_btn:
    if canvas_result.image_data is not None:
        if np.max(canvas_result.image_data) == 0:
            st.warning("🍃 ¡El lienzo está vacío! Las hadas necesitan ver un número dibujado para cantar su respuesta.")
        else:
            with st.spinner("✨ Las estrellas fugaces están descifrando tu dibujo..."):
                image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                
                img_array = np.array(image_resized) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                
                prediction = model.predict(img_array, verbose=0)
                digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                alternative = np.argsort(prediction[0])[-2]

            # ── Cuento de Resultados Mágico ──────────────────────────────────
            st.markdown('<div class="fairy-card">', unsafe_allow_html=True)
            st.markdown("### 📖 Capítulo Revelado: El Mensaje del Bosque")
            
            st.markdown(
                f'<div class="story-bubble">'
                f'🌲✨ ¡El anciano búho del gran árbol ha hablado! Tras mirar atentamente tu trazo magico, '
                f'ha descubierto que dibujaste el número <b>{digit}</b> con una seguridad del <b>{confidence:.1f}%</b>. '
                f'Los duendes celebran bailando en círculo bajo los hongos luminosos.'
                f'</div>',
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            c_m1, c_m2, c_m3 = st.columns(3)
            with c_m1:
                st.metric(label="Número Encontrado", value=str(digit))
            with c_m2:
                st.metric(label="Magia Revelada", value=f"{confidence:.1f}%")
            with c_m3:
                st.metric(label="Adivinanza Vecina", value=str(alternative))
            
            st.markdown("---")
            st.markdown("#### 📊 El Coro de los Animales del Bosque (Probabilidades)")
            prob_df = pd.DataFrame({
                'Número': [str(i) for i in range(10)],
                'Canto (%)': prediction[0] * 100
            })
            st.bar_chart(prob_df.set_index('Número'), color="#66bb6a")
            
            fairy_c1, fairy_c2 = st.columns(2)
            with fairy_c1:
                st.write("**Tu dibujo original en el claro del bosque:**")
                st.image(canvas_result.image_data, width=140)
            with fairy_c2:
                st.write("**La esencia pura del número (28x28):**")
                st.image(image_resized, width=140)
                
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ No se encontraron huellas de magia en el lienzo.")
