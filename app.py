import streamlit as st
import cv2
import numpy as np
import pickle
import mediapipe as mp
import json
import itertools
import copy
import requests
from streamlit_lottie import st_lottie

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Gestura AI",
    page_icon="📡",
    layout="centered"
)

# --- 2. ASSETS & STYLING ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load specific animations
lottie_ai = load_lottieurl("https://lottie.host/02e6973e-2b76-4d04-8b6f-453713607062/7t1zJq09W8.json")

# Custom CSS for Neon Glow and "Glass" cards
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
    }
    .gesture-text {
        font-size: 40px;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .confidence-text {
        font-size: 18px;
        color: #b0b0b0;
        margin-top: 5px;
    }
    .instruction-text {
        font-size: 14px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_resources():
    try:
        model_dict = pickle.load(open('./gesture_model.p', 'rb'))
        model = model_dict['model']
        with open('./gesture_map.json', 'r') as f:
            gesture_map = json.load(f)
        return model, gesture_map
    except Exception as e:
        return None, None

model, gesture_map = load_resources()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# --- 4. NORMALIZATION FUNCTION ---
def normalize_hand(landmark_list):
    temp_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_list[0][0], temp_list[0][1]
    for i in range(len(temp_list)):
        temp_list[i][0] -= base_x
        temp_list[i][1] -= base_y
    max_value = 0
    flattened = list(itertools.chain.from_iterable(temp_list))
    for val in flattened:
        if abs(val) > max_value:
            max_value = abs(val)
    if max_value > 0:
        for i in range(len(temp_list)):
            temp_list[i][0] /= max_value
            temp_list[i][1] /= max_value
    return list(itertools.chain.from_iterable(temp_list))

# --- 5. UI LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    if lottie_ai:
        st_lottie(lottie_ai, height=150, key="ai_anim")
    else:
        st.write("🤖")

with col2:
    st.title("Gestura AI")
    st.write("Real-time Sign Language Translator")
    st.markdown('<p class="instruction-text">Ensure your hand is clearly visible and well-lit.</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🧠 Knowledge Base")
    st.write("This AI knows the following signs:")
    if gesture_map:
        for k, v in gesture_map.items():
            st.code(f"{v}", language="text")
    else:
        st.error("Model not loaded properly.")

# --- CAMERA & PREDICTION ---
img_file_buffer = st.camera_input("Scanner", label_visibility="hidden")

if img_file_buffer is not None:
    # Processing Spinner
    with st.spinner("Analyzing Geometry..."):
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        result_name = "Scanning..."
        confidence = 0.0
        glow_color = "#444" # Default grey

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                landmark_list = []
                for lm in hand_lms.landmark:
                    landmark_list.append([lm.x, lm.y, lm.z])
                
                processed_input = normalize_hand(landmark_list)
                input_array = np.array([processed_input])
                
                probabilities = model.predict_proba(input_array)[0]
                best_id = np.argmax(probabilities)
                confidence = probabilities[best_id]
                result_name = gesture_map.get(str(best_id), "Unknown")

                # Dynamic Styling based on confidence
                if confidence > 0.8:
                    glow_color = "#00ff88" # Neon Green
                    text_effect = "linear-gradient(45deg, #00ff88, #00d2ff)"
                else:
                    glow_color = "#ff4b4b" # Neon Red
                    result_name = "Uncertain"
                    text_effect = "linear-gradient(45deg, #ff4b4b, #ff9068)"

    # --- RESULT CARD DISPLAY (HTML INJECTION) ---
    st.markdown(f"""
    <div class="result-card" style="border: 1px solid {glow_color}; box-shadow: 0 0 15px {glow_color}40;">
        <h3 style="margin:0; color:#aaa; font-size:16px;">DETECTED GESTURE</h3>
        <h1 class="gesture-text" style="background: {text_effect}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {result_name}
        </h1>
        <p class="confidence-text">Confidence: {confidence*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Default Empty State Card
    st.markdown("""
    <div class="result-card">
        <h3 style="color:#666;">Waiting for Input...</h3>
        <p style="color:#444; font-size:12px;">Capture an image to start translation</p>
    </div>
    """, unsafe_allow_html=True)