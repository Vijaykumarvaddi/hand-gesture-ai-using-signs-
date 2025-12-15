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
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load animation (Safe Mode)
lottie_ai = load_lottieurl("https://lottie.host/02e6973e-2b76-4d04-8b6f-453713607062/7t1zJq09W8.json")

# Custom CSS - SIMPLIFIED FOR VISIBILITY
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .result-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.2);
        margin-top: 20px;
    }
    .gesture-text {
        font-size: 50px;
        font-weight: 800;
        margin: 0;
        /* Removed the complex gradient text fill that causes invisible text */
        color: #ffffff; 
        text-shadow: 0 0 10px rgba(255,255,255,0.5);
    }
    .confidence-text {
        font-size: 20px;
        color: #cccccc;
        margin-top: 10px;
        font-style: italic;
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
        st.write("🤖 AI Ready")

with col2:
    st.title("Gestura AI")
    st.write("Real-time Sign Language Translator")

# --- SIDEBAR DEBUG ---
with st.sidebar:
    st.header("System Status")
    if model:
        st.success("Model Loaded ✅")
    else:
        st.error("Model Missing ❌")
        
    st.write("---")
    st.write("Available Signs:")
    if gesture_map:
        for k, v in gesture_map.items():
            st.code(f"{v}", language="text")

# --- CAMERA & PREDICTION ---
img_file_buffer = st.camera_input("Scanner", label_visibility="visible")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Defaults
    result_name = "No Hand Detected"
    confidence = 0.0
    text_color = "#ff4b4b" # Red by default
    glow_color = "#ff4b4b"

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

            # High Confidence = Green / Low = Orange
            if confidence > 0.8:
                text_color = "#00ff88" # Neon Green
                glow_color = "#00ff88"
            else:
                text_color = "#ffcc00" # Orange
                glow_color = "#ffcc00"
                result_name = f"{result_name} (?)"

    # --- RESULT CARD (Clean & Visible) ---
    st.markdown(f"""
    <div class="result-card" style="border-color: {glow_color}; box-shadow: 0 0 20px {glow_color}40;">
        <p style="color:#aaa; font-size:14px; margin-bottom:5px;">PREDICTED SIGN</p>
        <h1 class="gesture-text" style="color: {text_color}; text-shadow: 0 0 15px {glow_color};">
            {result_name}
        </h1>
        <p class="confidence-text">Confidence: {confidence*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
