import streamlit as st
import cv2
import numpy as np
import pickle
import mediapipe as mp
import json
import itertools
import copy
import gc  # Garbage Collector

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Gestura AI",
    page_icon="📡",
    layout="centered"
)

# --- 2. LIGHTWEIGHT MODEL LOADING ---
@st.cache_resource
def load_resources():
    try:
        # Load model
        model_dict = pickle.load(open('./gesture_model.p', 'rb'))
        model = model_dict['model']
        
        # Load labels
        with open('./gesture_map.json', 'r') as f:
            gesture_map = json.load(f)
            
        return model, gesture_map
    except Exception as e:
        return None, None

model, gesture_map = load_resources()

# --- 3. MEDIAPIPE SETUP (OPTIMIZED) ---
mp_hands = mp.solutions.hands
# model_complexity=0 is the "Lite" version (Faster & Less RAM)
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5,
    model_complexity=0 
)

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
st.title("Gestura AI")
st.write("Liter version Optimized for Speed & Low Memory")

# --- 6. OPTIMIZED CAMERA INPUT ---
img_file_buffer = st.camera_input("Scanner", label_visibility="visible")

if img_file_buffer is not None:
    # 1. Read Image
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # 2. Resize Image (Crucial for Memory Saving)
    # Reducing 1080p/720p to small size saves ~70% RAM
    cv2_img = cv2.resize(cv2_img, (640, 480))
    
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    result_name = "No Hand Detected"
    confidence = 0.0
    color = "#ff4b4b"

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

            if confidence > 0.8:
                color = "#00ff88"
            else:
                color = "#ffcc00"
                result_name += " (?)"

    # Result Card
    st.markdown(f"""
    <div style="background-color: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 20px; text-align: center; border: 1px solid {color};">
        <h2 style="color: {color}; margin:0;">{result_name}</h2>
        <p style="color: #ccc; margin:0;">{confidence*100:.1f}% Confidence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. FORCE MEMORY CLEANUP
    del bytes_data, cv2_img, img_rgb, results
    gc.collect()
