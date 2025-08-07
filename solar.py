import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.title("SolarGuard: Solar Panel Defect Detection")

# Load model
model = load_model("solar_panel_transfer_learning_model.h5")
class_names = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-covered"]

# Upload image
uploaded_file = st.file_uploader("Upload a solar panel image", type=["jpg", "png"])

if uploaded_file:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    pred = model.predict(img_array).argmax(axis=1)[0]
    result = class_names[pred]

    st.image(img_display, caption="Uploaded Image")
    st.write(f"**Prediction**: {result}")
    
    if result in ["Bird-drop","Dusty","Snow-covered"]:
        st.write("**Recommendation**: Schedule the cleaning immediately.")
    elif result in ["Electrical-damage", "Physical-Damage"]:
        st.write("**Recommendation**: Schedule the repair or replacement of the solar panel.")
    else:
        st.write("**Recommendation**: Solar panel is in Good Condition.")