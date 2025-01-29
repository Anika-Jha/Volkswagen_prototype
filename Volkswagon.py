#Volkswagen Adaptive Learning System and Emotion-Aware Cabin Environment prototype.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import cv2
import streamlit as st
import time
# Step 1: Prepare Driving Data
#taking random data , can be modified
def generate_driving_data():
    data = {
        'speed': [30, 60, 45, 70, 55, 80, 40, 65],
        'brake_intensity': [0.2, 0.8, 0.5, 0.9, 0.6, 0.4, 0.3, 0.7],
        'steering_angle': [5, 15, 10, 20, 12, 25, 8, 18],
        'label': ['calm', 'aggressive', 'calm', 'aggressive', 'calm', 'aggressive', 'calm', 'aggressive']
    }
    return pd.DataFrame(data)

# Step 2: Train the Driving Style Model
def train_driving_model(df):
    X = df[['speed', 'brake_intensity', 'steering_angle']]
    y = df['label']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Generate data and train model
driving_data = generate_driving_data()
driving_model = train_driving_model(driving_data)
# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Dummy function to simulate emotion detection
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Simulate emotion based on face position
    for (x, y, w, h) in faces:
        if x % 2 == 0:
            return 'happy'
        else:
            return 'stressed'
    return 'neutral'
def adjust_cabin_settings(driving_style, emotion):
    # Define cabin adjustments based on driving style and emotion
    adjustments = {
        'calm': {'lighting': 'soft', 'music': 'relaxing'},
        'aggressive': {'lighting': 'dim', 'music': 'neutral'},
        'happy': {'lighting': 'bright', 'music': 'upbeat'},
        'stressed': {'lighting': 'soothing', 'music': 'calming'}
    }
    
    # Determine settings based on detected style and emotion
    settings = adjustments.get(driving_style, {}).copy()
    if emotion in adjustments:
        settings.update(adjustments[emotion])
    
    return settings
# Streamlit Dashboard
st.title("Volkswagen Adaptive Learning & Emotion-Aware Cabin Prototype")

# Sidebar for Driving Style Simulation
st.sidebar.header("Simulate Driving Style")
speed = st.sidebar.slider("Speed", 0, 100, 50)
brake_intensity = st.sidebar.slider("Brake Intensity", 0.0, 1.0, 0.5)
steering_angle = st.sidebar.slider("Steering Angle", 0, 30, 10)

# Predict Driving Style
new_data = np.array([[speed, brake_intensity, steering_angle]])
predicted_style = driving_model.predict(new_data)[0]
st.write(f"Detected Driving Style: **{predicted_style}**")

# Real-Time Emotion Detection
st.header("Real-Time Emotion Detection")
st.write("Turn on your webcam to detect emotion in real-time")

# Webcam Capture
cap = cv2.VideoCapture(0)
emotion = 'neutral'
if st.button('Start Emotion Detection'):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        emotion = detect_emotion(frame)
        
        # Display the video with emotion label
        cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        st.image(frame, channels="BGR")
        
        # Stop capture after 5 seconds for demo purposes
        time.sleep(5)
        break
    cap.release()

st.write(f"Detected Emotion: **{emotion}**")

# Display Adjusted Cabin Settings
cabin_settings = adjust_cabin_settings(predicted_style, emotion)
st.subheader("Adjusted Cabin Settings")
st.write(f"Lighting: {cabin_settings.get('lighting', 'default')}")
st.write(f"Music: {cabin_settings.get('music', 'default')}")
