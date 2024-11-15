# Volkswagen_prototype

Volkswagen Adaptive Learning System and Emotion-Aware Cabin Environment prototype. This prototype leverages machine learning, real-time emotion detection, and a Streamlit-based user interface to simulate how a smart car system can enhance driving safety, comfort, and personalization.

Objective
The goal of the prototype is to:

Analyze driving behavior to classify it as either "calm" or "aggressive" using machine learning.
Detect driver emotions (like "happy" or "stressed") in real-time using facial recognition via a webcam.
Adjust in-cabin settings (like lighting and music) based on both driving behavior and detected emotions to improve the driving experience.
Provide a user-friendly interface to visualize how these adaptive systems work using Streamlit.

Working Breakdown
Step 1: Simulating Driving Behavior Data

The first part of the prototype is about classifying driving behavior using a machine learning model.

Data Preparation:
The system uses a simulated dataset that includes features like:
Speed (e.g., how fast the car is moving).
Brake Intensity (e.g., how hard the brakes are applied).
Steering Angle (e.g., how sharply the car is turning).
Each record is labeled as either "calm" or "aggressive" based on the driving pattern.
Model Training:
We use a Random Forest Classifier to learn from this dataset.
The model is trained to recognize patterns in driving behavior to classify new input data as either "calm" or "aggressive".

Step 2: Real-Time Emotion Detection Using a Webcam

The second part focuses on detecting the driver's current emotional state using a webcam.

Facial Recognition:
The system uses OpenCV with a pre-trained Haar Cascade Classifier to detect faces.
Once a face is detected, it uses simulated logic to determine the driver's emotion.
If the face position is detected in certain areas, it classifies the emotion as "happy" or "stressed" (for simplicity, we're using dummy logic here, but this can be replaced with a real deep learning emotion recognition model).

Step 3: Adjusting Cabin Settings Based on Inputs

The system then combines the outputs from the driving behavior model and the emotion detection system to adjust the in-cabin settings.

Cabin Settings Adjustment:
The system maps driving styles and emotions to specific cabin adjustments.
The adjustments include:
Lighting: Changes to soft, bright, or dim lighting based on the context.
Music: Adjusts the music to relaxing, upbeat, neutral, or calming tracks.
<img width="735" alt="Screenshot 2024-11-15 at 10 49 57 PM" src="https://github.com/user-attachments/assets/70bacb70-410d-4fde-b204-3cc9d02e44ef">


Step 4: Streamlit User Interface

The final part of the prototype is the Streamlit-based user interface that allows users to interact with the system.

User Input Simulation:
The interface provides sliders for users to simulate driving data (like speed, brake intensity, and steering angle).
It also includes a button to activate the webcam for real-time emotion detection.

Dynamic Output:
The system predicts the driving style based on the slider inputs and displays it.
It captures the emotion using the webcam and displays the detected emotion.
Finally, it shows the adjusted cabin settings based on the driving behavior and emotion.
User Interaction Flow:
Adjust Sliders: The user sets the speed, brake intensity, and steering angle using sliders.
Predict Driving Style: The system uses the trained model to predict if the driving is "calm" or "aggressive".
Start Emotion Detection: The user activates the webcam to detect their current emotion.
View Adjustments: The system dynamically adjusts and displays the cabin settings based on the detected inputs.
