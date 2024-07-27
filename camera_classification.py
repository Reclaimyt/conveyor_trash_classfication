import cv2
import streamlit as st
from PIL import Image
from mqtt_config import send_to_mqtt
import tensorflow.keras.applications.mobilenet_v2 as mobilenetv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import time

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('../base_model.h5')
    return model

# Potrait Mode
def bounding_box(frame):
    height, width, _ = frame.shape
    roi_width = int(width * 0.65)  # 65% of the width
    roi_height = int(height * 0.2)  # 2`0% of the height
    top = (height - roi_height) // 2
    left = (width - roi_width) // 2

    cv2.rectangle(frame, (left, top), (left + roi_width, top + roi_height), (0, 255, 0), 2)
    roi_for_prediction = frame[top:top + roi_height, left:left + roi_width]

    return frame, roi_for_prediction

def process_frame(frame):
    height, width, _ = frame.shape
    new_dim = min(height, width)
    top = (height - new_dim) // 2
    left = (width - new_dim) // 2
    cropped_frame = frame[top:top + new_dim, left:left + new_dim]

    return cropped_frame

# Predict function
def predict(model, cropped_frame):
    frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    
    # Convert frame to PIL image and resize
    img_arr = Image.fromarray(frame_rgb)
    img = img_arr.resize((224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_confidence = np.max(predictions, axis=1)

    return predicted_class, predicted_confidence

# Streamlit app
st.title("Webcam Live Feed with Trash Classification")
run = st.checkbox('Run')

FRAME_WINDOW = st.image([])

# Threshold confidence
CONFIDENCE_THRESHOLD = 0.80  
CONFIDENCE_THRESHOLD_TO_DB = 0.95

MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "/sic/kelompok15/conveyor"

model = load_model()
camera = cv2.VideoCapture(0)

predictions_placeholder = st.empty()
message_placeholder = st.empty()

CATEGORIES = {
    0: 'organik',
    1: 'organik',
    2: 'b3',
    3: 'anorganik',
    4: 'organik',
    5: 'organik',
    6: 'b3',
    7: 'anorganik',
    8: 'organik',
    9: 'anorganik',
    10: 'anorganik',
    11: 'organik',
    12: 'organik',
    13: 'anorganik',
}

while run:
    ret, frame = camera.read()
    # if not ret:
    #     st.write("Failed to read frame from camera.")
    #     continue

    frame, roi_for_prediction = bounding_box(frame)
    cropped_frame = process_frame(frame)

    frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    
    FRAME_WINDOW.image(img.resize((600, 600)))
    
    results, confidence = predict(model, roi_for_prediction)

    if results is not None:
        label = results[0]
        confidence = confidence[0]
    
        with predictions_placeholder.container():
            if confidence >= CONFIDENCE_THRESHOLD:
                st.write("Predictions:")
                st.write(f"Class: {label}")
                st.write(f"Category: {CATEGORIES[label]}")
                st.write(f"Confidence: {confidence * 100:.2f}%")

                if confidence >= CONFIDENCE_THRESHOLD_TO_DB:
                    send_to_mqtt(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, f"{CATEGORIES[label]}")

                    camera.release()
                    
                    with message_placeholder:
                        st.write("Data sent to MQTT. Pausing camera for 3 seconds.")
                    
                    time.sleep(3)
                    message_placeholder.empty()
                    camera = cv2.VideoCapture(0)
            else:
                st.write("Predictions:")
                st.write("Confidence below threshold, no prediction displayed.")
                st.write("")
else:
    st.write('Stopped')
    camera.release()
