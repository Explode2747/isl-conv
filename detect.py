# live_alphabet_detection.py

import cv2
import numpy as np
import tensorflow as tf
import json

# Load the trained model
model = tf.keras.models.load_model('alphabet_classification_model.h5')

# Load the labels from the JSON file
with open('labels.json', 'r') as f:
    labels = json.load(f)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame from webcam
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (64, 64))  # Resize to match the model's input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    reshaped_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(reshaped_frame)
    predicted_label = labels[np.argmax(prediction)]

    # Display the prediction on the frame
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Alphabet Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
