import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import urllib.request

# --- CONFIGURATION ---
MODEL_PATH = 'models/emotion_model.h5'
HAARCASCADE_PATH = 'haarcascade/haarcascade_frontalface_default.xml'
HAARCASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
IMG_SIZE = 48
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def download_haarcascade():
    """
    Downloads the Haar Cascade XML file if it doesn't exist.
    """
    if not os.path.exists(HAARCASCADE_PATH):
        print("Haar Cascade not found. Downloading...")
        os.makedirs(os.path.dirname(HAARCASCADE_PATH), exist_ok=True)
        try:
            urllib.request.urlretrieve(HAARCASCADE_URL, HAARCASCADE_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading Haar Cascade: {e}")
            print("Please manually download 'haarcascade_frontalface_default.xml' and place it in the 'haarcascade' folder.")
            exit()

def load_emotion_model():
    """
    Loads the trained Keras model.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found.")
        print("Please run 'python train_model.py' first to train the model.")
        return None
    
    try:
        model = load_model(MODEL_PATH, compile=False) # compile=False avoids optimizer warning if just for inference
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main():
    # 1. Setup Resources
    download_haarcascade()
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    
    model = load_emotion_model()
    if model is None:
        return

    # 2. Start Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting video stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # 3. Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract ROI (Region of Interest)
            roi_gray = gray[y:y+h, x:x+w]
            try:
                # Resize to 48x48
                roi_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
                
                # Normalize and Shape
                roi_gray = roi_gray.astype('float32') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)  # Add batch dimension
                roi_gray = np.expand_dims(roi_gray, axis=-1) # Add channel dimension -> (1, 48, 48, 1)

                # 4. Predict
                prediction = model.predict(roi_gray, verbose=0)
                max_index = int(np.argmax(prediction))
                predicted_emotion = EMOTIONS[max_index]
                confidence = prediction[0][max_index] * 100

                # 5. Display
                label = f"{predicted_emotion} ({confidence:.1f}%)"
                
                # Draw Box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw Label Background (for readability)
                cv2.rectangle(frame, (x, y-25), (x+w, y), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            except Exception as e:
                print(f"Prediction Error: {e}")

        # Show Output
        cv2.imshow('Face Emotion Detection', frame)

        # Build in quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
