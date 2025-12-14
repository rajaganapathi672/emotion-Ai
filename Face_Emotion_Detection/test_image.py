import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import argparse

# --- CONFIGURATION ---
MODEL_PATH = 'models/emotion_model.h5'
HAARCASCADE_PATH = 'haarcascade/haarcascade_frontalface_default.xml'
IMG_SIZE = 48
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_single_image(image_path):
    # Check dependencies
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(HAARCASCADE_PATH):
        print(f"Error: Haar cascade not found at {HAARCASCADE_PATH}. Run app.py once to download it.")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load Model
    print("Loading model...")
    model = load_model(MODEL_PATH, compile=False)
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)

    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No faces detected in the image.")
        return

    print(f"Detected {len(faces)} faces.")

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Preprocess
        roi_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0) # Batch
        roi_gray = np.expand_dims(roi_gray, axis=-1) # Channel

        # Predict
        prediction = model.predict(roi_gray, verbose=0)
        max_index = int(np.argmax(prediction))
        predicted_emotion = EMOTIONS[max_index]
        confidence = prediction[0][max_index] * 100

        print(f"Face at ({x},{y}): {predicted_emotion} ({confidence:.2f}%)")

        # Draw
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        label = f"{predicted_emotion}: {confidence:.1f}%"
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show result
    cv2.imshow('Test Image Result', img)
    print("Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use argparse for command line flexibility
    parser = argparse.ArgumentParser(description="Test Emotion Detection on an Image")
    parser.add_argument("image_path", help="Path to the image file")
    
    # If run without arguments, you can manually set a default path for testing here or prompted
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <path_to_image>")
        # Prompt user if no arg provided (easier for beginners)
        path = input("Enter path to image: ").strip()
        if path:
             predict_single_image(path)
    else:
        args = parser.parse_args()
        predict_single_image(args.image_path)
