# Face Emotion Detection System

A real-time Face Emotion Detection application using Python, OpenCV, and TensorFlow/Keras. This project uses a Convolutional Neural Network (CNN) to classify facial expressions into 7 categories: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

## Features
- **Real-time Detection**: Uses webcam to detect faces and predict emotions instantly.
- **Deep Learning**: Custom CNN architecture trained from scratch.
- **Face Detection**: Utilizes OpenCV's Haar Cascades for efficient face localization.
- **Robustness**: Handles multiple faces and various lighting conditions (to an extent).
- **CPU/GPU Support**: Automatically detects and uses GPU if available.

## Project Structure
```
face emotion ai/
│
├── Face_Emotion_Detection/    # Main Source Code
│   ├── app.py                 # Main application script (Webcam)
│   ├── train_model.py         # Script to train the CNN model
│   ├── test_image.py          # Script to test on static images
│   ├── requirements.txt       # List of dependencies
│   └── models/                # Trained model directory
│
├── haarcascade/               # Face detection XML
│   └── haarcascade_frontalface_default.xml
│
└── README.md                  # Project documentation
```

## Prerequisites
- Python 3.7+
- A webcam (for real-time detection)

## Installation

1.  **Clone the project**:
    ```bash
    git clone https://github.com/rajaganapathi672/emotion-Ai.git
    cd "face emotion ai"
    ```

2.  **Navigate to the source folder**:
    ```bash
    cd Face_Emotion_Detection
    ```

3.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### Step 1: Prepare the Dataset
**Critical Step**: You need a dataset to train the model. The FER-2013 dataset is a popular choice.
1.  Download a dataset (e.g., from Kaggle).
2.  Organize the images into `dataset/train` and `dataset/test` folders inside `Face_Emotion_Detection`.

### Step 2: Train the Model
Once the dataset is ready, run the training script:
```bash
python train_model.py
```
- This will train the CNN for 50 epochs (configurable).
- The best model will be saved to `models/emotion_model.h5`.

### Step 3: Run Real-Time Detection
Start the webcam application:
```bash
python app.py
```
*(Or `app_web.py` if using the web interface)*

### Step 4: Test on a Single Image
To test a specific image file:
```bash
python test_image.py path/to/your/image.jpg
```

## Implementation Details
- **Architecture**: The CNN consists of 4 convolutional blocks with BatchNormalization, MaxPooling, and Dropout regularization.
- **Input**: Images are converted to grayscale and resized to 48x48 pixels.

## Disclaimer
This project is for educational purposes.

---
**Date**: December 2025
