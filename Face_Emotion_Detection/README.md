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
Face_Emotion_Detection/
│
├── app.py                     # Main application script (Webcam)
├── train_model.py             # Script to train the CNN model
├── test_image.py              # Script to test on static images
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation
│
├── models/
│   └── emotion_model.h5       # Trained model (generated after training)
│
├── dataset/                   # Dataset directory (You must populate this)
│   ├── train/                 # Training images
│   │   ├── angry/
│   │   ├── ...
│   └── test/                  # Validation/Test images
│       ├── angry/
│       ├── ...
│
└── haarcascade/               # Face detection XML (auto-downloaded)
```

## Prerequisities
- Python 3.7+
- A webcam (for real-time detection)

## Installation

1.  **Clone/Download** the project.
2.  **Navigate** to the project folder:
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
2.  Organize the images into the `dataset/train` and `dataset/test` folders.
3.  Ensure each emotion has its own subfolder (angry, disgust, fear, happy, neutral, sad, surprise) containing the images.

### Step 2: Train the Model
Once the dataset is ready, run the training script:
```bash
python train_model.py
```
- This will train the CNN for 50 epochs (configurable).
- The best model will be saved to `models/emotion_model.h5`.
- **Note**: This may take time depending on your hardware (GPU recommended).

### Step 3: Run Real-Time Detection
Start the webcam application:
```bash
python app.py
```
- A window will open showing the webcam feed with bounding boxes and emotion labels.
- Press **'q'** to quit the application.

### Step 4: Test on a Single Image
To test a specific image file:
```bash
python test_image.py path/to/your/image.jpg
```

## Implementation Details
- **Architecture**: The CNN consists of 4 convolutional blocks with BatchNormalization, MaxPooling, and Dropout regularization to prevent overfitting.
- **Input**: Images are converted to grayscale and resized to 48x48 pixels.
- **Data Augmentation**: `train_model.py` uses random rotations, shifts, and flips to improve model generalization.

## Disclaimer
This project is for educational and academic purposes. The accuracy of the model depends heavily on the quality and size of the dataset used for training.

---
**Author**: [Your Name/ID]
**Date**: December 2025
