import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64
EPOCHS = 2  # Reduced to 2 for quick testing
NUM_CLASSES = 7
LEARNING_RATE = 0.0001
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
MODEL_PATH = 'models/emotion_model.h5'

# Define emotions (must match folder names)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def build_model():
    """
    Builds a CNN model for Face Emotion Detection.
    """
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        
        # Block 1
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Fully Connected Layers
        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=LEARNING_RATE),
                  metrics=['accuracy'])
    
    return model

def train():
    """
    Trains the CNN model using data from dataset/train and dataset/test.
    """
    # Check if dataset directories exist
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print(f"ERROR: Dataset not found. Please create '{TRAIN_DIR}' and '{TEST_DIR}' with emotion subfolders.")
        return

    # Data Augmentation for Training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Rescaling for Validation (No augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    print("Loading Validation Data...")
    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Build and Train Model
    model = build_model()
    model.summary()

    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # Save Model
    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Model saved successfully!")

    # Plot Training Results
    plot_results(history)

def plot_results(history):
    """
    Plots training accuracy and loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU Detected: {gpus}")
    else:
        print("No GPU detected, running on CPU.")
        
    train()
