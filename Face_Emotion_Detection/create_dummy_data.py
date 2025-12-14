import cv2
import numpy as np
import os

# Define output directories
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 48
NUM_IMAGES_PER_CLASS = 10  # Very small number just to make code run

def create_dummy_images(base_dir):
    for emotion in EMOTIONS:
        dir_path = os.path.join(base_dir, emotion)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        print(f"Generating images for {base_dir}/{emotion}...")
        for i in range(NUM_IMAGES_PER_CLASS):
            # Create a random noise image
            img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            
            # Save it
            filename = f"{emotion}_{i}.jpg"
            cv2.imwrite(os.path.join(dir_path, filename), img)

def main():
    print("Generating dummy dataset...")
    create_dummy_images(TRAIN_DIR)
    create_dummy_images(TEST_DIR)
    print("Dummy dataset created successfully!")
    print("You can now run 'python train_model.py' to test the pipeline.")
    print("WARNING: The model will not learn actual emotions from this noise.")

if __name__ == "__main__":
    main()
