import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
MODEL_PATH = 'mango_cnn_model.h5'  # your model
IMAGE_SIZE = (224, 224)  # change if your model input size is different
CLASS_NAMES = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

FOLDER_PATH = 'test'  # folder with images

# --- LOAD MODEL ---
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- LOAD AND PREPROCESS IMAGE ---
def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# --- PREDICT ---
def predict(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]
    return predicted_class, confidence

# --- PROCESS FOLDER ---
def process_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the folder.")
        return

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img_array, img = load_and_preprocess(img_path)
        predicted_class, confidence = predict(img_array)
        
        # Display image with prediction
        plt.imshow(img)
        plt.title(f"{img_file}\nPrediction: {CLASS_NAMES[predicted_class]} ({confidence*100:.2f}%)")
        plt.axis('off')
        plt.show()

# --- MAIN ---
if __name__ == "__main__":
    if not os.path.isdir(FOLDER_PATH):
        print(f"Folder '{FOLDER_PATH}' does not exist.")
    else:
        process_folder(FOLDER_PATH)
