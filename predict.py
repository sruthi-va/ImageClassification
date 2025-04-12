import sys
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

# Parameters
IMG_SIZE = (128, 128)
MODEL_PATH = 'saved_model/best_model.h5'  # or 'final_model.h5'
CLASS_NAMES = ['cat', 'dog']  # Update this to match your classes
NUM_CLASSES = len(CLASS_NAMES)

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Make batch of 1
    return img_array

def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"Image path '{img_path}' does not exist.")
        return

    model = load_model(MODEL_PATH)
    img_array = load_and_preprocess_image(img_path)
    
    prediction = model.predict(img_array)
    
    if NUM_CLASSES > 1:
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
    else:
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]

    print(f"Predicted: {CLASS_NAMES[predicted_class]} ({confidence*100:.2f}% confidence)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
    else:
        predict_image(sys.argv[1])
