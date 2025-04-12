import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
MODEL_PATH = 'saved_model/best_model.h5'
TEST_DIR = 'data/test'
NUM_CLASSES = 2  # Adjust for your case

# Load model
model = load_model(MODEL_PATH)

# Prepare test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical' if NUM_CLASSES > 1 else 'binary',
    shuffle=False
)

# Predictions
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1) if NUM_CLASSES > 1 else (y_pred > 0.5).astype(int).reshape(-1)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
