import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_model

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2  # Change if you have more classes
TRAIN_DIR = 'data/train'
VALID_DIR = 'data/test'  # Or use validation_split from training data

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical' if NUM_CLASSES > 1 else 'binary'
)

valid_gen = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical' if NUM_CLASSES > 1 else 'binary'
)

# Build and train the model
model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES)

checkpoint = ModelCheckpoint(
    'saved_model/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# Save final model
model.save('saved_model/final_model.h5')
