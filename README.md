# 🧠 Image Classification with TensorFlow & Keras

This project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images into categories (e.g., cats vs. dogs). It includes data preprocessing, training, evaluation, and prediction scripts, and works with custom image datasets.

---

## Project Structure

```
image-classification-project/
├── data/
│   ├── train/                # training images organized by class subfolders
│   └── test/                 # testing/validation images organized similarly
├── saved_model/             # saved model files (.h5)
├── model.py                 # CNN model definition
├── train.py                 # training script
├── predict.py               # run predictions on new images
├── evaluate.py              # generate classification report & confusion matrix
├── utils.py                 # helper functions for visualization and preprocessing
├── requirements.txt         # Python package dependencies
└── README.md
```

---

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Dataset Format

This project expects image folders structured like this:

```
data/
├── train/
│   ├── class_a/
│   └── class_b/
└── test/
    ├── class_a/
    └── class_b/
```

Replace `class_a`, `class_b` with your actual class names (e.g., `cats`, `dogs`).

---

## Training the Model

To train the CNN model:

```bash
python train.py
```

This will:
- Load images from `data/train/` and `data/test/`
- Train a model for 10 epochs
- Save the best model to `saved_model/best_model.h5`
- Save the final model to `saved_model/final_model.h5`

---

## Evaluating the Model

To evaluate the trained model:

```bash
python evaluate.py
```

This will:
- Load the best model from `saved_model/`
- Run it on the test set
- Print a classification report
- Display a confusion matrix

---

## Making Predictions

To run predictions on new images:

```bash
python predict.py path/to/image.jpg
```

Make sure the image is preprocessed similarly (128x128 by default).

---

## Visualizing Training

In `train.py`, after training, you can optionally call this in an interactive environment to view accuracy/loss:

```python
from utils import plot_training_history
plot_training_history(history)
```

---

## Modify for Your Use Case

- Adjust `NUM_CLASSES` in `train.py` and `evaluate.py` for your dataset.
- You can increase `EPOCHS`, tweak model layers in `model.py`, or modify augmentation in `train.py`.

---

## References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras ImageDataGenerator](https://keras.io/api/preprocessing/image/)
- [scikit-learn Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
