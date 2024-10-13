# Lung Colon Image Classification Using InceptionV3

A deep learning project that uses transfer learning with **InceptionV3** to classify lung and colon cancer histopathological images into three categories. This project is built using **TensorFlow** and **Keras**.

## Overview

This project focuses on using a pre-trained **InceptionV3** model for image classification to identify lung and colon cancer types from medical histopathological images. The model is fine-tuned for this specific task and achieves high accuracy through transfer learning.

---

## Dataset

The **Lung Colon Image Dataset** consists of three categories:
- **Lung Adenocarcinoma**
- **Lung Squamous Cell Carcinoma**
- **Colon Adenocarcinoma**


---

### Prerequisites

Make sure you have the following installed:
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy, Pandas, Matplotlib
  
## Model Architecture

We use **InceptionV3**, a pre-trained model on the **ImageNet** dataset, and fine-tune it for our three-class image classification task. The model architecture includes:

- **Input Layer**: Resized images to 256x256 pixels.
- **Base Model**: InceptionV3 without the top layers (pre-trained on ImageNet).
- **Flattening Layer**: To flatten the output from the base model.
- **Fully Connected Layers**:
  - Dense layer with 256 units, **ReLU** activation, and **Batch Normalization**.
  - Dense layer with 128 units, **ReLU** activation, and **Dropout** (0.3).
- **Output Layer**: Softmax layer for multi-class classification (three categories).

### Model Summary:

- **Input shape**: (256, 256, 3)
- **Optimizer**: Adam
- **Loss function**: Categorical Crossentropy
- **Metrics**: Accuracy

---

## Training and Evaluation

The model is trained using the following configuration:

- **Image Size**: 256x256 pixels
- **Training/Validation Split**: 80% training, 20% validation
- **Batch Size**: 64
- **Epochs**: 10 (with early stopping)
- **Callbacks**: 
  - Early stopping 
  - ReduceLROnPlateau 
  - Custom callback to stop training when validation accuracy exceeds 90%

### Training Visualization:
- Training is visualized using **accuracy** and **loss plots**.

### Validation Performance:
- Validation performance is evaluated using:
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-Score)

---

## Results

After training, the model achieved a validation accuracy of over 90%. Below are the key results:

- **Confusion Matrix**: Displays the number of correct and incorrect predictions.
- **Classification Report**: Shows **precision**, **recall**, and **F1-score** for each category (Lung Adenocarcinoma, Lung Squamous Cell Carcinoma, Colon Adenocarcinoma).


