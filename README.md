# Plant Detection and Classification using YOLO and MobileNetV2

This project is a **Plant Detection and Classification System** that identifies different plant species from images or real-time video streams. The system is built using YOLO-style datasets for pre-processing and leverages a **transfer learning approach** with the **MobileNetV2** model for classification. Additionally, it integrates a **real-time detection application** using OpenCV for live predictions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Directory Structure](#directory-structure)
5. [Installation](#installation)
6. [Usage](#usage)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Training](#model-training)
    - [Real-Time Prediction](#real-time-prediction)
7. [Results](#results)
8. [Limitations](#limitations)
9. [Future Work](#future-work)
10. [Acknowledgements](#acknowledgements)

---

## Project Overview

This system is designed to detect and classify plant species based on images. It uses:

- **YOLO-style datasets** for structured input data preparation.
- **MobileNetV2**, a lightweight deep learning model, for transfer learning and fine-tuning.
- **OpenCV**, for real-time video stream analysis and classification.

The project comprises:
1. Dataset preparation using YOLO annotations.
2. A training pipeline for building a classifier.
3. A real-time application for live predictions using webcam input.

---

## Features

1. **Data Preprocessing**: Converts YOLO-style annotations into a class-based directory structure.
2. **Transfer Learning**: Leverages a pre-trained MobileNetV2 model for efficient and accurate classification.
3. **Real-Time Detection**: Uses OpenCV to capture and predict live camera feed.
4. **Visualization**: Displays accuracy and loss trends during training and validation.

---

## Prerequisites

### Tools and Libraries

- Python 3.7+
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- Roboflow dataset (or equivalent YOLO-style dataset)

---

## Directory Structure

The project directory is organized as follows:

```
PlantDetection/
│
├── script.py                  # Real-time prediction script
├── dataset_preprocessing.py   # YOLO dataset preprocessing script
├── train_model.py             # Model training script
├── plant_classifier_transfer_model.h5  # Trained model (after running train_model.py)
├── README.md                  # Project documentation
│
├── datasets/                  # Root folder for datasets
│   ├── otrain/                # Training data
│   ├── oval/                  # Validation data
│   └── otest/                 # Test data
│
└── results/                   # Training results and plots
```

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/saadhussain01306/Plant-Detection_YoLo.git
   cd Plant-Detection_YoLo
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow opencv-python matplotlib
   ```

3. Ensure you have the YOLO-style dataset ready or download it from [Roboflow](https://roboflow.com/).

---

## Usage

### Data Preprocessing

1. Place your YOLO dataset (with `images` and `labels` folders) in the `datasets` folder.
2. Run the dataset preprocessing script:
   ```python
   python dataset_preprocessing.py
   ```
   This script organizes the dataset into a class-based folder structure for training.

---

### Model Training

1. Update the paths for `train_dir`, `val_dir`, and `test_dir` in `train_model.py`.
2. Run the training script:
   ```python
   python train_model.py
   ```
3. Training consists of:
   - Feature extraction with frozen MobileNetV2 layers.
   - Fine-tuning for additional performance improvements.

The trained model will be saved as `plant_classifier_transfer_model.h5`.

---

### Real-Time Prediction

1. Run the real-time prediction script:
   ```python
   python script.py
   ```
2. Instructions:
   - **Press 'Enter'**: Capture a frame and predict the plant class.
   - **Press 'q'**: Quit the application.

---

## Results

### Model Accuracy
- Initial Training Accuracy: ~85%
- Fine-Tuned Accuracy: ~92%

### Sample Predictions
For a test dataset containing 10 classes (`anthurium`, `clivia`, etc.), the model predicts:
- True Class: `orchid`, Predicted: `orchid` (Confidence: 96.5%)
- True Class: `violet`, Predicted: `violet` (Confidence: 91.8%)

### Training Visualizations
- Accuracy and loss trends are plotted during training.

---

## Limitations

1. Limited to 10 predefined classes.
2. Requires a well-labeled YOLO-style dataset.
3. Model performance may degrade on low-quality or unseen data.

---

## Future Work

1. Extend the dataset for more plant species.
2. Implement a mobile-friendly application for on-the-go plant classification.
3. Add bounding box detection for object localization.

---
