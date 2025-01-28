# Malicious URL Detector

## Overview
This project is a **Malicious URL Detector** that classifies URLs into different categories based on various URL features. It employs two machine learning approaches:
1. **Gradient Boosting Classifier (Traditional Machine Learning Model)**
2. **Convolutional Neural Network (CNN)**

The project leverages both traditional feature-based techniques and deep learning models to ensure high accuracy and robustness. The models are fine-tuned for optimal performance.

---

## Features
- **Gradient Boosting Classifier**: Trained using engineered features such as URL length, domain tokens, and path tokens.
- **Convolutional Neural Network (CNN)**: Trained on character-level sequences of URL strings to capture context and structure.
- **Model Comparison**: Visualization of performance comparison between the two models.
- **Robust Preprocessing**: Handles missing data and encodes categorical labels efficiently.
- **Save and Reuse**: Models, tokenizers, and label encoders are saved for future use.

---

## Dataset
The project uses the **ISCX-URL2016 dataset**, which contains labeled URLs categorized into different types.  
Ensure that the dataset (`ISCX-URL2016_All.csv`) is placed in the same directory as the code.

---
## How It Works

### Data Preprocessing:
1. Load the dataset and select relevant columns.
2. Handle missing values and encode labels.

### Gradient Boosting Model:
1. Extract numerical features from the URLs.
2. Fine-tune the model using Grid Search.
3. Evaluate the model and display classification metrics.

### Convolutional Neural Network:
1. Tokenize the URL strings into character-level sequences.
2. Train the CNN using an embedding layer, convolutional layers, and dense layers.
3. Evaluate the model and display classification metrics.

### Comparison:
1. Visualize the accuracy of both models using bar charts.
