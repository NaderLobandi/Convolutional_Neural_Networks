# CIFAR-10 Image Classification

## Overview

This project aims to develop and compare machine learning models for image classification on the CIFAR-10 dataset. We focused on a Support Vector Machine (SVM) model, an Artificial Neural Network (ANN) using Keras, and a Convolutional Neural Network (CNN) with a pre-trained ResNet model. Special attention was given to hyperparameter tuning using GridSearchCV to optimize model performance.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is split into 50,000 training images and 10,000 test images.

## Methods

### ResNet Model

-   Utilized TensorFlow and Keras for model building.
-   Implemented residual blocks with convolutional layers, batch normalization, ReLU activation, and skip connections.
-   Applied global average pooling and an output layer for classification.
-   Used learning rate scheduling and early stopping to optimize performance.

### CNN Model

-   Built using Keras and TensorFlow.
-   Employed convolutional layers with pooling layers and a flattening layer.
-   Added fully connected layers with dropout for regularization.
-   Compiled with Adam optimizer and categorical crossentropy loss.

### SVM with GridSearchCV

-   Used scikit-learn and Keras for the SVM model.
-   Preprocessed images and flattened them for the SVM.
-   Applied GridSearchCV for hyperparameter optimization.

### ANN

-   Developed using the Keras Sequential API.
-   Normalized pixel values and flattened images.
-   Configured with Dense and Dropout layers.
-   Compiled and trained with categorical crossentropy loss and Adam optimizer.

### KNN

-   Implemented k-Nearest Neighbors classifier using scikit-learn.
-   Standardized features and defined hyperparameter grid.
-   Performed GridSearchCV for optimization.

## Results

-   ResNet showed the highest test accuracy (82.11%) and reasonable computational time.
-   CNN followed with a test accuracy of 67.34%.
-   SVM, ANN, and KNN models demonstrated varied performance with differing computational demands.

## Conclusion

The project demonstrates the effectiveness of CNNs, particularly ResNet, in image classification tasks, outperforming traditional machine learning models in accuracy and computational efficiency.

## Requirements

-   TensorFlow
-   Keras
-   scikit-learn
-   NumPy

## Usage

Instructions on how to set up, train, and evaluate the models are provided in each subdirectory.
