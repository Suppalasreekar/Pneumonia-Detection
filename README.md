
# Pneumonia Detection

This project involves building and training models for pneumonia detection using chest X-ray images. It employs a Convolutional Neural Network (CNN) for feature extraction and a Random Forest classifier for final prediction. The notebook includes steps for data preprocessing, feature extraction, model training, and evaluation, along with visualization of training results.

## Project Overview

This project demonstrates:
1. Loading and preprocessing an image dataset for pneumonia detection.
2. Using a CNN model for feature extraction from the images.
3. Training a Random Forest model on the extracted features for final classification.
4. Evaluating model accuracy on test data.
5. Plotting training history to analyze performance.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- Scikit-Learn
- Matplotlib
- Numpy

Install the dependencies using:
```bash
pip install tensorflow keras scikit-learn matplotlib numpy
```

## File Structure

- `cnn.ipynb`: Jupyter notebook containing the code for feature extraction using CNN, training the Random Forest model, and evaluating the combined model performance.

## Model Architecture

1. **Feature Extraction**: A Convolutional Neural Network (CNN) is used to extract relevant features from the chest X-ray images.
2. **Classification**: A Random Forest model is trained on the extracted features to classify images for pneumonia detection.

Refer to the notebook for detailed layer-by-layer specifications and parameters for both models.

## Results and Evaluation

The combined model (CNN + Random Forest) achieved an accuracy of **98%** on the test data.

## Contributing

Feel free to fork this repository and submit pull requests for improvements or new features.
