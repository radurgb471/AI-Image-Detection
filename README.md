# AI Image Detector - Forensic Analysis

This project is a digital forensics tool designed to identify whether an image is real or AI-generated. It analyzes the statistical properties of image gradients and uses a machine learning model to provide a classification.

## Overview

The detection process involves:
1. Converting the image to a luminance map.
2. Applying Sobel filters to calculate horizontal and vertical gradients.
3. Extracting statistical features, specifically the covariance matrix and the anisotropy score.
4. Classifying the data using a Random Forest model.

## Requirements

Ensure you have Python installed along with the following libraries:
- opencv-python
- numpy
- pandas
- scikit-learn
- matplotlib
- joblib

Install them using:
```bash
pip install opencv-python numpy pandas scikit-learn matplotlib joblib
