# AI Image Detector - Forensic Analysis (Version 2)

This project is an advanced digital forensics tool designed to distinguish between real images and AI-generated content. Version 2 improves upon previous methods by combining spatial statistical analysis (gradients) with frequency domain analysis using Fast Fourier Transform (FFT).

## Overview

The detection system extracts five specific features to classify images:
1. Spatial Gradients (var_x, var_y, cov_xy): Captured using Sobel filters to measure texture consistency.
2. Anisotropy: Analyzes the directionality of image structures.
3. FFT Score: Identifies artificial patterns and periodic artifacts in the frequency domain that are typical of AI generation processes.

## Requirements

The project requires Python and the following libraries:
- opencv-python
- numpy
- pandas
- scikit-learn
- matplotlib
- joblib

Install the dependencies using:
```bash
pip install opencv-python numpy pandas scikit-learn matplotlib joblib
