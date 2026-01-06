# Digit Recognizer

A machine learning project for recognizing handwritten digits using deep learning.

## Project Overview

This project implements a digit recognition model trained on handwritten digit data. The model uses PyTorch to classify images of digits (0-9) with high accuracy.

## Files

- **main.ipynb** - Jupyter notebook containing the model training pipeline, data exploration, and evaluation
- **digit_recognizer_model.pth** - Pre-trained PyTorch model weights
- **train.csv** - Training dataset with pixel values and digit labels
- **test.csv** - Test dataset for making predictions
- **sample_submission.csv** - Sample submission file format for predictions

## Model Architecture

The model is a convolutional neural network (CNN) built with PyTorch, designed to:
- Accept 28x28 pixel grayscale images as input
- Extract features through convolutional layers
- Classify digits 0-9 with high accuracy

## Getting Started

### Requirements

- Python 3.7+
- PyTorch
- Pandas
- NumPy
- Jupyter Notebook
- scikit-learn (optional, for additional evaluation metrics)

### Installation

```bash
pip install torch pandas numpy jupyter scikit-learn
```

### Usage

1. Open the notebook in Jupyter:
```bash
jupyter notebook main.ipynb
```

2. Run the cells to:
   - Load and explore the training data
   - Train the model (or load pre-trained weights)
   - Make predictions on the test set
   - Generate submission file

## Dataset Format

**Training Data (train.csv):**
- First column: digit label (0-9)
- Remaining columns: pixel values (0-255) for a 28x28 image flattened to 784 values

**Test Data (test.csv):**
- 784 pixel values per row (no labels)

**Submission Format (sample_submission.csv):**
- ImageId: Test image index
- Label: Predicted digit (0-9)

## Results

The model achieves competitive accuracy on the digit recognition task. Detailed performance metrics and confusion matrices can be found in the main.ipynb notebook.

## License

This project is provided as-is for educational and research purposes.
