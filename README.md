# Logistic Regression from First Principles

A clean, educational implementation of logistic regression built from scratch using NumPy, demonstrating fundamental machine learning concepts without relying on high-level ML libraries for the core algorithm.

## Overview

![S Shaped Curve](assets/logisticRegression.png)

This project implements logistic regression from first principles, providing an excellent foundation for understanding how binary classification algorithms work under the hood. The implementation includes gradient descent optimization and is validated against real-world datasets.

## Features

- **Core Algorithm Implementation**: Sigmoid function, gradient calculation, and gradient descent optimizer
- **Probability Estimation**: Probabilistic predictions with adjustable decision threshold
- **Real-world Validation**: Evaluated on the breast cancer classification dataset
- **Data Preprocessing**: Standardized features for optimal performance
- **Educational Focus**: Well-commented code with clear mathematical concepts

## Algorithm Details

### Logistic Regression Model

The logistic regression model uses the sigmoid function to map predictions to probability space:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z = X\theta$ represents the linear combination of features and parameters.

### Gradient Descent Optimization

The model is trained using gradient descent with the following update rule:

$$\theta := \theta - \alpha \nabla J(\theta)$$

Where the gradient is computed as:

$$\nabla J(\theta) = \frac{1}{m} X^T(\sigma(X\theta) - y)$$

## Installation

### Prerequisites

- Python 3.7+

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd logsisticRegressionFromScratch
```

2. Create and activate a virtual environment:
```bash
python -m venv log_reg
log_reg\Scripts\activate
```

3. Install required dependencies:
```bash
pip install numpy matplotlib seaborn scikit-learn
```

## Usage

Run the main script to train the logistic regression model on the breast cancer dataset:

```bash
python main.py
```

### Expected Output

```
Training accuracy: 0.9540
Testing accuracy: 0.9298
✓ Saved: confusion_matrices.png
✓ Saved: roc_curve.png
✓ Saved: accuracy_comparison.png
✓ Saved: probability_distribution.png
✓ Saved: metrics_heatmap.png
```

### Code Example

```python
from main import gradient_descent, predict, predict_proba
import numpy as np

# Train the model
theta = gradient_descent(X_train, y_train, alpha=0.1, num_iter=1000)

# Make predictions
y_pred = predict(X_test, theta, threshold=0.5)

# Get probability estimates
y_proba = predict_proba(X_test, theta)
```

## Project Structure

```
logsisticRegressionFromScratch/
├── main.py                 # Core implementation and training script
├── README.md              # This file
├── assets/                # Project assets (optional)
└── log_reg/              # Python virtual environment
```

## Implementation Details

### Key Functions

- **`sigmoid(z)`**: Applies the sigmoid activation function
- **`calculate_gradient(theta, X, y)`**: Computes gradients for the logistic loss function
- **`gradient_descent(X, y, alpha, num_iter, tol)`**: Trains the model using gradient descent
  - `alpha`: Learning rate (default: 0.1)
  - `num_iter`: Maximum iterations (default: 1)
  - `tol`: Convergence tolerance (default: 1e-7)
- **`predict_proba(X, theta)`**: Returns probability predictions
- **`predict(X, theta, threshold)`**: Returns binary class predictions (default threshold: 0.5)

## Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** from scikit-learn containing:
- **Samples**: 569 instances
- **Features**: 30 features derived from digitized images
- **Task**: Binary classification (malignant/benign)
- **Train/Test Split**: 80/20

## Results

The model achieves strong performance on the breast cancer dataset:
- **Training Accuracy**: ~95%
- **Testing Accuracy**: ~93%

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | Latest | Numerical computations |
| Matplotlib | Latest | Visualization |
| Seaborn | Latest | Statistical visualization |
| scikit-learn | Latest | Data preprocessing and evaluation |

## Learning Outcomes

By studying this implementation, you'll understand:

1. The mathematical foundations of logistic regression
2. How gradient descent optimization works
3. The role of the sigmoid function in binary classification
4. Feature scaling and its importance
5. Probability-based vs. hard classification
6. Model evaluation metrics (accuracy)

## Educational Resources

- Understand how ML algorithms are implemented at a low level
- Modify hyperparameters to see their impact on model performance
- Extend the implementation with additional features (regularization, cross-validation, etc.)

## Future Enhancements

- [ ] L1/L2 regularization implementation
- [ ] K-fold cross-validation
- [ ] Additional evaluation metrics (precision, recall, F1-score, ROC-AUC)
- [ ] Visualization of decision boundaries
- [ ] Support for multi-class classification
- [ ] Stochastic gradient descent variant

## License

This project is provided as-is for educational purposes.

## Author

Created as an educational machine learning project demonstrating implementation of logistic regression from first principles.

---

**Note**: This implementation is designed for educational purposes. For production use, consider using optimized libraries like scikit-learn, XGBoost, or TensorFlow.
