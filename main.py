import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def calculate_gradient(theta, X, y):
    m = y.size # number of instances
    return (X.T @ (sigmoid(X @ theta) - y)) / m

def gradient_descent(X, y, alpha=0.1, num_iter=1, tol=1e-7):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])

    for i in range(num_iter):
        grad = calculate_gradient(theta, X_b, y)
        theta -= alpha * grad

        if np.linalg.norm(grad) < tol:
            break
    
    return theta

def predict_proba(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)

def predict(X, theta, threshold=0.5):
    return (predict_proba(X, theta) >= threshold).astype(int)


# Using sklearn for preprocessing of data & evaluating our Logistic Regression Algorithm
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_fscore_support

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

theta_hat = gradient_descent(X_train_scaled, y_train, alpha=0.1)

# Make Predictions
y_pred_train = predict(X_train_scaled, theta_hat)
y_pred_test= predict(X_test_scaled, theta_hat)
y_pred_proba_train = predict_proba(X_train_scaled, theta_hat)
y_pred_proba_test = predict_proba(X_test_scaled, theta_hat)

train_acc= accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"Training accuracy: {train_acc}")
print(f"Testing accuracy: {test_acc}")

# Create plots directory
os.makedirs('plots',exist_ok=True)

# Set seabrorn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# 1. Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

sns.heatmap(cm_train, annot=True, fmt='d', cmap="Blues", ax=axes[0], cbar=True)
axes[0].set_title('Training Confusion Matrix', fontsize=16)
axes[0].set_ylabel("True label")
axes[0].set_xlabel("Predicted label")

sns.heatmap(cm_test, annot=True, fmt='d', cmap="RdYlGn", ax=axes[1], cbar=True)
axes[1].set_title("Testing Confusion Matrix", fontsize=16)
axes[1].set_ylabel("True label")
axes[1].set_xlabel("Predicted label")

plt.tight_layout()
plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices.png")
plt.close()

# 2. ROC Curve
fig, ax = plt.subplots(figsize=(10, 7))

fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)

auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

ax.plot(fpr_train, tpr_train, linewidth=2.5, label=f"Train (AUC = {auc_train:.3f})", color='#2E86AB')
ax.plot(fpr_test, tpr_test, linewidth=2.5, label=f"Test (AUC = {auc_test:.3f})", color= '#A23B72')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label="Random Classifier")

ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel("True Positive Rate", fontsize=12, fontweight='bold')
ax.set_title("ROC Curve - Logistic Regression", fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curve.png")
plt.close()


