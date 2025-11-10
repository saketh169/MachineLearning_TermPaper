"""
================================================================================
LOGISTIC REGRESSION CLASSIFIER (From Scratch)
================================================================================
Parametric, Gradient-Based Binary Classifier
Uses gradient descent with L2 regularization to find optimal coefficients
Performance: 80-95% accuracy on various datasets
================================================================================
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import Tuple, Dict


# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch.
    Uses gradient descent with optional L2 regularization.
    """
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000, 
                 lambda_: float = 0.01, tolerance: float = 1e-5):
        """
        Args:
            learning_rate: Step size for gradient descent
            iterations: Maximum number of iterations
            lambda_: L2 regularization strength
            tolerance: Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_ = lambda_
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function: 1 / (1 + e^(-z))"""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Binary cross-entropy loss with L2 regularization"""
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        
        # Prevent log(0) by clipping
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        # Binary cross-entropy
        loss = -(1/m) * np.sum(y * np.log(predictions) + 
                               (1 - y) * np.log(1 - predictions))
        
        # Add L2 regularization
        regularization = (self.lambda_ / (2*m)) * np.sum(self.weights ** 2)
        
        return loss + regularization
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classifier using gradient descent"""
        m, n = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n)
        self.bias = 0
        
        print(f"\n{'='*70}")
        print(f"Training Logistic Regression")
        print(f"{'='*70}")
        print(f"Samples: {m}, Features: {n}")
        
        # Gradient descent
        for iteration in range(self.iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Compute gradients
            dw = (1/m) * np.dot(X.T, (predictions - y)) + (self.lambda_/m) * self.weights
            db = (1/m) * np.sum(predictions - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute and store loss
            loss = self.compute_cost(X, y)
            self.loss_history.append(loss)
            
            if iteration % 100 == 0:
                print(f"  Iteration {iteration:4d} | Loss: {loss:.6f}")
            
            # Early stopping
            if iteration > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tolerance:
                print(f"  Converged at iteration {iteration}")
                break
        
        print(f"Training complete. Final Loss: {self.loss_history[-1]:.6f}\n")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of class 1"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class (0 or 1)"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare dataset"""
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int32)
    
    # Normalize features (important for gradient descent)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8  # Avoid division by zero
    X = (X - X_mean) / X_std
    
    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, 
                     test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray]:
    """Split data into train and test sets"""
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    
    split_idx = int(len(X) * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate accuracy, precision, and F1-score"""
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """Main pipeline"""
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION CLASSIFIER - FROM SCRATCH")
    print("="*70)
    
    # Dataset paths
    datasets_dir = os.path.join(os.path.dirname(__file__), '..', 'Datasets')
    datasets = {
        'bank': 'bank_cleaned.csv',
        'heart': 'heart_cleaned.csv',
        'wine': 'wine_cleaned.csv',
        'breast': 'breast_cleaned.csv',
        'churn': 'churn_cleaned.csv'
    }
    
    results = {}
    
    for name, filename in datasets.items():
        filepath = os.path.join(datasets_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"\n✗ {name.upper()}: File not found - {filepath}")
            continue
        
        try:
            print(f"\n{'='*70}")
            print(f"Dataset: {name.upper()}")
            print(f"{'='*70}")
            
            # Load and split data
            X, y = load_dataset(filepath)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            print(f"Loaded: {X.shape}")
            print(f"Train: {X_train.shape} | Test: {X_test.shape}")
            print(f"Class distribution - Train: {np.unique(y_train, return_counts=True)}")
            
            # Train classifier
            lr = LogisticRegression(learning_rate=0.1, iterations=500, lambda_=0.001)
            lr.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = lr.predict(X_train)
            y_pred_test = lr.predict(X_test)
            
            # Evaluate
            train_metrics = calculate_metrics(y_train, y_pred_train)
            test_metrics = calculate_metrics(y_test, y_pred_test)
            
            print(f"\nTRAIN METRICS:")
            print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
            print(f"  Precision: {train_metrics['precision']:.4f}")
            print(f"  Recall:    {train_metrics['recall']:.4f}")
            print(f"  F1-Score:  {train_metrics['f1']:.4f}")
            
            print(f"\nTEST METRICS:")
            print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall:    {test_metrics['recall']:.4f}")
            print(f"  F1-Score:  {test_metrics['f1']:.4f}")
            
            results[name] = test_metrics
            
        except Exception as e:
            print(f"\n✗ {name.upper()}: Error - {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    summary_text = ""
    summary_text += f"\n{'='*70}\n"
    summary_text += "SUMMARY - TEST SET PERFORMANCE\n"
    summary_text += f"{'='*70}\n"
    for name, metrics in results.items():
        line = f"{name.upper():10} | Accuracy: {metrics['accuracy']:.4f} | " \
               f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | " \
               f"F1: {metrics['f1']:.4f}\n"
        summary_text += line
    summary_text += f"{'='*70}\n"
    
    print(summary_text)
    
    # Write to output.txt
    output_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'output.txt')
    
    with open(output_file, 'w') as f:
        f.write("LOGISTIC REGRESSION CLASSIFIER - RESULTS\n")
        f.write(f"{'='*70}\n\n")
        f.write(summary_text)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
