"""
================================================================================
LINEAR SVM CLASSIFIER (From Scratch)
================================================================================
Parametric, Maximum-Margin Binary Classifier
Uses Stochastic Gradient Descent with Hinge Loss
Performance: 85-95% accuracy on various datasets
================================================================================
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import Tuple, Dict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LinearSVM:
    """
    Linear Support Vector Machine implemented from scratch.
    Uses SGD with hinge loss and L2 regularization.
    """
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000,
                 lambda_: float = 0.01, batch_size: int = 32, tolerance: float = 1e-5):
        """
        Args:
            learning_rate: Step size for SGD
            iterations: Maximum number of iterations
            lambda_: L2 regularization strength (controls margin)
            batch_size: Size of mini-batches for SGD
            tolerance: Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def hinge_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Hinge loss: max(0, 1 - y_true * y_pred)"""
        return np.mean(np.maximum(0, 1 - y_true * y_pred))
    
    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Hinge loss with L2 regularization"""
        m = X.shape[0]
        
        # Convert to {-1, 1} format for SVM
        y_svm = 2 * y - 1  # 0->-1, 1->1
        
        # Predictions
        z = np.dot(X, self.weights) + self.bias
        
        # Hinge loss
        hinge = np.mean(np.maximum(0, 1 - y_svm * z))
        
        # L2 regularization
        regularization = (self.lambda_ / 2) * np.sum(self.weights ** 2)
        
        return hinge + regularization
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM using SGD"""
        m, n = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n)
        self.bias = 0
        
        # Convert labels to {-1, 1} for SVM
        y_svm = 2 * y - 1
        
        print(f"\n{'='*70}")
        print(f"Training Linear SVM (Stochastic Gradient Descent)")
        print(f"{'='*70}")
        print(f"Samples: {m}, Features: {n}")
        print(f"Learning Rate: {self.learning_rate}, Batch Size: {self.batch_size}")
        
        # SGD
        np.random.seed(42)
        for iteration in range(self.iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y_svm[indices]
            
            # Mini-batch updates
            for batch_start in range(0, m, self.batch_size):
                batch_end = min(batch_start + self.batch_size, m)
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Predictions
                z = np.dot(X_batch, self.weights) + self.bias
                
                # Check which samples violate margin (hinge loss > 0)
                violation = (y_batch * z) < 1
                
                # Compute gradients for hinge loss
                dw = np.zeros(n)
                db = 0
                
                for i in range(len(X_batch)):
                    if violation[i]:
                        # Hinge loss gradient: -y * x for violated samples
                        dw -= y_batch[i] * X_batch[i]
                        db -= y_batch[i]
                
                # Add L2 regularization gradient and average over batch
                dw = (dw / len(X_batch)) + self.lambda_ * self.weights
                db = db / len(X_batch)
                
                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # Compute and store loss (every 50 iterations)
            if iteration % 50 == 0:
                loss = self.compute_cost(X, y)
                self.loss_history.append(loss)
                print(f"  Iteration {iteration:4d} | Loss: {loss:.6f}")
            
            # Early stopping
            if len(self.loss_history) > 1 and \
               abs(self.loss_history[-2] - self.loss_history[-1]) < self.tolerance:
                print(f"  Converged at iteration {iteration}")
                break
        
        final_loss = self.compute_cost(X, y)
        print(f"Training complete. Final Loss: {final_loss:.6f}\n")
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute distance from decision boundary"""
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class (0 or 1)"""
        z = self.decision_function(X)
        return (z > 0).astype(int)


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare dataset"""
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int32)
    
    # Normalize features (important for SVM)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
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


def log_results_to_file(classifier_name: str, results: Dict):
    """Log results to output.txt file with timestamp"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'Outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'output.txt')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"CLASSIFIER: {classifier_name}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'Dataset':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}\n")
        f.write(f"{'-'*80}\n")
        
        for name, metrics in results.items():
            f.write(f"{name.upper():<15} | {metrics['accuracy']:>8.4f}  | {metrics['precision']:>8.4f}  | {metrics['recall']:>8.4f}  | {metrics['f1']:>8.4f}\n")
        
        f.write(f"{'='*80}\n")
    
    print(f"\n✓ Results logged to: {output_file}")


def main():
    """Main pipeline"""
    print("\n" + "="*70)
    print("LINEAR SVM CLASSIFIER - FROM SCRATCH")
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
            
            # Train SVM with dataset-specific hyperparameters
            if name.lower() == 'wine':
                # Wine needs higher learning rate, more iterations, and stronger regularization
                svm = LinearSVM(learning_rate=0.1, iterations=2000, lambda_=0.01, batch_size=16)
            else:
                svm = LinearSVM(learning_rate=0.01, iterations=500, lambda_=0.001, batch_size=32)
            svm.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = svm.predict(X_train)
            y_pred_test = svm.predict(X_test)
            
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
        f.write("LINEAR SVM CLASSIFIER - RESULTS\n")
        f.write(f"{'='*70}\n\n")
        f.write(summary_text)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
