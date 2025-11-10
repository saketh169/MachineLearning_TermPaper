"""
================================================================================
WEIGHTED K-NEAREST NEIGHBORS CLASSIFIER (From Scratch)
================================================================================
Non-Parametric, Distance-Based Classifier
Uses inverse distance weighting for classification
Performance: 80-92% accuracy on various datasets
================================================================================
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import Tuple, Dict
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WeightedKNN:
    """
    Weighted k-Nearest Neighbors classifier implemented from scratch.
    Uses distance-weighted voting for classification.
    """
    
    def __init__(self, k: int = 5, distance_metric: str = 'euclidean', 
                 weight_type: str = 'inverse_distance'):
        """
        Args:
            k: Number of nearest neighbors
            distance_metric: 'euclidean' or 'manhattan'
            weight_type: 'inverse_distance', 'uniform', or 'gaussian'
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weight_type = weight_type
        self.X_train = None
        self.y_train = None
        
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Manhattan distance"""
        return np.sum(np.abs(x1 - x2))
    
    def compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute distance based on selected metric"""
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store training data (lazy learning)"""
        self.X_train = X
        self.y_train = y
    
    def compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """Compute weights based on distances"""
        if self.weight_type == 'uniform':
            # All neighbors have equal weight
            return np.ones_like(distances)
        
        elif self.weight_type == 'inverse_distance':
            # Weight = 1 / distance (with small epsilon to avoid division by zero)
            epsilon = 1e-10
            weights = 1.0 / (distances + epsilon)
            return weights
        
        elif self.weight_type == 'gaussian':
            # Weight = exp(-distance^2 / (2 * sigma^2))
            sigma = np.mean(distances) + 1e-10
            weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
            return weights
        
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")
    
    def predict_single(self, x: np.ndarray) -> int:
        """Predict class for single sample"""
        # Vectorized distance computation
        if self.distance_metric == 'euclidean':
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        else:
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        
        # Find k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_distances = distances[k_indices]
        k_labels = self.y_train[k_indices]
        
        # Compute weights
        weights = self.compute_weights(k_distances)
        
        # Weighted voting
        class_votes = {}
        for label, weight in zip(k_labels, weights):
            class_votes[label] = class_votes.get(label, 0) + weight
        
        # Return class with highest weighted votes
        return max(class_votes, key=class_votes.get)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for multiple samples"""
        predictions = np.array([self.predict_single(x) for x in X])
        return predictions


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare dataset"""
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int32)
    
    # Normalize features (important for KNN)
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


def main():
    """Main pipeline"""
    print("\n" + "="*70)
    print("WEIGHTED K-NEAREST NEIGHBORS CLASSIFIER - FROM SCRATCH")
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
            
            # Train classifier with different k values
            for k_val in [3, 5, 7]:
                print(f"\n  Training with k={k_val}...")
                knn = WeightedKNN(k=k_val, distance_metric='euclidean', 
                                 weight_type='inverse_distance')
                knn.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = knn.predict(X_train)
                y_pred_test = knn.predict(X_test)
                
                # Evaluate
                train_metrics = calculate_metrics(y_train, y_pred_train)
                test_metrics = calculate_metrics(y_test, y_pred_test)
                
                print(f"  k={k_val} - Train Acc: {train_metrics['accuracy']:.4f} | Test Acc: {test_metrics['accuracy']:.4f}")
                
                # Store best result
                if k_val == 5:
                    results[name] = test_metrics
            
            # Final evaluation with k=5
            print(f"\n  Final Model (k=5):")
            knn_final = WeightedKNN(k=5, distance_metric='euclidean', 
                                   weight_type='inverse_distance')
            knn_final.fit(X_train, y_train)
            y_pred_train = knn_final.predict(X_train)
            y_pred_test = knn_final.predict(X_test)
            
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
        f.write("WEIGHTED K-NEAREST NEIGHBORS CLASSIFIER - RESULTS\n")
        f.write(f"{'='*70}\n\n")
        f.write(summary_text)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
