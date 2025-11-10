"""
================================================================================
CLASSIFIER VISUALIZATION WITH DECISION BOUNDARIES
================================================================================
Interactive visualization tool for 5 binary classification datasets
Plots feature space with decision boundaries from all 4 classifiers
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
from typing import Tuple, List, Dict

# Add classifiers directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ================================================================================
# IMPORT CLASSIFIERS
# ================================================================================

import importlib.util

def load_classifier_class(filename, classname):
    """Dynamically load classifier class from file"""
    filepath = os.path.join(os.path.dirname(__file__), 'Classifiers', filename)
    spec = importlib.util.spec_from_file_location("classifier_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, classname)

LogisticRegression = load_classifier_class('30_LogisticRegression.py', 'LogisticRegression')
LinearSVM = load_classifier_class('30_LinearSVM.py', 'LinearSVM')
WeightedKNN = load_classifier_class('30_WeightedKNN.py', 'WeightedKNN')
DecisionTree = load_classifier_class('30_DecisionTree.py', 'DecisionTree')


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Load dataset and return features, labels, column names, and target column name"""
    df = pd.read_csv(filepath)
    
    # Get column names
    columns = df.columns.tolist()
    
    # Use LAST column as target (all datasets should have label in last column)
    target_col = columns[-1]
    
    # Separate features and target
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int32)
    
    # Get feature names (all except last column)
    feature_names = columns[:-1]
    
    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    return X, y, feature_names, target_col


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


def plot_confusion_matrices(fig, gs, all_results: Dict[str, np.ndarray], dataset_name: str):
    """Plot confusion matrices for all classifiers in a 2x2 grid in columns 1-2"""
    
    classifiers = list(all_results.keys())
    
    for idx, clf_name in enumerate(classifiers):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        cm = all_results[clf_name]
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0)
        ax.set_title(f'{clf_name}', fontsize=10, fontweight='bold')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=9, fontweight='bold')
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels(['True 0', 'True 1'])
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('Actual', fontsize=8)


def plot_prediction_pie_charts(fig, gs, all_results: Dict[str, np.ndarray], dataset_name: str):
    """Plot pie charts showing prediction distribution for each classifier in columns 3-4"""
    
    classifiers = list(all_results.keys())
    
    for idx, clf_name in enumerate(classifiers):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col + 2])  # Offset by 2 columns
        
        cm = all_results[clf_name]
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        
        sizes = [TP, TN, FP, FN]
        labels = ['TP', 'TN', 'FP', 'FN']
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
        
        # Only show non-zero values
        non_zero_sizes = []
        non_zero_labels = []
        non_zero_colors = []
        
        for size, label, color in zip(sizes, labels, colors):
            if size > 0:
                non_zero_sizes.append(size)
                non_zero_labels.append(f'{label}\n({size})')
                non_zero_colors.append(color)
        
        if non_zero_sizes:
            wedges, texts, autotexts = ax.pie(non_zero_sizes, labels=non_zero_labels, 
                                             colors=non_zero_colors, autopct='%1.1f%%',
                                             startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
            
            for autotext in autotexts:
                autotext.set_fontsize(7)
                autotext.set_fontweight('bold')
            
            for text in texts:
                text.set_fontsize(6)
        
        ax.set_title(f'{clf_name}', fontsize=10, fontweight='bold')


def plot_individual_metrics_bars(fig, gs, all_results: Dict[str, Dict[str, float]], dataset_name: str):
    """Plot individual metrics bar charts for each classifier in columns 5-6"""
    
    classifiers = list(all_results.keys())
    
    for idx, clf_name in enumerate(classifiers):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col + 4])  # Offset by 4 columns
        
        metrics = all_results[clf_name]
        metric_names = ['Acc', 'Prec', 'Rec', 'F1']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1']]
        
        bars = ax.bar(metric_names, metric_values, 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                     alpha=0.7, edgecolor='black', linewidth=1, width=0.5)
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.1%}', ha='center', va='bottom', 
                   fontsize=6, fontweight='bold')
        
        ax.set_title(f'{clf_name}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=8, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45, pad=3)


def visualize_dataset(dataset_name: str, filepath: str):
    """Visualize dataset with confusion matrices, pie charts, and bar charts for each classifier"""
    
    print(f"\n{'='*70}")
    print(f"Loading Dataset: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    X, y, feature_names, target_col = load_dataset(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {len(feature_names)} features")
    print(f"Target: {target_col}")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Visualization: Confusion Matrices, Prediction Distributions, Individual Metrics")
    
    # Train classifiers and collect results
    classifiers = [
        ('Logistic Regression', LogisticRegression(learning_rate=0.1, iterations=500, lambda_=0.001)),
        ('Linear SVM', LinearSVM(learning_rate=0.01, iterations=500, lambda_=0.001, batch_size=32)),
        ('Weighted k-NN', WeightedKNN(k=5, weight_type='inverse_distance')),
        ('Decision Tree', DecisionTree(max_depth=8, min_samples_split=5, min_samples_leaf=2)),
    ]
    
    all_results = {}
    confusion_matrices = {}
    
    for idx, (clf_name, clf) in enumerate(classifiers):
        print(f"\n[{idx+1}/4] Training {clf_name}...", end=' ')
        
        try:
            # Train classifier (suppress output)
            import io
            from contextlib import redirect_stdout
            
            with redirect_stdout(io.StringIO()):
                clf.fit(X_train, y_train)
            
            # Get predictions on test set
            y_pred_test = clf.predict(X_test)
            
            # Calculate confusion matrix
            cm = np.zeros((2, 2), dtype=int)
            for true, pred in zip(y_test, y_pred_test):
                cm[int(true), int(pred)] += 1
            
            confusion_matrices[clf_name] = cm
            
            # Calculate metrics
            TP = cm[1, 1]
            TN = cm[0, 0]
            FP = cm[0, 1]
            FN = cm[1, 0]
            
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            all_results[clf_name] = metrics
            
            print(f"✓ Test Accuracy: {accuracy:.2%}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            # Add default metrics and confusion matrix for failed classifier
            all_results[clf_name] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
            confusion_matrices[clf_name] = np.zeros((2, 2), dtype=int)
    
    # Create visualization with 2 rows x 6 columns layout
    fig = plt.figure(figsize=(24, 10))
    fig.patch.set_facecolor('white')
    
    # Add main title
    fig.suptitle(f'{dataset_name.upper()} Dataset - Classifier Analysis',
                fontsize=18, fontweight='bold', y=0.98)
    
    # Create subplots layout: 2 rows, 6 columns
    gs = fig.add_gridspec(2, 6, hspace=0.4, wspace=0.3, 
                         top=0.88, bottom=0.08, left=0.05, right=0.95)
    
    # Columns 1-2: Confusion matrices (2x2 grid)
    plot_confusion_matrices(fig, gs, confusion_matrices, dataset_name)
    
    # Columns 3-4: Prediction distribution pie charts (2x2 grid)
    plot_prediction_pie_charts(fig, gs, confusion_matrices, dataset_name)
    
    # Columns 5-6: Individual metrics bar charts (2x2 grid)
    plot_individual_metrics_bars(fig, gs, all_results, dataset_name)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), 'Outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'{dataset_name}_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_file}")
    
    plt.show()
    
    # Exit automatically after file is saved and displayed
    sys.exit(0)


def main():
    """Main pipeline"""
    try:
        print("\n" + "="*70)
        print("CLASSIFIER VISUALIZATION TOOL")
        print("="*70)
        
        # Available datasets
        datasets_dir = os.path.join(os.path.dirname(__file__), 'Datasets')
        datasets = {
            '1': ('bank', 'bank_cleaned.csv'),
            '2': ('heart', 'heart_cleaned.csv'),
            '3': ('wine', 'wine_cleaned.csv'),
            '4': ('breast', 'breast_cleaned.csv'),
            '5': ('churn', 'churn_cleaned.csv'),
        }
        
        print("\nAvailable Datasets:")
        print("-" * 70)
        for key, (name, filename) in datasets.items():
            filepath = os.path.join(datasets_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                print(f"  [{key}] {name.upper():10} - {df.shape[0]:5} samples × {df.shape[1]:2} features")
            else:
                print(f"  [{key}] {name.upper():10} - NOT FOUND")
        
        # Get user input
        print("\n" + "="*70)
        choice = input("Select dataset to visualize (1-5): ").strip()
        
        if choice not in datasets:
            print("❌ Invalid choice!")
            return
        
        dataset_name, filename = datasets[choice]
        filepath = os.path.join(datasets_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ Dataset file not found: {filepath}")
            return
        
        # Visualize
        visualize_dataset(dataset_name, filepath)
        
        print(f"\n{'='*70}\n")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
