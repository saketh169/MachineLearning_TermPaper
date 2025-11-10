"""
================================================================================
DATASET CLEANUP & PREPROCESSING
================================================================================
Purpose: Clean and preprocess 5 datasets for binary classification
- Remove duplicates
- Handle missing values  
- Encode categorical variables
- Drop unnecessary ID columns
- Binarize targets
- Save cleaned datasets with '_cleaned' suffix
================================================================================
"""

import pandas as pd
import numpy as np
import os

DATASET_DIR = "Datasets"

# ================================================================================
# STEP 0: DATASET CONFIGURATIONS
# ================================================================================

# Dataset configurations with processing steps
DATASETS = {
    'wine': {
        'file': 'wine.csv',
        'target': 'quality',
        'steps': ['remove_duplicates', 'handle_missing', 'binarize_target'],
        'threshold': 5,
    },
    'heart': {
        'file': 'heart.csv',
        'target': 'target',
        'steps': ['handle_missing', 'ensure_binary'],
    },
    'bank': {
        'file': 'bank.csv',
        'target': 'class',
        'steps': ['handle_missing'],
    },
    'breast': {
        'file': 'breast.csv',
        'target': 'diagnosis',
        'steps': ['drop_cols', 'handle_missing', 'encode_diagnosis'],
    },
    'churn': {
        'file': 'churn.csv',
        'target': 'churn',
        'steps': ['drop_cols', 'handle_missing', 'encode_categorical', 'scale_features', 'balance_classes'],
    },
}


# ================================================================================
# STEP 1: REMOVE DUPLICATES
# ================================================================================

def remove_duplicates(df, name):
    """Remove duplicate rows."""
    print(f"  [1] Removing duplicates...")
    removed = len(df) - len(df.drop_duplicates())
    df = df.drop_duplicates(keep='first')
    print(f"      Removed {removed} duplicates")
    return df


# ================================================================================
# STEP 2: HANDLE MISSING VALUES
# ================================================================================

def handle_missing(df, name):
    """Handle missing values - drop empty columns, impute numeric with median."""
    print(f"  [2] Handling missing values...")
    
    # Drop empty columns
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        print(f"      Dropping empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)
    
    # Impute missing numeric columns with median
    missing_cols = df.columns[df.isna().any()].tolist()
    if missing_cols:
        for col in missing_cols:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"      Imputed {col} with median")
    
    print(f"      Total NaNs remaining: {df.isna().sum().sum()}")
    return df


# ================================================================================
# STEP 3: BINARIZE TARGET VARIABLES
# ================================================================================

def binarize_target(df, name, target_col, threshold):
    """Convert target to binary 0/1 based on threshold."""
    print(f"  [3] Binarizing target '{target_col}' (> {threshold})...")
    df[target_col] = (df[target_col] > threshold).astype(int)
    print(f"      Values: {sorted(df[target_col].unique())}")
    return df


def ensure_binary(df, name, target_col):
    """Ensure target is binary 0/1."""
    print(f"  [2] Ensuring binary target...")
    df[target_col] = df[target_col].astype(int)
    print(f"      Values: {sorted(df[target_col].unique())}")
    return df


# ================================================================================
# STEP 4: ENCODE CATEGORICAL VARIABLES
# ================================================================================

def encode_diagnosis(df, name):
    """Encode diagnosis: M->1, B->0."""
    print(f"  [3] Encoding diagnosis...")
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


def encode_categorical(df, name, target_col=None):
    """One-hot encode categorical columns, ensuring target stays last."""
    print(f"  [3] Encoding categorical variables...")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target column from categorical encoding if it's there
    if target_col and target_col in cat_cols:
        cat_cols.remove(target_col)
        print(f"      Excluding target column '{target_col}' from encoding")
    
    if cat_cols:
        print(f"      Encoding: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, dtype=int)
    
    # Ensure target column is the last column
    if target_col and target_col in df.columns:
        cols = [col for col in df.columns if col != target_col] + [target_col]
        df = df[cols]
        print(f"      Moved target '{target_col}' to last position")
    
    return df


# ================================================================================
# STEP 5: BALANCE CLASSES
# ================================================================================

def balance_classes(df, name, target_col):
    """Balance classes by keeping all minority samples and randomly selecting majority samples."""
    print(f"  [5] Balancing classes for target '{target_col}'...")
    
    if target_col not in df.columns:
        print(f"      Target column '{target_col}' not found, skipping balancing")
        return df
    
    # Get class distributions
    class_counts = df[target_col].value_counts().sort_index()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    
    print(f"      Original: Class {int(minority_class)}: {minority_count}, Class {int(majority_class)}: {majority_count}")
    
    if minority_count >= majority_count * 0.8:  # If already reasonably balanced
        print(f"      Dataset already balanced, skipping")
        return df
    
    # Keep all minority class samples
    minority_df = df[df[target_col] == minority_class]
    
    # Randomly select majority samples to match minority count
    majority_df = df[df[target_col] == majority_class].sample(n=minority_count, random_state=42)
    
    # Combine and shuffle
    df_balanced = pd.concat([minority_df, majority_df], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Verify balancing
    balanced_counts = df_balanced[target_col].value_counts().sort_index()
    print(f"      Balanced: {dict(balanced_counts)}")
    print(f"      Final shape: {df_balanced.shape}")
    
    return df_balanced


# ================================================================================
# STEP 4: SCALE FEATURES
# ================================================================================

def scale_features(df, name):
    """Scale continuous features using StandardScaler, leave binary features as-is."""
    print(f"  [4] Scaling continuous features...")
    
    # Define feature types for each dataset
    feature_types = {
        'churn': {
            'continuous': ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary'],
            'binary': ['credit_card', 'active_member', 'country_France', 'country_Germany', 'country_Spain', 'gender_Female', 'gender_Male']
        },
        'wine': {
            'continuous': ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                          'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
                          'pH', 'sulphates', 'alcohol'],
            'binary': []
        },
        'heart': {
            'continuous': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
            'binary': ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        },
        'bank': {
            'continuous': ['variance', 'skewness', 'curtosis', 'entropy'],
            'binary': []
        },
        'breast': {
            'continuous': [col for col in df.columns if col not in ['diagnosis'] and not col.startswith('Unnamed')],
            'binary': []
        }
    }
    
    if name in feature_types:
        config = feature_types[name]
        continuous_cols = [col for col in config['continuous'] if col in df.columns]
        binary_cols = [col for col in config['binary'] if col in df.columns]
        
        print(f"      Continuous features: {continuous_cols}")
        print(f"      Binary features: {binary_cols}")
        
        # Scale continuous features
        if continuous_cols:
            for col in continuous_cols:
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std() + 1e-8  # Avoid division by zero
                    df[col] = (df[col] - mean_val) / std_val
                    print(f"        Scaled {col}: mean={mean_val:.2f}, std={std_val:.2f}")
        
        # Binary features remain as-is (0/1)
        if binary_cols:
            print(f"      Binary features kept as 0/1: {binary_cols}")
    
    return df


def drop_cols(df, name):
    """Drop ID columns."""
    print(f"  [1] Dropping ID columns...")
    id_cols = ['id', 'customer_id', 'ID', 'Id']
    cols_to_drop = [col for col in id_cols if col in df.columns]
    if cols_to_drop:
        print(f"      Dropped: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


# ================================================================================
# STEP 6: PROCESS DATASET - Apply all cleaning steps
# ================================================================================

def process_dataset(name, config):
    """Process single dataset."""
    print(f"\n{'='*70}")
    print(f"Processing: {name.upper()}")
    print(f"{'='*70}")
    
    # Load
    filepath = os.path.join(DATASET_DIR, config['file'])
    df = pd.read_csv(filepath)
    print(f"  Loaded: {df.shape} | Columns: {list(df.columns)}")
    
    # Apply steps
    for step in config['steps']:
        if step == 'remove_duplicates':
            df = remove_duplicates(df, name)
        elif step == 'handle_missing':
            df = handle_missing(df, name)
        elif step == 'binarize_target':
            df = binarize_target(df, name, config['target'], config['threshold'])
        elif step == 'ensure_binary':
            df = ensure_binary(df, name, config['target'])
        elif step == 'encode_diagnosis':
            df = encode_diagnosis(df, name)
        elif step == 'drop_cols':
            df = drop_cols(df, name)
        elif step == 'encode_categorical':
            df = encode_categorical(df, name, config.get('target'))
        elif step == 'scale_features':
            df = scale_features(df, name)
        elif step == 'balance_classes':
            df = balance_classes(df, name, config.get('target'))
    
    # Save
    output_file = os.path.join(DATASET_DIR, config['file'].replace('.csv', '_cleaned.csv'))
    df.to_csv(output_file, index=False)
    print(f"\n  Saved: {output_file}")
    print(f"  Final shape: {df.shape}")
    print(f"  NaNs: {df.isna().sum().sum()}")
    
    return df


# ================================================================================
# STEP 7: MAIN PIPELINE - Run cleanup on all datasets
# ================================================================================

def main():
    """Main pipeline."""
    print("\n" + "="*70)
    print("DATASET CLEANUP PIPELINE")
    print("="*70)
    
    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: {DATASET_DIR} directory not found!")
        return
    
    # Process all datasets
    results = {}
    for name, config in DATASETS.items():
        try:
            df = process_dataset(name, config)
            results[name] = ('SUCCESS', df.shape)
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results[name] = ('FAILED', str(e))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, (status, info) in results.items():
        print(f"  {name.upper():10} | {status:8} | {info}")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()