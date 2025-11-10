"""
Data Profiling - Generate profile reports for cleaned datasets
"""

import subprocess
import sys
import os

# Install packages
packages = ['pydantic-settings', 'ydata-profiling']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

from ydata_profiling import ProfileReport
import pandas as pd

# Generate profiles
datasets = ['wine', 'heart', 'bank', 'breast', 'churn']
os.makedirs('Datasets/profiles', exist_ok=True)

print("\nGenerating Profile Reports...")
print("=" * 50)

for name in datasets:
    df = pd.read_csv(f'Datasets/{name}_cleaned.csv')
    prof = ProfileReport(df, minimal=True)
    prof.to_file(f'Datasets/profiles/{name}_profile.html')
    print(f"  {name.upper():10} -> {name}_profile.html")

print("=" * 50)
print("Done!\n")

