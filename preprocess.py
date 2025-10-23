from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import sys
import os
from sklearn.preprocessing import StandardScaler

# Load data from local CSV file
data_file = 'BEED_Data.csv'

try:
    df = pd.read_csv(data_file)
    print(f"BEED dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading data from file: {e}")
    sys.exit(1)

# Display basic information
print(f"\nDataset Info:")
print(f"- Number of samples: {len(df)}")
print(f"- Number of features: {len(df.columns) - 1}")
print(f"- Target variable: y")
print(f"- Target classes: {sorted(df['y'].unique())}")
print(f"\nTarget distribution:")
print(df['y'].value_counts().sort_index())

# Preprocessing Steps
# Check for missing values
missing_values = df.isnull().sum()
if missing_values.any():
    print(f"\nMissing values found:")
    print(missing_values[missing_values > 0])
    df.dropna(inplace=True)
    print(f"Rows after removing missing values: {len(df)}")
else:
    print("\nNo missing values found.")

# Separate features and target
X = df.drop('y', axis=1)
y = df['y']

# Standardize features (recommended for many ML algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create processed dataframe
df_processed = pd.DataFrame(X_scaled, columns=X.columns)
df_processed['y'] = y.values

# Save the preprocessed file
os.makedirs('data', exist_ok=True)
df_processed.to_csv('data/preprocessed_beed.csv', index=False)
print(f"\nPreprocessing complete!")
print(f"Preprocessed file saved to: data/preprocessed_beed.csv")