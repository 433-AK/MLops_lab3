import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import lightgbm as lgb
import mlflow

# Enable autologging for LightGBM
mlflow.lightgbm.autolog()

# Set experiment name
mlflow.set_experiment("BEED Classification")

# Load preprocessed data
df = pd.read_csv('data/preprocessed_beed.csv')
print(f"Dataset loaded: {df.shape}")
print(f"Target distribution:\n{df['y'].value_counts().sort_index()}")

# Prepare features and target
X = df.drop('y', axis=1)
y = df['y']

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Start MLflow run
with mlflow.start_run(run_name="LightGBM Autolog"):
    # LightGBM classifier for multiclass classification
    model = lgb.LGBMClassifier(
        random_state=42,
        objective='multiclass',  # multiclass classification
        num_class=4,  # 4 classes (0, 1, 2, 3)
        n_estimators=100,
        verbose=-1  # suppress output
    )
    
    # Autologging captures training metrics by providing an eval_set
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss'
    )
    
    # Manually log extra metrics that autolog might not capture
    y_pred = model.predict(X_test)
    
    # Calculate metrics for multiclass classification
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_macro", precision_macro)
    mlflow.log_metric("recall_macro", recall_macro)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("f1_weighted", f1_weighted)
    
    # Print results
    print(f"\n=== Model Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
