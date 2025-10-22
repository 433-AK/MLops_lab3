import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import mlflow
import joblib  # Used for saving and loading scikit-learn models

# Set experiment name
mlflow.set_experiment("BEED Classification")

# Load preprocessed data
df = pd.read_csv('data/preprocessed_beed.csv')
print(f"Dataset loaded: {df.shape}")
print(f"Target distribution:\n{df['y'].value_counts().sort_index()}")

# Prepare features and target
X = df.drop('y', axis=1)
y = df['y']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Start MLflow run
with mlflow.start_run(run_name="Logistic Regression Baseline"):
    mlflow.set_tag("model_type", "Logistic Regression")
    mlflow.set_tag("dataset", "BEED")
    
    # Model parameters (multi_class='multinomial' for multiclass classification)
    params = {
        "solver": "lbfgs",
        "multi_class": "multinomial",
        "max_iter": 1000,
        "random_state": 42
    }
    mlflow.log_params(params)
    
    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("f1_weighted", f1_weighted)
    mlflow.log_metric("precision_macro", precision_macro)
    mlflow.log_metric("recall_macro", recall_macro)
    
    # Print results
    print(f"\n=== Model Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    # Save the model using joblib
    joblib.dump(lr, "model.joblib")
    mlflow.log_artifact("model.joblib")
    
    print(f"\nModel saved to: model.joblib")