from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import mlflow
import mlflow.sklearn
import itertools

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

# Define hyperparameter grid
n_estimators_options = [50, 100, 200]
max_depth_options = [10, 20, 30, None]

# Track best model
best_accuracy = 0
best_params = {}
best_model = None

print(f"\n{'='*70}")
print(f"Starting Random Forest Hyperparameter Tuning")
print(f"Testing {len(n_estimators_options)} n_estimators × {len(max_depth_options)} max_depth = {len(n_estimators_options) * len(max_depth_options)} combinations")
print(f"{'='*70}\n")

# Start parent run for hyperparameter tuning
with mlflow.start_run(run_name="Random Forest Hyperparameter Tuning") as parent_run:
    mlflow.set_tag("model_type", "Random Forest")
    mlflow.set_tag("tuning", "nested_runs")
    mlflow.set_tag("dataset", "BEED")
    
    # Log the hyperparameter search space
    mlflow.log_param("n_estimators_options", str(n_estimators_options))
    mlflow.log_param("max_depth_options", str(max_depth_options))
    mlflow.log_param("total_combinations", len(n_estimators_options) * len(max_depth_options))
    
    run_count = 0
    
    # Nested runs for each hyperparameter combination
    for n_estimators, max_depth in itertools.product(n_estimators_options, max_depth_options):
        run_count += 1
        
        with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}", nested=True) as child_run:
            # Set parameters
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": 42,
                "n_jobs": -1
            }
            
            # Log parameters
            mlflow.log_params(params)
            
            print(f"[{run_count}/{len(n_estimators_options) * len(max_depth_options)}] Training with n_estimators={n_estimators}, max_depth={max_depth}")
            
            # Train the model
            rf = RandomForestClassifier(**params)
            rf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = rf.predict(X_test)
            
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
            
            print(f"   Accuracy: {accuracy:.4f}, F1 (Macro): {f1_macro:.4f}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                best_model = rf
                print(f"   *** New best model! ***")
    
    # Log best model metrics to parent run
    print(f"\n{'='*70}")
    print(f"Hyperparameter Tuning Complete!")
    print(f"{'='*70}")
    print(f"\n=== Best Model Configuration ===")
    print(f"n_estimators: {best_params['n_estimators']}")
    print(f"max_depth: {best_params['max_depth']}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    # Make predictions with best model
    y_pred_best = best_model.predict(X_test)
    
    # Calculate final metrics for best model
    best_f1_macro = f1_score(y_test, y_pred_best, average='macro')
    best_f1_weighted = f1_score(y_test, y_pred_best, average='weighted')
    best_precision_macro = precision_score(y_test, y_pred_best, average='macro')
    best_recall_macro = recall_score(y_test, y_pred_best, average='macro')
    
    # Log best parameters and metrics to parent run
    mlflow.log_param("best_n_estimators", best_params['n_estimators'])
    mlflow.log_param("best_max_depth", best_params['max_depth'])
    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_metric("best_f1_macro", best_f1_macro)
    mlflow.log_metric("best_f1_weighted", best_f1_weighted)
    mlflow.log_metric("best_precision_macro", best_precision_macro)
    mlflow.log_metric("best_recall_macro", best_recall_macro)
    
    # Log best model
    mlflow.sklearn.log_model(best_model, "best_random_forest_model")
    
    # Print detailed results
    print(f"\n=== Best Model Performance ===")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"F1 Score (Macro): {best_f1_macro:.4f}")
    print(f"F1 Score (Weighted): {best_f1_weighted:.4f}")
    print(f"Precision (Macro): {best_precision_macro:.4f}")
    print(f"Recall (Macro): {best_recall_macro:.4f}")
    print(f"\n=== Classification Report (Best Model) ===")
    print(classification_report(y_test, y_pred_best))
    
    print(f"\nBest model saved to MLflow")
    print(f"Parent Run ID: {parent_run.info.run_id}")
