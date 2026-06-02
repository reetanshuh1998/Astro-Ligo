import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_xgboost(dataset_path, model_dir="data/models"):
    """
    Trains XGBoost models to predict neutron star Radius (R) and log10(Lambda)
    given stellar Mass (M) and core EOS parameters.
    """
    print("Loading dataset for XGBoost training...")
    df = pd.read_csv(dataset_path)
    
    # 1. Feature Engineering
    # Inputs: Mass only (real-world universal curve fitting)
    X_cols = ["M"]
    # Targets: Radius and log10_Lambda (highly recommended for dynamic range balancing)
    y_cols = ["R", "log10_Lambda"]
    
    # Shuffling and splitting (75% train, 10% validation, 15% test)
    # Using deterministic seed for reproducibility
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df_shuffled)
    n_train = int(n * 0.75)
    n_val = int(n * 0.10)
    
    train_df = df_shuffled.iloc[:n_train]
    val_df = df_shuffled.iloc[n_train:n_train+n_val]
    test_df = df_shuffled.iloc[n_train+n_val:]
    
    X_train, y_train = train_df[X_cols], train_df[y_cols]
    X_val, y_val = val_df[X_cols], val_df[y_cols]
    X_test, y_test = test_df[X_cols], test_df[y_cols]
    
    os.makedirs(model_dir, exist_ok=True)
    
    models = {}
    metrics = {}
    
    # 2. Train independent XGBoost regressors for each target
    for target in y_cols:
        print(f"Training XGBoost Regressor for target '{target}'...")
        
        # Hyperparameters optimized for scientific physics tabular datasets
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, 
            y_train[target],
            eval_set=[(X_val, y_val[target])],
            verbose=False
        )
        
        # Evaluate on test set
        preds = model.predict(X_test)
        true = y_test[target]
        
        mae = mean_absolute_error(true, preds)
        rmse = np.sqrt(mean_squared_error(true, preds))
        r2 = r2_score(true, preds)
        
        metrics[target] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"  Test metrics for {target}:")
        print(f"    MAE:  {mae:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    R2:   {r2:.4f}")
        
        # Save model
        model_path = os.path.join(model_dir, f"xgboost_{target}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        models[target] = model
        
    # Save the test partition for uniform downstream evaluation and plots
    test_df.to_csv(os.path.join(model_dir, "test_split.csv"), index=False)
    
    print("XGBoost training successfully completed.")
    return models, metrics

if __name__ == "__main__":
    # Test execution
    train_xgboost("data/processed/ns_dataset_test.csv")
