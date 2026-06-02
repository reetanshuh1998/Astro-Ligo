import os
import pickle
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm, kstest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ns_infer.models.train_dnn import DeterministicMLP
from ns_infer.models.train_bnn import EnsembleGaussianMLP, predict_ensemble

def evaluate_canonical_stars(ensemble, scaler_X, scaler_y, device):
    """
    Evaluates ensembled predictions at exactly 1.40 M_sun and 2.08 M_sun.
    Propagates both epistemic and aleatoric uncertainty by drawing samples
    from the ensembled Gaussian distributions.
    """
    # Create DataFrame with exact target masses
    X_canonical = pd.DataFrame({"M": [1.40, 2.08]})
    X_scaled = scaler_X.transform(X_canonical)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    # We will collect samples of physical R and Lambda to form full posterior ensembles
    r_samples = {1.40: [], 2.08: []}
    lambda_samples = {1.40: [], 2.08: []}
    
    # Put ensemble members in eval mode, keep dropout active for MC sampling
    for member in ensemble:
        member.eval()
        for m in member.modules():
            if isinstance(m, nn.Dropout):
                m.train()
                
    mc_runs = 50
    samples_per_run = 200
    np.random.seed(42)
    
    with torch.no_grad():
        for member in ensemble:
            for _ in range(mc_runs):
                pred_mean, pred_logvar = member(X_tensor)
                
                # Unscale to physical units
                # pred_mean: shape (2, 2) [batch, targets]
                # pred_logvar: shape (2, 2)
                mean_phys = pred_mean.cpu().numpy() * scaler_y.scale_ + scaler_y.mean_
                std_phys = np.sqrt(np.exp(pred_logvar.cpu().numpy())) * scaler_y.scale_
                
                for idx, mass in enumerate([1.40, 2.08]):
                    # Draw samples from the predicted Gaussian distributions (aleatoric + epistemic)
                    r_draws = np.random.normal(mean_phys[idx, 0], std_phys[idx, 0], size=samples_per_run)
                    log10_l_draws = np.random.normal(mean_phys[idx, 1], std_phys[idx, 1], size=samples_per_run)
                    
                    r_samples[mass].extend(r_draws)
                    lambda_samples[mass].extend(10**log10_l_draws)
                    
    # Compute summary statistics
    stats = {}
    for mass in [1.40, 2.08]:
        r_arr = np.array(r_samples[mass])
        l_arr = np.array(lambda_samples[mass])
        
        # We can compute mean, std, and credible intervals (16th, 50th, 84th percentiles for 68% CI)
        r_mean, r_std = np.mean(r_arr), np.std(r_arr)
        r_p16, r_p50, r_p84 = np.percentile(r_arr, [16, 50, 84])
        
        l_mean, l_std = np.mean(l_arr), np.std(l_arr)
        l_p16, l_p50, l_p84 = np.percentile(l_arr, [16, 50, 84])
        
        # 90% credible intervals (5th and 95th percentiles)
        r_p05, r_p95 = np.percentile(r_arr, [5, 95])
        l_p05, l_p95 = np.percentile(l_arr, [5, 95])
        
        stats[mass] = {
            "R": {
                "mean": float(r_mean),
                "std": float(r_std),
                "median": float(r_p50),
                "ci_68_lower": float(r_p16),
                "ci_68_upper": float(r_p84),
                "ci_90_lower": float(r_p05),
                "ci_90_upper": float(r_p95)
            },
            "Lambda": {
                "mean": float(l_mean),
                "std": float(l_std),
                "median": float(l_p50),
                "ci_68_lower": float(l_p16),
                "ci_68_upper": float(l_p84),
                "ci_90_lower": float(l_p05),
                "ci_90_upper": float(l_p95)
            }
        }
    return stats

def evaluate_models(model_dir="data/models", output_dir="data/metrics"):
    """
    Performs joint evaluation of XGBoost, DNN, and BNN Ensemble models on the test split.
    Computes accuracy, inference speed, and uncertainty calibration diagnostics.
    """
    print("Beginning downstream model evaluations on the test split...")
    test_split_path = os.path.join(model_dir, "test_split.csv")
    if not os.path.exists(test_split_path):
        print(f"Error: test split not found at {test_split_path}. Run training scripts first.")
        return None
        
    test_df = pd.read_csv(test_split_path)
    X_cols = ["M"]
    y_cols = ["R", "log10_Lambda"]
    
    X_test = test_df[X_cols]
    y_test = test_df[y_cols]
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {}
    
    # -------------------------------------------------------------
    # 1. EVALUATE XGBOOST
    # -------------------------------------------------------------
    xgb_models = {}
    xgb_preds = {}
    xgb_start_time = time.time()
    
    try:
        for target in y_cols:
            model_path = os.path.join(model_dir, f"xgboost_{target}.pkl")
            with open(model_path, "rb") as f:
                xgb_models[target] = pickle.load(f)
            xgb_preds[target] = xgb_models[target].predict(X_test)
            
        xgb_runtime_ms = (time.time() - xgb_start_time) / len(test_df) * 1000.0
        
        xgb_metrics = {}
        for target in y_cols:
            mae = mean_absolute_error(y_test[target], xgb_preds[target])
            rmse = np.sqrt(mean_squared_error(y_test[target], xgb_preds[target]))
            r2 = r2_score(y_test[target], xgb_preds[target])
            xgb_metrics[target] = {"MAE": mae, "RMSE": rmse, "R2": r2}
            
        results["XGBoost"] = {
            "metrics": xgb_metrics,
            "runtime_single_ms": xgb_runtime_ms
        }
        print("  XGBoost evaluated successfully.")
    except Exception as e:
        print(f"  Error evaluating XGBoost: {e}")
        
    # -------------------------------------------------------------
    # 2. EVALUATE DETERMINISTIC DNN
    # -------------------------------------------------------------
    try:
        with open(os.path.join(model_dir, "dnn_scaler_X.pkl"), "rb") as f:
            dnn_scaler_X = pickle.load(f)
        with open(os.path.join(model_dir, "dnn_scaler_y.pkl"), "rb") as f:
            dnn_scaler_y = pickle.load(f)
            
        dnn_model = DeterministicMLP(input_dim=1, output_dim=2).to(device)
        dnn_model.load_state_dict(torch.load(os.path.join(model_dir, "dnn_model.pt"), map_location=device))
        dnn_model.eval()
        
        X_test_scaled = dnn_scaler_X.transform(X_test)
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        
        dnn_start_time = time.time()
        with torch.no_grad():
            dnn_out = dnn_model(X_tensor).cpu().numpy()
        dnn_runtime_ms = (time.time() - dnn_start_time) / len(test_df) * 1000.0
        
        dnn_out_phys = dnn_out * dnn_scaler_y.scale_ + dnn_scaler_y.mean_
        
        dnn_metrics = {}
        for idx, target in enumerate(y_cols):
            mae = mean_absolute_error(y_test[target], dnn_out_phys[:, idx])
            rmse = np.sqrt(mean_squared_error(y_test[target], dnn_out_phys[:, idx]))
            r2 = r2_score(y_test[target], dnn_out_phys[:, idx])
            dnn_metrics[target] = {"MAE": mae, "RMSE": rmse, "R2": r2}
            
        results["DNN"] = {
            "metrics": dnn_metrics,
            "runtime_single_ms": dnn_runtime_ms
        }
        print("  Deterministic DNN evaluated successfully.")
    except Exception as e:
        print(f"  Error evaluating DNN: {e}")
        
    # -------------------------------------------------------------
    # 3. EVALUATE BNN GAUSSIAN DEEP ENSEMBLE + MC DROPOUT
    # -------------------------------------------------------------
    try:
        with open(os.path.join(model_dir, "bnn_scaler_X.pkl"), "rb") as f:
            bnn_scaler_X = pickle.load(f)
        with open(os.path.join(model_dir, "bnn_scaler_y.pkl"), "rb") as f:
            bnn_scaler_y = pickle.load(f)
            
        # Reconstruct ensemble members
        ensemble = []
        for i in range(5):
            member = EnsembleGaussianMLP(input_dim=1, num_targets=2).to(device)
            member.load_state_dict(torch.load(os.path.join(model_dir, f"bnn_member_{i}.pt"), map_location=device))
            ensemble.append(member)
            
        bnn_start_time = time.time()
        # predict_ensemble handles decoupling of epistemic/aleatoric uncertainty
        pred_mean, pred_std, std_aleatoric, std_epistemic = predict_ensemble(
            ensemble, X_test, bnn_scaler_X, bnn_scaler_y, device, mc_runs=20
        )
        bnn_runtime_ms = (time.time() - bnn_start_time) / len(test_df) * 1000.0
        
        bnn_metrics = {}
        calibration = {}
        
        for idx, target in enumerate(y_cols):
            mae = mean_absolute_error(y_test[target], pred_mean[:, idx])
            rmse = np.sqrt(mean_squared_error(y_test[target], pred_mean[:, idx]))
            r2 = r2_score(y_test[target], pred_mean[:, idx])
            bnn_metrics[target] = {"MAE": mae, "RMSE": rmse, "R2": r2}
            
            # Uncertainty Calibration evaluation
            errors = y_test[target] - pred_mean[:, idx]
            stds = pred_std[:, idx]
            
            z_scores = errors / (stds + 1e-10)
            
            cov_50 = np.mean(np.abs(z_scores) <= 0.67449)
            cov_90 = np.mean(np.abs(z_scores) <= 1.64485)
            
            pit_values = norm.cdf(z_scores)
            ks_stat, p_val = kstest(pit_values, 'uniform')
            
            calibration[target] = {
                "coverage_50_expected": 0.50,
                "coverage_50_actual": float(cov_50),
                "coverage_90_expected": 0.90,
                "coverage_90_actual": float(cov_90),
                "ks_stat": float(ks_stat),
                "pit_ks_pvalue": float(p_val)
            }
            
        results["Ensemble_BNN"] = {
            "metrics": bnn_metrics,
            "calibration": calibration,
            "runtime_single_ms": bnn_runtime_ms
        }
        
        # Save full predictions for analysis and plots
        pred_df = test_df.copy()
        pred_df["xgb_R"] = xgb_preds["R"]
        pred_df["xgb_log10_Lambda"] = xgb_preds["log10_Lambda"]
        
        pred_df["dnn_R"] = dnn_out_phys[:, 0]
        pred_df["dnn_log10_Lambda"] = dnn_out_phys[:, 1]
        
        pred_df["bnn_R"] = pred_mean[:, 0]
        pred_df["bnn_log10_Lambda"] = pred_mean[:, 1]
        pred_df["bnn_R_std"] = pred_std[:, 0]
        pred_df["bnn_log10_Lambda_std"] = pred_std[:, 1]
        
        pred_df["bnn_R_std_aleatoric"] = std_aleatoric[:, 0]
        pred_df["bnn_R_std_epistemic"] = std_epistemic[:, 0]
        pred_df["bnn_log10_Lambda_std_aleatoric"] = std_aleatoric[:, 1]
        pred_df["bnn_log10_Lambda_std_epistemic"] = std_epistemic[:, 1]
        
        pred_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
        print("  Ensemble Gaussian BNN evaluated successfully.")
        
        # ---- Canonical Predictions & Comparison Table Extraction ----
        print("  Evaluating canonical neutron star constraints at 1.40 M_sun and 2.08 M_sun...")
        canonical_stats = evaluate_canonical_stars(ensemble, bnn_scaler_X, bnn_scaler_y, device)
        results["Canonical_Predictions"] = canonical_stats
        
        r14_mean = canonical_stats[1.40]["R"]["mean"]
        r14_std = canonical_stats[1.40]["R"]["std"]
        r14_p16 = canonical_stats[1.40]["R"]["ci_68_lower"]
        r14_p84 = canonical_stats[1.40]["R"]["ci_68_upper"]
        
        l14_mean = canonical_stats[1.40]["Lambda"]["mean"]
        l14_std = canonical_stats[1.40]["Lambda"]["std"]
        l14_p05 = canonical_stats[1.40]["Lambda"]["ci_90_lower"]
        l14_p95 = canonical_stats[1.40]["Lambda"]["ci_90_upper"]
        
        r208_mean = canonical_stats[2.08]["R"]["mean"]
        r208_std = canonical_stats[2.08]["R"]["std"]
        r208_p16 = canonical_stats[2.08]["R"]["ci_68_lower"]
        r208_p84 = canonical_stats[2.08]["R"]["ci_68_upper"]
        
        table_str = f"""
========================================================================================
                 PUBLICATION-READY CANONICAL NEUTRON STAR CONSTRAINTS
========================================================================================
Observational Quantity        ML Predicted Value (BNN)       Official Collaboration Value
----------------------------------------------------------------------------------------
At M = 1.40 M_sun:
  Radius (R_1.4)              {r14_mean:5.2f} ± {r14_std:4.2f} km             11.90 ± 1.40 km  (LIGO GW170817)
                              [68% CI: {r14_p16:5.2f} - {r14_p84:5.2f} km]     12.71 ± 1.15 km  (NICER J0030)
                              
  Tidal Deformability (L_1.4)  {int(l14_mean):3d} ± {int(l14_std):2d}                    300 -190/+420    (LIGO GW170817)
                              [90% CI: {int(l14_p05):3d} - {int(l14_p95):3d}]           [90% CI: 110 - 720]

At M = 2.08 M_sun:
  Radius (R_2.08)             {r208_mean:5.2f} ± {r208_std:4.2f} km             12.39 ± 0.85 km  (NICER J0740)
                              [68% CI: {r208_p16:5.2f} - {r208_p84:5.2f} km]     [68% CI: 11.54 - 13.24 km]
========================================================================================
"""
        print(table_str)
        
        # Save table to file
        table_path = os.path.join(output_dir, "canonical_comparison_table.txt")
        with open(table_path, "w") as f_tbl:
            f_tbl.write(table_str)
        print(f"  Canonical comparison table saved to {table_path}")
        results["Canonical_ASCII_Table"] = table_str
        
    except Exception as e:
        print(f"  Error evaluating Ensemble BNN: {e}")
        import traceback
        traceback.print_exc()
        
    # Write metrics summary as JSON
    with open(os.path.join(output_dir, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Comparison report saved to {os.path.join(output_dir, 'model_comparison.json')}")
    return results

if __name__ == "__main__":
    evaluate_models()
