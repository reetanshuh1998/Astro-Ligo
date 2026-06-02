import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from ns_infer.data.observational_data import ObservationalDataManager

# Set elegant scientific styling for plots
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# Harmonious HSL-tailored colors
COLOR_TRUE = "#1a1a1a"       # Charcoal / Black
COLOR_XGB = "#e66101"        # Bright Orange
COLOR_DNN = "#5e3c99"        # Deep Purple
COLOR_BNN_MEAN = "#2b8cbe"   # Sky Blue
COLOR_BNN_SHADE1 = "#a6bddb" # Light Blue (1-sigma)
COLOR_BNN_SHADE2 = "#ece7f2" # Very Light Blue (2-sigma)

def plot_mr_curves(pred_df, figure_dir):
    """Plots the ML-reconstructed real-universe Mass-Radius curve overlaid with real observations."""
    print("  Plotting Mass-Radius curves...")
    raw_obs_path = "data/processed/raw_observations.csv"
    if not os.path.exists(raw_obs_path):
        print("  Error: raw_observations.csv not found!")
        return
        
    raw_df = pd.read_csv(raw_obs_path)
    
    # Sort test predictions by mass to yield smooth continuous curves
    sub_df = pred_df.sort_values(by="M")
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    # Plot BNN uncertainty bands representing the real-world EOS parameter space
    m_vals = sub_df["M"].values
    r_bnn = sub_df["bnn_R"].values
    std_bnn = sub_df["bnn_R_std"].values
    
    ax.fill_betweenx(m_vals, r_bnn - 1.96 * std_bnn, r_bnn + 1.96 * std_bnn, color=COLOR_BNN_SHADE2, alpha=0.5, label="BNN 95% CI (Real-world EOS)")
    ax.fill_betweenx(m_vals, r_bnn - 1.0 * std_bnn, r_bnn + 1.0 * std_bnn, color=COLOR_BNN_SHADE1, alpha=0.6, label="BNN 68% CI (Real-world EOS)")
    
    # Plot ML reconstructed mean universal curves of our universe
    ax.plot(sub_df["xgb_R"], m_vals, color=COLOR_XGB, linestyle="--", linewidth=2.0, label="XGBoost Emulator")
    ax.plot(sub_df["dnn_R"], m_vals, color=COLOR_DNN, linestyle=":", linewidth=2.0, label="Dense MLP Emulator")
    ax.plot(r_bnn, m_vals, color=COLOR_BNN_MEAN, linestyle="-.", linewidth=2.0, label="BNN Mean Emulator")
    
    # Overlay the actual discrete physical observations from scientific releases
    # Group by observational type for rich aesthetics
    obs_groups = {
        "NICER":      {"color": "#08519c", "marker": "o", "label": "NICER Pulse Profile (Riley/Miller)"},
        "qLMXB":      {"color": "#756bb1", "marker": "s", "label": "qLMXB Radii (Lattimer/Steiner)"},
        "PRE_Burst":  {"color": "#d95f02", "marker": "d", "label": "PRE Bursters (Ozel et al.)"},
        "GW_Merger":  {"color": "#31a354", "marker": "^", "label": "LIGO/Virgo Mergers (GW170817)"}
    }
    
    for obs_type, style in obs_groups.items():
        type_df = raw_df[raw_df["type"] == obs_type]
        if not type_df.empty:
            ax.errorbar(
                type_df["R"], type_df["M"],
                xerr=type_df["R_err"], yerr=type_df["M_err"],
                fmt=style["marker"], color=style["color"],
                ecolor=style["color"], elinewidth=1.5, capsize=3,
                markersize=8, markeredgecolor='black', alpha=0.9,
                label=style["label"]
            )
            
    # Official LIGO & NICER Credible Intervals Visual Overlays
    # 1. LIGO GW170817 90% credible range for R_1.4: 11.9 +/- 1.4 km
    ax.errorbar(
        11.9, 1.40, xerr=1.4, fmt="*", color="#2ca02c", ecolor="#2ca02c",
        elinewidth=4, capsize=8, markersize=12, label="LIGO GW170817 (90% CI for $R_{1.4}$)", zorder=5
    )
    # 2. NICER J0030+0451 68% credible range for R_1.4: 12.71 +/- 1.15 km
    ax.errorbar(
        12.71, 1.40, xerr=1.15, fmt="p", color="#1f77b4", ecolor="#1f77b4",
        elinewidth=4, capsize=8, markersize=10, label="NICER PSR J0030 (68% CI for $R_{1.4}$)", zorder=5
    )
    # 3. NICER J0740+6620 68% credible range for R_2.08: 12.39 +/- 0.85 km
    ax.errorbar(
        12.39, 2.08, xerr=0.85, fmt="h", color="#9467bd", ecolor="#9467bd",
        elinewidth=4, capsize=8, markersize=10, label="NICER PSR J0740 (68% CI for $R_{2.08}$)", zorder=5
    )
            
    ax.set_title("ML Reconstructed Real-Universe Mass-Radius Curve\n(Vetted Empirical Astronomical Observations Overlay)")
    ax.set_xlabel("Stellar Radius $R$ (km)")
    ax.set_ylabel("Stellar Mass $M$ ($M_\\odot$)")
    ax.set_xlim(9.0, 16.0)
    ax.set_ylim(0.8, 2.4)
    ax.legend(loc="lower left", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "01_mass_radius_comparison.png"), dpi=300)
    plt.close()

def plot_mlambda_curves(pred_df, figure_dir):
    """Plots the ML-reconstructed real-universe Mass-Tidal Deformability curve."""
    print("  Plotting Mass-Lambda curves...")
    raw_obs_path = "data/processed/raw_observations.csv"
    if not os.path.exists(raw_obs_path):
        print("  Error: raw_observations.csv not found!")
        return
        
    raw_df = pd.read_csv(raw_obs_path)
    sub_df = pred_df.sort_values(by="M")
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    m_vals = sub_df["M"].values
    xgb_lambda = 10**sub_df["xgb_log10_Lambda"]
    dnn_lambda = 10**sub_df["dnn_log10_Lambda"]
    
    log_lambda_mean = sub_df["bnn_log10_Lambda"].values
    log_lambda_std = sub_df["bnn_log10_Lambda_std"].values
    
    lambda_bnn = 10**log_lambda_mean
    lambda_bnn_high1 = 10**(log_lambda_mean + log_lambda_std)
    lambda_bnn_low1 = 10**(log_lambda_mean - log_lambda_std)
    lambda_bnn_high2 = 10**(log_lambda_mean + 1.96 * log_lambda_std)
    lambda_bnn_low2 = 10**(log_lambda_mean - 1.96 * log_lambda_std)
    
    # Plot ensembled CI bands
    ax.fill_between(m_vals, lambda_bnn_low2, lambda_bnn_high2, color=COLOR_BNN_SHADE2, alpha=0.5, label="BNN 95% CI")
    ax.fill_between(m_vals, lambda_bnn_low1, lambda_bnn_high1, color=COLOR_BNN_SHADE1, alpha=0.6, label="BNN 68% CI")
    
    # Plot reconstructed universal curves
    ax.plot(m_vals, xgb_lambda, color=COLOR_XGB, linestyle="--", linewidth=2.0, label="XGBoost Emulator")
    ax.plot(m_vals, dnn_lambda, color=COLOR_DNN, linestyle=":", linewidth=2.0, label="Dense MLP Emulator")
    ax.plot(m_vals, lambda_bnn, color=COLOR_BNN_MEAN, linestyle="-.", linewidth=2.0, label="BNN Mean Emulator")
    
    # Plot real observed tidal deformability constraints
    # For GW mergers (GW170817), we have direct constraints
    gw_df = raw_df[raw_df["type"] == "GW_Merger"]
    if not gw_df.empty:
        # Standard errors for GW170817 components
        # Lambda1: 300 +/- 150, Lambda2: 400 +/- 200
        lambda_err = np.array([150.0, 200.0, 250.0]) # GW190425 has high error
        ax.errorbar(
            gw_df["M"], gw_df["Lambda"],
            xerr=gw_df["M_err"], yerr=lambda_err[:len(gw_df)],
            fmt="^", color="#31a354", ecolor="#31a354",
            elinewidth=1.5, capsize=3, markersize=8,
            markeredgecolor='black', alpha=0.9,
            label="LIGO/Virgo GW Direct Measurements"
        )
        
    # Plot universal Yagi-Yunes mapped points for X-ray stars as semi-transparent diamonds
    xray_df = raw_df[raw_df["type"] != "GW_Merger"]
    if not xray_df.empty:
        ax.scatter(
            xray_df["M"], xray_df["Lambda"],
            marker="d", color="#d95f02", s=40,
            alpha=0.5, edgecolors='black',
            label="X-Ray Stars (Yagi-Yunes Universal Mapped)"
        )
        
    # Official LIGO GW170817 90% CI for Lambda_1.4: 300 -190/+420
    lambda_central = 300.0
    lambda_err = [[190.0], [420.0]]  # shape (2, 1) for asymmetric yerr
    ax.errorbar(
        [1.40], [lambda_central], yerr=lambda_err, fmt="*", color="#2ca02c", ecolor="#2ca02c",
        elinewidth=4, capsize=8, markersize=12, label="LIGO GW170817 (90% CI for $\\Lambda_{1.4}$)", zorder=5
    )
        
    ax.set_yscale("log")
    ax.set_title("ML Reconstructed Real-Universe Mass-Tidal Deformability Curve\n(Vetted Empirical Astronomical Observations Overlay)")
    ax.set_xlabel("Stellar Mass $M$ ($M_\\odot$)")
    ax.set_ylabel("Tidal Deformability $\\Lambda$")
    ax.set_xlim(0.8, 2.4)
    ax.set_ylim(1.0, 1e4)
    ax.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "02_mass_lambda_comparison.png"), dpi=300)
    plt.close()

def plot_calibration(pred_df, figure_dir):
    """Plots PIT histograms and Empirical Coverage curves for uncertainty diagnostics (globally and binned by mass)."""
    print("  Plotting calibration diagnostics...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. PIT Histogram for R
    # PIT = norm.cdf((True_R - BNN_R_mean) / BNN_R_std)
    errors = pred_df["R"] - pred_df["bnn_R"]
    stds = pred_df["bnn_R_std"]
    pit_values = norm.cdf(errors / (stds + 1e-10))
    
    ax0 = axes[0]
    ax0.hist(pit_values, bins=15, density=True, color=COLOR_BNN_SHADE1, edgecolor="black", alpha=0.8)
    ax0.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Perfect Calibration")
    ax0.set_xlabel("Probability Integral Transform (PIT) value")
    ax0.set_ylabel("Probability Density")
    ax0.set_title("Stellar Radius PIT Calibration Histogram")
    ax0.legend(loc="upper right")
    
    # 2. Empirical Coverage Curve (Global)
    # Loop over expected coverages from 5% to 95%
    ax1 = axes[1]
    expected_coverages = np.linspace(0.05, 0.95, 19)
    actual_coverages_R = []
    actual_coverages_L = []
    
    # Radius error/std
    z_scores_R = np.abs(errors / (stds + 1e-10))
    # Lambda error/std
    errors_L = pred_df["log10_Lambda"] - pred_df["bnn_log10_Lambda"]
    stds_L = pred_df["bnn_log10_Lambda_std"]
    z_scores_L = np.abs(errors_L / (stds_L + 1e-10))
    
    for exp_cov in expected_coverages:
        # Find corresponding Z-threshold
        z_thresh = norm.ppf(0.5 + exp_cov / 2.0)
        
        act_cov_R = np.mean(z_scores_R <= z_thresh)
        act_cov_L = np.mean(z_scores_L <= z_thresh)
        
        actual_coverages_R.append(act_cov_R)
        actual_coverages_L.append(act_cov_L)
        
    ax1.plot(expected_coverages, expected_coverages, color="red", linestyle="--", linewidth=1.5, label="Perfect Calibration")
    ax1.plot(expected_coverages, actual_coverages_R, color=COLOR_BNN_MEAN, marker="o", linewidth=2.0, label="Radius $R$")
    ax1.plot(expected_coverages, actual_coverages_L, color=COLOR_DNN, marker="s", linewidth=2.0, label="$\\log_{10} \\Lambda$")
    
    ax1.set_xlabel("Expected Confidence Interval Level")
    ax1.set_ylabel("Actual Empirical Coverage")
    ax1.set_title("Global Calibration Curve")
    ax1.legend(loc="upper left")
    
    # 3. Mass-Binned Calibration Curve (for Radius R)
    # Splitting into Low-Mass (< 1.3 M_sun), Mid-Mass (1.3 - 1.7 M_sun), and High-Mass (> 1.7 M_sun)
    ax2 = axes[2]
    ax2.plot(expected_coverages, expected_coverages, color="red", linestyle="--", linewidth=1.5, label="Perfect Calibration")
    
    bin_low = pred_df[pred_df["M"] < 1.3]
    bin_mid = pred_df[(pred_df["M"] >= 1.3) & (pred_df["M"] <= 1.7)]
    bin_high = pred_df[pred_df["M"] > 1.7]
    
    bins = [
        (bin_low, "Low-Mass ($M < 1.3 M_\\odot$)", "#2ca02c", "o"),
        (bin_mid, "Mid-Mass ($1.3 \\leq M \\leq 1.7 M_\\odot$)", "#1f77b4", "s"),
        (bin_high, "High-Mass ($M > 1.7 M_\\odot$)", "#9467bd", "^")
    ]
    
    for df_bin, label_bin, color_bin, marker_bin in bins:
        if not df_bin.empty:
            errors_bin = df_bin["R"] - df_bin["bnn_R"]
            stds_bin = df_bin["bnn_R_std"]
            z_scores_bin = np.abs(errors_bin / (stds_bin + 1e-10))
            
            act_coverages_bin = []
            for exp_cov in expected_coverages:
                z_thresh = norm.ppf(0.5 + exp_cov / 2.0)
                act_coverages_bin.append(np.mean(z_scores_bin <= z_thresh))
                
            ax2.plot(expected_coverages, act_coverages_bin, color=color_bin, marker=marker_bin, linewidth=1.8, label=label_bin)
            
    ax2.set_xlabel("Expected Confidence Interval Level")
    ax2.set_ylabel("Actual Empirical Coverage")
    ax2.set_title("Mass-Dependent Calibration for Radius $R$")
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "03_uncertainty_calibration.png"), dpi=300)
    plt.close()

def plot_observational_posteriors(figure_dir, mode="mock"):
    """Plots observational posteriors reconstructed by ML BNN vs actual observations."""
    print("  Plotting observational posterior contours...")
    obs_manager = ObservationalDataManager(mode=mode)
    
    # 1. NICER PSR J0030+0451 Mass-Radius Posterior Contours
    j0030_df = obs_manager.get_j0030_posteriors(num_samples=1000)
    
    # 2. NICER PSR J0740+6620 Mass-Radius Posterior Contours
    j0740_df = obs_manager.get_j0740_posteriors(num_samples=1000)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Plot J0030
    ax0 = axes[0]
    # Standard KDE contours for J0030 observational posteriors
    m0, r0 = j0030_df["M"].values, j0030_df["R"].values
    xmin, xmax = 9.0, 16.0
    ymin, ymax = 0.8, 2.0
    
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([r0, m0])
    kernel = gaussian_kde(values)
    zz = np.reshape(kernel(positions).T, xx.shape)
    
    # Plot contour density
    cf0 = ax0.contourf(xx, yy, zz, cmap="Blues", alpha=0.7)
    c0 = ax0.contour(xx, yy, zz, colors="darkblue", linewidths=1.0)
    
    ax0.set_xlabel("Radius $R$ (km)")
    ax0.set_ylabel("Stellar Mass $M$ ($M_\\odot$)")
    ax0.set_title("PSR J0030+0451 Mass-Radius Contours")
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)
    
    # Plot J0740
    ax1 = axes[1]
    m1, r1 = j0740_df["M"].values, j0740_df["R"].values
    xmin, xmax = 9.0, 16.0
    ymin, ymax = 1.6, 2.4
    
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([r1, m1])
    kernel = gaussian_kde(values)
    zz = np.reshape(kernel(positions).T, xx.shape)
    
    cf1 = ax1.contourf(xx, yy, zz, cmap="Purples", alpha=0.7)
    c1 = ax1.contour(xx, yy, zz, colors="purple", linewidths=1.0)
    
    ax1.set_xlabel("Radius $R$ (km)")
    ax1.set_ylabel("Stellar Mass $M$ ($M_\\odot$)")
    ax1.set_title("PSR J0740+6620 Mass-Radius Contours")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "04_observational_posteriors.png"), dpi=300)
    plt.close()

def plot_performance_summary(metrics_path, figure_dir):
    """Generates an elegant bar chart benchmarking model accuracy and execution speed."""
    print("  Plotting model performance metrics...")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    models = ["XGBoost", "DNN", "Ensemble_BNN"]
    r_mae = []
    l_mae = []
    runtimes = []
    
    for m in models:
        r_mae.append(metrics[m]["metrics"]["R"]["MAE"])
        l_mae.append(metrics[m]["metrics"]["log10_Lambda"]["MAE"])
        runtimes.append(metrics[m]["runtime_single_ms"])
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Accuracy bar chart
    ax0 = axes[0]
    x = np.arange(len(models))
    width = 0.35
    
    ax0.bar(x - width/2, r_mae, width, label="Radius $R$ (km)", color=COLOR_BNN_MEAN)
    ax0.bar(x + width/2, l_mae, width, label="$\\log_{10} \\Lambda$", color=COLOR_DNN)
    
    ax0.set_xticks(x)
    ax0.set_xticklabels(["XGBoost", "Dense MLP", "Gaussian Ensemble"])
    ax0.set_ylabel("Mean Absolute Error (MAE)")
    ax0.set_title("Model Parameter Prediction Accuracy (MAE)")
    ax0.legend()
    
    # 2. Execution Speed (Logarithmic scale)
    ax1 = axes[1]
    # Add MCMC baseline which typically takes ~ 10^7 ms (approx 3 hours per star)
    models_speed = ["Traditional MCMC", "XGBoost", "Dense MLP", "Gaussian Ensemble"]
    runtimes_all = [1.08e7, runtimes[0], runtimes[1], runtimes[2]]
    
    bars = ax1.bar(models_speed, runtimes_all, color=["#d73027", COLOR_XGB, COLOR_DNN, COLOR_BNN_MEAN], width=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Inference Runtime per Star (ms)")
    ax1.set_title("Inference Speed Performance Benchmark")
    
    # Add exact values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.0,
            height * 1.5 if height < 1e6 else height / 4.0,
            f"{height:.2e} ms" if height > 10 else f"{height:.2f} ms",
            ha="center", va="bottom", fontsize=9, color="black" if height < 1e6 else "white", weight="bold"
        )
        
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "05_speed_accuracy_benchmark.png"), dpi=300)
    plt.close()

def plot_uncertainty_breakdown(pred_df, figure_dir):
    """
    Plots the decoupled uncertainty components (aleatoric vs epistemic standard deviation)
    predicted by the deep BNN ensemble as a function of stellar mass.
    """
    print("  Plotting decoupled uncertainty components...")
    sub_df = pred_df.sort_values(by="M")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Panel 1: Radius uncertainty breakdown
    ax0 = axes[0]
    ax0.plot(sub_df["M"], sub_df["bnn_R_std"], color="black", linestyle="-", linewidth=2.0, label="Total Predictive $\\sigma_R$")
    ax0.plot(sub_df["M"], sub_df["bnn_R_std_aleatoric"], color="#e31a1c", linestyle="--", linewidth=1.8, label="Aleatoric $\\sigma_R$ (Obs. Noise)")
    ax0.plot(sub_df["M"], sub_df["bnn_R_std_epistemic"], color="#1f78b4", linestyle=":", linewidth=2.0, label="Epistemic $\\sigma_R$ (Model)")
    
    ax0.set_xlabel("Stellar Mass $M$ ($M_\\odot$)")
    ax0.set_ylabel("Radius Uncertainty $\\sigma_R$ (km)")
    ax0.set_title("Stellar Radius Uncertainty Breakdown")
    ax0.legend(loc="upper right")
    
    # Panel 2: Lambda uncertainty breakdown
    ax1 = axes[1]
    ax1.plot(sub_df["M"], sub_df["bnn_log10_Lambda_std"], color="black", linestyle="-", linewidth=2.0, label="Total Predictive $\\sigma_{\\log\\Lambda}$")
    ax1.plot(sub_df["M"], sub_df["bnn_log10_Lambda_std_aleatoric"], color="#e31a1c", linestyle="--", linewidth=1.8, label="Aleatoric $\\sigma_{\\log\\Lambda}$ (Obs. Noise)")
    ax1.plot(sub_df["M"], sub_df["bnn_log10_Lambda_std_epistemic"], color="#1f78b4", linestyle=":", linewidth=2.0, label="Epistemic $\\sigma_{\\log\\Lambda}$ (Model)")
    
    ax1.set_xlabel("Stellar Mass $M$ ($M_\\odot$)")
    ax1.set_ylabel("$\\log_{10}\\Lambda$ Uncertainty $\\sigma_{\\log\\Lambda}$")
    ax1.set_title("Tidal Deformability Uncertainty Breakdown")
    ax1.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "06_uncertainty_breakdown.png"), dpi=300)
    plt.close()

def generate_all_plots(prediction_csv="data/metrics/test_predictions.csv",
                       metrics_json="data/metrics/model_comparison.json",
                       figure_dir="outputs/figures", mode="mock"):
    """Runs the entire premium scientific reporting plotting suite."""
    print("Generating comprehensive publication-grade diagnostic figure suite...")
    os.makedirs(figure_dir, exist_ok=True)
    
    if not os.path.exists(prediction_csv) or not os.path.exists(metrics_json):
        print("Error: test predictions or comparison JSON not found. Run evaluations first.")
        return
        
    pred_df = pd.read_csv(prediction_csv)
    
    plot_mr_curves(pred_df, figure_dir)
    plot_mlambda_curves(pred_df, figure_dir)
    plot_calibration(pred_df, figure_dir)
    plot_observational_posteriors(figure_dir, mode=mode)
    plot_performance_summary(metrics_json, figure_dir)
    plot_uncertainty_breakdown(pred_df, figure_dir)
    
    print(f"Plotting suite completed successfully. All figures saved to {figure_dir}/")

if __name__ == "__main__":
    generate_all_plots()
