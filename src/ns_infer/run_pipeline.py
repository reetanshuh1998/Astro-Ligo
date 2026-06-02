import os
import argparse
import time
from ns_infer.data.dataset_builder import build_dataset
from ns_infer.data.observational_data import ObservationalDataManager
from ns_infer.models.train_xgboost import train_xgboost
from ns_infer.models.train_dnn import train_dnn
from ns_infer.models.train_bnn import train_bnn
from ns_infer.evaluation.compare_models import evaluate_models
from ns_infer.evaluation.plot_results import generate_all_plots

def main():
    """
    Main orchestrator CLI for the neutron star machine learning parameter estimation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="ns_infer: End-to-end physics-informed neutron star ML parameter estimation."
    )
    parser.add_argument(
        "--mode", type=str, choices=["mock", "download"], default="mock",
        help="Observational data mode: 'mock' (local distributions) or 'download' (fetch official releases)."
    )
    parser.add_argument(
        "--num-eos", type=int, default=100,
        help="Number of distinct piecewise polytropic core EOS models to sample for the dataset."
    )
    parser.add_argument(
        "--num-stars", type=int, default=20,
        help="Number of stellar configurations to generate per valid EOS model (dynamic mass balancing)."
    )
    parser.add_argument(
        "--epochs", type=int, default=80,
        help="Number of training epochs for deep neural network models."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size for PyTorch models."
    )
    parser.add_argument(
        "--lr", type=float, default=0.003,
        help="Learning rate for optimization loops."
    )
    parser.add_argument(
        "--skip-gen", action="store_true",
        help="Skip synthetic data generation if dataset file already exists."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("      NS_INFER: NEUTRON STAR PARAMETER ESTIMATION ML PIPELINE")
    print("=" * 70)
    print(f"  Execution Mode:         {args.mode.upper()}")
    print(f"  Target core EOS models: {args.num_eos}")
    print(f"  Stars per EOS:          {args.num_stars}")
    print(f"  Training Epochs:        {args.epochs}")
    print(f"  Seed:                   {args.seed}")
    print("-" * 70)
    
    start_time = time.time()
    
    # Define directories
    processed_dir = "data/processed"
    model_dir = "data/models"
    metrics_dir = "data/metrics"
    figure_dir = "outputs/figures"
    
    dataset_path = os.path.join(processed_dir, "ns_dataset.csv")
    
    # -----------------------------------------------------------------
    # STEP 1: Synthetic Data Generation
    # -----------------------------------------------------------------
    if args.skip_gen and os.path.exists(dataset_path):
        print(f"Skipping synthetic data generation. Using existing file: {dataset_path}")
    else:
        print("\n[STEP 1] Generating physical synthetic training database...")
        build_dataset(
            output_path=dataset_path,
            num_eos=args.num_eos,
            num_stars_per_eos=args.num_stars,
            seed=args.seed
        )
        
    # -----------------------------------------------------------------
    # STEP 2: Observational Posterior Manager
    # -----------------------------------------------------------------
    print("\n[STEP 2] Fetching/Managing observational posterior catalogs...")
    obs_manager = ObservationalDataManager(mode=args.mode)
    obs_manager.fetch_pe_samples()
    
    # -----------------------------------------------------------------
    # STEP 3: Machine Learning Model Training
    # -----------------------------------------------------------------
    print("\n[STEP 3] Commencing model training scripts...")
    # A. XGBoost Baseline
    train_xgboost(dataset_path=dataset_path, model_dir=model_dir)
    
    # B. Deterministic Deep Neural Network (Dense MLP)
    train_dnn(
        dataset_path=dataset_path,
        model_dir=model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # C. Bayesian-like Deep Ensemble + MC Dropout
    train_bnn(
        dataset_path=dataset_path,
        model_dir=model_dir,
        num_members=5,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # -----------------------------------------------------------------
    # STEP 4: Joint Evaluation & Speed Benchmarks
    # -----------------------------------------------------------------
    print("\n[STEP 4] Executing model evaluations and calibration checks...")
    evaluate_models(model_dir=model_dir, output_dir=metrics_dir)
    
    # -----------------------------------------------------------------
    # STEP 5: High-Quality Diagnostic Plots Generation
    # -----------------------------------------------------------------
    print("\n[STEP 5] Generating premium publication-grade reporting plots...")
    generate_all_plots(
        prediction_csv=os.path.join(metrics_dir, "test_predictions.csv"),
        metrics_json=os.path.join(metrics_dir, "model_comparison.json"),
        figure_dir=figure_dir,
        mode=args.mode
    )
    
    total_duration = time.time() - start_time
    print("=" * 70)
    print("  NEUTRON STAR ML INFERENCE PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  Total pipeline execution time: {total_duration:.2f} seconds.")
    print(f"  All report figures saved to:   {figure_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
