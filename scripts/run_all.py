import subprocess
import sys
import os
import argparse
from typing import Optional

python_exe = os.path.join(".venv", "Scripts" if sys.platform == "win32" else "bin", "python.exe" if sys.platform == "win32" else "python")

def cleanup_legacy_plots(dataset: str):
    """Inline replacement for cleanup_legacy_reports.py"""
    base = os.path.join('reports', dataset)
    if not os.path.isdir(base): return
    for fname in os.listdir(base):
        if fname.lower().endswith('.png') and (
            fname.startswith('kinship_distribution_') or
            fname.startswith('feature_importance_') or
            fname.startswith('confusion_matrix_')
        ):
            try:
                os.remove(os.path.join(base, fname))
            except OSError: pass

def run_for_dataset(ds: str, epochs: Optional[str], train_device: Optional[str], eval_device: Optional[str], 
                   special_epochs: Optional[str], prune: bool, only_rf: bool, data_type: str):
    
    print(f"=== Pipeline Start: {ds} (Type: {data_type}) ===")

    # Step 1: Data Prep
    print(f"Step 1: Data Preparation")
    subprocess.run([python_exe, "scripts/data_prep.py", ds, "--type", data_type], check=True)
    subprocess.run([python_exe, "scripts/data_prep.py", ds, "--type", data_type, "--drop-un"], check=True)

    cleanup_legacy_plots(ds)

    # Step 2: EDA (Moved to analysis folder)
    print(f"Step 2: EDA")
    # Note: Ensure scripts/analysis/eda.py exists
    subprocess.run([python_exe, "scripts/analysis/eda.py", ds, "--scenario", "included"], check=True)
    subprocess.run([python_exe, "scripts/analysis/eda.py", ds, "--scenario", "noUN"], check=True)

    # Step 3: Feature Selection (Kept in root as it generates pickle files needed for training)
    print(f"Step 3: Feature Selection")
    subprocess.run([python_exe, "scripts/feature_selection.py", ds, "--scenario", "included"], check=True)
    subprocess.run([python_exe, "scripts/feature_selection.py", ds, "--scenario", "noUN"], check=True)

    # Step 4 & 5: Train and Evaluate
    modes = ["zero", "weighted", "smote", "overunder"]
    for scenario in ["included", "noUN"]:
        for mode in modes:
            print(f"Step 4/5: Train/Eval {ds} [{scenario} - {mode}]")
            
            # Build Train Command
            train_cmd = [python_exe, "scripts/train_models.py", ds, mode, "--scenario", scenario]
            if epochs: train_cmd += ["--epochs", str(epochs)]
            if train_device: train_cmd += ["--train-device", train_device]
            if special_epochs: train_cmd += ["--special-epochs", str(special_epochs)]
            if only_rf: train_cmd += ["--only-randomforest"]
            subprocess.run(train_cmd, check=True)

            # Build Eval Command
            eval_cmd = [python_exe, "scripts/evaluate_models.py", ds, mode, "--scenario", scenario]
            if eval_device: eval_cmd += ["--eval-device", eval_device]
            if only_rf: eval_cmd += ["--only-randomforest"]
            subprocess.run(eval_cmd, check=True)

    # Step 6: Reporting
    print(f"Step 6: Building Report")
    report_cmd = [python_exe, "scripts/build_report.py", ds]
    if prune: report_cmd.append("--prune")
    subprocess.run(report_cmd, check=True)

    print(f"=== Pipeline Complete: {ds} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run full genomic classification pipeline')
    parser.add_argument('dataset', type=str, help="Dataset ID (e.g., cM_1) or 'all'")
    parser.add_argument('--type', type=str, choices=['standard', 'percentile'], default='standard', help="Data source type")
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--train-device', type=str, choices=['cpu','cuda'], default=None)
    parser.add_argument('--eval-device', type=str, choices=['cpu','cuda'], default=None)
    parser.add_argument('--special-epochs', type=int, default=None)
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--only-randomforest', action='store_true')
    args = parser.parse_args()

    # Env var fallbacks
    epochs = args.epochs if args.epochs is not None else os.environ.get('TRAIN_EPOCHS', '1')
    train_dev = args.train_device or os.environ.get('TRAIN_DEVICE')
    eval_dev = args.eval_device or os.environ.get('EVAL_DEVICE')

    datasets = ["cM_1", "cM_3", "cM_6", "cM_10"] if args.dataset.lower() == 'all' else [args.dataset]

    for ds in datasets:
        run_for_dataset(ds, epochs, train_dev, eval_dev, args.special_epochs, args.prune, args.only_randomforest, args.type)
