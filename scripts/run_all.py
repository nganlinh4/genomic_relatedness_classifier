import subprocess
import sys
import os
import argparse
from typing import Optional

python_exe = os.path.join(".venv", "Scripts" if sys.platform == "win32" else "bin", "python.exe" if sys.platform == "win32" else "python")


def run_for_dataset(ds: str, epochs: Optional[str], train_device: Optional[str], eval_device: Optional[str], special_epochs: Optional[str], prune: bool, only_rf: bool):
    print(f"Running Step 1: Data Preparation for {ds}")
    # Two scenarios: included (default) and noUN (drop UN)
    subprocess.run([python_exe, "scripts/data_prep.py", ds], check=True)
    subprocess.run([python_exe, "scripts/data_prep.py", ds, "--drop-un"], check=True)

    # Clean legacy plots so final report only references organized directories
    print(f"Cleaning legacy plots for {ds}")
    subprocess.run([python_exe, "scripts/cleanup_legacy_reports.py", ds], check=False)

    print(f"Running Step 2: EDA for {ds}")
    subprocess.run([python_exe, "scripts/eda.py", ds, "--scenario", "included"], check=True)
    subprocess.run([python_exe, "scripts/eda.py", ds, "--scenario", "noUN"], check=True)

    print(f"Running Step 3: Feature Selection for {ds}")
    subprocess.run([python_exe, "scripts/feature_selection.py", ds, "--scenario", "included"], check=True)
    subprocess.run([python_exe, "scripts/feature_selection.py", ds, "--scenario", "noUN"], check=True)

    modes = ["zero", "weighted", "smote", "overunder"]
    for scenario in ["included", "noUN"]:
        for mode in modes:
            print(f"Running Step 4-5: Train and Evaluate for {ds} ({scenario}) with {mode}")
            train_cmd = [python_exe, "scripts/train_models.py", ds, mode, "--scenario", scenario]
            if epochs:
                train_cmd += ["--epochs", str(epochs)]
            if train_device:
                train_cmd += ["--train-device", train_device]
            if special_epochs:
                train_cmd += ["--special-epochs", str(special_epochs)]
            if only_rf:
                train_cmd += ["--only-randomforest"]
            subprocess.run(train_cmd, check=True)

            eval_cmd = [python_exe, "scripts/evaluate_models.py", ds, mode, "--scenario", scenario]
            if eval_device:
                eval_cmd += ["--eval-device", eval_device]
            if only_rf:
                eval_cmd += ["--only-randomforest"]
            subprocess.run(eval_cmd, check=True)

    print(f"Building consolidated report for {ds}")
    report_cmd = [python_exe, "scripts/build_report.py", ds]
    if prune:
        report_cmd.append("--prune")
    subprocess.run(report_cmd, check=True)

    print(f"All steps completed for {ds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run full pipeline')
    parser.add_argument('dataset', type=str, help="cM_1, cM_3, cM_6 or 'all'")
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (default from TRAIN_EPOCHS env or 1)')
    parser.add_argument('--train-device', type=str, choices=['cpu','cuda'], default=None, help='Training device (must be cuda; cpu will abort in training scripts)')
    parser.add_argument('--eval-device', type=str, choices=['cpu','cuda'], default=None, help='Evaluation device (cuda recommended)')
    parser.add_argument('--special-epochs', type=int, default=None, help='Epoch override only for UN-included + oversampling modes (smote/overunder)')
    parser.add_argument('--prune', action='store_true', help='Delete per-mode evaluation JSONs after consolidation')
    parser.add_argument('--only-randomforest', action='store_true', help='Train/evaluate only Random Forest (skip MLP/CNN)')
    args = parser.parse_args()

    # Fallback to env for epochs if not provided
    epochs = args.epochs if args.epochs is not None else os.environ.get('TRAIN_EPOCHS', '1')
    train_device = args.train_device or os.environ.get('TRAIN_DEVICE')
    eval_device = args.eval_device or os.environ.get('EVAL_DEVICE')

    if args.dataset.lower() == 'all':
        for ds in ["cM_1", "cM_3", "cM_6"]:
            run_for_dataset(ds, epochs, train_device, eval_device, args.special_epochs, args.prune, args.only_randomforest)
    else:
        run_for_dataset(args.dataset, epochs, train_device, eval_device, args.special_epochs, args.prune, args.only_randomforest)