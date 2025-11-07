import subprocess
import sys
import os
import argparse
from typing import Optional


def run_for_dataset(ds: str, epochs: Optional[str], train_device: Optional[str], eval_device: Optional[str], special_epochs: Optional[str], prune: bool):
    print(f"Running Step 1: Data Preparation for {ds}")
    # Two scenarios: included (default) and noUN (drop UN)
    subprocess.run(["./.venv/Scripts/python.exe", "scripts/data_prep.py", ds], check=True)
    subprocess.run(["./.venv/Scripts/python.exe", "scripts/data_prep.py", ds, "--drop-un"], check=True)

    print(f"Running Step 2: EDA for {ds}")
    subprocess.run(["./.venv/Scripts/python.exe", "scripts/eda.py", ds, "--scenario", "included"], check=True)
    subprocess.run(["./.venv/Scripts/python.exe", "scripts/eda.py", ds, "--scenario", "noUN"], check=True)

    print(f"Running Step 3: Feature Selection for {ds}")
    subprocess.run(["./.venv/Scripts/python.exe", "scripts/feature_selection.py", ds, "--scenario", "included"], check=True)
    subprocess.run(["./.venv/Scripts/python.exe", "scripts/feature_selection.py", ds, "--scenario", "noUN"], check=True)

    modes = ["zero", "weighted", "smote", "overunder"]
    for scenario in ["included", "noUN"]:
        for mode in modes:
            print(f"Running Step 4-5: Train and Evaluate for {ds} ({scenario}) with {mode}")
            train_cmd = ["./.venv/Scripts/python.exe", "scripts/train_models.py", ds, mode, "--scenario", scenario]
            if epochs:
                train_cmd += ["--epochs", str(epochs)]
            if train_device:
                train_cmd += ["--train-device", train_device]
            if special_epochs:
                train_cmd += ["--special-epochs", str(special_epochs)]
            subprocess.run(train_cmd, check=True)

            eval_cmd = ["./.venv/Scripts/python.exe", "scripts/evaluate_models.py", ds, mode, "--scenario", scenario]
            if eval_device:
                eval_cmd += ["--eval-device", eval_device]
            subprocess.run(eval_cmd, check=True)

    print(f"Building consolidated report for {ds}")
    report_cmd = ["./.venv/Scripts/python.exe", "scripts/build_report.py", ds]
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
    args = parser.parse_args()

    # Fallback to env for epochs if not provided
    epochs = args.epochs if args.epochs is not None else os.environ.get('TRAIN_EPOCHS', '1')
    train_device = args.train_device or os.environ.get('TRAIN_DEVICE')
    eval_device = args.eval_device or os.environ.get('EVAL_DEVICE')

    if args.dataset.lower() == 'all':
        for ds in ["cM_1", "cM_3", "cM_6"]:
            run_for_dataset(ds, epochs, train_device, eval_device, args.special_epochs, args.prune)
    else:
        run_for_dataset(args.dataset, epochs, train_device, eval_device, args.special_epochs, args.prune)