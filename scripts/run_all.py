import subprocess
import sys
import os
import argparse
from typing import Optional


def run_for_dataset(ds: str, epochs: Optional[str], train_device: Optional[str], eval_device: Optional[str]):
    print(f"Running Step 1: Data Preparation for {ds}")
    subprocess.run(["./.venv/Scripts/python.exe", "scripts/data_prep.py", ds], check=True)

    print(f"Running Step 2: EDA for {ds}")
    subprocess.run(["./.venv/Scripts/python.exe", "scripts/eda.py", ds], check=True)

    print(f"Running Step 3: Feature Selection for {ds}")
    subprocess.run(["./.venv/Scripts/python.exe", "scripts/feature_selection.py", ds], check=True)

    modes = ["zero", "weighted", "smote"]
    for mode in modes:
        print(f"Running Step 4-5: Train and Evaluate for {ds} with {mode}")
        train_cmd = ["./.venv/Scripts/python.exe", "scripts/train_models.py", ds, mode]
        if epochs:
            train_cmd += ["--epochs", str(epochs)]
        if train_device:
            train_cmd += ["--train-device", train_device]
        subprocess.run(train_cmd, check=True)

        eval_cmd = ["./.venv/Scripts/python.exe", "scripts/evaluate_models.py", ds, mode]
        if eval_device:
            eval_cmd += ["--eval-device", eval_device]
        subprocess.run(eval_cmd, check=True)

    print(f"Building consolidated report for {ds}")
    subprocess.run(["./.venv/Scripts/python.exe", "scripts/build_report.py", ds], check=True)

    print(f"All steps completed for {ds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run full pipeline')
    parser.add_argument('dataset', type=str, help="cM_1, cM_3, cM_6 or 'all'")
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (default from TRAIN_EPOCHS env or 1)')
    parser.add_argument('--train-device', type=str, choices=['cpu','cuda'], default=None, help='Training device')
    parser.add_argument('--eval-device', type=str, choices=['cpu','cuda'], default=None, help='Evaluation device')
    args = parser.parse_args()

    # Fallback to env for epochs if not provided
    epochs = args.epochs if args.epochs is not None else os.environ.get('TRAIN_EPOCHS', '1')
    train_device = args.train_device or os.environ.get('TRAIN_DEVICE')
    eval_device = args.eval_device or os.environ.get('EVAL_DEVICE')

    if args.dataset.lower() == 'all':
        for ds in ["cM_1", "cM_3", "cM_6"]:
            run_for_dataset(ds, epochs, train_device, eval_device)
    else:
        run_for_dataset(args.dataset, epochs, train_device, eval_device)