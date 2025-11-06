import subprocess
import sys

if len(sys.argv) < 2:
    print("Usage: python run_all.py <dataset>")
    print("dataset: cM_1, cM_3, cM_6")
    sys.exit(1)

dataset = sys.argv[1]

# Step 1: Data Preparation
print(f"Running Step 1: Data Preparation for {dataset}")
subprocess.run(["./.venv/Scripts/python.exe", f"scripts/data_prep_{dataset}.py"], check=True)

# Step 2: EDA
print(f"Running Step 2: EDA for {dataset}")
subprocess.run(["./.venv/Scripts/python.exe", f"scripts/eda_{dataset}.py"], check=True)

# Step 3: Feature Selection
print(f"Running Step 3: Feature Selection for {dataset}")
subprocess.run(["./.venv/Scripts/python.exe", f"scripts/feature_selection_{dataset}.py"], check=True)

# Step 4: Train and Evaluate for different imbalance modes
modes = ["zero", "weighted", "smote"]
for mode in modes:
    print(f"Running Step 4-5: Train and Evaluate for {dataset} with {mode}")
    subprocess.run(["./.venv/Scripts/python.exe", "scripts/train_models.py", dataset, mode], check=True)
    subprocess.run(["./.venv/Scripts/python.exe", "scripts/evaluate_models.py", dataset, mode], check=True)

# Step 5: Generate final results table
print(f"Generating final results table for {dataset}")
subprocess.run(["./.venv/Scripts/python.exe", "scripts/generate_results_table.py", dataset], check=True)

print(f"All steps completed for {dataset}")