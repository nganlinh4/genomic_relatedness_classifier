# Genomic Relatedness Classifier

This project builds and evaluates models to predict kinship using IBD metrics and distributional statistics.

## Project Structure

- `data/raw/` — Raw input data files
  - `merged_info.out` — Unzipped distributional statistics file
  - `merged_info.out.zip` — Original zipped file
  - `model_input_with_kinship_filtered_cM_*.csv` — Kinship data for cM thresholds (1, 3, 6)

- `data/processed/` — Processed datasets and intermediate artifacts
  - `merged_cM_*.csv` — Merged datasets ready for modeling
  - `top_features_*.pkl`, `scaler_*.pkl` — Feature selection outputs
  - `evaluation_results_*_*.json` — Per-mode eval JSONs (temporary; consolidated after build)

- `models/<dataset>/<mode>/` — Trained model weights (`mlp.pth`, `cnn.pth`)

- `reports/<dataset>/` — Consolidated results and plots
  - `results.json` — Machine-readable consolidated results
  - `results.md` — Human-readable summary with links to confusion matrices
  - `feature_importance_<dataset>.png`, `kinship_distribution_<dataset>.png`
  - `<mode>/confusion_matrix_*.png` — Per-mode confusion matrix images

- `scripts/` — Orchestration and processing
  - `run_all.py` — End-to-end pipeline runner
  - `data_prep.py`, `eda.py`, `feature_selection.py` — Preprocessing steps
  - `train_models.py`, `evaluate_models.py` — Train and evaluate per mode
  - `build_report.py` — Consolidate per-mode JSONs into a single report

- `docs/` — Documentation (`plan.md`)

- `.venv/` — Python virtual environment (created with uv)

## Setup

1. Install uv: `pip install uv`
2. Create venv: `uv venv`
3. Activate: `.venv\Scripts\activate` (Windows)
4. Install deps: `uv pip install pandas`

## Usage

End-to-end run for a dataset (uses the repo's .venv):
```
./.venv/Scripts/python.exe scripts/run_all.py cM_1
```

Run for all datasets:
```
./.venv/Scripts/python.exe scripts/run_all.py all
```

Optional flags for control (CLI overrides env vars):
- `--epochs <int>` — number of training epochs (default 1 or `$env:TRAIN_EPOCHS`)
- `--train-device cpu|cuda` — training device (default cuda; CUDA is required)
- `--eval-device cpu|cuda` — evaluation device (default cuda; CUDA is required)

Examples:
```
# 50 epochs on CPU for training, evaluate on CPU
./.venv/Scripts/python.exe scripts/run_all.py cM_1 --epochs 50 --train-device cpu --eval-device cpu

# Use environment vars instead (PowerShell):
$env:TRAIN_EPOCHS=25; $env:TRAIN_DEVICE="cpu"; $env:EVAL_DEVICE="cpu"; \
  ./.venv/Scripts/python.exe scripts/run_all.py cM_3
```

Outputs are consolidated under `reports/<dataset>/`:
- `results.json` — comprehensive machine-readable results
- `results.md` — human-readable summary with metric tables and confusion matrices
- `feature_importance_<dataset>.png` — top feature importances
- `kinship_distribution_<dataset>.png` — target distribution

## Notes

- Defaults prioritize fast prototyping with strict GPU usage: 1 epoch, CUDA-only for train/eval.
- Model weights are stored under `models/<dataset>/<mode>/mlp.pth` and `cnn.pth`.
- Per-mode evaluation JSONs are cleaned up after consolidation; the canonical outputs are under `reports/<dataset>/`.
- The previous dataset-specific scripts and legacy generators were removed in favor of the generalized pipeline.

### Git LFS (optional)

This repo ignores large artifacts by default (`models/**`, image files under `reports/**`, and `data/raw/**`). If you prefer tracking these in Git with Large File Storage (LFS), enable LFS and add patterns like below to `.gitattributes`:

```
models/** filter=lfs diff=lfs merge=lfs -text
reports/**/*.png filter=lfs diff=lfs merge=lfs -text
data/raw/** filter=lfs diff=lfs merge=lfs -text
```

PowerShell (Windows):
```
git lfs install
git add .gitattributes
git commit -m "Enable LFS for models/images/raw data"
```