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

- `models/<dataset>/<mode>/` — Trained model weights (`mlp.pth`, `cnn.pth`) — generated on run (git-ignored)

- `reports/<dataset>/` — Consolidated results and plots — generated on run (git-ignored)
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

### Project structure (Mermaid)

```mermaid
graph TD
  A[Repo root]
  A --> B[data/]
  B --> B1[raw/ — input CSVs and merged_info.out]
  B --> B2[processed/ — merged_{dataset}.csv, top_features_{dataset}.pkl, scaler_{dataset}.pkl, evaluation_results_*.json]

  A --> C[scripts/ — pipeline]
  subgraph SCRIPTS [scripts]
    C1[data_prep.py — parse merged_info.out + join labeled pairs → merged_{dataset}.csv]
    C2[eda.py — kinship distribution plot → reports/{dataset}/kinship_distribution_{dataset}.png]
    C3[feature_selection.py — RF importance; save top_features & scaler; plot]
    C4[train_models.py — train MLP/1D-CNN per imbalance mode]
    C5[evaluate_models.py — evaluate; robust OvR AUC; save confusion matrices + per-mode JSON]
    C6[build_report.py — aggregate → reports/{dataset}/results(.json/.md/.pdf) + results_KR(.md/.pdf)]
    C7[run_all.py — orchestrate full pipeline for one dataset or all]
  end

  A --> D[models/ — model weights per dataset/mode (git-ignored)]
  A --> E[reports/ — consolidated results and plots (git-ignored)]
  E --> E1[{dataset}/results(.md/.pdf), results_KR(.md/.pdf)]
  E --> E2[{dataset}/feature_importance_{dataset}.png, kinship_distribution_{dataset}.png]
  E --> E3[{dataset}/{mode}/confusion_matrix_*.png]
  A --> F[docs/plan.md, docs/plan_KR.md]
```

Note: A Korean version of this README is available at `README_KR.md`.

## Setup

Prereqs:
- Python 3.10–3.12
- NVIDIA GPU with a compatible CUDA driver

Steps (Windows PowerShell):
1) Install uv (optional, used below to manage the venv)
  - `pip install uv`
2) Create and activate a virtual environment
  - `uv venv`
  - `.\.venv\Scripts\activate`
3) Install PyTorch with CUDA (follow the official selector for your CUDA/driver)
  - Example (CUDA 12.1): `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
4) Install Python dependencies
  - `pip install pandas scikit-learn imbalanced-learn matplotlib seaborn`
5) Optional: for PDF export of reports, install Node.js (for `npx md-to-pdf`)

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
- `--train-device cuda` — training device (CUDA is required; CPU is disabled by policy)
- `--eval-device cuda` — evaluation device (CUDA is required; CPU is disabled by policy)

Examples:
```
# 50 epochs on GPU
./.venv/Scripts/python.exe scripts/run_all.py cM_1 --epochs 50 --train-device cuda --eval-device cuda

# Use environment vars instead (PowerShell):
$env:TRAIN_EPOCHS=25; $env:TRAIN_DEVICE="cuda"; $env:EVAL_DEVICE="cuda"; \
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
- Generated artifacts in `models/` and `reports/` are ignored by git to keep the repo light.

### Interpreting results and class imbalance

- Many datasets here are highly imbalanced (e.g., majority class > 80%).
- 'zero' mode performs no rebalancing and can appear strong on accuracy while failing minority classes — treat as a baseline only.
- 'weighted' mode (class-weighted loss) and 'smote' mode (oversampling on train split) address imbalance differently; review macro/weighted F1 and per-class metrics.
- AUC is computed robustly per class (OvR) with safe fallbacks, so it will always be numeric; prefer macro/weighted AUC for imbalanced settings.

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