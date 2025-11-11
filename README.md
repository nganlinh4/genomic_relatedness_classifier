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
  - `results.md`, `results_KR.md` — Markdown reports (EN/KR)
  - `results.pdf`, `results_KR.pdf` — Optional PDFs via `npx md-to-pdf`
  - `assets/<scenario>/feature_importance_<dataset>_<scenario>.svg|png`
  - `assets/<scenario>/kinship_distribution_<dataset>_<scenario>.svg|png`
  - `plots/confusion/<scenario>/<mode>/confusion_matrix_*.svg|png`

- `scripts/` — Orchestration and processing
  - `run_all.py` — End-to-end pipeline runner
  - `data_prep.py`, `eda.py`, `feature_selection.py` — Preprocessing steps
  - `train_models.py`, `evaluate_models.py` — Train and evaluate per mode
  - `build_report.py` — Consolidate per-mode JSONs into a single report

- `docs/` — Documentation (`plan.md`)

- `.venv/` — Python virtual environment (created with uv)

Note: A Korean version of this README is available at `README_KR.md`.

## Setup

Prereqs:
- Python 3.11 (repo uses a 3.11 .venv)
- NVIDIA GPU with a compatible CUDA driver

Steps (Windows PowerShell):
1) Install uv (project-local environment manager)
   
   ```powershell
   pip install uv
   ```

2) Create and activate the virtual environment at the repo root

   ```powershell
   uv venv
   .\.venv\Scripts\activate
   ```

3) Install project dependencies from pyproject/uv.lock (fast, reproducible)

   ```powershell
   uv lock ; uv sync
   ```

4) Install PyTorch with CUDA (managed separately to avoid CPU wheels)
   
   Use the official index for your CUDA toolchain; example below for CUDA 12.1:

   ```powershell
   uv pip install --index-url https://download.pytorch.org/whl/cu121 `
     torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
   ```

5) Optional: for PDF export of reports, install Node.js (enables `npx md-to-pdf`)

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
- `--special-epochs <int>` — override epochs only for UN-included + oversampling (smote/overunder)
- `--only-randomforest` — train/evaluate only Random Forest (skip MLP/CNN)
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
- `results.md` / `results_KR.md` — human-readable summaries with metric tables and plots
- Scenario plots: `assets/<scenario>/feature_importance_*.svg|png` and `kinship_distribution_*.svg|png`
- Confusion matrices: `plots/confusion/<scenario>/<mode>/confusion_matrix_*.svg|png`

### Scenarios and modes

- Scenarios:
  - `included`: keep rows labeled `UN` to measure impact of the ambiguous class
  - `noUN`: drop `UN` before split/training
- Modes (imbalance strategies):
  - `zero`: no rebalancing (baseline; may favor majority)
  - `weighted`: class-weighted loss only
  - `smote`: oversample the training split via SMOTE
  - `overunder`: SMOTE then ENN/Tomek cleanup (over + under)

Reports embed scenario plots in a two-column layout and prefer SVG for crisp text; confusion matrices are included per mode/model.

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
- AUC is computed robustly per class (OvR) with safe fallbacks (never N/A); prefer macro/weighted AUC for imbalanced settings.

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

## Reproducibility (pyproject.toml + uv.lock)

- This repo includes a `pyproject.toml` with the exact Python version range and core deps, plus a `uv.lock` generated by `uv lock` for reproducible resolution.
- PyTorch GPU wheels are not listed in `pyproject.toml` on purpose: the default PyPI index resolves to CPU wheels. Install CUDA-enabled wheels via the official index (see Setup step 4).
- `.venv/` is git-ignored; never commit virtual environments.

Typical workflow to refresh your environment:

```powershell
# Sync core Python deps from lockfile
uv sync

# Ensure CUDA-enabled PyTorch (adjust versions/index for your CUDA)
uv pip install --index-url https://download.pytorch.org/whl/cu121 `
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121

# Verify GPU availability
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```