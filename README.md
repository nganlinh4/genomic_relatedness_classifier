# Genomic Relatedness Classifier

This project builds and evaluates models to predict kinship using IBD metrics and distributional statistics.

## Project Structure

- `data/raw/` - Raw input data files
  - `merged_info.out` - Unzipped distributional statistics file
  - `merged_info.out.zip` - Original zipped file
  - `model_input_with_kinship_filtered_cM_*.csv` - Kinship data for different cM thresholds (1, 3, 6)

- `data/processed/` - Processed datasets
  - `merged_cM_*.csv` - Merged datasets ready for modeling

- `scripts/` - Python scripts for data processing
  - `data_prep_cM_*.py` - Data preparation scripts for each cM dataset
  - `verify.py` - Verification script

- `docs/` - Documentation
  - `plan.md` - Project plan and steps

- `.venv/` - Python virtual environment (created with uv)

## Setup

1. Install uv: `pip install uv`
2. Create venv: `uv venv`
3. Activate: `.venv\Scripts\activate` (Windows)
4. Install deps: `uv pip install pandas`

## Usage

Run data preparation for cM_1:
```
python scripts/data_prep_cM_1.py
```

Verify the output:
```
python scripts/verify.py
```

## Next Steps

- Repeat data preparation for cM_3 and cM_6 datasets
- Perform EDA
- Feature selection and preprocessing
- Model building and evaluation