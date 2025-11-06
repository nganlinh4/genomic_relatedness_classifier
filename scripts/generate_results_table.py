import json
import sys

if len(sys.argv) < 2:
    print("Usage: python generate_results_table.py <dataset>")
    print("dataset: cM_1, cM_3, cM_6")
    sys.exit(1)

dataset = sys.argv[1]
modes = ["zero", "weighted", "smote"]

# Collect all results
all_results = []
for mode in modes:
    try:
        with open(f'data/processed/evaluation_results_{dataset}_{mode}.json', 'r') as f:
            data = json.load(f)
            for model_name, metrics in data['models'].items():
                all_results.append({
                    'Model': model_name,
                    'Mode': mode,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'F1-Score': f"{metrics['f1_score']:.4f}",
                    'AUC': f"{metrics['auc']:.4f}" if metrics['auc'] else 'N/A'
                })
    except FileNotFoundError:
        print(f"Warning: Results file for {dataset} {mode} not found.")

# Generate markdown table
table = f"# Final Results for {dataset}\n\n"
table += "| Model | Mode | Accuracy | F1-Score | AUC |\n"
table += "|-------|------|----------|----------|-----|\n"
for result in all_results:
    table += f"| {result['Model']} | {result['Mode']} | {result['Accuracy']} | {result['F1-Score']} | {result['AUC']} |\n"

# Write to file
with open(f'results_{dataset}.md', 'w') as f:
    f.write(table)

print(f"Results table saved to results_{dataset}.md")