import os
import sys
import json
from datetime import datetime


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_report.py <dataset>")
        print("dataset: cM_1, cM_3, cM_6")
        sys.exit(1)

    dataset = sys.argv[1]
    modes = ["zero", "weighted", "smote"]

    # Gather per-mode results
    per_mode = {}
    for mode in modes:
        path = os.path.join('data', 'processed', f'evaluation_results_{dataset}_{mode}.json')
        if os.path.exists(path):
            per_mode[mode] = load_json(path)
        else:
            print(f"Warning: Missing results JSON for {dataset} {mode}: {path}")

    if not per_mode:
        print("No results found to aggregate.")
        sys.exit(1)

    # Build consolidated structure
    any_result = next(iter(per_mode.values()))
    label_names = any_result.get('label_names', [])
    device = any_result.get('device')

    consolidated = {
        'dataset': dataset,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'device': device,
        'label_names': label_names,
        'modes': {}
    }

    for mode, res in per_mode.items():
        consolidated['modes'][mode] = {
            'val_samples': res.get('val_samples'),
            'val_class_distribution': res.get('val_class_distribution'),
            'models': res.get('models', {})
        }

    # Write consolidated JSON
    out_dir = os.path.join('reports', dataset)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(consolidated, f, indent=2)
    print(f"Wrote consolidated results to {json_path}")

    # Write Markdown summary
    md_path = os.path.join(out_dir, 'results.md')
    lines = []
    lines.append(f"# Results for {dataset}\n")
    lines.append(f"Generated: {consolidated['generated_at']}  ")
    lines.append(f"Device: {device}\n")
    if label_names:
        lines.append(f"Labels: {', '.join(label_names)}\n")

    for mode in modes:
        if mode not in consolidated['modes']:
            continue
        mode_block = consolidated['modes'][mode]
        lines.append(f"## Mode: {mode}\n")
        lines.append(f"Validation samples: {mode_block.get('val_samples')}  ")
        vcd = mode_block.get('val_class_distribution', {})
        if vcd:
            parts = [f"{k}={v}" for k, v in vcd.items()]
            lines.append(f"Class distribution: {', '.join(parts)}\n")

        # Table header
        lines.append("| Model | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |")
        lines.append("|-------|----------|---------------|------------|----------------|-------------|-------------|")
        for model_key, metrics in mode_block.get('models', {}).items():
            acc = metrics.get('accuracy')
            f1w = metrics.get('f1_weighted')
            f1m = metrics.get('f1_macro')
            aw = metrics.get('auc_weighted')
            am = metrics.get('auc_macro')
            ai = metrics.get('auc_micro')
            def fmt(x):
                return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else ("N/A" if x is None else str(x))
            lines.append(f"| {model_key} | {fmt(acc)} | {fmt(f1w)} | {fmt(f1m)} | {fmt(aw)} | {fmt(am)} | {fmt(ai)} |")

        # Confusion matrices
        lines.append("")
        for model_key, metrics in mode_block.get('models', {}).items():
            cm_path = metrics.get('confusion_matrix_path')
            if cm_path and os.path.exists(cm_path):
                rel_path = os.path.relpath(cm_path, start=out_dir)
                lines.append(f"<details><summary>Confusion Matrix: {model_key}</summary>")
                lines.append("")
                lines.append(f"![]({rel_path})")
                lines.append("")
                lines.append("</details>")
        lines.append("")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Wrote Markdown report to {md_path}")

    # Optionally delete separate per-mode JSONs to reduce clutter
    for mode in list(per_mode.keys()):
        path = os.path.join('data', 'processed', f'evaluation_results_{dataset}_{mode}.json')
        try:
            os.remove(path)
            print(f"Removed {path}")
        except OSError:
            pass


if __name__ == '__main__':
    main()
