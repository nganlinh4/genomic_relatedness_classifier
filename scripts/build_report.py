import os
import sys
import json
import shutil
import subprocess
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

    # Class imbalance note (use any available distribution)
    any_mode_key = next(iter(consolidated['modes'])) if consolidated['modes'] else None
    vcd_any = consolidated['modes'][any_mode_key].get('val_class_distribution', {}) if any_mode_key else {}
    if vcd_any:
        total = sum(int(v) for v in vcd_any.values())
        maj = max(int(v) for v in vcd_any.values()) if vcd_any else 0
        share = (maj / total) if total else 0
        lines.append("## Note on class imbalance\n")
        if share >= 0.8:
            lines.append(f"Validation set is severely imbalanced (majority class ≈ {share*100:.1f}%). Metrics from 'zero' mode are likely biased toward the majority class. Prefer macro/weighted metrics and inspect per-class results. Consider 'weighted' or 'smote' modes for more balanced learning.\n")
        elif share >= 0.6:
            lines.append(f"Validation set is imbalanced (majority class ≈ {share*100:.1f}%). 'Zero' mode can be biased; review macro/weighted metrics and per-class metrics.\n")
        else:
            lines.append("Some class imbalance may still exist. Review macro/weighted metrics alongside accuracy.\n")
        lines.append("\n")

    # Dataset-level plots (EDA and Feature Importance)
    eda_plot = os.path.join('reports', dataset, f'kinship_distribution_{dataset}.png')
    fi_plot = os.path.join('reports', dataset, f'feature_importance_{dataset}.png')
    if os.path.exists(eda_plot) or os.path.exists(fi_plot):
        lines.append("## Dataset-level plots\n")
        if os.path.exists(eda_plot):
            rel_eda = os.path.relpath(eda_plot, start=out_dir)
            lines.append(f"<details><summary>Kinship Distribution</summary>\n\n![]({rel_eda})\n\n</details>")
        if os.path.exists(fi_plot):
            rel_fi = os.path.relpath(fi_plot, start=out_dir)
            lines.append(f"<details><summary>Top Feature Importances</summary>\n\n![]({rel_fi})\n\n</details>")
        lines.append("")

    # Executive summary: best model per mode and overall recommendation
    try:
        lines.append("## Executive summary\n")
        lines.append("| Mode | Best Model | Accuracy | F1 (weighted) | AUC (weighted) |")
        lines.append("|------|------------|----------|---------------|----------------|")
        summary = []
        for mode in modes:
            mb = consolidated['modes'].get(mode)
            if not mb:
                continue
            best_key = None
            best = None
            for mk, metrics in mb.get('models', {}).items():
                if metrics is None:
                    continue
                if best is None or (metrics.get('f1_weighted') or 0) > (best.get('f1_weighted') or 0):
                    best = metrics
                    best_key = mk
            if best_key:
                summary.append((mode, best_key, best))
                def fmt(x):
                    return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else ("N/A" if x is None else str(x))
                lines.append(f"| {mode} | {best_key} | {fmt(best.get('accuracy'))} | {fmt(best.get('f1_weighted'))} | {fmt(best.get('auc_weighted'))} |")

        # Simple recommendation: pick the highest F1-weighted; if it's 'zero' and imbalance is high, caution
        rec = None
        if summary:
            rec = max(summary, key=lambda t: (t[2].get('f1_weighted') or 0))
            mode_name, model_name, metrics = rec
            lines.append("")
            lines.append("### Recommendation\n")
            lines.append(f"Consider {model_name} in '{mode_name}' mode as the current best performer (F1_w={metrics.get('f1_weighted'):.4f}).")
            if vcd_any:
                if maj / total >= 0.6 and mode_name == 'zero':
                    lines.append("Given the observed imbalance, treat 'zero' as a baseline; prefer 'weighted' or 'smote' in longer training runs if performance is close.")
            lines.append("")
    except Exception:
        # Do not fail report generation on summary issues
        pass

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

        # Mode caveats
        if mode == 'zero':
            lines.append("> Caveat: 'zero' mode uses no rebalancing and may favor the majority class; treat results as a baseline.\n")
        elif mode == 'weighted':
            lines.append("> Note: 'weighted' mode applies class-weighted loss to mitigate imbalance; with very few epochs it may underperform transiently.\n")
        elif mode == 'smote':
            lines.append("> Note: 'smote' mode oversamples the training split; evaluation remains on the original (possibly imbalanced) validation set.\n")

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

    # Optionally convert Markdown to PDF using npx md-to-pdf if available (no output flags)
    try:
        npx_path = shutil.which('npx')
        if npx_path:
            pdf_path = os.path.join(out_dir, 'results.pdf')
            default_pdf = os.path.splitext(md_path)[0] + '.pdf'
            print("Converting Markdown to PDF via npx md-to-pdf (no output flag)...")
            # Always call without any output flag; most versions write <md>.pdf next to the input
            subprocess.run([npx_path, 'md-to-pdf', md_path], check=False)
            # If default PDF exists, rename/move to results.pdf
            if os.path.exists(default_pdf):
                try:
                    if default_pdf != pdf_path:
                        # Remove existing target if present, then move
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)
                        shutil.move(default_pdf, pdf_path)
                except Exception:
                    pass
            if os.path.exists(pdf_path):
                print(f"Wrote PDF report to {pdf_path}")
            else:
                print("md-to-pdf did not produce a PDF. You can run it manually: npx md-to-pdf <path-to-results.md>")
        else:
            print("npx not found; skipping PDF generation. Install Node.js and run 'npx md-to-pdf' to produce PDF.")
    except Exception as e:
        print(f"PDF generation step skipped due to error: {e}")

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
