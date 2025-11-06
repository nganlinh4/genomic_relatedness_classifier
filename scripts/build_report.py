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

    out_dir = os.path.join('reports', dataset)
    os.makedirs(out_dir, exist_ok=True)

    if not per_mode:
        # Fallback: use existing consolidated results if present
        json_path_existing = os.path.join(out_dir, 'results.json')
        if os.path.exists(json_path_existing):
            consolidated = load_json(json_path_existing)
            label_names = consolidated.get('label_names', [])
            device = consolidated.get('device')
            print("Using existing consolidated results.json to regenerate Markdown (and KR/PDF variants).")
        else:
            print("No results found to aggregate and no existing consolidated results.json present.")
            sys.exit(1)
    else:
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
        json_path = os.path.join(out_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(consolidated, f, indent=2)
        print(f"Wrote consolidated results to {json_path}")

    # Helpers to format sections (EN and KR)
    def build_lines_en():
        lines = []
        lines.append(f"# Results for {dataset}\n")
        lines.append(f"Generated: {consolidated['generated_at']}  ")
        lines.append(f"Device: {device}\n")
        if label_names:
            lines.append(f"Labels: {', '.join(label_names)}\n")

        # Executive summary first
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

            # Simple recommendation
            rec = None
            any_mode_key = next(iter(consolidated['modes'])) if consolidated['modes'] else None
            vcd_any = consolidated['modes'][any_mode_key].get('val_class_distribution', {}) if any_mode_key else {}
            if summary:
                rec = max(summary, key=lambda t: (t[2].get('f1_weighted') or 0))
                mode_name, model_name, metrics = rec
                lines.append("")
                lines.append("### Recommendation\n")
                lines.append(f"Consider {model_name} in '{mode_name}' mode as the current best performer (F1_w={metrics.get('f1_weighted'):.4f}).")
                if vcd_any:
                    total = sum(int(v) for v in vcd_any.values())
                    maj = max(int(v) for v in vcd_any.values()) if vcd_any else 0
                    if total > 0 and maj / total >= 0.6 and mode_name == 'zero':
                        lines.append("Given the observed imbalance, treat 'zero' as a baseline; prefer 'weighted' or 'smote' in longer training runs if performance is close.")
                lines.append("")
        except Exception:
            pass

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

        # Dataset-level plots (EDA and Feature Importance) - not collapsed
        eda_plot = os.path.join('reports', dataset, f'kinship_distribution_{dataset}.png')
        fi_plot = os.path.join('reports', dataset, f'feature_importance_{dataset}.png')
        if os.path.exists(eda_plot) or os.path.exists(fi_plot):
            lines.append("## Dataset-level plots\n")
            items = []
            if os.path.exists(eda_plot):
                rel_eda = os.path.relpath(eda_plot, start=out_dir)
                items.append(("Kinship Distribution", rel_eda))
            if os.path.exists(fi_plot):
                rel_fi = os.path.relpath(fi_plot, start=out_dir)
                items.append(("Top Feature Importances", rel_fi))
            if items:
                lines.append('<div style="display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap;">')
                for title_txt, img_path in items:
                    lines.append('<figure style="flex:1; min-width:320px; margin:0;">')
                    lines.append(f'<img src="{img_path}" style="width:100%; max-width:100%;" />')
                    lines.append(f'<figcaption style="text-align:center; font-size: 13px; margin-top:6px;">{title_txt}</figcaption>')
                    lines.append('</figure>')
                lines.append('</div>')
                lines.append("")

        # (Secondary) executive summary repeated after plots (kept for structure; minimal duplication OK)
        try:
            lines.append("## Executive summary (detailed)\n")
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

            # Confusion matrices - present in 2-column layout to save space
            cms = []
            for model_key, metrics in mode_block.get('models', {}).items():
                cm_path = metrics.get('confusion_matrix_path')
                if cm_path and os.path.exists(cm_path):
                    rel_path = os.path.relpath(cm_path, start=out_dir)
                    cms.append((model_key, rel_path))
            if cms:
                lines.append("")
                # Group into rows of 2; if last row has a single item, render it smaller and add an empty placeholder
                i = 0
                while i < len(cms):
                    remaining = len(cms) - i
                    if remaining >= 2:
                        row = cms[i:i+2]
                        lines.append('<div style="display:flex; gap:12px; align-items:flex-start;">')
                        for model_key, rel_path in row:
                            lines.append('<figure style="flex:1; margin:0;">')
                            lines.append(f'<img src="{rel_path}" style="width:100%; max-width:100%;" />')
                            lines.append(f'<figcaption style="text-align:center; font-size: 13px; margin-top:6px;">Confusion Matrix: {model_key}</figcaption>')
                            lines.append('</figure>')
                        lines.append('</div>')
                        lines.append("")
                        i += 2
                    else:
                        # One remaining: render small, add empty placeholder cell
                        model_key, rel_path = cms[i]
                        lines.append('<div style="display:flex; gap:12px; align-items:flex-start;">')
                        # Small figure (~70% width of its cell, centered)
                        lines.append('<figure style="flex:1; margin:0; display:flex; justify-content:center;">')
                        lines.append('<div style="width:70%;">')
                        lines.append(f'<img src="{rel_path}" style="width:100%; max-width:100%;" />')
                        lines.append(f'<div style="text-align:center; font-size: 13px; margin-top:6px;">Confusion Matrix: {model_key}</div>')
                        lines.append('</div>')
                        lines.append('</figure>')
                        # Empty placeholder cell to keep 2x2 grid feel
                        lines.append('<figure style="flex:1; margin:0;">&nbsp;</figure>')
                        lines.append('</div>')
                        lines.append("")
                        i += 1
                lines.append("")
        return lines

    def build_lines_kr():
        lines = []
        lines.append(f"# {dataset} 결과\n")
        lines.append(f"생성 시각: {consolidated['generated_at']}  ")
        lines.append(f"디바이스: {device}\n")
        if label_names:
            lines.append(f"레이블: {', '.join(label_names)}\n")

        # 요약 먼저 배치
        try:
            lines.append("## 요약\n")
            lines.append("| 모드 | 최고 모델 | 정확도 | F1 (가중치) | AUC (가중치) |")
            lines.append("|------|-----------|--------|------------|--------------|")
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

            if summary:
                rec = max(summary, key=lambda t: (t[2].get('f1_weighted') or 0))
                mode_name, model_name, metrics = rec
                lines.append("")
                lines.append("### 권장 사항\n")
                lines.append(f"현재 최고 성능 모델은 '{mode_name}' 모드의 {model_name} 입니다 (F1(가중치)={metrics.get('f1_weighted'):.4f}).")
                any_mode_key = next(iter(consolidated['modes'])) if consolidated['modes'] else None
                vcd_any = consolidated['modes'][any_mode_key].get('val_class_distribution', {}) if any_mode_key else {}
                if vcd_any:
                    total = sum(int(v) for v in vcd_any.values())
                    maj = max(int(v) for v in vcd_any.values()) if vcd_any else 0
                    if total > 0 and maj / total >= 0.6 and mode_name == 'zero':
                        lines.append("불균형을 고려하여 'zero'는 기준선으로만 사용하고, 장기 학습 시 'weighted' 또는 'smote'를 우선 고려하세요.")
                lines.append("")
        except Exception:
            pass

        any_mode_key = next(iter(consolidated['modes'])) if consolidated['modes'] else None
        vcd_any = consolidated['modes'][any_mode_key].get('val_class_distribution', {}) if any_mode_key else {}
        if vcd_any:
            total = sum(int(v) for v in vcd_any.values())
            maj = max(int(v) for v in vcd_any.values()) if vcd_any else 0
            share = (maj / total) if total else 0
            lines.append("## 클래스 불균형 참고\n")
            if share >= 0.8:
                lines.append(f"검증 세트의 불균형이 매우 큽니다 (최대 클래스 ≈ {share*100:.1f}%). 'zero' 모드는 다수 클래스에 치우칠 수 있으므로, 매크로/가중치 F1과 클래스별 지표를 함께 확인하세요. 유사 성능이라면 'weighted' 또는 'smote' 모드를 고려하세요.\n")
            elif share >= 0.6:
                lines.append(f"검증 세트에 불균형이 있습니다 (최대 클래스 ≈ {share*100:.1f}%). 'zero' 모드는 편향될 수 있으니 매크로/가중치 지표를 확인하세요.\n")
            else:
                lines.append("일부 불균형이 존재할 수 있습니다. 정확도와 함께 매크로/가중치 지표를 확인하세요.\n")
            lines.append("\n")

        eda_plot = os.path.join('reports', dataset, f'kinship_distribution_{dataset}.png')
        fi_plot = os.path.join('reports', dataset, f'feature_importance_{dataset}.png')
        if os.path.exists(eda_plot) or os.path.exists(fi_plot):
            lines.append("## 데이터셋 수준 플롯\n")
            items = []
            if os.path.exists(eda_plot):
                rel_eda = os.path.relpath(eda_plot, start=out_dir)
                items.append(("친족도 분포", rel_eda))
            if os.path.exists(fi_plot):
                rel_fi = os.path.relpath(fi_plot, start=out_dir)
                items.append(("상위 특성 중요도", rel_fi))
            if items:
                lines.append('<div style="display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap;">')
                for title_txt, img_path in items:
                    lines.append('<figure style="flex:1; min-width:320px; margin:0;">')
                    lines.append(f'<img src="{img_path}" style="width:100%; max-width:100%;" />')
                    lines.append(f'<figcaption style="text-align:center; font-size: 13px; margin-top:6px;">{title_txt}</figcaption>')
                    lines.append('</figure>')
                lines.append('</div>')
                lines.append("")

        try:
            lines.append("## 요약 (상세)\n")
            lines.append("| 모드 | 최고 모델 | 정확도 | F1 (가중치) | AUC (가중치) |")
            lines.append("|------|-----------|--------|------------|--------------|")
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

            if summary:
                rec = max(summary, key=lambda t: (t[2].get('f1_weighted') or 0))
                mode_name, model_name, metrics = rec
                lines.append("")
                lines.append("### 권장 사항\n")
                lines.append(f"현재 최고 성능 모델은 '{mode_name}' 모드의 {model_name} 입니다 (F1(가중치)={metrics.get('f1_weighted'):.4f}).")
                if vcd_any:
                    if maj / total >= 0.6 and mode_name == 'zero':
                        lines.append("불균형을 고려하여 'zero'는 기준선으로만 사용하고, 장기 학습 시 'weighted' 또는 'smote'를 우선 고려하세요.")
                lines.append("")
        except Exception:
            pass

        for mode in modes:
            if mode not in consolidated['modes']:
                continue
            mode_block = consolidated['modes'][mode]
            lines.append(f"## 모드: {mode}\n")
            lines.append(f"검증 샘플 수: {mode_block.get('val_samples')}  ")
            vcd = mode_block.get('val_class_distribution', {})
            if vcd:
                parts = [f"{k}={v}" for k, v in vcd.items()]
                lines.append(f"클래스 분포: {', '.join(parts)}\n")

            if mode == 'zero':
                lines.append("> 참고: 'zero' 모드는 재균형을 사용하지 않아 다수 클래스에 유리할 수 있습니다. 기준선으로 보세요.\n")
            elif mode == 'weighted':
                lines.append("> 참고: 'weighted' 모드는 클래스 가중 손실을 사용합니다. 에포크가 매우 적을 때 성능이 일시 저하될 수 있습니다.\n")
            elif mode == 'smote':
                lines.append("> 참고: 'smote' 모드는 학습 세트만 과샘플링하며, 검증은 원본 분포에서 수행합니다.\n")

            lines.append("| 모델 | 정확도 | F1 (가중치) | F1 (매크로) | AUC (가중치) | AUC (매크로) | AUC (마이크로) |")
            lines.append("|------|--------|------------|------------|--------------|--------------|----------------|")
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

            cms = []
            for model_key, metrics in mode_block.get('models', {}).items():
                cm_path = metrics.get('confusion_matrix_path')
                if cm_path and os.path.exists(cm_path):
                    rel_path = os.path.relpath(cm_path, start=out_dir)
                    cms.append((model_key, rel_path))
            if cms:
                lines.append("")
                i = 0
                while i < len(cms):
                    remaining = len(cms) - i
                    if remaining >= 2:
                        row = cms[i:i+2]
                        lines.append('<div style="display:flex; gap:12px; align-items:flex-start;">')
                        for model_key, rel_path in row:
                            lines.append('<figure style="flex:1; margin:0;">')
                            lines.append(f'<img src="{rel_path}" style="width:100%; max-width:100%;" />')
                            lines.append(f'<figcaption style="text-align:center; font-size: 13px; margin-top:6px;">혼동 행렬: {model_key}</figcaption>')
                            lines.append('</figure>')
                        lines.append('</div>')
                        lines.append("")
                        i += 2
                    else:
                        model_key, rel_path = cms[i]
                        lines.append('<div style="display:flex; gap:12px; align-items:flex-start;">')
                        lines.append('<figure style="flex:1; margin:0; display:flex; justify-content:center;">')
                        lines.append('<div style="width:70%;">')
                        lines.append(f'<img src="{rel_path}" style="width:100%; max-width:100%;" />')
                        lines.append(f'<div style="text-align:center; font-size: 13px; margin-top:6px;">혼동 행렬: {model_key}</div>')
                        lines.append('</div>')
                        lines.append('</figure>')
                        lines.append('<figure style="flex:1; margin:0;">&nbsp;</figure>')
                        lines.append('</div>')
                        lines.append("")
                        i += 1
                lines.append("")
        return lines

    # Write Markdown (EN)
    md_path = os.path.join(out_dir, 'results.md')
    lines_en = build_lines_en()
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines_en))
    print(f"Wrote Markdown report to {md_path}")

    # Write Markdown (KR)
    md_path_kr = os.path.join(out_dir, 'results_KR.md')
    lines_kr = build_lines_kr()
    with open(md_path_kr, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines_kr))
    print(f"Wrote Korean Markdown report to {md_path_kr}")

    # Optionally convert Markdown to PDF using npx md-to-pdf if available (no output flags)
    try:
        npx_path = shutil.which('npx')
        if npx_path:
            # English PDF
            pdf_path = os.path.join(out_dir, 'results.pdf')
            default_pdf = os.path.splitext(md_path)[0] + '.pdf'
            print("Converting Markdown to PDF via npx md-to-pdf (EN, no output flag)...")
            subprocess.run([npx_path, 'md-to-pdf', md_path], check=False)
            if os.path.exists(default_pdf):
                try:
                    if default_pdf != pdf_path:
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)
                        shutil.move(default_pdf, pdf_path)
                except Exception:
                    pass
            if os.path.exists(pdf_path):
                print(f"Wrote PDF report to {pdf_path}")
            else:
                print("md-to-pdf did not produce a PDF for EN. You can run it manually: npx md-to-pdf <path-to-results.md>")

            # Korean PDF
            pdf_path_kr = os.path.join(out_dir, 'results_KR.pdf')
            default_pdf_kr = os.path.splitext(md_path_kr)[0] + '.pdf'
            print("Converting Markdown to PDF via npx md-to-pdf (KR, no output flag)...")
            subprocess.run([npx_path, 'md-to-pdf', md_path_kr], check=False)
            if os.path.exists(default_pdf_kr):
                try:
                    if default_pdf_kr != pdf_path_kr:
                        if os.path.exists(pdf_path_kr):
                            os.remove(pdf_path_kr)
                        shutil.move(default_pdf_kr, pdf_path_kr)
                except Exception:
                    pass
            if os.path.exists(pdf_path_kr):
                print(f"Wrote PDF report to {pdf_path_kr}")
            else:
                print("md-to-pdf did not produce a PDF for KR. You can run it manually: npx md-to-pdf <path-to-results_KR.md>")
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
