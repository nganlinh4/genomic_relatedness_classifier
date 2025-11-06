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
                'train_samples_before': res.get('train_samples_before'),
                'train_class_distribution_before': res.get('train_class_distribution_before'),
                'train_samples_after': res.get('train_samples_after'),
                'train_class_distribution_after': res.get('train_class_distribution_after'),
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

        # Executive summary (merged with class-imbalance note and reliability)
        try:
            any_mode_key = next(iter(consolidated['modes'])) if consolidated['modes'] else None
            vcd_any = consolidated['modes'][any_mode_key].get('val_class_distribution', {}) if any_mode_key else {}
            total = sum(int(v) for v in vcd_any.values()) if vcd_any else 0
            maj = max(int(v) for v in vcd_any.values()) if vcd_any else 0
            share = (maj / total) if total else 0.0

            lines.append("## Executive summary\n")
            if vcd_any:
                if share >= 0.8:
                    lines.append(f"Note: validation set is severely imbalanced (majority ≈ {share*100:.1f}%). 'zero' mode is unreliable for minority classes.\n")
                elif share >= 0.6:
                    lines.append(f"Note: validation set is imbalanced (majority ≈ {share*100:.1f}%). Treat 'zero' as a baseline only.\n")
                else:
                    lines.append("Note: some imbalance may persist; prefer macro/weighted metrics alongside accuracy.\n")
                lines.append("")

            lines.append("| Mode | Best Model | Val N | Train N (pre→post) | Accuracy | F1 (weighted) | AUC (weighted) | Reliability |")
            lines.append("|------|------------|-------|---------------------|----------|---------------|----------------|-------------|")

            for mode in modes:
                mb = consolidated['modes'].get(mode)
                if not mb:
                    continue
                # pick best by weighted F1
                best_key = None
                best = None
                for mk, metrics in mb.get('models', {}).items():
                    if metrics is None:
                        continue
                    if best is None or (metrics.get('f1_weighted') or 0) > (best.get('f1_weighted') or 0):
                        best = metrics
                        best_key = mk
                if best_key is None:
                    continue
                def fmt(x):
                    return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else ("N/A" if x is None else str(x))
                val_n = mb.get('val_samples')
                pre_n = mb.get('train_samples_before')
                post_n = mb.get('train_samples_after')
                if mode == 'zero':
                    reli = "Baseline only (unreliable under imbalance)" if share >= 0.6 else "Baseline"
                elif mode == 'weighted':
                    reli = "Mitigates imbalance (class weights)"
                else:
                    reli = "Mitigates imbalance (oversampled train)"
                lines.append(f"| {mode} | {best_key} | {val_n} | {pre_n}→{post_n} | {fmt(best.get('accuracy'))} | {fmt(best.get('f1_weighted'))} | {fmt(best.get('auc_weighted'))} | {reli} |")
        except Exception:
            pass

        # Modes explained (for non-technical readers)
        lines.append("## Modes explained\n")
        lines.append("- zero: No rebalancing. Fast baseline; can be biased toward majority classes when data are imbalanced.\n")
        lines.append("- weighted: Uses class-weighted loss so mistakes on rare classes count more during training.\n")
        lines.append("- smote: Creates synthetic minority samples to balance the training set. Validation stays on original data.\n\n")

        # Key terms
        lines.append("## Key terms\n")
        lines.append("- Accuracy: Share of correct predictions. Can be misleading under heavy imbalance.\n")
        lines.append("- F1 (weighted): F1 averaged with class frequencies as weights. Better reflects performance under imbalance.\n")
        lines.append("- F1 (macro): Simple average of F1 across classes (each class equal weight).\n")
        lines.append("- AUC (weighted/macro/micro): One-vs-Rest ROC AUC, aggregated by class frequency (weighted), equal class weight (macro), or instance level (micro).\n\n")

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

        # Mode sections
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

            # Training counts (pre/post) and distribution
            pre_n = mode_block.get('train_samples_before')
            post_n = mode_block.get('train_samples_after')
            tcd_pre = mode_block.get('train_class_distribution_before', {})
            tcd_post = mode_block.get('train_class_distribution_after', {})
            if pre_n is not None and post_n is not None:
                lines.append(f"Training samples: pre={pre_n}, post={post_n}  ")
            if tcd_pre:
                parts = [f"{k}={v}" for k, v in tcd_pre.items()]
                lines.append(f"Train class dist (pre): {', '.join(parts)}  ")
            if tcd_post and tcd_post != tcd_pre:
                parts = [f"{k}={v}" for k, v in tcd_post.items()]
                lines.append(f"Train class dist (post): {', '.join(parts)}\n")

            # Mode caveats
            if mode == 'zero':
                lines.append("> Caveat: 'zero' mode uses no rebalancing and may favor the majority class; treat results as a baseline.\n")
            elif mode == 'weighted':
                lines.append("> Note: 'weighted' mode applies class-weighted loss to mitigate imbalance; with very few epochs it may underperform transiently.\n")
            elif mode == 'smote':
                lines.append("> Note: 'smote' mode oversamples the training split; evaluation remains on the original (possibly imbalanced) validation set.\n")

            # Table header
            lines.append("| Model | Val N | Train N (pre→post) | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |")
            lines.append("|-------|-------|---------------------|----------|---------------|------------|----------------|-------------|-------------|")
            for model_key, metrics in mode_block.get('models', {}).items():
                acc = metrics.get('accuracy')
                f1w = metrics.get('f1_weighted')
                f1m = metrics.get('f1_macro')
                aw = metrics.get('auc_weighted')
                am = metrics.get('auc_macro')
                ai = metrics.get('auc_micro')
                def fmt(x):
                    return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else ("N/A" if x is None else str(x))
                lines.append(f"| {model_key} | {mode_block.get('val_samples')} | {pre_n}→{post_n} | {fmt(acc)} | {fmt(f1w)} | {fmt(f1m)} | {fmt(aw)} | {fmt(am)} | {fmt(ai)} |")

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

        # 요약 먼저 배치 (불균형 참고와 통합, 신뢰도 포함)
        try:
            lines.append("## 요약\n")
            any_mode_key = next(iter(consolidated['modes'])) if consolidated['modes'] else None
            vcd_any = consolidated['modes'][any_mode_key].get('val_class_distribution', {}) if any_mode_key else {}
            total = sum(int(v) for v in vcd_any.values()) if vcd_any else 0
            maj = max(int(v) for v in vcd_any.values()) if vcd_any else 0
            share = (maj / total) if total else 0.0
            if vcd_any:
                if share >= 0.8:
                    lines.append(f"참고: 검증 세트의 불균형이 매우 큽니다 (최대 클래스 ≈ {share*100:.1f}%). 'zero' 모드 지표는 소수 클래스에 대해 신뢰하기 어렵습니다.\n")
                elif share >= 0.6:
                    lines.append(f"참고: 검증 세트에 불균형이 있습니다 (최대 클래스 ≈ {share*100:.1f}%). 'zero'는 기준선으로만 보세요.\n")
                else:
                    lines.append("참고: 일부 불균형이 있을 수 있으니 정확도와 함께 매크로/가중치 지표를 보세요.\n")
                lines.append("")

            lines.append("| 모드 | 최고 모델 | 검증 N | 학습 N (전→후) | 정확도 | F1 (가중치) | AUC (가중치) | 신뢰도 |")
            lines.append("|------|-----------|--------|-----------------|--------|------------|--------------|--------|")
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
                    val_n = mb.get('val_samples')
                    pre_n = mb.get('train_samples_before')
                    post_n = mb.get('train_samples_after')
                    if mode == 'zero':
                        reli = "기준선(불균형 시 신뢰 낮음)" if share >= 0.6 else "기준선"
                    elif mode == 'weighted':
                        reli = "불균형 완화(가중 손실)"
                    else:
                        reli = "불균형 완화(SMOTE 과샘플링)"
                    lines.append(f"| {mode} | {best_key} | {val_n} | {pre_n}→{post_n} | {fmt(best.get('accuracy'))} | {fmt(best.get('f1_weighted'))} | {fmt(best.get('auc_weighted'))} | {reli} |")
        except Exception:
            pass

        # 모드 설명 (비전문가용)
        lines.append("## 모드 설명\n")
        lines.append("- zero: 재균형 미적용. 빠른 기준선이나, 불균형 시 다수 클래스에 치우칠 수 있습니다.\n")
        lines.append("- weighted: 클래스 가중 손실로 소수 클래스 오류의 영향을 크게 반영합니다.\n")
        lines.append("- smote: 학습 세트만 합성 표본으로 균형화합니다. 검증은 원본 분포로 진행됩니다.\n\n")

        # 용어 설명
        lines.append("## 용어 설명\n")
        lines.append("- 정확도(Accuracy): 전체 예측 중 정답 비율. 불균형이 크면 왜곡될 수 있습니다.\n")
        lines.append("- F1(가중치): 클래스 빈도로 가중 평균한 F1. 불균형 상황을 더 잘 반영합니다.\n")
        lines.append("- F1(매크로): 클래스별 F1을 단순 평균(각 클래스 동일 가중).\n")
        lines.append("- AUC(가중치/매크로/마이크로): One-vs-Rest ROC AUC을 클래스 빈도 가중(가중치), 동일 가중(매크로), 개별 샘플 기준(마이크로)으로 집계.\n\n")

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
        # (Detailed summary and recommendation removed to avoid confusion)

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

            # 학습 데이터 수(전/후) 및 분포
            pre_n = mode_block.get('train_samples_before')
            post_n = mode_block.get('train_samples_after')
            tcd_pre = mode_block.get('train_class_distribution_before', {})
            tcd_post = mode_block.get('train_class_distribution_after', {})
            if pre_n is not None and post_n is not None:
                lines.append(f"학습 샘플 수: 전={pre_n}, 후={post_n}  ")
            if tcd_pre:
                parts = [f"{k}={v}" for k, v in tcd_pre.items()]
                lines.append(f"학습 클래스 분포(전): {', '.join(parts)}  ")
            if tcd_post and tcd_post != tcd_pre:
                parts = [f"{k}={v}" for k, v in tcd_post.items()]
                lines.append(f"학습 클래스 분포(후): {', '.join(parts)}\n")

            if mode == 'zero':
                lines.append("> 참고: 'zero' 모드는 재균형을 사용하지 않아 다수 클래스에 유리할 수 있습니다. 기준선으로 보세요.\n")
            elif mode == 'weighted':
                lines.append("> 참고: 'weighted' 모드는 클래스 가중 손실을 사용합니다. 에포크가 매우 적을 때 성능이 일시 저하될 수 있습니다.\n")
            elif mode == 'smote':
                lines.append("> 참고: 'smote' 모드는 학습 세트만 과샘플링하며, 검증은 원본 분포에서 수행합니다.\n")

            lines.append("| 모델 | 검증 N | 학습 N (전→후) | 정확도 | F1 (가중치) | F1 (매크로) | AUC (가중치) | AUC (매크로) | AUC (마이크로) |")
            lines.append("|------|--------|-----------------|--------|------------|------------|--------------|--------------|----------------|")
            for model_key, metrics in mode_block.get('models', {}).items():
                acc = metrics.get('accuracy')
                f1w = metrics.get('f1_weighted')
                f1m = metrics.get('f1_macro')
                aw = metrics.get('auc_weighted')
                am = metrics.get('auc_macro')
                ai = metrics.get('auc_micro')
                def fmt(x):
                    return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else ("N/A" if x is None else str(x))
                lines.append(f"| {model_key} | {mode_block.get('val_samples')} | {pre_n}→{post_n} | {fmt(acc)} | {fmt(f1w)} | {fmt(f1m)} | {fmt(aw)} | {fmt(am)} | {fmt(ai)} |")

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
