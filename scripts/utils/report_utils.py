import os
import json
from datetime import datetime


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def discover_results(dataset: str, processed_dir: str = 'data/processed'):
    """Discover scenario+mode evaluation result JSON files.

    Supports legacy naming (evaluation_results_<dataset>_<mode>.json) by assigning
    scenario 'included'. Returns list of tuples (path, scenario, mode, payload).
    """
    discovered = []
    for fname in os.listdir(processed_dir):
        if not fname.startswith(f'evaluation_results_{dataset}_') or not fname.endswith('.json'):
            continue
        parts = fname[:-5].split('_')
        if len(parts) < 5:
            # Legacy: evaluation_results_<dataset>_<mode>.json
            mode = parts[-1]
            scenario = 'included'
            path = os.path.join(processed_dir, fname)
            discovered.append((path, scenario, mode, load_json(path)))
            continue
        scenario = parts[-2]
        mode = parts[-1]
        path = os.path.join(processed_dir, fname)
        discovered.append((path, scenario, mode, load_json(path)))
    return discovered


def consolidate(discovered, dataset: str):
    if not discovered:
        return None
    any_result = discovered[0][3]
    label_names = any_result.get('label_names', [])
    device = any_result.get('device')
    consolidated = {
        'dataset': dataset,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'device': device,
        'label_names': label_names,
        'scenarios': {}
    }
    for path, scenario, mode, res in discovered:
        scen = consolidated['scenarios'].setdefault(scenario, {'modes': {}})
        scen['modes'][mode] = {
            'val_samples': res.get('val_samples'),
            'val_class_distribution': res.get('val_class_distribution'),
            'train_samples_before': res.get('train_samples_before'),
            'train_class_distribution_before': res.get('train_class_distribution_before'),
            'train_samples_after': res.get('train_samples_after'),
            'train_class_distribution_after': res.get('train_class_distribution_after'),
            'models': res.get('models', {})
        }
    return consolidated


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else ("N/A" if x is None else str(x))


def build_markdown_en(consolidated, reports_dir):
    dataset = consolidated['dataset']
    lines = []
    lines.append(f"# Results for {dataset}\n")
    lines.append(f"Generated: {consolidated['generated_at']}  ")
    lines.append(f"Device: {consolidated['device']}\n")
    if consolidated.get('label_names'):
        lines.append(f"Labels: {', '.join(consolidated['label_names'])}\n")
    lines.append("## Executive summary\n")
    lines.append('<div style="font-size:12px; line-height:1.25">')
    lines.append('<table style="font-size:12px; border-collapse:collapse;">')
    lines.append('<thead>')
    lines.append('<tr>'
                 '<th style="padding:2px 6px; font-size:12px;">Scenario</th>'
                 '<th style="padding:2px 6px; font-size:12px;">Mode</th>'
                 '<th style="padding:2px 6px; font-size:12px;">Best Model</th>'
                 '<th style="padding:2px 6px; font-size:12px;">Val N</th>'
                 '<th style="padding:2px 6px; font-size:12px;">Train N (pre→post)</th>'
                 '<th style="padding:2px 6px; font-size:12px;">Accuracy</th>'
                 '<th style="padding:2px 6px; font-size:12px;">F1 (weighted)</th>'
                 '<th style="padding:2px 6px; font-size:12px;">AUC (weighted)</th>'
                 '<th style="padding:2px 6px; font-size:12px;">Reliability</th>'
                 '</tr>')
    lines.append('</thead>')
    lines.append('<tbody>')
    for scenario_key, scen in consolidated.get('scenarios', {}).items():
        # Imbalance share
        vcd_any = None
        if scen['modes']:
            any_mode_key = next(iter(scen['modes']))
            vcd_any = scen['modes'][any_mode_key].get('val_class_distribution', {})
        total = sum(int(v) for v in vcd_any.values()) if vcd_any else 0
        maj = max(int(v) for v in vcd_any.values()) if vcd_any else 0
        share = (maj / total) if total else 0.0
        for mode_key, mb in scen.get('modes', {}).items():
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
            val_n = mb.get('val_samples')
            pre_n = mb.get('train_samples_before')
            post_n = mb.get('train_samples_after')
            if mode_key == 'zero':
                reli = "Baseline only (unreliable under imbalance)" if share >= 0.6 else "Baseline"
            elif mode_key == 'weighted':
                reli = "Mitigates imbalance (class weights)"
            elif mode_key == 'smote':
                reli = "Mitigates imbalance (oversampled train)"
            else:
                reli = "Mitigates imbalance (over+under sampling)"
            lines.append(
                f"<tr>"
                f"<td style='padding:2px 6px; font-size:12px;'>{scenario_key}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{mode_key}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{best_key}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{val_n}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{pre_n}→{post_n}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{_fmt(best.get('accuracy'))}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{_fmt(best.get('f1_weighted'))}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{_fmt(best.get('auc_weighted'))}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{reli}</td>"
                f"</tr>"
            )

    lines.append('</tbody>')
    lines.append('</table>')
    lines.append("</div>\n")
    lines.append("\n## Scenarios and modes\n")
    lines.append("- Scenarios: 'included' keeps UN-labeled rows; 'noUN' removes them prior to splits and training.\n")
    lines.append("- Modes: zero (no rebalancing), weighted (class-weighted loss), smote (oversampling), overunder (SMOTE + ENN/Tomek).\n\n")

    lines.append("## Key terms\n")
    lines.append("- Accuracy: Share of correct predictions. Can be misleading under heavy imbalance.\n")
    lines.append("- F1 (weighted): F1 averaged with class frequencies as weights. Better reflects performance under imbalance.\n")
    lines.append("- F1 (macro): Simple average of F1 across classes (each class equal weight).\n")
    lines.append("- AUC (weighted/macro/micro): OvR ROC AUC aggregated differently (class frequency, equal class, instance level).\n\n")

    # Scenario-level plots
    for scenario_key in [k for k in ['included','noUN'] if k in consolidated.get('scenarios', {})]:
        # New organized assets directories
        assets_dir = os.path.join(reports_dir, 'assets', scenario_key)
        eda_plot = os.path.join(assets_dir, f'kinship_distribution_{dataset}_{scenario_key}.png')
        fi_plot = os.path.join(assets_dir, f'feature_importance_{dataset}_{scenario_key}.png')
        if os.path.exists(eda_plot) or os.path.exists(fi_plot):
            lines.append(f"## Plots ({scenario_key})\n")
            items = []
            if os.path.exists(eda_plot):
                rel_eda = os.path.relpath(eda_plot, start=reports_dir)
                items.append(("Kinship Distribution", rel_eda))
            if os.path.exists(fi_plot):
                rel_fi = os.path.relpath(fi_plot, start=reports_dir)
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

    # Detailed sections
    for scenario_key, scen in consolidated.get('scenarios', {}).items():
        lines.append(f"## Scenario: {scenario_key}\n")
        for mode, mode_block in scen.get('modes', {}).items():
            lines.append(f"### Mode: {mode}\n")
            val_n = mode_block.get('val_samples')
            lines.append(f"Validation samples: {val_n}  ")
            vcd = mode_block.get('val_class_distribution', {})
            if vcd:
                parts = [f"{k}={v}" for k, v in vcd.items()]
                lines.append(f"Class distribution: {', '.join(parts)}\n")
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

            # Caveats
            if mode == 'zero':
                lines.append("> Caveat: 'zero' mode uses no rebalancing and may favor the majority class; treat results as a baseline.\n")
            elif mode == 'weighted':
                lines.append("> Note: 'weighted' mode applies class-weighted loss to mitigate imbalance; with very few epochs it may underperform transiently.\n")
            elif mode == 'smote':
                lines.append("> Note: 'smote' mode oversamples the training split; evaluation remains on the original (possibly imbalanced) validation set.\n")
            else:
                lines.append("> Note: 'overunder' combines oversampling and undersampling to clean ambiguous regions after SMOTE.\n")

            lines.append("| Model | Val N | Train N (pre→post) | Accuracy | F1 (weighted) | F1 (macro) | AUC (weighted) | AUC (macro) | AUC (micro) |")
            lines.append("|-------|-------|---------------------|----------|---------------|------------|----------------|-------------|-------------|")
            for model_key, metrics in mode_block.get('models', {}).items():
                lines.append(
                    f"| {model_key} | {val_n} | {pre_n}→{post_n} | {_fmt(metrics.get('accuracy'))} | {_fmt(metrics.get('f1_weighted'))} | {_fmt(metrics.get('f1_macro'))} | {_fmt(metrics.get('auc_weighted'))} | {_fmt(metrics.get('auc_macro'))} | {_fmt(metrics.get('auc_micro'))} |"
                )
            # Confusion matrices
            cms = []
            for model_key, metrics in mode_block.get('models', {}).items():
                cm_path = metrics.get('confusion_matrix_path')
                if cm_path and os.path.exists(cm_path):
                    rel_path = os.path.relpath(cm_path, start=reports_dir)
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
                            lines.append(f'<figcaption style="text-align:center; font-size: 13px; margin-top:6px;">Confusion Matrix: {model_key}</figcaption>')
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
                        lines.append(f'<div style="text-align:center; font-size: 13px; margin-top:6px;">Confusion Matrix: {model_key}</div>')
                        lines.append('</div>')
                        lines.append('</figure>')
                        lines.append('<figure style="flex:1; margin:0;">&nbsp;</figure>')
                        lines.append('</div>')
                        lines.append("")
                        i += 1
                lines.append("")
    return lines


def build_markdown_kr(consolidated, reports_dir):
    # Simplified Korean version mirroring EN structure.
    dataset = consolidated['dataset']
    lines = []
    lines.append(f"# {dataset} 결과\n")
    lines.append(f"생성 시각: {consolidated['generated_at']}  ")
    lines.append(f"디바이스: {consolidated['device']}\n")
    if consolidated.get('label_names'):
        lines.append(f"레이블: {', '.join(consolidated['label_names'])}\n")
    lines.append("## 요약\n")
    lines.append('<div style="font-size:12px; line-height:1.25">')
    lines.append('<table style="font-size:12px; border-collapse:collapse;">')
    lines.append('<thead>')
    lines.append('<tr>'
                 '<th style="padding:2px 6px; font-size:12px;">시나리오</th>'
                 '<th style="padding:2px 6px; font-size:12px;">모드</th>'
                 '<th style="padding:2px 6px; font-size:12px;">최고 모델</th>'
                 '<th style="padding:2px 6px; font-size:12px;">검증 N</th>'
                 '<th style="padding:2px 6px; font-size:12px;">학습 N (전→후)</th>'
                 '<th style="padding:2px 6px; font-size:12px;">정확도</th>'
                 '<th style="padding:2px 6px; font-size:12px;">F1 (가중치)</th>'
                 '<th style="padding:2px 6px; font-size:12px;">AUC (가중치)</th>'
                 '<th style="padding:2px 6px; font-size:12px;">신뢰도</th>'
                 '</tr>')
    lines.append('</thead>')
    lines.append('<tbody>')
    for scenario_key, scen in consolidated.get('scenarios', {}).items():
        vcd_any = None
        if scen['modes']:
            any_mode_key = next(iter(scen['modes']))
            vcd_any = scen['modes'][any_mode_key].get('val_class_distribution', {})
        total = sum(int(v) for v in vcd_any.values()) if vcd_any else 0
        maj = max(int(v) for v in vcd_any.values()) if vcd_any else 0
        share = (maj / total) if total else 0.0
        for mode_key, mb in scen.get('modes', {}).items():
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
            val_n = mb.get('val_samples')
            pre_n = mb.get('train_samples_before')
            post_n = mb.get('train_samples_after')
            if mode_key == 'zero':
                reli = "기준선(불균형 시 신뢰 낮음)" if share >= 0.6 else "기준선"
            elif mode_key == 'weighted':
                reli = "불균형 완화(가중 손실)"
            elif mode_key == 'smote':
                reli = "불균형 완화(SMOTE 과샘플링)"
            else:
                reli = "불균형 완화(과샘플+언더샘플)"
            lines.append(
                f"<tr>"
                f"<td style='padding:2px 6px; font-size:12px;'>{scenario_key}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{mode_key}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{best_key}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{val_n}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{pre_n}→{post_n}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{_fmt(best.get('accuracy'))}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{_fmt(best.get('f1_weighted'))}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{_fmt(best.get('auc_weighted'))}</td>"
                f"<td style='padding:2px 6px; font-size:12px;'>{reli}</td>"
                f"</tr>"
            )

    # Scenario plots
    lines.append('</tbody>')
    lines.append('</table>')
    lines.append("</div>\n")
    for scenario_key in [k for k in ['included','noUN'] if k in consolidated.get('scenarios', {})]:
        assets_dir = os.path.join(reports_dir, 'assets', scenario_key)
        eda_plot = os.path.join(assets_dir, f'kinship_distribution_{dataset}_{scenario_key}.png')
        fi_plot = os.path.join(assets_dir, f'feature_importance_{dataset}_{scenario_key}.png')
        if os.path.exists(eda_plot) or os.path.exists(fi_plot):
            lines.append(f"## 플롯 ({scenario_key})\n")
            items = []
            if os.path.exists(eda_plot):
                rel_eda = os.path.relpath(eda_plot, start=reports_dir)
                items.append(("친족도 분포", rel_eda))
            if os.path.exists(fi_plot):
                rel_fi = os.path.relpath(fi_plot, start=reports_dir)
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

    for scenario_key, scen in consolidated.get('scenarios', {}).items():
        lines.append(f"## 시나리오: {scenario_key}\n")
        for mode, mode_block in scen.get('modes', {}).items():
            lines.append(f"### 모드: {mode}\n")
            val_n = mode_block.get('val_samples')
            lines.append(f"검증 샘플 수: {val_n}  ")
            vcd = mode_block.get('val_class_distribution', {})
            if vcd:
                parts = [f"{k}={v}" for k, v in vcd.items()]
                lines.append(f"클래스 분포: {', '.join(parts)}\n")
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
            else:
                lines.append("> 참고: 'overunder'는 SMOTE 이후 ENN/Tomek으로 경계의 모호 표본을 제거합니다.\n")
            lines.append("| 모델 | 검증 N | 학습 N (전→후) | 정확도 | F1 (가중치) | F1 (매크로) | AUC (가중치) | AUC (매크로) | AUC (마이크로) |")
            lines.append("|------|--------|-----------------|--------|------------|------------|--------------|--------------|----------------|")
            for model_key, metrics in mode_block.get('models', {}).items():
                lines.append(
                    f"| {model_key} | {val_n} | {pre_n}→{post_n} | {_fmt(metrics.get('accuracy'))} | {_fmt(metrics.get('f1_weighted'))} | {_fmt(metrics.get('f1_macro'))} | {_fmt(metrics.get('auc_weighted'))} | {_fmt(metrics.get('auc_macro'))} | {_fmt(metrics.get('auc_micro'))} |"
                )
            cms = []
            for model_key, metrics in mode_block.get('models', {}).items():
                cm_path = metrics.get('confusion_matrix_path')
                if cm_path and os.path.exists(cm_path):
                    rel_path = os.path.relpath(cm_path, start=reports_dir)
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
