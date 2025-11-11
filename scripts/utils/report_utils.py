import os
import json
import re
from datetime import datetime
import pandas as pd


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
            'training_meta': res.get('training_meta'),
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
        lines.append(f"Labels: {', '.join(str(ln) for ln in consolidated['label_names'])}\n")
    # Data merging summary (computed on-the-fly from raw stats; no artifacts)
    try:
        raw_dir = os.path.join('data', 'raw')
        p_path = os.path.join(raw_dir, 'merged_info.out')
        a_path = os.path.join(raw_dir, 'merged_added_info.out')

        def _parse_stats_cols(path):
            rows = []
            if not os.path.exists(path):
                return pd.DataFrame(columns=['pair'])
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip() or 'allChr' not in line:
                        continue
                    m = re.search(r"\[(.*?)\]", line)
                    if not m:
                        continue
                    pair = m.group(1).replace(' ', '')
                    tail = line.split('allChr', 1)[1]
                    stats = {'pair': pair}
                    for mm in re.finditer(r'([^:]+?):(\S+)', tail):
                        key = mm.group(1).strip()
                        val = mm.group(2).strip()
                        try:
                            stats[key] = float(val)
                        except ValueError:
                            stats[key] = val
                    rows.append(stats)
            df = pd.DataFrame(rows)
            return df

        df_p = _parse_stats_cols(p_path)
        df_a = _parse_stats_cols(a_path)
        if not df_p.empty or not df_a.empty:
            p_cols = set(df_p.columns) - {'pair'}
            a_cols = set(df_a.columns) - {'pair'}
            common_cols = sorted(p_cols & a_cols)
            added_only = sorted(a_cols - p_cols)
            lines.append("## Data merging\n")
            lines.append("- Source stats files: merged_info.out + merged_added_info.out  ")
            lines.append(
                f"- Columns: primary={len(p_cols)}; added={len(a_cols)}; overlap={len(common_cols)}; added-only kept={len(added_only)}\n"
            )
            if added_only:
                lines.append("- Added-only columns: " + ", ".join(added_only) + "\n")
            lines.append("- Overlapping metrics are deduplicated (prefer primary, backfill missing from added).\n")
    except Exception:
        # Be quiet in reports if parsing fails
        pass
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

    # At-a-glance highlights for executives
    try:
        lines.append("### At a glance\n")
        for scenario_key, scen in consolidated.get('scenarios', {}).items():
            # Majority share on validation set (imbalance indicator)
            vcd_any = None
            if scen['modes']:
                any_mode_key = next(iter(scen['modes']))
                vcd_any = scen['modes'][any_mode_key].get('val_class_distribution', {})
            total = sum(int(v) for v in vcd_any.values()) if vcd_any else 0
            maj = max(int(v) for v in vcd_any.values()) if vcd_any else 0
            share = (maj / total) if total else 0.0
            # Best across all modes/models by weighted F1
            best_tuple = None  # (f1w, aucw, mode, model, valN)
            for mode_key, mb in scen.get('modes', {}).items():
                for model_key, metrics in (mb.get('models') or {}).items():
                    if not metrics:
                        continue
                    f1w = metrics.get('f1_weighted') or 0.0
                    aucw = metrics.get('auc_weighted') or 0.0
                    valn = mb.get('val_samples')
                    if (best_tuple is None) or (f1w > best_tuple[0]):
                        best_tuple = (f1w, aucw, mode_key, model_key, valn)
            if best_tuple:
                f1w, aucw, mode_key, model_key, valn = best_tuple
                lines.append(f"- {scenario_key}: Best {model_key} ({mode_key}) — F1w {f1w:.2f}, AUCw {aucw:.2f}; Val N {valn}; Majority share {share:.0%}")
        lines.append("")
    except Exception:
        pass

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
        assets_dir = os.path.join(reports_dir, 'assets', scenario_key)
        # Prefer SVG for crisp text; fallback to PNG if SVG missing
        eda_svg = os.path.join(assets_dir, f'kinship_distribution_{dataset}_{scenario_key}.svg')
        eda_png = os.path.join(assets_dir, f'kinship_distribution_{dataset}_{scenario_key}.png')
        fi_svg = os.path.join(assets_dir, f'feature_importance_{dataset}_{scenario_key}.svg')
        fi_png = os.path.join(assets_dir, f'feature_importance_{dataset}_{scenario_key}.png')
        eda_plot = eda_svg if os.path.exists(eda_svg) else eda_png
        fi_plot = fi_svg if os.path.exists(fi_svg) else fi_png
        if (eda_plot and os.path.exists(eda_plot)) or (fi_plot and os.path.exists(fi_plot)):
            lines.append(f"## Plots ({scenario_key})\n")
            items = []
            if eda_plot and os.path.exists(eda_plot):
                rel_eda = os.path.relpath(eda_plot, start=reports_dir)
                items.append(("Kinship Distribution", rel_eda))
            if fi_plot and os.path.exists(fi_plot):
                rel_fi = os.path.relpath(fi_plot, start=reports_dir)
                items.append(("Top Feature Importances", rel_fi))
            if items:
                lines.append('<div style="display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap;">')
                for title_txt, img_path in items:
                    lines.append('<figure style="flex:0 0 48%; max-width:48%; margin:0;">')
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
            # Training info: prefer RF training time when RF-only runs to avoid misleading deep-model epochs/time
            tmeta = mode_block.get('training_meta') or {}
            models_block = mode_block.get('models') or {}
            rf_metrics = models_block.get('RandomForest') or {}
            rf_time = rf_metrics.get('train_duration_seconds')
            only_rf = bool(tmeta.get('only_randomforest'))
            if only_rf and rf_time is not None:
                tdev = tmeta.get('device')
                lines.append(f"Training: RandomForest time={rf_time:.1f}s, device={tdev}  ")
            else:
                # Fallback to generic metadata when deep models were trained (or RF time missing)
                if tmeta:
                    ep = tmeta.get('epochs')
                    tdev = tmeta.get('device')
                    dur = tmeta.get('duration_seconds')
                    try:
                        lines.append(f"Training: epochs={ep}, device={tdev}, time={float(dur):.1f}s  ")
                    except Exception:
                        lines.append(f"Training: epochs={ep}, device={tdev}  ")
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
                if not metrics:  # Skip missing (e.g., pruned models when only RF trained)
                    continue
                lines.append(
                    f"| {model_key} | {val_n} | {pre_n}→{post_n} | {_fmt(metrics.get('accuracy'))} | {_fmt(metrics.get('f1_weighted'))} | {_fmt(metrics.get('f1_macro'))} | {_fmt(metrics.get('auc_weighted'))} | {_fmt(metrics.get('auc_macro'))} | {_fmt(metrics.get('auc_micro'))} |"
                )
            # Confusion matrices
            cms = []
            for model_key, metrics in mode_block.get('models', {}).items():
                if not metrics:
                    continue
                cm_path = metrics.get('confusion_matrix_path')
                cm_svg = metrics.get('confusion_matrix_path_svg')
                # Prefer SVG if exists on disk
                use_path = None
                if cm_svg and os.path.exists(cm_svg):
                    use_path = cm_svg
                elif cm_path and os.path.exists(cm_path):
                    use_path = cm_path
                if use_path:
                    rel_path = os.path.relpath(use_path, start=reports_dir)
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
                        # Single confusion matrix: center at same width as a half-row item (~48%)
                        lines.append('<div style="display:flex; gap:12px; align-items:flex-start; justify-content:center;">')
                        lines.append('<figure style="flex:0 0 48%; max-width:48%; margin:0;">')
                        lines.append(f'<img src="{rel_path}" style="width:100%; max-width:100%;" />')
                        lines.append(f'<figcaption style="text-align:center; font-size: 13px; margin-top:6px;">Confusion Matrix: {model_key}</figcaption>')
                        lines.append('</figure>')
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
        lines.append(f"레이블: {', '.join(str(ln) for ln in consolidated['label_names'])}\n")
    # 데이터 병합 요약 (원시 파일에서 즉시 계산; 별도 산출물 생성 안 함)
    try:
        raw_dir = os.path.join('data', 'raw')
        p_path = os.path.join(raw_dir, 'merged_info.out')
        a_path = os.path.join(raw_dir, 'merged_added_info.out')

        def _parse_stats_cols(path):
            rows = []
            if not os.path.exists(path):
                return pd.DataFrame(columns=['pair'])
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip() or 'allChr' not in line:
                        continue
                    m = re.search(r"\[(.*?)\]", line)
                    if not m:
                        continue
                    pair = m.group(1).replace(' ', '')
                    tail = line.split('allChr', 1)[1]
                    stats = {'pair': pair}
                    for mm in re.finditer(r'([^:]+?):(\S+)', tail):
                        key = mm.group(1).strip()
                        val = mm.group(2).strip()
                        try:
                            stats[key] = float(val)
                        except ValueError:
                            stats[key] = val
                    rows.append(stats)
            df = pd.DataFrame(rows)
            return df

        df_p = _parse_stats_cols(p_path)
        df_a = _parse_stats_cols(a_path)
        if not df_p.empty or not df_a.empty:
            p_cols = set(df_p.columns) - {'pair'}
            a_cols = set(df_a.columns) - {'pair'}
            common_cols = sorted(p_cols & a_cols)
            added_only = sorted(a_cols - p_cols)
            lines.append("## 데이터 병합\n")
            lines.append("- 원본 통계 파일: merged_info.out + merged_added_info.out  ")
            lines.append(
                f"- 컬럼 요약: 기본={len(p_cols)}; 추가={len(a_cols)}; 중복(겹침)={len(common_cols)}; 추가 전용 유지={len(added_only)}\n"
            )
            if added_only:
                lines.append("- 추가 전용 컬럼: " + ", ".join(added_only) + "\n")
            lines.append("- 겹치는 지표는 중복을 제거합니다(기본 우선, 누락 시 추가에서 보완).\n")
    except Exception:
        pass
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

    # 간단 요약 (의사결정자용)
    try:
        lines.append("### 한눈에 보기\n")
        for scenario_key, scen in consolidated.get('scenarios', {}).items():
            vcd_any = None
            if scen['modes']:
                any_mode_key = next(iter(scen['modes']))
                vcd_any = scen['modes'][any_mode_key].get('val_class_distribution', {})
            total = sum(int(v) for v in vcd_any.values()) if vcd_any else 0
            maj = max(int(v) for v in vcd_any.values()) if vcd_any else 0
            share = (maj / total) if total else 0.0
            best_tuple = None  # (f1w, aucw, mode, model, valN)
            for mode_key, mb in scen.get('modes', {}).items():
                for model_key, metrics in (mb.get('models') or {}).items():
                    if not metrics:
                        continue
                    f1w = metrics.get('f1_weighted') or 0.0
                    aucw = metrics.get('auc_weighted') or 0.0
                    valn = mb.get('val_samples')
                    if (best_tuple is None) or (f1w > best_tuple[0]):
                        best_tuple = (f1w, aucw, mode_key, model_key, valn)
            if best_tuple:
                f1w, aucw, mode_key, model_key, valn = best_tuple
                lines.append(f"- {scenario_key}: 최고 {model_key} ({mode_key}) — F1(가중) {f1w:.2f}, AUC(가중) {aucw:.2f}; 검증 N {valn}; 다수 클래스 비율 {share:.0%}")
        lines.append("")
    except Exception:
        pass
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
                # Enforce two-column layout for scenario assets
                lines.append('<div style="display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap;">')
                for title_txt, img_path in items:
                    lines.append('<figure style="flex:0 0 48%; max-width:48%; margin:0;">')
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
            # RF-only일 때는 RF 학습 시간으로 표기 (딥러닝 에포크/시간은 오해 소지)
            tmeta = mode_block.get('training_meta') or {}
            models_block = mode_block.get('models') or {}
            rf_metrics = models_block.get('RandomForest') or {}
            rf_time = rf_metrics.get('train_duration_seconds')
            only_rf = bool(tmeta.get('only_randomforest'))
            if only_rf and rf_time is not None:
                tdev = tmeta.get('device')
                lines.append(f"학습: RandomForest 시간={rf_time:.1f}초, 디바이스={tdev}  ")
            else:
                if tmeta:
                    ep = tmeta.get('epochs')
                    tdev = tmeta.get('device')
                    dur = tmeta.get('duration_seconds')
                    try:
                        lines.append(f"학습: 에포크={ep}, 디바이스={tdev}, 시간={float(dur):.1f}초  ")
                    except Exception:
                        lines.append(f"학습: 에포크={ep}, 디바이스={tdev}  ")
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
                if not metrics:
                    continue
                lines.append(
                    f"| {model_key} | {val_n} | {pre_n}→{post_n} | {_fmt(metrics.get('accuracy'))} | {_fmt(metrics.get('f1_weighted'))} | {_fmt(metrics.get('f1_macro'))} | {_fmt(metrics.get('auc_weighted'))} | {_fmt(metrics.get('auc_macro'))} | {_fmt(metrics.get('auc_micro'))} |"
                )
            cms = []
            for model_key, metrics in mode_block.get('models', {}).items():
                cm_path = metrics.get('confusion_matrix_path')
                cm_svg = metrics.get('confusion_matrix_path_svg')
                use_path = None
                if cm_svg and os.path.exists(cm_svg):
                    use_path = cm_svg
                elif cm_path and os.path.exists(cm_path):
                    use_path = cm_path
                if use_path:
                    rel_path = os.path.relpath(use_path, start=reports_dir)
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
                        # 단일 혼동 행렬: 반열(약 48%) 크기로 중앙 정렬
                        lines.append('<div style="display:flex; gap:12px; align-items:flex-start; justify-content:center;">')
                        lines.append('<figure style="flex:0 0 48%; max-width:48%; margin:0;">')
                        lines.append(f'<img src="{rel_path}" style="width:100%; max-width:100%;" />')
                        lines.append(f'<figcaption style="text-align:center; font-size: 13px; margin-top:6px;">혼동 행렬: {model_key}</figcaption>')
                        lines.append('</figure>')
                        lines.append('</div>')
                        lines.append("")
                        i += 1
                lines.append("")
    return lines
