import os
import sys
import json

# Ensure repository root (project root) on sys.path for intra-package imports.
# This script lives at <root>/scripts/build_report.py so one parent is project root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from scripts.utils.report_utils import discover_results, consolidate, build_markdown_en, build_markdown_kr
except ModuleNotFoundError:
    # Fallback if 'scripts' isn't treated as a package, try relative utils import
    try:
        from utils.report_utils import discover_results, consolidate, build_markdown_en, build_markdown_kr  # type: ignore
    except ModuleNotFoundError:
        raise


def main():
    """Build a consolidated report for a given dataset.

    Usage: python scripts/build_report.py <dataset> [--prune]
    If --prune is supplied, per-mode evaluation JSONs for the dataset are deleted
    after consolidation to reduce file sprawl.
    """
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_report.py <dataset> [--prune]")
        print("dataset: cM_1, cM_3, cM_6")
        sys.exit(1)

    dataset = sys.argv[1]
    prune = '--prune' in sys.argv

    # Discover scenario+mode evaluation JSON results
    discovered = discover_results(dataset)

    out_dir = os.path.join('reports', dataset)
    os.makedirs(out_dir, exist_ok=True)

    if not discovered:
        existing = os.path.join(out_dir, 'results.json')
        if os.path.exists(existing):
            with open(existing, 'r') as f:
                consolidated = json.load(f)
            print("Using existing consolidated results.json (no new evaluation files found).")
        else:
            print("No evaluation results found and no existing consolidated report.")
            sys.exit(1)
    else:
        consolidated = consolidate(discovered, dataset)
        json_path = os.path.join(out_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(consolidated, f, indent=2)
        print(f"Wrote consolidated results to {json_path}")

    # Build English and Korean markdown reports
    md_path_en = os.path.join(out_dir, 'results.md')
    lines_en = build_markdown_en(consolidated, out_dir)
    with open(md_path_en, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines_en))
    print(f"Wrote Markdown report to {md_path_en}")

    md_path_kr = os.path.join(out_dir, 'results_KR.md')
    lines_kr = build_markdown_kr(consolidated, out_dir)
    with open(md_path_kr, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines_kr))
    print(f"Wrote Korean Markdown report to {md_path_kr}")

    # Optional pruning of evaluation JSONs used for consolidation
    if prune and discovered:
        processed_dir = 'data/processed'
        removed = 0
        for path, scenario, mode, _ in discovered:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        print(f"Pruned {removed} evaluation JSON files for dataset {dataset}.")

    # Generate PDFs using md-to-pdf if available
    try:
        import shutil as _shutil
        import subprocess as _subp
        npx_cmd = _shutil.which('npx') or _shutil.which('npx.cmd')
        if npx_cmd:
            # English (default output naming by md-to-pdf)
            _subp.run([npx_cmd, 'md-to-pdf', os.path.join(out_dir, 'results.md')], check=False)
            # Korean
            _subp.run([npx_cmd, 'md-to-pdf', os.path.join(out_dir, 'results_KR.md')], check=False)
            print("PDFs generated via md-to-pdf (EN and KR) without explicit --output flags.")
        else:
            print("npx/md-to-pdf not found on PATH; skipping PDF generation. Install Node and md-to-pdf to enable.")
    except Exception as e:
        print(f"PDF generation skipped due to error: {e}")


if __name__ == '__main__':
    main()
