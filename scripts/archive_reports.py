#!/usr/bin/env python3
"""Archive kinship reports with automatic naming."""

import os
import shutil
import zipfile
import argparse
from datetime import datetime
from pathlib import Path


def archive_reports(name=None):
    """Create a zip of Korean PDFs from reports/ with renamed files."""
    if name is None:
        name = datetime.now().strftime('%y%m%d')
    
    repo_root = Path(__file__).parent.parent
    reports_dir = repo_root / 'reports'
    output_path = repo_root / f'kinship_reports_KR_{name}.zip'
    
    print(f"Creating {output_path.name}...")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for cM_dir in sorted(reports_dir.glob('cM_*')):
            if not cM_dir.is_dir():
                continue
            
            dataset = cM_dir.name
            kr_pdf = cM_dir / 'results_KR.pdf'
            
            if kr_pdf.exists():
                # Rename: remove_noise_cM_1_results_KR.pdf
                arcname = f'{name}_{dataset}_results_KR.pdf'
                zf.write(kr_pdf, arcname)
                print(f"  Added {arcname}")
    
    size_mb = output_path.stat().st_size / (1024*1024)
    print(f"\n✓ Created: {output_path.name} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Archive kinship reports')
    parser.add_argument('--name', type=str, default=None,
                       help='Prefix for files (default: today\'s date YYMMDD)')
    args = parser.parse_args()
    archive_reports(args.name)
