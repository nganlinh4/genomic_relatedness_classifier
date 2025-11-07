import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='EDA for dataset and scenario')
    parser.add_argument('dataset', type=str, help='cM_1, cM_3, cM_6')
    parser.add_argument('--scenario', type=str, choices=['included','noUN'], default='included', help='Scenario: included (default) or noUN')
    args = parser.parse_args()

    dataset = args.dataset
    suffix = '' if args.scenario == 'included' else '_noUN'
    merged_csv = os.path.join('data', 'processed', f'merged_{dataset}{suffix}.csv')

    df = pd.read_csv(merged_csv)
    kinship_counts = df['kinship'].value_counts()

    # Save plot to organized reports directory: reports/<dataset>/assets/<scenario>/
    out_dir = os.path.join('reports', dataset, 'assets', args.scenario)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'kinship_distribution_{dataset}_{args.scenario}.png')

    # Stronger, global font scaling so plots stay legible in a 2-column layout
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 28,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.dpi': 200,
    })
    # Slightly smaller canvas but much bigger fonts -> better readability when downscaled
    plt.figure(figsize=(10, 7))
    ax = kinship_counts.plot(kind='bar', color='#4C78A8')
    ax.set_title(f'Kinship Target Distribution ({dataset}, {args.scenario})', pad=12)
    ax.set_xlabel('Kinship Category', labelpad=10)
    ax.set_ylabel('Count', labelpad=10)
    ax.tick_params(axis='x', labelrotation=45)
    # Add value labels on bars for clarity
    try:
        ax.bar_label(ax.containers[0], fmt='%d', padding=3, fontsize=16)
    except Exception:
        pass
    ax.tick_params(axis='y')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print(f"Kinship distribution plot saved to {out_path}")
    print("Value counts:")
    print(kinship_counts)


if __name__ == '__main__':
    main()
