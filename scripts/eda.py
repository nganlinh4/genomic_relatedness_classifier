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

    # Larger, publication-ready fonts for 2-column report layout
    plt.figure(figsize=(12, 8), dpi=300)
    ax = kinship_counts.plot(kind='bar', color='#4C78A8')
    ax.set_title(f'Kinship Target Distribution ({dataset}, {args.scenario})', fontsize=22, pad=12)
    ax.set_xlabel('Kinship Category', fontsize=18, labelpad=10)
    ax.set_ylabel('Count', fontsize=18, labelpad=10)
    ax.tick_params(axis='x', labelrotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Kinship distribution plot saved to {out_path}")
    print("Value counts:")
    print(kinship_counts)


if __name__ == '__main__':
    main()
