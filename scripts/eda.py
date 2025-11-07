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

    # Save plot to reports directory
    out_dir = os.path.join('reports', dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'kinship_distribution_{dataset}_{args.scenario}.png')

    plt.figure(figsize=(10, 6))
    kinship_counts.plot(kind='bar')
    plt.title(f'Kinship Target Distribution ({dataset}, {args.scenario})')
    plt.xlabel('Kinship Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Kinship distribution plot saved to {out_path}")
    print("Value counts:")
    print(kinship_counts)


if __name__ == '__main__':
    main()
