import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/eda.py <dataset>")
        print("dataset: cM_1, cM_3, cM_6")
        sys.exit(1)

    dataset = sys.argv[1]
    merged_csv = os.path.join('data', 'processed', f'merged_{dataset}.csv')

    df = pd.read_csv(merged_csv)
    kinship_counts = df['kinship'].value_counts()

    # Save plot to reports directory
    out_dir = os.path.join('reports', dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'kinship_distribution_{dataset}.png')

    plt.figure(figsize=(10, 6))
    kinship_counts.plot(kind='bar')
    plt.title(f'Kinship Target Distribution ({dataset})')
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
