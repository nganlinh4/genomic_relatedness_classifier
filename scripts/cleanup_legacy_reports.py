import os
import argparse


def cleanup_dataset(dataset: str):
    base = os.path.join('reports', dataset)
    if not os.path.isdir(base):
        return
    removed = 0
    for fname in list(os.listdir(base)):
        if not fname.lower().endswith('.png'):
            continue
        if (
            fname.startswith('kinship_distribution_') or
            fname.startswith('feature_importance_') or
            fname.startswith('confusion_matrix_')
        ):
            try:
                os.remove(os.path.join(base, fname))
                removed += 1
            except OSError:
                pass
    print(f"Cleaned {removed} legacy plot files in {base}")


def main():
    parser = argparse.ArgumentParser(description='Remove legacy report plot files from reports/<dataset> root')
    parser.add_argument('dataset', type=str, help='Dataset id (e.g., cM_1, cM_3, cM_6)')
    args = parser.parse_args()
    cleanup_dataset(args.dataset)


if __name__ == '__main__':
    main()
