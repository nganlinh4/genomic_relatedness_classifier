import sys
import os
import argparse
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Feature selection per dataset and scenario')
    parser.add_argument('dataset', type=str, help='cM_1, cM_3, cM_6')
    parser.add_argument('--scenario', type=str, choices=['included','noUN'], default='included', help='Scenario: included (default) or noUN')
    args = parser.parse_args()

    dataset = args.dataset
    suffix = '' if args.scenario == 'included' else '_noUN'
    merged_csv = os.path.join('data', 'processed', f'merged_{dataset}{suffix}.csv')

    df = pd.read_csv(merged_csv)
    feature_cols = [col for col in df.columns if col not in ['pair', 'kinship']]
    X = df[feature_cols]
    y = df['kinship']

    # Encode y
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale
    scaler_all = StandardScaler()
    X_scaled = scaler_all.fit_transform(X)

    # Feature importance via RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y_encoded)
    importances = rf.feature_importances_

    feature_importance_df = (
        pd.DataFrame({'feature': feature_cols, 'importance': importances})
        .sort_values('importance', ascending=False)
    )

    # Select top 50 features
    top_features = feature_importance_df.head(50)['feature'].tolist()

    # Fit scaler on selected features
    scaler_selected = StandardScaler()
    scaler_selected.fit(df[top_features])

    # Persist
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    with open(os.path.join('data', 'processed', f'top_features_{dataset}{suffix}.pkl'), 'wb') as f:
        pickle.dump(top_features, f)
    with open(os.path.join('data', 'processed', f'scaler_{dataset}{suffix}.pkl'), 'wb') as f:
        pickle.dump(scaler_selected, f)

    # Plot feature importance into organized directory: reports/<dataset>/assets/<scenario>/
    out_dir = os.path.join('reports', dataset, 'assets', args.scenario)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'feature_importance_{dataset}_{args.scenario}.png')

    top20 = feature_importance_df.head(20)
    plt.figure(figsize=(12, 10), dpi=300)
    bars = plt.barh(top20['feature'][::-1], top20['importance'][::-1], color='#F58518')
    plt.title(f'Top 20 Feature Importances ({dataset}, {args.scenario})', fontsize=22, pad=14)
    plt.xlabel('Importance', fontsize=18, labelpad=10)
    plt.ylabel('Feature', fontsize=18, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)
    plt.gca().invert_yaxis()  # Highest at top
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print(f"Top 50 selected features saved to data/processed/top_features_{dataset}{suffix}.pkl")
    print(f"Scaler saved to data/processed/scaler_{dataset}{suffix}.pkl")
    print(f"Feature importance plot saved to {out_path}")
    print("Top 10 features:")
    print(feature_importance_df.head(10))


if __name__ == '__main__':
    main()
