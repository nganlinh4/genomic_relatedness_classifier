import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the merged dataset
df = pd.read_csv('data/processed/merged_cM_1.csv')

# Define features and target
feature_cols = [col for col in df.columns if col not in ['pair', 'kinship']]
X = df[feature_cols]
y = df['kinship']

# Encode target if needed (but RF can handle strings)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit RandomForest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y_encoded)

# Get feature importances
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

# Select top 50 features
top_features = feature_importance_df.head(50)['feature'].tolist()

# Fit scaler on selected features
scaler_selected = StandardScaler()
scaler_selected.fit(X[top_features])

# Save selected features
with open('data/processed/top_features_cM_1.pkl', 'wb') as f:
    pickle.dump(top_features, f)

# Save scaler
with open('data/processed/scaler_cM_1.pkl', 'wb') as f:
    pickle.dump(scaler_selected, f)

# Save feature importance plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df.head(20)['feature'], feature_importance_df.head(20)['importance'])
plt.title('Top 20 Feature Importances (cM_1)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('data/processed/feature_importance_cM_1.png')
# plt.show()

print("Top 50 selected features saved to data/processed/top_features_cM_1.pkl")
print("Scaler saved to data/processed/scaler_cM_1.pkl")
print("Feature importance plot saved to data/processed/feature_importance_cM_1.png")
print("Top 10 features:")
print(feature_importance_df.head(10))