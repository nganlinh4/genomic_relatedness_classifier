import pandas as pd

df = pd.read_csv('data/processed/merged_cM_1.csv')
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print('Sample rows:')
print(df.head())