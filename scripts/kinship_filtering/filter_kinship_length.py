import pandas as pd

# Read the kinship_UN.tsv file
kinship_file = '/home/moagen/linh/genomic_relatedness_classifier/data/raw/kinship_UN.tsv'
df = pd.read_csv(kinship_file, sep='\t')

# Identify rows where LENGTH_CM >= 50
rows_to_remove = df[df['LENGTH_CM'] >= 50]

# Get unique PAIR_IDs from removed rows
removed_pairs = rows_to_remove['PAIR_ID'].unique()

# Filter to keep only rows where LENGTH_CM < 50
df_filtered = df[df['LENGTH_CM'] < 50]

# Save filtered results
output_filtered = '/home/moagen/linh/genomic_relatedness_classifier/data/raw/kinship_UN_filtered_length50.tsv'
df_filtered.to_csv(output_filtered, sep='\t', index=False)
print(f"Filtered results saved to: {output_filtered}")
print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(df_filtered)}")
print(f"Removed rows: {len(rows_to_remove)}")

# Save list of removed pairs
output_removed_pairs = '/home/moagen/linh/genomic_relatedness_classifier/data/raw/kinship_UN_removed_pairs_length50.txt'
with open(output_removed_pairs, 'w') as f:
    f.write("Pairs with LENGTH_CM >= 50 (REMOVED)\n")
    f.write("=" * 50 + "\n\n")
    for pair in sorted(removed_pairs):
        pair_data = rows_to_remove[rows_to_remove['PAIR_ID'] == pair]
        max_length = pair_data['LENGTH_CM'].max()
        f.write(f"{pair} (Max LENGTH_CM: {max_length:.6f})\n")

print(f"\nRemoved pairs list saved to: {output_removed_pairs}")
print(f"\nUnique pairs removed: {len(removed_pairs)}")
print(f"\nRemoved pairs:")
for pair in sorted(removed_pairs):
    max_length = rows_to_remove[rows_to_remove['PAIR_ID'] == pair]['LENGTH_CM'].max()
    print(f"  {pair} (Max LENGTH_CM: {max_length:.6f})")
