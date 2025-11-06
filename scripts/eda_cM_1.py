import pandas as pd
import matplotlib.pyplot as plt

# Load the merged dataset
df = pd.read_csv('data/processed/merged_cM_1.csv')

# Visualize kinship distribution
kinship_counts = df['kinship'].value_counts()

plt.figure(figsize=(10, 6))
kinship_counts.plot(kind='bar')
plt.title('Kinship Target Distribution (cM_1)')
plt.xlabel('Kinship Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig('data/processed/kinship_distribution_cM_1.png')
# plt.show()  # Commented out to avoid hanging in terminal

print("Kinship distribution plot saved to data/processed/kinship_distribution_cM_1.png")
print("Value counts:")
print(kinship_counts)