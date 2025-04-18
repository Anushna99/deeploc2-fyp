import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Excel file
file_path = "classified_results.csv"  # Update this with your actual file path
df = pd.read_csv(file_path)

# Convert 'true_label_flags' column from string to lists
df['true_label_flags'] = df['true_label_flags'].apply(eval)  # Converts string lists to Python lists

# Check if there is at least one False in 'true_label_flags'
df['has_false'] = df['true_label_flags'].apply(lambda x: any(not flag for flag in x))

# Create a contingency table (crosstab) for heatmap
heatmap_data = pd.crosstab(df['has_false'], df['result'])

# Plot the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt='d')
plt.xlabel("Classification Result")
plt.ylabel("Has At Least One False in True Label Flags")
plt.title("Heatmap of True Label Flags vs. Classification Certainty")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Save the heatmap as an image
heatmap_path = "heatmap.png"
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")

print(f"Heatmap saved as {heatmap_path}")
