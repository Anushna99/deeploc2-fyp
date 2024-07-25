import pandas as pd
import os

# Data from the user's output
data = {
    "Metric": [
        "NumLabels", "NumLabelsTest", "ACC_membrane", "MCC_membrane", "ACC_subloc", 
        "HammLoss_subloc", "Jaccard_subloc", "MicroF1_subloc", "MacroF1_subloc", 
        "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", 
        "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", 
        "Peroxisome"
    ],
    "Mean": [
        1.27, 1.28, 0.88, 0.70, 0.52, 0.93, 0.67, 0.71, 0.63, 0.61, 0.66, 0.85, 0.64, 
        0.73, 0.89, 0.50, 0.21, 0.36, 0.45
    ],
    "StdDev": [
        0.03, 0.03, 0.01, 0.01, 0.02, 0.00, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.01, 
        0.04, 0.02, 0.02, 0.04, 0.06, 0.04
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the output directory and file path
output_dir = "/outputs/training_results"
output_file = os.path.join(output_dir, "results_from_swissprot.csv")

# Create the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
df.to_csv(output_file, index=False)

print(f"CSV file saved as {output_file}")
