import numpy as np
import pandas as pd
import re
from src.utils import *
import os

output_dir = "uncertainty_estimation_optimal/"
os.makedirs(output_dir, exist_ok=True) 
# Load the combined CSV file
file_path = "/home/cseroot/pasindumadusha.20/deeploc2-fyp/outputs/uncertainty_results/hpa/merged_predictions_of_ensembles_with_stats.csv" 
df = pd.read_csv(file_path)

# Function to extract numerical values from the formatted string
def extract_values(cell):
    match = re.search(r"mean:\s*([\d\.]+),\s*var:\s*([\d\.]+)", str(cell))
    if match:
        mean = float(match.group(1))
        variance = float(match.group(2))
        return mean, variance
    return None, None  # If pattern not found

# Initialize a dictionary to store processed data
formatted_results = {"ACC": df["ACC"]}  # Keep the protein IDs unchanged

# Initialize lists to store global uncertainty values
all_means = []
all_variances = []

# Process each class label separately
for col in df.columns[1:]:  # Skip the 'ACC' column
    means = []
    variances = []
    
    # Extract values for each row
    for cell in df[col]:
        mean, var = extract_values(cell)
        means.append(mean)
        variances.append(var)
    
    # Convert to NumPy arrays for efficient calculations
    means = np.array(means)
    variances = np.array(variances)

    # Store per-class mean and variance
    all_means.append(means)
    all_variances.append(variances)

    # Store the formatted values (for per-class details)
    formatted_results[col] = [
        f"mean: {m:.6f}, var: {v:.6f}" for m, v in zip(means, variances)
    ]

# Convert all mean and variance lists into NumPy arrays
all_means = np.array(all_means).T  # Shape (N, 10)
all_variances = np.array(all_variances).T  # Shape (N, 10)

# Compute **Global Predictive Entropy** (Total Uncertainty) → One value per ID
predictive_entropy = -np.sum(all_means * np.log(all_means + 1e-8), axis=1)

# Add new columns for global uncertainty values
formatted_results["Predictive_Entropy"] = predictive_entropy

# Convert results to DataFrame
formatted_results_df = pd.DataFrame(formatted_results)

# Save results to CSV
formatted_results_df.to_csv(f"{output_dir}uncertainty_results_with_predictions.csv", index=False)

# Print a sample of the results
print(formatted_results_df.head())

df = formatted_results_df

# Define thresholds for uncertainty (e.g., 25th and 75th percentiles)
low_uncertainty_threshold = np.float64(1.9626403829134766)
high_uncertainty_threshold = np.float64(2.7960446073763996)

# Categorize uncertainty levels
def categorize_uncertainty(value):
    if value < low_uncertainty_threshold:
        return "Low"
    elif value > high_uncertainty_threshold:
        return "High"
    else:
        return "Moderate"

df["Uncertainty_Level"] = df["Predictive_Entropy"].apply(categorize_uncertainty)
# Count each uncertainty level
uncertainty_counts = df["Uncertainty_Level"].value_counts().to_dict()
low_count = uncertainty_counts.get("Low", 0)
moderate_count = uncertainty_counts.get("Moderate", 0)
high_count = uncertainty_counts.get("High", 0)

# Save with summary header
main_csv_path = os.path.join(output_dir, "uncertainty_results_with_optimal_thresholds.csv")
with open(main_csv_path, "w") as f:
    f.write(f"# Low Threshold: {low_uncertainty_threshold:.6f}\n")
    f.write(f"# High Threshold: {high_uncertainty_threshold:.6f}\n")
    f.write(f"# Counts - Low: {low_count}, Moderate: {moderate_count}, High: {high_count}\n")
    df.to_csv(f, index=False)

print(f"Saved full categorized results with header to: {main_csv_path}")

# Split into three separate dataframes
df_low = df[df["Uncertainty_Level"] == "Low"]
df_moderate = df[df["Uncertainty_Level"] == "Moderate"]
df_high = df[df["Uncertainty_Level"] == "High"]

# Save to separate CSV files
df_low.to_csv(f"{output_dir}low_uncertainty.csv", index=False)
df_moderate.to_csv(f"{output_dir}moderate_uncertainty.csv", index=False)
df_high.to_csv(f"{output_dir}high_uncertainty.csv", index=False)

# Print summary of split data
print(f"Low Uncertainty IDs: {len(df_low)} saved to low_uncertainty.csv")
print(f"Moderate Uncertainty IDs: {len(df_moderate)} saved to moderate_uncertainty.csv")
print(f"High Uncertainty IDs: {len(df_high)} saved to high_uncertainty.csv")


binary_predictions = get_binary_predictions_uncertain(df_moderate, output_dir, true_labels_csv= '/home/cseroot/pasindumadusha.20/deeploc2-fyp/original_model_results/data_files/hpa_testset.csv', model='Fast', uncertainty="moderate")
calculate_metrics(binary_predictions, output_dir, "hpa", "moderate")

# plot_reliability_curve("/home/cseroot/pasindumadusha.20/deeploc2-fyp/uncertainty_estimation/predictions_with_true_labels_uncertainty_moderate.csv", "/home/cseroot/pasindumadusha.20/deeploc2-fyp/uncertainty_estimation/")

