# import pandas as pd
# import numpy as np
# import os
from src.utils import *

# output_dir = "validation_set_tuning/"
# os.makedirs(output_dir, exist_ok=True)  

# model_name = "Fast"
# dataset_name = "swissprot"
# true_labels_csv = '/home/cseroot/pasindumadusha.20/deeploc2-fyp/data_files/multisub_5_partitions_unique.csv'


# # Load the CSV files
# parent_df = pd.read_csv('/home/cseroot/pasindumadusha.20/deeploc2-fyp/data_files/multisub_5_partitions_unique.csv')
# child_df = pd.read_csv('/home/cseroot/pasindumadusha.20/deeploc2-fyp/outputs/esm1b/ensemble_0/model_0_results.csv')
# df = child_df
# df = df.iloc[:, :12]
# df.columns = [df.columns[0]] + [col.replace("pred_", "") for col in df.columns[1:]]

# # Compute predictive entropy
# probs = df.iloc[:, 1:12].values
# df['Predictive_Entropy'] = -np.sum(probs * np.log(probs + 1e-8), axis=1)

# # Define threshold sets (as tuples of (low%, high%))
# threshold_sets = {
#     "25_75": (25, 75),
#     "30_70": (30, 70),
#     "20_80": (20, 80)
# }

# # Categorize function
# def categorize_uncertainty(value, low_th, high_th):
#     if value < low_th:
#         return "Low"
#     elif value > high_th:
#         return "High"
#     else:
#         return "Moderate"

# # Loop through each threshold set
# for name, (low_pct, high_pct) in threshold_sets.items():
#     low_th = np.percentile(df["Predictive_Entropy"], low_pct)
#     high_th = np.percentile(df["Predictive_Entropy"], high_pct)
    
#     df_copy = df.copy()
#     df_copy["Uncertainty_Level"] = df_copy["Predictive_Entropy"].apply(
#         lambda x: categorize_uncertainty(x, low_th, high_th)
#     )

#      # Count the number of samples in each category
#     counts = df_copy["Uncertainty_Level"].value_counts().to_dict()
#     low_count = counts.get("Low", 0)
#     moderate_count = counts.get("Moderate", 0)
#     high_count = counts.get("High", 0)

#     # Write to CSV with custom header
#     output_file = os.path.join(output_dir, f"entropy_categorized_{name}.csv")
#     with open(output_file, 'w') as f:
#         f.write(f"# Low Threshold: {low_th:.6f}\n")
#         f.write(f"# High Threshold: {high_th:.6f}\n")
#         f.write(f"# Counts - Low: {low_count}, Moderate: {moderate_count}, High: {high_count}\n")
#         df_copy.to_csv(f, index=False)


#     print(f"Saved: entropy_categorized_{name}.csv (Thresholds: {low_pct}th - {high_pct}th)")
#     # Split into DataFrames
#     for level, df_part in zip(["low", "moderate", "high"], [
#         df_copy[df_copy["Uncertainty_Level"] == "Low"],
#         df_copy[df_copy["Uncertainty_Level"] == "Moderate"],
#         df_copy[df_copy["Uncertainty_Level"] == "High"]
#     ]):
#         # Save file
#         csv_path = os.path.join(output_dir, f"entropy_{name}_{level}.csv")
#         df_part.to_csv(csv_path, index=False)

#         # Log
#         print(f"Saved: {csv_path}")

#         # Call prediction and metric functions
#         tag = f"{name}_{level}"  # e.g., Fast_25_75_low_swissprot

#         print(f"Running predictions and metrics for: {tag}")
#         binary_predictions = get_binary_predictions_uncertain(
#             df_part,
#             output_dir,
#             true_labels_csv=true_labels_csv,
#             model=model_name,
#             uncertainty=tag,
#             minimal=True
#         )
#         calculate_metrics(binary_predictions, output_dir, dataset_name, uncertainty=tag)

# import pandas as pd
# import numpy as np
# import os

# # Setup
# output_dir = "validation_set_tuning/"
# os.makedirs(output_dir, exist_ok=True)

# log_file = os.path.join(output_dir, "grid_search_log.txt")
# open(log_file, "w").close()  # Clear previous log if exists

# model_name = "Fast"
# dataset_name = "swissprot"
# true_labels_csv = '/home/cseroot/pasindumadusha.20/deeploc2-fyp/data_files/multisub_5_partitions_unique.csv'

# # Load the prediction CSV
# child_df = pd.read_csv('/home/cseroot/pasindumadusha.20/deeploc2-fyp/outputs/esm1b/ensemble_0/model_0_results.csv')
# df = child_df.iloc[:, :12]
# df.columns = [df.columns[0]] + [col.replace("pred_", "") for col in df.columns[1:]]

# # Compute predictive entropy
# probs = df.iloc[:, 1:12].values
# df['Predictive_Entropy'] = -np.sum(probs * np.log(probs + 1e-8), axis=1)

# # Narrowed range for grid search (around 20–80 percentiles)
# entropy_values = df["Predictive_Entropy"]
# percentile_low_range = np.arange(18, 23, 1)   # 18%, 19%, ..., 22%
# percentile_high_range = np.arange(77, 83, 1)  # 77%, 78%, ..., 82%

# best_micro_f1 = -1
# best_thresholds = None

# def categorize_uncertainty(value, low_th, high_th):
#     if value < low_th:
#         return "Low"
#     elif value > high_th:
#         return "High"
#     else:
#         return "Moderate"

# # Grid search loop
# for low_pct in percentile_low_range:
#     for high_pct in percentile_high_range:
#         if low_pct >= high_pct:
#             continue

#         low_th = np.percentile(entropy_values, low_pct)
#         high_th = np.percentile(entropy_values, high_pct)

#         df_copy = df.copy()
#         df_copy["Uncertainty_Level"] = df_copy["Predictive_Entropy"].apply(
#             lambda x: categorize_uncertainty(x, low_th, high_th)
#         )

#         df_low = df_copy[df_copy["Uncertainty_Level"] == "Low"]
#         if df_low.empty:
#             continue

#         tag = f"gs_low_{low_pct}_{high_pct}"
#         print(f"Testing thresholds: low={low_th:.4f}, high={high_th:.4f} ({tag})")

#         try:
#             binary_predictions = get_binary_predictions_uncertain(
#                 df_low,
#                 output_dir,
#                 true_labels_csv=true_labels_csv,
#                 model=model_name,
#                 uncertainty=tag,
#                 minimal=True
#             )

#             metrics_df = calculate_metrics(binary_predictions, output_dir, dataset_name, uncertainty=tag)
#             micro_f1 = metrics_df.loc[metrics_df["Metric"] == "MicroF1", "Value"].values[0]

#             # Logging
#             log_line = f"{tag} | Low Th: {low_th:.6f}, High Th: {high_th:.6f} | Micro-F1: {micro_f1:.4f}\n"
#             with open(log_file, "a") as logf:
#                 logf.write(log_line)

#             print(log_line.strip())

#             # Update best score
#             if micro_f1 > best_micro_f1:
#                 best_micro_f1 = micro_f1
#                 best_thresholds = (low_th, high_th)

#         except Exception as e:
#             print(f"Error at {tag}: {e}")
#             continue

# # Final best result
# print(f"\n✅ Best Micro-F1: {best_micro_f1:.4f} with thresholds {best_thresholds}")

# Configuration
output_dir = "validation_set_tuning_optimal/"
os.makedirs(output_dir, exist_ok=True)

model_name = "Fast"
dataset_name = "swissprot"
true_labels_csv = '/home/cseroot/pasindumadusha.20/deeploc2-fyp/data_files/multisub_5_partitions_unique.csv'

# Optimal thresholds
low_th = np.float64(1.9626403829134766)
high_th = np.float64(2.7960446073763996)
threshold_name = "optimal"

# Load predictions
child_df = pd.read_csv('/home/cseroot/pasindumadusha.20/deeploc2-fyp/outputs/esm1b/ensemble_0/model_0_results.csv')
df = child_df.iloc[:, :12]
df.columns = [df.columns[0]] + [col.replace("pred_", "") for col in df.columns[1:]]

# Compute predictive entropy
probs = df.iloc[:, 1:12].values
df['Predictive_Entropy'] = -np.sum(probs * np.log(probs + 1e-8), axis=1)

# Categorize
def categorize_uncertainty(value, low_th, high_th):
    if value < low_th:
        return "Low"
    elif value > high_th:
        return "High"
    else:
        return "Moderate"

# Apply categorization
df_copy = df.copy()
df_copy["Uncertainty_Level"] = df_copy["Predictive_Entropy"].apply(
    lambda x: categorize_uncertainty(x, low_th, high_th)
)

# Save full categorized CSV with header
counts = df_copy["Uncertainty_Level"].value_counts().to_dict()
with open(os.path.join(output_dir, f"entropy_categorized_{threshold_name}.csv"), "w") as f:
    f.write(f"# Low Threshold: {low_th:.6f}\n")
    f.write(f"# High Threshold: {high_th:.6f}\n")
    f.write(f"# Counts - Low: {counts.get('Low', 0)}, Moderate: {counts.get('Moderate', 0)}, High: {counts.get('High', 0)}\n")
    df_copy.to_csv(f, index=False)

print(f"Saved: entropy_categorized_{threshold_name}.csv")

# Split and process each uncertainty level
for level, df_part in zip(["low", "moderate", "high"], [
    df_copy[df_copy["Uncertainty_Level"] == "Low"],
    df_copy[df_copy["Uncertainty_Level"] == "Moderate"],
    df_copy[df_copy["Uncertainty_Level"] == "High"]
]):
    csv_path = os.path.join(output_dir, f"entropy_{threshold_name}_{level}.csv")
    df_part.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Run predictions + metrics
    tag = f"{threshold_name}_{level}"
    print(f"Running predictions and metrics for: {tag}")
    binary_predictions = get_binary_predictions_uncertain(
        df_part,
        output_dir,
        true_labels_csv=true_labels_csv,
        model=model_name,
        uncertainty=tag,
        minimal=True
    )
    calculate_metrics(binary_predictions, output_dir, dataset_name, uncertainty=tag)