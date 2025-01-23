# import pandas as pd
# import os
from src.utils import *

# def concatenate_csv_files(folder_path, output_file):
#     # Get all CSV files in the folder
#     csv_files = [file for file in os.listdir(folder_path) if file.endswith('results.csv')]
    
#     # Ensure there are at least 5 CSV files
#     if len(csv_files) < 5:
#         print("The folder must contain at least 5 CSV files.")
#         return
    
#     # Read and concatenate the first 5 CSV files
#     df_list = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files[:5]]
#     concatenated_df = pd.concat(df_list, ignore_index=True)

#     # Rename columns for the range from 2nd to 11th (index 1 to 10 in zero-based indexing)
#     column_indices_to_rename = range(1, 12)  # 2nd to 11th columns
#     for idx in column_indices_to_rename:
#         original_name = concatenated_df.columns[idx]
#         concatenated_df.rename(columns={original_name: original_name.replace("pred_", "")}, inplace=True)
    
    
#     # Extract pred_loc and true_loc columns
#     pred_loc_columns = [col for col in concatenated_df.columns if col.startswith("pred_loc_")]
#     true_loc_columns = [col for col in concatenated_df.columns if col.startswith("true_loc_")]

#     # Map pred_loc and true_loc columns to their corresponding labels
#     labels = [col.replace("pred_loc_", "") for col in pred_loc_columns]

#     # Create predicted_label and true_label columns
#     def get_labels(row, columns, label_names):
#         return ", ".join([label for col, label in zip(columns, label_names) if row[col] == 1])

#     concatenated_df["predicted_label"] = concatenated_df.apply(lambda row: get_labels(row, pred_loc_columns, labels), axis=1)
#     concatenated_df["true_label"] = concatenated_df.apply(lambda row: get_labels(row, true_loc_columns, labels), axis=1)

#     # Drop the original pred_loc and true_loc columns
#     concatenated_df.drop(columns=pred_loc_columns + true_loc_columns, inplace=True)
    
#     # Save the concatenated DataFrame to a new CSV
#     concatenated_df.to_csv(output_file, index=False)
#     print(f"Concatenated CSV saved to: {output_file}")
#     return concatenated_df
    

# # Specify folder containing CSV files and output file path
# folder_path = "/home/cseroot/pasindumadusha.20/deeploc2-fyp/outputs/esm1b/ensemble_4"  # Replace with your folder path
# output_file = "/home/cseroot/pasindumadusha.20/deeploc2-fyp/outputs/uncertainity_results/Fast/swissprot/model_5/predictions_with_true_labels.csv"  # Replace with your desired output file name
# output_folder = "/home/cseroot/pasindumadusha.20/deeploc2-fyp/outputs/uncertainity_results/Fast/swissprot/model_5"
# # Run the function
# concatenated_df = concatenate_csv_files(folder_path, output_file)
# calculate_metrics(concatenated_df, output_folder, "swissprot")
# print("Generating calibaration curve...")
# plot_combined_calibration_curve(concatenated_df, output_folder)

# merge_df = merge_prediction_files(selected_folder, required_files, uncertainty_results_path)

# # calculate variance distribution over each classes
# plot_variance_distribution(merge_df, uncertainty_results_path)

import pandas as pd
import numpy as np
import os

def compute_statistics(base_folder, output_file):
    # Initialize an empty list to store DataFrames
    dataframes = []
    
    # Iterate through model folders (model_1 to model_5)
    for i in range(1, 6):  # From model_1 to model_5
        model_folder = os.path.join(base_folder, f"model_{i}")
        file_path = os.path.join(model_folder, "predictions_with_true_labels.csv")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        # Read the CSV file and append it to the list
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    # Get the class column names (ignoring "ACC")
    class_columns = [
        "Membrane", "Cytoplasm", "Nucleus", "Extracellular", 
        "Cell membrane", "Mitochondrion", "Plastid", "Endoplasmic reticulum", 
        "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"
    ]

    # Initialize a dictionary to store results
    results = []

    # Process each sequence (ACC) in the first DataFrame
    for index, acc in enumerate(dataframes[0]["ACC"]):
        row = {"ACC": acc}
        for col in class_columns:
            # Gather the values for the current column from all DataFrames
            values = [df.iloc[index][col] for df in dataframes]
            
            # Compute mean and variance
            mean_val = np.mean(values)
            var_val = np.var(values)

            # Format the results
            row[col] = f"({', '.join([f'{v:.8f}' for v in values])}) mean: {mean_val:.8f}, var: {var_val:.8f}"
        
        # Append the row to the results
        results.append(row)

    # Create a DataFrame from the results
    output_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return output_df

# Specify base folder and output file
base_folder = "/home/cseroot/pasindumadusha.20/deeploc2-fyp/outputs/uncertainity_results/Fast/swissprot"  # Replace with the folder containing model_1 to model_5
output_file = "/home/cseroot/pasindumadusha.20/deeploc2-fyp/outputs/uncertainity_results/Fast/swissprot/merged_predictions_of_ensembles_with_stats.csv"  # Replace with the desired output file name

# Run the function
output_df = compute_statistics(base_folder, output_file)

# calculate variance distribution over each classes
plot_variance_distribution(output_df, base_folder)
print("Calculating metrics for the ensemble results...")
true_labels_csv = '/home/cseroot/pasindumadusha.20/deeploc2-fyp/data_files/multisub_5_partitions_unique.csv'
binary_predictions = get_binary_predictions(output_df, base_folder, true_labels_csv, "swissprot")

calculate_metrics(binary_predictions, base_folder, "swissprot")

print("Generating calibaration curve...")
plot_combined_calibration_curve(binary_predictions, base_folder)
