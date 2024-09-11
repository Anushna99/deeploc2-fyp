import pandas as pd
import os
from src.constants import *
import numpy as np
from scipy.stats import entropy

def process_csv_file(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Define the categories (localization names)
    categories = CATEGORIES

    # Create the 'true prediction' column
    df['true_prediction'] = df.apply(lambda row: ','.join([category for category in categories if row[f'true_loc_{category}']]), axis=1)

    # Create the 'predicted location' column
    df['predicted_location'] = df.apply(lambda row: ','.join([category for category in categories if row[f'pred_loc_{category}']]), axis=1)

    # Select the columns in the desired order
    columns_order = ['ACC', 'true_prediction', 'predicted_location'] + [f'pred_{category}' for category in categories]
    df = df[columns_order]

    # Save the processed DataFrame to the output file
    df.to_csv(output_file, index=False)
    print(f"Processed file saved as {output_file}")

def process_all_files(input_folder, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each CSV file in the input folder
    for i, file_name in enumerate(os.listdir(input_folder)):
        if file_name.endswith('output_predictions_with_true_values.csv'):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, f"output_predictions_model_{i}.csv")
            process_csv_file(input_file, output_file)

def aggregate_predictions(input_folder, output_file):
    # Define the categories (localization names)
    categories = CATEGORIES

    # Initialize a dictionary to store aggregated data
    aggregated_data = {}

    # Process each CSV file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.startswith('output_predictions_model') :
            
            input_file = os.path.join(input_folder, file_name)
            df = pd.read_csv(input_file)

            # Iterate through each row in the DataFrame
            for _, row in df.iterrows():
                acc = row['ACC']

                if acc not in aggregated_data:
                    aggregated_data[acc] = {
                        'true_prediction': row['true_prediction'],
                        'predicted_location': row['predicted_location'],
                        **{f'pred_{category}': [] for category in categories}
                    }

                # Append the prediction values to the corresponding lists
                for category in categories:
                    aggregated_data[acc][f'pred_{category}'].append(row[f'pred_{category}'])

    # Create a new DataFrame to store the aggregated results
    aggregated_df = pd.DataFrame.from_dict(aggregated_data, orient='index').reset_index().rename(columns={'index': 'ACC'})

    # Ensure that each list of predictions is converted to a comma-separated string
    for category in categories:
        aggregated_df[f'pred_{category}'] = aggregated_df[f'pred_{category}'].apply(lambda x: ','.join(map(str, x)))

    # Save the aggregated DataFrame to the output file
    aggregated_df.to_csv(output_file, index=False)
    print(f"Aggregated predictions saved as {output_file}")

def aggregate_predictions_and_add_labels(input_folder, output_file, aggregated_predictions_file):
    # Define the categories (localization names)
    categories = CATEGORIES

    # Initialize a dictionary to store aggregated data
    aggregated_data = {}

    # Process each CSV file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('output_predictions.csv') and not file_name.startswith('aggregated_predictions'):
            
            input_file = os.path.join(input_folder, file_name)
            df = pd.read_csv(input_file)

            # Iterate through each row in the DataFrame
            for _, row in df.iterrows():
                acc = row['ACC']

                if acc not in aggregated_data:
                    aggregated_data[acc] = {f'pred_{category}': [] for category in categories}

                # Append the prediction values to the corresponding lists
                for category in categories:
                    aggregated_data[acc][f'pred_{category}'].append(row[category])

    # Create a new DataFrame to store the aggregated results
    aggregated_df = pd.DataFrame.from_dict(aggregated_data, orient='index').reset_index().rename(columns={'index': 'ACC'})

    # Ensure that each list of predictions is converted to a comma-separated string
    for category in categories:
        aggregated_df[f'pred_{category}'] = aggregated_df[f'pred_{category}'].apply(lambda x: ','.join(map(str, x)))

    # Load the aggregated_predictions.csv to get true_prediction and predicted_location
    labels_df = pd.read_csv(aggregated_predictions_file)
    labels_df = labels_df[['ACC', 'true_prediction', 'predicted_location']]

    # Merge the labels_df with aggregated_df based on ACC
    final_df = pd.merge(aggregated_df, labels_df, on='ACC', how='left')

    # Reorder columns to have ACC, true_prediction, predicted_location, followed by prediction columns
    columns_order = ['ACC', 'true_prediction', 'predicted_location'] + [f'pred_{category}' for category in categories]
    final_df = final_df[columns_order]

    # Save the final DataFrame to the output file
    final_df.to_csv(output_file, index=False)
    print(f"Final aggregated predictions with labels saved as {output_file}")

def calculate_uncertainty(input_file, output_file):
    # Load the aggregated predictions
    df = pd.read_csv(input_file)

    # Define the categories (localization names)
    categories = [col.replace('pred_', '') for col in df.columns if col.startswith('pred_')]

    # Process each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize lists to store mean predictions and variance uncertainty
        mean_predictions = []
        variance_uncertainties = []

        for category in categories:
            # Ensure the value is a string, then split and convert to float
            value = row[f'pred_{category}']
            if isinstance(value, str):
                pred_values = list(map(float, value.split(',')))
            else:
                pred_values = [float(value)]  # Handle the case where the value is a single float

            # Calculate the mean prediction for the current category
            mean_pred = np.mean(pred_values)
            mean_predictions.append(mean_pred)

            # Calculate variance uncertainty
            variance_uncertainty = np.var(pred_values)
            variance_uncertainties.append(variance_uncertainty)

            # Update the DataFrame with the formatted mean prediction and variance uncertainty
            df.at[index, f'pred_{category}'] = f"{mean_pred:.8f} (var:{variance_uncertainty:.8f})"

    # Save the updated DataFrame to the output file
    df.to_csv(output_file, index=False)
    print(f"Mean predictions and variance uncertainties saved to {output_file}")


input_folder = 'outputs/esm1b'
output_folder = 'outputs/training_results'
## create output files combining true values
# process_all_files(input_folder, output_folder)
aggregated_predictions_file = os.path.join(output_folder, 'aggregated_predictions_&_true_values.csv') 
## save a one file combinging all the true labels and predictions
# aggregate_predictions(input_folder=output_folder, output_file=aggregated_predictions_file)

# # # The output file path
final_output_file = os.path.join(output_folder, 'final_aggregated_predictions_with_labels.csv')

# # Run the function
# aggregate_predictions_and_add_labels(input_folder, final_output_file, aggregated_predictions_file)

input_file_for_uncertainity = final_output_file
output_file_for_uncertainity_results = os.path.join(output_folder, 'uncertainity_values.csv')
calculate_uncertainty(input_file_for_uncertainity, output_file_for_uncertainity_results)

