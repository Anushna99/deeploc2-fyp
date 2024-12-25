import pandas as pd

# Load your CSV
df = pd.read_csv('hpa_for_thresholding.csv')

# Ensure the predicted_label column is treated as a string
df['predicted_label'] = df['predicted_label'].astype(str)

# Define a function to filter columns based on the predicted label and keep necessary columns
def filter_columns(row):
    # Handle cases where predicted_label might be NaN or None
    if pd.isna(row['predicted_label']):
        return pd.Series()
    predicted_labels = row['predicted_label'].split(', ')
    relevant_cols = [col for col in df.columns if any(label in col for label in predicted_labels)]
    # Include predicted_label and true_label columns in the output
    relevant_cols.extend(['predicted_label', 'true_label'])
    return pd.Series({col: row[col] for col in relevant_cols})

# Apply the function to each row
filtered_df = df.apply(filter_columns, axis=1)

# Save the filtered DataFrame
filtered_df.to_csv('filtered_file_2.csv', index=False)

print("Filtered CSV with necessary columns saved successfully.")
