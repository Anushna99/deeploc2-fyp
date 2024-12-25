import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Load your CSV file
df = pd.read_csv('filtered_file_2.csv')

# Split data into thresholding and testing sets
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)  # 70% for training, 30% for testing
df_test.to_csv('initial_test_set.csv', index=False)
def extract_and_compare_confidences(row):
    predicted_classes = str(row['predicted_label']).split(', ')
    true_classes = set(str(row['true_label']).split(', '))
    confidences = []
    true_label_flags = []

    for cls in predicted_classes:
        if cls in row and isinstance(row[cls], str):
            var_string = str(row[cls]).split('var:')[-1].strip('()')
            variance = float(var_string)
            confidence = 1 - variance
            is_in_true = cls in true_classes
            confidences.append(confidence)
            true_label_flags.append(is_in_true)

    return pd.Series([confidences, true_label_flags], index=['confidences', 'true_label_flags'])

df_train[['confidences', 'true_label_flags']] = df_train.apply(extract_and_compare_confidences, axis=1)

def calculate_cost(tau, w1=1.0, w2=1.0):
    rejected = df_train['confidences'].explode() < tau
    accepted = ~rejected
    true_labels = df_train['true_label_flags'].explode()
    FP = ((true_labels == False) & rejected).sum()
    FN = ((true_labels == True) & accepted).sum()
    return w1 * FP + w2 * FN

tau_values = np.linspace(df_train['confidences'].explode().min(), df_train['confidences'].explode().max(), 100)
costs = [calculate_cost(tau) for tau in tau_values]
min_cost_index = np.argmin(costs)
optimal_tau = tau_values[min_cost_index]
# Apply optimal tau to the test dataset
df_test[['confidences', 'true_label_flags']] = df_test.apply(extract_and_compare_confidences, axis=1)
df_test['result'] = df_test['confidences'].apply(lambda x: 'certain' if all(conf >= optimal_tau for conf in x) else 'uncertain')

# Save the results to a new CSV file
df_test.to_csv('classified_results.csv', index=False)
print(f"Optimal tau for confidence: {optimal_tau}")
print("Results have been saved to 'classified_results.csv'.")

print(df_test.head())
import matplotlib.pyplot as plt
all_confidences = df_test['confidences'].explode().astype(float).dropna()
plt.figure(figsize=(10, 6))
plt.hist(all_confidences, bins=50, alpha=0.75, color='blue')
plt.title('Histogram of Confidence Values')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
plt.savefig('confidence_histogram.png')
plt.close()  # Close the plot to free up memory