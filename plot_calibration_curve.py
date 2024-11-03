import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Load the provided test data
true_data = pd.read_csv('hpa_testset.csv')
predicted_data = pd.read_csv('./outputs/training_results/hpa_test_results_dropout.csv')

# Extract common class names from both datasets
class_names = [col for col in true_data.columns if col in predicted_data.columns]

# Initialize a figure for plotting calibration curves
plt.figure(figsize=(10, 8))

# Initialize variables to store best thresholds and accuracies
best_thresholds = {}
accuracy_results = {}

# Iterate through each class
for class_name in class_names:
    # Extract true labels and predicted probabilities for the current class
    y_true = true_data[class_name]
    y_pred_prob = predicted_data[class_name]

    # Find the best threshold by maximizing MCC
    best_mcc = -1
    best_threshold = 0.5  # Default threshold if MCC is not improved
    for threshold in np.arange(0.1, 0.9, 0.1):
        y_pred = (y_pred_prob >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    # Store the best threshold for this class
    best_thresholds[class_name] = best_threshold

    # Calculate accuracy using the best threshold
    y_pred_best = (y_pred_prob >= best_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred_best) * 100  # Accuracy in percentage
    accuracy_results[class_name] = accuracy

    # Compute the calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_prob, n_bins=10)

    # Plot the calibration curve for the current class
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f"{class_name} (Thresh={best_threshold:.2f})")

# Plot the perfect calibration line
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')

# Set labels and title
plt.xlabel('Mean predicted value')
plt.ylabel('Fraction of positives')
plt.legend(title='Class')
plt.title('Calibration Curves for DeepLoc2 Classes with Optimized Thresholds')

# Save the plot as a PNG file in the outputs folder
plt.savefig('./outputs/calibration_curves_all_classes_best_mcc.png')

# Display the plot
plt.show()

# Print the best threshold and accuracy for each class
print("Best Threshold and Accuracy for each class:")
for class_name in class_names:
    print(f"{class_name}: Best Threshold = {best_thresholds[class_name]:.2f}, Accuracy = {accuracy_results[class_name]:.2f}%")
