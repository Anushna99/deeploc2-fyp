import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Load the provided test data
true_data = pd.read_csv('hpa_testset.csv')
predicted_data = pd.read_csv('./outputs/results_hpa_testset_20240620_013643.csv/results_20240619-204146.csv')

# Define the class names
class_names = ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane',
               'Mitochondrion', 'Plastid', 'Endoplasmic reticulum',
               'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome']

# Initialize a figure for plotting
plt.figure(figsize=(10, 8))

# Iterate through each class
for class_name in class_names:
    # Extract true labels and predicted probabilities for the current class
    y_true = true_data[class_name]
    y_pred_prob = predicted_data[class_name]

    # Compute the calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_prob, n_bins=10)

    # Plot the calibration curve for the current class
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=class_name)

# Plot the perfect calibration line
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')

# Set labels and title
plt.xlabel('Mean predicted value')
plt.ylabel('Fraction of positives')
plt.legend(title='Class')
plt.title('Calibration Curves for DeepLoc2 Classes')

# Save the plot as a PNG file in the outputs folder
plt.savefig('./outputs/calibration_curves_all_classes.png')

# Display the plot
plt.show()