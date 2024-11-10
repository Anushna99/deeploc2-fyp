import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import os

def extract_true_labels(true_labels_csv):
    """Extract true labels for each protein and format as a binary array for each class."""
    true_df = pd.read_csv(true_labels_csv)
    class_columns = true_df.columns[1:-2]  # Exclude 'sid', 'Lengths', and 'fasta' columns

    true_labels_dict = {}
    for _, row in true_df.iterrows():
        protein_id = row['sid']
        true_locations = [1 if row[class_name] == 1 else 0 for class_name in class_columns]
        true_labels_dict[protein_id] = true_locations

    return true_labels_dict, class_columns.tolist()

def extract_predictions(predictions_csv, class_columns):
    """Extract predicted probabilities for each protein and convert to binary predictions."""
    pred_df = pd.read_csv(predictions_csv)
    
    # Create a dictionary to store predictions
    pred_labels_dict = {}
    for _, row in pred_df.iterrows():
        protein_id = row['Protein_ID']
        pred_probs = [row[class_name] for class_name in class_columns]
        pred_labels_dict[protein_id] = pred_probs

    return pred_labels_dict

def plot_calibration_curve(true_labels_csv, predictions_csv, n_bins=10):
    """Plot calibration curves for each class and a separate overall calibration curve."""
    
    # Define a fixed color map for each class
    color_map = {
    'Cytoplasm': 'dodgerblue',         # A bright but soft blue
    'Nucleus': 'crimson',              # Softer than pure red, with good contrast
    'Extracellular': 'forestgreen',    # A deep green that's more subdued
    'Cell membrane': 'mediumorchid',   # A softer purple
    'Mitochondrion': 'darkorange',     # Muted orange with good visibility
    'Plastid': 'darkgoldenrod',        # A rich gold/brown tone
    'Endoplasmic reticulum': 'teal',   # Soft but distinctive blue-green
    'Lysosome/Vacuole': 'slategray',   # Neutral gray with a hint of blue
    'Golgi apparatus': 'mediumvioletred', # Softer magenta
    'Peroxisome': 'gold'               # Bright yellow with strong contrast
    }
    
    # Extract true labels and class names
    true_labels_dict, class_labels = extract_true_labels(true_labels_csv)
    
    # Extract predicted probabilities
    pred_labels_dict = extract_predictions(predictions_csv, class_labels)

    # Prepare data for overall calibration
    all_true_labels = []
    all_pred_probs = []
    
    # 1. Plot individual calibration curves for each class
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_labels):
        class_true = []
        class_pred = []
        
        for protein_id in true_labels_dict.keys():
            if protein_id in pred_labels_dict:
                class_true.append(true_labels_dict[protein_id][i])
                class_pred.append(pred_labels_dict[protein_id][i])

        # Append to overall data for combined calibration curve
        all_true_labels.extend(class_true)
        all_pred_probs.extend(class_pred)
        
        # Calculate calibration curve for the class
        prob_true, prob_pred = calibration_curve(class_true, class_pred, n_bins=n_bins, strategy='uniform')
        
        # Plot calibration curve for the class with a fixed color
        plt.plot(prob_pred, prob_true, marker='o', label=class_name, color=color_map.get(class_name, 'gray'))
    
    # Plot the diagonal for perfect calibration
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    
    # Set plot labels and title for class-specific plot
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("True Frequency")
    plt.title("Calibration Plot for Each Class - HPA Original Model")
    plt.legend(loc="best")
    
    # Save the class-specific plot
    class_plot_path = os.path.join('.', "original_model_calibration_plot_classes_hpa_testset.png")
    plt.tight_layout()
    plt.savefig(class_plot_path)
    plt.close()
    
    print(f"Class-specific calibration plot saved to {class_plot_path}")
    
    # 2. Plot overall calibration curve
    plt.figure(figsize=(10, 8))
    
    # Calculate overall calibration curve
    prob_true, prob_pred = calibration_curve(all_true_labels, all_pred_probs, n_bins=n_bins, strategy='uniform')
    
    # Plot overall calibration curve
    plt.plot(prob_pred, prob_true, marker='o', linestyle='--', color='black', label='Overall Calibration')
    
    # Plot the diagonal for perfect calibration
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    
    # Set plot labels and title for overall plot
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("True Frequency")
    plt.title("Overall Calibration Plot - HPA Original Model")
    plt.legend(loc="best")
    
    # Save the overall plot
    overall_plot_path = os.path.join('.', "original_model_calibration_plot_overall_hpa_testset.png")
    plt.tight_layout()
    plt.savefig(overall_plot_path)
    plt.close()
    
    print(f"Overall calibration plot saved to {overall_plot_path}")

true_data = 'hpa_testset.csv'
prediction_data = 'outputs/results_hpa_testset_20240620_013643.csv/results_20240619-204146.csv'
plot_calibration_curve(true_data, prediction_data)
