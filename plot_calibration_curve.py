import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import os

def extract_true_labels(true_labels_csv):
    """Extract true labels for each protein and format as a binary array for each class."""
    true_df = pd.read_csv(true_labels_csv)
    class_columns = true_df.columns[5:-1]  # Exclude 'sid', 'Lengths', and 'fasta' columns

    true_labels_dict = {}
    for _, row in true_df.iterrows():
        protein_id = row['ACC']
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
    """Plot calibration curves for each class and a separate overall calibration curve with calibration metrics."""
    
    # Define a fixed color map for each class
    color_map = {
    'Cytoplasm': 'dodgerblue',         
    'Nucleus': 'crimson',              
    'Extracellular': 'forestgreen',    
    'Cell membrane': 'mediumorchid',   
    'Mitochondrion': 'darkorange',     
    'Plastid': 'darkgoldenrod',        
    'Endoplasmic reticulum': 'teal',   
    'Lysosome/Vacuole': 'slategray',   
    'Golgi apparatus': 'mediumvioletred', 
    'Peroxisome': 'gold'               
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
    
    brier_scores = {}  # Store Brier scores for each class
    
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
        
        # Calculate Brier score for each class
        brier_score = brier_score_loss(class_true, class_pred)
        brier_scores[class_name] = brier_score
        
        # Plot calibration curve for the class with a fixed color
        plt.plot(prob_pred, prob_true, marker='o', label=f"{class_name} (Brier: {brier_score:.3f})", color=color_map.get(class_name, 'gray'))
    
    # Plot the diagonal for perfect calibration
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    
    # Set plot labels and title for class-specific plot
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("True Frequency")
    plt.title("Calibration Plot for Each Class - swissprot Original Model - Accurate")
    plt.legend(loc="best")
    
    # Save the class-specific plot
    class_plot_path = os.path.join('.', "original_model_calibration_plot_classes_swissprot_testset_accurate.png")
    plt.tight_layout()
    plt.savefig(class_plot_path)
    plt.close()
    
    print(f"Class-specific calibration plot saved to {class_plot_path}")
    
    # 2. Plot overall calibration curve
    plt.figure(figsize=(10, 8))
    
    # Calculate overall calibration curve
    prob_true, prob_pred = calibration_curve(all_true_labels, all_pred_probs, n_bins=n_bins, strategy='uniform')
    
    # Calculate overall Brier score
    overall_brier_score = brier_score_loss(all_true_labels, all_pred_probs)
    
    # Plot overall calibration curve
    plt.plot(prob_pred, prob_true, marker='o', linestyle='--', color='black', label=f'Overall Calibration (Brier: {overall_brier_score:.3f})')
    
    # Plot the diagonal for perfect calibration
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    
    # Set plot labels and title for overall plot
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("True Frequency")
    plt.title("Overall Calibration Plot - swissprot Original Model - Accurate")
    
    # Add Brier score as a text annotation below the plot
    plt.figtext(0.5, -0.05, f"Overall Brier Score: {overall_brier_score:.3f}", ha="center", fontsize=12)
    plt.legend(loc="best")
    
    # Save the overall plot
    overall_plot_path = os.path.join('.', "original_model_calibration_plot_overall_swissprot_testset_accurate.png")
    plt.tight_layout()
    plt.savefig(overall_plot_path)
    plt.close()
    
    print(f"Overall calibration plot saved to {overall_plot_path}")

true_data = 'data_files/multisub_5_partitions_unique.csv'
prediction_data = 'results_deeploc_swissprot_clipped4K.csv'
plot_calibration_curve(true_data, prediction_data)
