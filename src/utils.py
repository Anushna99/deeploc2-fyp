from src.model import *
from src.data import DataloaderHandler
import pickle
from transformers import T5EncoderModel, T5Tokenizer, logging
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, matthews_corrcoef
from sklearn.calibration import calibration_curve
import itertools

class ModelAttributes:
    '''
    A class named ModelAttributes is defined to store various attributes related to the model.
    Parameters Being Returned
        model_type: The type of model being used (FAST or ACCURATE).
        class_type: The class representing the model architecture (e.g., ESM1bFrozen for the FAST model).
        alphabet: The set of characters or tokens used to encode sequences, loaded from a pickle file for FAST or from a pre-trained tokenizer for ACCURATE.
        embedding_file: The path to the embeddings file specific to the model type.
        save_path: The directory where model checkpoints will be saved.
        outputs_save_path: The directory where output predictions will be saved.
        clip_len: The length to which input sequences will be clipped for the model.
        embed_len: The length of the embeddings used by the model.
        
        clip_len: This parameter determines the maximum length to which input sequences will be clipped. 
        Sequences longer than this length will be truncated, and shorter sequences will be padded. 
        Adjusting this value can help the model focus on a specific length of sequences,
        potentially improving performance for datasets with sequences of similar lengths.

        embed_len: This parameter defines the length of the embeddings that represent the sequences.
        The embedding length is crucial as it affects the dimensionality of the input to the model. 
        A higher embedding length can capture more features but might increase computational complexity.
    '''
    def __init__(self, 
                 model_type: str,
                 class_type: pl.LightningModule, 
                 alphabet, 
                 embedding_file: str, 
                 dataset: str,
                 save_path: str,
                 outputs_save_path: str,
                 clip_len: int,
                 embed_len: int) -> None:
        self.model_type = model_type
        self.class_type = class_type 
        self.alphabet = alphabet
        self.embedding_file = embedding_file
        self.dataset = dataset
        self.save_path = save_path
        if not os.path.exists(f"{self.save_path}"):
            os.makedirs(f"{self.save_path}")
        self.ss_save_path = os.path.join(self.save_path, "signaltype")
        if not os.path.exists(f"{self.ss_save_path}"):
            os.makedirs(f"{self.ss_save_path}")

        self.outputs_save_path = outputs_save_path

        if not os.path.exists(f"{outputs_save_path}"):
            os.makedirs(f"{outputs_save_path}")
        self.clip_len = clip_len
        self.embed_len = embed_len
        

def get_train_model_attributes(model_type):
    '''
    This function returns an instance of ModelAttributes based on the specified model_type.
    '''
    if model_type == FAST:
        # opens the file named ESM1b_alphabet.pkl in binary read mode ("rb"). 
        # The with statement ensures that the file is properly closed after its contents are read, even if an error occurs.
        # and deserialize the file and saved to python obj.
        with open("models/ESM1b_alphabet.pkl", "rb") as f:
            alphabet = pickle.load(f)
        return ModelAttributes(
            model_type,
            ESM1bFrozen, # this is model architecture for esmb1.
            alphabet,
            EMBEDDINGS[FAST]["embeds"],
            "swissprot",
            "models/models_esm1b",
            "outputs/esm1b/",
            1022,
            1280
        )
    elif model_type == ACCURATE:
        alphabet = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
        
        return ModelAttributes(
            model_type,
            ProtT5Frozen,
            alphabet,
            EMBEDDINGS[ACCURATE]["embeds"], 
            "swissprot",           
            "models/models_prott5",
            "outputs/prott5/",
            4000,
            1024
        )
    else:
        raise Exception("wrong model type provided expected Fast,Accurate got", model_type)
    
def get_test_model_attributes(model_type, data):
    '''
    New function specifically for testing, supporting both SwissProt and HPA datasets.
    '''
    if model_type == FAST:
        with open("models/ESM1b_alphabet.pkl", "rb") as f:
            alphabet = pickle.load(f)
        
        # Switch between SwissProt and HPA embeddings for Fast model
        if data == "swissprot":
            embedding_file = EMBEDDINGS[FAST]["embeds"]
            save_path = "models/models_test_swissprot_esm1b"
            outputs_save_path = "outputs/test_swissprot_esm1b/"
            dataset = data
        elif data == "hpa":
            embedding_file = EMBEDDINGS[TEST_ESM]["embeds"]
            save_path = "models/models_test_hpa_esm1b"
            outputs_save_path = "outputs/test_hpa_esm1b/"
            dataset = data
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        return ModelAttributes(
            model_type="Fast",
            class_type=ESM1bFrozen,
            alphabet=alphabet,
            embedding_file=embedding_file,
            dataset=data,
            save_path=save_path,
            outputs_save_path=outputs_save_path,
            clip_len=1022,
            embed_len=1280
        )

    elif model_type == ACCURATE:
        alphabet = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)

        # Switch between SwissProt and HPA embeddings for Accurate model
        if data == "swissprot":
            embedding_file = EMBEDDINGS[ACCURATE]["embeds"]
            save_path = "models/models_test_swissprot_prott5"
            outputs_save_path = "outputs/test_swissprot_prott5/"
            dataset = data
        elif data == "hpa":
            embedding_file = EMBEDDINGS[TEST_PROTT5]["embeds"]
            save_path = "models/models_test_hpa_prott5"
            outputs_save_path = "outputs/test_hpa_prott5/"
            dataset = data
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        return ModelAttributes(
            model_type="Accurate",
            class_type=ProtT5Frozen,
            alphabet=alphabet,
            embedding_file=embedding_file,
            dataset = data,
            save_path=save_path,
            outputs_save_path=outputs_save_path,
            clip_len=4000,
            embed_len=1024
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

def save_fasta_to_csv(fasta_dict, outputs_save_path, type):
    data = []
    for protein_id, sequence in fasta_dict.items():
        data.append([protein_id, sequence])
    
    # Define column names
    column_names = ['Protein_ID', 'Sequence']
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=column_names)
    
    output_file = os.path.join(outputs_save_path, f'fasta_readings_{type}.csv')
    
    # Ensure the output directory exists
    os.makedirs(outputs_save_path, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"FASTA sequences saved to {output_file}")

def format_suffix(batch_size, reg_loss_mult, sup_loss_mult):
    # Format batch size directly
    batch_suffix = f"B{batch_size}"
    
    # Format reg_loss_mult and sup_loss_mult according to specified rules
    reg_suffix = f"R{str(reg_loss_mult).replace('0.', '')}"  # Removes '0.' for compact representation
    sup_suffix = f"S{str(sup_loss_mult).replace('0.', '')}"  # Same logic for sup_loss_mult

    return f"{batch_suffix}_{reg_suffix}_{sup_suffix}"

suffix = format_suffix(BATCH_SIZE, REG_LOSS_MULT, SUP_LOSS_MULT)

def merge_prediction_files(folder_path, required_files, output_folder):
    """Merge prediction files from a folder, calculate mean and variance, and save the merged CSV."""
    merged_data = {}

    # Read each file and merge by ACC
    for file in required_files:
        file_path = os.path.join(folder_path, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col="ACC")
            for col in df.columns:
                if col not in merged_data:
                    merged_data[col] = []
                merged_data[col].append(df[col])

    # Prepare the ACC column separately to ensure it is first in the final DataFrame
    acc_index = merged_data['Membrane'][0].index
    results = {'ACC': acc_index}

    # Calculate mean and variance, then add formatted results for each class to the DataFrame
    for col, data in merged_data.items():
        combined = pd.concat(data, axis=1)
        results[col] = [
            f"({', '.join([f'{v:.8f}' for v in row.values])}) mean: {np.mean(row.values):.8f}, var: {np.var(row.values):.8f}"
            for _, row in combined.iterrows()
        ]

    # Save merged DataFrame with calculated statistics to output folder
    final_df = pd.DataFrame(results)
    output_file = os.path.join(output_folder, f"merged_predictions_of_ensembles_with_stats_{suffix}.csv")
    # Write the hyperparameter configuration as the first line of the CSV
    with open(output_file, 'w') as f:
        f.write(f"# Hyperparameter Configuration: BATCH_SIZE={BATCH_SIZE}, REG_LOSS_MULT={REG_LOSS_MULT}, SUP_LOSS_MULT={SUP_LOSS_MULT}\n")
        final_df.to_csv(f, index=False)
    
    print(f"Merged predictions of ensembles with statistics saved to {output_file}")

    return final_df

def plot_variance_distribution(df, output_folder):
    """Plot the variance distribution for each class based on the merged CSV file and save summary statistics."""
    
    # Ensure the output folder and graphs subfolder exist
    graphs_folder = os.path.join(output_folder, f"graphs_{suffix}")
    os.makedirs(graphs_folder, exist_ok=True)
    
    # DataFrame to store mean and std deviation of variance for each class
    variance_stats = pd.DataFrame(columns=['Category', 'Mean Variance', 'Std Deviation'])
    
    # Loop over each class (excluding the 'ACC' column)
    for column in df.columns[1:]:
        # Extract variance values for each class by parsing the column
        variance_values = df[column].apply(lambda x: float(x.split("var: ")[1][:-1]))

        # Calculate mean and standard deviation of the variances
        mean_variance = variance_values.mean()
        std_deviation = variance_values.std()
        
        # Append the stats to the DataFrame using concat
        new_row = pd.DataFrame({
            'Category': [column],
            'Mean Variance': [mean_variance],
            'Std Deviation': [std_deviation]
        })
        variance_stats = pd.concat([variance_stats, new_row], ignore_index=True)
        
        # Plot the variance distribution
        plt.figure(figsize=(8, 6))
        plt.hist(variance_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.suptitle(
            f'Variance Distribution for {column}',
            fontsize=14, y=1.05
        )
        plt.title(
            f"Hyperparameter Configuration:\n BATCH_SIZE={BATCH_SIZE}, REG_LOSS_MULT={REG_LOSS_MULT}, SUP_LOSS_MULT={SUP_LOSS_MULT}",
            fontsize=10
        )
        plt.xlabel('Variance')
        plt.ylabel('Frequency')

        # Replace '/' with '_' in column name for file name safety
        safe_column_name = column.replace("/", "_")
        output_path = os.path.join(graphs_folder, f'{safe_column_name}_variance_distribution_{suffix}.png')
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        plt.savefig(output_path)
        plt.close()
        print(f'Saved variance distribution plot for {column} at {output_path}')

    # Save the summary statistics as a CSV file
    stats_csv_path = os.path.join(output_folder, f"variance_statistics_{suffix}.csv")
    with open(stats_csv_path, 'w') as f:
        f.write(f"# Hyperparameter Configuration: BATCH_SIZE={BATCH_SIZE}, REG_LOSS_MULT={REG_LOSS_MULT}, SUP_LOSS_MULT={SUP_LOSS_MULT}\n")
        variance_stats.to_csv(f, index=False)

    print(f"Variance statistics saved to {stats_csv_path}")

esm1b_label_thresholds = np.array([0.45380859, 0.46953125, 0.52753906, 0.64638672, 
                            0.52368164, 0.63730469, 0.65859375, 0.62783203, 
                            0.56484375, 0.66777344, 0.71679688])
prott5_label_threshold = np.array([0.45717773, 0.47612305, 0.50136719, 0.61728516, 0.56464844, 
                                   0.62197266, 0.63945312, 0.60898438, 0.58476562, 0.64941406, 0.73642578])
class_labels = CATEGORIES

def extract_true_labels(true_labels_csv):
    """
    Extract true labels from the provided CSV file and return as a dictionary.
    Each ACC will map to its corresponding true locations as a list of strings.
    """
    # Load the true labels CSV
    true_df = pd.read_csv(true_labels_csv)
    
    # Define the class columns in the true labels CSV file
    class_columns = CATEGORIES

    # Ensure all required columns exist, filling missing ones with `0`s
    for col in class_columns:
        if col not in true_df.columns:
            true_df[col] = 0  # Fill missing class columns with 0s
    
    # Create a dictionary to store true labels
    true_labels_dict = {}
    
    # Iterate over each row and extract true labels based on binary values
    for _, row in true_df.iterrows():
        acc = row['sid']
        true_locations = [class_columns[i] for i, val in enumerate(row[class_columns]) if val == 1]
        true_labels_dict[acc] = ', '.join(true_locations) if true_locations else "None"
    
    return true_labels_dict

def get_binary_predictions(merged_df, output_folder, label_thresholds=esm1b_label_thresholds, true_labels_csv='/home/pasindumadusha_20/deeploc2-fyp/hpa_testset.csv'):
    # Initialize a list to store the final results with the desired structure
    results = []

    # Loop over each row to calculate mean values and predicted labels
    for idx, row in merged_df.iterrows():
        acc = row["ACC"]
        row_data = {"ACC": acc}  # Initialize row data with ACC
        
        # Initialize a list to store the predicted classes
        predicted_labels = []

        # Loop over each class to apply threshold and extract mean values
        for i, class_name in enumerate(class_labels):
            # Calculate mean prediction value for the current class
            mean_value = np.mean([float(val) for val in row[class_name].split(" mean: ")[0][1:-1].split(",")])
            row_data[class_name] = mean_value  # Add mean value to the row
            
            # Apply threshold to decide if this class is predicted
            if mean_value >= label_thresholds[i]:
                predicted_labels.append(class_name)
        
        # Join predicted class names with commas and store them in `predicted_label`
        row_data["predicted_label"] = ", ".join(predicted_labels) if predicted_labels else "None"
        
        # Append the row data to the results list
        results.append(row_data)

    # Convert the results list to a DataFrame
    binary_df = pd.DataFrame(results)

    print(binary_df.head(10))  # Display the first 10 rows for verification
    
    # Extract true labels from the CSV
    true_labels_dict = extract_true_labels(true_labels_csv)

    # Map the true labels to the binary_df based on the ACC column
    binary_df['true_label'] = binary_df['ACC'].map(true_labels_dict)

    print(binary_df.head(10))

    output_path = os.path.join(output_folder, f"predictions_with_true_labels_{suffix}.csv")
     # Write the hyperparameter configuration as the first line of the CSV
    with open(output_path, 'w') as f:
        f.write(f"# Hyperparameter Configuration: BATCH_SIZE={BATCH_SIZE}, REG_LOSS_MULT={REG_LOSS_MULT}, SUP_LOSS_MULT={SUP_LOSS_MULT}\n")
        binary_df.to_csv(f, index=False)
    
    print(f"Binary predictions with mean values and true labels saved to: {output_path}")
    return binary_df

def calculate_metrics(data_df, output_folder):
    # Define class labels excluding the absent ones
    filtered_class_labels = [label for label in class_labels if label not in ["Membrane", "Extracellular", "Plastid"]]

    # Initialize a dictionary to store metrics
    metrics = {
        "Metric": ["Subset Accuracy", "Jaccard", "MicroF1", "MacroF1"] + [f"MCC_{label}" for label in filtered_class_labels],
        "Value": []
    }

    # Helper function to convert labels to binary arrays for each class
    def multilabel_to_binary_array(labels, all_labels):
        return [1 if label in labels else 0 for label in all_labels]

    # Convert 'true_label' and 'predicted_label' columns to binary arrays
    data_df['true_binary'] = data_df['true_label'].apply(lambda x: multilabel_to_binary_array(x.split(', '), filtered_class_labels))
    data_df['predicted_binary'] = data_df['predicted_label'].apply(lambda x: multilabel_to_binary_array(x.split(', '), filtered_class_labels))
    
    true_binary_matrix = np.array(data_df['true_binary'].to_list())
    predicted_binary_matrix = np.array(data_df['predicted_binary'].to_list())

    # Subset accuracy (exact match ratio)
    subset_accuracy = np.mean(np.all(true_binary_matrix == predicted_binary_matrix, axis=1))
    metrics["Value"].append(subset_accuracy)

    # Jaccard Index (samples average for multilabel)
    jaccard = jaccard_score(true_binary_matrix, predicted_binary_matrix, average='samples')
    metrics["Value"].append(jaccard)

    # Micro and Macro F1-Scores for multilabel
    micro_f1 = f1_score(true_binary_matrix, predicted_binary_matrix, average='micro')
    macro_f1 = f1_score(true_binary_matrix, predicted_binary_matrix, average='macro')
    metrics["Value"].extend([micro_f1, macro_f1])

    # Calculate MCC for each present class
    for i, class_name in enumerate(filtered_class_labels):
        mcc = matthews_corrcoef(true_binary_matrix[:, i], predicted_binary_matrix[:, i])
        metrics["Value"].append(mcc)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Save the results to a CSV file
    output_path = os.path.join(output_folder, f"metrics_table_{suffix}.csv")
    # Write the hyperparameter configuration as the first line of the CSV
    with open(output_path, 'w') as f:
        f.write(f"# Hyperparameter Configuration: BATCH_SIZE={BATCH_SIZE}, REG_LOSS_MULT={REG_LOSS_MULT}, SUP_LOSS_MULT={SUP_LOSS_MULT}\n")
        metrics_df.to_csv(f, index=False)
    
    print(f"Metrics table saved to: {output_path}")

        
def plot_combined_calibration_curve(data_df, output_folder, n_bins=10):
    """
    Plot a combined calibration curve for all classes in one plot.
    
    Parameters:
        data_df (pd.DataFrame): DataFrame containing mean predicted probabilities and true labels for each class.
        output_folder (str): Path to save the calibration plot.
        n_bins (int): Number of bins for calibration.
    """
    plt.figure(figsize=(10, 8))

    # Loop through each class and calculate calibration data
    for class_name in class_labels:
        # Extract mean predicted probabilities and true labels for the class
        prob_col = class_name
        true_col = "true_label"

        # Convert the true label to binary format for each class
        data_df[f'{class_name}_true_binary'] = data_df[true_col].apply(lambda x: 1 if class_name in x else 0)

        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(data_df[f'{class_name}_true_binary'], data_df[prob_col], n_bins=n_bins, strategy='uniform')

        # Plot calibration curve
        plt.plot(prob_pred, prob_true, marker='o', label=class_name)

    # Plot the diagonal for perfect calibration
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")

    # Set plot labels and title
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("True Frequency")
    plt.suptitle(
        "Combined Calibration Plot for All Classes",
        fontsize=14, y=1.05
    )
    plt.title(
        f"Hyperparameter Configuration:\n BATCH_SIZE={BATCH_SIZE}, REG_LOSS_MULT={REG_LOSS_MULT}, SUP_LOSS_MULT={SUP_LOSS_MULT}",
        fontsize=10
    )
    plt.legend(loc="best")
    
    # Save the plot
    output_path = os.path.join(output_folder, f"combined_calibration_plot_{suffix}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    plt.close()

    print(f"Combined calibration plot saved to {output_path}")

def iterate_hyperparams(batch_sizes, sup_loss_mults, reg_loss_mults):
    """
    Generator function to iterate over combinations of hyperparameters and assign
    them to BATCH_SIZE, SUP_LOSS_MULT, and REG_LOSS_MULT.
    
    Yields:
        BATCH_SIZE, SUP_LOSS_MULT, REG_LOSS_MULT: A combination of the parameters
    """
    for batch_size, sup_loss_mult, reg_loss_mult in itertools.product(batch_sizes, sup_loss_mults, reg_loss_mults):
        yield batch_size, sup_loss_mult, reg_loss_mult
