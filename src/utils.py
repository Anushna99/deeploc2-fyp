from src.model import *
from src.data import DataloaderHandler
import pickle
from transformers import T5EncoderModel, T5Tokenizer, logging
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt

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
    output_file = os.path.join(output_folder, "merged_predictions_of_ensembles_with_stats.csv")
    final_df.to_csv(output_file, index=False)
    print(f"Merged predictions of ensembles with statistics saved to {output_file}")

def plot_variance_distribution(merged_csv_file, output_folder):
    """Plot the variance distribution for each class based on the merged CSV file and save summary statistics."""
    # Load the merged CSV file
    df = pd.read_csv(merged_csv_file)
    
    # Ensure the output folder and graphs subfolder exist
    graphs_folder = os.path.join(output_folder, "graphs")
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
        plt.title(f'Variance Distribution for {column}')
        plt.xlabel('Variance')
        plt.ylabel('Frequency')

        # Replace '/' with '_' in column name for file name safety
        safe_column_name = column.replace("/", "_")
        output_path = os.path.join(graphs_folder, f'{safe_column_name}_variance_distribution.png')
        plt.savefig(output_path)
        plt.close()
        print(f'Saved variance distribution plot for {column} at {output_path}')

    # Save the summary statistics as a CSV file
    stats_csv_path = os.path.join(output_folder, "variance_statistics.csv")
    variance_stats.to_csv(stats_csv_path, index=False)
    print(f"Variance statistics saved to {stats_csv_path}")
