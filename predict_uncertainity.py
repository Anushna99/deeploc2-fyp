from src.data import *
from src.utils import *
from src.eval_utils import *
from src.embedding import *
from src.metrics import *
import argparse

def check_folder_exists(folder_path):
    """Check if the specified folder exists."""
    if not os.path.exists(folder_path):
        print(f"Error: Folder does not exist at path: {folder_path}")
        return False
    print(f"Folder exists at path: {folder_path}")
    return True

def check_required_files(folder_path, required_files):
    """Check if all required files exist in the specified folder."""
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(folder_path, f))]
    if missing_files:
        print("Some required files are missing in the results folder:")
        for file in missing_files:
            print(f"- {file}")
        return False
    print("All required prediction files are present.")
    return True

<<<<<<< Updated upstream
def prepare_uncertainty_results_path(dataset_name):
    """Prepare the dataset-specific directory inside 'uncertainity_results'."""
    suffix = format_suffix(BATCH_SIZE, REG_LOSS_MULT, SUP_LOSS_MULT)
    uncertainty_results_path = os.path.join("outputs", f"uncertainity_results/hyperparameter_ensemble", dataset_name, suffix)
=======
def prepare_uncertainty_results_path(dataset_name, model_type, suffix):
    """Prepare the dataset-specific and model-specific directory inside 'uncertainity_results'."""
    uncertainty_results_path = os.path.join("outputs", "uncertainity_results/hyperparameter_ensemble", model_type, dataset_name, suffix)
    true_labels_csv = '/home/cseroot/pasindumadusha.20/deeploc2-fyp/original_model_results/data_files/hpa_testset.csv' if dataset_name == 'hpa' else 'pasindumadusha.20/deeploc2-fyp/data_files/multisub_5_partitions_unique.csv'
>>>>>>> Stashed changes
    os.makedirs(uncertainty_results_path, exist_ok=True)
    print(f"Results will be saved in: {uncertainty_results_path}")
    return uncertainty_results_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--dataset", 
        choices=["swissprot", "hpa"], 
        required=True, 
        help="Dataset to use (SwissProt or HPA)."
    )
    args = parser.parse_args()

    suffix = format_suffix(BATCH_SIZE, REG_LOSS_MULT, SUP_LOSS_MULT)
    test_results_saved_folder = {
<<<<<<< Updated upstream
        "swissprot": "outputs/test_swissprot_esm1b",
        "hpa": "outputs/test_hpa_esm1b"
=======
        'Accurate': {
            "swissprot": f"outputs/test_swissprot_prott5/{suffix}",
            "hpa": f"outputs/test_hpa_prott5/{suffix}"
        },
        'Fast': {
            "swissprot": f"outputs/test_swissprot_esm1b/{suffix}",
            "hpa": f"outputs/test_hpa_esm1b/{suffix}"
        }
>>>>>>> Stashed changes
    }

    selected_folder = test_results_saved_folder.get(args.dataset)
    required_files = [f"{i}_1Layer_output_predictions.csv" for i in range(5)]

    if check_folder_exists(selected_folder) and check_required_files(selected_folder, required_files):
        print("proceeding with calculating uncertainity...")

<<<<<<< Updated upstream
        uncertainty_results_path = prepare_uncertainty_results_path(args.dataset)
=======
        uncertainty_results_path, true_labels_csv = prepare_uncertainty_results_path(args.dataset, args.model, suffix)
>>>>>>> Stashed changes

        # Merge prediction files and save the output to the relevant folder
        merge_df = merge_prediction_files(selected_folder, required_files, uncertainty_results_path)

        # calculate variance distribution over each classes
        plot_variance_distribution(merge_df, uncertainty_results_path)

        # calculate metrics
        print("Calculating metrics for the ensemble results...")
        binary_predictions = get_binary_predictions(merge_df, uncertainty_results_path)

        calculate_metrics(binary_predictions, uncertainty_results_path)

        print("Generating calibaration curve...")
        plot_combined_calibration_curve(binary_predictions, uncertainty_results_path)

        




    

    

