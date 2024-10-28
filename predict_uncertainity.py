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

def prepare_uncertainty_results_path(dataset_name):
    """Prepare the dataset-specific directory inside 'uncertainity_results'."""
    uncertainty_results_path = os.path.join("outputs", "uncertainity_results", dataset_name)
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

    test_results_saved_folder = {
        "swissprot": "outputs/test_swissprot_esm1b",
        "hpa": "outputs/test_hpa_esm1b"
    }

    selected_folder = test_results_saved_folder.get(args.dataset)
    required_files = [f"{i}_1Layer_output_predictions.csv" for i in range(5)]

    if check_folder_exists(selected_folder) and check_required_files(selected_folder, required_files):
        print("proceeding with calculating uncertainity...")

        uncertainty_results_path = prepare_uncertainty_results_path(args.dataset)

        # Merge prediction files and save the output to the relevant folder
        merge_prediction_files(selected_folder, required_files, uncertainty_results_path)

        merged_csv_path = os.path.join(uncertainty_results_path, "merged_predictions_of_ensembles_with_stats.csv")
        # calculate variance distribution over each classes
        plot_variance_distribution(merged_csv_path, uncertainty_results_path)




    

    

