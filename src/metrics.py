from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss, matthews_corrcoef, confusion_matrix, roc_auc_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report
from sklearn import metrics
import pickle
from src.constants import *
from src.utils import *
import numpy as np
import pandas as pd
import os
import json


# taken from https://www.kaggle.com/cpmpml/optimizing-probabilities-for-best-mcc
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)
        
def get_best_threshold_mcc(y_true, y_prob):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0

    y_pred = (y_prob >= best_proba).astype(int)
    score = matthews_corrcoef(y_true, y_pred)
    # print(score, best_mcc)
    # plt.plot(mccs)
    return best_proba

def get_optimal_threshold(output_df, data_df):
    '''
    To find the threshold that maximizes the difference between the 
    True Positive Rate (TPR) and False Positive Rate (FPR) on the Receiver Operating Characteristic (ROC) curve.
    The optimal threshold is chosen where the difference between TPR and FPR is maximized, balancing sensitivity and specificity
    Finds the optimal threshold for each category based on the difference between TPR and FPR.
    '''
    test_df = data_df.merge(output_df)
    
    predictions = np.stack(test_df["preds"].to_numpy())
    actuals = np.stack(test_df["Target"].to_numpy())
    
    optimal_thresholds = np.zeros((11,))
    for i in range(11):
        fpr, tpr, thresholds = metrics.roc_curve(actuals[:, i], predictions[:, i])
        optimal_idx = np.argmax(tpr - fpr)
        optimal_thresholds[i] = thresholds[optimal_idx]

    return optimal_thresholds

def get_optimal_threshold_pr(output_df, data_df):
    '''
    To find the threshold that maximizes the F1 score (a balance between precision and recall) on the Precision-Recall (PR) curve.
    The optimal threshold is chosen where the F1 score, which is the harmonic mean of precision and recall, is maximized.
    '''
    test_df = data_df.merge(output_df)
    
    predictions = np.stack(test_df["preds"].to_numpy())
    actuals = np.stack(test_df["Target"].to_numpy())
    
    optimal_thresholds = np.zeros((11,))
    for i in range(11):
        pr, re, thresholds = metrics.precision_recall_curve(actuals[:, i], predictions[:, i])
        fscores = (2 * pr * re) / (pr + re)
        optimal_idx = np.argmax(fscores)
        optimal_thresholds[i] = thresholds[optimal_idx]

    return optimal_thresholds

import numpy as np

def get_optimal_threshold_mcc(output_df, data_df):
    '''
    To find the threshold that maximizes the Matthews Correlation Coefficient (MCC), which measures the quality of binary classifications.
    '''
    test_df = data_df.merge(output_df)  
    print('output df', output_df)
    print('testdf', test_df)
    
    predictions = np.stack(test_df["preds"].to_numpy())
    actuals = np.stack(test_df["Target"].to_numpy())
    
    optimal_thresholds = np.zeros((11,))
    for i in range(11):
        optimal_thresholds[i] = get_best_threshold_mcc(actuals[:, i], predictions[:, i])

    return optimal_thresholds

def calculate_sl_metrics_fold(test_df, thresholds):
    print("Computing fold")
    predictions = np.stack(test_df["preds"].to_numpy())
    outputs = predictions>thresholds
    actuals = np.stack(test_df["Target"].to_numpy())

    # Convert predictions, outputs, and actuals back to DataFrames
    preds_df = pd.DataFrame(predictions, columns=[
        'Membrane', 'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane',
        'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole',
        'Golgi apparatus', 'Peroxisome'
    ])

    outputs_df = pd.DataFrame(outputs, columns=preds_df.columns)
    actuals_df = pd.DataFrame(actuals, columns=preds_df.columns)

    # Combine all into a single DataFrame
    combined_df = pd.concat([test_df[['ACC']], preds_df, outputs_df, actuals_df], axis=1)

    # Rename columns for clarity
    combined_df.columns = ['ACC'] + \
                          [f'pred_{col}' for col in preds_df.columns] + \
                          [f'pred_loc_{col}' for col in preds_df.columns] + \
                          [f'true_loc_{col}' for col in preds_df.columns]

    ypred_membrane = outputs[:, 0]
    ypred_subloc = outputs[:,1:]
    y_membrane = actuals[:, 0]
    y_subloc = actuals[:,1:]

    metrics_dict = {}

    metrics_dict["NumLabels"] = y_subloc.sum(1).mean()
    metrics_dict["NumLabelsTest"] = ypred_subloc.sum(1).mean()
    metrics_dict["ACC_membrane"] = (ypred_membrane == y_membrane).mean()
    metrics_dict["MCC_membrane"] = matthews_corrcoef(y_membrane, ypred_membrane)
    metrics_dict["ACC_subloc"] = (np.all((ypred_subloc == y_subloc), axis=1)).mean()
    metrics_dict["HammLoss_subloc"] = 1-hamming_loss(y_subloc, ypred_subloc)
    metrics_dict["Jaccard_subloc"] = jaccard_score(y_subloc, ypred_subloc, average="samples")
    metrics_dict["MicroF1_subloc"] = f1_score(y_subloc, ypred_subloc, average="micro")
    metrics_dict["MacroF1_subloc"] = f1_score(y_subloc, ypred_subloc, average="macro")
    for i in range(10):
      metrics_dict[f"{CATEGORIES[1+i]}"] = matthews_corrcoef(y_subloc[:,i], ypred_subloc[:,i])

    # for i in range(10):
    #    metrics_dict[f"{categories[1+i]}"] = roc_auc_score(y_subloc[:,i], predictions[:,i+1])
    return metrics_dict, combined_df

def calculate_sl_metrics(model_attrs: ModelAttributes, datahandler: DataloaderHandler, thresh_type="mcc", inner_i="1Layer"):
    with open(os.path.join(model_attrs.outputs_save_path, f"thresholds_sl_{thresh_type}.pkl"), "rb") as f:
        threshold_dict = pickle.load(f)
    
    # Calculate mean thresholds
    threshold_values = np.array(list(threshold_dict.values()))
    print("Mean of the threshold values per dimension:")
    for i, threshold_mean in enumerate(threshold_values.mean(axis=0)):
        print(f"Dimension {i}: {threshold_mean}")
    print("Overall mean of threshold values:", threshold_values.mean())
    
    # Initialize an empty dictionary to store metrics for each fold
    metrics_dict_list = {}
    
    # Initialize an empty list to collect all data frames for each partition
    full_data_df = []
    
    # Iterate over the outer cross-validation folds (assuming 5-fold CV)
    for outer_i in range(5):
        # Get the data partition for the current fold
        data_df = datahandler.get_partition(outer_i)
        
        # Load the corresponding SL output predictions
        output_df = pd.read_pickle(os.path.join(model_attrs.outputs_save_path, f"{outer_i}_{inner_i}.pkl"))
        
        # Merge the data partition with the SL predictions
        data_df = data_df.merge(output_df)
        
        # Append the merged data frame to the list of full data
        full_data_df.append(data_df)
        
        # Get the threshold for the current fold from the threshold dictionary
        threshold = threshold_dict[f"{outer_i}_{inner_i}"]
        
        # Calculate SL metrics for the current fold using the merged data and threshold
        metrics_dict, combined_df = calculate_sl_metrics_fold(data_df, threshold)

        # Save the combined DataFrame as a CSV file
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = os.path.join(model_attrs.outputs_save_path, f"{outer_i}_{inner_i}_{current_timestamp}_output_predictions_with_true_values.csv")
        
        if not os.path.exists(os.path.dirname(output_csv_path)):
            os.makedirs(os.path.dirname(output_csv_path))
        
        combined_df.to_csv(output_csv_path, index=False)
        print(f"Saved predictions of {outer_i} {inner_i} to {output_csv_path}")
        
        # Accumulate the metrics for each fold in a dictionary
        for k in metrics_dict:
            metrics_dict_list.setdefault(k, []).append(metrics_dict[k])

    # # Combine all data frames from each fold into a single data frame
    # combined_df = pd.concat(full_data_df, ignore_index=True)
    
    # # Save the combined data frame with specific columns to a CSV file
    # output_csv_path = os.path.join(model_attrs.outputs_save_path, "combined_true_and_predictions.csv")
    # combined_df[['ACC', 'True_Label', 'preds']].to_csv(output_csv_path, index=False)
    # print(f"Saved full data to {output_csv_path}")

    output_dict = {}
    for k in metrics_dict_list:
        output_dict[k] = [f"{round(np.array(metrics_dict_list[k]).mean(), 2):.2f} pm {round(np.array(metrics_dict_list[k]).std(), 2):.2f}"]

    print(pd.DataFrame(output_dict).to_latex())
    for k in metrics_dict_list:
        print("{0:21s} : {1}".format(k, f"{round(np.array(metrics_dict_list[k]).mean(), 2):.2f} + {round(np.array(metrics_dict_list[k]).std(), 2):.2f}"))
    for k in metrics_dict_list:
        print("{0}".format(f"{round(np.array(metrics_dict_list[k]).mean(), 2):.2f} + {round(np.array(metrics_dict_list[k]).std(), 2):.2f}"))


def calculate_ss_metrics_fold(y_test, y_test_preds, thresh):
    y_preds = y_test_preds > thresh

    metrics_dict = {}

    metrics_dict["microF1"] = f1_score(y_test, y_preds, average="micro")
    metrics_dict["macroF1"] = f1_score(y_test, y_preds, average="macro")
    metrics_dict["accuracy"] = (np.all((y_preds == y_test), axis=1)).mean()

    for j in range(len(SS_CATEGORIES)-1):
        metrics_dict[f"{SS_CATEGORIES[j+1]}"]  = matthews_corrcoef(y_preds[:, j],y_test[:, j])

    return metrics_dict

def calculate_ss_metrics(model_attrs: ModelAttributes, datahandler: DataloaderHandler, thresh_type="mcc"):
    with open(os.path.join(model_attrs.outputs_save_path, f"thresholds_ss_{thresh_type}.pkl"), "rb") as f:
        threshold_dict = pickle.load(f)
    # print(np.array(list(threshold_dict.values())).mean(0))
    metrics_dict_list = {}
    thresh = np.array([threshold_dict[k] for k in SS_CATEGORIES[1:]])
    
    for outer_i in range(5):
        _,_,_, y_test = datahandler.get_swissprot_ss_xy(model_attrs.outputs_save_path, outer_i)
        y_test_preds = pickle.load(open(f"{model_attrs.outputs_save_path}/ss_{outer_i}.pkl", "rb"))
        metrics_dict = calculate_ss_metrics_fold(y_test, y_test_preds, thresh)
        for k in metrics_dict:
            metrics_dict_list.setdefault(k, []).append(metrics_dict[k])

    output_dict = {}
    for k in metrics_dict_list:
        output_dict[k] = [f"{round(np.array(metrics_dict_list[k]).mean(), 2):.2f} pm {round(np.array(metrics_dict_list[k]).std(), 2):.2f}"]
    print(pd.DataFrame(output_dict).to_latex())

def save_protein_predictions_to_csv(model_attrs, datahandler, inner_i="1Layer"):
    full_data_df = []
    for outer_i in range(5):
        # Get the data partition for the current fold
        data_df = datahandler.get_partition(outer_i)
        
        # Load the corresponding SL output predictions
        output_df = pd.read_pickle(os.path.join(model_attrs.outputs_save_path, f"{outer_i}_{inner_i}.pkl"))
        
        # Merge the data partition with the SL predictions
        data_df = data_df.merge(output_df)
        
        # Append the merged data frame to the list of full data
        full_data_df.append(data_df)
    
    # Concatenate all data partitions into a single DataFrame
    full_data_df = pd.concat(full_data_df, axis=0)
    
    # Assume that the predicted probabilities for each category are in the correct order in the DataFrame
    predictions_df = full_data_df[['Protein_ID', 'Localizations'] + [f'pred_{cat}' for cat in CATEGORIES]]
    
    # Rename columns to match the desired format
    predictions_df.columns = ['Protein_ID', 'Localizations'] + CATEGORIES
    
    # Save to CSV
    csv_file_path = "protein_predictions.csv"  # Specify the desired path and filename
    predictions_df.to_csv(csv_file_path, index=False)

    print(f"Protein predictions saved to {csv_file_path}")
