import os
import tqdm
import pandas as pd
import numpy as np
import pickle
import torch
from src.utils import ModelAttributes
from src.data import DataloaderHandler
from src.metrics import *
from datetime import datetime

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "cpu"
    dtype=torch.bfloat16
else:
    device = "cpu"
    dtype=torch.bfloat16

def predict_sl_values(dataloader, model, outputs_save_path, outer_i, inner_i):
    output_dict = {}
    annot_dict = {}
    pool_dict = {}
    with torch.no_grad():
      for i, (toks, lengths, np_mask, targets, targets_seq, labels) in tqdm.tqdm(enumerate(dataloader)):
        # Enables automatic mixed precision, which improves performance and reduces memory usage by using both 16-bit and 32-bit floating-point operations.
        with torch.autocast(device_type=device,dtype=dtype):
            # y_pred: The raw prediction scores or logits from the model.
            # y_pool: The pooled outputs from the model, which could represent aggregated information from different parts of the input.
            # y_attn: Attention weights or scores, though not used further in this function.
            y_pred, y_pool, y_attn = model.predict(toks.to(device), lengths.to(device), np_mask.to(device))
        x = torch.sigmoid(y_pred).float().cpu().numpy() # Applies the sigmoid function to y_pred, converting raw logits to probabilities between 0 and 1.
        for j in range(len(labels)):
            if len(labels) == 1:
                output_dict[labels[j]] = x
                pool_dict[labels[j]] = y_pool.float().cpu().numpy()
                annot_dict[labels[j]] = y_attn[:lengths[j]].float().cpu().numpy()
            else:
                output_dict[labels[j]] = x[j]
                pool_dict[labels[j]] = y_pool[j].float().cpu().numpy()
                annot_dict[labels[j]] = y_attn[j,:lengths[j]].float().cpu().numpy()

    output_df = pd.DataFrame(output_dict.items(), columns=['ACC', 'preds'])
    annot_df = pd.DataFrame(annot_dict.items(), columns=['ACC', 'pred_annot'])
    pool_df = pd.DataFrame(pool_dict.items(), columns=['ACC', 'embeds'])

    # current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_csv_path = os.path.join(outputs_save_path, f"{outer_i}_{inner_i}_{current_timestamp}_output_predictions.csv")
    # if not os.path.exists(os.path.dirname(output_csv_path)):
    #         os.makedirs(os.path.dirname(output_csv_path))
    
    localization_columns = CATEGORIES

    # Create a DataFrame for the predictions
    preds_df = pd.DataFrame(output_df['preds'].to_list(), columns=localization_columns)
    
    # Combine the ACC column with the predictions DataFrame
    output_df_csv = pd.concat([output_df[['ACC']], preds_df], axis=1)

    # Save the DataFrame
    # output_df_csv.to_csv(output_csv_path, index=False)
    # print(f"Saved predictions to {output_csv_path}")

    return output_df.merge(annot_df).merge(pool_df)
    
def generate_sl_outputs(
        model_attrs: ModelAttributes, 
        datahandler: DataloaderHandler, 
        thresh_type="mcc", 
        inner_i="1Layer", 
        reuse=False):
    '''
    This function generates predictions and optimal thresholds for different models and saves the results.
    '''
    
    threshold_dict = {}
        
    for outer_i in range(5):
        # use 5 trained models to generate predictions
        print("Generating output for ensemble model", outer_i)
        # this load the same test sets each time that used for trainnng the model
        dataloader, data_df = datahandler.get_partition_dataloader_inner(outer_i)
        if not os.path.exists(os.path.join(model_attrs.outputs_save_path, f"inner_{outer_i}_{inner_i}.pkl")):
            # path to the model i trained checkpoint
            path = f"{model_attrs.save_path}/{outer_i}_{inner_i}.ckpt"
            # evaluate from that checkpoint
            model = model_attrs.class_type.load_from_checkpoint(path).to(device).eval()
            # predict the location values
            pred_df = predict_sl_values(dataloader, model, model_attrs.outputs_save_path, outer_i, inner_i)
            # save the file for laterr
            pred_df.to_pickle(os.path.join(model_attrs.outputs_save_path, f"inner_{outer_i}_{inner_i}.pkl"))
        else:
            pred_df = pd.read_pickle(os.path.join(model_attrs.outputs_save_path, f"inner_{outer_i}_{inner_i}.pkl"))

        # caculate the thresholds by using truth values and predictions
        if thresh_type == "roc":
            thresholds = get_optimal_threshold(pred_df, data_df)
        elif thresh_type == "pr":
            thresholds = get_optimal_threshold_pr(pred_df, data_df)
        else:
            thresholds = get_optimal_threshold_mcc(pred_df, data_df)
        threshold_dict[f"{outer_i}_{inner_i}"] = thresholds

        # if the predictions already generated then save them
        if not os.path.exists(os.path.join(model_attrs.outputs_save_path, f"{outer_i}_{inner_i}.pkl")):
            dataloader, data_df = datahandler.get_partition_dataloader(outer_i)
            output_df = predict_sl_values(dataloader, model, model_attrs.outputs_save_path, outer_i, inner_i)
            output_df.to_pickle(os.path.join(model_attrs.outputs_save_path, f"{outer_i}_{inner_i}.pkl"))


    # Convert the threshold dictionary to a DataFrame
    threshold_df = pd.DataFrame.from_dict(threshold_dict, orient='index', columns=[
        CATEGORIES
    ])

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save the thresholds as a CSV file
    threshold_csv_path = os.path.join(model_attrs.outputs_save_path, f"thresholds_sl_{current_timestamp}_{thresh_type}.csv")
    threshold_df.to_csv(threshold_csv_path)

    print(f"Thresholds saved to {threshold_csv_path}")

    with open(os.path.join(model_attrs.outputs_save_path, f"thresholds_sl_{thresh_type}.pkl"), "wb") as f:
        pickle.dump(threshold_dict, f)

def predict_ss_values(X, model):
    X_tensor = torch.tensor(X, device=device).float()
    y_preds = torch.sigmoid(model(X_tensor))
    return y_preds.detach().cpu().numpy()

def generate_ss_outputs(
        model_attrs: ModelAttributes, 
        datahandler: DataloaderHandler, 
        thresh_type="mcc", 
        inner_i="1Layer", 
        reuse=False):
    
    threshold_dict = {}
    if not os.path.exists(f"{model_attrs.outputs_save_path}"):
        os.makedirs(f"{model_attrs.outputs_save_path}")
    for outer_i in range(5):
        print("Generating output for ensemble model", outer_i)
        X_train, y_train, X_test, y_test = datahandler.get_swissprot_ss_xy(model_attrs.outputs_save_path, outer_i)
        path = f"{model_attrs.save_path}/signaltype/{outer_i}.ckpt"
        model = SignalTypeMLP.load_from_checkpoint(path).to(device).eval()
        
        y_train_preds = predict_ss_values(X_train, model)
        thresh = np.zeros((9,))
        threshold_dict = {}
        #print("thresholds")
        for type_i in range(9):
            thresh[type_i] = get_best_threshold_mcc(y_train[:, type_i], y_train_preds[:, type_i])
            threshold_dict[SS_CATEGORIES[type_i+1]] = thresh[type_i]
            #print(SS_CATEGORIES[type_i+1], thresh[type_i])
        y_test_preds = predict_ss_values(X_test, model)
        pickle.dump(y_test_preds, open(f"{model_attrs.outputs_save_path}/ss_{outer_i}.pkl", "wb"))

    with open(os.path.join(model_attrs.outputs_save_path, f"thresholds_ss_mcc.pkl"), "wb") as f:
        pickle.dump(threshold_dict, f)
