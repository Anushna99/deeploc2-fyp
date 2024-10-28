from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.data import *
from src.utils import *
from src.eval_utils import *
from src.embedding import *
from src.metrics import *
import argparse
import subprocess
import os
import warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m","--model", 
        default="Fast",
        choices=['Accurate', 'Fast'],
        type=str,
        help="Model to use."
    )
    parser.add_argument(
        "-d", "--dataset", 
        choices=["swissprot", "hpa"], 
        required=True, 
        help="Dataset to use (SwissProt or HPA)."
    )
    args = parser.parse_args()

    model_attrs = get_test_model_attributes(model_type=args.model, data=args.dataset)
    print("All Model Attributes:")
    print(vars(model_attrs))
    if not os.path.exists(model_attrs.embedding_file):
        print("Embeddings not found, generating......")
        generate_embeddings(model_attrs)
        print("Embeddings created!")
    else:
        print("Using existing embeddings")
    
    if not os.path.exists(model_attrs.embedding_file):
        raise Exception("Embeddings could not be created. Verify that data_files/embeddings/<MODEL_DATASET> is deleted")
    
    datahandler = DataloaderHandler(
        clip_len=model_attrs.clip_len, 
        alphabet=model_attrs.alphabet, 
        embedding_file=model_attrs.embedding_file,
        embed_len=model_attrs.embed_len
    )

    print("Using trained models to generate outputs for localization")
    generate_sl_predictions(model_attrs=model_attrs, datahandler=datahandler)
    print("Generated outputs for testing the model is done.")
