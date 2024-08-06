from src.model import *
from src.data import DataloaderHandler
import pickle
from transformers import T5EncoderModel, T5Tokenizer, logging
import os
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
                 save_path: str,
                 outputs_save_path: str,
                 clip_len: int,
                 embed_len: int) -> None:
        self.model_type = model_type
        self.class_type = class_type 
        self.alphabet = alphabet
        self.embedding_file = embedding_file
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
            "models/models_prott5",
            "outputs/prott5/",
            4000,
            1024
        )
    else:
        raise Exception("wrong model type provided expected Fast,Accurate got", model_type)
    

