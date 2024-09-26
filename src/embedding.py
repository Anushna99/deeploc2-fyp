import h5py
import torch
from esm import Alphabet, FastaBatchedDataset, pretrained
from transformers import T5EncoderModel, T5Tokenizer
import tqdm
from src.data import *
from src.utils import *
import os
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "cpu"
    dtype=torch.bfloat16
else:
    device = "cpu"
    dtype=torch.bfloat16

def embed_esm1b(embed_dataloader, out_file):
    model, _ = pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")
    model.eval().to(device)
    embed_h5 = h5py.File(out_file, "w")
    try:
        # Uses torch.autocast to enable mixed-precision training (if applicable) for more efficient computation.
        with torch.autocast(device_type=device,dtype=dtype):
            with torch.no_grad():
                for i, (toks, lengths, np_mask, labels) in tqdm.tqdm(enumerate(embed_dataloader)):
                    # Passes the tokenized sequences through the model to obtain embeddings.
                    # repr_layers=[33] specifies that embeddings from layer 33 should be extracted.
                    # Converts the embeddings to NumPy arrays.
                    embed = model(toks.to(device), repr_layers=[33])["representations"][33].float().cpu().numpy()
                    for j in range(len(labels)):
                        # removing start and end tokens and
                        # For each sequence in the batch, stores the embeddings in the HDF5 file.
                        # Removes start and end tokens from the embeddings (embed[j, 1:1+lengths[j]]).
                        # Converts embeddings to np.float16 for storage efficiency.
                        embed_h5[labels[j]] = embed[j, 1:1+lengths[j]].astype(np.float16)
        embed_h5.close()
    except:
        os.system(f"rm {out_file}")
        raise Exception("Failed to create embeddings")
    

def embed_prott5(embed_dataloader, out_file):
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model.eval().to(device)
    embed_h5 = h5py.File(out_file, "w")
    try:
        with torch.autocast(device_type=device,dtype=dtype):
            with torch.no_grad():
                for i, (toks, lengths, np_mask, labels) in tqdm.tqdm(enumerate(embed_dataloader)):
                    embed = model(input_ids=torch.tensor(toks['input_ids'], device=device),
                    attention_mask=torch.tensor(toks['attention_mask'], 
                        device=device)).last_hidden_state.float().cpu().numpy()
                    for j in range(len(labels)):
                        # removing end tokens
                        embed_h5[labels[j]] = embed[j, :lengths[j]].astype(np.float16)
        embed_h5.close()
    except:
        os.system(f"rm {out_file}")
        raise Exception("Failed to create embeddings")

def generate_embeddings(model_attrs: ModelAttributes, is_training: bool):
    '''
    This function generates embeddings based on the model type (either FAST or ACCURATE).
    '''
    if is_training:
        embedding_type = model_attrs.model_type  
    else:
        if model_attrs.model_type == FAST :
            embedding_type = TEST_ESM
        else :
            embedding_type = TEST_PROTT5

    # Example usage of embedding_type (or return it if needed)
    print(f"Embedding type selected: {embedding_type}")

    fasta_dict = read_fasta(EMBEDDINGS[embedding_type]["source_fasta"])
    # store the fasta readings to a csv file
    fasta_format = os.path.splitext(os.path.basename(EMBEDDINGS[embedding_type]["source_fasta"]))[0]
    save_fasta_to_csv(fasta_dict=fasta_dict, outputs_save_path=model_attrs.outputs_save_path, type = fasta_format)
    # Converts the dictionary into a DataFrame with columns ACC and Sequence
    test_df = pd.DataFrame(fasta_dict.items(), columns=['ACC', 'Sequence'])

    # create a embedded batches
    embed_dataset = FastaBatchedDatasetTorch(test_df)
    # create sequence batches with token size of 8196
    embed_batches = embed_dataset.get_batch_indices(8196, extra_toks_per_seq=1)
    print('model type', model_attrs.model_type)
    if model_attrs.model_type == FAST:
        embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverter(model_attrs.alphabet), batch_sampler=embed_batches)
        #For each batch, runs the model to get representations and stores them in the HDF5 file.
        embed_esm1b(embed_dataloader, EMBEDDINGS[embedding_type]["embeds"])
    elif model_attrs.model_type == ACCURATE:
        embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverterProtT5(model_attrs.alphabet), batch_sampler=embed_batches)
        embed_prott5(embed_dataloader, EMBEDDINGS[embedding_type]["embeds"])
    else:
        raise Exception("wrong model type provided expected Fast,Accurate got", model_attrs.model_type)
    
