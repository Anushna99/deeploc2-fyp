import pickle
import torch
from Bio import SeqIO
import re
import pandas as pd
import time 
import os
class FastaBatchedDatasetTorch(torch.utils.data.Dataset):
    def __init__(self, data_df):
        '''Initializes the dataset with a DataFrame data_df containing the sequences and their corresponding identifiers (ACC).
        '''
        self.data_df = data_df

    def __len__(self):
        '''
        Returns the number of sequences in the dataset.
        '''
        return len(self.data_df)
    
    def shuffle(self):
        '''
        Shuffles the DataFrame to randomize the order of sequences.
        '''
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx):
        '''
        Returns the sequence and its corresponding identifier for a given index idx.
        '''
        return self.data_df["Sequence"][idx], self.data_df["ACC"][idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        '''
        generate batches of indices that can be used to load data efficiently.
        '''

        # calculate each sequence's length and pair it with sequence index
        sizes = [(len(s), i) for i, s in enumerate(self.data_df["Sequence"])]
        # order it from most to least
        sizes.sort(reverse=True)
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0
        start = 0
        #start = random.randint(0, len(sizes))
        for j in range(len(sizes)):
            i = (start + j) % len(sizes)
            sz = sizes[i][0]
            idx = sizes[i][1]    
            sz += extra_toks_per_seq
            if (max(sz, max_len) * (len(buf) + 1) > toks_per_batch):
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(idx)

        _flush_current_buf()
        return batches

class BatchConverterProtT5(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        #print(len(raw_batch[0]), raw_batch[1], raw_batch[2])
        max_len = max(len(seq_str) for seq_str, _ in raw_batch)
        labels = []
        lengths = []
        strs = []
        for i, (seq_str, label) in enumerate(raw_batch):
            #seq_str = seq_str[1:]
            labels.append(label)
            lengths.append(len(seq_str))
            strs.append(seq_str)
        
        proteins = [" ".join(list(item)) for item in strs]
        proteins = [re.sub(r"[UZOB]", "X", sequence) for sequence in proteins]
        ids = self.alphabet.batch_encode_plus(proteins, add_special_tokens=True, padding=True)
        non_pad_mask = torch.tensor(ids['input_ids']) > -100 # B, T

        return ids, torch.tensor(lengths), non_pad_mask, labels


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch):
        # raw_batch is a list of tuples (sequence_string, label)
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        #print(len(raw_batch[0]), raw_batch[1], raw_batch[2])
        max_len = max(len(seq_str) for seq_str, _ in raw_batch)
        tokens = torch.empty((batch_size, max_len + int(self.alphabet.prepend_bos) + \
            int(self.alphabet.append_eos)), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        lengths = []
        strs = []
        for i, (seq_str, label) in enumerate(raw_batch):
            #seq_str = seq_str[1:]
            labels.append(label)
            lengths.append(len(seq_str))
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor([self.alphabet.get_idx(s) for s in seq_str], dtype=torch.int64)
            tokens[i, int(self.alphabet.prepend_bos) : len(seq_str) + int(self.alphabet.prepend_bos)] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_str) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        
        non_pad_mask = ~tokens.eq(self.alphabet.padding_idx) &\
         ~tokens.eq(self.alphabet.cls_idx) &\
         ~tokens.eq(self.alphabet.eos_idx)# B, T
        
        # Return the processed tensors: tokens (tokenized sequences), lengths (lengths of the sequences), 
        # non_pad_mask (mask for non-padding tokens), and labels (which contain the ACC values).
        return tokens, torch.tensor(lengths), non_pad_mask, labels

def read_fasta(fastafile):
    """Parse a file with sequences in FASTA format and store in a dict"""
    proteins = list(SeqIO.parse(fastafile, "fasta"))
    res = {}
    for prot in proteins:
        res[str(prot.id)] = str(prot.seq)
    return res

# with open("/tools/src/deeploc-2.0/models/ESM1b_alphabet.pkl", "rb") as f:
#     alphabet = pickle.load(f)

###################################
#######   TRAINING STUFF  #########
###################################

import h5py
import numpy as np
import pickle5
from sklearn.model_selection import ShuffleSplit
from src.constants import *

def get_swissprot_df(clip_len):  
    with open(SIGNAL_DATA, "rb") as f:
        # This DataFrame contains annotations related to the proteins.
        annot_df = pd.compat.pickle_compat.load(f)
    nes_exclude_list = ['Q7TPV4','P47973','P38398','P38861','Q16665','O15392','Q9Y8G3','O14746','P13350','Q06142']
    swissprot_exclusion_list = ['Q04656-5','O43157','Q9UPN3-2']

    #These functions ensure that sequences longer than clip_len are trimmed to include only the middle portion.
    def clip_middle_np(x):
        if len(x)>clip_len:
            x = np.concatenate((x[:clip_len//2],x[-clip_len//2:]), axis=0)
        return x
    def clip_middle(x):
      if len(x)>clip_len:
          x = x[:clip_len//2] + x[-clip_len//2:]
      return x
 
    annot_df["TargetAnnot"] = annot_df["ANNOT"].apply(lambda x: clip_middle_np(x))
    data_df = pd.read_csv(LOCALIZATION_DATA)
    data_df["Sequence"] = data_df["Sequence"].apply(lambda x: clip_middle(x))
    data_df["Target"] = data_df[CATEGORIES].values.tolist()    

    annot_df = annot_df[~annot_df.ACC.isin(nes_exclude_list)].reset_index(drop=True)
    data_df = data_df[~data_df.ACC.isin(swissprot_exclusion_list)].reset_index(drop=True)
    data_df = data_df.merge(annot_df[["ACC", "ANNOT", "Types", "TargetAnnot"]], on="ACC", how="left")
    data_df['TargetAnnot'] = data_df['TargetAnnot'].fillna(0)

    print(data_df.head(10))

    # embedding_fasta = read_fasta(f"{embedding_path}/remapped_sequences_file.fasta")
    # embedding_df = pd.DataFrame(embedding_fasta.items(), columns=["details", "RawSeq"])
    # embedding_df["Hash"] = embedding_df.details.apply(lambda x: x.split()[0])
    # embedding_df["ACC"] = embedding_df.details.apply(lambda x: x.split()[1])
    # data_df = data_df.merge(embedding_df[["ACC", "Hash"]]).reset_index(drop=True)

    return data_df

def convert_to_binary(x):
    types_binary = np.zeros((len(SS_CATEGORIES)-1,))
    for c in x.split("_"):
      types_binary[SS_CATEGORIES.index(c)-1] = 1
    return types_binary

def get_swissprot_ss_Xy(save_path, fold, clip_len):
    with open(SIGNAL_DATA, "rb") as f:
        annot_df = pickle5.load(f)
    nes_exclude_list = ['Q7TPV4','P47973','P38398','P38861','Q16665','O15392','Q9Y8G3','O14746','P13350','Q06142']
    swissprot_exclusion_list = ['Q04656-5','O43157','Q9UPN3-2']
    def clip_middle_np(x):
        if len(x)>clip_len:
            x = np.concatenate((x[:clip_len//2],x[-clip_len//2:]), axis=0)
        return x
    def clip_middle(x):
      if len(x)>clip_len:
          x = x[:clip_len//2] + x[-clip_len//2:]
      return x
    
    train_annot_pred_df = pd.read_pickle(os.path.join(save_path, f"inner_{fold}_1Layer.pkl"))
    test_annot_pred_df = pd.read_pickle(os.path.join(save_path, f"{fold}_1Layer.pkl"))
    assert train_annot_pred_df.merge(test_annot_pred_df, on="ACC").empty == True

    
    filt_annot_df = annot_df[annot_df["Types"]!=""].reset_index(drop=True)
    seq_df = filt_annot_df.merge(train_annot_pred_df)
    seq_df["Sequence"] = seq_df["Sequence"].apply(lambda x: clip_middle(x))
    seq_df["Target"] = seq_df[CATEGORIES].values.tolist()
    seq_df["TargetSignal"] = seq_df["Types"].apply(lambda x: convert_to_binary(x))

    annot_true_df = seq_df
    X_true_train, y_true_train = np.concatenate((np.stack(annot_true_df["embeds"].to_numpy()), np.stack(annot_true_df["Target"].to_numpy())), axis=1) , np.stack(annot_true_df["TargetSignal"].to_numpy())
    annot_pred_df = seq_df
    X_pred_target = np.stack(annot_true_df["preds"].to_numpy())# > threshold_dict[f"{i}_multidct"]
    X_pred_train, y_pred_train = np.concatenate((np.stack(annot_pred_df["embeds"].to_numpy()), X_pred_target), axis=1), np.stack(annot_pred_df["TargetSignal"].to_numpy())

    seq_df = filt_annot_df.merge(test_annot_pred_df)
    seq_df["Sequence"] = seq_df["Sequence"].apply(lambda x: clip_middle(x))
    seq_df["Target"] = seq_df[CATEGORIES].values.tolist()
    seq_df["TargetSignal"] = seq_df["Types"].apply(lambda x: convert_to_binary(x))

    annot_test_df = seq_df
    X_test_target = np.stack(annot_test_df["preds"].to_numpy())# > threshold_dict[f"{i}_multidct"]
    X_test, y_test = np.concatenate((np.stack(annot_test_df["embeds"].to_numpy()), X_test_target), axis=1), np.stack(annot_test_df["TargetSignal"].to_numpy())
    
    X_train = np.concatenate((X_true_train, X_pred_train), axis=0)
    y_train = np.concatenate((y_true_train, y_pred_train), axis=0)
    #print(X_train.shape, X_test.shape)

    return X_train, y_train, X_test, y_test


class EmbeddingsLocalizationDataset(torch.utils.data.Dataset):
    """
    Dataset of protein embeddings and the corresponding subcellular localization label.
    """

    def __init__(self, embedding_file, data_df) -> None:
        super().__init__()
        self.data_df = data_df
        self.embeddings_file = embedding_file
    
    def __getitem__(self, index: int):
        '''
        For a given index, it retrieves the corresponding sequence, embedding, target, target annotation, 
        and protein ID (ACC) from the DataFrame and the embeddings file.
        '''
        embedding = np.array(self.embeddings_file[self.data_df["ACC"][index]]).copy()
        # Check if the 'Target' column exists, otherwise return None
        target = self.data_df["Target"][index] if "Target" in self.data_df.columns else None
        # Check if the 'TargetAnnot' column exists, otherwise return None
        target_annot = self.data_df["TargetAnnot"][index] if "TargetAnnot" in self.data_df.columns else None

        return self.data_df["Sequence"][index], embedding, target, target_annot, self.data_df["ACC"][index]
    
    def get_batch_indices(self, toks_per_batch, max_batch_size, extra_toks_per_seq=0):
        '''
        This method generates indices for batching the data. It sorts sequences by length and groups them 
        into batches that fit within a specified number of tokens per batch (toks_per_batch) and a maximum batch size (max_batch_size).
        '''
        sizes = [(len(s), i) for i, s in enumerate(self.data_df["Sequence"])]
        sizes.sort(reverse=True)
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0
        start = 0
        #start = random.randint(0, len(sizes))
        for j in range(len(sizes)):
            i = (start + j) % len(sizes)
            sz = sizes[i][0]
            idx = sizes[i][1]    
            sz += extra_toks_per_seq
            if (max(sz, max_len) * (len(buf) + 1) > toks_per_batch) or len(buf) >= max_batch_size:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(idx)

        _flush_current_buf()
        return batches

    def __len__(self) -> int:
        return len(self.data_df)
    
class EmbeddingsPreProcessDataset(torch.utils.data.Dataset):
    """
    Dataset of protein embeddings and the corresponding subcellular localization label.
    """

    def __init__(self, embedding_file) -> None:
        super().__init__()
        self.embeddings_file = embedding_file
        self.data_df = pd.DataFrame({
            "ACC": list(self.embeddings_file.keys())
        })
    
    def __getitem__(self, index: int):
        '''
        For a given index, it retrieves the corresponding embedding, np_mask, and ACC from the embeddings file.
        '''
        acc = self.data_df["ACC"][index]
        embedding = np.array(self.embeddings_file[acc]).copy()
        np_mask = np.ones(embedding.shape[0], dtype=bool)  # Create a mask of the same length as the embedding

        return embedding, len(embedding), np_mask, acc
    
    def get_batch_indices(self, toks_per_batch, max_batch_size, extra_toks_per_seq=0):
        '''
        This method generates indices for batching the data. It sorts sequences by length and groups them 
        into batches that fit within a specified number of tokens per batch (toks_per_batch) and a maximum batch size (max_batch_size).
        '''
        sizes = [(len(self.embeddings_file[acc]), i) for i, acc in enumerate(self.data_df["ACC"])]
        sizes.sort(reverse=True)
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0
        
        for j in range(len(sizes)):
            sz = sizes[j][0]
            idx = sizes[j][1]
            sz += extra_toks_per_seq
            if (max(sz, max_len) * (len(buf) + 1) > toks_per_batch) or len(buf) >= max_batch_size:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(idx)

        _flush_current_buf()
        return batches

    def __len__(self) -> int:
        return len(self.data_df)

class TrainBatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, embed_len):
        self.alphabet = alphabet
        self.embed_len = embed_len

    def __call__(self, raw_batch):
        batch_size = len(raw_batch)
        max_len = max(len(seq_str) for seq_str, _, _, _, _ in raw_batch)
        embedding_tensor = torch.zeros((batch_size, max_len, self.embed_len), dtype=torch.float32)
        np_mask = torch.zeros((batch_size, max_len))
        target_annots = torch.zeros((batch_size, max_len), dtype=torch.int64)
        labels = []
        lengths = []
        strs = []
        targets = torch.zeros((batch_size, 11), dtype=torch.float32)
        for i, (seq_str, embedding, target, target_annot, label) in enumerate(raw_batch):
            #seq_str = seq_str[1:]
            labels.append(label)
            lengths.append(len(seq_str))
            strs.append(seq_str)
            targets[i] = torch.tensor(target)
            embedding_tensor[i, :len(seq_str)] = torch.tensor(np.array(embedding))
            target_annots[i, :len(seq_str)] = torch.tensor(target_annot)
            np_mask[i, :len(seq_str)] = 1
        np_mask = np_mask == 1
        return embedding_tensor, torch.tensor(lengths), np_mask, targets, target_annots, labels
    
class TestBatchConverter:
    def __init__(self, embed_len):
        self.embed_len = embed_len

    def __call__(self, batch):
        """
        Takes a list of tuples (embedding, length, np_mask, acc) and collates them into a batch.
        """
        # Unzip the batch into separate lists
        embeddings, lengths, np_masks, labels = zip(*batch)

        # Pad the embeddings to the maximum length in the batch
        max_len = max(lengths)
        padded_embeddings = torch.zeros((len(embeddings), max_len, self.embed_len), dtype=torch.float32)
        np_masks_padded = torch.zeros((len(np_masks), max_len), dtype=torch.bool)

        for i in range(len(embeddings)):
            padded_embeddings[i, :lengths[i]] = torch.tensor(embeddings[i], dtype=torch.float32)
            np_masks_padded[i, :lengths[i]] = torch.tensor(np_masks[i], dtype=torch.bool)

        # Convert lengths to tensor
        lengths_tensor = torch.tensor(lengths, dtype=torch.int64)

        return padded_embeddings, lengths_tensor, np_masks_padded, list(labels)
    
class SignalTypeDataset(torch.utils.data.Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y
    
    def __getitem__(self, index: int):
        return torch.tensor(self.X[index]).float(), torch.tensor(self.y[index]).float()

    def __len__(self):
        return self.X.shape[0]


class DataloaderHandler:
    '''
    The DataloaderHandler class is designed to handle the data loading and processing necessary for training and validation of a model.
    Parameters:
        clip_len: Maximum length of sequences to be processed.
        alphabet: Object containing information about token indices (e.g., padding, start, end tokens).
        embedding_file: Path to the file where embeddings are stored.
        embed_len: Length of the embeddings.
    '''
    def __init__(self, clip_len, alphabet, embedding_file, embed_len, num_workers=7) -> None:
        self.clip_len = clip_len
        self.alphabet = alphabet
        self.embedding_file = embedding_file
        self.embed_len = embed_len
        self.num_workers = num_workers  # Add num_workers parameter

    def get_train_val_dataloaders(self, outer_i):
        data_df = get_swissprot_df(self.clip_len) #Calls get_swissprot_df function to obtain a DataFrame containing SwissProt data with sequences clipped to clip_len.
        
        train_df = data_df[data_df.Partition != outer_i].reset_index(drop=True) # Splits the data into training and validation sets based on outer_i.
        #ShuffleSplit is used to create a validation set of size 2048.

        X = np.stack(train_df["ACC"].to_numpy())
        sss_tt = ShuffleSplit(n_splits=1, test_size=2048, random_state=0)
        
        (split_train_idx, split_val_idx) = next(sss_tt.split(X))
        split_train_df =  train_df.iloc[split_train_idx].reset_index(drop=True)
        split_val_df = train_df.iloc[split_val_idx].reset_index(drop=True)

        print("Trainng data frame", split_train_df.head(10))
        print("validation data frme", split_val_df.head(10))

        # can help you understand how the data is distributed and whether there are any imbalances.
        print("Train DataFrame Categories Mean:")
        print(split_train_df[CATEGORIES].mean())
        print("\nValidation DataFrame Categories Mean:")
        print(split_val_df[CATEGORIES].mean())

        embedding_file = h5py.File(self.embedding_file, "r")
        # create dataset objects for training and validation.
        train_dataset = EmbeddingsLocalizationDataset(embedding_file, split_train_df)
        train_batches = train_dataset.get_batch_indices(4096*4, BATCH_SIZE, extra_toks_per_seq=0)
        # The DataLoader class is used to create iterable data loaders for both training and validation datasets. 
        # The collate_fn parameter specifies how to combine multiple data samples into a single batch, and batch_sampler 
        # is used to control the batching process.
        train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=TrainBatchConverter(self.alphabet, self.embed_len), batch_sampler=train_batches, num_workers=self.num_workers,
            pin_memory=True) # use pin_memory to utilize more gpu

        val_dataset = EmbeddingsLocalizationDataset(embedding_file, split_val_df)
        val_batches = val_dataset.get_batch_indices(4096*4, BATCH_SIZE, extra_toks_per_seq=0)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                     collate_fn=TrainBatchConverter(self.alphabet, self.embed_len), 
                                                     batch_sampler=val_batches,
                                                    num_workers=self.num_workers,
                                                    pin_memory=True) # use pin_memory to utilize more gpu
        return train_dataloader, val_dataloader

    def get_partition(self, outer_i):
        data_df = get_swissprot_df(self.clip_len )
        test_df = data_df[data_df.Partition == outer_i].reset_index(drop=True)
        return test_df

    def get_partition_dataloader(self, outer_i):
        data_df = get_swissprot_df(self.clip_len)
        test_df = data_df[data_df.Partition == outer_i].reset_index(drop=True)
        
        embedding_file = h5py.File(self.embedding_file, "r")
        test_dataset = EmbeddingsLocalizationDataset(embedding_file, test_df)
        test_batches = test_dataset.get_batch_indices(4096*4, BATCH_SIZE, extra_toks_per_seq=0)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=TrainBatchConverter(self.alphabet, self.embed_len), batch_sampler=test_batches, num_workers=self.num_workers,
            pin_memory=True)
        return test_dataloader, test_df

    def get_partition_dataloader_inner(self, partition_i):
        data_df = get_swissprot_df(self.clip_len)
        test_df = data_df[data_df.Partition != partition_i].reset_index(drop=True)
        embedding_file = h5py.File(self.embedding_file, "r")
        test_dataset = EmbeddingsLocalizationDataset(embedding_file, test_df)
        test_batches = test_dataset.get_batch_indices(4096*4, BATCH_SIZE, extra_toks_per_seq=0)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=TrainBatchConverter(self.alphabet, self.embed_len), batch_sampler=test_batches, num_workers=self.num_workers,
            pin_memory=True)

        return test_dataloader, test_df
    
    def get_test_dataloader(self, model_attrs):
        if (model_attrs.dataset == 'swissprot'):
            if (model_attrs.model_type == FAST):
                embedding_file = h5py.File('data_files/embeddings/esm1b_swissprot.h5', "r")
            else:
                embedding_file = h5py.File('data_files/embeddings/prott5_swissprot.h5', "r")
        else:
            if (model_attrs.model_type == FAST):
                embedding_file = h5py.File('data_files/embeddings/esm1b_hpa.h5', "r")
            else:
                embedding_file = h5py.File('data_files/embeddings/prott5_hpa.h5', "r")
        print('embedding file selected: ', embedding_file)
        test_dataset = EmbeddingsPreProcessDataset(embedding_file)
        test_batches = test_dataset.get_batch_indices(4096*4, BATCH_SIZE, extra_toks_per_seq=0)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=TestBatchConverter(self.embed_len), batch_sampler=test_batches, num_workers=self.num_workers,
            pin_memory=True)

        return test_dataloader
    
    def get_ss_train_val_dataloader(self, save_path, outer_i):
        X, y, _, _ = get_swissprot_ss_Xy(save_path, outer_i, clip_len=self.clip_len)
        sss_tt = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        
        (split_train_idx, split_val_idx) = next(sss_tt.split(y))
        split_train_X, split_train_y =  X[split_train_idx], y[split_train_idx]
        split_val_X, split_val_y = X[split_val_idx], y[split_val_idx]

        print(split_train_X.shape, split_train_y.shape, split_val_X.shape, split_val_y.shape)
        
        train_dataset = SignalTypeDataset(split_train_X, split_train_y)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=BATCH_SIZE,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True)

        val_dataset = SignalTypeDataset(split_val_X, split_val_y)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=BATCH_SIZE,
            num_workers=self.num_workers,
            pin_memory=True)
        
        return train_dataloader, val_dataloader

    def get_ss_test_dataloader(self, save_path, outer_i):
        _, _, X, y = get_swissprot_ss_Xy(save_path, outer_i, clip_len=self.clip_len)
        
        print(X.shape, y.shape)
        val_dataset = SignalTypeDataset(X, y)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=X.shape[0],
            num_workers=self.num_workers,
            pin_memory=True)

        return val_dataloader
    
    def get_swissprot_ss_xy(self, save_path, outer_i):
        return get_swissprot_ss_Xy(save_path=save_path, fold=outer_i, clip_len=self.clip_len)
    

class HPATestDataset(torch.utils.data.Dataset):
    """
    Dataset to handle HPA test embeddings and sequence identifiers.
    """

    def __init__(self, embedding_file):
        super().__init__()
        self.embedding_file = embedding_file
        self.keys = list(embedding_file.keys())

    def __getitem__(self, index):
        """
        Returns the sequence embedding and its identifier (ACC) for a given index.
        """
        acc = self.keys[index]
        embedding = self.embedding_file[acc][:]
        return embedding, acc

    def __len__(self):
        return len(self.keys)


    def get_hpa_test_dataloader(alphabet, embed_len, batch_size=128, num_workers=7):
        """
        This function creates a DataLoader for the HPA test dataset.
        Args:
            alphabet: Object containing token indices and other attributes for batch conversion.
            embed_len: Length of the embedding dimension.
            batch_size: Number of sequences in each batch.
            num_workers: Number of workers for data loading.
        Returns:
            test_dataloader: DataLoader to iterate through HPA test data.
        """

        # Load the HPA embedding file
        embedding_file = h5py.File('data_files/embeddings/esm1b_hpa.h5', "r")
        
        keys = list(embedding_file.keys())
        print(f"Number of sequences in the embedding file: {len(keys)}")
        # Create the test dataset
        test_dataset = HPATestDataset(embedding_file)

        # Define a collate function to convert raw batches into processed batches
        def collate_fn(raw_batch):
            """
            Converts the raw batch of (embedding, ACC) into tensors compatible with the model.
            """
            batch_size = len(raw_batch)
            embeddings, acc_list = zip(*raw_batch)
            max_len = max(embedding.shape[0] for embedding in embeddings)

            # Prepare tensors
            embedding_tensor = torch.zeros((batch_size, max_len, embed_len), dtype=torch.float32)
            lengths = torch.zeros(batch_size, dtype=torch.int64)

            for i, embedding in enumerate(embeddings):
                length = embedding.shape[0]
                embedding_tensor[i, :length, :] = torch.tensor(embedding)
                lengths[i] = length

            # Create a non-padding mask (all True since embeddings do not have padding)
            non_pad_mask = torch.ones_like(embedding_tensor[:, :, 0], dtype=torch.bool)

            return embedding_tensor, lengths, non_pad_mask, acc_list

        # Create the DataLoader
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        num_sequences = len(test_dataloader.dataset)
        print(f"Number of sequences in the dataloader: {num_sequences}")

        return test_dataloader














