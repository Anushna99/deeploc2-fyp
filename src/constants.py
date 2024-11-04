
CATEGORIES = ["Membrane","Cytoplasm","Nucleus","Extracellular","Cell membrane","Mitochondrion","Plastid","Endoplasmic reticulum","Lysosome/Vacuole","Golgi apparatus","Peroxisome"]
SS_CATEGORIES = ["NULL", "SP", "TM", "MT", "CH", "TH", "NLS", "NES", "PTS", "GPI"] 

FAST = "Fast"
ACCURATE = "Accurate"
TEST_ESM = "Test_esm"
TEST_PROTT5 = "Test_prott5"

# An HDF5 (Hierarchical Data Format version 5) file, denoted with the .h5 extension, is a binary data format used for storing 
# large amounts of numerical data.
EMBEDDINGS = {
    FAST: {
        "embeds": "data_files/embeddings/esm1b_swissprot.h5",
        "config": "swissprot_esm1b.yaml",
        "source_fasta": "data_files/deeploc_swissprot_clipped1k.fasta"
    },
    ACCURATE: {
        "embeds": "data_files/embeddings/prott5_swissprot.h5",
        "config": "swissprot_prott5.yaml",
        "source_fasta": "data_files/deeploc_swissprot_clipped4k.fasta"
    },
    TEST_ESM:{
        "embeds": "data_files/embeddings/esm1b_hpa.h5",
        "config": "hpa_esm1b.yaml",
        "source_fasta": "data_files/deeploc_hpa_clipped1k.fasta"
    },
    TEST_PROTT5:{
        "embeds": "data_files/embeddings/prott5_hpa.h5",
        "config": "hpa_t5.yaml",
        "source_fasta": "data_files/deeploc_hpa_clipped4k.fasta"
    }
}

SIGNAL_DATA = "data_files/multisub_ninesignals.pkl"
LOCALIZATION_DATA = "./data_files/multisub_5_partitions_unique.csv"

# hyper parameters that can change
BATCH_SIZE = 256 
SUP_LOSS_MULT = 0.05
REG_LOSS_MULT = 0.05

batch_sizes = [64, 128, 256] 
sup_loss_mults = [0.05, 0.1, 0.2]
reg_loss_mults = [0.05, 0.1, 0.2]
