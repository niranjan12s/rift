import torch
import os 


# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_FILE = "/content/training_data.txt" 
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

VOCAB_SIZE = 100          
MAX_SEQ_LEN = 16           # sequence length (tokens per sample) - Reduced for memory
BATCH_SIZE = 16        # Reduced batch size to mitigate memory issues
NUM_DIMENSIONS = 8        # wave num_dimensions - Reduced for memory
WAVELET_PARAMS = 3         # amplitude, omega, phase (A, w, phi)
HIDDEN_DIM = 64
EPOCHS = 20
LR = 1e-6 # Further reduced learning rate to mitigate nan loss
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 1.0
TIME_STEPS = 16            # number of time steps for wave emission (tunable) - Reduced for memory
NUM_SAMPLES = 8000 # Set NUM_SAMPLES to 8000 as requested
# Loss weights
LAMBDA_TOKEN = 1.0
LAMBDA_WAVE_RECON = 0.5
LAMBDA_SEM_ALIGN = 0.2
# diversity / contrastive optional (not enabled by default)
USE_DIVERSITY_PENALTY = True
LAMBDA_DIVERSITY = 0.1

# Sampling params
SAMPLE_MAX_TOKENS = 40
TOP_K = 50
TOP_P = 0.9
TEMPERATURE = 0.8
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# --- END CONFIG ---