import torch
import os
from torchvision import transforms

os.environ["IPDB_CONTEXT_SIZE"] = "7"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z_DIM = 64
HIDDEN_DIM = 64
BETA_1, BETA_2 = 0.5, 0.999
DISPLAY_STEP = 1
N_EPOCHS = 100
BATCH_SIZE = 32
LR = 2e-4
C_LAMBDA = 10  # weighting the gradient penalty
CRIT_REPEATS = 5  # 1 GENERATOR TRAININGs = 5 CRITIC TRAININGs
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
DSET_SUBSET_SIZE = 1000
LOG_PATH = "./logs"
BASE_DIR = "."
CHECKPOINT_DIR = "./checkpoints"
IMAGE_DIR = "./images"
