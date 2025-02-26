import torch
import torch.nn.functional as F


# Training configurations
TRAIN_SEQUENCES = 10
TRAIN_STEPS = 300
UPDATE_INTERVAL = 30

# Loss configurations
DISCOUNT = 0.85
REWARD_SCALE = 0.5
CONTINUATION_SCALE = 0.5
DYNAMICS_SCALE = 1
REP_SCALE = 0.1
AUTOENCODER_SCALE = 0.1
USE_SYMLOG = True

# World configurations
obs_space = 18
Z_SIZE = 320  # Size of internal representation, maybe needs to be bigger
H_SIZE = 64  # Size of memory
A_SIZE = 4

# Dreaming configurations
DREAM_STEPS = 15

# Network configurations

# RewardNet
RNET_HIDDEN_SIZE = 32
RNET_OUTPUT_SIZE = 1
RNET_CRITERION = torch.nn.MSELoss()

# ContinuationNet
CNET_HIDDEN_SIZE = 32
CNET_OUTPUT_SIZE = 1
def cLoss(output, target): weights = torch.where(target == 1, 15, 1); return F.binary_cross_entropy(output, target, weight=weights)
CNET_CRITERION = cLoss

# DynamicsPredictor
DNET_HIDDEN_LAYER_1 = 48
DNET_HIDDEN_LAYER_2 = 32
#DNET_CRITERION = torch.nn.MSELoss()
def dLoss(output, target): return F.kl_div(F.log_softmax(output, dim=-1), F.softmax(target, dim=-1), reduction="batchmean")
DNET_CRITERION = dLoss

# SequencePredictor
SNET_NUM_LAYERS = 2

# Autoencoder
AE_HIDDEN_SIZE_1 = 32
AE_HIDDEN_SIZE_2 = 64
AE_HIDDEN_SIZE_3 = 128
AE_CRITERION = torch.nn.L1Loss()

# Environment information
MEAN = [-2.33932778e+00, -1.89206378e-02,  1.79042931e-04,  2.54489586e-01,
-4.94385641e-03, -2.68711885e-03,  2.33085515e+00, -3.62293576e-02,
  1.20932573e-03, -2.51300624e-01, -6.64584314e-03,  3.80889795e-04,
  2.26191062e-02, -2.79261558e-03, -2.93161766e-02, -1.24017527e-03,
  9.24450000e-02,  8.59350000e-02]
STD = [0.93985386, 1.50506239, 0.61684625, 2.00974784, 2.08708858, 3.08088639,
 0.9392112, 1.48261537, 0.61751399, 2.106373,   2.19956273, 3.05867426,
 1.60724903, 0.87539951, 4.05959484, 3.09234419, 1.0836379, 1.05379423]

# Optimizer
OPTIMIZER = torch.optim.Adam
SCHEDULER = torch.optim.lr_scheduler.StepLR
LR = 0.0005

# Save configurations
FILE_NAME = 'world_model.pth'
LOG_FILE_NAME = 'world_model.log'
AE_FILE_NAME = 'models/autoencoderL320.pth'