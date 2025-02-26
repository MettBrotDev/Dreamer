import torch
import torch.nn.functional as F

DEVICE = torch.set_default_device('cuda')

# Training configurations
TRAIN_SEQUENCES = 400
TRAIN_STEPS = 150
UPDATE_INTERVAL = 50

# Loss configurations
DISCOUNT = 0.99
DYNAMICS_SCALE = 1.0
REPRESENTATION_SCALE = 0.2
PREDICTION_SCALE = 1.0
REWARD_SCALE = 1.0
CONTINUATION_SCALE = 1.0
RECONSTRUCTION_SCALE = 1.0

USE_SYMLOG = True

# Optimizer
OPTIMIZER = torch.optim.Adam
SCHEDULER = torch.optim.lr_scheduler.StepLR
LR = 0.001

# Tau for gumble softmax
TAU_MIN = 0.1
TAU_MAX = 1.0
ANNEAL = 0.9
ANNEALING_STOP = TRAIN_STEPS * TRAIN_SEQUENCES * ANNEAL

# Kl divergence
FREE_NATS = 1.0
KL_USE_ANNEALING = True
KL_K = 0.001
KL_X_POINT = 0.4
KL_X0 = TRAIN_STEPS * TRAIN_SEQUENCES * KL_X_POINT

# World configurations
obs_space = 3
Z_SLOTS = 10        # Size of latent space
Z_CATEGORIES = 10
EMBED_DIM = 24
H_SIZE = 64        # Size of memory
A_SIZE = 1

# Dreaming configurations
DREAM_STEPS = 15

# Network configurations

# RewardNet
RNET_HIDDEN_SIZE = 128
RNET_OUTPUT_SIZE = 1

# ContinuationNet
CNET_HIDDEN_SIZE = 128
CNET_OUTPUT_SIZE = 1
# Adjust weight due to class imbalance
CNET_GOAL_WEIGHT = 10

# DynamicsPredictor
DNET_HIDDEN_LAYER_1 = 128
DNET_HIDDEN_LAYER_2 = 256

# SequencePredictor
SNET_NUM_LAYERS = 2

# Autoencoder
AE_HIDDEN_SIZE_1 = 128
AE_HIDDEN_SIZE_2 = 256

# Actor
ACTOR_HIDDEN_SIZE_1 = 128
NoiseEps = 0.1

# Critic
CRITIC_HIDDEN_SIZE_1 = 128


# SAT ---- DEPRECATED
SAT_NUM_HEADS = 4
SAT_NUM_LAYERS = 1



# Save configurations
FILE_NAME = 'world_model.pth'
FILE_NAME2 = 'world_model2.pth'
FILE_NAME3 = 'world_model3.pth'
LOG_FILE_NAME = 'world_model.log' # Entire loss log
LOG_DIR = 'world_model_losses.log' # Different losses
LOG_DIR2 = 'world_model_losses2.log' # Different losses
LOG_DIR3 = 'world_model_losses3.log' # Different losses
AE_FILE_NAME = 'models/autoencoderL320.pth'