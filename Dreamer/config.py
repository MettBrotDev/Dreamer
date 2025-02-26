import torch
import torch.nn.functional as F

DEVICE = torch.set_default_device('cuda')

# Training configurations
TRAIN_SEQUENCES = 500
TRAIN_STEPS = 200
UPDATE_INTERVAL = 50

# Loss configurations
DISCOUNT = 0.95
DYNAMICS_SCALE = 2.5
REPRESENTATION_SCALE = 0.2
PREDICTION_SCALE = 1.0
REWARD_SCALE = 0.5
CONTINUATION_SCALE = 0.5
RECONSTRUCTION_SCALE = 2.0

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
KL_X_POINT = 0.3
KL_X0 = TRAIN_STEPS * TRAIN_SEQUENCES * KL_X_POINT

# World configurations
obs_space = 18
Z_SLOTS = 10        # Size of latent space
Z_CATEGORIES = 10
EMBED_DIM = 24
H_SIZE = 48        # Size of memory
A_SIZE = 4

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

# Critic
CRITIC_HIDDEN_SIZE_1 = 128


# SAT ---- DEPRECATED
SAT_NUM_HEADS = 4
SAT_NUM_LAYERS = 1

# Environment information
# I calculated them over 1 million steps and then manually decreased some standart deviations to make the model pay more attention 
MEAN = [-2.35774315e+00, -3.00303251e-02, -2.78874670e-03,  2.48868849e-01,
 -5.22817959e-03,  2.06030181e-03,  2.33903901e+00, -2.56888424e-02,
 -3.01197069e-03, -2.52728237e-01, -5.72023838e-03,  3.54300698e-04,
 -4.95584178e-03, -6.62168269e-03,  1.06919653e-02, -1.79777017e-05,
  8.72990000e-02,  8.85220000e-02]
STD = [0.93167627, 1.49304449, 0.61912315, 2.20160867, 2.3096616,  3.5772082,
 0.93621738, 1.49037185, 0.61797483, 2.20455734, 2.29323621, 3.56556584,
 1.60528252, 0.86930457, 6.51361546, 4.93583941, 1.06175045, 1.06668458]

# Save configurations
FILE_NAME = 'world_model.pth'
FILE_NAME2 = 'world_model2.pth'
LOG_FILE_NAME = 'world_model.log' # Total loss log not used anymore
LOG_DIR1 = 'world_model_losses1.log'
LOG_DIR2 = 'world_model_losses2.log'
AE_FILE_NAME = 'models/autoencoderL320.pth'