import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import config as config
import numpy as np
import math


'''
Neurnonal network to predic the reward.
We need this to predict which reward we would get based on our "dreaming" into the future.

In and outputs:
Its inputs will be the current innner representation of the world z (maybe the memory of the agent h).
The output will be a single prediction of the reward which will be used to train the agent.

Training: 
We will train the reward network by using the reward signal from the environment.
We will run 32 sequences of 300 steps and use that as one training batch.
'''

class reward_net(nn.Module):
    def __init__(self, z_slots=config.Z_SLOTS, embed_dim=config.EMBED_DIM, hidden_size=config.RNET_HIDDEN_SIZE, output_size=config.RNET_OUTPUT_SIZE):
        super(reward_net, self).__init__()
        self.fc1 = nn.Linear(z_slots* embed_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, 1)
        self.fc_logvar = nn.Linear(hidden_size, 1)
        self.fc_logvar.is_logvar = True
        self.activation = nn.LeakyReLU()
    
    def forward(self, z_embedded):
        z_embedded = z_embedded.flatten()
        x = self.activation(self.fc1(z_embedded))
        x = self.activation(self.fc2(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        # Add some stability with clamping
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mean, logvar
    



'''
Neurnonal network to get the world state z out of the internal representation h.
We need this to get the world state after predicting the next state h.

In and outputs:
Its inputs will be the inner world representation h.
The output will be the world state z.

Training: 
We will train the dynamics predictor by using the world state z from the environment using the MSE.
'''
class DynamicsPredictor(nn.Module):
    def __init__(self, latent_embedding,input_size=config.H_SIZE,  hidden_size1=config.DNET_HIDDEN_LAYER_1, hidden_size2=config.DNET_HIDDEN_LAYER_2, num_categories=config.Z_CATEGORIES, output_slots=config.Z_SLOTS):
        super(DynamicsPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc_logits = nn.Linear(hidden_size2, num_categories * output_slots)
        self.activation = nn.LeakyReLU()
        self.num_categories = num_categories
        self.output_slots = output_slots
        self.embedding = latent_embedding
    
    def forward(self, x, steps, hard=False):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        logits = self.fc_logits(x)
        logits = logits.view(-1, self.output_slots, self.num_categories)
        z_probs = F.gumbel_softmax(logits, tau=tau_scheduler(steps), hard=hard)

        z_embedded = self.embedding(z_probs)

        return z_embedded, z_probs
    
    

'''
Neurnonal network to predic the continuation.
If the output is 0 that means the game doesnt continue after this step. (else 1)
We mainly need this for our sequence Predictor to know when the environment resets.

In and outputs:
Its inputs will be the current innner representation of the world z (maybe the memory of the agent h).
The output will be a single prediction of the continuation which will be used to train the agent.
This output is binary so its either 0 or 1.

Training: 
We will train the reward network by using the continuation signal from the environment.
'''
class ContinuationNet(nn.Module):
    def __init__(self, z_slots=config.Z_SLOTS, embed_dim=config.EMBED_DIM, hidden_size=config.CNET_HIDDEN_SIZE, output_size=config.CNET_OUTPUT_SIZE):
        super(ContinuationNet, self).__init__()
        self.fc1 = nn.Linear(z_slots * embed_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, z_embedded):
        z_embedded = z_embedded.flatten()
        x = self.activation(self.fc1(z_embedded))
        logits = self.fc2(x)
        probs = torch.sigmoid(logits)
        c = probs + (torch.bernoulli(probs) - probs).detach()
        return c, probs

'''
Neurnonal network to predict the next inner representation of the world h.
We need this to predict the next state of the world based on the current state.

In and outputs:
Its inputs will be the current innner representation of the world h, as well as the action taken and z.
The output will be the next inner representation of the world h.

Training:
We combine all of the other losses into one and train the agent with that.
'''
class SequencePredictor(nn.Module):
    def __init__(self, z_slots=config.Z_SLOTS, embed_dim=config.EMBED_DIM):
        super(SequencePredictor, self).__init__()
        self.gru = nn.GRU(input_size= z_slots * embed_dim + config.A_SIZE + config.CNET_OUTPUT_SIZE, hidden_size=config.H_SIZE, num_layers=config.SNET_NUM_LAYERS, batch_first=True)
        self.layer_norm = nn.LayerNorm(config.H_SIZE)

    def forward(self, z_embedded, a, c, h_prev):
        z_flat = z_embedded.clone().view(1, -1)    # Forgot why i dont use flatten here, no time to test though
        a = a.unsqueeze(0)
        c = c.unsqueeze(0)
        z_a_c = torch.cat((z_flat, a, c), dim=1).unsqueeze(0)
        _, h_next = self.gru(z_a_c, h_prev)
        h_next = self.layer_norm(h_next)
        return h_next.squeeze(0)
    

''''
Binary Autoencoder to encode and decode the obeservation of the environment.

In and outputs:
Its inputs will be the current observation of the environment.
The output will be the encoded observation of the environment in form of a binary matrix

Training:
We can automaticly train this one using the reconstruction loss.'''
class BinaryAutoencoder(nn.Module):
    def __init__(self, latent_embedding, num_categories=config.Z_CATEGORIES, z_slots=config.Z_SLOTS, embed_dim=config.EMBED_DIM):
        super(BinaryAutoencoder, self).__init__()

        self.num_categories = num_categories
        self.output_slots = z_slots

        self.encoder = nn.Sequential(
            nn.Linear(config.obs_space, config.AE_HIDDEN_SIZE_1),
            nn.ReLU(),
            nn.Linear(config.AE_HIDDEN_SIZE_1, config.AE_HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(config.AE_HIDDEN_SIZE_2, config.AE_HIDDEN_SIZE_2),
            nn.ReLU()
        )
        self.encoder_logits = nn.Linear(config.AE_HIDDEN_SIZE_2, num_categories * z_slots)
        self.embedding = latent_embedding

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * z_slots, config.AE_HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(config.AE_HIDDEN_SIZE_2, config.AE_HIDDEN_SIZE_1),
            nn.Sigmoid()
        )
        self.fc_dec_mean = nn.Linear(config.AE_HIDDEN_SIZE_1, config.obs_space)
        self.fc_dec_logvar = nn.Linear(config.AE_HIDDEN_SIZE_1, config.obs_space)
        self.fc_dec_logvar.is_logvar = True
    
    def forward(self, x, steps, hard=False):
        x = self.normalize(x)
        x = self.encoder(x)
        logits = self.encoder_logits(x) * 3
        logits = logits.clone().view(-1, self.output_slots, self.num_categories)
        z_probs = F.gumbel_softmax(logits, tau=tau_scheduler(steps), hard=hard)

        z_embedded = self.embedding(z_probs)

        x_mean, x_logvar = self.decode(z_embedded)

        return z_embedded, z_probs, x_mean, x_logvar
    
    def decode(self, z_embedded):
        x = F.relu(self.decoder(z_embedded.clone().flatten()))
        x_mean = self.fc_dec_mean(x)
        x_logvar = self.fc_dec_logvar(x)
        x = self.denormalize(x_mean)
        x = torch.clamp(x, min=-10, max=10)
        return x_mean, x_logvar
    
    '''Normalization for more stable training.'''
    def normalize(self, x):
        return symlog(x)

    def denormalize(self, x_norm):
        return symexp(x_norm)
    

'''Transformer to interpret the latent space z, which consists of one hot vectors now.
EDIT: I dont know what tf i was thinking, i wanted to add complexity because i was underfitting but that was NOT it.'''
class SlotAttentionTransformer(nn.Module):
    def __init__(self, num_slots=config.Z_SLOTS, slot_dim=config.Z_CATEGORIES, num_heads=config.SAT_NUM_HEADS, num_layers=config.SAT_NUM_LAYERS):
        super(SlotAttentionTransformer, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=slot_dim, nhead=num_heads), num_layers=num_layers)

    def forward(self, z):
        # Batch dimension if not present
        if z.dim() == 2:
            z = z.unsqueeze(0)
        z = self.encoder(z)
        return z


class LatentEmbedding(nn.Module):
    def __init__(self, categories=config.Z_CATEGORIES, embed_dim=config.EMBED_DIM):
        super(LatentEmbedding, self).__init__()
        self.embedding = nn.Embedding(categories, embed_dim)

    def forward(self, z):
        embedded = torch.matmul(z, self.embedding.weight)
        return embedded
        

'''SymLog function to scale the reward and loss.
They use this in the Dreamer implementation to stabilize their values.'''
def symlog(x, eps=1e-5):
    return torch.sign(x) * torch.log1p(x.abs() + eps)

def symexp(x, eps=1e-5):
    return torch.sign(x) * (torch.exp(x.abs()) - 1 + eps)


'''Changes the tau value for the gumbel softmax over time.
This is used to make the model more deterministic and therefore hopefully more accurate over time.'''
def tau_scheduler(steps, tau_min=config.TAU_MIN, tau_max=config.TAU_MAX, decay_steps=config.ANNEALING_STOP):
    return max(tau_min, tau_max - (tau_max - tau_min) * steps / decay_steps)


'''Compute the kl divergence loss which is used in the dreamer v3 paper.
It takes 2 probability tensors and computes the kl divergence between them.'''
def compute_kl_divergence(q_probs, p_probs, steps, clip_threshold=config.FREE_NATS):

    # Clamp to avoid infinite loss
    q_probs = q_probs.clone().clamp(min=1e-8, max=1-1e-8)
    p_probs = p_probs.clone().clamp(min=1e-8, max=1-1e-8)

    q_dist = dist.Categorical(probs=q_probs)
    p_dist = dist.Categorical(probs=p_probs)
    
    kl_per_slot = dist.kl_divergence(q_dist, p_dist)

    kl_weight = logistic_kl_weight(steps) 
    if config.KL_USE_ANNEALING:
        kl_per_slot = kl_weight * kl_per_slot
    
    # Clip the KL divergence, important: Clipping for each value, that was wrong before and caused the model to have issues learning
    kl_clipped = torch.clamp(kl_per_slot, max=clip_threshold)

    kl_clipped_mean = kl_clipped.mean()
    kl_mean = kl_per_slot.mean()

    if kl_clipped.isinf().any():
        print("KL divergence is infinite")
    
    return kl_clipped_mean, kl_mean



''''Logistic reduction of the kl weight in the first x0 steps.
K is the slope of the function and x0 the point where the function is at 0.5.'''
def logistic_kl_weight(step, k=config.KL_K, x0=config.KL_X0):
    return max(1, float(1 / (1 + math.exp(-k * (step - x0)))))







'''Actor and Critic networks for the agent.'''

class Actor(nn.Module):
    def __init__(self, h_size, a_size, hidden_size=config.ACTOR_HIDDEN_SIZE_1):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(h_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, a_size),
            #nn.Tanh()  # I thought that was smart but thinking about it a second time, this would potentially hinder my predictions and not help them
        )
        
    def forward(self, h):
        return self.net(h)

class Critic(nn.Module):
    def __init__(self, h_size, hidden_size=config.CRITIC_HIDDEN_SIZE_1):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(h_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, h):
        return self.net(h)