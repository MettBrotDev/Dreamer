import torch 
import torch.nn as nn
import torch.nn.functional as F
import config
import numpy as np


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
    def __init__(self, input_size=config.Z_SIZE, hidden_size=config.RNET_HIDDEN_SIZE, output_size=config.RNET_OUTPUT_SIZE):
        super(reward_net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        x = torch.clamp(x, min=-10, max=10)
        x = x.squeeze(-1)
        return x
    
    def computeLoss(self, output, target):
        if config.USE_SYMLOG:
            output = symlog(output)
            target = symlog(target)
        return config.RNET_CRITERION(output, target)



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
    def __init__(self, input_size=config.H_SIZE, output_size=config.Z_SIZE):
        super(DynamicsPredictor, self).__init__()
        
        self.dynamics = nn.Sequential(
            nn.Linear(config.H_SIZE, config.DNET_HIDDEN_LAYER_1),
            nn.LeakyReLU(),
            nn.Linear(config.DNET_HIDDEN_LAYER_1, config.DNET_HIDDEN_LAYER_2),
            nn.LeakyReLU(),
            nn.Linear(config.DNET_HIDDEN_LAYER_2, config.Z_SIZE),
            nn.Sigmoid()
        )
    
    def forward(self, x, use_binarization=True):
        latent_z = self.dynamics(x)
        if use_binarization:
            # Binarize using a straight-through estimator (STE)
            binary_z = (latent_z > 0.5).float() + latent_z - latent_z.detach()
        else:
            binary_z = latent_z
        return binary_z
    
    def computeLoss(self, output, target):
        if config.USE_SYMLOG and False:
            output = symlog(output)
            target = symlog(target)
        return config.DNET_CRITERION(output, target)
    

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
class continuation_net(nn.Module):
    def __init__(self, input_size=config.Z_SIZE, hidden_size=config.CNET_HIDDEN_SIZE, output_size=config.CNET_OUTPUT_SIZE):
        super(continuation_net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = x.squeeze(-1)
        return x
    
    def computeLoss(self, output, target):                     # Adjust weight due to class imbalance
        return config.CNET_CRITERION(output, target)


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
    def __init__(self):
        super(SequencePredictor, self).__init__()
        self.gru = nn.GRU(input_size=config.Z_SIZE + config.A_SIZE + config.CNET_OUTPUT_SIZE, hidden_size=config.H_SIZE, num_layers=config.SNET_NUM_LAYERS, batch_first=True)

    def forward(self, z, a, c, h_prev):
        z_a_c = torch.cat((z, a, c), dim=0).unsqueeze(0)
        h, h_next = self.gru(z_a_c, h_prev)
        return h, h_next
    

''''
Binary Autoencoder to encode and decode the obeservation of the environment.

In and outputs:
Its inputs will be the current observation of the environment.
The output will be the encoded observation of the environment in form of a binary matrix

Training:
We can automaticly train this one using the reconstruction loss.'''
class BinaryAutoencoder(nn.Module):
    def __init__(self):
        super(BinaryAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(config.obs_space, config.AE_HIDDEN_SIZE_1),
            nn.ReLU(),
            nn.Linear(config.AE_HIDDEN_SIZE_1, config.AE_HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(config.AE_HIDDEN_SIZE_2, config.AE_HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(config.AE_HIDDEN_SIZE_3, config.Z_SIZE),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(config.Z_SIZE, config.AE_HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(config.AE_HIDDEN_SIZE_3, config.AE_HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(config.AE_HIDDEN_SIZE_2, config.AE_HIDDEN_SIZE_1),
            nn.ReLU(),
            nn.Linear(config.AE_HIDDEN_SIZE_1, config.obs_space),
        )
        
    def forward(self, x, use_binarization=True):
        latent_z = self.encoder(x)

        if use_binarization:
            # Binarize using a straight-through estimator (STE)
            binary_z = (latent_z > 0.5).float() + latent_z - latent_z.detach()
        else:
            binary_z = latent_z
        
        output = self.decoder(binary_z)
        return latent_z, output
    
    def decode(self, z):
        return self.decoder(z)

        
    def computeLoss(self, x):
        z_encoded, x_reconstructed = self.forward(x)
        return config.AE_CRITERION(x_reconstructed, x)


'''SymLog function to scale the reward and loss.
They use this in the Dreamer implementation to stabilize their rewards.'''
def symlog(x, eps=1e-5):
    return torch.sign(x) * torch.log1p(x.abs() + eps)





    


# Old code for the reward net
# --------------------------------------------------------------------------------------------------------------------------------

    '''def train_episode(self, batch):
        train, test = train_test_split(batch, test_size=0.2)
        train = [(torch.tensor(x[0], dtype=torch.float), torch.tensor(x[1], dtype=torch.float).unsqueeze(-1)) for x in train]
        self.train_batch(train)
        test = [(torch.tensor(x[0], dtype=torch.float), torch.tensor(x[1], dtype=torch.float).unsqueeze(-1)) for x in test]

        validation_loss = self.validate_batch(test)
        print('RNET Validation loss: ', validation_loss)


    def train_batch(self, train_data):
        optimizer = config.RNET_OPTIMIZER(self.parameters(), lr=config.RNET_LR)
        criterion = config.RNET_CRITERION
        optimizer.zero_grad()

        for i in range(len(train_data)):
            output = self.forward(train_data[i][0])
            loss = criterion(output, train_data[i][1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
    
    def validate_batch(self, val_data):
        criterion = config.RNET_CRITERION
        total_loss = 0
        for i in range(len(val_data)):
            output = self.forward(val_data[i][0])
            loss = criterion(output, val_data[i][1])
            total_loss += loss.item()
        return total_loss / len(val_data)

    def predict(self, z):
        input = torch.tensor(z, dtype=torch.float)
        with torch.no_grad():
            return self.forward(input).item()'''