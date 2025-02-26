import os
import torch
import torch.nn as nn
import torch.optim as optim
import hockey.hockey_env as h_env
import DreamingNets  
import numpy as np
import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
print("Using device:", device)

def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def generate_data(num_samples):
    data = np.random.randn(num_samples, config.obs_space).astype(np.float32)
    return data


def compute_env_normalization(env, num_samples=1000):
    obs_list = []
    obs, info = env.reset()
    for _ in range(num_samples):
        obs_list.append(obs)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, info = env.reset()
    obs_array = np.array(obs_list)
    mean = np.mean(obs_array, axis=0)
    std = np.std(obs_array, axis=0) + 1e-6 # Avoid division by 0
    return mean, std


def train_autoencoder(num_steps=200000, batch_size=32, use_binarization=False, log_filename="training_log.txt"):

    env = h_env.HockeyEnv()
    player1 = h_env.BasicOpponent(weak=False)
    player2 = h_env.BasicOpponent(weak=True)

    autoencoder = DreamingNets.BinaryAutoencoder().to(device)
    autoencoder.apply(initialize_weights)

    loss_fn = config.AE_CRITERION

    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.9)


    norm_mean = config.MEAN
    norm_std = config.STD
    print("Computed normalization stats:")
    print("Mean:", norm_mean)
    print("Std: ", norm_std)


    autoencoder.train()
    
    obs_buffer = []
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    with open(log_filename, "a") as log_file:
        for step in range(1, num_steps + 1):
            obs_norm = (np.array(obs) - norm_mean) / norm_std
            #obs_norm = np.array(obs) / 5
            obs_buffer.append(obs_norm)
            
            a1 = player2.act(obs)
            a2 = player2.act(obs_agent2)
            obs, reward, done, truncated, info = env.step(np.hstack([a1,a2]))
            obs_agent2 = env.obs_agent_two()
            if done or truncated:
                obs, info = env.reset()

            if len(obs_buffer) >= batch_size:
                batch_obs = np.array(obs_buffer)
                batch_tensor = torch.tensor(batch_obs, dtype=torch.float32, device=device)
                
                latent, reconstruction = autoencoder(batch_tensor, use_binarization=use_binarization)
                recon_loss = loss_fn(reconstruction, batch_tensor)
                latent_reg = torch.mean(torch.abs(latent))
                loss = recon_loss + 0.0001 * latent_reg 
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=0.9)
                optimizer.step()
                scheduler.step()
                
                obs_buffer = []
                
                if step % 1000 == 0:
                    print(f"Step {step}/{num_steps} - Recon Loss: {recon_loss.item():.4f} - Total Loss: {loss.item():.4f}")
                    log_line = f"{step}, {loss.item():.6f}\n"
                    log_file.write(log_line)
                    log_file.flush()
    
    # Save the trained autoencoder model
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "autoencoder.pth")
    torch.save(autoencoder.state_dict(), save_path)
    print(f"Autoencoder saved at {save_path}")
    env.close()

    
if __name__ == "__main__":
    train_autoencoder(num_steps=3000000, batch_size=100, use_binarization=True,  log_filename="autoencoder_log.txt")