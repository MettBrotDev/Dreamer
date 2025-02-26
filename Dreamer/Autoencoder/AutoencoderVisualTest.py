import torch
import torch.nn as nn
import numpy as np
import hockey.hockey_env as h_env
import config
import DreamingNets 
import os
from PIL import Image
from ModifyHockeyEnv import ModifiedHockeyEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
torch.set_default_device(device)
print("Device:", device)

def visualizeSamples(env, model, observations, num_samples=5):
    """
    For a given autoencoder model and a batch of observations,
    prints out the original and the reconstructed observations.
    """
    model.eval()
    with torch.no_grad():
        obs_norm = normalize(observations)
        internal_rep, reconstructions = model.forward(obs_norm)
    
    for _ in range(10):
        i = np.random.randint(0, len(observations))
        orig = observations[i].cpu().numpy()
        recon = denormalize(reconstructions[i]).cpu().numpy()
        print(f"Sample {i}:")
        print("Original:", orig)
        print("Reconstructed:", recon)
        print("Difference:", orig - recon)
        print("-" * 40)

        env.set_state(orig.astype(np.float64))
        img = env.render(mode='rgb_array')
        img = Image.fromarray(img)
        img.save(f"VisualSamples/{i}_orig.png")

        env.set_state(recon.astype(np.float64))
        img = env.render(mode='rgb_array')
        img = Image.fromarray(img)
        img.save(f"VisualSamples/{i}_recon.png")
        
def normalize(x):
    mean = torch.tensor(config.MEAN, dtype=torch.float32, device=device)
    std = torch.tensor(config.STD, dtype=torch.float32, device=device)
    return (x - mean) / std

def denormalize(x_norm):
    mean = torch.tensor(config.MEAN, dtype=torch.float32, device=device)
    std = torch.tensor(config.STD, dtype=torch.float32, device=device)
    return x_norm * std + mean


def main():
    autoencoder = DreamingNets.BinaryAutoencoder().to(device)
    
    model_path = config.AE_FILE_NAME if hasattr(config, 'AE_FILE_NAME') else "models/autoencoder.pth"
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location=device)
        autoencoder.load_state_dict(state_dict)
        print(f"Loaded autoencoder model from {model_path}")
    else:
        print("No saved model found at", model_path)
        return

    env = ModifiedHockeyEnv() 
    player1 = h_env.BasicOpponent(weak=False)
    player2 = h_env.BasicOpponent(weak=True)
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    observations = []
    for i in range(100):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        observations.append(obs_tensor)
        a1 = player2.act(obs)
        a2 = player2.act(obs_agent2)
        obs, reward, done, truncated, info = env.step(np.hstack([a1,a2]))
        obs_agent2 = env.obs_agent_two()
        if done or truncated:
            obs, info = env.reset()
    
    observations = torch.stack(observations)

    visualizeSamples(env, autoencoder, observations, num_samples=5)
    
    env.close()

if __name__ == '__main__':
    main()