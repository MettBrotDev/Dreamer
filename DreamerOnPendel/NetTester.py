import gymnasium as gym
import torch
import numpy as np
import config as config
from NetTraining import WorldModel  # Your world model class
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
import os

def plot_states(reduced_states, title):
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_states[:, 0], reduced_states[:, 1], alpha=0.5)
    plt.title(title)
    plt.savefig(f"Plot-{title}.png")
    plt.show()

def test_actor(plots=False, images=False): 
    env = gym.make("Pendulum-v1", render_mode='human')
    #env = gym.wrappers.RecordVideo(env, video_folder='videos/', episode_trigger=lambda e: True)
    obs = env.reset()[0]
    done = False

    z_states = []
    h_states = []

    world = WorldModel()
    world.loadModel(config.FILE_NAME2)
    
    h = torch.zeros(config.SNET_NUM_LAYERS, 1, config.H_SIZE, device=config.DEVICE)

    c = torch.tensor([0.0], dtype=torch.float32, device=config.DEVICE)
    
    for step in range(500):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=config.DEVICE)
        
        z_embedded, z_probs, x_mean, x_logvar = world.autoencoder.forward(obs_tensor, world.steps, hard=True)
        
        action = world.agent.act(h, z_embedded)
        action_np = action.cpu().detach().numpy()

        print(action_np)
        
        result = env.step(action_np)
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, done, truncated, info = result
            done = done or truncated
        
        env.render()

        c = torch.tensor([done], dtype=torch.float32, device=config.DEVICE)
        h = world.sequence_predictor.forward(z_embedded, action, c, h).detach()

        z_states.append(z_embedded)
        h_states.append(h)

        
        if done:
            print("Episode finished after {} timesteps".format(step+1))
            break

    z_states = torch.stack(z_states).cpu().detach().numpy()
    h_states = torch.stack(h_states).cpu().detach().numpy()

    z_states = z_states.reshape(z_states.shape[0], -1)
    h_states = h_states.reshape(h_states.shape[0], -1)

    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)

    z_pca = pca.fit_transform(z_states)
    z_tsne = tsne.fit_transform(z_states)
    h_pca = pca.fit_transform(h_states)
    h_tsne = tsne.fit_transform(h_states)

    if plots:
        plot_states(z_pca, "PCA of z states")
        plot_states(z_tsne, "t-SNE of z states")
        plot_states(h_pca, "PCA of h states")
        plot_states(h_tsne, "t-SNE of h states") 

    '''normal_img = env.render()

    env.unwrapped.state = obs_tensor.cpu().detach().numpy().astype(np.float64)

    test_img = env.render()

    z_sampple, z_probs, obs_rec, _ = world.autoencoder.forward(obs_tensor, 0, hard=True)
    env.unwrapped.state = obs_rec.cpu().detach().numpy().astype(np.float64)
    autoencoded_img = env.render()

    world.dream_storage = []
    dreamed_imgs = []
    world.z = z_sampple
    world.h = h
    world.dream(done)
    sequence = world.dream_storage
    for state in sequence:
        h = state[0]
        z = state[1]
        z_probs = state[2]
        r_mean = state[3]
        r_logvar = state[4]
        c = state[5]
        c_probs = state[6]

        dreamed_decoded, decode_var = world.autoencoder.decode(z)
        env.unwrapped.state = dreamed_decoded.cpu().detach().numpy().astype(np.float64)
        dreamed_imgs.append(env.render())
    
    if images:
        saved_images_dir = "./test_images/"
        os.makedirs(saved_images_dir, exist_ok=True)
        normal_filename = os.path.join(saved_images_dir, "anormal.png")
        test_filename = os.path.join(saved_images_dir, "aotest.png")
        autoenc_filename = os.path.join(saved_images_dir, "autoencoded.png")
        Image.fromarray(normal_img).save(normal_filename)
        Image.fromarray(test_img).save(test_filename)
        Image.fromarray(autoencoded_img).save(autoenc_filename)
        for i, dreamed_img in enumerate(dreamed_imgs):
            dreamed_filename = os.path.join(saved_images_dir, f"dreamed_{i:02d}.png")
            Image.fromarray(dreamed_img).save(dreamed_filename)'''

    env.close()



    
if __name__ == '__main__':
    test_actor(plots=True, images=False)
