import config
import DreamingNets
import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym
import torch
import os
from torch import nn
from PIL import Image
from torch.nn import functional as F
import torch.distributions as dist
from ModifyHockeyEnv import ModifiedHockeyEnv
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from ActorCriticAgent import Agent

'''Cuda stuff'''
torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print("Device: " + str(device))

class WorldModel:
    def __init__(self):
        self.latent_embedding = DreamingNets.LatentEmbedding()
        self.autoencoder = DreamingNets.BinaryAutoencoder(self.latent_embedding)
        self.sequence_predictor = DreamingNets.SequencePredictor()
        self.reward_net = DreamingNets.reward_net()
        self.continuation_net = DreamingNets.ContinuationNet()
        self.dynamics_predictor = DreamingNets.DynamicsPredictor(self.latent_embedding)

        self.latent_embedding.apply(self.initialize_weights)
        self.sequence_predictor.apply(self.initialize_weights)
        self.reward_net.apply(self.initialize_weights)
        self.continuation_net.apply(self.initialize_weights)
        self.dynamics_predictor.apply(self.initialize_weights)
        self.autoencoder.apply(self.initialize_weights)

        self.parameters = self.collect_parameters(self.sequence_predictor, self.reward_net, self.continuation_net, self.dynamics_predictor, self.autoencoder, self.latent_embedding)
        self.optimizer = config.OPTIMIZER(self.parameters, lr=config.LR)
        self.scheduler = config.SCHEDULER(self.optimizer, step_size=30, gamma=0.95)

        self.h = torch.zeros(config.SNET_NUM_LAYERS, 1, config.H_SIZE)
        self.h_next = torch.zeros(config.SNET_NUM_LAYERS, 1, config.H_SIZE)
        self.z = torch.zeros(config.Z_SLOTS, config.EMBED_DIM, requires_grad=True)
        self.c_next = torch.zeros(1)
        self.dream_next = []
        self.dream_storage = []

        self.steps = 0

        self.agent = Agent(h_size=config.H_SIZE, z_slots=config.Z_SLOTS, embed_dim=config.EMBED_DIM,
                           a_size=config.A_SIZE, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))



        print('World model initialized')
    def collect_parameters(self, *models):
        all_params = []
        for module in models:
            all_params += list(module.parameters())
        # remove duplicates (else you get problems with embedding being in multiple modules)
        unique_params = {id(p): p for p in all_params}.values()
        parameters = list(unique_params)
        return parameters


    # helps initializing the weights of the networks
    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


    def dream(self, c):
        # Dreaming
        dreamed_sequence = []
        h_prev = self.h
        z_prev = self.z
        c = torch.tensor([c], dtype=torch.float32)

        for i in range(config.DREAM_STEPS):
            a = self.agent.act(h_prev, z_prev)              

            h_next = self.sequence_predictor.forward(z_prev, a, c, h_prev)

            z_sample, z_probs = self.dynamics_predictor.forward(h_next[-1], 0, hard=True)

            r_mean, r_logvar = self.reward_net.forward(z_sample)

            c_sample, c_probs = self.continuation_net.forward(z_sample)
            c = c_sample
            
            dreamed_sequence.append((h_next.clone(), z_sample.clone(), z_probs.clone(), r_mean.clone(), r_logvar.clone(), c_sample.clone(), c_probs.clone()))

            h_prev = h_next
            z_prev = z_sample
            if i == 0:
                self.h_next = h_next
                self.c_next = c
                self.dream_next = dreamed_sequence
        self.dream_storage.append(dreamed_sequence)

    def loadModel(self):
        if not os.path.isfile(config.FILE_NAME):
            print('No model found')
            return
        print(f"Loading model {config.FILE_NAME}")
        whole_model = torch.load(config.FILE_NAME)
        self.sequence_predictor.load_state_dict(whole_model['sequence_predictor'])
        self.reward_net.load_state_dict(whole_model['reward_net'])
        self.dynamics_predictor.load_state_dict(whole_model['dynamics_predictor'])
        self.continuation_net.load_state_dict(whole_model['continuation_net'])
        self.autoencoder.load_state_dict(whole_model['autoencoder'])
        self.latent_embedding.load_state_dict(whole_model['embedding'])
        self.agent.actor.load_state_dict(whole_model['actor'])
        self.agent.critic.load_state_dict(whole_model['critic'])
        print('Models loaded')
    

    def plot_states(self, reduced_states, title):
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_states[:, 0], reduced_states[:, 1], alpha=0.5)
        plt.title(title)
        plt.savefig(f"Plot-{title}.png")
        plt.show()


    def test_run(self, images=False, plots=False):
        z_states = []
        h_states = []

        env = ModifiedHockeyEnv()
        player1 = h_env.BasicOpponent(weak=True)
        self.actor = player1
        player2 = h_env.BasicOpponent(weak=False)

        obs1, info = env.reset()
        obs2 = env.obs_agent_two()
        self.h = torch.zeros(config.SNET_NUM_LAYERS, 1, config.H_SIZE)
        self.z, _, _, _ = self.autoencoder.forward(torch.tensor(obs1, dtype=torch.float32), 0, hard=True)
        done = 0
        for i in range(300):
            print("Collecting state: ", i)
            if done:
                env.reset()
            action1 = self.agent.act(self.h, self.z).cpu().detach().numpy()
            action2 = player2.act(obs2)
            obs1, reward, done, t, info = env.step(np.hstack([action1, action2]))
            obs2 = env.obs_agent_two()
            obs_tensor = torch.tensor(obs1, dtype=torch.float32)
            z_sample, z_probs, obs_rec, obs_var = self.autoencoder.forward(obs_tensor, 0, hard=True)
            self.h = self.h_next
            self.z = z_sample


            self.dream(done)
            for state in self.dream_storage[0]:
                h = state[0]
                z = state[1]
                z_probs = state[2]
                z_states.append(z)
                h_states.append(h)

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
            self.plot_states(z_pca, "PCA of z states")
            self.plot_states(z_tsne, "t-SNE of z states")
            self.plot_states(h_pca, "PCA of h states")
            self.plot_states(h_tsne, "t-SNE of h states")  


        normal_img = env.render(mode='rgb_array')

        z_sampple, z_probs, obs_rec, _ = self.autoencoder.forward(obs_tensor, 0, hard=True)
        env.set_state(obs_rec.cpu().detach().numpy().astype(np.float64))
        autoencoded_img = env.render(mode='rgb_array')

        self.dream_storage = []
        dreamed_imgs = []
        self.dream(done)
        sequence = self.dream_storage[0]
        for state in sequence:
            h = state[0]
            z = state[1]
            z_probs = state[2]
            r_mean = state[3]
            r_logvar = state[4]
            c = state[5]
            c_probs = state[6]

            dreamed_decoded, decode_var = self.autoencoder.decode(z)
            env.set_state(dreamed_decoded.cpu().detach().numpy().astype(np.float64))
            dreamed_imgs.append(env.render(mode='rgb_array'))

        if images:
            saved_images_dir = "./test_images/"
            os.makedirs(saved_images_dir, exist_ok=True)
            normal_filename = os.path.join(saved_images_dir, "anormal.png")
            autoenc_filename = os.path.join(saved_images_dir, "autoencoded.png")
            Image.fromarray(normal_img).save(normal_filename)
            Image.fromarray(autoencoded_img).save(autoenc_filename)
            for i, dreamed_img in enumerate(dreamed_imgs):
                dreamed_filename = os.path.join(saved_images_dir, f"dreamed_{i:02d}.png")
                Image.fromarray(dreamed_img).save(dreamed_filename)

        env.close()




if __name__ == '__main__':
    world = WorldModel()
    world.loadModel()
    world.test_run(images=False, plots=True)