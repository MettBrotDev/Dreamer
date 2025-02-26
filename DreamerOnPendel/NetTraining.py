import config as config
import DreamingNets
import numpy as np
import gymnasium as gym
import torch
import os
from torch import nn
from PIL import Image
from torch.nn import functional as F
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from ActorCriticAgent import Agent

'''Cuda stuff'''
torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
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

        self.writer = SummaryWriter(log_dir=config.LOG_DIR3)

        self.h = torch.zeros(config.SNET_NUM_LAYERS, 1, config.H_SIZE)
        self.h_next = torch.zeros(config.SNET_NUM_LAYERS, 1, config.H_SIZE)
        self.z = torch.zeros(config.Z_SLOTS, config.EMBED_DIM, requires_grad=True)
        self.c_next = torch.zeros(1)
        self.action_next = torch.zeros(config.A_SIZE)
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
            nn.init.kaiming_uniform_(layer.weight, a=0.01, nonlinearity='relu')
            if hasattr(layer, 'is_logvar') and layer.is_logvar:
                nn.init.constant_(layer.bias, 0.2)
            else:
                nn.init.zeros_(layer.bias)

    def dream(self, c):
        # Dreaming
        dreamed_sequence = []
        h_prev = self.h
        z_prev = self.z
        c = torch.tensor([c], dtype=torch.float32)

        for i in range(config.DREAM_STEPS):

            a = self.agent.act(h_prev, z_prev).detach()

            h_next = self.sequence_predictor.forward(z_prev, a, c, h_prev)

            z_embedded, z_probs = self.dynamics_predictor(h_next[-1], self.steps)
            
            r_mean, r_logvar = self.reward_net(z_embedded)

            c_sample, c_probs = self.continuation_net(z_embedded)
            c = c_sample
            
            dreamed_sequence.append((h_next.detach(), z_embedded.detach(), z_probs.detach(), r_mean.detach(), r_logvar.detach(), c_sample.detach(), c_probs.detach()))

            # Since we are no longer training on multi step dreams we detach all further dreams from eachother
            h_prev = h_next.detach()
            z_prev = z_embedded.detach()
            if i == 0:
                self.h_next = h_next
                self.c_next = c
                self.dream_next = [h_next, z_embedded, z_probs, r_mean, r_logvar, c_sample, c_probs]
                self.action_next = a
        self.dream_storage = dreamed_sequence

    # Prediction loss
    def computePredicitonLoss(self, obs, z_embedded, r_mean, r_logvar, r, c_probs, c, summary=False):

        # Loss for the reconstruction 
        x_mean, x_logvar = self.autoencoder.decode(z_embedded)
        x_std = torch.exp(0.5 * x_logvar)

        obs_dist = dist.Normal(x_mean, x_std)
        nll_x = - obs_dist.log_prob(obs).mean()

        if nll_x.isinf():
            print("NLL_X is inf")


        # Loss for the reward
        r = torch.tensor([r], dtype=torch.float32)
        r_std = torch.exp(0.5 * r_logvar)
        r_dist = dist.Normal(r_mean, r_std)
        nll_r = - r_dist.log_prob(r).mean()
        nll_r = torch.clamp(nll_r, max=10.0)

        # Loss for the continuation
        c = torch.tensor([c], dtype=torch.float32)
        c_dist = dist.Bernoulli(c_probs)
        weight = torch.where(c == 1, torch.tensor(config.CNET_GOAL_WEIGHT), torch.tensor(1.0))
        nll_c = - (c_dist.log_prob(c) * weight).mean()
        nll_c = torch.clamp(nll_c, max=10.0)

        if summary:
            self.writer.add_scalar('Loss/NLL_X', nll_x, self.steps)
            self.writer.add_scalar('Loss/NLL_R', nll_r, self.steps)
            self.writer.add_scalar('Loss/NLL_C', nll_c, self.steps)
            self.writer.add_histogram('Latent/h', self.h, self.steps)
            self.writer.add_histogram('Latent/z', self.z, self.steps)
        
        return nll_x * config.RECONSTRUCTION_SCALE + nll_r * config.REWARD_SCALE + nll_c * config.CONTINUATION_SCALE
    
    # dynamics and representation loss
    def computeDynamicsLoss(self, z_enc_probs, z_dyn_probs, summary=False):
        z_enc_probs = z_enc_probs.detach()
        loss, unclippedLoss = DreamingNets.compute_kl_divergence(z_enc_probs, z_dyn_probs, self.steps)

        if summary:
            self.writer.add_scalar('Loss/Dynamics', loss, self.steps)
            self.writer.add_scalar('Loss/Dynamics_unclipped', unclippedLoss, self.steps)

        return loss
    
    def computeRepresentationLoss(self, z_enc_probs, z_dyn_probs, summary=False):
        z_dyn_probs = z_dyn_probs.detach()
        loss, unclippedLoss = DreamingNets.compute_kl_divergence(z_enc_probs, z_dyn_probs, self.steps)

        return loss

    def trainingStep(self, obs, r, c, summary=False):
        # Training step using stochastic latent extraction
        z_enc, z_enc_probs, _, _ = self.autoencoder.forward(obs, self.steps)
        new_dream_space = []

        h_pred, z_embedded, z_dyn_probs, r_mean, r_logvar, c_sample, c_probs = self.dream_next

        prediction_loss = self.computePredicitonLoss(obs, z_embedded, r_mean, r_logvar, r, c_probs, c, summary=summary)
        dynamics_loss = self.computeDynamicsLoss(z_enc_probs, z_dyn_probs, summary=summary)
        representation_loss = self.computeRepresentationLoss(z_enc_probs, z_dyn_probs, summary=summary)

        total_loss = prediction_loss * config.PREDICTION_SCALE + dynamics_loss * config.DYNAMICS_SCALE + representation_loss * config.REPRESENTATION_SCALE

        # Actor critic update
        actor_loss, critic_loss = self.agent.update_from_dream(self.dream_storage)

        if summary:
            self.writer.add_scalar('Loss/Actor', actor_loss, self.steps)
            self.writer.add_scalar('Loss/Critic', critic_loss, self.steps)
            self.writer.add_scalar('Loss/Total', total_loss, self.steps)

        if c.item() != 0:
            c_sample, c_probs = self.continuation_net.forward(z_enc)
            r_mean, r_logvar = self.reward_net.forward(z_enc)
            print(f"Goal scored || c_probs: {c_probs} || r: {r} || r_mean: {r_mean} || r_logvar: {r_logvar}")

        
        self.dream_space = new_dream_space

        self.dream(c)

        self.steps += 1

        return total_loss

    def trainingRound(self, env):
        total_loss = torch.tensor(0.0)
        self.h = torch.zeros(config.SNET_NUM_LAYERS, 1, config.H_SIZE)
        self.h_next = torch.zeros(config.SNET_NUM_LAYERS, 1, config.H_SIZE)
        self.z = torch.zeros(config.Z_SLOTS, config.EMBED_DIM, requires_grad=True)

        # Setup environment
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, requires_grad=True)
        self.z, _, _, _ = self.autoencoder.forward(obs_t, self.steps)
        c = torch.tensor([0], dtype=torch.float32)

        # First initializations
        self.dream(c)

        # Train
        for i in range(config.TRAIN_STEPS):
            if c.item() != 0: 
                env.reset()
            # Using stored action to ensure consistency with the dream, since there is noise on the action otherwise
            action = self.action_next
            obs, r, c, t, _ = env.step(action)
            if t: break

            c = torch.tensor([c], dtype=torch.float32)
            obs_t = torch.tensor(obs, dtype=torch.float32)
            self.z, _,  _, _ = self.autoencoder.forward(obs_t, self.steps)
            self.h = self.h_next  # next latent state as current latent state

            total_loss = total_loss + self.trainingStep(obs_t, r, c, summary=(i % config.UPDATE_INTERVAL == 0 or True))

            if i % config.UPDATE_INTERVAL == 0 and i != 0:
                total_loss = total_loss / config.UPDATE_INTERVAL
                with open(config.LOG_FILE_NAME, "a") as log_file:
                    log_line = f"{self.steps}, {total_loss.item():.6f}\n"
                    log_file.write(log_line)
                    log_file.flush()
                print('Step: ' + str(i) + ' Loss: ' + str(total_loss))
                self.optimizer.zero_grad()
                total_loss.backward()
                # I would usually clip for stability but they are explicitly only clipping rep and dyn loss so thats what im gonna do
                # torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=1.0)
                self.optimizer.step()
                if self.steps >  2000:
                    self.scheduler.step()

                total_loss = torch.tensor(0.0, requires_grad=True)
                self.full_detach()

        # Using leftover data
        total_loss = total_loss / config.UPDATE_INTERVAL
        with open(config.LOG_FILE_NAME, "a") as log_file:
            log_line = f"{self.steps}, {total_loss.item():.6f}\n"
            log_file.write(log_line)
            log_file.flush()
        print('Step: ' + str(i) + ' Loss: ' + str(total_loss))
        self.optimizer.zero_grad()
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=0.9)
        self.optimizer.step()

        total_loss = torch.tensor(0.0, requires_grad=True)
        self.full_detach()
        env.close()

    def full_detach(self):
        # Detaches everything
        self.dream_space = []
        self.h = self.h.detach()
        self.z = self.z.detach()            # not really needed but just to be sure
        self.h_next = self.h_next.detach()  # not needed either, since it gets overwritten before its used anyway
        if self.dream_next:
            self.dream_next = [x.detach() for x in self.dream_next]

    def trainIteration(self):
        env = gym.make('Pendulum-v1')
        self.steps = 0
        for i in range(config.TRAIN_SEQUENCES):
            self.trainingRound(env)
            print(f'Sequence {i} done')
        print('Training done')
        self.saveModel()

    def saveModel(self):
        whole_model = {'sequence_predictor': self.sequence_predictor.state_dict(),
                       'reward_net': self.reward_net.state_dict(),
                       'dynamics_predictor': self.dynamics_predictor.state_dict(),
                       'continuation_net': self.continuation_net.state_dict(),
                       'autoencoder': self.autoencoder.state_dict(),
                       'embedding': self.latent_embedding.state_dict(),
                       'actor': self.agent.actor.state_dict(),
                       'critic': self.agent.critic.state_dict()}
        torch.save(whole_model, config.FILE_NAME3)
        print('Models saved')

    def loadModel(self, file_name=config.FILE_NAME3):
        if not os.path.isfile(file_name):
            print('No model found')
            return
        print(f"Loading model {file_name}")
        whole_model = torch.load(file_name)
        self.sequence_predictor.load_state_dict(whole_model['sequence_predictor'])
        self.reward_net.load_state_dict(whole_model['reward_net'])
        self.dynamics_predictor.load_state_dict(whole_model['dynamics_predictor'])
        self.continuation_net.load_state_dict(whole_model['continuation_net'])
        self.autoencoder.load_state_dict(whole_model['autoencoder'])
        self.latent_embedding.load_state_dict(whole_model['embedding'])
        self.agent.actor.load_state_dict(whole_model['actor'])
        self.agent.critic.load_state_dict(whole_model['critic'])
        print('Models loaded')

    # Deprecated code, used for pretrained autoencoder that doesnt exist anymore
    def loadAutoencoder(self):
        if not os.path.isfile(config.AE_FILE_NAME):
            print('No model found')
            return
        print(f"Loading model {config.AE_FILE_NAME}")
        model = torch.load(config.AE_FILE_NAME)
        self.autoencoder.load_state_dict(model)
        print('Autoencoder loaded')
        # Optionally freeze the autoencoder if needed later on
        # for param in self.autoencoder.parameters():
        #     param.requires_grad = False



if __name__ == '__main__':
    world = WorldModel()
    #world.loadModel()
    world.trainIteration()
