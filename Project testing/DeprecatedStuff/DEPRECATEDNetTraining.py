import config
import DreamingNets
import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym
import torch
import os
from torch import nn
from PIL import Image

'''Cuda stuff'''
torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print("Device: " + str(device))

class WorldModel:
    def __init__(self):
        self.sequence_predictor = DreamingNets.SequencePredictor()
        self.reward_net = DreamingNets.reward_net()
        self.continuation_net = DreamingNets.continuation_net()
        self.dynamics_predictor = DreamingNets.DynamicsPredictor()

        self.sequence_predictor.apply(self.initialize_weights)
        self.reward_net.apply(self.initialize_weights)
        self.continuation_net.apply(self.initialize_weights)
        self.dynamics_predictor.apply(self.initialize_weights)

        self.parameters = list(self.sequence_predictor.parameters()) + list(self.reward_net.parameters()) + list(self.continuation_net.parameters()) + list(self.dynamics_predictor.parameters())
        self.optimizer = config.OPTIMIZER(self.parameters, lr=config.LR, weight_decay=config.WEIGHT_DECAY)

        self.h = torch.zeros(config.SNET_NUM_LAYERS, config.H_SIZE)
        self.next_h = torch.zeros(config.SNET_NUM_LAYERS, config.H_SIZE)
        self.z = torch.zeros(config.Z_SIZE, requires_grad=True)
        self.c_next = torch.zeros(1)
        self.dream_space = []

        self.actor = h_env.BasicOpponent(weak=True)

        print('World model initialized')

    # helps initializing the weights of the networks
    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    

    # DreamSpace:
    # A list of lists. Each list is a dream sequence.
    # Each dream sequence is a list of tuples (h, z, r).
    # The first sequence is the oldest dream.


    def dream(self, c):
        # Dreaming
        dreamed_sequence = []
        h_prev = self.h
        z_prev = self.z
        c = torch.tensor([c], dtype=torch.float32)

        for i in range(config.DREAM_STEPS):
            a = torch.tensor(self.actor.act(z_prev.cpu().detach().numpy()), dtype=torch.float32)

            _, h_next = self.sequence_predictor.forward(z_prev, a, c, h_prev)
            z = self.dynamics_predictor.forward(h_next[-1])
            r = self.reward_net.forward(z.unsqueeze(0))
            c = self.continuation_net.forward(z.unsqueeze(0))

            dreamed_sequence.append((h_next.clone(), z.clone(), r.clone(), c.clone()))

            h_prev = h_next
            z_prev = z
            if i == 0:
                self.h_next = h_next
                self.c_next = c
        self.dream_space.append(dreamed_sequence)


    def computeLosses(self, z_pred, z, r_pred, r, c_pred, c):
        loss = 0

        # Normalize rewards
        r = r / 10
        r = torch.tensor([r], dtype=torch.float32)

        c = torch.tensor([c], dtype=torch.float32)

        # Reward loss
        loss += self.reward_net.computeLoss(r_pred, r) * config.REWARD_SCALE
        # Continuation loss
        loss += self.continuation_net.computeLoss(c_pred, c) * config.CONTINUATION_SCALE
        # Dynamics loss
        loss += self.dynamics_predictor.computeLoss(z_pred, z) * config.DYNAMICS_SCALE

        return loss


    def trainingStep(self, z, r, c):
        # Training
        total_loss = torch.tensor(0.0)
        new_dream_space = []

        for seq in self.dream_space:
                if not seq:
                    continue
                
                h_pred, z_pred, r_pred, c_pred = seq[0]
                seq = seq[1:]
                
                loss = self.computeLosses(z_pred, z, r_pred, r, c_pred, c) * np.power(config.DISCOUNT, 20 - len(seq))
                total_loss = total_loss + loss
                
                if seq:
                    new_dream_space.append(seq)

        self.dream_space = new_dream_space
        self.dream(c) 
        return total_loss

    
    def trainingSingle(self, env):
        total_loss = torch.tensor(0.0)
        self.h = torch.zeros(config.SNET_NUM_LAYERS, config.H_SIZE)

        #Setup environment
        obs, info = env.reset()
        obs_agent2 = env.obs_agent_two()
        player2 = h_env.BasicOpponent(weak=True)
        self.z = torch.tensor(obs, dtype=torch.float32, requires_grad=True)
        c = torch.tensor([0], dtype=torch.float32)

        #First initializations
        self.dream(c)

        #Train
        for i in range(config.TRAIN_STEPS):
            #get next state
            if c: 
                print(f'Goal achived, resetting || Prediction: {self.c_next}')
                env.reset()
            a1 = self.getAction(obs)
            a2 = player2.act(obs_agent2)
            obs, r, c, t, info = env.step(np.hstack([a1,a2]))
            obs_agent2 = env.obs_agent_two()
            if t: break

            self.z = torch.tensor(obs, dtype=torch.float32)
            self.h = self.h_next #Use latest h from dream space
            
            total_loss = total_loss + self.trainingStep(self.z, r, c)

            if i % config.UPDATE_INTERVAL == 0:
                print('Step: ' + str(i) + ' Loss: ' + str(total_loss))
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Reset everything
                total_loss = torch.tensor(0.0, requires_grad=True)
                self.dream_space = []
                self.h = self.h.detach()
                self.z = self.z.detach() #not really needed but just to be sure
                self.h_next = self.h_next.detach()

        # Use remaining data
        print('Step: ' + str(i) + ' Loss: ' + str(total_loss))
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Reset everything
        total_loss = torch.tensor(0.0, requires_grad=True)
        self.dream_space = []
        self.h = self.h.detach()
        self.z = self.z.detach() #not really needed but just to be sure
        self.h_next = self.h_next.detach()     
        env.close()


    def testRun(self):
        env = h_env.HockeyEnv()
        obs, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        player2 = h_env.BasicOpponent(weak=True)

        self.z = torch.tensor(obs, dtype=torch.float32, requires_grad=True)
        self.h = torch.zeros(config.SNET_NUM_LAYERS, config.H_SIZE)
        c = torch.tensor([0], dtype=torch.float32)

        self.dream(c)

        for i in range(20):
            img = env.render(mode='rgb_array')
            img = Image.fromarray(img)
            img.save(f'./test/img{i}.png')
            a1 = self.getAction(obs)
            a2 = player2.act(obs_agent2)
            obs, r, c, t, info = env.step(np.hstack([a1,a2]))
            obs_agent2 = env.obs_agent_two()
            if t: break

            self.z = torch.tensor(obs, dtype=torch.float32)
            self.h = self.h_next

            self.trainingStep(self.z, r, c)

        a1 = self.getAction(obs)
        a2 = player2.act(obs_agent2)
        obs, r, c, t, info = env.step(np.hstack([a1,a2]))
        obs_agent2 = env.obs_agent_two()

        self.h = self.h_next
        self.z = torch.tensor(obs, dtype=torch.float32)


        obs = self.z.cpu().detach().numpy()
        obs = obs.astype(np.float64)
        env.set_state(obs)

        img_base = env.render(mode='rgb_array')
        img_base = Image.fromarray(img_base)
        img_base.save(f'./test/img_base2.png')

        env_list = []

        for i, seq in enumerate(self.dream_space):
                if not seq:
                    continue
                
                h_pred, z_pred, r_pred, c_pred = seq[0]
                
                env = h_env.HockeyEnv()
                obs = z_pred.cpu().detach().numpy()
                obs = obs.astype(np.float64)
                env.set_state(obs)
                env_list.append(env)

                # print the difference for each value of z to the z_pred
                diff = z_pred - self.z
                print('Differences : ' + str(diff)) 

                #print('Real      r: ' + str(r))
                #print('Predicted r: ' + str(r_pred))
                
                loss = self.computeLosses(z_pred, self.z, r_pred, r, c_pred, c)

                print('Loss: ' + str(loss))
        
        for i, env in enumerate(env_list):
            #print('Render: ' + str(i))
            img = env.render(mode='rgb_array')
            img = Image.fromarray(img)
            img.save(f'./test/dream{20 - i}.png')
        
        
                

    '''Not sure why i even have this function right now but im sure it will be useful later on'''
    def trainIteration(self):
        env = h_env.HockeyEnv()
        for i in range(config.TRAIN_SEQUENCES):
            self.trainingSingle(env)
            print(f'Sequence {i} done')
        print('Training done')
        self.saveModel()

    def getAction(self, obs):
        return self.actor.act(obs)
    

    def saveModel(self):
        whole_model = {'sequence_predictor': self.sequence_predictor.state_dict(), 'reward_net': self.reward_net.state_dict(), 'dynamics_predictor': self.dynamics_predictor.state_dict()}
        torch.save(whole_model, config.FILE_NAME)
        print('Models saved')


    def loadModel(self):
        #check for file 
        if not os.path.isfile(config.FILE_NAME):
            print('No model found')
            return
        print(f"Loading model {config.FILE_NAME}")
        whole_model = torch.load(config.FILE_NAME)
        self.sequence_predictor.load_state_dict(whole_model['sequence_predictor'])
        self.reward_net.load_state_dict(whole_model['reward_net'])
        self.dynamics_predictor.load_state_dict(whole_model['dynamics_predictor'])
        print('Models loaded')


if __name__ == '__main__':
    world = WorldModel()
    world.loadModel()
    world.trainIteration()
    world.testRun()
