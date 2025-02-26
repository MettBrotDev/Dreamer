import config as config
import DreamingNets
import torch
import PinkNoiseAtBenno as PinkNoise



class Agent:
    def __init__(self, h_size, z_slots, embed_dim, a_size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.actor = DreamingNets.Actor(h_size + (z_slots * embed_dim), a_size).to(self.device)
        self.critic = DreamingNets.Critic(h_size + (z_slots * embed_dim)).to(self.device)
        self.actor_optimizer = config.OPTIMIZER(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = config.OPTIMIZER(self.critic.parameters(), lr=1e-4)
        self.discount = config.DISCOUNT
        self.PinkNoise = PinkNoise.PinkNoise((config.A_SIZE, config.TRAIN_STEPS))
    
    def get_combined(self, h, z_embedded):
        if h.dim() > 2:
            h_last = h[-1]
        else:
            h_last = h
        z_embedded_flat = z_embedded.flatten(start_dim=1)
        return torch.cat([h_last, z_embedded_flat], dim=-1)
    
    def act(self, h, z_embedded):
        input_vector = self.get_combined(h, z_embedded).to(self.device)
        action = self.actor(input_vector)
        noise = config.NoiseEps * self.PinkNoise()
        noise = torch.tensor(noise, dtype=torch.float32, device=self.device)
        action = action + noise
        return action.squeeze(0)
    

    def update_from_dream(self, dream_sequence):
        # Update the critic and actor using the dream trajectory produced by the world model.
        imagined_values = []
        rewards = []
        combined_list = []
        
        for step in dream_sequence:
            h_step, z_sample, z_probs, r_mean, r_logvar, c_sample, c_probs, a = step

            r_mean = DreamingNets.symlog(r_mean) if config.USE_SYMLOG else r_mean
            r_mean = r_mean.detach()
            h_step = h_step.detach()
            z_sample = z_sample.detach()
            
            combined = self.get_combined(h_step, z_sample)
            combined_list.append(combined)
            value = self.critic(combined, a)
            imagined_values.append(value)
            rewards.append(r_mean)
        
        imagined_values = torch.stack(imagined_values)  
        rewards = torch.stack(rewards)
        T = rewards.size(0)

       # Compute TD targets for critic
        targets = []
        for t in range(T-1):
            a_next = self.actor(combined_list[t+1])
            q_next = self.critic(combined_list[t+1], a_next).detach()
            target = rewards[t] + self.discount * q_next
            targets.append(target)
        targets.append(rewards[-1].unsqueeze(0))
        targets = torch.stack(targets)
        
        # Critic loss: mean squared error between predicted values and targets, both symlogged.
        critic_loss = ((imagined_values - targets) ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor loss: maximize the expected return
        actor_loss = 0
        for t in range(T):
            a_t = self.actor(combined_list[t])
            q_t = self.critic(combined_list[t], a_t)
            actor_loss -= q_t.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()