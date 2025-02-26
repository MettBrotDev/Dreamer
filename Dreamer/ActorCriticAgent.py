import config
import DreamingNets
import torch



class Agent:
    def __init__(self, h_size, z_slots, embed_dim, a_size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.actor = DreamingNets.Actor(h_size + (z_slots * embed_dim), a_size).to(self.device)
        self.critic = DreamingNets.Critic(h_size + (z_slots * embed_dim)).to(self.device)
        self.actor_optimizer = config.OPTIMIZER(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = config.OPTIMIZER(self.critic.parameters(), lr=1e-4)
        self.discount = config.DISCOUNT
    
    def get_combined(self, h, z_embedded):
        if h.dim() > 2:
            h_last = h[-1]
        else:
            h_last = h
        z_embedded_flat = z_embedded.flatten(start_dim=1)
        return torch.cat([h_last, z_embedded_flat], dim=-1)
    
    def act(self, h, z_embedded):
        input_vector = self.get_combined(h, z_embedded).to(self.device)
        with torch.no_grad():
            action = self.actor(input_vector)
        return action.squeeze(0)
    

    def update_from_dream(self, dream_sequence):
        # Update the critic and actor using the dream trajectory produced by the world model.

        imagined_values = []
        rewards = []
        combined_list = []
        
        for step in dream_sequence:
            h_step, z_sample, z_probs, r_mean, r_logvar, c_sample, c_probs = step
            h_step = h_step
            z_sample = z_sample
            r_mean = r_mean

            r_mean = DreamingNets.symlog(r_mean) if config.USE_SYMLOG else r_mean
            
            combined = self.get_combined(h_step, z_sample)
            combined_list.append(combined)
            value = self.critic(combined)
            imagined_values.append(value)
            rewards.append(r_mean)
        
        imagined_values = torch.stack(imagined_values)  
        rewards = torch.stack(rewards)
        
        # Compute discounted returns
        T = rewards.size(0)
        returns = []
        R = imagined_values[-1].detach()
        for t in reversed(range(T)):
            R = rewards[t] + self.discount * R
            returns.insert(0, R)
        returns = torch.stack(returns)

        if config.USE_SYMLOG:
            returns = DreamingNets.symlog(returns)
        
        # Critic loss: mean squared error between predicted values and returns, both symlogged.
        critic_loss = ((imagined_values - returns.detach()) ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss: encourage actions that lead to higher critic value.
        actor_loss = - DreamingNets.symexp(self.critic(combined_list[0]).mean()) if config.USE_SYMLOG else - self.critic(combined_list[0]).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()