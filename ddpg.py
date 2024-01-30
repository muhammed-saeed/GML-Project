
import torch
import torch.nn.functional as F
import torch.optim as optim
from ou_noise import OUNoise
import numpy as np
from params import  (
   TAU, LR_CRITIC, LR_ACTOR,  
   OU_MU, OU_THETA, OU_SIGMA,
   GAMMA, DEVICE
)

class DDPG():

    def __init__(self, agent_id, model, action_size, random_seed):
        self.id = agent_id

        # Actor Neural Network (Regular and target)
        self.actor_regular = model.actor_regular
        self.actor_target = model.actor_target
        self.actor_optimizer = optim.Adam(self.actor_regular.parameters(), lr=LR_ACTOR)
        
        # Critic Neural Network (Regular and target)
        self.critic_regular = model.critic_regular
        self.critic_target = model.critic_target
        self.critic_optimizer = optim.Adam(self.critic_regular.parameters(), lr=LR_CRITIC)
     
        # Exploration noise
        self.noise = OUNoise(action_size, random_seed, OU_MU, OU_THETA, OU_SIGMA)
        
        # Ensure that both networks have the same weights
        self.deep_copy(self.actor_target, self.actor_regular)
        self.deep_copy(self.critic_target, self.critic_regular)
        
        
    def act(self, states, noise_value, add_noise=True):
        states = torch.from_numpy(states).float().to(DEVICE)
        self.actor_regular.eval()
        
        with torch.no_grad():
            action = self.actor_regular(states).cpu().data.numpy()
        
        self.actor_regular.train()
        
        if add_noise:
            # Include exploration noise
            action += noise_value * self.noise.sample()

        # Clip action to the right interval
        return np.clip(action, -1, 1)

    def learn(self, memory, agent_id, experiences, all_next_actions, all_actions):
        states, actions, rewards, next_states, dones = experiences
        
        # Update the critic neural network
        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id]).to(DEVICE)
        actions_next = torch.cat(all_next_actions, dim=1).to(DEVICE)

        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, actions_next)
            
        Q_expected = self.critic_regular(states, actions)
        # Compute Q targets for current states filtered by agent id
        Q_targets = rewards.index_select(1, agent_id) + (GAMMA * Q_targets_next * (1 - dones.index_select(1, agent_id)))
        
        # Calculate the critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        
        # Minimize the loss
        critic_loss.backward()
        # Critic gradient clipping to 1
        torch.nn.utils.clip_grad_norm_(self.critic_regular.parameters(), 1) 
        self.critic_optimizer.step()
        
        # Update the actor neural network
        self.actor_optimizer.zero_grad()
        # Detach actions of other agents
        actions_pred = [actions if i == self.id else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(DEVICE)
        actor_loss = -self.critic_regular(states, actions_pred).mean()
        
        # Minimize the loss function
        actor_loss.backward()

        self.actor_optimizer.step()

        # Update target network using the soft update approach (slowly updating)
        self.soft_update(self.critic_regular, self.critic_target)
        self.soft_update(self.actor_regular, self.actor_target)
        
        
    def soft_update(self, local_model, target_model):
        # Update the target network slowly to improve the stability
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU) * target_param.data)

    def deep_copy(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


