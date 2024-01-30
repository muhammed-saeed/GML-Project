# -*- coding: utf-8 -*-
import torch
from replay_buffer import ReplayBuffer
from ddpg import DDPG
import numpy as np
from actor_critic import ActorCritic
from params import  (
    BUFFER_SIZE, BATCH_SIZE, 
    EPSILON, UPDATE_EVERY, 
    EPSILON_MIN, EXPLORATION_STEPS,
    N_EXPERIENCES, DEVICE  
)


class MADDPG():
    def __init__(self, state_size, action_size, n_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed
        # Number of agents
        self.n_agents = n_agents
         # Exploration noise
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.exploration_steps = EXPLORATION_STEPS
        self.epsilon_decay = (self.epsilon - (self.epsilon_min) * N_EXPERIENCES) / (self.exploration_steps ) 
        self.noise_enabled = True
        # Timestep progressive counter
        self.timestep_counter = 0
        # Setup n DDPG agents
        self.agents = self.setup_agents()
        # Experience Replay
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def setup_agents(self):
        agents = []
        for i in range(self.n_agents):
            model = ActorCritic(
                n_agents=self.n_agents, state_size=self.state_size, action_size=self.action_size, seed=self.random_seed
            )
            agents.append(DDPG(i, model, self.action_size, self.random_seed))
        return agents
        
    def step(self, states, actions, rewards, next_states, dones):
        # Flat states and next states
        states = states.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        # Add experience to the buffer
        self.memory.add(states, actions, rewards, next_states, dones)
        
        self.timestep_counter = self.timestep_counter + 1 

        # Learn from our buffer if possible
        if len(self.memory) > BATCH_SIZE and self.timestep_counter % UPDATE_EVERY == 0:
            # Sample experiences for each agent
            for _ in range(N_EXPERIENCES):
                experiences = [self.memory.sample() for _ in range(self.n_agents)]
                self.learn(experiences)
                
    
    def act(self, states):
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act(state, noise_value=self.epsilon, add_noise=self.noise_enabled)
            actions.append(action)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
                
        # Return flattened actions
        return np.array(actions).reshape(1, -1)
    
    def checkpoint(self):
        # Save actor and critic weights for each agent
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_regular.state_dict(),  "actor_agent_{}.pth".format(i))
            torch.save(agent.critic_regular.state_dict(), "critic_agent_{}.pth".format(i))
            
    def load_weights(self):
        # Load Weights
        for i, agent in enumerate(self.agents):
            agent.actor_regular.load_state_dict(torch.load("actor_agent_{}.pth".format(i)))
            agent.critic_regular.load_state_dict(torch.load("critic_agent_{}.pth".format(i)))

    def learn(self, experiences):
        next_actions = []
        actions = []
        for i, agent in enumerate(self.agents):
            states, _ , _ , next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(DEVICE)            
            
            state = states.reshape(-1, self.action_size, self.state_size).index_select(1, agent_id).squeeze(1)
            action = agent.actor_regular(state)
            actions.append(action)

            next_state = next_states.reshape(-1, self.action_size, self.state_size).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            next_actions.append(next_action)
                       
        # Call to the method learn for each agent using
        # the related experiences and all actions/next actions
        for i, agent in enumerate(self.agents):
            agent.learn(self.memory, i, experiences[i], next_actions, actions)


