# -*- coding: utf-8 -*-
from actor import Actor
from critic import Critic
from params import DEVICE

class ActorCritic():

    def __init__(self, n_agents, state_size, action_size, seed):
        critic_input_size = (state_size+action_size)*n_agents
        
        self.actor_regular = Actor(state_size, action_size, seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, seed).to(DEVICE)
        
        self.critic_regular = Critic(critic_input_size, seed).to(DEVICE)
        self.critic_target = Critic(critic_input_size, seed).to(DEVICE)