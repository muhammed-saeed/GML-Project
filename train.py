from unity_env import init_environment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from maddpg import MADDPG
from params import (
    GOAL, SCORE_AVERAGED, 
    PRINT_EVERY, N_EPISODES, 
    MAX_TIMESTEPS, UNITY_EXE_PATH,
    BUFFER_SIZE
)

def train(n_episodes=N_EPISODES):
    global_scores = []
    averaged_scores = []
    scores_deque = deque(maxlen=SCORE_AVERAGED)
    max_reward = 0.0
    
    for episode in range(1, N_EPISODES + 1):
        episode_rewards = []
        
        # Get the current states for each agent
        states = env.reset(train_mode=True)[brain_name].vector_observations 

        for t in range(MAX_TIMESTEPS):
            # Act according to our policy
            actions = agent.act(states)
            # Send the decided actions to all the agents
            env_info = env.step(actions)[brain_name]        
            # Get next state for each agent
            next_states = env_info.vector_observations     
            # Get rewards obtained from each agent
            rewards = env_info.rewards          
            # Info about if an env is done
            dones = env_info.local_done   
            # Learn from the collected experience
            agent.step(states, actions, rewards, next_states, dones)
            # Update current states
            states = next_states   
            # Add the rewards recieved
            episode_rewards.append(rewards) 
            
            # Stop the loop if an agent is done               
            if np.any(dones):                          
                break

        # After each episode, we add up the rewards that each agent received (without discounting), 
        # to get a score for each agent. This yields 2 (potentially different) scores. 
        # We then take the maximum of these 2 scores.
        episode_reward = np.max(np.sum(np.array(episode_rewards), axis=0))
        # Store episode results
        global_scores.append(episode_reward) 
        scores_deque.append(episode_reward) 
        avg_score = np.mean(scores_deque)
        averaged_scores.append(avg_score)
        
        if episode_reward > max_reward:
            max_reward = episode_reward
            
        if episode % PRINT_EVERY == 0:
            print('Episode {}\tAverage Score: {:.3f} MaxReward: {:.3f} Buffer : {}/{} Noise: {:.3f} Timestep: {}.'.format(
                episode, avg_score, max_reward, len(agent.memory), BUFFER_SIZE, agent.epsilon, agent.timestep_counter))  
            
        if avg_score >= GOAL:  
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, avg_score))
            agent.checkpoint()
            break
            
    return global_scores, averaged_scores

# Init the Tennis environment and get agents, state and action info
env, brain_name, n_agents, state_size, action_size = init_environment(UNITY_EXE_PATH)
agent = MADDPG(state_size=state_size, action_size=action_size, n_agents=n_agents, random_seed=89)
# Train the agent and get the results
scores, averages = train()

# Plot Statistics (Global scores and averaged scores)
plt.subplot(2, 1, 2)
plt.plot(np.arange(1, len(scores) + 1), averages)
plt.ylabel('Tennis Environment Average Score')
plt.xlabel('Episode #')
plt.savefig("/local/musaeed/tennis-maddpg/maddpgTennisGameScore.png")
plt.show()