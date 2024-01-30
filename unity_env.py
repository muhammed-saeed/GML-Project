from unityagents import UnityEnvironment

def init_environment(executable_path, train=True): 
    
    # Init the Reacher Unity Environment
    env = UnityEnvironment(file_name=executable_path, no_graphics=True)
    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # Reset the environment
    env_info = env.reset(train_mode=train)[brain_name]
    # Number of agents
    n_agents = len(env_info.agents)
    # Size of each action
    action_size = brain.vector_action_space_size
    
    # Get state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    
    return env, brain_name, n_agents, state_size, action_size