import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Unity env executable path
UNITY_EXE_PATH = '/local/musaeed/Tennis_Linux/Tennis.x86_64'
# Environment Goal
GOAL = 0.51
# Averaged score
SCORE_AVERAGED = 100
# Let us know the progress each 100 episodes
PRINT_EVERY = 50
# Number of episode for training
N_EPISODES = 3000
# Max Timesteps
MAX_TIMESTEPS = 1000
# Replay Buffer Size
BUFFER_SIZE = 12000
# Minibatch Size
BATCH_SIZE = 256 
# Discount Gamma
GAMMA = 0.995 
# Soft Update Value
TAU = 1e-3
# Learning rates for each NN      
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
# Update network every X intervals
UPDATE_EVERY = 2
# Learn from batch of experiences n_experiences times
N_EXPERIENCES = 4
# Noise parameters
OU_MU = 0.0
# Volatility
OU_SIGMA = 0.2       
# Speed of mean reversion   
OU_THETA = 0.15 
# Noise exploration parameters
EPSILON = 1
EPSILON_MIN = 0.01
# Exploration steps related to N_EXPERIENCES
EXPLORATION_STEPS = 12000 
