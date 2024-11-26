'''
Script for training the agent for snake using various methods
'''
# run on cpu
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from tqdm import tqdm
from collections import deque
import pandas as pd
import time
from utils import play_game2
from game_environment import Snake, SnakeNumpy
import tensorflow as tf
from agent import DeepQLearningAgent
import json

# Set the seed for reproducibility 
tf.random.set_seed(42)

# Load the training configuration
version = 'v17.1'

# get training configurations
with open('model_config/{:s}.json'.format(version), 'r') as f:
    config = json.load(f)
board_size = config['board_size']
frames = config['frames'] 
max_time_limit = config['max_time_limit']
supervised = bool(config['supervised'])
n_actions = config['n_actions']
obstacles = bool(config['obstacles'])
buffer_size = config['buffer_size']

# Define Training Parameters
episodes = 100_000 # Total number of episodes
log_frequency = 500 # Frequency of logging metrics
games_eval = 8 # Number of games for evaluation
epsilon, epsilon_end = 1, 0.01 # Epsilon range for Epsilon Greedy Policy
decay = 0.97 # Decay rate for Epsilon
reward_type = 'current' # Reward type for training
n_games_training = 8 * 16 # Number of games for training
sample_actions = False # Sample actions for training

# Initialize the Agent
agent = DeepQLearningAgent(
    board_size=board_size, 
    frames=frames, 
    n_actions=n_actions, 
    buffer_size=buffer_size, 
    version=version
)

# agent.print_models()

# Load pretrained model and buffer if supervised learning is enabled
if(supervised):
    epsilon = 0.01 # Lower exploartion for pre-trained model
    try:
        agent.load_model(file_path=f'models/{version}')
        agent.load_buffer(file_path=f'models/{version}', iteration=1)
    except FileNotFoundError:
        print('Pre-trained model or buffer NOT found. Training from scratch.')
else:
    # Fill the replay buffer with initial experiences for non-supervised training
    initial_games = 512
    env_init = SnakeNumpy(
        board_size=board_size,
        frames=frames,
        max_time_limit=max_time_limit,
        games=initial_games,
        frame_mode=True,
        obstacles=obstacles,
        version=version
    )
    start_time = time.time()
    _ = play_game2(
        env=env_init,
        agent=agent,
        n_actions=n_actions,
        n_games=initial_games,
        record=True,
        epsilon=epsilon,
        verbose=True,
        reset_seed=False,
        frame_mode=True,
        total_frames=initial_games * 64
    )
    print(f'Initial buffer filled with {initial_games * 64} frames in {time.time() - start_time:.2f} seconds.')

# decay = np.exp(np.log((epsilon_end/epsilon))/episodes)

# Set up the environments for training and evaluation
env_train = SnakeNumpy(
    board_size=board_size, 
    frames=frames,
    max_time_limit=max_time_limit,
    games=n_games_training,
    frame_mode=True,
    obstacles=obstacles,
    version=version
)
env_eval = SnakeNumpy(
    board_size=board_size,
    frames=frames,
    max_time_limit=max_time_limit,
    games=games_eval,
    frame_mode=True,
    obstacles=obstacles,
    version=version
)

# Initialize the model logs
model_logs = {
    'iteration': [],
    'reward_mean': [],
    'length_mean': [],
    'games': [],
    'loss': []
}

# Training Loop
for index in tqdm(range(episodes)):
    
    # Play games and collect experiences
    _, _, _ = play_game2(
        env=env_train,
        agent=agent,
        n_actions=n_actions,
        epsilon=epsilon,
        n_games=n_games_training,
        record=True,
        sample_actions=sample_actions,
        reward_type=reward_type,
        frame_mode=True,
        total_frames=n_games_training,
        stateful=True
    )
    
    # Train the agent
    loss = agent.train_agent(batch_size=64)
    
    # Evaluate the agent periodically
    if (index+1) % log_frequency == 0:
        rewards, lengths, games = play_game2(
            env=env_eval,
            agent=agent,
            n_actions=n_actions,
            n_games=games_eval,
            epsilon=-1, # No exploration during evaluation
            record=False,
            sample_actions=False,
            frame_mode=True,
            total_frames=-1,
            total_games=games_eval
        )
        avg_reward = round(int(rewards) / games, 2)
        avg_length = round(int(lengths) / games, 2)
        
        # Log results
        model_logs['iteration'].append(index+1)
        model_logs['reward_mean'].append(avg_reward)
        model_logs['length_mean'].append(avg_length)
        model_logs['games'].append(games)
        model_logs['loss'].append(loss)
        
        pd.DataFrame(model_logs).to_csv(
            f'model_logs/{version}.csv',
            index=False
        )
        
        # Save the model and update the target network
        agent.update_target_net()
        agent.save_model(file_path=f'models/{version}', iteration=(index+1))
        
    # Decay the epsilon value
    epsilon = max(epsilon * decay, epsilon_end)
