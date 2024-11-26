"""
store all the agents here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import json
from replay_buffer import ReplayBufferNumpy

class DQNModel(nn.Module):
    """ Deep Q Network Model to approximate the Q-values"""
    def __init__(self, board_size, n_frames, n_actions, version):
        super(DQNModel, self).__init__()
        self.board_size = board_size
        self.n_frames = n_frames
        self.n_actions = n_actions

        # Load the model architecture from JSON
        with open(f'model_config/{version}.json', 'r') as f:
            config = json.load(f)

        layers = []
        in_channels = n_frames  # Starting input channels (frames)
        
        # Iterate through the layers specified in the JSON
        for layer_name, layer_config in config['model'].items():
            if "Conv2D" in layer_name:
                # Add Conv2D layer
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=layer_config['filters'],
                        kernel_size=tuple(layer_config['kernel_size']),
                        stride=layer_config.get('stride', 1),  # Default stride 1
                        padding=layer_config.get('padding', 0)
                    )
                )
                # Add activation function
                if layer_config['activation'] == 'relu':
                    layers.append(nn.ReLU())
                elif layer_config['activation'] == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif layer_config['activation'] == 'tanh':
                    layers.append(nn.Tanh())
                in_channels = layer_config['filters']  # Update in_channels for next layer
                
            elif "Flatten" in layer_name:
                # Add Flatten layer
                layers.append(nn.Flatten())

            elif "Dense" in layer_name:
                # Compute in_channels dynamically after Flatten
                if len(layers) > 0 and isinstance(layers[-1], nn.Flatten):
                    dummy_input = torch.zeros(1, n_frames, board_size, board_size)
                    with torch.no_grad():
                        in_channels = nn.Sequential(*layers)(dummy_input).numel()

                # Add Dense (Linear) layer
                layers.append(
                    nn.Linear(
                        in_features=in_channels,
                        out_features=layer_config['units']
                    )
                )
                # Add activation function for Dense layer
                if layer_config['activation'] == 'relu':
                    layers.append(nn.ReLU())
                elif layer_config['activation'] == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif layer_config['activation'] == 'tanh':
                    layers.append(nn.Tanh())
                in_channels = layer_config['units']  # Update in_channels for next layer

        # Store the sequential feature extractor
        self.features = nn.Sequential(*layers)        
        
        # Final output layer for Q-values
        self.q_values = nn.Linear(in_channels, n_actions)

    def forward(self, x):
        x = self.features(x)
        return self.q_values(x)

class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    <Other Agents Removed for Clarity of the Assignment>
    
    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """ initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._n_frames, self._board_size, self._board_size) # (C, H, W)
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size ** 2).reshape(self._board_size, -1)
        self._version = version

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        iteration = 0 if iteration is None else iteration
        with open(f"{file_path}/buffer_{iteration:04d}", 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        iteration = 0 if iteration is None else iteration
        with open(f"{file_path}/buffer_{iteration:04d}", 'rb') as f:
            self._buffer = pickle.load(f)

class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values

    Attributes
    ----------
    _model : TensorFlow Graph
        Stores the graph of the DQN model
    _target_net : TensorFlow Graph
        Stores the target network graph of the DQN model
    """
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True, version=''):
        """Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks
        """
        super().__init__(board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net, version=version)
        # Initialize the models, loss function, optimizer and device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using {self.device} for training")
        self._model = self._agent_model().to(self.device)
        self.optimizer = optim.Adam(self._model.parameters(), lr=0.001)
        self.criterion = nn.SmoothL1Loss() # Huber Loss for PyTorch
        if self._use_target_net:
            self._target_net = self._agent_model().to(self.device)
            self.update_target_net

    def _prepare_input(self, board):
        """Reshape input and normalize
        
        Parameters
        ----------
        board : Numpy array
            The board state to process

        Returns
        -------
        board : Numpy array
            Processed and normalized board
        """
        board = board.astype(np.float32) / 4.0 # Normalize       
        if board.ndim == 3:
            board = np.expand_dims(board, axis=0) # Add batch dimension
        board = torch.tensor(board).permute(0, 3, 1, 2).to(self.device) # (B, H, W, C) -> (B, C, H, W)
        return board

    def _get_model_outputs(self, board, model=None):
        """Get action values from the DQN model

        Parameters
        ----------
        board : Numpy array
            The board state for which to predict action values
        model : PyTorch model, optional
            The model to use for prediction, default is self._model

        Returns
        -------
        model_outputs : Numpy array
            Predicted model outputs on board,
            of shape (board.shape[0], num_actions)
        """
        # Prepare the input by normalizing and converting to tensor
        board = self._prepare_input(board)
        
        # Use the default model if none is provided
        if model is None:
            model = self._model
        
        # Forward pass through the model to get predictions
        with torch.no_grad():  # No gradients required for inference
            model_outputs = model(board).cpu().numpy()  # Convert tensor to numpy
        
        return model_outputs

    def _normalize_board(self, board):
        """Normalize the board before input to the network
        
        Parameters
        ----------
        board : Numpy array
            The board state to normalize

        Returns
        -------
        board : Numpy array
            The copy of board state after normalization
        """
        # return board.copy()
        # return((board/128.0 - 1).copy())
        return board.astype(np.float32)/4.0

    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value
        
        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Numpy array
            Selected action using the argmax function
        """
        board = self._prepare_input(board)
        
        with torch.no_grad():
            model_outputs = self._model(board).cpu().numpy()
        
        # Mask illegal moves with -inf and select the action with the maximum value
        masked_outputs = np.where(legal_moves==1, model_outputs, -np.inf)
        return np.argmax(masked_outputs, axis=1)

    def _agent_model(self):
        """Returns the model which evaluates Q values for a given state input

        Returns
        -------
        model : TensorFlow Graph
            DQN model graph
        """
        return DQNModel(self._board_size, self._n_frames, self._n_actions, self._version)


    def set_weights_trainable(self):
        """Set selected layers to non trainable and compile the model"""
        for param in self._model.parameters():
            param.requires_grad = False
        # the last dense layers should be trainable
        for name, param in self._model.named_parameters():
            if 'action_prev_dense' in name or 'action_values' in name:
                param.requires_grad = True



    def get_action_proba(self, board, values=None):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        values : None, optional
            Kept for consistency with other agent classes
        
        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        # Prepare the input
        board = self._prepare_input(board)
        
        # Forward pass through the model
        with torch.no_grad():
            model_outputs = self._model(board).cpu()
            
        # Subtracting max for numerical stability, then taking softmax
        model_outputs = torch.softmax(model_outputs, dim=1).cpu().numpy()
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        """Save the current models to disk using tensorflow's
        inbuilt save model function (saves in h5 format)
        saving weights instead of model as cannot load compiled
        model with any kind of custom object (loss or metric)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        iteration = 0 if iteration is None else iteration
        torch.save(self._model.state_dict(), f"{file_path}/model_{iteration:04d}.pt")
        if(self._use_target_net):
            torch.save(self._target_net.state_dict(), f"{file_path}/model_{iteration:04d}_target.pt")

    def load_model(self, file_path='', iteration=None):
        """ load any existing models, if available """
        """Load models from disk using tensorflow's
        inbuilt load model function (model saved in h5 format)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """
        iteration = 0 if iteration is None else iteration
        self._model.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}.pt", map_location=self.device))
        if(self._use_target_net):
            self._target_net.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}_target.pt", map_location=self.device))

    def print_models(self):
        """Print the current models using summary method"""
        print('Training Model')
        print(self._model)
        if(self._use_target_net):
            print('Target Network')
            print(self._target_net)

    def train_agent(self, batch_size=32, reward_clip=False):
        """Train the model by sampling from buffer and return the error.
        We are predicting the expected future discounted reward for all
        actions with our model. The target for training the model is calculated
        in two parts:
        1) dicounted reward = current reward + 
                        (max possible reward in next state) * gamma
           the next reward component is calculated using the predictions
           of the target network (for stability)
        2) rewards for only the action take are compared, hence while
           calculating the target, set target value for all other actions
           the same as the model predictions
        
        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer, should be small
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Whether to clip the rewards using the numpy sign command
            rewards > 0 -> 1, rewards <0 -> -1, rewards == 0 remain same
            this setting can alter the learned behaviour of the agent

        Returns
        -------
            loss : float
            The current error (error metric is defined in reset_models)
        """
        # Sample a batch of experiences
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        
        # Convert to PyTorch tensors and move to device
        s = self._prepare_input(s) # Current state
        next_s = self._prepare_input(next_s) # Next state
        a = torch.tensor(a, dtype=torch.long).to(self.device) # Actions
        r = torch.tensor(r, dtype=torch.float32).to(self.device) # Rewards
        done = torch.tensor(done, dtype=torch.float32).to(self.device) # Game over indicator
        legal_moves = torch.tensor(legal_moves, dtype=torch.bool).to(self.device) # Legal moves mask
        gamma_tensor = torch.tensor(self._gamma, dtype=torch.float32).to(self.device) # Discount factor
        
        if(reward_clip):
            r = torch.sign(r) # Clip rewards to -1, 0, 1 if enabled
        
        # Approximate Q-values for the current state
        model_outputs = self._model(s)
        
        # Approximate Q-values for the next state
        with torch.no_grad():
            next_model_outputs = self._target_net(next_s) if self._use_target_net else self._model(next_s)
        
        # Mask illegal moves with -inf and select the action with the maximum value
        max_model_output_index = torch.argmax(torch.where(legal_moves, model_outputs, float("-inf")), dim=1)
        max_model_output = next_model_outputs[torch.arange(next_model_outputs.shape[0]), max_model_output_index]
        
        # Compute target Q-values (expected Q-values) using the Bellman equation
        expected_value = r.squeeze() + gamma_tensor * max_model_output * (1 - done.squeeze())
        
        # Compute Q-values for the actions taken
        output_q_value = model_outputs[torch.arange(model_outputs.size(0)), torch.argmax(a, dim=1)]
        
        # Compute loss
        loss = self.criterion(output_q_value, expected_value)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def compare_weights(self):
        """Simple utility function to check if the model and target 
        network have the same weights or not
        """
        for i in range(len(self._model.layers)):
            for j in range(len(self._model.layers[i].weights)):
                c = (self._model.layers[i].weights[j].numpy() == \
                     self._target_net.layers[i].weights[j].numpy()).all()
                print('Layer {:d} Weights {:d} Match : {:d}'.format(i, j, int(c)))

    def copy_weights_from_agent(self, agent_for_copy):
        """Update weights between competing agents which can be used
        in parallel training
        """
        assert isinstance(agent_for_copy, self), "Agent type is required for copy"

        self._model.set_weights(agent_for_copy._model.get_weights())
        self._target_net.set_weights(agent_for_copy._model_pred.get_weights())
        