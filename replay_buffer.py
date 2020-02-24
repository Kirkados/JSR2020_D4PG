"""
Generates and manages the large experience replay buffer.

The experience replay buffer holds all the data that is dumped into it from the 
many agents who are running episodes of their own. The learner then trains off 
this heap of data continually and in its own thread.

@author: Kirk Hovell (khovell@gmail.com)
"""

import random
import numpy as np

from collections import deque

from settings import Settings

class ReplayBuffer():
    # Generates and manages a non-prioritized replay buffer
    
    def __init__(self):
        # Generate the buffer
        self.buffer = deque(maxlen = Settings.REPLAY_BUFFER_SIZE)

    # Query how many entries are in the buffer
    def how_filled(self):
        return len(self.buffer)
    
    # Add new experience to the buffer
    def add(self, experience):
        self.buffer.append(experience)
        
    # Randomly sample data from the buffer
    def sample(self):
        # Decide how much data to sample
        # (maybe the buffer doesn't contain enough samples yet to fill a MINI_BATCH)
        batch_size = min(Settings.MINI_BATCH_SIZE, len(self.buffer)) 
        # Sample the data
        sampled_batch = np.asarray(random.sample(self.buffer, batch_size))

        # Unpack the training data
        states_batch           = np.stack(sampled_batch[:, 0])
        actions_batch          = np.stack(sampled_batch[:, 1])
        rewards_batch          = sampled_batch[:, 2]
        next_states_batch      = np.stack(sampled_batch[:, 3])
        dones_batch            = np.stack(sampled_batch[:,4])
        gammas_batch           = np.reshape(sampled_batch[:, 5], [-1, 1])

        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, gammas_batch
        