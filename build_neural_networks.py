"""
These two Classes build the Actor network and Critic network.
The actor network receives the state and calculates the action.
The critic network receives the state and action and calculates a q-distribution.

The q-distribution represents the probability that the value of this state-action pair
is in a certain bin. Each of the outputs corresponds to a probability that the true
value lies in a given bin. This strategy yields better results than simply estimating
the value of the state-action pair, as we have a full distribution to work with rather
than just the mean.

@author: Kirk Hovell (khovell@gmail.com)
"""

import tensorflow as tf

from settings import Settings


class BuildActorNetwork:
    
    def __init__(self, state, scope):
        """ 
        The actor receives the state and outputs the action 
        """

        self.state = state
        self.scope = scope
        
        # Making sure all variables generated here are under the name "scope"
        with tf.variable_scope(self.scope):
            
            # The first layer is the state (input)
            self.layer = self.state
            
            # If learning from pixels include convolutional layers
            if Settings.LEARN_FROM_PIXELS:
                
                # Build convolutional layers
                for i, conv_layer_settings in enumerate(Settings.CONVOLUTIONAL_LAYERS):
                    self.layer = tf.layers.conv2d(inputs = self.layer,
                                             activation = tf.nn.relu,
                                             name = 'conv_layer' + str(i),
                                             **conv_layer_settings) 
                    # ** means that named arguments are being passed to the function.
                    # The conv2d function is able to accept the keywords.
                
                # Flattening image into a column for subsequent fully-connected layers
                self.layer = tf.layers.flatten(self.layer) 
            
            
            # Building fully-connected layers
            for i, number_of_neurons in enumerate(Settings.ACTOR_HIDDEN_LAYERS):
                self.layer = tf.layers.dense(inputs = self.layer,
                                        units = number_of_neurons,
                                        activation = tf.nn.relu,
                                        name = 'fully_connected_layer_' + str(i))
            
            # Convolutional layers (optional) have been applied, followed by fully-connected hidden layers
            # The final layer goes from the output of the last fully-connected layer
            # to the action size. It is squished with a tanh and then scaled to the action range.
            # Tanh forces output between -1 and 1, which I need to scale to the action range
            self.actions_out_unscaled = tf.layers.dense(inputs = self.layer,
                                                   units = Settings.ACTION_SIZE,
                                                   activation = tf.nn.tanh,
                                                   name = 'output_layer') 

            # Scaling actions to the correct range
            self.action_scaled = tf.multiply(0.5, tf.multiply(self.actions_out_unscaled, Settings.ACTION_RANGE) + Settings.LOWER_ACTION_BOUND + Settings.UPPER_ACTION_BOUND) # for tanh
            
            # Grab all the parameters from this neural network
            self.parameters = tf.trainable_variables(scope = self.scope)
    
    def generate_training_function(self, dQ_dAction):
        # Develop the operation that trains the actor one step.
        with tf.variable_scope(self.scope):
            with tf.variable_scope('Training'):
                # Choosing an AdamOptimizer to perform stochastic gradient descent
                self.optimizer = tf.train.AdamOptimizer(Settings.ACTOR_LEARNING_RATE)
                # Calculating the gradients for each parameter. This uses the 
                # dQ_dAction (action gradients) that are received from the critic.
                # The actor gradients are the derivative of the reward with respect
                # to each actor parameter. Negative dQ_dAction is used to perform
                # gradient ascent instead of gradient descent.
                self.actor_gradients = tf.gradients(self.action_scaled, self.parameters, -dQ_dAction)  
                
                # The actor gradients are summed over the batch, so we must divide by 
                # the batch size to get the mean gradients.
                self.actor_gradients_scaled = list(map(lambda x: tf.divide(x, Settings.MINI_BATCH_SIZE), self.actor_gradients)) # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients
                
                # Apply the gradients to each parameter!
                actor_training_function = self.optimizer.apply_gradients(zip(self.actor_gradients_scaled, self.parameters))
                 
                return actor_training_function
                  
            
class BuildQNetwork:
    
    def __init__(self, state, action, scope):
        
        self.state = state
        self.action = action
        self.scope = scope
        """
        Defines a critic network that predicts the q-distribution (expected return)
        from a given state and action. 
        
        The network archetectire is modified from the D4PG paper. The state goes through
        two layers on its own before being added to the action who has went through
        one layer. Then, the sum of the two goes through the final layer. Note: the 
        addition happend before the relu.
        """
        with tf.variable_scope(self.scope):
            # Two sides flow through the network independently.
            self.state_side  = self.state
            self.action_side = self.action
            
            ######################
            ##### State Side #####
            ######################
            # If learning from pixels (a state-only feature), use convolutional layers
            if Settings.LEARN_FROM_PIXELS:            
                # Build convolutional layers
                for i, conv_layer_settings in enumerate(Settings.CONVOLUTIONAL_LAYERS):
                    self.state_side = tf.layers.conv2d(inputs = self.state_side,
                                                       activation = tf.nn.relu,
                                                       name = 'state_conv_layer' + str(i),
                                                       **conv_layer_settings) # the "**" allows the passing of keyworded arguments
                
                # Flattening image into a column for subsequent layers 
                self.state_side = tf.layers.flatten(self.state_side) 
                    
            # Fully connected layers on state side from the second layer onwards Settings.CRITIC_HIDDEN_LAYERS[1:]
            for i, number_of_neurons in enumerate(Settings.CRITIC_HIDDEN_LAYERS):
                self.state_side = tf.layers.dense(inputs = self.state_side,
                                                  units = number_of_neurons,
                                                  activation = None,
                                                  name = 'state_fully_connected_layer_' + str(i))
                # Perform a relu unless this is the layer that is being added to the action side
                if i < (len(Settings.CRITIC_HIDDEN_LAYERS) - 1):
                    self.state_side = tf.nn.relu(self.state_side)
            
            #######################
            ##### Action Side #####
            #######################
            # Fully connected layers on action side
            for i, number_of_neurons in enumerate(Settings.CRITIC_HIDDEN_LAYERS[1:]):
                self.action_side = tf.layers.dense(inputs = self.action_side,
                                                   units = number_of_neurons,
                                                   activation = None,
                                                   name = 'action_fully_connected_layer_' + str(i))
                # Perform a relu unless this is the layer that is being added to the action side
                if i < (len(Settings.CRITIC_HIDDEN_LAYERS) - 2):
                    self.action_side = tf.nn.relu(self.action_side)
            
            ################################################
            ##### Combining State Side and Action Side #####
            ################################################            
            self.layer = tf.add(self.state_side, self.action_side)
            self.layer = tf.nn.relu(self.layer)
            
            #################################################
            ##### Final Layer to get Value Distribution #####
            #################################################
            # Calculating the final layer logits as an intermediate step,
            # since the cross_entropy loss function needs logits.
            self.q_distribution_logits = tf.layers.dense(inputs = self.layer,
                                                         units = Settings.NUMBER_OF_BINS,
                                                         activation = None,
                                                         name = 'output_layer')
            
            # Calculating the softmax of the last layer to convert logits to a probability distribution.
            # Softmax ensures that all outputs add up to 1, relative to their weights
            self.q_distribution = tf.nn.softmax(self.q_distribution_logits, name = 'output_probabilities') 
            
            # The value bins that each probability corresponds to.
            self.bins = tf.lin_space(Settings.MIN_V, Settings.MAX_V, Settings.NUMBER_OF_BINS)
            
            # Getting the parameters from the critic
            self.parameters = tf.trainable_variables(scope=self.scope)
            
            # Calculating the derivative of the q-distribution with respect to the action input.
            # It is weighted by the bins to give the derivative of the expected value with respect
            # to the input actions. This is used in the actor training.
            self.dQ_dAction = tf.gradients(self.q_distribution, self.action, self.bins) # also known as action gradients
            
            
    def generate_training_function(self, target_q_distribution, target_bins, importance_sampling_weights):
        # Create the operation that trains the critic one step.        
        with tf.variable_scope(self.scope):
            with tf.variable_scope('Training'):
                
                # Choosing the Adam optimizer to perform stochastic gradient descent
                self.optimizer = tf.train.AdamOptimizer(Settings.CRITIC_LEARNING_RATE)               
                
                # Project the target distribution onto the bounds of the original network
                projected_target_distribution = l2_project(target_bins, target_q_distribution, self.bins)  
                
                # Calculate the cross entropy loss between the projected distribution and the main q_network!
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.q_distribution_logits, labels = tf.stop_gradient(projected_target_distribution))
                
                # A loss correction is needed if we use a prioritized replay buffer
                # to account for the bias introduced by the prioritized sampling.
                if Settings.PRIORITY_REPLAY_BUFFER:
                    # Correct prioritized loss bias using importance sampling
                    self.weighted_loss = self.loss * importance_sampling_weights
                else:
                    self.weighted_loss = self.loss

                # Taking the average across the batch
                self.mean_loss = tf.reduce_mean(self.weighted_loss)
                
                # Optionally perform L2 regularization, where the network is 
                # penalized for having large parameters
                if Settings.L2_REGULARIZATION:
                    self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.parameters if 'kernel' in v.name]) * Settings.L2_REG_PARAMETER
                else:
                    self.l2_reg_loss = 0.0
                    
                # Add up the final loss function
                self.total_loss = self.mean_loss + self.l2_reg_loss
                 
                # Set the optimizer to minimize the total loss, and do so by modifying the critic parameter.
                critic_training_function = self.optimizer.minimize(self.total_loss, var_list=self.parameters)
                  
                return critic_training_function, projected_target_distribution


# Projection function used by the critic training function
'''
## l2_projection ##
# Taken from: https://github.com/deepmind/trfl/blob/master/trfl/dist_value_ops.py
# Projects the target distribution onto the support of the original network [Vmin, Vmax]
'''

def l2_project(z_p, p, z_q):
    """Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.
    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).
    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.
    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.
    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    # Broadcasting of tensors is used extensively in the code below. To avoid
    # accidental broadcasting along unintended dimensions, tensors are defensively
    # reshaped to have equal number of dimensions (3) throughout and intended
    # shapes are indicated alongside tensor definitions. To reduce verbosity,
    # extra dimensions of size 1 are inserted by indexing with `None` instead of
    # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
    # `[k, l]' to one of shape `[k, 1, l]`).
    
    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[0], z_q[-1]
    d_pos = tf.concat([z_q, vmin[None]], 0)[1:]  # 1 x Kq x 1
    d_neg = tf.concat([vmax[None], z_q], 0)[:-1]  # 1 x Kq x 1
    # Clip z_p to be in new support range (vmin, vmax).
    z_p = tf.clip_by_value(z_p, vmin, vmax)[:, None, :]  # B x 1 x Kp
    
    # Get the distance between atom values in support.
    d_pos = (d_pos - z_q)[None, :, None]  # z_q[i+1] - z_q[i]. 1 x B x 1
    d_neg = (z_q - d_neg)[None, :, None]  # z_q[i] - z_q[i-1]. 1 x B x 1
    z_q = z_q[None, :, None]  # 1 x Kq x 1
    
    # Ensure that we do not divide by zero, in case of atoms of identical value.
    d_neg = tf.where(d_neg > 0, 1./d_neg, tf.zeros_like(d_neg))  # 1 x Kq x 1
    d_pos = tf.where(d_pos > 0, 1./d_pos, tf.zeros_like(d_pos))  # 1 x Kq x 1
    
    delta_qp = z_p - z_q   # clip(z_p)[j] - z_q[i]. B x Kq x Kp
    d_sign = tf.cast(delta_qp >= 0., dtype=p.dtype)  # B x Kq x Kp
    
    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
    # Shape  B x Kq x Kp.
    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]  # B x 1 x Kp.
    return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * p, 2)