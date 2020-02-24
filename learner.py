"""
This Class builds the Learner which consititutes the Critic, the Agent, and their
target networks. Additionally, it samples data from the replay_buffer and trains
both the Critic and Agent neural networks.

When a Learner instance is created, all the appropriate networks and training
operations are built. Then, simply run Learner.run() to initiate the continuous
training process.

Adapted from msinto93, and SuRELI's github code. Many thanks!

Training:
    The critic is trained using supervised learning to minimize the
    cross-entropy loss between the Q value and the target value
    y = r_t + gamma * Q_target(next_state, Action(next_state))

    To train the actor, we apply the policy gradient
    Grad = grad(Q(s,a), A)) * grad(A, params)

@author: Kirk Hovell (khovell@gmail.com)
"""

import tensorflow as tf
import numpy as np
import time
import multiprocessing
import queue # for empty error catching

from build_neural_networks import BuildActorNetwork, BuildQNetwork
from settings import Settings

class Learner:
    def __init__(self, sess, saver, replay_buffer, writer):
        print("Initialising learner...")

        # Saving items to the self. object for future use
        self.sess = sess
        self.saver = saver
        self.replay_buffer = replay_buffer
        self.writer = writer

        with tf.variable_scope("Preparing_placeholders"):
            # Defining placeholders for training
#            self.state_placeholder                       = tf.placeholder(dtype = tf.float32, shape = [Settings.MINI_BATCH_SIZE, Settings.STATE_SIZE], name = "state_placeholder") # the '*' unpacks the STATE_SIZE list (incase it's pixels of higher dimension)
#            self.action_placeholder                      = tf.placeholder(dtype = tf.float32, shape = [Settings.MINI_BATCH_SIZE, Settings.ACTION_SIZE], name = "action_placeholder") # placeholder for actions
#            self.target_bins_placeholder                 = tf.placeholder(dtype = tf.float32, shape = [Settings.MINI_BATCH_SIZE, Settings.NUMBER_OF_BINS], name = "target_bins_placeholder") # Bin values of target network with Bellman update applied
#            self.target_q_distribution_placeholder       = tf.placeholder(dtype = tf.float32, shape = [Settings.MINI_BATCH_SIZE, Settings.NUMBER_OF_BINS], name = "target_q_distribution_placeholder")  # Future q-distribution from target critic
#            self.dQ_dAction_placeholder                  = tf.placeholder(dtype = tf.float32, shape = [Settings.MINI_BATCH_SIZE, Settings.ACTION_SIZE], name = "dQ_dAction_placeholder") # Gradient of critic predicted value with respect to input actions
#            self.importance_sampling_weights_placeholder = tf.placeholder(dtype = tf.float32, shape =  Settings.MINI_BATCH_SIZE, name = "importance_sampling_weights_placeholder") # [PRIORITY_REPLAY_BUFFER only] Holds the weights that are used to remove bias from priority sampling

            self.state_placeholder                       = tf.placeholder(dtype = tf.float32, shape = [None, Settings.STATE_SIZE], name = "state_placeholder") # the '*' unpacks the STATE_SIZE list (incase it's pixels of higher dimension)
            self.action_placeholder                      = tf.placeholder(dtype = tf.float32, shape = [None, Settings.ACTION_SIZE], name = "action_placeholder") # placeholder for actions
            self.target_bins_placeholder                 = tf.placeholder(dtype = tf.float32, shape = [None, Settings.NUMBER_OF_BINS], name = "target_bins_placeholder") # Bin values of target network with Bellman update applied
            self.target_q_distribution_placeholder       = tf.placeholder(dtype = tf.float32, shape = [None, Settings.NUMBER_OF_BINS], name = "target_q_distribution_placeholder")  # Future q-distribution from target critic
            self.dQ_dAction_placeholder                  = tf.placeholder(dtype = tf.float32, shape = [Settings.MINI_BATCH_SIZE, Settings.ACTION_SIZE], name = "dQ_dAction_placeholder") # Gradient of critic predicted value with respect to input actions
            self.importance_sampling_weights_placeholder = tf.placeholder(dtype = tf.float32, shape =  None, name = "importance_sampling_weights_placeholder") # [PRIORITY_REPLAY_BUFFER only] Holds the weights that are used to remove bias from priority sampling

        # The reward options that the distributional critic predicts the liklihood of being in
        self.bins = np.linspace(Settings.MIN_V, Settings.MAX_V, Settings.NUMBER_OF_BINS, dtype = np.float32)

        ######################################################
        ##### Build the networks and training operations #####
        ######################################################
        self.build_main_networks()
        self.build_target_networks()

        # Build the operation to update the target network parameters
        self.build_target_parameter_update_operations()

        # Create operstions for Tensorboard logging
        self.writer = writer
        self.create_summary_functions()

        print("Learner created!")


    def create_summary_functions(self):

        # Creates the operation that, when run, will log the appropriate data to tensorboard
        with tf.variable_scope("Logging_Learning"):
            # The critic loss during training is the only logged item
            self.iteration_loss_placeholder = tf.placeholder(tf.float32)
            self.iteration_loss_summary = tf.summary.scalar("Loss", self.iteration_loss_placeholder)
            self.iteration_summary = tf.summary.merge([self.iteration_loss_summary])


    def build_main_networks(self):
        ##################################
        #### Build the learned critic ####
        ##################################
        self.critic = BuildQNetwork(self.state_placeholder, self.action_placeholder, scope='learner_critic_main')

        # Build the critic training function
        self.train_critic_one_step, self.projected_target_distribution = self.critic.generate_training_function(self.target_q_distribution_placeholder, self.target_bins_placeholder, self.importance_sampling_weights_placeholder)

        #################################
        #### Build the learned actor ####
        #################################
        self.actor = BuildActorNetwork(self.state_placeholder, scope='learner_actor_main')

        # Build the actor training function
        self.train_actor_one_step = self.actor.generate_training_function(self.dQ_dAction_placeholder)


    def build_target_networks(self):
        ###########################################
        #### Build the target actor and critic ####
        ###########################################
        self.target_critic = BuildQNetwork(self.state_placeholder, self.action_placeholder, scope='learner_critic_target')
        self.target_actor  = BuildActorNetwork(self.state_placeholder, scope='learner_actor_target')


    def build_target_parameter_update_operations(self):
        # Build operations that either
            # 1) initialize target networks to be identical to main networks 2) slowly
            # 2) slowly copy main network parameters to target networks according to Settings.TARGET_NETWORK_TAU
        main_parameters = self.actor.parameters + self.critic.parameters
        target_parameters = self.target_actor.parameters + self.target_critic.parameters

        # Build operation that fully copies the main network parameters to the targets [Option 1 above]
        initialize_target_network_parameters = []
        # Looping across all variables in the main critic and main actor
        for source_variable, destination_variable in zip(main_parameters, target_parameters):
            initialize_target_network_parameters.append(destination_variable.assign(source_variable))

        # Build operation that slowly updates target networks according to Settings.TARGET_NETWORK_TAU [Option 2 above]
        update_target_network_parameters = []
        # Looping across all variables in the main critic and main actor
        for source_variable, destination_variable in zip(main_parameters, target_parameters):
            # target = tau*main + (1 - tau)*target
            update_target_network_parameters.append(destination_variable.assign((tf.multiply(source_variable, Settings.TARGET_NETWORK_TAU) + tf.multiply(destination_variable, 1. - Settings.TARGET_NETWORK_TAU))))

        # Save both operations to self object for later use
        self.initialize_target_network_parameters = initialize_target_network_parameters
        self.update_target_network_parameters = update_target_network_parameters

    def generate_queue(self):
        # Generate the queues responsible for communicating with the learner
        self.agent_to_learner = multiprocessing.Queue(maxsize = 1)
        self.learner_to_agent = multiprocessing.Queue(maxsize = 1)

        return self.agent_to_learner, self.learner_to_agent

    def run(self, stop_run_flag, replay_buffer_dump_flag, starting_training_iteration):
        # Continuously train the actor and the critic, by applying stochastic gradient
        # descent to batches of data sampled from the replay buffer

        # Initializing the counter of training iterations
        self.total_training_iterations = starting_training_iteration

        # Starting time
        start_time = time.time()

        # Initialize the target networks to be identical to the main networks
        self.sess.run(self.initialize_target_network_parameters)

        # Setup priority replay buffer parameters, if used
        if Settings.PRIORITY_REPLAY_BUFFER:
            # When priority_beta = 1, the priority sampling bias is fully accounted for.
            # We slowly anneal priority_beta towards 1.0 over the course of training.
            # Lower beta allows the prioritized samples to be weighted unfairly,
            # but this can help training, at least initially.
            priority_beta = Settings.PRIORITY_BETA_START # starting beta value
            beta_increment = (Settings.PRIORITY_BETA_END - Settings.PRIORITY_BETA_START) / Settings.MAX_TRAINING_ITERATIONS # to increment on each iteration
        else:
            # If we aren't using a priority buffer, set the importance sampled weights to ones for the entire run
            weights_batch = np.ones(shape = Settings.MINI_BATCH_SIZE)


        ###############################
        ##### Start Training Loop #####
        ###############################
        while self.total_training_iterations < Settings.MAX_TRAINING_ITERATIONS and not stop_run_flag.is_set():

            # Check if the agent wants some q-distributions calculated
            try:
                state_log, action_log, next_state_log, reward_log, done_log, gamma_log = self.agent_to_learner.get(False)

                # Reshapping
                gamma_log = np.reshape(gamma_log,  [-1, 1])

                # Get the online q-distribution
                critic_distribution = self.sess.run(self.critic.q_distribution, feed_dict = {self.state_placeholder: state_log, self.action_placeholder: action_log}) # [episode length, number of bins]

                # Clean next actions from the target actor
                clean_next_actions = self.sess.run(self.target_actor.action_scaled, {self.state_placeholder:next_state_log}) # [episode length, num_actions]

                # Get the target q-distribution
                target_critic_distribution = self.sess.run(self.target_critic.q_distribution, feed_dict = {self.state_placeholder:state_log, self.action_placeholder:clean_next_actions}) # [episode length, number of bins]

                # Create batch of bins [see further description below]
                target_bins = np.repeat(np.expand_dims(self.bins, axis = 0), len(reward_log), axis = 0) # [episode length, number_of_bins]
                target_bins[done_log, :] = 0.0
                target_bins = np.expand_dims(reward_log, axis = 1) + (target_bins*gamma_log)

                # Calculating the bellman distribution (r + gamma*target_q_distribution). The critic loss is with respect to this projection.
                projected_target_distribution = self.sess.run(self.projected_target_distribution, feed_dict = {self.target_q_distribution_placeholder: target_critic_distribution, self.target_bins_placeholder: target_bins})

                # Calculating the loss at each timestep
                weights_batch = weights_batch = np.ones(shape = len(reward_log))
                loss_log = self.sess.run(self.critic.loss, feed_dict = {self.state_placeholder:state_log, self.action_placeholder:action_log, self.target_q_distribution_placeholder:target_critic_distribution, self.target_bins_placeholder:target_bins, self.importance_sampling_weights_placeholder:weights_batch})

                # Send the results back to the agent
                self.learner_to_agent.put((critic_distribution, target_critic_distribution, projected_target_distribution, loss_log))

            except queue.Empty:
                # If queue was empty, do nothing
                pass

            # If we don't have enough data yet to train OR we want to wait before we start to train
            if (self.replay_buffer.how_filled() < Settings.MINI_BATCH_SIZE) or (self.replay_buffer.how_filled() < Settings.REPLAY_BUFFER_START_TRAINING_FULLNESS):
                continue # Skip this training iteration. Wait for more training data.

            # Sample a mini-batch of data from the replay_buffer
            if Settings.PRIORITY_REPLAY_BUFFER:
                sampled_batch = self.replay_buffer.sample(priority_beta)
                weights_batch = sampled_batch[6] # [priority-only data] used for removing bias in prioritized data
                index_batch   = sampled_batch[7] # [priority-only data] used for updating priorities
            else:
                sampled_batch = self.replay_buffer.sample()

            # Unpack the training data
            states_batch           = sampled_batch[0]
            actions_batch          = sampled_batch[1]
            rewards_batch          = sampled_batch[2]
            next_states_batch      = sampled_batch[3]
            dones_batch            = sampled_batch[4]
            gammas_batch           = sampled_batch[5]

            ###################################
            ##### Prepare Critic Training #####
            ###################################
            # Get clean next actions by feeding the next states through the target actor
            clean_next_actions = self.sess.run(self.target_actor.action_scaled, {self.state_placeholder:next_states_batch}) # [batch_size, num_actions]

            # Get the next q-distribution by passing the next states and clean next actions through the target critic
            target_critic_distribution = self.sess.run(self.target_critic.q_distribution, {self.state_placeholder:next_states_batch, self.action_placeholder:clean_next_actions}) # [batch_size, number_of_bins]

            # Create batch of bins
            target_bins = np.repeat(np.expand_dims(self.bins, axis = 0), Settings.MINI_BATCH_SIZE, axis = 0) # [batch_size, number_of_bins]

            # If this data in the batch corresponds to the end of an episode (dones_batch[i] = True),
            # set all the bins to 0.0. This will eliminate the inclusion of the predicted future
            # reward when computing the bellman update (i.e., the predicted future rewards are only
            # the current reward, since we aren't continuing the episode any further).
            target_bins[dones_batch, :] = 0.0

            # Bellman projection. reward + gamma^N*bin -> The new
            # expected reward, according to the recently-received reward.
            # If the new reward is outside of the current bin, then we will
            # adjust the probability that is assigned to the bin.
            target_bins = np.expand_dims(rewards_batch, axis = 1) + (target_bins*gammas_batch)

            #####################################
            ##### TRAIN THE CRITIC ONE STEP #####
            #####################################
            critic_loss, _ = self.sess.run([self.critic.loss, self.train_critic_one_step], {self.state_placeholder:states_batch, self.action_placeholder:actions_batch, self.target_q_distribution_placeholder:target_critic_distribution, self.target_bins_placeholder:target_bins, self.importance_sampling_weights_placeholder:weights_batch})


            ##################################
            ##### Prepare Actor Training #####
            ##################################
            # Get clean actions that the main actor would have taken for this batch of states if there were no noise added
            clean_actions = self.sess.run(self.actor.action_scaled, {self.state_placeholder:states_batch})

            # Calculate the derivative of the main critic's q-value with respect to these actions
            dQ_dAction = self.sess.run(self.critic.dQ_dAction, {self.state_placeholder:states_batch, self.action_placeholder:clean_actions}) # also known as action gradients

            ####################################
            ##### TRAIN THE ACTOR ONE STEP #####
            ####################################
            self.sess.run(self.train_actor_one_step, {self.state_placeholder:states_batch, self.dQ_dAction_placeholder:dQ_dAction[0]})


            # If it's time to update the target networks
            if self.total_training_iterations % Settings.UPDATE_TARGET_NETWORKS_EVERY_NUM_ITERATIONS == 0:
                # Update target networks according to TAU!
                self.sess.run(self.update_target_network_parameters)

            # If we're using a priority buffer, tend to it now.
            if Settings.PRIORITY_REPLAY_BUFFER:
                # The priority replay buffer ranks the data according to how unexpected they were
                # An unexpected data point will have high loss. Now that we've just calculated the loss,
                # update the priorities in the replay buffer.
                self.replay_buffer.update_priorities(index_batch, (np.abs(critic_loss)+Settings.PRIORITY_EPSILON))

                # Increment priority beta value slightly closer towards 1.0
                priority_beta += beta_increment

                # If it's time to check if the prioritized replay buffer is overful
                if Settings.PRIORITY_REPLAY_BUFFER and (self.total_training_iterations % Settings.DUMP_PRIORITY_REPLAY_BUFFER_EVER_NUM_ITERATIONS == 0):
                    # If the buffer is overfilled
                    if (self.replay_buffer.how_filled() > Settings.REPLAY_BUFFER_SIZE):
                        # Make the agents wait before adding any more data to the buffer
                        replay_buffer_dump_flag.clear()
                        # How overful is the buffer?
                        samples_to_remove = self.replay_buffer.how_filled() - Settings.REPLAY_BUFFER_SIZE
                        # Remove the appropriate number of samples
                        self.replay_buffer.remove(samples_to_remove)
                        # Allow the agents to continue now that the buffer is ready
                        replay_buffer_dump_flag.set()

            # If it's time to log the training performance to TensorBoard
            if self.total_training_iterations % Settings.LOG_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS == 0:
                # Logging the mean critic loss across the batch
                summary = self.sess.run(self.iteration_summary, feed_dict = {self.iteration_loss_placeholder: np.mean(critic_loss)})
                self.writer.add_summary(summary, self.total_training_iterations)

            # If it's time to save a checkpoint. Be it a regular checkpoint, the final planned iteration, or the final unplanned iteration
            if (self.total_training_iterations % Settings.SAVE_CHECKPOINT_EVERY_NUM_ITERATIONS == 0) or (self.total_training_iterations == Settings.MAX_TRAINING_ITERATIONS) or stop_run_flag.is_set():
                # Save the state of all networks and note the training iteration
                self.saver.save(self.total_training_iterations, self.state_placeholder, self.actor.action_scaled)

            # If it's time to print the training performance to the screen
            if self.total_training_iterations % Settings.DISPLAY_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS == 0:
                print("Trained actor and critic %i iterations in %.2f minutes, at %.3f s/iteration. Now at iteration %i." % (Settings.DISPLAY_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS, (time.time() - start_time)/60, (time.time() - start_time)/Settings.DISPLAY_TRAINING_PERFORMANCE_EVERY_NUM_ITERATIONS, self.total_training_iterations))
                start_time = time.time() # resetting the timer for the next PERFORMANCE_UPDATE_EVERY_NUM_ITERATIONS of iterations

            # Incrementing training iteration counter
            self.total_training_iterations += 1

        # If we are done training
        print("Learner finished after running " + str(self.total_training_iterations) + " training iterations!")

        # Flip the flag signalling all agents to stop
        stop_run_flag.set()


"""
My version of the critic training is below. I find it more legible than the
above training operations, but it does not run as fast as the above implementation
by msinto93. So, I have opted to use theirs. I've verified that the same inputs
always produces the same outputs between the two implementations.
"""

# Alternate implementation of D4PG critic training
"""
with tf.variable_scope("Train_Critic"): # grouping tensorboard graph

            ##################################################
            ###### Generating Updated Bin Probabilities ######
            ##################################################

            # Initializing the matrix that will hold the new bin probabilities as they get generated
            new_bin_probabilities = tf.zeros([Settings.MINI_BATCH_SIZE, Settings.NUMBER_OF_BINS])

            # For each bin, project where that bin's probability should end up after receiving the reward
            # by calculating the new expected reward. Then, find out what bin the projection lies in.
            # Then, distribute the probability into that bin. Then, build a loss function to minimize
            # the difference between the current distribution and the calculated distribution.
            for this_bin in range(Settings.NUMBER_OF_BINS): # for each bin

                # Bellman projection. reward + gamma^N*not_done*bin -> The new
                # expected reward, according to the recently-received reward.
                # If the new reward is outside of the current bin, then we will
                # adjust the probability that is assigned to the bin.
                # If the episode terminates here, the new expected reward from
                # this state-action pair is just the reward.
                projection = self.reward_placeholder + (self.discount_factor_placeholder)*(1.0 - self.done_placeholder)*self.bins[this_bin]

                # Clipping projected reward to its bounds.
                projection = tf.squeeze(tf.clip_by_value(projection, Settings.MIN_V, Settings.MAX_V)) # Squeezing -> shape [batch_size]

                # Which bin number the projected value ends up in (so we know which bin to increase the probability of)
                new_bin = (projection - Settings.MIN_V)/self.bin_width # shape [batch_size]

                # However, it is unlikely the new bin number will lie directly
                # on an existing bin number. Therefore, determine the nearby
                # bins so we know where we should distribute the probability into.
                adjacent_bin_upper = tf.ceil(new_bin) # shape [batch_size]
                adjacent_bin_lower = tf.floor(new_bin) # shape [batch_size]

                # Checking if the upper and lower bins are the same bin (!!!).
                # This occurs when the projection lies directly on a bin.
                # Common causes are: 1) The reward is large and pushes the projection
                # to one of the bounds (where it is clipped). 2) There is a
                # reward of 0 for bin[i] = 0.
                are_bins_identical = tf.equal(adjacent_bin_upper, adjacent_bin_lower) # shape [batch_size]
                are_bins_different = tf.logical_not(are_bins_identical)               # shape [batch_size]

                # Generating two one-hot matrices that will be used to place the
                # projected next-state probabilities from the target critic
                # network into the appropriate bin. The appropriate bin is the
                # one who we would like to increase their probability.
                # Only one element in each row is a 1, all others are 0.
                one_hot_upper = tf.one_hot(tf.to_int32(adjacent_bin_upper), depth = Settings.NUMBER_OF_BINS) # shape [batch_size, #atoms]
                one_hot_lower = tf.one_hot(tf.to_int32(adjacent_bin_lower), depth = Settings.NUMBER_OF_BINS) # shape [batch_size, #atoms]

                # Disributing the next-state bin probabilities (from the target
                # q_network) into both bins dictated by the projection.
                # Accumulating the new bin probabilities as we loop through all bins.
                # Note: the "upper" part gets multiplied by the one_hot_lower because
                #       the (upper - new_bin) is essentially "how far" the new bin is from the
                #       upper bin. Therefore, the larger that number, the more we
                #       should put in the lower bin.
                # This accumulation only applies to samples in the batch that
                # have been assigned different bins (by multiplying by are_bins_different)
                new_bin_probabilities += tf.reshape(self.target_q_network[:, this_bin] * (adjacent_bin_upper - new_bin) * tf.to_float(are_bins_different), [-1, 1]) * one_hot_lower # [batch_size, 1] * [batch_size, #atoms] = [batch_size, #atoms]
                new_bin_probabilities += tf.reshape(self.target_q_network[:, this_bin] * (new_bin - adjacent_bin_lower) * tf.to_float(are_bins_different), [-1, 1]) * one_hot_upper # [batch_size, 1] * [batch_size, #atoms] = [batch_size, #atoms]

                # If, by chance, the new_bin lies directly on a bin, then the
                # adjacent_bin_upper and adjacent_bin_lower will be identical.
                # In that case, the full next-state probability is added to that
                # bin.
                new_bin_probabilities += tf.reshape(self.target_q_network[:, this_bin] * tf.to_float(are_bins_identical), [-1, 1]) * one_hot_upper # [batch_size, 1] * [batch_size, #atoms] = [batch_size, #atoms]


            ###########################################
            ##### Generating Critic Loss Function #####
            ###########################################

            # DEBUGGING
            #self.TEST_PROBS = new_bin_probabilities
            # END DEBUGGING

            # We've now got the new distribution (bin probabilities),
            # now we must generate a loss function for the critic!
            self.critic_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.q_network_logits,
                                                                            labels = tf.stop_gradient(new_bin_probabilities)) # not sure if tf.stop_gradients is needed, but it certainly doesn't hurt

            # Taking the mean loss across the batch
            self.critic_loss = tf.reduce_mean(self.critic_losses)

            # Optional L2 Regularization
            if Settings.L2_REGULARIZATION:
                # Penalize the critic for having large weights -> L2 Regularization
                self.critic_loss += l2_regularization(self.critic_parameters)


            ##############################################################################
            ##### Develop the Operation that Trains the Critic with Gradient Descent #####
            ##############################################################################
            self.critic_trainer        = tf.train.AdamOptimizer(Settings.CRITIC_LEARNING_RATE)
            self.train_critic_one_step = self.critic_trainer.minimize(self.critic_loss, var_list = self.critic_parameters) # RUN THIS TO TRAIN THE CRITIC ONE STEP
            #self.train_critic_one_step = self.critic_trainer.minimize(self.critic_loss) # RUN THIS TO TRAIN THE CRITIC ONE STEP


"""