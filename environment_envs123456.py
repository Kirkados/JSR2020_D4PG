
"""
This script sets the environmental parameters for training a deep guidance
implementation on a spacecraft robotics proximity operations task.

The details are presented in a 2020 paper under review in the Journal of
Guidance, Control, and Dynamics titled "Deep Reinforcement Learning for 
Spacecraft Proximity Operations Guidance"

The task is currently setup to have a chaser spacecraft track a rotating 
target while avoiding and obstacle.

@author: Kirk Hovell (khovell@gmail.com)
"""
import numpy as np
import os
import signal
import multiprocessing
from scipy.integrate import odeint # Numerical integrator

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

class Environment:

    def __init__(self):
        ##################################
        ##### Environment Properties #####
        ##################################
        self.TOTAL_STATE_SIZE         = 8 # [x, y, theta, desired_x_error, desired_y_error, desired_theta_error, obstable_distance_x, obstacle_distance_y]
        self.IRRELEVANT_STATES        = [0,1,2] # No obstacle: [0,1,2,6,7] ; Yes obstacle: [0,1,2]
        self.STATE_SIZE               = self.TOTAL_STATE_SIZE - len(self.IRRELEVANT_STATES) # total number of relevant states
        self.ACTION_SIZE              = 3 # [x_dot, y_dot, theta_dot]
        self.LOWER_ACTION_BOUND       = np.array([-0.1, -0.1, -10*np.pi/180]) # [m/s, m/s, rad/s] stationary=[-0.05, -0.05, -10*np.pi/180]; rotating=[-0.1, -0.1, -10*np.pi/180]
        self.UPPER_ACTION_BOUND       = np.array([ 0.1,  0.1,  10*np.pi/180]) # [m/s, m/s, rad/s] stationary=[ 0.05,  0.05,  10*np.pi/180]; rotating=[ 0.1,  0.1,  10*np.pi/180]
        self.LOWER_STATE_BOUND        = np.array([  0.,   0., -4*2*np.pi,  0.,  0., -4*2*np.pi,  0., 0. ]) # [m, m, rad, m, m, rad, m, m]
        self.UPPER_STATE_BOUND        = np.array([ 3.7,  2.4,  4*2*np.pi, 3.7, 2.4,  4*2*np.pi, 3.7, 2.4]) # [m, m, rad, m, m, rad, m, m]
        self.NORMALIZE_STATE          = True # Normalize state on each timestep to avoid vanishing gradients
        self.RANDOMIZE                = True # whether or not to RANDOMIZE the state & target location
        self.NOMINAL_INITIAL_POSITION = np.array([3.0, 1.0, 0.0])
        self.NOMINAL_TARGET_POSITION  = np.array([1.85, 1.2, 0]) # stationary=[1.85, 0.6, np.pi/2]; rotating=[1.85, 1.2, 0]
        self.MIN_V                    = -1000. # -350
        self.MAX_V                    =  100.
        self.N_STEP_RETURN            =   1
        self.DISCOUNT_FACTOR          =   0.95**(1/self.N_STEP_RETURN)
        self.TIMESTEP                 =   0.2 # [s]
        self.TARGET_REWARD            =   1. # reward per second
        self.FALL_OFF_TABLE_PENALTY   =   0.
        self.END_ON_FALL              = False # end episode on a fall off the table
        self.GOAL_REWARD              =   0.
        self.NEGATIVE_PENALTY_FACTOR  = 1.5 # How much of a factor to additionally penalize negative rewards
        self.MAX_NUMBER_OF_TIMESTEPS  = 900 # per episode -- 450 for stationary, 900 for rotating
        self.ADDITIONAL_VALUE_INFO    = False # whether or not to include additional reward and value distribution information on the animations
        self.REWARD_TYPE              = True # True = Linear; False = Exponential
        self.REWARD_WEIGHTING         = [0.5, 0.5, 0.1] # How much to weight the rewards in the state
        self.REWARD_MULTIPLIER        = 250 # how much to multiply the differential reward by
        
        # Obstacle properties
        self.USE_OBSTACLE              = True # Also change self.IRRELEVANT_STATES
        self.OBSTABLE_PENALTY          = 15 # [rewards/second] How bad is it to collide with the obstacle?
        self.OBSTABLE_DISTANCE         = 0.2 # [m] radius of which the obstacle penalty will be applied
        self.OBSTACLE_INITIAL_POSITION = np.array([1.2, 1.2]) # [m]
        self.OBSTABLE_VELOCITY         = np.array([0.0 , 0.0]) # [m/s]

        # Test time properties
        self.TEST_ON_DYNAMICS         = True # Whether or not to use full dynamics along with a PD controller at test time
        self.KINEMATIC_NOISE          = False # Whether or not to apply noise to the kinematics in order to simulate a poor controller
        self.KINEMATIC_NOISE_SD       = [0.02, 0.02, np.pi/100] # The standard deviation of the noise that is to be applied to each element in the state
        self.FORCE_NOISE_AT_TEST_TIME = False # [Default -> False] Whether or not to force kinematic noise to be present at test time

        # PD Controller Gains
        self.KP                       = 0 # PD controller gain
        self.KD                       = 2.0 # PD controller gain
        self.CONTROLLER_ERROR_WEIGHT  = [1, 1, 0.05] # How much to weight each error signal (for example, to weight the angle error less than the position error)      
        
        # Physical properties
        self.LENGTH  = 0.3  # [m] side length
        self.MASS    = 10   # [kg]
        self.INERTIA = 1/12*self.MASS*(self.LENGTH**2 + self.LENGTH**2) # 0.15 [kg m^2]
        
        # Target collision properties
        self.TARGET_COLLISION_DISTANCE = self.LENGTH # [m] how close chaser and target need to be before a penalty is applied
        self.TARGET_COLLISION_PENALTY  = 15           # [rewards/second] penalty given for colliding with target  

        # Additional properties
        self.PHASE_1_TIME             = 90 # [s] the time to automatically switch from phase 0 to phase 1--45 for stationary; 90 for rotating
        self.DOCKING_TOO_FAST_PENALTY = 0 # [rewards/s] penalty for docking too quickly
        self.MAX_DOCKING_SPEED        = [0.02, 0.02, 10]
        self.TARGET_ANGULAR_VELOCITY  = 0.0698 #[rad/s] constant target angular velocity stationary: 0 ; rotating: 0.0698
        self.PENALIZE_VELOCITY        = True # Should the velocity be penalized with severity proportional to how close it is to the desired location? Added Dec 11 2019
        self.VELOCITY_PENALTY         = [0.5, 0.5, 0.0] # [x, y, theta] stationary: [0.5, 0.5, 0.5/250] ; rotating [0.5, 0.5, 0] Amount the chaser should be penalized for having velocity near the desired location

    ###################################
    ##### Seeding the environment #####
    ###################################
    def seed(self, seed):
        np.random.seed(seed)

    ######################################
    ##### Resettings the Environment #####
    ######################################
    def reset(self, use_dynamics, test_time):
        # This method resets the state and returns it
        """ NOTES:
               - if use_dynamics = True -> use dynamics
               - if test_time = True -> do not add "controller noise" to the kinematics
        """
        # Setting the default to be kinematics
        self.dynamics_flag = False

        # Resetting phase number so we complete phase 0 before moving on to phase 1
        self.phase_number = 0

        # Logging whether it is test time for this episode
        self.test_time = test_time

        # If we are randomizing the initial consitions and state
        if self.RANDOMIZE:
            # Randomizing initial state
            self.state = self.NOMINAL_INITIAL_POSITION + np.random.randn(3)*[0.3, 0.3, np.pi/2]
            # Randomizing target state
            self.target_location = self.NOMINAL_TARGET_POSITION + np.random.randn(3)*[0.15, 0.15, np.pi/12]

        else:
            # Constant initial state
            self.state = self.NOMINAL_INITIAL_POSITION
            # Constant target location
            self.target_location = self.NOMINAL_TARGET_POSITION

        # Obstacle initial location (not randomized)
        self.obstacle_location = self.OBSTACLE_INITIAL_POSITION
        
        # Docking port location
        self.docking_port = self.target_location + np.array([np.cos(self.target_location[2])*(self.LENGTH+0.2), np.sin(self.target_location[2])*(self.LENGTH+0.2), -np.pi])

        # Hold point location
        self.hold_point = self.docking_port + np.array([np.cos(self.target_location[2])*(self.LENGTH*2 - 0.1), np.sin(self.target_location[2])*(self.LENGTH*2 - 0.1), 0])

        # How long is the position portion of the state
        self.POSITION_STATE_LENGTH = len(self.state)

        if use_dynamics:
            # Setting the dynamics state to be equal, initially, to the kinematics state, plus the velocity initial conditions state
            velocity_initial_conditions = np.array([0., 0., 0.])
            self.state = np.concatenate((self.state, velocity_initial_conditions))
            """ Note: dynamics_state = [x, y, theta, xdot, ydot, thetadot] """
            self.dynamics_flag = True # for this episode, dynamics will be used

        # Resetting the time
        self.time = 0.

        # Resetting the differential reward
        self.previous_position_reward = [None, None, None]


    #####################################
    ##### Step the Dynamics forward #####
    #####################################
    def step(self, action):

        # Integrating forward one time step.
        # Returns initial condition on first row then next TIMESTEP on the next row
        #########################################
        ##### PROPAGATE KINEMATICS/DYNAMICS #####
        #########################################
        if self.dynamics_flag:
            # Additional parameters to be passed to the kinematics
            kinematics_parameters = [action]

            ############################
            #### PROPAGATE DYNAMICS ####
            ############################
            # First calculate the next guidance command
            guidance_propagation = odeint(kinematics_equations_of_motion, self.state[:self.POSITION_STATE_LENGTH], [self.time, self.time + self.TIMESTEP], args = (kinematics_parameters,), full_output = 0)

            # Saving the new guidance signal
            guidance_position = guidance_propagation[1,:]

            # Next, calculate the control effort
            control_effort = self.controller(guidance_position, action) # Passing the desired position and velocity (Note: the action is the desired velocity)

            # Anything additional that needs to be sent to the dynamics integrator
            dynamics_parameters = [control_effort, self.MASS, self.INERTIA]

            # Finally, propagate the dynamics forward one timestep
            next_states = odeint(dynamics_equations_of_motion, self.state, [self.time, self.time + self.TIMESTEP], args = (dynamics_parameters,), full_output = 0)

            # Saving the new state
            self.state = next_states[1,:]

        else:

            # Additional parameters to be passed to the kinematics
            kinematics_parameters = [action]

            # Dummy guidance position
            guidance_position = []

            ###############################
            #### PROPAGATE KINEMATICS #####
            ###############################
            next_states = odeint(kinematics_equations_of_motion, self.state, [self.time, self.time + self.TIMESTEP], args = (kinematics_parameters,), full_output = 0)

            # Saving the new state
            self.state = next_states[1,:]

            # Optionally, add noise to the kinematics to simulate "controller noise"
            if self.KINEMATIC_NOISE and (not self.test_time or self.FORCE_NOISE_AT_TEST_TIME):
                 # Add some noise to the position part of the state
                 self.state += np.random.randn(self.POSITION_STATE_LENGTH) * self.KINEMATIC_NOISE_SD

        # Done the differences between the kinematics and dynamics
        # Increment the timestep
        self.time += self.TIMESTEP

        # Calculating the reward for this state-action pair
        reward = self.reward_function(action)

        # Check if this episode is done
        done = self.is_done()

        # Check if Phase 1 was completed
        self.check_phase_number()
        
        # Step obstacle's position ahead one timestep
        self.obstacle_location += self.OBSTABLE_VELOCITY*self.TIMESTEP

        # Step target's attitude ahead one timestep
        self.target_location[2] += self.TARGET_ANGULAR_VELOCITY*self.TIMESTEP

        # Update the docking port target state
        self.docking_port = self.target_location + np.array([np.cos(self.target_location[2])*(self.LENGTH+0.2), np.sin(self.target_location[2])*(self.LENGTH+0.2), -np.pi])

        # Update the hold point location
        self.hold_point = self.docking_port + np.array([np.cos(self.target_location[2])*(self.LENGTH*2 - 0.1), np.sin(self.target_location[2])*(self.LENGTH*2 - 0.1), 0])


        # Return the (state, reward, done)
        return self.state, reward, done, guidance_position

    def check_phase_number(self):
        # If the time is past PHASE_1_TIME seconds, automatically enter phase 2
        if self.time >= self.PHASE_1_TIME and self.phase_number == 0:
            self.phase_number = 1
            self.previous_position_reward = [None, None, None] # Reset the reward function to avoid a major spike
            #print("Entering phase %i at time %f" %(self.phase_number, self.time))


    def controller(self, guidance_position, guidance_velocity):
        # This function calculates the control effort based on the state and the
        # desired position (guidance_command)

        position_error = guidance_position - self.state[:self.POSITION_STATE_LENGTH]
        velocity_error = guidance_velocity - self.state[self.POSITION_STATE_LENGTH:]

        # Using a PD controller on all states independently
        control_effort = self.KP * position_error*self.CONTROLLER_ERROR_WEIGHT + self.KD * velocity_error*self.CONTROLLER_ERROR_WEIGHT

        return control_effort

    def pose_error(self):
        """
        This method returns the pose error of the current state.
        Instead of returning [state, desired_state] as the state, I'll return
        [state, error]. The error will be more helpful to the policy I believe.
        """
        if self.phase_number == 0:
            return self.hold_point - self.state[:self.POSITION_STATE_LENGTH]
        elif self.phase_number == 1:
            return self.docking_port - self.state[:self.POSITION_STATE_LENGTH]


    def reward_function(self, action):
        # Returns the reward for this TIMESTEP as a function of the state and action

        # Sets the current location that we are trying to move to
        if self.phase_number == 0:
            desired_location = self.hold_point
        elif self.phase_number == 1:
            desired_location = self.docking_port


        current_position_reward = np.zeros(1)

        # Calculates a reward map
        if self.REWARD_TYPE:
            # Linear reward
            current_position_reward = -np.abs((desired_location - self.state[:self.POSITION_STATE_LENGTH])*self.REWARD_WEIGHTING)* self.TARGET_REWARD
        else:
            # Exponential reward
            current_position_reward = np.exp(-np.sum(np.absolute(desired_location - self.state[:self.POSITION_STATE_LENGTH])*self.REWARD_WEIGHTING)) * self.TARGET_REWARD

        reward = np.zeros(1)

        # If it's not the first timestep, calculate the differential reward
        if np.all([self.previous_position_reward[i] is not None for i in range(len(self.previous_position_reward))]):
            reward = (current_position_reward - self.previous_position_reward)*self.REWARD_MULTIPLIER
            for i in range(len(reward)):
                if reward[i] < 0:
                    reward[i]*= self.NEGATIVE_PENALTY_FACTOR

        self.previous_position_reward = current_position_reward

        # Collapsing to a scalar
        reward = np.sum(reward)

        # Giving a penalty for docking too quickly
        if self.phase_number == 1 and np.any(np.abs(action) > self.MAX_DOCKING_SPEED):
            reward -= self.DOCKING_TOO_FAST_PENALTY

        # Giving a massive penalty for falling off the table
        if self.state[0] > self.UPPER_STATE_BOUND[0] or self.state[0] < self.LOWER_STATE_BOUND[0] or self.state[1] > self.UPPER_STATE_BOUND[1] or self.state[1] < self.LOWER_STATE_BOUND[1]:
            reward -= self.FALL_OFF_TABLE_PENALTY/self.TIMESTEP

        # Giving a large reward for completing the task
        if np.sum(np.absolute(self.state[:self.POSITION_STATE_LENGTH] - desired_location)) < 0.01:
            reward += self.GOAL_REWARD
            
        # Giving a large penalty for colliding with the obstacle
        if np.linalg.norm(self.state[:self.POSITION_STATE_LENGTH-1] - self.obstacle_location) <= self.OBSTABLE_DISTANCE and self.USE_OBSTACLE:
            reward -= self.OBSTABLE_PENALTY
            
        # Giving a penalty for colliding with the target
        if np.linalg.norm(self.state[:self.POSITION_STATE_LENGTH-1] - self.target_location[:-1]) <= self.TARGET_COLLISION_DISTANCE:
            reward -= self.TARGET_COLLISION_PENALTY
            
        # Giving a penalty for high velocities near the target location
        if self.PENALIZE_VELOCITY:
            radius = np.linalg.norm(desired_location[:2]- self.target_location[:2]) # vector from the target to the desired location
            reference_velocity = self.TARGET_ANGULAR_VELOCITY*np.array([-radius*np.sin(self.target_location[2]), radius*np.cos(self.target_location[2]), 1])
            reward -= np.sum(np.abs(action - reference_velocity)/(self.pose_error()**2+0.01)*self.VELOCITY_PENALTY)

        # Multiplying the reward by the TIMESTEP to give the rewards on a per-second basis
        return (reward*self.TIMESTEP).squeeze()

    def is_done(self):
        # Checks if this episode is done or not
        """
            NOTE: THE ENVIRONMENT MUST RETURN done = True IF THE EPISODE HAS
                  REACHED ITS LAST TIMESTEP
        """

        # If we've fallen off the table, end the episode
        if self.state[0] > self.UPPER_STATE_BOUND[0] or self.state[0] < self.LOWER_STATE_BOUND[0] or self.state[1] > self.UPPER_STATE_BOUND[1] or self.state[1] < self.LOWER_STATE_BOUND[1]:
            done = self.END_ON_FALL
        else:
            done = False

        # If we've spun too many times
        if self.state[2] > self.UPPER_STATE_BOUND[2] or self.state[2] < self.LOWER_STATE_BOUND[2]:
            pass

        # If we've run out of timesteps
        if round(self.time/self.TIMESTEP) == self.MAX_NUMBER_OF_TIMESTEPS:
            done = True

        return done


    def generate_queue(self):
        # Generate the queues responsible for communicating with the agent
        self.agent_to_env = multiprocessing.Queue(maxsize = 1)
        self.env_to_agent = multiprocessing.Queue(maxsize = 1)

        return self.agent_to_env, self.env_to_agent
    
    def obstable_relative_location(self):
        # Returns the position of the obstacle with respect to the chaser
        relative_position = self.obstacle_location - self.state[:self.POSITION_STATE_LENGTH-1]
        
        return relative_position

    def run(self):
        ###################################
        ##### Running the environment #####
        ###################################
        """
        This method is called when the environment process is launched by main.py.
        It is responsible for continually listening for an input action from the
        agent through a Queue. If an action is received, it is to step the environment
        and return the results.
        """
        # Instructing this process to treat Ctrl+C events (called SIGINT) by going SIG_IGN (ignore).
        # This permits the process to continue upon a Ctrl+C event to allow for graceful quitting.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Loop until the process is terminated
        while True:
            # Blocks until the agent passes us an action
            action, *test_time = self.agent_to_env.get()

            if type(action) == bool:
                # The signal to reset the environment was received
                self.reset(action, test_time[0])
                # Return the results
                self.env_to_agent.put((np.append(self.state[:self.POSITION_STATE_LENGTH], np.append(self.pose_error(), self.obstable_relative_location())), self.target_location))

            else:
                ################################
                ##### Step the environment #####
                ################################
                next_state, reward, done, *guidance_position = self.step(action)

                # Return the results
                self.env_to_agent.put((np.append(next_state[:self.POSITION_STATE_LENGTH], np.append(self.pose_error(), self.obstable_relative_location())), reward, done, guidance_position))

###################################################################
##### Generating kinematics equations representing the motion #####
###################################################################
def kinematics_equations_of_motion(state, t, parameters):
    # From the state, it returns the first derivative of the state

    # Unpacking the action from the parameters
    action = parameters[0]

    # Building the derivative matrix. For kinematics, d(state)/dt = action = \dot{state}
    derivatives = action

    return derivatives


#####################################################################
##### Generating the dynamics equations representing the motion #####
#####################################################################
def dynamics_equations_of_motion(state, t, parameters):
    # state = [x, y, theta, xdot, ydot, thetadot]

    # Unpacking the state
    x, y, theta, xdot, ydot, thetadot = state
    control_effort, mass, inertia = parameters # unpacking parameters

    derivatives = np.array((xdot, ydot, thetadot, control_effort[0]/mass, control_effort[1]/mass, control_effort[2]/inertia)).squeeze()

    return derivatives


##########################################
##### Function to animate the motion #####
##########################################
def render(states, actions, desired_pose, instantaneous_reward_log, cumulative_reward_log, critic_distributions, target_critic_distributions, projected_target_distribution, bins, loss_log, guidance_position_log, episode_number, filename, save_directory):

    # Load in a temporary environment, used to grab the physical parameters
    temp_env = Environment()

    # Checking if we want the additional reward and value distribution information
    extra_information = temp_env.ADDITIONAL_VALUE_INFO

    # Unpacking state
    x, y, theta = states[:,0], states[:,1], states[:,2]

    # Extracting physical properties
    length = temp_env.LENGTH

    # Calculating spacecraft corner locations through time #

    # Corner locations in body frame
    r1_b = length/2.*np.array([[ 1.], [ 1.]]) # [2, 1]
    r2_b = length/2.*np.array([[ 1.], [-1.]])
    r3_b = length/2.*np.array([[-1.], [-1.]])
    r4_b = length/2.*np.array([[-1.], [ 1.]])

    # Rotation matrix (body -> inertial)
    C_Ib = np.moveaxis(np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta),  np.cos(theta)]]), source = 2, destination = 0) # [NUM_TIMESTEPS, 2, 2]

    # Rotating the corner locations to the inertial frame
    r1_I = np.matmul(C_Ib, r1_b) # [NUM_TIMESTEPS, 2, 1]
    r2_I = np.matmul(C_Ib, r2_b) # [NUM_TIMESTEPS, 2, 1]
    r3_I = np.matmul(C_Ib, r3_b) # [NUM_TIMESTEPS, 2, 1]
    r4_I = np.matmul(C_Ib, r4_b) # [NUM_TIMESTEPS, 2, 1]

    # Calculating desired pose #

    # Calculating target angles
    target_angles = desired_pose[2] + [temp_env.TARGET_ANGULAR_VELOCITY*i*temp_env.TIMESTEP for i in range(len(theta))]

    # Rotation matrix (body -> inertial)
    C_Ib = np.moveaxis(np.array([[np.cos(target_angles), -np.sin(target_angles)],
                     [np.sin(target_angles),  np.cos(target_angles)]]), source = 2, destination = 0) # [NUM_TIMESTEPS, 2, 2]

    # Rotating corner locations to the inertial frame
    r1_des = np.matmul(C_Ib, r1_b) # [2, 1]
    r2_des = np.matmul(C_Ib, r2_b) # [2, 1]
    r3_des = np.matmul(C_Ib, r3_b) # [2, 1]
    r4_des = np.matmul(C_Ib, r4_b) # [2, 1]
    
    # Calculating obstacle position at all points in time
    obstacle_x = temp_env.OBSTACLE_INITIAL_POSITION[0] + [temp_env.OBSTABLE_VELOCITY[0]*i*temp_env.TIMESTEP for i in range(len(theta))] # x = v_x*t
    obstacle_y = temp_env.OBSTACLE_INITIAL_POSITION[1] + [temp_env.OBSTABLE_VELOCITY[1]*i*temp_env.TIMESTEP for i in range(len(theta))] # y = v_y*t

    # Assembling desired pose into lists
    #r_des_x = [r2_des[0], r3_des[0], r4_des[0], r1_des[0]] + desired_pose[0]
    #r_des_y = [r2_des[1], r3_des[1], r4_des[1], r1_des[1]] + desired_pose[1]
    #r_des_front_x = [r1_des[0], r2_des[0]] + desired_pose[0]
    #r_des_front_y = [r1_des[1], r2_des[1]] + desired_pose[1]

    # Table edges
    #table = np.array([[0,0], [3.5, 0], [3.5, 2.41], [0, 2.41], [0, 0]])

    # Generating figure window
    figure = plt.figure(constrained_layout = True)
    figure.set_size_inches(5, 4, True)

    if extra_information:
        grid_spec = gridspec.GridSpec(nrows = 2, ncols = 3, figure = figure)
        subfig1 = figure.add_subplot(grid_spec[0,0], aspect = 'equal', autoscale_on = False, xlim = (0, 3.4), ylim = (0, 2.4))
        #subfig1 = figure.add_subplot(grid_spec[0,0], aspect = 'equal', autoscale_on = False, xlim = (-2, 2), ylim = (-2, 2))
        subfig2 = figure.add_subplot(grid_spec[0,1], xlim = (np.min([np.min(instantaneous_reward_log), 0]) - (np.max(instantaneous_reward_log) - np.min(instantaneous_reward_log))*0.02, np.max([np.max(instantaneous_reward_log), 0]) + (np.max(instantaneous_reward_log) - np.min(instantaneous_reward_log))*0.02), ylim = (-0.5, 0.5))
        subfig3 = figure.add_subplot(grid_spec[0,2], xlim = (np.min(loss_log)-0.01, np.max(loss_log)+0.01), ylim = (-0.5, 0.5))
        subfig4 = figure.add_subplot(grid_spec[1,0], ylim = (0, 1.02))
        subfig5 = figure.add_subplot(grid_spec[1,1], ylim = (0, 1.02))
        subfig6 = figure.add_subplot(grid_spec[1,2], ylim = (0, 1.02))

        # Setting titles
        subfig1.set_xlabel("X Position (m)",    fontdict = {'fontsize': 8})
        subfig1.set_ylabel("Y Position (m)",    fontdict = {'fontsize': 8})
        subfig2.set_title("Timestep Reward",    fontdict = {'fontsize': 8})
        subfig3.set_title("Current loss",       fontdict = {'fontsize': 8})
        subfig4.set_title("Q-dist",             fontdict = {'fontsize': 8})
        subfig5.set_title("Target Q-dist",      fontdict = {'fontsize': 8})
        subfig6.set_title("Bellman projection", fontdict = {'fontsize': 8})

        # Changing around the axes
        subfig1.tick_params(labelsize = 8)
        subfig2.tick_params(which = 'both', left = False, labelleft = False, labelsize = 8)
        subfig3.tick_params(which = 'both', left = False, labelleft = False, labelsize = 8)
        subfig4.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = False, labelsize = 8)
        subfig5.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = False, labelsize = 8)
        subfig6.tick_params(which = 'both', left = False, labelleft = False, right = True, labelright = True, labelsize = 8)

        # Adding the grid
        subfig4.grid(True)
        subfig5.grid(True)
        subfig6.grid(True)

        # Setting appropriate axes ticks
        subfig2.set_xticks([np.min(instantaneous_reward_log), 0, np.max(instantaneous_reward_log)] if np.sign(np.min(instantaneous_reward_log)) != np.sign(np.max(instantaneous_reward_log)) else [np.min(instantaneous_reward_log), np.max(instantaneous_reward_log)])
        subfig3.set_xticks([np.min(loss_log), np.max(loss_log)])
        subfig4.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig4.tick_params(axis = 'x', labelrotation = -90)
        subfig4.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        subfig5.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig5.tick_params(axis = 'x', labelrotation = -90)
        subfig5.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        subfig6.set_xticks([bins[i*5] for i in range(round(len(bins)/5) + 1)])
        subfig6.tick_params(axis = 'x', labelrotation = -90)
        subfig6.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])

    else:
        subfig1 = figure.add_subplot(1, 1, 1, aspect = 'equal', autoscale_on = False, xlim = (0, 3.4), ylim = (0, 2.4), xlabel = 'X Position (m)', ylabel = 'Y Position (m)')

    # Defining plotting objects that change each frame
    body,       = subfig1.plot([], [], color = 'k', linestyle = '-', linewidth = 2) # Note, the comma is needed
    front_face, = subfig1.plot([], [], color = 'g', linestyle = '-', linewidth = 2) # Note, the comma is needed
    body_dot    = subfig1.scatter([], [], color = 'r')

    if extra_information:
        reward_bar           = subfig2.barh(y = 0, height = 0.2, width = 0)
        loss_bar             = subfig3.barh(y = 0, height = 0.2, width = 0)
        q_dist_bar           = subfig4.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        target_q_dist_bar    = subfig5.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        projected_q_dist_bar = subfig6.bar(x = bins, height = np.zeros(shape = len(bins)), width = bins[1]-bins[0])
        time_text            = subfig1.text(x = 0.2, y = 0.91, s = '', fontsize = 8, transform=subfig1.transAxes)
        reward_text          = subfig1.text(x = 0.0,  y = 1.02, s = '', fontsize = 8, transform=subfig1.transAxes)
    else:
        time_text    = subfig1.text(x = 0.03, y = 0.96, s = '', fontsize = 8, transform=subfig1.transAxes)
        reward_text  = subfig1.text(x = 0.62, y = 0.96, s = '', fontsize = 8, transform=subfig1.transAxes)
        episode_text = subfig1.text(x = 0.40, y = 1.02, s = '', fontsize = 8, transform=subfig1.transAxes)

    # Plotting constant items once (background, etc)
    #table,             = subfig1.plot(table[:,0], table[:,1],       color = 'k', linestyle = '-', linewidth = 3)
    desired_pos,       = subfig1.plot([], [],             color = 'r', linestyle = '-', linewidth = 2)
    desired_pos_front, = subfig1.plot([], [], color = 'k', linestyle = '-', linewidth = 2)

    # Function called once to initialize axes as empty
    def initialize_axes():
        body.set_data([], [])
        front_face.set_data([], [])
        time_text.set_text('')

        if not extra_information:
            episode_text.set_text('Episode ' + str(episode_number))

        return body, front_face, time_text, body_dot

    # Function called repeatedly to draw each frame
    def render_one_frame(frame, *fargs):
        temp_env = fargs[0] # Extract environment from passed args

        # Draw the spacecraft body
        thisx = [r2_I[frame,0,0], r3_I[frame,0,0], r4_I[frame,0,0], r1_I[frame,0,0]] + x[frame]
        thisy = [r2_I[frame,1,0], r3_I[frame,1,0], r4_I[frame,1,0], r1_I[frame,1,0]] + y[frame]
        body.set_data(thisx, thisy)

        # Draw the front face of the spacecraft body in a different colour
        thisx = [r1_I[frame,0,0], r2_I[frame,0,0]] + x[frame]
        thisy = [r1_I[frame,1,0], r2_I[frame,1,0]] + y[frame]
        front_face.set_data(thisx, thisy)

        # Draw the target body
        thisx = [r2_des[frame,0,0], r3_des[frame,0,0], r4_des[frame,0,0], r1_des[frame,0,0]] + desired_pose[0]
        thisy = [r2_des[frame,1,0], r3_des[frame,1,0], r4_des[frame,1,0], r1_des[frame,1,0]] + desired_pose[1]
        desired_pos.set_data(thisx, thisy)

        # Draw the front face of the target body in a different colour
        thisx = [r1_des[frame,0,0], r2_des[frame,0,0]] + desired_pose[0]
        thisy = [r1_des[frame,1,0], r2_des[frame,1,0]] + desired_pose[1]
        desired_pos_front.set_data(thisx, thisy)

        body_dot.set_offsets(np.hstack((x[frame],y[frame])))
        if frame != 0:
            subfig1.patches.clear() # remove the last frame's arrow
        # Update the time text
        time_text.set_text('Time = %.1f s' %(frame*temp_env.TIMESTEP))
        
        if temp_env.USE_OBSTACLE:
            obstacle_dot = plt.Circle((obstacle_x[frame], obstacle_y[frame]), radius = np.max([0.01,temp_env.OBSTABLE_DISTANCE - np.sqrt(2)*temp_env.LENGTH/2]), fill = False, color = 'k')
            subfig1.add_patch(obstacle_dot)

        # Update the reward text
        reward_text.set_text('Total reward = %.1f' %cumulative_reward_log[frame])

        if extra_information:
            # Updating the instantaneous reward bar graph
            reward_bar[0].set_width(instantaneous_reward_log[frame])
            # And colouring it appropriately
            if instantaneous_reward_log[frame] < 0:
                reward_bar[0].set_color('r')
            else:
                reward_bar[0].set_color('g')

            # Updating the loss bar graph
            loss_bar[0].set_width(loss_log[frame])

            # Updating the q-distribution plot
            for this_bar, new_value in zip(q_dist_bar, critic_distributions[frame,:]):
                this_bar.set_height(new_value)

            # Updating the target q-distribution plot
            for this_bar, new_value in zip(target_q_dist_bar, target_critic_distributions[frame, :]):
                this_bar.set_height(new_value)

            # Updating the projected target q-distribution plot
            for this_bar, new_value in zip(projected_q_dist_bar, projected_target_distribution[frame, :]):
                this_bar.set_height(new_value)

        # If dynamics are present, draw an arrow showing the location of the guided position
        if temp_env.TEST_ON_DYNAMICS:
            
            
            position_arrow = plt.Arrow(x[frame], y[frame], guidance_position_log[frame,0] - x[frame], guidance_position_log[frame,1] - y[frame], width = 0.06, color = 'k')

#            # Adding the rotational arrow
#            angle_error = guidance_position_log[-1,2] - theta[frame]
#            style="Simple,tail_width=0.5,head_width=4,head_length=8"
#            kw = dict(arrowstyle=style, color="k")
#            start_point = np.array((x[frame] + temp_env.LENGTH, y[frame]))
#            end_point   = np.array((x[frame] + temp_env.LENGTH*np.cos(angle_error), y[frame] + temp_env.LENGTH*np.sin(angle_error)))
#            half_chord = np.linalg.norm(end_point - start_point)/2
#            mid_dist = temp_env.LENGTH - (temp_env.LENGTH**2 - half_chord**2)**0.5
#            ratio = mid_dist/(half_chord)
#            ratio = 1
#            rotation_arrow = patches.FancyArrowPatch((x[frame] + temp_env.LENGTH, y[frame]), (x[frame] + temp_env.LENGTH*np.cos(angle_error), y[frame] + temp_env.LENGTH*np.sin(angle_error)), connectionstyle = "arc3,rad=" + str(ratio), **kw)
            subfig1.add_patch(position_arrow)
#            subfig1.add_patch(rotation_arrow)

        # Since blit = True, must return everything that has changed at this frame
        return body, front_face, time_text, body_dot

    # Generate the animation!
    fargs = [temp_env] # bundling additional arguments
    animator = animation.FuncAnimation(figure, render_one_frame, frames = np.linspace(0, len(states)-1, len(states)).astype(int),
                                       blit = True, init_func = initialize_axes, fargs = fargs)
    """
    frames = the int that is passed to render_one_frame. I use it to selectively plot certain data
    fargs = additional arguments for render_one_frame
    interval = delay between frames in ms
    """

    # Save the animation!
    try:
        # Save it to the working directory [have to], then move it to the proper folder
        animator.save(filename = filename + '_episode_' + str(episode_number) + '.mp4', fps = 30, dpi = 100)
        # Make directory if it doesn't already exist
        os.makedirs(os.path.dirname(save_directory + filename + '/videos/'), exist_ok=True)
        # Move animation to the proper directory
        os.rename(filename + '_episode_' + str(episode_number) + '.mp4', save_directory + filename + '/videos/episode_' + str(episode_number) + '.mp4')
    except:
        print("Skipping animation for episode %i due to an error" %episode_number)
        # Try to delete the partially completed video file
        try:
            os.remove(filename + '_episode_' + str(episode_number) + '.mp4')
        except:
            pass

    del temp_env
    plt.close(figure)